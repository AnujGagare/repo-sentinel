"""
FastAPI backend for Repo Sentinel.

Endpoints:
  GET  /healthz        -> liveness/readiness probe for deployment platforms
  POST /query           -> ask a question, get a grounded answer + citations
                           (rate-limited, API-key gated when configured)
  GET  /status          -> current indexed commit, chunk count, last index time
  POST /reindex          -> trigger incremental re-index (API-key gated when
                           configured -- this re-embeds files, so it's not
                           free to spam)
  GET  /debug_retrieve   -> diagnostic: shows retrieval at each pipeline
                           stage separately (BM25-only / vector-only /
                           fused / reranked), used to root-cause "why
                           didn't chunk X get retrieved" instead of guessing
  GET  /eval             -> run the eval harness against the current index

CORS is left open (`allow_origins=["*"]`) since this is a portfolio demo
meant to be hit from a static frontend on a different origin -- tighten
this if you ever put real data behind it.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_settings
from logging_config import configure_logging
from api.security import RateLimiter, get_client_id, make_api_key_dependency
from api.repo_jobs import RepoJobManager
from api.repo_validation import InvalidRepoUrlError
from indexing.pipeline import IndexingPipeline
from indexing.embedder import embed_query
from retrieval.hybrid_retriever import reciprocal_rank_fusion
from retrieval.reranker import rerank
from generation.llm_client import generate_answer
from eval.run_eval import load_eval_set, evaluate_retrieval

settings, ollama_settings, groq_settings = load_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="Repo Sentinel API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

require_api_key = make_api_key_dependency(settings.api_key)
rate_limiter = RateLimiter(max_per_minute=settings.rate_limit_per_minute)

pipeline: IndexingPipeline | None = None
startup_error: str | None = None
# Guards swapping `pipeline` when a new user-submitted repo finishes
# indexing (see /index_repo below) -- without this, a request reading
# `pipeline` mid-swap could see a half-updated state.
pipeline_lock = threading.Lock()


def _index_new_repo(repo_path: str, job) -> dict:
    """Called by RepoJobManager once a user-submitted repo has been cloned.
    Runs a full index against it and, on success, atomically swaps it in
    as the active repo. Indexes the WHOLE cloned repo (index_subdir=None)
    rather than assuming a package-subfolder structure like the FastAPI
    demo config does -- we can't know an arbitrary repo's layout."""
    global pipeline
    new_pipeline = IndexingPipeline(
        repo_path,
        persist_dir=f"{settings.index_dir}_job_{job.job_id}",
        index_subdir=None,
        auto_summarize=settings.auto_summarize,
    )
    result = new_pipeline.full_index()
    with pipeline_lock:
        pipeline = new_pipeline
    return result


repo_job_manager = RepoJobManager(
    clone_root=f"{settings.index_dir}_repos",
    on_ready=_index_new_repo,
)


@app.on_event("startup")
def startup():
    global pipeline, startup_error
    logger.info("Starting up. repo_path=%s index_subdir=%s backend=%s",
                settings.repo_path, settings.index_subdir, settings.llm_backend)
    try:
        pipeline = IndexingPipeline(
            settings.repo_path,
            persist_dir=settings.index_dir,
            index_subdir=settings.index_subdir,
            auto_summarize=settings.auto_summarize,
        )
        # Index on startup if nothing's indexed yet (first deploy). Subsequent
        # updates should go through /reindex (triggered by a git hook) rather
        # than re-running full_index every restart.
        if pipeline.watcher.last_indexed_commit() is None:
            logger.info("No prior index found -- running full index...")
            result = pipeline.full_index()
            logger.info("Full index complete: %s", result)
        else:
            logger.info("Existing index found at commit %s", pipeline.watcher.last_indexed_commit())
    except Exception as e:
        # Don't let a startup failure silently produce a half-broken app --
        # log it clearly and let /healthz report unhealthy, rather than
        # every endpoint failing with an unexplained 503.
        logger.exception("Startup failed")
        startup_error = str(e)


@app.get("/healthz")
def healthz():
    """Liveness/readiness probe. Returns 200 only when the pipeline is
    actually usable -- deployment platforms (Render, k8s, etc.) use this
    to decide whether to route traffic to this instance."""
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Startup failed: {startup_error}")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Still starting up")
    return {"status": "ok", "chunks_indexed": pipeline.vector_store.count()}


def _require_ready() -> IndexingPipeline:
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {startup_error}")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not ready")
    return pipeline


class QueryRequest(BaseModel):
    question: str
    top_k_retrieve: int = 30
    top_k_rerank: int = 5


class IndexRepoRequest(BaseModel):
    repo_url: str


class IndexJobResponse(BaseModel):
    job_id: str
    status: str
    detail: str | None = None
    error: str | None = None
    repo_url: str


class Citation(BaseModel):
    file_path: str
    symbol_name: str
    start_line: int
    end_line: int
    source_snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    indexed_commit: str
    latency_ms: int


def _retrieve_and_rerank(pipeline: IndexingPipeline, question: str, top_k_retrieve: int, top_k_rerank: int):
    query_vec = embed_query(question)
    vector_results = pipeline.vector_store.query(query_vec, top_k=top_k_retrieve)
    bm25_results = pipeline.bm25.search(question, top_k=top_k_retrieve)

    # Backfill metadata for chunks BM25 found but vector search didn't --
    # otherwise those entries would carry empty metadata (see
    # hybrid_retriever docstring / the "None::None" bug this fixes).
    vector_ids = {r["chunk_id"] for r in vector_results}
    bm25_only_ids = [cid for cid, _ in bm25_results if cid not in vector_ids]
    extra_lookup = pipeline.vector_store.get_by_ids(bm25_only_ids)

    fused = reciprocal_rank_fusion(bm25_results, vector_results, top_k=top_k_retrieve, extra_doc_lookup=extra_lookup)
    return rerank(question, fused, top_k=top_k_rerank)


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(req: QueryRequest, request: Request, pipeline: IndexingPipeline = Depends(_require_ready)):
    rate_limiter.check(get_client_id(request))

    if not req.question or not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    if len(req.question) > 2000:
        raise HTTPException(status_code=422, detail="question too long (max 2000 characters)")

    t0 = time.time()
    top_chunks = _retrieve_and_rerank(pipeline, req.question, req.top_k_retrieve, req.top_k_rerank)
    if not top_chunks:
        raise HTTPException(status_code=404, detail="No relevant code found for this question")

    try:
        answer = generate_answer(req.question, top_chunks)
    except Exception:
        logger.exception("Generation failed for question: %s", req.question)
        raise HTTPException(status_code=502, detail="The language model backend failed to respond. Try again shortly.")

    latency_ms = int((time.time() - t0) * 1000)
    logger.info("Query answered in %dms: %s", latency_ms, req.question[:80])

    citations = [
        Citation(
            file_path=c.metadata.get("file_path", "?"),
            symbol_name=c.metadata.get("symbol_name", "?"),
            start_line=c.metadata.get("start_line", 0),
            end_line=c.metadata.get("end_line", 0),
            source_snippet=c.metadata.get("source_preview", ""),
        )
        for c in top_chunks
    ]

    return QueryResponse(
        answer=answer,
        citations=citations,
        indexed_commit=pipeline.watcher.last_indexed_commit() or "unknown",
        latency_ms=latency_ms,
    )


@app.get("/status")
def status(pipeline: IndexingPipeline = Depends(_require_ready)):
    return {
        "indexed_commit": pipeline.watcher.last_indexed_commit(),
        "current_repo_commit": pipeline.watcher.current_commit(),
        "chunks_indexed": pipeline.vector_store.count(),
        "is_stale": pipeline.watcher.last_indexed_commit() != pipeline.watcher.current_commit(),
        "auto_summarize_enabled": pipeline.auto_summarize,
        "summaries_cached": len(pipeline._summary_cache),
    }


@app.post("/reindex", dependencies=[Depends(require_api_key)])
def reindex(pipeline: IndexingPipeline = Depends(_require_ready)):
    logger.info("Reindex triggered")
    result = pipeline.incremental_index()
    logger.info("Reindex result: %s", result)
    return result


@app.get("/debug_retrieve")
def debug_retrieve(question: str, top_k: int = 20, pipeline: IndexingPipeline = Depends(_require_ready)):
    """
    Diagnostic endpoint: shows retrieval results at each stage separately
    (BM25-only, vector-only, then fused+reranked) so a "why didn't chunk X
    show up" question can actually be answered instead of guessed at.
    If a chunk is missing even from the raw BM25/vector lists, the issue
    is retrieval depth or embedding relevance. If it's present there but
    drops out after reranking, the issue is the reranker's scoring.
    """
    query_vec = embed_query(question)
    vector_results = pipeline.vector_store.query(query_vec, top_k=top_k)
    bm25_results = pipeline.bm25.search(question, top_k=top_k)

    vector_ids = {r["chunk_id"] for r in vector_results}
    bm25_only_ids = [cid for cid, _ in bm25_results if cid not in vector_ids]
    extra_lookup = pipeline.vector_store.get_by_ids(bm25_only_ids)

    vector_lookup = {r["chunk_id"]: r for r in vector_results}

    def bm25_label(cid: str) -> str:
        meta = (extra_lookup.get(cid) or vector_lookup.get(cid) or {}).get("metadata", {})
        return f"{meta.get('file_path', '?')}::{meta.get('symbol_name', '?')}" if meta else cid

    fused = reciprocal_rank_fusion(bm25_results, vector_results, top_k=top_k, extra_doc_lookup=extra_lookup)
    reranked = rerank(question, list(fused), top_k=top_k)

    return {
        "question": question,
        "bm25_only_top_k": [f"{bm25_label(cid)} (score={score:.2f})" for cid, score in bm25_results],
        "vector_only_top_k": [f"{r['metadata'].get('file_path')}::{r['metadata'].get('symbol_name')}" for r in vector_results],
        "after_fusion_top_k": [f"{c.metadata.get('file_path')}::{c.metadata.get('symbol_name')} (rrf={c.rrf_score:.4f})" for c in fused],
        "after_rerank_top_k": [f"{c.metadata.get('file_path')}::{c.metadata.get('symbol_name')} (rerank_score={getattr(c, 'rerank_score', 0):.3f})" for c in reranked],
    }


@app.get("/eval")
def run_eval(pipeline: IndexingPipeline = Depends(_require_ready)):
    eval_set = load_eval_set(settings.eval_set_path)

    def retrieve_fn(question: str) -> list[dict]:
        chunks = _retrieve_and_rerank(pipeline, question, top_k_retrieve=40, top_k_rerank=10)
        return [c.metadata for c in chunks]

    return evaluate_retrieval(eval_set, retrieve_fn)


@app.post("/index_repo", response_model=IndexJobResponse, dependencies=[Depends(require_api_key)])
def index_repo(req: IndexRepoRequest, request: Request):
    """
    Starts indexing a user-supplied git repo in the background and returns
    a job_id immediately -- see api/repo_jobs.py for why this is async
    rather than blocking. Poll GET /index_repo/{job_id} for progress.

    Gated behind the API key (when configured) AND rate-limited, same as
    /query -- this is a genuinely expensive operation (a clone, a full
    embedding pass, and potentially dozens of LLM calls for undocumented
    -function summaries), so it needs at least the same protection as the
    per-question endpoint, arguably more.
    """
    rate_limiter.check(get_client_id(request))
    try:
        job = repo_job_manager.start_job(req.repo_url)
    except InvalidRepoUrlError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return IndexJobResponse(job_id=job.job_id, status=job.status, detail=job.detail,
                             error=job.error, repo_url=job.repo_url)


@app.get("/index_repo/{job_id}", response_model=IndexJobResponse)
def index_repo_status(job_id: str):
    job = repo_job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return IndexJobResponse(job_id=job.job_id, status=job.status, detail=job.detail,
                             error=job.error, repo_url=job.repo_url)
