"""
Indexing pipeline: the orchestrator that ties together

    git_watcher (what changed) 
      -> code_chunker / doc_chunker (turn files into chunks)
      -> embedder (turn chunks into vectors)
      -> vector_store + bm25_index (make chunks searchable)

Supports two modes:
  - full_index(): index every file from scratch (first run)
  - incremental_index(): only touch files that changed since last indexed
    commit (used by the file watcher / git hook for live updates)

Note on repo_path vs. index_subdir:
GitWatcher requires the actual git repository ROOT (where .git lives) --
git diff/commit lookups fail otherwise. But you often only want to index a
subfolder of that repo (e.g. FastAPI's core `fastapi/` package, not its
`tests/` or `docs_src/`). `index_subdir`, if given, restricts which
changed files actually get chunked, while git operations still run
against the true repo root.

This is the piece that makes the "live-updating" claim of the project real
rather than aspirational -- run incremental_index() after every commit and
the index (and therefore every subsequent answer) reflects current code.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chunking.code_chunker import chunk_python_file, CodeChunk
from indexing.embedder import embed_texts
from indexing.vector_store import VectorStore
from indexing.bm25_index import BM25Index
from indexing.git_watcher import GitWatcher
from generation.llm_client import generate_summary

logger = logging.getLogger(__name__)


class IndexingPipeline:
    def __init__(self, repo_path: str, persist_dir: str = "./data/index", index_subdir: str | None = None,
                 auto_summarize: bool = True):
        self.repo_path = Path(repo_path)
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        # e.g. "fastapi" to only index files under <repo_path>/fastapi/
        self.index_subdir = index_subdir.strip("/\\") if index_subdir else None
        # See generation/llm_client.py:generate_summary -- enriches
        # undocumented functions with an auto-generated one-line summary
        # so BM25/embeddings have a natural-language signal to match on.
        # Disable for faster indexing when you don't need the extra recall
        # (e.g. quick local iteration on chunking/retrieval logic itself).
        self.auto_summarize = auto_summarize

        self.watcher = GitWatcher(str(repo_path), state_dir=str(self.persist_dir))
        self.vector_store = VectorStore(persist_dir=str(self.persist_dir / "chroma"))
        self.bm25_path = self.persist_dir / "bm25.pkl"
        self.bm25 = BM25Index.load(str(self.bm25_path)) if self.bm25_path.exists() else BM25Index()

        # Cache of chunk_id -> generated summary, so re-indexing never
        # re-generates a summary for a chunk whose content hasn't changed
        # (chunk_id is content-hashed -- see code_chunker.py). This is what
        # keeps auto-summarization a one-time cost per chunk, not a
        # per-reindex cost.
        self.summary_cache_path = self.persist_dir / "summary_cache.json"
        self._summary_cache: dict[str, str] = (
            json.loads(self.summary_cache_path.read_text()) if self.summary_cache_path.exists() else {}
        )

        # BM25 needs the FULL corpus text to rebuild (it's not naturally
        # incremental like the vector store), so we keep a chunk_id -> text
        # map on disk alongside it.
        self._chunk_text_cache: dict[str, str] = {}

    def _save_summary_cache(self) -> None:
        self.summary_cache_path.write_text(json.dumps(self._summary_cache))

    def _get_or_generate_summary(self, chunk: CodeChunk) -> str:
        if chunk.chunk_id in self._summary_cache:
            return self._summary_cache[chunk.chunk_id]
        try:
            summary = generate_summary(chunk.source, chunk.symbol_name)
        except Exception as e:
            # Never let a summary-generation hiccup (LLM down, timeout,
            # etc.) break indexing -- fall back to no summary for this
            # chunk rather than failing the whole pipeline.
            logger.warning("Summary generation failed for %s: %s", chunk.chunk_id, e)
            summary = ""
        self._summary_cache[chunk.chunk_id] = summary
        return summary

    def _in_scope(self, rel_path: str) -> bool:
        if self.index_subdir is None:
            return True
        normalized = rel_path.replace("\\", "/")
        return normalized.startswith(self.index_subdir + "/")

    def _filter_scope(self, paths: list[str]) -> list[str]:
        return [p for p in paths if self._in_scope(p)]


    def _chunk_file(self, rel_path: str) -> list[CodeChunk]:
        full_path = self.repo_path / rel_path
        if not full_path.exists():
            return []
        try:
            source = full_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return []
        if rel_path.endswith(".py"):
            return chunk_python_file(rel_path, source)
        return []  # markdown chunker plugs in here (see doc_chunker.py)

    def _index_files(self, file_paths: list[str]) -> int:
        all_chunks: list[CodeChunk] = []
        for fp in file_paths:
            # Remove old chunks for this file first (handles the case where
            # a function was deleted/renamed within an otherwise-modified file)
            self.vector_store.delete_chunks_for_file(fp)
            all_chunks.extend(self._chunk_file(fp))

        if not all_chunks:
            return 0

        # Count upfront so progress logging can show "N/M" -- without this,
        # auto-summarization (one sequential LLM call per undocumented
        # function) is completely invisible while running: verified in
        # real testing that a local, GPU-VRAM-constrained model can take
        # long enough per call that several minutes of silence looks
        # indistinguishable from a hang, even though it's working correctly.
        needs_summary = [
            c for c in all_chunks
            if not c.docstring and self.auto_summarize and c.symbol_type in ("function", "method")
        ]
        already_cached = sum(1 for c in needs_summary if c.chunk_id in self._summary_cache)
        to_generate = len(needs_summary) - already_cached
        if to_generate > 0:
            logger.info(
                "Generating summaries for %d undocumented function(s)/method(s) "
                "(%d already cached from a prior run). This calls the LLM once per "
                "function and can take a while on a slower local model -- progress "
                "logs below.", to_generate, already_cached,
            )

        texts = []
        summary_progress = 0
        for c in all_chunks:
            description = c.docstring
            if not description and self.auto_summarize and c.symbol_type in ("function", "method"):
                was_cached = c.chunk_id in self._summary_cache
                description = self._get_or_generate_summary(c)
                if not was_cached:
                    summary_progress += 1
                    logger.info("  [%d/%d] generated summary for %s", summary_progress, to_generate, c.symbol_name)
            # Repeat symbol_name once extra -- a small, cheap BM25 boost for
            # exact-name-style queries, on top of whatever natural-language
            # signal the docstring/summary provides.
            texts.append(f"{c.symbol_name} {c.symbol_name} {description or ''}\n{c.source}")

        if self.auto_summarize:
            self._save_summary_cache()

        logger.info("Embedding %d chunks...", len(texts))
        embeddings = embed_texts(texts)

        self.vector_store.upsert_chunks(
            chunk_ids=[c.chunk_id for c in all_chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c.to_metadata() for c in all_chunks],
        )

        for c, t in zip(all_chunks, texts):
            self._chunk_text_cache[c.chunk_id] = t

        return len(all_chunks)

    def _rebuild_bm25_from_vector_store(self) -> None:
        """
        BM25 index is cheap to rebuild fully (milliseconds even for
        thousands of chunks, as measured earlier), so rather than
        maintaining a separate incremental BM25 structure, we just pull
        the current full chunk set out of the vector store (source of
        truth) and rebuild BM25 from it every time indexing runs.
        """
        # ChromaDB doesn't expose "get everything" cleanly for huge
        # collections, but for a single-repo project size this is fine.
        raw = self.vector_store.collection.get()
        text_map = dict(zip(raw["ids"], raw["documents"]))
        self.bm25.build(text_map)
        self.bm25.save(str(self.bm25_path))

    def full_index(self) -> dict:
        t0 = time.time()
        changes = self.watcher.get_changes_since_last_index()
        files = self._filter_scope(changes.files_to_reindex())
        n_chunks = self._index_files(files)
        self._rebuild_bm25_from_vector_store()
        self.watcher.mark_indexed(self.watcher.current_commit())
        return {
            "mode": "full",
            "files_indexed": len(files),
            "chunks_indexed": n_chunks,
            "commit": self.watcher.current_commit(),
            "elapsed_seconds": round(time.time() - t0, 2),
        }

    def incremental_index(self) -> dict:
        t0 = time.time()
        changes = self.watcher.get_changes_since_last_index()
        if changes.is_empty:
            return {"mode": "incremental", "status": "no changes", "commit": self.watcher.current_commit()}

        files_to_remove = self._filter_scope(changes.files_to_remove())
        files_to_reindex = self._filter_scope(changes.files_to_reindex())

        for fp in files_to_remove:
            self.vector_store.delete_chunks_for_file(fp)

        n_chunks = self._index_files(files_to_reindex)
        self._rebuild_bm25_from_vector_store()
        self.watcher.mark_indexed(self.watcher.current_commit())

        return {
            "mode": "incremental",
            "files_changed": len(files_to_reindex) + len(files_to_remove),
            "files_reindexed": len(files_to_reindex),
            "files_removed": len(files_to_remove),
            "chunks_indexed": n_chunks,
            "from_commit": changes.from_commit,
            "to_commit": changes.to_commit,
            "elapsed_seconds": round(time.time() - t0, 2),
        }


if __name__ == "__main__":
    import sys as _sys
    repo = _sys.argv[1] if len(_sys.argv) > 1 else "data/fastapi"
    pipeline = IndexingPipeline(repo, persist_dir="./data/index_test")
    print("Running full index (this calls the real embedding model)...")
    result = pipeline.full_index()
    print(result)
