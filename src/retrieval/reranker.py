"""
Reranker: takes the top-N candidates from hybrid retrieval and re-scores
them with a cross-encoder model for higher precision.

Why this is a separate step from retrieval:
Embedding search and BM25 both score a query against documents
independently (bi-encoder style) -- fast, but approximate. A cross-encoder
instead feeds the (query, document) pair together into one model, letting
it directly reason about their relationship -- much more accurate, but too
slow to run against an entire corpus. So the standard pattern is:
  1. Cheap retrieval (BM25 + embeddings) narrows corpus -> top ~20 candidates
  2. Expensive-but-accurate reranker narrows top ~20 -> top ~5 for the LLM

This is consistently one of the highest-leverage additions to a RAG
pipeline's answer quality, which is why it's worth having as its own
explicit stage (and its own eval metric) rather than skipping it.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_RERANKER_MODEL)
    return _model


def rerank(query: str, candidates: list, top_k: int = 5) -> list:
    """
    candidates: list of RetrievedChunk (from hybrid_retriever.py), or any
    object with a `.document` attribute.
    Returns the same objects, re-sorted, truncated to top_k, with a
    `.rerank_score` attribute attached.
    """
    if not candidates:
        return []
    model = _get_model()
    pairs = [(query, c.document) for c in candidates]
    scores = model.predict(pairs)

    for c, s in zip(candidates, scores):
        c.rerank_score = float(s)

    return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)[:top_k]


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class FakeChunk:
        chunk_id: str
        document: str
        rerank_score: float = 0.0

    query = "how does FastAPI resolve dependency injection"
    candidates = [
        FakeChunk("a", "def solve_dependencies(request, dependant): walks the dependency tree and resolves each Depends() marker recursively"),
        FakeChunk("b", "def format_error_message(exc): converts an exception into a JSON-serializable error response"),
        FakeChunk("c", "class Depends: marker class used in function signatures to declare a dependency to be injected"),
    ]

    print("Reranking (downloads cross-encoder model on first run)...")
    results = rerank(query, candidates, top_k=3)
    for r in results:
        print(f"  {r.rerank_score:.3f}  {r.chunk_id}: {r.document[:70]}")
