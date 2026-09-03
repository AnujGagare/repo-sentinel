"""
Hybrid retriever: combines BM25 (keyword) and vector (semantic) search
results using Reciprocal Rank Fusion (RRF).

Why RRF and not just averaging scores:
BM25 scores and cosine-distance scores live on completely different scales
(BM25 is unbounded, distance is 0-2ish) so naively averaging them is
meaningless. RRF sidesteps this by only using each result's *rank position*
in its own list, not its raw score -- a well-established, simple, robust
way to merge heterogeneous rankers. Formula for each doc:

    RRF_score(d) = sum over each ranker r of  1 / (k + rank_r(d))

where k is a small constant (60 is the standard default from the original
RRF paper) that dampens the influence of very top-ranked outliers.
"""

from __future__ import annotations

from dataclasses import dataclass


RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk_id: str
    document: str
    metadata: dict
    rrf_score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None


def reciprocal_rank_fusion(
    bm25_results: list[tuple[str, float]],       # [(chunk_id, score), ...] from BM25Index.search
    vector_results: list[dict],                   # from VectorStore.query
    top_k: int = 20,
    extra_doc_lookup: dict[str, dict] | None = None,  # backfill for BM25-only hits, see get_by_ids()
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    bm25_ranks: dict[str, int] = {}
    vector_ranks: dict[str, int] = {}
    doc_lookup: dict[str, dict] = dict(extra_doc_lookup or {})

    for rank, (chunk_id, _score) in enumerate(bm25_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        bm25_ranks[chunk_id] = rank

    for rank, result in enumerate(vector_results, start=1):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        vector_ranks[cid] = rank
        doc_lookup[cid] = result

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]

    out = []
    for cid in ranked_ids:
        doc_info = doc_lookup.get(cid, {"document": "", "metadata": {}})
        out.append(RetrievedChunk(
            chunk_id=cid,
            document=doc_info.get("document", ""),
            metadata=doc_info.get("metadata", {}),
            rrf_score=scores[cid],
            bm25_rank=bm25_ranks.get(cid),
            vector_rank=vector_ranks.get(cid),
        ))
    return out


if __name__ == "__main__":
    # A chunk that ranks decently on BOTH keyword and semantic search should
    # outrank one that only wins on a single ranker -- that's the whole
    # point of hybrid search, so we test for it explicitly.
    bm25_results = [("chunk_A", 9.5), ("chunk_B", 4.0), ("chunk_C", 1.2)]
    vector_results = [
        {"chunk_id": "chunk_B", "document": "doc B", "metadata": {}},
        {"chunk_id": "chunk_A", "document": "doc A", "metadata": {}},
        {"chunk_id": "chunk_D", "document": "doc D", "metadata": {}},
    ]

    fused = reciprocal_rank_fusion(bm25_results, vector_results)
    print("Fused ranking:")
    for r in fused:
        print(f"  {r.chunk_id}  rrf={r.rrf_score:.4f}  bm25_rank={r.bm25_rank}  vector_rank={r.vector_rank}")

    assert fused[0].chunk_id == "chunk_A", "chunk appearing near top of both lists should win"
    print("PASS: chunk strong in both rankers correctly wins fusion")
