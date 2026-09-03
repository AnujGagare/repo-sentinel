from retrieval.hybrid_retriever import reciprocal_rank_fusion


def test_chunk_strong_in_both_rankers_wins_fusion():
    bm25_results = [("chunk_A", 9.5), ("chunk_B", 4.0), ("chunk_C", 1.2)]
    vector_results = [
        {"chunk_id": "chunk_B", "document": "doc B", "metadata": {}},
        {"chunk_id": "chunk_A", "document": "doc A", "metadata": {}},
        {"chunk_id": "chunk_D", "document": "doc D", "metadata": {}},
    ]
    fused = reciprocal_rank_fusion(bm25_results, vector_results)
    assert fused[0].chunk_id == "chunk_A"


def test_bm25_only_chunk_has_no_metadata_without_backfill():
    """Documents the exact failure this project shipped and then fixed:
    a chunk found only by BM25 (not vector search) has no document/metadata
    UNLESS extra_doc_lookup is supplied. This isn't the desired end state --
    it's the bug. See test_bm25_only_chunk_gets_metadata_from_backfill for
    the fix, exercised through the same function."""
    bm25_results = [("chunk_X", 5.0)]
    vector_results = []
    fused = reciprocal_rank_fusion(bm25_results, vector_results)
    assert fused[0].document == ""
    assert fused[0].metadata == {}


def test_bm25_only_chunk_gets_metadata_from_backfill():
    bm25_results = [("chunk_X", 5.0)]
    vector_results = []
    extra_lookup = {"chunk_X": {"document": "real content", "metadata": {"file_path": "x.py"}}}
    fused = reciprocal_rank_fusion(bm25_results, vector_results, extra_doc_lookup=extra_lookup)
    assert fused[0].document == "real content"
    assert fused[0].metadata == {"file_path": "x.py"}


def test_respects_top_k_limit():
    bm25_results = [(f"chunk_{i}", 10.0 - i) for i in range(30)]
    fused = reciprocal_rank_fusion(bm25_results, [], top_k=5)
    assert len(fused) == 5


def test_empty_inputs_return_empty_list():
    assert reciprocal_rank_fusion([], []) == []


def test_ranks_recorded_correctly():
    bm25_results = [("a", 5.0), ("b", 3.0)]
    vector_results = [{"chunk_id": "b", "document": "doc b", "metadata": {}}]
    fused = reciprocal_rank_fusion(bm25_results, vector_results)
    by_id = {c.chunk_id: c for c in fused}
    assert by_id["a"].bm25_rank == 1
    assert by_id["a"].vector_rank is None
    assert by_id["b"].bm25_rank == 2
    assert by_id["b"].vector_rank == 1
