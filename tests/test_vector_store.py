import numpy as np

from indexing.vector_store import VectorStore


def _make_store(tmp_persist_dir):
    return VectorStore(persist_dir=f"{tmp_persist_dir}/chroma", collection_name="test")


def test_upsert_and_count(tmp_persist_dir):
    store = _make_store(tmp_persist_dir)
    embeds = np.random.rand(3, 384)
    store.upsert_chunks(
        chunk_ids=["a", "b", "c"],
        embeddings=embeds,
        documents=["doc a", "doc b", "doc c"],
        metadatas=[{"file_path": "a.py"}, {"file_path": "b.py"}, {"file_path": "c.py"}],
    )
    assert store.count() == 3


def test_upsert_with_empty_list_is_a_noop(tmp_persist_dir):
    store = _make_store(tmp_persist_dir)
    store.upsert_chunks(chunk_ids=[], embeddings=np.empty((0, 384)), documents=[], metadatas=[])
    assert store.count() == 0


def test_query_returns_nearest_neighbor_first(tmp_persist_dir):
    store = _make_store(tmp_persist_dir)
    embeds = np.random.rand(3, 384)
    store.upsert_chunks(
        chunk_ids=["a", "b", "c"],
        embeddings=embeds,
        documents=["doc a", "doc b", "doc c"],
        metadatas=[{"file_path": "a.py"}, {"file_path": "b.py"}, {"file_path": "c.py"}],
    )
    results = store.query(embeds[0], top_k=1)
    assert results[0]["chunk_id"] == "a"
    assert results[0]["distance"] == 0.0


def test_delete_chunks_for_file_only_removes_matching_file(tmp_persist_dir):
    store = _make_store(tmp_persist_dir)
    embeds = np.random.rand(3, 384)
    store.upsert_chunks(
        chunk_ids=["a", "b", "c"],
        embeddings=embeds,
        documents=["doc a", "doc b", "doc c"],
        metadatas=[{"file_path": "auth.py"}, {"file_path": "auth.py"}, {"file_path": "routing.py"}],
    )
    store.delete_chunks_for_file("auth.py")
    assert store.count() == 1

    remaining = store.get_by_ids(["c"])
    assert "c" in remaining
    assert remaining["c"]["metadata"]["file_path"] == "routing.py"


def test_get_by_ids_returns_only_requested_chunks(tmp_persist_dir):
    store = _make_store(tmp_persist_dir)
    embeds = np.random.rand(3, 384)
    store.upsert_chunks(
        chunk_ids=["a", "b", "c"],
        embeddings=embeds,
        documents=["doc a", "doc b", "doc c"],
        metadatas=[{"file_path": "a.py"}, {"file_path": "b.py"}, {"file_path": "c.py"}],
    )
    result = store.get_by_ids(["a", "c"])
    assert set(result.keys()) == {"a", "c"}
    assert result["a"]["document"] == "doc a"


def test_get_by_ids_with_empty_list_returns_empty_dict(tmp_persist_dir):
    store = _make_store(tmp_persist_dir)
    assert store.get_by_ids([]) == {}


def test_upsert_is_idempotent_for_same_chunk_id(tmp_persist_dir):
    """This is what makes incremental re-indexing correct: re-upserting a
    chunk with an unchanged chunk_id (content-hashed -- see code_chunker.py)
    should update in place, not create a duplicate entry."""
    store = _make_store(tmp_persist_dir)
    embeds = np.random.rand(1, 384)
    store.upsert_chunks(chunk_ids=["a"], embeddings=embeds, documents=["v1"], metadatas=[{"file_path": "a.py"}])
    store.upsert_chunks(chunk_ids=["a"], embeddings=embeds, documents=["v2 updated"], metadatas=[{"file_path": "a.py"}])
    assert store.count() == 1
    result = store.get_by_ids(["a"])
    assert result["a"]["document"] == "v2 updated"
