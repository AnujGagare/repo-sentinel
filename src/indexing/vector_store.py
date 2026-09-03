"""
Vector store wrapper around ChromaDB (local, embedded -- no separate server
process needed, which keeps deployment simple on a free host).

Key design point: `upsert_chunks` is idempotent and keyed by chunk_id
(the content-hashed id from code_chunker.py). This is what makes
incremental re-indexing correct:
  - If a chunk's content hasn't changed, its chunk_id is identical, so
    upserting it again is a no-op in practice.
  - If a chunk's content changed, its chunk_id changes too (it's a hash of
    the content), so the old entry is simply never referenced again --
    we explicitly delete stale chunk_ids for files that were reindexed,
    via `delete_chunks_for_file`.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, persist_dir: str = "./data/chroma", collection_name: str = "repo_sentinel"):
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert_chunks(self, chunk_ids: list[str], embeddings, documents: list[str], metadatas: list[dict]) -> None:
        if not chunk_ids:
            return
        self.collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete_chunks_for_file(self, file_path: str) -> None:
        """Remove all indexed chunks belonging to a given file (used when a
        file changes or is deleted, before re-adding its fresh chunks)."""
        self.collection.delete(where={"file_path": file_path})

    def query(self, query_embedding, top_k: int = 20) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding],
            n_results=top_k,
        )
        out = []
        for i in range(len(results["ids"][0])):
            out.append({
                "chunk_id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return out

    def get_by_ids(self, chunk_ids: list[str]) -> dict[str, dict]:
        """Fetch document + metadata for a specific set of chunk_ids.
        Used to backfill metadata for chunks that BM25 found but vector
        search didn't (so hybrid fusion never has to fall back to an
        empty/unknown chunk)."""
        if not chunk_ids:
            return {}
        result = self.collection.get(ids=chunk_ids)
        return {
            cid: {"document": doc, "metadata": meta}
            for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
        }

    def count(self) -> int:
        return self.collection.count()



if __name__ == "__main__":
    import numpy as np
    import shutil

    test_dir = "/tmp/repo_sentinel_vecstore_test"
    shutil.rmtree(test_dir, ignore_errors=True)

    store = VectorStore(persist_dir=test_dir, collection_name="test")
    fake_embeds = np.random.rand(3, 384)
    store.upsert_chunks(
        chunk_ids=["a", "b", "c"],
        embeddings=fake_embeds,
        documents=["solve_dependencies code", "get_current_user code", "Depends class code"],
        metadatas=[{"file_path": "routing.py"}, {"file_path": "auth.py"}, {"file_path": "params.py"}],
    )
    print(f"Stored {store.count()} chunks")

    results = store.query(fake_embeds[0], top_k=2)
    print("Query results for vector 'a':")
    for r in results:
        print(f"  {r['chunk_id']}  distance={r['distance']:.4f}  {r['document']}")
    assert results[0]["chunk_id"] == "a", "nearest neighbor search broken"
    print("PASS: exact match returned as top result")

    # test deletion
    store.delete_chunks_for_file("auth.py")
    print(f"After deleting auth.py chunks: {store.count()} remain")
    assert store.count() == 2
    print("PASS: file-scoped deletion works")
