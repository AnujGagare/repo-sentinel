"""
BM25 keyword index.

Why this matters alongside embeddings:
Embedding search is great at "semantic" matches (concepts, paraphrases) but
often weak on exact identifiers -- e.g. a query for `solve_dependencies`
might retrieve semantically-similar-but-wrong functions instead of the
literal function of that name. BM25 (classic TF-IDF-family keyword scoring)
nails exact term matches reliably and cheaply, which is why production RAG
systems combine both ("hybrid search") rather than relying on embeddings
alone.

This module tokenizes each chunk's source (+ symbol name, weighted extra)
and builds an in-memory BM25 index. It's intentionally simple: for a
single-repo project, an in-memory index rebuilt on incremental updates is
fast enough and avoids extra infra.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(text: str) -> list[str]:
    """
    Code-aware tokenizer: splits on non-identifier chars, and additionally
    splits snake_case and camelCase identifiers into sub-tokens so that a
    query for "solve dependencies" still matches `solve_dependencies`.
    """
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        tokens.append(raw.lower())
        # split snake_case
        if "_" in raw:
            tokens.extend(p.lower() for p in raw.split("_") if p)
        # split camelCase
        camel_parts = re.findall(r"[A-Z][a-z0-9]*|[a-z0-9]+", raw)
        if len(camel_parts) > 1:
            tokens.extend(p.lower() for p in camel_parts)
    return tokens


class BM25Index:
    def __init__(self):
        self.chunk_ids: list[str] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunk_id_to_text: dict[str, str]) -> None:
        """
        chunk_id_to_text: mapping of chunk_id -> searchable text.
        We weight the symbol name by repeating it, so exact-name queries
        score even higher (a common, effective BM25 trick).

        Handles an empty corpus explicitly -- rank_bm25's BM25Okapi divides
        by the corpus size internally (average document length) and raises
        a bare ZeroDivisionError with no useful message if given zero
        documents. Found via real testing: submitting a repo with no
        .py/.md files (e.g. a repo containing only a plain README) to the
        /index_repo feature produced exactly this crash. An empty corpus
        isn't a program error -- it's a legitimate (if unhelpful) outcome
        that search() should handle by simply returning no results.
        """
        self.chunk_ids = list(chunk_id_to_text.keys())
        self._corpus_tokens = [tokenize(t) for t in chunk_id_to_text.values()]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Returns [(chunk_id, score), ...] sorted descending by score.
        Note: filters out non-positive scores. At real corpus sizes this
        only excludes genuinely irrelevant results, but be aware that BM25's
        classic IDF formula can produce negative scores for terms that
        appear in most/all documents in the corpus (most likely with very
        small or near-duplicate-heavy corpora) -- see
        tests/test_bm25_index.py::test_degenerate_single_document_corpus_returns_no_results
        for the specific edge case this guards against."""
        if not self.chunk_ids:
            return []  # never built, or built with an empty corpus -- either way, nothing to search
        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [(cid, score) for cid, score in ranked[:top_k] if score > 0]

    def save(self, path: str) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: str) -> "BM25Index":
        return pickle.loads(Path(path).read_bytes())


if __name__ == "__main__":
    # quick self-test
    idx = BM25Index()
    idx.build({
        "chunk_1": "def solve_dependencies(request, dependant): resolve the dependency tree",
        "chunk_2": "def get_current_user(token: str) -> User: decode jwt and return user",
        "chunk_3": "class Depends: marker class used to declare a dependency",
    })
    results = idx.search("solve dependencies")
    print("Query: 'solve dependencies'")
    for cid, score in results:
        print(f"  {cid}: {score:.3f}")
