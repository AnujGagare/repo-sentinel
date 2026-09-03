from indexing.bm25_index import BM25Index, tokenize


def test_tokenize_splits_snake_case():
    tokens = tokenize("solve_dependencies")
    assert "solve" in tokens
    assert "dependencies" in tokens


def test_tokenize_splits_camel_case():
    tokens = tokenize("getCurrentUser")
    assert "get" in tokens
    assert "current" in tokens
    assert "user" in tokens


def test_exact_name_match_ranks_highest():
    idx = BM25Index()
    idx.build({
        "chunk_1": "def solve_dependencies(request, dependant): resolve the dependency tree",
        "chunk_2": "def get_current_user(token: str) -> User: decode jwt and return user",
        "chunk_3": "class Depends: marker class used to declare a dependency",
    })
    results = idx.search("solve dependencies")
    assert results[0][0] == "chunk_1"


def test_search_returns_empty_for_no_match():
    idx = BM25Index()
    idx.build({"chunk_1": "completely unrelated content about widgets"})
    results = idx.search("xyzabc123 nonsense query")
    assert results == []


def test_save_and_load_roundtrip(tmp_persist_dir):
    # NOTE: uses 3 varied documents, not 1-2. BM25's classic IDF formula
    # trends toward zero (or negative) for very small corpora regardless
    # of term rarity -- e.g. with exactly 2 docs, any term appearing in
    # just 1 of them lands at EXACTLY idf=0 by the formula's arithmetic
    # (log(1.5) - log(1.5) = 0), which this suite's own >0 score filter
    # then excludes. Not a bug in this code -- a well-known property of
    # BM25 at small corpus sizes. See test_degenerate_single_document_corpus
    # below for that behavior documented explicitly with a minimal repro.
    idx = BM25Index()
    idx.build({
        "chunk_1": "solve dependencies function walks the tree",
        "chunk_2": "get current user decode jwt token",
        "chunk_3": "completely different content about widgets and gadgets",
    })
    path = f"{tmp_persist_dir}/bm25.pkl"
    idx.save(path)

    loaded = BM25Index.load(path)
    results = loaded.search("solve dependencies")
    assert len(results) == 1
    assert results[0][0] == "chunk_1"


def test_degenerate_single_document_corpus_returns_no_results():
    """Documents a known BM25 limitation rather than hiding it: when a
    query term appears in every document in the corpus (trivially true
    with a single-document corpus), the classic Robertson-Sparse-Jones IDF
    formula produces a negative score, and rank_bm25's own epsilon-floor
    safeguard can't rescue it because the corpus-average IDF is negative
    too. This has no practical impact at real corpus sizes (hundreds of
    code chunks, where no term appears in anywhere near all of them) --
    asserted here so a future change to the >0 score filter in
    BM25Index.search() doesn't silently start returning nonsensical
    negative-relevance results without someone noticing."""
    idx = BM25Index()
    idx.build({"chunk_1": "solve dependencies function"})
    results = idx.search("solve dependencies")
    assert results == []


def test_empty_corpus_does_not_crash_build_or_search():
    """Regression test for a real crash found via real testing: submitting
    a repo with no .py/.md files (e.g. octocat/Hello-World, which has only
    a plain README) to /index_repo caused rank_bm25's BM25Okapi to divide
    by the corpus size internally and raise a bare ZeroDivisionError.
    An empty corpus is a legitimate outcome, not a program error."""
    idx = BM25Index()
    idx.build({})  # should not raise
    results = idx.search("anything")
    assert results == []


def test_search_before_build_returns_empty_rather_than_raising():
    idx = BM25Index()
    assert idx.search("anything") == []



    """Not a strict perf assertion (CI machines vary), but guards against
    an accidental O(n^2)-or-worse regression that would make this
    unusable at real corpus sizes."""
    import time
    corpus = {f"chunk_{i}": f"function number {i} does something with widgets and gadgets" for i in range(500)}
    idx = BM25Index()
    t0 = time.time()
    idx.build(corpus)
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"indexing 500 chunks took {elapsed:.2f}s, expected well under 5s"
