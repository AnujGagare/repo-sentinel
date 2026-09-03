from chunking.code_chunker import chunk_python_file, chunk_python_repo


SAMPLE_SOURCE = '''
def top_level_function(x: int) -> int:
    """A documented top-level function."""
    return x + 1


def undocumented_function(x):
    return x * 2


class Foo:
    """A documented class."""

    def __init__(self, x):
        self.x = x

    def method_a(self):
        """Method with a docstring."""
        return self.x
'''


def test_chunks_top_level_function_with_docstring():
    chunks = chunk_python_file("sample.py", SAMPLE_SOURCE)
    match = [c for c in chunks if c.symbol_name == "top_level_function"]
    assert len(match) == 1
    assert match[0].symbol_type == "function"
    assert match[0].docstring == "A documented top-level function."
    assert match[0].parent_class is None


def test_chunks_undocumented_function_with_empty_docstring():
    chunks = chunk_python_file("sample.py", SAMPLE_SOURCE)
    match = [c for c in chunks if c.symbol_name == "undocumented_function"]
    assert len(match) == 1
    assert match[0].docstring is None


def test_chunks_class_and_its_methods_separately():
    chunks = chunk_python_file("sample.py", SAMPLE_SOURCE)
    class_chunk = [c for c in chunks if c.symbol_type == "class" and c.symbol_name == "Foo"]
    method_chunks = [c for c in chunks if c.parent_class == "Foo"]
    assert len(class_chunk) == 1
    assert {c.symbol_name for c in method_chunks} == {"Foo.__init__", "Foo.method_a"}


def test_to_metadata_includes_clean_source_preview():
    """The frontend's expandable citation view needs pure code, not the
    combined symbol_name+summary+source blob used for retrieval (see
    pipeline.py) -- to_metadata() must expose a separate, clean field."""
    chunks = chunk_python_file("sample.py", SAMPLE_SOURCE)
    match = next(c for c in chunks if c.symbol_name == "top_level_function")
    meta = match.to_metadata()
    assert "source_preview" in meta
    assert "def top_level_function" in meta["source_preview"]
    assert len(meta["source_preview"]) <= 1500


def test_to_metadata_never_contains_none_values():
    """Regression test: ChromaDB's metadata validator rejects None outright
    (only str/int/float/bool allowed). A top-level function's parent_class
    is None before normalization -- to_metadata() must convert it to ''."""
    chunks = chunk_python_file("sample.py", SAMPLE_SOURCE)
    for c in chunks:
        meta = c.to_metadata()
        assert all(v is not None for v in meta.values()), f"None value in metadata for {c.symbol_name}: {meta}"


def test_chunk_id_is_stable_for_identical_content():
    chunks_a = chunk_python_file("sample.py", SAMPLE_SOURCE)
    chunks_b = chunk_python_file("sample.py", SAMPLE_SOURCE)
    ids_a = {c.symbol_name: c.chunk_id for c in chunks_a}
    ids_b = {c.symbol_name: c.chunk_id for c in chunks_b}
    assert ids_a == ids_b


def test_chunk_id_changes_when_source_changes():
    original = chunk_python_file("sample.py", SAMPLE_SOURCE)
    modified_source = SAMPLE_SOURCE.replace("return x + 1", "return x + 2")
    modified = chunk_python_file("sample.py", modified_source)

    orig_id = next(c.chunk_id for c in original if c.symbol_name == "top_level_function")
    new_id = next(c.chunk_id for c in modified if c.symbol_name == "top_level_function")
    assert orig_id != new_id, "chunk_id should change when the function body changes (used for incremental reindexing)"


def test_skips_files_with_syntax_errors():
    chunks = chunk_python_file("broken.py", "def broken(:\n    pass")
    assert chunks == []


def test_repo_walk_ignores_configured_directories(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "real.py").write_text("def real_fn(): pass")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_real.py").write_text("def test_fn(): pass")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.py").write_text("def cached_fn(): pass")

    chunks = chunk_python_repo(str(tmp_path))
    symbol_names = {c.symbol_name for c in chunks}
    assert "real_fn" in symbol_names
    assert "test_fn" not in symbol_names  # tests/ is excluded by default
    assert "cached_fn" not in symbol_names  # __pycache__ is excluded


def test_groups_near_duplicate_methods_by_docstring_similarity():
    """The FastAPI-class regression test: a class with several methods
    sharing a near-identical docstring template (differing only in one
    word, like get/post/put/patch's HTTP-verb docstrings) should collapse
    into a single method_group chunk instead of flooding retrieval with
    near-duplicate individual chunks."""
    source = '''
class Router:
    def get(self, path):
        """
        Register a GET endpoint. This is a long shared docstring template
        that describes routing behavior in extensive detail, repeated
        nearly verbatim across every HTTP verb method below it.
        """
        pass

    def post(self, path):
        """
        Register a POST endpoint. This is a long shared docstring template
        that describes routing behavior in extensive detail, repeated
        nearly verbatim across every HTTP verb method below it.
        """
        pass

    def put(self, path):
        """
        Register a PUT endpoint. This is a long shared docstring template
        that describes routing behavior in extensive detail, repeated
        nearly verbatim across every HTTP verb method below it.
        """
        pass

    def unrelated_method(self):
        """Does something completely different with no shared wording at all."""
        pass
'''
    chunks = chunk_python_file("router.py", source)
    groups = [c for c in chunks if c.symbol_type == "method_group"]
    assert len(groups) == 1
    assert "get" in groups[0].symbol_name and "post" in groups[0].symbol_name and "put" in groups[0].symbol_name

    # the unrelated method should NOT be swept into the group
    individual = [c for c in chunks if c.symbol_type == "method"]
    assert any(c.symbol_name == "Router.unrelated_method" for c in individual)
