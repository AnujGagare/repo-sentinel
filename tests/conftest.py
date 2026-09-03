"""
Shared pytest fixtures.

Heavy external dependencies (the embedding model download, Ollama/Groq
network calls) are deliberately mocked throughout this suite rather than
skipped -- this keeps tests fast, deterministic, and runnable in CI
without needing a GPU, a running Ollama server, or network access to
Hugging Face. This is standard practice, not a workaround: unit tests
should isolate the code under test from external services.

For true end-to-end verification against the real embedding model and a
real LLM, see the manual test procedure in README.md's "Local development
setup" section -- that's a deliberate, separate layer of testing this
suite doesn't replace.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def tmp_persist_dir(tmp_path):
    """A throwaway directory for anything that needs to write to disk
    (ChromaDB, BM25 pickle, caches) -- pytest cleans this up automatically."""
    d = tmp_path / "persist"
    d.mkdir()
    return str(d)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Ensure tests never accidentally pick up real credentials or hit
    real services, regardless of what's in the actual environment."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("REPO_SENTINEL_LLM_BACKEND", "ollama")
