import pytest

from config import Settings, load_settings


def test_default_settings_load_successfully(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    settings, ollama, groq = load_settings()
    assert settings.llm_backend == "ollama"
    assert settings.auto_summarize is False


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        Settings(llm_backend="not_a_real_backend")


def test_backend_is_case_insensitive():
    settings = Settings(llm_backend="OLLAMA")
    assert settings.llm_backend == "ollama"


def test_invalid_log_level_rejected():
    with pytest.raises(ValueError):
        Settings(log_level="NOT_A_LEVEL")


def test_groq_backend_without_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("REPO_SENTINEL_LLM_BACKEND", "groq")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        load_settings()


def test_groq_backend_with_api_key_succeeds(monkeypatch):
    monkeypatch.setenv("REPO_SENTINEL_LLM_BACKEND", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    settings, ollama, groq = load_settings()
    assert settings.llm_backend == "groq"
    assert groq.api_key == "test-key-123"
