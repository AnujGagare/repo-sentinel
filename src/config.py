"""
Centralized application configuration.

Replaces scattered `os.environ.get(...)` calls throughout the codebase
with a single validated settings object. Benefits over scattered env
reads, which matter for a project meant to run reliably outside your own
laptop:

  - Fails FAST and LOUD at startup if something required is missing or
    malformed (e.g. GROQ_API_KEY unset while backend=groq), instead of
    failing confusingly mid-request the first time a user asks a question.
  - One place to see every configurable value and its default -- useful
    both for you maintaining this and for anyone reviewing the code.
  - Type-validated (pydantic) instead of "hope the string parses as an
    int later."

Values are read from environment variables, with an optional `.env` file
(see `.env.example`) loaded automatically for local development. In
production (Render, etc.) you set real environment variables instead of
shipping a `.env` file.
"""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REPO_SENTINEL_", env_file=".env", extra="ignore")

    # --- Repo / indexing ---
    repo_path: str = Field(default="./data/fastapi", description="Path to the git repo root to watch")
    index_subdir: str | None = Field(default="fastapi", description="Subfolder within repo_path to actually index")
    index_dir: str = Field(default="./data/index", description="Where the vector store / BM25 / caches persist")
    auto_summarize: bool = Field(default=False, description="Auto-generate summaries for undocumented functions (adds one LLM call per undocumented function/method at index time -- slow on a local model, off by default for that reason)")
    embedding_model: str = Field(
        default="jinaai/jina-embeddings-v2-base-code",
        description="Embedding model to use. The code-specific default needs "
                    "meaningfully more RAM than fits in Render's free-tier 512MB "
                    "limit -- override to a smaller model (e.g. "
                    "sentence-transformers/all-MiniLM-L6-v2) for memory-constrained "
                    "deployments. See render.yaml.",
    )
    eval_set_path: str = Field(default="./src/eval/eval_set.json")

    # --- Generation backend ---
    llm_backend: str = Field(default="ollama", description="'ollama' (local dev) or 'groq' (deployed)")

    # --- API security (see api/security.py) ---
    api_key: str | None = Field(
        default=None,
        description="If set, required as X-API-Key header for mutating/expensive endpoints "
                    "(/query, /reindex). Unset = open access, fine for local dev only.",
    )
    rate_limit_per_minute: int = Field(default=20, description="Max /query calls per minute per client IP")

    # --- Logging ---
    log_level: str = Field(default="INFO")

    @field_validator("llm_backend")
    @classmethod
    def _validate_backend(cls, v: str) -> str:
        v = v.lower()
        if v not in ("ollama", "groq"):
            raise ValueError(f"llm_backend must be 'ollama' or 'groq', got {v!r}")
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"log_level must be a standard logging level, got {v!r}")
        return v


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_", env_file=".env", extra="ignore")
    url: str = Field(default="http://localhost:11434")
    model: str = Field(default="llama3.1:8b")


class GroqSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GROQ_", env_file=".env", extra="ignore")
    api_key: str | None = Field(default=None)
    model: str = Field(default="llama-3.1-8b-instant")


def load_settings() -> tuple[Settings, OllamaSettings, GroqSettings]:
    """
    Loads and cross-validates all settings. Raises a clear error at import
    time (not mid-request) if the configuration is internally inconsistent
    -- e.g. backend=groq but no GROQ_API_KEY provided.
    """
    settings = Settings()
    ollama = OllamaSettings()
    groq = GroqSettings()

    if settings.llm_backend == "groq" and not groq.api_key:
        raise RuntimeError(
            "REPO_SENTINEL_LLM_BACKEND=groq but GROQ_API_KEY is not set. "
            "Set it in your environment or .env file before starting the app."
        )

    return settings, ollama, groq


if __name__ == "__main__":
    settings, ollama, groq = load_settings()
    print("Settings loaded successfully:")
    print(settings.model_dump())
    print(ollama.model_dump())
    print({**groq.model_dump(), "api_key": "***" if groq.api_key else None})
