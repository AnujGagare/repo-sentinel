"""
LLM generation client.

Switches backend based on the REPO_SENTINEL_LLM_BACKEND env var:
  - "ollama" (default, for local development): calls a locally running
    Ollama server, free, fully offline once the model is pulled.
  - "groq": calls Groq's hosted API (free tier), used for the deployed
    version since free cloud hosts can't run a local LLM continuously.

Both backends are called through the same `generate_answer()` function so
the rest of the app (API layer, eval harness) doesn't need to know or care
which one is active -- this is the standard adapter pattern for swapping
LLM providers without touching business logic.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import load_settings

logger = logging.getLogger(__name__)

_settings, _ollama, _groq = load_settings()

BACKEND = _settings.llm_backend
OLLAMA_URL = _ollama.url
OLLAMA_MODEL = _ollama.model
GROQ_API_KEY = _groq.api_key
GROQ_MODEL = _groq.model
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Transient network/timeout errors (a model warming up, a momentary Groq
# rate limit, Ollama still loading the model into memory) are common
# enough in practice that failing on the first hiccup would be a poor
# production experience. Retry a small, bounded number of times with
# backoff before giving up -- deliberately NOT retrying on 4xx errors
# (bad request, auth failure), since retrying those just wastes time on
# something that will never succeed.
_MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 2


def _with_retries(fn, *args, **kwargs):
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except requests.HTTPError as e:
            if e.response is not None and 400 <= e.response.status_code < 500:
                raise  # don't retry client errors -- they won't self-resolve
            last_exc = e
        except requests.RequestException as e:
            last_exc = e
        if attempt < _MAX_RETRIES:
            wait = _RETRY_BACKOFF_SECONDS * (attempt + 1)
            logger.warning("LLM call failed (attempt %d/%d), retrying in %ds: %s",
                           attempt + 1, _MAX_RETRIES + 1, wait, last_exc)
            time.sleep(wait)
    raise last_exc


SYSTEM_PROMPT = """You are a code assistant that answers questions about a specific codebase.

Rules:
1. Answer ONLY using the provided code/doc excerpts below. Do not use outside knowledge about this or any other library.
2. Every claim you make must cite the file path and line numbers it came from, in the format (file.py:12-34).
3. If the provided excerpts don't contain enough information to answer, say so explicitly instead of guessing.
4. Be concise and technical -- this is for a developer audience.
"""


def _build_user_prompt(question: str, chunks: list) -> str:
    context_blocks = []
    for c in chunks:
        meta = c.metadata
        loc = f"{meta.get('file_path', '?')}:{meta.get('start_line', '?')}-{meta.get('end_line', '?')}"
        context_blocks.append(f"--- {loc} ---\n{c.document}")

    context = "\n\n".join(context_blocks)
    return f"""Code/doc excerpts:

{context}

Question: {question}"""


def generate_answer(question: str, chunks: list) -> str:
    """chunks: list of RetrievedChunk (post-rerank), most relevant first."""
    user_prompt = _build_user_prompt(question, chunks)

    if BACKEND == "groq":
        return _with_retries(_call_groq, user_prompt)
    return _with_retries(_call_ollama, user_prompt)


def _call_ollama(user_prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def _call_groq(user_prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable not set")
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def generate_summary(code: str, symbol_name: str) -> str:
    """
    Generates a short one-line natural-language summary for an undocumented
    function/method, used to enrich its indexed text.

    Why this exists: diagnosed via /debug_retrieve that `solve_dependencies`
    (no docstring) never surfaced in retrieval for the query "how does
    FastAPI resolve dependencies" -- because its indexed text was just the
    bare function name + code, with no natural-language bridge between the
    query's wording ("resolve") and the code's wording ("solve"), and
    nothing for embeddings to latch onto semantically either. A short
    auto-generated summary gives both BM25 and the embedding model that
    missing natural-language signal.

    Deliberately a SEPARATE, smaller call from generate_answer() -- small
    prompt (one function's code, no large multi-chunk context), so even on
    a modest local model this is fast per-call. Called once per
    undocumented chunk and cached (see pipeline.py's summary cache), so
    it's a one-time cost, not a per-query cost.
    """
    prompt = f"""In ONE short sentence (under 20 words), describe what this function does. Use plain, general language a developer might naturally use to describe or search for it -- don't just restate the function name.

Function: {symbol_name}

Code:
{code[:2000]}

One-sentence summary:"""

    if BACKEND == "groq":
        return _with_retries(_call_groq_completion, prompt)
    return _with_retries(_call_ollama_completion, prompt)


def _call_ollama_completion(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=150,  # observed real response times up to ~90s on a VRAM-constrained
                      # local GPU (partial CPU offload) during actual testing of this
                      # project -- a shorter timeout would trigger unnecessary retries
                      # (doubling wait time) on a call that was simply slow, not failing
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def _call_groq_completion(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable not set")
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    from dataclasses import dataclass, field

    @dataclass
    class FakeChunk:
        document: str
        metadata: dict = field(default_factory=dict)

    fake_chunks = [
        FakeChunk(
            document="def solve_dependencies(request, dependant):\n    values = {}\n    for sub_dependant in dependant.dependencies:\n        resolve(sub_dependant)\n    return values",
            metadata={"file_path": "dependencies/utils.py", "start_line": 400, "end_line": 410},
        )
    ]
    prompt = _build_user_prompt("How does dependency resolution work?", fake_chunks)
    print("Constructed prompt (this part needs no network, verified locally):\n")
    print(prompt)
    print(f"\nActive backend: {BACKEND}")
    print("(Actual generation call requires Ollama running locally, or GROQ_API_KEY set)")
