"""
API security: key-based auth + basic rate limiting for endpoints that
cost real money or compute (/query hits an LLM; /reindex re-embeds
files).

Why this exists: once this is deployed publicly (see README deployment
section), /query is reachable by anyone with the URL and calls a paid-tier
-capable LLM API (Groq) on every request. Left completely open, that's a
real cost/abuse exposure, not a hypothetical one -- a portfolio demo is
exactly the kind of URL that gets scraped and hit by bots. Two
independent, deliberately simple layers:

  1. API key (optional): if REPO_SENTINEL_API_KEY is set, requests must
     include a matching `X-API-Key` header. Unset by default so local
     dev stays frictionless; set it for the deployed version.
  2. Rate limiting (always on): a basic in-memory sliding-window limiter
     per client IP. Deliberately NOT using a heavier solution (Redis-backed
     limiter, etc.) -- this app runs as a single process on a single free
     -tier instance, so in-memory state is the right amount of complexity
     for its actual deployment shape. If this ever runs multi-instance,
     this would need to move to shared storage; noted here rather than
     silently wrong.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request


class RateLimiter:
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._hits: dict[str, deque] = defaultdict(deque)

    def check(self, client_id: str) -> None:
        now = time.time()
        window = self._hits[client_id]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.max_per_minute:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({self.max_per_minute}/min). Try again shortly.",
            )
        window.append(now)


def make_api_key_dependency(expected_key: str | None):
    """
    Returns a FastAPI dependency that enforces the X-API-Key header when
    expected_key is set, and is a no-op when it isn't (local dev mode).
    """
    async def verify(x_api_key: str | None = Header(default=None)):
        if expected_key is None:
            return  # auth disabled -- local dev
        if x_api_key != expected_key:
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")
    return verify


def get_client_id(request: Request) -> str:
    """Best-effort client identifier for rate limiting. Falls back to a
    shared bucket if no address is available (e.g. some test clients)."""
    if request.client:
        return request.client.host
    return "unknown"
