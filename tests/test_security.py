import asyncio

import pytest
from fastapi import HTTPException

from api.security import RateLimiter, make_api_key_dependency


def test_rate_limiter_allows_up_to_limit():
    limiter = RateLimiter(max_per_minute=3)
    for _ in range(3):
        limiter.check("client_a")  # should not raise


def test_rate_limiter_blocks_after_limit_exceeded():
    limiter = RateLimiter(max_per_minute=3)
    for _ in range(3):
        limiter.check("client_a")
    with pytest.raises(HTTPException) as exc_info:
        limiter.check("client_a")
    assert exc_info.value.status_code == 429


def test_rate_limiter_tracks_clients_independently():
    limiter = RateLimiter(max_per_minute=1)
    limiter.check("client_a")
    limiter.check("client_b")  # different client, independent bucket, should not raise


def test_rate_limiter_window_expires_old_hits():
    limiter = RateLimiter(max_per_minute=1)
    limiter.check("client_a")
    # simulate time passing by manually rewinding the recorded hit timestamp
    limiter._hits["client_a"][0] -= 61
    limiter.check("client_a")  # should not raise -- old hit has expired out of the window


def test_api_key_dependency_disabled_when_no_key_configured():
    dependency = make_api_key_dependency(expected_key=None)
    # should not raise regardless of what header is (or isn't) provided
    asyncio.run(dependency(x_api_key=None))
    asyncio.run(dependency(x_api_key="anything"))


def test_api_key_dependency_rejects_missing_header_when_key_configured():
    dependency = make_api_key_dependency(expected_key="secret123")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(dependency(x_api_key=None))
    assert exc_info.value.status_code == 401


def test_api_key_dependency_rejects_wrong_key():
    dependency = make_api_key_dependency(expected_key="secret123")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(dependency(x_api_key="wrong"))
    assert exc_info.value.status_code == 401


def test_api_key_dependency_accepts_correct_key():
    dependency = make_api_key_dependency(expected_key="secret123")
    asyncio.run(dependency(x_api_key="secret123"))  # should not raise
