"""
Validates user-supplied git repo URLs before cloning them.

Why this needs to be strict: this is the one place in the app where a
completely untrusted string (typed by any visitor) turns into a network
request the SERVER makes on the visitor's behalf. Without validation this
is a classic SSRF vector -- e.g. a "repo URL" of
`http://169.254.169.254/latest/meta-data/` would make the server itself
fetch cloud metadata (credentials, instance info) on a request from
someone who was never supposed to have that access. This applies whether
this is deployed on Render, AWS, GCP, or anywhere else with a metadata
endpoint.

Rules, in order of how much damage skipping them could cause:
  1. Must be http(s), not file://, ssh://, git://, or anything else that
     could reach local files or bypass network-level protections.
  2. Must not resolve to a private/internal/link-local address --
     blocks localhost, RFC1918 private ranges, and the cloud metadata
     address specifically.
  3. Must look like an actual git hosting URL (github.com, gitlab.com,
     bitbucket.org, or a self-hosted-looking path) rather than an
     arbitrary URL that happens to be syntactically valid.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = {"http", "https"}

ALLOWED_HOST_SUFFIXES = (
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "codeberg.org",
    "sr.ht",
)

_REPO_PATH_RE = re.compile(r"^/[\w.\-]+/[\w.\-]+(?:\.git)?/?$")


class InvalidRepoUrlError(ValueError):
    pass


def _is_private_or_reserved(host: str) -> bool:
    """Resolves the hostname and checks whether ANY of its addresses are
    private/loopback/link-local -- checking the hostname string alone
    isn't enough, since DNS rebinding could point an allowed-looking
    hostname at a private address."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise InvalidRepoUrlError(f"could not resolve host: {host}")

    for info in infos:
        ip_str = info[4][0]
        ip = ipaddress.ip_address(ip_str)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return True
    return False


def validate_repo_url(url: str) -> str:
    """Returns the normalized URL if valid, raises InvalidRepoUrlError otherwise."""
    url = url.strip()
    if not url:
        raise InvalidRepoUrlError("repo URL is empty")

    parsed = urlparse(url)

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise InvalidRepoUrlError(
            f"URL scheme must be http or https, got {parsed.scheme!r}"
        )

    if not parsed.hostname:
        raise InvalidRepoUrlError("URL has no host")

    host = parsed.hostname.lower()

    if not any(host == suffix or host.endswith("." + suffix) for suffix in ALLOWED_HOST_SUFFIXES):
        raise InvalidRepoUrlError(
            f"host {host!r} is not an allowed git hosting provider "
            f"(allowed: {', '.join(ALLOWED_HOST_SUFFIXES)})"
        )

    if not _REPO_PATH_RE.match(parsed.path):
        raise InvalidRepoUrlError(
            f"path {parsed.path!r} doesn't look like a repo path (expected /owner/repo)"
        )

    if _is_private_or_reserved(host):
        raise InvalidRepoUrlError(f"host {host!r} resolves to a private/internal address, refusing to clone")

    return f"{parsed.scheme}://{host}{parsed.path}"


if __name__ == "__main__":
    valid_examples = [
        "https://github.com/fastapi/fastapi",
        "https://github.com/fastapi/fastapi.git",
        "https://gitlab.com/someorg/somerepo",
    ]
    invalid_examples = [
        "file:///etc/passwd",
        "http://169.254.169.254/latest/meta-data/",
        "http://localhost:8000/",
        "http://127.0.0.1/",
        "https://evil.com/github.com/fastapi/fastapi",
        "ssh://github.com/fastapi/fastapi",
        "https://github.com/",
        "not a url at all",
    ]

    print("Valid URLs (should all pass):")
    for u in valid_examples:
        try:
            print(f"  OK: {validate_repo_url(u)}")
        except InvalidRepoUrlError as e:
            print(f"  UNEXPECTED REJECTION: {u} -> {e}")

    print("\nInvalid URLs (should all be rejected):")
    for u in invalid_examples:
        try:
            result = validate_repo_url(u)
            print(f"  UNEXPECTED ACCEPTANCE: {u} -> {result}")
        except InvalidRepoUrlError as e:
            print(f"  correctly rejected: {u} ({e})")
