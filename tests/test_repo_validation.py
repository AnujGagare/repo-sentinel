import pytest

from api.repo_validation import validate_repo_url, InvalidRepoUrlError


@pytest.mark.parametrize("url", [
    "https://github.com/fastapi/fastapi",
    "https://github.com/fastapi/fastapi.git",
    "https://gitlab.com/someorg/somerepo",
    "https://bitbucket.org/someorg/somerepo",
])
def test_accepts_valid_repo_urls(url):
    validate_repo_url(url)  # should not raise


@pytest.mark.parametrize("url,reason", [
    ("file:///etc/passwd", "non-http(s) scheme"),
    ("ssh://github.com/fastapi/fastapi", "non-http(s) scheme"),
    ("http://169.254.169.254/latest/meta-data/", "cloud metadata address"),
    ("http://localhost:8000/", "localhost"),
    ("http://127.0.0.1/", "loopback address"),
    ("https://evil.com/github.com/fastapi/fastapi", "wrong host entirely"),
    ("https://github.com.evil.com/fastapi/fastapi", "subdomain trick"),
    ("https://notgithub.com/fastapi/fastapi", "similar-looking wrong host"),
    ("https://github.com/", "no repo path"),
    ("https://github.com/fastapi/fastapi/tree/main", "deep link, not repo root"),
    ("https://github.com/fastapi/fastapi/../../../etc/passwd", "path traversal attempt"),
    ("not a url at all", "not a URL"),
    ("", "empty string"),
])
def test_rejects_dangerous_or_malformed_urls(url, reason):
    with pytest.raises(InvalidRepoUrlError):
        validate_repo_url(url)


def test_syntactically_valid_but_nonexistent_repo_passes_url_validation():
    """validate_repo_url only checks that a URL is SAFE to attempt cloning
    (right scheme, allowed host, not a private/internal address) -- it does
    NOT check that the repo actually exists, since that requires an actual
    network request (the clone attempt itself). A well-formed URL to a
    nonexistent repo correctly passes here and fails later, at clone time,
    with a distinct error path."""
    validate_repo_url("https://github.com/thisorgdoesnotexist12345/repo")  # should not raise
