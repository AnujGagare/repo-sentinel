import time
from unittest.mock import patch

import pytest

from api.repo_jobs import RepoJobManager, MAX_INDEXABLE_FILES
from api.repo_validation import InvalidRepoUrlError


def _wait_for_terminal_status(manager, job_id, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get_job(job_id)
        if job.status in ("ready", "failed"):
            return job
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} did not reach a terminal status within {timeout}s")


def test_rejects_invalid_url_before_starting_a_job(tmp_path):
    manager = RepoJobManager(str(tmp_path / "repos"), on_ready=lambda path, job: {})
    with pytest.raises(InvalidRepoUrlError):
        manager.start_job("file:///etc/passwd")


def test_successful_job_reaches_ready_and_calls_on_ready(tmp_path):
    clone_root = tmp_path / "repos"
    on_ready_calls = []

    def fake_clone(url, path, depth):
        # simulate git.Repo.clone_from -- create the target dir with a
        # couple of files, as a real clone would
        from pathlib import Path
        Path(path).mkdir(parents=True)
        (Path(path) / "main.py").write_text("def hello(): pass")

    def on_ready(repo_path, job):
        on_ready_calls.append(repo_path)
        return {"chunks_indexed": 1}

    manager = RepoJobManager(str(clone_root), on_ready=on_ready)

    with patch("api.repo_jobs.git.Repo.clone_from", side_effect=fake_clone):
        job = manager.start_job("https://github.com/fastapi/fastapi")
        final = _wait_for_terminal_status(manager, job.job_id)

    assert final.status == "ready"
    assert final.result == {"chunks_indexed": 1}
    assert len(on_ready_calls) == 1


def test_failed_clone_marks_job_failed_and_cleans_up_directory(tmp_path):
    import git as git_module

    clone_root = tmp_path / "repos"
    manager = RepoJobManager(str(clone_root), on_ready=lambda path, job: {})

    def failing_clone(url, path, depth):
        raise git_module.GitCommandError("clone", "simulated clone failure")

    with patch("api.repo_jobs.git.Repo.clone_from", side_effect=failing_clone):
        job = manager.start_job("https://github.com/fastapi/fastapi")
        final = _wait_for_terminal_status(manager, job.job_id)

    assert final.status == "failed"
    assert final.error is not None
    assert not (clone_root / job.job_id).exists()  # cleaned up, not left behind


def test_oversized_repo_is_rejected_after_clone(tmp_path):
    clone_root = tmp_path / "repos"

    def fake_clone_many_files(url, path, depth):
        from pathlib import Path
        p = Path(path)
        p.mkdir(parents=True)
        for i in range(MAX_INDEXABLE_FILES + 10):
            (p / f"file_{i}.py").write_text("pass")

    manager = RepoJobManager(str(clone_root), on_ready=lambda path, job: {})

    with patch("api.repo_jobs.git.Repo.clone_from", side_effect=fake_clone_many_files):
        job = manager.start_job("https://github.com/fastapi/fastapi")
        final = _wait_for_terminal_status(manager, job.job_id)

    assert final.status == "failed"
    assert "exceeding the limit" in final.error


def test_on_ready_exception_marks_job_failed_without_leaking_internals(tmp_path):
    clone_root = tmp_path / "repos"

    def fake_clone(url, path, depth):
        from pathlib import Path
        Path(path).mkdir(parents=True)
        (Path(path) / "main.py").write_text("def hello(): pass")

    def broken_on_ready(repo_path, job):
        raise RuntimeError("some internal embedding model detail that shouldn't leak")

    manager = RepoJobManager(str(clone_root), on_ready=broken_on_ready)

    with patch("api.repo_jobs.git.Repo.clone_from", side_effect=fake_clone):
        job = manager.start_job("https://github.com/fastapi/fastapi")
        final = _wait_for_terminal_status(manager, job.job_id)

    assert final.status == "failed"
    assert "internal embedding model detail" not in final.error
    assert final.error == "internal error during indexing -- check server logs"


def test_empty_repo_with_no_indexable_files_fails_clearly(tmp_path):
    """Regression test: octocat/Hello-World (a real repo used for manual
    testing) has only a plain README with no extension -- zero .py/.md
    files. Previously this reached full_index() and crashed deep inside
    BM25 with a division-by-zero. Now caught early with a clear,
    actionable message before any indexing is attempted."""
    clone_root = tmp_path / "repos"

    def fake_clone_no_code(url, path, depth):
        from pathlib import Path
        p = Path(path)
        p.mkdir(parents=True)
        (p / "README").write_text("no extension, won't match .py or .md")

    on_ready_calls = []
    manager = RepoJobManager(str(clone_root), on_ready=lambda path, job: on_ready_calls.append(path))

    with patch("api.repo_jobs.git.Repo.clone_from", side_effect=fake_clone_no_code):
        job = manager.start_job("https://github.com/octocat/Hello-World")
        final = _wait_for_terminal_status(manager, job.job_id)

    assert final.status == "failed"
    assert "no Python" in final.error or "no indexable" in final.error.lower()
    assert on_ready_calls == []  # never even attempted indexing



    manager = RepoJobManager(str(tmp_path / "repos"), on_ready=lambda path, job: {})
    assert manager.get_job("nonexistent-job-id") is None
