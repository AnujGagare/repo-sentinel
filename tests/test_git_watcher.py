import subprocess

import pytest

from indexing.git_watcher import GitWatcher


def _init_repo(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)


def _commit_all(path, message):
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=path, check=True)


@pytest.fixture
def git_repo(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _init_repo(repo_dir)
    (repo_dir / "module_a.py").write_text("def a(): pass")
    (repo_dir / "module_b.py").write_text("def b(): pass")
    (repo_dir / "README.md").write_text("# hi")
    (repo_dir / "ignored.txt").write_text("not tracked by our extension filter")
    _commit_all(repo_dir, "initial commit")
    return repo_dir


def test_initial_index_reports_all_matching_files(git_repo, tmp_persist_dir):
    watcher = GitWatcher(str(git_repo), state_dir=tmp_persist_dir)
    changes = watcher.get_changes_since_last_index()
    files = set(changes.files_to_reindex())
    assert files == {"module_a.py", "module_b.py", "README.md"}
    assert "ignored.txt" not in files  # not a .py or .md file


def test_no_changes_after_marking_indexed(git_repo, tmp_persist_dir):
    watcher = GitWatcher(str(git_repo), state_dir=tmp_persist_dir)
    watcher.mark_indexed(watcher.current_commit())
    changes = watcher.get_changes_since_last_index()
    assert changes.is_empty


def test_detects_exactly_one_changed_file(git_repo, tmp_persist_dir):
    """The core incremental-indexing claim: after a commit touching one
    file, exactly that file (not the whole repo) should be reported."""
    watcher = GitWatcher(str(git_repo), state_dir=tmp_persist_dir)
    watcher.mark_indexed(watcher.current_commit())

    (git_repo / "module_a.py").write_text("def a(): return 'changed'")
    _commit_all(git_repo, "modify module_a")

    changes = watcher.get_changes_since_last_index()
    assert changes.files_to_reindex() == ["module_a.py"]
    assert changes.files_to_remove() == []


def test_detects_deleted_file(git_repo, tmp_persist_dir):
    watcher = GitWatcher(str(git_repo), state_dir=tmp_persist_dir)
    watcher.mark_indexed(watcher.current_commit())

    (git_repo / "module_b.py").unlink()
    _commit_all(git_repo, "delete module_b")

    changes = watcher.get_changes_since_last_index()
    assert "module_b.py" in changes.files_to_remove()
    assert "module_b.py" not in changes.files_to_reindex()


def test_detects_added_file(git_repo, tmp_persist_dir):
    watcher = GitWatcher(str(git_repo), state_dir=tmp_persist_dir)
    watcher.mark_indexed(watcher.current_commit())

    (git_repo / "module_c.py").write_text("def c(): pass")
    _commit_all(git_repo, "add module_c")

    changes = watcher.get_changes_since_last_index()
    assert "module_c.py" in changes.files_to_reindex()
