"""
Git-diff based incremental change detector.

Why this exists:
A naive "RAG over a repo" project re-embeds the entire codebase every time
you want fresh answers -- wasteful and slow as a repo grows. Production
systems instead track what actually changed and update only those chunks.

This module wraps GitPython to answer: "between commit A and commit B,
which files were added, modified, or deleted?" That result feeds directly
into the indexing pipeline, which will:
  - re-chunk + re-embed only ADDED/MODIFIED files
  - remove index entries for DELETED files
  - leave everything else untouched

We track the "last indexed commit" in a small state file so re-indexing
can always compute a correct diff, even if the watcher process restarts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import git


STATE_FILE_NAME = ".repo_sentinel_state.json"


@dataclass
class RepoChanges:
    added: list[str]
    modified: list[str]
    deleted: list[str]
    renamed: list[tuple[str, str]]  # (old_path, new_path)
    from_commit: str
    to_commit: str

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.modified or self.deleted or self.renamed)

    def files_to_reindex(self) -> list[str]:
        """Files whose chunks need to be re-embedded."""
        return self.added + self.modified + [new for _, new in self.renamed]

    def files_to_remove(self) -> list[str]:
        """Files whose chunks should be purged from the index."""
        return self.deleted + [old for old, _ in self.renamed]


class GitWatcher:
    def __init__(self, repo_path: str, state_dir: str | None = None):
        self.repo_path = Path(repo_path)
        self.repo = git.Repo(repo_path)
        self.state_path = Path(state_dir or repo_path) / STATE_FILE_NAME

    def current_commit(self) -> str:
        return self.repo.head.commit.hexsha

    def last_indexed_commit(self) -> str | None:
        if not self.state_path.exists():
            return None
        return json.loads(self.state_path.read_text()).get("last_indexed_commit")

    def mark_indexed(self, commit_sha: str) -> None:
        self.state_path.write_text(json.dumps({"last_indexed_commit": commit_sha}))

    def get_changes_since_last_index(self, extensions: tuple[str, ...] = (".py", ".md")) -> RepoChanges:
        """
        Compute file-level changes between the last indexed commit and HEAD.
        If no prior index exists, everything currently tracked counts as
        "added" (i.e. a full initial index).
        """
        head_sha = self.current_commit()
        last_sha = self.last_indexed_commit()

        if last_sha is None:
            all_files = [
                f for f in self._list_tracked_files(self.repo.head.commit)
                if f.endswith(extensions)
            ]
            return RepoChanges(
                added=all_files, modified=[], deleted=[], renamed=[],
                from_commit="(none)", to_commit=head_sha,
            )

        if last_sha == head_sha:
            return RepoChanges(added=[], modified=[], deleted=[], renamed=[],
                                from_commit=last_sha, to_commit=head_sha)

        old_commit = self.repo.commit(last_sha)
        new_commit = self.repo.commit(head_sha)
        diff_index = old_commit.diff(new_commit)

        added, modified, deleted, renamed = [], [], [], []
        for diff in diff_index:
            path = diff.b_path or diff.a_path
            if not path.endswith(extensions):
                continue
            if diff.new_file:
                added.append(diff.b_path)
            elif diff.deleted_file:
                deleted.append(diff.a_path)
            elif diff.renamed_file:
                renamed.append((diff.a_path, diff.b_path))
            else:
                modified.append(diff.b_path)

        return RepoChanges(
            added=added, modified=modified, deleted=deleted, renamed=renamed,
            from_commit=last_sha, to_commit=head_sha,
        )

    def _list_tracked_files(self, commit) -> list[str]:
        return [item.path for item in commit.tree.traverse() if item.type == "blob"]


if __name__ == "__main__":
    import sys
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    watcher = GitWatcher(repo_path, state_dir="/tmp")
    changes = watcher.get_changes_since_last_index()
    print(f"From {changes.from_commit} -> {changes.to_commit}")
    print(f"  to re-index: {len(changes.files_to_reindex())} files")
    print(f"  to remove:   {len(changes.files_to_remove())} files")
    if changes.files_to_reindex():
        print("  sample:", changes.files_to_reindex()[:5])
