"""
Manages the lifecycle of "index a new repo" requests as background jobs.

Why this is a job/polling pattern, not a normal request/response: cloning
+ chunking + embedding + auto-summarization for a new repo can take
anywhere from under a minute to 20+ minutes (observed directly during this
project's own development, dominated by embedding-model cold start and
per-function LLM summary calls). A single HTTP request blocking that long
would hit client/proxy timeouts and tie up a worker the whole time. So:

    POST /index_repo          -> returns a job_id immediately
    GET  /index_repo/{job_id} -> poll this for status until "ready" or "failed"

Jobs run in a background thread (FastAPI's BackgroundTasks), not inside
the request handler. Job state lives in memory (a dict) -- acceptable for
this project's actual single-process deployment shape; would need to move
to shared storage (Redis, a DB) if ever run multi-instance, same caveat as
the rate limiter (see api/security.py).

Only ONE repo is "active" (chat-answerable) at a time in this design --
starting a new index job replaces whatever was previously active once it
completes. True concurrent multi-repo support is a materially bigger
feature (per-repo storage isolation, tenancy, cleanup policies) --
deliberately out of scope here, see README.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import git

from api.repo_validation import validate_repo_url, InvalidRepoUrlError

logger = logging.getLogger(__name__)

# Bounds how large a repo we'll commit to fully indexing -- without this, a
# single request could hand the server an enormous repo and tie up CPU/LLM
# calls for a very long time. Chosen generously enough for most real
# projects while still being a real, enforced limit rather than an
# unbounded promise.
MAX_INDEXABLE_FILES = 800


@dataclass
class IndexJob:
    job_id: str
    repo_url: str
    status: str = "queued"  # queued -> cloning -> indexing -> ready | failed
    error: str | None = None
    detail: str | None = None
    created_at: float = field(default_factory=time.time)
    result: dict | None = None


class RepoJobManager:
    def __init__(self, clone_root: str, on_ready):
        """
        clone_root: directory under which each job's repo is cloned
                    (data/repos/<job_id>/)
        on_ready: callback(repo_path: str, job: IndexJob) -> dict, called
                  once cloning succeeds -- this is where the caller (main.py)
                  actually runs IndexingPipeline and swaps the active repo.
                  Kept as a callback rather than importing IndexingPipeline
                  directly here, to keep this module focused on job/clone
                  lifecycle, not indexing logic.
        """
        self.clone_root = Path(clone_root)
        self.clone_root.mkdir(parents=True, exist_ok=True)
        self.on_ready = on_ready
        self._jobs: dict[str, IndexJob] = {}
        self._lock = threading.Lock()

    def start_job(self, repo_url: str) -> IndexJob:
        normalized_url = validate_repo_url(repo_url)  # raises InvalidRepoUrlError if unsafe

        job_id = uuid.uuid4().hex[:12]
        job = IndexJob(job_id=job_id, repo_url=normalized_url)
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> IndexJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _set_status(self, job: IndexJob, status: str, detail: str | None = None):
        job.status = status
        job.detail = detail
        logger.info("[job %s] %s%s", job.job_id, status, f" -- {detail}" if detail else "")

    def _run_job(self, job: IndexJob):
        repo_dir = self.clone_root / job.job_id
        try:
            self._set_status(job, "cloning", job.repo_url)
            git.Repo.clone_from(job.repo_url, str(repo_dir), depth=1)

            file_count = sum(1 for _ in repo_dir.rglob("*.py")) + sum(1 for _ in repo_dir.rglob("*.md"))
            if file_count == 0:
                raise ValueError(
                    "this repo has no Python (.py) or Markdown (.md) files to index -- "
                    "try a repo with actual source code"
                )
            if file_count > MAX_INDEXABLE_FILES:
                raise ValueError(
                    f"repo has {file_count} indexable files, exceeding the limit of "
                    f"{MAX_INDEXABLE_FILES} for this demo deployment"
                )

            self._set_status(job, "indexing", f"{file_count} files to process")
            result = self.on_ready(str(repo_dir), job)

            job.result = result
            self._set_status(job, "ready")

        except (InvalidRepoUrlError, git.GitCommandError, ValueError) as e:
            job.error = str(e)
            self._set_status(job, "failed", str(e))
            shutil.rmtree(repo_dir, ignore_errors=True)
        except Exception as e:
            # Anything unexpected -- log the full thing server-side but
            # don't leak internals to the job status response.
            logger.exception("[job %s] unexpected failure", job.job_id)
            job.error = "internal error during indexing -- check server logs"
            self._set_status(job, "failed")
            shutil.rmtree(repo_dir, ignore_errors=True)
