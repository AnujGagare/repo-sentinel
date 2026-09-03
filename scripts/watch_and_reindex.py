"""
Polling-based alternative to the git post-commit hook.

Some repos (e.g. a clone you don't control, or a demo environment) aren't
convenient to install a git hook into. This script polls the repo's HEAD
commit every N seconds and triggers /reindex when it changes -- same end
result as the hook, just on a delay instead of instantaneous.

Usage:
    python scripts/watch_and_reindex.py --repo ./data/fastapi --interval 10
"""

from __future__ import annotations

import argparse
import subprocess
import time

import requests


def get_head_commit(repo_path: str) -> str:
    result = subprocess.run(
        ["git", "-C", repo_path, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Path to the watched git repo")
    parser.add_argument("--interval", type=int, default=10, help="Poll interval in seconds")
    parser.add_argument("--api-base", default="http://localhost:8000")
    args = parser.parse_args()

    last_seen = get_head_commit(args.repo)
    print(f"[watcher] starting, current HEAD = {last_seen[:8]}, polling every {args.interval}s")

    while True:
        time.sleep(args.interval)
        current = get_head_commit(args.repo)
        if current != last_seen:
            print(f"[watcher] change detected: {last_seen[:8]} -> {current[:8]}, triggering reindex")
            try:
                resp = requests.post(f"{args.api_base}/reindex", timeout=120)
                print(f"[watcher] reindex result: {resp.json()}")
            except requests.RequestException as e:
                print(f"[watcher] reindex call failed: {e}")
            last_seen = current


if __name__ == "__main__":
    main()
