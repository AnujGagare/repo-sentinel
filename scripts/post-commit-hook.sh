#!/bin/bash
# Post-commit hook: calls the running Repo Sentinel API's /reindex endpoint
# every time a commit lands, so the index (and therefore every subsequent
# answer) reflects the new code within seconds instead of on a fixed poll
# interval.
#
# Install by copying this file to <repo>/.git/hooks/post-commit and making
# it executable:
#   cp scripts/post-commit-hook.sh <target_repo>/.git/hooks/post-commit
#   chmod +x <target_repo>/.git/hooks/post-commit

API_BASE="${REPO_SENTINEL_API_BASE:-http://localhost:8000}"

echo "[repo-sentinel] commit detected, triggering re-index..."
curl -s -X POST "${API_BASE}/reindex" | python3 -m json.tool
