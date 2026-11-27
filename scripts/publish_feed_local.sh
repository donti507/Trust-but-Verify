#!/usr/bin/env bash
# Build the feed in temporary detached worktrees and advance both refs together.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
python_bin="${PYTHON_BIN:-python3}"

if [[ $# -gt 0 && ( $# -ne 2 || "$1" != --date ) ]]; then
    echo "Usage: $0 [--date YYYY-MM-DD]" >&2
    exit 2
fi
publication="${2:-$(date -u +%F)}"
if ! weekday="$("$python_bin" - "$publication" <<'PY'
from datetime import date
import sys

value = date.fromisoformat(sys.argv[1])
if value.isoformat() != sys.argv[1]:
    raise ValueError("Use YYYY-MM-DD")
print(value.weekday())
PY
)"; then
    echo "Invalid publication date. Use YYYY-MM-DD." >&2
    exit 2
fi
if (( weekday >= 5 )); then
    echo "Skipped: weekend"
    exit 0
fi

scratch="$(mktemp -d "${TMPDIR:-/tmp}/trustswap-feed.XXXXXXXX")"
data="$scratch/feed-data"
web="$scratch/gh-pages"
cleanup() {
    git worktree remove --force "$data" 2>/dev/null || true
    git worktree remove --force "$web" 2>/dev/null || true
    rm -rf -- "$scratch"
}
trap cleanup EXIT

branch_exists() {
    local status=0
    git ls-remote --exit-code --heads origin "refs/heads/$1" >/dev/null || status=$?
    if (( status != 0 && status != 2 )); then
        exit "$status"
    fi
    return "$status"
}
checkout_branch() {
    git fetch --no-tags origin "refs/heads/$1"
    git worktree add --detach "$2" FETCH_HEAD
}

if ! branch_exists feed-data; then
    echo "Skipped: feed-data branch and artifact bundle are missing"
    exit 0
fi
checkout_branch feed-data "$data"
day_file="feed/$publication.json"
if [[ -f "$data/$day_file" ]]; then
    echo "Skipped: date already published"
    exit 0
fi
if [[ ! -f "$data/artifacts/model.json" || ! -f "$data/artifacts/replay.json" ]]; then
    echo "Skipped: feed artifact bundle is missing"
    exit 0
fi

new_pages=false
if branch_exists gh-pages; then
    checkout_branch gh-pages "$web"
else
    git init -q "$web"
    new_pages=true
fi

"$python_bin" scripts/daily_feed.py --date "$publication" --feed-dir "$data/feed" \
    --site "$web/index.html" --artifacts "$data/artifacts"
if [[ ! -f "$data/$day_file" ]]; then
    exit 0  # The feed command already logged the reason.
fi
git -C "$data" add -- "$day_file" feed/history.json
git -C "$data" commit -m "Publish rates feed for $publication"
git -C "$web" add -- index.html
if ! git -C "$web" diff --cached --quiet; then
    git -C "$web" -c user.name="$(git var GIT_AUTHOR_IDENT | sed 's/ <.*//')" \
        -c user.email="$(git var GIT_AUTHOR_IDENT | sed 's/.*<\([^>]*\)>.*/\1/')" \
        commit -m "Refresh dashboard for $publication"
fi
data_sha="$(git -C "$data" rev-parse HEAD)"
web_sha="$(git -C "$web" rev-parse HEAD)"
if "$new_pages"; then
    git fetch "$web" HEAD
fi
git push --atomic origin "$data_sha:refs/heads/feed-data" "$web_sha:refs/heads/gh-pages"
