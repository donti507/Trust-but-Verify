"""Feed publisher tests using local bare Git repositories and an offline runner."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from trustswap.feed.pipeline import BudgetedClient, FeedSkip

PUBLISHER = Path(__file__).resolve().parents[1] / "scripts" / "publish_feed_local.sh"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True,
                          capture_output=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    remote = tmp_path / "origin"
    remote.mkdir()
    git(remote, "init", "--bare")
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "remote", "add", "origin", str(remote))
    scripts = repo / "scripts"
    scripts.mkdir()
    shutil.copy2(PUBLISHER, scripts)
    (scripts / "daily_feed.py").write_text(
        """
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
for arg in ('date', 'feed-dir', 'site', 'artifacts'):
    parser.add_argument('--' + arg, required=True)
args = parser.parse_args()
if args.date == '2026-11-24':
    print("Skipped: DTCC file unavailable")
elif args.date == '2026-11-25':
    print("Skipped: PCSS request failed (APIStatusError, HTTP 403): Forbidden")
else:
    feed = Path(args.feed_dir)
    feed.mkdir(exist_ok=True)
    (feed / (args.date + '.json')).write_text('{"trades": 5}')
    (feed / 'history.json').write_text('[]')
    Path(args.site).write_text('<h1>Fictional confirmations</h1>')
    print('Published 5 fictional confirmations')
""",
        encoding="utf-8",
    )
    git(repo, "add", "scripts")
    git(repo, "commit", "-m", "Fixture source")
    git(repo, "switch", "--orphan", "feed-data")
    artifacts = repo / "artifacts"
    artifacts.mkdir()
    for name in ("model.json", "replay.json"):
        (artifacts / name).write_text("{}")
    git(repo, "add", "artifacts")
    git(repo, "commit", "-m", "Fixture bundle")
    git(repo, "push", "origin", "feed-data")
    git(repo, "switch", "main")
    return repo, remote


def run(repo, day, monkeypatch):
    monkeypatch.setenv("PYTHON_BIN", sys.executable)
    return subprocess.run(["bash", "scripts/publish_feed_local.sh", "--date", day], cwd=repo,
                          capture_output=True, text=True, check=True).stdout


def refs(remote):
    return git(remote, "for-each-ref", "--format=%(refname):%(objectname)", "refs/heads")


def test_skip_weekend_no_file_or_pcss_failure(checkout, monkeypatch):
    repo, remote = checkout
    before = refs(remote)
    for day, message in (("2026-11-22", "weekend"), ("2026-11-24", "DTCC file unavailable"),
                         ("2026-11-25", "HTTP 403")):
        assert message in run(repo, day, monkeypatch)
        assert refs(remote) == before
        assert git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1
    assert "gh-pages" not in before


def test_publish_creates_both_refs_and_rerun_changes_nothing(checkout, monkeypatch):
    repo, remote = checkout
    assert "Published 5" in run(repo, "2026-11-23", monkeypatch)
    first = refs(remote)
    assert git(remote, "show", "feed-data:feed/2026-11-23.json") == '{"trades": 5}'
    assert "Fictional confirmations" in git(remote, "show", "gh-pages:index.html")
    assert "already published" in run(repo, "2026-11-23", monkeypatch)
    assert refs(remote) == first


def test_provider_diagnostics_never_echo_credentials(tmp_path, monkeypatch):
    key = "fictional-test-token-012345"
    monkeypatch.setenv("PCSS_API_KEY", key)

    class APIStatusError(Exception):
        status_code = 403

    class Provider:
        params = {"max_tokens": 20}

        def complete(self, *args, **kwargs):
            raise APIStatusError(f"Authorization: Bearer {key}; password=unexpected-secret")

    with pytest.raises(FeedSkip) as failure:
        BudgetedClient(Provider(), tmp_path / "ledger", 60_000).complete([], temperature=0, seed=0)
    message = str(failure.value)
    assert "APIStatusError, HTTP 403): Forbidden" in message
    assert all(term not in message for term in (key, "Authorization", "Bearer", "unexpected-secret"))
