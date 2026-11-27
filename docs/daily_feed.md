# Daily rates feed

The feed selects five complete vanilla
trades from the previous US business day's public DTCC CFTC rates file.
Documents and parties are fictional. Two documents get a seeded defect
apiece. A defect counts as caught when the review rule flags one of its
affected fields. Feed documents stay in the `live_feed` collection.

## 1. Export the feed bundle locally

Use the trusted frontier full-logistic model from the local Step 14 run and
the replay embedded in the current Pages site.

```sh
git fetch origin gh-pages
python scripts/export_feed_artifacts.py \
  --model results/models/frontier/full_logistic.joblib \
  --output results/feed_artifacts
```

The exporter reads `origin/gh-pages:index.html` by default. For a first
deployment with no `gh-pages` branch, build the local generated demo with
30 cached replay confirmations and pass `--replay results/demo/index.html`.
The bundle contains only `model.json` and `replay.json`. It contains numeric
model parameters and the fictional replay evidence already present in the
site. It rejects real-collection documents and omits raw caches, local paths,
and unneeded extraction metadata.

## 2. Push the bundle to feed-data

For an existing feed-data branch, from the source checkout:

```sh
git fetch origin feed-data
git worktree add -b local-feed-data ../tbv-feed-data origin/feed-data
mkdir -p ../tbv-feed-data/artifacts
cp results/feed_artifacts/{model,replay}.json ../tbv-feed-data/artifacts/
git -C ../tbv-feed-data add artifacts/model.json artifacts/replay.json
git -C ../tbv-feed-data commit -m "Install portable feed artifacts"
git -C ../tbv-feed-data push origin HEAD:feed-data
```

For a new branch, create an empty local repository first. Set its `origin` to
the same repository, then copy and commit the two files with the commands above.

```sh
git init --initial-branch=feed-data ../tbv-feed-data
git -C ../tbv-feed-data remote add origin "$(git remote get-url origin)"
```

## 3. Run from your laptop

In **Settings → Pages**, choose **Deploy from a branch → gh-pages → / (root)**.
Install the project's dependencies and authenticate Git for both branch pushes.
Set `PCSS_API_KEY` and `PCSS_BASE_URL` in your environment. Run from the source checkout:

```sh
scripts/publish_feed_local.sh
scripts/publish_feed_local.sh --date 2026-11-23
```

The date is an optional UTC publication date. The script uses temporary
worktrees for the two branches, reads the portable bundle from `feed-data`,
and pushes both refs atomically only after the full feed was written.
Weekends, published dates, missing files and PCSS failures exit successfully
with a skip message. Provider failures show the exception type, HTTP status
when available, and a short safe diagnosis. Keep `cache/live_feed` between
runs so successful model responses and the token budget can be reused.

### WSL weekday schedule

With the WSL distribution and cron service running, add the following to
`crontab -e`. Cron uses the distribution's local timezone for the 08:00 start.
Replace the example paths. Your private environment file should export
`PCSS_API_KEY` and `PCSS_BASE_URL` and be readable only by your account.
Use a noninteractive Git credential helper and the installed virtual environment.

```cron
SHELL=/bin/bash
0 8 * * 1-5 cd /home/you/Trust-but-Verify && mkdir -p cache && . /home/you/.config/trustswap/feed.env && PYTHON_BIN=/home/you/Trust-but-Verify/.venv/bin/python scripts/publish_feed_local.sh >>cache/feed-cron.log 2>&1
```

### Optional manual Actions run

The Actions workflow runs only through **Run workflow**. Set the
`PCSS_API_KEY` and `PCSS_BASE_URL` Actions secrets to try from that runner.
GitHub's built-in token can write branch refs with repository permission,
but its pushes do not trigger branch-based Pages builds. A manually triggered
Actions publication that should also update Pages needs a narrowly scoped
`FEED_PUBLISH_TOKEN` Actions secret. The laptop script uses your existing Git
authentication.
