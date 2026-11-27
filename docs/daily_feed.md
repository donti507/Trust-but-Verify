# Daily rates feed

The weekday workflow starts at 06:15 UTC. It selects five complete vanilla
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

## 3. Set secrets and dispatch

Add the `PCSS_API_KEY` and `PCSS_BASE_URL` secrets in the repository's
**Settings → Secrets and variables → Actions**. Under **Settings → Pages**,
select **GitHub Actions** as the build and deployment source. The workflow
publishes `gh-pages` and deploys the same HTML with the built-in token.

In **Actions → Daily synthetic rates feed**, select **Run workflow**.
The date is an optional UTC publication date. The workflow verifies the
JSON model hash, uses its portable probability calculation, and pushes the
feed and Pages branches in one atomic Git push. Missing artifacts, weekends,
missing source files, provider errors and budget limits leave both branches
unchanged. Re-running a published date is a no-op.

To prepare the same feed and page locally without pushing:

```sh
python scripts/daily_feed.py --feed-dir ../tbv-feed-data/feed \
  --site /tmp/tbv-feed-preview.html --artifacts ../tbv-feed-data/artifacts
```

The adapter accepts conventions and frequencies supported by the existing
renderer. The business-day convention and centers are fictional rendering
assumptions. Successful model responses are cached by input hash. Keep the
local `cache/live_feed` directory across manual runs for token accounting.
