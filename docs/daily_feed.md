# Daily rates feed

The weekday workflow starts at 06:15 UTC. It reads the previous US business day's CFTC cumulative rates publication, selects five complete vanilla trades deterministically, and renders fictional parties. Two documents receive one seeded defect apiece. The feed is operational demonstration data in the `live_feed` collection. A defect counts as caught when the policy sends at least one affected field to review.

## Configure publication

- Set the `FRONTIER_BASE_URL` and `FRONTIER_API_KEY` Actions secrets for the pinned PCSS endpoint.
- Upload a trusted `feed-bootstrap` Actions artifact containing `index.html` built with 30 generated replay confirmations and the saved frontier `full_logistic.joblib`. Set `FEED_BOOTSTRAP_RUN` to its run ID.
- Configure GitHub Pages to serve the root of `gh-pages`. Use a narrowly scoped `FEED_PUBLISH_TOKEN` Actions secret if branch-based Pages builds must trigger from the automated commit. GitHub does not trigger Pages builds from a commit made by the built-in token.

Run the workflow manually from Actions with an optional UTC publication date. Both branches update in one atomic Git push. On a weekday with a complete run, `feed-data` receives the date file and history and `gh-pages` receives the offline site. Completed dates are no-ops on replay. The local command prepares the same files without pushing:

```sh
python scripts/daily_feed.py --feed-dir /tmp/feed-data/feed --site /tmp/gh-pages/index.html \
  --replay /path/to/bootstrap/index.html --risk-model /path/to/bootstrap/full_logistic.joblib
```

The source adapter accepts the currencies, conventions, rate indices and frequencies supported by the existing renderer and normalizers. Publication omits incomplete, capped-notional, package and nonstandard rows. The business-day convention and centers are fictional rendering assumptions. Provider failures, missing files and an insufficient daily budget skip publication. An input reservation and the reported usage are recorded under `cache/live_feed`; keep that cache across manual invocations.
