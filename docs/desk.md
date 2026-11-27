# Local Synthetic demo desk

Install the extra with `python -m pip install -e '.[desk]'`. Create local users with
`python -m trustswap.desk create-user USERNAME ROLE`. The role is `analyst`,
`supervisor`, or `auditor`; the CLI prompts for a bcrypt password. Set a random
session secret of at least 32 characters in `DESK_SECRET_KEY`, then run
`python -m trustswap.desk` and open `http://localhost:8000`. The app binds to
loopback. `DESK_DB` or `--db` selects the SQLite database.

Analysts accept fields or submit proposed booking changes with a note.
Supervisors approve or reject proposals with a reason and can adjust the
policy threshold. Every field must be accepted or approved before a trade is
affirmed. Auditors can inspect the read-only event log. The desk collection and
its SQLite state are isolated from the research pipeline.

## Seed and import

Load the generated replay page and optional dated feed publications. This adds
new confirmations and creates matching fictional bookings, with seeded
differences on about one in five new trades. Existing trades and review states
are kept on subsequent runs.

```sh
python -m trustswap.desk seed --replay results/demo/index.html --feed-dir /path/to/feed-data/feed
python scripts/import_bookings.py /path/to/fictional-bookings.csv
```

Supervisors can also import a CSV from Overview. Use an optional `trade_id`
column and field-path headers such as `trade_date`,
`floating_leg.notional_amount`, `floating_leg.notional_currency` and
`termination_date`. Blank values are missing. Rates use decimals unless the
value explicitly includes a percent or basis-point unit. Without a trade ID,
a complete normalized date, notional, currency and maturity key must match
uniquely. The importer reports rejected lines with reasons, including attempts to
replace a booking after review starts. Valid rows are applied together. The CLI
returns exit code 1 when rows are rejected. Use only fictional party names and sample addresses.

Auditors and supervisors download CSV or JSON from Audit log. Exports preserve event order and
the database blocks updates and deletes on audit records.

Without arguments, `seed` reads `results/demo/index.html` and `feed/`. If the feed directory is absent, it uses feed days embedded in the replay page.
