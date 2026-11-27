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

On a trade's Review page, select a highlight in the confirmation to jump to the
field. Red callouts mark fields flagged by the risk policy or the booking
comparison. Field cards show the original wording and its normalized form when
it differs. Progress counts accepted and approved fields. Analysts can use
**Accept all matched low-risk fields** to clear open or rejected fields that
match the booking and pass the current risk policy. The desk audits each
acceptance individually. To recover highlights for an older seed, rerun the seed
command against the same replay and feed files.

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

## Upload and export confirmations

Set `DESK_UPLOAD_DIR` to an absolute path outside any Git checkout. The desk saves
uploads there by content hash with a 10 MB file limit. The Overview upload form is
available to analysts and supervisors. Use fictional documents and the fictional
party names already used by the demo.

FIN MT360/MT361 and FpML swaps are parsed locally. The desk supports two-stream
fixed/floating and floating/floating terms with constant notionals. It reads FIN
party and notional fields and infers the payment interval from regular dated
schedules. Exported FIN files contain the desk confirmation terms in tagged text.
FpML imports and exports cover the same terms in two swap streams. Additional
settlement instructions and complex schedules are outside the desk profile. These
exports are for local interchange and review. [SWIFT message reference](https://www2.swift.com/knowledgecentre/rest/v1/publications/us3mb_20230720/2.0/us3mb_20230720.pdf?logDownload=true) · [FpML confirmation examples](https://www.fpml.org/spec/fpml-5-12-4-rec-1/html/confirmation/fpml-5-12-examples.html)

A structured file can use its embedded trade ID, or you can enter a booking trade
ID. The desk also tries the existing normalized booking key. When no booking
matches, import a booking CSV for the new trade before clearing discrepancies.
The Review page offers FIN and FpML downloads after affirmation.

PDF, PNG, JPEG and TXT are saved locally with external extraction disabled by
default. To process a fictional sample with the pinned extraction and scoring
pipeline, set `DESK_ALLOW_EXTERNAL_LLM=true`, `DESK_RISK_MODEL` to the verified
full-logistic JSON artifact described in `docs/daily_feed.md`, and
`PCSS_BASE_URL` and `PCSS_API_KEY` in the desk process environment. Restart the
desk and upload the sample. The desk reads the PDF text layer first and uses local
Tesseract for scans. Install Tesseract on the host for OCR. External extraction
sends extracted text to the configured provider and caches completions by input
hash under the private upload directory.

## Exceptions and response deadlines

The exception queue tracks trades with open review flags or booking differences.
The case priority is the largest severity weight times field risk on the trade.
Supervisors assign cases to analysts. Analysts see assigned and unassigned cases
from My queue and can filter overdue cases. Overview shows the SLA buckets and
average time from opening to resolution.

`configs/desk.yaml` sets deadlines in UTC. Critical cases have four elapsed hours,
major cases have one business day on the configured QuantLib calendar, and minor
cases have two elapsed days. The age turns amber halfway to the deadline and red
at the deadline.

An assigned analyst or supervisor can open a counterparty draft or download it
as an `.eml` file with fictional addresses. Drafting records an audit event and
sets the case to Waiting for counterparty. Record the reply to return the case to
Assigned. Pending corrections route to Pending approval. Clearing the outstanding
issues resolves the case.

## Trade history and lifecycle

Open an affirmed trade and select a lifecycle event. An analyst may enter an
amended field manually or attach a fictional confirmation. Attachments use the
desk upload pipeline and show explicitly revised fields against the current
booked version. A partial termination enters the remaining amount for equal
same-currency notionals. For a novation, select the outgoing party and use a
fictional replacement such as Synthetic Party C.

The supervisor sees the booked and proposed values in Approvals. Approval
creates a new immutable version in History. A rejection keeps the old version
and records the reason. A full termination locks the trade after approval.

## FX forwards, cross-currency swaps and CDS

Run `python -m trustswap.desk seed-products` for 10 fictional confirmations of
each additional product. Use Overview to open them or upload a labelled UTF-8
TXT confirmation with a `Product: FX forward`, `Product: Cross-currency swap`
or `Product: Credit default swap` line. Label each stated field as in the
seeded confirmations. The desk reads those terms locally. With external
extraction enabled, the desk can first classify unlabelled text and then
extract terms using the product prompt. Both calls are cached by input hash.
Scanned documents use the upload workflow above.

These products display **Uncalibrated**. An invalid term, a missing critical
term, a failed product check or a booking difference sends the field for
review. FX uses the configured quote-per-base rate, with a tolerance of 20%.
Cross-currency swaps require two distinct currencies and explicit initial
and final exchange flags. CDS uses the configured fictional reference entities,
spread and upfront terms, and QuantLib twentieth-IMM maturity dates. Reference
rates and entities are in `configs/desk.yaml`.

Booking CSV headers use product prefixes such as `fx_forward.forward_rate`,
`xccy_swap.exchange_initial` and `cds.reference_entity`. A product booking can
use its trade ID or that product's normalized booking key. The desk compares
these products in their own fields. FIN, FpML and lifecycle operations use the
interest-rate-swap profile above.
