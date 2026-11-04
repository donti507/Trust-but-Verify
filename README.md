# Trust but Verify

Domain-grounded verification and cost-aware abstention for AI-extracted interest rate swap terms.
An LLM proposes field values, a deterministic QuantLib/CDM verifier checks them, and a calibrated,
cost-aware selector decides which fields a human must review, with a threshold chosen by conformal
risk control.

- Plan and step-by-step instructions: Technical Runbook (see AGENTS.md for the link)
- Status: Phase 0 (setup)
- Licence: to be decided after checking the participating institutions' IP policies

## Quick start

On Ubuntu/Debian, install the system libraries, fonts and OCR engine before
installing Python dependencies (CI uses the same packages):

```bash
sudo apt-get update
sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 fonts-dejavu-core tesseract-ocr
```

Synthetic PDF templates use DejaVu Serif with ligatures disabled to keep extracted
text and field spans consistent across development and CI machines.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest
python run.py --help
```
Heavy dependencies (QuantLib, vLLM, OCR) are in `requirements.txt` and `requirements-gpu.txt`.

## How to run: EDGAR scope (Step 3A)

From the repository root, set a descriptive SEC User-Agent with your real name or
organisation and contact email, then run the standard-library-only script:

```bash
export SEC_USER_AGENT="Your Name contact@example.com"
python scripts/edgar_scope.py
# Optionally download the first 10 candidates:
python scripts/edgar_scope.py --download 10
```

The script searches four quoted swap phrases from 2001 through today's date in
8-K, 10-Q and 10-K filings and their exhibits. It writes deduplicated
`data/real/candidates.csv` and prints unique-document counts by phrase, year and
root filing form. `matched_phrases` is a semicolon-separated union of phrase
matches; `exhibit_file_name` identifies the matched document (including primary
filing documents when applicable). Counts per phrase can overlap.

Requests share a four-per-second limit, including downloads and retries;
429/503 responses trigger exponential backoff and respect Retry-After. Search
pagination splits date ranges above EDGAR's result-window cap. If even one day
exceeds that cap, or a response is partial or changes while paging, the script
fails rather than writing a misleading partial candidate CSV. Rerun the search
after transient failures. No User-Agent means no requests or output writes.

Downloads go to `data/raw/edgar/`; `manifest.jsonl` records URLs, paths and SHA-256
hashes of the original bytes. Existing downloads are reused only after hash
verification; unmanifested or altered files cause an error and are never
replaced. `--output PATH` and `--raw-dir PATH` override the default destinations.
All data outputs remain Git-ignored. The script does not assign splits, label
real documents, or inspect `real_test`. Tests use saved synthetic JSON responses
and a mocked HTTP layer; this development task does not run the live search.

## How to run: EDGAR screen (Step 3B)

After Step 3A, with the project installed and `SEC_USER_AGENT` still set:

```bash
python scripts/edgar_screen.py --seed 20261026
```

This takes 75 unique candidates from each era (2001–2005, 2006–2008,
2009–2021, 2022–today), for 300 total. It refuses undersized strata before
network access. Downloads reuse Step 3A's four-per-second client and verified,
non-overwriting `data/raw/edgar/manifest.jsonl`. Step 2 writes unassigned real
records, text and offset maps; unchanged files can be reused on a retry.
Failed candidates are replaced by seeded, unused candidates from the same era,
maintaining 75 successful documents per era and 300 total when the pool permits.
If an era is exhausted, the script still screens all successes and saves
`screen.csv`, then reports the shortfall without duplicating rows or borrowing
from another era.

`data/real/screen.csv` records term flags, page count/basis, score, era and a
heuristic triage label. All four term groups plus fewer than 40 pages means
`likely_confirmation`; at least two groups means `maybe`; otherwise `unlikely`.
PDF page counts come directly from PyMuPDF. HTML/TXT counts are
`ceil(len(text) / 3000)`, marked `estimated_chars`. Screening never renders
HTML/TXT with WeasyPrint; these counts are character-based triage estimates.

`data/real/manual_check.csv` has 60 seeded random rows, 20 per label, with an
empty `is_confirmation` column for a human. If a label has fewer than 20 rows,
the script saves/prints the screen results and reports the shortage without
writing a partial manual sample. It never duplicates examples or invents human
labels. Existing manual-check files are preserved while screening resumes; move one
aside only if you want a new manual sample. Screening does not write gold labels or assign train/test splits,
and refuses to reuse a document that already has a split.

Use `--data-root`, `--candidates` or `--raw-dir` for alternate local paths.
Tests use only fictional saved fixtures and mocked HTTP, not SEC downloads.

## How to run: offline manual review

```bash
python scripts/review_helper.py
# Optional: custom CSV, plain highlighting, or revisit saved skips
python scripts/review_helper.py --csv data/real/manual_check.csv --no-color --revisit-skipped
```

The helper reads local `text_path` files only: it never downloads or opens URLs.
It shows the URL, era, screen label, page count/basis, the first 60 lines, and all
later lines containing the requested swap terms, with highlights and original
line numbers. Non-TTY/`--no-color` output uses `[[phrase]]` highlighting.

Enter `y` (confirmation), `n` (not), or `s` (skip), optionally with a product key
on the **same line**, then press Enter: `y 1`, `n 4`, or `s`. Product keys are
`1` irs_fixed_float, `2` basis, `3` cap_floor, `4` other. Every answer is saved
atomically to the same CSV, preserving other columns and answers. EOF/Ctrl+C
stops safely; rerunning skips completed answers and saved skips. Use
`--revisit-skipped` to return to the latter. Missing text files stay unanswered.
The final summary shows yes/no counts per screen label and era, plus skip and
unanswered totals. These human triage decisions do not create trade gold labels.

## How to run: pilot annotation (Step 4)

```bash
python scripts/select_pilot.py
python scripts/annotate.py --annotator A1
# A second human works independently, without viewing A1:
python scripts/annotate.py --annotator A2
python scripts/agreement.py
```

Read [the annotation guide](docs/annotation_guide.md) first. Pilot selection targets
6 documents from each pre-2022 era and up to 2 from 2022+, preferring reviewed
fixed/float confirmations within each era. Unfilled slots are redistributed
without exceeding the recent-era cap. Before adding an unchecked candidate,
the selector displays its first 40 saved-text lines and every keyword match,
then requires y/n acceptance as a suitable fixed/float confirmation. Accepted
rows record that human confirmation. Existing pilot files are protected; use
`--top-up` to replace rejected rows or `--output` for a different pilot.

The annotation CLI shows keyword contexts, asks for exact saved-text quotes,
lets you choose among repeated occurrences, and confirms normalised values before
saving. `show` displays the whole text; `absent`, `ref`, `amb` and `back` control
status/navigation. At any prompt, `reject <reason>` marks the current pilot
document unsuitable and skips its remaining fields. Type `\n` for saved line breaks, or `text:<literal>` to quote
reserved command words. A bare rate asks for a document-supported unit; it never
uses a verifier or market default. `ref` leaves the span null and allows an
unresolved null value. Enter the document-supported flags at the end.

Each field is saved atomically to `data/real/annotations/A1.jsonl` (or A2).
EOF/Ctrl+C stops safely; rerun the same command to resume. Changed source text
or concurrent annotation edits are rejected. Unanswered fields remain unannotated,
not absent. A1 never reads A2's file. No real gold is published automatically.

Agreement writes per-field normalised value agreement and status Cohen's kappa
to `data/real/agreement.csv`, plus every value/status/span/flag or missing-answer
disagreement to `data/real/adjudication.csv` for the domain lead. Unpaired fields
are excluded from paired metrics. Kappa is blank when undefined (no pairs or
constant identical status marginals). Different source-text versions are reported
and excluded from comparisons. Existing reports are protected; preserve lead
edits before requesting regeneration with `--overwrite`. All tools are offline.

## Replacing pilot covers and rejected documents

During annotation, enter `reject cover 8-K only` (or another specific reason).
This saves `rejected_reason` in `pilot20.csv`, skips the document for both
annotators, and retains existing partial annotations. Then choose one workflow:

```bash
# Fill rejected slots from other candidates, retaining all active pilot rows:
python scripts/select_pilot.py --top-up

# Or resolve a still-present cover to its actual referenced exhibit:
# SEC_USER_AGENT must identify you and your contact email (see EDGAR setup).
python scripts/resolve_exhibit.py <cover_doc_id>
```

Top-up excludes current and previously rejected IDs/URLs, keeps the recent-era
cap, previews every unchecked replacement, and saves only a complete 20-document
pilot. EOF/declined choices that exhaust the pool leave the prior pilot intact.
Removed cover/rejection metadata is retained in `data/real/pilot_rejections.csv`.

The exhibit resolver checks the local cover's confirmation/exhibit citation,
reads the SEC accession directory (and filing document index if needed), and
matches descriptions, filenames or the cited exhibit number. Multiple matches
require a choice. It uses the shared paced SEC client and SHA-256 manifest,
ingests via the bounded no-OCR screening pipeline, and requires a local-text
preview acceptance before replacing the cover in the same pilot position.
Decline leaves the pilot unchanged and retains cached downloads/text. The new
row records `replaces_doc_id` and `cover_document_url`. If you already topped up
the cover's slot, resolve only a cover still present in the current pilot.

Agreement reports default to non-rejected IDs in the current `pilot20.csv` when
present; retained retired-cover annotations do not enter its metrics. All tests
mock SEC responses and use fictional local fixtures; no live requests are made
while developing these tools.

## Layout
```
configs/   experiment configs (YAML)
data/      raw, real, synthetic, adversarial, ref  (never committed)
cache/     every model output, keyed by input hash  (never committed)
prereg/    frozen hypotheses, thresholds, hashes
results/   run outputs  (never committed)
src/trustswap/{common,ingest,extract,verify,score,decide,synth,finetune,eval}
tests/
```

### Failed EDGAR downloads and submission recovery

Remote HTTP/network failures after retries append URL, status (blank if unknown)
and reason to `data/real/download_failures.csv`; other candidates continue.
On an exhibit 404, both scripts try the accession's full `.txt` submission. They
select the original `<DOCUMENT>` block by exhibit filename first, or the first
block containing one of that candidate's matched phrases if no filename matches.
The selected block is saved as a separate `.full_submission.txt` file with its
own SHA-256. Its manifest entry records `source="full_submission"`, actual source
URL, selected filename/method, and the full submission's hash. Normal downloads
record `source="exhibit"`. Existing raw files are never overwritten.

Screening ingests the selected SGML block as text even when the missing exhibit
was named `.pdf`. Its CSV records `download_source`, `resolved_source_url`, and
`selected_filename`, and the document record cites the actual source URL.
Each command ends with **Downloaded**, **Recovered via fallback**, and **Failed**
counts. Downloaded includes verified cached files and recoveries; recovered is a
subset. Failed counts terminally unsuccessful candidates, not retry requests.
The append-only failure log also records 404s that were subsequently recovered.
Missing-block and fallback failures are logged and trigger same-era replacement
in screening. Local manifest-integrity conflicts still stop rather than replace
or silently trust altered files. No live SEC requests run in the tests.

### Screening limits and resume

Each document's ingest and score run in a separate process with a 60-second
limit; a stuck native parser is terminated. Timeout and parsing errors append
URL, status (`timeout`/`error`) and reason to `data/real/screen_failures.csv`, and
the next document continues. New ingest files are staged and promoted only after
successful processing, so timeouts cannot append partial document records.

Screening disables OCR (ordinary Step 2 ingestion keeps OCR enabled by default).
A PDF with any page lacking a text layer receives `needs_ocr`, with no OCR call
or incomplete text record. Files larger than 15,000,000 bytes receive `too_large`
and are not ingested. These labels appear in `screen.csv` and summary counts but
are excluded from the three-label manual-check sample.

Successful screen rows are checkpointed after each document. Resume reuses
previous rows or saved ingested text without extracting original files again;
already-ingested documents never trigger another download, even if raw files are
missing. Existing manual-review answers are retained. Progress prints every 10
documents, including cached results, skips and failures.

### Human-reviewed model proposals

Use an already-installed model served by Ollama, or explicitly select the
OpenAI-compatible API (requires `OPENAI_API_KEY`; the SDK also reads
`OPENAI_BASE_URL` for your gateway). Install `requirements-dev.txt` first:

```bash
python scripts/annotate_auto.py --annotator A1 --model local:qwen2.5:7b --dry-run
python scripts/annotate_auto.py --annotator A1 --model local:qwen2.5:7b
python scripts/annotate_auto.py --annotator A1 --model openai:gemini-2.0-flash --dry-run
```

For each field, review the raw proposal, normalised value, exact spans and three
context lines. Use y to accept, n for manual correction, or s to skip. Repeated
values require choosing an occurrence. Feature flags remain manual. Rerun to
resume; add `--revisit-skipped` to review skips. Responses use the Step 0 cache;
dry-run displays proposals and fills the cache without annotation/log writes.
Human decisions go to the usual `A1.jsonl`, with a separate releasable
`A1.proposals.jsonl` provenance log. See
[the assisted annotation protocol](docs/annotation_provenance.md) for limitations
and the paper disclosure: “Annotations were LLM-proposed and human-confirmed;
the proposal log is released.”
