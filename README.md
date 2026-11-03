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

`data/real/screen.csv` records term flags, page count/basis, score, era and a
heuristic triage label. All four term groups plus fewer than 40 pages means
`likely_confirmation`; at least two groups means `maybe`; otherwise `unlikely`.
PDF page counts are exact. HTML/TXT counts are marked `estimated_text_layout`:
A4, 16 mm margins, 10 pt DejaVu Serif, ingested text only, with remote resources
blocked. These are triage estimates, not claims about a source HTML page count.

`data/real/manual_check.csv` has 60 seeded random rows, 20 per label, with an
empty `is_confirmation` column for a human. If a label has fewer than 20 rows,
the script saves/prints the screen results and reports the shortage without
writing a partial manual sample. It never duplicates examples or invents human
labels. Existing manual-check files are protected: preserve/move one before
rerunning. Screening does not write gold labels or assign train/test splits,
and refuses to reuse a document that already has a split.

Use `--data-root`, `--candidates` or `--raw-dir` for alternate local paths.
Tests use only fictional saved fixtures and mocked HTTP, not SEC downloads.

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
