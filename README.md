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
