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
