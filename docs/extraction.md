# Extraction (Steps 10-12)

`src/trustswap/extract/` turns a document into extraction records
(`cache/extract/<model>/<doc_id>.jsonl`, one line per sample k).

* **Prompt** `prompts/v1.txt` + `prompts/fewshot_v1.json`. The document is shown
  with numbered lines between `<<<DOCUMENT_START>>>` / `<<<DOCUMENT_END>>>`;
  delimiter look-alikes inside it are defanged and the model is told to ignore
  instructions in the document. Two few-shot examples (one fixed/float, one
  float/float) come from *train* template families, with answers built from the
  gold trade (`prompt.gold_answer`, never from a model). Regenerate with
  `python -c "from pathlib import Path; from trustswap.extract.prompt import build_fewshot; build_fewshot(Path('src/trustswap/extract/prompts/fewshot_v1.json'))"`
  and bump the prompt version when anything in the prompt changes.
* **Spans.** The model returns each value *verbatim* plus the line it is on;
  `parse.locate` computes the character offsets. This replaces asking the model
  for raw offsets (unreliable for LLMs) while keeping the runbook's record
  format. `span_found` is false when the value text is not in the document.
* **Values** are normalised with the Step 1 normalisers; failures keep
  `value: null` and record `normalise_error` and the `raw_value`.
* **Schema validity** = JSON matching `fields.output_schema()` (strict-mode
  compatible, 36 field paths) with every confidence in [0, 1].
* **Uncertainty**: `verbal_conf` (model's stated confidence) and `mean_logprob`
  (mean token log-prob of the value literal, when the provider returns log-probs).
* **Clients**: one OpenAI-compatible client for both tracks, configured in
  `configs/models.yaml` (JSON-schema mode or vLLM `guided_json`; log-probs on or
  off). Keys come only from environment variables named there.
* **Cache**: raw completions in `cache/raw/<model>/`, keyed by model id, prompt
  version, full prompt, temperature, k and seed. Re-runs never call the model.
  `real_test` is refused until Step 19.

Step 10 check: `python scripts/extract.py --model frontier --docs data/synthetic/docs.jsonl --limit 10`
and the same with `--model open_weights` must both print `schema-valid: 10/10 = 100.0%`.

## Pinning a model (`scripts/pin_model.py`)

The tool probes an OpenAI-compatible endpoint (seed, JSON-schema mode, vLLM
`guided_json`, plain JSON mode, log-probs) and writes the strongest working
settings for one track into `configs/models.yaml`, keeping comments:

    python scripts/pin_model.py --track frontier --name DeepSeek-V4.1-Flash \
        --version deepseek-ai/DeepSeek-V4.1-Flash-1017

Both tracks may point at the PCSS AI Hub (`https://llm.hpc.psnc.pl/v1`), an
OpenAI-compatible service for open-weight models. In `json_object` mode the
schema is not enforced by the server; the parser still validates every output.
