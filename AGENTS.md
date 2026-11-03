# Instructions for AI coding agents (Codex and others)

This repository implements the *Trust but Verify* research project. The full plan is the
Technical Runbook: https://claude.ai/artifact/1BNdD4TFWgzVVhZoEMpMxq

## How to work
- Do exactly one runbook step per task. Implement the step's "Check" as pytest tests.
- Follow the file formats in the runbook section "Interfaces and file formats".
- Keep functions pure where possible; no network calls or paid APIs inside tests.
- Run `pytest` before proposing a change. CI must pass.

## Five rules that never bend
1. Never train on, tune on, or look at results from the `real_test` split before Step 19
   (pre-registration) is Done.
2. Never let the verifier edit an extracted value. It only reports a verdict and a residual.
3. Never let an LLM write a gold label. Real labels come from human annotators; synthetic labels
   come from the structured trade the document was rendered from.
4. Never mix collections. Real, synthetic and adversarial results are always reported separately.
5. Cache every model output by input hash. Re-running an analysis must never re-call a model.

## Also
- Never commit anything under data/, cache/ or results/, and never commit credentials.
- Finance logic (day counts, calendars, schedules) must be tested against QuantLib, not hand-rolled.
