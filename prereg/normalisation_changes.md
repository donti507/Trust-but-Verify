# Extraction normalisation changes

**2026-11-28 (before S19).** The extraction parser trims boundary separators from
date, notional amount, fixed-rate and spread values before typed parsing.
Attached negative signs and decimal points remain part of numeric values.
The original extraction text, token probabilities and evidence offsets are
retained. This change is in extraction post-processing. The v1 prompt, shared
Step 1 normalisers and V1–V7 rule code retain their existing hashes and behaviour.

To regenerate the paired comparison from the original parsed extraction cache,
run `python scripts/replay_normalisation.py` from the source checkout. It writes
`results/normalisation_replay/comparison.md` and keeps the before and after
outputs separate. Use `--data` and `--cache` for a different local input tree.
The comparison uses full logistic risk and a fixed λ of 1 by default. The replay
reads eligible synthetic and adversarial documents and cached samples. It
leaves the source cache and data untouched.
