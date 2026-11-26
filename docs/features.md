# Feature table (Step 13)

`python scripts/build_features.py --model <frontier|open_weights>` writes
`cache/features/<model>.parquet` (one row per document and gold field path) and
`results/features_<model>.md` (label rate per collection/split and per field).

| prefix | group | columns |
| --- | --- | --- |
| `u_` | uncertainty | mean token log-prob of the k = 0 value; share of K samples equal to the modal value (K = 1 gives 1.0); agreement with the other model's k = 0 value; verbalised confidence; predicted null |
| `g_` | grounding | log(1 + V2 residual); evidence span found in the text |
| `v_` | verdicts | per rule V1..V7, one-hot PASS / HARD / SOFT / UNDET / NONE (rule silent on the field) |
| `rho_` | residuals | per rule, log(1 + abs(residual)), 0 when null |
| `tau_` | field type | one-hot field name, leg and default severity (critical / major / minor) |
| `q_` | document | OCR flag, log text length, template family seen in training, schema-valid k = 0 output, retried after the output limit |

Label `e = 1` when the normalised k = 0 value differs from gold, or gold is
ambiguous (the correct action is then to escalate). Missing inputs (no log-probs,
schema-invalid output, other model absent) are filled with a neutral value and
flagged in a `*_missing` column, so the table has no missing values. Metadata
columns (`doc_id, field, model, collection, split, defect, is_control, k_samples`)
are for splitting and reporting, never features. Collections stay separate:
risk models train on synthetic `train`, tune on `dev`, and report `syn_test` and
`adv_test` apart (Step 14).

# Risk models (Step 14)

`python scripts/train_risk.py --model <frontier|open_weights>` fits four calibrated
models of P(e = 1): {base, full} x {L2 logistic, LightGBM}. base uses the u, g, tau
and q columns (the uncertainty-only comparator for H1); full adds the verifier
columns v and rho. Models are fitted on synthetic train; hyperparameters (logistic C;
LightGBM leaves and minimum child size) are chosen on dev by log loss; probabilities
are calibrated on dev (Platt scaling for logistic, isotonic for LightGBM).

Outputs: `results/models/<model>/*.joblib`, `results/models/<model>/metrics.json`,
`results/risk_<model>.md` (ECE with 15 bins, Brier, log loss, AUROC, AUPRC on dev,
syn_test and adv_test), and `cache/risk/<model>.parquet` (calibrated P(error) per
field for every variant; the input of Step 15). Dev scores are optimistic because
calibration is fitted there; syn_test and adv_test are the honest numbers. With few
dev errors, isotonic calibration is coarse; this is reported, not tuned away.
