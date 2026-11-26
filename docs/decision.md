# Decision rule (Step 15)

`src/trustswap/decide/policy.py`: ESCALATE a field if `p_hat * c_bar > lambda * r`,
else ACCEPT. `c_bar` is the expected cost of accepting a wrong value (the `likely`
column of `configs/costs.yaml`, by field severity) and `r` the review cost. The
costs are placeholders (critical 100, major 10, minor 0.2, review 1, B = 100) until
the domain-lead cost session sets `status: confirmed`.

* Field mode: only escalated fields are reviewed; an accepted wrong field costs c.
* Trade mode: a document goes straight through only if every field is accepted.
* Document loss at lambda: `min(B, sum of accepted e * c) / B`, in [0, 1]; it never
  decreases as lambda grows. lambda = 0 escalates every field with p_hat > 0 and
  gives zero loss.
* Lambda grid (for Step 16): 0 plus 200 log-spaced values from 1e-3 to 1e3.
* Sensitivity: costs x0.5, x2, and critical/major swapped.

`python scripts/decide.py --model open_weights --variant full_logistic` writes
`results/decision_<model>.md` and a decision log
`results/<run_id>/decisions.jsonl` (doc_id, field, value, p_hat, c_bar, lambda,
action, rules fired, evidence span) used by the demo.
