# Verifier rule families

Each applicable rule returns a per-field verdict (`PASS`, `HARD`, `SOFT`, `UNDET`)
and a numeric residual or null. Rules never edit values. `UNDET` means a rule
cannot apply. V1–V2 are generic validation; V3–V7 are domain rules.

| Rule | Checks | Residual | Verdict |
| --- | --- | --- | --- |
| V1 Schema | Required fields present; codes valid in FpML enumerations | null | HARD on missing/invalid |
| V2 Grounding | Value appears in or normalises from its cited span | Normalised edit distance for strings; relative difference for numbers | HARD above threshold; SOFT for near matches |
| V3 Temporal | trade_date ≤ effective_date < termination_date; stated tenor matches dates | Days | HARD |
| V4 Calendar | Regenerate schedule with QuantLib (dates, frequency, convention, centers); compare with explicit dates | Days off | HARD when explicit dates differ |
| V5 Accrual | Recompute stated amounts with extracted day count and notional | Currency units | HARD beyond rounding |
| V6 Cross-field | Index/currency match; designated maturity/reset tenor match; leg notionals equal unless amortising | Categorical | HARD or SOFT |
| V7 Convention prior | Day count/calendar match market standard for index/currency | Categorical | SOFT only |

Expected rule families per defect are H4 predictions to test, not observed rule
outcomes or facts. Their single editable mapping is `EXPECTED_RULES` in
`trustswap.synth.defects`. `silent` predicts no rule detection. Amendment and
swapped-leg defects may be silent; prompt injection is a security test, not a
rule test. Unusual-but-correct controls expect **no HARD verdict**; V7 may be
SOFT. This step does not implement, run or disable verifier rules.
