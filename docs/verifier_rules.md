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
SOFT.

## Implementation status (Step 7)

V1-V4 are implemented in `src/trustswap/verify/` as pure functions
`rule(trade, text) -> {field_path: RuleResult}`; `trustswap.verify.run.verify`
runs them and refuses any rule that modifies the trade. V5-V7 follow in Step 8.

| Rule | HARD | SOFT | UNDET | Residual |
| --- | --- | --- | --- | --- |
| V1 | required field missing; invalid currency, day count, convention, period, floating rate option; non-positive notional | `ambiguous` status; well-formed centre outside the local list | never | null |
| V2 | span does not normalise, lies outside the text, or differs beyond rounding | relative difference <= 0.005; text edit distance <= 0.20 | no value or no span | relative difference; days for dates; edit distance for text; 0/1 for codes |
| V3 | trade date after effective date; termination not after effective | tenor < 7 days or > 50 years | dates missing | days |
| V4 | a date stated on a schedule line is off every leg's regenerated schedule | effective/termination date not a business day (real confirmations state unadjusted dates) | centres outside the local calendar list | days |

V4 reads dates only from schedule lines (payment date, period end, calculation
period, reset date, first/last period, commencing, stub). Dates on lines that
mention a stub are passed to QuantLib as `firstDate`/`nextToLastDate`, so agreed
stubs pass. Thresholds are pre-registration candidates.

First Step 7 observation (one synthetic parent, both severities): V1-V4 flag
non-standard calendar (V4), stub ambiguity (V4), OCR digit corruption (V2), scale
error (V2) and missing required terms (V1); all twelve controls produce no HARD.
Contradiction is silent under V2 because each competing amount is individually
grounded; this is evidence for H4, to be measured properly in Step 9.
