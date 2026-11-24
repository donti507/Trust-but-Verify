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

## Implementation status (Steps 7-8)

V1-V4 are implemented in `src/trustswap/verify/` as pure functions
`rule(trade, text) -> {field_path: RuleResult}`; `trustswap.verify.run.verify`
runs them and refuses any rule that modifies the trade. V5-V7 were added in Step 8.

| Rule | HARD | SOFT | UNDET | Residual |
| --- | --- | --- | --- | --- |
| V1 | required field missing; invalid currency, day count, convention, period, floating rate option; non-positive notional | `ambiguous` status; well-formed centre outside the local list | never | null |
| V2 | span does not normalise, lies outside the text, or differs beyond rounding | relative difference <= 0.005; text edit distance <= 0.20 | no value or no span | relative difference; days for dates; edit distance for text; 0/1 for codes |
| V3 | trade date after effective date; termination not after effective | tenor < 7 days or > 50 years | dates missing | days |
| V4 | a date stated on a schedule line is off every leg's regenerated schedule | effective/termination date not a business day (real confirmations state unadjusted dates) | centres outside the local calendar list | days |
| V5 | a stated fixed amount is more than 1% from the closest recomputed period amount | between max(1 unit, 0.01%) and 1% | no fixed amount stated; inputs missing; ACT/ACT.ICMA | currency units |
| V6 | index currency differs from leg currency; leg notionals differ; payer is not the other leg's receiver; a stated notional matches no extracted notional | legacy IBOR with fallback wording naming an RFR; LIBOR on a trade dated 2022+; leg currencies differ; notional differences when the text says amortising/accreting | index, currency or party missing | 1.0 categorical; relative difference for notionals |
| V7 | never | floating day count not the index standard; centres missing the index home centre or adding centres outside the currency/era norm; fixed day count outside 30/360, ACT/360, ACT/365.FIXED | unknown index or currency; value missing | 1.0 categorical |

V4 reads dates only from schedule lines (payment date, period end, calculation
period, reset date, first/last period, commencing, stub). Dates on lines that
mention a stub are passed to QuantLib as `firstDate`/`nextToLastDate`, so agreed
stubs pass. Thresholds are pre-registration candidates.

V5 reads only lines that state a fixed amount ("fixed amount", "interest
amount", "payment amount", "amount payable", "coupon amount", "accrued") with an
explicit currency; floating amounts need a rate fixing and are not checked. If
the line gives a period (two dates), that period is accrued; otherwise the
closest period of the regenerated fixed schedule counts. V6 and V5 read money
only when a currency code or symbol is attached, and never partially read OCR
damage ("EUR 388,OOO,OOO" is skipped, not read as EUR 388). V6 does not check
designated maturity against reset tenor: the schema has no designated-maturity
field, and synthetic basis legs legitimately reset and pay on different tenors.

First Step 7 observation (one synthetic parent, both severities): V1-V4 flag
non-standard calendar (V4), stub ambiguity (V4), OCR digit corruption (V2), scale
error (V2) and missing required terms (V1); all twelve controls produce no HARD.
Contradiction is silent under V2 because each competing amount is individually
grounded; this is evidence for H4, to be measured properly in Step 9.

Step 8 observations (one synthetic EUR fixed/float parent, both severities; the
same pattern on a second parent). These are early H4 evidence, to be measured
properly in Step 9:

* As predicted: contradiction fires V6 (the competing notional matches no
  extraction); legacy index fallback fires V6 (SOFT); unusual day count fires V7.
* Not predicted: multiple notionals (predicted silent) and scale error
  (predicted V2, V5) both fire V6 through the stated-notional check. OCR digit
  corruption is silent under V5-V7.
* V5 is UNDET on every synthetic document because the templates state no
  payment amounts. V5 evidence must come from real confirmations that state
  fixed amounts, or from a template family that adds them.
* Stub ambiguity and amendment are silent under V5-V7. Non-standard calendar
  defects and their controls are both SOFT under V7, so V7 alone cannot separate
  them; V4 does.
* No control produces a HARD verdict under any of V1-V7; 1,000 clean sampled
  trades produce no V5-V7 flags.
