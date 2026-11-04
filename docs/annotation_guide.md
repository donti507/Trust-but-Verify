# Pilot annotation guide (Step 4)

Use only the saved UTF-8 text identified by the pilot row. Offsets are zero-based
Unicode character offsets `[start, end)`, not bytes, PDF coordinates, or offsets
into a browser/HTML source. The span is **the exact text of the value in the
saved text file**: `text[start:end]` must reproduce the entered quote. Labels
come only from the document and its identified incorporated references, never
from a verifier, a model prediction, or an assumed market convention.

## Four statuses

- **explicit**: the applicable value is stated in this document. Store its
  normalised value and exact evidence span. The span normally excludes the field
  label; currency/rate units can be included when part of the stated value.
- **by_reference**: the term is supplied by an identified incorporated definition,
  such as the specified ISDA Definitions edition. Leave `span` null. Enter a
  value only if supported by that reference; otherwise leave value null. Never
  substitute a remembered/default market convention or run a verifier to fill it.
- **ambiguous**: competing statements, unresolved amendment precedence, units or
  reference/fallback applicability prevent a unique answer. Store null value and
  span; the downstream correct action is ESCALATE. Do not select a plausible
  value just to complete the form.
- **absent**: there is no applicable statement or supported incorporation supplying
  this term. Both value and span are null. Absence is a positive human decision,
  not an unanswered annotation; never invent zero spreads or counterparties.

A clearly controlling amendment governs the amended field. Equal-authority
conflicts remain ambiguous. Keyword search is navigation only: no keyword hit
is not evidence of absence. Read enough surrounding text, including leg headings,
to determine the term's scope and whether it governs this transaction.

## Workflow and special features

Run A1 and A2 independently, without viewing the other's annotations or verifier
output. Type an exact saved-text quote; if repeated, choose its actual occurrence
from the displayed offsets/context. Inspect the normalised proposal and confirm
with y before it is saved. Use absent, ref, amb, show, or back. Reserved literal
values can be entered as `text:<literal>`. Use `\n` to enter saved line breaks.
Unsupported normalisation is a tooling issue: stop/report it rather than mark a
clear term absent or ambiguous just to bypass the parser.

After the fields, explicitly enter the document-supported `flags` list:
`amortising`, `cap`, `floor`, or none. Flag an amortising schedule even if only an
initial notional fits the scalar schema. Caps/floors do not become the fixed rate;
record the stated underlying terms and flag the optional feature. These flags
are stored alongside the trade fields, not invented as extra swap economics.
Complex features never justify replacing document labels with verifier values.
Real-test documents are excluded from this pilot. This tool creates independent
human annotation snapshots, not adjudicated gold; the domain lead resolves
all disagreements before any real gold record is published.

## trade_date

**Definition:** Date the parties entered into the transaction, not the filing or signature-upload date.

**Usually appears:** Opening trade details or letter heading.

**Examples and boundary rules:** Trade Date: 20-Mar-2012 → 2012-03-20.

## effective_date

**Definition:** Contractual start date on which the swap becomes effective.

**Usually appears:** Transaction details, commencement clause or schedule heading.

**Examples and boundary rules:** Effective Date: 2012-03-22 → 2012-03-22. Do not calculate T+2 yourself.

## termination_date

**Definition:** Contractual maturity/end date applicable to the transaction, including a clearly controlling amendment.

**Usually appears:** Transaction details, maturity clause or amendment.

**Examples and boundary rules:** Termination Date: March 22 2017 → 2017-03-22.

## fixed_leg.payer

**Definition:** Legal party paying this leg, including a branch if stated as part of its identity.

**Usually appears:** Leg heading, Fixed/Floating Amount Payer or Paying Party clause.

**Examples and boundary rules:** Fixed Amount Payer: Example Bank A → Example Bank A. Do not infer it from the other leg.

## fixed_leg.receiver

**Definition:** Legal party receiving this leg.

**Usually appears:** Leg heading or Receiving Party clause.

**Examples and boundary rules:** Receiver: Example Company B → Example Company B. Do not infer it merely because another party pays.

## fixed_leg.notional_amount

**Definition:** Principal used to calculate this leg, not a settlement payment or market value.

**Usually appears:** Notional, Notional Amount or Calculation Amount row.

**Examples and boundary rules:** USD 250,000,000 / 250 million / USD 250mm → Decimal 250000000. For amortisation, use an expressly designated initial principal and flag amortising; otherwise mark competing scalar amounts ambiguous.

## fixed_leg.notional_currency

**Definition:** Currency of this leg’s principal.

**Usually appears:** Currency row or currency prefix/suffix on the notional.

**Examples and boundary rules:** USD / US$ → USD; € → EUR. Cite the currency token itself when possible.

## fixed_leg.payment_frequency

**Definition:** Regular interval between payments for this leg, not the rate reset interval or a one-off stub.

**Usually appears:** Payment Dates, Payment Frequency or Calculation Period payment clause.

**Examples and boundary rules:** semi-annual / 6M → 6M; annual → 1Y. Do not derive it by regenerating a schedule.

## fixed_leg.reset_frequency

**Definition:** Fixed legs do not normally reset. Mark absent when inapplicable; never copy the floating reset frequency. If an unusual document expressly states a fixed-leg reset term, capture the statement and flag it for the domain lead rather than correct it from a verifier.

**Usually appears:** Fixed-leg terms, if mentioned at all.

**Examples and boundary rules:** No fixed-leg reset statement → absent with null value/span.

## fixed_leg.day_count_fraction

**Definition:** Contractual fraction/basis for accruing this leg’s interest.

**Usually appears:** Day Count Fraction or Calculation Basis row.

**Examples and boundary rules:** Actual/360 → ACT/360; Actual/365 (Fixed) → ACT/365.FIXED. ACT/365 without a resolved qualifier is ambiguous.

## fixed_leg.business_day_convention

**Definition:** Rule used to adjust the leg’s stated dates when they are not business days.

**Usually appears:** Business Day Convention or Date Adjustment clause.

**Examples and boundary rules:** Modified Following → MODFOLLOWING. Do not choose a convention from market practice.

## fixed_leg.business_centers

**Definition:** All stated centers/calendars defining business days for this leg.

**Usually appears:** Business Centers/Centres, Financial Centres or Business Day definition.

**Examples and boundary rules:** London and New York → [GBLO, USNY]. Capture the whole combined phrase, not just the first city.

## fixed_leg.fixed_rate

**Definition:** Contractual fixed interest rate on the fixed leg, excluding spreads, caps and floors.

**Usually appears:** Fixed Rate or Fixed Interest Rate row.

**Examples and boundary rules:** 4.25% / 425 bps → Decimal 0.0425. If the token is bare, state its unit only when the document/header supports it.

## floating_leg.payer

**Definition:** Legal party paying this leg, including a branch if stated as part of its identity.

**Usually appears:** Leg heading, Fixed/Floating Amount Payer or Paying Party clause.

**Examples and boundary rules:** Floating Amount Payer: Example Bank A → Example Bank A. Do not infer it from the other leg.

## floating_leg.receiver

**Definition:** Legal party receiving this leg.

**Usually appears:** Leg heading or Receiving Party clause.

**Examples and boundary rules:** Receiver: Example Company B → Example Company B. Do not infer it merely because another party pays.

## floating_leg.notional_amount

**Definition:** Principal used to calculate this leg, not a settlement payment or market value.

**Usually appears:** Notional, Notional Amount or Calculation Amount row.

**Examples and boundary rules:** USD 250,000,000 / 250 million / USD 250mm → Decimal 250000000. For amortisation, use an expressly designated initial principal and flag amortising; otherwise mark competing scalar amounts ambiguous.

## floating_leg.notional_currency

**Definition:** Currency of this leg’s principal.

**Usually appears:** Currency row or currency prefix/suffix on the notional.

**Examples and boundary rules:** USD / US$ → USD; € → EUR. Cite the currency token itself when possible.

## floating_leg.payment_frequency

**Definition:** Regular interval between payments for this leg, not the rate reset interval or a one-off stub.

**Usually appears:** Payment Dates, Payment Frequency or Calculation Period payment clause.

**Examples and boundary rules:** semi-annual / 6M → 6M; annual → 1Y. Do not derive it by regenerating a schedule.

## floating_leg.reset_frequency

**Definition:** Regular frequency at which the floating rate is reset; this is distinct from payment frequency.

**Usually appears:** Reset Dates, Fixing or Reset Frequency clause.

**Examples and boundary rules:** quarterly / 3 months → 3M. An initial stub is not a different regular tenor.

## floating_leg.day_count_fraction

**Definition:** Contractual fraction/basis for accruing this leg’s interest.

**Usually appears:** Day Count Fraction or Calculation Basis row.

**Examples and boundary rules:** Actual/360 → ACT/360; Actual/365 (Fixed) → ACT/365.FIXED. ACT/365 without a resolved qualifier is ambiguous.

## floating_leg.business_day_convention

**Definition:** Rule used to adjust the leg’s stated dates when they are not business days.

**Usually appears:** Business Day Convention or Date Adjustment clause.

**Examples and boundary rules:** Modified Following → MODFOLLOWING. Do not choose a convention from market practice.

## floating_leg.business_centers

**Definition:** All stated centers/calendars defining business days for this leg.

**Usually appears:** Business Centers/Centres, Financial Centres or Business Day definition.

**Examples and boundary rules:** London and New York → [GBLO, USNY]. Capture the whole combined phrase, not just the first city.

## floating_leg.floating_rate_option

**Definition:** Named contractual floating benchmark/index, including a clearly triggered fallback.

**Usually appears:** Floating Rate Option, Reference Index or fallback clause.

**Examples and boundary rules:** USD SOFR → USD-SOFR. A legacy/fallback choice with no controlling trigger is ambiguous; never select an index from the trade date alone.

## floating_leg.spread

**Definition:** Additive margin on this floating leg, including its sign.

**Usually appears:** Spread or Margin row.

**Examples and boundary rules:** minus 50 bps → Decimal -0.005. No stated margin is absent, not automatically zero.

## notice_address

**Definition:** Address designated for transaction notices; exclude unrelated issuer/SEC/counsel addresses.

**Usually appears:** Notices, Address for Notices or party contact section.

**Examples and boundary rules:** Copy the exact address block. Type \n for saved line breaks; preserve the whole evidence span.
