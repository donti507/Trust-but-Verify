# Interfaces and file formats

## Document record (Step 2)

Write one JSON object per line to `data/<collection>/docs.jsonl`.
The Pydantic v2 definition is `trustswap.common.records.DocumentRecord`.

```json
{"doc_id":"real_0042","collection":"real","family_id":"fam_017","split":"real_cal","source_url":"https://www.sec.gov/Archives/...","sha256":"<64 lowercase hexadecimal characters>","text_path":"data/real/text/real_0042.txt","ocr":false,"parent_id":null,"defect":null}
```

- `collection`: `real`, `synthetic`, or `adversarial`. Collections remain separate.
- `split`: `train`, `dev`, `syn_test` for synthetic documents (by template family);
  `real_cal`, `real_test` for real documents (by document family); `adv_test` for
  adversarial documents. Ingestion leaves both `split` and `family_id` null.
- `source_url`: original URL for real documents; null otherwise. Ingestion never
  fetches it. Callers supply locally available source files.
- `parent_id` and `defect`: only for adversarial documents; null otherwise.
- `sha256`: SHA-256 of the original file bytes, before extraction or normalisation.
- `text_path`: path to UTF-8 text. `ocr` is true if any page required OCR.

`ingest_document(path, collection=..., output_dir="data", doc_id=...)` saves text
under `<output_dir>/<collection>/text/`, appends a record to `docs.jsonl`, and
returns `(text, meta)`. `meta` includes all record fields plus `source_text` and
`offset_map`. If omitted, the ID is `<collection>_<original-byte-sha256>`.
Existing text and offset files are never overwritten. No writes are allowed
inside `data/raw/`, including through symlinks.

## Text offsets

`<doc_id>.offsets.json`, next to the text, contains `source_text` (visible text
before whitespace normalisation) and `offset_map`. Each entry `[a,b]` maps one
saved character to `[a,b)` in `source_text`. Non-whitespace characters map
one-to-one; collapsed whitespace maps to the entire original whitespace run.
Leading/trailing whitespace is removed, horizontal runs collapse to a space,
and runs containing line breaks collapse to a newline. Table cells are separated
by ` | ` and table rows by newlines.

Spans `[start,end)` use Unicode character positions into saved text; slicing
`text[start:end]` returns the extracted phrase. To recover its source range, use
`offset_map[start][0]` and `offset_map[end-1][1]` for a nonempty span. These are
positions in extracted text, not PDF byte offsets, bounding boxes, or HTML markup.

## Synthetic trades (Step 5A)

`sample_trade(random.Random(seed))` returns an `InterestRateSwap`. Fixed/float
records use `fixed_leg` and `floating_leg`; float/float records use
`floating_leg` and `floating_leg_2`, with `fixed_leg` null. Exactly one of
`fixed_leg` and `floating_leg_2` must be present. Every applicable extracted field is explicit with a non-null value and a null
span before rendering. Fixed-leg reset frequency is absent with null value and
span; fixed legs do not reset.

Sampling uses the market assumptions in `synth/markets.py`, QuantLib settlement
calendars, business-day trade dates, T+0 effectiveness for GBP and T+2
elsewhere, and convention-adjusted yearly tenors. The USD LIBOR calendar requires both New York and London to be open.
Era rate ranges are illustrative synthetic assumptions, not historical curves.
The RNG is supplied by the caller; no global seed or evaluation date is changed.
Fixed/float is sampled with probability 85%, basis swaps with 15%. Basis sampling
is conditional on IBOR-eligible currency/era combinations: distinct reset tenors
of the same IBOR, or USD LIBOR versus SOFR in 2018–2021. GBP uses LIBOR before
2022 and SONIA from 2022. Fixed legs use typical market frequency/day-count pairs
80% of the time, with weighted alternative choices for the remaining 20%.

## Gold records (Step 5)

Write one JSON object per document per line in `data/<collection>/gold.jsonl`.
The Pydantic definition is `trustswap.common.records.GoldRecord`.

```json
{"doc_id":"syn_000001","collection":"synthetic","trade":{"...":"InterestRateSwap.model_dump(mode=\"json\")"},"generator":{"seed":20261026,"sampler_version":"0.1","template_family":null}}
```

`trade` is the complete Step 1 schema serialized with `model_dump(mode="json")`;
all term fields contain `value`, `span`, and `status`. Status is `explicit`,
`by_reference`, `ambiguous`, or `absent`. An ambiguous field's correct action is
**ESCALATE**. Spans are half-open character offsets into saved document text, or
null. Step 5A leaves all spans null; part B fills them after rendering.

`generator` exists only for synthetic/adversarial records. Real records instead
contain `annotators` (for example `["A1", "A2"]`) and `adjudicated` (a boolean).
Real gold comes exclusively from human annotators, never an LLM or verifier.

`write_gold_record(trade, doc_id=..., seed=..., output_dir="data")` appends a
synthetic record, preserving Decimal values as JSON strings and dates as ISO
strings. It also accepts `collection="adversarial"` and optional template family
metadata, but rejects real labels. The seed describes the caller's RNG stream;
the sampler version is `0.1`. No raw files are modified.

## Synthetic rendering (Step 5B)

`render_trade(trade, seed=..., doc_id=..., template_family=..., output_dir="data")`
returns a `RenderResult` containing HTML/PDF paths, ingested text, a
`DocumentRecord`, and a `GoldRecord`. Omitting `template_family` chooses a family
deterministically from the rendering seed. The input trade is never mutated.
Applicable term fields must have explicit, non-null values. Absent fields are
not rendered and retain null spans.

Artifacts are written to `data/synthetic/html/<doc_id>.html`,
`data/synthetic/pdf/<doc_id>.pdf`, and `data/synthetic/text/<doc_id>.txt` with its
`.offsets.json` sidecar. Document and gold records append to the existing JSONL
files. Documents receive `family_id`, `split`, and original-final-PDF SHA-256;
gold receives `generator.template_family` and a span for every term. Ingestion
accepts optional `family_id` and `split` supplied by rendering; its defaults
remain null for ordinary source-document ingestion.

The immutable family manifest is `trustswap.synth.templates.FAMILY_SPLITS`:

| Family | Split |
| --- | --- |
| label_value | train |
| two_column | train |
| dealer_caps | train |
| legal_preamble | dev |
| letter | syn_test |
| term_sheet | syn_test |

Variants and wording changes stay within their family's split. Future families
must receive one manifest assignment; callers cannot override the split. This
initial set has six families, with twelve planned later.

Every PDF starts with `SYNTHETIC DOCUMENT: fictional parties, generated for research`.
Jinja templates escape supplied text. WeasyPrint uses only packaged templates,
inline CSS and local system fonts; its resource fetcher rejects all URLs.

During layout, invisible link annotations mark field identities and rectangles.
PDF glyph positions identify the exact occurrence of each value, including
repeated amounts and party names. The renderer translates these positions through
Step 2's whitespace map to saved-text spans. It removes the annotations, ingests
the final PDF, and checks that removing markers did not change the extracted
text. Neither the final HTML nor PDF contains marker metadata. Spans refer to
the final PDF's ingested text, not to the HTML source or the input trade.

Dates, notionals, rates, frequencies, day counts and business centers have varied
human-readable formats. Tests compare the span text using Step 1 normalisers;
index spelling and center-list order are compared in canonical form. The
renderer rejects missing, overlapping or mismatched markers and refuses existing
artifact paths and all writes under `data/raw/`.

PDF creation and modification metadata are fixed at `2026-01-01T00:00:00Z`.
The PDF identifier is SHA-256 of `trustswap-pdf:<rendering seed>`. Marker removal
preserves that identifier. Fixed-seed tests compare HTML bytes, ingested text, all gold spans and gold
records. PDF bytes and their hashes are not determinism assertions: PDF encoders
may vary bytes while preserving identical text and spans. CI fixes
`PYTHONHASHSEED=0`.

## Adversarial injector (Step 6)

`trustswap.synth.defects` exposes one function per type: `contradiction`,
`amendment`, `non_standard_calendar`, `swapped_legs`, `scale_error`,
`unusual_day_count`, `stub_ambiguity`, `multiple_notionals`,
`legacy_index_fallback`, `ocr_digit_corruption`, `missing_information`, and
`prompt_injection`. Each accepts `(parent_trade, render_seed, severity=1,
parent_id=None, control=False, designed_by="team")` (options are keyword-only).
It returns a `DefectCase` with separate displayed `trade`, structured `gold`,
`label`, rendering edits/notes, metadata, and an untouched copy of the parent.
Omitted parent IDs are deterministic hashes of the structured clean trade.

All cases have severity 1 or 2. Examples include an amendment of one business day
versus one year; scale factors ×10 versus ×1000; one OCR digit substitution versus
all zero digits; a missing notice address versus one seed-selected domain term; and one versus three injected
instructions. OCR corruption here simulates letter/digit corruption in a text
layer; it does not pretend that raster OCR ran (`ocr` reports actual ingestion).

`render_case(case, output_dir="data", template_family=...)` uses the Step 5B
renderer. It first saves the clean twin in `synthetic`, or verifies existing
parent records, PDF hash, structured gold, rendering seed, family and artifact
presence before reusing it. A missing or changed twin is never silently replaced.
Children go only to `adversarial`, with document split `adv_test`. No real data or
real labels are inspected. Rendering never calls an API or executes document text.

Both the document and gold JSONL record include these top-level fields:

| Field | Meaning |
| --- | --- |
| parent_id | Existing clean synthetic document ID |
| defect | One of the twelve function/type names |
| severity | 1 or 2, interpreted within the defect type |
| affected_fields | Nonempty list of Step 1 paths, e.g. `floating_leg.notional_amount` |
| expected_rules | Nonempty list of V1–V7 identifiers or `["silent"]` |
| designed_by | `team` or caller-supplied `blind` designation |
| is_control | True for an unusual-but-correct matched control |

These additional fields are omitted from legacy/non-adversarial records when
unset. `AdversarialMetadata` validates complete Step 6 provenance. Historical
Step 2/5A adversarial records with only parent/defect metadata remain readable.
New Step 6 records always include the complete metadata, in both JSONL files.

Gold is derived by structured edits, never from a model's extraction or verifier:

- Amendments explicitly supersede the former date; gold holds the amended date.
- Contradictions, competing notionals, ambiguous effective/termination boundaries and untriggered
  fallback clauses have ambiguous affected gold with null value/span; correct
  action is ESCALATE.
- Missing fields have absent gold with null value/span, and are omitted from the
  PDF. The renderer rejects omissions paired with non-absent gold.
- OCR and scale corruption retain the true parent amount in gold. Swapped role
  summaries reverse payer/receiver on both legs, keeping the trade internally
  consistent; all four role fields are affected. Gold retains the authoritative
  parent allocation, also stated in the text. The higher severity repeats the
  reversed allocation in the closing summary.
  Corrupted spans may deliberately fail grounding against the true gold value.
- Prompt-injection text is inert document content. Economic values/statuses in
  gold are unchanged; spans are recomputed in the new document. No instructions
  from the document are followed.
- Nonstandard-calendar defects place an explicit date on an added center's
  actual QuantLib holiday; correct controls recompute settlement using the joint
  calendar. Rare valid day counts use ACT/ACT.ISDA or 30E/360. Stub controls state
  an explicit short initial or final period with unambiguous effective/termination
  dates. Defects make the effective date (severity 1) or termination date
  (severity 2) ambiguous; reset frequency is unchanged. Expected rule is V4.

`make_control(type, parent, seed, ...)` produces a matched control for every type.
Controls have correct explicit gold (except inapplicable fixed resets), preserved
clean twins, and no expected HARD result. Calendar/day-count controls may receive
V7 SOFT; other controls predict silence. No control is relabelled as a defect
merely because it is rare. Expected-rule mappings are editable H4 predictions in
one code table, `EXPECTED_RULES`; see [verifier_rules.md](verifier_rules.md) for
V1–V7 definitions. This step creates test cases, not a verifier implementation.

Missing-information severity 1 omits only `notice_address`. Severity 2 omits
exactly one of `floating_leg.business_centers`, `floating_leg.day_count_fraction`,
or `floating_leg.spread`, selected deterministically by the parent's render seed.
The omitted term alone is listed in `affected_fields` and has absent/null gold;
the notice address remains present at severity 2.
