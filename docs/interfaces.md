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
`fixed_leg` and `floating_leg_2` must be present. Every extracted field is
explicit with a non-null value and a null span before rendering.

Sampling uses the market assumptions in `synth/markets.py`, QuantLib settlement
calendars, two business days to effectiveness, and convention-adjusted yearly
tenors. The USD LIBOR calendar requires both New York and London to be open.
Era rate ranges are illustrative synthetic assumptions, not historical curves.
The RNG is supplied by the caller; no global seed or evaluation date is changed.

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
