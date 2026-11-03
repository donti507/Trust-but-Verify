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
