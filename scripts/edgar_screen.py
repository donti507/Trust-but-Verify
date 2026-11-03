#!/usr/bin/env python3
"""Offline-testable EDGAR screening; run explicitly to download real candidates.

Screen labels are heuristic triage, never gold labels or split assignments.
Only operators fill manual_check.csv's is_confirmation column.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import date
import hashlib
from html import escape
import json
import os
from pathlib import Path
from random import Random
import re
import tempfile

# Support both `python scripts/edgar_screen.py` and import as scripts.edgar_screen.
if __package__:
    from . import edgar_scope
else:
    import edgar_scope

from trustswap.common.records import DocumentRecord
from trustswap.ingest.to_text import ingest_document, to_text

DEFAULT_SEED = 20261026
ERA_QUOTAS = {"2001–2005": 75, "2006–2008": 75, "2009–2021": 75, "2022+": 75}
LABELS = ("likely_confirmation", "maybe", "unlikely")
TERMS = {
    "has_notional": r"\b(?:notional|calculation) amount\b",
    "has_termination": r"\btermination date\b",
    "has_effective": r"\beffective date\b",
    "has_rate": r"\b(?:fixed rate|floating rate option)\b",
}
SCREEN_COLUMNS = (*edgar_scope.COLUMNS, "doc_id", "era", "raw_path", "sha256", "text_path",
                  "ocr", "page_count", "page_count_basis", *TERMS, "under_40_pages", "score", "label")


def era_for(filing_date: str, *, as_of: date | None = None) -> str | None:
    value = date.fromisoformat(filing_date)
    if value < date(2001, 1, 1) or value > (as_of or date.today()):
        return None
    if value.year <= 2005:
        return "2001–2005"
    if value.year <= 2008:
        return "2006–2008"
    if value.year <= 2021:
        return "2009–2021"
    return "2022+"


def read_candidates(path: str | Path) -> list[edgar_scope.Candidate]:
    """Validate identities/URLs and deduplicate before assigning sample slots."""
    documents = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(edgar_scope.COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError("candidate CSV is missing required Step 3A columns")
        for row in reader:
            candidate = edgar_scope.candidate_from_hit({
                "_id": row["accession_number"] + ":" + row["exhibit_file_name"],
                "_source": {"ciks": [row["cik"]], "display_names": [row["company"]],
                            "root_forms": [row["form"]], "file_date": row["filing_date"]},
            })
            if candidate.document_url != row["document_url"]:
                raise ValueError("candidate URL does not match its SEC document identity")
            candidate.matched_phrases.update(filter(None, row["matched_phrases"].split("; ")))
            existing = documents.get(candidate.document_url)
            if existing:
                if (existing.accession_number, existing.cik, existing.company, existing.form, existing.filing_date) != (
                    candidate.accession_number, candidate.cik, candidate.company, candidate.form, candidate.filing_date):
                    raise ValueError("conflicting duplicate candidate metadata")
                existing.matched_phrases.update(candidate.matched_phrases)
            else:
                documents[candidate.document_url] = candidate
    return sorted(documents.values(), key=lambda candidate: candidate.document_url)


def stratified_sample(candidates, *, seed=DEFAULT_SEED, quotas=None, as_of=None):
    quotas = ERA_QUOTAS if quotas is None else quotas
    if set(quotas) != set(ERA_QUOTAS) or any(not isinstance(n, int) or n < 0 for n in quotas.values()):
        raise ValueError("quotas must give a nonnegative count for each era")
    buckets = {era: {} for era in ERA_QUOTAS}
    for candidate in candidates:
        era = era_for(candidate.filing_date, as_of=as_of)
        if era:
            buckets[era][candidate.document_url] = candidate
    shortages = [f"{era}: need {quotas[era]}, have {len(buckets[era])}"
                 for era in ERA_QUOTAS if len(buckets[era]) < quotas[era]]
    if shortages:
        raise ValueError("insufficient candidates for stratified sample; " + "; ".join(shortages))
    rng, selected = Random(seed), []
    for era in ERA_QUOTAS:
        pool = sorted(buckets[era].values(), key=lambda candidate: candidate.document_url)
        selected.extend(rng.sample(pool, quotas[era]))
    rng.shuffle(selected)
    return selected


def score_document(text: str, page_count: int) -> dict:
    if page_count < 1:
        raise ValueError("document must have a positive page count")
    normalized = " ".join(text.split())
    flags = {name: bool(re.search(pattern, normalized, re.I)) for name, pattern in TERMS.items()}
    matches = sum(flags.values())
    short = page_count < 40
    label = "likely_confirmation" if matches == 4 and short else "maybe" if matches >= 2 else "unlikely"
    return {**flags, "under_40_pages": short, "score": matches + int(short), "label": label}


def count_pages(path: str | Path, text: str) -> tuple[int, str]:
    """Native PDF pages; conservative, reproducible print layout for HTML/TXT.

    Non-PDF source documents have no intrinsic page count. Estimate from the
    ingested text, not untrusted HTML/CSS/images that could fetch remote URLs.
    """
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        import pymupdf
        with pymupdf.open(path) as document:
            return document.page_count, "pdf"
    from weasyprint import HTML

    def no_resources(url, *args, **kwargs):
        raise ValueError("screen page estimation forbids external resources")

    html = ('<!doctype html><meta charset="utf-8"><style>'
            '@page { size: A4; margin: 16mm; } '
            'body { font-family: "DejaVu Serif", serif; font-size: 10pt; line-height: 1.3; '
            'font-variant-ligatures: none; } '
            'pre { font: inherit; white-space: pre-wrap; overflow-wrap: anywhere; }'
            '</style><pre>' + escape(text) + '</pre>')
    return len(HTML(string=html, url_fetcher=no_resources).render().pages), "estimated_text_layout"


def _ingest_or_reuse(candidate, raw_path, root, existing_records):
    doc_id = "real_" + hashlib.sha256(candidate.document_url.encode()).hexdigest()[:24]
    text_path = root / "real/text" / f"{doc_id}.txt"
    sidecar_path = text_path.with_suffix(".offsets.json")
    records = [record for record in existing_records if record.doc_id == doc_id]
    if records:
        if len(records) != 1:
            raise ValueError("duplicate ingested document ID")
        record = records[0]
        if record.collection != "real" or record.source_url != candidate.document_url or record.split is not None:
            raise ValueError("existing document is not an unassigned candidate; refusing split access")
        text, meta = to_text(raw_path)
        if (record.sha256 != meta["sha256"] or not text_path.is_file() or not sidecar_path.is_file()
                or text_path.read_text(encoding="utf-8") != text
                or Path(record.text_path).resolve() != text_path.resolve()):
            raise ValueError("existing ingested artifacts differ; refusing overwrite")
        offsets = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if offsets != {"source_text": meta["source_text"], "offset_map": [list(pair) for pair in meta["offset_map"]]}:
            raise ValueError("existing offset map differs; refusing overwrite")
        return text, {**meta, **record.model_dump()}
    text, meta = ingest_document(raw_path, collection="real", source_url=candidate.document_url,
                                 doc_id=doc_id, output_dir=root)
    existing_records.append(DocumentRecord.model_validate({key: value for key, value in meta.items()
                                                         if key in DocumentRecord.model_fields}))
    return text, meta


def screen_candidates(client, candidates, *, data_root="data", raw_dir=None, as_of=None):
    root = Path(data_root)
    raw_dir = Path(raw_dir) if raw_dir is not None else root / "raw/edgar"
    # Shared Step 3A client supplies the User-Agent, rate limit and SHA-256 manifest.
    edgar_scope.download_candidates(client, candidates, len(candidates), raw_dir)
    records_path = root / "real/docs.jsonl"
    records = [DocumentRecord.model_validate_json(line) for line in
               records_path.read_text(encoding="utf-8").splitlines()] if records_path.exists() else []
    rows = []
    for candidate in candidates:
        raw_path = raw_dir / f"{candidate.cik}_{candidate.accession_number}_{candidate.exhibit_file_name}"
        text, meta = _ingest_or_reuse(candidate, raw_path, root, records)
        pages, basis = count_pages(raw_path, text)
        rows.append({**candidate.row(), "doc_id": meta["doc_id"], "era": era_for(candidate.filing_date, as_of=as_of),
                     "raw_path": str(raw_path), "sha256": meta["sha256"], "text_path": meta["text_path"],
                     "ocr": meta["ocr"], "page_count": pages, "page_count_basis": basis,
                     **score_document(text, pages)})
    return rows


def manual_sample(rows, *, seed=DEFAULT_SEED, per_label=20):
    """Exactly per_label unique rows per label; never duplicate or invent labels."""
    if per_label < 1:
        raise ValueError("per_label must be positive")
    rng, selected = Random(seed), []
    for label in LABELS:
        pool = sorted((row for row in rows if row["label"] == label), key=lambda row: row["document_url"])
        if len({row["document_url"] for row in pool}) < per_label:
            raise ValueError(f"manual sample needs {per_label} {label} rows; only {len(pool)} available. "
                             "screen.csv is retained; manual_check.csv was not changed")
        # Screening rows are unique by construction; guard direct callers too.
        unique = {row["document_url"]: row for row in pool}
        selected.extend({**row, "is_confirmation": ""} for row in rng.sample(list(unique.values()), per_label))
    rng.shuffle(selected)
    return selected


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def print_summary(rows):
    print(f"Screened documents: {len(rows)}")
    labels, eras = Counter(row["label"] for row in rows), Counter(row["era"] for row in rows)
    print("Count per label:")
    for label in LABELS:
        print(f"  {label}: {labels[label]}")
    print("Count per era:")
    for era in ERA_QUOTAS:
        print(f"  {era}: {eras[era]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args(argv)
    manual_path = args.data_root / "real/manual_check.csv"
    try:
        client = edgar_scope.SECClient()
        # Protect any existing human work, including an as-yet-unfilled review CSV.
        if manual_path.exists():
            raise FileExistsError(f"manual review file already exists; preserve it before rerunning: {manual_path}")
        candidates = read_candidates(args.candidates or args.data_root / "real/candidates.csv")
        selected = stratified_sample(candidates, seed=args.seed)
        rows = screen_candidates(client, selected, data_root=args.data_root, raw_dir=args.raw_dir)
        write_csv(args.data_root / "real/screen.csv", rows, SCREEN_COLUMNS)
        print_summary(rows)
        manual = manual_sample(rows, seed=args.seed)
        write_csv(manual_path, manual, (*SCREEN_COLUMNS, "is_confirmation"))
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
