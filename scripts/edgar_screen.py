#!/usr/bin/env python3
"""Offline-testable EDGAR screening; run explicitly to download real candidates.

Screen labels are heuristic triage, never gold labels or split assignments.
Only operators fill manual_check.csv's is_confirmation column.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
from datetime import date
import hashlib
import math
import multiprocessing
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
from trustswap.ingest.to_text import ingest_document

DEFAULT_SEED = 20261026
ERA_QUOTAS = {"2001–2005": 75, "2006–2008": 75, "2009–2021": 75, "2022+": 75}
LABELS = ("likely_confirmation", "maybe", "unlikely")
SCREEN_LABELS = (*LABELS, "needs_ocr", "too_large")
MAX_FILE_BYTES = 15_000_000
SCREEN_TIMEOUT_SECONDS = 60
TERMS = {
    "has_notional": r"\b(?:notional|calculation) amount\b",
    "has_termination": r"\btermination date\b",
    "has_effective": r"\beffective date\b",
    "has_rate": r"\b(?:fixed rate|floating rate option)\b",
}
SCREEN_COLUMNS = (*edgar_scope.COLUMNS, "doc_id", "era", "raw_path", "sha256", "text_path",
                  "download_source", "resolved_source_url", "selected_filename", "ocr", "page_count", "page_count_basis", *TERMS, "under_40_pages", "score", "label")


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
    if page_count < 0:
        raise ValueError("document page count must be nonnegative")
    normalized = " ".join(text.split())
    flags = {name: bool(re.search(pattern, normalized, re.I)) for name, pattern in TERMS.items()}
    matches = sum(flags.values())
    short = page_count < 40
    label = "likely_confirmation" if matches == 4 and short else "maybe" if matches >= 2 else "unlikely"
    return {**flags, "under_40_pages": short, "score": matches + int(short), "label": label}


def count_pages(path: str | Path, text: str) -> tuple[int, str]:
    """No HTML layout/rendering: use native PDF pages or character estimates."""
    if Path(path).suffix.lower() == ".pdf":
        import pymupdf
        with pymupdf.open(path) as document:
            return document.page_count, "pdf"
    return math.ceil(len(text) / 3000), "estimated_chars"


def _doc_id(candidate):
    return "real_" + hashlib.sha256(candidate.document_url.encode()).hexdigest()[:24]


def _records(root):
    path = root / "real/docs.jsonl"
    return [DocumentRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _matching_records(candidate, records):
    return [record for record in records if record.doc_id == _doc_id(candidate)
            or record.source_url == candidate.document_url]


def _resume_entry(candidate, records, manifest, raw_dir):
    """An ingested document must never trigger a new HTTP request, even if raw is missing."""
    record = next(iter(_matching_records(candidate, records)), None)
    if record is None:
        entry = manifest.get(candidate.document_url)
        if entry and Path(entry["path"]).is_file() and Path(entry["path"]).stat().st_size > MAX_FILE_BYTES:
            return entry  # Do not read a huge cached file merely to classify it too_large.
        return None
    if candidate.document_url in manifest:
        return manifest[candidate.document_url]
    source = "full_submission" if record.source_url == edgar_scope.full_submission_url(candidate) else "exhibit"
    return {"document_url": candidate.document_url, "path": str(edgar_scope._raw_path(candidate, raw_dir, source)),
            "sha256": record.sha256, "source": source, "source_url": record.source_url}


def _cached_rows(root):
    path = root / "real/screen.csv"
    if not path.exists():
        return {}
    result = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for original in csv.DictReader(handle):
            row = {key: original.get(key, "") for key in SCREEN_COLUMNS}
            for key in ("page_count", "score"):
                row[key] = int(row[key]) if row[key] else ""
            for key in ("ocr", *TERMS, "under_40_pages"):
                row[key] = row[key].lower() == "true" if row[key] else ""
            result[row["document_url"]] = row
    return result


def _base_row(candidate, entry, as_of):
    return {**candidate.row(), "doc_id": _doc_id(candidate), "era": era_for(candidate.filing_date, as_of=as_of),
            "raw_path": entry["path"], "sha256": entry["sha256"], "text_path": "",
            "download_source": entry.get("source", "exhibit"),
            "resolved_source_url": entry.get("source_url", candidate.document_url),
            "selected_filename": entry.get("selected_filename", candidate.exhibit_file_name), "ocr": False}


def _special_row(base, label, pages="", basis="not_screened"):
    return {**base, "page_count": pages, "page_count_basis": basis, **{key: "" for key in TERMS},
            "under_40_pages": "", "score": "", "label": label}


def _screen_one(payload):
    """Worker-side parsing and scoring; new files are confined to staging."""
    candidate, entry = payload["candidate"], payload["entry"]
    root, staging = Path(payload["root"]), Path(payload["staging"])
    raw_path = Path(entry["path"])
    base = _base_row(candidate, entry, payload["as_of"])
    if raw_path.exists() and raw_path.stat().st_size > MAX_FILE_BYTES:
        return {"row": _special_row(base, "too_large"), "record": None}
    record = payload["record"]
    if record is not None:
        record = DocumentRecord.model_validate(record)
        base["doc_id"] = record.doc_id
        if record.collection != "real" or record.split is not None or record.source_url != base["resolved_source_url"]:
            raise ValueError("existing document is not an unassigned candidate; refusing split access")
        text_path = root / "real/text" / f"{record.doc_id}.txt"
        if Path(record.text_path).resolve() != text_path.resolve() or not text_path.with_suffix(".offsets.json").is_file():
            raise ValueError("existing ingest artifacts are missing or have unexpected paths")
        text = text_path.read_text(encoding="utf-8")  # No source extraction or OCR on resume.
        meta = record.model_dump()
        needs_ocr = record.ocr or (raw_path.suffix.lower() == ".pdf" and not text.strip())
        new_record = None
    else:
        text, meta = ingest_document(raw_path, collection="real", source_url=base["resolved_source_url"],
                                     doc_id=base["doc_id"], output_dir=staging, ocr_enabled=False)
        if meta["sha256"] != entry["sha256"]:
            raise ValueError("source bytes changed since download")
        needs_ocr = meta["needs_ocr"]
        new_record = {key: value for key, value in meta.items() if key in DocumentRecord.model_fields}
    pages, basis = count_pages(raw_path, text)
    if needs_ocr:
        return {"row": _special_row(base, "needs_ocr", pages, basis), "record": None}
    row = {**base, "text_path": str(root / "real/text" / f"{base['doc_id']}.txt"),
           "page_count": pages, "page_count_basis": basis, **score_document(text, pages)}
    return {"row": row, "record": new_record}


def _screen_worker(payload, connection):
    try:
        connection.send(_screen_one(payload))
    except Exception as error:
        connection.send({"error": f"{type(error).__name__}: {error}"})
    finally:
        connection.close()


def _run_document(payload, *, timeout=SCREEN_TIMEOUT_SECONDS, worker=None):
    """Hard process deadline also bounds native parsers; no abandoned worker threads."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=worker or _screen_worker, args=(payload, sender), daemon=True)
    try:
        process.start()
        sender.close()
        if not receiver.poll(timeout):
            raise TimeoutError(f"ingest and screening exceeded {timeout} seconds")
        result = receiver.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()


def _commit_ingest(result, staging, root, records):
    if result["record"] is None:
        return
    record = DocumentRecord.model_validate(result["record"])
    source = staging / "real/text" / f"{record.doc_id}.txt"
    target = root / "real/text" / source.name
    destinations = (target, target.with_suffix(".offsets.json"))
    if any(path.exists() for path in destinations):
        raise FileExistsError("existing ingest files will not be overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    source.with_suffix(".offsets.json").rename(destinations[1])
    record = record.model_copy(update={"text_path": str(target)})
    with (root / "real/docs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")
    records.append(record)


def download_sample(client, selected, pool, *, seed=DEFAULT_SEED, data_root="data", raw_dir=None, as_of=None):
    """Try each unique URL once, replacing terminal failures within its era."""
    root = Path(data_root)
    raw_dir = Path(raw_dir) if raw_dir is not None else root / "raw/edgar"
    records = _records(root)
    manifest_path = raw_dir / "manifest.jsonl"
    manifest = {row["document_url"]: row for row in (json.loads(line) for line in
                manifest_path.read_text(encoding="utf-8").splitlines())} if manifest_path.exists() else {}
    selected_urls = {item.document_url for item in selected}
    reserves = {era: {} for era in ERA_QUOTAS}
    for item in pool:
        era = era_for(item.filing_date, as_of=as_of)
        if era and item.document_url not in selected_urls:
            reserves[era][item.document_url] = item
    rng = Random(seed)
    for era in ERA_QUOTAS:
        ordered = sorted(reserves[era].values(), key=lambda item: item.document_url)
        rng.shuffle(ordered)
        reserves[era] = deque(ordered)
    pending, attempted, successes = deque(selected), set(), []
    report = edgar_scope.DownloadReport()
    while pending:
        item = pending.popleft()
        if item.document_url in attempted:
            continue
        attempted.add(item.document_url)
        resumed = _resume_entry(item, records, manifest, raw_dir)
        result = edgar_scope.DownloadReport(successes={item.document_url: resumed}) if resumed else edgar_scope.download_candidates(
            client, [item], 1, raw_dir, failure_path=root / "real/download_failures.csv")
        report.merge(result)
        if item.document_url in result.successes:
            successes.append(item)
        else:
            era = era_for(item.filing_date, as_of=as_of)
            if era and reserves[era]:
                pending.append(reserves[era].popleft())
    return successes, report


def screen_candidates(client, candidates, *, data_root="data", raw_dir=None, as_of=None, download_report=None,
                      timeout_seconds=SCREEN_TIMEOUT_SECONDS):
    root = Path(data_root)
    raw_dir = Path(raw_dir) if raw_dir is not None else root / "raw/edgar"
    records = _records(root)
    manifest_path = raw_dir / "manifest.jsonl"
    manifest = {row["document_url"]: row for row in (json.loads(line) for line in
                manifest_path.read_text(encoding="utf-8").splitlines())} if manifest_path.exists() else {}
    if download_report is None:
        report = edgar_scope.DownloadReport()
        for candidate in candidates:
            resumed = _resume_entry(candidate, records, manifest, raw_dir)
            report.merge(edgar_scope.DownloadReport(successes={candidate.document_url: resumed}) if resumed else
                         edgar_scope.download_candidates(client, [candidate], 1, raw_dir,
                                                         failure_path=root / "real/download_failures.csv"))
    else:
        report = download_report
    checkpoint = _cached_rows(root)
    rows = []
    (root / "real").mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(candidates, 1):
        try:
            if candidate.document_url not in report.successes:
                continue
            entry = report.successes[candidate.document_url]
            matching = _matching_records(candidate, records)
            if len(matching) > 1:
                raise ValueError("duplicate ingested document ID")
            record = matching[0] if matching else None
            if record:
                if record.split is not None:
                    raise ValueError("existing document is assigned; refusing split access")
                if record.source_url != entry.get("source_url", candidate.document_url) or record.sha256 != entry["sha256"]:
                    raise ValueError("existing document provenance does not match the download")
                text_path = root / "real/text" / f"{record.doc_id}.txt"
                if (Path(record.text_path).resolve() != text_path.resolve() or not text_path.is_file()
                        or not text_path.with_suffix(".offsets.json").is_file()):
                    raise ValueError("existing ingest artifacts are missing or have unexpected paths")
            cached = checkpoint.get(candidate.document_url)
            raw_path = Path(entry["path"])
            if raw_path.exists() and raw_path.stat().st_size > MAX_FILE_BYTES:
                base = _base_row(candidate, entry, as_of)
                if record:
                    base["doc_id"] = record.doc_id
                row = _special_row(base, "too_large")
            elif (cached and cached["sha256"] == entry["sha256"] and cached["ocr"] is False
                  and (not record or cached["doc_id"] == record.doc_id)
                  and (not record or not record.ocr or cached["label"] == "needs_ocr")
                  and cached["page_count_basis"] in {"estimated_chars", "pdf", "not_screened"}
                  and (record or cached["label"] in {"needs_ocr", "too_large"})):
                row = cached
            else:
                with tempfile.TemporaryDirectory(prefix=".screen-", dir=root / "real") as directory:
                    staging = Path(directory)
                    payload = {"candidate": candidate, "entry": entry, "root": str(root), "staging": str(staging),
                               "record": record.model_dump() if record else None, "as_of": as_of}
                    result = _run_document(payload, timeout=timeout_seconds)
                    _commit_ingest(result, staging, root, records)
                    row = result["row"]
            rows.append(row)
            checkpoint[candidate.document_url] = row
            write_csv(root / "real/screen.csv", list(checkpoint.values()), SCREEN_COLUMNS)
        except Exception as error:
            edgar_scope._log_failure(root / "real/screen_failures.csv", candidate.document_url, error,
                                    status="timeout" if isinstance(error, TimeoutError) else "error")
        finally:
            if index % 10 == 0:
                print(f"Screening progress: {index}/{len(candidates)} documents; {len(rows)} results", flush=True)
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
    for label in SCREEN_LABELS:
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
    report = None
    try:
        client = edgar_scope.SECClient()
        candidates = read_candidates(args.candidates or args.data_root / "real/candidates.csv")
        selected = stratified_sample(candidates, seed=args.seed)
        successful, report = download_sample(client, selected, candidates, seed=args.seed,
                                             data_root=args.data_root, raw_dir=args.raw_dir)
        rows = screen_candidates(client, successful, data_root=args.data_root, raw_dir=args.raw_dir,
                                 download_report=report)
        write_csv(args.data_root / "real/screen.csv", rows, SCREEN_COLUMNS)
        print_summary(rows)
        if len(successful) != len(selected):
            target = Counter(era_for(item.filing_date) for item in selected)
            actual = Counter(row["era"] for row in rows)
            shortage = "; ".join(f"{era}: {actual[era]}/{target[era]}" for era in ERA_QUOTAS if actual[era] < target[era])
            raise ValueError(f"same-era replacement pool exhausted ({shortage}); successful screen.csv retained")
        if manual_path.exists():
            print(f"Keeping existing manual review file: {manual_path}")
        else:
            manual = manual_sample(rows, seed=args.seed)
            write_csv(manual_path, manual, (*SCREEN_COLUMNS, "is_confirmation"))
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    finally:
        if report is not None:
            report.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
