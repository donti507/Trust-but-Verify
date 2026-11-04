"""Local pilot I/O shared by the annotation command-line tools."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

from trustswap.common.annotations import AnnotationRecord


def read_csv_snapshot(path):
    raw = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    columns = list(reader.fieldnames or [])
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("malformed CSV")
    return columns, rows, hashlib.sha256(raw).hexdigest()


def read_csv(path):
    columns, rows, _ = read_csv_snapshot(path)
    return columns, rows


def atomic_text(path, text, *, expected=None, check=False):
    path = Path(path)
    def guard():
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if check and digest != expected:
            raise RuntimeError("output changed on disk; refusing to overwrite concurrent edits")
    guard()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        guard()
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, columns, rows, **options):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return atomic_text(path, buffer.getvalue(), **options)


def read_annotations(path):
    path = Path(path)
    if not path.exists():
        return {}, None
    raw = path.read_bytes()
    records = {}
    for line in raw.decode("utf-8").splitlines():
        record = AnnotationRecord.model_validate_json(line)
        if record.doc_id in records:
            raise ValueError("duplicate annotation document")
        records[record.doc_id] = record
    return records, hashlib.sha256(raw).hexdigest()


def save_annotations(path, records, expected):
    text = "".join(record.model_dump_json() + "\n" for record in records.values())
    return atomic_text(path, text, expected=expected, check=True)


def rejection_path(pilot_path):
    return Path(pilot_path).parent / "pilot_rejections.csv"


def rejection_history(pilot_path):
    path = rejection_path(pilot_path)
    return read_csv(path)[1] if path.exists() else []


def archive_rejections(pilot_path, rows):
    """Preserve removed rows and reasons; never delete their annotation files."""
    path = rejection_path(pilot_path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    old_columns, old_rows = read_csv(path) if path.exists() else ([], [])
    combined = {row["doc_id"]: row for row in old_rows}
    for row in rows:
        combined[row["doc_id"]] = {**combined.get(row["doc_id"], {}), **row}
    columns = list(dict.fromkeys([*old_columns, *(key for row in combined.values() for key in row)]))
    if combined:
        write_csv(path, columns, [{key: row.get(key, "") for key in columns} for row in combined.values()],
                  expected=expected, check=True)
