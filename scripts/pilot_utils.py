"""Local pilot I/O shared by the annotation command-line tools."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

from trustswap.common.annotations import AnnotationRecord


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("malformed CSV")
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
