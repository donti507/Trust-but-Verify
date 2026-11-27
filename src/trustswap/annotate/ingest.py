"""Import an explicit local manifest, with no network or model processing."""
import csv
from pathlib import Path
from urllib.parse import urlsplit

from trustswap.annotate import store
from trustswap.common.hashing import sha256_bytes
from trustswap.ingest.to_text import to_text

FORMATS = (".html", ".htm", ".txt", ".pdf")


def _path(root, manifest, raw, doc):
    if not raw:
        matches = [root / f"{doc}{suffix}" for suffix in FORMATS if (root / f"{doc}{suffix}").is_file()]
        if len(matches) != 1:
            raise ValueError(f"{doc} needs an unambiguous local filename in the manifest")
        return matches[0].resolve()
    hints = (root / raw, manifest.parent / raw, Path(raw))
    for hint in hints:
        target = hint.resolve()
        if target.is_relative_to(root) and target.is_file():
            return target
    raise ValueError(f"{doc} is missing or its file is outside the raw directory")


def import_documents(db, directory, manifest, *, random_seed=None):
    root, manifest = Path(directory).resolve(), Path(manifest).resolve()
    if not root.is_dir():
        raise ValueError("The raw document directory does not exist")
    with manifest.open(encoding="utf-8-sig", newline="") as source:
        entries = csv.DictReader(source)
        if "doc_id" not in (entries.fieldnames or []):
            raise ValueError("The manifest must contain doc_id")
        rows = list(entries)
    prepared = {}
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("The manifest has an invalid CSV row")
        # Skip locked metadata before resolving or reading a raw document.
        if row.get("split", "").strip().casefold() == "real_test":
            continue
        if row.get("collection", "").strip() not in ("", "real"):
            raise ValueError("Only the real collection belongs in this annotation tool")
        doc = row["doc_id"].strip()
        if not store.DOC_ID.fullmatch(doc) or doc in prepared:
            raise ValueError("Invalid or duplicate manifest document ID")
        raw = next((row[key].strip() for key in ("raw_path", "local_path", "file_path", "filename", "path")
                    if row.get(key, "").strip()), "")
        path = _path(root, manifest, raw, doc)
        if not path.is_relative_to(root) or path.suffix.lower() not in FORMATS:
            raise ValueError("Choose a local HTML, text or PDF document inside the raw directory")
        if "real_test" in {part.casefold() for part in path.relative_to(root).parts}:
            continue
        url = next((row[key].strip() for key in ("resolved_source_url", "source_url", "document_url", "url")
                    if row.get(key, "").strip()), None)
        if url and (urlsplit(url).scheme not in {"https", "http"} or not urlsplit(url).hostname):
            raise ValueError("Source URLs must be HTTP(S) provenance")
        text, meta = to_text(path, ocr_enabled=False)
        expected = (row.get("sha256") or "").strip().lower()
        if expected and expected != meta["sha256"]:
            raise ValueError(f"The manifest SHA-256 does not match {doc}")
        if not text:
            raise ValueError(f"{doc} has no visible text")
        prepared[doc] = (meta["sha256"], sha256_bytes(text.encode()), url, text, store.dump(meta["offset_map"]))
    added = 0
    with store.transaction(db):
        store.seed(db, store.seed(db) if random_seed is None else random_seed)
        for doc, content in sorted(prepared.items()):
            prior = db.execute("SELECT raw_hash,text_hash,url FROM documents WHERE id=?", (doc,)).fetchone()
            if prior is not None:
                if tuple(prior) != content[:3]:
                    raise ValueError(f"Source changed for {doc}. Use a new document ID")
                continue
            if store.sealed(db):
                raise ValueError("The cohort is locked. Use a fresh database to add documents")
            db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?)", (doc, *content))
            added += 1
    return added
