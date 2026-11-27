"""Read fictional confirmation uploads and attach them to the desk's bookings."""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from trustswap.desk import imports, matcher, store, upload_review
from trustswap.desk.formats import fpml, read_terms, swift
from trustswap.ingest.to_text import normalise_whitespace, to_text

MAX_BYTES = 10 * 1024 * 1024
MAX_TEXT = 300_000
MAX_PIXELS = 25_000_000
ROOT = Path(__file__).resolve().parents[3]
STRUCTURED = {"swift", "fpml"}
SUFFIXES = {"pdf": ".pdf", "png": ".png", "jpeg": ".jpg", "swift": ".fin", "fpml": ".xml", "text": ".txt"}


@dataclass(frozen=True)
class Upload:
    kind: str
    digest: str
    file: Path
    text: str | None
    fields: dict | None
    reference: str | None


def upload_directory(value=None):
    configured = value if value is not None else os.environ.get("DESK_UPLOAD_DIR")
    if not configured:
        raise ValueError("Set DESK_UPLOAD_DIR to a private directory outside the checkout")
    directory = Path(configured).expanduser().resolve()
    if directory.is_relative_to(ROOT) or ROOT.is_relative_to(directory) or any(
        (parent / ".git").exists() for parent in (directory, *directory.parents)
    ):
        raise ValueError("Desk uploads must be stored outside Git checkouts")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


def sniff(payload, filename):
    if not payload or len(payload) > MAX_BYTES:
        raise ValueError("Choose a nonempty confirmation of at most 10 MB")
    ext = Path(filename or "").suffix.lower()
    for kind, magic, extensions in (
        ("pdf", b"%PDF-", {".pdf"}),
        ("png", b"\x89PNG\r\n\x1a\n", {".png"}),
        ("jpeg", b"\xff\xd8\xff", {".jpg", ".jpeg"}),
    ):
        if payload.startswith(magic):
            if ext not in extensions:
                raise ValueError("The file contents do not match its extension")
            return kind, None
    if ext not in {".txt", ".fin", ".xml"}:
        raise ValueError("Choose a PDF, PNG, JPEG, TXT, FIN or FpML XML file with a matching extension")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("Text, FIN and XML uploads must use UTF-8") from None
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", text) or not text.strip():
        raise ValueError("The upload is not a text document")
    if ext == ".xml" and text.lstrip().startswith("<"):
        return "fpml", text
    if ext in {".fin", ".txt"} and swift.MESSAGE.search(text):
        return "swift", text
    if ext == ".txt" and not text.lstrip().startswith(("<", "{1:", "{2:", "%PDF-")):
        return "text", text
    raise ValueError("The file contents do not match its extension or are not a supported format")


def _atomic_write(path, payload):
    if path.is_symlink():
        raise ValueError("The configured upload directory contains a symbolic link")
    if path.exists():
        return
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        scratch = Path(stream.name)
        try:
            stream.write(payload)
            stream.close()
            scratch.replace(path)
        finally:
            scratch.unlink(missing_ok=True)


def _ocr_image(path, kind):
    from PIL import Image
    import pytesseract

    with Image.open(path) as image:
        if image.format != {"png": "PNG", "jpeg": "JPEG"}[kind] or image.width * image.height > MAX_PIXELS:
            raise ValueError("The scanned image has an unsupported format or size")
        return pytesseract.image_to_string(image, timeout=30)


def _pdf(path):
    import pymupdf

    with pymupdf.open(path, filetype="pdf") as pdf:
        if pdf.is_encrypted or not 0 < len(pdf) <= 100:
            raise ValueError("Choose an unencrypted confirmation of at most 100 pages")
        if any(page.rect.width * page.rect.height * (300 / 72) ** 2 > MAX_PIXELS for page in pdf):
            raise ValueError("The PDF page size exceeds the desk scan limit")
    return to_text(path)


def _check_fictional(fields):
    for path, field in fields.items():
        try:
            imports._fictional(path, imports._canonical(path, field["confirmation"]))
        except ValueError:
            raise ValueError("Upload a confirmation with the desk's fictional parties and example addresses") from None


def read_upload(payload, filename):
    kind, text = sniff(payload, filename)
    digest = sha256(payload).hexdigest()
    directory = upload_directory()
    file = directory / f"{digest}{SUFFIXES[kind]}"
    if kind in STRUCTURED:
        try:
            parsed = (fpml if kind == "fpml" else swift).parse(text)
            fields = upload_review.rows_for(parsed.trade, parsed.fields)
        except (ValueError, ArithmeticError, RuntimeError, KeyError):
            raise ValueError("This confirmation does not contain a supported, safe swap") from None
        _check_fictional(fields)
        _atomic_write(file, payload)
        return Upload(kind, digest, file, text, fields, parsed.reference)
    # Saving the upload is local even when extraction is disabled.
    _atomic_write(file, payload)
    if not upload_review.allow_external():
        return Upload(kind, digest, file, None, None, None)
    ocr = False
    try:
        if kind == "pdf":
            text, metadata = _pdf(file)
            ocr = metadata["ocr"]
        elif kind in {"png", "jpeg"}:
            text, _ = normalise_whitespace(_ocr_image(file, kind))
            ocr = True
        else:
            text, _ = normalise_whitespace(text)
    except Exception:
        raise ValueError("The desk could not read the document or scan") from None
    if not text or len(text) > MAX_TEXT:
        raise ValueError("The document has no readable text or exceeds the extraction limit")
    fields = upload_review.scorer(directory)(f"desk-{digest[:32]}", text, ocr)
    _check_fictional(fields)
    return Upload(kind, digest, file, text, fields, None)


def _find_trade(db, reference, fields):
    by_trade = {}
    for row in db.execute("SELECT trade_id,path,booking FROM fields"):
        by_trade.setdefault(row["trade_id"], {})[row["path"]] = json.loads(row["booking"])
    booked = [{"trade_id": trade_id, "fields": values} for trade_id, values in by_trade.items()]
    return matcher.link_booking(reference, {path: row["confirmation"] for path, row in fields.items()}, booked)


def attach(db, upload, actor, requested_id=None):
    reference = requested_id or upload.reference or f"desk-{upload.digest[:20]}"
    try:
        imports._trade_id(reference)
    except ValueError:
        raise ValueError("Enter a valid booking trade ID") from None
    with store.transaction(db):
        matched = _find_trade(db, reference, upload.fields)
        trade_id = matched["trade_id"] if matched else reference
        current = db.execute("SELECT sha256 FROM desk_uploads WHERE trade_id=?", (trade_id,)).fetchone()
        if current and current["sha256"] == upload.digest:
            return trade_id
        trade = db.execute("SELECT assigned_to,state FROM trades WHERE id=?", (trade_id,)).fetchone()
        if trade is None:
            store.new_trade(db, trade_id, "synthetic", upload.text, upload.fields)
        else:
            if trade["assigned_to"] not in {None, actor} and db.execute(
                "SELECT role FROM users WHERE id=?", (actor,)
            ).fetchone()[0] != "supervisor":
                raise ValueError("This trade is assigned to another analyst")
            states = db.execute("SELECT path,state FROM fields WHERE trade_id=?", (trade_id,)).fetchall()
            if trade["state"] != "Open" or any(row["state"] != "open" for row in states):
                raise ValueError("The booking already has review decisions")
            db.execute("UPDATE trades SET source=? WHERE id=?", (upload.text, trade_id))
            db.execute("DELETE FROM field_evidence WHERE trade_id=?", (trade_id,))
            db.execute("UPDATE fields SET confirmation='null',p=0 WHERE trade_id=?", (trade_id,))
            for path, row in upload.fields.items():
                db.execute("""INSERT INTO fields(trade_id,path,confirmation,booking,p,c,r)
                           VALUES (?,?,?,'null',?,?,?)
                           ON CONFLICT (trade_id,path) DO UPDATE SET
                           confirmation=excluded.confirmation,p=excluded.p,c=excluded.c,r=excluded.r""",
                           (trade_id, path, json.dumps(row["confirmation"]), row["p"], row["c"], row["r"]))
            store.save_evidence(db, trade_id, upload.fields)
        db.execute("""INSERT INTO desk_uploads(trade_id,sha256,kind) VALUES (?,?,?)
                   ON CONFLICT(trade_id) DO UPDATE SET sha256=excluded.sha256,kind=excluded.kind""",
                   (trade_id, upload.digest, upload.kind))
        store.event(db, actor, "confirmation_uploaded", trade_id, sha256=upload.digest, kind=upload.kind,
                    matched_booking=matched is not None)
    return trade_id


def export(db, trade_id, kind):
    terms = {}
    for row in db.execute("SELECT * FROM fields WHERE trade_id=?", (trade_id,)):
        value = row["booking"] if row["state"] == "approved" else row["confirmation"]
        terms[row["path"]] = json.loads(value)
    for path, term in terms.items():
        imports._fictional(path, term)
    trade = read_terms(trade_id, terms, "").trade
    exporter = fpml if kind == "fpml" else swift
    try:
        return exporter.export(trade_id, trade)
    except (ArithmeticError, KeyError, RuntimeError):
        raise ValueError("This affirmed trade is missing terms required for export") from None
