"""Local document ingestion with offsets into extracted, visible text.

Offsets are Unicode character positions, not byte positions or HTML/PDF offsets.
Each saved character maps to a half-open range in the pre-normalisation text.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

from trustswap.common.hashing import sha256_bytes
from trustswap.common.records import AdversarialMetadata, DocumentRecord


def normalise_whitespace(source: str) -> tuple[str, list[tuple[int, int]]]:
    """Collapse horizontal whitespace and blank lines, retaining line boundaries."""
    output: list[str] = []
    offsets: list[tuple[int, int]] = []
    previous_end = 0
    for token in re.finditer(r"\S+", source):
        if output:
            gap = source[previous_end:token.start()]
            if gap:
                output.append("\n" if "\n" in gap or "\r" in gap else " ")
                offsets.append((previous_end, token.start()))
        output.append(token.group())
        offsets.extend((index, index + 1) for index in range(token.start(), token.end()))
        previous_end = token.end()
    return "".join(output), offsets


def _html_text(data: bytes | str) -> str:
    from bs4 import BeautifulSoup, NavigableString

    soup = BeautifulSoup(data, "html.parser")
    for hidden in soup.select('script, style, template, head, [hidden], [aria-hidden="true"]'):
        hidden.decompose()
    # Replace complete rows, so inline markup inside a cell never inserts a delimiter.
    for row in list(soup.find_all("tr")):
        cells = row.find_all(["td", "th"], recursive=False)
        row.replace_with(NavigableString("\n" + " | ".join(
            " ".join(cell.stripped_strings) for cell in cells
        ) + "\n"))
    for br in soup.find_all("br"):
        br.replace_with(NavigableString("\n"))
    for block in soup.find_all(["p", "div", "section", "h1", "h2", "h3", "li"]):
        block.insert_before(NavigableString("\n"))
        block.insert_after(NavigableString("\n"))
    return soup.get_text()


def _decode_text(data: bytes) -> tuple[str, str]:
    """Try strict decoders in order; Latin-1 handles every remaining byte."""
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        # Preserve the previous handling of UTF-8 byte-order marks.
        if encoding == "utf-8":
            text = text.removeprefix("\ufeff")
        return text, encoding
    raise AssertionError("Latin-1 must decode every byte")


def _edgar_text(text: str) -> str:
    """Keep each SGML TEXT payload in order, excluding filing headers."""
    if not re.search(r"<DOCUMENT\s*>", text, re.I) or not re.search(r"<TEXT\s*>", text, re.I):
        return text
    payloads = re.findall(r"<TEXT\s*>(.*?)</TEXT\s*>", text, re.I | re.S)
    if not payloads:
        raise ValueError("EDGAR filing has no complete TEXT section")
    return "\n".join(
        _html_text(payload) if re.search(r"</?[A-Za-z][^>]*>", payload) else payload
        for payload in payloads
    )


def _pdf_text(data: bytes) -> tuple[str, bool]:
    import pdfplumber
    import pymupdf

    plumber = None
    try:
        try:
            plumber = pdfplumber.open(io.BytesIO(data))
        except Exception:
            # A parser failure must still allow the independent PyMuPDF reader.
            pass
        pages = []
        ocr = False
        with pymupdf.open(stream=data, filetype="pdf") as document:
            for index, page in enumerate(document):
                text = ""
                if plumber is not None:
                    try:
                        text = plumber.pages[index].extract_text() or ""
                    except Exception:
                        pass
                if not text.strip():
                    text = page.get_text("text")
                if not text.strip():
                    import pytesseract
                    from PIL import Image

                    pixmap = page.get_pixmap(dpi=300, alpha=False, colorspace=pymupdf.csRGB)
                    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                    text = pytesseract.image_to_string(image)
                    ocr = True
                pages.append(text)
        return "\n\n".join(pages), ocr
    finally:
        if plumber is not None:
            plumber.close()


def to_text(path: str | Path) -> tuple[str, dict]:
    """Read a document without modifying it; return text and provenance metadata."""
    path = Path(path)
    data = path.read_bytes()
    suffix = path.suffix.lower()
    ocr = False
    encoding = None
    if suffix in {".html", ".htm"}:
        source = _html_text(data)
    elif suffix == ".txt":
        source, encoding = _decode_text(data)
        source = _edgar_text(source)
    elif suffix == ".pdf":
        source, ocr = _pdf_text(data)
    else:
        raise ValueError(f"Unsupported document extension: {suffix}")
    text, offsets = normalise_whitespace(source)
    return text, {"sha256": sha256_bytes(data), "ocr": ocr, "encoding": encoding,
                  "offset_map": offsets, "source_text": source}


def _protect_raw(path: Path) -> None:
    resolved = path.resolve()
    if any(parent.name == "raw" and parent.parent.name == "data"
           for parent in (resolved, *resolved.parents)):
        raise ValueError("Output must not be inside data/raw/")


def ingest_document(path: str | Path, *, collection: str, output_dir: str | Path = "data",
                    doc_id: str | None = None, source_url: str | None = None,
                    parent_id: str | None = None, defect: str | None = None,
                    family_id: str | None = None, split: str | None = None,
                    adversarial: AdversarialMetadata | None = None) -> tuple[str, dict]:
    """Save text and offsets, then append a DocumentRecord to collection/docs.jsonl.

    output_dir is the data root. Input and existing text/offset files are never
    overwritten. split and family_id remain null until later runbook steps.
    Returned metadata additionally includes source_text and offset_map.
    """
    if collection not in {"real", "synthetic", "adversarial"}:
        raise ValueError("collection must be real, synthetic or adversarial; unsafe path components")
    text, meta = to_text(path)
    doc_id = f"{collection}_{meta['sha256']}" if doc_id is None else doc_id
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", doc_id):
        raise ValueError("doc_id must be a safe, nonempty path component")
    directory = Path(output_dir) / collection
    text_path = directory / "text" / f"{doc_id}.txt"
    record_path = directory / "docs.jsonl"
    offsets_path = text_path.with_suffix(".offsets.json")
    for target in (text_path, offsets_path, record_path):
        _protect_raw(target)
        if target.resolve() == Path(path).resolve():
            raise ValueError("Output must not overwrite the source")
    for target in (text_path, offsets_path):
        if target.exists():
            raise FileExistsError(target)
    extra = adversarial.model_dump() if adversarial else {}
    if adversarial:
        parent_id, defect = extra.pop("parent_id"), extra.pop("defect")
    record = DocumentRecord(doc_id=doc_id, collection=collection, sha256=meta["sha256"],
                            text_path=str(text_path), ocr=meta["ocr"], source_url=source_url,
                            parent_id=parent_id, defect=defect, family_id=family_id, split=split, **extra)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    with text_path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
    with offsets_path.open("x", encoding="utf-8") as handle:
        json.dump({"source_text": meta["source_text"], "offset_map": meta["offset_map"]}, handle, ensure_ascii=False)
    with record_path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")
    return text, {**meta, **record.model_dump()}
