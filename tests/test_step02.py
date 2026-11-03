"""Step 2: real local fixtures, extraction, offsets and immutable inputs."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from trustswap.ingest.to_text import ingest_document, normalise_whitespace, to_text

FIXTURES = Path(__file__).parent / "fixtures"
PHRASES = ("Notional Amount", "USD 100,000,000")


@pytest.fixture
def text_pdf(tmp_path):
    path = tmp_path / "confirmation.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.drawString(72, 740, "Notional Amount")
    pdf.drawString(72, 720, "USD 100,000,000")
    pdf.save()
    return path


def assert_spans(text, meta):
    assert len(meta["offset_map"]) == len(text)
    for phrase in PHRASES:
        start = text.index(phrase)
        end = start + len(phrase)
        assert text[start:end] == phrase
        source_start = meta["offset_map"][start][0]
        source_end = meta["offset_map"][end - 1][1]
        assert normalise_whitespace(meta["source_text"][source_start:source_end])[0] == phrase
    for character, (start, end) in zip(text, meta["offset_map"]):
        assert 0 <= start < end <= len(meta["source_text"])
        source = meta["source_text"][start:end]
        assert source.isspace() if character.isspace() else source == character


@pytest.mark.parametrize("kind", ["html", "txt", "pdf"])
def test_ingest_document(kind, tmp_path, text_pdf):
    source = text_pdf if kind == "pdf" else FIXTURES / f"confirmation.{kind}"
    original = source.read_bytes()
    text, meta = ingest_document(source, collection="synthetic", output_dir=tmp_path / "text")
    assert_spans(text, meta)
    assert source.read_bytes() == original
    assert meta["sha256"] == hashlib.sha256(original).hexdigest()
    assert meta["doc_id"] == "synthetic_" + meta["sha256"]
    assert meta["ocr"] is False
    assert Path(meta["text_path"]).read_text(encoding="utf-8") == text
    record = json.loads((Path(meta["text_path"]).parent.parent / "docs.jsonl").read_text())
    assert record == {key: meta[key] for key in ("doc_id", "collection", "family_id", "split", "source_url", "sha256", "text_path", "ocr", "parent_id", "defect")}
    assert record["split"] is None and record["family_id"] is None
    sidecar = json.loads(Path(meta["text_path"]).with_suffix(".offsets.json").read_text())
    assert sidecar["offset_map"] == [list(pair) for pair in meta["offset_map"]]
    assert sidecar["source_text"] == meta["source_text"]


def test_html_table_rows_and_inline_markup():
    text, _ = to_text(FIXTURES / "confirmation.html")
    assert "Notional Amount | USD 100,000,000\nFixed Rate | 1.80000 percent" in text
    assert "invisible text" not in text
    assert "Ignored title" not in text


@pytest.mark.parametrize("source,expected", [
    ("  A\t B \r\n\n C  ", "A B\nC"), ("", ""), (" \n\t", ""),
    ("é\u00a0£100", "é £100"),
])
def test_whitespace_offsets(source, expected):
    text, offsets = normalise_whitespace(source)
    assert text == expected
    assert len(offsets) == len(text)
    for char, (start, end) in zip(text, offsets):
        assert source[start:end].isspace() if char.isspace() else source[start:end] == char


@pytest.mark.parametrize("failure", ["empty", "page_error", "open_error"])
def test_pymupdf_fallback(text_pdf, monkeypatch, failure):
    import pdfplumber

    if failure == "open_error":
        def broken_open(*args, **kwargs):
            raise ValueError("test parser failure")
        monkeypatch.setattr(pdfplumber, "open", broken_open)
    else:
        def extraction(*args, **kwargs):
            if failure == "page_error":
                raise ValueError("test page failure")
            return ""
        monkeypatch.setattr(pdfplumber.page.Page, "extract_text", extraction)
    text, meta = to_text(text_pdf)
    assert_spans(text, meta)
    assert meta["ocr"] is False


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract is not installed")
@pytest.mark.parametrize("mixed", [False, True])
def test_scanned_page_ocr(tmp_path, mixed):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1800, 400), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=60)
    draw.text((40, 50), "Notional Amount", font=font, fill="black")
    draw.text((40, 170), "USD 100,000,000", font=font, fill="black")
    path = tmp_path / "scanned.pdf"
    pdf = canvas.Canvas(str(path))
    if mixed:
        pdf.drawString(72, 740, "Text layer page")
        pdf.showPage()
    pdf.drawImage(ImageReader(image), 30, 500, width=540, height=120)
    pdf.save()
    text, meta = to_text(path)
    assert meta["ocr"] is True
    assert_spans(text, meta)
    if mixed:
        assert text.startswith("Text layer page\n")


def test_raw_source_is_immutable_and_output_is_refused(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    source = raw / "confirmation.txt"
    original = FIXTURES.joinpath("confirmation.txt").read_bytes()
    source.write_bytes(original)
    ingest_document(source, collection="synthetic", output_dir=tmp_path / "processed")
    with pytest.raises(ValueError, match="data/raw"):
        ingest_document(source, collection="synthetic", output_dir=raw)
    assert source.read_bytes() == original
    assert list(raw.iterdir()) == [source]


def test_outputs_never_overwrite_inputs_or_existing_records(tmp_path):
    source = tmp_path / "synthetic" / "text" / "same.txt"
    source.parent.mkdir(parents=True)
    source.write_text("original")
    with pytest.raises(ValueError, match="source"):
        ingest_document(source, collection="synthetic", output_dir=tmp_path, doc_id="same")
    assert source.read_text() == "original"
    ingest_document(source, collection="real", output_dir=tmp_path)
    with pytest.raises(FileExistsError):
        ingest_document(source, collection="real", output_dir=tmp_path)


@pytest.mark.parametrize("collection,doc_id", [("../raw", "id"), ("real", "../id"), ("", "id")])
def test_unsafe_identifiers(tmp_path, collection, doc_id):
    with pytest.raises(ValueError, match="path component"):
        ingest_document(FIXTURES / "confirmation.txt", collection=collection, doc_id=doc_id, output_dir=tmp_path)


def test_unsupported_format(tmp_path):
    source = tmp_path / "unknown.bin"
    source.write_bytes(b"unknown")
    with pytest.raises(ValueError, match="Unsupported"):
        to_text(source)


def test_jsonl_append_and_collection_isolation(tmp_path):
    source = FIXTURES / "confirmation.txt"
    for collection in ("real", "synthetic", "adversarial"):
        for doc_id in ("first", "second"):
            _, meta = ingest_document(source, collection=collection, output_dir=tmp_path,
                                      doc_id=doc_id,
                                      source_url="https://example.com/confirmation" if collection == "real" else None,
                                      parent_id="source_1" if collection == "adversarial" else None,
                                      defect="changed_rate" if collection == "adversarial" else None)
            assert meta["collection"] == collection
        records = [json.loads(line) for line in (tmp_path / collection / "docs.jsonl").read_text().splitlines()]
        assert [record["doc_id"] for record in records] == ["first", "second"]
        assert all(record["collection"] == collection for record in records)


@pytest.mark.parametrize("updates", [
    {"collection": "unknown"}, {"split": "syn_test"},
    {"collection": "synthetic", "source_url": "https://example.com"},
    {"parent_id": "parent"}, {"defect": "changed_rate"}, {"sha256": "bad"},
])
def test_invalid_document_records(updates):
    from pydantic import ValidationError
    from trustswap.common.records import DocumentRecord

    record = {"doc_id": "real_0042", "collection": "real", "sha256": "a" * 64,
              "text_path": "data/real/text/real_0042.txt", "ocr": False}
    with pytest.raises(ValidationError):
        DocumentRecord.model_validate({**record, **updates})


def test_raw_output_via_symlink_is_refused(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(raw, target_is_directory=True)
    with pytest.raises(ValueError, match="data/raw"):
        ingest_document(FIXTURES / "confirmation.txt", collection="real", output_dir=alias)
    assert not list(raw.iterdir())
