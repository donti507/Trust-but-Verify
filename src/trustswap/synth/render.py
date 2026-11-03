"""Render structured synthetic trades with spans anchored to PDF glyphs.

Hidden markers are link annotations, not visible token strings. Their rectangles
identify occurrences even when both legs contain identical amounts or parties.
Annotations are removed before ingestion of the final PDF.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import io
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
import pdfplumber
import pymupdf
from weasyprint import HTML

from trustswap.common.records import DocumentRecord, GoldRecord
from trustswap.common.schema import ExtractedField, InterestRateSwap
from trustswap.ingest.to_text import ingest_document, normalise_whitespace, to_text, _protect_raw
from trustswap.synth.gold import write_gold_record
from trustswap.synth.templates import FAMILY_SPLITS

DISCLAIMER = "SYNTHETIC DOCUMENT: fictional parties, generated for research"
MARKER_PREFIX = "urn:trustswap:field:"


@dataclass(frozen=True)
class RenderResult:
    html_path: Path
    pdf_path: Path
    text: str
    document: DocumentRecord
    gold: GoldRecord


def trade_fields(trade: InterestRateSwap) -> dict[str, ExtractedField]:
    """Return term paths in schema order, omitting absent leg containers."""
    result = {}
    for name in type(trade).model_fields:
        value = getattr(trade, name)
        if isinstance(value, ExtractedField):
            result[name] = value
        elif value is not None:
            for term in type(value).model_fields:
                result[f"{name}.{term}"] = getattr(value, term)
    return result


_LABELS = {
    "trade_date": ("Trade Date", "Transaction Date"),
    "effective_date": ("Effective Date", "Commencement Date"),
    "termination_date": ("Termination Date", "Maturity Date"),
    "payer": ("Payer", "Paying Party"), "receiver": ("Receiver", "Receiving Party"),
    "notional_amount": ("Notional Amount", "Notional", "Calculation Amount"),
    "notional_currency": ("Notional Currency", "Currency"),
    "payment_frequency": ("Payment Frequency", "Payment Interval"),
    "reset_frequency": ("Reset Frequency", "Rate Reset Interval"),
    "day_count_fraction": ("Day Count Fraction", "Day Count Basis"),
    "business_day_convention": ("Business Day Convention", "Date Adjustment"),
    "business_centers": ("Business Centers", "Financial Centres"),
    "fixed_rate": ("Fixed Rate", "Fixed Interest Rate"),
    "floating_rate_option": ("Floating Rate Option", "Reference Index"),
    "spread": ("Spread", "Margin"), "notice_address": ("Notice Address", "Address for Notices"),
}


def _display(name: str, value, currency: str | None, rng: Random) -> str:
    if isinstance(value, date):
        months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
        return rng.choice((value.isoformat(), f"{value.day:02d}-{months[value.month - 1]}-{value.year}"))
    if name == "notional_amount":
        number = format(value, ",")
        prefix = {"USD": "US$", "EUR": "€", "GBP": "£"}.get(currency, currency + " ")
        return rng.choice((f"{currency} {number}", f"{number} {currency}",
                           f"{prefix}{format(value / Decimal(1000000), 'f')} million"))
    if name in {"fixed_rate", "spread"}:
        return rng.choice((f"{value * 100:f}%", f"{value * 100:f} percent", f"{value * 10000:f} bps"))
    if name in {"payment_frequency", "reset_frequency"}:
        aliases = {"3M": "quarterly", "6M": "semi-annual", "1Y": "annual"}
        return rng.choice((value, aliases.get(value, value)))
    if name == "day_count_fraction":
        return rng.choice((value, {"30/360": "30/360 (Bond Basis)", "ACT/360": "Actual/360",
                                   "ACT/365.FIXED": "Actual/365 (Fixed)"}[value]))
    if name == "business_day_convention":
        return {"MODFOLLOWING": "Modified Following", "FOLLOWING": "Following"}[value]
    if name == "business_centers":
        centers = {"USNY": "New York", "GBLO": "London", "EUTA": "TARGET", "JPTO": "Tokyo"}
        return " and ".join(centers[code] for code in value)
    return str(value)


def _context(trade: InterestRateSwap, seed: int, family: str) -> list[dict]:
    rng = Random(seed)
    sections = {}
    titles = {"document": "Transaction Details", "fixed_leg": "Fixed Leg",
              "floating_leg": "Floating Leg 1", "floating_leg_2": "Floating Leg 2"}
    for path, field in trade_fields(trade).items():
        if field.status != "explicit" or field.value is None:
            raise ValueError("Step 5B requires explicit, non-null structured terms")
        container, _, name = path.rpartition(".")
        container = container or "document"
        label = rng.choice(_LABELS[name])
        if family == "legal_preamble" and name == "notional_amount":
            label = "Calculation Amount"
        if family == "legal_preamble" and name == "payment_frequency":
            label = "Calculation Period payment interval"
        currency = getattr(trade, container).notional_currency.value if container != "document" else None
        sections.setdefault(container, {"title": titles[container], "rows": []})["rows"].append(
            {"path": path, "label": label, "text": _display(name, field.value, currency, rng)})
    return list(sections.values())


def _html(sections: list[dict], family: str, *, markers: bool) -> str:
    env = Environment(loader=PackageLoader("trustswap.synth.templates", "."),
                      autoescape=select_autoescape(default=True), undefined=StrictUndefined)
    return env.get_template(f"{family}.html.j2").render(
        sections=sections, markers=markers, disclaimer=DISCLAIMER)


def _no_resources(url, *args, **kwargs):
    raise ValueError(f"External resources are forbidden in synthetic rendering: {url}")


def _pdf_and_spans(marked_html: str) -> tuple[bytes, str, dict[str, tuple[int, int]]]:
    """Read annotation rectangles and PDF text maps, then discard all markers."""
    marked_pdf = HTML(string=marked_html, url_fetcher=_no_resources).write_pdf()
    source_parts = []
    raw_ranges: dict[str, list[int]] = {}
    source_offset = 0
    with pymupdf.open(stream=marked_pdf, filetype="pdf") as document:
        with pdfplumber.open(io.BytesIO(marked_pdf)) as plumber:
            for page, text_page in zip(document, plumber.pages):
                links = [(link, link.get("uri", "")[len(MARKER_PREFIX):])
                         for link in page.get_links() if link.get("uri", "").startswith(MARKER_PREFIX)]
                textmap = text_page.get_textmap()
                raw = "".join(piece for piece, _ in textmap.tuples)
                if raw != text_page.extract_text():
                    raise ValueError("unsupported PDF text-map ordering")
                cursor = source_offset
                for piece, char in textmap.tuples:
                    if char is not None and not piece.isspace():
                        x = (char["x0"] + char["x1"]) / 2
                        y = (char["top"] + char["bottom"]) / 2
                        matches = [key for link, key in links if pymupdf.Point(x, y) in link["from"]]
                        if len(matches) > 1:
                            raise ValueError("overlapping field markers")
                        if matches:
                            raw_ranges.setdefault(matches[0], []).extend(range(cursor, cursor + len(piece)))
                    cursor += len(piece)
                source_parts.append(raw)
                source_offset += len(raw) + 2  # Step 2 joins PDF pages with two newlines.
                for link, _ in links:
                    page.delete_link(link)
        # No generated timestamps or random file ID: deterministic PDF bytes.
        clean_pdf = document.tobytes(garbage=4, deflate=True, no_new_id=True)
    source = "\n\n".join(source_parts)
    text, offsets = normalise_whitespace(source)
    spans = {}
    for key, positions in raw_ranges.items():
        start, end = min(positions), max(positions) + 1
        selected = [i for i, (a, b) in enumerate(offsets) if a >= start and b <= end]
        if not selected:
            raise ValueError(f"empty rendered field: {key}")
        spans[key] = (selected[0], selected[-1] + 1)
    return clean_pdf, text, spans


def render_trade(trade: InterestRateSwap, *, seed: int, doc_id: str,
                 template_family: str | None = None, output_dir: str | Path = "data") -> RenderResult:
    """Render and save a synthetic PDF, HTML, ingested text and both JSONL records.

    Family splits are fixed in templates.FAMILY_SPLITS. No user-supplied split
    overrides, network resources, raw-file writes or output overwrites are allowed.
    """
    family = template_family or Random(seed).choice(tuple(FAMILY_SPLITS))
    if family not in FAMILY_SPLITS:
        raise ValueError(f"unknown template family: {family}")
    root = Path(output_dir)
    # Validate identifiers before using them in paths.
    DocumentRecord(doc_id=doc_id, collection="synthetic", sha256="0" * 64, text_path="pending", ocr=False)
    html_path = root / "synthetic" / "html" / f"{doc_id}.html"
    pdf_path = root / "synthetic" / "pdf" / f"{doc_id}.pdf"
    for path in (html_path, pdf_path, root / "synthetic" / "text" / f"{doc_id}.txt",
                 root / "synthetic" / "text" / f"{doc_id}.offsets.json"):
        _protect_raw(path)
        if path.exists():
            raise FileExistsError(path)
    sections = _context(trade, seed, family)
    pdf, expected_text, spans = _pdf_and_spans(_html(sections, family, markers=True))
    expected_fields = {row["path"]: row["text"] for section in sections for row in section["rows"]}
    if spans.keys() != expected_fields.keys():
        raise ValueError("some rendered field markers are missing from the PDF")
    for key, phrase in expected_fields.items():
        start, end = spans[key]
        if normalise_whitespace(expected_text[start:end])[0] != normalise_whitespace(phrase)[0]:
            raise ValueError(f"PDF layout changed the visible field value: {key}")
    annotated = trade.model_copy(deep=True)
    for key, field in trade_fields(annotated).items():
        field.span = spans[key]
    annotated = InterestRateSwap.model_validate(annotated.model_dump())
    with TemporaryDirectory(prefix="trustswap-render-") as directory:
        temporary = Path(directory) / "document.pdf"
        temporary.write_bytes(pdf)
        text, _ = to_text(temporary)
        if text != expected_text:
            raise ValueError("removing markers changed the final PDF text")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with html_path.open("x", encoding="utf-8") as handle:
        handle.write(_html(sections, family, markers=False))
    with pdf_path.open("xb") as handle:
        handle.write(pdf)
    text, meta = ingest_document(pdf_path, collection="synthetic", output_dir=root, doc_id=doc_id,
                                 family_id=family, split=FAMILY_SPLITS[family])
    document = DocumentRecord.model_validate({key: meta[key] for key in DocumentRecord.model_fields})
    gold = write_gold_record(annotated, doc_id=doc_id, seed=seed, output_dir=root, template_family=family)
    return RenderResult(html_path, pdf_path, text, document, gold)
