"""Rendered PDF gold spans are checked against independently ingested text."""
from collections import defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
from random import Random

import pymupdf
import pytest

from trustswap.common import normalise
from trustswap.common.records import DocumentRecord, GoldRecord
from trustswap.ingest.to_text import to_text
from trustswap.synth.render import DISCLAIMER, MARKER_PREFIX, render_trade, trade_fields
from trustswap.synth.sampler import sample_trade
from trustswap.synth.templates import FAMILY_SPLITS


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    root = tmp_path_factory.mktemp("rendered")
    rng = Random(507)
    families = tuple(FAMILY_SPLITS)
    results = []
    originals = []
    for index in range(50):
        trade = sample_trade(rng)
        originals.append(trade)
        results.append(render_trade(trade, seed=20261026 + index, doc_id=f"syn_{index:06d}",
                                    template_family=families[index % len(families)], output_dir=root))
    return root, originals, results


def canonical(name, value):
    if name.endswith("_date"):
        return normalise.normalise_date(value)
    if name in {"fixed_rate", "spread"}:
        return normalise.normalise_rate(value)
    if name == "notional_amount":
        return normalise.normalise_notional(value)[0]
    if name in {"payment_frequency", "reset_frequency"}:
        return normalise.normalise_frequency(value)
    if name == "day_count_fraction":
        return normalise.normalise_day_count(value)
    if name == "business_day_convention":
        return normalise.normalise_business_day_convention(value)
    if name == "business_centers":
        return normalise.normalise_business_centers(value)
    if name == "floating_rate_option":
        return normalise.normalise_floating_rate_option(value)
    return value.strip()


def test_fifty_trade_spans_normalise_to_gold(rendered):
    _, originals, results = rendered
    assert len(results) == 50
    assert any(result.gold.trade.floating_leg_2 is not None for result in results)
    for original, result in zip(originals, results):
        text, meta = to_text(result.pdf_path)
        assert text == result.text == Path(result.document.text_path).read_text(encoding="utf-8")
        assert meta["ocr"] is False
        assert text.splitlines()[0] == DISCLAIMER
        occupied = []
        for path, field in trade_fields(result.gold.trade).items():
            if field.status == "absent":
                assert path == "fixed_leg.reset_frequency"
                assert field.value is None and field.span is None
                continue
            assert field.status == "explicit"
            assert field.span is not None
            start, end = field.span
            assert 0 <= start < end <= len(text)
            phrase = text[start:end]
            name = path.split(".")[-1]
            expected = field.value
            if isinstance(expected, date):
                expected = expected.isoformat()
            elif name == "business_centers":
                expected = sorted(expected)
            elif name == "floating_rate_option":
                expected = normalise.normalise_floating_rate_option(expected)
            assert canonical(name, phrase) == expected, (path, phrase, expected)
            if name == "notional_amount":
                leg_name = path.split(".")[0]
                assert normalise.normalise_notional(phrase)[1] == getattr(original, leg_name).notional_currency.value
            assert all(end <= a or start >= b for a, b in occupied)
            occupied.append((start, end))
            assert trade_fields(original)[path].span is None  # no mutation of sampled gold
            assert trade_fields(original)[path].value == field.value


def test_records_and_split_isolation(rendered):
    root, _, results = rendered
    docs = [DocumentRecord.model_validate_json(line) for line in
            (root / "synthetic/docs.jsonl").read_text().splitlines()]
    gold = [GoldRecord.model_validate_json(line) for line in
            (root / "synthetic/gold.jsonl").read_text().splitlines()]
    assert len(docs) == len(gold) == 50
    assignments = defaultdict(set)
    for document, record, result in zip(docs, gold, results):
        assert record.doc_id == document.doc_id
        assert record.generator.template_family == document.family_id
        assert document.split == FAMILY_SPLITS[document.family_id]
        assignments[document.family_id].add(document.split)
        assert document.sha256 == hashlib.sha256(result.pdf_path.read_bytes()).hexdigest()
        assert document.collection == record.collection == "synthetic"
        assert Path(document.text_path).with_suffix(".offsets.json").is_file()
    assert set(assignments) == set(FAMILY_SPLITS)
    assert all(len(splits) == 1 for splits in assignments.values())
    assert {doc.split for doc in docs} == {"train", "dev", "syn_test"}


def test_final_artifacts_have_no_markers(rendered):
    for result in rendered[2]:
        assert MARKER_PREFIX not in result.html_path.read_text()
        assert MARKER_PREFIX.encode() not in result.pdf_path.read_bytes()
        assert MARKER_PREFIX not in result.text
        with pymupdf.open(result.pdf_path) as document:
            assert all(not page.get_links() for page in document)


def test_fixed_seed_determinism(tmp_path):
    left = render_trade(sample_trade(Random(40)), seed=40, doc_id="same", output_dir=tmp_path / "left")
    right = render_trade(sample_trade(Random(40)), seed=40, doc_id="same", output_dir=tmp_path / "right")
    assert left.html_path.read_bytes() == right.html_path.read_bytes()
    assert left.text == right.text == to_text(left.pdf_path)[0] == to_text(right.pdf_path)[0]
    assert {path: field.span for path, field in trade_fields(left.gold.trade).items()} == {
        path: field.span for path, field in trade_fields(right.gold.trade).items()}
    assert left.gold == right.gold


def test_template_wording_and_number_variation(rendered):
    _, _, results = rendered
    combined = "\n".join(result.text for result in results)
    assert "Notional Amount" in combined
    assert "Calculation Amount" in combined
    assert "Notional:" in combined or "Notional /" in combined
    assert "The purpose of this letter agreement" in combined
    assert "Calculation Period" in combined
    assert "DEALER CONFIRMATION" in combined
    import re
    assert re.search(r"USD [\d,]+", combined)
    assert re.search(r"[\d,]+ USD", combined)
    assert re.search(r"US\$[\d.]+ million", combined)


def test_unknown_family_and_raw_output_are_refused(tmp_path):
    trade = sample_trade(Random(1))
    with pytest.raises(ValueError, match="unknown template"):
        render_trade(trade, seed=1, doc_id="syn_1", template_family="unknown", output_dir=tmp_path)
    with pytest.raises(ValueError, match="data/raw"):
        render_trade(trade, seed=1, doc_id="syn_1", output_dir=tmp_path / "data/raw")
    assert not (tmp_path / "data").exists()


def test_existing_output_is_not_overwritten(rendered):
    root, originals, results = rendered
    before = results[0].pdf_path.read_bytes()
    with pytest.raises(FileExistsError):
        render_trade(originals[0], seed=1, doc_id="syn_000000", output_dir=root)
    assert results[0].pdf_path.read_bytes() == before


def test_html_escaping_and_no_external_resources(tmp_path, monkeypatch):
    import socket
    from trustswap.synth.render import _no_resources

    def no_network(*args, **kwargs):
        raise AssertionError("renderer attempted network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    trade = sample_trade(Random(5))
    trade.notice_address.value = '<script>alert("fiction")</script>'
    result = render_trade(trade, seed=5, doc_id="escaped", template_family="label_value", output_dir=tmp_path)
    assert "&lt;script&gt;" in result.html_path.read_text()
    start, end = result.gold.trade.notice_address.span
    assert result.text[start:end] == trade.notice_address.value
    with pytest.raises(ValueError, match="forbidden"):
        _no_resources("https://example.com/image.png")


def test_fixed_reset_is_not_rendered(rendered):
    from bs4 import BeautifulSoup
    from trustswap.synth.render import _context

    fixed = [(trade, result) for trade, result in zip(rendered[1], rendered[2]) if trade.fixed_leg is not None]
    assert fixed
    for trade, result in fixed:
        sections = _context(trade, result.gold.generator.seed, result.document.family_id)
        assert all(row["path"] != "fixed_leg.reset_frequency" for section in sections for row in section["rows"])
        field = result.gold.trade.fixed_leg.reset_frequency
        assert field.status == "absent" and field.value is None and field.span is None
        soup = BeautifulSoup(result.html_path.read_text(), "html.parser")
        heading = next(h for h in soup.find_all("h2") if h.get_text().strip().lower() == "fixed leg")
        content = []
        for sibling in heading.next_siblings:
            if getattr(sibling, "name", None) == "h2":
                break
            content.append(sibling.get_text() if hasattr(sibling, "get_text") else str(sibling))
        assert "reset" not in " ".join(content).lower()


def test_pdf_metadata_and_seed_identifier(tmp_path):
    trade = sample_trade(Random(40))
    result = render_trade(trade, seed=40, doc_id="metadata", output_dir=tmp_path)
    with pymupdf.open(result.pdf_path) as document:
        assert document.metadata["creationDate"] == "D:20260101000000Z"
        assert document.metadata["modDate"] == "D:20260101000000Z"
        identifier = hashlib.sha256(b"trustswap-pdf:40").hexdigest()
        trailer_id = document.xref_get_key(-1, "ID")[1].upper()
        assert f"({identifier.upper()})" in trailer_id or f"<{identifier.encode().hex().upper()}>" in trailer_id
