"""Fictional pilot documents and simulated operator input; no real labels/network."""
import hashlib
import io
import json
from pathlib import Path
import socket

import pytest

from scripts import annotate, agreement, select_pilot
from scripts.pilot_utils import read_annotations, read_csv, write_csv
from trustswap.common.annotations import AnnotationRecord, FIELDS, normalise_field, validated_field

FIXTURE = Path(__file__).parent / "fixtures/pilot/confirmation.txt"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *args, **kwargs: pytest.fail("pilot tools must be offline"))


def inputs(*values):
    iterator = iter(values)
    def ask(prompt):
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError from None
    return ask


@pytest.fixture
def pilot(tmp_path):
    text = tmp_path / "confirmation.txt"
    text.write_bytes(FIXTURE.read_bytes())
    row = {"doc_id": "fixture_1", "document_url": "https://example.invalid/fictional", "era": "2001–2005",
           "label": "likely_confirmation", "text_path": str(text)}
    path = tmp_path / "pilot20.csv"
    write_csv(path, tuple(row), [row])
    return path, text, tmp_path / "annotations"


def blank(doc_id="fixture_1", annotator="A1", **updates):
    return AnnotationRecord(doc_id=doc_id, annotator=annotator, text_path=str(FIXTURE),
                            text_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), **updates)


def test_select_twenty_stratified_prioritised_and_flagged():
    screen, manual = [], []
    for era in select_pilot.ERAS:
        for i in range(12):
            url = f"https://example.invalid/{era}/{i:02d}"
            screen.append({"doc_id": f"d_{select_pilot.ERAS.index(era)}_{i}", "document_url": url,
                           "era": era, "label": "likely_confirmation", "text_path": "fictional.txt"})
            if i < 4:
                manual.append({"document_url": url, "is_confirmation": "y", "product_type": "irs_fixed_float"})
            elif i == 11:
                manual.append({"document_url": url, "is_confirmation": "n", "product_type": "other"})
    selected = select_pilot.select_pilot(screen, manual, seed=4)
    assert len(selected) == len({row["doc_id"] for row in selected}) == 20
    from collections import Counter
    assert Counter(row["era"] for row in selected) == dict(zip(select_pilot.ERAS, (6, 6, 6, 2)))
    assert sum(row["selection_reason"] == "manual_confirmation" for row in selected) == 14
    assert all(row["manual_check_required"] == "true" for row in selected if row["selection_reason"] == "heuristic_topup")
    assert not any(row["is_confirmation"] == "n" for row in selected)
    assert selected == select_pilot.select_pilot(list(reversed(screen)), manual, seed=4)


def test_pilot_redistributes_without_exceeding_recent_cap():
    rows = [{"doc_id": f"d{i}", "document_url": str(i), "era": "2009–2021" if i < 21 else "2022+",
             "label": "likely_confirmation", "text_path": "fixture.txt"} for i in range(40)]
    selected = select_pilot.select_pilot(rows, [])
    assert len(selected) == 20 and sum(row["era"] == "2022+" for row in selected) == 2
    with pytest.raises(ValueError, match="fewer than 20"):
        select_pilot.select_pilot(rows[21:], [])


def test_span_choice_normalisation_and_per_field_save(pilot):
    path, source, output_dir = pilot
    text = source.read_text()
    quote = "USD 250,000,000"
    found = annotate.occurrences(text, quote)
    assert len(found) == 2
    output = io.StringIO()
    answer = annotate.ask_field("floating_leg.notional_amount", text, inputs(quote, "2", "y"), output)
    assert answer["span"] == list(found[1]) and answer["value"] == "250000000"
    assert text[slice(*answer["span"])] == quote
    counter = 0
    def ask(prompt):
        nonlocal counter
        values = ["20-Mar-2012", "y", "absent", "y"]
        if counter == 2:
            records, _ = read_annotations(output_dir / "A1.jsonl")
            assert records["fixture_1"].fields["trade_date"]["value"] == "2012-03-20"
        if counter >= len(values):
            raise EOFError
        result = values[counter]
        counter += 1
        return result
    assert annotate.annotate(path, annotator="A1", output_dir=output_dir, input_fn=ask, output=io.StringIO()) == 0
    records, _ = read_annotations(output_dir / "A1.jsonl")
    record = records["fixture_1"]
    assert set(record.fields) == {"trade_date", "effective_date"}
    assert not record.complete
    assert record.fields["effective_date"]["status"] == "absent"
    assert "termination_date" not in record.fields


def test_reference_ambiguous_and_back_resume(pilot):
    path, _, output_dir = pilot
    annotate.annotate(path, annotator="A1", output_dir=output_dir,
                      input_fn=inputs("20-Mar-2012", "y", "back", "absent", "y", "ref", "22-Mar-2012", "y", "amb", "y"),
                      output=io.StringIO())
    records, _ = read_annotations(output_dir / "A1.jsonl")
    record = records["fixture_1"]
    assert record.fields["trade_date"]["status"] == "absent"
    assert record.fields["effective_date"] == {"value": "2012-03-22", "span": None, "status": "by_reference"}
    assert record.fields["termination_date"]["status"] == "ambiguous"
    annotate.annotate(path, annotator="A1", output_dir=output_dir, input_fn=inputs("absent", "y"), output=io.StringIO())
    resumed, _ = read_annotations(output_dir / "A1.jsonl")
    assert "fixed_leg.payer" in resumed["fixture_1"].fields


def test_complete_flags_and_trade_roundtrip(pilot):
    path, _, output_dir = pilot
    answers = [value for _ in FIELDS for value in ("absent", "y")] + ["amortising, cap, floor", "y"]
    annotate.annotate(path, annotator="A1", output_dir=output_dir, input_fn=inputs(*answers), output=io.StringIO())
    records, _ = read_annotations(output_dir / "A1.jsonl")
    record = records["fixture_1"]
    assert record.complete and record.flags == ["amortising", "cap", "floor"]
    assert record.as_trade().trade_date.status == "absent"
    before = (output_dir / "A1.jsonl").read_bytes()
    annotate.annotate(path, annotator="A1", output_dir=output_dir, input_fn=lambda _: pytest.fail("complete file should resume without prompts"), output=io.StringIO())
    assert (output_dir / "A1.jsonl").read_bytes() == before


def test_changed_source_refused_and_other_annotator_not_read(pilot):
    path, source, output_dir = pilot
    annotate.annotate(path, annotator="A1", output_dir=output_dir, input_fn=inputs("absent", "y"), output=io.StringIO())
    (output_dir / "A2.jsonl").write_text("not read by A1")
    annotate.annotate(path, annotator="A1", output_dir=output_dir, input_fn=inputs("absent", "y"), output=io.StringIO())
    assert len(read_annotations(output_dir / "A1.jsonl")[0]["fixture_1"].fields) == 2
    source.write_text(source.read_text() + "changed")
    with pytest.raises(ValueError, match="text changed"):
        annotate.annotate(path, annotator="A1", output_dir=output_dir, input_fn=inputs(), output=io.StringIO())


def test_field_normalisers_and_null_span_reference():
    assert str(normalise_field("fixed_leg.fixed_rate", "4.25%")) == "0.0425"
    assert normalise_field("fixed_leg.notional_currency", "US$") == "USD"
    assert normalise_field("floating_leg.business_centers", "New York and London") == ["GBLO", "USNY"]
    with pytest.raises(ValueError, match="by-reference"):
        validated_field("trade_date", "2012-03-20", [1, 2], "by_reference")
    with pytest.raises(ValueError, match="explicit"):
        validated_field("trade_date", "2012-03-20", None, "explicit")


def test_agreement_normalised_values_status_spans_flags_and_unpaired():
    a = blank(fields={"fixed_leg.fixed_rate": validated_field("fixed_leg.fixed_rate", "0.04250", [5, 10], "explicit"),
                      "trade_date": validated_field("trade_date", None, None, "absent")}, flags=["cap"], flags_reviewed=True)
    b = blank(annotator="A2", fields={"fixed_leg.fixed_rate": validated_field("fixed_leg.fixed_rate", "0.0425", [5, 10], "explicit"),
                      "trade_date": validated_field("trade_date", None, None, "ambiguous"),
                      "effective_date": validated_field("effective_date", None, None, "absent")}, flags=[], flags_reviewed=True)
    metrics, disagreements = agreement.compare({a.doc_id: a}, {b.doc_id: b})
    rates = next(row for row in metrics if row["field"] == "fixed_leg.fixed_rate")
    assert rates["value_exact_rate"] == 1 and rates["field_exact_matches"] == 1
    assert rates["status_kappa"] is None  # Constant status has undefined kappa.
    assert {row["field"] for row in disagreements} == {"trade_date", "effective_date", "flags"}
    assert any(row["reason"] == "missing_A1" for row in disagreements)
    b.fields["fixed_leg.fixed_rate"]["span"] = [20, 25]
    _, disagreements = agreement.compare({a.doc_id: a}, {b.doc_id: b})
    assert any(row["field"] == "fixed_leg.fixed_rate" and row["reason"] == "span" for row in disagreements)


def test_kappa_known_cases_and_source_versions():
    assert agreement.cohen_kappa(["explicit", "absent"], ["absent", "explicit"]) == -1
    assert agreement.cohen_kappa(["explicit", "explicit", "absent", "absent"], ["explicit", "absent", "explicit", "absent"]) == 0
    assert agreement.cohen_kappa([], []) is None
    a, b = blank(), blank(annotator="A2")
    b.text_sha256 = "0" * 64
    metrics, disagreements = agreement.compare({a.doc_id: a}, {b.doc_id: b})
    assert disagreements[0]["reason"] == "source_version"
    assert all(row["paired"] == 0 for row in metrics)


def test_agreement_cli_writes_lead_disagreements(tmp_path):
    a = blank(fields={"trade_date": validated_field("trade_date", None, None, "absent")})
    b = blank(annotator="A2", fields={"trade_date": validated_field("trade_date", None, None, "ambiguous")})
    left, right = tmp_path / "A1.jsonl", tmp_path / "A2.jsonl"
    left.write_text(a.model_dump_json() + "\n")
    right.write_text(b.model_dump_json() + "\n")
    output, metrics = tmp_path / "adjudication.csv", tmp_path / "agreement.csv"
    assert agreement.main(["--a1", str(left), "--a2", str(right), "--output", str(output), "--metrics", str(metrics), "--pilot", str(tmp_path / "no_pilot.csv")]) == 0
    _, rows = read_csv(output)
    assert rows[0]["field"] == "trade_date" and "status" in rows[0]["reason"]
    assert len(read_csv(metrics)[1]) == len(FIELDS)


def test_complete_explicit_fixture_annotation_and_spans(pilot):
    from decimal import Decimal
    path, source, output_dir = pilot
    quotes = {
        "trade_date": "20-Mar-2012", "effective_date": "22-Mar-2012", "termination_date": "22-Mar-2017",
        "fixed_leg.payer": "Example Bank A", "fixed_leg.receiver": "Example Company B",
        "fixed_leg.notional_amount": "USD 250,000,000", "fixed_leg.notional_currency": "USD",
        "fixed_leg.payment_frequency": "semi-annual", "fixed_leg.day_count_fraction": "30/360",
        "fixed_leg.business_day_convention": "Modified Following", "fixed_leg.business_centers": "New York",
        "fixed_leg.fixed_rate": "4.25%", "floating_leg.payer": "Example Company B",
        "floating_leg.receiver": "Example Bank A", "floating_leg.notional_amount": "USD 250,000,000",
        "floating_leg.notional_currency": "USD", "floating_leg.payment_frequency": "quarterly",
        "floating_leg.reset_frequency": "3 months", "floating_leg.day_count_fraction": "Actual/360",
        "floating_leg.business_day_convention": "Modified Following", "floating_leg.business_centers": "New York",
        "floating_leg.floating_rate_option": "USD-LIBOR-BBA", "floating_leg.spread": "25 bps",
        "notice_address": "1 Example Street\nExample City",
    }
    text, responses = source.read_text(), []
    for field in FIELDS:
        if field == "fixed_leg.reset_frequency":
            responses.extend(["absent", "y"])
            continue
        quote = quotes[field]
        responses.append(quote.replace("\n", "\\n"))
        matches = annotate.occurrences(text, quote)
        if len(matches) > 1:
            occurrence = 2 if field.startswith("floating_leg.") else 1
            if field.endswith("notional_currency"):
                occurrence = 4 if field.startswith("floating_leg.") else 2
            responses.append(str(occurrence))
        responses.append("y")
    responses.extend(["none", "y"])
    annotate.annotate(path, annotator="A1", output_dir=output_dir, input_fn=inputs(*responses), output=io.StringIO())
    record = read_annotations(output_dir / "A1.jsonl")[0]["fixture_1"]
    assert record.complete
    trade = record.as_trade()
    assert trade.fixed_leg.fixed_rate.value == Decimal("0.0425")
    assert trade.floating_leg.spread.value == Decimal("0.0025")
    assert trade.fixed_leg.notional_amount.value == Decimal("250000000")
    assert trade.fixed_leg.payment_frequency.value == "6M"
    assert trade.floating_leg.reset_frequency.value == "3M"
    for field, answer in record.fields.items():
        if field in quotes:
            assert text[slice(*answer["span"])] == quotes[field]


def test_selector_cli_csv_flags_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    rows = [{"doc_id": f"fixture_{era}_{i}", "document_url": f"fixture://{era}/{i}", "era": era,
             "text_path": str(FIXTURE), "label": "likely_confirmation"} for era in select_pilot.ERAS for i in range(10)]
    screen, manual, output = tmp_path / "screen.csv", tmp_path / "manual.csv", tmp_path / "pilot.csv"
    write_csv(screen, tuple(rows[0]), rows)
    write_csv(manual, ("document_url", "is_confirmation", "product_type"), [])
    assert select_pilot.main(["--screen", str(screen), "--manual", str(manual), "--output", str(output)]) == 0
    selected = read_csv(output)[1]
    assert len(selected) == 20 and all(row["not_manually_checked"] == "false" for row in selected)
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        select_pilot.main(["--screen", str(screen), "--manual", str(manual), "--output", str(output)])
    assert output.read_bytes() == before


def test_unresolved_reference_and_document_header_rate_units():
    output = io.StringIO()
    reference = annotate.ask_field("floating_leg.day_count_fraction", "ISDA Definitions apply", inputs("ref", "", "y"), output)
    assert reference == {"value": None, "span": None, "status": "by_reference"}
    text = "Fixed Rate (decimal): 0.0425"
    rate = annotate.ask_field("fixed_leg.fixed_rate", text, inputs("0.0425", "decimal", "y"), output)
    assert rate["value"] == "0.0425" and text[slice(*rate["span"])] == "0.0425"


@pytest.mark.parametrize("entered", [" 27-Mar-17", "\t27-Mar-17  ", "  text: 27-Mar-17  "])
def test_annotation_trims_value_before_span_search(entered):
    text = "Trade Date: 27-Mar-17\nEffective Date: 29-Mar-17"
    answer = annotate.ask_field("trade_date", text, inputs(entered, "y"), io.StringIO())
    assert answer["value"] == "2017-03-27"
    assert text[slice(*answer["span"])] == "27-Mar-17"
    assert answer["span"][0] == text.index("27-Mar-17")


def test_annotation_trims_reference_value():
    answer = annotate.ask_field("effective_date", "See incorporated schedule", inputs("ref", " 01-Apr-22 ", "y"), io.StringIO())
    assert answer == {"value": "2022-04-01", "span": None, "status": "by_reference"}
