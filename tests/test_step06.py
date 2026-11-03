"""Step 6: structured gold rules, real rendering and clean-twin provenance."""
import json
from pathlib import Path
from random import Random

import pytest
import QuantLib as ql
from pydantic import ValidationError

from trustswap.common.normalise import normalise_day_count, normalise_notional
from trustswap.common.records import AdversarialMetadata, DocumentRecord, GoldRecord
from trustswap.synth.defects import (
    DEFECT_TYPES, EXPECTED_RULES, INJECTORS, PROMPT_INJECTION, SEVERITIES,
    amendment, contradiction, make_control, missing_information, ocr_digit_corruption,
    prompt_injection, render_case,
)
from trustswap.ingest.to_text import to_text
from trustswap.synth.markets import calendar_for
from trustswap.synth.render import DISCLAIMER, trade_fields
from trustswap.synth.sampler import sample_trade


@pytest.fixture(scope="module")
def parent():
    return sample_trade(Random(40))


@pytest.fixture(scope="module")
def rendered(parent, tmp_path_factory):
    root = tmp_path_factory.mktemp("step06")
    results = {}
    for kind in DEFECT_TYPES:
        for severity in SEVERITIES:
            for control in (False, True):
                case = INJECTORS[kind](parent, 40, severity=severity, parent_id="syn_parent", control=control)
                result = render_case(case, output_dir=root, template_family="label_value")
                results[kind, severity, control] = (case, result)
    return root, results


def test_every_defect_and_control_renders(rendered):
    root, results = rendered
    assert len(results) == 12 * 2 * 2
    for (kind, severity, control), (case, result) in results.items():
        assert result.pdf_path.is_file() and result.html_path.is_file()
        assert result.text.startswith(DISCLAIMER)
        assert result.document.collection == result.gold.collection == "adversarial"
        assert result.document.split == "adv_test"
        assert result.document.defect == result.gold.defect == kind
        assert result.document.severity == result.gold.severity == severity
        assert result.document.is_control == result.gold.is_control == control
        assert result.document.affected_fields == result.gold.affected_fields == case.metadata.affected_fields
        assert result.document.expected_rules == result.gold.expected_rules
        assert result.document.designed_by == result.gold.designed_by == "team"
        for path in case.metadata.affected_fields:
            assert path in trade_fields(result.gold.trade)
        if not control:
            assert result.document.expected_rules == list(EXPECTED_RULES[kind])
    docs = [DocumentRecord.model_validate_json(line) for line in
            (root / "adversarial/docs.jsonl").read_text().splitlines()]
    gold = [GoldRecord.model_validate_json(line) for line in
            (root / "adversarial/gold.jsonl").read_text().splitlines()]
    assert len(docs) == len(gold) == 48
    assert {record.defect for record in docs if record.is_control} == set(DEFECT_TYPES)


def test_parents_exist_and_are_preserved(parent, rendered):
    root, results = rendered
    docs = [json.loads(line) for line in (root / "synthetic/docs.jsonl").read_text().splitlines()]
    labels = [GoldRecord.model_validate_json(line) for line in (root / "synthetic/gold.jsonl").read_text().splitlines()]
    assert len(docs) == len(labels) == 1
    assert docs[0]["doc_id"] == "syn_parent"
    assert Path(docs[0]["text_path"]).is_file()
    for _, result in results.values():
        assert result.document.parent_id == result.gold.parent_id == docs[0]["doc_id"]
    for path, term in trade_fields(parent).items():
        assert term.span is None
        assert trade_fields(labels[0].trade)[path].value == term.value
        assert trade_fields(labels[0].trade)[path].status == term.status


@pytest.mark.parametrize("severity", SEVERITIES)
def test_gold_semantics(parent, rendered, severity):
    _, results = rendered
    for kind in ("contradiction", "multiple_notionals", "stub_ambiguity", "legacy_index_fallback"):
        case, result = results[kind, severity, False]
        for path in case.metadata.affected_fields:
            term = trade_fields(result.gold.trade)[path]
            assert term.status == "ambiguous" and term.value is None and term.span is None
    _, amended = results["amendment", severity, False]
    assert amended.gold.trade.termination_date.value != parent.termination_date.value
    start, end = amended.gold.trade.termination_date.span
    from trustswap.common.normalise import normalise_date
    assert normalise_date(amended.text[start:end]) == amended.gold.trade.termination_date.value.isoformat()
    case, missing = results["missing_information", severity, False]
    for path in case.metadata.affected_fields:
        term = trade_fields(missing.gold.trade)[path]
        assert term.status == "absent" and term.value is None and term.span is None
    if severity == 1:
        assert parent.notice_address.value not in missing.text
    else:
        assert parent.notice_address.value in missing.text
    _, ocr = results["ocr_digit_corruption", severity, False]
    term = ocr.gold.trade.floating_leg.notional_amount
    assert term.value == parent.floating_leg.notional_amount.value
    phrase = ocr.text[slice(*term.span)]
    assert "O" in phrase
    with pytest.raises(ValueError):
        normalise_notional(phrase)
    _, injection = results["prompt_injection", severity, False]
    assert injection.text.count(PROMPT_INJECTION) == (1 if severity == 1 else 3)
    for path, term in trade_fields(parent).items():
        actual = trade_fields(injection.gold.trade)[path]
        assert actual.value == term.value and actual.status == term.status


def test_correct_controls_and_quantlib_calendars(rendered):
    for (kind, severity, control), (case, result) in rendered[1].items():
        if not control:
            continue
        assert all(term.status != "ambiguous" for term in trade_fields(result.gold.trade).values())
        assert result.document.expected_rules in (["silent"], ["V7"])
        if kind == "non_standard_calendar":
            trade = result.gold.trade
            calendar = calendar_for(trade.floating_leg.business_centers.value)
            value = trade.effective_date.value
            assert calendar.isBusinessDay(ql.Date(value.day, value.month, value.year))
        if kind == "unusual_day_count":
            term = result.gold.trade.floating_leg.day_count_fraction
            assert normalise_day_count(result.text[slice(*term.span)]) == term.value
            counter = ql.ActualActual(ql.ActualActual.ISDA) if severity == 1 else ql.Thirty360(ql.Thirty360.European)
            assert counter.yearFraction(ql.Date(1, 1, 2024), ql.Date(1, 2, 2024)) > 0
        if kind == "stub_ambiguity":
            assert ("Agreed short initial stub:" if severity == 1 else "Agreed short final stub:") in result.text
            assert "regular reset tenor remains" in result.text


def test_nonstandard_calendar_defect_has_closed_center(rendered):
    for severity in SEVERITIES:
        trade = rendered[1]["non_standard_calendar", severity, False][1].gold.trade
        value = trade.effective_date.value
        assert not calendar_for(trade.floating_leg.business_centers.value).isBusinessDay(
            ql.Date(value.day, value.month, value.year))


def test_severities_are_distinct_and_seed_is_deterministic(parent):
    for name, injector in INJECTORS.items():
        first = injector(parent, 40, severity=1)
        again = injector(parent, 40, severity=1)
        second = injector(parent, 40, severity=2)
        assert first == again
        assert (first.trade, first.gold.trade, first.field_text, first.omitted_fields, first.notes) != (
            second.trade, second.gold.trade, second.field_text, second.omitted_fields, second.notes), name


def test_rendering_determinism(parent, tmp_path):
    case = contradiction(parent, 40)
    left = render_case(case, output_dir=tmp_path / "left")
    right = render_case(contradiction(parent, 40), output_dir=tmp_path / "right")
    assert left.html_path.read_bytes() == right.html_path.read_bytes()
    assert left.text == right.text == to_text(left.pdf_path)[0] == to_text(right.pdf_path)[0]
    assert {path: term.span for path, term in trade_fields(left.gold.trade).items()} == {
        path: term.span for path, term in trade_fields(right.gold.trade).items()}
    assert left.gold == right.gold


def test_changed_parent_is_not_replaced(parent, rendered):
    root, _ = rendered
    changed = parent.model_copy(deep=True)
    changed.notice_address.value = "Changed parent"
    with pytest.raises(ValueError, match="clean twin"):
        render_case(contradiction(changed, 40, parent_id="syn_parent"), output_dir=root, template_family="label_value")


def test_metadata_validation_and_blind_design(parent):
    case = missing_information(parent, 40, designed_by="blind")
    assert case.gold.designed_by == "blind"
    invalid = case.metadata.model_dump()
    invalid["expected_rules"] = ["V8"]
    with pytest.raises(ValidationError):
        AdversarialMetadata.model_validate(invalid)
    invalid["expected_rules"] = ["silent", "V1"]
    with pytest.raises(ValidationError):
        AdversarialMetadata.model_validate(invalid)
    with pytest.raises(ValueError, match="severity"):
        amendment(parent, 40, severity=3)


def test_controls_ground_every_explicit_gold_span(rendered):
    from trustswap.common import normalise as n

    normalisers = {"trade_date": n.normalise_date, "effective_date": n.normalise_date,
                   "termination_date": n.normalise_date, "notional_amount": lambda s: n.normalise_notional(s)[0],
                   "fixed_rate": n.normalise_rate, "spread": n.normalise_rate,
                   "payment_frequency": n.normalise_frequency, "reset_frequency": n.normalise_frequency,
                   "day_count_fraction": n.normalise_day_count, "business_centers": n.normalise_business_centers,
                   "business_day_convention": n.normalise_business_day_convention,
                   "floating_rate_option": n.normalise_floating_rate_option}
    for (kind, severity, control), (_, result) in rendered[1].items():
        if not control:
            continue
        for path, term in trade_fields(result.gold.trade).items():
            if term.status == "absent":
                assert term.span is None
                continue
            assert term.span is not None
            phrase = result.text[slice(*term.span)]
            name = path.split(".")[-1]
            value = normalisers.get(name, lambda s: s)(phrase)
            expected = term.value
            if name.endswith("_date"):
                expected = expected.isoformat()
            elif name == "business_centers":
                expected = sorted(expected)
            elif name == "floating_rate_option":
                expected = expected.upper()
            assert value == expected, (kind, severity, path)


def test_scale_and_swapped_role_gold_preserves_truth(parent, rendered):
    for severity in SEVERITIES:
        _, scale = rendered[1]["scale_error", severity, False]
        term = scale.gold.trade.floating_leg.notional_amount
        assert term.value == parent.floating_leg.notional_amount.value
        assert normalise_notional(scale.text[slice(*term.span)])[0] == term.value * (10 if severity == 1 else 1000)
        case, swapped = rendered[1]["swapped_legs", severity, False]
        for path in case.metadata.affected_fields:
            assert trade_fields(swapped.gold.trade)[path].value == trade_fields(parent)[path].value
            assert "Authoritative" in swapped.text


@pytest.mark.parametrize("severity", SEVERITIES)
@pytest.mark.parametrize("seed", [40, 2])
def test_swapped_legs_reverse_both_roles_and_remain_consistent(seed, severity):
    parent = sample_trade(Random(seed))
    case = INJECTORS["swapped_legs"](parent, seed, severity=severity)
    legs = [name for name in ("fixed_leg", "floating_leg", "floating_leg_2") if getattr(parent, name) is not None]
    expected = {f"{leg}.{role}" for leg in legs for role in ("payer", "receiver")}
    assert len(case.metadata.affected_fields) == 4
    assert set(case.metadata.affected_fields) == expected
    for name in legs:
        before, after = getattr(parent, name), getattr(case.trade, name)
        assert after.payer.value == before.receiver.value
        assert after.receiver.value == before.payer.value
    first, second = (getattr(case.trade, leg) for leg in legs)
    assert first.payer.value == second.receiver.value
    assert first.receiver.value == second.payer.value


def test_missing_domain_information_is_seeded_and_absent(parent):
    seen = set()
    for seed in range(30):
        case = missing_information(parent, seed, severity=2)
        assert len(case.omitted_fields) == 1
        path = case.omitted_fields[0]
        seen.add(path)
        term = trade_fields(case.gold.trade)[path]
        assert term.status == "absent" and term.value is None and term.span is None
        assert case.gold.trade.notice_address.value == parent.notice_address.value
        assert case.metadata.affected_fields == [path]
        assert missing_information(parent, seed, severity=2) == case
    assert seen == {"floating_leg.business_centers", "floating_leg.day_count_fraction", "floating_leg.spread"}
    lowest = missing_information(parent, 40, severity=1)
    assert lowest.metadata.affected_fields == ["notice_address"]


@pytest.mark.parametrize("severity,path", [(1, "effective_date"), (2, "termination_date")])
def test_stub_ambiguity_targets_date_boundaries(parent, severity, path):
    case = INJECTORS["stub_ambiguity"](parent, 40, severity=severity)
    assert case.metadata.affected_fields == [path]
    assert case.metadata.expected_rules == ["V4"]
    assert trade_fields(case.gold.trade)[path].status == "ambiguous"
    assert case.gold.trade.floating_leg.reset_frequency == parent.floating_leg.reset_frequency
    control = make_control("stub_ambiguity", parent, 40, severity=severity)
    assert trade_fields(control.gold.trade)[path].value == trade_fields(parent)[path].value
    assert trade_fields(control.gold.trade)[path].status == "explicit"
