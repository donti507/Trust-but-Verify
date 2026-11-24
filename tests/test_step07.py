"""Step 7: verifier rules V1-V4.

Adversarial checks evaluate the gold labels against the rendered document, with
ambiguous fields filled by the displayed value (what a literal reader would
take). Only the defect/rule pairs that V1-V4 can detect by construction are
asserted; the remaining H4 predictions (e.g. contradiction -> V2) are research
outcomes measured in Step 9, not test requirements.
"""
from datetime import date
from decimal import Decimal
from random import Random

import pytest

from trustswap.synth import defects as D
from trustswap.synth.render import render_trade, trade_fields
from trustswap.synth.sampler import sample_trade
from trustswap.synth.templates import FAMILY_SPLITS
from trustswap.verify.base import RuleResult, worst
from trustswap.verify.run import RULES, fired, to_record, verify
from trustswap.verify.v1_schema import v1_schema
from trustswap.verify.v2_grounding import v2_grounding
from trustswap.verify.v3_temporal import v3_temporal
from trustswap.verify.v4_calendar import explicit_dates, v4_calendar


def _flags(results, verdicts=("HARD", "SOFT")):
    return [(rule, path, r.verdict, r.detail) for rule, fields in results.items()
            for path, r in fields.items() if r.verdict in verdicts]


def _fixed_trade(seed=3):
    rng = Random(seed)
    while True:
        trade = sample_trade(rng)
        if trade.fixed_leg is not None:
            return trade


def _set(trade, path, **changes):
    field = trade_fields(trade)[path]
    for key, value in changes.items():
        setattr(field, key, value)
    return trade


# ---------- clean data ----------

def test_clean_sampled_trades_never_flag_v1_v3_v4():
    rng = Random(20261124)
    for _ in range(500):
        results = verify(sample_trade(rng), "", ("V1", "V3", "V4"))
        assert _flags(results) == []


def test_rules_never_modify_the_trade():
    trade = _fixed_trade()
    before = trade.model_dump_json()
    for rule in RULES.values():
        rule(trade, "")
    assert trade.model_dump_json() == before


def test_clean_rendered_documents_pass_all_rules(tmp_path):
    for i, family in enumerate(FAMILY_SPLITS):
        trade = sample_trade(Random(100 + i))
        rendered = render_trade(trade, seed=100 + i, doc_id=f"clean_{i}", template_family=family, output_dir=tmp_path)
        results = verify(rendered.gold.trade, rendered.text)
        assert _flags(results) == [], family
        for path, field in trade_fields(rendered.gold.trade).items():
            if field.value is not None:
                assert results["V2"][path].verdict == "PASS", (family, path)


# ---------- V1 schema ----------

def test_v1_missing_required_field_is_hard():
    trade = _set(_fixed_trade(), "fixed_leg.fixed_rate", value=None, span=None, status="absent")
    assert v1_schema(trade)["fixed_leg.fixed_rate"].verdict == "HARD"


def test_v1_invalid_codes_are_hard():
    trade = _fixed_trade()
    _set(trade, "floating_leg.notional_currency", value="XXQ")
    _set(trade, "floating_leg.floating_rate_option", value="USD-FAKE")
    _set(trade, "fixed_leg.day_count_fraction", value="ACT/999")
    _set(trade, "fixed_leg.payment_frequency", value="quarterly")
    results = v1_schema(trade)
    for path in ("floating_leg.notional_currency", "floating_leg.floating_rate_option",
                 "fixed_leg.day_count_fraction", "fixed_leg.payment_frequency"):
        assert results[path].verdict == "HARD", path


def test_v1_status_handling():
    trade = _fixed_trade()
    _set(trade, "fixed_leg.day_count_fraction", value=None, span=None, status="by_reference")
    _set(trade, "fixed_leg.fixed_rate", value=None, span=None, status="ambiguous")
    _set(trade, "notice_address", value=None, span=None, status="absent")
    results = v1_schema(trade)
    assert results["fixed_leg.day_count_fraction"].verdict == "PASS"
    assert results["fixed_leg.fixed_rate"].verdict == "SOFT"
    assert results["notice_address"].verdict == "PASS"


def test_v1_unknown_but_wellformed_center_is_soft():
    trade = _set(_fixed_trade(), "floating_leg.business_centers", value=["CHZU"])
    assert v1_schema(trade)["floating_leg.business_centers"].verdict == "SOFT"


# ---------- V2 grounding ----------

TEXT = ("Notional Amount: USD 100,000,000\nFixed Rate: 1.80000 percent\n"
        "Termination Date: 23 June 2016\nFixed Rate Payer: TriMas Company LLC\nSpread: None\n")


def _span(phrase):
    start = TEXT.index(phrase)
    return (start, start + len(phrase))


def test_v2_grounded_values_pass_and_wrong_values_fail():
    trade = _fixed_trade()
    _set(trade, "fixed_leg.notional_amount", value=Decimal("100000000"), span=_span("USD 100,000,000"))
    _set(trade, "fixed_leg.fixed_rate", value=Decimal("0.018"), span=_span("1.80000 percent"))
    _set(trade, "termination_date", value=date(2016, 6, 23), span=_span("23 June 2016"))
    _set(trade, "fixed_leg.payer", value="TriMas Company LLC", span=_span("TriMas Company LLC"))
    _set(trade, "floating_leg.spread", value=Decimal(0), span=_span("None"))
    ok = v2_grounding(trade, TEXT)
    for path in ("fixed_leg.notional_amount", "fixed_leg.fixed_rate", "termination_date",
                 "fixed_leg.payer", "floating_leg.spread"):
        assert ok[path].verdict == "PASS", path

    _set(trade, "fixed_leg.notional_amount", value=Decimal("10000000"))
    _set(trade, "termination_date", value=date(2021, 6, 23))
    bad = v2_grounding(trade, TEXT)
    assert bad["fixed_leg.notional_amount"].verdict == "HARD"
    assert bad["fixed_leg.notional_amount"].residual == pytest.approx(9.0)
    assert bad["termination_date"].verdict == "HARD"
    assert bad["termination_date"].residual == 1826.0


def test_v2_missing_span_is_undet_and_bad_span_is_hard():
    trade = _fixed_trade()
    _set(trade, "fixed_leg.fixed_rate", span=None)
    _set(trade, "fixed_leg.notional_amount", span=_span("TriMas Company LLC"))
    _set(trade, "trade_date", span=(len(TEXT) - 2, len(TEXT) + 50))
    results = v2_grounding(trade, TEXT)
    assert results["fixed_leg.fixed_rate"].verdict == "UNDET"
    assert results["fixed_leg.notional_amount"].verdict == "HARD"
    assert results["trade_date"].verdict == "HARD"


def test_v2_near_match_is_soft():
    trade = _set(_fixed_trade(), "fixed_leg.payer", value="TriMas Company LLP", span=_span("TriMas Company LLC"))
    assert v2_grounding(trade, TEXT)["fixed_leg.payer"].verdict == "SOFT"


# ---------- V3 temporal ----------

def test_v3_ordering_violations_are_hard():
    trade = _fixed_trade()
    eff = trade.effective_date.value
    _set(trade, "termination_date", value=eff)
    assert v3_temporal(trade)["termination_date"].verdict == "HARD"

    trade = _fixed_trade()
    _set(trade, "trade_date", value=date(trade.effective_date.value.year + 1, 1, 15))
    result = v3_temporal(trade)
    assert result["trade_date"].verdict == "HARD"
    assert result["trade_date"].residual > 0


def test_v3_implausible_tenor_is_soft_and_missing_is_undet():
    trade = _fixed_trade()
    _set(trade, "termination_date", value=date(trade.effective_date.value.year + 80, 1, 15))
    assert v3_temporal(trade)["termination_date"].verdict == "SOFT"
    _set(trade, "effective_date", value=None, span=None, status="absent")
    _set(trade, "trade_date", value=None, span=None, status="absent")
    assert v3_temporal(trade)["trade_date"].verdict == "UNDET"


# ---------- V4 calendar ----------

def _keybank_trade():
    """Shape of the annotated 2017 KeyBank confirmation: monthly, New York, Modified Following."""
    trade = _fixed_trade()
    for path, value in (("trade_date", date(2017, 3, 27)), ("effective_date", date(2017, 4, 1)),
                        ("termination_date", date(2022, 4, 1))):
        _set(trade, path, value=value)
    for leg in ("fixed_leg", "floating_leg"):
        _set(trade, f"{leg}.payment_frequency", value="1M")
        _set(trade, f"{leg}.business_centers", value=["USNY"])
        _set(trade, f"{leg}.business_day_convention", value="MODFOLLOWING")
    return trade


def test_v4_real_unadjusted_confirmation_is_not_hard():
    text = ("Fixed Rate Payer Payment Dates: Monthly, on the 1st, commencing on 01-May-17 up to and including "
            "the Termination date, subject to adjustment in accordance with Modified Following\n")
    results = v4_calendar(_keybank_trade(), text)
    assert results["effective_date"].verdict == "SOFT"   # 1 April 2017 is a Saturday
    assert all(r.verdict != "HARD" for r in results.values())


def test_v4_off_schedule_payment_date_is_hard():
    text = "Fixed Rate Payer Payment Dates: commencing on 15-May-17, then monthly\n"
    result = v4_calendar(_keybank_trade(), text)["effective_date"]
    assert result.verdict == "HARD"
    assert result.residual == 14.0


def test_v4_unknown_center_is_undet():
    trade = _keybank_trade()
    for leg in ("fixed_leg", "floating_leg"):
        _set(trade, f"{leg}.business_centers", value=["CHZU"])
    assert all(r.verdict == "UNDET" for r in v4_calendar(trade, "").values())


def test_explicit_date_extraction_only_reads_schedule_lines():
    text = "Trade Date: 27-Mar-17\nPayment Dates: commencing 2017-05-01\nAgreed short initial stub: 2017-04-01 to 2017-05-15\n"
    stated, stubs = explicit_dates(text)
    assert stated == [date(2017, 5, 1)]
    assert stubs == [date(2017, 4, 1), date(2017, 5, 15)]


# ---------- adversarial defects ----------

def _literal_extraction(case, rendered):
    gold = rendered.gold.trade.model_copy(deep=True)
    for path, field in trade_fields(gold).items():
        if field.status == "ambiguous":
            field.value, field.status, field.span = trade_fields(case.trade)[path].value, "explicit", None
    return gold


@pytest.fixture(scope="module")
def adversarial(tmp_path_factory):
    root = tmp_path_factory.mktemp("adv")
    parent = sample_trade(Random(7))
    cases = {}
    for kind in D.DEFECT_TYPES:
        for severity in (1, 2):
            case = D.INJECTORS[kind](parent, 41, severity=severity)
            rendered = D.render_case(case, output_dir=root)
            cases[(kind, severity, False)] = (case, verify(_literal_extraction(case, rendered), rendered.text))
        control = D.make_control(kind, parent, 41)
        rendered = D.render_case(control, output_dir=root)
        cases[(kind, 1, True)] = (control, verify(_literal_extraction(control, rendered), rendered.text))
    return cases


DETECTABLE = [  # (defect, severity, rule): predictions V1-V4 can satisfy by construction
    ("non_standard_calendar", 1, "V4"), ("non_standard_calendar", 2, "V4"),
    ("stub_ambiguity", 1, "V4"), ("stub_ambiguity", 2, "V4"),
    ("ocr_digit_corruption", 1, "V2"), ("ocr_digit_corruption", 2, "V2"),
    ("scale_error", 1, "V2"), ("scale_error", 2, "V2"),
    ("missing_information", 2, "V1"),
]


@pytest.mark.parametrize("kind,severity,rule", DETECTABLE)
def test_defects_fire_their_expected_rule(adversarial, kind, severity, rule):
    case, results = adversarial[(kind, severity, False)]
    assert rule in case.metadata.expected_rules
    assert fired(results, rule, case.metadata.affected_fields)


@pytest.mark.parametrize("kind", D.DEFECT_TYPES)
def test_controls_have_no_hard_verdict(adversarial, kind):
    _, results = adversarial[(kind, 1, True)]
    assert _flags(results, ("HARD",)) == []


# ---------- helpers ----------

def test_worst_and_record_format():
    combined = worst(RuleResult(verdict="PASS", residual=0.0, detail="a"),
                     RuleResult(verdict="HARD", residual=3.0, detail="b"),
                     RuleResult(verdict="HARD", residual=5.0, detail="c"))
    assert combined.verdict == "HARD" and combined.residual == 5.0
    record = to_record("doc_1", {"V3": {"termination_date": combined}})
    assert record == {"doc_id": "doc_1", "rules": {"V3": {"termination_date": {
        "verdict": "HARD", "residual": 5.0, "detail": combined.detail}}}}
