"""Step 8: verifier rules V5 (accrual), V6 (cross-field), V7 (convention prior).

Adversarial assertions cover only the defect/rule pairs that V5-V7 can detect by
construction. Other outcomes (e.g. multiple_notionals firing V6 although H4
predicts "silent") are recorded as observations in docs/verifier_rules.md and
measured in Step 9; they are not test requirements.
"""
from datetime import date
from decimal import Decimal
from random import Random

import pytest

from trustswap.synth import defects as D
from trustswap.synth.render import trade_fields
from trustswap.synth.sampler import sample_trade
from trustswap.verify.amounts import currency_amounts
from trustswap.verify.run import HARD_RULES, RULES, fired, verify
from trustswap.verify.v4_calendar import _schedules
from trustswap.verify.v5_accrual import accrual, v5_accrual
from trustswap.verify.v6_crossfield import v6_crossfield
from trustswap.verify.v7_convention import v7_convention


def _flags(results, verdicts=("HARD", "SOFT")):
    return [(rule, path, r.verdict, r.detail) for rule, fields in results.items()
            for path, r in fields.items() if r.verdict in verdicts]


def _trade(currency="USD", fixed=True, seed=0):
    rng = Random(seed)
    while True:
        trade = sample_trade(rng)
        if trade.floating_leg.notional_currency.value == currency and (trade.fixed_leg is not None) == fixed:
            return trade


def _set(trade, path, **changes):
    field = trade_fields(trade)[path]
    for key, value in changes.items():
        setattr(field, key, value)
    return trade


def _first_period_amount(trade):
    leg = trade.fixed_leg
    dates = sorted(_schedules(trade.effective_date.value, trade.termination_date.value, leg, [])[0])
    amount = accrual(leg.notional_amount.value, leg.fixed_rate.value, leg.day_count_fraction.value, dates[0], dates[1])
    return amount, dates[0], dates[1]


# ---------- clean data ----------

def test_rules_are_registered():
    assert list(RULES) == ["V1", "V2", "V3", "V4", "V5", "V6", "V7"]
    assert "V7" not in HARD_RULES


def test_clean_sampled_trades_never_flag_v5_v7():
    rng = Random(20261125)
    for _ in range(1000):
        assert _flags(verify(sample_trade(rng), "", ("V5", "V6", "V7"))) == []


def test_currency_amounts_reads_only_money():
    assert currency_amounts("Notional Amount: USD 50,000,000") == [(Decimal(50_000_000), "USD")]
    assert currency_amounts("Notional: 388,000,000 EUR") == [(Decimal(388_000_000), "EUR")]
    assert currency_amounts("EUR 388.5 million") == [(Decimal("388500000.0"), "EUR")]
    assert currency_amounts("Notional: EUR 388,OOO,OOO") == []  # no partial read of OCR damage
    assert currency_amounts("Trade Date 2017, rate 2.5%, ACT/360") == []


# ---------- V5 accrual ----------

def test_v5_stated_fixed_amount_recomputes():
    trade = _trade()
    amount, *_ = _first_period_amount(trade)
    text = f"Fixed Amount: USD {amount:,.2f}"
    result = v5_accrual(trade, text)["fixed_leg.fixed_rate"]
    assert result.verdict == "PASS" and result.residual < 1.0


def test_v5_wrong_amount_is_hard_and_near_miss_is_soft():
    trade = _trade()
    amount, *_ = _first_period_amount(trade)
    hard = v5_accrual(trade, f"Fixed Amount: USD {amount * 1.10:,.2f}")["fixed_leg.notional_amount"]
    assert hard.verdict == "HARD" and hard.residual > 0
    # the closest regenerated period may differ by a day or two, so stay inside 1% of every candidate
    soft = v5_accrual(trade, f"Fixed Amount: USD {amount * 1.004:,.2f}")["fixed_leg.notional_amount"]
    assert soft.verdict in {"SOFT", "PASS"}
    assert soft.verdict != "HARD"


def test_v5_uses_a_stated_period():
    trade = _trade()
    amount, start, end = _first_period_amount(trade)
    line = f"Fixed Amount for the period {start} to {end}: USD {amount:,.2f}"
    assert v5_accrual(trade, line)["fixed_leg.day_count_fraction"].verdict == "PASS"
    wrong_basis = _set(trade.model_copy(deep=True), "fixed_leg.day_count_fraction",
                       value="ACT/365.FIXED" if trade.fixed_leg.day_count_fraction.value != "ACT/365.FIXED" else "ACT/360")
    assert v5_accrual(wrong_basis, line)["fixed_leg.day_count_fraction"].verdict in {"SOFT", "HARD"}


def test_v5_undet_without_amounts_and_empty_for_basis_swaps():
    assert v5_accrual(_trade(), "Notional Amount: USD 10,000,000")["fixed_leg.fixed_rate"].verdict == "UNDET"
    assert v5_accrual(_trade(fixed=False), "Fixed Amount: USD 100") == {}


# ---------- V6 cross-field ----------

def test_v6_index_currency_mismatch_is_hard():
    trade = _set(_trade(), "floating_leg.floating_rate_option", value="GBP-SONIA")
    assert v6_crossfield(trade, "")["floating_leg.floating_rate_option"].verdict == "HARD"


def test_v6_leg_notionals_must_agree_unless_amortising():
    trade = _trade()
    _set(trade, "fixed_leg.notional_amount", value=trade.floating_leg.notional_amount.value * 2)
    assert v6_crossfield(trade, "")["fixed_leg.notional_amount"].verdict == "HARD"
    assert v6_crossfield(trade, "The Notional Amount amortises per Schedule A.")["fixed_leg.notional_amount"].verdict == "SOFT"


def test_v6_payer_receiver_mirror():
    trade = _trade()
    _set(trade, "fixed_leg.receiver", value=trade.fixed_leg.payer.value)
    results = v6_crossfield(trade, "")
    assert results["fixed_leg.payer"].verdict == "HARD"


def test_v6_stated_notional_must_match_extraction():
    trade = _trade()
    value = trade.floating_leg.notional_amount.value
    assert v6_crossfield(trade, f"Notional Amount: USD {value:,}")["floating_leg.notional_amount"].verdict == "PASS"
    result = v6_crossfield(trade, f"Notional Amount: USD {value * 10:,}")["floating_leg.notional_amount"]
    assert result.verdict == "HARD" and result.residual == pytest.approx(9.0)  # relative to the extracted value
    # amounts in another currency are not compared
    assert v6_crossfield(trade, "Notional Amount: JPY 1,000")["floating_leg.notional_amount"].verdict == "PASS"


def test_v6_legacy_index_with_fallback_or_after_cessation_is_soft():
    trade = _trade()
    _set(trade, "floating_leg.floating_rate_option", value="USD-LIBOR-BBA")
    _set(trade, "trade_date", value=date(2019, 3, 1))
    assert v6_crossfield(trade, "")["floating_leg.floating_rate_option"].verdict == "PASS"
    text = "Fallback from USD-LIBOR-BBA to USD-SOFR may apply."
    assert v6_crossfield(trade, text)["floating_leg.floating_rate_option"].verdict == "SOFT"
    _set(trade, "trade_date", value=date(2023, 3, 1))
    assert v6_crossfield(trade, "")["floating_leg.floating_rate_option"].verdict == "SOFT"


# ---------- V7 convention prior ----------

def test_v7_unusual_conventions_are_soft_never_hard():
    trade = _trade()
    _set(trade, "floating_leg.day_count_fraction", value="30E/360")
    _set(trade, "fixed_leg.day_count_fraction", value="ACT/ACT.ISDA")
    _set(trade, "fixed_leg.business_centers", value=["USNY", "JPTO"])
    results = v7_convention(trade, "")
    assert results["floating_leg.day_count_fraction"].verdict == "SOFT"
    assert results["fixed_leg.day_count_fraction"].verdict == "SOFT"
    assert results["fixed_leg.business_centers"].verdict == "SOFT"
    assert all(r.verdict != "HARD" for r in results.values())


def test_v7_usd_libor_london_and_new_york_is_standard():
    trade = _trade()
    _set(trade, "trade_date", value=date(2019, 3, 1))
    _set(trade, "floating_leg.floating_rate_option", value="USD-LIBOR-BBA")
    _set(trade, "floating_leg.business_centers", value=["USNY", "GBLO"])
    assert v7_convention(trade, "")["floating_leg.business_centers"].verdict == "PASS"
    _set(trade, "floating_leg.business_centers", value=["USNY"])
    assert v7_convention(trade, "")["floating_leg.business_centers"].verdict == "SOFT"


def test_v7_missing_values_are_undet():
    trade = _set(_trade(), "floating_leg.day_count_fraction", value=None, span=None, status="absent")
    assert v7_convention(trade, "")["floating_leg.day_count_fraction"].verdict == "UNDET"


# ---------- adversarial defects ----------

def _literal_extraction(case, rendered):
    gold = rendered.gold.trade.model_copy(deep=True)
    for path, field in trade_fields(gold).items():
        if field.status == "ambiguous":
            field.value, field.status, field.span = trade_fields(case.trade)[path].value, "explicit", None
    return gold


@pytest.fixture(scope="module")
def adversarial(tmp_path_factory):
    root = tmp_path_factory.mktemp("adv8")
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


DETECTABLE = [  # predictions V5-V7 satisfy by construction
    ("contradiction", 1, "V6"), ("contradiction", 2, "V6"),
    ("legacy_index_fallback", 1, "V6"), ("legacy_index_fallback", 2, "V6"),
    ("unusual_day_count", 1, "V7"), ("unusual_day_count", 2, "V7"),
]


@pytest.mark.parametrize("kind,severity,rule", DETECTABLE)
def test_defects_fire_their_expected_rule(adversarial, kind, severity, rule):
    case, results = adversarial[(kind, severity, False)]
    assert rule in case.metadata.expected_rules
    assert fired(results, rule, case.metadata.affected_fields)


@pytest.mark.parametrize("kind", D.DEFECT_TYPES)
def test_controls_have_no_hard_verdict_under_all_rules(adversarial, kind):
    _, results = adversarial[(kind, 1, True)]
    assert _flags(results, ("HARD",)) == []


def test_v7_is_never_hard_on_any_case(adversarial):
    for _, results in adversarial.values():
        assert all(r.verdict != "HARD" for r in results["V7"].values())
