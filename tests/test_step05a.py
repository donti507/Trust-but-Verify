"""Step 5A sampling checks against QuantLib calendars and market assumptions."""
from datetime import date
from decimal import Decimal
from random import Random

import pytest
from pydantic import ValidationError

from trustswap.common.schema import InterestRateSwap
from trustswap.synth.markets import (
    FIXED_DAY_COUNTS, FREQUENCIES, MARKETS, TENORS, calendar_for, market_for,
)
from trustswap.synth.sampler import rate_bounds, sample_trade


@pytest.fixture(scope="module")
def trades():
    rng = Random(507)
    return [sample_trade(rng) for _ in range(1000)]


def fields(value):
    if isinstance(value, dict):
        if "status" in value:
            yield value
        else:
            for child in value.values():
                yield from fields(child)


def test_thousand_trades_validate(trades):
    assert len(trades) == 1000
    for trade in trades:
        assert InterestRateSwap.model_validate_json(trade.model_dump_json()) == trade
        for field in fields(trade.model_dump()):
            assert field["status"] == "explicit"
            assert field["span"] is None
            assert field["value"] is not None
    assert any(trade.fixed_leg is not None for trade in trades)
    assert any(trade.floating_leg_2 is not None for trade in trades)


def test_markets_and_values(trades):
    currencies = set()
    for trade in trades:
        currency = trade.floating_leg.notional_currency.value
        currencies.add(currency)
        market = market_for(currency, trade.trade_date.value)
        legs = [trade.fixed_leg or trade.floating_leg_2, trade.floating_leg]
        assert legs[0].payer.value == legs[1].receiver.value
        assert legs[0].receiver.value == legs[1].payer.value
        assert legs[0].notional_amount.value == legs[1].notional_amount.value
        for leg in legs:
            assert leg.notional_currency.value == currency
            assert leg.business_centers.value == list(market.centers)
            assert leg.business_day_convention.value in {"MODFOLLOWING", "FOLLOWING"}
            assert leg.payment_frequency.value in FREQUENCIES
            assert leg.reset_frequency.value in FREQUENCIES
            assert Decimal(5_000_000) <= leg.notional_amount.value <= Decimal(1_000_000_000)
            assert leg.notional_amount.value % Decimal(1_000_000) == 0
            if hasattr(leg, "floating_rate_option"):
                if leg is trade.floating_leg:
                    assert leg.floating_rate_option.value == market.floating_option
                else:
                    assert leg.floating_rate_option.value in {market.floating_option, "USD-SOFR"}
                    if leg.floating_rate_option.value != market.floating_option:
                        assert currency == "USD" and date(2018, 1, 1) <= trade.trade_date.value < date(2022, 1, 1)
                assert leg.day_count_fraction.value == market.floating_day_count
                assert Decimal("-0.005") <= leg.spread.value <= Decimal("0.02")
            else:
                assert leg.day_count_fraction.value in FIXED_DAY_COUNTS
                low, high = rate_bounds(trade.trade_date.value)
                assert Decimal(low) / 10000 <= leg.fixed_rate.value <= Decimal(high) / 10000
                assert leg.fixed_rate.value == leg.fixed_rate.value.quantize(Decimal("0.0001"))
    assert currencies == set(MARKETS)


def test_dates_against_quantlib(trades):
    import QuantLib as ql

    def qdate(value):
        return ql.Date(value.day, value.month, value.year)

    for trade in trades:
        assert date(2005, 1, 1) <= trade.trade_date.value <= date(2026, 12, 31)
        assert trade.trade_date.value <= trade.effective_date.value < trade.termination_date.value
        calendar = calendar_for(trade.floating_leg.business_centers.value)
        effective = qdate(trade.effective_date.value)
        termination = qdate(trade.termination_date.value)
        assert calendar.isBusinessDay(effective)
        assert calendar.isBusinessDay(termination)
        lag = 0 if trade.floating_leg.notional_currency.value == "GBP" else 2
        assert effective == calendar.advance(qdate(trade.trade_date.value), lag, ql.Days)
        convention = {"MODFOLLOWING": ql.ModifiedFollowing, "FOLLOWING": ql.Following}[
            trade.floating_leg.business_day_convention.value]
        assert any(termination == calendar.advance(effective, ql.Period(years, ql.Years), convention)
                   for years in TENORS)


def test_fixed_seed():
    left, right = Random(42), Random(42)
    assert [sample_trade(left).model_dump_json() for _ in range(50)] == [
        sample_trade(right).model_dump_json() for _ in range(50)]


def test_libor_cutover():
    assert market_for("USD", date(2021, 12, 31)).floating_option == "USD-LIBOR-BBA"
    assert market_for("USD", date(2021, 12, 31)).centers == ("USNY", "GBLO")
    assert market_for("USD", date(2022, 1, 1)).floating_option == "USD-SOFR"
    assert market_for("USD", date(2022, 1, 1)).centers == ("USNY",)


def test_exactly_two_legs(trades):
    fixed = next(trade for trade in trades if trade.fixed_leg is not None)
    floating = next(trade for trade in trades if trade.floating_leg_2 is not None)
    invalid = fixed.model_dump()
    invalid["floating_leg_2"] = floating.floating_leg_2.model_dump()
    with pytest.raises(ValidationError, match="either"):
        InterestRateSwap.model_validate(invalid)
    invalid["floating_leg_2"] = None
    invalid["fixed_leg"] = None
    with pytest.raises(ValidationError, match="either"):
        InterestRateSwap.model_validate(invalid)


def test_gold_jsonl(trades, tmp_path):
    import json
    from trustswap.common.records import GoldRecord
    from trustswap.synth.gold import write_gold_record

    for index, trade in enumerate(trades[:10]):
        record = write_gold_record(trade, doc_id=f"syn_{index:06d}", seed=507, output_dir=tmp_path)
        assert record.trade == trade
    lines = (tmp_path / "synthetic" / "gold.jsonl").read_text().splitlines()
    assert len(lines) == 10
    for index, line in enumerate(lines):
        data = json.loads(line)
        assert set(data) == {"doc_id", "collection", "trade", "generator"}
        assert data["doc_id"] == f"syn_{index:06d}"
        assert data["collection"] == "synthetic"
        assert data["trade"] == trades[index].model_dump(mode="json")
        assert data["generator"] == {"seed": 507, "sampler_version": "0.1", "template_family": None}
        assert GoldRecord.model_validate_json(line).trade == trades[index]


def test_gold_collections_and_provenance(trades, tmp_path):
    from trustswap.common.records import GoldRecord
    from trustswap.synth.gold import write_gold_record

    record = write_gold_record(trades[0], doc_id="adv_1", seed=507, collection="adversarial",
                               template_family="family_1", output_dir=tmp_path)
    assert record.generator.template_family == "family_1"
    assert (tmp_path / "adversarial" / "gold.jsonl").is_file()
    assert not (tmp_path / "synthetic").exists()
    with pytest.raises(ValueError, match="only"):
        write_gold_record(trades[0], doc_id="real_1", seed=507, collection="real", output_dir=tmp_path)
    invalid = record.model_dump()
    invalid["generator"] = None
    with pytest.raises(ValidationError, match="generator"):
        GoldRecord.model_validate(invalid)


def test_gold_refuses_raw_outputs(trades, tmp_path):
    from trustswap.synth.gold import write_gold_record

    with pytest.raises(ValueError, match="data/raw"):
        write_gold_record(trades[0], doc_id="syn_1", seed=507, output_dir=tmp_path / "data" / "raw")
    assert not (tmp_path / "data").exists()


def test_trade_dates_are_business_days(trades):
    import QuantLib as ql

    for trade in trades:
        value = trade.trade_date.value
        assert value.weekday() < 5
        calendar = calendar_for(trade.floating_leg.business_centers.value)
        assert calendar.isBusinessDay(ql.Date(value.day, value.month, value.year))


def test_basis_share_and_distinct_legs(trades):
    basis = [trade for trade in trades if trade.floating_leg_2 is not None]
    assert 0.10 <= len(basis) / len(trades) <= 0.20
    for trade in basis:
        first, second = trade.floating_leg, trade.floating_leg_2
        assert (first.floating_rate_option.value, first.reset_frequency.value) != (
            second.floating_rate_option.value, second.reset_frequency.value)
        if first.floating_rate_option.value == second.floating_rate_option.value:
            assert "IBOR" in first.floating_rate_option.value
        else:
            assert trade.trade_date.value.year in {2018, 2019, 2020, 2021}
            assert {first.floating_rate_option.value, second.floating_rate_option.value} == {
                "USD-SOFR", "USD-LIBOR-BBA"}


@pytest.mark.parametrize("currency,candidate,expected", [
    ("USD", date(2024, 7, 6), date(2024, 7, 8)),
    ("USD", date(2024, 7, 4), date(2024, 7, 5)),
    ("GBP", date(2021, 12, 31), date(2021, 12, 31)),
    ("GBP", date(2022, 1, 1), date(2022, 1, 4)),
    ("USD", date(2021, 12, 31), date(2022, 1, 3)),
])
def test_roll_forward_weekends_holidays_and_cutovers(currency, candidate, expected):
    from trustswap.synth.markets import roll_trade_date

    assert roll_trade_date(currency, candidate) == expected


def test_gbp_cutover_and_settlement(trades):
    before = market_for("GBP", date(2021, 12, 31))
    after = market_for("GBP", date(2022, 1, 1))
    assert before.floating_option == "GBP-LIBOR-BBA"
    assert after.floating_option == "GBP-SONIA"
    assert before.floating_day_count == after.floating_day_count == "ACT/365.FIXED"
    assert before.centers == after.centers == ("GBLO",)
    gbp = [trade for trade in trades if trade.floating_leg.notional_currency.value == "GBP"]
    assert gbp
    assert all(trade.effective_date.value == trade.trade_date.value for trade in gbp)


def test_typical_fixed_conventions_and_variation(trades):
    buckets = {}
    for trade in trades:
        if trade.fixed_leg is None:
            continue
        currency = trade.fixed_leg.notional_currency.value
        libor = trade.floating_leg.floating_rate_option.value == "USD-LIBOR-BBA"
        expected = {
            "USD": {("6M", "30/360")} if libor else {("1Y", "ACT/360")},
            "EUR": {("1Y", "30/360")},
            "GBP": {("6M", "ACT/365.FIXED"), ("1Y", "ACT/365.FIXED")},
        }[currency]
        leg = trade.fixed_leg
        typical = (leg.payment_frequency.value, leg.day_count_fraction.value) in expected
        buckets.setdefault((currency, libor), []).append(typical)
    assert len(buckets) == 4
    for results in buckets.values():
        assert 0.65 <= sum(results) / len(results) <= 0.95


def test_sampled_indices_are_recognised(trades):
    from trustswap.common.normalise import normalise_floating_rate_option

    for trade in trades:
        for leg in (trade.floating_leg, trade.floating_leg_2):
            if leg is not None:
                code = leg.floating_rate_option.value
                assert normalise_floating_rate_option(code) == code.upper()
