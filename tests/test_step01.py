"""Step 01: handwritten swap examples and deterministic normalisation."""
from decimal import Decimal

import pytest
from pydantic import ValidationError

from trustswap.common.schema import ExtractedField, InterestRateSwap
from trustswap.common.normalise import (
    normalise_date, normalise_rate, normalise_notional,
    normalise_day_count, normalise_business_day_convention,
)

# Twenty independently specified examples, including partial and uncertain terms.
RECORDS = [
    ("2024-01-02", "USD", "250000000", "0.0425", "explicit"),
    ("2024-02-29", "EUR", "1000000", "0.03", "explicit"),
    ("2024-03-01", "GBP", "2000000", "0.04", "by_reference"),
    ("2024-04-03", "JPY", "500000000", "0.001", "ambiguous"),
    ("2024-05-06", "CHF", "3000000", "-0.002", "explicit"),
    ("2024-06-07", "CAD", "4000000", "0", "explicit"),
    ("2024-07-08", "AUD", "5000000", "0.05", "explicit"),
    ("2024-08-09", "NZD", "6000000", "0.06", "by_reference"),
    ("2024-09-10", "SEK", "7000000", "0.035", "explicit"),
    ("2024-10-11", "NOK", "8000000", "0.045", "ambiguous"),
    ("2024-11-12", "DKK", "9000000", "0.025", "explicit"),
    ("2024-12-13", "PLN", "10000000", "0.055", "explicit"),
    ("2025-01-14", "CZK", "11000000", "0.065", "by_reference"),
    ("2025-02-15", "HUF", "12000000", "0.075", "explicit"),
    ("2025-03-16", "SGD", "13000000", "0.015", "explicit"),
    ("2025-04-17", "HKD", "14000000", "0.085", "ambiguous"),
    ("2025-05-18", "ZAR", "15000000", "0.095", "explicit"),
    ("2025-06-19", "MXN", "16000000.25", "0.105", "explicit"),
    ("2025-07-20", None, None, None, "absent"),
    (None, "USD", "0", "0.00001", "by_reference"),
]


def field(value, status="explicit"):
    if value is None:
        return {"value": None, "span": None, "status": "absent"}
    return {"value": value, "span": [2, 12] if status == "explicit" else None, "status": status}


@pytest.mark.parametrize("trade,currency,amount,rate,status", RECORDS)
def test_swap_round_trip(trade, currency, amount, rate, status):
    common = {
        "payer": field("Party A"), "receiver": field("Party B"),
        "notional_amount": field(amount, status), "notional_currency": field(currency, status),
        "payment_frequency": field("P6M"), "reset_frequency": field("P3M"),
        "day_count_fraction": field("ACT/360"),
        "business_day_convention": field("MODFOLLOWING"),
        "business_centers": field(["USNY", "GBLO"]),
    }
    record = {
        "trade_date": field(trade), "effective_date": field("2025-08-01"),
        "termination_date": field("2030-08-01"),
        "fixed_leg": dict(common, fixed_rate=field(rate, status)),
        "floating_leg": dict(common, payer=field("Party B"), receiver=field("Party A"),
                             floating_rate_option=field("USD-SOFR", "by_reference"),
                             spread=field("0.001")),
        "notice_address": field(None),
    }
    swap = InterestRateSwap.model_validate(record)
    assert InterestRateSwap.model_validate_json(swap.model_dump_json()) == swap
    assert swap.fixed_leg.notional_amount.value == (Decimal(amount) if amount is not None else None)
    assert swap.floating_leg.floating_rate_option.status == "by_reference"


@pytest.mark.parametrize("value", ["4.25%", "0.0425", "425 bps"])
def test_rates(value):
    assert normalise_rate(value) == Decimal("0.0425")


@pytest.mark.parametrize("value,currency", [("USD 250,000,000", "USD"), ("250 million", None), ("USD 250mm", "USD")])
def test_notionals(value, currency):
    assert normalise_notional(value) == (Decimal("250000000"), currency)


@pytest.mark.parametrize("value", ["2024-02-29", "29 February 2024", "Feb 29, 2024", "2024/02/29"])
def test_dates(value):
    assert normalise_date(value) == "2024-02-29"


def test_conventions():
    assert normalise_day_count("Actual/360") == "ACT/360"
    assert normalise_day_count("ACT/360") == "ACT/360"
    assert normalise_business_day_convention("Modified Following") == "MODFOLLOWING"
    assert normalise_business_day_convention("MODFOLLOWING") == "MODFOLLOWING"


@pytest.mark.parametrize("fn,value", [(normalise_date, "01/02/2024"), (normalise_date, "2023-02-29"), (normalise_rate, "bad"), (normalise_notional, "USD 25,00"), (normalise_notional, "XYZ 250"), (normalise_day_count, "unknown"), (normalise_business_day_convention, "unknown")])
def test_invalid_normalisation(fn, value):
    with pytest.raises(ValueError):
        fn(value)


@pytest.mark.parametrize("data", [{"span": [-1, 4], "status": "explicit"}, {"span": [4, 4], "status": "explicit"}, {"status": "invalid"}, {"value": "x", "status": "absent"}])
def test_invalid_fields(data):
    with pytest.raises(ValidationError):
        ExtractedField[str].model_validate(data)


def test_currency_override():
    assert normalise_notional("250 million", "usd") == (Decimal("250000000"), "USD")
    with pytest.raises(ValueError):
        normalise_notional("USD 250mm", "EUR")
