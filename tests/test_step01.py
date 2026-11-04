"""Step 01: handwritten swap examples and deterministic normalisation."""
from decimal import Decimal

import pytest
from pydantic import ValidationError

from trustswap.common.schema import ExtractedField, InterestRateSwap
from trustswap.common.normalise import (
    normalise_date, normalise_rate, normalise_notional,
    normalise_day_count, normalise_business_day_convention,
    normalise_frequency, normalise_business_centers, normalise_floating_rate_option,
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


@pytest.mark.parametrize("value", ["4.25%", "425 bps"])
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


@pytest.mark.parametrize("text,expected", [
    ("4.25 percent", "0.0425"), ("4.25 per cent", "0.0425"),
    ("4.25 percent per annum", "0.0425"), ("4.25% p.a.", "0.0425"),
    ("425 bp", "0.0425"), ("425 bps", "0.0425"),
    ("425 basis points", "0.0425"), ("plus 4.25 percent", "0.0425"),
    ("minus 425 bp", "-0.0425"), ("(4.25%)", "-0.0425"),
    ("1.80000 percent", "0.0180000"),
])
def test_extended_rates(text, expected):
    assert normalise_rate(text) == Decimal(expected)


@pytest.mark.parametrize("value", ["0.0425", "4.25", Decimal("0.0425"), 0, 4.25])
def test_bare_rates_are_ambiguous(value):
    with pytest.raises(ValueError, match="^ambiguous rate unit$"):
        normalise_rate(value)


def test_explicit_rate_units():
    assert normalise_rate("0.0425", unit="decimal") == Decimal("0.0425")
    assert normalise_rate("4.25", unit="percent") == Decimal("0.0425")
    with pytest.raises(ValueError, match="Conflicting"):
        normalise_rate("4.25%", unit="decimal")
    with pytest.raises(ValueError, match="Unsupported"):
        normalise_rate("4.25", unit="unknown")


@pytest.mark.parametrize("text,amount,currency", [
    ("USD100,000,000", "100000000", "USD"),
    ("$250 million", "250000000", "USD"),
    ("US$250 million", "250000000", "USD"),
    ("U.S.$250 million", "250000000", "USD"),
    ("€250 million", "250000000", "EUR"),
    ("£250 million", "250000000", "GBP"),
    ("250 million USD", "250000000", "USD"),
    ("250m", "250000000", None), ("250mm", "250000000", None),
])
def test_extended_notionals(text, amount, currency):
    assert normalise_notional(text) == (Decimal(amount), currency)


@pytest.mark.parametrize("text", ["250M", "USD 250 M", "250 M USD"])
def test_ambiguous_notional_multiplier(text):
    with pytest.raises(ValueError, match="ambiguous"):
        normalise_notional(text)


@pytest.mark.parametrize("text,expected", [
    ("Actual/365 (Fixed)", "ACT/365.FIXED"),
    ("Actual/365 [Fixed]", "ACT/365.FIXED"),
    ("Actual/Actual (ISDA)", "ACT/ACT.ISDA"),
    ("Actual/Actual [ISDA]", "ACT/ACT.ISDA"),
    ("Bond Basis", "30/360"), ("30/360 (Bond Basis)", "30/360"),
    ("Act/Act ICMA", "ACT/ACT.ICMA"),
])
def test_extended_day_counts(text, expected):
    assert normalise_day_count(text) == expected
    assert normalise_day_count(expected) == expected


def test_ambiguous_day_count():
    with pytest.raises(ValueError, match="ambiguous"):
        normalise_day_count("ACT/365")


def test_business_day_suffix():
    assert normalise_business_day_convention("Modified Following Business Day Convention") == "MODFOLLOWING"


@pytest.mark.parametrize("text", ["20th March 2012", "20-Mar-2012", "March 20 2012"])
def test_extended_dates(text):
    assert normalise_date(text) == "2012-03-20"


def test_numeric_month_day_date_rejected():
    with pytest.raises(ValueError):
        normalise_date("03/20/2012")


@pytest.mark.parametrize("text,expected", [
    ("quarterly", "3M"), ("semi-annual", "6M"), ("3 months", "3M"),
    ("3M", "3M"), ("annual", "1Y"),
])
def test_frequencies(text, expected):
    assert normalise_frequency(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("London", ["GBLO"]), ("New York", ["USNY"]),
    ("TARGET", ["EUTA"]), ("Tokyo", ["JPTO"]),
    ("London and New York, TARGET and Tokyo", ["EUTA", "GBLO", "JPTO", "USNY"]),
    ("London, GBLO", ["GBLO"]),
])
def test_business_centers(text, expected):
    assert normalise_business_centers(text) == expected


@pytest.mark.parametrize("text", [
    "USD-LIBOR-BBA", "USD-SOFR", "USD-SOFR-COMPOUND",
    "EUR-EURIBOR-Reuters", "EUR-EuroSTR", "GBP-SONIA",
    "  usd  sofr compound  ",
])
def test_floating_rate_options(text):
    assert normalise_floating_rate_option(text) == "-".join(text.upper().split())


@pytest.mark.parametrize("fn,text", [
    (normalise_frequency, "sometimes"), (normalise_frequency, "0M"),
    (normalise_business_centers, "unknown"),
    (normalise_floating_rate_option, "USD-UNKNOWN"),
])
def test_unknown_new_normalisers(fn, text):
    with pytest.raises(ValueError):
        fn(text)


def test_floating_rate_reference_is_used(tmp_path, monkeypatch):
    from trustswap.common import normalise

    reference = tmp_path / "floating_rate_options.txt"
    reference.write_text("TEST-CODE\n", encoding="utf-8")
    monkeypatch.setattr(normalise, "_FLOATING_RATE_CODES", reference)
    assert normalise_floating_rate_option("test code") == "TEST-CODE"
    with pytest.raises(ValueError):
        normalise_floating_rate_option("USD-SOFR")


@pytest.mark.parametrize("value,expected", [
    ("27-Mar-17", "2017-03-27"), ("01-Apr-22", "2022-04-01"),
    ("15-Jun-98", "1998-06-15"), ("27-Mar-2017", "2017-03-27"),
    ("01-Jan-00", "2000-01-01"), ("31-Dec-49", "2049-12-31"),
    ("01-Jan-50", "1950-01-01"), ("15-Jun-68", "1968-06-15"),
    ("31-Dec-99", "1999-12-31"), (" 27-Mar-17 ", "2017-03-27"),
])
def test_named_month_short_year_pivot(value, expected):
    assert normalise_date(value) == expected


@pytest.mark.parametrize("value", ["03/04/17", "03-04-17", "29-Feb-50"])
def test_numeric_dates_and_invalid_short_year_date_refused(value):
    with pytest.raises(ValueError):
        normalise_date(value)


@pytest.mark.parametrize("value", ["None", "N/A", "Not Applicable", "none", "n/a", "NOT APPLICABLE", "  None  "])
def test_explicit_no_spread_wording_is_zero(value):
    assert normalise_rate(value, is_spread=True) == Decimal(0)


@pytest.mark.parametrize("value", ["None", "N/A", "Not Applicable"])
def test_no_spread_wording_is_not_a_generic_zero_rate(value):
    with pytest.raises(ValueError):
        normalise_rate(value)


@pytest.mark.parametrize("value", [None, "", "absent"])
def test_missing_spread_does_not_invent_zero(value):
    with pytest.raises(ValueError):
        normalise_rate(value, is_spread=True)
