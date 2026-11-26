"""Read the published CFTC cumulative rates file into a narrow vanilla IRS schema."""
from __future__ import annotations

import csv

import QuantLib as ql
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from io import BytesIO, TextIOWrapper
import random
import re
from urllib.error import HTTPError
from urllib.request import urlopen
import zipfile

from trustswap.common.normalise import normalise_day_count, normalise_floating_rate_option
from trustswap.common.schema import InterestRateSwap
from trustswap.synth.markets import MARKETS

URL = "https://pddata.dtcc.com/ppd/api/report/cumulative/cftc/CFTC_CUMULATIVE_RATES_{day}.zip"
MAX_BYTES = 64 * 1024 * 1024
DAY_COUNTS = {"A004": "ACT/360"}
PERIODS = {"DAIL": "D", "WEEK": "W", "MNTH": "M", "YEAR": "Y"}


class NoFile(RuntimeError):
    """The requested daily publication is unavailable."""


def previous_business_day(day: date) -> date:
    candidate = day - timedelta(days=1)
    calendar = ql.UnitedStates(ql.UnitedStates.FederalReserve)
    while not calendar.isBusinessDay(ql.Date(candidate.day, candidate.month, candidate.year)):
        candidate -= timedelta(days=1)
    return candidate


def download(day: date) -> bytes:
    try:
        with urlopen(URL.format(day=day.strftime("%Y_%m_%d")), timeout=60) as response:
            data = response.read(MAX_BYTES + 1)
    except HTTPError as exc:
        if exc.code in (403, 404):
            raise NoFile(f"DTCC file unavailable for {day}") from None
        raise
    if len(data) > MAX_BYTES:
        raise ValueError("DTCC archive exceeds size limit")
    return data


def _term(value):
    return {"value": value, "status": "explicit"}


def _decimal(value):
    result = Decimal(value.replace(",", ""))
    if not result.is_finite():
        raise ValueError("Non-finite reported value")
    return result


def _frequency(row, prefix, leg):
    period = row[f"{prefix} period-leg {leg}"]
    count = int(row[f"{prefix} period multiplier-leg {leg}"])
    if count < 1 or count > 120:
        raise ValueError("Unsupported frequency multiplier")
    return f"{count}{PERIODS[period]}"


def parse_trade(row):
    """Require explicit dates, principal and rate terms. Fill fictional metadata only."""
    product = row["upi fisn"]
    match = re.fullmatch(r"NA/Swap (OIS|Fxd Flt|Basis|Flt Flt) ([A-Z]{3})", product)
    if not match or row["asset class"] != "IR" or row["action type"] != "NEWT":
        return None
    if any(row.get(name, "").upper() != "FALSE" for name in (
        "non-standardized term indicator", "package indicator",
    )) or any(row.get(name) for name in ("embedded option type", "option type")):
        return None
    currency = match[2]
    market = MARKETS[currency]
    trade_date = date.fromisoformat(row["execution timestamp"][:10])
    effective = date.fromisoformat(row["effective date"])
    maturity = date.fromisoformat(row["expiration date"])
    if not trade_date <= effective < maturity:
        return None
    if any(row[f"notional currency-leg {leg}"] != currency for leg in (1, 2)):
        return None
    amounts = [_decimal(row[f"notional amount-leg {leg}"]) for leg in (1, 2)]
    if not amounts[0] == amounts[1] > 0:
        return None
    fixed = [leg for leg in (1, 2) if row[f"fixed rate-leg {leg}"]]
    basis = match[1] in {"Basis", "Flt Flt"}
    if len(fixed) != (0 if basis else 1):
        return None
    values = {
        "trade_date": _term(trade_date), "effective_date": _term(effective), "termination_date": _term(maturity),
        "notice_address": _term("Synthetic Party A, 1 Example Street, Example City"),
    }

    for leg in (1, 2):
        is_fixed = leg in fixed
        kind = "fixed" if is_fixed else "floating"
        day_count = row[f"{kind} rate day count convention-leg {leg}"]
        day_count = normalise_day_count(DAY_COUNTS.get(day_count, day_count))
        a, b = ("Synthetic Party A", "Synthetic Party B") if leg == 1 else ("Synthetic Party B", "Synthetic Party A")
        terms = {
            "payer": a, "receiver": b, "notional_amount": amounts[leg - 1], "notional_currency": currency,
            "payment_frequency": _frequency(row, f"{kind} rate payment frequency", leg),
            "day_count_fraction": day_count, "business_day_convention": "MODFOLLOWING",
            "business_centers": list(market.centers),
        }
        if is_fixed:
            terms["fixed_rate"] = _decimal(row[f"fixed rate-leg {leg}"])
            name = "fixed_leg"
        else:
            option = row.get(f"underlier id-leg {leg}") or (row["upi underlier name"] if not basis else "")
            terms["floating_rate_option"] = normalise_floating_rate_option(option)
            terms["reset_frequency"] = _frequency(row, "floating rate reset frequency", leg)
            spread = _decimal(row.get(f"spread-leg {leg}") or "0")
            notation = row.get(f"spread notation-leg {leg}", "")
            if spread and notation not in {"1", "3"}:
                return None
            terms["spread"] = spread if notation != "3" else spread / Decimal(10000)
            name = "floating_leg_2" if basis and leg == 2 else "floating_leg"
        values[name] = {key: _term(value) for key, value in terms.items()}
    return InterestRateSwap.model_validate(values)


def select_trades(data: bytes, day: date, count: int = 5) -> list[InterestRateSwap]:
    """Sample deterministically from unique complete trades regardless of archive row order."""
    trades = {}
    with zipfile.ZipFile(BytesIO(data)) as archive:
        files = [entry for entry in archive.infolist() if entry.filename.lower().endswith(".csv")]
        if len(files) != 1 or files[0].file_size > 4 * MAX_BYTES:
            raise ValueError("Unexpected DTCC archive structure")
        with archive.open(files[0]) as source:
            for raw in csv.DictReader(TextIOWrapper(source, encoding="utf-8-sig", newline="")):
                row = {key.strip().casefold(): (value or "").strip() for key, value in raw.items() if key}
                try:
                    trade = parse_trade(row)
                except (KeyError, ValueError, InvalidOperation):
                    continue
                if trade:
                    trades.setdefault(trade.model_dump_json(), trade)
    if len(trades) < count:
        raise NoFile(f"Only {len(trades)} eligible vanilla rates trades for {day}")
    return random.Random(day.isoformat()).sample([trades[key] for key in sorted(trades)], count)
