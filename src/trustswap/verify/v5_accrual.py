"""V5 accrual: recompute stated fixed amounts from the extracted notional, fixed
rate, day count and schedule.

Residual: currency units between the stated amount and the closest recomputed
amount.

* Only lines that state a payment amount ("fixed amount", "interest amount",
  "payment amount", "amount payable", "coupon amount", "accrued") with an explicit
  currency are checked. Floating amounts need a rate fixing and are not checked.
* If the line states a period (two dates), that period is accrued. Otherwise the
  stated amount is compared with every period of the regenerated fixed schedule
  (adjusted and unadjusted), and the closest one counts.
* PASS within max(1 currency unit, 0.01%); SOFT within 1% (period or rounding
  conventions can explain it); HARD beyond 1%. No stated amount -> UNDET.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import QuantLib as ql

from trustswap.common.normalise import normalise_date
from trustswap.common.schema import InterestRateSwap
from trustswap.verify.amounts import currency_amounts, lines_with
from trustswap.verify.base import RuleResult, worst
from trustswap.verify.v4_calendar import _DATE_RE, _q, _schedules, explicit_dates

AMOUNT_KEYWORDS = ("fixed amount", "interest amount", "payment amount", "amount payable",
                   "coupon amount", "accrued")
_EXCLUDE = ("notional", "calculation amount", "floating amount", "floating rate")
_DAY_COUNTERS = {
    "ACT/360": lambda: ql.Actual360(),
    "ACT/365.FIXED": lambda: ql.Actual365Fixed(),
    "ACT/ACT.ISDA": lambda: ql.ActualActual(ql.ActualActual.ISDA),
    "30/360": lambda: ql.Thirty360(ql.Thirty360.BondBasis),
    "30E/360": lambda: ql.Thirty360(ql.Thirty360.European),
    "30E/360.ISDA": lambda: ql.Thirty360(ql.Thirty360.German),
}
ABS_TOLERANCE = 1.0
REL_PASS = 0.0001
REL_SOFT = 0.01
FIELDS = ("fixed_leg.fixed_rate", "fixed_leg.notional_amount", "fixed_leg.day_count_fraction")


def _line_period(line: str) -> tuple[date, date] | None:
    days = []
    for match in _DATE_RE.finditer(line):
        try:
            days.append(date.fromisoformat(normalise_date(match.group(0))))
        except ValueError:
            continue
    return (min(days), max(days)) if len(days) >= 2 and min(days) < max(days) else None


def accrual(notional: Decimal, rate: Decimal, day_count: str, start: date, end: date) -> float:
    counter = _DAY_COUNTERS[day_count]()
    return float(notional) * float(rate) * counter.yearFraction(_q(start), _q(end))


def v5_accrual(trade: InterestRateSwap, text: str) -> dict[str, RuleResult]:
    def every(result: RuleResult) -> dict[str, RuleResult]:
        return {path: result for path in FIELDS}

    leg = trade.fixed_leg
    if leg is None:
        return {}
    stated = [(amount, code, line) for line in lines_with(text, AMOUNT_KEYWORDS, _EXCLUDE)
              for amount, code in currency_amounts(line)]
    if not stated:
        return every(RuleResult(verdict="UNDET", detail="no fixed amount stated in the text"))
    notional, rate = leg.notional_amount.value, leg.fixed_rate.value
    day_count, currency = leg.day_count_fraction.value, leg.notional_currency.value
    eff, term = trade.effective_date.value, trade.termination_date.value
    if None in (notional, rate, eff, term) or day_count not in _DAY_COUNTERS or not eff < term:
        return every(RuleResult(verdict="UNDET", detail="inputs missing or day count not recomputable"))

    periods = None
    results = []
    for amount, code, line in stated:
        if currency and code != currency:
            continue
        period = _line_period(line)
        if period is not None:
            candidates = [accrual(notional, rate, day_count, *period)]
        else:
            if periods is None:
                _, stubs = explicit_dates(text)
                schedules = _schedules(eff, term, leg, stubs) or []
                periods = {(a, b) for s in schedules for a, b in zip(sorted(s), sorted(s)[1:])}
            candidates = [accrual(notional, rate, day_count, a, b) for a, b in periods]
        if not candidates:
            continue
        expected = min(candidates, key=lambda c: abs(c - float(amount)))
        residual = abs(expected - float(amount))
        if residual <= max(ABS_TOLERANCE, REL_PASS * abs(expected)):
            results.append(RuleResult(verdict="PASS", residual=residual, detail="stated amount recomputes"))
        else:
            verdict = "SOFT" if residual <= REL_SOFT * abs(expected) else "HARD"
            results.append(RuleResult(verdict=verdict, residual=residual,
                                      detail=f"stated {code} {amount:,} vs recomputed {expected:,.2f}"))
    if not results:
        return every(RuleResult(verdict="UNDET", detail="no checkable amount in the fixed leg currency"))
    return every(worst(*results))
