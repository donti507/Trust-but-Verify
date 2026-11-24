"""V2 grounding: the value appears in, or normalises from, its cited evidence span.

Residuals: relative difference for amounts and rates (absolute difference when the
stated value is zero), days for dates, normalised edit distance for free text,
0/1 for categorical codes. Thresholds below are pre-registration candidates.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

from trustswap.common.normalise import (
    normalise_business_centers, normalise_business_day_convention, normalise_date,
    normalise_day_count, normalise_floating_rate_option, normalise_frequency,
    normalise_notional, normalise_rate,
)
from trustswap.common.schema import InterestRateSwap
from trustswap.verify.base import RuleResult, iter_fields

NUMERIC_SOFT = 0.005   # relative difference treated as rounding
TEXT_SOFT = 0.20       # normalised edit distance treated as a near match
_SYMBOLS = {"$": "USD", "US$": "USD", "U.S.$": "USD", "€": "EUR", "£": "GBP"}


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _relative(found: Decimal, value: Decimal) -> float:
    if value == 0:
        return float(abs(found - value))
    return float(abs(found - value) / abs(value))


def _numeric(found: Decimal, value: Decimal) -> RuleResult:
    residual = _relative(found, value)
    if residual == 0:
        return RuleResult(verdict="PASS", residual=0.0, detail="span normalises to the value")
    if residual <= NUMERIC_SOFT:
        return RuleResult(verdict="SOFT", residual=residual, detail="span differs from value by rounding")
    return RuleResult(verdict="HARD", residual=residual, detail=f"span states {found}, not {value}")


def _categorical(found, value) -> RuleResult:
    if found == value:
        return RuleResult(verdict="PASS", residual=0.0, detail="span normalises to the value")
    return RuleResult(verdict="HARD", residual=1.0, detail=f"span states {found!r}, not {value!r}")


def _text(snippet: str, value: str) -> RuleResult:
    a, b = " ".join(snippet.split()).casefold(), " ".join(str(value).split()).casefold()
    if b and b in a:
        return RuleResult(verdict="PASS", residual=0.0, detail="value appears in the span")
    residual = edit_distance(a, b) / max(len(a), len(b), 1)
    if residual <= TEXT_SOFT:
        return RuleResult(verdict="SOFT", residual=residual, detail="span is a near match")
    return RuleResult(verdict="HARD", residual=residual, detail="value does not appear in the span")


def _check(name: str, snippet: str, value) -> RuleResult:
    if isinstance(value, date):
        found = date.fromisoformat(normalise_date(snippet.strip()))
        days = abs((found - value).days)
        if days == 0:
            return RuleResult(verdict="PASS", residual=0.0, detail="span normalises to the date")
        return RuleResult(verdict="HARD", residual=float(days), detail=f"span states {found}, not {value}")
    if name == "notional_amount":
        found, _ = normalise_notional(snippet, None)
        return _numeric(found, value)
    if name in {"fixed_rate", "spread"}:
        return _numeric(normalise_rate(snippet, is_spread=name == "spread"), value)
    if name == "notional_currency":
        upper = snippet.upper()
        codes = {code for symbol, code in _SYMBOLS.items() if symbol in snippet}
        if str(value).upper() in upper or str(value).upper() in codes:
            return RuleResult(verdict="PASS", residual=0.0, detail="currency appears in the span")
        return RuleResult(verdict="HARD", residual=1.0, detail=f"{value} does not appear in the span")
    if name in {"payment_frequency", "reset_frequency"}:
        return _categorical(normalise_frequency(snippet), value)
    if name == "day_count_fraction":
        return _categorical(normalise_day_count(snippet), value)
    if name == "business_day_convention":
        return _categorical(normalise_business_day_convention(snippet), value)
    if name == "business_centers":
        return _categorical(normalise_business_centers(snippet), sorted(value))
    if name == "floating_rate_option":
        return _categorical(normalise_floating_rate_option(snippet), str(value).upper())
    return _text(snippet, value)


def v2_grounding(trade: InterestRateSwap, text: str) -> dict[str, RuleResult]:
    results: dict[str, RuleResult] = {}
    for path, field in iter_fields(trade).items():
        name = path.rpartition(".")[2]
        if field.value is None:
            results[path] = RuleResult(verdict="UNDET", detail="no value to ground")
            continue
        if field.span is None:
            results[path] = RuleResult(verdict="UNDET", detail="no evidence span cited")
            continue
        start, end = field.span
        if end > len(text):
            results[path] = RuleResult(verdict="HARD", detail="evidence span lies outside the document")
            continue
        snippet = text[start:end]
        try:
            results[path] = _check(name, snippet, field.value)
        except (ValueError, ArithmeticError) as error:
            results[path] = RuleResult(verdict="HARD", detail=f"span does not normalise: {error}")
    return results
