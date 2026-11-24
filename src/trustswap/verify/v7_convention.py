"""V7 convention prior: day counts and business centres should match the market
standard for the index and currency. Unusual is not wrong, so V7 is SOFT only.

Residual: 1.0 when the field departs from the standard, 0.0 otherwise.

* Floating leg day count must equal the index standard (e.g. SOFR ACT/360,
  SONIA ACT/365.FIXED).
* Business centres must include the index's home centre and add nothing beyond
  the centres usual for that currency and trade-date era (e.g. USD LIBOR:
  New York + London).
* Fixed leg day count should be 30/360, ACT/360 or ACT/365.FIXED.
* Unknown index or currency -> UNDET.

The table lives in trustswap/ref/conventions.yaml for domain-lead review.
"""
from __future__ import annotations

from importlib.resources import files

import yaml

from trustswap.common.schema import InterestRateSwap
from trustswap.synth.markets import market_for
from trustswap.verify.base import RuleResult, present_legs

CONVENTIONS_FILE = files("trustswap.ref").joinpath("conventions.yaml")


def load_conventions() -> dict:
    return yaml.safe_load(CONVENTIONS_FILE.read_text(encoding="utf-8"))


_TABLE = load_conventions()
INDEX_STANDARD = {name.upper(): (row["day_count"], set(row["centers"])) for name, row in _TABLE["index"].items()}
HOME_CENTER = {code: set(centers) for code, centers in _TABLE["currency"].items()}
FIXED_DAY_COUNTS = tuple(_TABLE["fixed_day_counts"])


def _allowed(currency: str | None, trade) -> set[str] | None:
    if currency not in HOME_CENTER:
        return None
    allowed = set(HOME_CENTER[currency])
    if trade.trade_date.value is not None:
        allowed |= set(market_for(currency, trade.trade_date.value).centers)
    return allowed


def _centers(path: str, centers, required: set[str], allowed: set[str]) -> RuleResult:
    if not centers:
        return RuleResult(verdict="UNDET", detail="no business centres")
    have = set(centers)
    if not required <= have:
        return RuleResult(verdict="SOFT", residual=1.0, detail=f"missing usual centre {'+'.join(sorted(required - have))}")
    if not have <= allowed | required:
        return RuleResult(verdict="SOFT", residual=1.0, detail=f"extra centre {'+'.join(sorted(have - allowed - required))}")
    return RuleResult(verdict="PASS", residual=0.0, detail="usual business centres")


def v7_convention(trade: InterestRateSwap, text: str) -> dict[str, RuleResult]:
    out: dict[str, RuleResult] = {}
    for name in present_legs(trade):
        leg = getattr(trade, name)
        currency = leg.notional_currency.value
        allowed = _allowed(currency, trade)
        dc_path, bc_path = f"{name}.day_count_fraction", f"{name}.business_centers"
        if hasattr(leg, "floating_rate_option"):
            standard = INDEX_STANDARD.get((leg.floating_rate_option.value or "").upper())
            if standard is None or allowed is None:
                out[dc_path] = out[bc_path] = RuleResult(verdict="UNDET", detail="no convention prior for this index")
                continue
            day_count, required = standard
            out[dc_path] = (RuleResult(verdict="UNDET", detail="day count missing")
                            if leg.day_count_fraction.value is None else
                            RuleResult(verdict="PASS", residual=0.0, detail="standard index day count")
                            if leg.day_count_fraction.value == day_count else
                            RuleResult(verdict="SOFT", residual=1.0,
                                       detail=f"{leg.day_count_fraction.value} where {day_count} is standard"))
            out[bc_path] = _centers(bc_path, leg.business_centers.value, required, allowed | required)
        else:
            if allowed is None:
                out[dc_path] = out[bc_path] = RuleResult(verdict="UNDET", detail="no convention prior for this currency")
                continue
            dc = leg.day_count_fraction.value
            out[dc_path] = (RuleResult(verdict="PASS", residual=0.0, detail="usual fixed day count")
                            if dc in FIXED_DAY_COUNTS else
                            RuleResult(verdict="UNDET" if dc is None else "SOFT", residual=None if dc is None else 1.0,
                                       detail=f"unusual fixed day count {dc}"))
            out[bc_path] = _centers(bc_path, leg.business_centers.value, HOME_CENTER[currency], allowed)
    return out
