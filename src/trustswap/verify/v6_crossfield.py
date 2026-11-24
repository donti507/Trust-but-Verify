"""V6 cross-field: extracted fields must agree with each other and with every
notional the document states.

Residual: 1.0 for a categorical mismatch; relative difference for notionals.

* Floating rate option currency must equal the leg currency (HARD).
* A legacy IBOR (LIBOR, EURIBOR) next to fallback wording that names another
  index, or a LIBOR on a trade dated 2022 or later, is SOFT: a fallback may govern.
* Designated maturity vs reset tenor is not checked: the schema has no
  designated-maturity field (synthetic basis legs reset and pay on different tenors).
* Leg notionals must be equal and in one currency (HARD; SOFT if the text says
  the notional amortises or accretes; currency mismatch SOFT).
* Each leg's payer must be the other leg's receiver; a leg cannot pay itself (HARD).
* Every amount on a notional line ("notional", "calculation amount") must match
  an extracted notional in the same currency (HARD; SOFT if amortising).
"""
from __future__ import annotations

from datetime import date
from trustswap.common.schema import InterestRateSwap
from trustswap.verify.amounts import currency_amounts, lines_with
from trustswap.verify.base import RuleResult, present_legs, worst

RFR = ("SOFR", "SONIA", "EUROSTR", "ESTR", "€STR")
LEGACY = ("LIBOR", "EURIBOR")
LIBOR_NEW_TRADE_CUTOFF = date(2022, 1, 1)
NOTIONAL_KEYWORDS = ("notional", "calculation amount")
VARYING = ("amortis", "amortiz", "accret")


def _same(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and a.strip().lower() == b.strip().lower()


def _rel(a, b) -> float:
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-9)


def v6_crossfield(trade: InterestRateSwap, text: str) -> dict[str, RuleResult]:
    legs = present_legs(trade)
    out: dict[str, list[RuleResult]] = {}

    def add(path: str, result: RuleResult) -> None:
        out.setdefault(path, []).append(result)

    ok = lambda detail: RuleResult(verdict="PASS", residual=0.0, detail=detail)  # noqa: E731
    varying = any(word in text.lower() for word in VARYING)
    fallback_lines = [line.upper() for line in lines_with(text, ("fallback",))]

    # Index against currency, legacy index.
    for name in legs:
        leg = getattr(trade, name)
        if not hasattr(leg, "floating_rate_option"):
            continue
        option, currency = leg.floating_rate_option.value, leg.notional_currency.value
        path = f"{name}.floating_rate_option"
        if option is None or currency is None:
            add(path, RuleResult(verdict="UNDET", detail="index or currency missing"))
        elif option.split("-")[0].upper() != currency.upper():
            add(path, RuleResult(verdict="HARD", residual=1.0, detail=f"index {option} does not match currency {currency}"))
        else:
            add(path, ok("index matches currency"))
        if option and any(tag in option.upper() for tag in LEGACY):
            other = [line for line in fallback_lines if any(r in line for r in RFR)]
            if other:
                add(path, RuleResult(verdict="SOFT", residual=1.0, detail=f"fallback wording may replace legacy index {option}"))
            trade_day = trade.trade_date.value
            if "LIBOR" in option.upper() and trade_day is not None and trade_day >= LIBOR_NEW_TRADE_CUTOFF:
                add(path, RuleResult(verdict="SOFT", residual=1.0, detail=f"LIBOR index on a trade dated {trade_day}"))

    # Leg notionals and currencies.
    if len(legs) == 2:
        a, b = (getattr(trade, name) for name in legs)
        ca, cb = a.notional_currency.value, b.notional_currency.value
        if ca and cb and ca != cb:
            for name in legs:
                add(f"{name}.notional_currency", RuleResult(verdict="SOFT", residual=1.0, detail="legs in different currencies"))
        elif ca and cb:
            for name in legs:
                add(f"{name}.notional_currency", ok("leg currencies agree"))
        na, nb = a.notional_amount.value, b.notional_amount.value
        if na is not None and nb is not None:
            if na == nb:
                result = ok("leg notionals agree")
            else:
                result = RuleResult(verdict="SOFT" if varying else "HARD", residual=_rel(na, nb),
                                    detail="leg notionals differ" + (" (amortising wording present)" if varying else ""))
            for name in legs:
                add(f"{name}.notional_amount", result)

        # Payer/receiver mirror.
        for name, leg, other in ((legs[0], a, b), (legs[1], b, a)):
            if _same(leg.payer.value, leg.receiver.value):
                add(f"{name}.payer", RuleResult(verdict="HARD", residual=1.0, detail="leg payer equals its receiver"))
            if leg.payer.value is None or other.receiver.value is None:
                add(f"{name}.payer", RuleResult(verdict="UNDET", detail="party missing"))
            elif _same(leg.payer.value, other.receiver.value):
                add(f"{name}.payer", ok("payer is the other leg's receiver"))
            else:
                add(f"{name}.payer", RuleResult(verdict="HARD", residual=1.0,
                                                detail="payer is not the other leg's receiver"))

    # Every stated notional must match an extracted notional in that currency.
    stated = [m for line in lines_with(text, NOTIONAL_KEYWORDS) for m in currency_amounts(line)]
    for amount, code in stated:
        matching = [name for name in legs if getattr(trade, name).notional_currency.value == code
                    and getattr(trade, name).notional_amount.value is not None]
        if not matching:
            continue
        values = [getattr(trade, name).notional_amount.value for name in matching]
        if amount in values:
            result = ok("stated notional matches")
        else:
            residual = min(_rel(amount, v) for v in values)
            result = RuleResult(verdict="SOFT" if varying else "HARD", residual=residual,
                                detail=f"text states notional {code} {amount:,}, extracted {values[0]:,}")
        for name in matching:
            add(f"{name}.notional_amount", result)

    return {path: worst(*results) for path, results in out.items()}
