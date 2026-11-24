"""V3 temporal: trade_date <= effective_date < termination_date, with a plausible tenor.

Residual: the number of days by which an ordering is violated (0 when satisfied).
The schema has no stated-tenor field, so the tenor check is a plausibility band.
"""
from __future__ import annotations

from trustswap.common.schema import InterestRateSwap
from trustswap.verify.base import RuleResult, worst

MIN_TENOR_DAYS = 7
MAX_TENOR_DAYS = 50 * 366
DATE_FIELDS = ("trade_date", "effective_date", "termination_date")


def v3_temporal(trade: InterestRateSwap, text: str | None = None) -> dict[str, RuleResult]:
    values = {name: getattr(trade, name).value for name in DATE_FIELDS}
    checks: dict[str, list[RuleResult]] = {name: [] for name in DATE_FIELDS}
    trade_d, eff, term = values["trade_date"], values["effective_date"], values["termination_date"]

    if trade_d is not None and eff is not None:
        gap = (trade_d - eff).days
        result = (RuleResult(verdict="HARD", residual=float(gap), detail=f"trade date is {gap} days after the effective date")
                  if gap > 0 else RuleResult(verdict="PASS", residual=0.0, detail="trade date on or before effective date"))
        checks["trade_date"].append(result)
        checks["effective_date"].append(result)
    if eff is not None and term is not None:
        tenor = (term - eff).days
        if tenor <= 0:
            result = RuleResult(verdict="HARD", residual=float(1 - tenor),
                                detail="termination date is not after the effective date")
        elif tenor < MIN_TENOR_DAYS or tenor > MAX_TENOR_DAYS:
            result = RuleResult(verdict="SOFT", residual=float(tenor), detail=f"implausible tenor of {tenor} days")
        else:
            result = RuleResult(verdict="PASS", residual=0.0, detail="effective date before termination date")
        checks["effective_date"].append(result)
        checks["termination_date"].append(result)

    return {name: worst(*found) if found else RuleResult(verdict="UNDET", detail="dates needed for ordering are missing")
            for name, found in checks.items()}
