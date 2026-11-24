"""V1 schema: required fields are present and codes are valid (generic validation)."""
from __future__ import annotations

import re

from trustswap.common.normalise import _CURRENCIES, normalise_floating_rate_option
from trustswap.common.schema import InterestRateSwap
from trustswap.verify.base import RuleResult, iter_fields

DOCUMENT_REQUIRED = {"trade_date", "effective_date", "termination_date"}
LEG_REQUIRED = {"payer", "receiver", "notional_amount", "notional_currency", "payment_frequency",
                "day_count_fraction", "business_day_convention", "business_centers"}
FIXED_REQUIRED = {"fixed_rate"}
FLOATING_REQUIRED = {"floating_rate_option", "reset_frequency"}

DAY_COUNTS = {"ACT/360", "ACT/365.FIXED", "ACT/ACT.ISDA", "ACT/ACT.ICMA", "30/360", "30E/360", "30E/360.ISDA"}
CONVENTIONS = {"FOLLOWING", "MODFOLLOWING", "PRECEDING", "MODPRECEDING", "NONE", "NEAREST"}
CENTERS = {"USNY", "GBLO", "EUTA", "JPTO"}  # local reference list; extend with FpML centres as needed
_FREQUENCY = re.compile(r"[1-9]\d*[DWMY]")


def is_required(path: str) -> bool:
    container, _, name = path.rpartition(".")
    if not container:
        return name in DOCUMENT_REQUIRED
    if name in LEG_REQUIRED:
        return True
    if container == "fixed_leg":
        return name in FIXED_REQUIRED
    return name in FLOATING_REQUIRED


def _code_problem(name: str, value) -> tuple[str, str] | None:
    """Return (verdict, reason) when a present value is not a valid code."""
    if name == "notional_currency" and str(value).upper() not in _CURRENCIES:
        return "HARD", f"{value!r} is not an ISO 4217 currency code"
    if name == "notional_amount" and value <= 0:
        return "HARD", "notional must be positive"
    if name == "day_count_fraction" and value not in DAY_COUNTS:
        return "HARD", f"{value!r} is not a supported day count fraction"
    if name == "business_day_convention" and value not in CONVENTIONS:
        return "HARD", f"{value!r} is not a business day convention"
    if name in {"payment_frequency", "reset_frequency"} and not _FREQUENCY.fullmatch(str(value)):
        return "HARD", f"{value!r} is not a period code such as 3M or 1Y"
    if name == "business_centers":
        if not value:
            return "HARD", "business centers list is empty"
        malformed = [c for c in value if not re.fullmatch(r"[A-Z]{4}", c)]
        if malformed:
            return "HARD", f"malformed business center codes {malformed}"
        unknown = [c for c in value if c not in CENTERS]
        if unknown:
            return "SOFT", f"business centers {unknown} are not in the local reference list"
    if name == "floating_rate_option":
        try:
            normalise_floating_rate_option(str(value))
        except ValueError:
            return "HARD", f"{value!r} is not in the floating rate option reference list"
    if name in {"payer", "receiver", "notice_address"} and not str(value).strip():
        return "HARD", f"{name} is blank"
    return None


def v1_schema(trade: InterestRateSwap, text: str | None = None) -> dict[str, RuleResult]:
    results: dict[str, RuleResult] = {}
    for path, field in iter_fields(trade).items():
        name = path.rpartition(".")[2]
        required = is_required(path)
        if field.value is None:
            if field.status == "by_reference":
                results[path] = RuleResult(verdict="PASS", detail="incorporated by reference")
            elif field.status == "ambiguous":
                results[path] = RuleResult(verdict="SOFT", detail="field marked ambiguous")
            elif required:
                results[path] = RuleResult(verdict="HARD", detail="required field is missing")
            else:
                results[path] = RuleResult(verdict="PASS", detail="optional field absent")
            continue
        problem = _code_problem(name, field.value)
        if problem:
            results[path] = RuleResult(verdict=problem[0], detail=problem[1])
        else:
            results[path] = RuleResult(verdict="PASS", detail="present and well-formed")
    return results
