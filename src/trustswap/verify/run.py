"""Run the verifier rules on one extracted trade.

Rules never edit values: ``verify`` checks that the trade is unchanged afterwards.
"""
from __future__ import annotations

from trustswap.common.schema import InterestRateSwap
from trustswap.verify.base import RuleResult
from trustswap.verify.v1_schema import v1_schema
from trustswap.verify.v2_grounding import v2_grounding
from trustswap.verify.v3_temporal import v3_temporal
from trustswap.verify.v4_calendar import v4_calendar
from trustswap.verify.v5_accrual import v5_accrual
from trustswap.verify.v6_crossfield import v6_crossfield
from trustswap.verify.v7_convention import v7_convention

RULES = {"V1": v1_schema, "V2": v2_grounding, "V3": v3_temporal, "V4": v4_calendar,
         "V5": v5_accrual, "V6": v6_crossfield, "V7": v7_convention}
GENERIC_RULES = ("V1", "V2")
DOMAIN_RULES = ("V3", "V4", "V5", "V6", "V7")
HARD_RULES = ("V1", "V2", "V3", "V4", "V5", "V6")  # V7 is SOFT only


def verify(trade: InterestRateSwap, text: str, rules: tuple[str, ...] = tuple(RULES)) -> dict[str, dict[str, RuleResult]]:
    """Return {rule: {field_path: RuleResult}} for the requested rules."""
    before = trade.model_dump_json()
    results = {name: RULES[name](trade, text) for name in rules}
    if trade.model_dump_json() != before:
        raise RuntimeError("a verifier rule modified the trade")
    return results


def to_record(doc_id: str, results: dict[str, dict[str, RuleResult]]) -> dict:
    """Verifier record in the runbook format: {doc_id, rules: {rule: {field: {verdict, residual}}}}."""
    return {"doc_id": doc_id,
            "rules": {rule: {path: {"verdict": r.verdict, "residual": r.residual, "detail": r.detail}
                             for path, r in fields.items()}
                      for rule, fields in results.items()}}


def fired(results: dict[str, dict[str, RuleResult]], rule: str, fields: list[str] | None = None,
          verdicts: tuple[str, ...] = ("HARD", "SOFT")) -> bool:
    """True when `rule` reports one of `verdicts` on any of `fields` (all fields if None)."""
    found = results.get(rule, {})
    paths = found if fields is None else [p for p in fields if p in found]
    return any(found[p].verdict in verdicts for p in paths)
