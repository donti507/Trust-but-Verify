"""Shared types for verifier rules (Step 7).

Every rule is a pure function ``rule(trade, text) -> dict[path, RuleResult]``.
Rules read the trade and the document text and never modify either.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from trustswap.common.schema import ExtractedField, InterestRateSwap

Verdict = Literal["PASS", "HARD", "SOFT", "UNDET"]
LEGS = ("fixed_leg", "floating_leg", "floating_leg_2")
# Higher rank wins when two checks report on the same field.
_RANK = {"UNDET": 0, "PASS": 1, "SOFT": 2, "HARD": 3}


class RuleResult(BaseModel):
    """One rule's finding for one field. Residual units are rule-specific."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    verdict: Verdict
    residual: float | None = None
    detail: str


def iter_fields(trade: InterestRateSwap) -> dict[str, ExtractedField]:
    """Field paths in schema order, skipping legs that are not present."""
    result: dict[str, ExtractedField] = {}
    for name in type(trade).model_fields:
        value = getattr(trade, name)
        if isinstance(value, ExtractedField):
            result[name] = value
        elif value is not None:
            for term in type(value).model_fields:
                result[f"{name}.{term}"] = getattr(value, term)
    return result


def present_legs(trade: InterestRateSwap) -> list[str]:
    return [name for name in LEGS if getattr(trade, name) is not None]


def worst(*results: RuleResult) -> RuleResult:
    """Combine checks on one field: the most severe verdict wins, residuals take the max."""
    top = max(results, key=lambda r: _RANK[r.verdict])
    residuals = [r.residual for r in results if r.residual is not None and r.verdict == top.verdict]
    return RuleResult(verdict=top.verdict, residual=max(residuals) if residuals else top.residual,
                      detail=top.detail)


def is_flagged(result: RuleResult) -> bool:
    """A rule 'fires' when it reports HARD or SOFT."""
    return result.verdict in {"HARD", "SOFT"}
