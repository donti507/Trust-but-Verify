"""Deterministic swap confirmation formats for the local desk."""
from dataclasses import dataclass
from decimal import Decimal

from trustswap.common.schema import InterestRateSwap
from trustswap.desk import matcher
from trustswap.extract.parse import locate, trade_from_fields
from trustswap.synth.render import trade_fields


@dataclass(frozen=True)
class StructuredTrade:
    reference: str | None
    trade: InterestRateSwap
    fields: dict


def read_terms(reference, terms, source, raw=None):
    """Convert transport terms to the existing typed swap and retain source evidence."""
    fields = {}
    raw = raw or {}
    for path, value in terms.items():
        if matcher.missing(value):
            continue
        snippet = raw.get(path, value)
        span = locate(snippet, None, source) if isinstance(snippet, str) else None
        normalized = matcher.normalize(path, format(value, "f") if isinstance(value, Decimal) else value)
        fields[path] = {"value": normalized, "raw_value": snippet,
                        "span": span, "span_found": span is not None}
    trade = trade_from_fields(fields)
    for path, field in trade_fields(trade).items():
        if path in fields:
            fields[path]["value"] = field.model_dump(mode="json")["value"]
    return StructuredTrade(reference, trade, fields)


def flatten(trade):
    return {path: field.model_dump(mode="json")["value"] for path, field in trade_fields(trade).items()
            if field.value is not None}
