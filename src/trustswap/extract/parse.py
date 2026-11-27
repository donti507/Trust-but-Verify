"""Parse a model completion into an extraction record (Step 10).

* Schema validity: the completion is JSON that matches ``output_schema()`` and
  every confidence lies in [0, 1]. Invalid completions are still recorded.
* Span: the verbatim value is located on (or just after) the line the model
  named; offsets are into the original document text. ``span_found`` is False
  when the value text is not in the document.
* mean_logprob: mean log-probability of the tokens that produced the field's
  "value" literal, when the provider returns token log-probs.
* Values are normalised with the Step 1 normalisers; a value that does not
  normalise keeps ``value: null`` and records ``normalise_error``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import date
from importlib.resources import files
import json
import re

import jsonschema

from trustswap.common.annotations import normalise_field
from trustswap.extract.fields import PATHS, STRUCTURE_LEGS, output_schema
from trustswap.extract.prompt import line_starts

_VALIDATOR = jsonschema.Draft202012Validator(output_schema())
_DATES = {"trade_date", "effective_date", "termination_date"}
_RATES = {"fixed_rate", "spread"}
_SIGNED = _RATES | {"notional_amount"}
_EDGE = re.compile(r"^[/:–.,;\s-]+|[/:–.,;\s-]+$")
_SIGNED_PREFIX = re.compile(r"^(?:[/:–,;\s]|-(?!\d|\.\d)|\.(?!\d))+")
_SUFFIX = re.compile(r"[/:–.,;\s-]+$")
_OPTION_SPELLING = {line.strip().upper(): line.strip() for line in
                    files("trustswap.ref").joinpath("floating_rate_options.txt").read_text(encoding="utf-8").splitlines()
                    if line.strip()}


@dataclass
class Parsed:
    schema_valid: bool
    errors: list[str] = field(default_factory=list)
    structure: str | None = None
    fields: dict[str, dict] = field(default_factory=dict)


def locate(value: str, line: int | None, text: str) -> tuple[int, int] | None:
    """Offsets of the verbatim value, searching from the named line, then anywhere."""
    words = value.split()
    if not words:
        return None
    pattern = re.compile(r"\s+".join(re.escape(w) for w in words))
    starts = line_starts(text)
    windows = []
    if line is not None and 1 <= line <= len(starts):
        begin = starts[line - 1]
        end = starts[line + 2] if line + 2 < len(starts) else len(text)  # value may wrap onto 2 more lines
        windows.append((begin, end))
    windows.append((0, len(text)))
    for flags in (0, re.IGNORECASE):
        for begin, end in windows:
            match = re.compile(pattern.pattern, flags).search(text, begin, end)
            if match and match.start() < end:
                return match.start(), match.end()
    return None


def _value_ranges(raw: str) -> dict[str, tuple[int, int]]:
    """Character range of each field's "value" literal inside the raw completion."""
    ranges = {}
    for path in PATHS:
        match = re.search(r'"%s"\s*:\s*\{\s*"value"\s*:\s*(null|"(?:[^"\\]|\\.)*")' % re.escape(path), raw)
        if match:
            ranges[path] = match.span(1)
    return ranges


def _mean_logprobs(raw: str, tokens: list[tuple[str, float]] | None) -> dict[str, float]:
    if not tokens or "".join(t for t, _ in tokens) != raw:
        return {}
    spans, offset = [], 0
    for token, logprob in tokens:
        spans.append((offset, offset + len(token), logprob))
        offset += len(token)
    result = {}
    for path, (start, end) in _value_ranges(raw).items():
        values = [lp for a, b, lp in spans if a < end and b > start]
        if values:
            result[path] = sum(values) / len(values)
    return result


def _jsonable(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, date):
        return value.isoformat()
    return value


def trim_value(path: str, raw_value: str) -> str:
    """Remove boundary punctuation only from extracted dates, amounts and rates."""
    name = path.rsplit(".", 1)[-1]
    if name in _DATES:
        return _EDGE.sub("", raw_value)
    if name not in _SIGNED:
        return raw_value
    # An attached minus or decimal point is part of the number. A dash
    # separated from digits by whitespace is a document delimiter.
    prefixed = _SIGNED_PREFIX.sub("", raw_value)
    trimmed = _SUFFIX.sub("", prefixed)
    if name in _RATES and trimmed.casefold().endswith(" p.a") and prefixed[len(trimmed):].startswith("."):
        return trimmed + "."
    return trimmed


def normalise_value(path: str, raw_value: str):
    value = _jsonable(normalise_field(path, trim_value(path, raw_value)))
    if path.endswith("floating_rate_option"):
        value = _OPTION_SPELLING.get(value, value)  # reference-list spelling
    return value


def parse_completion(raw: str, text: str, tokens: list[tuple[str, float]] | None = None) -> Parsed:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as error:
        return Parsed(schema_valid=False, errors=[f"not JSON: {error}"])
    errors = [e.message for e in _VALIDATOR.iter_errors(data)]
    if errors:
        return Parsed(schema_valid=False, errors=errors[:10])
    bad_conf = [p for p, f in data["fields"].items() if not 0 <= f["confidence"] <= 1]
    if bad_conf:
        return Parsed(schema_valid=False, errors=[f"confidence outside [0, 1]: {bad_conf[:5]}"])

    logprobs = _mean_logprobs(raw, tokens)
    legs = STRUCTURE_LEGS[data["structure"]]
    fields = {}
    for path in PATHS:
        answer = data["fields"][path]
        leg = path.split(".")[0] if "." in path else None
        if leg is not None and leg not in legs:
            continue  # leg not part of the stated structure
        raw_value = answer["value"]
        record = {"value": None, "raw_value": raw_value, "span": None, "span_found": False,
                  "line": answer["line"], "verbal_conf": answer["confidence"],
                  "mean_logprob": logprobs.get(path)}
        if raw_value is not None and raw_value.strip():
            span = locate(raw_value, answer["line"], text)
            record["span"], record["span_found"] = (list(span) if span else None), span is not None
            try:
                record["value"] = normalise_value(path, raw_value)
            except (ValueError, ArithmeticError) as error:
                record["normalise_error"] = str(error)
        fields[path] = record
    return Parsed(schema_valid=True, structure=data["structure"], fields=fields)


def to_trade(parsed: Parsed):
    """InterestRateSwap for the verifier: normalised values, located spans, null -> absent."""
    if not parsed.schema_valid:
        return None
    return trade_from_fields(parsed.fields)


def trade_from_fields(fields: dict[str, dict]):
    """Build a trade from extraction-record fields ({path: {"value", "span", ...}})."""
    from trustswap.common.schema import InterestRateSwap

    record: dict = {}
    for path, item in fields.items():
        value = item.get("value")
        status = "explicit" if value is not None else "absent"
        span = item.get("span") if value is not None else None
        term = {"value": value, "span": span, "status": status}
        if "." in path:
            leg, name = path.split(".", 1)
            record.setdefault(leg, {})[name] = term
        else:
            record[path] = term
    return InterestRateSwap.model_validate(record)
