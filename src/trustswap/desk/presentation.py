"""Readable desk labels, comparison values and source highlights."""
from collections import defaultdict
from decimal import Decimal
import json

from trustswap.decide.policy import escalate
from trustswap.desk import matcher
from trustswap.desk.imports import field_label
from trustswap.desk.products import TERMS
from trustswap.extract.parse import locate

CENTERS = {"EUTA": "TARGET (EUTA)", "USNY": "New York (USNY)", "GBLO": "London (GBLO)", "JPTO": "Tokyo (JPTO)"}
CONVENTIONS = {
    "MODFOLLOWING": "Modified Following", "MODPRECEDING": "Modified Preceding",
    "FOLLOWING": "Following", "PRECEDING": "Preceding", "NONE": "Unadjusted", "NEAREST": "Nearest",
}
GROUPS = ("Trade dates", "Fixed leg", "Floating leg", "Product terms", "Parties")


def value_text(value):
    if matcher.missing(value):
        return "Missing"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return str(value)


def comparison_value(path, value, raw=None):
    original = value if raw is None else raw
    display = value_text(original)
    try:
        canonical = matcher.normalize(path, value)
    except (ValueError, ArithmeticError):
        canonical = None
    if canonical is None:
        return display, None
    term = path.rsplit(".", 1)[-1]
    if term == "business_centers":
        normalized = ", ".join(CENTERS.get(code, code) for code in canonical)
    elif term == "business_day_convention":
        normalized = CONVENTIONS.get(canonical, canonical)
    elif isinstance(canonical, Decimal):
        normalized = format(canonical, "f")
    else:
        normalized = value_text(canonical)
    return display, normalized if normalized != display else None


def comparison(row, threshold):
    field = dict(row)
    path = field["path"]
    confirm, booked = json.loads(field["confirmation"]), json.loads(field["booking"])
    raw = json.loads(field["raw"]) if field.get("raw") is not None else None
    field.update(
        confirmation=confirm, booking=booked, raw_value=raw,
        label=field_label(path), match=matcher.match(path, confirm, booked),
        risk=bool(field.get("product_check")) if path in TERMS else
        bool(escalate(field["p"], field["c"], field["r"], threshold)),
        confirmation_display=comparison_value(path, confirm, raw), booking_display=comparison_value(path, booked),
    )
    field["flagged"] = field["risk"] or field["match"] != matcher.MATCH
    return field


def source_groups(source, fields):
    grouped = defaultdict(list)
    spans = []
    for index, field in enumerate(fields):
        field["anchor"] = f"field-{index}"
        path = field["path"]
        if path in TERMS:
            group = {"trade_date": "Trade dates", "payer": "Parties", "receiver": "Parties"}.get(
                TERMS[path].normalizer, "Product terms")
        elif "." not in path and path.endswith("_date"):
            group = "Trade dates"
        elif path.rsplit(".", 1)[-1] in {"payer", "receiver", "notice_address"}:
            group = "Parties"
        elif path.startswith("fixed_leg."):
            group = "Fixed leg"
        elif path.startswith("floating_leg"):
            group = "Floating leg"
        else:
            group = "Parties"
        grouped[group].append(field)
        span = json.loads(field["span"]) if field.get("span") else None
        raw = field["raw_value"] or field["confirmation"]
        if not (isinstance(span, list) and len(span) == 2 and all(type(i) is int for i in span)
                and 0 <= span[0] < span[1] <= len(source)):
            span = locate(raw, None, source) if isinstance(raw, str) and raw.strip() else None
        if span:
            spans.append((*span, field))
    boundaries = sorted({0, len(source), *(offset for start, end, _ in spans for offset in (start, end))})
    segments = []
    for start, end in zip(boundaries, boundaries[1:]):
        overlapping = [field for left, right, field in spans if left <= start and end <= right]
        chosen = next((field for field in overlapping if field["flagged"]), overlapping[0] if overlapping else None)
        segments.append((source[start:end], chosen))
    return [(name, grouped[name]) for name in GROUPS if grouped[name]], segments
