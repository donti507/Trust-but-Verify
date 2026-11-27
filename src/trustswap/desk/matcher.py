"""Compare normalized confirmation and booking fields without changing either."""
from decimal import Decimal
import re

from trustswap.common.annotations import normalise_field
from trustswap.common.normalise import normalise_rate
from trustswap.desk.products import KEYS, TERMS

MATCH = "MATCH"
MISMATCH = "MISMATCH"
MISSING_CONFIRMATION = "MISSING_IN_CONFIRMATION"
MISSING_BOOKING = "MISSING_IN_BOOKING"
RATE_TOLERANCE = Decimal("0.00001")
KEY_FIELDS = (
    "trade_date", "floating_leg.notional_amount", "floating_leg.notional_currency", "termination_date",
)


def clean(value):
    return re.sub(r"^[\s/:|]+", "", value).strip() if isinstance(value, str) else value


def missing(value):
    return value is None or (isinstance(value, str) and not clean(value))


def term_name(field):
    spec = TERMS.get(field)
    return spec.normalizer if spec else field.rsplit(".", 1)[-1]


def normalize(field, value):
    if missing(value):
        return None
    value = clean(value)
    name = term_name(field)
    if name == "flag":
        flag = str(value).casefold()
        if flag in {"true", "yes", "1"}:
            return True
        if flag in {"false", "no", "0"}:
            return False
        raise ValueError("Use yes or no for an exchange flag")
    if name == "decimal":
        number = Decimal(str(value))
        if not number.is_finite() or number <= 0:
            raise ValueError("Expected a positive rate")
        return number
    if name in {"fixed_rate", "spread"}:
        text = str(value).strip()
        inferred = bool(re.search(r"%|per ?cent|bps?|basis points?", text, re.I))
        return normalise_rate(text, unit=None if inferred else "decimal", is_spread=name == "spread")
    if name == "business_centers" and isinstance(value, list):
        value = ", ".join(value)
    return normalise_field(name, str(value))


def match(field, confirmed, booked):
    if missing(confirmed):
        return MATCH if missing(booked) else MISSING_CONFIRMATION
    if missing(booked):
        return MISSING_BOOKING
    try:
        left, right = normalize(field, confirmed), normalize(field, booked)
    except (ValueError, ArithmeticError):
        return MISMATCH
    if term_name(field) in {"fixed_rate", "spread"}:
        return MATCH if abs(left - right) <= RATE_TOLERANCE else MISMATCH
    return MATCH if left == right else MISMATCH


def booking_key(fields):
    products = {path.split(".", 1)[0] for path in fields if path in TERMS}
    if len(products) > 1:
        return None
    product = next(iter(products), None)
    keys = tuple(f"{product}.{name}" for name in KEYS[product]) if product else KEY_FIELDS
    try:
        values = tuple(normalize(field, fields.get(field)) for field in keys)
    except (ValueError, ArithmeticError):
        return None
    if any(value is None for value in values):
        return None
    return (product, *values) if product else values


def link_booking(trade_id, fields, bookings):
    exact = [booking for booking in bookings if booking.get("trade_id") == trade_id]
    if exact:
        return exact[0] if len(exact) == 1 else None
    key = booking_key(fields)
    if key is None:
        return None
    nearby = [booking for booking in bookings if booking_key(booking["fields"]) == key]
    return nearby[0] if len(nearby) == 1 else None
