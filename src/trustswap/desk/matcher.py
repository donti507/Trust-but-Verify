"""Compare normalized confirmation and booking fields without changing either."""
from decimal import Decimal

from trustswap.common.annotations import normalise_field
from trustswap.common.normalise import normalise_rate

MATCH = "MATCH"
MISMATCH = "MISMATCH"
MISSING_CONFIRMATION = "MISSING_IN_CONFIRMATION"
MISSING_BOOKING = "MISSING_IN_BOOKING"
RATE_TOLERANCE = Decimal("0.00001")
KEY_FIELDS = (
    "trade_date", "floating_leg.notional_amount", "floating_leg.notional_currency", "termination_date",
)


def missing(value):
    return value is None or (isinstance(value, str) and not value.strip())


def normalize(field, value):
    if missing(value):
        return None
    name = field.rsplit(".", 1)[-1]
    if name in {"fixed_rate", "spread"}:
        text = str(value).strip()
        inferred = "%" in text or "bp" in text.casefold()
        return normalise_rate(text, unit=None if inferred else "decimal", is_spread=name == "spread")
    if name == "business_centers" and isinstance(value, list):
        value = ", ".join(value)
    return normalise_field(field, str(value))


def match(field, confirmed, booked):
    if missing(confirmed):
        return MATCH if missing(booked) else MISSING_CONFIRMATION
    if missing(booked):
        return MISSING_BOOKING
    try:
        left, right = normalize(field, confirmed), normalize(field, booked)
    except (ValueError, ArithmeticError):
        return MISMATCH
    if field.rsplit(".", 1)[-1] in {"fixed_rate", "spread"}:
        return MATCH if abs(left - right) <= RATE_TOLERANCE else MISMATCH
    return MATCH if left == right else MISMATCH


def booking_key(fields):
    try:
        values = tuple(normalize(field, fields.get(field)) for field in KEY_FIELDS)
    except (ValueError, ArithmeticError):
        return None
    return values if all(value is not None for value in values) else None


def link_booking(trade_id, fields, bookings):
    exact = [booking for booking in bookings if booking.get("trade_id") == trade_id]
    if exact:
        return exact[0] if len(exact) == 1 else None
    key = booking_key(fields)
    if key is None:
        return None
    nearby = [booking for booking in bookings if booking_key(booking["fields"]) == key]
    return nearby[0] if len(nearby) == 1 else None
