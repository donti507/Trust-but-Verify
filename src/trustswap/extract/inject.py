"""Extracted-field injection (Step 6 experiment family 2, applied after Step 12).

Copies of k = 0 extraction records get exactly one planted error each; the
evidence span is left pointing at the true text, as when a model misreads a
value. Copies are stored apart from real outputs (cache/extract_injected/) and
always tagged with the injection, so the two experiment families never mix.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from random import Random

KINDS = ("scale", "near_number", "date_shift", "code_swap", "party_swap", "drop_value")
_CODES = {
    "day_count_fraction": ["ACT/360", "ACT/365.FIXED", "30/360", "30E/360", "ACT/ACT.ISDA"],
    "payment_frequency": ["1M", "3M", "6M", "1Y"],
    "reset_frequency": ["1M", "3M", "6M", "1Y"],
    "business_day_convention": ["MODFOLLOWING", "FOLLOWING", "PRECEDING"],
    "notional_currency": ["USD", "EUR", "GBP", "JPY"],
    "floating_rate_option": ["USD-SOFR", "EUR-EURIBOR-Reuters", "GBP-SONIA", "USD-LIBOR-BBA"],
}


def _candidates(fields: dict, kind: str) -> list[str]:
    present = [p for p, f in fields.items() if f.get("value") is not None]
    name = lambda p: p.rsplit(".", 1)[-1]  # noqa: E731
    if kind in {"scale", "near_number"}:
        return [p for p in present if name(p) in {"notional_amount", "fixed_rate"}]
    if kind == "date_shift":
        return [p for p in present if name(p).endswith("_date")]
    if kind == "code_swap":
        return [p for p in present if name(p) in _CODES]
    if kind == "party_swap":
        return [p for p in present if name(p) == "payer"
                and fields.get(p.replace("payer", "receiver"), {}).get("value") not in (None, fields[p]["value"])]
    if kind == "drop_value":
        return [p for p in present if name(p) != "notice_address"]
    raise ValueError(kind)


def inject(record: dict, kind: str, rng: Random) -> dict | None:
    """A copy of the record with one planted error, or None if the kind does not apply."""
    if not record.get("schema_valid"):
        return None
    fields = record["fields"]
    paths = sorted(_candidates(fields, kind))
    if not paths:
        return None
    path = rng.choice(paths)
    copy = deepcopy(record)
    target = copy["fields"][path]
    original = target["value"]
    name = path.rsplit(".", 1)[-1]
    changed = {path: None}
    if kind == "scale":
        target["value"] = original * (1000 if name == "notional_amount" else 100)
    elif kind == "near_number":
        target["value"] = round(original * 1.1, 2) if name == "notional_amount" else round(original + 0.001, 6)
    elif kind == "date_shift":
        day = date.fromisoformat(original)
        target["value"] = (day + timedelta(days=rng.choice((1, -1, 30)))).isoformat()
    elif kind == "code_swap":
        options = [c for c in _CODES[name] if c.upper() != str(original).upper()]
        target["value"] = rng.choice(options)
    elif kind == "party_swap":
        receiver = path.replace("payer", "receiver")
        target["value"], copy["fields"][receiver]["value"] = (copy["fields"][receiver]["value"], original)
        changed[receiver] = None
    elif kind == "drop_value":
        target["value"], target["span"] = None, None
    copy["injection"] = {"kind": kind, "fields": sorted(changed), "path": path,
                         "original": original, "injected": target["value"]}
    return copy


def inject_all(record: dict, copies: int, seed: int) -> list[dict]:
    """Up to `copies` injected variants of one record, each a different kind."""
    rng = Random(seed)
    kinds = list(KINDS)
    rng.shuffle(kinds)
    result = []
    for kind in kinds:
        if len(result) == copies:
            break
        variant = inject(record, kind, rng)
        if variant is not None:
            result.append(variant)
    return result
