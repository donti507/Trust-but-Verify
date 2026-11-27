"""Seed fictional trades and import local booking snapshots."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from io import StringIO
import json
import math
from pathlib import Path
from random import Random
import re

from trustswap.decide.policy import load_costs
from trustswap.desk import exceptions, matcher, store
from trustswap.desk.products import TERMS, cost_term
from trustswap.extract.fields import PATHS

TRADE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
LEGS = {"fixed_leg": "fixed leg", "floating_leg": "floating leg", "floating_leg_2": "second floating leg"}
PARTIES = {f"Synthetic Party {letter}" for letter in "ABCD"}
FIELD_PATHS = set(PATHS)
BOOKING_PATHS = FIELD_PATHS | TERMS.keys()
DIFFERENCES = ("notional", "rate", "date", "convention")


def field_label(path):
    if path in TERMS:
        return TERMS[path].label
    leg, _, field = path.rpartition(".")
    label = field.replace("_", " ").capitalize()
    return f"{label} ({LEGS[leg]})" if leg else label


LABELS = {field_label(path): path for path in PATHS}


def _trade_id(value):
    if not isinstance(value, str) or not TRADE_ID.fullmatch(value):
        raise ValueError("Invalid trade identifier")
    return value


ADDRESSES = tuple(f"{name}, 1 Example Street, Example City" for name in sorted(PARTIES))


def _ocr_key(value):
    # Fold common OCR confusions and spacing so noisy copies map to the fictional original.
    return re.sub(r"\s+", "", value).lower().translate(str.maketrans("l|i0", "111o"))


def _canonical(path, value):
    """Return the fictional booking value for an extracted value, allowing OCR noise."""
    if matcher.missing(value) or not isinstance(value, str):
        return value
    term = path.rsplit(".", 1)[-1]
    options = ADDRESSES if term == "notice_address" else PARTIES if term in {"payer", "receiver"} else ()
    for option in options:
        if _ocr_key(value) == _ocr_key(option):
            return option
    return value


def _fictional(path, value):
    if matcher.missing(value):
        return
    term = matcher.term_name(path)
    if term in {"payer", "receiver"} and (not isinstance(value, str) or value not in PARTIES):
        raise ValueError("Booking parties must use the fictional party names")
    if term == "notice_address" and not (isinstance(value, str) and value in ADDRESSES):
        raise ValueError("Use a fictional example address")


def _replay(path, include_feed=False):
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="demo-data">(.*?)</script>', text, re.S)
    if not match or "Synthetic document, fictional parties" not in text:
        raise ValueError("Replay input must be a generated offline demo")
    payload = json.loads(match[1])
    categories = {"Clean synthetic": "synthetic", "Noisy synthetic": "synthetic", "Adversarial": "adversarial"}
    for doc in payload["documents"]:
        collection = categories.get(doc.get("category"))
        if collection is None or doc.get("collection", collection) != collection:
            raise ValueError("Replay must contain only generated research documents")

        yield doc, collection
    if include_feed:
        for day in (payload.get("feed") or {}).get("days", []):
            yield from _feed_day(day, date.fromisoformat(day["date"]).isoformat())


def _feed_day(payload, day):
    if payload.get("collection") != "live_feed" or payload.get("date") != day:
        raise ValueError("Feed input must be a dated live_feed publication")
    for doc in payload["documents"]:
        if doc.get("collection") != "live_feed":
            raise ValueError("Feed file contains another collection")
        yield doc, "live_feed"


def _feed(directory):
    for path in sorted(Path(directory).glob("????-??-??.json")):
        date.fromisoformat(path.stem)
        payload = json.loads(path.read_text(encoding="utf-8"))
        yield from _feed_day(payload, path.stem)


def _evidence(doc):
    fields = {}
    for field in doc["fields"]:
        path = field.get("field") or LABELS.get(field.get("name"))
        if path not in FIELD_PATHS or path in fields:
            raise ValueError("Unsupported or duplicate confirmation field")
        raw = field.get("value")
        if isinstance(raw, str) and raw.strip().lower() == "none":
            raw = None
        booked = _canonical(path, raw)
        _fictional(path, booked)
        p, c, r = (float(field[key]) for key in ("p", "c_bar", "r"))
        if not (0 <= p <= 1 and all(math.isfinite(x) and x >= 0 for x in (c, r))):
            raise ValueError("Invalid cached field risk or cost")
        fields[path] = {
            "confirmation": raw, "booking": booked, "p": p, "c": c, "r": r,
            "raw": field.get("raw_value", raw), "span": field.get("span"),
        }
    if not fields:
        raise ValueError("A confirmation needs extracted fields")
    return fields


def _possible_differences(fields):
    candidates = {}
    for path, field in sorted(fields.items()):
        term = path.rsplit(".", 1)[-1]
        try:
            value = matcher.normalize(path, field["booking"])
        except (ValueError, ArithmeticError):
            continue
        if value is None:
            continue
        if term == "notional_amount":
            candidates.setdefault("notional", (path, format(value + Decimal("1"), "f")))
        elif term in {"fixed_rate", "spread"}:
            candidates.setdefault("rate", (path, format(value + Decimal("0.0001"), "f")))
        elif term.endswith("_date"):
            candidates.setdefault("date", (path, (date.fromisoformat(value) + timedelta(days=1)).isoformat()))
        elif term == "business_day_convention":
            changed = "FOLLOWING" if value == "MODFOLLOWING" else "MODFOLLOWING"
            candidates.setdefault("convention", (path, changed))
    return candidates


def seed(db, replay, feed_dir=None):
    external_feed = feed_dir is not None and Path(feed_dir).is_dir()
    documents = list(_replay(replay, include_feed=not external_feed))
    if external_feed:
        documents.extend(_feed(feed_dir))
    prepared = {}
    for doc, source in documents:
        key = _trade_id(doc["id"])
        if key in prepared:
            raise ValueError("Duplicate source trade ID")
        prepared[key] = (source, doc["text"], _evidence(doc))
    with store.transaction(db):
        known = {row["id"]: (row["source_collection"], row["source"])
                 for row in db.execute("SELECT id,source_collection,source FROM trades")}
        for trade in sorted(prepared.keys() & known.keys()):
            source, text, fields = prepared[trade]
            if (source, text) == known[trade]:
                matching = {
                    row["path"]: fields[row["path"]]
                    for row in db.execute("SELECT path,confirmation FROM fields WHERE trade_id=?", (trade,))
                    if row["path"] in fields and json.loads(row["confirmation"]) == fields[row["path"]]["confirmation"]
                }
                store.save_evidence(db, trade, matching)
        new = sorted(prepared.keys() - known.keys())
        candidates = {trade: _possible_differences(prepared[trade][2]) for trade in new}
        eligible = [trade for trade in new if candidates[trade]]
        planted = Random("desk-v1").sample(eligible, min(len(eligible), round(len(new) * 0.2)))
        changes = {}
        counts = dict.fromkeys(DIFFERENCES, 0)
        for trade in planted:
            kind = min(candidates[trade], key=lambda item: (counts[item], DIFFERENCES.index(item)))
            counts[kind] += 1
            changes[trade] = (kind, *candidates[trade][kind])
        for trade in new:
            source, text, fields = prepared[trade]
            if trade in changes:
                kind, path, value = changes[trade]
                fields[path]["booking"] = value
            store.new_trade(db, trade, source, text, fields)
            if trade in changes:
                store.event(db, None, "booking_difference_planted", trade, path, kind=kind)
            exceptions.reconcile(db, trade)
    return len(new), len(planted)


@dataclass(frozen=True)
class RejectedRow:
    line: int
    reason: str


@dataclass(frozen=True)
class ImportReport:
    matched: int
    updated: int
    rejected: list[RejectedRow]


def parse_bookings(source):
    reader = csv.DictReader(source)
    columns = reader.fieldnames or []
    if (not columns or len(set(columns)) != len(columns)
            or not set(columns) <= BOOKING_PATHS | {"trade_id"}):
        raise ValueError("CSV headers must be distinct supported field paths and optional trade_id")
    rows, rejected = [], []
    for row in reader:
        line = reader.line_num
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Column count differs from the header")
            trade_id = row.pop("trade_id", "").strip()
            if trade_id:
                _trade_id(trade_id)
            fields = {path: value.strip() or None for path, value in row.items()}
            if not any(value is not None for value in fields.values()):
                raise ValueError("Booking row has no values")
            for path, value in fields.items():
                _fictional(path, value)
                matcher.normalize(path, value)
            if not trade_id and matcher.booking_key(fields) is None:
                raise ValueError("Bookings without a trade ID need a complete matching key")
        except (ValueError, ArithmeticError) as error:
            rejected.append(RejectedRow(line, str(error)))
        else:
            rows.append((line, trade_id or None, fields))
    return rows, rejected


def import_bookings(db, source, actor=None):
    bookings, rejected = parse_bookings(source)
    costs = load_costs()
    matched = updated = 0
    with store.transaction(db):
        trades = {row[0]: [] for row in db.execute("SELECT id FROM trades")}
        for field in db.execute("SELECT * FROM fields ORDER BY trade_id,path"):
            trades[field["trade_id"]].append(field)
        keyed = {}
        for trade_id, lines in trades.items():
            key = matcher.booking_key({row["path"]: json.loads(row["confirmation"]) for row in lines})
            if key is not None:
                keyed.setdefault(key, []).append(trade_id)
        candidates = {}
        for line, booking_id, fields in bookings:
            matches = [booking_id] if booking_id in trades else keyed.get(matcher.booking_key(fields), [])
            if len(matches) != 1:
                rejected.append(RejectedRow(line, "Matching key is ambiguous" if matches else "No matching trade"))
                continue
            candidates.setdefault(matches[0], []).append((line, booking_id, fields))

        for trade_id, choices in candidates.items():
            if len(choices) != 1:
                rejected.extend(RejectedRow(line, "More than one booking row matches this trade")
                                for line, _, _ in choices)
                continue
            line, booking_id, fields = choices[0]
            existing = trades[trade_id]
            product = db.execute("SELECT product FROM desk_products WHERE trade_id=?", (trade_id,)).fetchone()
            allowed = {key for key in TERMS if key.startswith(product["product"] + ".")} if product else FIELD_PATHS
            if not fields.keys() <= allowed:
                rejected.append(RejectedRow(line, "Booking fields do not match the trade product"))
                continue
            previous = {row["path"]: json.loads(row["booking"]) for row in existing}
            paths = sorted(set(previous) | set(fields))
            if all(previous.get(path) == fields.get(path) for path in paths):
                matched += 1
                continue
            if any(row["state"] != "open" for row in existing):
                rejected.append(RejectedRow(line, "A booking cannot be replaced after review has started"))
                continue
            for path in paths:
                value = json.dumps(fields.get(path), ensure_ascii=False, allow_nan=False)
                if path in previous:
                    db.execute("UPDATE fields SET booking=? WHERE trade_id=? AND path=?", (value, trade_id, path))
                else:
                    name = cost_term(path)
                    db.execute("INSERT INTO fields (trade_id,path,confirmation,booking,p,c,r) VALUES (?,?,'null',?,0,?,?)",
                               (trade_id, path, value, costs.c_bar(name), costs.r(name)))
            store.event(db, actor, "booking_imported", trade_id, booking_id=booking_id,
                        match="trade_id" if booking_id == trade_id else "normalized_key",
                        before=previous, after=fields)
            exceptions.reconcile(db, trade_id, actor)
            matched += 1
            updated += 1
    return ImportReport(matched, updated, sorted(rejected, key=lambda row: row.line))


def import_upload(db, payload, actor):
    return import_bookings(db, StringIO(payload.decode("utf-8-sig")), actor)
