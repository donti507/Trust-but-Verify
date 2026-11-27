"""Four-eyes lifecycle changes and immutable snapshots of booked swap terms."""
from datetime import date
from decimal import Decimal
import json

from trustswap.decide.policy import escalate
from trustswap.desk import imports, matcher, store

EVENTS = {
    "amendment": "Amendment", "partial": "Partial termination",
    "full": "Full termination", "novation": "Novation",
}


def current(db, trade_id):
    return db.execute("""SELECT * FROM desk_trade_versions WHERE trade_id=?
                         ORDER BY version DESC LIMIT 1""", (trade_id,)).fetchone()


def terminated(db, trade_id):
    cap = current(db, trade_id)
    return cap is not None and cap["event_type"] == "full"


def booked_terms(db, trade_id):
    return {row["path"]: json.loads(row["booking"]) for row in
            db.execute("SELECT path,booking FROM fields WHERE trade_id=?", (trade_id,))}


def _date(value):
    try:
        cleaned = matcher.normalize("effective_date", value)
        if cleaned is None:
            raise ValueError("Missing date")
        return date.fromisoformat(cleaned)
    except (ValueError, TypeError):
        raise ValueError("Enter a valid effective date") from None


def _valid_effective(terms, effective):
    start, end = (_date(terms.get(key)) for key in ("effective_date", "termination_date"))
    if not start <= effective <= end:
        raise ValueError("The effective date must be within the booked trade's life")


def _normalized(path, value):
    imports._fictional(path, value)
    try:
        value = matcher.normalize(path, value)
    except (ValueError, ArithmeticError):
        raise ValueError("Enter a valid value for the selected field") from None
    if value is None:
        raise ValueError("Changes must contain a value")
    return format(value, "f") if isinstance(value, Decimal) else value


def _changes(event_type, effective, terms, form, upload):
    if event_type == "amendment":
        if upload:
            requested = {key: row["confirmation"] for key, row in upload.fields.items()
                         if row["confirmation"] is not None}
        else:
            requested = {form.get("field"): form.get("value")}
        if not requested or set(requested) - imports.FIELD_PATHS or set(requested) - terms.keys():
            raise ValueError("Select a field in the current booking")
        updated = {path: _normalized(path, value) for path, value in requested.items()}
    elif event_type == "full":
        updated = {"termination_date": effective.isoformat()}
    elif event_type == "partial":
        notionals = {path: _normalized(path, value) for path, value in terms.items()
                     if path.endswith(".notional_amount") and value is not None}
        currencies = {value for path, value in terms.items() if path.endswith(".notional_currency") and value}
        if len(notionals) != 2 or len(currencies) != 1 or len(set(map(Decimal, notionals.values()))) != 1:
            raise ValueError("This partial termination form requires two equal notionals in one currency")
        remaining = _normalized("notional_amount", form.get("value"))
        if not 0 < Decimal(remaining) < min(map(Decimal, notionals.values())):
            raise ValueError("The remaining notional must be positive and below the current notional")
        updated = dict.fromkeys(notionals, remaining)
    else:
        outgoing, incoming = (form.get(key, "").strip() for key in ("outgoing", "incoming"))
        current_parties = {value for key, value in terms.items() if key.endswith((".payer", ".receiver")) and value}
        if outgoing not in current_parties or incoming not in (imports.PARTIES - current_parties):
            raise ValueError("Choose a current party and a different fictional replacement")
        updated = {key: incoming for key, value in terms.items()
                   if key.endswith((".payer", ".receiver")) and value == outgoing}
        notice = terms.get("notice_address")
        if notice and notice.startswith(outgoing + ","):
            updated["notice_address"] = incoming + notice.removeprefix(outgoing)
    combined = {**terms, **updated}
    if not _date(combined.get("trade_date")) <= _date(combined.get("effective_date")) <= _date(
        combined.get("termination_date")
    ):
        raise ValueError("Trade, effective and termination dates must be ordered")
    diff = {key: value for key, value in updated.items() if matcher.match(key, terms.get(key), value) != matcher.MATCH}
    if not diff and event_type != "full":
        raise ValueError("The proposed terms match the current booking")
    return diff


def _insert_version(db, trade_id, version, event_type, effective, terms, actor):
    db.execute("""INSERT INTO desk_trade_versions(trade_id,version,event_type,effective,terms,actor)
                  VALUES (?,?,?,?,?,?)""", (trade_id, version, event_type, effective, json.dumps(terms), actor))


def _validate_trade(db, trade_id):
    record = db.execute("SELECT state FROM trades WHERE id=?", (trade_id,)).fetchone()
    if record is None or record["state"] != "Affirmed":
        raise ValueError("Lifecycle events require an affirmed trade")
    if terminated(db, trade_id):
        raise ValueError("The trade is terminated and read-only")


def propose(db, trade_id, event_type, form, actor, upload=None):
    _validate_trade(db, trade_id)
    if event_type not in EVENTS:
        raise ValueError("Select a lifecycle event")
    if upload is not None and event_type != "amendment":
        raise ValueError("Attachments are supported for amendments")
    note = form.get("note", "").strip()
    if not note:
        raise ValueError("Add a note to explain the change")
    if db.execute("SELECT 1 FROM desk_lifecycle WHERE trade_id=? AND status='Pending approval'",
                  (trade_id,)).fetchone():
        raise ValueError("This trade already has an event pending approval")
    terms = booked_terms(db, trade_id)
    effective = _date(form.get("effective"))
    _valid_effective(terms, effective)
    updated = _changes(event_type, effective, terms, form, upload)
    version = current(db, trade_id)
    if version is None:
        _insert_version(db, trade_id, 1, "original", terms["effective_date"], terms, actor=None)
    base_version = version["version"] if version else 1
    threshold = db.execute("SELECT value FROM settings WHERE name='lambda'").fetchone()[0]
    risk_flags = [key for key in updated if upload and
                  escalate(upload.fields[key]["p"], upload.fields[key]["c"], upload.fields[key]["r"], threshold)]
    differences = {key: {"before": terms.get(key), "after": value} for key, value in updated.items()}
    if event_type == "full" and not differences:
        differences["termination_date"] = {"before": terms["termination_date"], "after": terms["termination_date"]}
    inserted = db.execute("""INSERT INTO desk_lifecycle(trade_id,event_type,effective,base_version,differences,
                            risk_flags,upload_sha256,proposer,note) VALUES (?,?,?,?,?,?,?,?,?)""",
                          (trade_id, event_type, effective.isoformat(), base_version, json.dumps(differences),
                           json.dumps(risk_flags), upload.digest if upload else None, actor, note))
    store.event(db, actor, "lifecycle_proposed", trade_id, event_id=inserted.lastrowid, event_type=event_type,
                effective=effective.isoformat(), changed_fields=list(differences),
                upload_sha256=upload.digest if upload else None)


def decide(db, event_id, actor, approve, reason):
    row = db.execute("SELECT * FROM desk_lifecycle WHERE id=?", (event_id,)).fetchone()
    if row is None or row["status"] != "Pending approval":
        raise ValueError("This lifecycle event is no longer pending")
    if row["proposer"] == actor:
        raise PermissionError("Self-approval is forbidden")
    reason = reason.strip()
    if not reason:
        raise ValueError("Enter a reason for the decision")
    trade_id = row["trade_id"]
    _validate_trade(db, trade_id)
    if approve:
        base = current(db, trade_id)
        terms = booked_terms(db, trade_id)
        if base["version"] != row["base_version"] or terms != json.loads(base["terms"]):
            raise ValueError("The booked version changed. Reject this event and propose it again")
        _valid_effective(terms, _date(row["effective"]))
        for path, diff in json.loads(row["differences"]).items():
            terms[path] = diff["after"]
            db.execute("UPDATE fields SET booking=?,state='approved',reason=? WHERE trade_id=? AND path=?",
                       (json.dumps(diff["after"]), reason, trade_id, path))
        _insert_version(db, trade_id, base["version"] + 1, row["event_type"], row["effective"], terms, actor)
    state = "Approved" if approve else "Rejected"
    db.execute("UPDATE desk_lifecycle SET status=?, reviewer=?, reason=? WHERE id=?", (state, actor, reason, event_id))
    store.event(db, actor, "lifecycle_approved" if approve else "lifecycle_rejected", trade_id,
                event_id=event_id, event_type=row["event_type"], reason=reason, proposer=row["proposer"])


def requests(db, trade_id=None):
    query = """SELECT desk_lifecycle.*,users.name AS analyst FROM desk_lifecycle
               JOIN users ON users.id=proposer"""
    tail = " WHERE status='Pending approval'" if trade_id is None else " WHERE trade_id=?"
    params = () if trade_id is None else (trade_id,)
    results = []
    for row in db.execute(query + tail + " ORDER BY desk_lifecycle.id DESC", params):
        results.append({**dict(row), "label": EVENTS[row["event_type"]],
                        "changes": json.loads(row["differences"]), "risk_flags": json.loads(row["risk_flags"])})
    return results


def versions(db, trade_id):
    return [
        {**dict(row), "label": EVENTS.get(row["event_type"], "Original booking"),
         "terms": json.loads(row["terms"])}
        for row in db.execute("""SELECT desk_trade_versions.*,users.name AS approved_by
                                 FROM desk_trade_versions LEFT JOIN users ON users.id=actor
                                 WHERE trade_id=? ORDER BY version DESC""", (trade_id,))
    ]
