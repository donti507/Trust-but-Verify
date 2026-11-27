"""Desk-only exception cases and SLA calculations."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from functools import lru_cache
import json
from pathlib import Path

import QuantLib as ql
import yaml

from trustswap.decide.policy import escalate
from trustswap.desk import matcher, store
from trustswap.desk.products import TERMS
from trustswap.score.features import SEVERITY
from trustswap.synth.markets import calendar_for

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "desk.yaml"
SEVERITY_WEIGHT = {"critical": 3, "major": 2, "minor": 1}
ACTIVE_FIELDS = {"open", "pending", "rejected"}
STATUS = ("New", "Assigned", "Waiting for counterparty", "Pending approval", "Resolved")
AGE_BUCKETS = ("On track", "Due soon", "Overdue")


@lru_cache
def sla():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["sla"]


def utc_now():
    return datetime.now(timezone.utc)


def due_at(created, severity, config=None):
    config = sla() if config is None else config
    if severity == "critical":
        return created + timedelta(hours=config["critical_hours"])
    if severity == "minor":
        return created + timedelta(days=config["minor_days"])
    if severity != "major":
        raise ValueError("Unknown SLA severity")
    calendar = calendar_for(config["business_centers"])
    date = ql.Date(created.day, created.month, created.year)
    due = calendar.advance(date, config["major_business_days"], ql.Days).to_date()
    return datetime.combine(due, created.timetz())


def _terms(db, trade_id):
    return list(db.execute('''SELECT fields.*,
                           json_extract(desk_products.checks, '$."' || fields.path || '"') AS product_check
                           FROM fields LEFT JOIN desk_products USING(trade_id) WHERE trade_id=?''', (trade_id,)))


def _disputed(row, threshold):
    if row["state"] not in ACTIVE_FIELDS:
        return False
    return bool(row["product_check"]) or bool(escalate(row["p"], row["c"], row["r"], threshold)) or matcher.match(
        row["path"], json.loads(row["confirmation"]), json.loads(row["booking"])
    ) != matcher.MATCH


def _severity(field):
    return TERMS[field].severity if field in TERMS else SEVERITY[field.rsplit(".", 1)[-1]]


def reconcile(db, trade_id=None, actor=None, *, now=None):
    """Synchronize cases inside the same transaction as the desk action."""
    now = utc_now() if now is None else now
    threshold = db.execute("SELECT value FROM settings WHERE name='lambda'").fetchone()[0]
    query = "SELECT id,assigned_to FROM trades"
    trades = db.execute(query if trade_id is None else query + " WHERE id=?", () if trade_id is None else (trade_id,))
    for trade in trades:
        key, owner = trade["id"], trade["assigned_to"]
        fields = _terms(db, key)
        disputed = [row for row in fields if _disputed(row, threshold)]
        current = db.execute("SELECT * FROM desk_cases WHERE trade_id=?", (key,)).fetchone()
        if not disputed and current is None:
            continue
        pending = any(row["state"] == "pending" for row in fields)
        if not disputed and not pending:
            if current["status"] != "Resolved":
                db.execute("UPDATE desk_cases SET status='Resolved',resolved_at=? WHERE trade_id=?",
                           (now.isoformat(), key))
                store.event(db, actor, "case_resolved", key, previous=current["status"])
            continue
        if disputed:
            severity = max((_severity(row["path"]) for row in disputed), key=SEVERITY_WEIGHT.get)
            priority = max(SEVERITY_WEIGHT[_severity(row["path"])] *
                           (1.0 if row["product_check"] else row["p"]) for row in disputed)
        else:
            severity, priority = current["severity"], current["priority"]
        new = current is None or current["status"] == "Resolved"
        created = now if new else datetime.fromisoformat(current["created_at"])
        deadline = due_at(created, severity)
        if pending:
            status = "Pending approval"
        elif current is not None and current["status"] == "Waiting for counterparty":
            status = current["status"]
        else:
            status = "Assigned" if owner else "New"
        db.execute("""INSERT INTO desk_cases VALUES (?,?,?,?,?,?,?,NULL)
                   ON CONFLICT(trade_id) DO UPDATE SET status=excluded.status,owner=excluded.owner,
                   priority=excluded.priority,severity=excluded.severity,created_at=excluded.created_at,
                   due_at=excluded.due_at,resolved_at=NULL""",
                   (key, status, owner, priority, severity, created.isoformat(), deadline.isoformat()))
        if new:
            store.event(db, actor, "case_opened", key, severity=severity, status=status,
                        due_at=deadline.isoformat())
        elif current["status"] != status:
            store.event(db, actor, "case_status_changed", key, previous=current["status"], status=status)


def analyst_options(db):
    return list(db.execute("SELECT id,name FROM users WHERE role='analyst' ORDER BY name"))


def assign(db, trade_id, owner, actor):
    case = db.execute("SELECT * FROM desk_cases WHERE trade_id=?", (trade_id,)).fetchone()
    if case is None or case["status"] == "Resolved":
        raise ValueError("Only an open exception case can be assigned")
    if not 0 < owner < 2**63 or not db.execute(
        "SELECT 1 FROM users WHERE id=? AND role='analyst'", (owner,)
    ).fetchone():
        raise ValueError("Select an analyst to assign this case")
    if case["owner"] == owner:
        return
    status = "Assigned" if case["status"] == "New" else case["status"]
    db.execute("UPDATE desk_cases SET owner=?,status=? WHERE trade_id=?", (owner, status, trade_id))
    db.execute("UPDATE trades SET assigned_to=? WHERE id=?", (owner, trade_id))
    store.event(db, actor, "case_assigned", trade_id, previous_owner=case["owner"], owner=owner, status=status)


def _assigned_case(db, trade_id, actor, role, allowed):
    case = db.execute("SELECT * FROM desk_cases WHERE trade_id=?", (trade_id,)).fetchone()
    if (case is None or case["status"] not in allowed or case["owner"] is None
            or (role == "analyst" and case["owner"] != actor)):
        raise ValueError("This action needs a case assigned to an analyst in the appropriate state")
    return case


def chase(db, trade_id, actor, role):
    case = _assigned_case(db, trade_id, actor, role, {"Assigned", "Waiting for counterparty"})
    threshold = db.execute("SELECT value FROM settings WHERE name='lambda'").fetchone()[0]
    from trustswap.desk.imports import field_label
    from trustswap.desk.presentation import value_text

    details = []
    for row in _terms(db, trade_id):
        if _disputed(row, threshold):
            confirmed, booked = (value_text(json.loads(row[key])) for key in ("confirmation", "booking"))
            details.append(f"{field_label(row['path'])}. You stated {confirmed}. We booked {booked}.")
    if not details:
        raise ValueError("There are no outstanding fields to chase")
    message = EmailMessage()
    message["From"] = "desk@example.invalid"
    message["To"] = "counterparty@example.invalid"
    message["Subject"] = f"Fictional confirmation follow-up for {trade_id}"
    lines = ["Hello Synthetic Counterparty,", "",
             f"Could you confirm the following terms for fictional trade {trade_id}?", "",
             *(f"- {text}" for text in details), "", "Thank you,", "Synthetic Desk"]
    message.set_content("\n".join(lines))
    db.execute("UPDATE desk_cases SET status='Waiting for counterparty' WHERE trade_id=?", (trade_id,))
    store.event(db, actor, "case_chase_drafted", trade_id, previous=case["status"],
                status="Waiting for counterparty", fields=len(details))
    return message


def acknowledge(db, trade_id, actor, role):
    _assigned_case(db, trade_id, actor, role, {"Waiting for counterparty"})
    db.execute("UPDATE desk_cases SET status='Assigned' WHERE trade_id=?", (trade_id,))
    store.event(db, actor, "case_reply_recorded", trade_id, status="Assigned")


def queue(db, *, analyst=None, now=None):
    now = utc_now() if now is None else now
    query = """SELECT desk_cases.*,users.name AS owner_name FROM desk_cases
               LEFT JOIN users ON owner=users.id"""
    params = ()
    if analyst is not None:
        query += " WHERE owner IS NULL OR owner=?"
        params = (analyst,)
    query += """ ORDER BY priority DESC, CASE severity WHEN 'critical' THEN 1 WHEN 'major' THEN 2 ELSE 3 END,
                due_at, trade_id"""
    result = []
    for row in db.execute(query, params):
        case = dict(row)
        created, deadline = (datetime.fromisoformat(row[name]) for name in ("created_at", "due_at"))
        case["age_hours"] = max(0, (now - created).total_seconds() / 3600)
        case["bucket"] = None
        if row["status"] != "Resolved":
            case["bucket"] = "Overdue" if now >= deadline else (
                "Due soon" if now >= created + (deadline - created) / 2 else "On track"
            )
        result.append(case)
    return result


def metrics(cases):
    buckets = Counter(case["bucket"] for case in cases if case["bucket"])
    hours = defaultdict(list)
    counts = Counter()
    for case in cases:
        owner = case["owner_name"] or "Unassigned"
        counts[owner] += bool(case["bucket"])
        if case["resolved_at"]:
            duration = datetime.fromisoformat(case["resolved_at"]) - datetime.fromisoformat(case["created_at"])
            hours[owner].append(duration.total_seconds() / 3600)
    durations = [hour for observations in hours.values() for hour in observations]
    return {"buckets": {key: buckets[key] for key in AGE_BUCKETS}, "overdue": buckets["Overdue"],
            "mean_hours": sum(durations) / len(durations) if durations else None,
            "owners": [(owner, count, sum(hours[owner]) / len(hours[owner]) if hours[owner] else None)
                       for owner, count in sorted(counts.items())]}
