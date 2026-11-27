"""Desk exception queues, SLA calendars and local chase drafts."""
from datetime import datetime, timedelta, timezone
from email import policy
from email.parser import BytesParser

import pytest

from test_desk import AMOUNT, accounts, auth, path, post, setup
from trustswap.desk import exceptions, store

FRIDAY = datetime(2026, 7, 3, 15, tzinfo=timezone.utc)
TRADE = "/trade/fictional-01"


def test_sla_dates_across_weekend_and_configured_business_holiday():
    import QuantLib as ql

    config = {"critical_hours": 4, "minor_days": 2, "major_business_days": 1, "business_centers": ["USNY"]}
    thursday = FRIDAY - timedelta(days=1)
    ql_date = ql.Date(thursday.day, thursday.month, thursday.year)
    ql_deadline = ql.UnitedStates(ql.UnitedStates.Settlement).advance(ql_date, 1, ql.Days).to_date()
    assert ql_deadline.isoformat() == "2026-07-06"
    assert exceptions.due_at(thursday, "major", config) == datetime.combine(ql_deadline, thursday.timetz())
    assert exceptions.due_at(FRIDAY, "critical", config) == FRIDAY + timedelta(hours=4)
    assert exceptions.due_at(FRIDAY, "minor", config) == FRIDAY + timedelta(days=2)


def snapshot(db, trade="fictional-01"):
    row = db.execute("SELECT * FROM desk_cases WHERE trade_id=?", (trade,)).fetchone()
    return dict(row)


def test_priority_uses_severity_times_risk_and_age_filter(setup, monkeypatch):
    app, db_path = setup
    now = FRIDAY + timedelta(hours=16)
    monkeypatch.setattr(exceptions, "utc_now", lambda: now)
    db = store.connect(db_path)
    with store.transaction(db):
        store.new_trade(db, "fictional-02", "synthetic", "Fictional date", {
            "trade_date": {"confirmation": "2026-07-03", "booking": "2026-07-06", "p": 0.40, "c": 100, "r": 1},
        })
        store.new_trade(db, "fictional-03", "synthetic", "Fictional mistaken notional", {
            AMOUNT: {"confirmation": "USD 2,000,000", "booking": "1,900,000", "p": 0.35, "c": 100, "r": 1},
        })
        exceptions.reconcile(db, "fictional-02", now=FRIDAY + timedelta(hours=13))
        exceptions.reconcile(db, "fictional-01", now=FRIDAY + timedelta(hours=13))
        exceptions.reconcile(db, "fictional-03", now=FRIDAY + timedelta(hours=12))
    queue = exceptions.queue(db, now=now)
    assert [row["trade_id"] for row in queue] == ["fictional-03", "fictional-02", "fictional-01"]
    assert [row["priority"] for row in queue] == pytest.approx([1.05, 0.8, 0.6])
    assert [row["bucket"] for row in queue] == ["Overdue", "On track", "Due soon"]
    stats = exceptions.metrics(queue)
    assert stats["buckets"] == {"On track": 1, "Due soon": 1, "Overdue": 1}
    analyst = auth(app, "analyst")
    response = analyst.get("/?overdue=1").get_data(as_text=True)
    table = response.split("Exception queue", 1)[1].split("Confirmations", 1)[0]
    assert "fictional-03" in table and "fictional-01" not in table and "fictional-02" not in table
    assert "Show all open cases" in table
    db.close()


def test_roles_and_case_transitions_with_chase_drafts(setup, monkeypatch):
    app, db_path = setup
    clock = [FRIDAY]
    monkeypatch.setattr(exceptions, "utc_now", lambda: clock[0])
    db = store.connect(db_path)
    with store.transaction(db):
        exceptions.reconcile(db)
        password = db.execute("SELECT password FROM users WHERE role='analyst'").fetchone()[0]
        db.execute("INSERT INTO users(name,role,password) VALUES ('backup-analyst','analyst',?)", (password,))
    ids = dict(db.execute("SELECT name,id FROM users"))
    analyst, supervisor, auditor = (auth(app, role) for role in store.ROLES)
    backup = auth(app, "backup-analyst")
    assert snapshot(db)["status"] == "New"
    assign, chase, reply = (f"{TRADE}/case/{action}" for action in ("assign", "chase", "reply"))
    for url in (assign, chase, reply):
        assert supervisor.post(url).status_code == 400
    assert post(analyst, assign, owner=ids["analyst"]).status_code == 403
    assert post(auditor, chase, mode="draft").status_code == 403
    assert post(analyst, chase, mode="draft").status_code == 409
    assert post(supervisor, assign, owner=ids["auditor"]).status_code == 400
    assert post(supervisor, assign, owner=ids["analyst"]).status_code == 302
    assert snapshot(db)["owner"] == ids["analyst"] and snapshot(db)["status"] == "Assigned"
    assert b"fictional-01" not in backup.get("/").data
    assert post(backup, chase, mode="draft").status_code == 403
    assert post(analyst, reply).status_code == 409
    draft = post(analyst, chase, mode="draft")
    assert draft.status_code == 200
    assert b"You stated USD 1,000,000. We booked 900,000." in draft.data
    assert b"counterparty@example.invalid" in draft.data
    downloaded = post(analyst, chase, mode="download")
    assert downloaded.status_code == 200
    assert "counterparty-draft.eml" in downloaded.headers["Content-Disposition"]
    parsed = BytesParser(policy=policy.default).parsebytes(downloaded.data)
    assert parsed["To"] == "counterparty@example.invalid"
    assert "Notional amount (floating leg)" in parsed.get_content()
    assert snapshot(db)["status"] == "Waiting for counterparty"
    assert post(analyst, reply).status_code == 302
    assert snapshot(db)["status"] == "Assigned"
    assert post(analyst, path("review"), action="propose", value="1000000", note="Confirmed").status_code == 302
    assert snapshot(db)["status"] == "Pending approval"
    assert post(analyst, chase, mode="draft").status_code == 409
    assert post(supervisor, path("approval"), action="reject", reason="Check again").status_code == 302
    assert snapshot(db)["status"] == "Assigned"
    assert post(analyst, path("review"), action="propose", value="1000000", note="Rechecked").status_code == 302
    clock[0] += timedelta(hours=2)
    assert post(supervisor, path("approval"), action="approve", reason="Confirmed").status_code == 302
    assert snapshot(db)["status"] == "Resolved"
    assert post(supervisor, assign, owner=ids["backup-analyst"]).status_code == 400
    assert post(supervisor, chase, mode="draft").status_code == 409
    metrics = exceptions.metrics(exceptions.queue(db))
    assert metrics["mean_hours"] == 2
    assert metrics["owners"] == [("analyst", 0, 2)]
    events = [r[0] for r in db.execute("SELECT action FROM audit WHERE action LIKE 'case_%' ORDER BY id")]
    assert events == [
        "case_opened", "case_assigned", "case_chase_drafted", "case_chase_drafted", "case_reply_recorded",
        "case_status_changed", "case_status_changed", "case_status_changed", "case_resolved",
    ]
    db.close()
