"""Local HTTP and SQLite acceptance tests. No server, network or model is needed."""
import json
import sqlite3

import pytest

pytest.importorskip("flask")
pytest.importorskip("bcrypt")

from trustswap.desk import matcher, store
from trustswap.desk.app import create_app

RATE = "fixed_leg.fixed_rate"
AMOUNT = "floating_leg.notional_amount"
PASSWORD = "fictional-password"


@pytest.fixture(scope="module")
def accounts():
    import bcrypt
    return bcrypt.hashpw(PASSWORD.encode(), bcrypt.gensalt(rounds=4))


@pytest.fixture
def setup(tmp_path, accounts):
    path = tmp_path / "desk.db"
    app = create_app(path, secret="z" * 32)
    db = store.connect(path)
    with db:
        for name in store.ROLES:
            db.execute("INSERT INTO users(name,role,password) VALUES (?,?,?)", (name, name, accounts))
        store.new_trade(db, "fictional-01", "synthetic", "Synthetic Party A · fictional terms", {
            AMOUNT: {"confirmation": "USD 1,000,000", "booking": "900,000", "p": 0.2, "c": 100, "r": 1},
            RATE: {"confirmation": "4.00%", "booking": "0.04", "p": 0.001, "c": 100, "r": 1},
        })
    db.close()
    return app, path


def csrf(client):
    with client.session_transaction() as session:
        return session["csrf"]


def auth(app, role):
    client = app.test_client()
    assert client.get("/login").status_code == 200
    login = client.post("/login", data={"csrf": csrf(client), "username": role, "password": PASSWORD})
    assert login.status_code == 302
    assert "HttpOnly" in login.headers["Set-Cookie"] and "SameSite=Strict" in login.headers["Set-Cookie"]
    client.get("/")
    return client


def path(action, field=AMOUNT):
    return f"/trade/fictional-01/field/{field}/{action}"


def post(client, route, **fields):
    return client.post(route, data={"csrf": csrf(client), **fields})


@pytest.mark.parametrize("role", store.ROLES)
def test_permissions_matrix_and_csrf(setup, role):
    app, _ = setup
    client = auth(app, role)
    assert client.get("/").status_code == client.get("/trade/fictional-01").status_code == 200
    assert client.get("/approvals").status_code == (200 if role == "supervisor" else 403)
    assert client.get("/audit").status_code == (200 if role in {"auditor", "supervisor"} else 403)
    assert post(client, "/settings/lambda", **{"lambda": "2"}).status_code == (302 if role == "supervisor" else 403)
    assert post(client, path("review"), action="accept").status_code == (302 if role == "analyst" else 403)
    assert post(client, path("approval"), action="approve", reason="Checked").status_code == (
        409 if role == "supervisor" else 403
    )
    for route in ("/login", "/logout", "/settings/lambda", path("review"), path("approval")):
        assert client.post(route, data={}).status_code == 400
        assert client.post(route, data={"csrf": "forged"}).status_code == 400


def test_four_eyes_rejection_approval_affirmation_and_restart(setup):
    app, db_path = setup
    analyst, reviewer, auditor = [auth(app, role) for role in store.ROLES]
    rendered = analyst.get("/trade/fictional-01").get_data(as_text=True)
    assert "Risk policy: Review" in rendered and "Booking comparison: MISMATCH" in rendered
    assert post(analyst, path("review"), action="propose", value="1,000,000").status_code == 400
    assert post(analyst, path("review"), action="propose", value="1,000,000", note="Source confirmed").status_code == 302
    db = store.connect(db_path)
    assert db.execute("SELECT state FROM trades").fetchone()[0] == "Open"
    with db:
        db.execute("UPDATE users SET role='supervisor' WHERE name='analyst'")
    assert post(analyst, path("approval"), action="approve", reason="Should fail").status_code == 403
    with db:
        db.execute("UPDATE users SET role='analyst' WHERE name='analyst'")
    assert post(reviewer, path("approval"), action="reject", reason="Recheck").status_code == 302
    assert post(analyst, path("review"), action="propose", value="1,000,000", note="Rechecked").status_code == 302
    assert post(reviewer, path("approval"), action="approve", reason="Checked").status_code == 302
    assert db.execute("SELECT state FROM trades").fetchone()[0] == "Open"
    assert post(analyst, path("review", RATE), action="accept").status_code == 302
    db.close()
    restarted = create_app(db_path, secret="z" * 32)
    assert "Affirmed" in auth(restarted, "analyst").get("/").get_data(as_text=True)
    log = auditor.get("/audit").get_data(as_text=True)
    assert all(action in log for action in ("correction_proposed", "correction_rejected", "correction_approved", "trade_affirmed"))
    db = store.connect(db_path)
    assert json.loads(db.execute("SELECT booking FROM fields WHERE path=?", (AMOUNT,)).fetchone()[0]) == "1,000,000"
    for sql in ("UPDATE audit SET action='tamper' WHERE id=1", "DELETE FROM audit WHERE id=1",
                "INSERT OR REPLACE INTO audit(id,action,detail) VALUES(1,'tamper','{}')"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            with db:
                db.execute(sql)
    db.close()


def test_matcher_statuses_tolerances_and_fuzzy_key():
    assert matcher.match(AMOUNT, "USD 1,000,000", "1000000") == "MATCH"
    assert matcher.match(AMOUNT, "1000000", "1000000.01") == "MISMATCH"
    assert matcher.match(RATE, "4%", "0.04001") == "MATCH"
    assert matcher.match(RATE, "4%", "0.040011") == "MISMATCH"
    assert matcher.match("fixed_leg.business_day_convention", "Modified Following", "MODFOLLOWING") == "MATCH"
    assert matcher.match("trade_date", "2026-11-23", "24 November 2026") == "MISMATCH"
    assert matcher.match(AMOUNT, None, "100") == "MISSING_IN_CONFIRMATION"
    assert matcher.match(AMOUNT, "100", "") == "MISSING_IN_BOOKING"
    fields = dict(zip(matcher.KEY_FIELDS, ["2026-11-23", "1,000,000", "USD", "2031-11-23"]))
    similar = {"trade_id": "other", "fields": {**fields, AMOUNT: "1000000"}}
    explicit = {"trade_id": "target", "fields": {}}
    assert matcher.link_booking("target", fields, [similar, explicit]) is explicit
    assert matcher.link_booking("target", fields, [similar]) is similar
    assert matcher.link_booking("target", fields, [similar, {**similar, "trade_id": "ambiguous"}]) is None
