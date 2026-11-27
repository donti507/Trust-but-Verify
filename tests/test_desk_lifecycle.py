"""Offline lifecycle workflow and booked version history."""
from io import BytesIO
import json
import sqlite3

import pytest

from test_desk import accounts, auth, csrf, post, setup
from test_desk_formats import XML, storage
from trustswap.desk import lifecycle, store, upload_review
from trustswap.desk.formats import fpml
from trustswap.desk.formats import flatten

TRADE = "FICT-01"
REVIEW = f"/trade/{TRADE}"
HISTORY = f"{REVIEW}/history"
PROPOSE = f"{REVIEW}/lifecycle/propose"
RATE = "fixed_leg.fixed_rate"


@pytest.fixture
def affirmed(setup):
    app, db_path = setup
    parsed = fpml.parse(XML)
    with store.connect(db_path) as db:
        terms = {path: {"confirmation": value, "booking": value, "p": 0, "c": 100, "r": 1}
                 for path, value in flatten(parsed.trade).items()}
        store.new_trade(db, TRADE, "synthetic", XML, terms)
        db.execute("UPDATE fields SET state='accepted' WHERE trade_id=?", (TRADE,))
        store.affirm_if_complete(db, TRADE, None)
    db.close()
    clients = {role: auth(app, role) for role in store.ROLES}
    return db_path, clients


def propose(client, kind, date="2026-07-01", **custom):
    return post(client, PROPOSE, event_type=kind, effective=date, note="Fictional lifecycle request", **custom)


def sign(client, db, action="approve"):
    event_id = db.execute("SELECT MAX(id) FROM desk_lifecycle").fetchone()[0]
    return post(client, f"/lifecycle/{event_id}/decision", action=action, reason="Checked booked terms")


def test_amendment_upload_diffs_against_booking_and_preserves_versions(affirmed, storage, monkeypatch):
    db_path, clients = affirmed
    analyst, supervisor = clients["analyst"], clients["supervisor"]
    db = store.connect(db_path)
    revised = XML.replace("<initialValue>0.04</initialValue>", "<initialValue>0.05</initialValue>")

    def never(*args):
        pytest.fail("The structured amendment must remain local")

    monkeypatch.setattr(upload_review, "scorer", never)
    response = analyst.post(PROPOSE, data={
        "event_type": "amendment", "effective": "2026-07-01", "note": "Confirm updated fixed rate",
        "fictional": "yes", "confirmation": (BytesIO(revised.encode()), "amendment.xml"), "csrf": csrf(analyst),
    })
    assert response.status_code == 302 and response.location == HISTORY
    pending = lifecycle.requests(db)[0]
    assert pending["upload_sha256"]
    assert pending["changes"] == {RATE: {"before": "0.04", "after": "0.05"}}
    assert lifecycle.booked_terms(db, TRADE)[RATE] == "0.04"
    inbox = supervisor.get("/approvals").data
    assert b"Lifecycle approvals" in inbox and b"Booked version 1" in inbox and b"0.05" in inbox
    assert propose(analyst, "amendment", field=RATE, value="6%").status_code == 400
    assert sign(supervisor, db).status_code == 302
    assert lifecycle.booked_terms(db, TRADE)[RATE] == "0.05"
    versions = lifecycle.versions(db, TRADE)
    assert [(row["version"], row["terms"][RATE]) for row in versions] == [(2, "0.05"), (1, "0.04")]
    for sql in ("UPDATE desk_trade_versions SET effective='1900-01-01'", "DELETE FROM desk_trade_versions"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            with db:
                db.execute(sql)
    history = clients["auditor"].get(HISTORY).data
    assert b"Version 2" in history and b"Version 1" in history and b"Approved by supervisor" in history
    exported = analyst.get(f"{REVIEW}/export/fpml")
    assert exported.status_code == 200
    assert flatten(fpml.parse(exported.get_data(as_text=True)).trade)[RATE] == "0.05"
    db.close()


def test_roles_validation_and_termination_lock(affirmed):
    db_path, clients = affirmed
    analyst, supervisor, auditor = (clients[role] for role in store.ROLES)
    db = store.connect(db_path)
    assert analyst.post(PROPOSE, data={}).status_code == 400
    assert propose(supervisor, "full").status_code == propose(auditor, "full").status_code == 403
    assert propose(analyst, "full", "2027-04-01").status_code == 400
    response = propose(analyst, "partial", value="1,000,000")
    assert response.status_code == 400 and b"below the current notional" in response.data
    assert b'value="1,000,000"' in response.data
    assert propose(analyst, "partial", value="750,000").status_code == 302
    with db:
        db.execute("UPDATE users SET role='supervisor' WHERE name='analyst'")
    assert sign(analyst, db).status_code == 403
    with db:
        db.execute("UPDATE users SET role='analyst' WHERE name='analyst'")
    assert sign(auditor, db).status_code == 403
    assert sign(supervisor, db).status_code == 302
    latest = lifecycle.versions(db, TRADE)[0]
    assert latest["terms"]["floating_leg.notional_amount"] == latest["terms"]["fixed_leg.notional_amount"] == "750000"
    assert propose(analyst, "novation", outgoing="Synthetic Party A", incoming="Synthetic Party C").status_code == 302
    assert sign(supervisor, db, "reject").status_code == 302
    assert lifecycle.versions(db, TRADE)[0]["version"] == 2
    assert propose(analyst, "novation", outgoing="Synthetic Party A", incoming="Synthetic Party C").status_code == 302
    assert sign(supervisor, db).status_code == 302
    assert lifecycle.versions(db, TRADE)[0]["terms"]["fixed_leg.payer"] == "Synthetic Party C"
    assert propose(analyst, "full", "2026-07-15").status_code == 302
    assert not lifecycle.terminated(db, TRADE)
    assert sign(supervisor, db).status_code == 302
    assert lifecycle.terminated(db, TRADE)
    assert lifecycle.versions(db, TRADE)[0]["terms"]["termination_date"] == "2026-07-15"
    assert b"Terminated" in analyst.get("/").data
    page = analyst.get(REVIEW).data
    assert b"Terminated" in page and b"Propose lifecycle change" not in page
    assert propose(analyst, "amendment", field=RATE, value="5%").status_code == 409
    assert post(analyst, f"{REVIEW}/accept-matched").status_code == 409
    assert post(analyst, f"{REVIEW}/field/{RATE}/review", action="accept").status_code == 409
    assert sign(supervisor, db).status_code == 400
    expected = ["lifecycle_proposed", "lifecycle_approved", "lifecycle_proposed", "lifecycle_rejected",
                "lifecycle_proposed", "lifecycle_approved", "lifecycle_proposed", "lifecycle_approved"]
    rows = db.execute("SELECT action,detail FROM audit WHERE action LIKE 'lifecycle_%' ORDER BY id").fetchall()
    assert [row["action"] for row in rows] == expected
    assert json.loads(rows[-1]["detail"])["event_type"] == "full"
    db.close()
