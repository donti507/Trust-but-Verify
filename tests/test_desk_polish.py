"""Desk review presentation and bulk clearing on fictional fixtures."""
import json

from test_desk import AMOUNT, RATE, accounts, auth, path, post, setup

import pytest

from trustswap.desk import matcher, presentation, store

TRADE = "/trade/fictional-01"
CENTER = "fixed_leg.business_centers"
CONVENTION = "fixed_leg.business_day_convention"


@pytest.mark.parametrize(
    ("field", "raw", "booked"),
    [
        ("trade_date", "/ 2023-12-21", "2023-12-21"),
        ("trade_date", " : 21-Dec-2023 ", "2023-12-21"),
        (RATE, "| 4 percent", "0.04"),
        (RATE, "4 per cent", "0.04"),
        (RATE, "400 bps", "0.04"),
        (CONVENTION, ": Modified Following", "MODFOLLOWING"),
        (CENTER, "/ London and TARGET", ["EUTA", "GBLO"]),
    ],
)
def test_equivalent_formats_match(field, raw, booked):
    assert matcher.match(field, raw, booked) == "MATCH"


def test_readable_values_and_evidence(setup):
    app, db_path = setup
    text = "Trade date / 2023-12-21\nFinancial centres: London and TARGET\nUSD 1,000,000"
    raw_date = "/ 2023-12-21"
    start = text.index(raw_date)
    db = store.connect(db_path)
    with db:
        store.new_trade(db, "fictional-02", "synthetic", text, {
            "trade_date": {"confirmation": "2023-12-21", "booking": "2023-12-21", "raw": raw_date,
                           "span": [start, start + len(raw_date)], "p": 0, "c": 100, "r": 1},
            CENTER: {"confirmation": ["EUTA", "GBLO"], "booking": "London and TARGET",
                     "raw": "London and TARGET", "p": 0, "c": 100, "r": 1},
            AMOUNT: {"confirmation": "USD 1,000,000", "booking": "900,000", "p": 0.2, "c": 100, "r": 1},
        })
    db.close()
    html = auth(app, "analyst").get("/trade/fictional-02").get_data(as_text=True)
    assert 'href="#field-2" class="source-highlight ok"' in html
    assert 'href="#field-1" class="source-callout review">Sent to review' in html
    assert 'id="field-2"' in html
    assert 'Normalized: 2023-12-21' in html
    assert 'Business centers (fixed leg)' in html and 'TARGET (EUTA), London (GBLO)' in html
    assert 'Trade dates' in html and 'Floating leg' in html and '0 of 3 fields cleared' in html
    assert presentation.comparison_value(CONVENTION, "MODFOLLOWING") == ("MODFOLLOWING", "Modified Following")


def test_validation_errors_keep_input(setup):
    app, db_path = setup
    analyst, supervisor = (auth(app, role) for role in ("analyst", "supervisor"))
    response = post(analyst, path("review"), action="propose", value=" 1,000,000 ", note=" ")
    assert response.status_code == 400
    assert b'role="alert">Proposed corrections need a value and note' in response.data
    assert b'value=" 1,000,000 "' in response.data
    response = post(analyst, path("review"), action="propose", value="Wrong <input>", note="Source & booking")
    assert response.status_code == 400
    assert b'Wrong &lt;input&gt;' in response.data and b'Source &amp; booking' in response.data
    response = post(supervisor, "/settings/lambda", **{"lambda": "infinite?"})
    assert response.status_code == 400
    assert b'Desk overview' in response.data and b'value="infinite?"' in response.data
    assert post(analyst, path("review"), action="propose", value="1,000,000", note="Checked").status_code == 302
    response = post(supervisor, path("approval"), action="invalid", reason="Keep my reason")
    assert response.status_code == 400
    assert b'Approvals inbox' in response.data and b'>Keep my reason</textarea>' in response.data
    db = store.connect(db_path)
    assert db.execute("SELECT state FROM fields WHERE path=?", (AMOUNT,)).fetchone()[0] == "pending"
    db.close()


def test_bulk_accept_checks_both_conditions_and_audits_each_field_once(setup):
    app, db_path = setup
    db = store.connect(db_path)
    with db:
        db.executemany("INSERT INTO fields(trade_id,path,confirmation,booking,p,c,r,state) VALUES "
                       "('fictional-01',?,?,?,?,100,1,?)", [
            ("trade_date", '"2023-12-21"', '"2023-12-22"', 0.001, "open"),
            ("effective_date", '"2023-12-21"', '"2023-12-21"', 0.02, "open"),
            ("termination_date", '"2025-12-21"', '"2025-12-21"', 0.01, "rejected"),
            (CENTER, '"TARGET"', '"EUTA"', 0.001, "pending"),
        ])
    route = f"{TRADE}/accept-matched"
    for role in ("supervisor", "auditor"):
        client = auth(app, role)
        assert b"Accept all matched low-risk fields" not in client.get(TRADE).data
        assert post(client, route).status_code == 403
    client = auth(app, "analyst")
    assert b"Accept all matched low-risk fields" in client.get(TRADE).data
    assert client.post(route).status_code == 400
    assert post(client, route).status_code == 302
    assert post(client, route).status_code == 302
    assert b"2 of 6 fields cleared" in client.get(TRADE).data
    assert dict(db.execute("SELECT path,state FROM fields")) == {
        AMOUNT: "open", RATE: "accepted", "trade_date": "open", "effective_date": "open",
        "termination_date": "accepted", CENTER: "pending",
    }
    events = list(db.execute("SELECT field,detail FROM audit WHERE action='field_accepted' ORDER BY field"))
    assert [(row["field"], json.loads(row["detail"])) for row in events] == [
        (RATE, {"via": "matched_low_risk"}), ("termination_date", {"via": "matched_low_risk"}),
    ]
    db.close()


def test_bulk_accept_affirms_a_trade_when_it_clears_the_last_field(setup):
    app, db_path = setup
    client = auth(app, "analyst")
    assert post(client, path("review"), action="accept").status_code == 302
    assert post(client, f"{TRADE}/accept-matched").status_code == 302
    assert b"Affirmed" in client.get(TRADE).data
    db = store.connect(db_path)
    assert db.execute("SELECT COUNT(*) FROM audit WHERE action='trade_affirmed'").fetchone()[0] == 1
    db.close()


def test_reseed_recovers_evidence_for_existing_confirmation(setup, tmp_path):
    from scripts.build_demo import render_page
    from trustswap.desk import imports

    _, db_path = setup
    text = "Synthetic Party A · fictional terms"
    document = {"id": "fictional-01", "category": "Clean synthetic", "text": text, "fields": [
        {"field": AMOUNT, "value": "USD 1,000,000", "p": 0.2, "c_bar": 100, "r": 1},
        {"field": RATE, "value": "4.00%", "raw_value": "4 percent", "p": 0.001, "c_bar": 100, "r": 1},
    ]}
    replay = tmp_path / "index.html"
    replay.write_text(render_page([document], 100), encoding="utf-8")
    db = store.connect(db_path)
    with db:
        db.execute("UPDATE fields SET state='accepted' WHERE path=?", (RATE,))
    try:
        assert imports.seed(db, replay) == (0, 0)
        row = db.execute("SELECT state,raw FROM fields JOIN field_evidence USING (trade_id,path) "
                         "WHERE path=?", (RATE,)).fetchone()
        assert tuple(row) == ("accepted", '"4 percent"')
    finally:
        db.close()
