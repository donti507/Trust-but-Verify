"""Offline seed, booking import and audit export tests."""
import csv
from io import BytesIO, StringIO
import json

import pytest

pytest.importorskip("flask")
pytest.importorskip("bcrypt")

from scripts.build_demo import render_page
from scripts import import_bookings as booking_cli
from test_desk import PASSWORD, auth, csrf, post, setup, accounts
from trustswap.desk import exports, imports, matcher, store
from trustswap.desk.__main__ import main
from trustswap.desk.matcher import KEY_FIELDS

AMOUNT = "floating_leg.notional_amount"
RATE = "fixed_leg.fixed_rate"


def doc(index, live=False):
    values = dict(zip(KEY_FIELDS, ["2026-11-20", 1_000_000 + index, "USD", "2031-11-20"]))
    values.update({RATE: "0.04", "fixed_leg.business_day_convention": "MODFOLLOWING"})
    fields = [{"field": path, "name": path, "value": value, "p": 0.01, "c_bar": 10, "r": 1}
              for path, value in values.items()]
    row = {"id": f"{'feed' if live else 'replay'}-{index:02d}", "text": "Synthetic Party A",
           "category": "Live feed" if live else "Clean synthetic", "fields": fields}
    if live:
        row["collection"] = "live_feed"
    return row


def test_seed_replay_and_feed_has_exact_cohort_and_deterministic_differences(tmp_path):
    replay = tmp_path / "index.html"
    replay.write_text(render_page([doc(index) for index in range(30)], 100), encoding="utf-8")
    feed = tmp_path / "feed"
    feed.mkdir()
    day = "2026-11-23"
    (feed / f"{day}.json").write_text(json.dumps({"collection": "live_feed", "date": day,
                                                "documents": [doc(i, live=True) for i in range(5)]}))
    path = tmp_path / "desk.db"
    store.initialize(path)
    db = store.connect(path)
    try:
        assert imports.seed(db, replay, feed) == (35, 7)
        assert imports.seed(db, replay, feed) == (0, 0)
        assert dict(db.execute("SELECT source_collection,count(*) FROM trades GROUP BY source_collection")) == {
            "synthetic": 30, "live_feed": 5,
        }
        assert {row[0] for row in db.execute("SELECT DISTINCT collection FROM trades")} == {"desk"}
        changes = [json.loads(row[0])["kind"] for row in db.execute(
            "SELECT detail FROM audit WHERE action='booking_difference_planted'"
        )]
        assert len(changes) == 7 and set(changes) == set(imports.DIFFERENCES)
        fields = db.execute("SELECT trade_id,path,confirmation,booking FROM fields").fetchall()
        mismatched = {r[0] for r in fields if matcher.match(r[1], json.loads(r[2]), json.loads(r[3])) != "MATCH"}
        assert len(mismatched) == 7
        other_path = tmp_path / "other.db"
        store.initialize(other_path)
        other = store.connect(other_path)
        try:
            imports.seed(other, replay, feed)
            assert list(other.execute("SELECT trade_id,path,confirmation,booking FROM fields")) == fields
        finally:
            other.close()
    finally:
        db.close()


def test_csv_linked_by_normalized_key_and_missing_confirmation(tmp_path):
    replay = tmp_path / "index.html"
    source = doc(0)
    source["fields"] = [field for field in source["fields"] if field["field"] != RATE]
    replay.write_text(render_page([source], 100), encoding="utf-8")
    path = tmp_path / "desk.db"
    store.initialize(path)
    db = store.connect(path)
    imports.seed(db, replay)
    fields = ",".join([*KEY_FIELDS, RATE])
    rows = fields + '\n20 November 2026,"1,000,000",USD,2031-11-20,4%\n'
    assert imports.import_bookings(db, StringIO(rows)) == imports.ImportReport(1, 1, [])
    field = db.execute("SELECT confirmation,booking FROM fields WHERE path=?", (RATE,)).fetchone()
    assert json.loads(field[0]) is None and json.loads(field[1]) == "4%"
    assert imports.import_bookings(db, StringIO(rows)) == imports.ImportReport(1, 0, [])
    with db:
        db.execute("UPDATE fields SET state='accepted' WHERE path=?", (RATE,))
    rejected = imports.import_bookings(db, StringIO(rows.replace("4%", "5%"))).rejected
    assert rejected == [imports.RejectedRow(2, "A booking cannot be replaced after review has started")]
    row = db.execute("SELECT booking FROM fields WHERE path=?", (RATE,)).fetchone()
    assert json.loads(row[0]) == "4%"
    db.close()


def test_upload_permission_csrf_and_auditor_exports(setup):
    app, path = setup
    content = f"trade_id,{AMOUNT},{RATE}\nfictional-01,1000000,0.04\n".encode()
    for role in ("supervisor", "analyst", "auditor"):
        client = auth(app, role)
        invalid = client.post("/bookings/import", data={"bookings": (BytesIO(content), "bookings.csv")})
        assert invalid.status_code == 400
        uploaded = client.post("/bookings/import", data={
            "csrf": csrf(client), "bookings": (BytesIO(content), "bookings.csv"),
        })
        assert uploaded.status_code == (302 if role == "supervisor" else 403)
        for kind in ("json", "csv"):
            output = client.get(f"/audit.{kind}")
            assert output.status_code == (200 if role in {"auditor", "supervisor"} else 403)
            if role in {"auditor", "supervisor"}:
                assert output.headers["Content-Disposition"] == f"attachment; filename=desk-audit.{kind}"
                if kind == "json":
                    rows = output.get_json()
                    audit_db = store.connect(path)
                    try:
                        stored = [dict(row) for row in exports.audit_rows(audit_db)]
                    finally:
                        audit_db.close()
                    for row in stored:
                        row["detail"] = json.loads(row["detail"])
                    assert rows == stored
                    assert any(row["action"] == "booking_imported" for row in rows)
                else:
                    rows = list(csv.DictReader(StringIO(output.get_data(as_text=True))))
                    assert any(row["action"] == "booking_imported" for row in rows)
                    audit_db = store.connect(path)
                    try:
                        expected = [{key: "" if value is None else str(value) for key, value in dict(row).items()}
                                    for row in exports.audit_rows(audit_db)]
                    finally:
                        audit_db.close()
                    assert rows == expected
    db = store.connect(path)
    text = db.execute("SELECT booking FROM fields WHERE path=?", (AMOUNT,)).fetchone()[0]
    assert json.loads(text) == "1000000"
    db.close()


def test_create_user_cli_and_fictional_party_validation(tmp_path, monkeypatch):
    path = tmp_path / "desk.db"
    monkeypatch.setattr("getpass.getpass", lambda prompt: PASSWORD)
    main(["--db", str(path), "create-user", "example-auditor", "auditor"])
    db = store.connect(path)
    row = db.execute("SELECT name,password,role FROM users").fetchone()
    assert row["name"] == "example-auditor" and row["role"] == "auditor"
    assert row["password"].startswith(b"$2") and PASSWORD.encode() not in row["password"]
    _, rejected = imports.parse_bookings(StringIO("trade_id,floating_leg.payer\nfictional-01,Unknown party\n"))
    assert "fictional" in rejected[0].reason
    db.close()



def test_csv_cli_reports_rejected_lines_and_validates_headers(setup, tmp_path, capsys):
    _, path = setup
    file = tmp_path / "bookings.csv"
    file.write_text(f"trade_id,{AMOUNT},{RATE}\nfictional-01,1000000,0.04\nmissing,wrong,0.04\n"
                    "unmatched,800000,0.04\n", encoding="utf-8")
    assert booking_cli.main([str(file), "--db", str(path)]) == 1
    output = capsys.readouterr().out
    assert "Updated 1. Rejected 2." in output
    assert "Line 3:" in output and "Line 4: No matching trade" in output
    file.write_text("trade_id,unsupported\nfictional-01,invalid\n", encoding="utf-8")
    assert booking_cli.main([str(file), "--db", str(path)]) == 1
    assert "CSV headers" in capsys.readouterr().out
