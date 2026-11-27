"""Offline fixtures for Step 4 screening, blind labelling, and adjudication."""
from contextlib import closing
import csv
from decimal import Decimal
from io import BytesIO
import json
import re
import sqlite3

import bcrypt
import pytest

from trustswap.annotate import adjudicate, checker, ingest, labels, stats, store, workflow
from trustswap.annotate.app import create_app
from trustswap.common.agreement import cohen_kappa
from trustswap.common.hashing import sha256_bytes
from trustswap.common.records import GoldRecord

PASS = {"answers": {key: True for key, _ in labels.CHECKLIST}, "include": True,
        "reason": "Fictional example is eligible", "duplicate_of": ""}
PASSWORD = "fictional-password"
RATE = "fixed_leg.fixed_rate"


@pytest.fixture
def corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        fields = ["doc_id", "path", "source_url", "sha256", "split"]
        csv_rows = csv.DictWriter(stream, fieldnames=fields)
        csv_rows.writeheader()
        for i in range(30):
            name = f"mock-{i:02}"
            text = (f"<p>Fictional 📄 {name}</p><p>Trade Date: 04-Jan-2024</p>   "
                    "<p>Effective Date: 08-Jan-2024</p>\n<p>Termination Date: 2029-01-08</p>"
                    "<p>Fixed Rate: 4 percent</p><p>Notional Amount: USD 1,000,000</p>"
                    "<script>hidden sample</script>")
            (raw / f"{name}.html").write_text(text, encoding="utf-8")
            csv_rows.writerow({"doc_id": name, "path": f"{name}.html", "source_url": f"https://example.invalid/{name}",
                               "sha256": sha256_bytes(text.encode())})
        # A path that does not exist proves the locked row is skipped before I/O.
        csv_rows.writerow({"doc_id": "locked", "path": "does-not-exist", "split": "real_test"})
    return raw, manifest


def initialize(path, corpus):
    app = create_app(path, secret="k" * 32)
    with closing(store.connect(path)) as db, store.transaction(db):
        hashed = bcrypt.hashpw(PASSWORD.encode(), bcrypt.gensalt(rounds=4))
        db.executemany("INSERT INTO users VALUES (?,?,?)", [(role.lower(), role, hashed) for role in store.ROLES])
    with closing(store.connect(path)) as db:
        assert ingest.import_documents(db, *corpus, random_seed=42) == 30
    return app


def screen_all(db, *, exclude=None):
    known = [row[0] for row in db.execute("SELECT id FROM documents")]
    with store.transaction(db):
        for name in known:
            item = PASS
            if name == exclude:
                item = {**PASS, "include": False, "reason": "Duplicate fictional exhibit",
                        "answers": {**PASS["answers"], "latest": False}, "duplicate_of": known[0]}
            workflow.screen_document(db, name, "a1", item)


@pytest.fixture
def local(tmp_path, corpus):
    db_path = tmp_path / "annotation.db"
    app = initialize(db_path, corpus)
    with closing(store.connect(db_path)) as db:
        screen_all(db)
    return app, db_path


def auth(app, role):
    client = app.test_client()
    assert client.get("/login").status_code == 200
    response = post(client, "/login", username=role.lower(), password=PASSWORD)
    assert response.status_code == 302
    assert "HttpOnly" in response.headers["Set-Cookie"] and "SameSite=Strict" in response.headers["Set-Cookie"]
    client.get("/")
    return client


def token(client):
    with client.session_transaction() as state:
        return state["csrf"]


def post(client, url, **form):
    return client.post(url, data={"csrf": token(client), **form})


def save(client, doc, form, revision=0, *, submit=False):
    return client.post(f"/documents/{doc}/save", json={"form": form, "revision": revision, "submit": submit, "delta": 2000},
                       headers={"X-CSRF-Token": token(client)})


def view(client, doc):
    html = client.get(f"/documents/{doc}").get_data(as_text=True)
    return json.loads(re.search(r'<script type="application/json" id="editor-data">(.*?)</script>', html, re.S)[1])


def complete(text, structure="fixed_float"):
    form = labels.empty_form()
    form.update(structure=structure, flags_reviewed=True)
    for path in labels.paths(structure):
        term = checker.suggestion(path, text) or {"status": "absent", "value": "", "span": None, "note": "Not stated"}
        form["fields"][path] = {**term, "confirmed": True}
    return form


def egress(client):
    response = client.get("/exchange/export")
    assert response.status_code == 200
    return response.data


def ingress(client, content):
    return post(client, "/exchange/import", labels=(BytesIO(content), "labels.jsonl"))


def test_independent_checker_boundaries_and_parse_flags():
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-c", """
import sys
from trustswap.annotate import checker
blocked = ('trustswap.extract', 'trustswap.score', 'trustswap.verify', 'openai', 'httpx')
assert not [name for name in sys.modules if name.startswith(blocked)]
"""], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    text = ("Trade Date: 2025-01-03\nEffective Date: 2025-01-02\nTermination Date: 2025-01-01\n"
            "Fixed Rate: 4 percent\nNotional Amount: USD 10,000")
    fields = complete(text)["fields"]
    fields[RATE]["value"] = "0.05"
    fields["fixed_leg.notional_amount"]["value"] = "lots"
    fields["fixed_leg.notional_currency"]["value"] = "???"
    findings = checker.check(text, fields)
    assert "The selected text does not support this value." in findings[RATE]
    assert "A labelled term in the document has a different value." in findings[RATE]
    assert findings["fixed_leg.notional_amount"] == ["The value could not be parsed."]
    assert findings["fixed_leg.notional_currency"] == ["The value could not be parsed."]
    assert all(any("date ≤" in error for error in findings[path]) for path in checker.DATE_NAMES)


def test_ingest_and_inclusion_checklist(tmp_path, corpus):
    app = initialize(tmp_path / "annotation.db", corpus)
    a1, a2 = (auth(app, role) for role in ("A1", "A2"))
    url = "/inclusion/mock-29"
    assert a2.get(url).status_code == 403
    response = a1.get(url)
    assert b"One specific trade" in response.data and b"hidden sample" not in response.data
    checks = {key: "yes" for key in PASS["answers"]}
    assert post(a1, url, **checks, decision="include", reason="").status_code == 400
    with closing(store.connect(tmp_path / "annotation.db")) as db:
        before = workflow.document(db, "mock-29")
        assert sha256_bytes(before["text"].encode()) == before["text_hash"]
        assert len(json.loads(before["offsets"])) == len(before["text"])
        assert "hidden sample" not in before["text"]
        assert workflow.document(db, "locked") is None
        assert a1.get("/documents/mock-29").status_code == 404
        screen_all(db, exclude="mock-29")
        assert store.sealed(db)
        assert db.execute("SELECT count(*) FROM cohort").fetchone()[0] == 29
        assert db.execute("SELECT count(*) FROM cohort WHERE overlap=1").fetchone()[0] == 10
        assert json.loads(db.execute("SELECT record FROM screening WHERE doc='mock-29'").fetchone()[0])["duplicate_of"] == "mock-00"
        assert ingest.import_documents(db, *corpus, random_seed=42) == 0


def test_assignments_blinding_and_role_permissions(local):
    app, path = local
    a1, a2, adj = (auth(app, role) for role in store.ROLES)
    with closing(store.connect(path)) as db:
        assigned = [tuple(row) for row in db.execute("SELECT * FROM cohort ORDER BY doc")]
    assert assigned == store.assign([row[0] for row in reversed(assigned)], 42)
    assert sum(row[1] for row in assigned) == 11
    assert sum(row[2] for row in assigned) == 10
    overlap, control = (next(row[0] for row in assigned if row[index]) for index in (1, 2))
    prefill = next(doc for doc, _, flag in assigned if not flag)
    private = next(doc for doc, share, _ in assigned if not share)
    assert a2.get(f"/documents/{private}").status_code == 404
    assert post(a2, f"/documents/{private}/start", mode="prefill").status_code == 404
    assert adj.get(f"/documents/{overlap}").status_code == 403
    assert a1.get(f"/adjudication/{overlap}").status_code == 403
    assert a2.get("/gold.jsonl").status_code == 403
    assert adj.get("/exchange/export").status_code == 403
    assert a1.post(f"/documents/{prefill}/start", data={"mode": "prefill"}).status_code == 400
    for client, doc, mode in ((a1, prefill, "prefill"), (a1, control, "control"), (a2, overlap, "blind")):
        assert post(client, f"/documents/{doc}/start", mode="prefill").status_code == 302
        initial = view(client, doc)
        assert initial["mode"] == mode and not initial["checks"]
        assert bool(initial["suggestions"]) == (mode == "prefill")
        assert initial["original"] is None
    assert b"Agreement on the overlap" not in a1.get("/stats").data
    assert b"Agreement on the overlap" in adj.get("/stats").data


def test_autosave_recheck_and_immutable_audit(local):
    app, path = local
    client = auth(app, "a1")
    with closing(store.connect(path)) as db:
        doc = db.execute("SELECT doc FROM cohort WHERE control=0 ORDER BY doc").fetchone()[0]
        text = workflow.document(db, doc)["text"]
        form = complete(text)
        form["fields"][RATE]["value"] = "0.05"
        form["cursor"] = RATE
        assert post(client, f"/documents/{doc}/start", mode="blind").status_code == 302
        offset = form["fields"][RATE]["span"]
        selected = client.post(f"/documents/{doc}/selection", json={"path": RATE, "span": offset},
                               headers={"X-CSRF-Token": token(client)})
        assert selected.get_json() == {"value": "0.04"}
        assert save(client, doc, form).status_code == 200
        assert view(client, doc)["form"]["cursor"] == RATE
        assert view(client, doc)["checks"] == {}
        assert save(client, doc, form).status_code == 409
        assert save(client, doc, form, 1, submit=True).get_json()["stage"] == "review"
        assert RATE in view(client, doc)["checks"]
        form["fields"][RATE]["value"] = "0.04"
        assert save(client, doc, form, 2, submit=True).status_code == 400
        form["reviewed"] = list(view(client, doc)["checks"])
        assert save(client, doc, form, 2, submit=True).status_code == 400
        form["reasons"][RATE] = "Re-read the source rate"
        assert save(client, doc, form, 2, submit=True).get_json()["stage"] == "final"
        row = db.execute("SELECT * FROM audit WHERE action='field_changed'").fetchone()
        assert (row["field"], row["user"], row["reason"]) == (RATE, "a1", "Re-read the source rate")
        assert json.loads(row["before"])["value"] == "0.05"
        assert json.loads(row["after"])["value"] == "0.04"
        assert row["at"]
        assert stats.progress(db, "A1")["changed"] == 1.0
        for sql in ("UPDATE audit SET reason='tampered'", "DELETE FROM audit",
                    "INSERT OR REPLACE INTO audit SELECT * FROM audit WHERE id=1"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"), db:
                db.execute(sql)


def test_control_bypasses_checker_and_prefill_needs_human_confirmation(local, monkeypatch):
    app, path = local
    a1 = auth(app, "a1")
    with closing(store.connect(path)) as db:
        control, prefill = [db.execute("SELECT doc FROM cohort WHERE control=? LIMIT 1", (flag,)).fetchone()[0]
                            for flag in (1, 0)]
        text = workflow.document(db, control)["text"]
        form = complete(text)
        form["fields"][RATE]["value"] = "0.05"
        with monkeypatch.context() as disabled:
            def unexpected(*args):
                pytest.fail("Blind controls must not run the checker or generate pre-fill")
            disabled.setattr(checker, "check", unexpected)
            disabled.setattr(checker, "suggestion", unexpected)
            post(a1, f"/documents/{control}/start", mode="prefill")
            assert view(a1, control)["suggestions"] == {}
            assert save(a1, control, form, submit=True).get_json()["stage"] == "final"
        form = complete(workflow.document(db, prefill)["text"])
        form["fields"][RATE]["confirmed"] = False
        post(a1, f"/documents/{prefill}/start", mode="prefill")
        assert save(a1, prefill, form, submit=True).status_code == 400
        form["fields"][RATE]["confirmed"] = True
        form["fields"][RATE]["value"] = "0.05"
        assert save(a1, prefill, form, submit=True).get_json()["stage"] == "review"
        actions = dict(db.execute("SELECT field,action FROM audit WHERE action LIKE 'prefill_%' AND doc=?", (prefill,)))
        assert actions[RATE] == "prefill_changed" and actions["trade_date"] == "prefill_accepted"


def test_offline_label_exchange_adjudication_and_gold(local, tmp_path, corpus):
    app, path = local
    other_path = tmp_path / "other.db"
    remote = initialize(other_path, corpus)
    a1, adj = (auth(app, role) for role in ("A1", "ADJ"))
    a2 = auth(remote, "A2")
    with closing(store.connect(path)) as db:
        doc = db.execute("SELECT doc FROM cohort WHERE overlap=1 AND control=0 ORDER BY doc").fetchone()[0]
        source = workflow.document(db, doc)
        left = complete(source["text"])
        post(a1, f"/documents/{doc}/start", mode="blind")
        assert save(a1, doc, left, submit=True).get_json()["stage"] == "final"
        packet = egress(a1)
        assert source["text"].encode() not in packet
        assert b"labels_a1.jsonl" in a1.get("/exchange/export").headers["Content-Disposition"].encode()
        assert ingress(a2, packet).status_code == 200
        assert b"Imported 0 finalized documents." in ingress(a2, packet).data
        assert post(a2, f"/documents/{doc}/start", mode="prefill").status_code == 302
        assert view(a2, doc)["mode"] == "blind"
        assert view(a2, doc)["form"]["fields"] == {}
        right = complete(source["text"])
        right["fields"][RATE]["value"] = "0.05"
        assert save(a2, doc, right, submit=True).get_json()["stage"] == "review"
        right["reviewed"] = list(view(a2, doc)["checks"])
        assert save(a2, doc, right, 1, submit=True).get_json()["stage"] == "final"
        returned = egress(a2)
        broken = [json.loads(line) for line in returned.splitlines()]
        broken[1]["text_hash"] = "0" * 64
        invalid = "".join(json.dumps(row) + "\n" for row in broken).encode()
        assert ingress(adj, invalid).status_code == 400 and store.get_work(db, doc, "a2") is None
        assert ingress(adj, returned).status_code == 200
        assert ingress(adj, returned).status_code == 200
        conflict = [json.loads(line) for line in returned.splitlines()]
        for snapshot in ("form", "submitted"):
            conflict[1]["data"][snapshot]["fields"]["floating_leg.payer"]["note"] = "Another answer"
        assert ingress(adj, b"".join((json.dumps(row) + "\n").encode() for row in conflict)).status_code == 400
        assert adj.get("/gold.jsonl").data == b""
        conflicts = adjudicate.pending_documents(db)
        assert conflicts == [(doc, 1)]
        page = adj.get(f"/adjudication/{doc}").get_data(as_text=True)
        assert "Use A1" in page and "Use A2" in page and "0.05" in page
        assert "<h2>Trade date</h2>" not in page
        assert "0.05" not in a1.get(f"/documents/{doc}").get_data(as_text=True)
        action = f"/adjudication/{doc}/decide"
        assert post(a1, action, field=RATE, choice="A1", reason="Void").status_code == 403
        assert post(adj, action, field=RATE, choice="manual", status="explicit", reason="").status_code == 400
        start, end = left["fields"][RATE]["span"]
        assert post(adj, action, field=RATE, choice="manual", status="explicit", value="0.04",
                    start=start, end=end, reason="The source explicitly states four percent").status_code == 302
        gold = GoldRecord.model_validate_json(adj.get("/gold.jsonl").data)
        assert gold.trade.fixed_leg.fixed_rate.value == Decimal("0.04")
        assert gold.adjudicated and gold.annotators == ["a1", "a2", "adj"]
        assert gold.sha256 == source["raw_hash"] and gold.source_url == source["url"]
        paired, metrics = stats.field_agreement(db)
        assert paired == stats.field_agreement(db, "blind")[0] == 1
        assert stats.field_agreement(db, "control")[0] == 0
        rate = next(row for row in metrics if row["field"] == RATE)
        assert rate["agreement"] == 0 and rate["kappa"] is None
        assert adjudicate.pending_documents(db) == []


def test_status_kappa_known_cases():
    assert cohen_kappa(["absent", "explicit"], ["explicit", "absent"]) == -1
    assert cohen_kappa(["absent", "explicit", "explicit", "absent"],
                       ["absent", "explicit", "absent", "explicit"]) == 0
    assert cohen_kappa(["absent"], ["absent"]) is None
    assert cohen_kappa([], []) is None


def test_values_can_be_inside_a_larger_unicode_span():
    source = "Example 📄. On 4 January 2024 the amount was USD 1.2 million at 4 percent."
    values = {"trade_date": "2024-01-04", "fixed_leg.notional_amount": "1200000",
              "fixed_leg.notional_currency": "USD", RATE: "0.04"}
    answer = {"status": "explicit", "span": [0, len(source)], "note": "", "confirmed": True}
    assert checker.check(source, {path: {**answer, "value": value} for path, value in values.items()}) == {}
    assert labels.normalize_selection("fixed_leg.notional_currency", "€") == "EUR"


def test_gold_of_unpaired_basis_swap_has_second_floating_leg(local):
    app, path = local
    a1, adj = (auth(app, role) for role in ("A1", "ADJ"))
    with closing(store.connect(path)) as db:
        doc = db.execute("SELECT doc FROM cohort WHERE overlap=0 AND control=1 LIMIT 1").fetchone()[0]
        text = workflow.document(db, doc)["text"]
    post(a1, f"/documents/{doc}/start", mode="blind")
    form = complete(text, "float_float")
    assert save(a1, doc, form, submit=True).get_json()["stage"] == "final"
    gold = GoldRecord.model_validate_json(adj.get("/gold.jsonl").data)
    assert gold.trade.floating_leg and gold.trade.floating_leg_2
    assert gold.trade.fixed_leg is None and not gold.adjudicated and gold.annotators == ["a1"]
