"""Private JSONL exchange for finalized human labels and the locked cohort."""
from datetime import datetime
import json

from trustswap.annotate import checker, labels, store, workflow
from trustswap.common.hashing import sha256_bytes

VERSION = 1


def corpus(db):
    return [dict(row) for row in db.execute("SELECT id,raw_hash,text_hash,url FROM documents ORDER BY id")]


def fingerprint(db):
    return sha256_bytes(store.dump({"seed": store.seed(db), "documents": corpus(db),
                                   "cohort": [dict(row) for row in db.execute("SELECT * FROM cohort ORDER BY doc")]}).encode())


def export(db, user, role):
    if role not in {"A1", "A2"} or user != store.user_for(db, role) or not store.sealed(db):
        raise ValueError("A locked cohort is required to export labels")
    screenings = None
    if role == "A1":
        screenings = {row["doc"]: json.loads(row["record"]) for row in db.execute("SELECT * FROM screening ORDER BY doc")}
    head = {"version": VERSION, "type": "annotation_exchange", "user": user, "role": role, "seed": store.seed(db),
            "cohort": fingerprint(db), "screenings": screenings}
    rows = [head]
    for row in db.execute("""SELECT work.doc,work.data,documents.raw_hash,documents.text_hash FROM work
                             JOIN documents ON documents.id=work.doc
                             WHERE user=? AND stage='final' ORDER BY work.doc""", (user,)):
        rows.append({"doc": row["doc"], "raw_hash": row["raw_hash"], "text_hash": row["text_hash"],
                     "data": json.loads(row["data"])})
    return "".join(store.dump(row) + "\n" for row in rows)


def _header(db, head, receiver):
    expected = {"version", "type", "user", "role", "seed", "cohort", "screenings"}
    if not isinstance(head, dict) or head.keys() != expected or type(head["version"]) is not int:
        raise ValueError("Invalid annotation exchange header")
    if head["version"] != VERSION or head["type"] != "annotation_exchange":
        raise ValueError("Unsupported annotation exchange format")
    role, author = head["role"], head["user"]
    if (not isinstance(role, str) or role not in {"A1", "A2"} or not isinstance(author, str)
            or not store.USER_ID.fullmatch(author) or author != store.user_for(db, role) or author == receiver):
        raise ValueError("The sender must be the registered other annotator")
    if type(head["seed"]) is not int or head["seed"] != store.seed(db):
        raise ValueError("The sender used a different assignment seed")
    if role == "A1":
        screenings = head["screenings"]
        known = {row["id"] for row in corpus(db)}
        if not isinstance(screenings, dict) or not known or screenings.keys() != known:
            raise ValueError("The inclusion packet must cover the same manifest")
        for doc, answer in screenings.items():
            answer = labels.screen(answer, doc, known)
            prior = db.execute("SELECT user,record FROM screening WHERE doc=?", (doc,)).fetchone()
            if prior is not None:
                if prior["user"] != author or json.loads(prior["record"]) != answer:
                    raise ValueError("A different inclusion decision already exists")
            else:
                if store.sealed(db):
                    raise ValueError("The inclusion cohort is already locked")
                db.execute("INSERT INTO screening VALUES (?,?,?)", (doc, author, store.dump(answer)))
                store.audit(db, doc, author, "screening_imported", after=answer, reason=answer["reason"])
        store.lock_cohort(db)
    elif head["screenings"] is not None:
        raise ValueError("Only A1 can transfer inclusion decisions")
    if not store.sealed(db) or head["cohort"] != fingerprint(db):
        raise ValueError("The locked cohort or document hashes differ. Import A1’s cohort first")
    return role, author


def _data(db, row, role):
    if not isinstance(row, dict) or set(row) != {"doc", "raw_hash", "text_hash", "data"}:
        raise ValueError("Invalid label record")
    doc = row["doc"]
    if not isinstance(doc, str) or not store.DOC_ID.fullmatch(doc):
        raise ValueError("Invalid transfer document ID")
    source = workflow.document(db, doc)
    assigned = workflow.assignment(db, doc, role)
    if not source or not assigned or (row["raw_hash"], row["text_hash"]) != (source["raw_hash"], source["text_hash"]):
        raise ValueError("Label document, assignment or hashes do not match")
    text = source["text"]
    data = row["data"]
    if not isinstance(data, dict) or set(data) != {"mode", "form", "submitted", "checks", "elapsed_ms", "changes"}:
        raise ValueError("Invalid final annotation snapshot")
    mode = data["mode"]
    if not isinstance(mode, str) or mode not in workflow.MODES or mode != workflow.selection_mode(
        assigned, role, mode == "prefill"
    ):
        raise ValueError("The label mode does not match this assignment")
    final = labels.form(data["form"], text, complete=True, prefill=mode == "prefill")
    original = labels.form(data["submitted"], text, complete=True, prefill=mode == "prefill")
    flags = {} if mode == "control" else checker.check(text, original["fields"])
    if original["reviewed"] or original["reasons"] or flags != data["checks"] or flags.keys() - set(final["reviewed"]):
        raise ValueError("Incomplete or inconsistent checker review")
    changed = workflow.compared_changes(original, final)
    if not flags and (changed or final["reviewed"] or final["reasons"]):
        raise ValueError("Changes require a submitted checker review")
    events = data["changes"]
    if not isinstance(events, dict) or events.keys() != changed.keys():
        raise ValueError("The change log does not match the final labels")
    for path, event in events.items():
        if not isinstance(event, dict) or set(event) != {"at", "reason"} or event["reason"] != changed[path]:
            raise ValueError("A logged field change needs its reason and time")
        stamp = datetime.fromisoformat(labels.string(event["at"], limit=48))
        if stamp.utcoffset() is None:
            raise ValueError("Change timestamps must include a timezone")
    labels.typed_fields(final["fields"])
    elapsed = workflow.milliseconds(data["elapsed_ms"], limit=2**53 - 1)
    return doc, text, {**data, "form": final, "submitted": original, "checks": flags, "elapsed_ms": elapsed}


def import_packet(db, content, receiver):
    try:
        rows = [json.loads(line) for line in content.split("\n") if line.strip()]
    except (TypeError, ValueError):
        raise ValueError("Choose a valid labels JSONL file") from None
    if not rows:
        raise ValueError("The labels file is empty")
    added, seen = 0, set()
    with store.transaction(db):
        role, author = _header(db, rows[0], receiver)
        for row in rows[1:]:
            doc, text, data = _data(db, row, role)
            if doc in seen:
                raise ValueError("The labels file repeats a document")
            seen.add(doc)
            prior = store.get_work(db, doc, author)
            if prior:
                if prior["stage"] != "final" or prior["data"] != data:
                    raise ValueError("A different annotation already exists. Existing labels were kept")
                continue
            db.execute("INSERT INTO work VALUES (?,?,'final',0,?)", (doc, author, store.dump(data)))
            if data["mode"] == "prefill":
                workflow._record_prefill(db, doc, author, text, data["submitted"])
            workflow.make_change_log(db, doc, author, data["submitted"], data["form"], data["changes"])
            store.audit(db, doc, author, "labels_imported", after={"received_by": receiver})
            added += 1
    return added
