"""Human screening, label submissions and re-check transitions."""
from datetime import datetime, timezone
import json

from trustswap.annotate import checker, labels, store

MODES = {"control", "blind", "prefill"}


def document(db, doc):
    return db.execute("SELECT * FROM documents WHERE id=?", (doc,)).fetchone()


def assignment(db, doc, role):
    row = db.execute("SELECT * FROM cohort WHERE doc=?", (doc,)).fetchone()
    if row is None or role not in {"A1", "A2"} or (role == "A2" and not row["overlap"]):
        return None
    return row


def selection_mode(assigned, role, prefill):
    if role == "A1" and assigned["control"]:
        return "control"
    return "prefill" if role == "A1" and prefill else "blind"


def screen_document(db, doc, user, form):
    if store.sealed(db):
        raise ValueError("The inclusion cohort is already locked")
    known = {row[0] for row in db.execute("SELECT id FROM documents")}
    if doc not in known or user != store.user_for(db, "A1"):
        raise ValueError("This document is outside your inclusion queue")
    record = labels.screen(form, doc, known)
    prior = db.execute("SELECT record FROM screening WHERE doc=?", (doc,)).fetchone()
    db.execute("""INSERT INTO screening VALUES (?,?,?)
                  ON CONFLICT(doc) DO UPDATE SET record=excluded.record""", (doc, user, store.dump(record)))
    store.audit(db, doc, user, "screened", before=json.loads(prior[0]) if prior else None,
                after=record, reason=record["reason"])
    store.lock_cohort(db)


def start(db, doc, user, assigned, role, prefill):
    mode = selection_mode(assigned, role, prefill)
    data = {"mode": mode, "form": labels.empty_form(), "submitted": None, "checks": {}, "elapsed_ms": 0, "changes": {}}
    db.execute("INSERT INTO work VALUES (?,?,'draft',0,?) ON CONFLICT(doc,user) DO NOTHING", (doc, user, store.dump(data)))


def milliseconds(delta, *, limit=300_000):
    if type(delta) is not int or not 0 <= delta <= limit:
        raise ValueError("Invalid annotation time")
    return delta


def compared_changes(original, final):
    if any(final[key] != original[key] for key in ("structure", "flags", "flags_reviewed")):
        raise ValueError("The structure and document characteristics are locked after submission")
    changes = {}
    for path, before in original["fields"].items():
        after = final["fields"][path]
        if labels.raw_identity(before) != labels.raw_identity(after):
            reason = final["reasons"].get(path)
            if not reason:
                raise ValueError(f"Enter a reason for changing {labels.label(path)}")
            changes[path] = reason
    return changes


def _record_prefill(db, doc, user, text, submitted):
    for path, answer in submitted["fields"].items():
        candidate = checker.suggestion(path, text)
        chosen = labels.raw_identity(answer)
        if candidate is None:
            store.audit(db, doc, user, "prefill_unavailable", field=path, after=chosen)
            continue
        same = all(chosen[key] == candidate[key] for key in ("status", "value", "span"))
        store.audit(db, doc, user, "prefill_accepted" if same else "prefill_changed",
                    field=path, before=candidate, after=chosen)


def make_change_log(db, doc, user, original, final, changes):
    for path, event in changes.items():
        store.audit(db, doc, user, "field_changed", field=path, before=labels.raw_identity(original["fields"][path]),
                    after=labels.raw_identity(final["fields"][path]), reason=event["reason"], at=event["at"])


def commit_work(db, doc, user, text, current, *, form, revision, delta, submit):
    if not current or current["stage"] == "final":
        raise ValueError("This annotation has no editable draft")
    if type(revision) is not int or revision != current["revision"]:
        raise RevisionError("A newer draft is already saved. Reload to resume it")
    stage, data = current["stage"], current["data"]
    final = labels.form(form, text, complete=submit, prefill=data["mode"] == "prefill")
    if stage == "review" and any(final[key] != data["submitted"][key]
                                 for key in ("structure", "flags", "flags_reviewed")):
        raise ValueError("The structure and document characteristics are locked after submission")
    data["elapsed_ms"] += milliseconds(delta)
    if submit and stage == "draft":
        final["reviewed"], final["reasons"] = [], {}
        data["submitted"] = final
        data["checks"] = {} if data["mode"] == "control" else checker.check(text, final["fields"])
        stage = "review" if data["checks"] else "final"
        if stage == "final":
            labels.typed_fields(final["fields"])
        if data["mode"] == "prefill":
            _record_prefill(db, doc, user, text, final)
        store.audit(db, doc, user, "submitted", after={"mode": data["mode"], "flagged": sorted(data["checks"])})
    elif submit:
        if data["checks"].keys() - set(final["reviewed"]):
            raise ValueError("Confirm that you re-checked every flagged field")
        changes = compared_changes(data["submitted"], final)
        labels.typed_fields(final["fields"])
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        data["changes"] = {path: {"reason": reason, "at": now} for path, reason in changes.items()}
        make_change_log(db, doc, user, data["submitted"], final, data["changes"])
        store.audit(db, doc, user, "reviewed", after={"rechecked": final["reviewed"]})
        stage = "final"
    data["form"] = final
    next_revision = revision + 1
    db.execute("UPDATE work SET stage=?,revision=?,data=? WHERE doc=? AND user=?",
               (stage, next_revision, store.dump(data), doc, user))
    return next_revision, stage


class RevisionError(ValueError):
    """Another tab saved a newer snapshot."""
