"""A separate human adjudication step for independently finalized labels."""
import json

from trustswap.annotate import labels, store, workflow
from trustswap.common.records import GoldRecord

STRUCTURE = "__structure__"
FLAGS = "__flags__"


def finalized(db, doc, role):
    user = store.user_for(db, role)
    work = store.get_work(db, doc, user) if user else None
    return work if work is not None and work["stage"] == "final" else None


def _decisions(db, doc):
    return {row["field"]: {"answer": json.loads(row["answer"]), "reason": row["reason"]} for row in
            db.execute("SELECT field,answer,reason FROM adjudication WHERE doc=?", (doc,))}


def comparison(db, doc):
    cohort = workflow.assignment(db, doc, "A2")
    if cohort is None:
        return None
    first, second = (finalized(db, doc, role) for role in ("A1", "A2"))
    if not first or not second:
        return None
    left, right = (work["data"]["form"] for work in (first, second))
    decisions = _decisions(db, doc)
    answers = {path: decision["answer"] for path, decision in decisions.items()}
    a, b = labels.typed_fields(left["fields"]), labels.typed_fields(right["fields"])
    differences = {}
    structure = left["structure"]
    if structure != right["structure"]:
        differences[STRUCTURE] = [structure, right["structure"]]
        # Shared term disputes can be settled immediately. Wait before showing
        # one-sided leg fields so discarded legs require no decision.
        structure = answers.get(STRUCTURE)
    if left["flags"] != right["flags"]:
        differences[FLAGS] = [left["flags"], right["flags"]]
    active = labels.paths(structure) if structure else [path for path in a if path in b]
    for path in active:
        if path not in a or path not in b or labels.alternative(a[path], b[path], path):
            differences[path] = [a.get(path), b.get(path)]
    pending = [path for path in differences if path not in answers]
    return {"left": left, "right": right, "a": a, "b": b, "structure": structure,
            "differences": differences, "answers": answers, "decisions": decisions, "pending": pending}


def decide(db, doc, path, user, source, choice, reason, manual):
    review = comparison(db, doc)
    if review is None or path not in review["differences"]:
        raise ValueError("Choose a field that is in dispute")
    reason = labels.string(reason).strip()
    if not reason:
        raise ValueError("Enter a reason for the decision")
    options = review["differences"][path]
    if choice in ("A1", "A2"):
        answer = options[choice == "A2"]
        if answer is None:
            raise ValueError("That annotator did not label this field")
    elif choice == "manual" and path not in {STRUCTURE, FLAGS}:
        answer = labels.typed(path, labels.answer(manual, source, complete=True))
    else:
        raise ValueError("Choose an annotator or enter a manual term")
    before = review["answers"].get(path)
    db.execute("""INSERT INTO adjudication VALUES (?,?,?,?,?)
                  ON CONFLICT(doc,field) DO UPDATE SET answer=excluded.answer,reason=excluded.reason""",
               (doc, path, user, store.dump(answer), reason))
    store.audit(db, doc, user, "adjudicated", field=path, before=before if before is not None else options,
                after=answer, reason=reason)


def pending_documents(db):
    output = []
    for row in db.execute("SELECT doc FROM cohort WHERE overlap=1 ORDER BY doc"):
        doc = row["doc"]
        review = comparison(db, doc)
        if review and review["pending"]:
            output.append((doc, len(review["pending"])))
    return output


def gold(db):
    for row in db.execute("""SELECT documents.*,cohort.overlap FROM documents
                             JOIN cohort ON cohort.doc=documents.id ORDER BY id"""):
        doc = row["id"]
        first = finalized(db, doc, "A1")
        if first is None:
            continue
        annotators = [first["user"]]
        form = first["data"]["form"]
        structure = form["structure"]
        terms = labels.typed_fields(form["fields"])
        adjudicated = False
        if row["overlap"]:
            second = finalized(db, doc, "A2")
            if not second:
                continue
            review = comparison(db, doc)
            if review["pending"]:
                continue
            annotators.append(second["user"])
            adjudicated = bool(review["differences"])
            if adjudicated:
                annotator = store.user_for(db, "ADJ")
                if annotator:
                    annotators.append(annotator)
            structure = review["structure"]
            terms = {path: review["answers"].get(path, terms.get(path)) for path in labels.paths(structure)}
        yield GoldRecord(doc_id=doc, collection="real", trade=labels.trade(structure, terms),
                         annotators=annotators, adjudicated=adjudicated,
                         sha256=row["raw_hash"], text_sha256=row["text_hash"], source_url=row["url"])
