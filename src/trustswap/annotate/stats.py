"""Human agreement on the completed seeded overlap and workflow statistics."""
from collections import defaultdict
import json

from trustswap.annotate import adjudicate, labels, store
from trustswap.common.agreement import cohen_kappa
from trustswap.extract.fields import PATHS


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def field_agreement(db, mode=None):
    pairs = defaultdict(list)
    overlap = 0
    for row in db.execute("SELECT doc FROM cohort WHERE overlap=1 ORDER BY doc"):
        first, second = (adjudicate.finalized(db, row["doc"], role) for role in ("A1", "A2"))
        if not first or not second or (mode is not None and first["data"]["mode"] != mode):
            continue
        overlap += 1
        left, right = (labels.typed_fields(work["data"]["form"]["fields"]) for work in (first, second))
        for path in left.keys() & right.keys():
            pairs[path].append((labels.identity(path, left[path]), labels.identity(path, right[path])))
    summary = []
    for path in PATHS:
        fields = pairs[path]
        summary.append({"field": path, "paired": len(fields),
                        "agreement": ratio(sum(a == b for a, b in fields), len(fields)),
                        "kappa": cohen_kappa([a[0] for a, _ in fields], [b[0] for _, b in fields])})
    return overlap, summary


def progress(db, role):
    user = store.user_for(db, role)
    assigned = db.execute("SELECT count(*) FROM cohort" + (" WHERE overlap=1" if role == "A2" else "")).fetchone()[0]
    screened = db.execute("SELECT count(*) FROM screening").fetchone()[0] if role == "A1" else None
    rows = list(db.execute("SELECT stage,data FROM work WHERE user=?", (user,))) if user else []
    final = 0
    active_ms = 0
    checked = flagged = checked_after_review = changed_after_flag = 0
    for row in rows:
        data = json.loads(row["data"])
        if data["submitted"] is not None and data["mode"] != "control":
            checked += len(data["submitted"]["fields"])
            flagged += len(data["checks"])
        if row["stage"] == "final":
            final += 1
            active_ms += data["elapsed_ms"]
            checked_after_review += len(data["checks"])
            changed_after_flag += len(data["checks"].keys() & data["changes"].keys())
    return {"role": role, "assigned": assigned, "screened": screened, "started": len(rows), "final": final,
            "flagged": ratio(flagged, checked), "changed": ratio(changed_after_flag, checked_after_review),
            "minutes": ratio(active_ms / 60_000, final)}
