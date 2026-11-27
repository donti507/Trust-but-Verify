"""Read and export the immutable SQLite audit event stream."""
import csv
from io import StringIO
import json
import re


def audit_rows(db):
    return db.execute(
        "SELECT audit.*,users.name FROM audit LEFT JOIN users ON actor=users.id ORDER BY audit.id"
    )


def export(db, kind):
    rows = [dict(row) for row in audit_rows(db)]
    if kind == "json":
        for row in rows:
            row["detail"] = json.loads(row["detail"])
        return json.dumps(rows, ensure_ascii=False, indent=2) + "\n", "application/json"
    if kind != "csv":
        raise ValueError("Unsupported audit format")
    fields = ["id", "at", "actor", "name", "action", "trade_id", "field", "detail"]
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        safe = {key: "'" + value if isinstance(value, str) and re.match(r"\s*[=+@-]", value) else value
                for key, value in row.items()}
        writer.writerow(safe)
    return output.getvalue(), "text/csv"
