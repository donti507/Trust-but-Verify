#!/usr/bin/env python3
"""Compare independent human annotations; disagreements are never auto-resolved."""
import argparse
from collections import Counter
import json
from pathlib import Path

from trustswap.common.annotations import FIELDS, canonical_value
if __package__:
    from .pilot_utils import read_annotations, write_csv, read_csv
else:
    from pilot_utils import read_annotations, write_csv, read_csv

ADJ_COLUMNS = ("doc_id", "field", "reason", "A1_status", "A1_value", "A1_span", "A2_status", "A2_value", "A2_span", "text_path")
METRIC_COLUMNS = ("field", "paired", "unpaired", "value_exact_matches", "value_exact_rate", "status_kappa", "span_exact_matches", "field_exact_matches")


def cohen_kappa(left, right):
    if len(left) != len(right):
        raise ValueError("status pairs must have equal lengths")
    count = len(left)
    if not count:
        return None
    a, b = Counter(left), Counter(right)
    observed = sum(x == y for x, y in zip(left, right)) / count
    expected = sum(a[status] * b[status] for status in a.keys() | b.keys()) / count ** 2
    return None if expected == 1 else (observed - expected) / (1 - expected)


def _disagreement(doc_id, field, reasons, a, b, text_path):
    def cell(item, key):
        if item is None:
            return "unannotated" if key == "status" else ""
        return json.dumps(item[key], ensure_ascii=False) if key != "status" else item[key]
    return {"doc_id": doc_id, "field": field, "reason": ";".join(reasons),
            **{f"{name}_{key}": cell(item, key) for name, item in (("A1", a), ("A2", b))
               for key in ("status", "value", "span")}, "text_path": text_path}


def compare(left, right):
    metrics, disagreements = [], []
    ids = sorted(left.keys() | right.keys())
    mismatches = {doc_id for doc_id in ids if doc_id in left and doc_id in right
                  and left[doc_id].text_sha256 != right[doc_id].text_sha256}
    for doc_id in sorted(mismatches):
        a, b = left[doc_id], right[doc_id]
        disagreements.append(_disagreement(doc_id, "__text__", ["source_version"],
            {"status": "version", "value": a.text_sha256, "span": None},
            {"status": "version", "value": b.text_sha256, "span": None}, a.text_path))
    for field in FIELDS:
        states_a, states_b, value_matches, span_matches, full_matches, unpaired = [], [], 0, 0, 0, 0
        for doc_id in ids:
            if doc_id in mismatches:
                continue
            a_record, b_record = left.get(doc_id), right.get(doc_id)
            a = a_record.fields.get(field) if a_record else None
            b = b_record.fields.get(field) if b_record else None
            if a is None and b is None:
                continue
            reasons = []
            if a is None or b is None:
                unpaired += 1
                reasons.append("missing_A1" if a is None else "missing_A2")
            else:
                states_a.append(a["status"])
                states_b.append(b["status"])
                same_value = canonical_value(field, a) == canonical_value(field, b)
                same_span = a["span"] == b["span"]
                value_matches += same_value
                span_matches += same_span
                full_matches += same_value and same_span and a["status"] == b["status"]
                if not same_value:
                    reasons.append("value")
                if a["status"] != b["status"]:
                    reasons.append("status")
                if not same_span:
                    reasons.append("span")
            if reasons:
                text_path = (a_record or b_record).text_path
                disagreements.append(_disagreement(doc_id, field, reasons, a, b, text_path))
        paired = len(states_a)
        metrics.append({"field": field, "paired": paired, "unpaired": unpaired,
                        "value_exact_matches": value_matches, "value_exact_rate": value_matches / paired if paired else None,
                        "status_kappa": cohen_kappa(states_a, states_b), "span_exact_matches": span_matches,
                        "field_exact_matches": full_matches})
    for doc_id in ids:
        if doc_id in mismatches:
            continue
        a, b = left.get(doc_id), right.get(doc_id)
        a_flags = a.flags if a and a.flags_reviewed else None
        b_flags = b.flags if b and b.flags_reviewed else None
        if a_flags != b_flags:
            disagreements.append(_disagreement(doc_id, "flags", ["flags"],
                {"status": "reviewed", "value": a_flags, "span": None} if a_flags is not None else None,
                {"status": "reviewed", "value": b_flags, "span": None} if b_flags is not None else None,
                (a or b).text_path))
    return metrics, disagreements


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1", type=Path, default=Path("data/real/annotations/A1.jsonl"))
    parser.add_argument("--a2", type=Path, default=Path("data/real/annotations/A2.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/real/adjudication.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("data/real/agreement.csv"))
    parser.add_argument("--pilot", type=Path, default=Path("data/real/pilot20.csv"), help="limit reports to current non-rejected pilot when present")
    parser.add_argument("--overwrite", action="store_true", help="replace previous generated reports; preserve lead edits first")
    args = parser.parse_args(argv)
    try:
        if not args.a1.is_file() or not args.a2.is_file():
            raise ValueError("both annotation files are required")
        left, _ = read_annotations(args.a1)
        right, _ = read_annotations(args.a2)
        if args.pilot.exists():
            _, pilot = read_csv(args.pilot)
            active = {row["doc_id"] for row in pilot if not row.get("rejected_reason", "").strip()
                      and row.get("split") != "real_test"}
            if not active:
                raise ValueError("pilot has no active documents; run select_pilot.py --top-up")
            left = {key: record for key, record in left.items() if key in active}
            right = {key: record for key, record in right.items() if key in active}
        if left and right and {record.annotator for record in left.values()} & {record.annotator for record in right.values()}:
            raise ValueError("agreement requires independent annotator identities")
        if not args.overwrite and (args.output.exists() or args.metrics.exists()):
            raise ValueError("report already exists; preserve domain-lead work before using --overwrite")
        metrics, disagreements = compare(left, right)
        write_csv(args.output, ADJ_COLUMNS, disagreements)
        write_csv(args.metrics, METRIC_COLUMNS, metrics)
        for row in metrics:
            print(f"{row['field']}: paired={row['paired']}, value_exact={row['value_exact_rate']}, status_kappa={row['status_kappa']}")
        print(f"Disagreements for domain lead: {len(disagreements)}")
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
