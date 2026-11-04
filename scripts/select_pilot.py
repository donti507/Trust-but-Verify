#!/usr/bin/env python3
"""Select a reproducible fixed/float pilot; no raw text or labels are inferred."""
import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path
from random import Random

if __package__:
    from .pilot_utils import read_csv, write_csv, rejection_history, archive_rejections
    from .review_helper import _decision, show_document, _safe
else:
    from pilot_utils import read_csv, write_csv, rejection_history, archive_rejections
    from review_helper import _decision, show_document, _safe

ERAS = ("2001–2005", "2006–2008", "2009–2021", "2022+")
SEED = 20261026


def select_pilot(screen, manual, *, seed=SEED, existing=(), excluded=(), confirm=None):
    existing = list(existing)
    if len(existing) > 20 or len({row["doc_id"] for row in existing}) != len(existing):
        raise ValueError("active pilot must have at most 20 unique documents")
    existing_ids = {row["doc_id"] for row in existing}
    existing_urls = {row["document_url"] for row in existing}
    excluded_ids = {row.get("doc_id") for row in excluded}
    excluded_urls = {row.get("document_url") for row in excluded}
    reviews = {}
    for row in manual:
        key = row["document_url"]
        if key in reviews:
            raise ValueError("duplicate URL in manual review CSV")
        reviews[key] = row
    buckets = {era: {True: [], False: []} for era in ERAS}
    seen_urls, seen_ids = set(), set()
    for source in sorted(screen, key=lambda row: row["document_url"]):
        if source.get("split") == "real_test" or source.get("collection", "real") != "real":
            continue
        if (source["doc_id"] in existing_ids | excluded_ids or source["document_url"] in existing_urls | excluded_urls
                or source.get("rejected_reason", "").strip()):
            continue
        if source["document_url"] in seen_urls or source["doc_id"] in seen_ids:
            raise ValueError("duplicate screen document")
        seen_urls.add(source["document_url"])
        seen_ids.add(source["doc_id"])
        review = reviews.get(source["document_url"], {})
        decision = _decision(review.get("is_confirmation", ""))
        product = review.get("product_type", "")
        preferred = decision == "y" and product == "irs_fixed_float"
        if decision == "n" or product in {"basis", "cap_floor", "other"}:
            continue
        if not preferred and source.get("label") != "likely_confirmation":
            continue
        era = source.get("era", "").replace("-", "–")
        if era not in buckets:
            raise ValueError("unknown pilot era")
        if not source.get("text_path"):
            continue
        row = {**source, "is_confirmation": review.get("is_confirmation", ""), "product_type": product,
               "selection_reason": "manual_confirmation" if preferred else "heuristic_topup",
               "not_manually_checked": str(not bool(decision)).lower(),
               "manual_check_required": str(not preferred).lower()}
        buckets[era][preferred].append(row)
    rng = Random(seed)
    pools = {}
    for era in ERAS:
        for preferred in (True, False):
            rng.shuffle(buckets[era][preferred])
        pools[era] = buckets[era][True] + buckets[era][False]
    selected = list(existing)
    counts = Counter(row["era"] for row in existing)
    if counts["2022+"] > 2:
        raise ValueError("existing pilot exceeds the recent-era cap")
    quotas = dict(zip(ERAS, (6, 6, 6, 2)))
    while len(selected) < 20:
        available = [era for era in ERAS if pools[era] and (era != "2022+" or counts[era] < 2)]
        if not available:
            raise ValueError("fewer than 20 eligible accepted documents after exclusions and the 2022+ cap")
        era = min(available, key=lambda era: (-(quotas[era] - counts[era]), ERAS.index(era)))
        row = pools[era].pop(0)
        if row["manual_check_required"] == "true" and confirm is not None:
            if not confirm(row):
                continue
            row = {**row, "is_confirmation": "y", "product_type": "irs_fixed_float",
                   "manual_check_required": "false", "not_manually_checked": "false",
                   "selection_reason": "interactive_confirmation"}
        selected.append(row)
        counts[era] += 1
    if not existing:
        rng.shuffle(selected)
    return selected


def preview_confirmation(row, csv_path, *, input_fn=None, output=None):
    input_fn = input if input_fn is None else input_fn
    output = sys.stdout if output is None else output
    try:
        show_document(row, csv_path, output=output, color=False, head_lines=40)
    except (OSError, ValueError) as error:
        print(f"Cannot preview; not added: {_safe(error)}", file=output)
        return False
    while True:
        print("Suitable fixed/float confirmation for the pilot? [y/n]: ", end="", file=output, flush=True)
        answer = input_fn("").strip().lower()
        if answer in {"y", "n"}:
            return answer == "y"
        print("Enter y or n; nothing is added without y.", file=output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, default=Path("data/real/screen.csv"))
    parser.add_argument("--manual", type=Path, default=Path("data/real/manual_check.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/real/pilot20.csv"))
    parser.add_argument("--top-up", action="store_true", help="replace rejected rows, retaining active pilot rows")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    try:
        columns, screen = read_csv(args.screen)
        _, manual = read_csv(args.manual)
        history = rejection_history(args.output)
        old_columns, old_rows, expected = [], [], None
        rejected, active = [], []
        if args.top_up:
            expected = hashlib.sha256(args.output.read_bytes()).hexdigest()
            old_columns, old_rows = read_csv(args.output)
            rejected = [row for row in old_rows if row.get("rejected_reason", "").strip()]
            active = [row for row in old_rows if not row.get("rejected_reason", "").strip()]
            if not rejected and len(active) == 20:
                print("No rejected documents; pilot is unchanged.")
                return 0
        elif args.output.exists():
            raise ValueError("pilot already exists; use --top-up for rejected documents")
        selected = select_pilot(screen, manual, seed=args.seed, existing=active, excluded=[*history, *rejected],
                                confirm=lambda row: preview_confirmation(row, args.screen))
        if args.top_up:
            active_ids = {row["doc_id"] for row in active}
            new = iter(row for row in selected if row["doc_id"] not in active_ids)
            ordered = [next(new) if row.get("rejected_reason", "").strip() else row for row in old_rows]
            selected = [*ordered, *new]
        extras = ("is_confirmation", "product_type", "selection_reason", "not_manually_checked", "manual_check_required", "rejected_reason")
        columns = list(dict.fromkeys([*old_columns, *columns, *extras]))
        rows = [{name: row.get(name, "") for name in columns} for row in selected]
        if rejected:
            if hashlib.sha256(args.output.read_bytes()).hexdigest() != expected:
                raise RuntimeError("pilot changed during top-up; not replacing it")
            archive_rejections(args.output, rejected)
        write_csv(args.output, columns, rows, expected=expected, check=True)
        print(f"Selected {len(rows)} documents: {dict(Counter(row['era'] for row in rows))}")
        print(f"Manual confirmation still required: {sum(row['manual_check_required'] == 'true' for row in rows)}")
    except (EOFError, KeyboardInterrupt):
        print("Selection stopped; pilot was not changed.")
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
