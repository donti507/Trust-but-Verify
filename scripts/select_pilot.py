#!/usr/bin/env python3
"""Select a reproducible fixed/float pilot; no raw text or labels are inferred."""
import argparse
from collections import Counter
from pathlib import Path
from random import Random

if __package__:
    from .pilot_utils import read_csv, write_csv
    from .review_helper import _decision
else:
    from pilot_utils import read_csv, write_csv
    from review_helper import _decision

ERAS = ("2001–2005", "2006–2008", "2009–2021", "2022+")
SEED = 20261026


def select_pilot(screen, manual, *, seed=SEED):
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
    selected, counts = [], Counter()
    for era, quota in zip(ERAS, (6, 6, 6, 2)):
        selected.extend(pools[era][:quota])
        counts[era] += min(quota, len(pools[era]))
        pools[era] = pools[era][quota:]
    while len(selected) < 20:
        available = [era for era in ERAS if pools[era] and (era != "2022+" or counts[era] < 2)]
        if not available:
            raise ValueError("fewer than 20 eligible documents after exclusions and the 2022+ cap")
        era = min(available, key=lambda era: (counts[era], ERAS.index(era)))
        selected.append(pools[era].pop(0))
        counts[era] += 1
    rng.shuffle(selected)
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, default=Path("data/real/screen.csv"))
    parser.add_argument("--manual", type=Path, default=Path("data/real/manual_check.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/real/pilot20.csv"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    try:
        columns, screen = read_csv(args.screen)
        _, manual = read_csv(args.manual)
        selected = select_pilot(screen, manual, seed=args.seed)
        extras = ("is_confirmation", "product_type", "selection_reason", "not_manually_checked", "manual_check_required")
        columns = list(dict.fromkeys([*columns, *extras]))
        rows = [{name: row.get(name, "") for name in columns} for row in selected]
        write_csv(args.output, columns, rows, expected=None, check=True)
        print(f"Selected {len(rows)} documents: {dict(Counter(row['era'] for row in rows))}")
        print(f"Manual confirmation still required: {sum(row['manual_check_required'] == 'true' for row in rows)}")
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
