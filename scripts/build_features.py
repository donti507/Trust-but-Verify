"""Step 13: build the feature table for one model.

    python scripts/build_features.py --model open_weights
    python scripts/build_features.py --model frontier

Reads data/{synthetic,adversarial}/ (docs + gold), cache/extract/<model>/ and
cache/verify/<model>/, and the other model's k = 0 outputs for cross-model
agreement. Writes cache/features/<model>.parquet and results/features_<model>.md.
Real documents are added later (real_cal only); real_test is never read.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from trustswap.extract.evaluate import load_gold, load_records
from trustswap.score.features import build_table, label_report

if __package__:
    from .full_extraction import all_docs
else:
    from full_extraction import all_docs

ROOT = Path(__file__).resolve().parents[1]
OTHER = {"frontier": "open_weights", "open_weights": "frontier"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(OTHER))
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    docs = all_docs(args.data)
    gold = {}
    for collection in ("synthetic", "adversarial"):
        path = args.data / collection / "gold.jsonl"
        if path.exists():
            gold.update(load_gold(path))
    records = load_records(args.cache / "extract" / args.model)
    other = load_records(args.cache / "extract" / OTHER[args.model])
    table = build_table(docs, gold, records, args.cache / "verify" / args.model, other)
    if table.isna().any().any():
        raise SystemExit("feature table has missing values")
    out = args.cache / "features" / f"{args.model}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    report = label_report(table, args.model)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / f"features_{args.model}.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
