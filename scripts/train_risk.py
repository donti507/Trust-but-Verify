"""Step 14: fit and calibrate the risk models for one extraction model.

    python scripts/train_risk.py --model open_weights

Reads cache/features/<model>.parquet; writes results/models/<model>/*.joblib,
results/models/<model>/metrics.json, results/risk_<model>.md and
cache/risk/<model>.parquet (calibrated P(error) per field for every model variant,
the input of Step 15).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    import joblib
    import pandas as pd

    from trustswap.score.risk import evaluate, fit_all, report

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["frontier", "open_weights"])
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    table = pd.read_parquet(args.cache / "features" / f"{args.model}.parquet")
    table = table[table.split != "real_test"]  # never touched before pre-registration
    models = fit_all(table)
    results = evaluate(models, table)
    out = args.results / "models" / args.model
    out.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, out / f"{name}.joblib")
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown = report(results, args.model, models)
    (args.results / f"risk_{args.model}.md").write_text(markdown, encoding="utf-8")

    predictions = table[["doc_id", "field", "model", "collection", "split", "defect", "is_control", "e"]].copy()
    for name, model in models.items():
        predictions[f"p_{name}"] = model.predict(table)
    (args.cache / "risk").mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.cache / "risk" / f"{args.model}.parquet", index=False)
    print(markdown)
    print(f"wrote {out} and {args.cache / 'risk' / f'{args.model}.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
