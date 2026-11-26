"""Step 18: compare abstention baselines using cached features and risks.

    python scripts/baselines.py --model open_weights
"""
from __future__ import annotations

import argparse
from math import isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("dev", "syn_test", "adv_test")
KEYS = ["doc_id", "field"]
PROBABILITIES = ["p_base_logistic", "p_full_logistic"]


def _number(value: float) -> str:
    return f"{value:.6g}" if isfinite(value) else "—"


def _report_split(split, table, results):
    from trustswap.eval.baselines import FIXED_METHODS, aurc

    lines = [
        f"## {split} ({table['doc_id'].nunique():,} documents, {len(table):,} fields)",
        "",
        "| Method | AURC | Cost-weighted AURC | Point coverage | Point risk | Point cost risk | "
        "CRC doc loss | CRC floor | Certified splits |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for method, result in results.items():
        curve, point = result["curve"], result["crc"]
        if method in FIXED_METHODS:
            row = curve.iloc[0]
            values = ["—", "—", f"{row.coverage:.2%}",
                      _number(row.risk), _number(row.cost_risk), "—", "—", "—"]
        else:
            values = [f"{aurc(curve):.6g}", f"{aurc(curve, weighted=True):.6g}"]
            if point is None:
                values += ["—"] * 6
            else:
                values += [f"{point['coverage']:.2%}", _number(point["risk"]), _number(point["cost_risk"]),
                           f"{point['mean_doc_loss']:.6g}",
                           f"{point['floor']:.6g}", f"{point['certified_fraction']:.2%}"]
        lines.append(f"| {method} | " + " | ".join(values) + " |")
    lines.append("")
    skipped = len(table) - results.get("self_consistency", {}).get("n_fields", 0)
    if skipped:
        lines += [f"Self-consistency skips {skipped:,} fields with k_samples = 1.", ""]
    if table["doc_id"].nunique() < 2 or ("self_consistency" in results and results["self_consistency"]["crc"] is None):
        lines += ["CRC requires at least two eligible documents per method.", ""]
    return lines


def main(argv=None) -> int:
    import pandas as pd

    from trustswap.decide.policy import COSTS_FILE, load_costs
    from trustswap.eval.baselines import CURVE_COLUMNS, evaluate_baselines

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=("frontier", "open_weights"))
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--costs", type=Path, default=COSTS_FILE)
    args = parser.parse_args(argv)

    costs = load_costs(args.costs)
    filename = f"{args.model}.parquet"
    filters = [("split", "in", SPLITS)]
    features = pd.read_parquet(args.cache / "features" / filename, engine="pyarrow", filters=filters)
    risks = pd.read_parquet(args.cache / "risk" / filename, engine="pyarrow",
                            filters=filters, columns=KEYS + ["split"] + PROBABILITIES)
    lines = [
        f"# Baselines ({args.model})",
        "",
        "Curves accept tied scores together. AURC uses the risk at the left endpoint between thresholds "
        "and extends the first observed risk to zero coverage.",
        "",
        "CRC at alpha = 0.05 averages held-out coverage and normalised document loss over 1,000 "
        "random document half-splits within each split. Each calibration half supplies 201 quantile thresholds. "
        "Point risks pool accepted held-out fields. Uncertified splits escalate every field.",
        "",
    ]
    curves = {}
    for split in SPLITS:
        feature_rows = features.loc[features["split"] == split]
        risk_rows = risks.loc[risks["split"] == split, KEYS + PROBABILITIES]
        table = feature_rows.merge(risk_rows, on=KEYS, validate="one_to_one")
        if len(table) != len(feature_rows) or len(table) != len(risk_rows):
            raise ValueError(f"Feature and risk cache keys differ for {split}")
        if table.empty:
            lines += [f"## {split}", "", "No cached rows.", ""]
            curves[split] = pd.DataFrame(columns=["method", *CURVE_COLUMNS])
            continue
        if "collection" in table and table["collection"].nunique(dropna=False) > 1:
            raise ValueError(f"{split} contains multiple collections")
        results = evaluate_baselines(table, costs)
        lines.extend(_report_split(split, table, results))
        curves[split] = pd.concat(
            [result["curve"].assign(method=method) for method, result in results.items()],
            ignore_index=True,
        )[["method", *CURVE_COLUMNS]]

    curve_dir = args.results / "curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    for split, curve in curves.items():
        curve.to_csv(curve_dir / f"{args.model}_{split}.csv", index=False)
    report = args.results / f"baselines_{args.model}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report} and curves in {curve_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
