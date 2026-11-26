"""Step 16: simulate conformal risk control on the synthetic test split.

    python scripts/crc.py --model open_weights --variant full_logistic --mode field
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (0.01, 0.02, 0.05, 0.1)


def main(argv=None) -> int:
    import pandas as pd

    from trustswap.decide.crc import simulate
    from trustswap.decide.policy import COSTS_FILE, load_costs

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=("frontier", "open_weights"))
    parser.add_argument("--variant", default="full_logistic")
    parser.add_argument("--mode", required=True, choices=("field", "trade"))
    parser.add_argument("--costs", type=Path, default=COSTS_FILE)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    p_col = f"p_{args.variant}"
    table = pd.read_parquet(
        args.cache / "risk" / f"{args.model}.parquet",
        engine="pyarrow",
        columns=["doc_id", "field", "split", "e", p_col],
        filters=[("split", "==", "syn_test")],
    )
    costs = load_costs(args.costs)
    automation_label = "Fields accepted" if args.mode == "field" else "Documents straight-through"
    lines = [
        f"# Conformal risk control ({args.model}, {args.variant}, {args.mode} mode)",
        "",
        "Synthetic results are a code check only. The claimed guarantee uses real_cal documents in Step 21.",
        "",
        "Alpha below the floor cannot be certified. Uncertified splits use lambda = 0. "
        "Pass means the observed mean test loss is at most alpha.",
        "",
        f"| Alpha | Floor | Mean lambda_hat | Mean test loss ± s.e. | {automation_label} | Certified splits | Check |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for alpha in ALPHAS:
        result = simulate(table, p_col, costs, args.mode, alpha)
        passed = result["mean_test_loss"] <= alpha
        lines.append(
            f"| {alpha:g} | {result['floor']:.6g} | {result['mean_lambda_hat']:.6g} | "
            f"{result['mean_test_loss']:.6g} ± {result['standard_error']:.6g} | "
            f"{result['mean_automation']:.2%} | {result['certified_fraction']:.2%} | "
            f"{'Pass' if passed else 'Fail'} |"
        )

    args.results.mkdir(parents=True, exist_ok=True)
    report = args.results / f"crc_{args.model}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
