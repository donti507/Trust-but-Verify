"""Step 15: apply the cost-aware decision rule to calibrated risks.

    python scripts/decide.py --model open_weights --variant full_logistic
    python scripts/decide.py --model open_weights --variant full_logistic --log-lambda 1 --log-split adv_test

Reads cache/risk/<model>.parquet and configs/costs.yaml. Writes
results/decision_<model>.md (automation vs loss at several lambdas, field and
trade modes, per split, plus cost sensitivity) and a decision log
results/<run_id>/decisions.jsonl (doc_id, field, value, p_hat, c_bar, lambda,
action, rules fired, evidence span) for one lambda and split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SHOWN = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)


def summary_table(table, p_col, costs, title) -> list[str]:
    from trustswap.decide.policy import evaluate_lambda

    lines = [f"### {title}", "",
             "| lambda | fields automated | docs straight-through | mean doc loss (field) | mean doc loss (trade) | "
             "errors accepted (field) | review cost/doc (field) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for lam in SHOWN:
        f = evaluate_lambda(table, p_col, lam, costs, "field")
        t = evaluate_lambda(table, p_col, lam, costs, "trade")
        lines.append(f"| {lam:g} | {100 * f['automated']:.1f}% | {100 * t['automated']:.1f}% | "
                     f"{f['mean_doc_loss']:.4f} | {t['mean_doc_loss']:.4f} | "
                     f"{f['errors_accepted']}/{f['errors_total']} | {f['review_cost_per_doc']:.2f} |")
    return lines + [""]


def decision_log(table, p_col, costs, lam, cache: Path, model: str) -> list[dict]:
    from trustswap.decide.policy import escalate, field_costs

    c, r = field_costs(table, costs)
    esc = escalate(table[p_col].to_numpy(float), c, r, lam)
    rows, extract_cache, verify_cache = [], {}, {}
    for (_, row), c_bar, flag in zip(table.iterrows(), c, esc):
        doc = row["doc_id"]
        if doc not in extract_cache:
            path = cache / "extract" / model / f"{doc}.jsonl"
            extract_cache[doc] = json.loads(path.read_text().splitlines()[0]) if path.exists() else {}
            vpath = cache / "verify" / model / f"{doc}.json"
            verify_cache[doc] = json.loads(vpath.read_text()) if vpath.exists() else {}
        item = (extract_cache[doc].get("fields") or {}).get(row["field"]) or {}
        fired = [f"{rule}:{found[row['field']]['verdict']}" for rule, found in verify_cache[doc].get("rules", {}).items()
                 if found.get(row["field"], {}).get("verdict") in {"HARD", "SOFT"}]
        rows.append({"doc_id": doc, "field": row["field"], "value": item.get("value"),
                     "raw_value": item.get("raw_value"), "p_hat": float(row[p_col]), "c_bar": float(c_bar),
                     "lambda": float(lam), "action": "ESCALATE" if flag else "ACCEPT", "rules_fired": fired,
                     "evidence_span": item.get("span"), "e": int(row["e"])})
    return rows


def main(argv=None) -> int:
    import pandas as pd

    from trustswap.decide.policy import load_costs

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["frontier", "open_weights"])
    parser.add_argument("--variant", default="full_logistic")
    parser.add_argument("--costs", type=Path, default=ROOT / "configs" / "costs.yaml")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--log-lambda", type=float, default=1.0)
    parser.add_argument("--log-split", default="adv_test")
    args = parser.parse_args(argv)

    table = pd.read_parquet(args.cache / "risk" / f"{args.model}.parquet")
    table = table[table.split != "real_test"]
    p_col = f"p_{args.variant}"
    costs = load_costs(args.costs)
    lines = [f"# Decision rule ({args.model}, {args.variant})", "",
             f"ESCALATE if p_hat * c_bar > lambda * r. Costs: `{args.costs.name}` (status: {costs.status}); "
             f"B = {costs.B:g}. Field mode reviews escalated fields only; trade mode sends the whole "
             "document to review if any field is escalated. Mean doc loss is min(B, accepted error cost)/B.", ""]
    for split in ("dev", "syn_test", "adv_test"):
        part = table[table.split == split]
        if len(part):
            lines += summary_table(part, p_col, costs, f"{split} ({part.doc_id.nunique():,} documents)")
    adv = table[table.split == "adv_test"]
    if len(adv):
        lines += ["## Cost sensitivity (adv_test)", ""]
        for label, variant in (("costs x0.5", costs.scaled(0.5)), ("costs x2", costs.scaled(2.0)),
                               ("critical <-> major swapped", costs.reordered(("major", "critical", "minor")))):
            lines += summary_table(adv, p_col, variant, label)
    markdown = "\n".join(lines)
    (args.results).mkdir(parents=True, exist_ok=True)
    (args.results / f"decision_{args.model}.md").write_text(markdown, encoding="utf-8")

    log_part = table[table.split == args.log_split]
    run_id = f"{args.model}_{args.variant}_lambda{args.log_lambda:g}_{args.log_split}"
    out = args.results / run_id / "decisions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = decision_log(log_part, p_col, costs, args.log_lambda, args.cache, args.model)
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")
    print(markdown)
    print(f"\nwrote results/decision_{args.model}.md and {out} ({len(rows):,} decisions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
