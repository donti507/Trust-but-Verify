"""Step 22: evaluate the five hypotheses from cached labels and predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from trustswap.decide.policy import COSTS_FILE, load_costs
from trustswap.eval.hypotheses import (
    COVERAGES, CRC_ALPHA, aurc_difference, crc_accepted, critical_errors, likelihood_ratio,
    matched_automation, ranking_transfer, silent_shares, wilson,
)
from trustswap.extract.evaluate import load_gold
from trustswap.synth.render import trade_fields

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/hypotheses.py"
SPLITS = {"syn_test": "synthetic", "adv_test": "adversarial"}
P_COLS = ["p_base_logistic", "p_full_logistic"]
KEYS = ["doc_id", "field", "collection", "split"]
AUDIT_COLUMNS = ["doc_id", "field", "defect", "value", "gold", "rules", "rule_gap"]
WORDING = {
    "H1": "Added verifier features and predictive performance",
    "H2": "Cost-aware and cost-blind rankings at matched automation",
    "H3": "Mean automated loss in the real evaluation",
    "H4": "Verifier silence among accepted critical errors",
    "H5": "Agreement between method rankings across collections",
}


def registration(path: Path):
    if not path.is_file():
        return "draft, not pre-registered", {key: (text, 0.05) for key, text in WORDING.items()}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = data.get("hypotheses", data)
    if isinstance(entries, list):
        entries = {item["id"]: item for item in entries}
    records = {}
    for key in WORDING:
        record = entries[key]
        text = record.get("wording") or record.get("statement") or record.get("text")
        alpha = float(record["alpha"])
        if not text or not 0 < alpha < 1:
            raise ValueError(f"{key} requires wording and an alpha in (0, 1)")
        records[key] = (text, alpha)
    return data.get("status", "draft"), records


def load_tables(cache: Path, model: str, *, status: str, include_real: bool = False):
    if include_real and status != "frozen":
        raise ValueError("real_test requires frozen pre-registration")
    splits = {**SPLITS, **({"real_test": "real"} if include_real else {})}
    feature_path = cache / "features" / f"{model}.parquet"
    risk_path = cache / "risk" / f"{model}.parquet"
    filters = [[("collection", "==", collection), ("split", "==", split)] for split, collection in splits.items()]
    features = pd.read_parquet(
        feature_path, engine="pyarrow",
        filters=filters + [[("collection", "==", "synthetic"), ("split", "==", "train")]],
    )
    risks = pd.read_parquet(risk_path, engine="pyarrow", columns=KEYS + P_COLS, filters=filters)
    train = features.loc[features["split"].eq("train")]
    held_out = features.loc[features["split"].ne("train")]
    joined = held_out.merge(risks, on=KEYS, validate="one_to_one")
    if len(joined) != len(held_out) or len(joined) != len(risks):
        raise ValueError("Feature and risk cache keys differ for the evaluation splits")
    return train, {split: joined.loc[joined["split"].eq(split)] for split in splits}, feature_path, risk_path


def audit_errors(errors: pd.DataFrame, cache: Path, data: Path, model: str) -> pd.DataFrame:
    gold = load_gold(data / "adversarial" / "gold.jsonl") if len(errors) else {}
    rows = []
    for doc_id, fields in errors.groupby("doc_id", sort=True):
        extract_path = cache / "extract" / model / f"{doc_id}.jsonl"
        verify_path = cache / "verify" / model / f"{doc_id}.json"
        samples = (json.loads(line) for line in extract_path.read_text(encoding="utf-8").splitlines() if line.strip())
        sample = next(record for record in samples if record["k"] == 0)
        values = sample.get("fields", {}) if sample.get("schema_valid", True) else {}
        verifier = json.loads(verify_path.read_text(encoding="utf-8"))["rules"]
        gold_fields = trade_fields(gold[doc_id].trade)
        for field in fields.itertuples(index=False):
            fired = [f"{rule}:{checks[field.field]['verdict']}" for rule, checks in sorted(verifier.items())
                     if checks.get(field.field, {}).get("verdict") in {"HARD", "SOFT"}]
            rows.append((doc_id, field.field, field.defect, values.get(field.field, {}).get("value"),
                         gold_fields[field.field].value, fired, ""))
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def write_audit(audit: pd.DataFrame, path: Path):
    output = audit.copy()
    if path.is_file():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        human = {(row.doc_id, row.field): row.rule_gap for row in old.itertuples()
                 if row.rule_gap}
        if not human.keys() <= set(zip(output["doc_id"], output["field"])):
            raise ValueError(f"Existing human annotations in {path} would be lost")
        output["rule_gap"] = [human.get(key, "") for key in zip(output["doc_id"], output["field"])]
    output["rules"] = output["rules"].map(json.dumps)
    output.to_csv(path, index=False)


def _number(value):
    return f"{value:.6g}" if np.isfinite(value) else "unavailable"


def _ci(estimate, low, high):
    return f"{_number(estimate)} [{_number(low)}, {_number(high)}]"


def _source(*paths):
    return f"Source: `{SCRIPT}` using " + ", ".join(f"`{path}`" for path in paths) + "."


def _verdict(supported):
    if supported is None:
        return "insufficient data"
    return "supported" if supported else "not supported"


def _real_verdict(path, model, status, alpha):
    if status != "frozen" or not path.is_file():
        return "pending Step 21", []
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("models", payload).get(model, payload)
    result = result.get("field", result)
    loss = next((result[key] for key in ("mean_doc_loss", "mean_automated_loss", "mean_loss") if key in result), None)
    if loss is None:
        return "insufficient data", [_source(path), "The evaluation has no mean automated loss."]
    loss = float(loss)
    supported = (loss <= alpha and result.get("certified", True)) if np.isfinite(loss) else None
    return _verdict(supported), [_source(path), f"Mean automated loss: {_number(loss)}."]


def build_report(args):
    prereg = args.prereg
    status, hypotheses = registration(prereg)
    real_path = args.results / "real_eval.json"
    if args.include_real and (status != "frozen" or not real_path.is_file()):
        raise ValueError("Real comparisons require frozen pre-registration and a Step 21 evaluation")
    train, tables, feature_path, risk_path = load_tables(args.cache, args.model, status=status,
                                                        include_real=args.include_real)
    costs = load_costs(args.costs)
    alpha = {key: value[1] for key, value in hypotheses.items()}
    options = {"n_resamples": args.n_resamples, "seed": args.seed}
    analyses = {}

    lrt = likelihood_ratio(train)
    h1 = [
        _source(feature_path),
        f"Synthetic train: Λ = {_number(lrt['lambda'])}, k = {lrt['df']}, p = {_number(lrt['p'])}. "
        "Both fits use C = 1. The chi-squared reference is an approximation for these penalized fits "
        "and field-level observations.",
        _source(feature_path, risk_path, args.costs),
        "Full minus base cost-weighted AURC with document-bootstrap intervals.",
    ]
    for split in SPLITS:
        table = tables[split]
        if table.empty:
            h1.append(f"- {split}: unavailable.")
        else:
            difference = aurc_difference(table, costs, **options)
            h1.append(f"- {split}: {_ci(difference.estimate, *difference.interval())}.")
    h1_support = lrt["p"] < alpha["H1"] if np.isfinite(lrt["p"]) else None
    analyses["H1"] = (_verdict(h1_support), h1)

    variants = {
        "Default": costs, "Costs ×0.5": costs.scaled(0.5), "Costs ×2": costs.scaled(2),
        "Critical/major swapped": costs.reordered(("major", "critical", "minor")),
    }
    h2 = [
        _source(feature_path, risk_path, args.costs),
        "Losses are clipped at B per document. The differences are cost-aware minus cost-blind. "
        "The verdict uses the one-sided bootstrap upper bounds for the default costs.",
        "",
        "| Split | Costs | Target | Automation | Aware loss [CI] | Blind loss [CI] | Difference [CI] |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    main_comparisons = []
    for split in SPLITS:
        if tables[split].empty:
            continue
        for name, variant in variants.items():
            fit = matched_automation(tables[split], variant, **options)
            lo, hi = fit.interval()
            for i, ((coverage, aware, blind, diff), target) in enumerate(zip(fit.estimate, COVERAGES)):
                h2.append(
                    f"| {split} | {name} | {target:.0%} | {coverage:.2%} | {_ci(aware, lo[i, 1], hi[i, 1])} | "
                    f"{_ci(blind, lo[i, 2], hi[i, 2])} | {_ci(diff, lo[i, 3], hi[i, 3])} |"
                )
            if name == "Default":
                main_comparisons.append((fit.quantile(1 - alpha["H2"])[:, 3] < 0).all())
    if len(main_comparisons) < len(SPLITS):
        h2 += ["", "Matched comparisons require cached rows from both evaluation splits."]
    analyses["H2"] = (_verdict(all(main_comparisons) if len(main_comparisons) == len(SPLITS) else None), h2)
    analyses["H3"] = _real_verdict(real_path, args.model, status, alpha["H3"])

    adversarial = tables["adv_test"]
    if adversarial.empty:
        accepted, folds = np.zeros(0, dtype=bool), ()
    else:
        accepted, folds = crc_accepted(adversarial, costs, seed=args.seed)
    audit = audit_errors(critical_errors(adversarial, accepted), args.cache, args.data, args.model)
    h4 = [
        _source(feature_path, risk_path, args.costs, args.cache / "extract" / args.model / "<doc_id>.jsonl",
                args.cache / "verify" / args.model / "<doc_id>.json", args.data / "adversarial" / "gold.jsonl"),
        f"CRC at α = {CRC_ALPHA:g} uses complementary adversarial document halves. Each error is counted "
        "at most once. An uncertified fold uses the Step 16 fallback.",
    ]
    if folds:
        h4.append("Fold thresholds: " + ", ".join(f"{fold['lambda']:.6g} ({'certified' if fold['certified'] else 'uncertified'})"
                                                  for fold in folds) + ".")
    h4 += ["", "| Defect | Silent | Accepted critical errors | Share [Wilson CI] |", "| --- | --- | --- | --- |"]
    shares = silent_shares(audit)
    for defect, (silent, n, share, low, high) in shares.items():
        h4.append(f"| {defect} | {silent} | {n} | {_ci(share, low, high)} |")
    silent, n, *_ = shares["Overall"]
    h4_support = wilson(silent, n, 1 - 2 * alpha["H4"])[0] > 0.5 if n and all(f["certified"] for f in folds) else None
    analyses["H4"] = (_verdict(h4_support), h4)

    h5 = [
        _source(feature_path, risk_path, args.costs),
        "Kendall tau-b compares cost-weighted AURCs. The no-abstention score is constant and the "
        "Verifin gate uses a binary score. Self-consistency uses fields with repeated samples. "
        "Each collection is bootstrapped independently.",
    ]
    h5_support = None
    for other in ("adv_test", "real_test"):
        if other not in tables:
            continue
        if tables["syn_test"].empty or tables[other].empty:
            h5.append(f"- syn_test versus {other}: unavailable.")
            continue
        tau, areas = ranking_transfer(tables["syn_test"], tables[other], costs, **options)
        h5 += ["", f"syn_test versus {other}: τ = {_ci(tau.estimate, *tau.interval())}.", "",
               "| Method | syn_test AURC | Comparison AURC |", "| --- | --- | --- |"]
        for method in sorted(areas[0], key=lambda key: (np.isnan(areas[0][key]), areas[0][key], key)):
            h5.append(f"| {method} | {_number(areas[0][method])} | {_number(areas[1][method])} |")
        if other == "adv_test" and np.isfinite(tau.estimate):
            h5_support = tau.quantile(alpha["H5"]) > 0
    analyses["H5"] = (_verdict(h5_support), h5)

    lines = [f"# Hypothesis tests ({args.model})", "", status, "",
             f"`{SCRIPT}` uses {args.n_resamples:,} document bootstrap resamples and reports 95% intervals.", ""]
    for key, (wording, level) in hypotheses.items():
        verdict, details = analyses[key]
        alpha_source = f"`{prereg}` via `{SCRIPT}`" if prereg.is_file() else f"`{SCRIPT}` draft default"
        lines += [f"## {key} — {wording}", "", f"**Verdict: {verdict}.** α = {level:g} ({alpha_source}).",
                  "", *details, ""]
    return "\n".join(lines), audit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("frontier", "open_weights"), default="open_weights")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--prereg", type=Path, default=ROOT / "prereg" / "hypotheses.yaml")
    parser.add_argument("--costs", type=Path, default=COSTS_FILE)
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-real", action="store_true")
    args = parser.parse_args(argv)
    if args.n_resamples < 1:
        parser.error("--n-resamples must be positive")
    report, audit = build_report(args)
    args.results.mkdir(parents=True, exist_ok=True)
    write_audit(audit, args.results / f"h4_errors_{args.model}.csv")
    path = args.results / "hypotheses.md"
    path.write_text(report, encoding="utf-8")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
