"""Step 23: build the offline ablation report from feature and extraction caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trustswap.decide.policy import COSTS_FILE, field_costs, load_costs
from trustswap.eval.ablations import (
    CRC_ALPHA, EVAL_SPLITS, FAMILIES, P_COL, SAMPLE_COUNTS,
    drop_family, fit_full, paired_models, prediction_deltas, review_mode_deltas, with_sample_count,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("open_weights", "frontier")
FILTERS = [
    [("collection", "==", "synthetic"), ("split", "in", ("train", "dev", "syn_test"))],
    [("collection", "==", "adversarial"), ("split", "==", "adv_test")],
]


def load_samples(directory: Path, doc_ids) -> dict:
    records = {}
    for doc_id in doc_ids:
        if Path(doc_id).name != doc_id:
            raise ValueError(f"Invalid document ID {doc_id!r}")
        path = directory / f"{doc_id}.jsonl"
        if path.is_file():
            with path.open(encoding="utf-8") as source:
                records[doc_id] = [json.loads(line) for line in source if line.strip()]
    return records


def _fit_ready(table):
    return all(table.loc[table["split"].eq(split), "e"].nunique() == 2 for split in ("train", "dev"))


def _metrics(result):
    cells = []
    for estimate, draws in zip(result.estimate, result.draws.T):
        finite = draws[np.isfinite(draws)]
        if not np.isfinite(estimate) or not len(finite):
            cells.append("Unavailable")
        else:
            low, high = np.quantile(finite, [0.025, 0.975])
            cells.append(f"{estimate:+.6g} [{low:+.6g}, {high:+.6g}]")
    return cells


def _row(*cells):
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _predict(table, model):
    predictions = {}
    for split in EVAL_SPLITS:
        held_out = table.loc[table["split"].eq(split)].copy()
        if not held_out.empty:
            held_out[P_COL] = model.predict(held_out)
        predictions[split] = held_out
    return predictions


def build_report(cache: Path, costs_path: Path, *, n_resamples=2000, seed=0) -> str:
    costs = load_costs(costs_path)
    options = {"n_resamples": n_resamples, "seed": seed}
    predictions, reviews, review_details, notes, sources = [], [], [], [], []
    full_by_model = {}

    def compare(name, model_name, reference, variant):
        for split in EVAL_SPLITS:
            ref, alt = reference[split], variant[split]
            if ref.empty:
                notes.append(f"{model_name}, {name}, {split} skipped because no eligible documents were cached.")
                continue
            c, _ = field_costs(ref, costs)
            delta = prediction_deltas(ref["doc_id"], ref[P_COL], ref["e"], alt[P_COL], alt["e"], c, **options)
            predictions.append(_row(name, model_name, split, ref["doc_id"].nunique(), *_metrics(delta)))

    for model_name in MODELS:
        feature_path = cache / "features" / f"{model_name}.parquet"
        if not feature_path.is_file():
            notes.append(f"{model_name} skipped because its feature cache is missing.")
            continue
        table = pd.read_parquet(feature_path, engine="pyarrow", filters=FILTERS).reset_index(drop=True)
        if not _fit_ready(table):
            notes.append(f"{model_name} skipped because synthetic train and dev cannot support fitting and calibration.")
            continue
        sample_path = cache / "extract" / model_name
        sources += [f"- `{feature_path}` and `{sample_path}/<doc_id>.jsonl`."]
        full_model = fit_full(table, seed=seed)
        full = _predict(table, full_model)
        full_by_model[model_name] = full
        for family in FAMILIES:
            variant = drop_family(table, family)
            compare(f"Drop {family}", model_name, full, _predict(variant, fit_full(variant, seed=seed)))

        samples = load_samples(sample_path, table["doc_id"].unique())
        for k in SAMPLE_COUNTS:
            variant = with_sample_count(table, samples, k)
            if not _fit_ready(variant):
                notes.append(f"{model_name}, K = {k} skipped because eligible caches do not cover "
                             "both classes in synthetic train and dev.")
                continue
            matched = table.loc[variant.index]
            baseline = full_model if len(matched) == len(table) else fit_full(matched, seed=seed)
            compare(f"K = {k}", model_name, _predict(matched, baseline),
                    _predict(variant, fit_full(variant, seed=seed)))

        for split in EVAL_SPLITS:
            held_out = full[split]
            n_docs = held_out["doc_id"].nunique()
            if n_docs < 2:
                notes.append(f"{model_name}, review modes, {split} skipped because CRC needs two or more documents.")
                continue
            delta, details = review_mode_deltas(held_out, costs, **options)
            reviews.append(_row(model_name, split, n_docs, *_metrics(delta)))
            for mode, detail in details.items():
                values = ", ".join(f"{value:.6g}" for value in detail["metrics"])
                folds = ", ".join(f"λ={fold['lambda']:.6g}, floor={fold['floor']:.6g}, "
                                  f"{'certified' if fold['certified'] else 'uncertified'}"
                                  for fold in detail["folds"])
                review_details.append(f"- {model_name}, {split}, {mode} mode: {values}. Folds: {folds}.")

    if not full_by_model:
        raise ValueError("No feature cache has usable synthetic train and dev rows")
    if len(full_by_model) == 2:
        for split in EVAL_SPLITS:
            left, right = (full_by_model[model][split] for model in MODELS)
            delta, n_docs = paired_models(left, right, costs, **options)
            if delta is None:
                notes.append(f"Frontier versus open-weights, {split} skipped because there are no shared documents.")
            else:
                predictions.append(_row("Frontier − open-weights", "Paired models", split, n_docs, *_metrics(delta)))
    else:
        notes.append("Frontier versus open-weights requires a usable feature cache for each model.")

    lines = [
        "# Ablations and controls", "",
        f"`scripts/ablations.py` uses {n_resamples:,} paired document bootstrap resamples per comparison "
        "and reports percentile 95% confidence intervals. Prediction intervals condition on the fitted models.",
        "", f"Cost configuration is `{costs_path}`.", "", *sources, "",
        "## Prediction ablations", "",
        "Deltas use the full model as the reference unless the row names a model comparison. "
        "AURC ranks fields by calibrated risk × cost. Each K comparison uses a full-cache agreement "
        "reference trained and evaluated on the same eligible documents. The model comparison uses "
        "shared document fields and each model's own error labels.", "",
        "| Ablation | Model | Split | Documents | Δ AUROC [95% CI] | Δ ECE [95% CI] | Δ cost-weighted AURC [95% CI] |",
        "| --- | --- | --- | --- | --- | --- | --- |", *predictions, "",
        "## Review modes", "",
        f"Step 16 CRC at α = {CRC_ALPHA:g} calibrates both modes on the same complementary document halves. "
        "Each document is evaluated once. Intervals condition on the selected thresholds. "
        "Field automation and document straight-through rates use the same denominators across modes.", "",
        "| Model | Split | Documents | Δ field automation [95% CI] | Δ straight-through [95% CI] | "
        "Δ review cost/doc [95% CI] | Δ loss [95% CI] |",
        "| --- | --- | --- | --- | --- | --- | --- |", *reviews, "",
        "Deltas are trade minus field. Per-mode values follow the same metric order.", "", *review_details,
    ]
    if notes:
        lines += ["", "## Unavailable comparisons", "", *(f"- {note}" for note in notes)]
    lines += [
        "", "## Pending: needs model calls", "",
        "- Repair loop: re-prompt once after a HARD verdict and compare the repaired extraction with the original.",
        "- Memorisation control: perturb numbers in eligible real documents and re-extract to test sensitivity.", "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--costs", type=Path, default=COSTS_FILE)
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.n_resamples < 1:
        parser.error("--n-resamples must be positive")
    report = build_report(args.cache, args.costs, n_resamples=args.n_resamples, seed=args.seed)
    args.results.mkdir(parents=True, exist_ok=True)
    output = args.results / "ablations.md"
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
