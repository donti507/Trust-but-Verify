"""Step 24: regenerate publication figures from cached study outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import yaml

from trustswap.decide.crc import floor
from trustswap.decide.policy import COSTS_FILE, load_costs
from trustswap.eval.baselines import FIXED_METHODS, METHODS
from trustswap.eval.hypotheses import _ranking_inputs

ROOT = Path(__file__).resolve().parents[3]
COLLECTIONS = {"syn_test": "synthetic", "adv_test": "adversarial", "real_test": "real"}
LABELS = dict(zip(METHODS, (
    "No abstention", "Log probability", "Self-consistency", "Verbal confidence",
    "ExtractConf", "Verifin gate", "Trust but Verify",
)))
COLORS = dict(zip(METHODS, ("#404040", "#0072B2", "#009E73", "#B14D85", "#C44E00", "#7B5F00", "#332288")))
STYLES = dict(zip(METHODS, ("-", "--", "-.", ":", "--", "-", "-")))
TAXONOMY_COLUMNS = ("Defect", "Critical errors", "Verifier silent", "Verifier fired", "Human marks", "Rule gaps")
KEYS = ["doc_id", "field", "collection", "split"]


def real_test_ready(prereg: Path, results: Path) -> bool:
    verifier = prereg / "verifier-v1.0.json"
    registration = prereg / "hypotheses.yaml"
    if not all(path.is_file() for path in (verifier, registration, results / "real_eval.json")):
        return False
    with verifier.open(encoding="utf-8") as source:
        verifier_status = json.load(source).get("status")
    with registration.open(encoding="utf-8") as source:
        hypotheses_status = (yaml.safe_load(source) or {}).get("status")
    return verifier_status == hypotheses_status == "frozen"


def _load_split(cache: Path, model: str, split: str):
    filters = [("split", "==", split), ("collection", "==", COLLECTIONS[split])]
    filename = f"{model}.parquet"
    features = pd.read_parquet(cache / "features" / filename, engine="pyarrow", filters=filters)
    risks = pd.read_parquet(cache / "risk" / filename, engine="pyarrow", filters=filters)
    p_columns = [col for col in risks if col.startswith(("p_base_", "p_full_"))]
    joined = features.merge(risks[KEYS + ["e"] + p_columns], on=KEYS, suffixes=("", "_risk"),
                            validate="one_to_one")
    if joined.empty or len(joined) != len(features) or len(joined) != len(risks):
        raise ValueError(f"{split} needs matching feature and risk cache rows")
    if not joined["e"].eq(joined["e_risk"]).all():
        raise ValueError(f"Feature and risk caches disagree on error labels in {split}")
    return joined


def frontier_stats(doc_ids, scores, error_cost, B, method, *, n_resamples=2000, seed=0):
    """Field automation and clipped document loss, with pointwise clustered intervals."""
    ids, code = np.unique(doc_ids, return_inverse=True)
    n_docs = len(ids)
    if not n_docs or n_resamples < 1:
        raise ValueError("Frontier requires at least one document and one resample")
    scores, error_cost = np.asarray(scores), np.asarray(error_cost)
    if method in FIXED_METHODS:
        accepted = scores == 0
        counts = np.bincount(code, weights=accepted, minlength=n_docs)[:, None]
        losses = np.bincount(code, weights=accepted * error_cost, minlength=n_docs)[:, None]
    else:
        quantiles = np.linspace(0, 1, 201)
        thresholds = np.unique(np.quantile(scores, quantiles, method="inverted_cdf"))
        n_points = len(thresholds) + 1
        first = 1 + np.searchsorted(thresholds, scores)
        bins = code * n_points + first
        counts = np.bincount(bins, minlength=n_docs * n_points).reshape(n_docs, n_points).cumsum(axis=1)
        losses = np.bincount(bins, weights=error_cost, minlength=n_docs * n_points)
        losses = losses.reshape(n_docs, n_points).cumsum(axis=1)

    loss = np.minimum(B, losses) / B
    totals = np.bincount(code)
    x = counts.sum(axis=0) / totals.sum()
    y = loss.mean(axis=0)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_docs, np.full(n_docs, 1.0 / n_docs), size=n_resamples).astype(float)
    boot_loss = weights @ loss / n_docs
    if method not in FIXED_METHODS:
        boot_x = (weights @ counts) / (weights @ totals)[:, None]
        boot_loss = np.array([np.interp(x, bx, by) for bx, by in zip(boot_x, boot_loss)])
    lo, hi = np.quantile(boot_loss, [0.025, 0.975], axis=0)
    return x, y, lo, hi


def _axes(ax, xlabel, ylabel):
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(axis="y", color="#e2e8ee", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _legend(ax, *, columns=2):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best", ncol=columns, framealpha=0.9, fontsize=8)


def plot_frontier(table, costs, *, n_resamples, seed):
    fig, ax = plt.subplots(figsize=(8.3, 5), layout="constrained")
    scores, e, c, multiple = _ranking_inputs(table, costs)
    docs = table["doc_id"].to_numpy()
    n_cal = table["doc_id"].nunique() // 2
    feasibility = floor(n_cal)
    ax.axhspan(0, feasibility, color="#dce4eb", alpha=0.75, label=f"Below CRC floor ({feasibility:.3g})", zorder=0)
    upper = feasibility
    for method in METHODS:
        eligible = multiple if method == "self_consistency" else np.ones(len(table), dtype=bool)
        if not eligible.any():
            continue
        x, y, lo, hi = frontier_stats(docs[eligible], scores[method][eligible], (e * c)[eligible],
                                      costs.B, method, n_resamples=n_resamples, seed=seed)
        upper = max(upper, float(max(y.max(), hi.max())))
        color = COLORS[method]
        if method in FIXED_METHODS:
            ax.vlines(x, lo, hi, color=color, linewidth=1.7, zorder=3)
            ax.scatter(x, y, marker="s" if method == "verifin_gate" else "D", color=color,
                       label=LABELS[method], zorder=4)
        else:
            ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0)
            ax.plot(x, y, color=color, linestyle=STYLES[method], label=LABELS[method], linewidth=1.65)
    _axes(ax, "Share of fields automated", "Mean normalized document loss")
    ax.set(xlim=(-0.015, 1.025), ylim=(0, min(1.04, upper * 1.08)))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Automation and document loss", loc="left", fontweight="bold")
    ax.text(0.0, -0.21, f"Pointwise 95% document-bootstrap intervals. CRC calibration n = {n_cal:,}.",
            transform=ax.transAxes, fontsize=8, color="#404c5b")
    _legend(ax)
    return fig


def plot_risk_coverage(curves):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), layout="constrained", sharex=True)
    for ax, column, title in zip(axes, ("risk", "cost_risk"), ("Uniform risk", "Cost-weighted risk")):
        for method in METHODS:
            rows = curves.loc[curves["method"].eq(method)].sort_values("coverage")
            if rows.empty or rows[column].isna().all():
                continue
            if method in FIXED_METHODS:
                ax.scatter(rows["coverage"], rows[column], label=LABELS[method], color=COLORS[method],
                           marker="s" if method == "verifin_gate" else "D", zorder=3)
            else:
                ax.plot(rows["coverage"], rows[column], label=LABELS[method],
                        color=COLORS[method], linestyle=STYLES[method])
        _axes(ax, "Coverage", "Error rate" if column == "risk" else "Error cost per accepted field")
        ax.set(xlim=(-0.015, 1.025), ylim=(0, None))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        if column == "risk":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(title, loc="left", fontweight="bold")
    _legend(axes[0])
    return fig


def reliability_bins(p, e, bins=15):
    p, e = np.asarray(p, dtype=float), np.asarray(e, dtype=float)
    indices = np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1], right=True)
    counts = np.bincount(indices, minlength=bins)
    active = counts > 0
    mean_p = np.bincount(indices, weights=p, minlength=bins)[active] / counts[active]
    mean_e = np.bincount(indices, weights=e, minlength=bins)[active] / counts[active]
    return mean_p, mean_e


def plot_reliability(table):
    variants = [("logistic", "Logistic"), ("lightgbm", "LightGBM")]
    variants = [(variant, title) for variant, title in variants
                if all(f"p_{feature}_{variant}" in table for feature in ("base", "full"))]
    if not variants:
        raise ValueError("Reliability requires a base and full probability column pair")
    fig, axes = plt.subplots(1, len(variants), figsize=(4.5 * len(variants), 4.3),
                             layout="constrained", squeeze=False)
    for ax, (variant, title) in zip(axes[0], variants):
        ax.plot([0, 1], [0, 1], color="#7c8794", linestyle=":", label="Perfect calibration")
        for feature, color, marker in (("base", COLORS["logprob"], "o"), ("full", COLORS["extractconf"], "s")):
            x, y = reliability_bins(table[f"p_{feature}_{variant}"], table["e"])
            ax.plot(x, y, color=color, marker=marker, label=feature.capitalize(), markersize=4)
        _axes(ax, "Mean predicted error probability", "Observed error rate")
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(f"{title} reliability · 15 bins", loc="left", fontweight="bold")
        _legend(ax, columns=1)
    return fig


def verifier_recall(docs_path: Path, verify_dir: Path):
    """Compute detection per cached, non-control document with a detectable expected rule."""
    totals = {}
    with docs_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            doc = json.loads(line)
            expected = set(doc.get("expected_rules") or []) - {"silent"}
            if doc.get("collection", "adversarial") != "adversarial" or doc.get("is_control") or not expected:
                continue
            doc_id = doc["doc_id"]
            if Path(doc_id).name != doc_id:
                raise ValueError(f"Invalid document ID {doc_id!r}")
            path = verify_dir / f"{doc_id}.json"
            if not path.is_file():
                continue
            rules = json.loads(path.read_text(encoding="utf-8"))["rules"]
            detected = any(
                rules.get(rule, {}).get(field, {}).get("verdict") in {"HARD", "SOFT"}
                for rule in expected for field in doc["affected_fields"]
            )
            counts = totals.setdefault(doc["defect"], [0, 0])
            counts[0] += int(detected)
            counts[1] += 1
    return dict(sorted(totals.items()))


def plot_verifier_recall(counts):
    fig, ax = plt.subplots(figsize=(7.6, max(3.0, len(counts) * 0.37 + 1.2)), layout="constrained")
    ax.set_title("Verifier recall by adversarial defect", loc="left", fontweight="bold")
    _axes(ax, "Share of eligible documents detected", "")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlim=(0, 1.16), xticks=np.linspace(0, 1, 6))
    if counts:
        for i, (defect, (hits, n)) in enumerate(counts.items()):
            ax.barh(i, hits / n, height=0.64, color=COLORS["logprob"])
            ax.text(hits / n + 0.018, i, f"{hits}/{n}", va="center", fontsize=9)
        ax.set_yticks(range(len(counts)), [name.replace("_", " ") for name in counts])
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No eligible verifier records", transform=ax.transAxes, ha="center")
    return fig


def taxonomy_table(errors):
    fired = errors["rules"].map(lambda text: bool(json.loads(text)) if text else False)
    marks = errors["rule_gap"].str.strip().str.casefold()
    summary = errors.assign(
        silent=~fired, fired=fired, marked=marks.ne(""),
        gap=marks.isin({"yes", "true", "1", "a rule could be written"}),
    ).groupby("defect", sort=True, dropna=False)
    rows = summary.agg(
        errors=("field", "size"), silent=("silent", "sum"), fired=("fired", "sum"),
        marked=("marked", "sum"), gaps=("gap", "sum"),
    )
    counts = [(defect.replace("_", " "), *values) for defect, *values in rows.itertuples()]
    counts.append(("Total", *rows.sum().tolist()))
    return counts


def plot_taxonomy(rows):
    fig, ax = plt.subplots(figsize=(10.4, max(2.4, 0.42 * len(rows) + 1.4)), layout="constrained")
    ax.set_title("Accepted critical errors by adversarial defect", loc="left", fontweight="bold", pad=12)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=TAXONOMY_COLUMNS, cellLoc="center",
                     colWidths=[0.24, 0.16, 0.16, 0.15, 0.15, 0.14], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e0e8")
        if row == 0:
            cell.set_facecolor("#263d55")
            cell.set_text_props(color="white", weight="bold")
        elif row == len(rows):
            cell.set_facecolor("#dce4eb")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f1f5f8")
        if col == 0:
            cell.get_text().set_ha("left")
    return fig


def _write_markdown(path, rows):
    lines = [
        "# H4 taxonomy", "",
        "Human marks count nonblank annotations. Rule gaps count affirmative human annotations.", "",
        "| " + " | ".join(TAXONOMY_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(TAXONOMY_COLUMNS)) + " |",
    ]
    lines += ["| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save(fig, out, name):
    try:
        fig.savefig(out / f"{name}.png", dpi=220)
        fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None})
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("open_weights", "frontier"), required=True)
    parser.add_argument("--split", choices=tuple(COLLECTIONS), default="syn_test")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--prereg", type=Path, default=ROOT / "prereg")
    parser.add_argument("--costs", type=Path, default=COSTS_FILE)
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.split == "real_test" and not real_test_ready(args.prereg, args.results):
        parser.error("real_test requires a frozen verifier, frozen pre-registration, and results/real_eval.json")
    if args.n_resamples < 1:
        parser.error("--n-resamples must be positive")

    model, split = args.model, args.split
    split_name = f"{model}_{split}"
    table = _load_split(args.cache, model, split)
    costs = load_costs(args.costs)
    curves = pd.read_csv(args.results / "curves" / f"{split_name}.csv")
    recalls = verifier_recall(args.data / "adversarial" / "docs.jsonl", args.cache / "verify" / model)
    errors = pd.read_csv(args.results / f"h4_errors_{model}.csv", dtype=str, keep_default_na=False)
    taxonomy = taxonomy_table(errors)

    out = args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.labelcolor": "#24374a"}):
        frontier = plot_frontier(table, costs, n_resamples=args.n_resamples, seed=args.seed)
        _save(frontier, out, f"automation_loss_{split_name}")
        _save(plot_risk_coverage(curves), out, f"risk_coverage_{split_name}")
        _save(plot_reliability(table), out, f"reliability_{split_name}")
        _save(plot_verifier_recall(recalls), out, f"verifier_recall_{model}")
        _save(plot_taxonomy(taxonomy), out, f"h4_taxonomy_{model}")
    _write_markdown(out / f"h4_taxonomy_{model}.md", taxonomy)
    print(f"Wrote figures and taxonomy table to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
