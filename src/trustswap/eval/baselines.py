"""Step 18: selective extraction baselines and their operating points."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trustswap.decide.crc import calibrate_lambda, floor
from trustswap.decide.policy import Costs, field_costs

CURVE_COLUMNS = ["coverage", "risk", "cost_risk"]
METHODS = ("no_abstention", "logprob", "self_consistency", "verbal", "extractconf", "verifin_gate", "ours")
FIXED_METHODS = ("no_abstention", "verifin_gate")


def risk_coverage(scores, e, c) -> pd.DataFrame:
    """Return the cumulative accepted-field metrics at each distinct score."""
    scores, e, c = np.asarray(scores), np.asarray(e), np.asarray(c)
    order = np.argsort(scores)
    sorted_scores = scores[order]
    if not len(order):
        return pd.DataFrame(columns=CURVE_COLUMNS, dtype=float)
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    accepted = ends + 1
    return pd.DataFrame({
        "coverage": accepted / len(order),
        "risk": e[order].cumsum()[ends] / accepted,
        "cost_risk": (e[order] * c[order]).cumsum()[ends] / accepted,
    })


def aurc(curve: pd.DataFrame, weighted: bool = False) -> float:
    """Integrate the left step between thresholds, extending the first risk to zero coverage."""
    if curve.empty:
        return np.nan
    coverage = curve["coverage"].to_numpy()
    risk = curve["cost_risk" if weighted else "risk"].to_numpy()
    return float(coverage[0] * risk[0] + np.diff(coverage) @ risk[:-1])


def _fixed_point(accepted, e, c) -> pd.DataFrame:
    count = np.count_nonzero(accepted)
    return pd.DataFrame([(
        count / len(accepted),
        e[accepted].mean() if count else np.nan,
        (e[accepted] * c[accepted]).mean() if count else np.nan,
    )], columns=CURVE_COLUMNS)


def crc_operating_point(
    doc_ids, scores, e, c, B: float, alpha: float = 0.05, *, n_splits: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Average held-out field coverage and clipped document loss over document half-splits."""
    unique_docs, doc_index = np.unique(doc_ids, return_inverse=True)
    n_docs = len(unique_docs)
    n_cal = n_docs // 2
    if n_cal < 1:
        raise ValueError("CRC needs at least two documents")
    if n_splits < 1:
        raise ValueError("n_splits must be positive")

    scores = np.asarray(scores, dtype=float)
    e = np.asarray(e)
    error_cost = e * np.asarray(c)
    quantiles = np.linspace(0, 1, 201)
    rng = np.random.default_rng(seed)
    coverage = np.empty(n_splits)
    test_loss = np.empty(n_splits)
    certified = np.empty(n_splits, dtype=bool)
    n_accepted = errors_accepted = 0
    cost_accepted = 0.0
    for i in range(n_splits):
        docs = rng.permutation(n_docs)
        cal_index = np.full(n_docs, -1)
        cal_index[docs[:n_cal]] = np.arange(n_cal)
        field_cal_index = cal_index[doc_index]
        on_cal = field_cal_index >= 0
        cal_scores = scores[on_cal]
        thresholds = np.quantile(cal_scores, quantiles, method="inverted_cdf")

        # Bin each field at its first accepting threshold, then accumulate per document.
        n_thresholds = thresholds.size + 1  # Column zero escalates every field.
        first = 1 + np.searchsorted(thresholds, cal_scores, side="left")
        bins = field_cal_index[on_cal] * n_thresholds + first
        totals = np.bincount(bins, weights=error_cost[on_cal], minlength=n_cal * n_thresholds)
        totals = totals.reshape(n_cal, n_thresholds).cumsum(axis=1)
        losses = np.minimum(totals, B) / B
        # CRC only needs the order. Indices also support repeated or infinite thresholds.
        selected, certified[i] = calibrate_lambda(losses, np.arange(n_thresholds), alpha)
        accepted = np.zeros(len(scores), dtype=bool)
        if selected > 0:
            accepted = (~on_cal) & (scores <= thresholds[int(selected) - 1])
        accepted_costs = error_cost[accepted]
        doc_loss = np.bincount(doc_index[accepted], weights=accepted_costs, minlength=n_docs)
        test_loss[i] = np.minimum(doc_loss[docs[n_cal:]], B).mean() / B
        count = np.count_nonzero(accepted)
        coverage[i] = count / np.count_nonzero(~on_cal)
        n_accepted += count
        errors_accepted += e[accepted].sum()
        cost_accepted += accepted_costs.sum()

    return {
        "coverage": float(coverage.mean()),
        "risk": float(errors_accepted / n_accepted) if n_accepted else np.nan,
        "cost_risk": float(cost_accepted / n_accepted) if n_accepted else np.nan,
        "mean_doc_loss": float(test_loss.mean()),
        "floor": floor(n_cal),
        "certified_fraction": float(certified.mean()),
    }


def evaluate_baselines(table: pd.DataFrame, costs: Costs, *, n_splits: int = 1000, seed: int = 0) -> dict:
    """Evaluate all applicable baselines for a single split of joined cache rows."""
    c, _ = field_costs(table, costs)
    e = table["e"].to_numpy()
    doc_ids = table["doc_id"].to_numpy()
    consistent = table["k_samples"].to_numpy() > 1
    hard_columns = table.filter(regex=r"^v_.+_HARD$")
    score_methods = {
        "logprob": np.where(table["u_logprob_missing"] == 1, np.inf, -table["u_logprob"]),
        "self_consistency": 1 - table.loc[consistent, "u_agreement"].to_numpy(),
        "verbal": 1 - table["u_verbal_conf"].to_numpy(),
        "extractconf": table["p_base_logistic"].to_numpy(),
        "ours": table["p_full_logistic"].to_numpy() * c,
    }
    fixed = {
        "no_abstention": np.ones(len(table), dtype=bool),
        "verifin_gate": ~hard_columns.eq(1).any(axis=1).to_numpy(),
    }
    results = {}
    for method in METHODS:
        if method in fixed:
            results[method] = {"curve": _fixed_point(fixed[method], e, c), "crc": None, "n_fields": len(e)}
            continue
        eligible = consistent if method == "self_consistency" else fixed["no_abstention"]
        scores = score_methods[method]
        if not len(scores):
            continue
        labels, costs_per_field, docs = e[eligible], c[eligible], doc_ids[eligible]
        curve = risk_coverage(scores, labels, costs_per_field)
        point = None
        if len(np.unique(docs)) >= 2:
            point = crc_operating_point(docs, scores, labels, costs_per_field, costs.B, n_splits=n_splits, seed=seed)
        results[method] = {"curve": curve, "crc": point, "n_fields": len(scores)}
    return results
