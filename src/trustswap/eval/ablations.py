"""Step 23 offline comparisons on cached swap extractions."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from trustswap.decide.crc import calibrate_lambda, doc_loss_matrix, floor
from trustswap.decide.policy import escalate, field_costs, lambda_grid
from trustswap.eval.baselines import aurc, risk_coverage
from trustswap.eval.hypotheses import cluster_bootstrap
from trustswap.extract.evaluate import consistency
from trustswap.score.features import RULES
from trustswap.score.risk import ece, fit

EVAL_SPLITS = ("syn_test", "adv_test")
FAMILIES = (*RULES, "grounding")
SAMPLE_COUNTS = (1, 3, 5)
P_COL = "p_full_logistic"
CRC_ALPHA = 0.05


def drop_family(table, family):
    if family not in FAMILIES:
        raise ValueError(f"Unknown feature family {family!r}")
    if family == "grounding":
        remove = [name for name in table if name.startswith("g_")]
    else:
        remove = [name for name in table if name.startswith(f"v_{family}_") or name == f"rho_{family}"]
    return table.drop(columns=remove)


def with_sample_count(table, records_by_doc, k):
    """Recompute agreement from the first k cached records of eligible documents."""
    if k < 1:
        raise ValueError("k must be positive")
    prefixes = {}
    for doc_id, records in records_by_doc.items():
        prefix = sorted(records, key=lambda record: record["k"])[:k]
        if [record["k"] for record in prefix] == list(range(k)):
            prefixes[doc_id] = prefix
    subset = table.loc[table["doc_id"].isin(prefixes)].copy()
    agreement = [
        consistency(prefixes[doc_id], field)
        for doc_id, field in subset[["doc_id", "field"]].itertuples(index=False, name=None)
    ]
    subset["u_agreement"] = [0.0 if value is None else value for value in agreement]
    subset["u_agreement_missing"] = [int(value is None) for value in agreement]
    subset["k_samples"] = k
    return subset


def fit_full(table, *, seed=0):
    synthetic = table.loc[table["collection"].eq("synthetic")]
    return fit(synthetic.loc[synthetic["split"].eq("train")],
               synthetic.loc[synthetic["split"].eq("dev")], "full", "logistic", seed=seed)


def prediction_deltas(doc_ids, p_ref, e_ref, p_variant, e_variant, costs, **options):
    """Variant minus reference AUROC, ECE and cost-weighted AURC, paired by document."""
    p_ref, e_ref, p_variant, e_variant, costs = map(np.asarray, (p_ref, e_ref, p_variant, e_variant, costs))

    def scores(p, e, c):
        auroc = roc_auc_score(e, p) if e.min() != e.max() else np.nan
        return np.array([auroc, ece(e, p), aurc(risk_coverage(p * c, e, c), weighted=True)])

    def delta(rows, _):
        c = costs[rows]
        return scores(p_variant[rows], e_variant[rows], c) - scores(p_ref[rows], e_ref[rows], c)

    return cluster_bootstrap(doc_ids, delta, **options)


def paired_models(open_weights, frontier, costs, **options):
    """Use each model's own error labels on the same document fields."""
    shared = set(open_weights["doc_id"]) & set(frontier["doc_id"])
    if not shared:
        return None, 0
    keys = ["doc_id", "field"]
    left = open_weights.loc[open_weights["doc_id"].isin(shared), keys + [P_COL, "e"]]
    right = frontier.loc[frontier["doc_id"].isin(shared), keys + [P_COL, "e"]]
    paired = left.merge(right, on=keys, validate="one_to_one", suffixes=("_open", "_frontier"))
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError("Models have different field keys for shared documents")
    c, _ = field_costs(paired, costs)
    result = prediction_deltas(paired["doc_id"], paired[f"{P_COL}_open"], paired["e_open"],
                               paired[f"{P_COL}_frontier"], paired["e_frontier"], c, **options)
    return result, len(shared)


def review_mode_deltas(table, costs, *, n_resamples=2000, seed=0):
    """Cross-fit CRC on shared halves and bootstrap paired held-out outcomes."""
    ids, code = np.unique(table["doc_id"], return_inverse=True)
    if len(ids) < 2:
        raise ValueError("CRC comparisons require at least two documents")
    grid = lambda_grid()
    c, r = field_costs(table, costs)
    p = table[P_COL].to_numpy(float)
    field_counts = np.bincount(code)
    halves = np.array_split(np.random.default_rng(seed).permutation(len(ids)), 2)
    outcomes, folds = {}, {}
    for mode in ("field", "trade"):
        _, losses = doc_loss_matrix(table, P_COL, costs, mode, grid)
        outcomes[mode] = np.zeros((len(ids), 4))
        folds[mode] = []
        for cal, test in (halves, halves[::-1]):
            lam, certified = calibrate_lambda(losses[cal], grid, CRC_ALPHA)
            escalated = escalate(p, c, r, lam)
            reviewed = np.bincount(code, weights=escalated, minlength=len(ids))
            if mode == "trade":
                escalated = (reviewed > 0)[code]
                reviewed = np.where(reviewed > 0, field_counts, 0)
            review_cost = np.bincount(code, weights=r * escalated, minlength=len(ids))
            outcomes[mode][test] = np.column_stack((
                field_counts[test] - reviewed[test], reviewed[test] == 0,
                review_cost[test], losses[test, np.searchsorted(grid, lam)],
            ))
            folds[mode].append({"lambda": lam, "certified": certified, "floor": floor(len(cal))})

    def summarize(mode, rows):
        values = outcomes[mode][rows]
        return np.r_[values[:, 0].sum() / field_counts[rows].sum(), values[:, 1:].mean(axis=0)]

    def difference(rows, _):
        return summarize("trade", rows) - summarize("field", rows)

    result = cluster_bootstrap(ids, difference, n_resamples=n_resamples, seed=seed)
    rows = np.arange(len(ids))
    details = {mode: {"metrics": summarize(mode, rows), "folds": folds[mode]} for mode in outcomes}
    return result, details
