"""Step 22 statistics on cached predictions and labelled documents."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import log_expit
from scipy.stats import chi2, kendalltau, norm

from trustswap.decide.crc import calibrate_lambda, doc_loss_matrix
from trustswap.decide.policy import Costs, escalate, field_costs, lambda_grid
from trustswap.eval.baselines import METHODS, aurc, risk_coverage
from trustswap.score.features import SEVERITY
from trustswap.score.risk import _logistic, columns

COVERAGES = (0.5, 0.7, 0.9)
CRC_ALPHA = 0.05


@dataclass(frozen=True)
class Bootstrap:
    estimate: np.ndarray
    draws: np.ndarray

    def interval(self, confidence: float = 0.95):
        """Percentile interval with undefined replicate statistics omitted."""
        tail = (1 - confidence) / 2
        if not np.isfinite(self.draws).any():
            return np.full((2, *self.estimate.shape), np.nan)
        return self.quantile([tail, 1 - tail])

    def quantile(self, q):
        return np.nanquantile(self.draws, q, axis=0)


def likelihood_ratio(train: pd.DataFrame, C: float = 1.0) -> dict:
    """Nested logistic fits with a common C and unpenalized log likelihoods."""
    e = train["e"].to_numpy(int)
    base_columns = columns(train, "base")
    base = [col for col in base_columns if train[col].nunique() > 1]
    added = [col for col in columns(train, "full") if col not in base_columns and train[col].nunique() > 1]
    if not added or len(np.unique(e)) != 2:
        return {"lambda": np.nan, "df": len(added), "p": np.nan}

    def loglik(names):
        if not names:
            p = e.mean()
            return float(np.log(np.where(e, p, 1 - p)).sum())
        features = train[names].to_numpy(float)
        logit = _logistic(C).fit(features, e).decision_function(features)
        return float(log_expit((2 * e - 1) * logit).sum())

    statistic = 2 * (loglik(base + added) - loglik(base))
    return {"lambda": statistic, "df": len(added), "p": float(chi2.sf(statistic, len(added)))}


def _document_codes(doc_ids):
    _, codes = np.unique(doc_ids, return_inverse=True)
    return codes


def _resamples(codes, n_resamples, rng):
    groups = np.split(np.argsort(codes, kind="stable"), np.bincount(codes).cumsum()[:-1])
    n_docs = len(groups)
    for _ in range(n_resamples):
        drawn = rng.integers(n_docs, size=n_docs)
        selected = [groups[index] for index in drawn]
        # Every draw contributes a separate document, even when an ID repeats.
        yield np.concatenate(selected), np.repeat(np.arange(n_docs), [len(rows) for rows in selected])


def cluster_bootstrap(doc_ids, statistic, *, n_resamples: int = 2000, seed: int = 0) -> Bootstrap:
    """Resample whole documents and evaluate a paired statistic on row indices and draw IDs."""
    codes = _document_codes(doc_ids)
    if not len(codes) or n_resamples < 1:
        raise ValueError("bootstrap requires at least one document and one resample")
    estimate = statistic(np.arange(len(codes)), codes)
    draws = [statistic(rows, docs) for rows, docs in _resamples(codes, n_resamples, np.random.default_rng(seed))]
    return Bootstrap(np.asarray(estimate), np.asarray(draws))


def aurc_difference(table, costs: Costs, **bootstrap_options) -> Bootstrap:
    c, _ = field_costs(table, costs)
    e = table["e"].to_numpy(int)
    base, full = (table[f"p_{kind}_logistic"].to_numpy(float) for kind in ("base", "full"))

    def difference(rows, _):
        return aurc(risk_coverage(full[rows], e[rows], c[rows]), weighted=True) - aurc(
            risk_coverage(base[rows], e[rows], c[rows]), weighted=True
        )

    return cluster_bootstrap(table["doc_id"], difference, **bootstrap_options)


def _matched_loss(order, error_cost, doc_codes, n_accept, B):
    accept = order[:n_accept]
    losses = np.bincount(doc_codes[accept], weights=error_cost[accept], minlength=doc_codes.max() + 1)
    return float(np.minimum(losses, B).mean() / B)


def matched_automation(table, costs: Costs, coverages=COVERAGES, **bootstrap_options) -> Bootstrap:
    """Rows correspond to targets and columns to coverage, aware loss, blind loss and difference."""
    c, _ = field_costs(table, costs)
    e = table["e"].to_numpy(int)
    p = table["p_full_logistic"].to_numpy(float)
    # Cache-order independent tie-break. It uses identifiers only.
    key = pd.MultiIndex.from_frame(table[["doc_id", "field"]])
    stable = np.empty(len(table), dtype=int)
    stable[key.argsort()] = np.arange(len(table))

    def points(rows, docs):
        n = len(rows)
        p_rows, c_rows = p[rows], c[rows]
        error_cost = e[rows] * c_rows
        aware = np.lexsort((stable[rows], p_rows * c_rows))
        blind = np.lexsort((stable[rows], p_rows))
        result = []
        for coverage in coverages:
            count = int(np.rint(coverage * n))
            aware_loss = _matched_loss(aware, error_cost, docs, count, costs.B)
            blind_loss = _matched_loss(blind, error_cost, docs, count, costs.B)
            result.append((count / n, aware_loss, blind_loss, aware_loss - blind_loss))
        return result

    return cluster_bootstrap(table["doc_id"], points, **bootstrap_options)


def crc_accepted(table, costs: Costs, *, seed: int = 0) -> tuple[np.ndarray, tuple[dict, ...]]:
    """Cross-fit the Step 16 field rule once across complementary document halves."""
    grid = lambda_grid()
    ids, loss = doc_loss_matrix(table, "p_full_logistic", costs, "field", grid)
    accepted = np.zeros(len(table), dtype=bool)
    if len(ids) < 2:
        return accepted, ()
    c, r = field_costs(table, costs)
    p = table["p_full_logistic"].to_numpy(float)
    doc_ids = table["doc_id"].to_numpy()
    halves = np.array_split(np.random.default_rng(seed).permutation(len(ids)), 2)
    folds = []
    for cal, test in (halves, halves[::-1]):
        lam, certified = calibrate_lambda(loss[cal], grid, CRC_ALPHA)
        on_test = np.isin(doc_ids, ids[test])
        accepted[on_test] = ~escalate(p[on_test], c[on_test], r[on_test], lam)
        folds.append({"lambda": lam, "certified": certified})
    return accepted, tuple(folds)


def critical_errors(table, accepted) -> pd.DataFrame:
    severity = table["field"].str.rsplit(".", n=1).str[-1].map(SEVERITY)
    return table.loc[np.asarray(accepted, dtype=bool) & table["e"].eq(1) & severity.eq("critical")]


def wilson(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if not total:
        return np.nan, np.nan
    p = successes / total
    z = norm.ppf((1 + confidence) / 2)
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def silent_shares(errors: pd.DataFrame) -> dict:
    """Summarise cached HARD/SOFT firings for accepted critical errors."""
    by_defect = [("Overall", errors), *errors.groupby("defect", sort=True, dropna=False)]
    results = {}
    for defect, rows in by_defect:
        n = len(rows)
        silent = int(rows["rules"].map(len).eq(0).sum())
        results[defect] = (silent, n, silent / n if n else np.nan, *wilson(silent, n))
    return results


def _ranking_inputs(table, costs):
    c, _ = field_costs(table, costs)
    e = table["e"].to_numpy(int)
    hard = table.filter(regex=r"^v_.+_HARD$").eq(1).any(axis=1).to_numpy()
    eligible = table["k_samples"].to_numpy() > 1
    scores = {
        "no_abstention": np.zeros(len(table)),
        "logprob": np.where(table["u_logprob_missing"].eq(1), np.inf, -table["u_logprob"]),
        "self_consistency": 1 - table["u_agreement"].to_numpy(),
        "verbal": 1 - table["u_verbal_conf"].to_numpy(),
        "extractconf": table["p_base_logistic"].to_numpy(),
        "verifin_gate": hard.astype(int),
        "ours": table["p_full_logistic"].to_numpy() * c,
    }
    return scores, e, c, eligible


def _areas(inputs, rows):
    scores, e, c, eligible = inputs
    return {
        method: aurc(risk_coverage(scores[part], e[part], c[part]), weighted=True)
        for method, scores in scores.items()
        for part in [rows[eligible[rows]] if method == "self_consistency" else rows]
    }


def ranking_transfer(left, right, costs: Costs, *, n_resamples: int = 2000, seed: int = 0):
    """Kendall tau-b over common methods, resampling each collection independently."""
    inputs = [_ranking_inputs(table, costs) for table in (left, right)]
    codes = [_document_codes(table["doc_id"]) for table in (left, right)]
    areas = [_areas(source, np.arange(len(code))) for source, code in zip(inputs, codes)]
    common = [method for method in METHODS if all(np.isfinite(area[method]) for area in areas)]

    def tau(a, b):
        if len(common) < 2:
            return np.nan
        return float(kendalltau([a[method] for method in common], [b[method] for method in common]).statistic)

    rng = np.random.default_rng(seed)
    samples = [_resamples(code, n_resamples, rng) for code in codes]
    draws = [tau(*(_areas(source, rows) for source, (rows, _) in zip(inputs, replicate)))
             for replicate in zip(*samples)]
    return Bootstrap(np.asarray(tau(*areas)), np.asarray(draws)), areas
