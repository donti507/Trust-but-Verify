"""Step 16: conformal risk control on synthetic test documents."""
from __future__ import annotations

import numpy as np

from trustswap.decide.policy import Costs, escalate, field_costs, lambda_grid


def _document_grid(table, p_col: str, costs: Costs, mode: str, grid):
    """Return document IDs, losses, accepted units and total units by document."""
    if mode not in ("field", "trade"):
        raise ValueError("mode must be 'field' or 'trade'")

    doc_ids, doc_index = np.unique(table["doc_id"].to_numpy(), return_inverse=True)
    grid = np.asarray(grid, dtype=float)
    c, r = field_costs(table, costs)
    p = table[p_col].to_numpy(float)
    accepted = ~escalate(p[:, None], c[:, None], r[:, None], grid[None, :])
    field_counts = np.bincount(doc_index, minlength=len(doc_ids))
    accepted_counts = np.zeros((len(doc_ids), len(grid)), dtype=int)
    np.add.at(accepted_counts, doc_index, accepted)

    error_cost = table["e"].to_numpy(int) * c
    if mode == "field":
        loss = np.zeros(accepted_counts.shape)
        np.add.at(loss, doc_index, np.where(accepted, error_cost[:, None], 0.0))
        automated, totals = accepted_counts, field_counts
    else:
        automated = accepted_counts == field_counts[:, None]
        totals = np.ones(len(doc_ids), dtype=int)
        doc_error_cost = np.bincount(doc_index, weights=error_cost, minlength=len(doc_ids))
        loss = np.where(automated, doc_error_cost[:, None], 0.0)

    return doc_ids, np.minimum(costs.B, loss) / costs.B, automated, totals


def doc_loss_matrix(table, p_col: str, costs: Costs, mode: str, grid) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted document IDs and their normalised losses at each grid point."""
    doc_ids, losses, _, _ = _document_grid(table, p_col, costs, mode, grid)
    return doc_ids, losses


def floor(n: int) -> float:
    """Smallest certifiable alpha with n calibration documents and loss bounded by one."""
    return 1.0 / (n + 1)


def calibrate_lambda(L_cal, grid, alpha: float) -> tuple[float, bool]:
    """Return the largest lambda in the passing prefix and whether it is certified."""
    grid = np.asarray(grid, dtype=float)
    losses = np.asarray(L_cal, dtype=float)
    if (
        grid.ndim != 1
        or grid.size == 0
        or grid[0] != 0
        or not np.isfinite(grid).all()
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("grid must start at zero and contain strictly increasing finite values")
    if losses.ndim != 2 or losses.shape[1] != grid.size or losses.shape[0] == 0:
        raise ValueError("L_cal must have at least one document and one column per grid value")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    bounds = (losses.sum(axis=0) + 1.0) / (len(losses) + 1)
    passing_prefix = np.logical_and.accumulate(bounds <= alpha)
    prefix_size = np.count_nonzero(passing_prefix)
    if prefix_size == 0:
        return 0.0, False
    return float(grid[prefix_size - 1]), True


def simulate(
    table, p_col: str, costs: Costs, mode: str, alpha: float, n_splits: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Summarise random document half-splits of the syn_test risk table.

    A table without split metadata is assumed to contain only syn_test rows.
    The standard error describes the split means and is undefined for one split.
    """
    if "split" in table:
        table = table.loc[table["split"] == "syn_test"]
    if table["doc_id"].nunique() < 2:
        raise ValueError("CRC simulation requires at least two syn_test documents")
    if n_splits < 1:
        raise ValueError("n_splits must be positive")

    grid = lambda_grid()
    doc_ids, losses, automated, totals = _document_grid(table, p_col, costs, mode, grid)
    n_docs = len(doc_ids)
    n_cal = n_docs // 2
    rng = np.random.default_rng(seed)
    test_losses = np.empty(n_splits)
    automation = np.empty(n_splits)
    lambdas = np.empty(n_splits)
    certified = np.empty(n_splits, dtype=bool)
    for split in range(n_splits):
        order = rng.permutation(n_docs)
        cal, test = order[:n_cal], order[n_cal:]
        lam, certified[split] = calibrate_lambda(losses[cal], grid, alpha)
        column = np.searchsorted(grid, lam)
        test_losses[split] = losses[test, column].mean()
        automation[split] = automated[test, column].sum() / totals[test].sum()
        lambdas[split] = lam

    standard_error = test_losses.std(ddof=1) / np.sqrt(n_splits) if n_splits > 1 else np.nan
    return {
        "floor": floor(n_cal),
        "mean_lambda_hat": float(lambdas.mean()),
        "mean_test_loss": float(test_losses.mean()),
        "standard_error": float(standard_error),
        "mean_automation": float(automation.mean()),
        "certified_fraction": float(certified.mean()),
    }
