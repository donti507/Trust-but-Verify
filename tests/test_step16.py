"""Step 16: conformal risk control checks on generated inputs."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scripts import crc as crc_script
from trustswap.decide.crc import calibrate_lambda, doc_loss_matrix, floor, simulate
from trustswap.decide.policy import evaluate_lambda, lambda_grid, load_costs

P_COL = "p_full_logistic"


def _risk_table(n_docs, seed=16):
    rng = np.random.default_rng(seed)
    fields = ("trade_date", "fixed_leg.notional_amount", "notice_address", "floating_leg.spread")
    rows = []
    for doc_id in range(n_docs):
        for field in fields:
            p = rng.uniform(0, 0.35)
            rows.append((doc_id, field, "syn_test", int(rng.random() < p), p))
    return pd.DataFrame(rows, columns=["doc_id", "field", "split", "e", P_COL])


@pytest.mark.parametrize("mode", ["field", "trade"])
def test_doc_loss_matrix_matches_policy_for_each_document(mode):
    table = pd.DataFrame([
        ("c", "trade_date", 1, 0.0),
        ("b", "notice_address", 1, 0.0),
        ("a", "fixed_leg.notional_amount", 1, 0.25),
        ("b", "fixed_leg.notional_amount", 0, 0.25),
        ("a", "trade_date", 1, 0.5),
    ], columns=["doc_id", "field", "e", P_COL])
    costs = replace(load_costs(), fields={"notional_amount": {"r": 2.0}})
    grid = [0.0, 5.0, 12.49, 12.5, 1000.0]
    doc_ids, losses = doc_loss_matrix(table, P_COL, costs, mode, grid)

    assert doc_ids.tolist() == ["a", "b", "c"]
    assert losses.shape == (len(doc_ids), len(grid))
    assert np.all((0.0 <= losses) & (losses <= 1.0))
    assert np.all(np.diff(losses, axis=1) >= 0)
    for row, doc_id in enumerate(doc_ids):
        document = table.loc[table["doc_id"] == doc_id]
        expected = [evaluate_lambda(document, P_COL, lam, costs, mode)["mean_doc_loss"] for lam in grid]
        np.testing.assert_allclose(losses[row], expected)


def test_calibration_floor_and_increasing_alpha():
    grid = [0.0, 0.25, 0.5, 1.0]
    losses = np.tile([0.0, 0.1, 0.2, 0.4], (9, 1))
    assert floor(9) == 0.1
    assert calibrate_lambda(losses, grid, alpha=0.09) == (0.0, False)
    assert calibrate_lambda(losses, grid, alpha=floor(9)) == (0.0, True)

    results = [calibrate_lambda(losses, grid, alpha) for alpha in (0.2, 0.3, 0.5)]
    assert results == [(0.25, True), (0.5, True), (1.0, True)]


def test_calibration_stops_at_first_failing_bound():
    grid = [0.0, 1.0, 2.0, 3.0]
    losses = np.tile([0.0, 0.1, 0.8, 0.05], (9, 1))
    assert calibrate_lambda(losses, grid, alpha=0.2) == (1.0, True)
    losses[:, 0] = 0.8
    assert calibrate_lambda(losses, grid, alpha=0.2) == (0.0, False)


@pytest.mark.parametrize("mode", ["field", "trade"])
def test_crc_passes_on_calibrated_synthetic_risks(mode):
    table, costs = _risk_table(500), load_costs()
    previous_lambda = 0.0
    for alpha in (0.01, 0.02, 0.05, 0.1):
        result = simulate(table, P_COL, costs, mode, alpha, n_splits=200, seed=23)
        assert result["floor"] == floor(250)
        assert result["certified_fraction"] == 1.0
        assert 0.0 < result["mean_test_loss"] <= alpha
        assert 0.0 < result["standard_error"]
        assert 0.0 < result["mean_automation"] <= 1.0
        assert previous_lambda <= result["mean_lambda_hat"]
        previous_lambda = result["mean_lambda_hat"]


@pytest.mark.parametrize("mode", ["field", "trade"])
def test_simulation_matches_held_out_policy_metrics(mode):
    table = _risk_table(17)
    # Vary document sizes so field-mode automation must be weighted by field count.
    table = table.drop(index=[0, 1, 4, 8, 9, 10])
    costs, grid = load_costs(), lambda_grid()
    doc_ids, losses = doc_loss_matrix(table, P_COL, costs, mode, grid)
    rng = np.random.default_rng(3)
    alpha = 0.2
    expected_losses, expected_automation, expected_lambdas = [], [], []
    for _ in range(5):
        order = rng.permutation(len(doc_ids))
        cal, test = order[:8], order[8:]
        lam, certified = calibrate_lambda(losses[cal], grid, alpha)
        assert certified
        held_out = table.loc[table["doc_id"].isin(doc_ids[test])]
        expected = evaluate_lambda(held_out, P_COL, lam, costs, mode)
        expected_losses.append(expected["mean_doc_loss"])
        expected_automation.append(expected["automated"])
        expected_lambdas.append(lam)

    result = simulate(table, P_COL, costs, mode, alpha, n_splits=5, seed=3)
    assert result["mean_test_loss"] == pytest.approx(np.mean(expected_losses))
    assert result["standard_error"] == pytest.approx(np.std(expected_losses, ddof=1) / np.sqrt(5))
    assert result["mean_automation"] == pytest.approx(np.mean(expected_automation))
    assert result["mean_lambda_hat"] == pytest.approx(np.mean(expected_lambdas))


def test_cli_writes_synthetic_report_with_uncertified_fallback(tmp_path):
    cache, results = tmp_path / "cache", tmp_path / "results"
    (cache / "risk").mkdir(parents=True)
    table = _risk_table(12)
    adversarial = _risk_table(600).assign(split="adv_test", field="outside_syn_test")
    pd.concat([table, adversarial]).to_parquet(cache / "risk" / "frontier.parquet", index=False)

    assert crc_script.main([
        "--model", "frontier", "--variant", "full_logistic", "--mode", "trade",
        "--cache", str(cache), "--results", str(results),
    ]) == 0
    report = (results / "crc_frontier.md").read_text()
    assert "Synthetic results are a code check only" in report
    assert "real_cal documents in Step 21" in report
    assert "Documents straight-through" in report
    rows = [line for line in report.splitlines() if line.startswith("| 0.")]
    assert len(rows) == 4
    for alpha, row in zip((0.01, 0.02, 0.05, 0.1), rows):
        assert row == f"| {alpha:g} | {floor(6):.6g} | 0 | 0 ± 0 | 0.00% | 0.00% | Pass |"
