"""Step 15: cost model and decision rule."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import decide as D
from trustswap.decide.policy import (Costs, escalate, evaluate_lambda, field_costs, frontier, lambda_grid,
                                     load_costs)


def _risk_table(n_docs=40, seed=0):
    rng = np.random.default_rng(seed)
    fields = ["trade_date", "fixed_leg.notional_amount", "notice_address", "floating_leg.spread"]
    rows = []
    for d in range(n_docs):
        for f in fields:
            p = float(rng.uniform(0, 0.4))
            rows.append({"doc_id": f"d{d}", "field": f, "split": "adv_test", "e": int(rng.random() < p),
                         "p_full_logistic": p})
    return pd.DataFrame(rows)


def test_costs_load_with_placeholder_ratios():
    costs = load_costs()
    assert costs.status == "placeholder"
    assert costs.c_bar("fixed_rate") == 100 and costs.c_bar("trade_date") == 10 and costs.c_bar("notice_address") == 0.2
    assert costs.r("trade_date") == 1.0 and costs.B > 0


def test_escalation_rule():
    assert list(escalate([0.5, 0.01, 0.0], [10, 10, 10], [1, 1, 1], 1.0)) == [True, False, False]
    assert list(escalate([0.2, 0.0], [1, 1], [1, 1], 0.0)) == [True, False]  # lambda 0: every p > 0 escalated


def test_lambda_zero_escalates_all_positive_risks_and_has_zero_loss():
    table = _risk_table()
    costs = load_costs()
    for mode in ("field", "trade"):
        result = evaluate_lambda(table, "p_full_logistic", 0.0, costs, mode)
        assert result["mean_doc_loss"] == 0.0 and result["errors_accepted"] == 0
    assert evaluate_lambda(table, "p_full_logistic", 0.0, costs, "field")["automated"] == pytest.approx(
        (table["p_full_logistic"] == 0).mean())


def test_loss_never_decreases_and_automation_never_falls_as_lambda_grows():
    table = _risk_table()
    costs = load_costs()
    for mode in ("field", "trade"):
        curve = frontier(table, "p_full_logistic", costs, mode)
        losses = [r["mean_doc_loss"] for r in curve]
        automated = [r["automated"] for r in curve]
        assert all(a <= b + 1e-12 for a, b in zip(losses, losses[1:]))
        assert all(a <= b + 1e-12 for a, b in zip(automated, automated[1:]))
        assert all(0 <= r["mean_doc_loss"] <= 1 for r in curve)
    huge = evaluate_lambda(table, "p_full_logistic", 1e9, costs, "field")
    assert huge["automated"] == 1.0 and huge["errors_accepted"] == table["e"].sum()


def test_trade_mode_reviews_whole_document():
    table = pd.DataFrame({"doc_id": ["a", "a", "b", "b"], "field": ["trade_date"] * 4, "split": "x",
                          "e": [0, 1, 0, 0], "p": [0.9, 0.0, 0.0, 0.0]})
    costs = load_costs()
    field = evaluate_lambda(table, "p", 1.0, costs, "field")
    trade = evaluate_lambda(table, "p", 1.0, costs, "trade")
    assert field["errors_accepted"] == 1 and field["mean_doc_loss"] > 0  # wrong field in doc a was accepted
    assert trade["mean_doc_loss"] == 0 and trade["automated"] == 0.5     # doc a reviewed whole, doc b straight-through


def test_cost_sensitivity_variants():
    costs = load_costs()
    assert costs.scaled(2.0).c_bar("fixed_rate") == 200 and costs.scaled(2.0).r("fixed_rate") == 1.0
    swapped = costs.reordered(("major", "critical", "minor"))
    assert swapped.c_bar("fixed_rate") == 10 and swapped.c_bar("trade_date") == 100
    c, r = field_costs(_risk_table(2), costs)
    assert list(c[:4]) == [10, 100, 0.2, 100] and set(r) == {1.0}


def test_grid_matches_runbook():
    grid = lambda_grid()
    assert len(grid) == 201 and grid[0] == 0 and grid[1] == pytest.approx(1e-3) and grid[-1] == pytest.approx(1e3)


def test_script_writes_report_and_decision_log(tmp_path):
    cache, results = tmp_path / "cache", tmp_path / "results"
    (cache / "risk").mkdir(parents=True)
    table = _risk_table(10)
    table["model"], table["collection"], table["defect"], table["is_control"] = "m", "adversarial", "none", 0
    table.to_parquet(cache / "risk" / "open_weights.parquet", index=False)
    assert D.main(["--model", "open_weights", "--cache", str(cache), "--results", str(results)]) == 0
    report = (results / "decision_open_weights.md").read_text()
    assert "### adv_test" in report and "Cost sensitivity" in report and "status: placeholder" in report
    log = [json.loads(l) for l in (results / "open_weights_full_logistic_lambda1_adv_test" / "decisions.jsonl")
           .read_text().splitlines()]
    assert len(log) == len(table)
    assert {"doc_id", "field", "value", "p_hat", "c_bar", "lambda", "action", "rules_fired", "evidence_span"} <= set(log[0])
    assert {r["action"] for r in log} <= {"ACCEPT", "ESCALATE"}
