"""Step 18: offline baseline checks on generated predictions."""
import numpy as np
import pandas as pd
import pytest

from scripts import baselines as cli
from trustswap.decide.policy import load_costs
from trustswap.eval.baselines import METHODS, aurc, crc_operating_point, evaluate_baselines, risk_coverage


def _table(n_docs=40, seed=18):
    rng = np.random.default_rng(seed)
    rows = []
    for doc_id in range(n_docs):
        for field in ("trade_date", "fixed_leg.notional_amount", "notice_address"):
            p = rng.uniform(0, 0.4)
            rows.append({
                "doc_id": doc_id, "field": field, "split": "syn_test", "collection": "synthetic",
                "e": int(rng.random() < p), "k_samples": 3, "u_logprob": -p,
                "u_logprob_missing": int(doc_id == 0), "u_agreement": 1 - p, "u_verbal_conf": 1 - p,
                "v_V1_HARD": int(p > 0.35), "v_V2_HARD": int(p < 0.01), "v_V1_SOFT": 1,
                "p_base_logistic": p, "p_full_logistic": p,
            })
    return pd.DataFrame(rows)


def test_all_baselines_produce_curves_and_fixed_methods_have_one_point():
    table = _table()
    costs = load_costs()
    results = evaluate_baselines(table, costs, n_splits=8)
    assert tuple(results) == METHODS
    for method, result in results.items():
        curve = result["curve"]
        assert list(curve.columns) == ["coverage", "risk", "cost_risk"]
        if method in ("no_abstention", "verifin_gate"):
            accepted = pd.Series(True, index=table.index)
            if method == "verifin_gate":
                accepted = table[["v_V1_HARD", "v_V2_HARD"]].eq(0).all(axis=1)
            assert len(curve) == 1 and result["crc"] is None
            assert curve.iloc[0].coverage == pytest.approx(accepted.mean())
            assert curve.iloc[0].risk == pytest.approx(table.loc[accepted, "e"].mean())
        else:
            assert len(curve) > 1 and curve.coverage.iloc[-1] == 1
            assert curve.coverage.is_monotonic_increasing
            assert result["crc"]["certified_fraction"] == 1.0
    assert results["logprob"]["curve"].coverage.iloc[-2] == pytest.approx(1 - 3 / len(table))


def test_self_consistency_requires_repeated_samples():
    table = _table(8)
    costs = load_costs()
    table.loc[table.doc_id == 0, "k_samples"] = 1
    results = evaluate_baselines(table, costs, n_splits=2)
    assert results["self_consistency"]["n_fields"] == len(table) - 3
    table["k_samples"] = 1
    assert "self_consistency" not in evaluate_baselines(table, costs, n_splits=2)


def test_perfect_score_has_zero_aurc_and_equal_scores_share_a_threshold():
    e = [0, 1, 0, 1]
    curve = risk_coverage(e, e, [40, 10, 20, 30])
    assert curve.coverage.tolist() == [0.5, 1.0]
    assert curve.risk.tolist() == [0.0, 0.5]
    assert curve.cost_risk.tolist() == [0.0, 10.0]
    assert aurc(curve) == aurc(curve, weighted=True) == 0.0


def test_random_ranking_has_aurc_near_error_rate():
    rng = np.random.default_rng(2)
    e = np.tile([0, 0, 0, 1, 1], 400)
    curve = risk_coverage(rng.random(len(e)), e, np.ones(len(e)))
    assert aurc(curve) == pytest.approx(e.mean(), abs=0.025)


def test_cost_weighted_ranking_improves_cost_weighted_aurc():
    table = _table(100)
    results = evaluate_baselines(table, load_costs(), n_splits=2)
    assert aurc(results["ours"]["curve"], weighted=True) < aurc(results["extractconf"]["curve"], weighted=True)


def test_crc_uses_calibration_scores_and_held_out_document_loss():
    n = 41
    seed = 47
    test_docs = np.random.default_rng(seed).permutation(n)[n // 2:]
    docs = np.repeat(np.arange(n), 2)
    on_test = np.isin(docs, test_docs)
    scores = np.ones(len(docs))
    scores[docs == test_docs[0]] = np.inf
    result = crc_operating_point(docs, scores, on_test, np.full(len(docs), 70.0), 100.0, n_splits=1, seed=seed)
    assert result["certified_fraction"] == 1.0
    assert result["floor"] == pytest.approx(1 / 21)
    assert result["coverage"] == pytest.approx(20 / 21)
    assert result["mean_doc_loss"] == pytest.approx(20 / 21)
    assert result["risk"] == 1.0
    assert result["cost_risk"] == 70.0


def test_crc_below_floor_escalates_all_fields():
    result = crc_operating_point(np.arange(10), np.full(10, -np.inf), np.ones(10), np.ones(10), 1.0, n_splits=2)
    assert result["floor"] > 0.05
    assert result["certified_fraction"] == 0
    assert result["coverage"] == result["mean_doc_loss"] == 0


def test_cli_joins_shuffled_caches_and_writes_split_reports(tmp_path):
    cache, results = tmp_path / "cache", tmp_path / "results"
    table = pd.concat([
        _table(10).assign(split="dev", doc_id=lambda rows: "dev-" + rows.doc_id.astype(str), k_samples=1),
        _table(10).assign(doc_id=lambda rows: "syn-" + rows.doc_id.astype(str)),
        _table(10).assign(split="adv_test", collection="adversarial", doc_id=lambda rows: "adv-" + rows.doc_id.astype(str)),
        _table(10).assign(split="train", field="excluded", doc_id=lambda rows: "train-" + rows.doc_id.astype(str)),
    ])
    risk_cols = ["doc_id", "field", "split", "p_base_logistic", "p_full_logistic"]
    for directory, data in (
        ("features", table.drop(columns=risk_cols[-2:])),
        ("risk", table[risk_cols].sample(frac=1, random_state=9)),
    ):
        (cache / directory).mkdir(parents=True)
        data.to_parquet(cache / directory / "open_weights.parquet", index=False)

    assert cli.main(["--model", "open_weights", "--cache", str(cache), "--results", str(results)]) == 0
    report = (results / "baselines_open_weights.md").read_text()
    assert "Self-consistency skips 30 fields with k_samples = 1." in report
    assert "CRC at alpha = 0.05" in report
    assert "## train" not in report
    for split in ("dev", "syn_test", "adv_test"):
        assert f"## {split}" in report
        csv = pd.read_csv(results / "curves" / f"open_weights_{split}.csv")
        assert list(csv.columns) == ["method", "coverage", "risk", "cost_risk"]
        expected = evaluate_baselines(table.loc[table.split == split], load_costs(), n_splits=2)
        assert set(csv.method) == set(expected)
        pd.testing.assert_frame_equal(
            csv.loc[csv.method == "extractconf"].drop(columns="method").reset_index(drop=True),
            expected["extractconf"]["curve"],
        )
