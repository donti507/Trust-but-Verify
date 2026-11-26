"""Step 14: risk models and calibration, on a feature table with a planted signal."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import train_risk as TR
from trustswap.score.risk import columns, ece, evaluate, fit, fit_all, metrics, report


def _table(n=6000, seed=0):
    """Errors depend weakly on uncertainty and strongly on a HARD verifier verdict."""
    rng = np.random.default_rng(seed)
    logprob = rng.normal(-0.3, 0.3, n)
    hard = rng.random(n) < 0.06
    noisy = rng.random(n) < 0.3
    score = -4.0 + 4.0 * hard + 0.8 * noisy - 2.0 * logprob
    e = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)
    split = rng.choice(["train", "dev", "syn_test", "adv_test"], n, p=[0.5, 0.2, 0.15, 0.15])
    return pd.DataFrame({
        "doc_id": [f"d{i // 25}" for i in range(n)], "field": "trade_date", "model": "m",
        "collection": np.where(split == "adv_test", "adversarial", "synthetic"), "split": split,
        "defect": "none", "is_control": 0, "k_samples": 1,
        "u_logprob": logprob, "u_agreement": 1.0, "g_span_found": 1.0, "g_v2_residual": 0.0,
        "v_V2_HARD": hard.astype(float), "v_V2_PASS": (~hard).astype(float), "rho_V2": hard * 1.5,
        "tau_trade_date": 1.0, "tau_sev_major": 1.0, "q_ocr": noisy.astype(float), "q_log_length": 7.0,
        "e": e,
    })


def test_feature_sets_are_nested():
    table = _table(200)
    base, full = set(columns(table, "base")), set(columns(table, "full"))
    assert base < full
    assert not any(c.startswith(("v_", "rho_")) for c in base)
    assert {"v_V2_HARD", "rho_V2"} <= full


def test_ece_and_metrics():
    assert ece([0, 1], [0.0, 1.0]) == 0.0
    assert ece([1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    m = metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["auroc"] == 1.0 and m["positives"] == 2 and 0 <= m["brier"] < 0.05


def test_full_model_uses_verifier_signal_and_is_calibrated():
    table = _table()
    models = fit_all(table)
    assert set(models) == {"base_logistic", "base_lightgbm", "full_logistic", "full_lightgbm"}
    results = evaluate(models, table)
    for split in ("syn_test", "adv_test"):
        r = results[split]
        assert r["full_logistic"]["auroc"] > r["base_logistic"]["auroc"] + 0.05
        assert r["full_lightgbm"]["auroc"] > r["base_lightgbm"]["auroc"] + 0.05
        for m in r.values():
            assert m["ece"] < 0.05  # calibrated on held-out data
    for model in models.values():
        p = model.predict(table)
        assert ((p >= 0) & (p <= 1)).all()
    assert "## syn_test" in report(results, "m", models)


def test_single_class_train_is_refused():
    table = _table(600)
    table.loc[table.split == "train", "e"] = 0
    with pytest.raises(ValueError, match="single class"):
        fit(table[table.split == "train"], table[table.split == "dev"], "base", "logistic")


def test_script_writes_models_metrics_and_predictions(tmp_path):
    cache, results = tmp_path / "cache", tmp_path / "results"
    (cache / "features").mkdir(parents=True)
    table = _table(3000)
    table.loc[table.index[:20], "split"] = "real_test"  # must be ignored
    table.to_parquet(cache / "features" / "open_weights.parquet", index=False)
    assert TR.main(["--model", "open_weights", "--cache", str(cache), "--results", str(results)]) == 0
    assert len(list((results / "models" / "open_weights").glob("*.joblib"))) == 4
    metrics_json = json.loads((results / "models" / "open_weights" / "metrics.json").read_text())
    assert set(metrics_json) == {"dev", "syn_test", "adv_test"}
    predictions = pd.read_parquet(cache / "risk" / "open_weights.parquet")
    assert "real_test" not in set(predictions.split)
    assert {"p_base_logistic", "p_full_lightgbm", "e"} <= set(predictions.columns)
    assert (results / "risk_open_weights.md").read_text().startswith("# Risk models (open_weights)")
