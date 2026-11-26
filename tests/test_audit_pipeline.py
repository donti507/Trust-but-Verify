"""Offline audit fixtures with separate generated collections."""
import json
import shutil

import joblib
import pandas as pd
import pytest

from scripts import audit_pipeline as audit
from scripts import crc, decide
from test_demo import _write_inputs, _write_jsonl
from trustswap.decide.policy import COSTS_FILE
from trustswap.score.risk import metrics
from trustswap.verify.evaluate import verifier_hash

MODEL = "open_weights"
NOTIONAL = "floating_leg.notional_amount"


class FixtureRiskModel:
    def __init__(self, variant):
        self.column = f"u_cached_{variant}"

    def predict(self, table):
        return table[self.column].to_numpy(float)


def _fixture(tmp_path):
    source, cache = _write_inputs(tmp_path, extra_samples=True)
    snapshot = cache / "audit_inputs"
    shutil.move(source, snapshot)
    for gold_path in snapshot.glob("*/gold.jsonl"):
        gold = audit.json_lines(gold_path)
        for row in gold:
            row["generator"] = {"seed": 1, "sampler_version": "fixture"}
            row["trade"]["fixed_leg"] = {}
            row["trade"]["trade_date"]["status"] = "explicit"
            row["trade"]["floating_leg"]["notional_amount"]["status"] = "explicit"
        _write_jsonl(gold_path, gold)
    risk_path = cache / "risk" / f"{MODEL}.parquet"
    risk = pd.read_parquet(risk_path)
    risk["p_base_logistic"] = risk["p_full_logistic"] / 2
    # An out-of-scope row with invalid schema would fail if it were ever audited.
    risk = pd.concat([risk, risk.iloc[:1].assign(split="real_test", collection="real", doc_id="blocked",
                                               field="invalid_field")], ignore_index=True)
    risk.to_parquet(risk_path, index=False)
    features = risk.drop(columns=["p_base_logistic", "p_full_logistic"]).assign(
        u_cached_base_logistic=risk["p_base_logistic"], u_cached_full_logistic=risk["p_full_logistic"],
    )
    feature_path = cache / "features" / f"{MODEL}.parquet"
    feature_path.parent.mkdir()
    features.to_parquet(feature_path, index=False)
    allowed = risk.loc[risk["split"].isin(audit.SPLITS)]
    results = tmp_path / "results"
    models = results / "models" / MODEL
    models.mkdir(parents=True)
    stats = {}
    for variant in ("base_logistic", "full_logistic"):
        joblib.dump(FixtureRiskModel(variant), models / f"{variant}.joblib")
        for split, rows in allowed.groupby("split"):
            stats.setdefault(split, {})[variant] = metrics(rows["e"], rows[f"p_{variant}"])
    (models / "metrics.json").write_text(json.dumps(stats), encoding="utf-8")

    decision_dir = results / f"{MODEL}_full_logistic_lambda1_adv_test"
    decisions = allowed.loc[allowed["split"].eq("adv_test")]
    _write_jsonl(decision_dir / "decisions.jsonl",
                 decide.decision_log(decisions, "p_full_logistic", audit.load_costs(COSTS_FILE), 1, cache, MODEL))
    crc.main(["--model", MODEL, "--cache", str(cache), "--results", str(results), "--mode", "trade"])
    _write_jsonl(cache / "verify_injected" / MODEL / "swap-conflict.jsonl", [
        {"doc_id": "swap-conflict", "injection": {"kind": "scale", "fields": [NOTIONAL]},
         "rules": {"V5": {NOTIONAL: {"verdict": "SOFT"}}}},
        {"doc_id": "swap-conflict", "injection": {"kind": "scale", "fields": [NOTIONAL]},
         "rules": {"V5": {"trade_date": {"verdict": "HARD"}}}},
    ])
    freeze = tmp_path / "verifier-v1.0.json"
    freeze.write_text(json.dumps({"status": "frozen", "verifier_hash": verifier_hash()}), encoding="utf-8")
    return cache, results, freeze, feature_path, decision_dir


@pytest.fixture
def cached_fixture(tmp_path):
    return _fixture(tmp_path)


def _run(cache, results, freeze, *_):
    code = audit.main(["--cache", str(cache), "--results", str(results), "--freeze", str(freeze),
                       "--model", MODEL])
    report = (results / "audit" / "report.md").read_text(encoding="utf-8")
    return code, report


def test_audit_checks_fixture_and_rebuilds_demo(cached_fixture, monkeypatch):
    parquet = pd.read_parquet

    def checked_read(path, **kwargs):
        assert kwargs["filters"] == [("split", "in", tuple(audit.SPLITS))]
        return parquet(path, **kwargs)

    def guarded_read(path, **kwargs):
        filters = kwargs["filters"]
        if filters and isinstance(filters[0], list):
            assert all("real" not in str(block) for block in filters)
            return parquet(path, **kwargs)
        return checked_read(path, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    code, report = _run(*cached_fixture)
    assert code == 0, report
    assert report.count("**PASS**") == 9
    assert "adversarial, scale: 1/2 caught (50.0%)" in report
    assert "AUROC" in report and "ECE" in report
    assert "fallback verified" in report
    assert "synthetic at λ=1:" in report and "adversarial at λ=1:" in report
    assert "blocked" not in report
    cache, results, *_ = cached_fixture
    assert (results / "audit" / MODEL / "index.html").is_file()
    assert not (cache.parent / "data").exists()


def test_audit_reports_tampered_metrics_labels_decisions_and_crc(cached_fixture):
    cache, results, freeze, feature_path, decision_dir = cached_fixture
    features = pd.read_parquet(feature_path)
    row = features["doc_id"].eq("swap-conflict") & features["field"].eq(NOTIONAL)
    features.loc[row, "e"] = 0
    features.to_parquet(feature_path, index=False)
    decision_file = decision_dir / "decisions.jsonl"
    logs = audit.json_lines(decision_file)
    logs[0]["action"] = "ESCALATE" if logs[0]["action"] == "ACCEPT" else "ACCEPT"
    _write_jsonl(decision_file, logs)
    crc_path = results / f"crc_{MODEL}.md"
    text = crc_path.read_text(encoding="utf-8").replace("| 0.01 |", "| 0.03 |")
    crc_path.write_text(text, encoding="utf-8")
    code, report = _run(cache, results, freeze)
    assert code == 1
    assert "Gold label disagrees" in report and "error labels differ" in report
    assert "action disagrees" in report and "CRC report differs" in report


def test_audit_rejects_wrong_collection_and_reports_missing_evidence(cached_fixture):
    cache, results, freeze, feature_path, _ = cached_fixture
    features = pd.read_parquet(feature_path)
    features.loc[features["split"].eq("syn_test"), "collection"] = "adversarial"
    features.to_parquet(feature_path, index=False)
    freeze.write_text(json.dumps({"status": "draft", "verifier_hash": verifier_hash()}), encoding="utf-8")
    code, report = _run(cache, results, freeze)
    assert code == 1
    assert "verifier record is still draft" in report
    assert "A split contains rows from the wrong collection" in report
    assert "Feature cache is missing or empty" in report



def test_audit_detects_saved_auc_discrepancy(cached_fixture):
    cache, results, freeze, *_ = cached_fixture
    path = results / "models" / MODEL / "metrics.json"
    metrics_report = audit.json_file(path)
    metrics_report["syn_test"]["full_logistic"]["auroc"] += 0.000002
    path.write_text(json.dumps(metrics_report), encoding="utf-8")
    code, report = _run(cache, results, freeze)
    assert code == 1
    assert "synthetic syn_test full_logistic: reported AUROC differs" in report


def test_certified_crc_rejects_a_nonmaximal_passing_threshold(tmp_path, monkeypatch):
    cache, results = tmp_path / "cache", tmp_path / "results"
    path = cache / "risk" / f"{MODEL}.parquet"
    path.parent.mkdir(parents=True)
    rows = pd.DataFrame([
        {"doc_id": f"d{i}", "field": NOTIONAL, "split": "syn_test", "collection": "synthetic",
         "p_full_logistic": 0.9 if i == 0 else 0.001, "e": int(i == 0)}
        for i in range(40)
    ])
    rows.to_parquet(path, index=False)
    crc.main(["--model", MODEL, "--cache", str(cache), "--results", str(results), "--mode", "field"])
    costs = audit.load_costs(COSTS_FILE)
    assert "α=0.05: 1000/1000 certified" in audit.check_crc(rows, costs, results, MODEL)
    original = audit.calibrate_lambda

    def nonmaximal(loss, grid, alpha):
        chosen, certified = original(loss, grid, alpha)
        return (0.0, certified) if certified else (chosen, certified)

    monkeypatch.setattr(audit, "calibrate_lambda", nonmaximal)
    with pytest.raises(ValueError, match="largest passing prefix"):
        audit.check_crc(rows, costs, results, MODEL)
