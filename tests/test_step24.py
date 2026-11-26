"""Step 24 checks for reproducible plots and the real-test gate."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from trustswap.decide.policy import evaluate_lambda, field_costs, load_costs
from trustswap.eval.baselines import METHODS
from trustswap.eval.plots import (
    frontier_stats, main, real_test_ready, reliability_bins, taxonomy_table, verifier_recall,
)

MODEL = "open_weights"
SPLIT = "syn_test"
NOTIONAL = "floating_leg.notional_amount"
DATE = "trade_date"


def _fixtures(root):
    cache, data, results, prereg = (root / folder for folder in ("cache", "data", "results", "prereg"))
    paths = (cache / "risk", cache / "features", cache / "verify" / MODEL, data / "adversarial",
             results / "curves", prereg)
    for path in paths:
        path.mkdir(parents=True)
    table = pd.DataFrame([
        {"doc_id": f"d{doc}", "field": field, "collection": "synthetic", "split": SPLIT,
         "e": int(doc == 0 and field == NOTIONAL), "k_samples": 3, "v_V5_HARD": int(doc == 0),
         "u_logprob": -0.1 * (doc + 1), "u_logprob_missing": 0, "u_agreement": 0.8,
         "u_verbal_conf": 0.75, "p_base_logistic": 0.04 * (doc + 1), "p_full_logistic": 0.06 * (doc + 1)}
        for doc in range(4) for field in (DATE, NOTIONAL)
    ])
    # The plots are scoped to the requested split at parquet read time.
    excluded = table.iloc[:1].assign(collection="real", split="real_test", field="excluded")
    caches = pd.concat([table, excluded], ignore_index=True)
    caches.drop(columns=["p_base_logistic", "p_full_logistic"]).to_parquet(paths[1] / f"{MODEL}.parquet")
    caches[["doc_id", "field", "collection", "split", "e", "p_base_logistic", "p_full_logistic"]].to_parquet(
        paths[0] / f"{MODEL}.parquet"
    )
    pd.DataFrame([
        {"method": method, "coverage": coverage, "risk": 0.08, "cost_risk": 0.6}
        for method in METHODS
        for coverage in ((1,) if method == "no_abstention" else (0.75,) if method == "verifin_gate" else (0.4, 1))
    ]).to_csv(paths[4] / f"{MODEL}_{SPLIT}.csv", index=False)
    docs = [
        {"doc_id": doc, "collection": "adversarial", "defect": "scale_error",
         "expected_rules": [expected], "affected_fields": [NOTIONAL], "is_control": control}
        for doc, expected, control in (("a1", "V5", False), ("a2", "V5", False),
                                       ("a3", "V5", True), ("silent", "silent", False))
    ]
    (paths[3] / "docs.jsonl").write_text("\n".join(map(json.dumps, docs)), encoding="utf-8")
    for doc, fired in (("a1", NOTIONAL), ("a2", DATE), ("a3", NOTIONAL), ("silent", NOTIONAL)):
        record = {"rules": {"V5": {fired: {"verdict": "SOFT"}}}}
        (paths[2] / f"{doc}.json").write_text(json.dumps(record), encoding="utf-8")
    pd.DataFrame([
        {"doc_id": "a1", "field": NOTIONAL, "defect": "scale_error", "rules": "[]", "rule_gap": "yes"},
        {"doc_id": "a2", "field": NOTIONAL, "defect": "scale_error", "rules": '["V5:SOFT"]', "rule_gap": ""},
    ]).to_csv(results / f"h4_errors_{MODEL}.csv", index=False)
    return cache, data, results, prereg


def _args(paths):
    cache, data, results, prereg = paths
    return ["--model", MODEL, "--cache", str(cache), "--data", str(data), "--results", str(results),
            "--prereg", str(prereg), "--n-resamples", "8"]


def test_command_creates_all_formats_and_regenerates(tmp_path):
    paths = _fixtures(tmp_path)
    command = [sys.executable, "-m", "trustswap.eval.plots", *_args(paths)]
    for _ in range(2):
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        assert "Wrote figures" in result.stdout
    figures = paths[2] / "figures"
    stems = [*(f"{name}_{MODEL}_{SPLIT}" for name in ("automation_loss", "risk_coverage", "reliability")),
             f"verifier_recall_{MODEL}", f"h4_taxonomy_{MODEL}"]
    for stem in stems:
        assert (figures / f"{stem}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert (figures / f"{stem}.pdf").read_bytes().startswith(b"%PDF-")
    assert len(list(figures.iterdir())) == 11
    taxonomy = (figures / f"h4_taxonomy_{MODEL}.md").read_text(encoding="utf-8")
    assert "| scale error | 2 | 1 | 1 | 1 | 1 |" in taxonomy


def test_real_gate_checks_both_freezes_and_evaluation_before_parquet_read(tmp_path, monkeypatch, capsys):
    paths = _fixtures(tmp_path)
    _, _, results, prereg = paths

    def reject_reads(*args, **kwargs):
        pytest.fail("Real-test cache was accessed before the freeze gate")

    monkeypatch.setattr(pd, "read_parquet", reject_reads)
    verifier = prereg / "verifier-v1.0.json"
    registration = prereg / "hypotheses.yaml"
    for verifier_status, prereg_status, evaluated in (
        ("draft", "frozen", True), ("frozen", "draft", True), ("frozen", "frozen", False),
    ):
        verifier.write_text(json.dumps({"status": verifier_status}), encoding="utf-8")
        registration.write_text(yaml.safe_dump({"status": prereg_status}), encoding="utf-8")
        real = results / "real_eval.json"
        if evaluated:
            real.write_text("{}", encoding="utf-8")
        else:
            real.unlink(missing_ok=True)
        with pytest.raises(SystemExit, match="2"):
            main(_args(paths) + ["--split", "real_test"])
        assert "requires a frozen verifier" in capsys.readouterr().err
    real.write_text("{}", encoding="utf-8")
    assert real_test_ready(prereg, results)


def test_frontier_agrees_with_policy_and_resamples_whole_documents():
    data = pd.DataFrame({"doc_id": ["a", "a", "b"], "field": [NOTIONAL, DATE, DATE],
                         "e": [1, 1, 0], "p": [0.02, 0.01, 0.01]})
    costs = load_costs()
    c, _ = field_costs(data, costs)
    x, y, low, high = frontier_stats(data["doc_id"], data["p"] * c, data["e"] * c,
                                     costs.B, "ours", n_resamples=120)
    assert x.tolist() == pytest.approx([0, 2 / 3, 1])
    for i, lam in enumerate((0, 0.1, 2)):
        expected = evaluate_lambda(data, "p", lam, costs, "field")
        assert y[i] == pytest.approx(expected["mean_doc_loss"])
        assert x[i] == pytest.approx(expected["automated"])
    assert low[-1] == 0 and high[-1] == 1


def test_recall_uses_affected_fields_and_excludes_controls_and_silent(tmp_path):
    cache, data, *_ = _fixtures(tmp_path)
    assert verifier_recall(data / "adversarial" / "docs.jsonl", cache / "verify" / MODEL) == {
        "scale_error": [1, 2],
    }


def test_reliability_bins_and_empty_taxonomy():
    predicted, observed = reliability_bins([0.0, 1 / 15, 0.5, 1.0], [0, 1, 0, 1])
    np.testing.assert_allclose(predicted, [1 / 30, 0.5, 1.0])
    np.testing.assert_allclose(observed, [0.5, 0.0, 1.0])
    assert taxonomy_table(pd.DataFrame(columns=["defect", "field", "rules", "rule_gap"])) == [
        ("Total", 0, 0, 0, 0, 0),
    ]
