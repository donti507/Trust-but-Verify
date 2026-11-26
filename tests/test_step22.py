"""Step 22 checks using generated tables and local cache fixtures."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import hypotheses as cli
from trustswap.decide.policy import load_costs
from trustswap.eval.hypotheses import (
    COVERAGES, cluster_bootstrap, critical_errors, likelihood_ratio,
    matched_automation, ranking_transfer, silent_shares,
)

NOTIONAL = "floating_leg.notional_amount"
DATE = "trade_date"


def test_lrt_detects_added_signal_and_ignores_noise_and_constants():
    rng = np.random.default_rng(22)
    n = 1800
    signal, noise, base = rng.normal(size=(3, n))
    p = 1 / (1 + np.exp(-2 * signal - 0.2 * base))
    e = rng.binomial(1, p)
    table = pd.DataFrame({"u_base": base, "e": e, "v_constant": 1})
    informative = likelihood_ratio(table.assign(rho_added=signal))
    null = likelihood_ratio(table.assign(rho_added=noise))
    assert informative["df"] == null["df"] == 1
    assert informative["lambda"] > 0 and informative["p"] < 0.01
    assert null["p"] > 0.05


def test_paired_cluster_bootstrap_covers_known_mean_difference():
    rng = np.random.default_rng(116)
    n = 100
    first = rng.normal(size=(n, 1)) + rng.normal(size=(n, 3))
    second = first + 0.3 + rng.normal(0, 0.2, size=(n, 1))
    values = (second - first).ravel()
    ids = np.repeat(np.arange(n), 3)

    def statistic(rows, docs):
        assert len(np.unique(docs)) == n
        assert np.all(np.bincount(docs) == 3)
        assert np.all(ids[rows].reshape(n, 3) == ids[rows].reshape(n, 3)[:, :1])
        return values[rows].mean()

    result = cluster_bootstrap(ids, statistic, n_resamples=400, seed=22)
    low, high = result.interval()
    assert low < 0.3 < high
    assert result.estimate == pytest.approx(values.mean())


def test_matched_automation_with_many_ties_and_all_cost_variants():
    costs = load_costs()
    n = 50
    # Blind ranking accepts all critical errors before any cheap fields.
    table = pd.DataFrame({
        "doc_id": np.tile(np.arange(n), 2), "field": [NOTIONAL] * n + [DATE] * n,
        "p_full_logistic": [0.2] * n + [0.3] * n, "e": [1] * n + [0] * n,
    })
    for config in (costs, costs.scaled(0.5), costs.scaled(2), costs.reordered(("major", "critical", "minor"))):
        fit = matched_automation(table, config, n_resamples=12)
        np.testing.assert_allclose(fit.estimate[:, 0], COVERAGES, atol=0.01)
        np.testing.assert_allclose(fit.draws[:, :, 0], np.broadcast_to(COVERAGES, (12, 3)), atol=0.01)
        if config is costs:
            assert (fit.estimate[:, 3] < 0).all()


def test_critical_error_filter_and_wilson_intervals():
    table = pd.DataFrame({
        "field": [NOTIONAL] * 5 + [DATE, NOTIONAL],
        "e": [1, 1, 1, 1, 0, 1, 1],
        "defect": ["scale"] * 3 + ["ocr"] * 4,
        "rules": [[], [], ["V5:HARD"], [], [], [], []],
    })
    errors = critical_errors(table, [True] * 6 + [False])
    shares = silent_shares(errors)
    assert shares["Overall"][:3] == (3, 4, 0.75)
    assert shares["Overall"][3:] == pytest.approx((0.30064184, 0.95441274))
    assert shares["scale"][:3] == pytest.approx((2, 3, 2 / 3))
    assert shares["ocr"][3:] == pytest.approx((0.20654931, 1.0))


def _risk_tables(root: Path):
    cache, data = root / "cache", root / "data"
    rng = np.random.default_rng(47)
    rows = []
    for split, collection in (("train", "synthetic"), ("syn_test", "synthetic"),
                              ("adv_test", "adversarial"), ("real_test", "real")):
        for index in range(40):
            for field in (DATE, NOTIONAL):
                flagged = index % 4 == 0
                e = int(index == 0 and field == NOTIONAL) if split == "adv_test" else int(flagged and field == NOTIONAL)
                rows.append({
                    "doc_id": f"{split}-{index:02d}", "split": split, "collection": collection,
                    "field": field, "e": e, "defect": "scale_error" if split == "adv_test" else "none",
                    "k_samples": 2, "u_logprob_missing": 0, "u_logprob": -rng.random(),
                    "u_agreement": rng.random(), "u_verbal_conf": rng.random(),
                    "v_V5_HARD": int(flagged), "rho_V5": rng.random(),
                    "p_base_logistic": rng.random(), "p_full_logistic": 0.02 if field == NOTIONAL else 0.005,
                })
    table = pd.DataFrame(rows)
    for subdir, sample in (
        ("features", table.drop(columns=cli.P_COLS)),
        ("risk", table.loc[table["split"].ne("train")].sample(frac=1, random_state=2)),
    ):
        (cache / subdir).mkdir(parents=True)
        sample.to_parquet(cache / subdir / "open_weights.parquet", index=False)
    doc = "adv_test-00"
    extract, verify = (cache / subdir / "open_weights" for subdir in ("extract", "verify"))
    for path in (extract, verify, data / "adversarial"):
        path.mkdir(parents=True)
    (extract / f"{doc}.jsonl").write_text(json.dumps({"k": 0, "fields": {NOTIONAL: {"value": 50}}}))
    (verify / f"{doc}.json").write_text(json.dumps({"rules": {"V5": {NOTIONAL: {"verdict": "SOFT"}}}}))
    gold = {
        "doc_id": doc, "collection": "adversarial", "generator": {"seed": 1, "sampler_version": "fixture"},
        "trade": {"fixed_leg": {}, "floating_leg": {"notional_amount": {"value": 25, "status": "explicit"}}},
    }
    (data / "adversarial" / "gold.jsonl").write_text(json.dumps(gold))
    return cache, data, table


def test_ranking_transfer_for_identical_collections(tmp_path):
    _, _, table = _risk_tables(tmp_path)
    sample = table.loc[table["split"].eq("syn_test")]
    result, areas = ranking_transfer(sample, sample.copy(), load_costs(), n_resamples=12)
    assert len(areas[0]) == 7
    assert result.estimate == pytest.approx(1)
    assert np.isfinite(result.interval()).all()


def test_cli_report_has_one_verdict_per_hypothesis_and_audits_errors(tmp_path):
    cache, data, _ = _risk_tables(tmp_path)
    result = tmp_path / "results"
    result.mkdir()
    # An unfrozen report must never open even a pre-existing real evaluation.
    (result / "real_eval.json").write_text("unreadable JSON")
    prereg = tmp_path / "hypotheses.yaml"
    options = ["--cache", str(cache), "--data", str(data), "--results", str(result),
               "--prereg", str(prereg), "--n-resamples", "12"]
    assert cli.main(options) == 0
    report_path = result / "hypotheses.md"
    report = report_path.read_text()
    assert "draft, not pre-registered" in report and "pending Step 21" in report
    assert re.findall(r"^## (H\d) —", report, re.MULTILINE) == list(cli.WORDING)
    assert report.count("**Verdict:") == len(cli.WORDING)
    assert all(source in report for source in ("scripts/hypotheses.py", "features/open_weights.parquet",
                                                "risk/open_weights.parquet", "gold.jsonl"))
    audit = pd.read_csv(result / "h4_errors_open_weights.csv", keep_default_na=False)
    assert list(audit.columns) == cli.AUDIT_COLUMNS
    assert audit.iloc[0].to_dict() == {
        "doc_id": "adv_test-00", "field": NOTIONAL, "defect": "scale_error",
        "value": 50, "gold": 25, "rules": '["V5:SOFT"]', "rule_gap": "",
    }

    prereg.write_text(yaml.safe_dump({
        "status": "frozen",
        "hypotheses": {key: {"wording": f"Registered wording for {key}", "alpha": 0.01} for key in cli.WORDING},
    }))
    (result / "real_eval.json").write_text(json.dumps({"mean_doc_loss": 0.02}))
    assert cli.main(options + ["--include-real"]) == 0
    report = report_path.read_text()
    assert "**Verdict: not supported.** α = 0.01" in report.split("## H3")[1].split("## H4")[0]
    assert "Registered wording for H1" in report
    assert "syn_test versus real_test" in report


def test_real_test_predicate_requires_frozen_registration(tmp_path, monkeypatch):
    cache, _, _ = _risk_tables(tmp_path)
    read_parquet = pd.read_parquet
    calls = []

    def check_filters(*args, **kwargs):
        calls.append(kwargs["filters"])
        return read_parquet(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", check_filters)
    with pytest.raises(ValueError, match="frozen"):
        cli.load_tables(cache, "open_weights", status="draft", include_real=True)
    assert calls == []
    _, tables, *_ = cli.load_tables(cache, "open_weights", status="frozen")
    assert set(tables) == {"syn_test", "adv_test"}
    assert all("real_test" not in str(filters) for filters in calls)
