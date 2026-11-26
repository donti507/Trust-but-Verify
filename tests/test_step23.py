"""Step 23 offline checks using generated features and extraction records."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import ablations as cli
from trustswap.decide.policy import evaluate_lambda, field_costs, load_costs
from trustswap.eval.ablations import (
    FAMILIES, P_COL, drop_family, fit_full, paired_models, prediction_deltas, review_mode_deltas,
    with_sample_count,
)

FIELDS = ("trade_date", "floating_leg.notional_amount")


def feature_table(seed=23, n_docs=70):
    rng = np.random.default_rng(seed)
    rows = []
    for split, collection in (("train", "synthetic"), ("dev", "synthetic"), ("syn_test", "synthetic"),
                              ("adv_test", "adversarial")):
        for doc_id in range(n_docs):
            for field in FIELDS:
                signal = rng.normal()
                e = int(rng.random() < 1 / (1 + np.exp(-3 * signal)))
                rows.append({
                    "doc_id": f"{split}-{doc_id}", "field": field, "split": split, "collection": collection,
                    "e": e, "k_samples": 5, "rho_V1": signal, "v_V1_HARD": int(signal > 0),
                    "rho_V2": rng.normal(), "v_V2_SOFT": 0, "g_span_found": rng.integers(2),
                    "u_agreement": 0.6, "u_agreement_missing": 0, "u_logprob": rng.normal(),
                })
    return pd.DataFrame(rows)


def test_drop_informative_rule_lowers_auroc_and_drop_noise_interval_includes_zero():
    data = feature_table(n_docs=250)
    test = data.loc[data["split"].eq("syn_test")]
    c, _ = field_costs(test, load_costs())
    p_full = fit_full(data).predict(test)
    for family in ("V1", "V2"):
        variant = drop_family(data, family)
        assert f"rho_{family}" not in variant and not any(name.startswith(f"v_{family}_") for name in variant)
        assert f"rho_{'V2' if family == 'V1' else 'V1'}" in variant
        p_drop = fit_full(variant).predict(drop_family(test, family))
        result = prediction_deltas(test["doc_id"], p_full, test["e"], p_drop, test["e"], c, n_resamples=160)
        low, high = result.interval()[:, 0]
        if family == "V1":
            assert result.estimate[0] < 0 and high < 0
        else:
            assert low <= 0 <= high
    assert "g_span_found" not in drop_family(data, "grounding")
    assert "v_V1_HARD" in drop_family(data, "grounding")


def test_recompute_agreement_matches_first_k_and_step13_missing_convention():
    field = FIELDS[1]
    table = pd.DataFrame({"doc_id": ["a", "b", "c"], "field": [field] * 3, "e": [0, 1, 0],
                          "u_agreement": [0.1, 0.2, 0.3], "u_agreement_missing": [0, 0, 0]})

    def record(k, value, valid=True):
        return {"k": k, "schema_valid": valid, "fields": {field: {"value": value}}}

    records = {
        "a": [record(4, 50), record(2, "25.00"), record(1, 50), record(0, 25), record(3, 50)],
        "b": [record(0, 10, False), record(1, 15), record(2, 15)],
        "c": [record(0, 20), record(2, 20)],
    }
    one = with_sample_count(table, records, 1)
    assert one["u_agreement"].tolist() == [1.0, 0.0, 1.0]
    assert one["u_agreement_missing"].tolist() == [0, 1, 0]
    three = with_sample_count(table, records, 3)
    assert three["doc_id"].tolist() == ["a", "b"]
    assert three["u_agreement"].tolist() == pytest.approx([2 / 3, 1.0])
    assert three["e"].tolist() == [0, 1]
    five = with_sample_count(table, records, 5)
    assert five["doc_id"].tolist() == ["a"]
    assert five["u_agreement"].tolist() == [0.6]
    assert table["u_agreement"].tolist() == [0.1, 0.2, 0.3]


def test_model_comparison_uses_only_shared_documents_and_each_models_labels():
    left = pd.DataFrame({"doc_id": ["a", "a", "b"], "field": [*FIELDS, FIELDS[0]],
                         P_COL: [0.1, 0.9, 0.3], "e": [0, 1, 1]})
    right = left.iloc[:2].assign(e=[1, 0])
    result, count = paired_models(left, right, load_costs(), n_resamples=3)
    assert count == 1
    assert result.estimate[0] == -1.0


def test_crc_comparison_matches_policy_on_held_out_halves():
    table = feature_table().query("split == 'syn_test'").copy()
    table[P_COL] = np.where(table["e"].eq(1), 0.9, 0.01)
    # Keep the same two fields per document to compare the automation denominators.
    costs = load_costs()
    seed = 23
    delta, details = review_mode_deltas(table, costs, n_resamples=12, seed=seed)
    ids = np.unique(table["doc_id"])
    halves = np.array_split(np.random.default_rng(seed).permutation(len(ids)), 2)
    for mode, result in details.items():
        held_metrics = []
        for fold, test in zip(result["folds"], halves[::-1]):
            held_out = table.loc[table["doc_id"].isin(ids[test])]
            policy = evaluate_lambda(held_out, P_COL, fold["lambda"], costs, mode)
            automated_metric = 0 if mode == "field" else 1
            held_metrics.append([policy["automated"], policy["review_cost_per_doc"], policy["mean_doc_loss"]])
            assert fold["certified"]
        assert np.mean(held_metrics, axis=0) == pytest.approx(result["metrics"][[automated_metric, 2, 3]])
    np.testing.assert_allclose(delta.estimate, details["trade"]["metrics"] - details["field"]["metrics"])
    assert np.isfinite(delta.interval()).all()


def test_cli_emits_delta_and_ci_for_every_available_ablation(tmp_path):
    cache = tmp_path / "cache"
    results = tmp_path / "results"
    (cache / "features").mkdir(parents=True)
    for model, num_samples, seed in (("open_weights", 3, 23), ("frontier", 5, 24)):
        table = feature_table(seed, n_docs=40)
        unrelated = table.iloc[:1].assign(doc_id="../excluded", split="real_test", collection="real")
        pd.concat([table, unrelated], ignore_index=True).to_parquet(cache / "features" / f"{model}.parquet")
        path = cache / "extract" / model
        path.mkdir(parents=True)
        for doc_id in table["doc_id"].unique():
            records = [{"k": k, "schema_valid": True,
                        "fields": {field: {"value": k % 2} for field in FIELDS}} for k in range(num_samples)]
            (path / f"{doc_id}.jsonl").write_text("\n".join(map(json.dumps, records)), encoding="utf-8")

    assert cli.main(["--cache", str(cache), "--results", str(results), "--n-resamples", "8"]) == 0
    report = (results / "ablations.md").read_text(encoding="utf-8")
    rows = report.splitlines()
    for model, counts in (("open_weights", (1, 3)), ("frontier", (1, 3, 5))):
        for split in ("syn_test", "adv_test"):
            for label in [*(f"Drop {family}" for family in FAMILIES), *(f"K = {k}" for k in counts)]:
                row = next(row for row in rows if row.startswith(f"| {label} | {model} | {split} |"))
                assert row.count("[") == 3
            review = next(row for row in rows if row.startswith(f"| {model} | {split} |"))
            assert review.count("[") == 4
    paired = [row for row in rows if row.startswith("| Frontier − open-weights |")]
    assert len(paired) == 2 and all(row.count("[") == 3 for row in paired)
    assert "open_weights, K = 5 skipped" in report
    assert "## Pending: needs model calls" in report
    assert "Repair loop" in report and "Memorisation control" in report
    assert "real_test" not in report
