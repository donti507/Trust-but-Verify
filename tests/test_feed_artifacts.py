"""Portable scoring parity and export gates, using offline fixtures only."""
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts import daily_feed, export_feed_artifacts
from scripts.build_demo import render_page
from trustswap.feed.artifacts import PortableRisk, export_model, replay_from_page
from trustswap.score.risk import fit

COLS = ("u_agreement", "v_V1_HARD")


@pytest.fixture
def fitted():
    rng = np.random.default_rng(104)
    samples = pd.DataFrame(rng.normal(size=(320, 2)), columns=COLS)
    samples["e"] = (samples[COLS[0]] + samples[COLS[1]] * 0.7 + rng.normal(size=len(samples)) > 0).astype(int)
    return fit(samples[:220], samples[220:], "full", "logistic"), samples


def documents():
    return [{
        "id": f"syn-{i:02d}", "source_path": "/unused/source", "category": "Clean synthetic",
        "collection": "synthetic", "split": "syn_test", "text": "Synthetic Party A · USD 50",
        "fields": [{"field": "floating_leg.notional_amount", "name": "Notional amount (floating leg)",
                    "value": 50, "raw_value": "USD 50", "span": [20, 26], "p": 0.04,
                    "c_bar": 100, "r": 1, "e": 0, "severity": "critical", "rules_fired": []}],
    } for i in range(30)]


def test_portable_predictions_match_saved_model_to_one_billionth(fitted):
    model, samples = fitted
    measurements = pd.concat([samples, pd.DataFrame([[-1e8, 0.0], [1e8, 0.0]], columns=COLS)])
    payload = json.loads(json.dumps(export_model(model)))
    portable = PortableRisk.from_payload(payload)
    assert portable.columns == COLS
    np.testing.assert_allclose(portable.predict(measurements), model.predict(measurements), atol=1e-9, rtol=0)
    payload["logistic"]["weights"][0] += 0.01
    with pytest.raises(ValueError, match="hash"):
        PortableRisk.from_payload(payload)


def test_export_reads_current_pages_ref_and_only_portable_fields(fitted, tmp_path, monkeypatch):
    model, samples = fitted
    saved = tmp_path / "full_logistic.joblib"
    joblib.dump(model, saved)
    html = render_page(documents(), 100, feed={"days": [{"collection": "live_feed"}], "history": []})

    def git_show(args, **kwargs):
        assert args == ["git", "show", "origin/gh-pages:index.html"]
        return SimpleNamespace(stdout=html)

    monkeypatch.setattr(export_feed_artifacts.subprocess, "run", git_show)
    out = tmp_path / "artifacts"
    assert export_feed_artifacts.main(["--model", str(saved), "--output", str(out)]) == 0
    assert {item.name for item in out.iterdir()} == {"model.json", "replay.json"}
    portable = PortableRisk.from_payload(json.loads((out / "model.json").read_text()))
    np.testing.assert_allclose(portable.predict(samples), model.predict(samples), atol=1e-9, rtol=0)
    replay = json.loads((out / "replay.json").read_text())
    assert len(replay["documents"]) == 30
    assert all("raw_value" not in item["fields"][0] and "source_path" not in item for item in replay["documents"])
    assert "live_feed" not in (out / "replay.json").read_text()


@pytest.mark.parametrize("override", [
    {"collection": "real"}, {"split": "real_test"}, {"id": "real-001"}, {"id": "real/doc"},
    {"category": "Real"},
])
def test_export_rejects_real_replay_documents(override):
    docs = documents()
    docs[5].update(override)
    with pytest.raises(ValueError, match="document|collection"):
        replay_from_page(render_page(docs, 100))


def test_runner_skips_missing_bundle_before_downloading(tmp_path, monkeypatch, capsys):
    def unexpected(*args):
        pytest.fail("DTCC download called before checking artifact availability")
    monkeypatch.setattr(daily_feed.dtcc, "download", unexpected)
    assert daily_feed.main(["--date", "2026-11-23", "--feed-dir", str(tmp_path / "feed"),
                            "--site", str(tmp_path / "index.html")]) == 0
    assert "bundle is missing" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_runner_rejects_tampered_bundle_before_fetch(fitted, tmp_path, monkeypatch):
    model, _ = fitted
    bundle = tmp_path / "artifacts"
    bundle.mkdir()
    payload = export_model(model)
    payload["logistic"]["intercept"] += 1
    (bundle / "model.json").write_text(json.dumps(payload))
    replay = replay_from_page(render_page(documents(), 100))
    (bundle / "replay.json").write_text(json.dumps(replay))
    monkeypatch.setattr(daily_feed.dtcc, "download", lambda _: pytest.fail("Invalid bundle caused a download"))
    with pytest.raises(ValueError, match="hash"):
        daily_feed.main(["--date", "2026-11-23", "--feed-dir", str(tmp_path / "feed"),
                         "--site", str(tmp_path / "index.html"), "--cache", str(tmp_path / "cache")])
    assert not (tmp_path / "feed").exists() and not (tmp_path / "index.html").exists()
