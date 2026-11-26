"""Parity and invariants for the actual browser JavaScript, without a DOM."""
from itertools import product
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from scripts import build_demo
from test_demo import _write_inputs
from trustswap.decide.policy import escalate, evaluate_lambda, load_costs

APP = Path(build_demo.FRONTEND) / "app.js"


def run_node(script, data):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed, so demo JavaScript logic cannot be tested")
    source = """
const app = require(process.argv[1]);
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
""" + script
    result = subprocess.run([node, "-e", source, str(APP)], input=json.dumps(data), text=True,
                            capture_output=True, check=True, timeout=15)
    return json.loads(result.stdout)


def test_js_decisions_match_policy_grid_including_ties():
    cases = list(product([0, 0.001, 0.2, 0.5, 1], [0.2, 1, 10, 100], [0, 0.5, 1, 2],
                         [0, 0.001, 0.1, 0.2, 0.5, 1, 2, 10, 100, 1000]))
    actual = run_node("console.log(JSON.stringify(input.map(args => app.decision(...args))));", cases)
    p, c, r, lam = np.asarray(cases).T
    expected = np.where(escalate(p, c, r, lam), "Review", "Accept")
    assert actual == expected.tolist()


def test_slider_document_counts_curves_and_percentages(tmp_path):
    data, cache = _write_inputs(tmp_path, extra_samples=True)
    costs = load_costs()
    docs = build_demo.load_documents(data, cache, "open_weights", costs, 30)
    report = run_node("""
const {docs, bound} = input;
const steps = Array.from({length: 62}, (_, i) => app.lambdaFor(i));
const facts = docs.map(doc => steps.map(lambda => app.facts(doc, lambda, bound)));
const groups = ['synthetic', 'adversarial'].map(type => docs.filter(
    doc => (doc.category === 'Adversarial') === (type === 'adversarial')));
const curves = groups.map(group => app.buildCurve(group, bound));
console.log(JSON.stringify({
  steps, facts, curves, overview: curves.map(curve => curve.map(p => app.pct(p.accepted / p.total))),
  rounded: app.pct(466 / 501), total: app.aggregate(facts.map(rows => rows[31])),
}));
""", {"docs": docs, "bound": costs.B})
    assert report["steps"][0] == 0 and report["steps"][-1] == 1000
    assert all(np.diff(report["steps"]) > 0)
    for doc, rows in zip(docs, report["facts"]):
        assert rows[0]["reviewed"] == len(doc["fields"])  # Fixture risks are all positive.
        assert rows[-1]["accepted"] == len(doc["fields"])
        table = pd.DataFrame(doc["fields"]).assign(doc_id=doc["id"])
        for lam, row in zip(report["steps"], rows):
            assert row["accepted"] + row["reviewed"] == row["total"] == len(doc["fields"])
            assert row["caught"] + row["missed"] == sum(field["e"] for field in doc["fields"])
            policy = evaluate_lambda(table, "p", lam, costs, "field")
            assert row["accepted"] / row["total"] == pytest.approx(policy["automated"])
            assert row["observed"] == pytest.approx(policy["mean_doc_loss"])
        for key, sign in (("accepted", 1), ("reviewed", -1), ("missed", 1)):
            assert (sign * np.diff([row[key] for row in rows]) >= 0).all()
    for curve, percentages in zip(report["curves"], report["overview"]):
        assert len(curve) == 62 and (np.diff([point["automation"] for point in curve]) >= 0).all()
        for point, shown in zip(curve, percentages):
            assert 0 <= point["observed"] <= 1 and 0 <= point["estimated"] <= 1
            assert float(shown.rstrip("%").replace(",", ".")) == pytest.approx(100 * point["accepted"] / point["total"], abs=0.05)
    assert report["rounded"] == "93%"
    assert report["total"]["total"] == sum(len(doc["fields"]) for doc in docs)
