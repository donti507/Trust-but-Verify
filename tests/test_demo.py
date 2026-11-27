"""Offline checks for the standalone demo builder."""
import json
import re

import pandas as pd
import pytest

from scripts import build_demo
from trustswap.decide.policy import escalate, load_costs


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_inputs(root, *, extra_samples=False):
    data, cache = root / "data", root / "cache"
    clean = (
        "swap-clean", "synthetic", "syn_test",
        "Interest Rate Swap Confirmation\n\n"
        "Counterparties: Alder Quay Bank and Kestrel Harbour Finance\n\n"
        "Trade date: 26 November 2026\n"
        "Effective date: 30 November 2026\n"
        "Termination date: 30 November 2031\n\n"
        "Floating leg notional: USD 25,000,000\n"
        "Floating rate: USD SOFR\n"
        "Payments: quarterly\n",
        "25,000,000", 25_000_000, 25_000_000, 0.006,
    )
    adversarial = (
        "swap-conflict", "adversarial", "adv_test",
        "Interest Rate Swap Confirmation\n\n"
        "Counterparties: Rowan Reach Bank and Wrenwater Capital\n\n"
        "Trade date: 26 November 2026\n"
        "Effective date: 30 November 2026\n"
        "Termination date: 30 November 2031\n\n"
        "Floating leg notional: USD 50,000,000\n"
        "Floating rate: USD SOFR\n"
        "Payments: quarterly\n\n"
        "Amendment: the floating leg notional is USD 5,000,000.\n"
        + ("Source note: </script> https://fictional.invalid\n" if not extra_samples else ""),
        "50,000,000", 50_000_000, 5_000_000, 0.62,
    )
    examples = [clean, adversarial]
    if extra_samples:
        examples += [
            ("swap-noisy_ocr2", "synthetic", "syn_test",
             "Swap confirmation · OCR copy\n\nCounterparties: Alder Quay Bank and Wrenwater Capital\n\n"
             "Trade date: 26 November 2026\n"
             "Floating leg notional: USD 2S,000,000\n",
             "2S,000,000", 2_000_000, 25_000_000, 0.33),
            ("z-clean-review", *clean[1:-1], 0.42),
        ]
    risks = []
    amount = "floating_leg.notional_amount"
    trade_date = "trade_date"
    source_date = "26 November 2026"
    manifests, gold = {}, {}
    for doc_id, collection, split, text, raw, value, gold_value, p in examples:
        base = data / collection
        text_path = base / "text" / f"{doc_id}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text, encoding="utf-8")
        manifest_path = root / "missing.txt" if collection == "adversarial" else text_path
        manifests.setdefault(collection, []).append({
            "doc_id": doc_id, "collection": collection, "split": split, "text_path": str(manifest_path),
        })
        gold.setdefault(collection, []).append({
            "doc_id": doc_id, "collection": collection,
            "trade": {"floating_leg": {"notional_amount": {"value": gold_value}},
                      "trade_date": {"value": "2026-11-26"}},
        })
        fields = {
            amount: {"raw_value": raw, "value": value, "span": [text.index(raw), text.index(raw) + len(raw)]},
            trade_date: {"raw_value": source_date, "value": "2026-11-26",
                         "span": [text.index(source_date), text.index(source_date) + len(source_date)]},
        }
        _write_jsonl(cache / "extract" / "open_weights" / f"{doc_id}.jsonl", [
            {"k": 1, "fields": {amount: {"value": -1}}},
            {"k": 0, "schema_valid": True, "fields": fields},
        ])
        verify_path = cache / "verify" / "open_weights" / f"{doc_id}.json"
        verify_path.parent.mkdir(parents=True, exist_ok=True)
        verify_path.write_text(json.dumps({"rules": {
            "V6": {amount: {"verdict": "HARD" if value != gold_value else "PASS"}},
            "V7": {trade_date: {"verdict": "PASS"}},
        }}), encoding="utf-8")
        for field, risk, error in ((amount, p, value != gold_value), (trade_date, 0.02, False)):
            risks.append({"doc_id": doc_id, "collection": collection, "split": split,
                          "field": field, "e": int(error), "p_full_logistic": risk})
    for collection in manifests:
        _write_jsonl(data / collection / "docs.jsonl", manifests[collection])
        _write_jsonl(data / collection / "gold.jsonl", gold[collection])
    # An excluded cache row has an undefined field cost. Any accidental processing fails.
    risks.append({"doc_id": "excluded", "collection": "real", "split": "real_cal",
                  "field": "unavailable", "e": 0, "p_full_logistic": 1.0})
    (cache / "risk").mkdir(parents=True)
    pd.DataFrame(risks).to_parquet(cache / "risk" / "open_weights.parquet", index=False)
    return data, cache


def test_demo_embeds_two_documents_and_needs_no_network(tmp_path):
    data, cache = _write_inputs(tmp_path)
    output = tmp_path / "results" / "demo" / "index.html"
    assert build_demo.main(["--data", str(data), "--cache", str(cache), "--output", str(output)]) == 0
    assert list(output.parent.iterdir()) == [output]
    html = output.read_text(encoding="utf-8")
    data_tag = re.search(r'<script type="application/json" id="demo-data">(.*?)</script>', html, re.DOTALL)
    documents = json.loads(data_tag[1])["documents"]
    assert [doc["id"] for doc in documents] == ["swap-clean", "swap-conflict"]
    assert "Synthetic document, fictional parties" in html
    assert not re.search(r"https?://", html, re.IGNORECASE)
    assert not re.search(r"<(?:script|link)\b[^>]+(?:src|href)=", html, re.IGNORECASE)
    assert "default-src 'none'" in html
    assert "excluded" not in html
    for doc in documents:
        assert doc["category"] in build_demo.CATEGORIES
        for field in doc["fields"]:
            assert 0 <= field["p"] <= 1
            assert field["e"] in (0, 1)
            assert field["severity"] in {"critical", "major", "minor"}
            assert set(field["rules_fired"]) <= set(build_demo.RULE_TEXT)
            if field["span"] is not None:
                a, b = field["span"]
                assert doc["text"][a:b] == field["raw_value"]
    clean, conflict = documents
    assert clean["fields"][0]["value"] == 25_000_000
    field = conflict["fields"][0]
    assert field["name"] == "Notional amount (floating leg)"
    assert (field["e"], field["severity"], field["rules_fired"]) == (1, "critical", ["V6"])
    start, end = field["span"]
    assert conflict["text"][start:end] == "50,000,000"
    assert "</script> https://fictional.invalid" in conflict["text"]


def test_selection_mixes_categories_and_prefers_review(tmp_path):
    data, cache = _write_inputs(tmp_path, extra_samples=True)
    documents = build_demo.load_documents(data, cache, "open_weights", load_costs(), 3)
    assert [doc["id"] for doc in documents] == ["z-clean-review", "swap-noisy_ocr2", "swap-conflict"]
    assert [doc["category"] for doc in documents] == list(build_demo.CATEGORIES)


@pytest.mark.parametrize("lam", [0, 0.001, 0.1, 1, 2, 62, 1000])
def test_browser_decision_mirror_matches_policy(lam):
    cases = [(0, 100, 1), (0.62, 100, 1), (0.2, 10, 2), (1, 0.2, 1), (0.001, 1, 1)]
    p, costs, review = zip(*cases)
    expected = escalate(p, costs, review, lam)
    assert [build_demo.decision(*case, lam) for case in cases] == [
        "Review" if flag else "Accept" for flag in expected
    ]


def test_stitch_views_and_chart_share_the_offline_data(tmp_path):
    from bs4 import BeautifulSoup

    data, cache = _write_inputs(tmp_path)
    costs = load_costs()
    documents = build_demo.load_documents(data, cache, "open_weights", costs, 2)
    html = build_demo.render_page(documents, 125.0, illustrative=True)
    page = BeautifulSoup(html, "html.parser")

    assert {view["id"] for view in page.select("[data-view]")} == {
        "overview", "confirmations", "method", "audit", "feed",
    }
    assert {link["href"][1:] for link in page.select("[data-nav]")} == {
        view["id"] for view in page.select("[data-view]")
    }
    assert page.select_one("svg#frontier")["viewbox"]
    assert page.select_one("#document-text") and page.select_one("#stages")
    payload = json.loads(page.select_one("#demo-data").get_text())
    assert payload["bound"] == 125.0
    assert payload["illustrative"] is True
    assert "__DEMO_" not in html
