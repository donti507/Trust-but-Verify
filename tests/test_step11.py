"""Step 11: extraction pilot (fake model, no network)."""
from datetime import date
from decimal import Decimal
import json
from pathlib import Path

import pytest

from scripts import pilot_extraction as P
from trustswap.common.cache import Cache
from trustswap.extract import clients as C
from trustswap.extract.evaluate import ModelReport, canonical, consistency, cost, load_gold, outcome
from trustswap.extract.prompt import DELIMITERS, gold_answer, number_lines, prompt_sha
from trustswap.extract.run import extract_document
from trustswap.synth.templates import FAMILY_SPLITS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    out = tmp_path_factory.mktemp("pilot")
    docs_file = P.make_synthetic(8, out)
    return out, docs_file


class Gold:
    """Fake model that answers each pilot document from its gold record."""
    params = {"structured": "json_schema"}

    def __init__(self, docs_file, wrong=None):
        self.answers, self.calls, self.wrong = {}, 0, wrong or {}
        gold = load_gold(docs_file.parent / "gold.jsonl")
        for doc in P.read_docs(docs_file):
            text = doc["_text_path"].read_text()
            answer = gold_answer(gold[doc["doc_id"]].trade, text)
            for path, value in self.wrong.get(doc["doc_id"], {}).items():
                answer["fields"][path]["value"] = value
            self.answers[number_lines(text)] = answer

    def complete(self, messages, *, temperature, seed):
        self.calls += 1
        document = messages[-1]["content"].rsplit(DELIMITERS[0], 1)[1]
        answer = next(a for marker, a in self.answers.items() if marker in document)
        return C.Completion(text=json.dumps(answer), input_tokens=3000, output_tokens=1000, latency_s=2.0)


def _use_fake(monkeypatch, model):
    spec = C.ModelSpec(key="frontier", name="fake", version="1", access_date="2026-11-24",
                       base_url_env=None, api_key_env="X")
    monkeypatch.setattr(P, "load_models", lambda: {"frontier": spec, "open_weights": spec})
    monkeypatch.setattr(P, "OpenAICompatClient", lambda _spec: model)


# ---------- comparison ----------

def test_canonical_comparison_rules():
    assert canonical("fixed_leg.fixed_rate", Decimal("0.0425")) == canonical("fixed_leg.fixed_rate", 0.0425)
    assert canonical("floating_leg.notional_amount", Decimal("1E+7")) == canonical("x.notional_amount", 10000000.0)
    assert canonical("trade_date", date(2024, 1, 2)) == canonical("trade_date", "2024-01-02")
    assert canonical("fixed_leg.business_centers", ["USNY", "GBLO"]) == canonical("x.business_centers", ["GBLO", "USNY"])
    assert canonical("floating_leg.floating_rate_option", "EUR-EURIBOR-Reuters") == canonical(
        "floating_leg.floating_rate_option", "EUR-EURIBOR-REUTERS")
    assert canonical("fixed_leg.payer", " Party  A ") == canonical("fixed_leg.payer", "party a")


def test_outcomes_for_absent_and_ambiguous_gold():
    from trustswap.common.schema import ExtractedField

    absent = ExtractedField()
    ambiguous = ExtractedField(status="ambiguous")
    explicit = ExtractedField(value=Decimal("5"), span=(0, 1), status="explicit")
    assert outcome("notice_address", None, absent) == "match"
    assert outcome("notice_address", "1 Street", absent) == "mismatch"  # invented value
    assert outcome("x.notional_amount", 5.0, explicit) == "match"
    assert outcome("x.notional_amount", None, explicit) == "mismatch"  # missed value
    assert outcome("x.notional_amount", 5.0, ambiguous) == "ambiguous"


def test_consistency_is_share_of_modal_value():
    def rec(v):
        return {"schema_valid": True, "fields": {"trade_date": {"value": v}}}
    records = [rec("2024-01-02"), rec("2024-01-02"), rec("2024-01-03"), {"schema_valid": False}]
    assert consistency(records, "trade_date") == pytest.approx(2 / 3)


def test_cost():
    assert cost(1_000_000, 500_000, {"input_per_million": 2.0, "output_per_million": 4.0}) == pytest.approx(4.0)
    assert cost(1, 1, None) is None


# ---------- pilot documents ----------

def test_pilot_documents_avoid_syn_test_and_are_idempotent(pilot):
    out, docs_file = pilot
    docs = P.read_docs(docs_file)
    assert len(docs) == 8
    assert {d["split"] for d in docs} <= {"train", "dev"} and "dev" in {d["split"] for d in docs}
    assert all(FAMILY_SPLITS[d["family_id"]] != "syn_test" for d in docs)
    P.make_synthetic(8, out)  # second call renders nothing new
    assert len(P.read_docs(docs_file)) == 8


# ---------- run + report ----------

def test_run_report_and_budget(pilot, tmp_path, monkeypatch):
    out, docs_file = pilot
    docs = P.read_docs(docs_file)
    first = docs[0]["doc_id"]
    model = Gold(docs_file, wrong={first: {"trade_date": "1-Jan-1999"}})
    _use_fake(monkeypatch, model)
    summary = P.run(("frontier",), docs, k=2, budget=None, workers=2, cache_root=tmp_path, log=lambda *_: None)
    assert summary["frontier"] == {"valid": 16, "total": 16, "fresh_tokens": 16 * 4000}
    again = P.run(("frontier",), docs, k=2, budget=None, workers=2, cache_root=tmp_path, log=lambda *_: None)
    assert again["frontier"]["fresh_tokens"] == 0 and model.calls == 16  # cache: no new calls

    markdown = P.report({"synthetic": docs_file, "real": tmp_path / "none.jsonl"}, tmp_path, tmp_path / "r.md")
    assert "## synthetic" in markdown and "## real" not in markdown
    assert "| trade_date |" in markdown and f"`trade_date` {first}: extracted `1-Jan-1999`" in markdown
    assert "cost/doc" in markdown and "100.0%" in markdown  # schema-valid

    budget_model = Gold(docs_file)
    _use_fake(monkeypatch, budget_model)
    limited = P.run(("frontier",), docs, k=1, budget=8000, workers=1, cache_root=tmp_path / "b",
                    log=lambda *_: None)
    assert limited["frontier"]["total"] < len(docs)  # stopped at the budget


def test_model_report_counts_accuracy_and_structure(pilot, tmp_path):
    _, docs_file = pilot
    docs = P.read_docs(docs_file)
    gold = load_gold(docs_file.parent / "gold.jsonl")
    model = Gold(docs_file)
    report = ModelReport(model="fake")
    for doc in docs:
        records = extract_document(doc["doc_id"], doc["_text_path"].read_text(), model, model_id="fake@1",
                                   cache=Cache(tmp_path / "raw"), out_dir=tmp_path / "ex", k_samples=2)
        report.add(doc["doc_id"], records, gold[doc["doc_id"]])
    assert report.accuracy == 1.0 and report.structure_match == len(docs)
    assert all(v == 1.0 for values in report.consistency.values() for v in values)


def test_real_pilot_must_come_from_real_cal(tmp_path):
    docs_file = tmp_path / "docs.jsonl"
    docs_file.write_text(json.dumps({"doc_id": "r1", "collection": "real", "split": None,
                                     "text_path": str(tmp_path / "r1.txt")}) + "\n")
    with pytest.raises(SystemExit, match="real_cal"):
        P.main(["run", "--docs", str(docs_file), "--yes", "--cache", str(tmp_path)])


# ---------- prompt freeze ----------

def test_prompt_sha_is_stable():
    assert prompt_sha() == prompt_sha() and len(prompt_sha()) == 64


def test_frozen_prompt_matches_its_hash():
    freeze = ROOT / "prereg" / "prompt-freeze.json"
    if not freeze.exists():
        pytest.skip("prompt not frozen yet")
    record = json.loads(freeze.read_text())
    from trustswap.extract.prompt import PROMPT_VERSION
    if record["prompt_version"] == PROMPT_VERSION:
        assert record["prompt_sha256"] == prompt_sha(), "prompt changed after freeze: create a new version"
