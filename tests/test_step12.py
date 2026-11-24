"""Step 12: datasets (clean, noisy, adversarial), frontier plan, verifier records,
extracted-field injection and completeness check. Fake model, no network."""
import json
from random import Random

import pytest

from scripts import build_datasets as B
from scripts import full_extraction as F
from scripts import pilot_extraction as P
from trustswap.extract import clients as C
from trustswap.extract.evaluate import load_gold
from trustswap.extract.inject import KINDS, inject, inject_all
from trustswap.extract.prompt import DELIMITERS, gold_answer, number_lines
from trustswap.synth.noise import add_noise, map_span
from trustswap.synth.render import trade_fields


# ---------- noise ----------

def test_noise_is_deterministic_and_maps_spans():
    text = "Notional Amount: USD 10,000,000\nFixed Rate: 4.2500%\nPayer: Synthetic Party A"
    noisy, mapping = add_noise(text, seed=3, severity=2)
    assert (noisy, mapping) == add_noise(text, seed=3, severity=2)
    assert len(mapping) == len(text) + 1 and mapping[-1] == len(noisy)
    start = text.index("Synthetic")
    span = map_span((start, start + len("Synthetic Party A")), mapping)
    assert span is not None and noisy[span[0]:span[1]].replace("\n", " ").split()
    assert "," in noisy and "10" in noisy.replace("l", "1").replace("O", "0")  # separators never swapped
    clean, _ = add_noise(text, seed=3, severity=1)
    assert sum(a != b for a, b in zip(clean, text)) <= sum(a != b for a, b in zip(noisy, text)) + 5


# ---------- datasets ----------

@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    B.build_synthetic(out, {"train": 3, "dev": 1, "syn_test": 2}, log=lambda *_: None)
    B.build_noisy(out, 1.0, log=lambda *_: None)
    B.build_adversarial(out, 1, log=lambda *_: None)
    B.build_noisy(out, 1.0, log=lambda *_: None)  # after adversarial: parent twins must be skipped
    return out


def test_datasets_are_complete_and_idempotent(data):
    syn = [json.loads(l) for l in (data / "synthetic" / "docs.jsonl").read_text().splitlines()]
    adv = [json.loads(l) for l in (data / "adversarial" / "docs.jsonl").read_text().splitlines()]
    clean = [d for d in syn if B.CLEAN_ID.fullmatch(d["doc_id"])]
    noisy = [d for d in syn if d["ocr"]]
    assert len(clean) == 6 and len(noisy) == 6
    assert {d["split"] for d in clean} == {"train", "dev", "syn_test"}
    assert len(adv) == 36 and all(d["split"] == "adv_test" for d in adv)
    gold = load_gold(data / "synthetic" / "gold.jsonl")
    for doc in noisy:  # gold values unchanged, spans point into the noisy text
        twin = gold[doc["doc_id"].rsplit("_ocr", 1)[0]]
        assert trade_fields(gold[doc["doc_id"]].trade)["trade_date"].value == trade_fields(twin.trade)["trade_date"].value
    assert B.build_synthetic(data, {"train": 3, "dev": 1, "syn_test": 2}, log=lambda *_: None) == 0
    assert B.build_noisy(data, 1.0, log=lambda *_: None) == 0
    assert B.build_adversarial(data, 1, log=lambda *_: None) == 0


def test_frontier_plan_is_stratified_and_within_budget(data):
    sizes = B.plan_frontier(data, budget=60_000, tokens_per_doc=6000)
    plan = (data / "s12" / "frontier_docs.txt").read_text().split()
    assert len(plan) == sum(sizes.values()) <= 10 and len(set(plan)) == len(plan)
    assert any(k.startswith("adv_test/") for k in sizes) and any(k.startswith("train/") for k in sizes)
    assert B.plan_frontier(data, budget=60_000, tokens_per_doc=6000) == sizes  # deterministic


# ---------- extraction, verify, inject, check ----------

class GoldModel:
    params = {"structured": "json_schema"}

    def __init__(self, data):
        self.answers = {}
        for collection in ("synthetic", "adversarial"):
            gold = load_gold(data / collection / "gold.jsonl")
            for doc in P.read_docs(data / collection / "docs.jsonl"):
                text = doc["_text_path"].read_text()
                self.answers[number_lines(text)] = gold_answer(gold[doc["doc_id"]].trade, text)

    def complete(self, messages, *, temperature, seed):
        document = messages[-1]["content"].rsplit(DELIMITERS[0], 1)[1]
        answer = next(a for key, a in self.answers.items() if key in document)
        return C.Completion(text=json.dumps(answer), input_tokens=3000, output_tokens=1000, latency_s=1.0)


def test_full_pipeline_on_a_plan(data, tmp_path, monkeypatch):
    spec = C.ModelSpec(key="frontier", name="fake", version="1", access_date="2026-11-24",
                       base_url_env=None, api_key_env="X")
    monkeypatch.setattr(P, "load_models", lambda: {"frontier": spec, "open_weights": spec})
    monkeypatch.setattr(P, "OpenAICompatClient", lambda _spec: GoldModel(data))
    B.plan_frontier(data, budget=120_000, tokens_per_doc=6000)
    plan = data / "s12" / "frontier_docs.txt"
    cache = tmp_path / "cache"
    assert F.main(["--data", str(data), "--cache", str(cache), "run", "--model", "frontier",
                   "--plan", str(plan), "--k", "1", "--yes"]) == 0
    assert F.main(["--data", str(data), "--cache", str(cache), "verify", "--model", "frontier"]) == 0
    planned = plan.read_text().split()
    for doc_id in planned:
        record = json.loads((cache / "verify" / "frontier" / f"{doc_id}.json").read_text())
        assert record["doc_id"] == doc_id and "rules" in record and record["verifier_hash"]
    clean_ids = [d for d in planned if d.startswith("syn_") and "_ocr" not in d]
    for doc_id in clean_ids:  # perfect extraction of clean synthetic: no HARD verdict
        record = json.loads((cache / "verify" / "frontier" / f"{doc_id}.json").read_text())
        assert not any(v["verdict"] == "HARD" for f in record["rules"].values() for v in f.values())
    problems, sheet = F.cmd_check("frontier", F.all_docs(data), cache, 1, planned, tmp_path / "spot.md", data,
                                  n_spot=3)
    assert problems == [] and "| [ ] |" in (tmp_path / "spot.md").read_text()
    markdown = F.cmd_inject("frontier", F.all_docs(data), cache, 2, tmp_path / "inj.md")
    assert "| injection | copies | detected |" in markdown
    injected = [json.loads(l) for p in (cache / "extract_injected" / "frontier").glob("*.jsonl")
                for l in p.read_text().splitlines()]
    assert injected and all("injection" in r for r in injected)
    # planted value changes on grounded fields are caught by the verifier
    scale = [l for p in (cache / "verify_injected" / "frontier").glob("*.jsonl")
             for l in map(json.loads, p.read_text().splitlines()) if l["injection"]["kind"] == "scale"]
    for record in scale:
        assert F.fired_on(record["rules"], record["injection"]["fields"])


def test_inject_kinds_change_exactly_the_tagged_fields():
    record = {"schema_valid": True, "fields": {
        "trade_date": {"value": "2024-01-02", "span": [0, 10]},
        "fixed_leg.notional_amount": {"value": 1000000.0, "span": [11, 20]},
        "fixed_leg.day_count_fraction": {"value": "ACT/360", "span": [21, 28]},
        "fixed_leg.payer": {"value": "A", "span": [29, 30]},
        "fixed_leg.receiver": {"value": "B", "span": [31, 32]},
    }}
    for kind in KINDS:
        variant = inject(record, kind, Random(1))
        assert variant is not None, kind
        meta = variant["injection"]
        changed = [p for p in record["fields"] if variant["fields"][p]["value"] != record["fields"][p]["value"]]
        assert sorted(changed) == meta["fields"], kind
    assert record["fields"]["fixed_leg.payer"]["value"] == "A"  # original untouched
    assert len(inject_all(record, 3, seed=5)) == 3
    assert inject({"schema_valid": False}, "scale", Random(1)) is None
