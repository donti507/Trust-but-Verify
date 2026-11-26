"""Step 13: feature builder (fake models on a small generated dataset, no network)."""
import json

import pandas as pd
import pytest

from scripts import build_datasets as B
from scripts import build_features as BF
from scripts import full_extraction as F
from scripts import pilot_extraction as P
from trustswap.common.cache import Cache
from trustswap.extract import clients as C
from trustswap.extract.evaluate import load_gold, load_records
from trustswap.extract.prompt import DELIMITERS, gold_answer, number_lines
from trustswap.extract.run import extract_document
from trustswap.score.features import (FIELD_NAMES, RULES, SEVERITIES, VERDICTS, build_table, feature_columns,
                                      label_report)
from trustswap.synth.render import trade_fields

WRONG_PATH = "trade_date"


class Model:
    """Answers from gold; optionally a wrong trade date, or broken JSON, for chosen documents."""
    params = {"structured": "json_schema"}

    def __init__(self, data, wrong=(), broken=(), name="m"):
        self.answers, self.name = {}, name
        for collection in ("synthetic", "adversarial"):
            gold = load_gold(data / collection / "gold.jsonl")
            for doc in P.read_docs(data / collection / "docs.jsonl"):
                text = doc["_text_path"].read_text()
                answer = gold_answer(gold[doc["doc_id"]].trade, text)
                if doc["doc_id"] in wrong:
                    answer["fields"][WRONG_PATH]["value"] = "1-Jan-1999"
                self.answers[number_lines(text)] = "{broken" if doc["doc_id"] in broken else json.dumps(answer)

    def complete(self, messages, *, temperature, seed, max_tokens=None):
        document = messages[-1]["content"].rsplit(DELIMITERS[0], 1)[1]
        text = next(a for key, a in self.answers.items() if key in document)
        tokens = [(text[i:i + 5], -0.2) for i in range(0, len(text), 5)]
        return C.Completion(text=text, tokens=tokens, finish_reason="stop", input_tokens=10, output_tokens=10)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    B.build_synthetic(data, {"train": 2, "dev": 1, "syn_test": 1}, log=lambda *_: None)
    B.build_adversarial(data, 1, log=lambda *_: None)
    B.build_noisy(data, 1.0, log=lambda *_: None)
    docs = F.all_docs(data)
    cache = tmp_path_factory.mktemp("cache")
    wrong, broken = {"syn_train_0000"}, {"syn_dev_0000"}
    for model_key, model in (("open_weights", Model(data, wrong, broken)), ("frontier", Model(data))):
        for doc in docs.values():
            extract_document(doc["doc_id"], doc["_text_path"].read_text(), model, model_id=f"{model_key}@1",
                             cache=Cache(cache / "raw" / model_key), out_dir=cache / "extract" / model_key,
                             k_samples=2 if model_key == "open_weights" else 1, split=doc.get("split"))
        F.cmd_verify(model_key, docs, cache)
    return data, cache, docs, wrong, broken


def _table(built):
    data, cache, docs, _, _ = built
    gold = {**load_gold(data / "synthetic" / "gold.jsonl"), **load_gold(data / "adversarial" / "gold.jsonl")}
    return build_table(docs, gold, load_records(cache / "extract" / "open_weights"),
                       cache / "verify" / "open_weights", load_records(cache / "extract" / "frontier")), gold


def test_one_row_per_gold_field_and_no_missing_values(built):
    table, gold = _table(built)
    assert not table.isna().any().any()
    expected = sum(len(trade_fields(gold[d].trade)) for d in table["doc_id"].unique())
    assert len(table) == expected
    assert table[["doc_id", "field"]].duplicated().sum() == 0


def test_one_hot_groups_sum_to_one(built):
    table, _ = _table(built)
    for rule in RULES:
        assert (table[[f"v_{rule}_{v}" for v in VERDICTS]].sum(axis=1) == 1).all()
    assert (table[[f"tau_{n}" for n in FIELD_NAMES]].sum(axis=1) == 1).all()
    assert (table[[f"tau_sev_{s}" for s in SEVERITIES]].sum(axis=1) == 1).all()


def test_labels_follow_gold(built):
    table, gold = _table(built)
    _, _, _, wrong, broken = built
    row = table[(table.doc_id == next(iter(wrong))) & (table.field == WRONG_PATH)].iloc[0]
    assert row["e"] == 1 and row["u_cross_agree"] == 0  # the other model got it right
    clean = table[(table.doc_id == "syn_train_0001")]
    assert clean["e"].sum() == 0 and clean["u_agreement"].eq(1.0).all() and clean["u_cross_agree"].eq(1).all()
    dead = table[table.doc_id == next(iter(broken))]
    assert dead["q_schema_valid"].eq(0).all() and dead["u_predicted_null"].eq(1).all()
    assert dead.loc[dead.field == "trade_date", "e"].iloc[0] == 1
    ambiguous = [(d, p) for d in table.doc_id.unique() for p, t in trade_fields(gold[d].trade).items()
                 if t.status == "ambiguous"]
    assert ambiguous
    for doc_id, path in ambiguous:
        assert table[(table.doc_id == doc_id) & (table.field == path)]["e"].iloc[0] == 1


def test_uncertainty_and_grounding_features(built):
    table, _ = _table(built)
    valid = table[table.q_schema_valid == 1]
    assert valid["u_logprob"].between(-0.21, -0.19).all() and valid["u_logprob_missing"].eq(0).all()
    assert table[table.q_schema_valid == 0]["u_logprob_missing"].eq(1).all()
    explicit = valid[(valid.u_predicted_null == 0) & (valid.e == 0) & valid.doc_id.str.fullmatch(r"syn_(train|dev|syn_test)_\d{4}")]
    assert explicit["g_span_found"].eq(1).all()
    assert table["q_ocr"].isin([0, 1]).all() and table.loc[table.doc_id.str.contains(r"_ocr\d$"), "q_ocr"].eq(1).all()
    assert table.loc[table.doc_id == "syn_train_0001", "q_family_seen"].eq(1).all()


def test_script_writes_parquet_and_report(built, tmp_path):
    data, cache, _, _, _ = built
    assert BF.main(["--model", "open_weights", "--data", str(data), "--cache", str(cache),
                    "--results", str(tmp_path)]) == 0
    table = pd.read_parquet(cache / "features" / "open_weights.parquet")
    assert len(table) > 0 and set(feature_columns(table)) and "e" in table
    report = (tmp_path / "features_open_weights.md").read_text()
    assert "Missing values: 0." in report and "| trade_date |" in report and "| adversarial | adv_test |" in report
    assert label_report(table, "x").startswith("# Features (x)")
