"""Synthetic cache replay of the paired before/after research pipeline."""
import json

import pandas as pd

from scripts import replay_normalisation as replay
from trustswap.common.annotations import normalise_field
from trustswap.extract.fields import PATHS
from trustswap.extract.parse import parse_completion, trade_from_fields


def make_fixture(root):
    data = root / "data"
    cache = root / "cache"
    for collection, splits in replay.SPLITS.items():
        docs, gold = [], []
        for split in splits:
            for index in range(3):
                doc_id = f"fixture_{split}_{index}"
                clean = {
                    "trade_date": "2012-04-13", "effective_date": "2012-04-16", "termination_date": "2017-04-14",
                    "fixed_leg.notional_amount": "USD 5,000,000", "fixed_leg.fixed_rate": "4.06%",
                    "fixed_leg.notional_currency": "USD", "floating_leg.notional_currency": "USD",
                    "floating_leg.floating_rate_option": "USD-LIBOR-BBA",
                }
                raw = dict(clean)
                dirty_path = ("trade_date", "fixed_leg.notional_amount", "fixed_leg.fixed_rate")[index]
                raw[dirty_path] = ("/ " if index == 0 else ": " if index == 1 else "- ") + clean[dirty_path]
                text = "\n".join(f"{path}: {raw_value}" for path, raw_value in raw.items())

                def answer(terms):
                    return {"structure": "fixed_float", "fields": {
                        path: {"value": terms.get(path), "line": None, "confidence": 0.75} for path in PATHS
                    }}

                # Gold is built from original structured values. The cached
                # fixture completions deliberately omit one term on both passes.
                expected = parse_completion(json.dumps(answer(clean)), text).fields
                submitted = {key: value for key, value in raw.items() if key != "termination_date"}
                parsed = parse_completion(json.dumps(answer(submitted)), text)
                try:
                    normalise_field(dirty_path, raw[dirty_path])
                except ValueError as error:
                    parsed.fields[dirty_path].update(value=None, normalise_error=str(error))
                text_file = data / collection / "text" / f"{doc_id}.txt"
                text_file.parent.mkdir(parents=True, exist_ok=True)
                text_file.write_text(text, encoding="utf-8")
                docs.append({"doc_id": doc_id, "collection": collection, "split": split,
                             "text_path": str(text_file), "ocr": False})
                gold.append({"doc_id": doc_id, "collection": collection,
                             "generator": {"seed": index, "sampler_version": "fixture"},
                             "trade": trade_from_fields(expected).model_dump(mode="json")})
                for model, k in (("frontier", 1), ("open_weights", 2)):
                    replay._write_jsonl(cache / "extract" / model / f"{doc_id}.jsonl", [
                        {"doc_id": doc_id, "model": model, "k": sample, "schema_valid": True,
                         "structure": parsed.structure, "fields": parsed.fields} for sample in range(k)
                    ])
        replay._write_jsonl(data / collection / "docs.jsonl", [
            *docs, {"doc_id": "locked_fixture", "collection": collection, "split": "real_test"},
        ])
        replay._write_jsonl(data / collection / "gold.jsonl", gold)
    for model in replay.MODELS:
        (cache / "extract" / model / "locked_fixture.jsonl").write_text("unreadable fixture", encoding="utf-8")
    return data, cache


def test_paired_cache_only_pipeline(tmp_path):
    data, cache = make_fixture(tmp_path)
    original_path = cache / "extract" / "frontier" / "fixture_train_0.jsonl"
    original = original_path.read_bytes()
    output = tmp_path / "replay"
    assert replay.main(["--data", str(data), "--cache", str(cache), "--out", str(output)]) == 0
    assert original_path.read_bytes() == original
    assert not (output / "after/cache/extract/frontier/locked_fixture.jsonl").exists()
    after = json.loads((output / "after/cache/extract/frontier/fixture_train_0.jsonl").read_text())
    before = json.loads(original)
    assert after["fields"]["trade_date"]["value"] == "2012-04-13"
    assert after["fields"]["trade_date"]["raw_value"] == before["fields"]["trade_date"]["raw_value"]
    assert after["fields"]["trade_date"]["span"] == before["fields"]["trade_date"]["span"]
    assert "normalise_error" not in after["fields"]["trade_date"]
    report = (output / "comparison.md").read_text()
    for stage in ("before", "after"):
        for model in replay.MODELS:
            for collection in replay.SPLITS:
                assert (output / stage / "results" / f"injection_{model}_{collection}.md").is_file()
    for model in replay.MODELS:
        paths = [output / name / "cache/risk" / f"{model}.parquet" for name in ("before", "after")]
        previous, current = map(pd.read_parquet, paths)
        assert current["e"].sum() < previous["e"].sum()
        assert f"| {model} | adversarial | adv_test |" in report
        assert f"| {model} | synthetic | syn_test |" in report
