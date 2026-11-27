"""Compare cached extraction records before and after separator cleanup.

Rebuild verification, injection and cache checks, features, risk and decisions
in a new directory. Only synthetic and adversarial study splits are eligible.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

from trustswap.decide.policy import evaluate_lambda, load_costs
from trustswap.extract.evaluate import load_gold
from trustswap.extract.parse import normalise_value, trim_value
from trustswap.score.risk import metrics

if __package__:
    from . import build_features, decide, full_extraction, train_risk
else:
    import build_features
    import decide
    import full_extraction
    import train_risk

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("frontier", "open_weights")
SPLITS = {"synthetic": ("train", "dev", "syn_test"), "adversarial": ("adv_test",)}
PROBABILITY = "p_full_logistic"
DOC_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def reparse(record: dict) -> dict:
    """Recompute affected typed values and preserve the cached evidence."""
    if not record.get("schema_valid"):
        return record
    fields = dict(record["fields"])
    for path, evidence in record["fields"].items():
        raw = evidence.get("raw_value")
        if not isinstance(raw, str) or not raw.strip() or trim_value(path, raw) == raw:
            continue
        amended = dict(evidence)
        try:
            amended["value"] = normalise_value(path, raw)
        except (ValueError, ArithmeticError) as error:
            amended.update(value=None, normalise_error=str(error))
        else:
            amended.pop("normalise_error", None)
        fields[path] = amended
    return {**record, "fields": fields}


def _local_path(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or "real_test" in resolved.parts:
        raise ValueError(f"Input escapes the eligible cohort: {path.name}")
    return resolved


def _records(file: Path, doc_id: str) -> list[dict]:
    records = sorted((json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()
                      if line.strip()), key=lambda record: record["k"])
    if (not records or [record["k"] for record in records] != list(range(len(records)))
            or any(record["doc_id"] != doc_id or record.get("split") == "real_test" for record in records)):
        raise ValueError(f"Invalid cached samples for {doc_id}")
    return records


def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def stage(data: Path, cache: Path, out: Path) -> dict:
    """Stage only files identified by eligible metadata, before any bulk loads."""
    data, cache = data.resolve(), cache.resolve()
    cohorts = {model: {} for model in MODELS}
    seen = set()
    for collection, splits in SPLITS.items():
        source = data / collection
        manifest = _local_path(source / "docs.jsonl", source)
        gold_file = _local_path(source / "gold.jsonl", source)
        gold = load_gold(gold_file)
        staged_docs = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            doc = json.loads(line)
            if doc.get("collection") != collection or doc.get("split") not in splits:
                continue
            doc_id = doc["doc_id"]
            if not DOC_ID.fullmatch(doc_id) or doc_id in seen:
                raise ValueError(f"Invalid or duplicate document ID: {doc_id}")
            seen.add(doc_id)
            files = {model: cache / "extract" / model / f"{doc_id}.jsonl" for model in MODELS}
            files = {model: file for model, file in files.items() if file.is_file()}
            if not files:
                continue
            if doc_id not in gold or gold[doc_id].collection != collection:
                raise ValueError(f"Missing matching gold for {doc_id}")
            text_path = Path(doc["text_path"]).resolve()
            if not text_path.is_relative_to(source.resolve()) or not text_path.is_file():
                text_path = source / "text" / f"{doc_id}.txt"
            text_path = _local_path(text_path, source.resolve())
            if not text_path.is_file():
                raise ValueError(f"Missing document text for {doc_id}")
            staged_docs.append({**doc, "text_path": str(text_path)})
            for model, file in files.items():
                before = _records(_local_path(file, file.parent), doc_id)
                cohorts[model][doc_id] = (collection, doc["split"], len(before))
                relative = Path("cache/extract") / model / file.name
                _write_jsonl(out / "before" / relative, before)
                _write_jsonl(out / "after" / relative, map(reparse, before))
        target = out / "data" / collection
        _write_jsonl(target / "docs.jsonl", staged_docs)
        _write_jsonl(target / "gold.jsonl", (gold[doc["doc_id"]].model_dump(mode="json") for doc in staged_docs))
    for model, docs in cohorts.items():
        missing = {"train", "dev"} - {split for _, split, _ in docs.values()}
        if missing:
            raise ValueError(f"{model} is missing cached records for {', '.join(sorted(missing))}")
    return cohorts


def run_pipeline(out: Path, cohorts: dict, lam: float, costs: Path) -> None:
    data = out / "data"
    docs = full_extraction.all_docs(data)
    for stage_name in ("before", "after"):
        cache, results = out / stage_name / "cache", out / stage_name / "results"
        common = ["--cache", str(cache), "--results", str(results)]
        for model, cohort in cohorts.items():
            print(f"{stage_name} {model}: verify and check {len(cohort)} cached documents", flush=True)
            full_extraction.cmd_verify(model, docs, cache)
            for collection in SPLITS:
                ids = [doc for doc, (group, _, _) in cohort.items() if group == collection]
                if not ids:
                    continue
                scoped_docs = {doc: docs[doc] for doc in ids}
                full_extraction.cmd_inject(model, scoped_docs, cache, 2, results / f"injection_{model}_{collection}.md")
                by_k = defaultdict(list)
                for doc in ids:
                    by_k[cohort[doc][2]].append(doc)
                for k, plan in sorted(by_k.items()):
                    sheet = results / f"spot_check_{model}_{collection}_k{k}.md"
                    problems, _ = full_extraction.cmd_check(model, scoped_docs, cache, k, plan, sheet, data)
                    if problems:
                        raise ValueError(f"Cache check failed for {stage_name} {model}: {problems[0]}")
        # Both models' verification precedes the cross-model feature build.
        for model in MODELS:
            args = ["--model", model, *common]
            build_features.main([*args, "--data", str(data)])
            train_risk.main(args)
            decide.main([*args, "--log-lambda", str(lam), "--costs", str(costs)])


def comparison(out: Path, lam: float, cost_file: Path) -> str:
    import pandas as pd

    costs = load_costs(cost_file)
    lines = ["# Extraction separator replay", "",
             f"Paired cached fields, calibrated full logistic risk, field review at λ = {lam:g}.",
             f"Costs from {cost_file.name}. Dev was used for calibration.", "",
             "| Extractor | Collection | Split | Fields | Error before | Error after | Review before | Review after | AUROC before | AUROC after |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in MODELS:
        before, after = [pd.read_parquet(out / name / "cache" / "risk" / f"{model}.parquet")
                         for name in ("before", "after")]
        keys = ["collection", "split", "doc_id", "field"]
        before, after = (table.sort_values(keys).reset_index(drop=True) for table in (before, after))
        if before.duplicated(keys).any() or not before[keys].equals(after[keys]):
            raise ValueError(f"Paired field coverage changed for {model}")
        for collection, splits in SPLITS.items():
            for split in splits:
                if split == "train":
                    continue
                parts = [table[(table["collection"] == collection) & (table["split"] == split)]
                         for table in (before, after)]
                if parts[0].empty:
                    lines.append("| " + " | ".join([model, collection, split, "0", *(["NA"] * 6)]) + " |")
                    continue
                errors = [f"{100 * part['e'].mean():.2f}%" for part in parts]
                reviews = [f"{100 * (1 - evaluate_lambda(part, PROBABILITY, lam, costs)['automated']):.2f}%"
                           for part in parts]
                auc = [metrics(part["e"], part[PROBABILITY])["auroc"] for part in parts]
                scores = [f"{value:.4f}" if math.isfinite(value) else "NA" for value in auc]
                row = [model, collection, split, str(len(parts[0])), *errors, *reviews, *scores]
                lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "normalisation_replay")
    parser.add_argument("--costs", type=Path, default=ROOT / "configs" / "costs.yaml")
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.lam) or args.lam < 0:
        parser.error("lambda must be finite and nonnegative")
    try:
        for path in (args.data / "synthetic", args.data / "adversarial",
                     *(args.cache / "extract" / model for model in MODELS)):
            if not path.is_dir():
                raise ValueError(f"Missing replay input directory: {path}")
        args.out.mkdir(parents=True, exist_ok=False)
        cohorts = stage(args.data, args.cache, args.out)
        run_pipeline(args.out, cohorts, args.lam, args.costs)
        report = comparison(args.out, args.lam, args.costs)
        (args.out / "comparison.md").write_text(report, encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
