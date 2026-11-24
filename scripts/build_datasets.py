"""Step 12 datasets: synthetic (clean + noisy), adversarial, and the frontier subset.

    python scripts/build_datasets.py synthetic --train 300 --dev 100 --syn-test 100
    python scripts/build_datasets.py noisy --fraction 0.5
    python scripts/build_datasets.py adversarial --parents 30
    python scripts/build_datasets.py plan --budget-tokens 3000000

Everything is deterministic from fixed seeds and idempotent: re-running skips
documents that already exist, so an interrupted build resumes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import re
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASE_SEED = 20261201
CLEAN_ID = re.compile(r"syn_(train|dev|syn_test)_\d{4}")  # parent twins of adversarial cases are excluded
# share of the frontier budget per group (adversarial is where errors live)
PLAN_SHARES = {"train": 0.30, "dev": 0.10, "syn_test": 0.12, "adv_test": 0.48}


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ids(path: Path) -> set[str]:
    return {r["doc_id"] for r in _jsonl(path)}


# ---------- synthetic ----------

def build_synthetic(out: Path, counts: dict[str, int], log=print) -> int:
    from trustswap.synth.render import render_trade
    from trustswap.synth.sampler import sample_trade
    from trustswap.synth.templates import FAMILY_SPLITS

    existing = _ids(out / "synthetic" / "docs.jsonl")
    made = 0
    for split, count in counts.items():
        families = [f for f, s in FAMILY_SPLITS.items() if s == split]
        for i in range(count):
            doc_id = f"syn_{split}_{i:04d}"
            if doc_id in existing:
                continue
            seed = BASE_SEED + hash_index(split, i)
            render_trade(sample_trade(Random(seed)), seed=seed, doc_id=doc_id,
                         template_family=families[i % len(families)], output_dir=out)
            made += 1
            if made % 50 == 0:
                log(f"  synthetic: {made} rendered")
    return made


def hash_index(split: str, i: int) -> int:
    return int(hashlib.sha256(f"{split}:{i}".encode()).hexdigest()[:8], 16)


# ---------- noisy ----------

def build_noisy(out: Path, fraction: float, log=print) -> int:
    """Noisy (scan-like) twins of clean synthetic documents; gold values unchanged, spans mapped."""
    from trustswap.common.records import DocumentRecord, GoldRecord
    from trustswap.synth.noise import add_noise, map_span
    from trustswap.synth.render import trade_fields

    base = out / "synthetic"
    docs, gold = _jsonl(base / "docs.jsonl"), {r["doc_id"]: r for r in _jsonl(base / "gold.jsonl")}
    existing = {d["doc_id"] for d in docs}
    clean = [d for d in docs if CLEAN_ID.fullmatch(d["doc_id"]) and not d.get("ocr")]
    made = 0
    with (base / "docs.jsonl").open("a", encoding="utf-8") as docs_out, \
            (base / "gold.jsonl").open("a", encoding="utf-8") as gold_out:
        for doc in clean:
            index = int(doc["doc_id"].rsplit("_", 1)[1])
            if (index * 0.6180339887) % 1 >= fraction:  # deterministic, evenly spread selection
                continue
            severity = 1 + index % 2
            noisy_id = f"{doc['doc_id']}_ocr{severity}"
            if noisy_id in existing:
                continue
            text_path = Path(doc["text_path"])
            if not text_path.exists():
                text_path = base / "text" / f"{doc['doc_id']}.txt"
            text, mapping = add_noise(text_path.read_text(encoding="utf-8"), seed=hash_index(noisy_id, 0),
                                      severity=severity)
            record = GoldRecord.model_validate(gold[doc["doc_id"]])
            trade = record.trade.model_copy(deep=True)
            for term in trade_fields(trade).values():
                term.span = map_span(term.span, mapping)
            noisy_path = base / "text" / f"{noisy_id}.txt"
            noisy_path.write_text(text, encoding="utf-8")
            doc_record = DocumentRecord(doc_id=noisy_id, collection="synthetic", family_id=doc["family_id"],
                                        split=doc["split"], sha256=hashlib.sha256(text.encode()).hexdigest(),
                                        text_path=str(noisy_path), ocr=True)
            gold_record = record.model_copy(update={"doc_id": noisy_id, "trade": trade})
            docs_out.write(doc_record.model_dump_json() + "\n")
            gold_out.write(gold_record.model_dump_json() + "\n")
            made += 1
    log(f"  noisy: {made} written")
    return made


# ---------- adversarial ----------

def build_adversarial(out: Path, parents: int, log=print) -> int:
    from trustswap.synth import defects as D
    from trustswap.synth.sampler import sample_trade

    existing = _ids(out / "adversarial" / "docs.jsonl")
    made = 0
    for p in range(parents):
        seed = BASE_SEED + 1_000_000 + p
        parent = sample_trade(Random(seed))
        cases = [D.INJECTORS[kind](parent, seed, severity=s) for kind in D.DEFECT_TYPES for s in D.SEVERITIES]
        cases += [D.make_control(kind, parent, seed) for kind in D.DEFECT_TYPES]
        for case in cases:
            if case.gold.doc_id in existing:
                continue
            D.render_case(case, output_dir=out)
            made += 1
        log(f"  adversarial: parent {p + 1}/{parents}")
    return made


# ---------- frontier plan ----------

def plan_frontier(out: Path, budget: int, tokens_per_doc: int, seed: int = BASE_SEED) -> dict[str, int]:
    """Deterministic stratified subset of documents for the budget-limited frontier model."""
    docs = _jsonl(out / "synthetic" / "docs.jsonl") + _jsonl(out / "adversarial" / "docs.jsonl")
    groups: dict[str, list[str]] = defaultdict(list)
    for doc in docs:
        split = doc.get("split")
        if split not in PLAN_SHARES:
            continue
        if split == "adv_test":
            key = f"adv_test/{doc.get('defect')}/{'control' if doc.get('is_control') else 'defect'}"
        else:
            key = f"{split}/{'noisy' if doc.get('ocr') else 'clean'}"
        groups[key].append(doc["doc_id"])
    total = budget // tokens_per_doc
    chosen, sizes = [], {}
    rng = Random(seed)
    for split, share in PLAN_SHARES.items():
        keys = sorted(k for k in groups if k.split("/")[0] == split)
        if not keys:
            continue
        per_group = int(total * share) // len(keys)
        for key in keys:
            members = sorted(groups[key])
            pick = members if len(members) <= per_group else rng.sample(members, per_group)
            chosen += sorted(pick)
            sizes[key] = len(pick)
    plan_path = out / "s12" / "frontier_docs.txt"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    return sizes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DATA)
    sub = parser.add_subparsers(dest="command", required=True)
    syn = sub.add_parser("synthetic")
    syn.add_argument("--train", type=int, default=300)
    syn.add_argument("--dev", type=int, default=100)
    syn.add_argument("--syn-test", type=int, default=100)
    noisy = sub.add_parser("noisy")
    noisy.add_argument("--fraction", type=float, default=0.5)
    adv = sub.add_parser("adversarial")
    adv.add_argument("--parents", type=int, default=30)
    plan = sub.add_parser("plan")
    plan.add_argument("--budget-tokens", type=int, default=3_000_000)
    plan.add_argument("--tokens-per-doc", type=int, default=6000)
    args = parser.parse_args(argv)

    if args.command == "synthetic":
        n = build_synthetic(args.out, {"train": args.train, "dev": args.dev, "syn_test": args.syn_test})
        print(f"synthetic: {n} new documents")
    elif args.command == "noisy":
        build_noisy(args.out, args.fraction)
    elif args.command == "adversarial":
        n = build_adversarial(args.out, args.parents)
        print(f"adversarial: {n} new documents")
    elif args.command == "plan":
        sizes = plan_frontier(args.out, args.budget_tokens, args.tokens_per_doc)
        for key, n in sizes.items():
            print(f"  {key}: {n}")
        print(f"frontier plan: {sum(sizes.values())} documents -> {args.out / 's12' / 'frontier_docs.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
