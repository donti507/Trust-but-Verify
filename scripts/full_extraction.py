"""Step 12: full extraction, verifier records, extracted-field injection, checks.

    # frontier (PCSS, budget-limited): the planned subset, K = 1
    python scripts/full_extraction.py run --model frontier --plan data/s12/frontier_docs.txt --k 1 --budget-tokens 3000000
    # open weights (Kaggle vLLM): every document, K = 5
    python scripts/full_extraction.py run --model open_weights --k 5 --workers 16
    # verifier records, injection experiment, completeness check + spot-check sheet
    python scripts/full_extraction.py verify --model frontier
    python scripts/full_extraction.py inject --model frontier
    python scripts/full_extraction.py check --model frontier --k 1 --plan data/s12/frontier_docs.txt

Outputs: cache/extract/<model>/, cache/verify/<model>/, cache/extract_injected/<model>/,
cache/verify_injected/<model>/, results/injection_tests_<model>.md, results/spot_check_<model>.md.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from random import Random
import sys

from trustswap.extract.evaluate import load_gold, load_records
from trustswap.extract.inject import KINDS, inject_all
from trustswap.extract.parse import trade_from_fields
from trustswap.extract.run import BLOCKED_SPLITS
from trustswap.verify.evaluate import verifier_hash
from trustswap.verify.run import to_record, verify

if __package__:
    from . import pilot_extraction as P
else:
    import pilot_extraction as P

ROOT = Path(__file__).resolve().parents[1]
COLLECTIONS = ("synthetic", "adversarial")


def all_docs(data: Path) -> dict[str, dict]:
    docs = {}
    for collection in COLLECTIONS:
        path = data / collection / "docs.jsonl"
        if path.exists():
            for doc in P.read_docs(path):
                if doc.get("split") not in BLOCKED_SPLITS:
                    docs[doc["doc_id"]] = doc
    return docs


def read_plan(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_record(record: dict, text: str) -> dict:
    """Verifier record for one extraction record (k = 0)."""
    base = {"doc_id": record["doc_id"], "model": record["model"], "k": record["k"],
            "verifier_hash": verifier_hash()}
    if not record.get("schema_valid"):
        return {**base, "schema_valid": False, "rules": {}}
    try:
        trade = trade_from_fields(record["fields"])
    except ValueError as error:  # values the schema cannot hold
        return {**base, "schema_valid": True, "trade_error": str(error)[:300], "rules": {}}
    return {**base, "schema_valid": True, **to_record(record["doc_id"], verify(trade, text))}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def cmd_verify(model: str, docs: dict, cache: Path) -> int:
    records = load_records(cache / "extract" / model)
    n = 0
    for doc_id, recs in records.items():
        if doc_id not in docs:
            continue
        text = docs[doc_id]["_text_path"].read_text(encoding="utf-8")
        write_json(cache / "verify" / model / f"{doc_id}.json", verify_record(recs[0], text))
        n += 1
    return n


def fired_on(rules: dict, fields: list[str]) -> set[str]:
    return {rule for rule, found in rules.items() for path in fields
            if found.get(path, {}).get("verdict") in {"HARD", "SOFT"}}


def cmd_inject(model: str, docs: dict, cache: Path, copies: int, out: Path) -> str:
    records = load_records(cache / "extract" / model)
    detected = defaultdict(Counter)
    for doc_id, recs in sorted(records.items()):
        if doc_id not in docs:
            continue
        seed = int(hashlib.sha256(f"{model}:{doc_id}".encode()).hexdigest()[:8], 16)
        variants = inject_all(recs[0], copies, seed)
        text = docs[doc_id]["_text_path"].read_text(encoding="utf-8")
        lines_ex, lines_ver = [], []
        for variant in variants:
            result = verify_record(variant, text)
            result["injection"] = variant["injection"]
            lines_ex.append(json.dumps(variant, sort_keys=True, default=str))
            lines_ver.append(json.dumps(result, sort_keys=True, default=str))
            kind = variant["injection"]["kind"]
            rules = fired_on(result.get("rules", {}), variant["injection"]["fields"])
            detected[kind]["total"] += 1
            detected[kind]["detected"] += bool(rules)
            for rule in rules:
                detected[kind][rule] += 1
        for folder, lines in (("extract_injected", lines_ex), ("verify_injected", lines_ver)):
            path = cache / folder / model / f"{doc_id}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    rules = [f"V{i}" for i in range(1, 8)]
    lines = [f"# Extracted-field injection ({model})", "",
             "One planted error per copy of a k = 0 extraction; span left on the true text.",
             "Detected = any rule reports HARD or SOFT on the injected field(s). Kept apart from",
             "document-perturbation results.", "",
             "| injection | copies | detected | " + " | ".join(rules) + " |",
             "| --- | --- | --- | " + " | ".join("---" for _ in rules) + " |"]
    for kind in KINDS:
        c = detected.get(kind)
        if not c:
            continue
        lines.append(f"| {kind} | {c['total']} | {100 * c['detected'] / c['total']:.0f}% | "
                     + " | ".join(f"{100 * c[r] / c['total']:.0f}%" for r in rules) + " |")
    markdown = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    return markdown


def cmd_check(model: str, docs: dict, cache: Path, k: int, plan: list[str] | None, out: Path,
              data: Path, n_spot: int = 20) -> tuple[list[str], str]:
    expected = plan if plan is not None else sorted(docs)
    records = load_records(cache / "extract" / model)
    problems = []
    for doc_id in expected:
        if len(records.get(doc_id, [])) != k:
            problems.append(f"{doc_id}: {len(records.get(doc_id, []))} records, expected {k}")
        if not (cache / "verify" / model / f"{doc_id}.json").exists():
            problems.append(f"{doc_id}: no verifier record")
    gold = {}
    for collection in COLLECTIONS:
        if (data / collection / "gold.jsonl").exists():
            gold.update(load_gold(data / collection / "gold.jsonl"))
    sample = Random(7).sample([d for d in expected if d in records and d in gold], min(n_spot, len(expected)))
    from trustswap.synth.render import trade_fields

    lines = [f"# Spot check ({model}): {len(sample)} documents", "",
             "Tick each row after reading the document text. `!` marks a disagreement with gold.", ""]
    for doc_id in sample:
        record = records[doc_id][0]
        verdicts = {}
        ver_path = cache / "verify" / model / f"{doc_id}.json"
        if ver_path.exists():
            for rule, found in json.loads(ver_path.read_text()).get("rules", {}).items():
                for path, item in found.items():
                    if item["verdict"] in {"HARD", "SOFT"}:
                        verdicts.setdefault(path, []).append(f"{rule}:{item['verdict']}")
        lines += [f"## {doc_id} ({docs[doc_id].get('split')})", f"text: `{docs[doc_id]['_text_path']}`", "",
                  "| ok? | field | extracted | gold | flags |", "| --- | --- | --- | --- | --- |"]
        for path, term in trade_fields(gold[doc_id].trade).items():
            item = record.get("fields", {}).get(path, {})
            value = item.get("value")
            gold_value = "AMBIGUOUS" if term.status == "ambiguous" else term.value
            mark = "" if str(value) == str(gold_value) else "!"
            lines.append(f"| [ ] | {path} | {item.get('raw_value')} | {gold_value} | {mark} {' '.join(verdicts.get(path, []))} |")
        lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return problems, str(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--model", required=True, choices=P.TRACKS)
    go.add_argument("--plan", type=Path)
    go.add_argument("--k", type=int, default=5)
    go.add_argument("--budget-tokens", type=int)
    go.add_argument("--workers", type=int, default=4)
    go.add_argument("--retry-max-tokens", type=int, default=16384,
                    help="retry calls cut off by the output limit once with this limit (0 = never)")
    go.add_argument("--yes", action="store_true")
    for name in ("verify", "inject", "check"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, choices=P.TRACKS)
    sub.choices["inject"].add_argument("--copies", type=int, default=2)
    sub.choices["check"].add_argument("--k", type=int, default=5)
    sub.choices["check"].add_argument("--plan", type=Path)
    args = parser.parse_args(argv)

    docs = all_docs(args.data)
    if args.command == "run":
        plan = read_plan(args.plan)
        selected = [docs[d] for d in plan if d in docs] if plan is not None else list(docs.values())
        need = P.estimate((args.model,), len(selected), args.k, args.cache)
        print(f"{len(selected)} documents x K={args.k}; estimated tokens (before cache): {need[args.model]:,}")
        if not args.yes and input("continue? [y/N] ").strip().lower() != "y":
            return 1
        P.run((args.model,), selected, args.k, args.budget_tokens, args.workers, args.cache,
              retry_max_tokens=args.retry_max_tokens or None)
    elif args.command == "verify":
        print(f"verifier records: {cmd_verify(args.model, docs, args.cache)}")
    elif args.command == "inject":
        print(cmd_inject(args.model, docs, args.cache, args.copies,
                         ROOT / "results" / f"injection_tests_{args.model}.md"))
    elif args.command == "check":
        problems, sheet = cmd_check(args.model, docs, args.cache, args.k, read_plan(args.plan),
                                    ROOT / "results" / f"spot_check_{args.model}.md", args.data)
        for line in problems[:20]:
            print("  " + line)
        print(f"{'OK' if not problems else f'{len(problems)} problems'}; spot-check sheet: {sheet}")
        return 0 if not problems else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
