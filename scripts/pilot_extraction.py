"""Step 11: extraction pilot.

    # 1. make 100 synthetic pilot documents (train + dev families, never syn_test)
    python scripts/pilot_extraction.py make-synthetic --n 100
    # 2. iterate the prompt on dev only (cheap: K = 1)
    python scripts/pilot_extraction.py run --model frontier --split dev --k 1
    python scripts/pilot_extraction.py report
    # 3. final pilot: K = 5, both models, with a token budget
    python scripts/pilot_extraction.py run --model both --k 5 --budget-tokens 3000000
    python scripts/pilot_extraction.py run --model both --k 5 --docs data/real/docs.jsonl --split real_cal --limit 20
    python scripts/pilot_extraction.py report
    # 4. freeze the prompt
    python scripts/pilot_extraction.py freeze

Raw completions are cached, so re-runs are free and an interrupted run resumes.
The token budget counts only fresh (uncached) calls in this run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import json
from pathlib import Path
from random import Random
import sys
import threading

import yaml

from trustswap.common.cache import Cache
from trustswap.extract.clients import OpenAICompatClient, load_models
from trustswap.extract.evaluate import ModelReport, load_gold, load_records, render_markdown
from trustswap.extract.prompt import PROMPT_VERSION, prompt_sha
from trustswap.extract.run import BLOCKED_SPLITS, extract_document

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "pilot"
FREEZE = ROOT / "prereg" / "prompt-freeze.json"
BASE_SEED = 20261124
TRACKS = ("frontier", "open_weights")


def read_docs(path: Path) -> list[dict]:
    docs = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for doc in docs:
        text_path = Path(doc["text_path"])
        if not text_path.exists():
            text_path = Path(path).parent / "text" / f"{doc['doc_id']}.txt"
        doc["_text_path"] = text_path
    return docs


# ---------- make-synthetic ----------

def make_synthetic(n: int, out: Path = PILOT) -> Path:
    """n documents spread over train and dev families (syn_test is never used for the pilot)."""
    from trustswap.synth.render import render_trade
    from trustswap.synth.sampler import sample_trade
    from trustswap.synth.templates import FAMILY_SPLITS

    families = [f for f, split in FAMILY_SPLITS.items() if split in {"train", "dev"}]
    docs_file = out / "synthetic" / "docs.jsonl"
    existing = {json.loads(line)["doc_id"] for line in docs_file.read_text().splitlines()} if docs_file.exists() else set()
    for i in range(n):
        doc_id = f"pilot_{i:04d}"
        if doc_id in existing:
            continue
        seed = BASE_SEED + i
        render_trade(sample_trade(Random(seed)), seed=seed, doc_id=doc_id,
                     template_family=families[i % len(families)], output_dir=out)
    return docs_file


# ---------- run ----------

def run(tracks, docs: list[dict], k: int, budget: int | None, workers: int, cache_root: Path, log=print,
        retry_max_tokens: int | None = None) -> dict:
    models = load_models()
    used = {track: 0 for track in tracks}
    lock = threading.Lock()
    stopped = threading.Event()
    summary = {}
    for track in tracks:
        spec = models[track]
        spec.check_pinned()
        client = OpenAICompatClient(spec)
        cache = Cache(cache_root / "raw" / track)
        out_dir = cache_root / "extract" / track
        valid = total = 0

        def one(doc):
            if stopped.is_set():
                return None
            text = doc["_text_path"].read_text(encoding="utf-8")
            return extract_document(doc["doc_id"], text, client, model_id=spec.model_id, cache=cache,
                                    out_dir=out_dir, k_samples=k, split=doc.get("split"),
                                    retry_max_tokens=retry_max_tokens)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(one, doc): doc for doc in docs}
            for done, future in enumerate(as_completed(futures), start=1):
                records = future.result()
                if records is None:
                    continue
                fresh = sum(r["fresh_tokens"] for r in records)
                with lock:
                    used[track] += fresh
                    valid += sum(r["schema_valid"] for r in records)
                    total += len(records)
                    if budget is not None and sum(used.values()) >= budget and not stopped.is_set():
                        stopped.set()
                        log(f"  token budget {budget:,} reached: stopping after running calls finish")
                if done % 10 == 0 or done == len(docs):
                    log(f"  {track}: {done}/{len(docs)} docs, fresh tokens {used[track]:,}")
        summary[track] = {"valid": valid, "total": total, "fresh_tokens": used[track]}
        log(f"{track}: schema-valid {valid}/{total}, fresh tokens {used[track]:,}")
        if stopped.is_set():
            break
    return summary


def estimate(tracks, n_docs: int, k: int, cache_root: Path) -> dict[str, int]:
    """Tokens needed, from the average of records already in the cache (6,000 per call if none)."""
    result = {}
    for track in tracks:
        records = [r for rs in load_records(cache_root / "extract" / track).values() for r in rs]
        per_call = (sum((r.get("input_tokens") or 0) + (r.get("output_tokens") or 0) for r in records) / len(records)
                    if records else 6000)
        result[track] = int(per_call * n_docs * k)
    return result


# ---------- report ----------

def report(collections: dict[str, Path], cache_root: Path, out: Path) -> str:
    models = load_models()
    pricing_file = ROOT / "configs" / "pricing.yaml"
    pricing = yaml.safe_load(pricing_file.read_text()) if pricing_file.exists() else {"currency": "", "models": {}}
    reports = {}
    for collection, docs_path in collections.items():
        if not docs_path.exists():
            continue
        docs = {d["doc_id"]: d for d in read_docs(docs_path) if d.get("split") not in BLOCKED_SPLITS}
        gold = load_gold(docs_path.parent / "gold.jsonl")
        for track in TRACKS:
            records = load_records(cache_root / "extract" / track)
            rep = ModelReport(model=models[track].name)
            for doc_id in sorted(docs):
                if doc_id in records and doc_id in gold:
                    rep.add(doc_id, records[doc_id], gold[doc_id])
            if rep.documents:
                reports[(collection, track)] = rep
    markdown = render_markdown(reports, prompt_version=PROMPT_VERSION, prompt_sha=prompt_sha(),
                               prices=pricing["models"], currency=pricing["currency"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    return markdown


# ---------- main ----------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make-synthetic")
    make.add_argument("--n", type=int, default=100)
    go = sub.add_parser("run")
    go.add_argument("--model", choices=[*TRACKS, "both"], default="both")
    go.add_argument("--docs", type=Path, default=PILOT / "synthetic" / "docs.jsonl")
    go.add_argument("--split", action="append", help="only these splits (repeatable)")
    go.add_argument("--limit", type=int)
    go.add_argument("--k", type=int, default=5)
    go.add_argument("--budget-tokens", type=int, help="stop after this many fresh tokens (all models)")
    go.add_argument("--workers", type=int, default=4)
    go.add_argument("--yes", action="store_true", help="skip the estimate confirmation")
    rep = sub.add_parser("report")
    rep.add_argument("--synthetic", type=Path, default=PILOT / "synthetic" / "docs.jsonl")
    rep.add_argument("--real", type=Path, default=ROOT / "data" / "real" / "docs.jsonl")
    rep.add_argument("--out", type=Path, default=ROOT / "results" / "pilot_extraction.md")
    sub.add_parser("freeze")
    for p in (go, rep):
        p.add_argument("--cache", type=Path, default=ROOT / "cache")
    args = parser.parse_args(argv)

    if args.command == "make-synthetic":
        print(f"wrote {make_synthetic(args.n)}")
        return 0
    if args.command == "run":
        tracks = TRACKS if args.model == "both" else (args.model,)
        docs = [d for d in read_docs(args.docs) if d.get("split") not in BLOCKED_SPLITS
                and (not args.split or d.get("split") in args.split)]
        if any(d["collection"] == "real" and d.get("split") != "real_cal" for d in docs):
            raise SystemExit("pilot real documents must come from real_cal only (use --split real_cal)")
        docs = docs[: args.limit] if args.limit else docs
        need = estimate(tracks, len(docs), args.k, args.cache)
        print(f"{len(docs)} documents x K={args.k}; estimated tokens (before cache): "
              + ", ".join(f"{t} {n:,}" for t, n in need.items()) + f" = {sum(need.values()):,}")
        if not args.yes and input("continue? [y/N] ").strip().lower() != "y":
            return 1
        run(tracks, docs, args.k, args.budget_tokens, args.workers, args.cache)
        return 0
    if args.command == "report":
        markdown = report({"synthetic": args.synthetic, "real": args.real}, args.cache, args.out)
        print(markdown)
        print(f"\nwrote {args.out}")
        return 0
    if args.command == "freeze":
        record = {"prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_sha(), "date": date.today().isoformat(),
                  "status": "frozen", "note": "Changing the prompt after this requires a new version (v2, ...)."}
        FREEZE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"froze prompt {PROMPT_VERSION}: {record['prompt_sha256'][:16]} -> {FREEZE}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
