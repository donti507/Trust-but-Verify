"""Run extraction for a set of documents (Step 10 check, later Steps 11-12).

    python scripts/extract.py --model frontier --docs data/synthetic/docs.jsonl --limit 10
    python scripts/extract.py --model open_weights --docs data/synthetic/docs.jsonl --limit 10 --k 5

Raw completions are cached in cache/raw/<model_key>/ and records written to
cache/extract/<model_key>/<doc_id>.jsonl. The real_test split is always refused.
Prints the schema-valid rate (Step 10 check: 100% on 10 documents per model).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from trustswap.common.cache import Cache
from trustswap.extract.clients import OpenAICompatClient, load_models
from trustswap.extract.run import BLOCKED_SPLITS, extract_document

ROOT = Path(__file__).resolve().parents[1]


def read_docs(path: Path) -> list[dict]:
    docs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for doc in docs:
        text_path = Path(doc["text_path"])
        if not text_path.exists():  # data copied from another machine: look next to docs.jsonl
            text_path = path.parent / "text" / f"{doc['doc_id']}.txt"
        doc["_text_path"] = text_path
    return docs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["frontier", "open_weights"])
    parser.add_argument("--docs", type=Path, required=True, help="docs.jsonl of one collection")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--k", type=int, default=1, help="samples per document (k=0 at temperature 0)")
    parser.add_argument("--split", action="append", help="only these splits (repeatable)")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    args = parser.parse_args(argv)

    spec = load_models()[args.model]
    spec.check_pinned()
    client = OpenAICompatClient(spec)
    cache = Cache(args.cache / "raw" / args.model)
    out_dir = args.cache / "extract" / args.model

    docs = [d for d in read_docs(args.docs) if d.get("split") not in BLOCKED_SPLITS
            and (not args.split or d.get("split") in args.split)][: args.limit]
    if any(d["collection"] == "real" for d in docs) and args.model == "frontier":
        print("note: real documents go to the frontier API; only public EDGAR filings are allowed")
    valid = total = 0
    for doc in docs:
        text = doc["_text_path"].read_text(encoding="utf-8")
        records = extract_document(doc["doc_id"], text, client, model_id=spec.model_id, cache=cache,
                                   out_dir=out_dir, k_samples=args.k, split=doc.get("split"))
        for record in records:
            total += 1
            valid += record["schema_valid"]
            if not record["schema_valid"]:
                print(f"  invalid: {doc['doc_id']} k={record['k']}: {record['errors'][:2]}")
        print(f"{doc['doc_id']}: {sum(r['schema_valid'] for r in records)}/{len(records)} schema-valid")
    rate = valid / total if total else 0.0
    print(f"schema-valid: {valid}/{total} = {100 * rate:.1f}%  ({spec.model_id}, records in {out_dir})")
    return 0 if total and valid == total else 1


if __name__ == "__main__":
    sys.exit(main())
