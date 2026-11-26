"""Build an offline HTML demo from generated documents and cached model outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd

from trustswap.decide.policy import COSTS_FILE, Costs, field_costs, load_costs
from trustswap.score.features import SEVERITY

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = Path(__file__).resolve().with_name("demo_frontend")
COLLECTIONS = {"synthetic": ("train", "dev", "syn_test"), "adversarial": ("adv_test",)}
CATEGORIES = ("Clean synthetic", "Noisy synthetic", "Adversarial")
RULE_TEXT = {
    "V1": "Required term missing or invalid.",
    "V2": "Value not found in the cited text.",
    "V3": "Dates out of order.",
    "V4": "Date off the payment schedule.",
    "V5": "Stated amount does not recompute.",
    "V6": "Terms contradict each other.",
    "V7": "Unusual convention for this market.",
}
LEGS = {"fixed_leg": "fixed leg", "floating_leg": "floating leg", "floating_leg_2": "second floating leg"}


def decision(p: float, c_bar: float, r: float, lam: float) -> str:
    """Mirror the browser decision, including acceptance at an exact tie."""
    return "Review" if p * c_bar > lam * r else "Accept"


def _jsonl(path: Path):
    if path.is_file():
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)


def _name(path: str) -> str:
    leg, _, field = path.rpartition(".")
    title = field.replace("_", " ").capitalize()
    return f"{title} ({LEGS[leg]})" if leg else title


def _span(span, length: int):
    if isinstance(span, list) and len(span) == 2 and all(type(x) is int for x in span):
        start, end = span
        if 0 <= start < end <= length:
            return span
    return None


def load_documents(data: Path, cache: Path, model: str, costs: Costs, n: int) -> list[dict]:
    """Select eligible cached documents by category, prioritizing review or errors at lambda 1."""
    risks = pd.read_parquet(
        cache / "risk" / f"{model}.parquet", engine="pyarrow",
        columns=["doc_id", "field", "collection", "split", "e", "p_full_logistic"],
        filters=[[("collection", "==", collection), ("split", "in", splits)]
                 for collection, splits in COLLECTIONS.items()],
    )
    c, r = field_costs(risks, costs)
    risks = risks.assign(c_bar=c, r=r)
    groups = {category: [] for category in CATEGORIES}
    for collection, splits in COLLECTIONS.items():
        base = data / collection
        gold_ids = {row["doc_id"] for row in _jsonl(base / "gold.jsonl") if row["collection"] == collection}
        manifests = {row["doc_id"]: row for row in _jsonl(base / "docs.jsonl")
                     if row["collection"] == collection and row.get("split") in splits}
        for doc_id, fields in risks.loc[risks["collection"] == collection].groupby("doc_id"):
            if doc_id not in gold_ids or doc_id not in manifests or not re.fullmatch(r"[\w][\w.-]*", doc_id):
                continue
            manifest = manifests[doc_id]
            if not fields["split"].eq(manifest["split"]).all():
                continue
            extract = cache / "extract" / model / f"{doc_id}.jsonl"
            verify = cache / "verify" / model / f"{doc_id}.json"
            path = Path(manifest["text_path"])
            if not path.is_file():
                path = base / "text" / f"{doc_id}.txt"
            if not (path.is_file() and extract.is_file() and verify.is_file()):
                continue
            if path.resolve().is_relative_to((data / "real").resolve()):
                continue

            category = "Adversarial" if collection == "adversarial" else (
                "Noisy synthetic" if doc_id.endswith(("_ocr1", "_ocr2")) else "Clean synthetic"
            )
            interesting = any(row.e or decision(row.p_full_logistic, row.c_bar, row.r, 1) == "Review"
                              for row in fields.itertuples(index=False))
            groups[category].append((not interesting, doc_id, path, extract, verify, fields))

    queues = {category: iter(sorted(rows, key=lambda row: row[:2])) for category, rows in groups.items()}
    documents = []
    while len(documents) < n:
        previous_count = len(documents)
        for category, queue in queues.items():
            for _, doc_id, path, extract, verify, fields in queue:
                sample = next((row for row in _jsonl(extract) if row["k"] == 0), None)
                if sample is None:
                    continue
                text = path.read_text(encoding="utf-8")
                rules = json.loads(verify.read_text(encoding="utf-8")).get("rules", {})
                extracted = sample.get("fields", {}) if sample.get("schema_valid", True) else {}
                entries = []
                for row in fields.itertuples(index=False):
                    item = extracted.get(row.field, {})
                    value = item.get("value")
                    entries.append({
                        "name": _name(row.field),
                        "value": item.get("raw_value") if value is None else value,
                        "span": _span(item.get("span"), len(text)),
                        "p": row.p_full_logistic, "c_bar": row.c_bar, "r": row.r,
                        "severity": SEVERITY[row.field.rsplit(".", 1)[-1]], "e": int(row.e),
                        "rules_fired": [rule for rule in RULE_TEXT
                                        if rules.get(rule, {}).get(row.field, {}).get("verdict") in {"HARD", "SOFT"}],
                    })
                documents.append({"id": doc_id, "category": category, "text": text, "fields": entries})
                break
            if len(documents) == n:
                break
        if previous_count == len(documents):
            break
    return documents


def render_page(documents: list[dict], bound: float, *, illustrative: bool = False) -> str:
    payload = json.dumps(
        {"documents": documents, "rules": RULE_TEXT, "bound": bound, "illustrative": illustrative},
        allow_nan=False, separators=(",", ":"),
    )
    # Keep data inside the JSON script element even if document text contains HTML.
    payload = payload.replace("<", "\\u003c")
    payload = re.sub(r"(?i)(https?):", lambda match: match[1] + r"\u003a", payload)
    template = (FRONTEND / "index.html").read_text(encoding="utf-8")
    components = {
        "CSS": (FRONTEND / "style.css").read_text(encoding="utf-8"),
        "JS": (FRONTEND / "app.js").read_text(encoding="utf-8"),
        "DATA": payload,
    }
    return re.sub(r"__DEMO_(CSS|JS|DATA)__", lambda match: components[match[1]], template)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("frontier", "open_weights"), default="open_weights")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--costs", type=Path, default=COSTS_FILE)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "demo" / "index.html")
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be positive")
    risk_path = args.cache / "risk" / f"{args.model}.parquet"
    if not risk_path.is_file():
        parser.error(f"cached risks are missing at {risk_path}")
    costs = load_costs(args.costs)
    documents = load_documents(args.data, args.cache, args.model, costs, args.n)
    if not documents:
        parser.error("no generated documents have matching risk, gold, text, extraction, and verifier records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_page(documents, costs.B), encoding="utf-8")
    print(f"Wrote {args.output} ({len(documents)} documents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
