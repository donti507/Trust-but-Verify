"""Score extraction records against gold (Step 11 pilot; reused in later steps).

Per field and model, using the k = 0 sample:
* ``match``       gold explicit/by_reference value equals the extracted value
                  after normalisation; or gold absent and nothing extracted.
* ``mismatch``    anything else (wrong value, missed value, invented value).
* ``ambiguous``   gold says the document is ambiguous: the correct action is
                  ESCALATE, so no extraction can "match"; counted separately.
Consistency = share of the K samples equal to the modal value (Step 13 feature).
Collections are never pooled (rule 4): every table is per collection.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

from trustswap.common.records import GoldRecord
from trustswap.extract.fields import PATHS
from trustswap.synth.render import trade_fields


def load_gold(path: Path) -> dict[str, GoldRecord]:
    return {record.doc_id: record for record in
            (GoldRecord.model_validate_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
             if line.strip())}


def load_records(directory: Path) -> dict[str, list[dict]]:
    """{doc_id: [record for k = 0..K-1]} from cache/extract/<model>/."""
    result = {}
    for path in sorted(Path(directory).glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        result[path.stem] = sorted(records, key=lambda r: r["k"])
    return result


def canonical(path: str, value):
    """Comparable form of a typed gold value or a JSON record value."""
    if value is None:
        return None
    name = path.rsplit(".", 1)[-1]
    if name in {"fixed_rate", "spread", "notional_amount"}:
        try:
            return Decimal(str(value)).normalize()
        except InvalidOperation:
            return str(value)
    if name.endswith("_date"):
        return value.isoformat() if isinstance(value, date) else str(value)
    if name == "business_centers":
        return tuple(sorted(value))
    if name == "floating_rate_option":
        return str(value).upper()
    return " ".join(str(value).split()).casefold()


def outcome(path: str, predicted, gold_term) -> str:
    if gold_term is not None and gold_term.status == "ambiguous":
        return "ambiguous"
    gold_value = None if gold_term is None else gold_term.value
    return "match" if canonical(path, predicted) == canonical(path, gold_value) else "mismatch"


def _predicted(record: dict, path: str):
    item = record.get("fields", {}).get(path) if record.get("schema_valid") else None
    return None if item is None else item.get("value")


def consistency(records: list[dict], path: str) -> float | None:
    values = [canonical(path, _predicted(r, path)) for r in records if r.get("schema_valid")]
    if not values:
        return None
    return Counter(map(repr, values)).most_common(1)[0][1] / len(values)


@dataclass
class ModelReport:
    model: str
    documents: int = 0
    samples: int = 0
    valid: int = 0
    structure_match: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    fresh_tokens: int = 0
    latency: list[float] = field(default_factory=list)
    fields: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    consistency: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    errors: dict[str, list[tuple]] = field(default_factory=lambda: defaultdict(list))

    def add(self, doc_id: str, records: list[dict], gold: GoldRecord) -> None:
        self.documents += 1
        for record in records:
            self.samples += 1
            self.valid += bool(record.get("schema_valid"))
            used = (record.get("input_tokens") or 0) + (record.get("output_tokens") or 0)
            self.input_tokens += record.get("input_tokens") or 0
            self.output_tokens += record.get("output_tokens") or 0
            if not record.get("cached", True):
                self.fresh_tokens += used
            if record.get("latency_s") is not None:
                self.latency.append(record["latency_s"])
        first = records[0]
        gold_structure = "fixed_float" if gold.trade.fixed_leg is not None else "float_float"
        self.structure_match += first.get("structure") == gold_structure
        gold_fields = trade_fields(gold.trade)
        for path in PATHS:
            term = gold_fields.get(path)
            if term is None:
                continue  # leg not in the gold trade
            predicted = _predicted(first, path)
            result = outcome(path, predicted, term)
            self.fields[path][result] += 1
            if result == "mismatch" and len(self.errors[path]) < 3:
                raw = (first.get("fields", {}).get(path) or {}).get("raw_value")
                self.errors[path].append((doc_id, raw, None if term.value is None else str(term.value)))
            share = consistency(records, path)
            if share is not None:
                self.consistency[path].append(share)

    @property
    def accuracy(self) -> float:
        totals = Counter()
        for counts in self.fields.values():
            totals.update(counts)
        scored = totals["match"] + totals["mismatch"]
        return totals["match"] / scored if scored else 0.0


def cost(tokens_in: int, tokens_out: int, price: dict | None) -> float | None:
    if not price:
        return None
    return tokens_in / 1e6 * price["input_per_million"] + tokens_out / 1e6 * price["output_per_million"]


def render_markdown(reports: dict[tuple[str, str], ModelReport], *, prompt_version: str, prompt_sha: str,
                    prices: dict[str, dict], currency: str) -> str:
    """reports keyed by (collection, model_key)."""
    lines = ["# Extraction pilot (Step 11)", "",
             f"Prompt `{prompt_version}` (sha256 `{prompt_sha[:16]}`). k = 0 at temperature 0 is scored; "
             "consistency uses all K samples. Collections are reported separately.", ""]
    for collection in sorted({c for c, _ in reports}):
        lines += [f"## {collection}", "",
                  "| model | docs | K | schema-valid | structure | field accuracy | latency (s, mean) | "
                  f"tokens/doc (in+out) | cost/doc ({currency}) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        keys = sorted(m for c, m in reports if c == collection)
        for key in keys:
            r = reports[(collection, key)]
            k = r.samples // r.documents if r.documents else 0
            per_doc = (r.input_tokens + r.output_tokens) / r.documents if r.documents else 0
            c = cost(r.input_tokens, r.output_tokens, prices.get(key))
            lat = sum(r.latency) / len(r.latency) if r.latency else float("nan")
            lines.append(f"| {r.model} | {r.documents} | {k} | {100 * r.valid / max(r.samples, 1):.1f}% | "
                         f"{100 * r.structure_match / max(r.documents, 1):.0f}% | **{100 * r.accuracy:.1f}%** | "
                         f"{lat:.1f} | {per_doc:,.0f} | "
                         + ("n/a" if c is None else f"{c / max(r.documents, 1):.4f}") + " |")
        lines += ["", "Per-field accuracy (k = 0; ambiguous gold excluded) and K-sample consistency:", "",
                  "| field | " + " | ".join(f"{reports[(collection, k)].model} acc | cons" for k in keys) + " |",
                  "| --- | " + " | ".join("--- | ---" for _ in keys) + " |"]
        for path in PATHS:
            cells = []
            present = False
            for key in keys:
                counts = reports[(collection, key)].fields.get(path)
                scored = (counts["match"] + counts["mismatch"]) if counts else 0
                present |= bool(counts)
                cons = reports[(collection, key)].consistency.get(path)
                cells.append((f"{100 * counts['match'] / scored:.0f}% ({counts['match']}/{scored})" if scored else "n/a")
                             + " | " + (f"{sum(cons) / len(cons):.2f}" if cons else "n/a"))
            if present:
                lines.append(f"| {path} | " + " | ".join(cells) + " |")
        lines += ["", "Example mismatches (doc, extracted text, gold):", ""]
        for key in keys:
            r = reports[(collection, key)]
            shown = [(p, e) for p in PATHS for e in r.errors.get(p, [])][:15]
            lines.append(f"**{r.model}**" + ("" if shown else ": none"))
            lines += [f"- `{p}` {doc}: extracted `{raw}`, gold `{gold}`" for p, (doc, raw, gold) in shown]
            lines.append("")
    return "\n".join(lines)
