"""Feed-only extraction, token accounting and review scoring."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from trustswap.common.cache import Cache
from trustswap.decide.policy import escalate, field_costs
from trustswap.extract.clients import OpenAICompatClient
from trustswap.extract.fields import output_schema
from trustswap.extract.parse import trade_from_fields
from trustswap.extract.run import extract_document
from trustswap.score.features import SEVERITY, field_rows
from trustswap.verify.run import to_record, verify


class FeedSkip(RuntimeError):
    """An expected condition that must leave the published feed untouched."""


class SingleAttemptClient(OpenAICompatClient):
    def _create(self, kwargs):
        # A provider retry could spend tokens without the daily ledger seeing it.
        with self._client() as client:
            return client.chat.completions.create(**kwargs)


class BudgetedClient:
    def __init__(self, client, ledger: Path, limit: int):
        self.client, self.path, self.limit = client, ledger, limit
        self.params = client.params

    def _write(self, count):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(str(count), encoding="ascii")
        tmp.replace(self.path)

    def complete(self, messages, *, temperature, seed):
        used = int(self.path.read_text()) if self.path.exists() else 0
        # UTF-8 request bytes provide a conservative input reservation. The output
        # reservation includes the provider's full configured completion allowance.
        wire = json.dumps([messages, output_schema(), self.params], ensure_ascii=False).encode()
        reservation = len(wire) + 512 + int(self.params["max_tokens"])
        if used + reservation > self.limit:
            raise FeedSkip("60k daily token cap leaves insufficient room for the next request")
        self._write(used + reservation)
        try:
            completion = self.client.complete(messages, temperature=temperature, seed=seed)
        except Exception:
            raise FeedSkip("PCSS request failed") from None
        if completion.input_tokens is not None and completion.output_tokens is not None:
            spent = completion.input_tokens + completion.output_tokens
            self._write(used + spent)
            if used + spent >= self.limit:
                raise FeedSkip("PCSS reported usage reached the daily token cap")
        return completion


@dataclass
class ReviewPipeline:
    client: BudgetedClient
    model_id: str
    cache: Cache
    risk: object
    costs: object
    threshold: float

    def __call__(self, confirmation):
        doc_id, text = confirmation.doc_id, confirmation.text
        with TemporaryDirectory(prefix="live-feed-extract-") as directory:
            records = extract_document(doc_id, text, self.client, model_id=self.model_id, cache=self.cache,
                                       out_dir=Path(directory), k_samples=1, prompt_version="v1")
        sample = records[0]
        if not sample["schema_valid"] or sample["finish_reason"] == "length":
            raise FeedSkip("PCSS returned an incomplete or invalid extraction")
        try:
            trade = trade_from_fields(sample["fields"])
        except ValueError:
            raise FeedSkip("PCSS returned incompatible trade fields") from None
        verified = to_record(doc_id, verify(trade, text))
        metadata = {"doc_id": doc_id, "collection": "live_feed", "split": None,
                    "defect": confirmation.defect, "ocr": False}
        features = pd.DataFrame(field_rows(metadata, confirmation.gold, records, verified, len(text)))
        p = self.risk.predict(features)
        c, r = field_costs(features, self.costs)
        mask = escalate(p, c, r, self.threshold)
        fields = []
        for row, probability, cost, review, sent in zip(features.itertuples(index=False), p, c, r, mask):
            term = row.field.rsplit(".", 1)[-1]
            item = sample["fields"].get(row.field, {})
            rules = [key for key, data in verified["rules"].items()
                     if data.get(row.field, {}).get("verdict") in {"HARD", "SOFT"}]
            leg, _, _ = row.field.rpartition(".")
            leg_name = {"fixed_leg": "fixed leg", "floating_leg": "floating leg",
                        "floating_leg_2": "second floating leg"}.get(leg)
            label = term.replace("_", " ").capitalize()
            fields.append({
                "field": row.field, "name": f"{label} ({leg_name})" if leg_name else label,
                "value": item.get("value"), "raw_value": item.get("raw_value"), "span": item.get("span"),
                "p": float(probability), "c_bar": float(cost), "r": float(review), "e": int(row.e),
                "severity": SEVERITY[term], "rules_fired": rules, "action": "Review" if sent else "Accept",
            })
        caught = any(field["field"] in confirmation.affected_fields and field["action"] == "Review"
                     for field in fields)
        return {"id": doc_id, "category": "Live feed", "collection": "live_feed", "text": text,
                "defect": confirmation.defect, "defect_caught": caught, "fields": fields}
