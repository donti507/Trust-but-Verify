"""Step 13: one feature row per (doc_id, field) for the risk model.

Groups (column prefixes):
  u_   uncertainty   mean token log-prob of the k = 0 value; share of K samples equal to
                     the modal value; cross-model agreement; verbalised confidence
  g_   grounding     V2 residual; evidence span found in the text
  v_   verdicts      per rule V1..V7: one-hot PASS / HARD / SOFT / UNDET / NONE (rule silent on field)
  rho_ residuals     per rule: log(1 + |residual|), 0 when null
  tau_ field type    one-hot field name and leg; one-hot severity class
  q_   document      OCR flag, log text length, template family seen in training,
                     schema-valid k = 0 output, output cut off by the length limit
Label e = 1 when the normalised k = 0 value differs from gold or gold is ambiguous.
Missing inputs are filled with a neutral value and flagged in a *_missing column, so
the table has no missing values.
"""
from __future__ import annotations

import math
from pathlib import Path

from trustswap.extract.evaluate import canonical, consistency, outcome
from trustswap.extract.fields import PATHS
from trustswap.synth.render import trade_fields
from trustswap.synth.templates import FAMILY_SPLITS

RULES = ("V1", "V2", "V3", "V4", "V5", "V6", "V7")
VERDICTS = ("PASS", "HARD", "SOFT", "UNDET", "NONE")
FIELD_NAMES = tuple(dict.fromkeys(p.rsplit(".", 1)[-1] for p in PATHS))
LEGS = ("top", "fixed_leg", "floating_leg", "floating_leg_2")
# Default severities from the runbook field table (to be confirmed in the cost session).
SEVERITY = {name: "critical" for name in ("payer", "receiver", "notional_amount", "notional_currency",
                                          "fixed_rate", "floating_rate_option", "spread")}
SEVERITY.update({name: "major" for name in ("trade_date", "effective_date", "termination_date",
                                            "payment_frequency", "reset_frequency", "day_count_fraction",
                                            "business_day_convention", "business_centers")})
SEVERITY["notice_address"] = "minor"
SEVERITIES = ("critical", "major", "minor")
META = ("doc_id", "field", "model", "collection", "split", "defect", "is_control", "k_samples")


def _value(record: dict | None, path: str):
    if not record or not record.get("schema_valid"):
        return None
    return (record.get("fields", {}).get(path) or {}).get("value")


def field_rows(doc: dict, gold, records: list[dict], verifier: dict | None, text_length: int,
               other_k0: dict | None = None) -> list[dict]:
    """Feature rows for one document (one row per gold field path)."""
    k0 = records[0]
    valid = bool(k0.get("schema_valid"))
    family = doc.get("family_id")
    seen = 1 if doc.get("collection") != "real" and FAMILY_SPLITS.get(family) == "train" else 0
    rules = (verifier or {}).get("rules", {})
    rows = []
    for path, term in trade_fields(gold.trade).items():
        item = (k0.get("fields", {}).get(path) or {}) if valid else {}
        name = path.rsplit(".", 1)[-1]
        leg = path.split(".")[0] if "." in path else "top"
        predicted = _value(k0, path)
        row = {
            "doc_id": doc["doc_id"], "field": path, "model": k0.get("model"),
            "collection": doc.get("collection"), "split": doc.get("split"),
            "defect": doc.get("defect") or "none", "is_control": int(bool(doc.get("is_control"))),
            "k_samples": len(records),
        }
        # u: uncertainty
        logprob = item.get("mean_logprob")
        row["u_logprob"] = float(logprob) if logprob is not None else 0.0
        row["u_logprob_missing"] = int(logprob is None)
        agree = consistency(records, path)
        row["u_agreement"] = float(agree) if agree is not None else 0.0
        row["u_agreement_missing"] = int(agree is None)
        if other_k0 is not None and other_k0.get("schema_valid"):
            row["u_cross_agree"] = int(canonical(path, predicted) == canonical(path, _value(other_k0, path)))
            row["u_cross_missing"] = 0
        else:
            row["u_cross_agree"], row["u_cross_missing"] = 0, 1
        conf = item.get("verbal_conf")
        row["u_verbal_conf"] = float(conf) if conf is not None else 0.0
        row["u_verbal_missing"] = int(conf is None)
        row["u_predicted_null"] = int(predicted is None)
        # g: grounding
        v2 = (rules.get("V2") or {}).get(path) or {}
        row["g_v2_residual"] = math.log1p(abs(v2["residual"])) if v2.get("residual") is not None else 0.0
        row["g_span_found"] = int(bool(item.get("span_found")))
        # v / rho: verdicts and residuals per rule
        for rule in RULES:
            found = (rules.get(rule) or {}).get(path)
            verdict = found["verdict"] if found else "NONE"
            for option in VERDICTS:
                row[f"v_{rule}_{option}"] = int(verdict == option)
            residual = found.get("residual") if found else None
            row[f"rho_{rule}"] = math.log1p(abs(residual)) if residual is not None else 0.0
        # tau: field type, leg, severity
        for option in FIELD_NAMES:
            row[f"tau_{option}"] = int(name == option)
        for option in LEGS:
            row[f"tau_leg_{option}"] = int(leg == option)
        for option in SEVERITIES:
            row[f"tau_sev_{option}"] = int(SEVERITY[name] == option)
        # q: document
        row["q_ocr"] = int(bool(doc.get("ocr")))
        row["q_log_length"] = math.log1p(text_length)
        row["q_family_seen"] = seen
        row["q_schema_valid"] = int(valid)
        row["q_retried_length"] = int(bool(k0.get("retried_after_length")))
        # label
        row["e"] = int(outcome(path, predicted, term) != "match")
        rows.append(row)
    return rows


def build_table(docs: dict[str, dict], gold: dict, records: dict[str, list[dict]], verify_dir: Path,
                other_records: dict[str, list[dict]] | None = None):
    """pandas DataFrame of every field row for documents with extraction, gold and text."""
    import json

    import pandas as pd

    rows = []
    for doc_id in sorted(records):
        if doc_id not in docs or doc_id not in gold:
            continue
        doc = docs[doc_id]
        verifier_path = Path(verify_dir) / f"{doc_id}.json"
        verifier = json.loads(verifier_path.read_text(encoding="utf-8")) if verifier_path.exists() else None
        text_length = len(doc["_text_path"].read_text(encoding="utf-8"))
        other = (other_records or {}).get(doc_id)
        rows += field_rows(doc, gold[doc_id], records[doc_id], verifier, text_length,
                           other[0] if other else None)
    table = pd.DataFrame(rows)
    feature_cols = [c for c in table.columns if c not in META and c != "e"]
    table[feature_cols] = table[feature_cols].astype(float)
    return table


def feature_columns(table) -> list[str]:
    return [c for c in table.columns if c not in META and c != "e"]


def label_report(table, model: str) -> str:
    """Markdown: label rate per collection/split and per field, plus the missing-value check."""
    lines = [f"# Features ({model})", "",
             f"Rows: {len(table):,} from {table['doc_id'].nunique():,} documents; "
             f"{len(feature_columns(table))} feature columns. "
             f"Missing values: {int(table.isna().sum().sum())}.", "",
             "Label e = 1: the k = 0 value differs from gold after normalisation, or gold is ambiguous.", "",
             "| collection | split | rows | error rate |", "| --- | --- | --- | --- |"]
    for (collection, split), part in table.groupby(["collection", "split"]):
        lines.append(f"| {collection} | {split} | {len(part):,} | {100 * part['e'].mean():.2f}% |")
    lines += ["", "| field | rows | error rate |", "| --- | --- | --- |"]
    for field, part in table.groupby("field", sort=False):
        lines.append(f"| {field} | {len(part):,} | {100 * part['e'].mean():.2f}% |")
    return "\n".join(lines) + "\n"
