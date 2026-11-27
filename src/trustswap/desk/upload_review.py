"""Desk-only text extraction and scoring through the pinned research components."""
from dataclasses import replace
from functools import partial
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pandas as pd

from trustswap.common.cache import Cache
from trustswap.decide.policy import load_costs
from trustswap.extract.clients import load_models
from trustswap.extract.parse import trade_from_fields
from trustswap.extract.prompt import prompt_sha
from trustswap.extract.run import extract_document
from trustswap.feed.artifacts import PortableRisk, read_json
from trustswap.feed.pipeline import SingleAttemptClient
from trustswap.score.features import field_rows
from trustswap.synth.render import trade_fields
from trustswap.verify.evaluate import verifier_hash
from trustswap.verify.run import to_record, verify

ROOT = Path(__file__).resolve().parents[3]


def allow_external():
    return os.environ.get("DESK_ALLOW_EXTERNAL_LLM", "").strip().lower() in {"true", "1"}


def rows_for(trade, source_fields, probabilities=None, costs=None):
    costs = costs or load_costs()
    fields = {}
    for path, term in trade_fields(trade).items():
        name = path.rsplit(".", 1)[-1]
        item = source_fields.get(path, {})
        fields[path] = {
            "confirmation": term.model_dump(mode="json")["value"], "booking": None,
            "raw": item.get("raw_value"), "span": item.get("span"),
            "p": float(probabilities[path]) if probabilities is not None else 0.0,
            "c": costs.c_bar(name), "r": costs.r(name),
        }
    return fields


def scorer(upload_dir):
    if not allow_external():
        raise ValueError("External text extraction is disabled")
    model_path = os.environ.get("DESK_RISK_MODEL")
    if not model_path:
        raise ValueError("Set DESK_RISK_MODEL to the full-logistic JSON artifact")
    try:
        risk = PortableRisk.from_payload(read_json(Path(model_path)))
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError):
        raise ValueError("The configured desk risk model could not be loaded or verified") from None
    prompt = json.loads((ROOT / "prereg/prompt-freeze.json").read_text(encoding="utf-8"))
    verify_freeze = json.loads((ROOT / "prereg/verifier-v1.0.json").read_text(encoding="utf-8"))
    if (prompt["status"] != "frozen" or prompt_sha("v1") != prompt["prompt_sha256"]
            or verifier_hash() != verify_freeze["verifier_hash"]):
        raise ValueError("The frozen prompt or verifier fingerprint differs")
    endpoint = os.environ.get("PCSS_BASE_URL", "")
    if not endpoint.startswith("https://") or not os.environ.get("PCSS_API_KEY"):
        raise ValueError("Set PCSS_BASE_URL to its HTTPS endpoint and provide PCSS_API_KEY")
    spec = replace(load_models()["frontier"], base_url_env="PCSS_BASE_URL", api_key_env="PCSS_API_KEY")
    spec.check_pinned()
    return partial(extract_and_score, client=SingleAttemptClient(spec, timeout=90), model_id=spec.model_id,
                   cache=Cache(upload_dir / "calls"), risk=risk, costs=load_costs())


def extract_and_score(doc_id, text, ocr, *, client, model_id, cache, risk, costs):
    # Enforce the opt-in here as well as at the upload boundary.
    if not allow_external():
        raise ValueError("External text extraction is disabled")
    try:
        with TemporaryDirectory(prefix="desk-extract-") as directory:
            records = extract_document(doc_id, text, client, model_id=model_id, cache=cache,
                                       out_dir=Path(directory), k_samples=1, prompt_version="v1")
    except Exception:
        raise ValueError("The configured extraction provider could not complete the request") from None
    first = records[0]
    if not first["schema_valid"] or first["finish_reason"] == "length":
        raise ValueError("The extraction was invalid or incomplete")
    trade = trade_from_fields(first["fields"])
    verified = to_record(doc_id, verify(trade, text))
    # Only the inventory of extracted paths is available at the desk. Drop the
    # research helper's comparison column before scoring and persist no labels.
    rows = field_rows({"doc_id": doc_id, "collection": "desk", "ocr": ocr},
                      SimpleNamespace(trade=trade), records, verified, len(text))
    features = pd.DataFrame(rows).drop(columns="e")
    probabilities = dict(zip(features["field"], risk.predict(features), strict=True))
    return rows_for(trade, first["fields"], probabilities, costs)
