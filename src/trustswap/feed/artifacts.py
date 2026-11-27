"""Portable numeric feed scoring and generated replay validation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.special import expit

from trustswap.score.risk import EPS
from trustswap.score.features import RULES
from trustswap.extract.fields import PATHS

MODEL_FORMAT = "trustswap.full-logistic.v1"
REPLAY_FORMAT = "trustswap.generated-replay.v1"
CATEGORIES = {
    "Clean synthetic": ("synthetic", {"train", "dev", "syn_test"}),
    "Noisy synthetic": ("synthetic", {"train", "dev", "syn_test"}),
    "Adversarial": ("adversarial", {"adv_test"}),
}
DOC_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
REAL_ID = re.compile(r"^real(?:[_.-]|$)", re.IGNORECASE)
REQUIRED_FIELD_KEYS = {"name", "value", "span", "p", "c_bar", "r", "e", "severity", "rules_fired"}


def serialized(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def model_hash(parameters):
    return hashlib.sha256(serialized(parameters).encode("utf-8")).hexdigest()


def export_model(model):
    """Extract the Step 14 scaler, logistic coefficients and Platt parameters."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if (model.feature_set, model.kind) != ("full", "logistic") or not isinstance(model.estimator, Pipeline):
        raise ValueError("Expected a saved full-logistic Step 14 risk model")
    steps = model.estimator.steps
    if len(steps) != 2:
        raise ValueError("Expected a scaler followed by a logistic model")
    scaler, estimator = (step[1] for step in steps)
    calibrator = model.calibrator
    if not isinstance(scaler, StandardScaler) or not all(
        isinstance(item, LogisticRegression) and list(item.classes_) == [0, 1]
        for item in (estimator, calibrator)
    ):
        raise ValueError("Expected a StandardScaler and two binary logistic fits with classes [0, 1]")

    columns = list(model.columns)
    if scaler.n_features_in_ != len(columns) or calibrator.n_features_in_ != 1:
        raise ValueError("Saved model feature dimensions differ")
    parameters = {
        "format": MODEL_FORMAT,
        "features": columns,
        "scaler": {
            "mean": scaler.mean_.tolist() if scaler.with_mean else [0.0] * len(columns),
            "scale": scaler.scale_.tolist() if scaler.with_std else [1.0] * len(columns),
        },
        "logistic": {"weights": estimator.coef_[0].tolist(), "intercept": float(estimator.intercept_[0])},
        "calibration": {"weight": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0]),
                        "clip": EPS},
    }
    payload = {**parameters, "sha256": model_hash(parameters)}
    PortableRisk.from_payload(payload)
    return payload


@dataclass(frozen=True)
class PortableRisk:
    columns: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    platt_weight: float
    platt_intercept: float
    clip: float

    @classmethod
    def from_payload(cls, payload):
        parameters = {key: value for key, value in payload.items() if key != "sha256"}
        digest = payload.get("sha256", "")
        if not isinstance(digest, str) or not hmac.compare_digest(digest.encode(), model_hash(parameters).encode()):
            raise ValueError("Feed model hash does not match its parameters")
        if parameters["format"] != MODEL_FORMAT:
            raise ValueError("Unsupported portable risk model format")
        columns = tuple(parameters["features"])
        if not columns or not all(isinstance(name, str) and name for name in columns) or len(set(columns)) != len(columns):
            raise ValueError("Invalid serialized feature list")
        scaler, logistic, calibration = (parameters[key] for key in ("scaler", "logistic", "calibration"))
        mean, scale, weights = (np.asarray(values, dtype=float) for values in
                                (scaler["mean"], scaler["scale"], logistic["weights"]))
        clip = float(calibration["clip"])
        scalars = [float(value) for value in (
            logistic["intercept"], calibration["weight"], calibration["intercept"], clip
        )]
        if not all(array.shape == (len(columns),) and np.isfinite(array).all()
                   for array in (mean, scale, weights)):
            raise ValueError("Serialized model arrays have invalid dimensions or values")
        if not np.all(scale > 0) or not all(map(math.isfinite, scalars)) or clip != EPS:
            raise ValueError("Serialized model constants are invalid")
        return cls(columns, mean, scale, weights, *scalars)

    def predict(self, table):
        x = table[list(self.columns)].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError("Feed features must be finite")
        raw = expit(((x - self.mean) / self.scale) @ self.weights + self.intercept)
        p = np.clip(raw, self.clip, 1 - self.clip)
        return expit(self.platt_weight * np.log(p / (1 - p)) + self.platt_intercept)


def replay_from_page(html):
    match = re.search(r'<script type="application/json" id="demo-data">(.*?)</script>', html, re.S)
    if not match or "Synthetic document, fictional parties" not in html:
        raise ValueError("Expected embedded data from a generated Trust but Verify replay page")
    return validate_replay({"format": REPLAY_FORMAT, **json.loads(match[1])})


def validate_replay(payload):
    if payload.get("format") != REPLAY_FORMAT or payload.get("illustrative"):
        raise ValueError("Expected the full generated replay, without illustrative predictions")
    documents = payload["documents"]
    if len(documents) != 30:
        raise ValueError("The replay must contain exactly 30 generated documents")
    bound = float(payload["bound"])
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("Invalid replay loss bound")
    cleaned, seen = [], set()
    for doc in documents:
        doc_id = doc["id"]
        category = doc.get("category")
        if category not in CATEGORIES or doc.get("collection") == "real":
            raise ValueError("Replay contains a document outside the permitted generated collections")
        expected, splits = CATEGORIES[category]
        if (doc.get("collection", expected) != expected or doc.get("split") not in splits | {None}
                or not isinstance(doc_id, str) or not DOC_ID.fullmatch(doc_id) or REAL_ID.match(doc_id)):
            raise ValueError("Replay contains real or unrecognized document provenance")
        if doc_id in seen or not isinstance(doc["text"], str) or not doc["fields"]:
            raise ValueError("Duplicate document or missing generated evidence")
        seen.add(doc_id)
        fields = []
        for field in doc["fields"]:
            clean = {key: field[key] for key in REQUIRED_FIELD_KEYS}
            p, c, r = (float(clean[key]) for key in ("p", "c_bar", "r"))
            span, value = clean["span"], clean["value"]
            if not (0 <= p <= 1 and all(math.isfinite(x) and x >= 0 for x in (c, r))
                    and clean["e"] in (0, 1) and clean["severity"] in {"critical", "major", "minor"}
                    and set(clean["rules_fired"]) <= set(RULES)):
                raise ValueError("Invalid replay field risk, cost, label or rules")
            if not isinstance(clean["name"], str) or (
                value is not None and not isinstance(value, (str, int, float))
                and not (isinstance(value, list) and all(isinstance(item, str) for item in value))
            ):
                raise ValueError("Replay field contains an unsupported value")
            if span is not None and not (
                isinstance(span, list) and len(span) == 2 and all(type(i) is int for i in span)
                and 0 <= span[0] < span[1] <= len(doc["text"])
            ):
                raise ValueError("Replay source span is invalid")
            if "field" in field:
                if field["field"] not in PATHS:
                    raise ValueError("Replay term path is unknown")
                clean["field"] = field["field"]
            fields.append(clean)
        cleaned.append({"id": doc_id, "category": category, "text": doc["text"], "fields": fields})
    return {"format": REPLAY_FORMAT, "bound": bound, "documents": cleaned}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
