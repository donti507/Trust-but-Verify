"""Conservative field decisions for additional products in the desk collection."""
from functools import lru_cache
import math

import yaml

from trustswap.decide.policy import load_costs
from trustswap.desk import matcher
from trustswap.desk.products import PRODUCTS, cds, cost_term, fx_forward, xccy_swap
from trustswap.desk.exceptions import CONFIG
from trustswap.extract.parse import locate


@lru_cache
def settings():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["products"]


def prepare(kind, text, raw):
    _, schema, definitions = PRODUCTS[kind]
    values, findings = {}, {}
    for key, definition in definitions.items():
        path = f"{kind}.{key}"
        try:
            normalized = matcher.normalize(path, raw.get(key))
            if definition.normalizer == "payer" or definition.normalizer == "receiver":
                from trustswap.desk.imports import _fictional

                _fictional(path, normalized)
            if definition.normalizer == "notional_amount" and key != "upfront" and normalized is not None:
                if normalized <= 0:
                    raise ValueError("Notionals must be positive")
        except (ValueError, TypeError, ArithmeticError):
            normalized = None
            findings[key] = "The stated term could not be validated."
        if normalized is None and definition.severity == "critical":
            findings.setdefault(key, "This critical term is missing.")
        values[key] = normalized

    terms = schema.model_validate(values)
    if kind == "fx_forward":
        checks = fx_forward.checks(terms, settings()["fx_reference_rates"])
    elif kind == "xccy_swap":
        checks = xccy_swap.checks(terms)
    else:
        checks = cds.checks(terms, settings()["cds_reference_entities"])
    findings.update(checks)
    costs = load_costs()
    rows = {}
    for key, value in terms.model_dump(mode="json").items():
        path = f"{kind}.{key}"
        evidence = raw.get(key)
        if not isinstance(evidence, (str, bool, int, float)) or isinstance(evidence, float) and not math.isfinite(evidence):
            evidence = None
        span = locate(evidence, None, text) if isinstance(evidence, str) and evidence.strip() else None
        rows[path] = {"confirmation": value, "booking": None, "p": 0.0, "c": costs.c_bar(cost_term(path)),
                      "r": costs.r(cost_term(path)), "raw": evidence, "span": span}
    return rows, {f"{kind}.{key}": reason for key, reason in findings.items()}
