"""Field registry for the desk's additional products."""
from dataclasses import dataclass
import re

from trustswap.desk.products import cds, fx_forward, xccy_swap


@dataclass(frozen=True)
class Term:
    label: str
    normalizer: str
    severity: str = "critical"


DATE = Term("Trade date", "trade_date", "major")
EFFECTIVE = Term("Effective date", "trade_date", "major")
MATURITY = Term("Maturity date", "trade_date", "major")

PRODUCTS = {
    "fx_forward": ("FX forward", fx_forward.FXForward, {"trade_date": DATE,
        "value_date": Term("Value date", "trade_date", "critical"),
        "base_currency": Term("Base currency", "notional_currency"),
        "quote_currency": Term("Quote currency", "notional_currency"),
        "base_amount": Term("Base amount", "notional_amount"),
        "quote_amount": Term("Quote amount", "notional_amount"),
        "forward_rate": Term("Forward rate", "decimal"),
        "buyer": Term("Buyer", "payer"), "seller": Term("Seller", "receiver")}),
    "xccy_swap": ("Cross-currency swap", xccy_swap.CrossCurrencySwap, {
        "trade_date": DATE, "effective_date": EFFECTIVE, "termination_date": MATURITY,
        "first_currency": Term("First leg currency", "notional_currency"),
        "second_currency": Term("Second leg currency", "notional_currency"),
        "first_notional": Term("First leg notional", "notional_amount"),
        "second_notional": Term("Second leg notional", "notional_amount"),
        "exchange_initial": Term("Initial notional exchange", "flag"),
        "exchange_final": Term("Final notional exchange", "flag"),
        "payer_1": Term("First leg payer", "payer"), "payer_2": Term("Second leg payer", "payer")}),
    "cds": ("Credit default swap", cds.CreditDefaultSwap, {
        "trade_date": DATE, "effective_date": EFFECTIVE, "termination_date": Term("Maturity date", "trade_date"),
        "notional_amount": Term("Notional amount", "notional_amount"),
        "notional_currency": Term("Notional currency", "notional_currency"),
        "reference_entity": Term("Reference entity", "text"),
        "spread": Term("Credit spread", "spread"), "upfront": Term("Upfront amount", "notional_amount"),
        "protection_buyer": Term("Protection buyer", "payer"),
        "protection_seller": Term("Protection seller", "receiver")}),
}
TERMS = {f"{kind}.{key}": field for kind, (_, _, fields) in PRODUCTS.items() for key, field in fields.items()}
KEYS = {
    "fx_forward": ("trade_date", "value_date", "base_currency", "quote_currency", "base_amount"),
    "xccy_swap": ("trade_date", "termination_date", "first_currency", "second_currency", "first_notional"),
    "cds": ("trade_date", "termination_date", "reference_entity", "notional_amount", "notional_currency"),
}
_PATTERNS = {
    "fx_forward": r"\b(?:fx|foreign exchange)[ -]+forward\b",
    "xccy_swap": r"\b(?:cross[- ]currency|xccy)(?: interest rate)?[- ]+swap\b",
    "cds": r"\b(?:credit[- ]default[- ]swap|cds)\b",
    "irs": r"\binterest rate swap\b|\birs\b",
}


def cost_term(path):
    field = TERMS.get(path)
    if field is None:
        return path.rsplit(".", 1)[-1]
    return "notional_amount" if field.severity == "critical" else "trade_date"


def detect(text):
    """Recognise an explicit product label before searching confirmation phrases."""
    header = re.search(r"(?im)^\s*product\s*:\s*(.+)$", text)
    sample = header.group(1).strip() if header else text
    if sample.lower() in PRODUCTS:
        return sample.lower()
    hits = [name for name, pattern in _PATTERNS.items() if re.search(pattern, sample, re.I)]
    if "xccy_swap" in hits and "irs" in hits:
        hits.remove("irs")
    return hits[0] if len(hits) == 1 else None


def local_terms(kind, text):
    spec = PRODUCTS[kind][2]
    names = {label.casefold().replace("_", " "): key for key, field in spec.items() for label in (key, field.label)}
    output = {}
    for line in text.splitlines():
        label, colon, raw = line.partition(":")
        key = names.get(label.strip().casefold().replace("_", " "))
        if colon and key:
            if key in output:
                raise ValueError("The confirmation repeats a product field")
            output[key] = raw.strip() or None
    return output
