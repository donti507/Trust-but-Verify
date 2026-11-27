"""Deterministic fictional fixtures for the additional desk products."""
from datetime import date, timedelta
from decimal import Decimal
import json

from trustswap.desk import exceptions, store
from trustswap.desk.products import PRODUCTS
from trustswap.desk.products.review import prepare


def _terms(product, index):
    day = date(2026, 1, 5) + timedelta(days=index)
    common = {"trade_date": day.isoformat()}
    amount = 1_000_000 + 100_000 * index
    if product == "fx_forward":
        return {**common, "value_date": (day + timedelta(days=30)).isoformat(), "base_currency": "USD",
                "quote_currency": "EUR", "base_amount": str(amount), "quote_amount": str(amount * Decimal("0.92")),
                "forward_rate": "0.92", "buyer": "Synthetic Party A", "seller": "Synthetic Party B"}
    common["effective_date"] = (day + timedelta(days=2)).isoformat()
    if product == "xccy_swap":
        return {**common, "termination_date": "2031-07-15", "first_currency": "USD", "second_currency": "EUR",
                "first_notional": str(amount), "second_notional": str(amount * Decimal("0.92")),
                "exchange_initial": "Yes", "exchange_final": "Yes", "payer_1": "Synthetic Party A",
                "payer_2": "Synthetic Party B"}
    return {**common, "termination_date": "2031-06-20", "notional_currency": "USD",
            "notional_amount": str(amount), "reference_entity": "Synthetic Reference A", "spread": "125 bps",
            "upfront": "0", "protection_buyer": "Synthetic Party A", "protection_seller": "Synthetic Party B"}


DEFECTS = {
    "fx_forward": ("forward_rate", "1.50"),
    "xccy_swap": ("second_currency", "USD"),
    "cds": ("termination_date", "2031-06-19"),
}


def seed(db):
    count = 0
    with store.transaction(db):
        for product, (name, _, definitions) in PRODUCTS.items():
            for index in range(10):
                trade_id = f"example-{product.replace('_', '-')}-{index + 1:02d}"
                if db.execute("SELECT 1 FROM trades WHERE id=?", (trade_id,)).fetchone():
                    continue
                booked = _terms(product, index)
                stated = dict(booked)
                if index == 9:
                    changed, value = DEFECTS[product]
                    stated[changed] = value
                text = "Synthetic demo. Fictional document and parties.\n" f"Product: {name}\n" + "\n".join(
                    f"{definitions[key].label}: {value}" for key, value in stated.items()
                )
                fields, findings = prepare(product, text, stated)
                booking, _ = prepare(product, "", booked)
                for key, field in fields.items():
                    field["booking"] = booking[key]["confirmation"]
                store.new_trade(db, trade_id, "synthetic", text, fields)
                db.execute("INSERT INTO desk_products(trade_id,product,checks) VALUES (?,?,?)",
                           (trade_id, product, json.dumps(findings)))
                store.event(db, None, "product_demo_seeded", trade_id, product=product)
                exceptions.reconcile(db, trade_id)
                count += 1
    return count
