"""Desk terms for a deliverable foreign exchange forward."""
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class FXForward(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    trade_date: date | None = None
    value_date: date | None = None
    base_currency: str | None = None
    quote_currency: str | None = None
    base_amount: Decimal | None = None
    quote_amount: Decimal | None = None
    forward_rate: Decimal | None = None
    buyer: str | None = None
    seller: str | None = None


def checks(terms, references):
    issues = {}
    if terms.trade_date and terms.value_date and terms.value_date < terms.trade_date:
        issues["value_date"] = "The value date is earlier than the trade date."
    pair = f"{terms.base_currency}/{terms.quote_currency}"
    if terms.base_currency and terms.base_currency == terms.quote_currency:
        issues["quote_currency"] = "The forward must exchange two different currencies."
    if terms.forward_rate is not None:
        if pair not in references:
            issues["forward_rate"] = "There is no configured rate for this currency pair."
        else:
            reference = Decimal(str(references[pair]))
            if not Decimal("0.80") * reference <= terms.forward_rate <= Decimal("1.20") * reference:
                issues["forward_rate"] = "The quoted rate is more than 20% from the configured reference."
    return issues
