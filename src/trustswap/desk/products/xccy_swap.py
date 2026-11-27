"""Desk terms for a constant-notional cross-currency swap."""
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class CrossCurrencySwap(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    trade_date: date | None = None
    effective_date: date | None = None
    termination_date: date | None = None
    first_currency: str | None = None
    second_currency: str | None = None
    first_notional: Decimal | None = None
    second_notional: Decimal | None = None
    exchange_initial: bool | None = None
    exchange_final: bool | None = None
    payer_1: str | None = None
    payer_2: str | None = None


def checks(terms):
    issues = {}
    if terms.first_currency and terms.first_currency == terms.second_currency:
        issues["second_currency"] = "The two swap legs must use different currencies."
    if terms.effective_date and terms.termination_date and terms.termination_date < terms.effective_date:
        issues["termination_date"] = "Maturity is earlier than the effective date."
    return issues
