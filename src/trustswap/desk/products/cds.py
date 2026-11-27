"""Desk terms for a single-name credit default swap."""
from datetime import date
from decimal import Decimal

import QuantLib as ql
from pydantic import BaseModel, ConfigDict


class CreditDefaultSwap(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    trade_date: date | None = None
    effective_date: date | None = None
    termination_date: date | None = None
    notional_amount: Decimal | None = None
    notional_currency: str | None = None
    reference_entity: str | None = None
    spread: Decimal | None = None
    upfront: Decimal | None = None
    protection_buyer: str | None = None
    protection_seller: str | None = None


def imm_maturity(maturity):
    """CDS roll dates use QuantLib's twentieth-IMM rule."""
    year = maturity.year
    if not 1901 <= year <= 2199:
        return False
    schedule = ql.Schedule(ql.Date(20, ql.March, year), ql.Date(20, ql.December, year), ql.Period(3, ql.Months),
                           ql.NullCalendar(), ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.TwentiethIMM, False)
    return maturity in (day.to_date() for day in schedule)


def checks(terms, entities):
    issues = {}
    if terms.reference_entity and terms.reference_entity not in entities:
        issues["reference_entity"] = "Use one of the configured fictional reference entities."
    if terms.termination_date:
        if not imm_maturity(terms.termination_date):
            issues["termination_date"] = "Maturity falls outside the standard twentieth-IMM roll dates."
        elif terms.effective_date and terms.termination_date < terms.effective_date:
            issues["termination_date"] = "Maturity is earlier than the effective date."
    if terms.spread is not None and terms.upfront is not None and terms.spread <= 0 and terms.upfront <= 0:
        issues["spread"] = "Specify a positive spread or upfront amount."
    return issues
