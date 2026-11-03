"""Seeded synthetic vanilla trades; no global RNG or QuantLib evaluation date."""
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
import math
from random import Random

from trustswap.common.schema import InterestRateSwap
from trustswap.synth.markets import (
    FIXED_DAY_COUNTS, FREQUENCIES, MARKETS, TENORS, calendar_for, market_for, roll_trade_date,
)


def _field(value):
    return {"value": value, "span": None, "status": "explicit"}


def rate_bounds(trade_date: date) -> tuple[int, int]:
    """Illustrative era ranges in basis points, bounded by 0.5% and 6%."""
    if trade_date.year <= 2008:
        return 250, 600
    if trade_date.year <= 2021:
        return 50, 350
    return 200, 600


def sample_trade(rng: Random) -> InterestRateSwap:
    """Sample 85% fixed/float and 15% IBOR basis swaps.

    Basis swaps use IBOR-eligible currency/era pairs. Overnight-only markets
    are excluded from that conditional draw rather than inventing tenor basis.
    """
    import QuantLib as ql

    first, last = date(2005, 1, 1), date(2026, 12, 31)
    fixed_float = rng.random() < 0.85
    while True:
        candidate = first + timedelta(days=rng.randint(0, (last - first).days))
        currency = rng.choice(tuple(MARKETS))
        trade_date = roll_trade_date(currency, candidate)
        market = market_for(currency, trade_date)
        if fixed_float or "LIBOR" in market.floating_option or "EURIBOR" in market.floating_option:
            break
    calendar = calendar_for(market.centers)
    convention = rng.choices(("MODFOLLOWING", "FOLLOWING"), weights=(9, 1), k=1)[0]
    ql_convention = {"MODFOLLOWING": ql.ModifiedFollowing, "FOLLOWING": ql.Following}[convention]
    ql_trade = ql.Date(trade_date.day, trade_date.month, trade_date.year)
    effective = calendar.advance(ql_trade, market.spot_days, ql.Days)
    tenor = rng.choice(TENORS)
    termination = calendar.advance(effective, ql.Period(tenor, ql.Years), ql_convention)
    amount = (Decimal(str(math.exp(rng.uniform(math.log(5_000_000), math.log(1_000_000_000)))))
              / Decimal(1_000_000)).quantize(Decimal(1), rounding=ROUND_HALF_UP) * Decimal(1_000_000)
    party_a, party_b = "Synthetic Party A", "Synthetic Party B"

    def leg(payer, receiver, floating):
        frequency = rng.choice(FREQUENCIES)
        day_count = market.floating_day_count if floating else rng.choices(
            FIXED_DAY_COUNTS, weights=market.fixed_day_count_weights, k=1)[0]
        if not floating:
            if rng.random() < 0.8:
                frequency, day_count = rng.choice(market.typical_fixed)
            else:
                alternatives = [(freq, dc) for freq in FREQUENCIES for dc in FIXED_DAY_COUNTS
                                if (freq, dc) not in market.typical_fixed]
                weights = [market.fixed_day_count_weights[FIXED_DAY_COUNTS.index(dc)]
                           for _, dc in alternatives]
                frequency, day_count = rng.choices(alternatives, weights=weights, k=1)[0]
        return {name: _field(value) for name, value in {
            "payer": payer, "receiver": receiver, "notional_amount": amount,
            "notional_currency": currency, "payment_frequency": frequency,
            "reset_frequency": frequency, "day_count_fraction": day_count,
            "business_day_convention": convention, "business_centers": list(market.centers),
        }.items()}

    floating = leg(party_b, party_a, True)
    floating.update(floating_rate_option=_field(market.floating_option), spread=_field(Decimal(0)))
    record = {"trade_date": _field(trade_date), "effective_date": _field(effective.to_date()),
              "termination_date": _field(termination.to_date()), "floating_leg": floating,
              "notice_address": _field("Synthetic Party A, 1 Example Street, Example City")}
    if fixed_float:
        fixed = leg(party_a, party_b, False)
        fixed["reset_frequency"] = {"value": None, "span": None, "status": "absent"}
        low, high = rate_bounds(trade_date)
        fixed["fixed_rate"] = _field((Decimal(rng.randint(low, high)) / Decimal(10000)).quantize(Decimal("0.0001")))
        record["fixed_leg"] = fixed
    else:
        second = leg(party_a, party_b, True)
        spread = Decimal(0) if rng.choice((True, False)) else Decimal(rng.randint(-50, 200)) / Decimal(10000)
        second_option = market.floating_option
        if currency == "USD" and date(2018, 1, 1) <= trade_date < date(2022, 1, 1) and rng.random() < 0.5:
            second_option = "USD-SOFR"
        else:
            second["reset_frequency"] = _field(rng.choice([
                frequency for frequency in FREQUENCIES
                if frequency != floating["reset_frequency"]["value"]]))
        second.update(floating_rate_option=_field(second_option), spread=_field(spread))
        record["floating_leg_2"] = second
    return InterestRateSwap.model_validate(record)
