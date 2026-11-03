"""Small sampling assumptions, not historical pricing or curve data."""
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Market:
    floating_option: str
    floating_day_count: str
    centers: tuple[str, ...]
    spot_days: int
    typical_fixed: tuple[tuple[str, str], ...]
    # Weights correspond to 30/360, ACT/360, ACT/365.FIXED.
    fixed_day_count_weights: tuple[int, int, int]


FIXED_DAY_COUNTS = ("30/360", "ACT/360", "ACT/365.FIXED")
FREQUENCIES = ("3M", "6M", "1Y")
TENORS = (2, 3, 5, 7, 10)
MARKETS = {
    "USD": Market("USD-SOFR", "ACT/360", ("USNY",), 2, (("1Y", "ACT/360"),), (7, 2, 1)),
    "EUR": Market("EUR-EURIBOR-Reuters", "ACT/360", ("EUTA",), 2, (("1Y", "30/360"),), (7, 2, 1)),
    "GBP": Market("GBP-SONIA", "ACT/365.FIXED", ("GBLO",), 0, (("6M", "ACT/365.FIXED"), ("1Y", "ACT/365.FIXED")), (1, 1, 8)),
}


def market_for(currency: str, trade_date: date) -> Market:
    if currency == "USD" and trade_date < date(2022, 1, 1):
        return Market("USD-LIBOR-BBA", "ACT/360", ("USNY", "GBLO"), 2, (("6M", "30/360"),), (7, 2, 1))
    if currency == "GBP" and trade_date < date(2022, 1, 1):
        return Market("GBP-LIBOR-BBA", "ACT/365.FIXED", ("GBLO",), 0,
                      (("6M", "ACT/365.FIXED"), ("1Y", "ACT/365.FIXED")), (1, 1, 8))
    return MARKETS[currency]


def calendar_for(centers):
    """Joint holiday calendar: every specified center must be open."""
    import QuantLib as ql

    calendars = {"USNY": lambda: ql.UnitedStates(ql.UnitedStates.Settlement),
                 "GBLO": ql.UnitedKingdom, "EUTA": ql.TARGET,
                 "JPTO": ql.Japan}
    selected = [calendars[center]() for center in centers]
    if not selected:
        raise ValueError("at least one business center is required")
    if len(selected) == 1:
        return selected[0]
    return ql.JointCalendar(*selected, ql.JoinHolidays)


def roll_trade_date(currency: str, candidate: date) -> date:
    """Roll forward using the applicable market, including across era cutovers."""
    import QuantLib as ql

    day = ql.Date(candidate.day, candidate.month, candidate.year)
    while True:
        market = market_for(currency, day.to_date())
        if calendar_for(market.centers).isBusinessDay(day):
            return day.to_date()
        day = day + 1
