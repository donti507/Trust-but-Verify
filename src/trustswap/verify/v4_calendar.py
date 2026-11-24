"""V4 calendar: dates must be business days and explicit schedule dates must lie
on the payment schedule regenerated with QuantLib.

Residual: days between a stated date and the nearest regenerated date (or the
nearest business day for the business-day check).

* Effective and termination dates should be good business days on the joint
  calendar of all legs' business centres. A miss is only SOFT, because real
  confirmations often state unadjusted dates (e.g. an effective date of
  1 April 2017, a Saturday, "subject to adjustment").
* Dates written on schedule lines ("payment date", "first period", "stub", ...)
  must match a regenerated schedule date of at least one leg (HARD otherwise).
  Dates on lines that mention a "stub" are treated as agreed stub boundaries and
  are passed to QuantLib as firstDate / nextToLastDate.
"""
from __future__ import annotations

from datetime import date
import re

import QuantLib as ql

from trustswap.common.normalise import normalise_date
from trustswap.common.schema import InterestRateSwap
from trustswap.synth.markets import calendar_for
from trustswap.verify.base import RuleResult, present_legs, worst
from trustswap.verify.v1_schema import CENTERS

SCHEDULE_KEYWORDS = ("payment date", "period end", "calculation period", "reset date", "stub",
                     "first period", "last period", "commencing", "period starts", "period ends")
_CONVENTIONS = {"MODFOLLOWING": ql.ModifiedFollowing, "FOLLOWING": ql.Following, "PRECEDING": ql.Preceding,
                "MODPRECEDING": ql.ModifiedPreceding, "NONE": ql.Unadjusted, "NEAREST": ql.Nearest}
_UNITS = {"D": ql.Days, "W": ql.Weeks, "M": ql.Months, "Y": ql.Years}
_MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
_DATE_PATTERNS = (
    r"\d{4}-\d{2}-\d{2}",
    r"\d{1,2}-" + _MONTHS + r"-\d{2,4}",
    r"\d{1,2}(?:st|nd|rd|th)?\s+" + _MONTHS + r",?\s+\d{4}",
    _MONTHS + r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}",
)
_DATE_RE = re.compile("|".join(f"(?:{p})" for p in _DATE_PATTERNS), re.IGNORECASE)


def _q(day: date) -> ql.Date:
    return ql.Date(day.day, day.month, day.year)


def explicit_dates(text: str) -> tuple[list[date], list[date]]:
    """Dates stated on schedule lines, split into (schedule dates, stub boundaries)."""
    schedule, stubs = [], []
    for line in text.splitlines():
        lower = line.lower()
        if not any(keyword in lower for keyword in SCHEDULE_KEYWORDS):
            continue
        for match in _DATE_RE.finditer(line):
            try:
                found = date.fromisoformat(normalise_date(match.group(0)))
            except ValueError:
                continue
            (stubs if "stub" in lower else schedule).append(found)
    return schedule, stubs


def _schedules(eff: date, term: date, leg, stubs: list[date]) -> list[set[date]] | None:
    """Adjusted and unadjusted schedules for one leg, with and without stub hints."""
    frequency = leg.payment_frequency.value
    centers = leg.business_centers.value
    if not frequency or not centers or any(c not in CENTERS for c in centers):
        return None
    match = re.fullmatch(r"([1-9]\d*)([DWMY])", frequency)
    if not match:
        return None
    period = ql.Period(int(match.group(1)), _UNITS[match.group(2)])
    calendar = calendar_for(centers)
    convention = _CONVENTIONS.get(leg.business_day_convention.value or "MODFOLLOWING", ql.ModifiedFollowing)
    variants = [(ql.Date(), ql.Date())]
    for boundary in stubs:
        if eff < boundary < term:
            variants += [(_q(boundary), ql.Date()), (ql.Date(), _q(boundary))]
    result = []
    for first, next_to_last in variants:
        for conv in (convention, ql.Unadjusted):
            try:
                schedule = ql.Schedule(_q(eff), _q(term), period, calendar, conv, conv,
                                       ql.DateGeneration.Backward, False, first, next_to_last)
            except RuntimeError:
                continue
            result.append({d.to_date() for d in schedule})
    return result


def v4_calendar(trade: InterestRateSwap, text: str) -> dict[str, RuleResult]:
    legs = present_legs(trade)
    eff, term = trade.effective_date.value, trade.termination_date.value
    results: dict[str, list[RuleResult]] = {"effective_date": [], "termination_date": []}
    for leg in legs:
        results[f"{leg}.business_centers"] = []
        results[f"{leg}.payment_frequency"] = []

    centers = sorted({c for leg in legs for c in (getattr(trade, leg).business_centers.value or [])})
    if not centers or any(c not in CENTERS for c in centers):
        undet = RuleResult(verdict="UNDET", detail="no supported business centers to build a calendar")
        return {path: undet for path in results}
    calendar = calendar_for(centers)
    for leg in legs:
        results[f"{leg}.business_centers"].append(RuleResult(verdict="PASS", detail="calendar built"))

    def business_day_check(day: date, label: str) -> RuleResult:
        if calendar.isBusinessDay(_q(day)):
            return RuleResult(verdict="PASS", residual=0.0, detail=f"{label} is a business day")
        nxt = calendar.adjust(_q(day), ql.Following).to_date()
        return RuleResult(verdict="SOFT", residual=float((nxt - day).days),
                          detail=f"{label} {day} is not a business day in {'+'.join(centers)}")

    if eff is not None:
        results["effective_date"].append(business_day_check(eff, "effective date"))
    if term is not None:
        results["termination_date"].append(business_day_check(term, "termination date"))

    if eff is not None and term is not None and eff < term:
        stated, stubs = explicit_dates(text)
        leg_schedules = {}
        for leg in legs:
            schedules = _schedules(eff, term, getattr(trade, leg), stubs)
            leg_schedules[leg] = schedules
            results[f"{leg}.payment_frequency"].append(
                RuleResult(verdict="PASS", detail="schedule regenerated") if schedules else
                RuleResult(verdict="UNDET", detail="schedule could not be regenerated"))
        all_dates = [d for schedules in leg_schedules.values() if schedules for s in schedules for d in s]
        if all_dates:
            worst_gap, worst_day = 0, None
            for day in stated + stubs:
                if not (eff <= day <= term):
                    continue
                gap = min(abs((day - d).days) for d in all_dates)
                if gap > worst_gap:
                    worst_gap, worst_day = gap, day
            schedule_result = (
                RuleResult(verdict="HARD", residual=float(worst_gap),
                           detail=f"stated schedule date {worst_day} is {worst_gap} days off the regenerated schedule")
                if worst_gap else RuleResult(verdict="PASS", residual=0.0, detail="stated schedule dates match"))
            if stated or stubs:
                results["effective_date"].append(schedule_result)
                results["termination_date"].append(schedule_result)

    return {path: worst(*found) if found else RuleResult(verdict="UNDET", detail="dates missing")
            for path, found in results.items()}
