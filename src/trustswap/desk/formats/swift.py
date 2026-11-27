"""Read and write the desk's constant-notional MT360/MT361 confirmation subset."""
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256
import re

import QuantLib as ql

from trustswap.desk import matcher
from trustswap.desk.formats import flatten, read_terms
from trustswap.synth.markets import calendar_for

MESSAGE = re.compile(r"\{2:[IO](360|361)|^\s*MT(360|361)\s*$", re.M)
TAG = re.compile(r"^:(\d{2}[A-Z]?):([^\r\n]*)", re.M)
LEGS = {"B": ("B", "fixed"), "C": ("B", "floating"),
        "E": ("A", "fixed"), "F": ("A", "floating")}
CONVENTIONS = {"MODIFIEDF": "MODFOLLOWING", "MODIFIEDP": "MODPRECEDING"}
DAY_COUNTS = {"AFI/365": "ACT/365.FIXED", "ACT/ACT": "ACT/ACT.ISDA"}
ADJUSTMENTS = {"NONE": ql.Unadjusted, "FOLLOWING": ql.Following, "MODFOLLOWING": ql.ModifiedFollowing,
               "PRECEDING": ql.Preceding, "MODPRECEDING": ql.ModifiedPreceding, "NEAREST": ql.Nearest}
UNITS = {"D": ql.Days, "W": ql.Weeks, "M": ql.Months, "Y": ql.Years}
FREQUENCIES = ("1D", "1W", "2W", "1M", "2M", "3M", "4M", "6M", "1Y", "2Y")


def _one(section, tag):
    values = section.get(tag, [])
    if len(values) > 1:
        raise ValueError(f"The confirmation repeats field {tag}")
    return values[0] if values else None


def _date(text):
    if text is None or not re.fullmatch(r"\d{8}", text):
        raise ValueError("A FIN date must have eight digits")
    return datetime.strptime(text, "%Y%m%d").date()


def _schedule(effective, termination, frequency, centers, convention):
    if not effective < termination or not 1901 <= effective.year <= termination.year <= 2199:
        raise ValueError("FIN schedule dates are invalid")
    calendar = calendar_for(centers) if centers else ql.NullCalendar()
    adjustment = ADJUSTMENTS[convention or "NONE"]
    period = ql.Period(int(frequency[:-1]), UNITS[frequency[-1]])
    schedule = ql.Schedule(ql.Date(effective.day, effective.month, effective.year),
                           ql.Date(termination.day, termination.month, termination.year),
                           period, calendar, adjustment, adjustment, ql.DateGeneration.Backward, False)
    return [day.to_date() for day in schedule][1:]


def _decimal(text):
    if text is None:
        return None
    if not re.fullmatch(r"N?\d+,\d*", text):
        raise ValueError("A FIN decimal requires a comma and optional N sign")
    return Decimal(text.replace("N", "-").replace(",", "."))


def _notional(text):
    if text is None or not re.fullmatch(r"[A-Z]{3}\d+,\d*", text):
        raise ValueError("Expected a FIN currency and positive notional amount")
    amount = _decimal(text[3:])
    if amount <= 0:
        raise ValueError("Notional must be positive")
    return text[:3], amount


def _sequence_values(section, general, kind, effective, termination):
    convention = _one(section, "14A") or _one(general, "14A")
    convention = CONVENTIONS.get(convention, convention)
    if convention:
        convention = matcher.normalize("business_day_convention", convention)
    if centers := section.get("22B", []):
        centers = matcher.normalize("business_centers", centers)
    day_count = _one(section, "14D")
    listed = [_date(text) for text in section.get("30F", [])]
    frequency = None
    if listed:
        if len(listed) > 2000:
            raise ValueError("The FIN payment schedule is too long")
        candidates = []
        for name in FREQUENCIES:
            if _schedule(effective, termination, name, centers, convention) == listed:
                candidates.append(name)
        # A single maturity payment may match several periods, so use the shortest.
        if not candidates:
            raise ValueError("The desk supports regular FIN payment schedules")
        frequency = candidates[0]
    fields = {"payment_frequency": frequency, "day_count_fraction": DAY_COUNTS.get(day_count, day_count),
              "business_day_convention": convention, "business_centers": centers or None}
    if kind == "fixed":
        fixed = _decimal(_one(section, "37M") or _one(section, "37U"))
        if fixed is None:
            raise ValueError("The fixed stream needs a FIN fixed rate")
        fields["fixed_rate"] = fixed / 100
    else:
        spread = _decimal(_one(section, "37R"))
        index = _one(section, "14F")
        if index is None:
            raise ValueError("The floating stream needs a FIN floating rate option")
        fields.update(floating_rate_option=index, reset_frequency=_one(section, "38E"),
                      spread=None if spread is None else spread / 100)
    return fields


def parse(source):
    message = MESSAGE.search(source)
    if message is None:
        raise ValueError("Expected an MT360 or MT361 confirmation")
    kind = message.group(1) or message.group(2)
    block = source.split("{4:", 1)[-1].split("-}", 1)[0]
    sections = defaultdict(lambda: defaultdict(list))
    seen = set()
    code = "A"
    for tag, value in TAG.findall(block):
        if len(tag) == 3 and tag.startswith("15"):
            if tag[-1] in seen or value.strip():
                raise ValueError("Invalid or repeated FIN sequence")
            code = tag[-1]
            seen.add(code)
        else:
            sections[code][tag].append(value.strip())
    if {code for code in sections} - {"A", "B", "C", "D", "E", "F", "G"}:
        raise ValueError("FIN contains additional terms outside the desk swap schema")
    general = sections["A"]
    if _one(general, "22A") not in {None, "NEWT"}:
        raise ValueError("The desk import expects a new FIN confirmation")
    if any(general.get(tag) for tag in ("77D", "72")):
        raise ValueError("Free-text FIN conditions require manual review")
    dates = {path: _date(_one(general, tag)) for path, tag in
             zip(("trade_date", "effective_date", "termination_date"), ("30T", "30V", "30P"))}
    party_a = _one(general, "82D") or _one(general, "82A")
    party_b = _one(general, "87D") or _one(general, "87A")
    if not party_a or not party_b or party_a == party_b:
        raise ValueError("FIN must identify two distinct parties")
    booked_b = _notional(_one(general, "32B"))
    booked_a = booked_b if kind == "360" else _notional(_one(general, "33B"))
    if kind == "361" and booked_a[0] == booked_b[0]:
        raise ValueError("MT361 must have two currencies")
    present = [code for code in LEGS if sections.get(code)]
    if len(present) != 2 or len({LEGS[code][0] for code in present}) != 2:
        raise ValueError("FIN must have one interest stream per payer")
    terms = {path: value.isoformat() for path, value in dates.items()}
    raw = {path: value.strftime("%Y%m%d") for path, value in dates.items()}
    floating = 0
    for code in sorted(present, key=lambda code: LEGS[code][0]):
        payer, side = LEGS[code]
        section = sections[code]
        if any(section.get(tag) for tag in ("37V", "37G", "37N", "22D", "38G", "38H", "30X")):
            raise ValueError("FIN rate, compounding, or stub terms are unsupported")
        leg = "fixed_leg" if side == "fixed" else ("floating_leg_2" if floating else "floating_leg")
        if f"{leg}.payer" in terms:
            raise ValueError("Fixed/fixed swaps are outside the desk's supported schema")
        floating += side == "floating"
        currency, notional = booked_a if payer == "A" else booked_b
        left, right = (party_a, party_b) if payer == "A" else (party_b, party_a)
        terms.update({f"{leg}.{path}": value for path, value in _sequence_values(
            section, general, side, dates["effective_date"], dates["termination_date"]).items()})
        terms.update({f"{leg}.payer": left, f"{leg}.receiver": right,
                      f"{leg}.notional_amount": notional, f"{leg}.notional_currency": currency})
    return read_terms(_one(general, "21N") or _one(general, "20"), terms, source, raw)


def _number(value):
    result = format(Decimal(value).copy_abs(), "f").replace(".", ",")
    return ("N" if Decimal(value) < 0 else "") + result + ("" if "," in result else ",")


def export(reference, trade):
    fields = flatten(trade)
    legs = {name: {key.rsplit(".", 1)[1]: value for key, value in fields.items() if key.startswith(f"{name}.")}
            for name in ("fixed_leg", "floating_leg", "floating_leg_2") if getattr(trade, name) is not None}
    by_party = list(legs.items())
    if "fixed_leg" in legs:
        by_party.reverse()  # Party B pays fixed in the desk's fixed/float export.
    (_, a), (_, b) = by_party
    if (not a.get("payer") or not b.get("payer") or a["payer"] == b["payer"]
            or a["payer"] != b.get("receiver") or b["payer"] != a.get("receiver")):
        raise ValueError("FIN export requires reciprocal payer and receiver terms")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,16}", reference):
        reference = "DSK" + sha256(reference.encode("utf-8")).hexdigest()[:13].upper()
    if any("\r" in value or "\n" in value or "\x00" in value for value in (a["payer"], b["payer"])):
        raise ValueError("FIN party names must be on one line")
    if not all(a.get(key) and b.get(key) for key in ("notional_amount", "notional_currency")):
        raise ValueError("A FIN export requires both notionals")
    same_currency = a["notional_currency"] == b["notional_currency"]
    if same_currency and Decimal(a["notional_amount"]) != Decimal(b["notional_amount"]):
        raise ValueError("MT360 requires equal notionals")
    dates = [date.fromisoformat(fields[path]) for path in ("trade_date", "effective_date", "termination_date")
             if path in fields]
    if len(dates) != 3:
        raise ValueError("FIN requires the trade, effective and termination dates")
    lines = ["MT360" if same_currency else "MT361", ":15A:", f":20:{reference}", ":22A:NEWT", f":21N:{reference}"]
    lines.extend(f":{tag}:{day:%Y%m%d}" for tag, day in zip(("30T", "30V", "30P"), dates))
    lines += [f":32B:{b['notional_currency']}{_number(b['notional_amount'])}"]
    if not same_currency:
        lines += [f":33B:{a['notional_currency']}{_number(a['notional_amount'])}"]
    lines += [f":82D:{a['payer']}", f":87D:{b['payer']}"]
    reversed_conventions = {value: code for code, value in CONVENTIONS.items()}
    reversed_day_counts = {value: code for code, value in DAY_COUNTS.items()}
    for (name, values), fixed_sequence, float_sequence in zip(by_party, "EB", "FC"):
        fixed = name == "fixed_leg"
        sequence = fixed_sequence if fixed else float_sequence
        lines.append(f":15{sequence}:")
        if fixed:
            lines.append(f":37M:{_number(Decimal(values['fixed_rate']) * 100)}")
        else:
            lines.append(f":14F:{values['floating_rate_option']}")
            if values.get("reset_frequency"):
                lines += [":14J:FIRST", f":38E:{values['reset_frequency']}"]
        convention = values.get("business_day_convention")
        centers = values.get("business_centers") or []
        if values.get("payment_frequency"):
            schedule = _schedule(dates[1], dates[2], values["payment_frequency"], centers, convention)
            lines.append(f":18A:{len(schedule)}")
            lines.extend(f":30F:{day:%Y%m%d}" for day in schedule)
        if values.get("day_count_fraction"):
            day_count = values["day_count_fraction"]
            lines.append(f":14D:{reversed_day_counts.get(day_count, day_count)}")
        if convention:
            lines.append(f":14A:{reversed_conventions.get(convention, convention)}")
        if centers:
            lines.append(f":18A:{len(centers)}")
            lines.extend(f":22B:{center}" for center in centers)
        if values.get("spread") is not None:
            lines.append(f":37R:{_number(Decimal(values['spread']) * 100)}")
    return "\n".join(lines) + "\n"
