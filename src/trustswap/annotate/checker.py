"""Independent regex and scalar parsing checks for human-submitted answers."""
from decimal import Decimal
import re

from trustswap.common.normalise import normalise_date, normalise_notional, normalise_rate

DATE_NAMES = {
    "trade_date": r"Trade Date|Transaction Date",
    "effective_date": r"Effective Date|Commencement Date",
    "termination_date": r"Termination Date|Maturity Date",
}
PATTERNS = {
    **DATE_NAMES,
    "notional_amount": r"Notional(?: Amount)?|Calculation Amount",
    "notional_currency": r"Notional Currency|Currency|Notional(?: Amount)?",
    "fixed_rate": r"Fixed(?: Interest)? Rate",
    "spread": r"Spread|Margin",
}
RATES = {"fixed_rate", "spread"}
_DATE_TOKEN = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}-[A-Za-z]{3}-\d{2,4}|"
    r"\d{1,2}(?:st|nd|rd|th)? [A-Za-z]+ \d{4}|[A-Za-z]+ \d{1,2}(?:st|nd|rd|th)?,? \d{4})\b",
    re.I,
)
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_AMOUNT_TOKEN = re.compile(rf"(?<![\w.,+-])[+-]?{_NUMBER}(?:[ \t]*(?:million|billion|thousand|mm|bn|[kmb]))?(?!\w|[.,]\d)", re.I)
_RATE_VALUE = r"(?:\d+(?:\.\d*)?|\.\d+)\s*(?:%|per ?cent|bps?|basis points?|decimal)(?!\w)"
_RATE_TOKEN = re.compile(rf"(?<![\w.(+-])(?:\(\s*{_RATE_VALUE}\s*\)|(?:plus\s+|minus\s+|[+-])?{_RATE_VALUE})", re.I)
_CURRENCY = r"(?:[A-Za-z]{3}|U\.S\.\$|US\$|\$|€|£)"
_CURRENCY_TOKEN = re.compile(r"(?<!\w)(?:[A-Z]{3}(?!\w)|U\.S\.\$|US\$|\$|€|£)")



def scalar(path, raw, *, label=False):
    name = path.rsplit(".", 1)[-1]
    text = str(raw).strip()
    if name in DATE_NAMES:
        return normalise_date(text)
    if name in RATES:
        unit = "decimal" if label and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text) else None
        return normalise_rate(text, unit=unit, is_spread=name == "spread")
    if name == "notional_amount":
        return normalise_notional(text)[0]
    if name == "notional_currency":
        if re.fullmatch(_CURRENCY, text, re.I):
            text += " 1"
        _, currency = normalise_notional(text)
        if currency is None:
            raise ValueError("No explicit currency")
        return currency
    return " ".join(text.casefold().split())


def candidates(path, text):
    """Find explicitly labelled candidates and retain offsets in clean text."""
    name = path.rsplit(".", 1)[-1]
    if name not in PATTERNS:
        return []
    pattern = rf"(?im)^[ \t]*(?:{PATTERNS[name]})[ \t]*[:|][ \t]*([^\n|]+)"
    matches = []
    for match in re.finditer(pattern, text):
        raw = match[1].rstrip()
        try:
            value = scalar(path, raw)
        except (ValueError, ArithmeticError):
            continue
        start = match.start(1)
        matches.append((value, [start, start + len(raw)]))
    return matches


def suggestion(path, text):
    found = candidates(path, text)
    if len({value for value, _ in found}) != 1:
        return None
    value, span = found[0]
    return {"status": "explicit", "value": format(value, "f") if isinstance(value, Decimal) else value,
            "span": span, "note": ""}


def _evidences(path, value, span):
    try:
        if scalar(path, span) == value:
            return True
    except (ValueError, ArithmeticError):
        pass
    name = path.rsplit(".", 1)[-1]
    if name in PATTERNS:
        # A person's selection may include a surrounding sentence or heading.
        token = (_DATE_TOKEN if name in DATE_NAMES else _RATE_TOKEN if name in RATES
                 else _AMOUNT_TOKEN if name == "notional_amount" else _CURRENCY_TOKEN)
        for match in token.finditer(span):
            try:
                if scalar(path, match[0]) == value:
                    return True
            except (ValueError, ArithmeticError):
                continue
        return False
    selected = scalar(path, span)
    return re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", selected) is not None


def check(text, fields):
    issues, parsed = {}, {}
    for path, field in fields.items():
        if field["status"] == "explicit":
            try:
                value = scalar(path, field["value"], label=True)
            except (ValueError, ArithmeticError):
                issues.setdefault(path, []).append("The value could not be parsed.")
                continue
            parsed[path] = value
            start, end = field["span"]
            if not _evidences(path, value, text[start:end]):
                issues.setdefault(path, []).append("The selected text does not support this value.")
        else:
            value = None
        found = {candidate for candidate, _ in candidates(path, text)}
        if found and value not in found:
            issues.setdefault(path, []).append("A labelled term in the document has a different value.")
    trade, effective, termination = (parsed.get(path) for path in DATE_NAMES)
    if ((trade and effective and trade > effective) or (effective and termination and effective >= termination)
            or (trade and termination and trade >= termination)):
        for path in DATE_NAMES:
            if path in parsed:
                issues.setdefault(path, []).append("Check that trade date ≤ effective date < termination date.")
    return issues
