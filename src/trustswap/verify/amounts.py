"""Find currency amounts written in document text (shared by V5 and V6).

Only amounts with an explicit currency (ISO code or $, EUR, GBP symbols) are read,
so years, rates and day counts are never mistaken for money.
"""
from __future__ import annotations

from decimal import Decimal
import re

from trustswap.common.normalise import normalise_notional

_TOKEN = r"(?:U\.S\.\$|US\$|\$|€|£|\b[A-Z]{3}\b)"
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_SCALE = r"(?:\s*(?:million|billion|thousand|mm|bn)\b)?"
_END = r"(?![.,]?\w)"  # "EUR 388,OOO" must not be read as EUR 388
_AMOUNT_RE = re.compile(
    rf"(?:{_TOKEN}\s?{_NUMBER}{_SCALE}{_END})|(?:\b{_NUMBER}{_SCALE}\s?{_TOKEN})")


def currency_amounts(line: str) -> list[tuple[Decimal, str]]:
    """(amount, currency) pairs in one line; unparseable or code-less matches are skipped."""
    found = []
    for match in _AMOUNT_RE.finditer(line):
        try:
            amount, code = normalise_notional(match.group(0))
        except ValueError:
            continue
        if code is not None:
            found.append((amount, code))
    return found


def lines_with(text: str, keywords: tuple[str, ...], exclude: tuple[str, ...] = ()) -> list[str]:
    """Lines that mention any keyword (case-insensitive) and none of the excluded words."""
    result = []
    for line in text.splitlines():
        lower = line.lower()
        if any(k in lower for k in keywords) and not any(x in lower for x in exclude):
            result.append(line)
    return result
