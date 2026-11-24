"""Scan-like text noise for synthetic documents (Step 12 hard synthetic set).

Clean rendered documents are trivially easy for strong LLMs, which leaves the
risk model nothing to learn from. ``add_noise`` imitates OCR output of a scanned
confirmation: character confusions (0/O, 1/l, 5/S, 8/B, rn/m), dropped and
doubled spaces, and broken lines. A careful human can still read every value,
so the gold values are unchanged; gold spans are mapped onto the noisy text.

This is a text-level proxy for scanning + OCR (no image pipeline), documented as
such. The ``ocr`` flag on the document record is set to True.
"""
from __future__ import annotations

from random import Random

RATES = {1: 0.02, 2: 0.05}  # per-character chance of an edit, by severity
_DIGIT = {"0": "O", "1": "l", "5": "S", "8": "B", "2": "Z", "6": "b"}
_LETTER = {"O": "0", "l": "1", "I": "1", "S": "5", "B": "8", "m": "rn", "e": "c", "a": "o"}


def add_noise(text: str, seed: int, severity: int = 1) -> tuple[str, list[int]]:
    """Noisy text and a map from every original offset (0..len) to its noisy offset."""
    if severity not in RATES:
        raise ValueError("severity must be 1 or 2")
    rng, rate = Random(seed), RATES[severity]
    out: list[str] = []
    mapping: list[int] = []
    length = 0
    for ch in text:
        mapping.append(length)
        piece = ch
        if rng.random() < rate:
            if ch in _DIGIT:
                piece = _DIGIT[ch]
            elif ch in _LETTER:
                piece = _LETTER[ch]
            elif ch == " ":
                piece = rng.choice(("", "  ", "\n"))
            # "," and "." are never swapped: "10,000,000" -> "10,000.000" would change the value
        out.append(piece)
        length += len(piece)
    mapping.append(length)
    return "".join(out), mapping


def map_span(span: tuple[int, int] | None, mapping: list[int]) -> tuple[int, int] | None:
    if span is None:
        return None
    start, end = mapping[span[0]], mapping[span[1]]
    return (start, end) if start < end else None
