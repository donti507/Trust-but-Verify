#!/usr/bin/env python3
"""Offline, human-only review of locally ingested EDGAR candidate text."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import os
from pathlib import Path
import re
import sys
import tempfile

PRODUCTS = {"1": "irs_fixed_float", "2": "basis", "3": "cap_floor", "4": "other"}
KEYWORDS = re.compile(r"Notional|Fixed\s+Rate|Floating\s+Rate\s+Option|Termination\s+Date|Trade\s+Date|Effective\s+Date", re.I)
ANSWER_COLUMNS = ("is_confirmation", "product_type", "review_status")


def _safe(text):
    """Display document content literally; do not pass terminal escape sequences."""
    return "".join(char if char.isprintable() or char == "\t" else f"\\x{ord(char):02x}" for char in str(text))


def _decision(value):
    value = value.strip().lower()
    if value in {"y", "yes", "true", "1"}:
        return "y"
    if value in {"n", "no", "false", "0"}:
        return "n"
    if not value:
        return ""
    raise ValueError(f"unrecognised is_confirmation value: {_safe(value)!r}")


def load_review(path):
    raw = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    columns = list(reader.fieldnames or [])
    if len(columns) != len(set(columns)):
        raise ValueError("duplicate CSV columns")
    required = {"document_url", "era", "label", "text_path"}
    if not required.issubset(columns):
        raise ValueError("manual-check CSV is missing URL, era, label or text_path columns")
    for name in ANSWER_COLUMNS:
        if name not in columns:
            columns.append(name)
    rows, urls = [], set()
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("malformed CSV row")
        if not row["document_url"] or row["document_url"] in urls:
            raise ValueError("manual-check rows must have unique, nonempty document URLs")
        urls.add(row["document_url"])
        for name in ANSWER_COLUMNS:
            row.setdefault(name, "")
        if row.get("split") == "real_test":
            raise ValueError("real_test review is excluded before Step 19")
        if row.get("collection", "real") != "real":
            raise ValueError("manual-check review must contain only real collection rows")
        decision = _decision(row["is_confirmation"])
        status = row["review_status"].strip().lower()
        if status not in {"", "answered", "skipped"} or (status == "answered" and not decision) or (status == "skipped" and decision):
            raise ValueError("inconsistent review status and confirmation answer")
        rows.append(row)
    return columns, rows, hashlib.sha256(raw).hexdigest()


def save_review(path, columns, rows, expected_digest):
    """Atomic per-answer save, refusing to overwrite concurrent human edits."""
    path = Path(path)
    def unchanged():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
            raise RuntimeError("CSV changed on disk during review; refusing to overwrite other edits")
    unchanged()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=".review-", delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        unchanged()
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_text_path(value, csv_path):
    if not value.strip() or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
        raise ValueError("text_path must name an already-downloaded local text file")
    value, csv_path = Path(value), Path(csv_path).resolve()
    if value.is_absolute():
        choices = [value]
    else:
        choices = [csv_path.parent / value, Path.cwd() / value]
        if csv_path.parent.name == "real" and csv_path.parent.parent.name == "data":
            choices.append(csv_path.parent.parent.parent / value)
    existing = {choice.resolve() for choice in choices if choice.is_file()}
    if len(existing) > 1:
        raise ValueError("ambiguous relative text_path; use an absolute path")
    if not existing:
        raise FileNotFoundError(f"local text file is missing: {_safe(value)}")
    return existing.pop()


def show_document(row, csv_path, *, output, color=False, head_lines=60):
    print(f"URL: {_safe(row['document_url'])}", file=output)
    print(f"Era: {_safe(row['era'])} | Screen label: {_safe(row['label'])}", file=output)
    print(f"Pages: {_safe(row.get('page_count') or 'unknown')} ({_safe(row.get('page_count_basis') or 'unknown basis')})", file=output)
    path = local_text_path(row["text_path"], csv_path)
    print(f"First {head_lines} lines, plus all matching lines (original line numbers):", file=output)
    previous = 0
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = _safe(line.rstrip("\r\n"))
            if number > head_lines and not KEYWORDS.search(line):
                continue
            if number > previous + 1:
                print(f"  ... lines {previous + 1}–{number - 1} omitted ...", file=output)
            highlighted = KEYWORDS.sub(lambda match: f"\033[1;33m{match.group()}\033[0m" if color else f"[[{match.group()}]]", line)
            print(f"{number:5d} | {highlighted}", file=output)
            previous = number


def parse_answer(value):
    match = re.fullmatch(r"\s*([yns])(?:[\s,]*([1-4]))?\s*", value, re.I)
    if not match:
        raise ValueError("Enter y, n or s, optionally followed by product 1–4 (for example: y 1).")
    key, product = match.groups()
    return key.lower(), PRODUCTS.get(product)


def print_summary(rows, output):
    yes = sum(_decision(row["is_confirmation"]) == "y" for row in rows)
    no = sum(_decision(row["is_confirmation"]) == "n" for row in rows)
    skipped = sum(row["review_status"].strip().lower() == "skipped" for row in rows)
    print(f"\nReview totals: yes={yes} no={no} skipped={skipped} unanswered={len(rows) - yes - no - skipped}", file=output)
    for field in ("label", "era"):
        print(f"Yes/no per {field}:", file=output)
        counts = Counter((row[field], _decision(row["is_confirmation"])) for row in rows)
        for group in sorted({row[field] for row in rows}):
            print(f"  {_safe(group)}: yes={counts[group, 'y']} no={counts[group, 'n']}", file=output)


def review(path="data/real/manual_check.csv", *, input_fn=None, output=None, color=None, revisit_skipped=False):
    path = Path(path)
    output = sys.stdout if output is None else output
    input_fn = input if input_fn is None else input_fn
    color = output.isatty() and "NO_COLOR" not in os.environ if color is None else color
    columns, rows, digest = load_review(path)
    exit_code = 0
    print("Products: 1 irs_fixed_float | 2 basis | 3 cap_floor | 4 other", file=output)
    try:
        for index, row in enumerate(rows, 1):
            if _decision(row["is_confirmation"]):
                continue
            if row["review_status"].strip().lower() == "skipped" and not revisit_skipped:
                continue
            print(f"\n--- Document {index}/{len(rows)} ---", file=output)
            try:
                show_document(row, path, output=output, color=color)
            except (OSError, ValueError) as error:
                print(f"Text unavailable; left unanswered: {_safe(error)}", file=output)
                continue
            while True:
                print("Answer y/n/s, optional product 1–4 (e.g. y 1); EOF/Ctrl+C stops: ", end="", file=output, flush=True)
                try:
                    answer, product = parse_answer(input_fn(""))
                except ValueError as error:
                    print(str(error), file=output)
                    continue
                updated = dict(row)
                updated["is_confirmation"] = "" if answer == "s" else answer
                updated["review_status"] = "skipped" if answer == "s" else "answered"
                if product is not None:
                    updated["product_type"] = product
                proposed = list(rows)
                proposed[index - 1] = updated
                digest = save_review(path, columns, proposed, digest)
                rows = proposed
                print("Saved.", file=output)
                break
    except EOFError:
        print("\nStopped; saved answers will be skipped on resume.", file=output)
    except KeyboardInterrupt:
        exit_code = 130
        print("\nInterrupted; saved answers will be skipped on resume.", file=output)
    finally:
        print_summary(rows, output)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/real/manual_check.csv"))
    parser.add_argument("--no-color", action="store_true", help="use [[phrase]] highlighting instead of ANSI")
    parser.add_argument("--revisit-skipped", action="store_true")
    args = parser.parse_args(argv)
    try:
        return review(args.csv, color=False if args.no_color else None, revisit_skipped=args.revisit_skipped)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(_safe(error))


if __name__ == "__main__":
    raise SystemExit(main())
