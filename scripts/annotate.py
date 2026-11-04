#!/usr/bin/env python3
"""Human pilot annotations only: exact saved-text spans, no verifier or network."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from trustswap.common.annotations import AnnotationRecord, FIELDS, FLAGS, normalise_field, validated_field

if __package__:
    from .pilot_utils import read_csv_snapshot, read_annotations, save_annotations, write_csv
    from .review_helper import local_text_path, _safe
else:
    from pilot_utils import read_csv_snapshot, read_annotations, save_annotations, write_csv
    from review_helper import local_text_path, _safe

HINTS = {
    "trade_date": ("trade date", "transaction date"), "effective_date": ("effective", "commencement"),
    "termination_date": ("termination", "maturity"), "payer": ("payer", "paying", "party"),
    "receiver": ("receiver", "receiving", "party"), "notional_amount": ("notional", "calculation amount"),
    "notional_currency": ("currency", "notional", "calculation amount"),
    "payment_frequency": ("payment", "period", "frequency"), "reset_frequency": ("reset", "fixing", "frequency"),
    "day_count_fraction": ("day count", "basis", "actual/", "act/", "30/360"),
    "business_day_convention": ("business day", "following", "preceding", "adjustment"),
    "business_centers": ("center", "centre", "london", "new york", "target", "tokyo"),
    "fixed_rate": ("fixed rate", "fixed interest"), "floating_rate_option": ("floating", "reference", "index", "libor", "sofr"),
    "spread": ("spread", "margin", "basis point"), "notice_address": ("notice", "address", "attention"),
}


class Back(Exception):
    pass


class Reject(Exception):
    pass


def show_hints(text, path, output):
    lines = text.splitlines()
    words = HINTS[path.split(".")[-1]] if path != "flags" else ("amortis", "amortiz", "cap", "floor")
    matches = [i for i, line in enumerate(lines) if any(word in line.lower() for word in words)]
    selected = sorted({j for i in matches[:30] for j in range(max(0, i - 1), min(len(lines), i + 2))})
    if not selected:
        print("No keyword hint found. Use show to inspect all saved text; do not infer absence from this search.", file=output)
    for i in selected:
        print(f"{i + 1:5d} | {_safe(lines[i])}", file=output)
    if len(matches) > 30:
        print("Additional keyword matches omitted; use show for the whole document.", file=output)


def occurrences(text, quote):
    if not quote:
        return []
    positions, start = [], 0
    while True:
        start = text.find(quote, start)
        if start < 0:
            return positions
        positions.append((start, start + len(quote)))
        start += 1


def ask_field(path, text, ask, output):
    while True:
        entered = ask("Exact value / absent / ref / amb / show / back / reject <reason>: ").strip()
        command = entered.strip().lower()
        if command == "show":
            for number, line in enumerate(text.splitlines(), 1):
                print(f"{number:5d} | {_safe(line)}", file=output)
            continue
        try:
            if command in {"absent", "amb"}:
                answer = validated_field(path, None, None, "absent" if command == "absent" else "ambiguous")
            elif command == "ref":
                raw = ask("Value supported by the incorporated reference (blank if unresolved): ").strip()
                value = _normalise_human(path, raw, ask) if raw.strip() else None
                answer = validated_field(path, value, None, "by_reference")
            else:
                raw = entered[5:] if entered.startswith("text:") else entered
                raw = raw.replace("\\n", "\n").strip()
                matches = occurrences(text, raw)
                if not matches:
                    raise ValueError("No exact occurrence in the saved text. Copy the value exactly (use \\n for line breaks).")
                if len(matches) == 1:
                    span = matches[0]
                else:
                    for index, (start, end) in enumerate(matches, 1):
                        print(f"{index}: [{start}, {end}) ...{_safe(text[max(0, start - 45):end + 45])}...", file=output)
                    while True:
                        choice = ask(f"Choose occurrence 1–{len(matches)}: ").strip()
                        if choice.isdigit() and 1 <= int(choice) <= len(matches):
                            span = matches[int(choice) - 1]
                            break
                        print("Choose an occurrence number, or back.", file=output)
                answer = validated_field(path, _normalise_human(path, raw, ask), span, "explicit")
                assert text[slice(*span)] == raw
            print("Proposed: " + _safe(json.dumps(answer, ensure_ascii=False)), file=output)
            if ask("Confirm this document-supported annotation? [y/n/back]: ").strip().lower() == "y":
                return answer
        except ValueError as error:
            print(_safe(error), file=output)


def _normalise_human(path, raw, ask):
    try:
        return normalise_field(path, raw)
    except ValueError as error:
        if str(error) != "ambiguous rate unit":
            raise
        unit = ask("Rate unit supported by document/header [percent/decimal/bps]: ").strip().lower()
        if unit not in {"percent", "decimal", "bps"}:
            raise ValueError("No supported unit chosen; use amb if the unit cannot be resolved")
        return normalise_field(path, raw, rate_unit=unit)


def ask_flags(ask, output):
    while True:
        value = ask("Flags: amortising, cap, floor (comma-separated), or none; back: ").strip().lower()
        values = [] if value == "none" else [part.strip().replace("amortizing", "amortising") for part in value.split(",")]
        if any(flag not in FLAGS for flag in values):
            print("Choose only document-supported flags or type none explicitly.", file=output)
            continue
        values = sorted(set(values))
        print("Proposed flags: " + json.dumps(values), file=output)
        if ask("Confirm flags? [y/n/back]: ").strip().lower() == "y":
            return values


def annotate(pilot_path, *, annotator, output_dir="data/real/annotations", input_fn=None, output=None):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", annotator):
        raise ValueError("invalid annotator ID")
    input_fn = input if input_fn is None else input_fn
    output = sys.stdout if output is None else output
    pilot_columns, pilot, pilot_digest = read_csv_snapshot(pilot_path)
    if len({row["doc_id"] for row in pilot}) != len(pilot):
        raise ValueError("duplicate pilot doc_id")
    if any(row.get("split") == "real_test" or row.get("collection", "real") != "real" for row in pilot):
        raise ValueError("pilot must contain real non-test documents only")
    path = Path(output_dir) / f"{annotator}.jsonl"
    records, digest = read_annotations(path)
    if any(record.annotator != annotator for record in records.values()):
        raise ValueError("annotation file belongs to another annotator")
    jobs = [(row, field) for row in pilot for field in (*FIELDS, "flags")]
    texts, index, force, exit_code = {}, 0, False, 0
    def ask(prompt):
        while True:
            print(prompt, end="", file=output, flush=True)
            value = input_fn("")
            command = value.strip()
            if command.lower() == "back":
                raise Back
            if command.lower() == "reject":
                print("Provide a reason: reject <reason>.", file=output)
                continue
            if command.lower().startswith("reject "):
                reason = command[7:].strip()
                if not reason:
                    print("A rejection reason is required.", file=output)
                    continue
                raise Reject(reason)
            return value
    try:
        while index < len(jobs):
            row, field = jobs[index]
            doc_id = row["doc_id"]
            if row.get("rejected_reason", "").strip():
                index += 1
                force = False
                continue
            previous = records.get(doc_id)
            if not force and previous and (previous.flags_reviewed if field == "flags" else field in previous.fields):
                index += 1
                continue
            if doc_id not in texts:
                source = local_text_path(row["text_path"], pilot_path)
                raw = source.read_bytes()
                texts[doc_id] = source, raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()
            source, text, text_hash = texts[doc_id]
            record = previous or AnnotationRecord(doc_id=doc_id, annotator=annotator, text_path=str(source),
                                                  text_sha256=text_hash, split=row.get("split") or None)
            if record.text_sha256 != text_hash:
                raise ValueError("saved text changed since annotation; do not reuse offsets across versions")
            print(f"\n{annotator} | {_safe(doc_id)} | {field}", file=output)
            print("Leg identity comes from document headings, never a verifier or market default.", file=output)
            if row.get("manual_check_required", "").lower() == "true":
                print("Pilot top-up: suitability still needs human confirmation against this document.", file=output)
            if force and field in record.fields:
                print("Current saved answer: " + _safe(json.dumps(record.fields[field])), file=output)
            show_hints(text, field, output)
            try:
                data = record.model_dump()
                if field == "flags":
                    data["flags"], data["flags_reviewed"] = ask_flags(ask, output), True
                else:
                    data["fields"][field] = ask_field(field, text, ask, output)
                updated = AnnotationRecord.model_validate(data)
                if hashlib.sha256(source.read_bytes()).hexdigest() != text_hash:
                    raise ValueError("saved text changed during annotation; not saving stale spans")
                proposed = {**records, doc_id: updated}
                digest = save_annotations(path, proposed, digest)
                records = proposed
                print("Saved field.", file=output)
                index += 1
                force = False
            except Reject as rejection:
                reason = str(rejection)
                proposed = [{**item, "rejected_reason": reason} if item["doc_id"] == doc_id else dict(item)
                            for item in pilot]
                pilot_columns = list(dict.fromkeys([*pilot_columns, "rejected_reason"]))
                serialised = [{key: item.get(key, "") for key in pilot_columns} for item in proposed]
                pilot_digest = write_csv(pilot_path, pilot_columns, serialised, expected=pilot_digest, check=True)
                row["rejected_reason"] = reason
                print(f"Rejected {_safe(doc_id)}: {_safe(reason)}. Run select_pilot.py --top-up for a replacement.", file=output)
                while index < len(jobs) and jobs[index][0]["doc_id"] == doc_id:
                    index += 1
                force = False
            except Back:
                if index == 0:
                    print("Already at the first field.", file=output)
                else:
                    index -= 1
                force = True
    except EOFError:
        print("\nStopped; rerun to resume unanswered fields.", file=output)
    except KeyboardInterrupt:
        exit_code = 130
        print("\nInterrupted; prior fields are saved.", file=output)
    print(f"Saved: {sum(len(record.fields) for record in records.values())} fields; "
          f"{sum(record.complete for record in records.values())} complete documents.", file=output)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--pilot", type=Path, default=Path("data/real/pilot20.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/real/annotations"))
    args = parser.parse_args(argv)
    try:
        return annotate(args.pilot, annotator=args.annotator, output_dir=args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(_safe(error))


if __name__ == "__main__":
    raise SystemExit(main())
