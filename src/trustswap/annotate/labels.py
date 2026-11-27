"""Validation and display of human-authored annotation forms."""
from decimal import Decimal

from trustswap.annotate import checker
from trustswap.common.annotations import FLAGS, normalise_field
from trustswap.common.schema import InterestRateSwap
from trustswap.extract.fields import LEG_CLASSES, PATHS, STRUCTURE_LEGS

CHECKLIST = (
    ("specific_trade", "One specific trade, rather than a form or template"),
    ("key_terms", "Trade date, notional and rate terms are present"),
    ("single_currency", "Both legs use the same currency"),
    ("swap", "Interest rate swap, with no cap, floor, swaption or CDS as the product"),
    ("latest", "Unique confirmation or the latest amendment"),
)
STATUSES = {"explicit", "absent", "ambiguous"}
LEG_LABELS = {"fixed_leg": "fixed leg", "floating_leg": "floating leg", "floating_leg_2": "second floating leg"}
STRUCTURES = {"fixed_float": "Fixed / floating", "float_float": "Floating / floating"}
GROUPS = ("Trade dates", "Fixed leg", "Floating leg", "Parties")


def paths(structure):
    return tuple(path for path in PATHS if "." not in path or path.split(".", 1)[0] in STRUCTURE_LEGS[structure])


def label(path):
    leg, _, name = path.rpartition(".")
    text = name.replace("_", " ").capitalize()
    return f"{text} ({LEG_LABELS[leg]})" if leg else text


def grouped():
    groups = {name: [] for name in GROUPS}
    for path in PATHS:
        leg, _, name = path.rpartition(".")
        group = ("Parties" if name in {"payer", "receiver", "notice_address"}
                 else "Trade dates" if not leg else "Fixed leg" if leg == "fixed_leg" else "Floating leg")
        groups[group].append((path, label(path)))
    return groups


def string(value, *, limit=4000):
    if not isinstance(value, str) or len(value) > limit or "\x00" in value:
        raise ValueError("Enter a valid text value")
    return value


def span(value, text):
    if not (isinstance(value, list) and len(value) == 2 and all(type(point) is int for point in value)
            and 0 <= value[0] < value[1] <= len(text)):
        raise ValueError("Select a valid source span")
    return value


def screen(payload, doc, known):
    if not isinstance(payload, dict) or set(payload) != {"answers", "include", "reason", "duplicate_of"}:
        raise ValueError("Complete the inclusion checklist and decision")
    answers = payload["answers"]
    if (not isinstance(answers, dict) or set(answers) != {key for key, _ in CHECKLIST}
            or any(type(value) is not bool for value in answers.values()) or type(payload["include"]) is not bool):
        raise ValueError("Answer all five checklist questions")
    reason = string(payload["reason"]).strip()
    duplicate = string(payload["duplicate_of"], limit=128).strip()
    if not reason:
        raise ValueError("Enter a reason for the decision")
    if payload["include"] and (not all(answers.values()) or duplicate):
        raise ValueError("Inclusion requires five yes answers and a unique document")
    if duplicate and (duplicate == doc or duplicate not in known or answers["latest"]):
        raise ValueError("Choose another document ID and mark the duplicate check no")
    return {**payload, "reason": reason, "duplicate_of": duplicate}


def answer(item, text, *, complete=False):
    if not isinstance(item, dict) or set(item) != {"status", "value", "note", "span", "confirmed"}:
        raise ValueError("Invalid field answer")
    status, value, note, offset, checked = (item[k] for k in ("status", "value", "note", "span", "confirmed"))
    if not isinstance(status, str) or status not in STATUSES | {""} or type(checked) is not bool:
        raise ValueError("Choose a valid field status")
    value, note = string(value), string(note)
    if offset is not None:
        span(offset, text)
    if complete:
        if status == "explicit":
            if not value.strip() or offset is None:
                raise ValueError("Enter the value and select its source text")
        elif status in {"absent", "ambiguous"}:
            if offset is not None or value or not note.strip():
                raise ValueError("Give a note for an absent or ambiguous field")
        else:
            raise ValueError("Choose a field status")
    return {"status": status, "value": value, "note": note, "span": offset, "confirmed": checked}


def normalize_selection(path, raw):
    name = path.rsplit(".", 1)[-1]
    if name not in checker.PATTERNS:
        return raw.strip()
    try:
        value = checker.scalar(path, raw)
        return format(value, "f") if isinstance(value, Decimal) else value
    except (ValueError, ArithmeticError):
        return raw.strip()


def _term_type(path):
    leg, _, field = path.rpartition(".")
    model = dict(LEG_CLASSES)[leg] if leg else InterestRateSwap
    return model.model_fields[field].annotation


def typed(path, item):
    status = item["status"]
    if status != "explicit":
        return {"status": status, "value": None, "span": None}
    raw = item["value"]
    try:
        if path.rsplit(".", 1)[-1] in checker.PATTERNS:
            value = checker.scalar(path, raw, label=True)
        else:
            try:
                value = normalise_field(path, raw)
            except ValueError:
                if path.endswith(".business_centers"):
                    raise
                value = " ".join(raw.split())
        if isinstance(value, Decimal) and not value.is_finite():
            raise ValueError("Non-finite value")
        term = _term_type(path).model_validate({"status": status, "value": value, "span": item["span"]})
    except (ValueError, ArithmeticError):
        raise ValueError(f"Enter a valid value for {label(path)}") from None
    return term.model_dump(mode="json")


def typed_fields(fields):
    return {path: typed(path, item) for path, item in fields.items()}


def identity(path, term):
    value = term["value"]
    if value is not None:
        if path.rsplit(".", 1)[-1] in {"notional_amount", "fixed_rate", "spread"}:
            value = Decimal(value)
        elif isinstance(value, list):
            value = tuple(value)
        elif isinstance(value, str):
            value = " ".join(value.casefold().split())
    return term["status"], value


def alternative(left, right, path):
    return identity(path, left) != identity(path, right) or left["span"] != right["span"]


def raw_identity(item):
    return {key: value for key, value in item.items() if key != "confirmed"}


def empty_form():
    return {"structure": "", "fields": {}, "flags": [], "flags_reviewed": False,
            "cursor": "", "reviewed": [], "reasons": {}}


def form(data, text, *, complete=False, prefill=False):
    if not isinstance(data, dict) or data.keys() != empty_form().keys():
        raise ValueError("Invalid annotation form")
    structure = data["structure"]
    if not isinstance(structure, str) or structure not in {*STRUCTURES, ""} or (complete and not structure):
        raise ValueError("Select the swap structure")
    active = set(paths(structure)) if structure else set(PATHS)
    if not isinstance(data["fields"], dict) or data["fields"].keys() - active:
        raise ValueError("Unexpected paths for this swap structure")
    if complete and data["fields"].keys() != active:
        raise ValueError("Answer every term before submitting")
    fields = {}
    for path, item in data["fields"].items():
        try:
            fields[path] = answer(item, text, complete=complete)
        except ValueError as error:
            raise ValueError(f"{label(path)}. {error}") from None
        if complete and prefill and not fields[path]["confirmed"]:
            raise ValueError(f"Confirm {label(path)} before submitting")
    flags, reviewed, reasons = data["flags"], data["reviewed"], data["reasons"]
    if (not isinstance(flags, list) or not all(isinstance(item, str) and item in FLAGS for item in flags)
            or type(data["flags_reviewed"]) is not bool or (complete and not data["flags_reviewed"])):
        raise ValueError("Review the document characteristics")
    if (not isinstance(reviewed, list) or not all(isinstance(path, str) and path in active for path in reviewed)
            or not isinstance(reasons, dict) or reasons.keys() - active):
        raise ValueError("Invalid re-check details")
    cursor = string(data["cursor"], limit=128)
    if cursor and cursor not in active:
        raise ValueError("Invalid resume position")
    return {**data, "fields": fields, "flags": sorted(set(flags)), "reviewed": sorted(set(reviewed)),
            "reasons": {path: string(reason).strip() for path, reason in reasons.items()}}


def trade(structure, fields):
    record = {leg: {} for leg in STRUCTURE_LEGS[structure]}
    for path in paths(structure):
        leg, _, key = path.rpartition(".")
        if leg:
            record[leg][key] = fields[path]
        else:
            record[key] = fields[path]
    return InterestRateSwap.model_validate(record)
