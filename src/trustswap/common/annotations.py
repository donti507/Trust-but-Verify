"""Partial human annotations: unanswered fields are never defaulted to absent."""
from decimal import Decimal
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trustswap.common import normalise as n
from trustswap.common.schema import InterestRateSwap, FixedLeg, FloatingLeg

FIELD_MODELS = {name: InterestRateSwap.model_fields[name].annotation
                for name in ("trade_date", "effective_date", "termination_date")}
for prefix, leg in (("fixed_leg", FixedLeg), ("floating_leg", FloatingLeg)):
    FIELD_MODELS.update({f"{prefix}.{name}": info.annotation for name, info in leg.model_fields.items()})
FIELD_MODELS["notice_address"] = InterestRateSwap.model_fields["notice_address"].annotation
FIELDS = tuple(FIELD_MODELS)
FLAGS = ("amortising", "cap", "floor")


def normalise_field(path, text, *, rate_unit=None):
    name = path.split(".")[-1]
    if name.endswith("_date"):
        return n.normalise_date(text)
    if name in {"fixed_rate", "spread"}:
        return n.normalise_rate(text, unit=rate_unit)
    if name == "notional_amount":
        return n.normalise_notional(text)[0]
    if name == "notional_currency":
        return n.normalise_notional(text.strip() + " 1")[1]
    if name in {"payment_frequency", "reset_frequency"}:
        return n.normalise_frequency(text)
    if name == "day_count_fraction":
        return n.normalise_day_count(text)
    if name == "business_day_convention":
        return n.normalise_business_day_convention(text)
    if name == "business_centers":
        return n.normalise_business_centers(text)
    if name == "floating_rate_option":
        return n.normalise_floating_rate_option(text)
    return " ".join(text.split())


def validated_field(path, value, span, status):
    field = FIELD_MODELS[path].model_validate({"value": value, "span": span, "status": status})
    if status == "explicit" and (field.value is None or field.span is None):
        raise ValueError("explicit annotation requires a value and exact source span")
    if status == "by_reference" and field.span is not None:
        raise ValueError("by-reference spans must be null")
    if status == "ambiguous" and (field.value is not None or field.span is not None):
        raise ValueError("ambiguous annotation has no chosen value or single span")
    return field.model_dump(mode="json")


def canonical_value(path, field):
    """Canonical comparison of already-typed JSON annotation values."""
    value = field["value"]
    if value is None:
        return None
    name = path.split(".")[-1]
    if name in {"fixed_rate", "spread", "notional_amount"}:
        return Decimal(str(value))
    if name == "business_centers":
        return tuple(n.normalise_business_centers(",".join(value)))
    return normalise_field(path, str(value))


class AnnotationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    collection: Literal["real"] = "real"
    annotator: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
    text_path: str
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["real_cal"] | None = None
    fields: dict[str, dict] = Field(default_factory=dict)
    flags: list[Literal["amortising", "cap", "floor"]] = Field(default_factory=list)
    flags_reviewed: bool = False
    complete: bool = False

    @model_validator(mode="after")
    def validate_answers(self):
        for path, answer in self.fields.items():
            if path not in FIELD_MODELS:
                raise ValueError(f"unknown schema path: {path}")
            if set(answer) != {"value", "span", "status"}:
                raise ValueError("each answered field requires exactly value, span and status")
            self.fields[path] = validated_field(path, **answer)
        self.flags = sorted(set(self.flags))
        self.complete = set(self.fields) == set(FIELDS) and self.flags_reviewed
        return self

    def as_trade(self):
        if not self.complete:
            raise ValueError("annotation is incomplete; unanswered fields are not absent")
        trade = {"fixed_leg": {}, "floating_leg": {}}
        for path, answer in self.fields.items():
            if "." in path:
                leg, name = path.split(".")
                trade[leg][name] = answer
            else:
                trade[path] = answer
        return InterestRateSwap.model_validate(trade)
