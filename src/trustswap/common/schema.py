"""Typed swap extraction records; spans are half-open character offsets."""
from datetime import date
from decimal import Decimal
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")
Status = Literal["explicit", "by_reference", "ambiguous", "absent"]


class ExtractedField(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")
    value: T | None = None
    span: tuple[int, int] | None = None
    status: Status = "absent"

    @model_validator(mode="after")
    def valid_span(self):
        if self.span is not None and not (0 <= self.span[0] < self.span[1]):
            raise ValueError("span must be nonnegative, nonempty half-open offsets")
        if self.status == "absent" and (self.value is not None or self.span is not None):
            raise ValueError("absent fields cannot have a value or span")
        return self


class Leg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payer: ExtractedField[str] = Field(default_factory=ExtractedField)
    receiver: ExtractedField[str] = Field(default_factory=ExtractedField)
    notional_amount: ExtractedField[Decimal] = Field(default_factory=ExtractedField)
    notional_currency: ExtractedField[str] = Field(default_factory=ExtractedField)
    payment_frequency: ExtractedField[str] = Field(default_factory=ExtractedField)
    reset_frequency: ExtractedField[str] = Field(default_factory=ExtractedField)
    day_count_fraction: ExtractedField[str] = Field(default_factory=ExtractedField)
    business_day_convention: ExtractedField[str] = Field(default_factory=ExtractedField)
    business_centers: ExtractedField[list[str]] = Field(default_factory=ExtractedField)


class FixedLeg(Leg):
    fixed_rate: ExtractedField[Decimal] = Field(default_factory=ExtractedField)


class FloatingLeg(Leg):
    floating_rate_option: ExtractedField[str] = Field(default_factory=ExtractedField)
    spread: ExtractedField[Decimal] = Field(default_factory=ExtractedField)


class InterestRateSwap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trade_date: ExtractedField[date] = Field(default_factory=ExtractedField)
    effective_date: ExtractedField[date] = Field(default_factory=ExtractedField)
    termination_date: ExtractedField[date] = Field(default_factory=ExtractedField)
    fixed_leg: FixedLeg | None = None
    floating_leg: FloatingLeg
    floating_leg_2: FloatingLeg | None = None
    notice_address: ExtractedField[str] = Field(default_factory=ExtractedField)

    @model_validator(mode="after")
    def exactly_two_legs(self):
        if (self.fixed_leg is None) == (self.floating_leg_2 is None):
            raise ValueError("provide either a fixed leg or a second floating leg")
        return self
