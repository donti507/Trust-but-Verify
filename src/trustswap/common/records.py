"""Document metadata shared by ingestion and later runbook steps."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from trustswap.common.schema import InterestRateSwap

Collection = Literal["real", "synthetic", "adversarial"]
Split = Literal["train", "dev", "syn_test", "real_cal", "real_test", "adv_test"]


Rule = Literal["V1", "V2", "V3", "V4", "V5", "V6", "V7", "silent"]
DEFECT_FIELDS = ("severity", "affected_fields", "expected_rules", "designed_by", "is_control")


class AdversarialMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parent_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    defect: Literal["contradiction", "amendment", "non_standard_calendar", "swapped_legs",
                    "scale_error", "unusual_day_count", "stub_ambiguity", "multiple_notionals",
                    "legacy_index_fallback", "ocr_digit_corruption", "missing_information", "prompt_injection"]
    severity: Literal[1, 2]
    affected_fields: list[str] = Field(min_length=1)
    expected_rules: list[Rule] = Field(min_length=1)
    designed_by: Literal["team", "blind"] = "team"
    is_control: bool = False

    @model_validator(mode="after")
    def rule_list(self):
        if "silent" in self.expected_rules and len(self.expected_rules) != 1:
            raise ValueError("silent cannot be combined with verifier rules")
        return self


def _adversarial_metadata(record):
    if record.collection != "adversarial" and (record.parent_id is not None or record.defect is not None):
        raise ValueError("parent_id and defect are only for adversarial records")
    present = any(getattr(record, key) is not None for key in DEFECT_FIELDS)
    if present:
        if record.collection != "adversarial":
            raise ValueError("defect metadata is only for adversarial records")
        AdversarialMetadata.model_validate({key: getattr(record, key)
            for key in ("parent_id", "defect", *DEFECT_FIELDS)})


def _omit_empty_defect_fields(data):
    for key in DEFECT_FIELDS:
        if data.get(key) is None:
            data.pop(key, None)
    return data


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    collection: Collection
    family_id: str | None = None
    split: Split | None = None
    source_url: str | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text_path: str
    ocr: bool
    parent_id: str | None = None
    defect: str | None = None
    severity: Literal[1, 2] | None = None
    affected_fields: list[str] | None = None
    expected_rules: list[Rule] | None = None
    designed_by: Literal["team", "blind"] | None = None
    is_control: bool | None = None

    @model_validator(mode="after")
    def collection_metadata(self):
        allowed = {"real": {"real_cal", "real_test"},
                   "synthetic": {"train", "dev", "syn_test"},
                   "adversarial": {"adv_test"}}
        if self.split is not None and self.split not in allowed[self.collection]:
            raise ValueError("split does not belong to collection")
        if self.collection != "real" and self.source_url is not None:
            raise ValueError("source_url is only for real documents")
        if self.collection != "adversarial" and (self.parent_id is not None or self.defect is not None):
            raise ValueError("parent_id and defect are only for adversarial documents")
        _adversarial_metadata(self)
        return self

    @model_serializer(mode="wrap")
    def serialise_defect_metadata(self, handler):
        return _omit_empty_defect_fields(handler(self))


class GeneratorMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    sampler_version: str
    template_family: str | None = None


class GoldRecord(BaseModel):
    """Gold terms and collection-specific provenance; never a verifier output."""
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    collection: Collection
    trade: InterestRateSwap
    generator: GeneratorMetadata | None = None
    annotators: list[str] | None = None
    adjudicated: bool | None = None
    parent_id: str | None = None
    defect: str | None = None
    severity: Literal[1, 2] | None = None
    affected_fields: list[str] | None = None
    expected_rules: list[Rule] | None = None
    designed_by: Literal["team", "blind"] | None = None
    is_control: bool | None = None

    @model_validator(mode="after")
    def provenance(self):
        if self.collection == "real":
            if self.generator is not None or not self.annotators or self.adjudicated is None:
                raise ValueError("real gold requires human annotators and adjudication metadata, without generator")
        elif self.generator is None or self.annotators is not None or self.adjudicated is not None:
            raise ValueError("generated gold requires generator metadata, without annotator metadata")
        _adversarial_metadata(self)
        return self

    @model_serializer(mode="wrap")
    def serialise_provenance(self, handler):
        record = handler(self)
        excluded = ("generator",) if self.collection == "real" else ("annotators", "adjudicated")
        for key in excluded:
            record.pop(key, None)
        for key in ("parent_id", "defect"):
            if record.get(key) is None:
                record.pop(key, None)
        return _omit_empty_defect_fields(record)
