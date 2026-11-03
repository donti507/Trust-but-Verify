"""Document metadata shared by ingestion and later runbook steps."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from trustswap.common.schema import InterestRateSwap

Collection = Literal["real", "synthetic", "adversarial"]
Split = Literal["train", "dev", "syn_test", "real_cal", "real_test", "adv_test"]


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
        return self


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

    @model_validator(mode="after")
    def provenance(self):
        if self.collection == "real":
            if self.generator is not None or not self.annotators or self.adjudicated is None:
                raise ValueError("real gold requires human annotators and adjudication metadata, without generator")
        elif self.generator is None or self.annotators is not None or self.adjudicated is not None:
            raise ValueError("generated gold requires generator metadata, without annotator metadata")
        return self

    @model_serializer(mode="wrap")
    def serialise_provenance(self, handler):
        record = handler(self)
        excluded = ("generator",) if self.collection == "real" else ("annotators", "adjudicated")
        for key in excluded:
            record.pop(key, None)
        return record
