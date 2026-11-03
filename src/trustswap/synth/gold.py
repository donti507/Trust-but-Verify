"""Gold labels come directly from generated structured trades, never extraction."""
from pathlib import Path

from trustswap.common.records import GoldRecord, GeneratorMetadata
from trustswap.common.schema import InterestRateSwap

SAMPLER_VERSION = "0.1"


def write_gold_record(trade: InterestRateSwap, *, doc_id: str, seed: int,
                      output_dir: str | Path = "data", collection: str = "synthetic",
                      template_family: str | None = None) -> GoldRecord:
    """Append one synthetic/adversarial record to <root>/<collection>/gold.jsonl.

    The seed identifies the caller's RNG stream, not an independently reseeded
    trade. Callers assign one doc_id per rendered document in later steps.
    This generator writer deliberately cannot create real-document labels.
    """
    if collection not in {"synthetic", "adversarial"}:
        raise ValueError("sampler gold may only be synthetic or adversarial")
    record = GoldRecord(doc_id=doc_id, collection=collection, trade=trade,
                        generator=GeneratorMetadata(seed=seed, sampler_version=SAMPLER_VERSION,
                                                    template_family=template_family))
    path = Path(output_dir) / collection / "gold.jsonl"
    resolved = path.resolve()
    if any(parent.name == "raw" and parent.parent.name == "data" for parent in resolved.parents):
        raise ValueError("gold output must not be inside data/raw/")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")
    return record
