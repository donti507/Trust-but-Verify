"""Render feed confirmations with the existing synthetic renderer in isolated scratch."""
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from trustswap.common.records import GoldRecord
from trustswap.synth import defects
from trustswap.synth.render import render_trade

DEFECTS = (defects.contradiction, defects.scale_error, defects.missing_information)


@dataclass(frozen=True)
class Confirmation:
    doc_id: str
    text: str
    gold: GoldRecord
    defect: str | None
    affected_fields: tuple[str, ...]


def render_batch(trades, day: date):
    rng = Random(day.isoformat() + ":render")
    planted = set(rng.sample(range(len(trades)), round(len(trades) * 0.4)))
    with TemporaryDirectory(prefix="live-feed-") as directory:
        for i, trade in enumerate(trades):
            doc_id = f"live-{day.isoformat()}-{i + 1:02d}"
            seed = rng.randrange(2**32)
            options = {}
            if i in planted:
                case = rng.choice(DEFECTS)(trade, seed, parent_id=doc_id)
                trade = case.trade
                options = dict(collection="adversarial", gold_trade=case.gold.trade,
                               field_text=case.field_text, omitted_fields=case.omitted_fields,
                               notes=case.notes, adversarial=case.metadata)
            result = render_trade(trade, doc_id=doc_id, seed=seed, output_dir=Path(directory), **options)
            metadata = options.get("adversarial")
            yield Confirmation(doc_id, result.text, result.gold, metadata.defect if metadata else None,
                               tuple(metadata.affected_fields) if metadata else ())
