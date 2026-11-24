"""Prompt construction for extraction (Step 10).

* Documents are shown with numbered lines between fixed delimiters; delimiter
  look-alikes inside the document are defanged in the displayed copy only.
* Few-shot examples come from synthetic documents in *train* template families
  only, with answers taken from the structured gold trade (never from a model).
* Every prompt has a version; the version and the full prompt text enter the
  cache key, so a changed prompt can never reuse an old output.
"""
from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
import re

from trustswap.extract.fields import PATHS

PROMPT_VERSION = "v1"
PROMPTS = files("trustswap.extract").joinpath("prompts")
DELIMITERS = ("<<<DOCUMENT_START>>>", "<<<DOCUMENT_END>>>")
_USER_MARK = "=== USER ==="


def line_starts(text: str) -> list[int]:
    """Character offset of the start of every line (line 1 starts at 0)."""
    return [0] + [m.end() for m in re.finditer(r"\n", text)]


def number_lines(text: str) -> str:
    """Document as shown to the model: 'N| line', with delimiter look-alikes defanged."""
    shown = text.replace("<<<", "< < <").replace(">>>", "> > >")
    return "\n".join(f"{i}| {line}" for i, line in enumerate(shown.split("\n"), start=1))


def load_template(version: str = PROMPT_VERSION) -> tuple[str, str]:
    raw = PROMPTS.joinpath(f"{version}.txt").read_text(encoding="utf-8")
    system, user = raw.split(_USER_MARK + "\n", 1)
    return system.strip() + "\n", user


def load_fewshot(version: str = PROMPT_VERSION) -> list[dict]:
    return json.loads(PROMPTS.joinpath(f"fewshot_{version}.json").read_text(encoding="utf-8"))


def _render_examples(examples: list[dict]) -> str:
    parts = []
    for i, example in enumerate(examples, start=1):
        parts.append(f"Example {i}.\n\n{DELIMITERS[0]}\n{number_lines(example['text'])}\n{DELIMITERS[1]}\n\n"
                     f"Answer:\n{json.dumps(example['answer'], ensure_ascii=False)}\n")
    return "\n".join(parts)


def build_messages(text: str, version: str = PROMPT_VERSION) -> list[dict]:
    system, user = load_template(version)
    system = system.replace("{field_list}", "\n".join(f"- {p}" for p in PATHS))
    user = (user.replace("{examples}", _render_examples(load_fewshot(version)))
                .replace("{document}", number_lines(text)))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def gold_answer(gold_trade, text: str) -> dict:
    """Ideal model answer for a synthetic document, from its structured gold trade."""
    from trustswap.synth.render import trade_fields

    starts = line_starts(text)
    fields = trade_fields(gold_trade)
    answer = {}
    for path in PATHS:
        term = fields.get(path)
        if term is None or term.span is None:
            answer[path] = {"value": None, "line": None, "confidence": 0.9}
            continue
        start, end = term.span
        line = max(i for i, offset in enumerate(starts, start=1) if offset <= start)
        answer[path] = {"value": text[start:end], "line": line, "confidence": 0.95}
    structure = "fixed_float" if gold_trade.fixed_leg is not None else "float_float"
    return {"structure": structure, "fields": answer}


def build_fewshot(out: Path, seeds=((11, "label_value"), (12, "two_column"))) -> list[dict]:
    """Render the few-shot examples (one fixed/float, one float/float) from train families."""
    import tempfile
    from random import Random

    from trustswap.synth.render import render_trade
    from trustswap.synth.sampler import sample_trade
    from trustswap.synth.templates import FAMILY_SPLITS

    examples = []
    with tempfile.TemporaryDirectory() as tmp:
        for (seed, family), want_fixed in zip(seeds, (True, False)):
            if FAMILY_SPLITS[family] != "train":
                raise ValueError(f"few-shot family {family} is not a train family")
            rng = Random(seed)
            while True:
                trade = sample_trade(rng)
                if (trade.fixed_leg is not None) == want_fixed:
                    break
            rendered = render_trade(trade, seed=seed, doc_id=f"fewshot_{seed}", template_family=family,
                                    output_dir=tmp)
            examples.append({"doc_id": rendered.gold.doc_id, "template_family": family, "split": "train",
                             "text": rendered.text, "answer": gold_answer(rendered.gold.trade, rendered.text)})
    out.write_text(json.dumps(examples, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return examples
