"""Extract one document K times through the cache and write extraction records (Step 10).

Record file: cache/extract/<model_key>/<doc_id>.jsonl, one line per sample k:
{"doc_id", "model", "prompt_version", "k", "temperature", "schema_valid", "errors",
 "structure", "fields": {path: {"value", "span", "verbal_conf", "mean_logprob", ...}},
 "finish_reason", "input_tokens", "output_tokens", "latency_s", "cache_key"}

k = 0 runs at temperature 0; k >= 1 at the sampling temperature. Raw completions
are cached by call_key(model_id, prompt_version, prompt, {temperature, k, seed}),
so re-running never calls the model again (rule 5).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from trustswap.common.cache import Cache
from trustswap.common.hashing import call_key
from trustswap.extract.clients import Completion
from trustswap.extract.parse import parse_completion
from trustswap.extract.prompt import PROMPT_VERSION, build_messages

BLOCKED_SPLITS = {"real_test"}  # rule 1: never before Step 19 pre-registration


class Client(Protocol):
    params: dict

    def complete(self, messages: list[dict], *, temperature: float, seed: int) -> Completion: ...


def temperature_for(k: int, sampling_temperature: float = 0.7) -> float:
    return 0.0 if k == 0 else sampling_temperature


def extract_document(doc_id: str, text: str, client: Client, *, model_id: str, cache: Cache,
                     out_dir: Path, k_samples: int = 1, sampling_temperature: float = 0.7,
                     prompt_version: str = PROMPT_VERSION, split: str | None = None) -> list[dict]:
    if split in BLOCKED_SPLITS:
        raise PermissionError(f"{doc_id}: split {split} is locked until pre-registration (Step 19)")
    messages = build_messages(text, prompt_version)
    prompt = json.dumps(messages, ensure_ascii=False)
    records = []
    for k in range(k_samples):
        temperature = temperature_for(k, sampling_temperature)
        params = {**client.params, "temperature": temperature, "k": k, "seed": k}
        key = call_key(model_id, prompt_version, prompt, params)
        cached = cache.has(key)
        raw = cache.get_or_compute(key, lambda: client.complete(messages, temperature=temperature, seed=k).to_cache())
        completion = Completion.from_cache(raw)
        parsed = parse_completion(completion.text, text, completion.tokens)
        records.append({
            "doc_id": doc_id, "model": model_id, "prompt_version": prompt_version, "k": k,
            "temperature": temperature, "schema_valid": parsed.schema_valid, "errors": parsed.errors,
            "structure": parsed.structure, "fields": parsed.fields,
            "finish_reason": completion.finish_reason, "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens, "latency_s": completion.latency_s, "cache_key": key, "cached": cached,
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{doc_id}.jsonl"
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records),
                   encoding="utf-8")
    tmp.replace(path)
    return records
