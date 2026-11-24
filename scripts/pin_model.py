"""Probe an OpenAI-compatible model and pin it in configs/models.yaml (Step 10).

Tests what the endpoint supports and writes the matching settings for one track:

    python scripts/pin_model.py --track frontier --name DeepSeek-V4.1-Flash \
        --version deepseek-ai/DeepSeek-V4.1-Flash-1017
    python scripts/pin_model.py --track open_weights --name Qwen3.8-27B --version Qwen/Qwen3.8-27B-FP8

Uses the base-URL and API-key environment variables named for that track in
configs/models.yaml (e.g. FRONTIER_BASE_URL / FRONTIER_API_KEY). Never prints keys.
Add --dry-run to only show the results.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import re
import sys

from trustswap.extract.clients import MODELS_CONFIG, _message, load_models
from trustswap.extract.fields import PATHS, output_schema

PROBE_PROMPT = ("Return JSON for an interest rate swap extraction where every field is null: "
                '{"structure": "fixed_float", "fields": {<each path>: {"value": null, "line": null, "confidence": 0.5}}}. '
                "Paths: " + ", ".join(PATHS))
STRUCTURED_ORDER = ("json_schema", "guided_json", "json_object")  # strongest first


def probe(client, name: str) -> dict:
    """{feature: (ok, note)} for seed, each structured mode and log-probs."""
    messages = [{"role": "user", "content": PROBE_PROMPT}]
    schema = output_schema()
    variants = {
        "plain": {},
        "seed": {"seed": 0},
        "json_schema": {"response_format": {"type": "json_schema", "json_schema": {
            "name": "swap_extraction", "schema": schema, "strict": True}}},
        "guided_json": {"extra_body": {"guided_json": schema}},
        "json_object": {"response_format": {"type": "json_object"}},
        "logprobs": {"logprobs": True},
    }
    results = {}
    for feature, extra in variants.items():
        try:
            response = client.chat.completions.create(model=name, messages=messages, max_tokens=3000,
                                                      temperature=0, **extra)
            choice = response.choices[0]
            text = choice.message.content or ""
            note = ""
            if feature in STRUCTURED_ORDER:
                try:
                    data = json.loads(text)
                    ok = isinstance(data, dict) and isinstance(data.get("fields"), dict)
                    note = "valid JSON" if ok else "JSON without 'fields'"
                except json.JSONDecodeError:
                    ok, note = False, "reply is not JSON"
            elif feature == "logprobs":
                ok = bool(choice.logprobs and getattr(choice.logprobs, "content", None))
                note = "" if ok else "accepted but no log-probs returned"
            else:
                ok = True
            results[feature] = (ok, note)
        except Exception as error:  # noqa: BLE001 - report every provider error the same way
            results[feature] = (False, _message(error)[:120])
        print(f"  {feature:12s} {'OK  ' if results[feature][0] else 'FAIL'} {results[feature][1]}", flush=True)
    return results


def choose(results: dict) -> dict:
    if not results["plain"][0]:
        raise SystemExit("the model does not answer at all: check the model name, URL and key")
    structured = next((mode for mode in STRUCTURED_ORDER if results[mode][0]), None)
    if structured is None:
        raise SystemExit("no JSON mode works for this model; choose another model")
    return {"structured": structured, "send_seed": results["seed"][0], "logprobs": results["logprobs"][0]}


def write_track(path: Path, track: str, values: dict) -> None:
    """Replace keys inside one track's block, keeping the file's comments and layout."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == f"{track}:")
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"^  \S", lines[i])), len(lines))
    block = lines[start + 1:end]
    for key, value in values.items():
        text = json.dumps(value) if isinstance(value, str) else str(value).lower()
        new = f"    {key}: {text}"
        for i, line in enumerate(block):
            if re.match(rf"^    {key}:", line):
                block[i] = new
                break
        else:
            block.append(new)
    path.write_text("\n".join(lines[:start + 1] + block + lines[end:]) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", required=True, choices=["frontier", "open_weights"])
    parser.add_argument("--name", required=True, help="model id the API expects")
    parser.add_argument("--version", required=True, help="pinned version, e.g. the Hugging Face repo@revision")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    import openai

    spec = load_models()[args.track]
    key = os.environ.get(spec.api_key_env)
    base_url = os.environ.get(spec.base_url_env) if spec.base_url_env else None
    if not key:
        raise SystemExit(f"set {spec.api_key_env} in ~/.bashrc first")
    client = openai.OpenAI(api_key=key, base_url=base_url, timeout=300, max_retries=2)
    print(f"probing {args.name} at {base_url or 'provider default'}")
    settings = choose(probe(client, args.name))
    values = {"name": args.name, "version": args.version, "access_date": date.today().isoformat(), **settings}
    print("settings:", values)
    if args.dry_run:
        return 0
    write_track(MODELS_CONFIG, args.track, values)
    print(f"pinned {args.track} in {MODELS_CONFIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
