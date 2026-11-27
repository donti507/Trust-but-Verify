"""Cached and explicitly opt-in product classification and extraction."""
from dataclasses import replace
import json
import os
from pathlib import Path

from trustswap.common.cache import Cache
from trustswap.common.hashing import call_key, sha256_bytes
from trustswap.desk.products import PRODUCTS
from trustswap.extract.clients import load_models
from trustswap.feed.pipeline import SingleAttemptClient

ROOT = Path(__file__).resolve().parents[4]
DETECTION_PROMPT = (
    "Classify the fictional confirmation. Respond with JSON containing only a product key. "
    "Its value is fx_forward, xccy_swap, cds, irs (single currency interest rate swap), or unknown."
)


def _spec():
    from trustswap.desk.upload_review import allow_external

    endpoint = os.environ.get("PCSS_BASE_URL", "")
    if not allow_external() or not endpoint.startswith("https://") or not os.environ.get("PCSS_API_KEY"):
        return None
    spec = replace(load_models()["frontier"], base_url_env="PCSS_BASE_URL", api_key_env="PCSS_API_KEY")
    spec.check_pinned()
    return spec


def _complete(spec, messages):
    with SingleAttemptClient(spec, timeout=90)._client() as client:
        reply = client.chat.completions.create(model=spec.name, messages=messages, temperature=0,
                                               max_tokens=spec.max_tokens, response_format={"type": "json_object"})
    if not reply.choices or reply.choices[0].finish_reason == "length" or not reply.choices[0].message.content:
        raise ValueError("Incomplete response")
    return reply.choices[0].message.content


def _extract(prompt, text, directory, spec):
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": text}]
    version = "desk-v1-" + sha256_bytes(prompt.encode())
    payload = json.dumps(messages, ensure_ascii=False)
    key = call_key(spec.model_id, version, payload, {"temperature": 0, "max_tokens": spec.max_tokens})
    try:
        content = Cache(directory / "product-calls").get_or_compute(key, lambda: _complete(spec, messages))
        result = json.loads(content)
    except Exception:
        raise ValueError("The configured provider could not process the fictional product confirmation") from None
    if not isinstance(result, dict):
        raise ValueError("The product extraction did not contain a JSON object")
    return result


def detect_remote(text, directory):
    spec = _spec()
    if spec is None:
        return None
    kind = _extract(DETECTION_PROMPT, text, directory, spec).get("product")
    return kind if isinstance(kind, str) and (kind in PRODUCTS or kind == "irs") else None


def extract_remote(kind, text, directory):
    spec = _spec()
    if spec is None:
        return {}
    if kind not in PRODUCTS:
        raise ValueError("Unknown desk product")
    prompt = (ROOT / "prompts" / f"desk_v1_{kind}.txt").read_text(encoding="utf-8")
    result = _extract(prompt, text, directory, spec)
    if result.keys() - PRODUCTS[kind][2].keys():
        raise ValueError("The product extraction included unexpected terms")
    return result
