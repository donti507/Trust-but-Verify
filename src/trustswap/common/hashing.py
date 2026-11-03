"""Stable SHA-256 hashes for files, text and model calls (Step 0)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def call_key(model: str, prompt_version: str, prompt: str, params: dict[str, Any] | None = None) -> str:
    """Key for one model call: identical inputs always give the identical key."""
    payload = json.dumps(
        {"model": model, "prompt_version": prompt_version, "prompt": prompt, "params": params or {}},
        sort_keys=True, ensure_ascii=False,
    )
    return sha256_text(payload)
