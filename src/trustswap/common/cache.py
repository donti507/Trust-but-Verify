"""Read-through cache for model outputs, keyed by input hash (Step 0).

Rule 5: re-running an analysis must never re-call a model. Every call goes through `get_or_compute`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


class Cache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> Any:
        return json.loads(self._path(key).read_text(encoding="utf-8"))

    def put(self, key: str, value: Any) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp.replace(p)  # atomic: a killed session never leaves a half-written entry

    def get_or_compute(self, key: str, fn: Callable[[], Any]) -> Any:
        if self.has(key):
            return self.get(key)
        value = fn()
        self.put(key, value)
        return value
