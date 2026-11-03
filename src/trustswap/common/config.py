"""Load experiment configs and fix the global seed (Step 0)."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # optional at Step 0
        np.random.seed(seed)
    except ImportError:
        pass
