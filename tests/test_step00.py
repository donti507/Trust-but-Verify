import json

from trustswap.common.cache import Cache
from trustswap.common.hashing import call_key, sha256_file, sha256_text

import run


def test_hash_is_stable_and_order_independent():
    a = call_key("m", "v1", "prompt", {"t": 0, "k": 1})
    b = call_key("m", "v1", "prompt", {"k": 1, "t": 0})
    assert a == b and len(a) == 64
    assert call_key("m", "v2", "prompt") != call_key("m", "v1", "prompt")


def test_file_hash_matches_text_hash(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("swap", encoding="utf-8")
    assert sha256_file(p) == sha256_text("swap")


def test_cache_never_recomputes(tmp_path):
    cache, calls = Cache(tmp_path), []
    fn = lambda: calls.append(1) or {"value": 0.0425}
    key = call_key("m", "v1", "doc")
    assert cache.get_or_compute(key, fn) == {"value": 0.0425}
    assert cache.get_or_compute(key, fn) == {"value": 0.0425}
    assert len(calls) == 1


def test_run_lists_all_steps():
    assert len(run.STEPS) == 27
    assert run.main(["s00"]) == 0
