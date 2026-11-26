"""Prepare the weekday DTCC feed and Pages snapshot without changing research data."""
import argparse
from datetime import date, datetime, timezone
from functools import cached_property
import fcntl
import json
import math
import os
from pathlib import Path
import re

import joblib
import yaml

from trustswap.common.cache import Cache
from trustswap.decide.policy import load_costs
from trustswap.extract.clients import load_models
from trustswap.extract.prompt import prompt_sha
from trustswap.verify.evaluate import verifier_hash
from trustswap.feed import dtcc
from trustswap.feed.pipeline import BudgetedClient, FeedSkip, ReviewPipeline, SingleAttemptClient
from trustswap.feed.publish import run
from trustswap.feed.render import render_batch

if __package__:
    from .build_demo import CATEGORIES, render_page
else:
    from build_demo import CATEGORIES, render_page

ROOT = Path(__file__).resolve().parents[1]


class Inputs:
    def __init__(self, args, threshold, limit):
        self.args, self.threshold, self.limit = args, threshold, limit

    @cached_property
    def replay(self):
        match = re.search(r'<script type="application/json" id="demo-data">(.*?)</script>',
                          self.args.replay.read_text(encoding="utf-8"), re.S)
        data = json.loads(match[1])
        docs = data["documents"]
        if len(docs) != 30 or any(doc["category"] not in CATEGORIES for doc in docs) or data.get("illustrative"):
            raise ValueError("Replay must contain 30 cached generated confirmations")
        return data

    @cached_property
    def scorer(self):
        prompt = json.loads((ROOT / "prereg" / "prompt-freeze.json").read_text())
        verifier = json.loads((ROOT / "prereg" / "verifier-v1.0.json").read_text())
        if prompt["status"] != "frozen" or prompt_sha("v1") != prompt["prompt_sha256"]:
            raise FeedSkip("Frozen v1 prompt fingerprint differs")
        if verifier_hash() != verifier["verifier_hash"]:
            raise FeedSkip("Pinned verifier fingerprint differs")
        spec = load_models()["frontier"]
        spec.check_pinned()
        if not os.environ.get(spec.base_url_env or "") or not os.environ.get(spec.api_key_env):
            raise FeedSkip("PCSS environment credentials or endpoint unavailable")
        today = datetime.now(timezone.utc).date()
        budget = BudgetedClient(SingleAttemptClient(spec, timeout=90), self.args.cache / "budget" / f"{today}.txt", self.limit)
        costs = load_costs()
        if costs.B != self.replay["bound"]:
            raise ValueError("Replay and feed must share the document loss bound")
        return ReviewPipeline(budget, spec.model_id, Cache(self.args.cache / "calls"),
                              joblib.load(self.args.risk_model), costs, self.threshold)

    def page(self, feed):
        return render_page(self.replay["documents"], self.replay["bound"], feed=feed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=datetime.now(timezone.utc).date())
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "feed.yaml")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache" / "live_feed")
    parser.add_argument("--feed-dir", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--risk-model", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    threshold, limit = float(config["lambda"]), int(config["daily_token_cap"])
    if not math.isfinite(threshold) or threshold < 0 or not 0 < limit <= 60000:
        parser.error("lambda must be finite and nonnegative, and the cap must be at most 60,000")
    inputs = Inputs(args, threshold, limit)
    args.cache.mkdir(parents=True, exist_ok=True)
    try:
        with (args.cache / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            line = run(args.date, args.feed_dir, args.site, threshold, download=dtcc.download,
                       select=dtcc.select_trades, render=render_batch, scorer=lambda doc: inputs.scorer(doc),
                       page=inputs.page, previous_day=dtcc.previous_business_day)
    except (dtcc.NoFile, FeedSkip) as exc:
        line = f"Skipped: {exc}"
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
