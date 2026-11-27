"""Export a trusted saved risk model and the generated replay for the feed runner."""
import argparse
from pathlib import Path
import subprocess

import joblib

from trustswap.feed.artifacts import export_model, replay_from_page, serialized
from trustswap.feed.publish import write

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "results/models/frontier/full_logistic.joblib")
    parser.add_argument("--replay", type=Path, help="Local generated HTML override for origin/gh-pages:index.html")
    parser.add_argument("--output", type=Path, default=ROOT / "results/feed_artifacts")
    args = parser.parse_args(argv)
    try:
        if args.replay:
            html = args.replay.read_text(encoding="utf-8")
        else:
            html = subprocess.run(["git", "show", "origin/gh-pages:index.html"], cwd=ROOT, text=True,
                                  capture_output=True, check=True).stdout
        replay = replay_from_page(html)
        model = export_model(joblib.load(args.model))
        replay_body, model_body = serialized(replay), serialized(model)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Artifact export failed: {exc}\n")
    write(args.output / "model.json", model_body + "\n")
    write(args.output / "replay.json", replay_body + "\n")
    print(f"Wrote model.json and replay.json to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
