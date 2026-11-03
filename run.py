"""Single entry point: python run.py <step> --config configs/default.yaml"""
from __future__ import annotations

import argparse
import sys

from trustswap.common.config import load_config, set_seed

STEPS = {
    "s00": "Environment and repository",
    "s01": "Field schema",
    "s02": "Ingest pipeline",
    "s03": "Real set: scoping and download",
    "s04": "Real set: annotation and splits",
    "s05": "Synthetic generator",
    "s06": "Adversarial injector",
    "s07": "Verifier V1 to V4",
    "s08": "Verifier V5 to V7",
    "s09": "Verifier tests and freeze",
    "s10": "Prompts and model clients",
    "s11": "Extraction pilot",
    "s12": "Full extraction",
    "s13": "Feature builder",
    "s14": "Risk models and calibration",
    "s15": "Cost model and decision rule",
    "s16": "Conformal risk control",
    "s17": "Track B gate and fine-tune",
    "s18": "Baselines",
    "s19": "Pre-registration",
    "s20": "Real-set extraction",
    "s21": "Real evaluation, run once",
    "s22": "Hypothesis tests",
    "s23": "Ablations and controls",
    "s24": "Figures and tables",
    "s25": "Paper 1",
    "s26": "Benchmark release",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Trust but Verify pipeline. Steps:\n" + "\n".join(f"  {k}  {v}" for k, v in STEPS.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("step", choices=sorted(STEPS))
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    print(f"[{args.step}] {STEPS[args.step]}: not implemented yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
