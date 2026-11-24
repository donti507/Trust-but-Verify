"""Step 9: run every verifier rule on clean synthetic gold and on adversarial cases.

Writes results/verifier_tests.md (not committed). Documents are generated fresh
from fixed seeds into a scratch directory, so the report is reproducible and
never touches data/.

    python scripts/verifier_tests.py                     # full run (~10 minutes)
    python scripts/verifier_tests.py --clean 12 --parents 2   # quick smoke run
    python scripts/verifier_tests.py --freeze            # also write prereg/verifier-v1.0.json (draft)
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
from random import Random
import subprocess
import sys
import tempfile

from trustswap.common.normalise import normalise_floating_rate_option, normalise_notional
from trustswap.synth import defects as D
from trustswap.synth.render import render_trade, trade_fields
from trustswap.synth.sampler import sample_trade
from trustswap.synth.templates import FAMILY_SPLITS
from trustswap.verify.evaluate import AdversarialReport, CleanReport, render_markdown, verifier_hash
from trustswap.verify.run import verify

ROOT = Path(__file__).resolve().parents[1]
BASE_SEED = 20261026  # configs/default.yaml


def _displayed_value(name: str, shown: str, current):
    """Value a literal reader takes from displayed text; None if the text does not normalise."""
    try:
        if name == "notional_amount":
            return normalise_notional(shown)[0]
        if name == "floating_rate_option":
            return normalise_floating_rate_option(shown)
        if name in {"payer", "receiver", "notice_address"}:
            return shown.strip()
        if isinstance(current, str):
            return shown.strip()
    except ValueError:
        return None
    return None


def literal_reading(case, rendered):
    """What an extractor that takes the document at face value would return.

    Starts from the gold labels; ambiguous fields take the displayed value, and
    fields whose displayed text was altered by the injector (case.field_text) take
    that text, normalised with the Step 1 normalisers. Text that does not
    normalise (e.g. OCR damage) leaves the gold value, i.e. a reader that repairs it.
    """
    gold = rendered.gold.trade.model_copy(deep=True)
    fields = trade_fields(gold)
    for path, term in fields.items():
        if term.status == "ambiguous":
            term.value, term.status, term.span = trade_fields(case.trade)[path].value, "explicit", None
    for path, shown in case.field_text.items():
        term = fields.get(path)
        if term is None or term.value is None:
            continue
        value = _displayed_value(path.rsplit(".", 1)[-1], shown, term.value)
        if value is not None:
            term.value, term.status = value, "explicit"
    return gold


def run_clean(n: int, scratch: Path, report: CleanReport, log=print) -> None:
    families = tuple(FAMILY_SPLITS)
    for i in range(n):
        seed = BASE_SEED + i
        family = families[i % len(families)]
        rendered = render_trade(sample_trade(Random(seed)), seed=seed, doc_id=f"clean_{i:05d}",
                                template_family=family, output_dir=scratch / "clean")
        report.add(rendered.gold.doc_id, family, verify(rendered.gold.trade, rendered.text))
        if (i + 1) % 50 == 0:
            log(f"  clean {i + 1}/{n}")


def run_adversarial(parents: int, scratch: Path, report: AdversarialReport, log=print) -> None:
    for i in range(parents):
        seed = BASE_SEED + 100_000 + i
        parent = sample_trade(Random(seed))
        out = scratch / f"adv_{i:04d}"  # one directory per parent: render_case needs one seed/family
        for kind in D.DEFECT_TYPES:
            for severity in D.SEVERITIES:
                case = D.INJECTORS[kind](parent, seed, severity=severity)
                rendered = D.render_case(case, output_dir=out)
                report.add_defect(kind, severity, case.metadata.expected_rules, case.metadata.affected_fields,
                                  verify(literal_reading(case, rendered), rendered.text))
            control = D.make_control(kind, parent, seed)
            rendered = D.render_case(control, output_dir=out)
            report.add_control(kind, control.metadata.expected_rules,
                               verify(literal_reading(control, rendered), rendered.text))
        log(f"  adversarial parent {i + 1}/{parents}")


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean", type=int, default=300, help="clean synthetic documents (default 300)")
    parser.add_argument("--parents", type=int, default=25, help="adversarial parent trades (default 25)")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "verifier_tests.md")
    parser.add_argument("--freeze", action="store_true", help="write prereg/verifier-v1.0.json as a draft")
    args = parser.parse_args(argv)

    clean, adversarial = CleanReport(), AdversarialReport()
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        print(f"clean synthetic: {args.clean} documents")
        run_clean(args.clean, scratch, clean)
        print(f"adversarial: {args.parents} parents x {len(D.DEFECT_TYPES)} defects x 2 severities + controls")
        run_adversarial(args.parents, scratch, adversarial)

    digest = verifier_hash()
    settings = f"{args.clean} clean documents, {args.parents} adversarial parents, base seed {BASE_SEED}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(clean, adversarial, digest=digest, settings=settings), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"clean pass rate: {100 * clean.pass_rate:.1f}%   verifier hash: {digest[:12]}")

    if args.freeze:
        record = {"tag": "verifier-v1.0", "status": "draft", "verifier_hash": digest,
                  "commit": _git_commit(), "date": date.today().isoformat(),
                  "clean_pass_rate": clean.pass_rate, "clean_documents": clean.documents,
                  "adversarial_parents": args.parents,
                  "domain_lead_review": None,
                  "note": "Set status to 'frozen' and fill domain_lead_review after sign-off, then tag."}
        path = ROOT / "prereg" / "verifier-v1.0.json"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0 if clean.pass_rate == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
