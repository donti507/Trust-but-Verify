"""Step 9: verifier tests and freeze.

The full run (300 clean documents, 25 adversarial parents) is
``python scripts/verifier_tests.py``; these tests run a small version of it.
"""
import json
from pathlib import Path

import pytest

from scripts.verifier_tests import literal_reading, run_adversarial, run_clean
from trustswap.synth import defects as D
from trustswap.verify.evaluate import AdversarialReport, CleanReport, render_markdown, verifier_hash

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "prereg" / "verifier-v1.0.json"


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("step9")
    clean, adversarial = CleanReport(), AdversarialReport()
    run_clean(12, scratch, clean, log=lambda *_: None)  # two documents per template family
    run_adversarial(1, scratch, adversarial, log=lambda *_: None)
    return clean, adversarial


def test_clean_synthetic_gold_passes_every_hard_rule(reports):
    clean, _ = reports
    assert clean.documents == 12 and len(clean.by_family) == 6
    assert clean.hard == []
    assert clean.pass_rate == 1.0


def test_recall_table_covers_every_defect_type(reports):
    _, adversarial = reports
    assert set(adversarial.defects) == set(D.DEFECT_TYPES)
    for kind in D.DEFECT_TYPES:
        assert set(adversarial.defects[kind]) == set(D.SEVERITIES)
    markdown = render_markdown(*reports, digest=verifier_hash(), settings="test")
    for kind in D.DEFECT_TYPES:
        assert f"| {kind} |" in markdown


def test_controls_raise_no_hard_verdict(reports):
    _, adversarial = reports
    assert set(adversarial.controls) == set(D.DEFECT_TYPES)
    assert not any(any_hard for rows in adversarial.controls.values() for any_hard, _, _ in rows)


def test_silent_predictions_never_count_as_detected(reports):
    _, adversarial = reports
    for kind, by_severity in adversarial.defects.items():
        for rows in by_severity.values():
            for expected, _, detected in rows:
                if expected == ("silent",):
                    assert detected is False, kind


def test_literal_reading_fills_only_ambiguous_fields(tmp_path):
    from random import Random

    from trustswap.synth.render import trade_fields
    from trustswap.synth.sampler import sample_trade

    parent = sample_trade(Random(5))
    case = D.INJECTORS["contradiction"](parent, 5, severity=1)
    rendered = D.render_case(case, output_dir=tmp_path)
    reading = literal_reading(case, rendered)
    fields = trade_fields(reading)
    assert fields["floating_leg.notional_amount"].status == "explicit"
    assert fields["floating_leg.notional_amount"].value == parent.floating_leg.notional_amount.value
    assert all(term.status != "ambiguous" for term in fields.values())


def test_verifier_hash_is_stable_hex():
    first, second = verifier_hash(), verifier_hash()
    assert first == second and len(first) == 64 and int(first, 16) >= 0


def test_frozen_verifier_matches_its_recorded_hash():
    """After sign-off (status 'frozen'), any rule change must be a logged bug fix with a new hash."""
    if not FREEZE.exists():
        pytest.skip("verifier not frozen yet")
    record = json.loads(FREEZE.read_text(encoding="utf-8"))
    if record.get("status") != "frozen":
        pytest.skip("freeze record is still a draft")
    assert record["verifier_hash"] == verifier_hash(), (
        "verifier changed after freeze: log the bug fix in prereg/verifier_changes.md and update the hash")


def test_literal_reading_takes_the_displayed_text(tmp_path):
    from random import Random

    from trustswap.synth.render import trade_fields
    from trustswap.synth.sampler import sample_trade

    parent = sample_trade(Random(20361026))  # a GBP-SONIA trade
    case = D.INJECTORS["legacy_index_fallback"](parent, 9, severity=1)
    rendered = D.render_case(case, output_dir=tmp_path / "a")
    option = trade_fields(literal_reading(case, rendered))["floating_leg.floating_rate_option"].value
    assert option.upper() == case.field_text["floating_leg.floating_rate_option"].upper()

    case = D.INJECTORS["scale_error"](parent, 9, severity=1)
    rendered = D.render_case(case, output_dir=tmp_path / "b")
    amount = trade_fields(literal_reading(case, rendered))["floating_leg.notional_amount"].value
    assert amount == parent.floating_leg.notional_amount.value * 10
