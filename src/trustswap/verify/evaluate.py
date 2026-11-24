"""Step 9: test the verifier on synthetic gold and adversarial cases, and freeze it.

* Clean check: every HARD-capable rule must PASS (or be UNDET) on every clean
  synthetic gold record. A HARD verdict is a generator bug or a rule bug.
* Adversarial check: each defect case is verified as a *literal reading* of the
  rendered document (ambiguous gold fields are filled with the displayed value,
  i.e. what an extractor that takes the text at face value would return).
  Recall per defect type = share of cases where an expected rule reports HARD or
  SOFT on an affected field. "silent" defects report how often anything fires.
* Controls (unusual but correct): false-alarm rate = share with any HARD; a
  second column counts any flag other than the V7 SOFT the control expects.
* Freeze: ``verifier_hash`` fingerprints the rule code and the convention table.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
from importlib.resources import files
from pathlib import Path

from trustswap.verify.base import RuleResult
from trustswap.verify.run import HARD_RULES, RULES

Results = dict[str, dict[str, RuleResult]]
VERIFY_DIR = Path(__file__).resolve().parent


def verifier_hash() -> str:
    """sha256 over the verifier source files and the V7 convention table, in a fixed order."""
    digest = hashlib.sha256()
    paths = sorted(p for p in VERIFY_DIR.glob("*.py") if p.name != "evaluate.py")
    for path in paths:
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    table = files("trustswap.ref").joinpath("conventions.yaml")
    digest.update(b"conventions.yaml\0" + table.read_bytes())
    return digest.hexdigest()


def hard_failures(results: Results) -> list[tuple[str, str, str]]:
    """(rule, field, detail) for every HARD verdict from a HARD-capable rule."""
    return [(rule, path, r.detail) for rule in HARD_RULES for path, r in results.get(rule, {}).items()
            if r.verdict == "HARD"]


def flags(results: Results, *, skip_v7_soft: bool = False) -> list[tuple[str, str, str]]:
    return [(rule, path, r.verdict) for rule, found in results.items() for path, r in found.items()
            if r.verdict in {"HARD", "SOFT"} and not (skip_v7_soft and rule == "V7" and r.verdict == "SOFT")]


def detected_by(results: Results, rule: str, fields: list[str]) -> bool:
    found = results.get(rule, {})
    return any(found[p].verdict in {"HARD", "SOFT"} for p in fields if p in found)


@dataclass
class CleanReport:
    documents: int = 0
    by_family: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    hard: list[tuple[str, str, str, str]] = field(default_factory=list)  # doc, rule, field, detail
    soft_docs: int = 0

    def add(self, doc_id: str, family: str, results: Results) -> None:
        self.documents += 1
        self.by_family[family] += 1
        self.hard += [(doc_id, *failure) for failure in hard_failures(results)]
        self.soft_docs += any(r.verdict == "SOFT" for found in results.values() for r in found.values())

    @property
    def pass_rate(self) -> float:
        failed = len({doc for doc, *_ in self.hard})
        return 1.0 - failed / self.documents if self.documents else 0.0


@dataclass
class AdversarialReport:
    # defect -> severity -> list of (expected rules, fired rules, detected)
    defects: dict[str, dict[int, list[tuple[tuple[str, ...], set[str], bool]]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    # defect -> list of (any HARD, any unexpected flag, fired rules)
    controls: dict[str, list[tuple[bool, bool, set[str]]]] = field(default_factory=lambda: defaultdict(list))

    def add_defect(self, kind: str, severity: int, expected: list[str], affected: list[str], results: Results) -> None:
        fired_rules = {rule for rule, found in results.items()
                       if any(r.verdict in {"HARD", "SOFT"} for r in found.values())}
        if expected == ["silent"]:
            detected = False
        else:
            detected = any(detected_by(results, rule, affected) for rule in expected)
        self.defects[kind][severity].append((tuple(expected), fired_rules, detected))

    def add_control(self, kind: str, expected: list[str], results: Results) -> None:
        allow_v7 = "V7" in expected
        any_hard = any(r.verdict == "HARD" for found in results.values() for r in found.values())
        unexpected = bool(flags(results, skip_v7_soft=allow_v7))
        fired_rules = {rule for rule, _, _ in flags(results)}
        self.controls[kind].append((any_hard, unexpected, fired_rules))


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}% ({part}/{whole})" if whole else "n/a"


def render_markdown(clean: CleanReport, adv: AdversarialReport, *, digest: str, settings: str) -> str:
    lines = ["# Verifier tests (Step 9)", "",
             f"Verifier hash: `{digest}`  ", f"Settings: {settings}", "",
             "Generated by `scripts/verifier_tests.py`. Synthetic and adversarial results are",
             "reported separately and never pooled with real documents.", "",
             "## 1. Clean synthetic gold", "",
             f"Documents: {clean.documents} ("
             + ", ".join(f"{fam} {n}" for fam, n in sorted(clean.by_family.items())) + ")  ",
             f"Pass rate (no HARD from {', '.join(HARD_RULES)}): **{100 * clean.pass_rate:.1f}%**  ",
             f"Documents with any SOFT: {_pct(clean.soft_docs, clean.documents)}", ""]
    if clean.hard:
        lines += ["| doc | rule | field | detail |", "| --- | --- | --- | --- |"]
        lines += [f"| {d} | {r} | {f} | {x} |" for d, r, f, x in clean.hard[:50]]
        lines.append("")
    lines += ["## 2. Recall per defect type", "",
              "Detected = an expected rule reports HARD or SOFT on an affected field. For",
              "defects predicted `silent`, recall is not defined; the last column shows how",
              "often each rule fired anyway (any field).", "",
              "| defect | expected | recall sev 1 | recall sev 2 | recall all | rules that fired |",
              "| --- | --- | --- | --- | --- | --- |"]
    for kind in sorted(adv.defects):
        by_sev = adv.defects[kind]
        rows = [row for sev in sorted(by_sev) for row in by_sev[sev]]
        expected = ", ".join(rows[0][0])
        if rows[0][0] == ("silent",):
            cells = ["n/a"] * 3
        else:
            cells = [_pct(sum(r[2] for r in by_sev.get(s, [])), len(by_sev.get(s, []))) for s in (1, 2)]
            cells.append(_pct(sum(r[2] for r in rows), len(rows)))
        rates = defaultdict(int)
        for row in rows:
            for rule in row[1]:
                rates[rule] += 1
        fired_text = ", ".join(f"{rule} {100 * rates[rule] / len(rows):.0f}%" for rule in RULES if rates[rule]) or "none"
        lines.append(f"| {kind} | {expected} | " + " | ".join(cells) + f" | {fired_text} |")
    lines += ["", "## 3. Unusual-but-correct controls", "",
              "False alarm = any HARD. Unexpected flag = any HARD or SOFT other than the V7",
              "SOFT that a control with unusual conventions is expected to raise.", "",
              "| control | false alarm (HARD) | unexpected flag | rules that fired |",
              "| --- | --- | --- | --- |"]
    total = hard_total = unexpected_total = 0
    for kind in sorted(adv.controls):
        rows = adv.controls[kind]
        hard = sum(r[0] for r in rows)
        unexpected = sum(r[1] for r in rows)
        total, hard_total, unexpected_total = total + len(rows), hard_total + hard, unexpected_total + unexpected
        rates = defaultdict(int)
        for row in rows:
            for rule in row[2]:
                rates[rule] += 1
        fired_text = ", ".join(f"{rule} {100 * rates[rule] / len(rows):.0f}%" for rule in RULES if rates[rule]) or "none"
        lines.append(f"| {kind} | {_pct(hard, len(rows))} | {_pct(unexpected, len(rows))} | {fired_text} |")
    lines.append(f"| **all** | **{_pct(hard_total, total)}** | **{_pct(unexpected_total, total)}** | |")
    lines += ["", "## 4. Freeze", "",
              "Domain-lead review of every rule, threshold and `trustswap/ref/conventions.yaml`",
              "is required before tagging `verifier-v1.0`. After the tag, rules change only for",
              "logged bug fixes (`prereg/verifier_changes.md`).", ""]
    return "\n".join(lines)
