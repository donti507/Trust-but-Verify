"""Pilot rejection, interactive replacement and mocked exhibit resolution."""
import hashlib
import io
import json
from pathlib import Path
import socket

import pytest

from scripts import annotate, agreement, edgar_scope, resolve_exhibit, select_pilot
from scripts.pilot_utils import read_csv, write_csv, read_annotations, rejection_history
from trustswap.common.annotations import AnnotationRecord, validated_field

FIXTURE = Path(__file__).parent / "fixtures/pilot/confirmation.txt"
EXHIBITS = Path(__file__).parent / "fixtures/exhibits"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *args, **kwargs: pytest.fail("live network forbidden"))
    monkeypatch.setattr(edgar_scope, "urlopen", lambda *args, **kwargs: pytest.fail("live HTTP forbidden"))


def inputs(*values):
    iterator = iter(values)
    def ask(prompt):
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError from None
    return ask


def test_reject_marks_pilot_skips_document_and_resumes(tmp_path):
    rows = [{"doc_id": f"fixture_{i}", "document_url": f"fixture://{i}", "text_path": str(FIXTURE), "era": "2001–2005"}
            for i in range(2)]
    pilot, output = tmp_path / "pilot20.csv", tmp_path / "annotations"
    write_csv(pilot, tuple(rows[0]), rows)
    display = io.StringIO()
    annotate.annotate(pilot, annotator="A1", output_dir=output,
                      input_fn=inputs("absent", "y", "reject cover 8-K only", "absent", "y"), output=display)
    saved = read_csv(pilot)[1]
    assert saved[0]["rejected_reason"] == "cover 8-K only" and saved[1]["rejected_reason"] == ""
    assert "select_pilot.py --top-up" in display.getvalue()
    records = read_annotations(output / "A1.jsonl")[0]
    assert set(records["fixture_0"].fields) == {"trade_date"}  # Partial work retained.
    assert set(records["fixture_1"].fields) == {"trade_date"}
    shown = io.StringIO()
    annotate.annotate(pilot, annotator="A2", output_dir=output, input_fn=inputs(), output=shown)
    assert "A2 | fixture_0" not in shown.getvalue() and "A2 | fixture_1" in shown.getvalue()


def test_reject_works_in_nested_confirmation_prompt(tmp_path):
    row = {"doc_id": "fixture_1", "text_path": str(FIXTURE), "document_url": "fixture://1"}
    pilot = tmp_path / "pilot.csv"
    write_csv(pilot, tuple(row), [row])
    annotate.annotate(pilot, annotator="A1", output_dir=tmp_path / "annotations",
                      input_fn=inputs("absent", "reject not a confirmation"), output=io.StringIO())
    assert read_csv(pilot)[1][0]["rejected_reason"] == "not a confirmation"
    assert not (tmp_path / "annotations/A1.jsonl").exists()


def pilot_pool():
    rows = []
    for era_index, era in enumerate(select_pilot.ERAS):
        for i in range(12):
            rows.append({"doc_id": f"fixture_{era_index}_{i}", "document_url": f"fixture://{era_index}/{i}",
                         "era": era, "text_path": str(FIXTURE), "label": "likely_confirmation"})
    return rows


def test_topup_retains_active_rows_excludes_history_and_requires_preview(tmp_path, monkeypatch, capsys):
    pool = pilot_pool()
    screen, manual, pilot = tmp_path / "screen.csv", tmp_path / "manual.csv", tmp_path / "pilot20.csv"
    write_csv(screen, tuple(pool[0]), pool)
    write_csv(manual, ("document_url", "is_confirmation", "product_type"), [])
    initial = select_pilot.select_pilot(pool, [])
    initial[0]["rejected_reason"] = "cover only"
    columns = list(dict.fromkeys([*(key for row in initial for key in row), "rejected_reason"]))
    write_csv(pilot, columns, [{key: row.get(key, "") for key in columns} for row in initial])
    old_active = {row["doc_id"]: {key: row.get(key, "") for key in columns} for row in initial[1:]}
    rejected_id = initial[0]["doc_id"]
    prompts = []
    def accept(prompt):
        prompts.append(prompt)
        return "y"
    monkeypatch.setattr("builtins.input", accept)
    assert select_pilot.main(["--screen", str(screen), "--manual", str(manual), "--output", str(pilot), "--top-up"]) == 0
    new = read_csv(pilot)[1]
    assert len(new) == 20 and len({row["doc_id"] for row in new}) == 20
    assert rejected_id not in {row["doc_id"] for row in new}
    for row in new:
        if row["doc_id"] in old_active:
            assert {key: row[key] for key in columns} == old_active[row["doc_id"]]
    additions = [row for row in new if row["doc_id"] not in old_active]
    assert len(additions) == len(prompts) == 1
    assert additions[0]["selection_reason"] == "interactive_confirmation"
    assert additions[0]["is_confirmation"] == "y"
    assert "First 40 lines" in capsys.readouterr().out
    assert rejection_history(pilot)[0]["doc_id"] == rejected_id
    before = pilot.read_bytes()
    assert select_pilot.main(["--screen", str(screen), "--manual", str(manual), "--output", str(pilot), "--top-up"]) == 0
    assert pilot.read_bytes() == before


def test_declined_preview_not_selected_and_interrupt_does_not_change_pilot(tmp_path, monkeypatch):
    pool = pilot_pool()
    declined = []
    def confirm(row):
        if not declined:
            declined.append(row["doc_id"])
            return False
        return True
    result = select_pilot.select_pilot(pool, [], confirm=confirm)
    assert declined[0] not in {row["doc_id"] for row in result}
    screen, manual, pilot = tmp_path / "screen.csv", tmp_path / "manual.csv", tmp_path / "pilot.csv"
    write_csv(screen, tuple(pool[0]), pool)
    write_csv(manual, ("document_url",), [])
    monkeypatch.setattr("builtins.input", inputs())
    assert select_pilot.main(["--screen", str(screen), "--manual", str(manual), "--output", str(pilot)]) == 130
    assert not pilot.exists()


class Clock:
    def __init__(self):
        self.now = 0.0
    def clock(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def cover(tmp_path):
    source = tmp_path / "cover.txt"
    source.write_bytes((EXHIBITS / "cover.txt").read_bytes())
    directory = "https://www.sec.gov/Archives/edgar/data/123456/000012345624000001/"
    row = {"doc_id": "cover_1", "document_url": directory + "cover.htm", "text_path": str(source),
           "era": "2022+", "form": "8-K", "cik": "0000123456", "accession_number": "0000123456-24-000001",
           "filing_date": "2024-03-01", "company": "Fictional Issuer", "label": "likely_confirmation"}
    pilot = tmp_path / "data/real/pilot20.csv"
    write_csv(pilot, tuple(row), [row])
    return pilot, row, directory, tmp_path / "data"


def test_exhibit_link_matching_description_filename_number_and_safe_urls(cover):
    _, _, directory, _ = cover
    text = (EXHIBITS / "cover.txt").read_text()
    found = resolve_exhibit.find_exhibits((EXHIBITS / "index.html").read_bytes(), directory, text)
    assert [item.name for item in found] == ["d123dex101.htm"]
    name_only = resolve_exhibit.find_exhibits(b'<a href="swap-confirmation.pdf">PDF</a>', directory, text)
    assert name_only[0].name == "swap-confirmation.pdf"
    number_only = resolve_exhibit.find_exhibits(b'<a href="d987ex101.htm">document</a>', directory, text)
    assert number_only[0].name == "d987ex101.htm"
    assert not resolve_exhibit.find_exhibits(b'<a href="d987ex1012.htm">not the cited exhibit</a>', directory, text)


def test_resolve_download_ingest_manifest_and_replace_only_after_acceptance(cover):
    pilot, row, directory, root = cover
    clock, calls, starts = Clock(), [], []
    def transport(url, headers):
        calls.append((url, headers))
        starts.append(clock.now)
        if url == directory:
            return edgar_scope.Response(200, {}, (EXHIBITS / "index.html").read_bytes())
        assert url == directory + "d123dex101.htm"
        return edgar_scope.Response(200, {}, FIXTURE.read_bytes().replace(b"\n", b"<br>"))
    client = edgar_scope.SECClient("Reviewer reviewer@example.invalid", transport=transport,
                                   clock=clock.clock, sleep=clock.sleep)
    replacement = resolve_exhibit.resolve("cover_1", pilot_path=pilot, data_root=root, client=client,
                                          input_fn=inputs("y"), output=io.StringIO())
    saved = read_csv(pilot)[1]
    assert len(saved) == 1 and saved[0]["doc_id"] == replacement["doc_id"] != "cover_1"
    assert saved[0]["replaces_doc_id"] == "cover_1" and saved[0]["rejected_reason"] == ""
    assert saved[0]["is_confirmation"] == "y" and saved[0]["manual_check_required"] == "false"
    assert Path(replacement["text_path"]).exists()
    entry = json.loads((root / "raw/edgar/manifest.jsonl").read_text())
    assert entry["sha256"] == hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
    assert all(headers["User-Agent"] == "Reviewer reviewer@example.invalid" for _, headers in calls)
    assert starts[1] - starts[0] >= 0.25
    assert rejection_history(pilot)[0]["doc_id"] == "cover_1"


def test_resolver_document_index_fallback_and_decline_preserves_cover(cover):
    pilot, row, directory, root = cover
    before, clock, urls = pilot.read_bytes(), Clock(), []
    def transport(url, headers):
        urls.append(url)
        if url == directory:
            return edgar_scope.Response(200, {}, (EXHIBITS / "directory.html").read_bytes())
        if url.endswith("-index.htm"):
            return edgar_scope.Response(200, {}, (EXHIBITS / "index.html").read_bytes())
        return edgar_scope.Response(200, {}, b"<html><p>Fixed Rate: 4%</p></html>")
    client = edgar_scope.SECClient("Test test@example.invalid", transport=transport, clock=clock.clock, sleep=clock.sleep)
    assert resolve_exhibit.resolve("cover_1", pilot_path=pilot, data_root=root, client=client,
                                   input_fn=inputs("n"), output=io.StringIO()) is None
    assert any(url.endswith("0000123456-24-000001-index.htm") for url in urls)
    assert pilot.read_bytes() == before and not rejection_history(pilot)
    assert (root / "raw/edgar/manifest.jsonl").exists()


def test_agreement_excludes_retired_cover_annotations(tmp_path):
    common = {"text_path": str(FIXTURE), "text_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest()}
    left = [AnnotationRecord(doc_id=doc, annotator="A1", **common,
              fields={"trade_date": validated_field("trade_date", None, None, "absent")}) for doc in ("active", "cover")]
    right = AnnotationRecord(doc_id="active", annotator="A2", **common,
                             fields={"trade_date": validated_field("trade_date", None, None, "absent")})
    a, b, pilot = tmp_path / "A1.jsonl", tmp_path / "A2.jsonl", tmp_path / "pilot.csv"
    a.write_text("".join(record.model_dump_json()+"\n" for record in left))
    b.write_text(right.model_dump_json()+"\n")
    write_csv(pilot, ("doc_id", "rejected_reason"), [{"doc_id": "active", "rejected_reason": ""}])
    output, metrics = tmp_path / "adjudication.csv", tmp_path / "agreement.csv"
    agreement.main(["--a1", str(a), "--a2", str(b), "--pilot", str(pilot), "--output", str(output), "--metrics", str(metrics)])
    assert not read_csv(output)[1]


def test_preview_shows_forty_lines_and_later_keyword_matches(tmp_path):
    lines = [f"Ordinary line {i}" for i in range(1, 76)]
    lines[69] = "Notional Amount: USD 100,000,000"
    text = tmp_path / "text.txt"
    text.write_text("\n".join(lines))
    row = {"document_url": "fixture://preview", "text_path": str(text), "era": "2001–2005", "label": "likely_confirmation"}
    shown = io.StringIO()
    assert select_pilot.preview_confirmation(row, tmp_path / "screen.csv", input_fn=inputs("y"), output=shown)
    assert "Ordinary line 40" in shown.getvalue() and "Ordinary line 41" not in shown.getvalue()
    assert "[[Notional]] Amount" in shown.getvalue()


def test_multiple_exhibits_require_human_choice(cover):
    pilot, row, directory, root = cover
    page = ((EXHIBITS / "index.html").read_text().replace("</table>",
        '<tr><td>Confirmation of another swap</td><td><a href="other-confirmation.htm">other-confirmation.htm</a></td></tr></table>')).encode()
    clock, urls = Clock(), []
    def transport(url, headers):
        urls.append(url)
        return edgar_scope.Response(200, {}, page if url == directory else b"<p>Fixed Rate: 4%</p>")
    client = edgar_scope.SECClient("Test test@example.invalid", transport=transport, clock=clock.clock, sleep=clock.sleep)
    replacement = resolve_exhibit.resolve("cover_1", pilot_path=pilot, data_root=root, client=client,
                                          input_fn=inputs("2", "y"), output=io.StringIO())
    assert replacement["document_url"] == directory + "other-confirmation.htm"
    assert directory + "d123dex101.htm" not in urls


def test_resolver_missing_user_agent_does_not_issue_request(cover, monkeypatch):
    pilot, _, _, root = cover
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    before = pilot.read_bytes()
    with pytest.raises(SystemExit) as error:
        resolve_exhibit.main(["cover_1", "--pilot", str(pilot), "--data-root", str(root)])
    assert error.value.code == 2 and pilot.read_bytes() == before
