"""EDGAR screening with fictional fixtures and a strictly mocked HTTP layer."""
from collections import Counter
import csv
from datetime import date
import hashlib
import json
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from scripts import edgar_scope, edgar_screen as screen

FIXTURES = Path(__file__).parent / "fixtures/edgar_screen"
AS_OF = date(2026, 11, 3)
YEARS = (2003, 2007, 2015, 2024)


class Clock:
    def __init__(self):
        self.now = 0.0
    def clock(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


def candidate(index, year, filename="likely.html"):
    accession = f"0000123456-{year % 100:02d}-{index:06d}"
    return edgar_scope.candidate_from_hit({
        "_id": f"{accession}:{filename}",
        "_source": {"ciks": ["0000123456"], "display_names": ["Fictional Issuer"],
                    "root_forms": ["8-K"], "file_date": f"{year}-03-01"},
    })


def small_candidates():
    return [candidate(era * 3 + label, year, filename)
            for era, year in enumerate(YEARS)
            for label, filename in enumerate(("likely.html", "maybe.txt", "unlikely.txt"))]


def fixture_client(calls, starts=None):
    clock = Clock()
    def transport(url, headers):
        calls.append((url, headers))
        if starts is not None:
            starts.append(clock.now)
        return edgar_scope.Response(200, {}, (FIXTURES / url.rsplit("/", 1)[-1]).read_bytes())
    return edgar_scope.SECClient("Screen tests test@example.invalid", transport=transport,
                                 clock=clock.clock, sleep=clock.sleep)


@pytest.fixture(autouse=True)
def no_live_http(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("live HTTP forbidden")
    monkeypatch.setattr(edgar_scope, "urlopen", forbidden)


def test_stratified_300_fixed_seed_and_recent_quota():
    candidates = [candidate(era * 150 + index, year) for era, year in enumerate(YEARS) for index in range(150)]
    first = screen.stratified_sample(candidates, seed=507, as_of=AS_OF)
    again = screen.stratified_sample(list(reversed(candidates)), seed=507, as_of=AS_OF)
    assert first == again
    assert len(first) == len({item.document_url for item in first}) == 300
    assert Counter(screen.era_for(item.filing_date, as_of=AS_OF) for item in first) == screen.ERA_QUOTAS
    assert screen.stratified_sample(candidates, seed=508, as_of=AS_OF) != first


@pytest.mark.parametrize("value,expected", [
    ("2001-01-01", "2001–2005"), ("2005-12-31", "2001–2005"),
    ("2006-01-01", "2006–2008"), ("2008-12-31", "2006–2008"),
    ("2009-01-01", "2009–2021"), ("2021-12-31", "2009–2021"),
    ("2022-01-01", "2022+"), ("2000-12-31", None), ("2027-01-01", None),
])
def test_era_boundaries(value, expected):
    assert screen.era_for(value, as_of=AS_OF) == expected


def test_insufficient_strata_and_duplicate_urls():
    with pytest.raises(ValueError, match="need 75"):
        screen.stratified_sample(small_candidates() * 100, as_of=AS_OF)


def test_read_candidates_validates_and_deduplicates(tmp_path):
    path = tmp_path / "candidates.csv"
    items = small_candidates()
    edgar_scope.write_candidates(items + [items[0]], path)
    assert len(screen.read_candidates(path)) == 12
    rows = list(csv.DictReader(path.open()))
    rows[0]["document_url"] = "https://example.invalid/not-sec"
    screen.write_csv(path, rows, edgar_scope.COLUMNS)
    with pytest.raises(ValueError, match="URL"):
        screen.read_candidates(path)


@pytest.mark.parametrize("text,pages,label,score", [
    ("Notional Amount Termination Date Effective Date Fixed Rate", 1, "likely_confirmation", 5),
    ("Calculation Amount Termination Date Effective Date Floating Rate Option", 39, "likely_confirmation", 5),
    ("Notional\nAmount Termination Date Effective Date Fixed Rate", 40, "maybe", 4),
    ("calculation amount effective date", 2, "maybe", 3),
    ("Notional Amount", 1, "unlikely", 2), ("nothing", 1, "unlikely", 1),
])
def test_scoring_and_page_boundary(text, pages, label, score):
    result = screen.score_document(text, pages)
    assert result["label"] == label and result["score"] == score
    assert result["under_40_pages"] == (pages < 40)


def test_native_pdf_pages_and_html_estimate(tmp_path):
    for pages in (39, 40):
        path = tmp_path / f"{pages}.pdf"
        pdf = canvas.Canvas(str(path))
        for index in range(pages):
            pdf.drawString(72, 720, f"Fictional page {index + 1}")
            pdf.showPage()
        pdf.save()
        assert screen.count_pages(path, "irrelevant") == (pages, "pdf")
    pages, basis = screen.count_pages(FIXTURES / "likely.html", "A short <img src='https://example.invalid'> text")
    assert pages == 1 and basis == "estimated_text_layout"
    long_pages, _ = screen.count_pages(FIXTURES / "maybe.txt", "Long document line\n" * 4000)
    assert long_pages >= 40


def test_mock_download_ingest_score_and_resume(tmp_path):
    calls, starts = [], []
    client = fixture_client(calls, starts)
    candidates = small_candidates()
    rows = screen.screen_candidates(client, candidates, data_root=tmp_path, as_of=AS_OF)
    assert Counter(row["label"] for row in rows) == {label: 4 for label in screen.LABELS}
    assert Counter(row["era"] for row in rows) == {era: 3 for era in screen.ERA_QUOTAS}
    assert all(right - left >= 0.25 for left, right in zip(starts, starts[1:]))
    assert all(headers["User-Agent"] == "Screen tests test@example.invalid" for _, headers in calls)
    assert len(calls) == 12
    raw_before = {str(path): path.read_bytes() for path in (tmp_path / "raw/edgar").iterdir()}
    for row in rows:
        assert row["sha256"] == hashlib.sha256(Path(row["raw_path"]).read_bytes()).hexdigest()
        assert Path(row["text_path"]).is_file()
        assert Path(row["text_path"]).with_suffix(".offsets.json").is_file()
    records = [json.loads(line) for line in (tmp_path / "real/docs.jsonl").read_text().splitlines()]
    assert len(records) == 12
    assert all(record["collection"] == "real" and record["split"] is None and record["family_id"] is None for record in records)
    again = screen.screen_candidates(client, candidates, data_root=tmp_path, as_of=AS_OF)
    assert again == rows and len(calls) == 12
    assert {str(path): path.read_bytes() for path in (tmp_path / "raw/edgar").iterdir()} == raw_before
    assert len((tmp_path / "real/docs.jsonl").read_text().splitlines()) == 12


def test_manual_60_balanced_unique_deterministic_and_blank():
    rows = [{"document_url": f"https://example.invalid/{label}/{i}", "label": label} for label in screen.LABELS for i in range(100)]
    selected = screen.manual_sample(rows, seed=507)
    assert len(selected) == len({row["document_url"] for row in selected}) == 60
    assert Counter(row["label"] for row in selected) == {label: 20 for label in screen.LABELS}
    assert all(row["is_confirmation"] == "" for row in selected)
    assert selected == screen.manual_sample(list(reversed(rows)), seed=507)
    assert all("is_confirmation" not in row for row in rows)
    with pytest.raises(ValueError, match="only 0 available"):
        screen.manual_sample([row for row in rows if row["label"] != "maybe"])


def test_main_mocked_csv_outputs_and_protected_manual_review(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SEC_USER_AGENT", "Screen tests test@example.invalid")
    candidates = small_candidates()
    edgar_scope.write_candidates(candidates, tmp_path / "real/candidates.csv")
    calls = []
    mocked_client = fixture_client(calls)
    monkeypatch.setattr(edgar_scope, "SECClient", lambda: mocked_client)
    # Use a small fixture sample; production defaults (300 and 60) are tested above.
    original_sample, original_manual = screen.stratified_sample, screen.manual_sample
    monkeypatch.setattr(screen, "stratified_sample", lambda items, **options: original_sample(items, quotas={era: 3 for era in screen.ERA_QUOTAS}, **options))
    monkeypatch.setattr(screen, "manual_sample", lambda rows, **options: original_manual(rows, per_label=1, **options))
    assert screen.main(["--data-root", str(tmp_path)]) == 0
    summary = capsys.readouterr().out
    assert "Screened documents: 12" in summary
    assert all(f"{label}: 4" in summary for label in screen.LABELS)
    with (tmp_path / "real/screen.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 12
    manual_path = tmp_path / "real/manual_check.csv"
    with manual_path.open() as handle:
        manual = list(csv.DictReader(handle))
    assert len(manual) == 3 and all(row["is_confirmation"] == "" for row in manual)
    saved = manual_path.read_bytes()
    with pytest.raises(SystemExit):
        screen.main(["--data-root", str(tmp_path)])
    assert manual_path.read_bytes() == saved
    assert len(calls) == 12


def test_missing_user_agent_refuses_without_outputs(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(SystemExit) as error:
        screen.main(["--data-root", str(tmp_path)])
    assert error.value.code == 2
    assert "SEC_USER_AGENT" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_manual_shortage_keeps_screen_results(tmp_path, monkeypatch, capsys):
    edgar_scope.write_candidates(small_candidates(), tmp_path / "real/candidates.csv")
    mocked_client = fixture_client([])
    monkeypatch.setattr(edgar_scope, "SECClient", lambda: mocked_client)
    original = screen.stratified_sample
    monkeypatch.setattr(screen, "stratified_sample", lambda items, **options: original(
        items, quotas={era: 3 for era in screen.ERA_QUOTAS}, **options))
    with pytest.raises(SystemExit) as error:
        screen.main(["--data-root", str(tmp_path)])
    assert error.value.code == 2
    assert (tmp_path / "real/screen.csv").is_file()
    assert not (tmp_path / "real/manual_check.csv").exists()
    output = capsys.readouterr()
    assert "Screened documents: 12" in output.out
    assert "manual sample needs 20" in output.err


def test_assigned_document_is_not_rescreened(tmp_path):
    candidates = small_candidates()[:1]
    client = fixture_client([])
    screen.screen_candidates(client, candidates, data_root=tmp_path)
    record_path = tmp_path / "real/docs.jsonl"
    record = json.loads(record_path.read_text())
    record["split"] = "real_cal"  # Fictional metadata, no actual dataset is read.
    record_path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="refusing split access"):
        screen.screen_candidates(client, candidates, data_root=tmp_path)


def test_same_era_topups_produce_300_successes(tmp_path):
    pool = [candidate(era * 150 + index, year) for era, year in enumerate(YEARS) for index in range(150)]
    selected = screen.stratified_sample(pool, seed=507, as_of=AS_OF)
    failed_urls = {item.document_url for item in selected[:20]}
    calls, clock = [], Clock()
    def transport(url, headers):
        calls.append(url)
        return edgar_scope.Response(503, {}, b"unavailable") if url in failed_urls else edgar_scope.Response(200, {}, b"ok")
    client = edgar_scope.SECClient("Tests test@example.invalid", transport=transport,
                                   clock=clock.clock, sleep=clock.sleep, max_retries=0)
    successful, report = screen.download_sample(client, selected, pool, seed=507, data_root=tmp_path, as_of=AS_OF)
    assert len(successful) == len({item.document_url for item in successful}) == 300
    assert Counter(screen.era_for(item.filing_date, as_of=AS_OF) for item in successful) == screen.ERA_QUOTAS
    assert len(report.successes) == 300 and len(report.failures) == 20
    assert failed_urls.isdisjoint(item.document_url for item in successful)
    assert len(calls) == 320
    manifest = tmp_path / "raw/edgar/manifest.jsonl"
    before = manifest.read_bytes()
    assert len(before.splitlines()) == 300
    again, _ = screen.download_sample(client, selected, list(reversed(pool)), seed=507, data_root=tmp_path, as_of=AS_OF)
    assert again == successful and manifest.read_bytes() == before
    assert len(calls) == 340  # Only previously unsuccessful URLs need new requests.


def test_full_submission_pdf_candidate_is_ingested_as_selected_sgml(tmp_path):
    item = candidate(1, 2024, "missing.pdf")
    item.matched_phrases.add(edgar_scope.PHRASES[0])
    full = (Path(__file__).parent / "fixtures/edgar/full_submission.txt").read_bytes()
    clock = Clock()
    def transport(url, headers):
        return edgar_scope.Response(404, {}, b"missing") if url == item.document_url else edgar_scope.Response(200, {}, full)
    client = edgar_scope.SECClient("Tests test@example.invalid", transport=transport,
                                   clock=clock.clock, sleep=clock.sleep)
    rows = screen.screen_candidates(client, [item], data_root=tmp_path, as_of=AS_OF)
    assert len(rows) == 1
    row = rows[0]
    assert row["download_source"] == "full_submission"
    assert row["resolved_source_url"] == edgar_scope.full_submission_url(item)
    assert row["selected_filename"] == "swap-ex101.htm"
    assert row["raw_path"].endswith(".full_submission.txt")
    assert row["page_count_basis"] == "estimated_text_layout"
    assert row["label"] == "likely_confirmation"
    assert "Primary filing" not in Path(row["text_path"]).read_text()
    record = json.loads((tmp_path / "real/docs.jsonl").read_text())
    assert record["source_url"] == row["resolved_source_url"]
    assert record["sha256"] == hashlib.sha256(Path(row["raw_path"]).read_bytes()).hexdigest()


def test_exhausted_pool_retains_screen_and_prints_download_counts(tmp_path, monkeypatch, capsys):
    items = small_candidates()
    edgar_scope.write_candidates(items, tmp_path / "real/candidates.csv")
    clock = Clock()
    def transport(url, headers):
        return edgar_scope.Response(403, {}, b"denied") if url == items[0].document_url else edgar_scope.Response(
            200, {}, (FIXTURES / url.rsplit("/", 1)[-1]).read_bytes())
    mocked = edgar_scope.SECClient("Tests test@example.invalid", transport=transport,
                                   clock=clock.clock, sleep=clock.sleep)
    monkeypatch.setattr(edgar_scope, "SECClient", lambda: mocked)
    original = screen.stratified_sample
    monkeypatch.setattr(screen, "stratified_sample", lambda items, **options: original(
        items, quotas={era: 3 for era in screen.ERA_QUOTAS}, **options))
    with pytest.raises(SystemExit) as error:
        screen.main(["--data-root", str(tmp_path)])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert "same-era replacement pool exhausted" in output.err
    assert "Downloaded: 11" in output.out and "Recovered via fallback: 0" in output.out and "Failed: 1" in output.out
    with (tmp_path / "real/screen.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 11
    with (tmp_path / "real/download_failures.csv").open() as handle:
        failures = list(csv.DictReader(handle))
    assert len(failures) == 1 and failures[0]["status"] == "403"
    assert not (tmp_path / "real/manual_check.csv").exists()
