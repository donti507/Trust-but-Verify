"""Offline EDGAR scoping tests. Every HTTP call uses a fixture transport."""
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from scripts import edgar_scope as edgar

FIXTURES = Path(__file__).parent / "fixtures/edgar"


def payload(name):
    return (FIXTURES / name).read_bytes()


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, delay):
        assert delay >= 0
        self.sleeps.append(delay)
        self.now += delay


def client(transport, clock=None, **options):
    clock = clock or Clock()
    return edgar.SECClient("Scope tests test@example.invalid", transport=transport,
                           clock=clock.clock, sleep=clock.sleep, wall_clock=lambda: 0, **options)


def fixture_transport(calls):
    def transport(url, headers):
        calls.append((url, headers))
        query = parse_qs(urlparse(url).query)
        phrase = query["q"][0]
        if phrase == f'"{edgar.PHRASES[0]}"':
            name = "page_1.json" if query["from"] == ["0"] else "page_2.json"
        elif phrase == f'"{edgar.PHRASES[1]}"':
            name = "overlap.json"
        else:
            name = "empty.json"
        return edgar.Response(200, {}, payload(name))
    return transport


@pytest.fixture(autouse=True)
def no_live_http(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("live HTTP is forbidden in tests")
    monkeypatch.setattr(edgar, "urlopen", blocked)


def test_pagination_phrase_dedup_and_request_scope():
    calls = []
    candidates = edgar.discover(client(fixture_transport(calls)), end_date=date(2026, 11, 3), page_size=2)
    assert len(candidates) == 3
    assert candidates[0].matched_phrases == set(edgar.PHRASES[:2])
    assert candidates[1].matched_phrases == {edgar.PHRASES[0]}
    assert [parse_qs(urlparse(url).query)["from"][0] for url, _ in calls] == ["0", "2", "0", "0", "0"]
    assert candidates[0].document_url == "https://www.sec.gov/Archives/edgar/data/123456/000012345620000001/swap-ex101.htm"
    assert candidates[0].company == "Example Issuer, Inc."
    assert candidates[0].cik == "0000123456"
    for url, headers in calls:
        query = parse_qs(urlparse(url).query)
        assert url.startswith(edgar.ENDPOINT)
        assert query["forms"] == ["8-K,10-Q,10-K"]
        assert query["startdt"] == ["2001-01-01"]
        assert query["enddt"] == ["2026-11-03"]
        assert headers["User-Agent"] == "Scope tests test@example.invalid"


def test_csv_and_summary(tmp_path, capsys):
    candidates = edgar.discover(client(fixture_transport([])))
    output = tmp_path / "real/candidates.csv"
    edgar.write_candidates(candidates, output)
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert set(rows[0]) == set(edgar.COLUMNS)
    assert rows[0]["matched_phrases"] == "; ".join(sorted(edgar.PHRASES[:2]))
    edgar.print_summary(candidates)
    summary = capsys.readouterr().out
    assert "Unique candidate documents: 3" in summary
    assert f"{edgar.PHRASES[0]}: 3" in summary
    assert f"{edgar.PHRASES[1]}: 1" in summary
    for label in ("2020", "2021", "2022", "8-K", "10-Q", "10-K"):
        assert f"{label}: 1" in summary


def test_rate_limit_includes_retries_and_downloads():
    clock, starts = Clock(), []
    responses = [edgar.Response(429, {"Retry-After": "2"}, b""),
                 edgar.Response(503, {}, b""), edgar.Response(200, {}, b"ok")]
    def transport(url, headers):
        starts.append(clock.now)
        return responses.pop(0) if responses else edgar.Response(200, {}, b"ok")
    sec = client(transport, clock)
    assert sec.get(edgar.ENDPOINT) == b"ok"
    assert sec.get("https://www.sec.gov/Archives/example.txt") == b"ok"
    assert starts == [0, 2, 4, 4.25]
    assert all(right - left >= 0.25 for left, right in zip(starts, starts[1:]))


def test_retry_after_http_date_and_retry_exhaustion():
    clock = Clock()
    replies = iter([edgar.Response(503, {"Retry-After": "Thu, 01 Jan 1970 00:00:10 GMT"}, b""),
                    edgar.Response(200, {}, b"ok")])
    assert client(lambda *args: next(replies), clock).get(edgar.ENDPOINT) == b"ok"
    assert clock.now == 10
    calls = []
    def unavailable(*args):
        calls.append(1)
        return edgar.Response(429, {}, b"")
    with pytest.raises(RuntimeError, match="HTTP 429"):
        client(unavailable, max_retries=2).get(edgar.ENDPOINT)
    assert len(calls) == 3


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_user_agent_refuses_before_http(monkeypatch, value, tmp_path, capsys):
    if value is None:
        monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    else:
        monkeypatch.setenv("SEC_USER_AGENT", value)
    with pytest.raises(SystemExit) as error:
        edgar.main(["--output", str(tmp_path / "candidates.csv")])
    assert error.value.code == 2
    assert "SEC_USER_AGENT" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_environment_user_agent(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Researcher researcher@example.invalid")
    calls = []
    sec = edgar.SECClient(transport=fixture_transport(calls))
    sec.search(edgar.PHRASES[2], date(2001, 1, 1), date(2026, 1, 1), 0, 100)
    assert calls[0][1]["User-Agent"] == "Researcher researcher@example.invalid"


def test_result_window_is_partitioned_without_overlapping_dates():
    calls = []
    def transport(url, headers):
        query = parse_qs(urlparse(url).query)
        period = (query["startdt"][0], query["enddt"][0])
        calls.append(period)
        name = "capped.json" if period == ("2020-01-01", "2020-01-02") else "overlap.json"
        return edgar.Response(200, {}, payload(name))
    hits = list(edgar.iter_hits(client(transport), "phrase", date(2020, 1, 1), date(2020, 1, 2)))
    assert len(hits) == 2
    assert calls == [("2020-01-01", "2020-01-02"), ("2020-01-01", "2020-01-01"), ("2020-01-02", "2020-01-02")]
    with pytest.raises(RuntimeError, match="cannot guarantee"):
        list(edgar.iter_hits(client(lambda *args: edgar.Response(200, {}, payload("capped.json"))),
                             "phrase", date(2020, 1, 1), date(2020, 1, 1)))


def test_repeated_and_partial_pages_fail_loudly():
    with pytest.raises(RuntimeError, match="repeated a page"):
        list(edgar.iter_hits(client(lambda *args: edgar.Response(200, {}, payload("page_1.json"))),
                             "phrase", date(2001, 1, 1), date(2026, 1, 1), page_size=2))
    body = {"timed_out": True, "hits": {"total": 0, "hits": []}}
    with pytest.raises(RuntimeError, match="partial"):
        list(edgar.iter_hits(client(lambda *args: edgar.Response(200, {}, json.dumps(body).encode())),
                             "phrase", date(2001, 1, 1), date(2026, 1, 1)))


def test_download_manifest_resume_and_never_overwrite(tmp_path):
    candidates = edgar.discover(client(fixture_transport([])))
    calls = []
    content = b"Original filing bytes \x96 remain unchanged"
    def transport(url, headers):
        calls.append(url)
        return edgar.Response(200, {}, content)
    sec = client(transport)
    root = tmp_path / "raw/edgar"
    edgar.download_candidates(sec, candidates, 2, root)
    entries = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    assert len(entries) == len(calls) == 2
    for entry in entries:
        assert Path(entry["path"]).read_bytes() == content
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()
    edgar.download_candidates(sec, candidates, 2, root)
    assert len(calls) == 2
    assert len((root / "manifest.jsonl").read_text().splitlines()) == 2
    Path(entries[0]["path"]).write_bytes(b"changed existing file")
    with pytest.raises(ValueError, match="refusing overwrite"):
        edgar.download_candidates(sec, candidates, 1, root)
    assert Path(entries[0]["path"]).read_bytes() == b"changed existing file"


def test_unmanifested_file_is_never_overwritten(tmp_path):
    candidate = edgar.candidate_from_hit(json.loads(payload("overlap.json"))["hits"]["hits"][0])
    target = tmp_path / f"{candidate.cik}_{candidate.accession_number}_{candidate.exhibit_file_name}"
    target.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        edgar.download_candidates(client(lambda *args: pytest.fail("download must not start")), [candidate], 1, tmp_path)
    assert target.read_bytes() == b"keep"


def test_main_mocked_end_to_end(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.invalid")
    monkeypatch.setattr(edgar, "_http_get", fixture_transport([]))
    clock = Clock()
    constructor = edgar.SECClient
    monkeypatch.setattr(edgar, "SECClient", lambda: constructor(clock=clock.clock, sleep=clock.sleep))
    assert edgar.main(["--output", str(tmp_path / "candidates.csv")]) == 0
    assert (tmp_path / "candidates.csv").exists()
    assert "Unique candidate documents: 3" in capsys.readouterr().out
