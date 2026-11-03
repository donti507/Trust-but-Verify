#!/usr/bin/env python3
"""Scope EDGAR swap confirmations. Network access occurs only when explicitly run.

Uses only the standard library. SEC fair-access guidance requires a descriptive
SEC_USER_AGENT identifying the operator and contact email. No classification,
labelling, split assignment or real_test access is performed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

ENDPOINT = "https://efts.sec.gov/LATEST/search-index"
PHRASES = (
    "Floating Rate Day Count Fraction", "Fixed Rate Day Count Fraction",
    "Interest Rate Swap Transaction", "Floating Rate Option",
)
FORMS = ("8-K", "10-Q", "10-K")
RESULT_WINDOW = 10_000
COLUMNS = ("accession_number", "cik", "company", "form", "filing_date",
           "exhibit_file_name", "document_url", "matched_phrases")


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict[str, str]
    body: bytes


def _http_get(url: str, headers: dict[str, str]) -> Response:
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=60) as response:
            return Response(response.status, dict(response.headers), response.read())
    except HTTPError as error:
        return Response(error.code, dict(error.headers), error.read())


class SECClient:
    """One shared, paced request stream for searches, retries and downloads."""
    def __init__(self, user_agent: str | None = None, *, transport=None,
                 clock=time.monotonic, sleep=time.sleep, wall_clock=time.time,
                 max_retries: int = 5):
        user_agent = user_agent if user_agent is not None else os.environ.get("SEC_USER_AGENT", "")
        if not user_agent.strip():
            raise ValueError("SEC_USER_AGENT must be set to an operator name and contact email")
        self.headers = {"User-Agent": user_agent, "Accept": "application/json, text/html;q=0.9, */*;q=0.8"}
        self.transport = transport or _http_get
        self.clock, self.sleep, self.wall_clock = clock, sleep, wall_clock
        self.max_retries = max_retries
        self.last_request = None

    def get(self, url: str) -> bytes:
        if urlparse(url).scheme != "https" or urlparse(url).hostname not in {"efts.sec.gov", "www.sec.gov"}:
            raise ValueError("only official SEC HTTPS destinations are allowed")
        for attempt in range(self.max_retries + 1):
            if self.last_request is not None:
                remaining = 0.25 - (self.clock() - self.last_request)
                if remaining > 0:
                    self.sleep(remaining)
            self.last_request = self.clock()
            response = self.transport(url, dict(self.headers))
            if response.status == 200:
                return response.body
            if response.status not in {429, 503} or attempt == self.max_retries:
                raise RuntimeError(f"SEC returned HTTP {response.status}; request failed: {url}")
            retry_after = next((value for key, value in response.headers.items() if key.lower() == "retry-after"), "")
            delay = min(60, 2 ** attempt)
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                try:
                    delay = max(delay, parsedate_to_datetime(retry_after).timestamp() - self.wall_clock())
                except (ValueError, TypeError, OverflowError):
                    pass
            self.sleep(delay)
        raise AssertionError("unreachable retry state")

    def search(self, phrase: str, start: date, end: date, offset: int, size: int) -> dict:
        parameters = {"q": f'"{phrase}"', "forms": ",".join(FORMS), "dateRange": "custom",
                      "startdt": start.isoformat(), "enddt": end.isoformat(), "from": offset, "size": size}
        return json.loads(self.get(ENDPOINT + "?" + urlencode(parameters)))


def _page(payload):
    if payload.get("timed_out") or payload.get("_shards", {}).get("failed", 0):
        raise RuntimeError("EDGAR search returned partial results")
    hits = payload["hits"]
    total = hits["total"]
    relation = total.get("relation", "eq") if isinstance(total, dict) else "eq"
    count = total["value"] if isinstance(total, dict) else total
    if not isinstance(count, int) or count < 0 or relation not in {"eq", "gte"}:
        raise ValueError("invalid EDGAR hit total")
    return count, relation, hits["hits"]


def iter_hits(client: SECClient, phrase: str, start: date, end: date, *, page_size=100):
    """Page fully, recursively subdividing inclusive dates past the result cap."""
    if not 1 <= page_size <= 100 or start > end:
        raise ValueError("invalid date range or page size")
    payload = client.search(phrase, start, end, 0, page_size)
    total, relation, hits = _page(payload)
    if total > RESULT_WINDOW or relation == "gte":
        if start == end:
            raise RuntimeError(f"EDGAR result cap on {start} for {phrase!r}; cannot guarantee complete retrieval")
        middle = start + (end - start) // 2
        yield from iter_hits(client, phrase, start, middle, page_size=page_size)
        yield from iter_hits(client, phrase, middle + timedelta(days=1), end, page_size=page_size)
        return
    offset = 0
    seen_pages = set()
    while offset < total:
        if not hits:
            raise RuntimeError("EDGAR pagination ended before the reported total")
        signature = tuple(hit["_id"] for hit in hits)
        if signature in seen_pages:
            raise RuntimeError("EDGAR repeated a page; refusing incomplete retrieval")
        seen_pages.add(signature)
        yield from hits
        offset += len(hits)
        if offset >= total:
            break
        payload = client.search(phrase, start, end, offset, page_size)
        new_total, new_relation, hits = _page(payload)
        if new_total != total or new_relation != "eq":
            raise RuntimeError("EDGAR results changed during pagination; rerun to obtain a complete search")


@dataclass
class Candidate:
    accession_number: str
    cik: str
    company: str
    form: str
    filing_date: str
    exhibit_file_name: str
    document_url: str
    matched_phrases: set[str] = field(default_factory=set)

    def row(self):
        return {**vars(self), "matched_phrases": "; ".join(sorted(self.matched_phrases))}


def candidate_from_hit(hit: dict) -> Candidate:
    source = hit["_source"]
    accession, separator, filename = hit["_id"].partition(":")
    accession = source.get("adsh", accession)
    filename = source.get("file_name", filename)
    ciks = source["ciks"]
    names = source["display_names"]
    roots = source.get("root_forms", [source.get("form", source.get("file_type"))])
    if isinstance(roots, str):
        roots = [roots]
    form = next((value for value in roots if value in FORMS), None)
    if not form:
        raise ValueError("EDGAR hit is outside the requested root filing forms")
    cik = str(ciks[0])
    if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession) or not re.fullmatch(r"\d{1,10}", cik):
        raise ValueError("invalid EDGAR accession or CIK")
    if not separator or not filename or Path(filename).name != filename or filename in {".", ".."} or "\\" in filename:
        raise ValueError("invalid EDGAR document filename")
    filing_date = date.fromisoformat(source["file_date"]).isoformat()
    company = re.sub(r"\s*\(CIK\s+\d+\)$", "", names[0]).strip()
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{quote(filename, safe='')}"
    return Candidate(accession, cik.zfill(10), company, form, filing_date, filename, url)


def discover(client: SECClient, *, end_date: date | None = None, phrases=PHRASES, page_size=100) -> list[Candidate]:
    documents = {}
    for phrase in phrases:
        for hit in iter_hits(client, phrase, date(2001, 1, 1), end_date or date.today(), page_size=page_size):
            candidate = candidate_from_hit(hit)
            existing = documents.setdefault(candidate.document_url, candidate)
            existing.matched_phrases.add(phrase)
    return sorted(documents.values(), key=lambda item: (item.filing_date, item.document_url))


def write_candidates(candidates: list[Candidate], path: str | Path):
    """Replace a completed candidate CSV atomically; never write partial searches."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=".candidates-", delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(candidate.row() for candidate in candidates)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def print_summary(candidates: list[Candidate]):
    print(f"Unique candidate documents: {len(candidates)}")
    print("Count per phrase:")
    for phrase in PHRASES:
        print(f"  {phrase}: {sum(phrase in item.matched_phrases for item in candidates)}")
    for label, counts in (("year", Counter(item.filing_date[:4] for item in candidates)),
                          ("form", Counter(item.form for item in candidates))):
        print(f"Count per {label}:")
        for name, count in sorted(counts.items()):
            print(f"  {name}: {count}")


def download_candidates(client: SECClient, candidates: list[Candidate], count: int,
                        raw_dir: str | Path = "data/raw/edgar"):
    if count < 0:
        raise ValueError("download count must be nonnegative")
    if not count:
        return
    root = Path(raw_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    existing = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry["document_url"] in existing:
                raise ValueError("duplicate URL in download manifest")
            existing[entry["document_url"]] = entry
    for candidate in candidates[:count]:
        target = root / f"{candidate.cik}_{candidate.accession_number}_{candidate.exhibit_file_name}"
        if candidate.document_url in existing:
            entry = existing[candidate.document_url]
            if str(target) != entry["path"] or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("existing download does not match its manifest; refusing overwrite")
            continue
        if target.exists():
            raise FileExistsError(f"unmanifested raw file will not be overwritten: {target}")
        body = client.get(candidate.document_url)
        with target.open("xb") as handle:
            handle.write(body)
        entry = {"document_url": candidate.document_url, "accession_number": candidate.accession_number,
                 "cik": candidate.cik, "path": str(target), "sha256": hashlib.sha256(body).hexdigest()}
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        existing[candidate.document_url] = entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", type=int, default=0, metavar="N", help="download first N candidates (default: none)")
    parser.add_argument("--output", type=Path, default=Path("data/real/candidates.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/edgar"))
    arguments = parser.parse_args(argv)
    if arguments.download < 0:
        parser.error("--download must be nonnegative")
    try:
        client = SECClient()
    except ValueError as error:
        parser.error(str(error))
    candidates = discover(client)
    write_candidates(candidates, arguments.output)
    print_summary(candidates)
    download_candidates(client, candidates, arguments.download, arguments.raw_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
