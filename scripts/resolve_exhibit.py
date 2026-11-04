#!/usr/bin/env python3
"""Resolve a pilot cover filing to a locally ingested confirmation exhibit.

Network requests occur only on explicit invocation, using the shared SEC client.
No annotation file is changed and no trade labels are inferred from index text.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from bs4 import BeautifulSoup

if __package__:
    from . import edgar_scope, edgar_screen
    from .pilot_utils import read_csv_snapshot, write_csv, archive_rejections, rejection_history
    from .review_helper import local_text_path, _safe
    from .select_pilot import preview_confirmation
else:
    import edgar_scope
    import edgar_screen
    from pilot_utils import read_csv_snapshot, write_csv, archive_rejections, rejection_history
    from review_helper import local_text_path, _safe
    from select_pilot import preview_confirmation


@dataclass(frozen=True)
class Exhibit:
    name: str
    url: str
    description: str
    score: int


def cover_identity(row):
    parsed = urlparse(row["document_url"])
    match = re.fullmatch(r"/Archives/edgar/data/(\d+)/(\d{18})/[^/]+", unquote(parsed.path))
    if parsed.scheme != "https" or parsed.hostname != "www.sec.gov" or not match:
        raise ValueError("cover URL must be an SEC archive document")
    cik, compact = match.groups()
    accession = f"{compact[:10]}-{compact[10:12]}-{compact[12:]}"
    if row.get("cik") and int(row["cik"]) != int(cik):
        raise ValueError("cover CIK metadata disagrees with its URL")
    if row.get("accession_number") and row["accession_number"] != accession:
        raise ValueError("cover accession metadata disagrees with its URL")
    if row.get("form") not in edgar_scope.FORMS:
        raise ValueError("cover must belong to an 8-K, 10-Q or 10-K filing")
    return cik.zfill(10), accession


def find_exhibits(index_bytes, index_url, cover_text, *, exclude_urls=()):
    """Description/name matches, or the exhibit number explicitly cited by the cover."""
    directory_path = urlparse(index_url).path
    references = set(re.findall(r"\bExhibit(?:\s+No\.?)?\s+(\d+\.\d+)\b", cover_text, re.I))
    # Some directory endpoints return the SEC directory JSON instead of HTML.
    if index_bytes.lstrip().startswith(b"{"):
        payload = json.loads(index_bytes)
        markup = "".join(f'<a href="{item["name"]}">{item["name"]}</a>' for item in payload["directory"]["item"])
        soup = BeautifulSoup(markup, "html.parser")
    else:
        soup = BeautifulSoup(index_bytes, "html.parser")
    matches = {}
    for link in soup.find_all("a", href=True):
        href = link["href"]
        url = urljoin(index_url, href)
        parsed = urlparse(url)
        if parsed.path in {"/ix", "/ixviewer/doc/action"}:
            target = parse_qs(parsed.query).get("doc", [""])[0]
            url = urljoin(index_url, target)
            parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "www.sec.gov" or parsed.query:
            continue
        path = unquote(parsed.path)
        if not path.startswith(directory_path):
            continue
        name = path[len(directory_path):]
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            continue
        if Path(name).suffix.lower() not in {".htm", ".html", ".txt", ".pdf"} or name.lower().endswith("-index.htm"):
            continue
        # Canonical URLs have no viewer query/fragment or alternate escaping.
        url = index_url + edgar_scope.quote(name, safe="")
        if url in exclude_urls:
            continue
        row = link.find_parent("tr")
        description = " ".join((row or link).stripped_strings)
        confirmation = bool(re.search("confirmation", name + " " + description, re.I))
        cited = False
        for number in references:
            first, second = number.split(".")
            if re.search(rf"ex(?:hibit)?[\s_.-]*{first}[_.-]*{second}(?!\d)", name, re.I) or re.search(
                rf"\bex(?:hibit)?[\s-]*{re.escape(number)}\b", description, re.I
            ):
                cited = True
        if not confirmation and not cited:
            continue
        score = 3 * confirmation + 2 * cited + bool(re.search(r"swap", description + " " + name, re.I))
        candidate = Exhibit(name, url, description, score)
        if url not in matches or matches[url].score < score:
            matches[url] = candidate
    return sorted(matches.values(), key=lambda item: (-item.score, item.name))


def resolve(doc_id, *, pilot_path="data/real/pilot20.csv", data_root="data", raw_dir=None,
            client=None, input_fn=None, output=None):
    output = sys.stdout if output is None else output
    input_fn = input if input_fn is None else input_fn
    pilot_path, root = Path(pilot_path), Path(data_root)
    client = edgar_scope.SECClient() if client is None else client
    columns, pilot, digest = read_csv_snapshot(pilot_path)
    covers = [row for row in pilot if row["doc_id"] == doc_id]
    if len(covers) != 1:
        raise ValueError("doc_id must identify exactly one pilot row")
    cover = covers[0]
    if cover.get("split") == "real_test" or cover.get("collection", "real") != "real":
        raise ValueError("only real non-test pilot documents may be resolved")
    cik, accession = cover_identity(cover)
    text = local_text_path(cover["text_path"], pilot_path).read_text(encoding="utf-8")
    if not re.search(r"\bexhibit\b.{0,250}\bconfirmation\b|\bconfirmation\b.{0,250}\bexhibit\b", text, re.I | re.S):
        raise ValueError("cover text does not cite a confirmation exhibit")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
    print(f"Filing index: {index_url}", file=output)
    excluded = {row["document_url"] for row in pilot}
    excluded.update(row["document_url"] for row in rejection_history(pilot_path))
    try:
        index_bytes = client.get(index_url)
    except edgar_scope.SECRequestError as error:
        if error.status != 404:
            raise
        index_bytes = b""
    choices = find_exhibits(index_bytes, index_url, text, exclude_urls=excluded)
    if not choices:
        # A bare archive listing may have opaque filenames and no descriptions.
        index_bytes = client.get(index_url + accession + "-index.htm")
        choices = find_exhibits(index_bytes, index_url, text, exclude_urls=excluded)
    if not choices:
        raise ValueError("no matching local-filing exhibit link found; pilot unchanged")
    for index, exhibit in enumerate(choices, 1):
        print(f"{index}: {_safe(exhibit.name)} | {_safe(exhibit.description)}", file=output)
    if len(choices) == 1:
        chosen = choices[0]
    else:
        while True:
            print("Choose exhibit number (blank cancels): ", end="", file=output, flush=True)
            value = input_fn("").strip()
            if not value:
                return None
            if value.isdigit() and 1 <= int(value) <= len(choices):
                chosen = choices[int(value) - 1]
                break
            print("Enter a listed exhibit number.", file=output)
    candidate = edgar_scope.candidate_from_hit({"_id": accession + ":" + chosen.name,
        "_source": {"ciks": [cik], "display_names": [cover.get("company", "")],
                    "root_forms": [cover["form"]], "file_date": cover["filing_date"]}})
    candidate.matched_phrases.update(edgar_scope.PHRASES)
    raw_dir = Path(raw_dir) if raw_dir is not None else root / "raw/edgar"
    report = edgar_scope.download_candidates(client, [candidate], 1, raw_dir,
                                            failure_path=root / "real/download_failures.csv")
    if candidate.document_url not in report.successes:
        raise ValueError("exhibit download failed; see download_failures.csv; pilot unchanged")
    rows = edgar_screen.screen_candidates(client, [candidate], data_root=root, raw_dir=raw_dir, download_report=report)
    if not rows or not rows[0].get("text_path"):
        raise ValueError("exhibit could not be ingested without OCR/size errors; pilot unchanged")
    replacement = rows[0]
    if replacement["era"] != cover["era"]:
        raise ValueError("resolved exhibit era differs from cover metadata; pilot unchanged")
    if any(row["doc_id"] == replacement["doc_id"] or row["document_url"] == replacement["document_url"] for row in pilot):
        raise ValueError("exhibit already belongs to the pilot")
    if not preview_confirmation(replacement, pilot_path, input_fn=input_fn, output=output):
        print("Exhibit not accepted; cover remains in pilot. Download and text are retained.", file=output)
        return None
    replacement = {**replacement, "is_confirmation": "y", "product_type": "irs_fixed_float",
                   "manual_check_required": "false", "not_manually_checked": "false",
                   "selection_reason": "exhibit_resolution", "rejected_reason": "",
                   "replaces_doc_id": doc_id, "cover_document_url": cover["document_url"]}
    result = [replacement if row["doc_id"] == doc_id else row for row in pilot]
    columns = list(dict.fromkeys([*columns, *(key for row in result for key in row)]))
    if hashlib.sha256(pilot_path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("pilot changed during exhibit resolution; not replacing it")
    archive_rejections(pilot_path, [{**cover, "rejected_reason": cover.get("rejected_reason") or
                                   "cover replaced by confirmation exhibit " + replacement["doc_id"]}])
    write_csv(pilot_path, columns, [{key: row.get(key, "") for key in columns} for row in result], expected=digest, check=True)
    print(f"Replaced {_safe(doc_id)} with {_safe(replacement['doc_id'])}. Existing annotations were retained.", file=output)
    return replacement


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("doc_id")
    parser.add_argument("--pilot", type=Path, default=Path("data/real/pilot20.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        resolve(args.doc_id, pilot_path=args.pilot, data_root=args.data_root, raw_dir=args.raw_dir)
    except (EOFError, KeyboardInterrupt):
        print("Stopped; pilot unchanged.")
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(_safe(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
