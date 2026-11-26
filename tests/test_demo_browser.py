"""Optional end-to-end smoke test with an already installed Chromium."""
import json
import os
from pathlib import Path
import re

import pytest

from scripts import build_demo
from test_demo import _write_inputs

pytestmark = pytest.mark.browser


def test_offline_browser_workflow(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    data, cache = _write_inputs(tmp_path)
    output = tmp_path / "index.html"
    build_demo.main(["--data", str(data), "--cache", str(cache), "--output", str(output)])

    with playwright.sync_playwright() as engine:
        executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or engine.chromium.executable_path
        if not Path(executable).is_file():
            pytest.skip("Playwright Chromium is not installed")
        with engine.chromium.launch(executable_path=executable) as browser:
            page = browser.new_page(accept_downloads=True)
            errors, requests = [], []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
            page.on("request", lambda req: requests.append(req.url) if re.match(r"https?://", req.url) else None)
            page.route(re.compile(r"https?://"), lambda route: route.abort())
            page.goto(output.as_uri())
            page.locator("#collection").select_option("Adversarial")
            assert page.locator("#overview h1").inner_text() == "Operations overview"
            kpis = page.locator(".metrics strong").all_text_contents()
            assert len(kpis) == 4 and all(re.fullmatch(r"[0-9.,]+%", value) for value in kpis)
            assert page.locator("#automated").inner_text() == "1 of 2 fields accepted"
            assert page.locator("#auto-rate").inner_text() == "50%"
            assert page.locator("#caught-rate").inner_text() == "100%"
            page.locator("#appetite").fill("0")
            assert page.locator("#missed").inner_text().startswith("0 known errors")
            page.locator("#appetite").fill("61")
            assert page.locator("#auto-rate").inner_text() == "100%"
            page.locator("#appetite").fill("31")
            page.get_by_role("button", name="Review trade").click()
            playwright.expect(page.locator("#confirmations")).to_be_visible()
            assert page.locator("#document-picker").input_value() == "swap-conflict"
            page.locator("#document-text .mark.review").click()
            assert "Notional amount" in page.locator(".field[aria-pressed=true]").inner_text()
            expect_text = page.locator("#explanation").inner_text()
            assert "Why this needs a person" in expect_text and "Terms contradict each other" in expect_text
            reviews = page.locator("#fields .pill.review").count() - page.locator("#fields summary .pill.review").count()
            assert reviews == page.locator("#source-callouts .source-callout").count() == 1
            callout = page.locator("#source-callouts .source-callout")
            assert "Sent to review" in callout.inner_text()
            assert callout.evaluate("(el) => getComputedStyle(el).borderLeftColor") == "rgb(180, 58, 57)"
            page.get_by_role("button", name="Confirm extraction").click()
            playwright.expect(page.locator(".detail-status")).to_have_text("Confirmed recorded in this session.")
            page.locator("#corrected-value").fill("5,000,000")
            page.locator("#review-note").fill("See amendment")
            page.get_by_role("button", name="Record correction").click()
            page.get_by_role("link", name="Overview", exact=True).click()
            with page.expect_download() as downloaded:
                page.get_by_role("button", name="Export audit report").click()
            assert downloaded.value.suggested_filename == "trust-but-verify-audit.json"
            payload = json.loads(Path(downloaded.value.path()).read_text(encoding="utf-8"))
            rows = [row for row in payload["records"] if row["annotation"]]
            assert len(rows) == 1 and rows[0]["action"] == "Review"
            assert rows[0]["annotation"] == {"action": "Correction", "note": "See amendment", "corrected": "5,000,000"}
            assert rows[0]["value"] == 50_000_000
            assert [event["action"] for event in payload["annotations"]] == ["Confirmed", "Correction"]
            assert all(event["field"] == "floating_leg.notional_amount" for event in payload["annotations"])
            page.get_by_role("link", name="How it works").click()
            page.get_by_role("button", name=re.compile("Decide")).click()
            trace = page.locator("#trace article").nth(3).inner_text()
            assert "p × cost (0.62 × 100 = 62) > λ × review cost (1 × 1 = 1)" in trace
            assert "Policy outcome: Review" in page.locator("#trace article").last.inner_text()
            assert errors == [] and requests == []
