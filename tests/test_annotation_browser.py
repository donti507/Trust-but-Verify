"""Optional Chromium check for annotation controls using only WSGI fixtures."""
from contextlib import closing
import os

import pytest

from test_annotation_tool import auth, corpus, local, post
from trustswap.annotate import store


@pytest.mark.browser
def test_editor_selection_keyboard_and_resume(local):
    playwright = pytest.importorskip("playwright.sync_api")
    app, path = local
    with closing(store.connect(path)) as db:
        doc = db.execute("SELECT doc FROM cohort WHERE control=0 ORDER BY doc").fetchone()[0]
    client = auth(app, "A1")
    post(client, f"/documents/{doc}/start", mode="blind")
    with playwright.sync_playwright() as runtime:
        executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
        try:
            browser = runtime.chromium.launch(executable_path=executable, args=["--no-sandbox"])
        except playwright.Error:
            pytest.skip("Install Playwright Chromium to run the offline annotation browser test")
        page = browser.new_page(offline=True)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def serve(route):
            assert route.request.url.startswith("http://localhost/")
            response = client.open(route.request.url, method=route.request.method,
                                   data=route.request.post_data or None, follow_redirects=True,
                                   headers={key: val for key, val in route.request.headers.items()
                                            if key in {"content-type", "x-csrf-token"}})
            response.direct_passthrough = False
            route.fulfill(status=response.status_code, headers=dict(response.headers), body=response.get_data())

        page.route("**/*", serve)
        page.goto(f"http://localhost/documents/{doc}")
        source = page.locator("#source-text")
        date_card = page.locator('[data-path="trade_date"]')
        effective = page.locator('[data-path="effective_date"]')
        page.locator("#structure").select_option("fixed_float")
        playwright.expect(date_card).to_be_visible()
        date_card.locator("[data-heading]").focus()
        source.evaluate("(node) => { const i = node.value.indexOf('04-Jan-2024'); node.setSelectionRange(i, i + 11); }")
        page.get_by_role("button", name="Use selected text").click()
        playwright.expect(date_card.locator("[data-value]")).to_have_value("2024-01-04")
        date_card.locator("[data-heading]").focus()
        page.keyboard.press("Tab")
        playwright.expect(effective.locator("[data-heading]")).to_be_focused()
        page.keyboard.press("a")
        playwright.expect(effective.locator("[data-status]")).to_have_value("absent")
        effective.locator("[data-note]").fill("Still checking the terms")
        playwright.expect(page.locator("#saved-state")).to_have_text("Saved locally")
        page.reload()
        playwright.expect(effective.locator("[data-note]")).to_have_value("Still checking the terms")
        playwright.expect(date_card.locator("[data-value]")).to_have_value("2024-01-04")
        date_card.locator("[data-value]").press("Enter")
        playwright.expect(page.locator("#editor-error")).to_contain_text("Choose a field status")
        assert not errors
        browser.close()
