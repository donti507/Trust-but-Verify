"""Uncalibrated desk products, from fictional files to booking comparison."""
import csv
from io import BytesIO, StringIO
import json

import pytest

from test_desk import accounts, auth, csrf, post, setup
from test_desk_formats import storage
from trustswap.desk import imports, matcher, store, uploads
from trustswap.desk.products import PRODUCTS, detect
from trustswap.desk.products import extract, review, seed


def as_document(kind, terms):
    name, _, specs = PRODUCTS[kind]
    return f"Synthetic demo and fictional parties.\nProduct: {name}\n" + "\n".join(
        f"{specs[key].label}: {value}" for key, value in terms.items() if value is not None
    )


@pytest.mark.parametrize(("kind", "field", "bad", "text"), [
    ("fx_forward", "value_date", "2025-01-01", "earlier"),
    ("fx_forward", "forward_rate", "1.50", "more than 20%"),
    ("xccy_swap", "exchange_initial", None, "critical term is missing"),
    ("xccy_swap", "second_currency", "USD", "different currencies"),
    ("cds", "reference_entity", "Synthetic Example Z", "configured fictional reference entities"),
    ("cds", "termination_date", "2031-06-19", "twentieth-IMM"),
    ("cds", "spread", "unclear rate", "could not be validated"),
])
def test_each_product_flags_only_its_bad_term(kind, field, bad, text):
    terms = seed._terms(kind, 0)
    assert review.prepare(kind, as_document(kind, terms), terms)[1] == {}
    terms[field] = bad
    source = as_document(kind, terms)
    assert detect(source) == kind
    fields, checks = review.prepare(kind, source, terms)
    assert list(checks) == [f"{kind}.{field}"]
    assert text in checks[f"{kind}.{field}"]
    assert all(row["p"] == 0 for row in fields.values())


def test_cds_maturity_uses_the_standard_quantlib_roll():
    from datetime import date
    import QuantLib as ql
    from trustswap.desk.products.cds import imm_maturity

    maturity = ql.cdsMaturity(ql.Date(5, ql.January, 2026), ql.Period(5, ql.Years), ql.DateGeneration.CDS2015).to_date()
    assert imm_maturity(maturity)
    assert not imm_maturity(date(maturity.year, maturity.month, 21))


def test_seed_products_cli_and_uncalibrated_review(setup, capsys):
    from trustswap.desk.__main__ import main

    app, db_path = setup
    main(["--db", str(db_path), "seed-products"])
    assert "Added 30" in capsys.readouterr().out
    main(["--db", str(db_path), "seed-products"])
    assert "Added 0" in capsys.readouterr().out
    db = store.connect(db_path)
    assert dict(db.execute("SELECT product,COUNT(*) FROM desk_products GROUP BY product")) == dict.fromkeys(PRODUCTS, 10)
    assert {row[0] for row in db.execute("SELECT DISTINCT collection FROM trades")} == {"desk"}
    for kind, slug in (("fx_forward", "fx-forward"), ("xccy_swap", "xccy-swap"), ("cds", "cds")):
        row = db.execute("SELECT checks FROM desk_products WHERE trade_id=?", (f"example-{slug}-10",)).fetchone()
        assert list(json.loads(row[0])) == [f"{kind}.{seed.DEFECTS[kind][0]}"]
    analyst, supervisor = (auth(app, role) for role in ("analyst", "supervisor"))
    assert post(supervisor, "/settings/lambda", **{"lambda": 1e9}).status_code == 302
    route = "/trade/example-fx-forward-10"
    page = analyst.get(route).data
    assert b"Uncalibrated" in page and b"Product check: The quoted rate is more than 20%" in page
    assert b"Risk policy:" not in page
    assert post(analyst, f"{route}/accept-matched").status_code == 302
    state = dict(db.execute("SELECT path,state FROM fields WHERE trade_id='example-fx-forward-10'"))
    assert state.pop("fx_forward.forward_rate") == "open"
    assert set(state.values()) == {"accepted"}
    assert b"FX forward" in supervisor.get("/").data
    db.close()


@pytest.mark.parametrize("kind", PRODUCTS)
def test_product_upload_matches_its_own_booking_keys(setup, storage, monkeypatch, kind):
    app, db_path = setup
    db = store.connect(db_path)
    assert seed.seed(db) == 30
    terms = seed._terms(kind, 0)
    text = as_document(kind, terms)

    def never(*args):
        pytest.fail("Labelled fictional product uploads must stay offline")

    monkeypatch.setattr(extract, "_complete", never)
    analyst = auth(app, "analyst")
    response = analyst.post("/confirmations/upload", data={
        "csrf": csrf(analyst), "fictional": "yes", "confirmation": (BytesIO(text.encode()), "fictional.txt"),
    })
    key = "example-" + kind.replace("_", "-") + "-01"
    assert response.status_code == 302 and response.location == f"/trade/{key}"
    assert db.execute("SELECT product FROM desk_products WHERE trade_id=?", (key,)).fetchone()[0] == kind
    assert b"Uncalibrated" in analyst.get(response.location).data
    prepared, _ = review.prepare(kind, text, terms)
    columns = {path: row["confirmation"] for path, row in prepared.items()}
    file = StringIO()
    writer = csv.DictWriter(file, fieldnames=list(columns))
    writer.writeheader()
    writer.writerow(columns)
    file.seek(0)
    assert imports.import_bookings(db, file).matched == 1
    incompatible = StringIO(f"trade_id,trade_date\n{key},2026-01-01\n")
    assert imports.import_bookings(db, incompatible).rejected[0].reason == "Booking fields do not match the trade product"
    db.close()


def test_bool_and_desk_specific_comparison():
    assert matcher.match("xccy_swap.exchange_final", "No", False) == "MATCH"
    assert matcher.match("xccy_swap.exchange_final", "No", True) == "MISMATCH"
    assert matcher.match("fx_forward.value_date", "/ 05-Jan-2026", "2026-01-05") == "MATCH"
    assert matcher.match("cds.spread", "125 bps", "0.0125") == "MATCH"


def test_opt_in_classification_and_product_extraction_cache(storage, monkeypatch):
    source = b"Fictional confirmation with no explicit product label."
    calls = []

    def stub(spec, messages):
        prompt = messages[0]["content"]
        calls.append(prompt)
        if prompt == extract.DETECTION_PROMPT:
            return '{"product":"fx_forward"}'
        assert "forward rate is quote currency" in prompt
        return json.dumps(seed._terms("fx_forward", 0))

    monkeypatch.setattr(extract, "_complete", stub)
    assert uploads.read_upload(source, "fictional.txt").fields is None
    assert not calls
    monkeypatch.setenv("DESK_ALLOW_EXTERNAL_LLM", "true")
    monkeypatch.setenv("PCSS_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("PCSS_API_KEY", "offline-test-placeholder")
    for _ in range(2):
        outcome = uploads.read_upload(source, "fictional.txt")
        assert outcome.product == "fx_forward" and outcome.checks == {}
    assert len(calls) == 2
    assert len(list((storage / "product-calls").rglob("*.json"))) == 2


def test_labelled_scan_uses_local_product_detection_with_external_off(storage, monkeypatch):
    from PIL import Image

    png = BytesIO()
    Image.new("RGB", (12, 12), "white").save(png, format="PNG")
    monkeypatch.setattr("pytesseract.image_to_string", lambda image, timeout: as_document("cds", seed._terms("cds", 0)))

    def never(*args):
        pytest.fail("A recognizable local scan must not call a model")

    monkeypatch.setattr(extract, "_complete", never)
    result = uploads.read_upload(png.getvalue(), "fictional.png")
    assert result.product == "cds" and result.checks == {}
