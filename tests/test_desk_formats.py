"""Fictional confirmation formats and offline desk upload processing."""
from io import BytesIO, StringIO

import pytest

pytest.importorskip("flask")
pytest.importorskip("bcrypt")
pytest.importorskip("defusedxml")

from test_desk import accounts, auth, setup
from trustswap.desk import imports, store, upload_review, uploads
from trustswap.desk.formats import flatten, fpml, swift

FIN = """MT360
:15A:
:20:FICT-01
:22A:NEWT
:21N:FICT-01
:30T:20260102
:30V:20260105
:30P:20270105
:32B:USD1000000,00
:82D:Synthetic Party A
:87D:Synthetic Party B
:15B:
:37M:4,00
:18A:2
:30F:20260706
:30F:20270105
:14D:ACT/360
:14A:FOLLOWING
:18A:1
:22B:USNY
:15F:
:14F:USD-SOFR
:14J:FIRST
:38E:3M
:18A:4
:30F:20260406
:30F:20260706
:30F:20261005
:30F:20270105
:14D:ACT/360
:14A:FOLLOWING
:18A:1
:22B:USNY
:37R:N0,10
"""
FIN_CROSS = FIN.replace("MT360", "MT361").replace(":82D:", ":33B:EUR900000,00\n:82D:").replace(
    ":14F:USD-SOFR", ":14F:EUR-EURIBOR-Reuters"
)
XML = """<?xml version="1.0"?>
<dataDocument xmlns="http://www.fpml.org/FpML-5/confirmation" fpmlVersion="5-12">
  <trade>
    <tradeHeader><partyTradeIdentifier><tradeId>FICT-01</tradeId></partyTradeIdentifier>
      <tradeDate>2026-01-02</tradeDate></tradeHeader>
    <swap>
      <swapStream>
        <payerPartyReference href="A"/><receiverPartyReference href="B"/>
        <calculationPeriodDates>
          <effectiveDate><unadjustedDate>2026-01-05</unadjustedDate></effectiveDate>
          <terminationDate><unadjustedDate>2027-01-05</unadjustedDate></terminationDate>
          <calculationPeriodDatesAdjustments><businessDayConvention>MODFOLLOWING</businessDayConvention>
            <businessCenters><businessCenter>USNY</businessCenter></businessCenters>
          </calculationPeriodDatesAdjustments>
        </calculationPeriodDates>
        <paymentDates><paymentFrequency><periodMultiplier>6</periodMultiplier><period>M</period></paymentFrequency></paymentDates>
        <calculationPeriodAmount><calculation>
          <notionalSchedule><notionalStepSchedule><initialValue>1000000</initialValue><currency>USD</currency></notionalStepSchedule></notionalSchedule>
          <fixedRateSchedule><initialValue>0.04</initialValue></fixedRateSchedule>
          <dayCountFraction>ACT/360</dayCountFraction>
        </calculation></calculationPeriodAmount>
      </swapStream>
      <swapStream>
        <payerPartyReference href="B"/><receiverPartyReference href="A"/>
        <calculationPeriodDates>
          <effectiveDate><unadjustedDate>2026-01-05</unadjustedDate></effectiveDate>
          <terminationDate><unadjustedDate>2027-01-05</unadjustedDate></terminationDate>
        </calculationPeriodDates>
        <calculationPeriodAmount><calculation>
          <notionalSchedule><notionalStepSchedule><initialValue>1000000</initialValue><currency>USD</currency></notionalStepSchedule></notionalSchedule>
          <floatingRateCalculation><floatingRateIndex>USD-SOFR</floatingRateIndex>
            <spreadSchedule><initialValue>0.001</initialValue></spreadSchedule>
          </floatingRateCalculation>
        </calculation></calculationPeriodAmount>
      </swapStream>
    </swap>
  </trade>
  <party id="A"><partyName>Synthetic Party A</partyName></party>
  <party id="B"><partyName>Synthetic Party B</partyName></party>
</dataDocument>"""


@pytest.mark.parametrize(("format_module", "source", "currency", "spread"), [
    (swift, FIN, "USD", "-0.001"),
    (swift, FIN_CROSS, "EUR", "-0.001"),
    (fpml, XML, "USD", "0.001"),
])
def test_parse_export_parse(format_module, source, currency, spread):
    from decimal import Decimal

    parsed = format_module.parse(source)
    fields = flatten(parsed.trade)
    assert parsed.reference == "FICT-01"
    assert Decimal(fields["fixed_leg.fixed_rate"]) == Decimal("0.04")
    assert fields["fixed_leg.payment_frequency"] == "6M"
    assert fields["floating_leg.notional_currency"] == currency
    assert Decimal(fields["floating_leg.spread"]) == Decimal(spread)
    output = format_module.export(parsed.reference, parsed.trade)
    reread = format_module.parse(output)
    assert reread.reference == parsed.reference
    assert flatten(reread.trade) == fields
    assert parsed.trade.trade_date.span is not None


def test_unsafe_xml_never_expands_external_entities(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("private fictional text", encoding="utf-8")
    xxe = f'<!DOCTYPE swap [<!ENTITY external SYSTEM "{outside.as_uri()}">]><swap>&external;</swap>'
    with pytest.raises(ValueError, match="unsafe"):
        fpml.parse(xxe)


@pytest.mark.parametrize(("filename", "payload"), [
    ("small.pdf", b"plain text pretending to be a PDF"),
    ("large.txt", b"x" * (uploads.MAX_BYTES + 1)),
    ("renamed.txt", b"\x89PNG\r\n\x1a\nfake"),
    ("binary.txt", b"\x00\xff"),
    ("x.xml", b"not XML"),
])
def test_wrong_type_and_oversize_are_rejected(filename, payload):
    with pytest.raises(ValueError):
        uploads.sniff(payload, filename)


@pytest.fixture
def storage(tmp_path, monkeypatch):
    directory = tmp_path / "private"
    monkeypatch.setenv("DESK_UPLOAD_DIR", str(directory))
    monkeypatch.delenv("DESK_ALLOW_EXTERNAL_LLM", raising=False)
    return directory


def test_private_storage_and_default_external_guard(storage, monkeypatch):
    def unavailable(*args, **kwargs):
        pytest.fail("An offline upload must not start OCR or a model")

    monkeypatch.setattr(upload_review, "scorer", unavailable)
    monkeypatch.setattr(uploads, "_pdf", unavailable)
    payload = b"%PDF-fake bytes"
    info = uploads.read_upload(payload, "../../confirmation.pdf")
    assert info.file.parent == storage and info.file.read_bytes() == payload
    assert info.fields is None and list(storage.iterdir()) == [info.file]
    with pytest.raises(ValueError, match="outside"):
        uploads.upload_directory(uploads.ROOT / "uploads")
    with pytest.raises(ValueError, match="disabled"):
        upload_review.extract_and_score("example", "", False, client=None, model_id=None,
                                        cache=None, risk=None, costs=None)


def test_image_ocr_is_local_and_receives_extracted_text(storage, monkeypatch):
    from PIL import Image

    data = BytesIO()
    Image.new("RGB", (20, 20), "white").save(data, format="PNG")
    monkeypatch.setenv("DESK_ALLOW_EXTERNAL_LLM", "true")
    monkeypatch.setattr("pytesseract.image_to_string", lambda image, timeout: "Synthetic Party A\n  Fictional trade")
    called = []

    def fake_pipeline(doc_id, text, ocr):
        called.append((text, ocr))
        return {}

    monkeypatch.setattr(upload_review, "scorer", lambda directory: fake_pipeline)
    uploads.read_upload(data.getvalue(), "fictional.png")
    assert called == [("Synthetic Party A\nFictional trade", True)]


def test_pdf_uses_text_layer_or_local_ocr(storage, monkeypatch):
    import pymupdf

    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Synthetic Party A")
        pdf.new_page()
        data = pdf.tobytes()
    monkeypatch.setenv("DESK_ALLOW_EXTERNAL_LLM", "1")
    monkeypatch.setattr("pytesseract.image_to_string", lambda image: "Fictional scan page")
    called = []

    def fake_pipeline(doc_id, text, ocr):
        called.append((text, ocr))
        return {}

    monkeypatch.setattr(upload_review, "scorer", lambda directory: fake_pipeline)
    uploads.read_upload(data, "fictional.pdf")
    assert called == [("Synthetic Party A\nFictional scan page", True)]


def test_upload_permissions_match_booking_and_only_export_affirmed(setup, storage, monkeypatch):
    from test_desk import csrf, post

    app, db_path = setup
    grader = auth(app, "auditor")
    analyst = auth(app, "analyst")
    assert b"External extraction is off" in analyst.get("/").data

    def submit(client, *, valid_csrf=True, filename="fictional.xml", data=XML.encode(), reference=None):
        fields = {"confirmation": (BytesIO(data), filename), "fictional": "yes"}
        if valid_csrf:
            fields["csrf"] = csrf(client)
        if reference is not None:
            fields["trade_id"] = reference
        return client.post("/confirmations/upload", data=fields)

    assert submit(grader).status_code == 403
    assert submit(analyst, valid_csrf=False).status_code == 400
    assert submit(analyst, filename="big.txt", data=b"x" * (uploads.MAX_BYTES + 1)).status_code == 400
    assert submit(analyst, filename="fictional.txt", data=b"Synthetic Party A").status_code == 202
    assert submit(analyst, filename="fictional.pdf", data=b"not a pdf").status_code == 400

    def never(*args, **kwargs):
        pytest.fail("Structured uploads must never use external extraction")

    monkeypatch.setenv("DESK_ALLOW_EXTERNAL_LLM", "true")
    monkeypatch.setattr(upload_review, "scorer", never)
    response = submit(analyst)
    assert response.status_code == 302 and response.location == "/trade/FICT-01"
    assert analyst.get("/trade/FICT-01/export/fpml").status_code == 409
    parsed = fpml.parse(XML)
    fields = flatten(parsed.trade)
    # Import the desk's fictional booking through the same CSV path as the CLI.
    import csv
    file = StringIO()
    writer = csv.DictWriter(file, fieldnames=["trade_id", *fields])
    writer.writeheader()
    writer.writerow({"trade_id": parsed.reference, **{
        key: ", ".join(value) if isinstance(value, list) else value for key, value in fields.items()
    }})
    file.seek(0)
    db = store.connect(db_path)
    try:
        assert imports.import_bookings(db, file).updated == 1
        before = dict(db.execute("SELECT path,booking FROM fields WHERE trade_id='FICT-01'"))
        assert submit(analyst).status_code == 302
        assert dict(db.execute("SELECT path,booking FROM fields WHERE trade_id='FICT-01'")) == before
        assert post(analyst, "/trade/FICT-01/accept-matched").status_code == 302
        assert db.execute("SELECT collection,state FROM trades WHERE id='FICT-01'").fetchone()[:] == ("desk", "Affirmed")
        assert db.execute("SELECT COUNT(*) FROM audit WHERE action='confirmation_uploaded'").fetchone()[0] == 1
        exported = grader.get("/trade/FICT-01/export/fpml")
        assert exported.status_code == 200 and "attachment" in exported.headers["Content-Disposition"]
        assert flatten(fpml.parse(exported.get_data(as_text=True)).trade) == fields
    finally:
        db.close()


def test_text_scoring_uses_frozen_pipeline_and_caches_by_input(tmp_path, monkeypatch):
    import json

    from trustswap.common.cache import Cache
    from trustswap.decide.policy import load_costs
    from trustswap.extract.clients import Completion
    from trustswap.extract.fields import PATHS

    monkeypatch.setenv("DESK_ALLOW_EXTERNAL_LLM", "true")
    terms = {"trade_date": "2026-01-02", "effective_date": "2026-01-05",
             "termination_date": "2027-01-05", "fixed_leg.fixed_rate": "4.00%",
             "floating_leg.floating_rate_option": "USD-SOFR"}
    source = "\n".join(terms.values())

    class OfflineClient:
        params = {}
        requests = 0

        def complete(self, messages, *, temperature, seed):
            self.requests += 1
            fields = {path: {"value": terms.get(path), "line": None, "confidence": 0.7} for path in PATHS}
            return Completion(json.dumps({"structure": "fixed_float", "fields": fields}), finish_reason="stop")

    class OfflineRisk:
        def predict(self, features):
            assert "e" not in features
            assert set(features["collection"]) == {"desk"}
            assert features["g_span_found"].sum() == len(terms)
            assert set(features["k_samples"]) == {1}
            return [0.25] * len(features)

    client = OfflineClient()
    parameters = {"client": client, "model_id": "fictional", "cache": Cache(tmp_path / "calls"),
                  "risk": OfflineRisk(), "costs": load_costs()}
    extracted = upload_review.extract_and_score("desk-1", source, False, **parameters)
    assert extracted == upload_review.extract_and_score("desk-2", source, False, **parameters)
    assert client.requests == 1
    assert extracted["fixed_leg.fixed_rate"]["p"] == 0.25
    assert extracted["trade_date"]["raw"] == "2026-01-02"
