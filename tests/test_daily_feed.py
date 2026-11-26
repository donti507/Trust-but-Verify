"""Offline feed tests with generated market rows and fake provider responses."""
import csv
from datetime import date
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from scripts import daily_feed
from trustswap.common.cache import Cache
from trustswap.decide.policy import load_costs
from trustswap.extract.clients import Completion
from trustswap.extract.prompt import gold_answer
from trustswap.feed import dtcc, pipeline
from trustswap.feed.publish import run
from trustswap.feed.render import render_batch

DAY = date(2026, 11, 23)


def market_file():
    buffer = io.StringIO()
    records = []
    for i in range(6):
        records.append({
            "UPI FISN": "NA/Swap OIS USD", "Asset Class": "IR", "Action type": "NEWT",
            "Non-standardized term indicator": "FALSE", "Package indicator": "FALSE",
            "Execution Timestamp": "2026-11-20T13:00:00Z", "Effective Date": "2026-11-24",
            "Expiration Date": "2031-11-24", "Notional currency-Leg 1": "USD",
            "Notional currency-Leg 2": "USD", "Notional amount-Leg 1": f"{20+i},000,000",
            "Notional amount-Leg 2": f"{20+i},000,000", "Fixed rate-Leg 1": "0.04", "Fixed rate-Leg 2": "",
            "Fixed rate day count convention-leg 1": "A004", "Floating rate day count convention-leg 2": "A004",
            "Fixed rate payment frequency period-Leg 1": "YEAR",
            "Fixed rate payment frequency period multiplier-Leg 1": "1",
            "Floating rate payment frequency period-Leg 2": "YEAR",
            "Floating rate payment frequency period multiplier-Leg 2": "1",
            "Floating rate reset frequency period-leg 2": "MNTH",
            "Floating rate reset frequency period multiplier-leg 2": "3", "UPI Underlier Name": "USD-SOFR",
        })
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records + [{**records[0], "UPI FISN": "NA/Option USD"}])
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr("rates.csv", buffer.getvalue())
    return content.getvalue()


def test_dtcc_selection_calendar_and_fictional_render():
    day = dtcc.previous_business_day(DAY)
    assert day == date(2026, 11, 20)
    assert dtcc.previous_business_day(date(2026, 7, 6)) == date(2026, 7, 3)
    trades = dtcc.select_trades(market_file(), day)
    assert trades == dtcc.select_trades(market_file(), day)
    assert len(trades) == 5 and all(trade.fixed_leg is not None for trade in trades)
    docs = list(render_batch(trades, DAY))
    assert sum(bool(doc.defect) for doc in docs) == 2
    assert all("Synthetic Party" in doc.text for doc in docs)


def test_publication_is_idempotent_and_weekend_is_a_noop(tmp_path):
    output, site = tmp_path / "feed", tmp_path / "index.html"
    requested = []
    kwargs = dict(
        download=lambda day: requested.append(day) or market_file(), select=dtcc.select_trades,
        render=lambda trades, day: enumerate(trades), previous_day=dtcc.previous_business_day,
        scorer=lambda item: {"id": str(item[0]), "collection": "live_feed",
                             "defect": "scale_error" if item[0] < 2 else None, "defect_caught": item[0] == 0,
                             "fields": [{"action": "Accept"}]},
        page=lambda feed: json.dumps(feed),
    )
    assert "weekend" in run(date(2026, 11, 22), output, site, 1.0, **kwargs)
    assert not requested and not output.exists()
    assert run(DAY, output, site, 1.0, **kwargs).startswith("Published 5")
    files = {file: file.read_bytes() for file in [*output.iterdir(), site]}
    assert "already published" in run(DAY, output, site, 1.0, **kwargs)
    assert requested == [date(2026, 11, 20)]
    assert all(path.read_bytes() == data for path, data in files.items())
    assert json.loads(site.read_text())["history"][0]["defects_caught"] == 1


def test_budget_before_provider_and_model_cache(tmp_path):
    confirmation = next(render_batch(dtcc.select_trades(market_file(), DAY), DAY))

    class Fake:
        params = {"max_tokens": 4096}
        calls = 0

        def complete(self, messages, **kwargs):
            self.calls += 1
            return Completion(json.dumps(gold_answer(confirmation.gold.trade, confirmation.text)),
                              input_tokens=100, output_tokens=100, finish_reason="stop")

    class Risk:
        def predict(self, rows):
            assert set(rows.collection) == {"live_feed"}
            return np.full(len(rows), 0.01)

    fake = Fake()
    ledger = tmp_path / "budget.txt"
    cached = Cache(tmp_path / "calls")
    restricted = pipeline.BudgetedClient(fake, ledger, 1)
    with pytest.raises(pipeline.FeedSkip, match="token cap"):
        restricted.complete([], temperature=0, seed=0)
    assert fake.calls == 0 and not ledger.exists()
    budget = pipeline.BudgetedClient(fake, ledger, 60_000)
    score = pipeline.ReviewPipeline(budget, "fixture", cached, Risk(), load_costs(), 1.0)
    doc = score(confirmation)
    assert doc == score(confirmation)
    assert fake.calls == 1 and ledger.read_text() == "200"
    assert doc["collection"] == "live_feed"


@pytest.mark.parametrize("error", [dtcc.NoFile("no DTCC file"), pipeline.FeedSkip("PCSS request failed"),
                                  pipeline.FeedSkip("60k daily token cap")])
def test_expected_failures_exit_zero_without_publication(tmp_path, monkeypatch, capsys, error):
    def stop(*args, **kwargs):
        raise error
    monkeypatch.setattr(daily_feed, "run", stop)
    args = ["--date", DAY.isoformat(), "--feed-dir", str(tmp_path / "feed"), "--site", str(tmp_path / "site"),
            "--replay", str(tmp_path / "unused"), "--risk-model", str(tmp_path / "unused")]
    assert daily_feed.main(args) == 0
    assert "Skipped:" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())
