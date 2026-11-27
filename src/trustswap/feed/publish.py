"""Prepare complete daily feed and standalone site snapshots before publishing."""
import json
from datetime import timedelta
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def publish(output, site, day, source, documents, threshold, page):
    day_key = day.isoformat()
    history_path = output / "history.json"
    history = read(history_path) if history_path.is_file() else []
    fields = [field for doc in documents for field in doc["fields"]]
    summary = {
        "date": day_key, "source_date": source.isoformat(), "trades": len(documents),
        "automation_pct": 100 * sum(field["action"] == "Accept" for field in fields) / len(fields),
        "defects_planted": sum(bool(doc["defect"]) for doc in documents),
        "defects_caught": sum(doc["defect_caught"] for doc in documents),
    }
    update = {"collection": "live_feed", "threshold": threshold, **summary, "documents": documents}
    history = sorted([row for row in history if row["date"] != day_key] + [summary], key=lambda row: row["date"])
    newest = max(row["date"] for row in history)
    from datetime import date

    first_day = (date.fromisoformat(newest) - timedelta(days=29)).isoformat()
    recent = [row for row in history if row["date"] >= first_day]
    days = [update if row["date"] == day_key else read(output / f"{row['date']}.json") for row in recent]
    html = page({"days": days, "history": recent})
    write(site, html)
    write(history_path, json.dumps(history, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write(output / f"{day_key}.json", json.dumps(update, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run(day, output, site, threshold, *, download, select, render, scorer, page, previous_day):
    if day.weekday() >= 5:
        return "Skipped: weekend"
    if (output / f"{day.isoformat()}.json").is_file():
        return "Skipped: date already published"
    source = previous_day(day)
    trades = select(download(source), source)
    documents = [scorer(doc) for doc in render(trades, day)]
    if len(documents) != 5 or sum(bool(doc["defect"]) for doc in documents) != 2:
        raise ValueError("A daily feed needs five documents and two planted defects")
    publish(output, site, day, source, documents, threshold, page)
    return f"Published {len(documents)} fictional confirmations for {day}"
