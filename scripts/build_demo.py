"""Build an offline HTML demo from generated documents and cached model outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd

from trustswap.decide.policy import COSTS_FILE, Costs, field_costs, load_costs
from trustswap.score.features import SEVERITY

ROOT = Path(__file__).resolve().parents[1]
COLLECTIONS = {"synthetic": ("train", "dev", "syn_test"), "adversarial": ("adv_test",)}
CATEGORIES = ("Clean synthetic", "Noisy synthetic", "Adversarial")
RULE_TEXT = {
    "V1": "Required term missing or invalid.",
    "V2": "Value not found in the cited text.",
    "V3": "Dates out of order.",
    "V4": "Date off the payment schedule.",
    "V5": "Stated amount does not recompute.",
    "V6": "Terms contradict each other.",
    "V7": "Unusual convention for this market.",
}
LEGS = {"fixed_leg": "fixed leg", "floating_leg": "floating leg", "floating_leg_2": "second floating leg"}


def decision(p: float, c_bar: float, r: float, lam: float) -> str:
    """Mirror the browser decision, including acceptance at an exact tie."""
    return "Review" if p * c_bar > lam * r else "Accept"


def _jsonl(path: Path):
    if path.is_file():
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)


def _name(path: str) -> str:
    leg, _, field = path.rpartition(".")
    title = field.replace("_", " ").capitalize()
    return f"{title} ({LEGS[leg]})" if leg else title


def _span(span, length: int):
    if isinstance(span, list) and len(span) == 2 and all(type(x) is int for x in span):
        start, end = span
        if 0 <= start < end <= length:
            return span
    return None


def load_documents(data: Path, cache: Path, model: str, costs: Costs, n: int) -> list[dict]:
    """Select eligible cached documents by category, prioritizing review or errors at lambda 1."""
    risks = pd.read_parquet(
        cache / "risk" / f"{model}.parquet", engine="pyarrow",
        columns=["doc_id", "field", "collection", "split", "e", "p_full_logistic"],
        filters=[[("collection", "==", collection), ("split", "in", splits)]
                 for collection, splits in COLLECTIONS.items()],
    )
    c, r = field_costs(risks, costs)
    risks = risks.assign(c_bar=c, r=r)
    groups = {category: [] for category in CATEGORIES}
    for collection, splits in COLLECTIONS.items():
        base = data / collection
        gold_ids = {row["doc_id"] for row in _jsonl(base / "gold.jsonl") if row["collection"] == collection}
        manifests = {row["doc_id"]: row for row in _jsonl(base / "docs.jsonl")
                     if row["collection"] == collection and row.get("split") in splits}
        for doc_id, fields in risks.loc[risks["collection"] == collection].groupby("doc_id"):
            if doc_id not in gold_ids or doc_id not in manifests or not re.fullmatch(r"[\w][\w.-]*", doc_id):
                continue
            manifest = manifests[doc_id]
            if not fields["split"].eq(manifest["split"]).all():
                continue
            extract = cache / "extract" / model / f"{doc_id}.jsonl"
            verify = cache / "verify" / model / f"{doc_id}.json"
            path = Path(manifest["text_path"])
            if not path.is_file():
                path = base / "text" / f"{doc_id}.txt"
            if not (path.is_file() and extract.is_file() and verify.is_file()):
                continue
            if path.resolve().is_relative_to((data / "real").resolve()):
                continue

            category = "Adversarial" if collection == "adversarial" else (
                "Noisy synthetic" if doc_id.endswith(("_ocr1", "_ocr2")) else "Clean synthetic"
            )
            interesting = any(row.e or decision(row.p_full_logistic, row.c_bar, row.r, 1) == "Review"
                              for row in fields.itertuples(index=False))
            groups[category].append((not interesting, doc_id, path, extract, verify, fields))

    queues = {category: iter(sorted(rows, key=lambda row: row[:2])) for category, rows in groups.items()}
    documents = []
    while len(documents) < n:
        previous_count = len(documents)
        for category, queue in queues.items():
            for _, doc_id, path, extract, verify, fields in queue:
                sample = next((row for row in _jsonl(extract) if row["k"] == 0), None)
                if sample is None:
                    continue
                text = path.read_text(encoding="utf-8")
                rules = json.loads(verify.read_text(encoding="utf-8")).get("rules", {})
                extracted = sample.get("fields", {}) if sample.get("schema_valid", True) else {}
                entries = []
                for row in fields.itertuples(index=False):
                    item = extracted.get(row.field, {})
                    value = item.get("value")
                    entries.append({
                        "name": _name(row.field),
                        "value": item.get("raw_value") if value is None else value,
                        "span": _span(item.get("span"), len(text)),
                        "p": row.p_full_logistic, "c_bar": row.c_bar, "r": row.r,
                        "severity": SEVERITY[row.field.rsplit(".", 1)[-1]], "e": int(row.e),
                        "rules_fired": [rule for rule in RULE_TEXT
                                        if rules.get(rule, {}).get(row.field, {}).get("verdict") in {"HARD", "SOFT"}],
                    })
                documents.append({"id": doc_id, "category": category, "text": text, "fields": entries})
                break
            if len(documents) == n:
                break
        if previous_count == len(documents):
            break
    return documents


def render_page(documents: list[dict]) -> str:
    payload = json.dumps({"documents": documents, "rules": RULE_TEXT}, allow_nan=False, separators=(",", ":"))
    # Keep data inside the JSON script element even if document text contains HTML.
    payload = payload.replace("<", "\\u003c")
    payload = re.sub(r"(?i)(https?):", lambda match: match[1] + r"\u003a", payload)
    return PAGE.replace("__DEMO_DATA__", payload)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("frontier", "open_weights"), default="open_weights")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--costs", type=Path, default=COSTS_FILE)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "demo" / "index.html")
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be positive")
    risk_path = args.cache / "risk" / f"{args.model}.parquet"
    if not risk_path.is_file():
        parser.error(f"cached risks are missing at {risk_path}")
    documents = load_documents(args.data, args.cache, args.model, load_costs(args.costs), args.n)
    if not documents:
        parser.error("no generated documents have matching risk, gold, text, extraction, and verifier records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_page(documents), encoding="utf-8")
    print(f"Wrote {args.output} ({len(documents)} documents)")
    return 0


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Trust but Verify</title>
<style>
:root {
    color-scheme: light;
    font-family: system-ui, sans-serif;
    color: #162638;
    background: #edf2f6;
}
* { box-sizing: border-box; }
body { margin: 0; }
header { background: #142b40; color: white; padding: 1.5rem max(1rem, calc((100% - 1320px) / 2)); }
header h1 { font: 650 clamp(1.8rem, 4vw, 2.5rem) Georgia, serif; margin: 0 0 .55rem; }
header p { display: inline-block; margin: 0; padding: .35rem .65rem; background: #25475a; border-radius: .3rem; }
main { width: min(1320px, calc(100% - 2rem)); margin: 1rem auto 2rem; }
.controls, .card, .panel { background: white; border: 1px solid #cad6e0; border-radius: .65rem; }
.controls { display: flex; flex-wrap: wrap; align-items: end; justify-content: space-between; gap: 1rem; padding: 1rem; }
label { display: block; font-weight: 650; margin-bottom: .3rem; }
.picker { min-width: min(100%, 26rem); }
.picker-controls { display: flex; flex-wrap: wrap; gap: .35rem; }
button, select, input { font: inherit; }
select, .nav { color: inherit; background: #fff; border: 1px solid #9aafbd; border-radius: .3rem; padding: .4rem .6rem; }
select { flex: 1; min-width: 10rem; max-width: 100%; }
button { cursor: pointer; }
button:disabled { cursor: default; opacity: .55; }
.slider { flex: 1 1 16rem; max-width: 26rem; }
.slider label { display: flex; flex-wrap: wrap; justify-content: space-between; gap: .25rem .8rem; }
output { color: #3c5366; font-size: .9rem; font-variant-numeric: tabular-nums; }
input[type=range] { width: 100%; min-height: 1.6rem; margin: 0; accent-color: #20676c; }
.cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; margin: .85rem 0; }
.card { margin: 0; padding: .7rem .9rem; }
.card dt { font-size: .92rem; color: #3c5366; }
.card dd { margin: .2rem 0 0; font-size: 1.65rem; line-height: 1.1; font-weight: 650; }
.workspace { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.18fr); gap: .85rem; align-items: start; }
.panel { min-width: 0; overflow: hidden; }
.panel-heading { padding: .85rem 1rem; border-bottom: 1px solid #dbe3ea; }
h2 { font-size: 1.15rem; margin: 0; }
.document { min-height: 15rem; max-height: 65vh; overflow: auto; margin: 0; padding: 1rem; white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.65; font-family: Georgia, serif; }
.detail { background: #f0f6f8; border-bottom: 1px solid #dbe3ea; padding: .7rem 1rem; }
.detail p { margin: .3rem 0 0; }
.detail ul { padding-left: 1.2rem; margin: .35rem 0 0; }
.table-scroll { overflow: auto; max-height: 55vh; }
table { width: 100%; border-collapse: collapse; text-align: left; font-size: .9rem; }
th, td { padding: .55rem .55rem; border-bottom: 1px solid #e3eaf0; vertical-align: top; overflow-wrap: anywhere; }
thead th { position: sticky; top: 0; background: #f6f8fa; }
tbody tr { cursor: pointer; }
tbody tr:hover, tbody tr.selected { background: #eaf2f7; }
.field-link, .mark { border: 0; color: inherit; font: inherit; padding: 0; background: none; text-align: left; }
.field-link { text-decoration: underline; text-underline-offset: .17em; }
.mark { border-radius: .16em; display: inline; white-space: pre-wrap; }
.badge { font-weight: 650; border-radius: .3rem; padding: .14rem .38rem; white-space: nowrap; }
.accept { background: #d8f0df; color: #14502e; }
.review { background: #f8dddf; color: #812c36; }
.mark.selected { box-shadow: 0 0 0 2px #195460; }
:focus-visible { outline: 3px solid #226879; outline-offset: 2px; }
@media (max-width: 860px) { .workspace { grid-template-columns: 1fr; } }
@media (max-width: 520px) { .cards { grid-template-columns: repeat(2, 1fr); } th, td { padding: .45rem; } }
</style>
</head>
<body>
<header>
    <h1>Trust but Verify</h1>
    <p>Synthetic document, fictional parties</p>
</header>
<main>
    <section class="controls" aria-label="Demo controls">
        <div class="picker">
            <label for="document-picker">Document</label>
            <div class="picker-controls">
                <button class="nav" id="previous" type="button">Previous</button>
                <select id="document-picker"></select>
                <button class="nav" id="next" type="button">Next</button>
            </div>
        </div>
        <div class="slider">
            <label for="appetite">Risk appetite <output id="threshold" for="appetite"></output></label>
            <input id="appetite" type="range" min="0" max="61" step="1" value="31">
        </div>
    </section>
    <section class="cards" aria-label="Decision summary">
        <dl class="card"><dt>Fields automated</dt><dd id="automated"></dd></dl>
        <dl class="card"><dt>Fields sent to a person</dt><dd id="reviewed"></dd></dl>
        <dl class="card"><dt>Errors caught</dt><dd id="caught"></dd></dl>
        <dl class="card"><dt>Errors that slipped through</dt><dd id="missed"></dd></dl>
    </section>
    <div class="workspace">
        <section class="panel" aria-labelledby="document-title">
            <div class="panel-heading"><h2 id="document-title">Document text</h2></div>
            <pre class="document" id="document-text" tabindex="0"></pre>
        </section>
        <section class="panel" aria-labelledby="fields-title">
            <div class="panel-heading"><h2 id="fields-title">Extracted fields</h2></div>
            <div class="detail" id="explanation" aria-live="polite" hidden></div>
            <div class="table-scroll" tabindex="0" aria-label="Field decisions">
                <table>
                    <thead><tr>
                        <th scope="col">Field</th><th scope="col">Extracted value</th><th scope="col">Risk</th>
                        <th scope="col">Cost class</th><th scope="col">Decision</th>
                    </tr></thead>
                    <tbody id="fields"></tbody>
                </table>
            </div>
        </section>
    </div>
</main>
<script type="application/json" id="demo-data">__DEMO_DATA__</script>
<script>
"use strict";

function decision(p, c_bar, r, lambda) {
    return p * c_bar > lambda * r ? "Review" : "Accept";
}

(function () {
    const byId = id => document.getElementById(id);
    const data = JSON.parse(byId("demo-data").textContent);
    const picker = byId("document-picker");
    const slider = byId("appetite");
    const previous = byId("previous");
    const next = byId("next");
    let selected = -1;

    function element(tag, text, className) {
        const node = document.createElement(tag);
        node.textContent = text;
        if (className) node.className = className;
        return node;
    }

    function percent(p) {
        return (100 * p).toLocaleString(undefined, {maximumFractionDigits: 1}) + "%";
    }

    function chooseField(index) {
        selected = index;
        render();
        byId("field-" + index).focus({preventScroll: true});
    }

    function renderText(record, decisions) {
        // Python span offsets count Unicode code points, rather than UTF-16 code units.
        const characters = Array.from(record.text);
        const bounds = new Set([0, characters.length]);
        record.fields.forEach(field => {
            if (field.span) field.span.forEach(offset => bounds.add(offset));
        });
        const positions = Array.from(bounds).sort((a, b) => a - b);
        const fragment = document.createDocumentFragment();
        for (let i = 1; i < positions.length; i++) {
            const start = positions[i - 1];
            const end = positions[i];
            const text = characters.slice(start, end).join("");
            const matches = [];
            record.fields.forEach((field, index) => {
                if (field.span && field.span[0] <= start && end <= field.span[1]) matches.push(index);
            });
            if (!matches.length) {
                fragment.appendChild(document.createTextNode(text));
                continue;
            }
            // For overlapping spans, show the selected field or prefer a review highlight.
            const review = matches.find(index => decisions[index] === "Review");
            const index = matches.includes(selected) ? selected : (review ?? matches[0]);
            const mark = element("button", text, "mark " + decisions[index].toLowerCase());
            if (selected === index) mark.classList.add("selected");
            mark.type = "button";
            mark.setAttribute("aria-label", record.fields[index].name + ": " + decisions[index] + ", " + text);
            mark.addEventListener("click", () => chooseField(index));
            fragment.appendChild(mark);
        }
        byId("document-text").replaceChildren(fragment);
    }

    function renderExplanation(field, action) {
        const panel = byId("explanation");
        panel.hidden = !field;
        panel.replaceChildren();
        if (!field) return;
        panel.append(
            element("strong", field.name + " · " + action),
            element("p", "Risk " + percent(field.p) + ". If wrong, cost is " + field.severity + ".")
        );
        if (field.rules_fired.length) {
            const list = document.createElement("ul");
            field.rules_fired.forEach(rule => list.appendChild(element("li", data.rules[rule])));
            panel.appendChild(list);
        } else {
            panel.appendChild(element("p", "No verifier warnings for this field."));
        }
    }

    function displayValue(value) {
        if (value === null) return "Missing";
        if (Array.isArray(value)) return value.join(", ");
        if (typeof value === "number") return value.toLocaleString(undefined, {maximumFractionDigits: 12});
        return String(value);
    }

    function render() {
        const record = data.documents[picker.selectedIndex];
        const step = Number(slider.value);
        const lambda = step === 0 ? 0 : 10 ** ((step - 31) / 10);
        byId("threshold").textContent = "Threshold " + Number(lambda.toPrecision(4)).toString();
        const decisions = record.fields.map(field => decision(field.p, field.c_bar, field.r, lambda));
        let automated = 0, reviewed = 0, caught = 0, missed = 0;
        const rows = document.createDocumentFragment();
        record.fields.forEach((field, index) => {
            const action = decisions[index];
            if (action === "Review") {
                reviewed++;
                caught += field.e;
            } else {
                automated++;
                missed += field.e;
            }
            const row = document.createElement("tr");
            if (index === selected) row.classList.add("selected");
            const name = element("th", "");
            name.scope = "row";
            const button = element("button", field.name, "field-link");
            button.id = "field-" + index;
            button.type = "button";
            button.setAttribute("aria-pressed", String(index === selected));
            name.appendChild(button);
            const badge = element("span", action, "badge " + action.toLowerCase());
            const actionCell = document.createElement("td");
            actionCell.appendChild(badge);
            row.append(name, element("td", displayValue(field.value)), element("td", percent(field.p)),
                element("td", field.severity), actionCell);
            row.addEventListener("click", () => chooseField(index));
            rows.appendChild(row);
        });
        byId("fields").replaceChildren(rows);
        byId("automated").textContent = automated;
        byId("reviewed").textContent = reviewed;
        byId("caught").textContent = caught;
        byId("missed").textContent = missed;
        renderText(record, decisions);
        renderExplanation(record.fields[selected], decisions[selected]);
        previous.disabled = picker.selectedIndex === 0;
        next.disabled = picker.selectedIndex === data.documents.length - 1;
    }

    function changeDocument() {
        selected = -1;
        byId("document-text").scrollTop = 0;
        byId("fields").closest(".table-scroll").scrollTop = 0;
        render();
    }

    data.documents.forEach(record => picker.appendChild(element("option", record.category + " · " + record.id)));
    picker.addEventListener("change", changeDocument);
    previous.addEventListener("click", () => { picker.selectedIndex--; changeDocument(); });
    next.addEventListener("click", () => { picker.selectedIndex++; changeDocument(); });
    slider.addEventListener("input", render);
    render();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
