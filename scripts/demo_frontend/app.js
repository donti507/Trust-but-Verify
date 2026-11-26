"use strict";

function decision(p, c_bar, r, lambda) {
  return p * c_bar > lambda * r ? "Review" : "Accept";
}

const lambdaFor = (step) => (step === 0 ? 0 : 10 ** ((step - 31) / 10));
const pct = (value) =>
  (100 * value).toLocaleString(undefined, { maximumFractionDigits: 1 }) + "%";
const lossPct = (value) =>
  (100 * value).toLocaleString(undefined, { maximumFractionDigits: 2 }) + "%";
function facts(record, threshold, bound) {
  let accepted = 0,
    caught = 0,
    missed = 0,
    actual = 0,
    estimated = 0;
  for (const field of record.fields) {
    if (decision(field.p, field.c_bar, field.r, threshold) === "Review") {
      caught += field.e;
    } else {
      accepted++;
      missed += field.e;
      actual += field.e * field.c_bar;
      estimated += field.p * field.c_bar;
    }
  }
  return {
    record,
    accepted,
    total: record.fields.length,
    reviewed: record.fields.length - accepted,
    caught,
    missed,
    observed: Math.min(actual, bound) / bound,
    estimated: Math.min(estimated, bound) / bound,
  };
}

function aggregate(rows) {
  const sums = {
    accepted: 0,
    total: 0,
    caught: 0,
    missed: 0,
    observed: 0,
    estimated: 0,
  };
  for (const row of rows) for (const name in sums) sums[name] += row[name];
  return {
    ...sums,
    automation: sums.accepted / sums.total,
    observed: sums.observed / rows.length,
    estimated: sums.estimated / rows.length,
  };
}

function buildCurve(documents, bound) {
  return Array.from({ length: 62 }, (_, step) =>
    aggregate(documents.map((doc) => facts(doc, lambdaFor(step), bound))),
  );
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { decision, facts, aggregate, lambdaFor, buildCurve, pct };
}
(function () {
  if (typeof document === "undefined") return;
  const byId = (id) => document.getElementById(id);
  const data = JSON.parse(byId("demo-data").textContent);
  if (data.illustrative) byId("sample-label").hidden = false;
  const picker = byId("document-picker");
  const slider = byId("appetite");
  const collection = byId("collection");
  const annotations = new Map();
  const auditEvents = [];
  const pageSize = 5;
  const key = (record, index) => JSON.stringify([record.id, index]);
  const cohort = (record) =>
    record.collection === "live_feed"
      ? "Live feed"
      : record.category === "Adversarial"
        ? "Adversarial"
        : "Synthetic";
  const lambda = () => lambdaFor(Number(slider.value));
  const number = (value) => Number(value.toPrecision(4)).toString();
  const valueText = (value) =>
    value == null
      ? "Missing"
      : Array.isArray(value)
        ? value.join(", ")
        : value.toLocaleString();
  let selectedDoc = data.documents[0];
  let selected = -1;
  let filter = "all";
  let page = 0;
  let stage = 0;
  let assurance = "flow";
  let curve = [];
  const feed = data.feed || { days: [], history: [] };
  let feedDay = feed.days.at(-1);

  function element(tag, text = "", className = "") {
    const node = document.createElement(tag);
    node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  function button(text, action, className) {
    const node = element("button", text, className);
    node.type = "button";
    node.addEventListener("click", action);
    return node;
  }

  function records() {
    return data.documents.filter(
      (record) => cohort(record) === collection.value,
    );
  }

  function fieldDecision(
    field,
    threshold = selectedDoc.collection === "live_feed"
      ? feedDay.threshold
      : lambda(),
  ) {
    return decision(field.p, field.c_bar, field.r, threshold);
  }

  function docFacts() {
    return records().map((record) => facts(record, lambda(), data.bound));
  }

  function rebuildCurve() {
    curve = buildCurve(records(), data.bound);
  }

  function renderText(record, decisions) {
    // Python span offsets count Unicode code points, rather than UTF-16 code units.
    const characters = Array.from(record.text);
    const bounds = new Set([0, characters.length]);
    record.fields.forEach((field) => {
      if (field.span) field.span.forEach((offset) => bounds.add(offset));
    });
    const positions = Array.from(bounds).sort((a, b) => a - b);
    const fragment = document.createDocumentFragment();
    // Fields sent to review get a callout box after the line that holds them.
    const pending = [];
    const flagged = new Set();
    function callout(index) {
      const field = record.fields[index];
      const box = element("button", "", "callout");
      box.type = "button";
      const reason = field.rules_fired.length
        ? field.rules_fired.map((rule) => data.rules[rule]).join(" ")
        : "High estimated error risk for a " + field.severity + " field.";
      box.append(
        element("span", "Sent to review", "callout-tag"),
        element("strong", field.name),
        element("span", reason + " Risk " + pct(field.p) + "."),
      );
      if (selected === index) box.classList.add("selected");
      box.addEventListener("click", () => chooseField(index, false));
      return box;
    }
    function plain(text) {
      const cut = pending.length ? text.indexOf("\n") : -1;
      if (cut < 0) {
        fragment.appendChild(document.createTextNode(text));
        return;
      }
      fragment.appendChild(document.createTextNode(text.slice(0, cut + 1)));
      pending.splice(0).forEach((index) => fragment.appendChild(callout(index)));
      fragment.appendChild(document.createTextNode(text.slice(cut + 1)));
    }
    for (let i = 1; i < positions.length; i++) {
      const start = positions[i - 1];
      const end = positions[i];
      const text = characters.slice(start, end).join("");
      const matches = [];
      record.fields.forEach((field, index) => {
        if (field.span && field.span[0] <= start && end <= field.span[1])
          matches.push(index);
      });
      matches.forEach((index) => {
        if (decisions[index] === "Review" && !flagged.has(index)) {
          flagged.add(index);
          pending.push(index);
        }
      });
      if (!matches.length) {
        plain(text);
        continue;
      }
      // For overlapping spans, show the selected field or prefer a review highlight.
      const review = matches.find((index) => decisions[index] === "Review");
      const index = matches.includes(selected)
        ? selected
        : (review ?? matches[0]);
      const mark = element(
        "button",
        text,
        "mark " + decisions[index].toLowerCase(),
      );
      if (selected === index) mark.classList.add("selected");
      mark.type = "button";
      mark.setAttribute(
        "aria-label",
        record.fields[index].name + ": " + decisions[index] + ", " + text,
      );
      mark.addEventListener("click", () => chooseField(index, false));
      fragment.appendChild(mark);
    }
    pending.splice(0).forEach((index) => fragment.appendChild(callout(index)));
    byId("document-text").replaceChildren(fragment);
    const callouts = byId("source-callouts");
    callouts.replaceChildren();
    record.fields.forEach((field, index) => {
      if (decisions[index] !== "Review") return;
      const item = element("li", "", "source-callout");
      item.append(
        element("strong", "Sent to review"),
        button(field.name, () => chooseField(index), "table-link"),
      );
      if (!field.span) item.append(element("span", "No source span"));
      callouts.append(item);
    });
    callouts.hidden = !callouts.childElementCount;
  }

  function drawChart() {
    const svg = byId("frontier");
    const kind = byId("loss-kind").value;
    const budgetInput = byId("budget");
    const budget = Math.max(
      0,
      Math.min(1, Number(budgetInput.value) / 100 || 0),
    );
    const maxLoss = Math.min(
      1,
      Math.max(0.005, budget, ...curve.map((point) => point[kind])) * 1.12,
    );
    const x = (share) => 65 + share * 526;
    const y = (loss) => 218 - (loss / maxLoss) * 190;
    const current = curve[Number(slider.value)];
    const svgNode = (tag, attrs, text = "") => {
      const node = document.createElementNS(svg.namespaceURI, tag);
      for (const [name, value] of Object.entries(attrs))
        node.setAttribute(name, value);
      node.textContent = text;
      return node;
    };
    svg.replaceChildren(
      svgNode(
        "title",
        { id: "frontier-title" },
        "Automation and loss by risk threshold",
      ),
    );
    const append = (tag, attrs, text) =>
      svg.appendChild(svgNode(tag, attrs, text));
    const line = (x1, y1, x2, y2, stroke, extra = {}) =>
      append("line", { x1, y1, x2, y2, stroke, ...extra });
    const label = (value, x, y, fill = "#405575", anchor = "end") =>
      append(
        "text",
        { x, y, fill, "text-anchor": anchor, "font-size": 12 },
        value,
      );
    const yBudget = y(budget);
    append("rect", {
      x: x(0),
      y: yBudget,
      width: x(1) - x(0),
      height: y(0) - yBudget,
      fill: "#d9eee7",
    });
    for (let i = 0; i <= 4; i++) {
      const yy = y((maxLoss * i) / 4);
      line(x(0), yy, x(1), yy, "#d5def2");
      label(lossPct((maxLoss * i) / 4), 57, yy + 4);
      label(pct(i / 4), x(i / 4), 241, "#405575", "middle");
    }
    line(x(0), yBudget, x(1), yBudget, "#b43a39", {
      "stroke-dasharray": "5 4",
    });
    label(
      "Budget " + lossPct(budget),
      x(1) - 4,
      Math.max(22, yBudget - 6),
      "#963333",
    );
    append("polyline", {
      points: curve
        .map((point) => x(point.automation) + "," + y(point[kind]))
        .join(" "),
      fill: "none",
      stroke: "#2563eb",
      "stroke-width": 3,
      "stroke-linejoin": "round",
    });
    const px = x(current.automation),
      py = y(current[kind]);
    line(px, py, px, y(0), "#5278c5", { "stroke-dasharray": "3 4" });
    append("circle", {
      cx: px,
      cy: py,
      r: 6,
      fill: "#004ac6",
      stroke: "#fff",
      "stroke-width": 2,
    });
    byId("chart-status").textContent =
      pct(current.automation) +
      " automated · " +
      lossPct(current[kind]) +
      " " +
      kind +
      " loss · " +
      (current[kind] <= budget ? "within budget" : "above budget");
  }

  function matchesFilter(row, type) {
    if (type === "review") return row.accepted < row.total;
    if (type === "straight") return row.accepted === row.total;
    if (type === "exceptions") return row.missed > 0;
    return true;
  }

  function tradeStatus(row) {
    if (row.missed) return ["Error slipped through", "exception"];
    if (row.accepted < row.total)
      return [row.total - row.accepted + " to review", "review"];
    return ["Straight-through", "accept"];
  }

  function reviewDocuments() {
    return selectedDoc.collection === "live_feed"
      ? feedDay.documents
      : records();
  }

  function fillPicker(docs) {
    picker.replaceChildren(
      ...docs.map((record) => {
        const option = element("option", record.id + " · " + record.category);
        option.value = record.id;
        return option;
      }),
    );
    picker.value = selectedDoc.id;
  }

  function renderFeed() {
    const date = byId("feed-date");
    date.replaceChildren(
      ...feed.days.map((day) => element("option", day.date)),
    );
    date.disabled = !feedDay;
    date.value = feedDay?.date ?? "";
    const body = byId("feed-queue");
    body.replaceChildren();
    byId("feed-summary").textContent = feedDay
      ? "Source day " + feedDay.source_date + " · λ " + feedDay.threshold
      : "No published feed yet";
    for (const record of feedDay?.documents ?? []) {
      const row = element("tr");
      const action = element("td");
      action.append(
        button("Review trade", () => showDocument(record), "primary"),
      );
      const [status, tone] = tradeStatus(
        facts(record, feedDay.threshold, data.bound),
      );
      const badge = element("td");
      badge.append(element("span", status, "pill " + tone));
      row.append(
        element("th", record.id),
        element("td", record.defect?.replaceAll("_", " ") ?? "None"),
        badge,
        action,
      );
      body.append(row);
    }
    const svg = byId("feed-history");
    svg.replaceChildren();
    if (!feed.history.length) return;
    const x = (i) => 48 + (i * 548) / Math.max(1, feed.history.length - 1);
    const y = (value) => 145 - 1.25 * value;
    const draw = (tag, attrs, value = "") => {
      const node = document.createElementNS(svg.namespaceURI, tag);
      for (const [name, content] of Object.entries(attrs))
        node.setAttribute(name, content);
      node.textContent = value;
      svg.append(node);
    };
    for (const value of [0, 25, 50, 75, 100]) {
      draw("line", {
        x1: 48,
        y1: y(value),
        x2: 596,
        y2: y(value),
        stroke: "#dde5f4",
      });
      draw(
        "text",
        { x: 4, y: y(value) + 4, fill: "#405575", "font-size": 12 },
        value + "%",
      );
    }
    for (const [color, value] of [
      ["#004ac6", (row) => row.automation_pct],
      ["#006243", (row) => (100 * row.defects_caught) / row.defects_planted],
    ]) {
      draw("polyline", {
        points: feed.history
          .map((row, i) => x(i) + "," + y(value(row)))
          .join(" "),
        fill: "none",
        stroke: color,
        "stroke-width": 3,
      });
      feed.history.forEach((row, i) =>
        draw("circle", { cx: x(i), cy: y(value(row)), r: 3, fill: color }),
      );
    }
    const last = feed.history.length - 1;
    [0, ...(last ? [last] : [])].forEach((i) =>
      draw(
        "text",
        {
          x: x(i),
          y: 169,
          fill: "#405575",
          "font-size": 12,
          "text-anchor": i ? "end" : "start",
        },
        feed.history[i].date,
      ),
    );
  }

  function showDocument(record) {
    selectedDoc = record;
    if (
      record.collection !== "live_feed" &&
      collection.value !== cohort(record)
    )
      changeCollection(cohort(record), record);
    selected = -1;
    fillPicker(reviewDocuments());
    if (location.hash === "#confirmations") renderReview();
    else location.hash = "confirmations";
  }

  function renderQueue(rows) {
    const query = byId("search").value.trim().toLocaleLowerCase();
    const matches = rows.filter(({ record }) => {
      return (record.id + " " + record.text)
        .toLocaleLowerCase()
        .includes(query);
    });
    document.querySelectorAll("[data-filter]").forEach((control) => {
      const type = control.dataset.filter;
      control.setAttribute("aria-pressed", String(type === filter));
      control.querySelector("span").textContent = matches.filter((row) =>
        matchesFilter(row, type),
      ).length;
    });
    const filtered = matches.filter((row) => matchesFilter(row, filter));
    page = Math.min(
      page,
      Math.max(0, Math.ceil(filtered.length / pageSize) - 1),
    );
    const start = page * pageSize;
    const displayed = filtered.slice(start, start + pageSize);
    const body = byId("queue");
    body.replaceChildren();
    for (const row of displayed) {
      const { record } = row;
      const tr = document.createElement("tr");
      const trade = element("th");
      trade.scope = "row";
      trade.append(button(record.id, () => showDocument(record), "table-link"));
      const notional = record.fields.find((field) =>
        field.name.startsWith("Notional amount"),
      );
      const status = element("td");
      const [name, tone] = tradeStatus(row);
      status.append(element("span", name, "pill " + tone));
      const progress = element(
        "td",
        row.accepted +
          " of " +
          row.total +
          " · " +
          pct(row.accepted / row.total),
      );
      const bar = element("span", "", "bar");
      const fill = element("span");
      fill.style.width = pct(row.accepted / row.total);
      bar.append(fill);
      progress.append(bar);
      const action = element("td");
      action.append(
        button(
          row.accepted < row.total ? "Review trade" : "View trade",
          () => showDocument(record),
          row.accepted < row.total ? "primary" : "",
        ),
      );
      tr.append(
        trade,
        element("td", record.category),
        element("td", valueText(notional?.value)),
        status,
        progress,
        action,
      );
      body.append(tr);
    }
    if (!displayed.length) {
      const tr = document.createElement("tr");
      const cell = element(
        "td",
        "No confirmations match this filter.",
        "empty",
      );
      cell.colSpan = 6;
      tr.append(cell);
      body.append(tr);
    }
    byId("queue-count").textContent = displayed.length
      ? "Showing " +
        (start + 1) +
        "–" +
        (start + displayed.length) +
        " of " +
        filtered.length +
        " confirmations"
      : "0 confirmations";
    byId("queue-prev").disabled = page === 0;
    byId("queue-next").disabled = start + pageSize >= filtered.length;
  }

  function renderOverview() {
    const rows = docFacts();
    const stats = aggregate(rows);
    const errors = stats.caught + stats.missed;
    byId("threshold").textContent = "λ " + number(lambda());
    slider.setAttribute(
      "aria-valuetext",
      "Threshold " +
        number(lambda()) +
        ", " +
        pct(stats.automation) +
        " automated",
    );
    byId("auto-rate").textContent = pct(stats.automation);
    byId("review-rate").textContent = pct(1 - stats.automation);
    byId("caught-rate").textContent = errors ? pct(stats.caught / errors) : "—";
    byId("loss-rate").textContent = lossPct(stats.observed);
    byId("automated").textContent =
      stats.accepted + " of " + stats.total + " fields accepted";
    byId("reviewed").textContent =
      stats.total - stats.accepted + " fields flagged for review";
    byId("caught").textContent =
      stats.caught + " of " + errors + " known errors sent to review";
    byId("missed").textContent =
      stats.missed +
      (stats.missed === 1 ? " known error slipped through" : " known errors slipped through");
    byId("policy-auto").textContent = pct(stats.automation);
    byId("policy-estimated").textContent = lossPct(stats.estimated);
    drawChart();
    renderQueue(rows);
  }

  function chooseField(index, showSource = true) {
    selected = index;
    renderReview();
    byId("field-" + index)?.focus({ preventScroll: true });
    if (showSource) {
      const highlight = byId("document-text").querySelector(".selected");
      highlight?.scrollIntoView?.({ block: "nearest" });
    }
  }

  function renderExplanation(field, action) {
    const panel = byId("explanation");
    panel.replaceChildren();
    panel.classList.toggle("needs-review", action === "Review");
    if (!field) return;
    const detailRow = element("div", "", "section-top");
    detailRow.append(
      element(
        "span",
        action === "Review" ? "Human review" : "Accepted by policy",
        "pill " + action.toLowerCase(),
      ),
      element("span", pct(field.p) + " risk", "pill blue"),
    );
    const title =
      action === "Review"
        ? "Why this needs a person"
        : "Why the policy accepts this field";
    panel.append(detailRow, element("h3", title + " — " + field.name));
    const list = document.createElement("ul");
    for (const rule of field.rules_fired) {
      list.append(
        element("li", rule + " · " + (data.rules[rule] ?? "Verifier warning.")),
      );
    }
    if (!list.childNodes.length)
      list.append(element("li", "No verifier warnings for this field."));
    list.append(
      element(
        "li",
        "Cost if wrong: " +
          field.severity +
          ". Threshold λ " +
          number(
            selectedDoc.collection === "live_feed"
              ? feedDay.threshold
              : lambda(),
          ) +
          ".",
      ),
    );
    panel.append(list);
    if (field.span) {
      const segment = Array.from(selectedDoc.text)
        .slice(...field.span)
        .join("");
      const cite = element("p", "Source evidence: " + segment, "notice");
      panel.append(cite);
    }
    const state = annotations.get(key(selectedDoc, selected));
    const note = element("textarea", state?.note ?? "");
    note.id = "review-note";
    note.rows = 2;
    const noteLabel = element("label", "Reviewer note (optional)");
    noteLabel.htmlFor = note.id;
    const corrected = element("input");
    corrected.id = "corrected-value";
    corrected.value = state?.corrected ?? "";
    const correctedLabel = element("label", "Proposed correction");
    correctedLabel.htmlFor = corrected.id;
    const actions = element("div", "", "actions");
    const feedback = element(
      "p",
      state
        ? state.action + " recorded in this session."
        : "Annotations stay in the session and are included in the audit export.",
      "detail-status",
    );
    feedback.setAttribute("role", "status");
    function save(name) {
      if (name === "Correction" && !corrected.value.trim()) {
        feedback.textContent = "Enter a proposed correction first.";
        corrected.focus();
        return;
      }
      const annotation = {
        action: name,
        note: note.value.trim(),
        corrected: name === "Correction" ? corrected.value.trim() : "",
      };
      annotations.set(key(selectedDoc, selected), annotation);
      auditEvents.push({
        trade_id: selectedDoc.id,
        collection: cohort(selectedDoc),
        field: field.field,
        lambda: lambda(),
        policy_decision: action,
        ...annotation,
      });
      feedback.textContent = name + " recorded in this session.";
    }
    actions.append(
      button("Confirm extraction", () => save("Confirmed"), "primary"),
      button("Record correction", () => save("Correction")),
    );
    panel.append(noteLabel, note, correctedLabel, corrected, actions, feedback);
  }

  function fieldGroup(field) {
    if (field.name.includes("(fixed leg)")) return "Fixed leg terms";
    if (field.name.includes("floating leg)")) return "Floating leg terms";
    if (/date/i.test(field.name)) return "Trade dates";
    return "Parties and settlement";
  }

  function renderReview() {
    const record = selectedDoc;
    const docs = reviewDocuments();
    const position = docs.indexOf(record);
    const row = facts(
      record,
      record.collection === "live_feed" ? feedDay.threshold : lambda(),
      data.bound,
    );
    const decisions = record.fields.map((field) => fieldDecision(field));
    if (selected < 0) {
      selected = decisions.indexOf("Review");
      if (selected < 0) selected = 0;
    }
    const [status, tone] = tradeStatus(row);
    byId("review-status").textContent = status;
    byId("review-status").className = "pill " + tone;
    byId("document-position").textContent =
      "Document " + (position + 1) + " of " + docs.length;
    byId("cleared").textContent =
      row.accepted +
      " of " +
      row.total +
      " fields automated · " +
      (row.total - row.accepted) +
      " pending policy review · " +
      row.caught +
      (row.caught === 1 ? " known error caught" : " known errors caught");
    byId("previous").disabled = position <= 0;
    byId("next").disabled = position >= docs.length - 1;
    const summary = byId("trade-summary");
    summary.replaceChildren();
    for (const [name, value] of [
      ["Trade ID", record.id],
      ["Case type", record.category],
      [
        "Extracted notional",
        valueText(
          record.fields.find((field) =>
            field.name.startsWith("Notional amount"),
          )?.value,
        ),
      ],
      [
        "Trade date",
        valueText(
          record.fields.find((field) => field.name === "Trade date")?.value,
        ),
      ],
    ]) {
      const item = element("div");
      item.append(element("dt", name), element("dd", value));
      const list = document.createElement("dl");
      list.append(item);
      summary.append(list);
    }
    const groups = new Map();
    const fieldsPanel = byId("fields");
    const closed = new Set(
      [...fieldsPanel.querySelectorAll("details:not([open])")].map(
        (group) => group.dataset.group,
      ),
    );
    fieldsPanel.replaceChildren();
    record.fields.forEach((field, index) => {
      const name = fieldGroup(field);
      if (!groups.has(name)) groups.set(name, []);
      groups.get(name).push(index);
    });
    for (const [name, indices] of groups) {
      const reviewCount = indices.filter(
        (index) => decisions[index] === "Review",
      ).length;
      const group = element("details", "", "panel group");
      group.dataset.group = name;
      group.open = !closed.has(name) || indices.includes(selected);
      const heading = element("summary", name);
      heading.append(
        element(
          "span",
          reviewCount
            ? reviewCount + " to review"
            : indices.length + " accepted",
          "pill " + (reviewCount ? "review" : "accept"),
        ),
      );
      group.append(heading);
      for (const index of indices) {
        const field = record.fields[index];
        const action = decisions[index];
        const control = button("", () => chooseField(index), "field");
        control.id = "field-" + index;
        control.setAttribute("aria-pressed", String(index === selected));
        const info = element("span", field.name);
        info.append(
          element(
            "small",
            pct(field.p) + " risk · " + field.severity + " cost",
          ),
        );
        control.append(
          info,
          element("span", valueText(field.value), "field-value"),
          element("span", action, "pill " + action.toLowerCase()),
        );
        group.append(control);
      }
      fieldsPanel.append(group);
    }
    renderText(record, decisions);
    renderExplanation(record.fields[selected], decisions[selected]);
  }

  const stageInfo = [
    [
      "Read",
      "Extract candidate trade terms and retain a span for source evidence.",
    ],
    [
      "Check",
      "Evaluate deterministic verifier rules and record their warnings.",
    ],
    [
      "Score",
      "Estimate field error probability from cached model and verifier signals.",
    ],
    [
      "Decide",
      "Send a field to review when estimated error cost exceeds the threshold times review cost.",
    ],
    [
      "Guarantee",
      "A separately calibrated conformal procedure can certify a loss target when enough calibration data are available.",
    ],
  ];

  function renderMethod() {
    const stages = byId("stages");
    stages.replaceChildren();
    stageInfo.forEach(([title, description], index) => {
      const tile = button(
        "",
        () => {
          stage = index;
          renderMethod();
          byId("stages").children[index].focus();
        },
        "stage",
      );
      tile.setAttribute("aria-pressed", String(index === stage));
      tile.append(
        element("span", String(index + 1), "stage-number"),
        element("strong", title),
        element("p", description),
      );
      stages.append(tile);
    });
    const record =
      selectedDoc.collection === "live_feed" ? records()[0] : selectedDoc;
    const field =
      record.fields[selected] ??
      record.fields.find(
        (field) => fieldDecision(field, lambda()) === "Review",
      ) ??
      record.fields[0];
    const action = fieldDecision(field, lambda());
    const errorCost = field.p * field.c_bar;
    const reviewCost = lambda() * field.r;
    const comparison =
      errorCost > reviewCost ? ">" : errorCost < reviewCost ? "<" : "=";
    byId("journey-title").textContent = "One field’s journey: " + field.name;
    byId("journey-state").textContent =
      action === "Review" ? "Escalation example" : "Acceptance example";
    byId("journey-state").className = "pill " + action.toLowerCase();
    const trace = byId("trace");
    trace.replaceChildren();
    for (const [i, content] of [
      [0, "Extracted value: " + valueText(field.value)],
      [
        1,
        field.rules_fired.length
          ? field.rules_fired.map((rule) => data.rules[rule]).join(" ")
          : "No verifier warnings.",
      ],
      [2, pct(field.p) + " estimated error risk"],
      [
        3,
        "p × cost (" +
          field.p +
          " × " +
          field.c_bar +
          " = " +
          errorCost +
          ") " +
          comparison +
          " λ × review cost (" +
          lambda() +
          " × " +
          field.r +
          " = " +
          reviewCost +
          ")",
      ],
      [4, "Policy outcome: " + action],
    ]) {
      const cell = element("article");
      cell.append(element("h3", stageInfo[i][0]), element("p", content));
      trace.append(cell);
    }
    byId("stage-detail").textContent = stageInfo[stage][1];
    const stats = aggregate(docFacts());
    const metricCards = byId("method-metrics");
    metricCards.replaceChildren();
    for (const [title, value, text] of [
      [
        "Fewer manual checks",
        pct(stats.automation),
        "Share of fields accepted automatically in this collection.",
      ],
      [
        "Mistakes caught in review",
        String(stats.caught),
        "Known errors among fields sent to a person.",
      ],
      [
        "Measured risk budget",
        lossPct(stats.observed),
        "Mean observed normalized document loss at this setting.",
      ],
    ]) {
      const card = element("article", "", "panel");
      card.append(
        element("h3", title),
        element("strong", value),
        element("p", text),
      );
      metricCards.append(card);
    }
    document.querySelectorAll("[data-assurance]").forEach((control) => {
      control.setAttribute(
        "aria-pressed",
        String(control.dataset.assurance === assurance),
      );
    });
    const summary = byId("assurance");
    summary.replaceChildren();
    if (assurance === "flow") {
      summary.append(
        element("h3", "Review exceptions with source evidence"),
        element(
          "p",
          "The confirmation view shows accepted fields and highlights fields that need review.",
        ),
      );
    } else {
      summary.append(
        element("h3", "Trace each decision to the cached analysis"),
        element(
          "p",
          "The audit shows the field, its current policy decision, verifier warnings, and any reviewer annotation.",
        ),
      );
    }
    const link = element(
      "a",
      assurance === "audit" ? "Open audit trail" : "Inspect confirmation",
      "button",
    );
    link.href = assurance === "audit" ? "#audit" : "#confirmations";
    const paragraph = element("p");
    paragraph.append(link);
    summary.append(paragraph);
  }

  function auditRecords() {
    return records().flatMap((record) =>
      record.fields.map((field, index) => ({
        record,
        field,
        action: fieldDecision(field),
        annotation: annotations.get(key(record, index)),
      })),
    );
  }

  function renderAudit() {
    const rows = auditRecords();
    byId("audit-count").textContent =
      rows.length + " fields · " + collection.value;
    const body = byId("audit-rows");
    body.replaceChildren();
    for (const { record, field, action, annotation } of rows) {
      const row = element("tr");
      const id = element("th");
      id.scope = "row";
      id.append(button(record.id, () => showDocument(record), "table-link"));
      const status = element("td");
      status.append(element("span", action, "pill " + action.toLowerCase()));
      row.append(
        id,
        element("td", field.name),
        status,
        element("td", field.rules_fired.join(", ") || "—"),
        element(
          "td",
          annotation
            ? [annotation.action, annotation.corrected, annotation.note]
                .filter(Boolean)
                .join(" · ")
            : "—",
        ),
      );
      body.append(row);
    }
  }

  function download(content, name, type) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = name;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function csvCell(value) {
    const text = String(value ?? "");
    const safe = /^[\s]*[=+\-@]/.test(text) ? "'" + text : text;
    return '"' + safe.replaceAll('"', '""') + '"';
  }

  function exportAudit() {
    const rows = [
      [
        "trade_id",
        "case_type",
        "field",
        "value",
        "risk",
        "severity",
        "lambda",
        "policy_decision",
        "rules",
        "known_error",
        "review_action",
        "correction",
        "note",
      ],
      ...auditRecords().map(({ record, field, action, annotation }) => [
        record.id,
        record.category,
        field.name,
        valueText(field.value),
        field.p,
        field.severity,
        lambda(),
        action,
        field.rules_fired.join(" "),
        field.e,
        annotation?.action,
        annotation?.corrected,
        annotation?.note,
      ]),
    ];
    download(
      rows.map((row) => row.map(csvCell).join(",")).join("\r\n"),
      "trust-but-verify-audit.csv",
      "text/csv;charset=utf-8",
    );
  }

  function exportAuditReport() {
    const rows = auditRecords().map(
      ({ record, field, action, annotation }) => ({
        trade_id: record.id,
        case_type: record.category,
        field: field.name,
        value: field.value,
        p: field.p,
        c_bar: field.c_bar,
        r: field.r,
        action,
        rules_fired: field.rules_fired,
        known_error: field.e,
        annotation: annotation ?? null,
      }),
    );
    const report = {
      collection: collection.value,
      lambda: lambda(),
      records: rows,
      annotations: auditEvents.filter(
        (event) => event.collection === collection.value,
      ),
    };
    download(
      JSON.stringify(report, null, 2),
      "trust-but-verify-audit.json",
      "application/json",
    );
  }

  const views = {
    overview: renderOverview,
    confirmations: renderReview,
    method: renderMethod,
    audit: renderAudit,
    feed: renderFeed,
  };
  function route() {
    const hash = location.hash.slice(1);
    const name = Object.hasOwn(views, hash) ? hash : "overview";
    document.querySelectorAll("[data-view]").forEach((view) => {
      view.hidden = view.id !== name;
    });
    document.querySelectorAll("[data-nav]").forEach((link) => {
      if (link.dataset.nav === name) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    views[name]();
  }

  function changeCollection(value, preferred) {
    collection.value = value;
    const docs = records();
    selectedDoc = preferred ?? docs[0];
    selected = -1;
    page = 0;
    filter = "all";
    picker.replaceChildren(
      ...docs.map((record) => {
        const option = element("option", record.id + " · " + record.category);
        option.value = record.id;
        return option;
      }),
    );
    picker.value = selectedDoc.id;
    rebuildCurve();
  }

  const available = [...new Set(data.documents.map(cohort))];
  available.forEach((name) => collection.append(element("option", name)));
  changeCollection(available[0]);
  collection.addEventListener("change", () => {
    changeCollection(collection.value);
    route();
  });
  picker.addEventListener("change", () =>
    showDocument(
      reviewDocuments().find((record) => record.id === picker.value),
    ),
  );
  byId("previous").addEventListener("click", () =>
    showDocument(reviewDocuments()[reviewDocuments().indexOf(selectedDoc) - 1]),
  );
  byId("next").addEventListener("click", () =>
    showDocument(reviewDocuments()[reviewDocuments().indexOf(selectedDoc) + 1]),
  );
  slider.addEventListener("input", renderOverview);
  byId("budget").addEventListener("input", drawChart);
  byId("loss-kind").addEventListener("change", drawChart);
  document.querySelectorAll("[data-filter]").forEach((control) =>
    control.addEventListener("click", () => {
      filter = control.dataset.filter;
      page = 0;
      renderQueue(docFacts());
    }),
  );
  document.querySelectorAll("[data-assurance]").forEach((control) =>
    control.addEventListener("click", () => {
      assurance = control.dataset.assurance;
      renderMethod();
    }),
  );
  byId("queue-prev").addEventListener("click", () => {
    page--;
    renderQueue(docFacts());
  });
  byId("queue-next").addEventListener("click", () => {
    page++;
    renderQueue(docFacts());
  });
  byId("search").addEventListener("input", () => {
    page = 0;
    if (location.hash && location.hash !== "#overview")
      location.hash = "overview";
    else renderQueue(docFacts());
  });
  byId("download").addEventListener("click", () =>
    download(
      selectedDoc.text,
      selectedDoc.id + ".txt",
      "text/plain;charset=utf-8",
    ),
  );
  byId("export-overview").addEventListener("click", exportAuditReport);
  byId("export-audit").addEventListener("click", exportAudit);
  byId("feed-date").addEventListener("change", (event) => {
    feedDay = feed.days.find((day) => day.date === event.target.value);
    if (selectedDoc.collection === "live_feed") {
      selectedDoc = feedDay.documents[0];
      selected = -1;
      fillPicker(reviewDocuments());
    }
    renderFeed();
  });
  window.addEventListener("hashchange", route);
  route();
})();
