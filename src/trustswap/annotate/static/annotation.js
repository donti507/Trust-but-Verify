"use strict";

const source = document.querySelector("#source-text");
const csrfToken = document.querySelector('meta[name="csrf-token"]').content;
const state = document.querySelector("#editor-data");

function codepointSpan() {
  if (!source || source.selectionStart === source.selectionEnd) return null;
  return [source.selectionStart, source.selectionEnd].map(index => Array.from(source.value.slice(0, index)).length);
}

async function post(url, payload, keepalive = false) {
  const body = JSON.stringify(payload);
  const response = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json", "X-CSRF-Token": csrfToken},
    body,
    keepalive: keepalive && new TextEncoder().encode(body).length < 60_000,
    credentials: "same-origin",
  });
  let result;
  try {
    result = await response.json();
  } catch {
    throw new Error("Reload the page before saving again.");
  }
  if (!response.ok) {
    const error = new Error(result.error);
    error.status = response.status;
    throw error;
  }
  return result;
}

function showMessage(element, text) {
  element.textContent = text;
  element.hidden = !text;
}

function startEditor(view) {
  const editor = document.querySelector("#editor");
  const structure = document.querySelector("#structure");
  const progress = document.querySelector("#field-progress");
  const count = document.querySelector("#progress-count");
  const savedState = document.querySelector("#saved-state");
  const errorMessage = document.querySelector("#editor-error");
  const submitButton = document.querySelector("#submit-labels");
  const readOnly = view.stage === "final";
  const fields = Array.from(editor.querySelectorAll("[data-path]"), card => ({
    card,
    path: card.dataset.path,
    heading: card.querySelector("[data-heading]"),
    status: card.querySelector("[data-status]"),
    value: card.querySelector("[data-value]"),
    note: card.querySelector("[data-note]"),
    confirmed: card.querySelector("[data-confirmed]"),
    rechecked: card.querySelector("[data-rechecked]"),
    reason: card.querySelector("[data-reason]"),
    spanButton: card.querySelector("[data-show-span]"),
    badge: card.querySelector("[data-unverified]"),
    offset: null,
  }));
  let active = null;
  let revision = view.revision;
  let saveTimer;
  let stopped = false;
  let submitting = false;
  let chain = Promise.resolve();
  let since = document.hidden ? null : performance.now();
  let measured = 0;
  const flags = Array.from(editor.querySelectorAll("[data-flag]"));
  const flagsReviewed = document.querySelector("#features-reviewed");
  structure.value = view.form.structure;
  flagsReviewed.checked = view.form.flags_reviewed;

  function appearance(field) {
    const hasValue = field.status.value === "explicit" || !field.status.value;
    field.card.querySelector("[data-value-label]").hidden = !hasValue;
    field.spanButton.hidden = !field.offset || !hasValue;
    field.badge.hidden = !(view.stage === "draft" && view.suggestions[field.path] && !field.confirmed?.checked);
  }

  for (const flag of flags) flag.checked = view.form.flags.includes(flag.dataset.flag);
  for (const field of fields) {
    const answer = view.form.fields[field.path] || view.suggestions[field.path];
    if (answer) {
      field.status.value = answer.status;
      field.value.value = answer.value;
      field.note.value = answer.note;
      field.offset = answer.span;
      if (field.confirmed) field.confirmed.checked = Boolean(answer.confirmed);
    }
    if (field.reason) field.reason.value = view.form.reasons[field.path] || "";
    if (field.rechecked) field.rechecked.checked = view.form.reviewed.includes(field.path);
    appearance(field);
  }

  function shown() {
    return fields.filter(field => !field.card.hidden);
  }

  function setActive(field, focus = false) {
    if (active) active.card.classList.remove("active");
    active = field;
    if (field) {
      field.card.classList.add("active");
      if (focus) field.heading.focus();
    }
  }

  function showFields() {
    const expected = new Set(view.paths[structure.value] || []);
    for (const field of fields) field.card.hidden = !expected.has(field.path);
    for (const group of editor.querySelectorAll("[data-group]")) {
      group.hidden = !group.querySelector("[data-path]:not([hidden])");
    }
    if (!active || active.card.hidden) {
      setActive(shown().find(field => field.path === view.form.cursor) || shown()[0]);
    }
    showProgress();
  }

  function snapshot() {
    const current = {structure: structure.value, fields: {}, cursor: active?.path || "",
      flags: flags.filter(flag => flag.checked).map(flag => flag.dataset.flag),
      flags_reviewed: flagsReviewed.checked, reviewed: [], reasons: {}};
    for (const field of shown()) {
      const explicit = field.status.value === "explicit";
      current.fields[field.path] = {status: field.status.value, value: explicit ? field.value.value : "",
        span: explicit ? field.offset : null, note: field.note.value, confirmed: Boolean(field.confirmed?.checked)};
      if (field.rechecked?.checked) current.reviewed.push(field.path);
      if (field.reason?.value.trim()) current.reasons[field.path] = field.reason.value;
    }
    return current;
  }

  function showProgress() {
    const visible = shown();
    const done = visible.filter(field => {
      const status = field.status.value;
      const complete = status === "explicit" ? field.value.value.trim() && field.offset : status && field.note.value.trim();
      return complete && (view.mode !== "prefill" || field.confirmed?.checked);
    }).length;
    progress.max = visible.length || 1;
    progress.value = done;
    count.textContent = done + " of " + visible.length + " fields answered";
  }

  function tick() {
    if (since !== null) measured += performance.now() - since;
    since = document.hidden ? null : performance.now();
  }

  function queueSave(submit = false) {
    if (readOnly || stopped || (!submit && submitting)) return;
    clearTimeout(saveTimer);
    chain = chain.then(async () => {
      if (stopped) return;
      tick();
      const delta = Math.min(Math.round(measured), 300_000);
      measured = 0;
      try {
        const result = await post(view.save, {form: snapshot(), revision, delta, submit}, !submit);
        revision = result.revision;
        savedState.textContent = "Saved locally";
        if (submit) {
          stopped = true;
          location.reload();
        }
      } catch (error) {
        measured += delta;
        savedState.textContent = error.status === 409 ? "Newer draft found" : "Save pending";
        if (error.status === 409) stopped = true;
        if (submit || stopped) {
          showMessage(errorMessage, error.message);
          errorMessage.scrollIntoView({block: "nearest"});
        }
      } finally {
        if (submit && !stopped) {
          submitting = false;
          submitButton.disabled = false;
        }
      }
    });
  }

  function edited() {
    showProgress();
    if (readOnly || stopped || submitting) return;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => queueSave(), 500);
  }

  async function useSelection() {
    const field = active;
    const offset = codepointSpan();
    if (readOnly || stopped || submitting || !field || !offset) return;
    if (field.selecting) return;
    field.selecting = true;
    try {
      const result = await post(view.selection, {path: field.path, span: offset});
      field.offset = offset;
      field.value.value = result.value;
      field.status.value = "explicit";
      if (field.confirmed) field.confirmed.checked = false;
      if (field.rechecked) field.rechecked.checked = false;
      appearance(field);
      showMessage(errorMessage, "");
      edited();
    } catch (error) {
      showMessage(errorMessage, error.message);
    } finally {
      field.selecting = false;
    }
  }

  for (const field of fields) {
    field.card.addEventListener("focusin", () => { setActive(field); if (!readOnly) edited(); });
    field.card.addEventListener("click", () => setActive(field));
    field.card.addEventListener("input", event => {
      if (event.target === field.value && !field.status.value) field.status.value = "explicit";
      if (event.target === field.status && field.status.value !== "explicit") {
        field.offset = null;
        field.value.value = "";
      }
      if ([field.value, field.status, field.note].includes(event.target)) {
        if (field.confirmed) field.confirmed.checked = false;
        if (field.rechecked) field.rechecked.checked = false;
      }
      appearance(field);
      edited();
    });
    field.spanButton.addEventListener("click", () => {
      if (!field.offset) return;
      const positions = field.offset.map(index => Array.from(source.value).slice(0, index).join("").length);
      source.focus();
      source.setSelectionRange(...positions);
    });
    field.card.addEventListener("keydown", event => {
      if (event.key === "Tab" && event.target === field.heading) {
        const ordered = shown();
        const next = ordered[ordered.indexOf(field) + (event.shiftKey ? -1 : 1)];
        if (next) { event.preventDefault(); setActive(next, true); }
      }
      if (!readOnly && event.key.toLowerCase() === "a" && (event.target === field.heading)) {
        event.preventDefault();
        field.status.value = "absent";
        field.status.dispatchEvent(new Event("input", {bubbles: true}));
        field.note.focus();
      }
    });
    if (readOnly) {
      for (const control of [field.status, field.value, field.note, field.confirmed, field.reason, field.rechecked]) {
        if (control) control.disabled = true;
      }
    }
  }

  structure.addEventListener("change", () => { showFields(); edited(); });
  for (const control of [...flags, flagsReviewed]) control.addEventListener("change", edited);
  showFields();
  if (active && view.form.cursor) active.heading.scrollIntoView({block: "nearest"});
  savedState.textContent = readOnly ? "Finalized" : revision ? "Saved locally" : "";
  editor.addEventListener("submit", event => {
    event.preventDefault();
    if (readOnly || submitting || stopped) return;
    submitting = true;
    submitButton.disabled = true;
    savedState.textContent = "Submitting…";
    queueSave(true);
  });
  if (readOnly) return;
  source.addEventListener("pointerup", useSelection);
  source.addEventListener("keyup", event => { if (event.shiftKey) useSelection(); });
  document.querySelector("#use-selection").addEventListener("click", useSelection);
  document.addEventListener("visibilitychange", () => { tick(); if (document.hidden) queueSave(); });
  window.addEventListener("pagehide", () => { tick(); queueSave(); });
  setInterval(() => { if (!document.hidden) queueSave(); }, 15_000);
}

function startAdjudication() {
  const cards = document.querySelectorAll("[data-adjudication]");
  if (!cards.length) return;
  const error = document.querySelector("#adjudication-error");
  let active = null;
  for (const card of cards) {
    card.addEventListener("focusin", () => {
      if (active) active.classList.remove("active");
      active = card;
      card.classList.add("active");
    });
    card.elements.choice.addEventListener("change", () => {
      const manual = card.querySelector("[data-manual]");
      if (manual) manual.hidden = card.elements.choice.value !== "manual";
    });
  }
  document.querySelector("#use-selection").addEventListener("click", async () => {
    const offset = codepointSpan();
    if (!offset || !active?.querySelector("[data-manual]")) return;
    const card = active;
    try {
      const result = await post(card.dataset.adjudication, {path: card.dataset.path, span: offset});
      card.elements.choice.value = "manual";
      card.elements.choice.dispatchEvent(new Event("change"));
      card.elements.status.value = "explicit";
      card.elements.value.value = result.value;
      card.elements.start.value = offset[0];
      card.elements.end.value = offset[1];
      showMessage(error, "");
    } catch (exception) {
      showMessage(error, exception.message);
      error.scrollIntoView();
    }
  });
}

if (state) startEditor(JSON.parse(state.textContent));
else startAdjudication();
