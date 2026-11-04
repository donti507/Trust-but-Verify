# LLM-proposed, human-confirmed annotations

Paper disclosure: “Annotations were LLM-proposed and human-confirmed; the proposal log is released.”

`scripts/annotate_auto.py` is an optional assisted protocol. It does not run the
verifier, infer gold from model confidence, or accept a proposal automatically.
The human checks the saved document, field scope, normalised value and exact
span, then explicitly accepts or corrects each field. An ABSENT proposal also
requires acceptance. A skipped field remains unanswered, not absent. Repeated
quotes require a human occurrence selection. Unmatched or unnormalisable
proposals cannot be accepted; use manual correction (including absent/ref/amb
and document-supported rate units) or skip. Feature flags are reviewed manually.

Keep A1 and A2 isolated from each other's labels. Assisted annotations must be
reported as assisted, not unaided independent extraction: a shared model/cache
can give both annotators identical proposals. Gold is still the human's reviewed
decision; the model response alone is never an annotation.

The default backend is an already-installed Ollama model on loopback. Nothing
installs or downloads a model. `--model openai:<model-id>` explicitly selects the
OpenAI-compatible Chat Completions API and sends the full document; it reads OPENAI_API_KEY and OPENAI_BASE_URL only
for an uncached request. No credentials are persisted. Operators must choose a
backend suitable for their documents before running it.

Step 0 Cache keys include model identity, endpoint/settings, prompt version,
field and full document text. Successful responses and failed attempts are
cached, so rerunning the same input does not repeat a model call. A per-key lock
prevents concurrent duplicate calls; after a killed process, inspect a stale
`.lock` before removing it. Model changes require a distinct model identity.
Changing input/settings deliberately creates a new cache key. Dry-run populates
this cache but writes neither annotations nor decision logs.

Annotations use the existing `A1.jsonl` format. `A1.proposals.jsonl` is a separate
provenance event log; `agreement.py` needs no changes. Each event contains doc_id,
annotator, field, source text path/hash, model/settings, prompt version/cache key,
raw proposal, normalised proposal, candidate spans/errors, human decision,
final value/annotation and exact final quote. Phases distinguish attempted,
skipped, cancelled, interrupted, confirmed and saved decisions. Count final
saved decisions, not every event, when summarising the log. No timestamps are
required. Source hashes link events to the saved text used for span lookup.

A confirmed event is durably written before the annotation snapshot. If saving
the snapshot fails, a rerun recovers only that already-human-confirmed decision.
Source changes and concurrent file edits abort instead of reusing stale spans
or overwriting another session. Existing annotations are retained; answered
fields are not proposed again. Skips persist unless `--revisit-skipped` is used.

Release the provenance logs alongside the benchmark and its source-text
identifiers when publishing; this tool does not publish artifacts. Keep runtime
data, caches, logs and credentials out of git. For unaided annotations continue
to use `annotate.py`; choose the study protocol before starting annotation.
