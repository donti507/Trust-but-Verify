#!/usr/bin/env python3
"""LLM proposals, human-only annotation decisions, and releasable provenance.

The default backend is a local Ollama server. No model is installed/downloaded.
Every response (including failed requests) is cached; tests supply mocked models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from trustswap.common.annotations import AnnotationRecord, FIELDS, normalise_field, validated_field
from trustswap.common.cache import Cache
from trustswap.common.hashing import call_key
if __package__:
    from . import annotate
    from .pilot_utils import read_csv, read_annotations, save_annotations, atomic_text
    from .review_helper import local_text_path, _safe
else:
    import annotate
    from pilot_utils import read_csv, read_annotations, save_annotations, atomic_text
    from review_helper import local_text_path, _safe

PROMPT_VERSION = "annotate_auto_v1"
DISCLOSURE = "Annotations were LLM-proposed and human-confirmed; the proposal log is released."


def make_prompt(field, text):
    return (f"Here is a swap confirmation. Extract the field `{field}` exactly as written in the text. "
            "Return only the value, nothing else. If not stated, return ABSENT.\n\n"
            "DOCUMENT TEXT:\n" + text)


class HTTPModel:
    """OpenAI-compatible SDK and local Ollama; credentials never enter cache keys."""
    def __init__(self, model="local:qwen2.5:7b", *, local_url="http://127.0.0.1:11434/api/generate", timeout=300):
        self.identity = model
        self.timeout = timeout
        provider, separator, self.name = model.partition(":")
        if not separator or not self.name or provider not in {"local", "ollama", "openai"}:
            raise ValueError("--model must be local:<installed-name> or openai:<model-id>")
        self.provider = provider
        if provider == "openai":
            self.endpoint = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            parsed = urlparse(self.endpoint)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.username or parsed.password or parsed.query or parsed.fragment):
                raise ValueError("OPENAI_BASE_URL must be an HTTP(S) base URL without embedded credentials")
            self.parameters = {"endpoint": self.endpoint.rstrip("/"), "api": "chat.completions", "max_tokens": 1024}
        else:
            parsed = urlparse(local_url)
            if (parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                    or parsed.username or parsed.password or parsed.query or parsed.fragment):
                raise ValueError("--local-url must be a loopback HTTP(S) endpoint without credentials")
            self.endpoint = local_url
            self.parameters = {"endpoint": local_url, "temperature": 0, "seed": 0, "num_predict": 1024}

    def __call__(self, prompt):
        headers = {"Content-Type": "application/json"}
        if self.provider == "openai":
            import openai
            # Let the SDK read OPENAI_API_KEY and OPENAI_BASE_URL itself.
            # Construct only on a cache miss, so cached runs need no credentials.
            with openai.OpenAI() as client:
                response = client.with_options(timeout=self.timeout, max_retries=0).chat.completions.create(
                    model=self.name, messages=[{"role": "user", "content": prompt}], max_tokens=1024,
                )
            if not response.choices:
                raise ValueError("frontier returned no choices")
            choice = response.choices[0]
            if choice.finish_reason == "length":
                raise ValueError("frontier output token limit reached")
            value = choice.message.content
        else:
            payload = {"model": self.name, "prompt": prompt, "stream": False,
                       "options": {key: value for key, value in self.parameters.items() if key != "endpoint"}}
            request = Request(self.endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read())
            if data.get("done_reason") == "length":
                raise ValueError("local output token limit reached")
            value = data.get("response")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("model returned no text value")
        return value


def cached_proposal(cache, model, field, text):
    prompt = make_prompt(field, text)
    key = call_key(model.identity, PROMPT_VERSION, prompt, getattr(model, "parameters", {}))
    hit = cache.has(key)
    def compute():
        try:
            raw = model(prompt)
            if not isinstance(raw, str):
                raise ValueError("model response must be text")
            return {"raw": raw, "error": None}
        except Exception as error:
            # Persist the failed attempt as well: a rerun must not repeat a call.
            return {"raw": None, "error": f"{type(error).__name__}: {error}"}
    if hit:
        return cache.get(key), key, True
    lock = cache.root / key[:2] / f"{key}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if cache.has(key):
            return cache.get(key), key, True
        raise RuntimeError(f"model call already pending for cache key {key}; no duplicate call made")
    try:
        os.close(descriptor)
        result = cache.get_or_compute(key, compute)
    finally:
        lock.unlink(missing_ok=True)
    return result, key, hit


def analyse(field, text, response):
    raw = response["raw"]
    result = {"raw": raw, "value": None, "status": None, "spans": [], "error": response.get("error")}
    if result["error"]:
        return result
    quote = raw.strip()
    if quote.upper() == "ABSENT":
        return {**result, "status": "absent"}
    result["spans"] = [list(span) for span in annotate.occurrences(text, quote)]
    if not result["spans"]:
        result["error"] = "proposal has no exact occurrence in saved text"
        return result
    try:
        answer = validated_field(field, normalise_field(field, quote), result["spans"][0], "explicit")
        result.update(value=answer["value"], status="explicit")
    except ValueError as error:
        result["error"] = str(error)
    return result


def show_proposal(field, text, proposal, output):
    print(f"\nField: {field}\nLLM raw answer: {_safe(proposal['raw'])}", file=output)
    print("Normalised value: " + _safe(json.dumps(proposal["value"])), file=output)
    print("Candidate spans: " + json.dumps(proposal["spans"]), file=output)
    if proposal["error"]:
        print("Cannot accept: " + _safe(proposal["error"]), file=output)
    lines = text.splitlines()
    anchors = [text.count("\n", 0, span[0]) for span in proposal["spans"]]
    if not anchors:
        words = annotate.HINTS[field.split(".")[-1]]
        anchors = [next((i for i, line in enumerate(lines) if any(word in line.lower() for word in words)), 0)]
    for number, anchor in enumerate(anchors, 1):
        first = max(0, min(anchor - 1, len(lines) - 3))
        print(f"Context {number} (up to 3 lines):", file=output)
        for index in range(first, min(first + 3, len(lines))):
            print(f"{index + 1:5d} | {_safe(lines[index])}", file=output)


class Provenance:
    def __init__(self, path, annotator):
        self.path = Path(path)
        raw = self.path.read_bytes() if self.path.exists() else None
        self.digest = hashlib.sha256(raw).hexdigest() if raw is not None else None
        self.events = [json.loads(line) for line in raw.decode().splitlines()] if raw else []
        if any(event.get("annotator") != annotator for event in self.events):
            raise ValueError("proposal log belongs to another annotator")

    def append(self, event):
        proposed = [*self.events, event]
        text = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in proposed)
        self.digest = atomic_text(self.path, text, expected=self.digest, check=True)
        self.events = proposed

    def latest(self):
        result = {}
        for event in self.events:
            result[event["doc_id"], event["field"]] = event
        return result


def _new_record(row, annotator, source, text_hash):
    return AnnotationRecord(doc_id=row["doc_id"], annotator=annotator, text_path=str(source),
                            text_sha256=text_hash, split=row.get("split") or None)


def _updated(record, field, answer):
    data = record.model_dump()
    if field == "flags":
        data["flags"], data["flags_reviewed"] = answer, True
    else:
        data["fields"][field] = answer
    return AnnotationRecord.model_validate(data)


def _final_value(field, answer):
    return answer if field == "flags" else answer["value"] if answer else None


def review_proposals(pilot_path, *, annotator, model, cache_dir="cache/annotation_proposals",
                     output_dir="data/real/annotations", dry_run=False, revisit_skipped=False,
                     input_fn=None, output=None):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", annotator):
        raise ValueError("invalid annotator ID")
    input_fn = input if input_fn is None else input_fn
    output = sys.stdout if output is None else output
    _, pilot = read_csv(pilot_path)
    if len({row["doc_id"] for row in pilot}) != len(pilot):
        raise ValueError("duplicate pilot document")
    if any(row.get("split") == "real_test" or row.get("collection", "real") != "real" for row in pilot):
        raise ValueError("only real non-test pilot documents may be reviewed")
    cache = Cache(cache_dir)
    target = Path(output_dir) / f"{annotator}.jsonl"
    records, digest = ({}, None) if dry_run else read_annotations(target)
    if any(record.annotator != annotator for record in records.values()):
        raise ValueError("annotation file belongs to another annotator")
    log = None if dry_run else Provenance(Path(output_dir) / f"{annotator}.proposals.jsonl", annotator)
    def ask(prompt):
        print(prompt.replace(" / reject <reason>", ""), end="", file=output, flush=True)
        value = input_fn("")
        if value.strip().lower() == "back":
            raise annotate.Back
        return value
    exit_code = 0
    try:
        for row in pilot:
            if row.get("rejected_reason", "").strip():
                continue
            source = local_text_path(row["text_path"], pilot_path)
            raw_text = source.read_bytes()
            text, text_hash = raw_text.decode("utf-8"), hashlib.sha256(raw_text).hexdigest()
            record = records.get(row["doc_id"]) or _new_record(row, annotator, source, text_hash)
            if record.text_sha256 != text_hash:
                raise ValueError("saved source text changed; previous spans cannot be reused")
            for field in (*FIELDS, "flags"):
                done = record.flags_reviewed if field == "flags" else field in record.fields
                latest = log.latest().get((record.doc_id, field)) if log else None
                # Recover a human-confirmed decision interrupted between log and snapshot.
                if latest and latest["phase"] == "confirmed":
                    if latest["text_sha256"] != text_hash:
                        raise ValueError("pending decision belongs to another source version")
                    final = latest["final_annotation"]
                    current = record.flags if field == "flags" else record.fields.get(field)
                    if not done:
                        record = _updated(record, field, final)
                        digest = save_annotations(target, {**records, record.doc_id: record}, digest)
                        records[record.doc_id] = record
                    log.append({**latest, "phase": "saved" if not done or current == final else "superseded"})
                    done = True
                if not dry_run and (done or (latest and latest["phase"] == "skipped"
                                           and latest["text_sha256"] == text_hash and not revisit_skipped)):
                    continue
                if field == "flags" and dry_run:
                    continue  # Feature flags are human-only, never LLM-proposed.
                if field == "flags":
                    proposal, key, hit = {"raw": None, "value": None, "status": None, "spans": [], "error": None}, None, False
                else:
                    response, key, hit = cached_proposal(cache, model, field, text)
                    proposal = analyse(field, text, response)
                    show_proposal(field, text, proposal, output)
                if dry_run:
                    continue
                while True:
                    try:
                        decision = "n" if field == "flags" else ask("Human decision [y accept / n correct / s skip / show full text]: ").strip().lower()
                    except annotate.Back:
                        print("Use n to correct this field, or s to skip.", file=output)
                        continue
                    if decision == "show":
                        for number, line in enumerate(text.splitlines(), 1):
                            print(f"{number:5d} | {_safe(line)}", file=output)
                        continue
                    if decision not in {"y", "n", "s"}:
                        print("Enter y, n or s. No automatic acceptance.", file=output)
                        continue
                    event = {"decision_id": hashlib.sha256(f"{annotator}:{record.doc_id}:{field}:{len(log.events)}".encode()).hexdigest()[:24],
                             "annotator": annotator, "doc_id": record.doc_id, "field": field,
                             "annotation_mode": "llm_proposed_human_confirmed", "text_sha256": text_hash,
                             "text_path": str(source), "model": model.identity, "parameters": getattr(model, "parameters", {}),
                             "prompt_version": PROMPT_VERSION, "cache_key": key, "cache_hit": hit,
                             "llm_proposal": proposal["raw"], "proposal_value": proposal["value"],
                             "proposal_spans": proposal["spans"], "proposal_error": proposal["error"],
                             "human_decision": decision, "final_value": None, "final_annotation": None, "phase": "decision"}
                    log.append(event)
                    if decision == "s":
                        log.append({**event, "phase": "skipped"})
                        break
                    if decision == "y" and proposal["error"]:
                        log.append({**event, "phase": "invalid_accept"})
                        print("Proposal cannot be accepted; correct it manually or skip.", file=output)
                        continue
                    try:
                        if field == "flags":
                            final = annotate.ask_flags(ask, output)
                        elif decision == "n":
                            final = annotate.ask_field(field, text, ask, output)
                        elif proposal["status"] == "absent":
                            final = validated_field(field, None, None, "absent")
                        else:
                            spans = proposal["spans"]
                            choice = 1
                            if len(spans) > 1:
                                while True:
                                    value = ask(f"Choose source occurrence 1–{len(spans)}: ").strip()
                                    if value.isdigit() and 1 <= int(value) <= len(spans):
                                        choice = int(value)
                                        break
                            final = validated_field(field, proposal["value"], spans[choice - 1], "explicit")
                    except annotate.Back:
                        log.append({**event, "phase": "cancelled"})
                        continue
                    except (EOFError, KeyboardInterrupt):
                        log.append({**event, "phase": "interrupted"})
                        raise
                    if hashlib.sha256(source.read_bytes()).hexdigest() != text_hash:
                        raise ValueError("source changed during review; decision not saved")
                    confirmed = {**event, "phase": "confirmed", "final_value": _final_value(field, final),
                                 "final_annotation": final,
                                 "final_quote": text[slice(*final["span"])] if field != "flags" and final["span"] else None}
                    log.append(confirmed)  # Provenance must exist before a human label is written.
                    proposed = _updated(record, field, final)
                    digest = save_annotations(target, {**records, record.doc_id: proposed}, digest)
                    records[record.doc_id] = record = proposed
                    log.append({**confirmed, "phase": "saved"})
                    print("Saved human decision and provenance.", file=output)
                    break
    except EOFError:
        print("\nStopped; confirmed fields are saved. Rerun to resume.", file=output)
    except KeyboardInterrupt:
        exit_code = 130
        print("\nInterrupted; confirmed fields are saved.", file=output)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--model", default="local:qwen2.5:7b", help="local:<installed-Ollama-model> or openai:<model-id>")
    parser.add_argument("--local-url", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--pilot", type=Path, default=Path("data/real/pilot20.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/real/annotations"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/annotation_proposals"))
    parser.add_argument("--dry-run", action="store_true", help="display proposals only; cache is still used, no annotation/log writes")
    parser.add_argument("--revisit-skipped", action="store_true")
    args = parser.parse_args(argv)
    try:
        model = HTTPModel(args.model, local_url=args.local_url)
        return review_proposals(args.pilot, annotator=args.annotator, model=model, cache_dir=args.cache_dir,
                                output_dir=args.output_dir, dry_run=args.dry_run, revisit_skipped=args.revisit_skipped)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(_safe(error))


if __name__ == "__main__":
    raise SystemExit(main())
