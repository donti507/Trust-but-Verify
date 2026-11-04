"""Fictional source documents and simulated humans/models; no network calls."""
import io
import json
import re
import socket
from pathlib import Path

import pytest

from scripts import annotate_auto as auto
from scripts.pilot_utils import read_annotations, write_csv
from trustswap.common.annotations import FIELDS
from trustswap.common.cache import Cache


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unexpected network call")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(auto, "urlopen", forbidden)


class Model:
    identity = "mock:fixture"
    parameters = {"temperature": 0}

    def __init__(self, values=None):
        self.values = values or {}
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        field = re.search(r"`([^`]+)`", prompt)[1]
        return self.values.get(field, "ABSENT")


def inputs(*values):
    values = iter(values)
    def ask(_):
        try:
            return next(values)
        except StopIteration:
            raise EOFError from None
    return ask


@pytest.fixture
def setup(tmp_path):
    text = tmp_path / "confirmation.txt"
    text.write_bytes((Path(__file__).parent / "fixtures/pilot/confirmation.txt").read_bytes())
    pilot = tmp_path / "pilot.csv"
    row = {"doc_id": "fixture_1", "text_path": str(text)}
    write_csv(pilot, list(row), [row])
    return pilot, text, tmp_path / "annotations", tmp_path / "cache"


def run(setup, model, *answers, **kwargs):
    pilot, _, output_dir, cache_dir = setup
    output = io.StringIO()
    result = auto.review_proposals(pilot, annotator="A1", model=model, output_dir=output_dir,
                                  cache_dir=cache_dir, input_fn=inputs(*answers), output=output, **kwargs)
    return result, output.getvalue()


def events(setup):
    return [json.loads(line) for line in (setup[2] / "A1.proposals.jsonl").read_text().splitlines()]


def record(setup):
    return read_annotations(setup[2] / "A1.jsonl")[0]["fixture_1"]


def test_dry_run_full_prompts_and_cache_without_label_or_log_writes(setup):
    model = Model({"trade_date": "20-Mar-2012"})
    _, output = run(setup, model, dry_run=True)
    assert len(model.calls) == len(FIELDS)
    source = setup[1].read_text()
    assert all(prompt.endswith(source) for prompt in model.calls)
    assert model.calls[0] == auto.make_prompt("trade_date", source)
    assert "2012-03-20" in output and "Context 1" in output
    assert not setup[2].exists()
    second = Model()
    run(setup, second, dry_run=True)
    assert second.calls == []


def test_human_accept_writes_compatible_annotation_and_provenance(setup):
    _, output = run(setup, Model({"trade_date": "20-Mar-2012"}), "y")
    answer = record(setup).fields["trade_date"]
    assert answer["value"] == "2012-03-20"
    assert setup[1].read_text()[slice(*answer["span"])] == "20-Mar-2012"
    saved = [e for e in events(setup) if e["phase"] == "saved"]
    assert saved[0]["human_decision"] == "y"
    assert saved[0]["llm_proposal"] == "20-Mar-2012"
    assert saved[0]["final_value"] == "2012-03-20"
    assert saved[0]["final_quote"] == "20-Mar-2012"
    assert "Candidate spans" in output


def test_manual_correction_and_trimmed_span(setup):
    run(setup, Model({"trade_date": "22-Mar-2012"}), "n", " 20-Mar-2012 ", "y")
    assert record(setup).fields["trade_date"]["value"] == "2012-03-20"
    saved = next(e for e in events(setup) if e["phase"] == "saved")
    assert saved["human_decision"] == "n" and saved["llm_proposal"] == "22-Mar-2012"
    assert saved["final_value"] == "2012-03-20"


def test_unmatched_proposal_cannot_be_accepted_and_skip_resumes(setup):
    model = Model({"trade_date": "invented date"})
    _, output = run(setup, model, "y", "s")
    assert not (setup[2] / "A1.jsonl").exists()
    assert "cannot be accepted" in output
    assert any(e["phase"] == "invalid_accept" for e in events(setup))
    rerun = Model()
    _, output = run(setup, rerun)
    assert "Field: trade_date" not in output
    assert rerun.calls == []  # next field was cached before the first EOF
    _, output = run(setup, Model(), "s", revisit_skipped=True)
    assert "Field: trade_date" in output


def test_repeated_quote_requires_human_occurrence_selection(setup):
    setup[1].write_text(setup[1].read_text() + "\nRepeated: 20-Mar-2012\n")
    run(setup, Model({"trade_date": "20-Mar-2012"}), "y", "2")
    assert record(setup).fields["trade_date"]["span"][0] == setup[1].read_text().rindex("20-Mar-2012")


def test_all_fields_and_flags_remain_agreement_compatible(setup):
    run(setup, Model(), *(["y"] * len(FIELDS)), "none", "y")
    result = record(setup)
    assert result.complete and result.flags_reviewed
    assert all(answer["status"] == "absent" for answer in result.fields.values())
    assert len([e for e in events(setup) if e["phase"] == "saved"]) == len(FIELDS) + 1


def test_failed_model_attempt_is_cached(tmp_path):
    class Broken(Model):
        def __call__(self, prompt):
            self.calls.append(prompt)
            raise RuntimeError("fixture outage")
    model = Broken()
    cache = Cache(tmp_path)
    first = auto.cached_proposal(cache, model, "trade_date", "fictional")
    second = auto.cached_proposal(cache, model, "trade_date", "fictional")
    assert len(model.calls) == 1
    assert first[0] == second[0] and second[2]
    assert "fixture outage" in first[0]["error"]


def test_confirmed_decision_recovers_after_snapshot_failure(setup, monkeypatch):
    original = auto.save_annotations
    def fail(*args):
        raise RuntimeError("simulated interrupted snapshot")
    monkeypatch.setattr(auto, "save_annotations", fail)
    with pytest.raises(RuntimeError, match="interrupted snapshot"):
        run(setup, Model({"trade_date": "20-Mar-2012"}), "y")
    assert events(setup)[-1]["phase"] == "confirmed"
    monkeypatch.setattr(auto, "save_annotations", original)
    model = Model()
    run(setup, model)
    assert record(setup).fields["trade_date"]["value"] == "2012-03-20"
    assert all("`trade_date`" not in prompt for prompt in model.calls)


@pytest.mark.parametrize("spec,response", [
    ("local:fixture", {"response": "20-Mar-2012"}),
])
def test_http_backends_mocked_only(spec, response, monkeypatch):
    requests = []
    def request(req, timeout):
        requests.append(req)
        return io.BytesIO(json.dumps(response).encode())
    monkeypatch.setattr(auto, "urlopen", request)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret")
    model = auto.HTTPModel(spec)
    assert model("full fictional document") == "20-Mar-2012"
    payload = json.loads(requests[0].data)
    assert payload.get("input", payload.get("prompt")) == "full fictional document"
    assert "fixture-secret" not in json.dumps(model.parameters)


def test_source_change_and_real_test_refused(setup):
    run(setup, Model(), "y")
    setup[1].write_text("changed")
    with pytest.raises(ValueError, match="source text changed"):
        run(setup, Model())
    row = {"doc_id": "fixture_1", "text_path": str(setup[1]), "split": "real_test"}
    write_csv(setup[0], list(row), [row])
    with pytest.raises(ValueError, match="non-test"):
        run(setup, Model())


def test_sdk_environment_chat_dry_run_and_cache(setup, monkeypatch):
    import openai
    from types import SimpleNamespace
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-sdk-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.invalid/v1")
    real_factory = openai.OpenAI
    clients, calls = [], []
    def factory():
        client = real_factory()
        clients.append(client)
        monkeypatch.setattr(client, "with_options", lambda **options: client)
        assert client.api_key == "fixture-sdk-secret"
        assert str(client.base_url) == "https://gateway.example.invalid/v1/"
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content="20-Mar-2012" if
                    "`trade_date`" in kwargs["messages"][0]["content"] else "ABSENT"))])
        monkeypatch.setattr(client.chat.completions, "create", create)
        return client
    monkeypatch.setattr(openai, "OpenAI", factory)
    model = auto.HTTPModel("openai:gemini-2.0-flash")
    _, output = run(setup, model, dry_run=True)
    assert "2012-03-20" in output
    assert len(calls) == len(FIELDS)
    assert all(call["model"] == "gemini-2.0-flash" for call in calls)
    assert all(call["messages"][0]["content"].endswith(setup[1].read_text()) for call in calls)
    assert not setup[2].exists()
    monkeypatch.delenv("OPENAI_API_KEY")
    run(setup, auto.HTTPModel("openai:gemini-2.0-flash"), dry_run=True)
    assert len(calls) == len(FIELDS)  # no SDK client or model call on rerun
    assert "fixture-sdk-secret" not in json.dumps(model.parameters)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://another.example.invalid/v1")
    other = auto.HTTPModel("openai:gemini-2.0-flash")
    assert model.parameters != other.parameters


def test_sdk_rejects_truncated_output(monkeypatch):
    import openai
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    client = MagicMock()
    client.__enter__.return_value = client
    client.with_options.return_value = client
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="length", message=SimpleNamespace(content="partial"))])
    monkeypatch.setattr(openai, "OpenAI", lambda: client)
    with pytest.raises(ValueError, match="token limit"):
        auto.HTTPModel("openai:gemini-2.0-flash")("fictional")
