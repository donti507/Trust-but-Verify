"""Model clients for extraction (Step 10).

Both tracks use the OpenAI-compatible chat API, which the frontier providers
(OpenAI, Gemini, Vertex AI) and a vLLM server all speak. They differ only in
configuration (configs/models.yaml):

* ``structured: json_schema``  -> response_format JSON-schema mode (frontier, recent vLLM)
* ``structured: guided_json``  -> vLLM's grammar-constrained ``guided_json``
* ``structured: json_object``  -> plain JSON mode (fallback; the parser still validates the schema)
* ``logprobs: true``           -> request token log-probs

Credentials come only from environment variables named in the config; they never
enter cache keys or records. Clients are built lazily, so cached runs need no key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import os
from pathlib import Path
import time

import yaml

from trustswap.extract.fields import output_schema

ROOT = Path(__file__).resolve().parents[3]
MODELS_CONFIG = ROOT / "configs" / "models.yaml"


@dataclass
class Completion:
    text: str
    tokens: list[tuple[str, float]] | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float | None = None

    def to_cache(self) -> dict:
        return {"text": self.text, "tokens": self.tokens, "finish_reason": self.finish_reason,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "latency_s": self.latency_s}

    @classmethod
    def from_cache(cls, data: dict) -> "Completion":
        tokens = [tuple(t) for t in data["tokens"]] if data.get("tokens") else None
        return cls(data["text"], tokens, data.get("finish_reason"), data.get("input_tokens"),
                   data.get("output_tokens"), data.get("latency_s"))


@dataclass
class ModelSpec:
    key: str                     # "frontier" or "open_weights"
    name: str                    # provider model id
    version: str                 # pinned version / snapshot
    access_date: str             # YYYY-MM-DD the version was pinned
    base_url_env: str | None     # env var with the API base URL (None: provider default)
    api_key_env: str             # env var with the API key
    structured: str = "json_schema"
    logprobs: bool = False
    max_tokens: int = 4096
    send_seed: bool = True       # Gemini's OpenAI-compatible API rejects "seed"
    extra_body: dict = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        """Identity used in records and cache keys: name@version."""
        return f"{self.name}@{self.version}"

    def check_pinned(self) -> None:
        if "PIN-ME" in (self.name, self.version, self.access_date):
            raise ValueError(f"pin {self.key} name, version and access_date in configs/models.yaml first")
        date.fromisoformat(self.access_date)


def load_models(path: Path = MODELS_CONFIG) -> dict[str, ModelSpec]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {key: ModelSpec(key=key, **{k: v for k, v in spec.items() if k != "notes"})
            for key, spec in data["models"].items()}


RETRY_WAITS = (5, 15, 30, 60, 120)  # seconds; overloaded providers (503/429) usually recover


def _message(error) -> str:
    """Provider's own error message (never contains the API key)."""
    body = getattr(error, "body", None)
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
    return str(error)


class OpenAICompatClient:
    """Chat completion with structured JSON output; returns text and token log-probs."""

    def __init__(self, spec: ModelSpec, timeout: float = 300, sleep=time.sleep):
        self.spec, self.timeout, self.sleep = spec, timeout, sleep
        self.params = {"structured": spec.structured, "logprobs": spec.logprobs,
                       "max_tokens": spec.max_tokens, "extra_body": spec.extra_body}

    def _client(self):
        import openai

        key = os.environ.get(self.spec.api_key_env)
        if not key:
            raise RuntimeError(f"set {self.spec.api_key_env} in your environment (never in code or chat)")
        base_url = os.environ.get(self.spec.base_url_env) if self.spec.base_url_env else None
        return openai.OpenAI(api_key=key, base_url=base_url, timeout=self.timeout, max_retries=0)

    def _create(self, kwargs: dict):
        """One API call with waits on overload / rate limits; clear messages for request errors."""
        import openai

        retryable = (openai.InternalServerError, openai.RateLimitError, openai.APIConnectionError,
                     openai.APITimeoutError)
        for attempt, wait in enumerate((*RETRY_WAITS, None), start=1):
            try:
                with self._client() as client:
                    return client.chat.completions.create(**kwargs)
            except retryable as error:
                if wait is None:
                    raise RuntimeError(f"{self.spec.name}: still failing after {attempt} tries: "
                                       f"{_message(error)}") from None
                print(f"  {self.spec.name} busy ({_message(error)[:80]}); retry {attempt} in {wait}s", flush=True)
                self.sleep(wait)
            except openai.APIStatusError as error:
                raise RuntimeError(f"{self.spec.name} rejected the request "
                                   f"({error.status_code}): {_message(error)}") from None

    def complete(self, messages: list[dict], *, temperature: float, seed: int) -> Completion:
        schema = output_schema()
        kwargs: dict = {"model": self.spec.name, "messages": messages, "temperature": temperature,
                        "max_tokens": self.spec.max_tokens}
        if self.spec.send_seed:
            kwargs["seed"] = seed
        extra = dict(self.spec.extra_body)
        if self.spec.structured == "json_schema":
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "swap_extraction", "schema": schema, "strict": True}}
        elif self.spec.structured == "guided_json":
            extra["guided_json"] = schema
        elif self.spec.structured == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        else:
            raise ValueError(f"unknown structured mode {self.spec.structured!r}")
        if self.spec.logprobs:
            kwargs["logprobs"] = True
        if extra:
            kwargs["extra_body"] = extra
        started = time.perf_counter()
        response = self._create(kwargs)
        latency = time.perf_counter() - started
        if not response.choices:
            raise RuntimeError("model returned no choices")
        choice = response.choices[0]
        tokens = None
        content = getattr(choice.logprobs, "content", None) if choice.logprobs else None
        if content:
            tokens = [(t.token, float(t.logprob)) for t in content]
        usage = response.usage
        return Completion(text=choice.message.content or "", tokens=tokens, finish_reason=choice.finish_reason,
                          input_tokens=getattr(usage, "prompt_tokens", None),
                          output_tokens=getattr(usage, "completion_tokens", None), latency_s=latency)
