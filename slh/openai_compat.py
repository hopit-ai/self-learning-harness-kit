"""Agent backend for self-served OpenAI-compatible endpoints (vLLM, SGLang).

``backend: openai_compatible`` with ``api_base`` (e.g. a Modal vLLM server's
``https://<app>.modal.run/v1``), ``model`` (the served model name; also the
price-table key) and ``capture: auto|messages|logprobs|tokens``.

The request is exactly what tau2's ``generate`` sends through litellm
(``to_litellm_messages``, each tool's ``openai_schema``, ``tool_choice:
auto`` when tools are present, temperature, per-trial seed, ``max_tokens``
when set, whitelisted extras) plus, by capture level, exactly the fields in
``budget.CAPTURE_FIELDS``:

- ``logprobs: true, top_logprobs: 0``: the log-prob of each sampled token;
- ``return_token_ids: true``: vLLM's extension returning
  ``prompt_token_ids`` and ``choices[i].token_ids``.

Server flags for RL-grade capture (vLLM): ``--return-tokens-as-token-ids``
(log-prob entries carry ``token_id:<n>``, a second source of sampled ids that
is cross-checked) and ``--logprobs-mode processed_logprobs`` (log-probs of the
distribution actually sampled from, after temperature). Sampling must not be
truncated: top_p 1.0 and no top_k (enforced by ``ModelConfig``), since
truncation breaks RL importance ratios.

The reply is parsed exactly as tau2's ``generate`` parses it (content, tool
calls with ``json.loads`` of the raw arguments), so parsing semantics are
unchanged by the serving switch. Every request goes through
``budget.metered_dispatch``: inside an authorized, reserved attempt, with
parameters re-checked against the config.

``capture: auto`` runs one probe per endpoint per run (``probe``), counted by
the meter, and requests only the level the probe verified. With a $0 cap the
probe is skipped and the level is ``messages``.

``serving_session`` (instead of ``api_base``): the endpoint launched by
``python -m slh serve start``. Before every request, inside the metered
attempt, the session's current window supplies the URL and bearer token, and
the request is refused (``slh.serving.ServingWindowClosed``, nothing sent)
once the window's request deadline has passed or the window is not serving.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from slh.budget import (
    CAPTURE_FIELDS,
    BudgetExceeded,
    Meter,
    ModelConfig,
    Overspend,
    UnmeteredCall,
    Usage,
    metered_dispatch,
)
from slh.serving import ServingWindowClosed, gate_for
from slh.tokens import LEVELS, extract, level_rank, logprobs_mode

PROBE_SYSTEM = "You are a connectivity probe. Reply with the single word OK."
PROBE_USER = "ping"
TIMEOUT_S = 600.0


# Named like litellm's errors so the meter treats them as transient.
class RateLimitError(Exception):
    pass


class ServiceUnavailableError(Exception):
    pass


class InternalServerError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class Timeout(Exception):
    pass


class EndpointError(Exception):
    """A non-transient HTTP error (4xx other than 429)."""


class ContextLengthError(EndpointError):
    """The server refused the request for its length (an HTTP 400 whose body names
    the context length): a property of this request, not of the endpoint."""


_CONTEXT_400 = re.compile(r"maximum context length|context length|context window|"
                          r"max_model_len|too many tokens|prompt is too long|"
                          r"input tokens exceed|longer than the maximum", re.IGNORECASE)


def _error_body(exc: Any) -> str:
    try:
        return exc.read(4000).decode("utf-8", "replace").strip()
    except Exception:  # noqa: BLE001 - the body is optional detail
        return ""


def post_json(url: str, body: dict[str, Any], headers: dict[str, str],
              timeout: float) -> dict[str, Any]:
    """POST a JSON body; return the JSON reply. Replaced in tests."""
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RateLimitError(f"HTTP 429 from {url}") from None
        if exc.code in (502, 503, 504):
            raise ServiceUnavailableError(f"HTTP {exc.code} from {url}") from None
        if exc.code >= 500:
            raise InternalServerError(f"HTTP {exc.code} from {url}") from None
        body = _error_body(exc)
        detail = f"HTTP {exc.code} from {url}" + (f": {body[:2000]}" if body else "")
        if exc.code == 400 and _CONTEXT_400.search(body):
            raise ContextLengthError(detail) from None
        raise EndpointError(detail) from None
    except TimeoutError:
        raise Timeout(f"timeout after {timeout}s from {url}") from None
    except urllib.error.URLError as exc:
        raise APIConnectionError(f"cannot reach {url}: {exc.reason}") from None


def parse_reply(response: dict[str, Any]) -> Any:
    """tau2's generate parsing, applied to a plain JSON response."""
    from tau2.data_model.message import AssistantMessage, ToolCall

    choice = response["choices"][0]
    message = choice["message"]
    if message.get("role", "assistant") != "assistant":
        raise ValueError("The response should be an assistant message")
    calls = [ToolCall(id=call["id"], name=call["function"]["name"],
                      arguments=json.loads(call["function"]["arguments"]))
             for call in message.get("tool_calls") or []]
    usage = response.get("usage") or {}
    return AssistantMessage(role="assistant", content=message.get("content"),
                            tool_calls=calls or None, cost=0.0,
                            usage={"completion_tokens": usage.get("completion_tokens"),
                                   "prompt_tokens": usage.get("prompt_tokens")},
                            raw_data={k: v for k, v in response.items()
                                      if k not in ("prompt_token_ids",)})


class OpenAICompatibleBackend:
    def __init__(self, config: ModelConfig, call_name: str, level: str = "messages") -> None:
        if config.backend != "openai_compatible":
            raise ValueError("OpenAICompatibleBackend needs an openai_compatible config")
        if level not in config.capture_levels():
            raise ValueError(f"capture level {level!r} not allowed by the config")
        self.config = config
        self.call_name = call_name
        self.level = level
        self.last_capture: dict[str, Any] | None = None
        self.gate = gate_for(config)
        self._served: tuple[str, str] | None = None  # (api_base, token) of the window
        # Optional pre-request check of the request body (e.g. the partner runner's
        # context-length guard); raises to refuse the request before it is sent.
        self.context_guard: Any = None

    def check_window(self) -> None:
        """Self-served windows only: refuse after the deadline (raises)."""
        if self.gate is not None:
            self._served = self.gate.check_request()

    def endpoint(self) -> str:
        if self.gate is not None:
            return self.gate.session.root.as_posix()
        return self.config.api_base

    def url(self) -> str:
        if self.gate is not None:
            if self._served is None:
                raise ServingWindowClosed("serving window not checked before the request")
            return self._served[0].rstrip("/") + "/chat/completions"
        return self.config.api_base.rstrip("/") + "/chat/completions"

    def headers(self) -> dict[str, str]:
        if self.gate is not None:
            if self._served is None:
                raise ServingWindowClosed("serving window not checked before the request")
            return {"Authorization": f"Bearer {self._served[1]}"}
        name = self.config.api_key_env
        return {"Authorization": f"Bearer {os.environ[name]}"} if name else {}

    def request_body(self, messages: list[Any], tools: list[Any] | None,
                     seed: int | None) -> dict[str, Any]:
        from tau2.utils.llm_utils import to_litellm_messages

        body: dict[str, Any] = {"model": self.config.model,
                                "messages": to_litellm_messages(messages)}
        if tools:
            body["tools"] = [tool.openai_schema for tool in tools]
            body["tool_choice"] = "auto"
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            body["max_tokens"] = self.config.max_tokens
        if seed is not None:
            body["seed"] = seed
        body.update(self.config.extra_params())
        body.update(CAPTURE_FIELDS[self.level])
        return body

    def complete(self, messages: list[Any], tools: list[Any] | None,
                 seed: int | None) -> tuple[Any, Usage | None]:
        body = self.request_body(messages, tools, seed)
        if self.context_guard is not None:
            self.context_guard(body)  # raises before anything is sent
        self._served = None
        response = metered_dispatch(
            body, lambda sent: post_json(self.url(), sent, self.headers(), TIMEOUT_S),
            gate=self.check_window)
        self.last_capture = extract(response)
        # Usage was recorded at dispatch; the meter charges from it.
        return parse_reply(response), None


def probe(backend: OpenAICompatibleBackend, meter: Meter) -> dict[str, Any]:
    """One metered request at the highest level the config allows; returns
    the verified level (never higher than what came back and was checked)."""
    from tau2.data_model.message import SystemMessage, UserMessage

    from slh.agent import metered_complete

    config = backend.config
    record: dict[str, Any] = {"endpoint": backend.endpoint(), "model": config.model,
                              "mode": config.capture,
                              "time_utc": dt.datetime.now(dt.UTC).isoformat(),
                              "logprobs_mode": logprobs_mode(
                                  config.logprobs_mode,
                                  {**config.extra_params(), "temperature": config.temperature})}
    if meter.cap_usd == 0:
        backend.level = "messages"
        return {**record, "probed": False, "requested_level": None,
                "detected_level": "messages", "reason": "skipped: $0 budget cap"}
    top = max(config.capture_levels(), key=level_rank)
    backend.level = top
    try:
        metered_complete(meter, "agent", backend,
                         [SystemMessage(role="system", content=PROBE_SYSTEM),
                          UserMessage(role="user", content=PROBE_USER)], None, 0)
        graded = backend.last_capture or {"level": "messages", "problems": []}
        detected = min(top, graded["level"], key=level_rank)
        problems = graded.get("problems", [])
        if graded["level"] != "messages" and not graded.get("trainable"):
            detected = "messages"
    except (BudgetExceeded, Overspend, UnmeteredCall):
        raise
    except Exception as exc:  # noqa: BLE001 - a failed probe falls back to messages
        detected, problems = "messages", [f"probe_failed:{type(exc).__name__}"]
    backend.level = detected
    return {**record, "probed": True, "requested_level": top, "detected_level": detected,
            "problems": problems, "levels": list(LEVELS)}
