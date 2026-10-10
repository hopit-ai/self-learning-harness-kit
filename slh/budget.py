"""Budget meter: the single accounting point for every model call.

Every call made by the agent, the user simulator or the grader's judge goes
through ``Meter.call``:

1. **Reserve.** Before each attempt the meter computes a reservation that is a
   true upper bound on the attempt's cost: input tokens are bounded by the
   UTF-8 byte length of the full serialized request (byte-level BPE never emits
   more tokens than bytes) plus fixed per-message, per-tool and template
   allowances; output tokens are bounded by the configured ``max_tokens`` or,
   when none is sent, by the model's ``max_output_tokens`` from the price table,
   plus any configured ``reasoning_reserve_tokens``. If spent + reservation
   would pass the cap, the attempt is refused before it is made.
2. **Persist the reservation.** Before a paid attempt is dispatched, the
   reservation is recorded as pending and the ``on_change`` hook persists it,
   so a process killed mid-call leaves a durable record that resume charges in
   full (``unreconciled_reservation``).
3. **Check at dispatch.** ``completion_guard`` wraps the provider completion
   function. Inside an authorized attempt it re-validates the exact outgoing
   parameters against the whitelist and the authorized config (model,
   max_tokens, reasoning-off, provider pin) and rejects anything else before
   it reaches the network.
4. **Charge at the wire.** The moment a provider response returns, its usage
   and any provider-reported dollar cost are recorded, before any parsing.
   The attempt is charged max(table-derived cost, provider-reported cost).
   Provider retries are forced off; the meter retries, reserving and charging
   every attempt. An attempt that fails with unknown usage costs its reservation.
5. **Check after.** Cost above the reservation, cumulative spend above the cap,
   or a provider-reported cost more than 1% above the table-derived cost
   (``price_table_underestimate``) raises ``Overspend`` and the run aborts.

**Parallel runs.** With a ``SharedLedger`` (``slh.ledger``) the cap check in
step 1 and the durable reservation in step 2 become one atomic, file-locked
append to the run's ledger, and the cap is checked against the committed spend
of *all* workers (charged plus every unsettled reservation). Step 4's charge
settles the reservation in the ledger.

**Rate limits.** Besides per-attempt backoff, a meter that sees
``pause_after`` consecutive transient failures (across calls) pauses for a
longer, jittered ``pause_s`` before trying again; that failed attempt does not
consume one of the call's retries. At most ``max_pauses`` pauses per session.

**Call capture.** After every attempt, ``on_attempt`` (if set) receives the
attempt's role, number, reservation, charged cost, error type, latency, the
request payload and the raw provider response fields recorded at the wire
before any parsing (``raw_response_record``).
"""

from __future__ import annotations

import contextlib
import contextvars
import datetime as dt
import json
import math
import random
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

ROLES = ("agent", "user", "judge")
# Fixed allowances on top of the byte bound, for chat-template tokens.
PER_MESSAGE_OVERHEAD_TOKENS = 16
PER_TOOL_OVERHEAD_TOKENS = 32
TEMPLATE_OVERHEAD_TOKENS = 512
UNDERESTIMATE_TOLERANCE = 0.01

# Parameters a config may add. Anything that can change the number or size of
# completions (n, max_completion_tokens, best_of, ...) is rejected, so the
# reservation stays an upper bound. ``reasoning`` and ``provider`` are
# OpenRouter request fields, accepted in exactly one form each (see
# _check_extra). They are sent top-level: litellm 1.82.6 drops a caller's
# ``extra_body`` for OpenRouter, so ``extra_body`` is rejected outright.
# ``chat_template_kwargs`` is the vLLM request field that switches a self-served
# Qwen3.5 to non-thinking mode; it is accepted on openai_compatible only, and
# only as exactly CHAT_TEMPLATE_NON_THINKING.
ALLOWED_EXTRA = frozenset({"top_p", "stop", "presence_penalty", "frequency_penalty",
                           "reasoning", "provider", "chat_template_kwargs"})
REASONING_OFF = {"enabled": False}
CHAT_TEMPLATE_NON_THINKING = {"enable_thinking": False}
MODEL_KEYS = frozenset({"backend", "model", "temperature", "seed", "max_tokens",
                        "reasoning_reserve_tokens", "retries", "api_base", "api_key_env",
                        "capture", "provider_advertises_logprobs", "logprobs_mode",
                        "checkpoint", "serving_session"})
BACKENDS = ("stub", "litellm", "openai_compatible")
# Capture modes per backend (see slh.tokens). Hosted APIs can give at most
# log-probs; token ids need a self-served OpenAI-compatible server.
CAPTURE_MODES = {"stub": ("messages",), "litellm": ("messages", "auto", "logprobs"),
                 "openai_compatible": ("messages", "auto", "logprobs", "tokens")}
# The only request fields capture may add, by level, exactly these values.
CAPTURE_FIELDS = {
    "messages": {},
    "logprobs": {"logprobs": True, "top_logprobs": 0},
    "tokens": {"logprobs": True, "top_logprobs": 0, "return_token_ids": True},
}
# Everything a metered request may carry at dispatch.
DISPATCH_KEYS = frozenset({"model", "messages", "tools", "tool_choice", "temperature",
                           "seed", "max_tokens", "num_retries"}) | ALLOWED_EXTRA
# Backoff between retries of a transient failure (rate limits, overloaded shared
# pools): exponential from BACKOFF_BASE_S, capped at BACKOFF_MAX_S, with jitter.
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 60.0
# Longer pause after a streak of transient failures (see the module docstring).
PAUSE_AFTER = 3
PAUSE_S = 120.0
MAX_PAUSES = 10
_sleep = time.sleep  # replaced in tests


def backoff_seconds(number: int) -> float:
    return min(BACKOFF_BASE_S * 2 ** number, BACKOFF_MAX_S) * random.uniform(1.0, 1.25)


def pause_seconds(base: float) -> float:
    return base * random.uniform(1.0, 1.25)


TRANSIENT_ERRORS = frozenset({
    "RateLimitError", "APIConnectionError", "Timeout", "APITimeoutError",
    "ServiceUnavailableError", "InternalServerError", "BadGatewayError",
})


class BudgetExceeded(RuntimeError):
    """A call was refused because it could take the run past its cap."""


class Overspend(RuntimeError):
    """A call cost more than its reservation, spend passed the cap, or the
    provider reported a higher cost than the price table."""


class UnmeteredCall(RuntimeError):
    """A provider call was attempted outside the budget meter."""


class DispatchRejected(RuntimeError):
    """The outgoing request differs from what the meter authorized."""


def freeze(value: Any) -> Any:
    """Deep-freeze mappings and sequences so config extras cannot be mutated."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(freeze(v) for v in value)
    return value


def thaw(value: Any) -> Any:
    """Fresh, mutable plain copies of frozen values (copy on read)."""
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    return value


def _exactly_json(a: Any, b: Any) -> bool:
    """Equal plain values with identical JSON types (False is not 0)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_exactly_json(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _check_extra(extra: Mapping[str, Any]) -> None:
    rejected = sorted(set(extra) - ALLOWED_EXTRA)
    if rejected:
        hint = (" (extra_body is dropped by litellm for OpenRouter; use top-level "
                "reasoning/provider)" if "extra_body" in rejected else "")
        raise ValueError(f"model config parameters not allowed: {rejected}{hint} "
                         f"(allowed: {sorted(ALLOWED_EXTRA)})")
    if "reasoning" in extra and thaw(extra["reasoning"]) != REASONING_OFF:
        raise ValueError(f"reasoning may only be {REASONING_OFF} (reasoning disabled)")
    if "chat_template_kwargs" in extra and not _exactly_json(
            thaw(extra["chat_template_kwargs"]), CHAT_TEMPLATE_NON_THINKING):
        raise ValueError(f"chat_template_kwargs may only be {CHAT_TEMPLATE_NON_THINKING} "
                         "(non-thinking mode)")
    if "provider" in extra:
        provider = thaw(extra["provider"])
        if (not isinstance(provider, dict) or set(provider) != {"order", "allow_fallbacks"}
                or provider["allow_fallbacks"] is not False
                or not isinstance(provider["order"], list) or len(provider["order"]) != 1
                or not isinstance(provider["order"][0], str) or not provider["order"][0]):
            raise ValueError("provider must pin exactly one provider: "
                             "{order: [<slug>], allow_fallbacks: false}")


@dataclass(frozen=True)
class ModelConfig:
    """One pinned model: who serves it and exactly what is sent.

    ``temperature`` and ``max_tokens`` are sent only when set (None = omit).
    ``extra`` is deep-frozen at construction and copied on read.
    """

    backend: str  # "stub", "litellm" or "openai_compatible"
    model: str
    temperature: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    reasoning_reserve_tokens: int = 0
    retries: int = 3
    extra: Mapping[str, Any] = field(default_factory=dict)
    # Token-level capture (slh.tokens). ``api_base`` and ``api_key_env`` are for
    # openai_compatible (self-served vLLM/SGLang) endpoints only.
    capture: str = "messages"
    api_base: str | None = None
    api_key_env: str | None = None
    provider_advertises_logprobs: bool = False
    logprobs_mode: str | None = None
    checkpoint: str | None = None
    # A self-served endpoint launched by ``python -m slh serve start``: the
    # session directory whose current window supplies the URL, the bearer
    # token and the deadline after which no request is sent (slh.serving).
    # Exclusive with api_base/api_key_env, so a run's config hash does not
    # change from one serving window to the next and the run stays resumable.
    serving_session: str | None = None

    def __post_init__(self) -> None:
        """Validate on every construction path, not only YAML loading."""
        if self.backend not in BACKENDS:
            raise ValueError(f"unknown backend {self.backend!r}")
        if self.capture not in CAPTURE_MODES[self.backend]:
            raise ValueError(f"capture {self.capture!r} is not available on backend "
                             f"{self.backend!r} (choose from {CAPTURE_MODES[self.backend]})")
        if self.backend == "openai_compatible":
            if self.serving_session is not None:
                if not isinstance(self.serving_session, str) or not self.serving_session:
                    raise ValueError("serving_session must be a nonempty path")
                if self.api_base is not None or self.api_key_env is not None:
                    raise ValueError("serving_session supplies the endpoint and its token; "
                                     "do not also set api_base/api_key_env")
            elif not isinstance(self.api_base, str) or not self.api_base.startswith(
                    ("http://", "https://")):
                raise ValueError("openai_compatible needs api_base: http(s)://host[:port]/v1 "
                                 "(or serving_session)")
            if set(self.extra) & {"provider", "reasoning"}:
                raise ValueError("provider/reasoning are OpenRouter fields; not allowed on "
                                 "openai_compatible")
            if self.capture in ("tokens", "auto") and thaw(self.extra).get("top_p", 1.0) != 1.0:
                # Truncated sampling distributions break RL importance ratios.
                raise ValueError("token capture requires top_p 1.0 (no truncation)")
        elif self.api_base is not None or self.api_key_env is not None:
            raise ValueError("api_base/api_key_env apply to openai_compatible only")
        elif self.serving_session is not None:
            raise ValueError("serving_session applies to openai_compatible only")
        if "chat_template_kwargs" in self.extra and self.backend != "openai_compatible":
            raise ValueError("chat_template_kwargs is a vLLM request field; allowed on "
                             "openai_compatible only")
        if self.logprobs_mode not in (None, "processed_logprobs", "raw_logprobs"):
            raise ValueError("logprobs_mode must be processed_logprobs or raw_logprobs")
        if self.provider_advertises_logprobs and self.backend != "litellm":
            raise ValueError("provider_advertises_logprobs applies to litellm models only")
        if self.capture == "logprobs" and self.backend == "litellm" and \
                not self.provider_advertises_logprobs:
            raise ValueError("capture: logprobs on a hosted provider needs "
                             "provider_advertises_logprobs: true (checked against the "
                             "provider's endpoint supported_parameters)")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model id must be a nonempty string")
        if not isinstance(self.extra, Mapping):
            raise ValueError("extra must be a mapping")
        _check_extra(self.extra)
        if self.backend == "litellm" and self.model.startswith("openrouter/") and (
                "provider" not in self.extra):
            raise ValueError("OpenRouter models must pin one provider (provider: {order: "
                             "[<slug>], allow_fallbacks: false}) so the price bound holds")
        object.__setattr__(self, "extra", freeze(self.extra))
        if self.temperature is not None and (
                isinstance(self.temperature, bool) or not isinstance(self.temperature, int | float)
                or not math.isfinite(self.temperature) or self.temperature < 0):
            raise ValueError("temperature must be a finite nonnegative number")
        for name in ("reasoning_reserve_tokens", "retries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.max_tokens is not None and (
                isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int)
                or self.max_tokens < 1):
            raise ValueError("max_tokens must be a positive integer or None")
        if self.seed is not None and (isinstance(self.seed, bool)
                                      or not isinstance(self.seed, int)):
            raise ValueError("seed must be an integer or None")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        missing = {"backend", "model"} - data.keys()
        if missing:
            raise ValueError(f"model config missing {sorted(missing)}")
        extra = {k: v for k, v in data.items() if k not in MODEL_KEYS}
        if data["backend"] in ("litellm", "openai_compatible") and \
                "reasoning_reserve_tokens" not in data:
            raise ValueError("provider model configs must state reasoning_reserve_tokens")
        temperature = data.get("temperature")
        max_tokens = data.get("max_tokens")
        return cls(
            backend=data["backend"], model=str(data["model"]),
            temperature=None if temperature is None else float(temperature),
            seed=data.get("seed"),
            max_tokens=None if max_tokens is None else int(max_tokens),
            reasoning_reserve_tokens=int(data.get("reasoning_reserve_tokens", 0)),
            retries=int(data.get("retries", 3)), extra=extra,
            capture=str(data.get("capture", "messages")), api_base=data.get("api_base"),
            api_key_env=data.get("api_key_env"),
            provider_advertises_logprobs=bool(data.get("provider_advertises_logprobs", False)),
            logprobs_mode=data.get("logprobs_mode"), checkpoint=data.get("checkpoint"),
            serving_session=data.get("serving_session"),
        )

    def extra_params(self) -> dict[str, Any]:
        """A fresh mutable copy of the extras, for building a request."""
        return thaw(self.extra)

    def to_dict(self) -> dict[str, Any]:
        out = {name: getattr(self, name) for name in
               ("backend", "model", "temperature", "seed", "max_tokens",
                "reasoning_reserve_tokens", "retries")}
        # Capture settings only when set, so existing configs keep their hashes.
        for name, default in (("capture", "messages"), ("api_base", None),
                              ("api_key_env", None), ("provider_advertises_logprobs", False),
                              ("logprobs_mode", None), ("checkpoint", None),
                              ("serving_session", None)):
            if getattr(self, name) != default:
                out[name] = getattr(self, name)
        out.update(self.extra_params())
        return out

    def capture_levels(self) -> tuple[str, ...]:
        """Capture levels this config may request (each with CAPTURE_FIELDS)."""
        if self.capture == "messages":
            return ("messages",)
        if self.backend == "litellm":
            advertised = self.provider_advertises_logprobs
            return ("messages", "logprobs") if advertised else ("messages",)
        return ("messages", "logprobs") if self.capture == "logprobs" else (
            "messages", "logprobs", "tokens")


@dataclass(frozen=True)
class Price:
    """USD per million tokens, the model's maximum output and context length."""

    input: float
    output: float
    cached_input: float
    max_output_tokens: int | None = None
    context_window: int | None = None

    def __post_init__(self) -> None:
        for name in ("input", "output", "cached_input"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not (
                    math.isfinite(value) and value >= 0):
                raise ValueError(f"price {name} must be a finite nonnegative number")
        for name in ("max_output_tokens", "context_window"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                      or value < 1):
                raise ValueError(f"{name} must be a positive integer")

    def usd(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        cached = min(cached_tokens, input_tokens)
        return (
            (input_tokens - cached) * self.input
            + cached * self.cached_input
            + output_tokens * self.output
        ) / 1_000_000


def load_prices(path: str | Path) -> dict[str, Price]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    models = raw.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"{path}: expected a 'models' mapping")
    prices = {}
    for model, entry in models.items():
        try:
            for name in ("max_output_tokens", "context_window"):
                if isinstance(entry.get(name), float):
                    raise ValueError(f"{name} must be an integer")
            price = Price(
                input=float(entry["input_per_mtok"]),
                output=float(entry["output_per_mtok"]),
                cached_input=float(entry.get("cached_input_per_mtok", entry["input_per_mtok"])),
                max_output_tokens=entry.get("max_output_tokens"),
                context_window=entry.get("context_window"),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: bad price entry for {model!r}: {exc}") from exc
        prices[str(model)] = price
    return prices


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class RoleTally:
    calls: int = 0
    failed_attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    usd: float = 0.0


def input_token_bound(payload: Any) -> int:
    """Upper bound on input tokens: UTF-8 bytes of the request plus allowances."""
    items = payload if isinstance(payload, list) else [payload]
    tools = sum(1 for item in items if isinstance(item, dict) and item.get("type") == "function")
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    return (len(body) + PER_MESSAGE_OVERHEAD_TOKENS * len(items)
            + PER_TOOL_OVERHEAD_TOKENS * tools + TEMPLATE_OVERHEAD_TOKENS)


def _read(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def usage_from_response(response: Any) -> Usage | None:
    """Read token usage from a raw provider response, or None if absent."""
    usage = _read(response, "usage")
    if usage is None:
        return None
    prompt, completion = _read(usage, "prompt_tokens"), _read(usage, "completion_tokens")
    if prompt is None or completion is None:
        return None
    cached = _read(_read(usage, "prompt_tokens_details"), "cached_tokens")
    return Usage(int(prompt), int(completion), int(cached or 0))


def reported_cost_from_response(response: Any) -> float | None:
    """Provider-reported dollars, if any (OpenRouter returns usage.cost).

    litellm also copies it into the hidden ``llm_provider-x-litellm-response-cost``
    header. OpenAI reports no cost; the price table is the only source there.
    """
    cost = _read(_read(response, "usage"), "cost")
    if cost is None:
        hidden = getattr(response, "_hidden_params", None) or {}
        cost = (hidden.get("additional_headers") or {}).get(
            "llm_provider-x-litellm-response-cost")
    try:
        cost = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        return None
    return cost if cost is not None and math.isfinite(cost) and cost >= 0 else None


def _plain(value: Any) -> Any:
    """JSON-safe plain copy of a provider object (pydantic models, dicts, lists)."""
    if hasattr(value, "model_dump"):
        with contextlib.suppress(Exception):
            value = value.model_dump()
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value if not (isinstance(value, float) and not math.isfinite(value)) else None
    return str(value)


def raw_response_record(response: Any) -> dict[str, Any]:
    """The provider response's own fields, read before tau2 parses anything.

    ``provider`` is the serving provider OpenRouter reports in its response
    body; ``model`` is the model id the response reports. Tool-call arguments
    stay the raw strings the model produced.
    """
    try:
        choices = _read(response, "choices") or []
        choice = choices[0] if choices else None
        message = _read(choice, "message")
        calls = []
        for call in _read(message, "tool_calls") or []:
            function = _read(call, "function")
            calls.append({"id": _read(call, "id"), "type": _read(call, "type") or "function",
                          "function": {"name": _read(function, "name"),
                                       "arguments": _read(function, "arguments")}})
        hidden = getattr(response, "_hidden_params", None) or {}
        provider = _read(response, "provider") or (
            (hidden.get("additional_headers") or {}).get("llm_provider-x-openrouter-provider"))
        record = {
            "response_id": _read(response, "id"),
            "model_reported": _read(response, "model"),
            "provider": provider,
            "finish_reason": _read(choice, "finish_reason"),
            "content": _read(message, "content"),
            "tool_calls": calls,
            "reasoning_content": _read(message, "reasoning_content"),
            "usage": _plain(_read(response, "usage")),
            "choices": len(choices),
        }
        record = _plain(record)
        # Graded token data (slh.tokens); stored apart from calls.jsonl.
        from slh.tokens import extract

        record["token_capture"] = extract(response)
        return record
    except Exception as exc:  # noqa: BLE001 - capture must never fail a charged call
        return {"capture_error": type(exc).__name__}


@dataclass
class _Attempt:
    config: ModelConfig
    responses: int = 0
    usage: Usage | None = None
    unknown_usage: bool = False
    reported_cost: float | None = None
    dispatched: bool = False
    rejected: bool = False
    raw: list[dict[str, Any]] = field(default_factory=list)
    wire_latency_s: float | None = None
    # The reservation covers exactly one dispatch of a request no larger than
    # the authorized one (its input-token bound), checked at the wire.
    authorized_bound: int | None = None
    authorized_sha256: str | None = None
    ledger: Any = None  # re-checked for a run halt immediately before sending

    def record(self, usage: Usage | None, reported_cost: float | None) -> None:
        self.responses += 1
        if reported_cost is not None:
            self.reported_cost = (self.reported_cost or 0.0) + reported_cost
        if usage is None:
            self.unknown_usage = True
            return
        if self.usage is None:
            self.usage = Usage()
        self.usage.input_tokens += usage.input_tokens
        self.usage.output_tokens += usage.output_tokens
        self.usage.cached_tokens += usage.cached_tokens


_attempt: contextvars.ContextVar[_Attempt | None] = contextvars.ContextVar(
    "slh_metered_attempt", default=None
)


def _exactly(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Equal keys and values with identical JSON types (True is not 1, 0 is not False)."""
    return set(a) == set(b) and all(type(a[k]) is type(b[k]) and a[k] == b[k] for k in a)


def check_capture_fields(kwargs: Mapping[str, Any], config: ModelConfig) -> dict[str, Any]:
    """The capture fields of a request must be exactly one allowed set; returns
    the request without them."""
    present = {k: kwargs[k] for k in ("logprobs", "top_logprobs", "return_token_ids")
               if k in kwargs}
    if not any(_exactly(present, CAPTURE_FIELDS[level]) for level in config.capture_levels()):
        raise DispatchRejected("request capture fields differ from what the config allows")
    return {k: v for k, v in kwargs.items() if k not in present}


def check_dispatch(kwargs: Mapping[str, Any], config: ModelConfig) -> None:
    """Re-validate the exact outgoing parameters against the authorized config."""
    kwargs = check_capture_fields(kwargs, config)
    unknown = sorted(set(kwargs) - DISPATCH_KEYS)
    if unknown:
        raise DispatchRejected(f"request parameters not allowed at dispatch: {unknown}")
    if kwargs.get("model") != config.model:
        raise DispatchRejected("request model differs from the authorized model")
    if kwargs.get("max_tokens") != config.max_tokens:
        raise DispatchRejected("request max_tokens differs from the authorized config")
    extras = {k: kwargs[k] for k in ALLOWED_EXTRA if k in kwargs}
    try:
        _check_extra(extras)
    except ValueError as exc:
        raise DispatchRejected(str(exc)) from None
    if thaw(extras) != {k: v for k, v in config.extra_params().items() if k in extras} or (
            set(extras) != set(config.extra)):
        raise DispatchRejected("request extras differ from the authorized config")


class Meter:
    """Per-run hard cap plus per-role usage tallies.

    ``ledger`` (a ``slh.ledger.SharedLedger``) makes the cap global across
    worker processes; without it the cap covers this meter's own spend.
    """

    def __init__(self, cap_usd: float, prices: dict[str, Price], ledger: Any = None,
                 pause_after: int = PAUSE_AFTER, pause_s: float = PAUSE_S,
                 max_pauses: int = MAX_PAUSES) -> None:
        if cap_usd < 0 or not math.isfinite(cap_usd):
            raise ValueError("budget cap must be a finite nonnegative number")
        if isinstance(pause_after, bool) or not isinstance(pause_after, int) or pause_after < 1:
            raise ValueError("pause_after must be a positive integer")
        if isinstance(max_pauses, bool) or not isinstance(max_pauses, int) or max_pauses < 0:
            raise ValueError("max_pauses must be a nonnegative integer")
        if not math.isfinite(pause_s) or pause_s < 0:
            raise ValueError("pause_s must be a finite nonnegative number")
        self.cap_usd = float(cap_usd)
        self.prices = prices
        self.ledger = ledger
        self.pause_after = pause_after
        self.pause_s = float(pause_s)
        self.max_pauses = max_pauses
        self.transient_streak = 0
        self.pauses: list[dict[str, Any]] = []
        # Called after every attempt with its capture fields (see module docstring).
        self.on_attempt: Callable[[dict[str, Any]], None] | None = None
        self._last_cost = 0.0
        self._ledger_charged = 0.0
        self.tally = {role: RoleTally() for role in ROLES}
        self.refused: list[dict[str, Any]] = []
        self.overspends: list[dict[str, Any]] = []
        self.underestimates: list[dict[str, Any]] = []
        self.unreconciled: list[dict[str, Any]] = []
        # The reservation of a paid attempt in flight, or None.
        self.pending: dict[str, Any] | None = None
        # Called when a paid reservation becomes pending and after every
        # charged attempt (success, failure or interrupt), so callers can
        # persist spend and pending reservations immediately.
        self.on_change: Callable[[Meter], None] | None = None

    @property
    def spent_usd(self) -> float:
        """This meter's own spend (this process, this session plus restored)."""
        return sum(t.usd for t in self.tally.values())

    def _run_spent(self) -> float:
        """Spend the cap applies to: the whole run's charged spend with a ledger."""
        return self._ledger_charged if self.ledger is not None else self.spent_usd

    def _notify(self) -> None:
        if self.on_change is not None:
            self.on_change(self)

    def price_for(self, model: str) -> Price:
        if model not in self.prices:
            raise BudgetExceeded(f"no price configured for model {model!r}; refusing to call")
        return self.prices[model]

    def output_token_bound(self, config: ModelConfig) -> int:
        price = self.price_for(config.model)
        limit = config.max_tokens if config.max_tokens is not None else price.max_output_tokens
        if limit is None:
            if price.output == 0 and price.input == 0:
                return 0
            raise BudgetExceeded(f"no output bound for {config.model!r}: set max_tokens or "
                                 "max_output_tokens in the price table")
        return limit + config.reasoning_reserve_tokens

    def reservation_usd(self, config: ModelConfig, request_payload: Any) -> float:
        price = self.price_for(config.model)
        return price.usd(input_token_bound(request_payload), self.output_token_bound(config))

    def call(
        self,
        role: str,
        config: ModelConfig,
        request_payload: Any,
        invoke: Callable[[], tuple[Any, Usage | None]],
    ) -> Any:
        """Reserve, persist, run ``invoke`` (with retries), charge, check.

        ``invoke`` returns (result, usage). Provider calls made inside it are
        checked and charged at the wire by ``completion_guard``; that usage
        takes precedence over what ``invoke`` returns.
        """
        if role not in self.tally:
            raise ValueError(f"unknown role {role!r}")
        reservation = self.reservation_usd(config, request_payload)
        if not math.isfinite(reservation) or reservation < 0:
            raise BudgetExceeded(f"non-finite reservation for {config.model!r}; refusing")
        attempts = config.retries + 1 if config.backend != "stub" else 1
        consumed = 0  # attempts that count against the retry budget
        number = 0  # attempts made
        while True:
            rid = self._authorize(role, config, reservation, number + 1)
            number += 1
            attempt = _Attempt(config=config,
                               authorized_bound=input_token_bound(request_payload),
                               authorized_sha256=_payload_sha256(request_payload),
                               ledger=self.ledger)
            token = _attempt.set(attempt)
            failure: BaseException | None = None
            result: Any = None
            started = time.perf_counter()
            try:
                result, returned_usage = invoke()
            except BaseException as exc:  # noqa: BLE001 - interrupts are charged too
                failure = exc
            finally:
                # Authorization ends with the attempt, whatever happened.
                _attempt.reset(token)
            elapsed = time.perf_counter() - started
            if failure is None:
                try:
                    self._charge(role, config, attempt, returned_usage, reservation,
                                 failed=False, rid=rid)
                finally:
                    self._emit(role, config, request_payload, attempt, number, reservation,
                               result, None, elapsed)
                self.transient_streak = 0
                return result
            interrupt = not isinstance(failure, Exception)
            try:
                try:
                    # A recorded response is charged its usage; otherwise (including an
                    # interrupt before any response) the full reservation.
                    self._charge(role, config, attempt, None, reservation, failed=True, rid=rid)
                except Overspend:
                    if not interrupt:
                        raise
            finally:
                self._emit(role, config, request_payload, attempt, number, reservation,
                           None, failure, elapsed)
            if attempt.responses:
                self.transient_streak = 0
            transient = type(failure).__name__ in TRANSIENT_ERRORS and attempt.responses == 0
            if interrupt or not transient:
                raise failure
            self.transient_streak += 1
            if (config.backend != "stub" and self.transient_streak >= self.pause_after
                    and len(self.pauses) < self.max_pauses):
                # A streak of transient failures: wait longer instead of burning
                # retries; this failed attempt does not count against them.
                wait = pause_seconds(self.pause_s)
                self.pauses.append({"role": role, "model": config.model,
                                    "after_consecutive_failures": self.transient_streak,
                                    "pause_s": round(wait, 3), "error_type":
                                        type(failure).__name__,
                                    "time_utc": dt.datetime.now(dt.UTC).isoformat()})
                self.transient_streak = 0
                self._notify()
                _sleep(wait)
                continue
            consumed += 1
            if consumed >= attempts:
                raise failure
            _sleep(backoff_seconds(consumed - 1))

    def _authorize(self, role: str, config: ModelConfig, reservation: float,
                   number: int) -> str | None:
        """Refuse an attempt that could pass the cap; else make its reservation
        durable before dispatch. Returns the ledger reservation id, if any."""
        if self.ledger is None:
            spent = self.spent_usd
            if not math.isfinite(spent) or spent + reservation > self.cap_usd + 1e-12:
                self.refused.append({"role": role, "model": config.model,
                                     "reservation_usd": reservation,
                                     "spent_usd": spent, "cap_usd": self.cap_usd})
                raise BudgetExceeded(
                    f"{role} call to {config.model!r} reserves up to ${reservation:.6f}; "
                    f"spent ${spent:.6f} of ${self.cap_usd:.6f} cap"
                )
            if reservation > 0:
                self.pending = {"role": role, "model": config.model,
                                "reservation_usd": reservation, "attempt": number,
                                "started_utc": dt.datetime.now(dt.UTC).isoformat()}
                self._notify()  # durable before dispatch
            return None
        try:
            if reservation > 0:
                rid = self.ledger.reserve(role, config.model, reservation, number)
            else:
                self.ledger.check()
                rid = None
        except BudgetExceeded:
            self.refused.append({"role": role, "model": config.model,
                                 "reservation_usd": reservation,
                                 "spent_usd": self.spent_usd, "cap_usd": self.cap_usd,
                                 "scope": "run (all workers)"})
            raise
        self._ledger_charged = self.ledger.charged_usd
        if rid is not None:
            self.pending = {"role": role, "model": config.model,
                            "reservation_usd": reservation, "attempt": number,
                            "reservation_id": rid,
                            "started_utc": dt.datetime.now(dt.UTC).isoformat()}
            self._notify()  # the ledger append above was already durable
        return rid

    def _emit(self, role: str, config: ModelConfig, request_payload: Any, attempt: _Attempt,
              number: int, reservation: float, result: Any, failure: BaseException | None,
              elapsed: float) -> None:
        if self.on_attempt is None:
            return
        self.on_attempt({
            "role": role, "config": config, "attempt": number,
            "reservation_usd": reservation, "charged_usd": self._last_cost,
            "error_type": None if failure is None else type(failure).__name__,
            "responses": list(attempt.raw), "result": result,
            "dispatched": attempt.dispatched,
            "rejected_before_dispatch": attempt.rejected and not attempt.dispatched,
            "attempt_s": elapsed, "wire_latency_s": attempt.wire_latency_s,
            "request_payload": request_payload,
        })

    def _charge(self, role: str, config: ModelConfig, attempt: _Attempt,
                returned: Usage | None, reservation: float, *, failed: bool,
                rid: str | None = None) -> None:
        tally = self.tally[role]
        if failed:
            tally.failed_attempts += 1
        else:
            tally.calls += 1
        usage = attempt.usage if attempt.responses else returned
        table_cost = None
        if failed and attempt.rejected and not attempt.dispatched:
            cost = 0.0  # rejected before dispatch: nothing reached the provider
        elif usage is None or attempt.unknown_usage:
            # Unknown usage (including a failure before any response) costs
            # the full reservation.
            cost = reservation
        else:
            tally.input_tokens += usage.input_tokens
            tally.output_tokens += usage.output_tokens
            tally.cached_tokens += usage.cached_tokens
            table_cost = self.price_for(config.model).usd(
                usage.input_tokens, usage.output_tokens, usage.cached_tokens)
            cost = table_cost
        underestimate = None
        if attempt.reported_cost is not None:
            if table_cost is not None and attempt.reported_cost > table_cost * (
                    1 + UNDERESTIMATE_TOLERANCE) + 1e-12:
                underestimate = {"role": role, "model": config.model,
                                 "table_usd": table_cost,
                                 "reported_usd": attempt.reported_cost}
                self.underestimates.append(underestimate)
            cost = max(cost, attempt.reported_cost)
        self._last_cost = cost
        tally.usd += cost
        if self.ledger is not None and (rid is not None or cost > 0):
            self._ledger_charged = self.ledger.settle(rid, cost, role, config.model)
        self.pending = None
        self._notify()
        if underestimate is not None:
            if self.ledger is not None:
                self.ledger.halt("price_table_underestimate")
            raise Overspend(f"provider-reported cost ${underestimate['reported_usd']:.6f} "
                            f"exceeds the price table's ${underestimate['table_usd']:.6f} "
                            f"for {config.model!r} (price_table_underestimate)")
        if (not math.isfinite(cost) or cost > reservation + 1e-12
                or self._run_spent() > self.cap_usd + 1e-12):
            event = {"role": role, "model": config.model, "cost_usd": cost,
                     "reservation_usd": reservation, "spent_usd": self.spent_usd,
                     "cap_usd": self.cap_usd}
            self.overspends.append(event)
            if self.ledger is not None:
                self.ledger.halt("overspend")
            raise Overspend(f"{role} call to {config.model!r} cost ${cost:.6f} against a "
                            f"${reservation:.6f} reservation; spent ${self.spent_usd:.6f} of "
                            f"${self.cap_usd:.6f} cap")

    def restore(self, spend: Mapping[str, Any]) -> None:
        """Resume: carry spend recorded by an earlier session of the same run.

        A pending reservation that was never resolved (the process died during
        a paid call) is charged in full and recorded as unreconciled.
        """
        for role, values in (spend.get("by_role") or {}).items():
            if role in self.tally:
                self.tally[role] = RoleTally(**{k: values.get(k, 0)
                                                for k in RoleTally.__dataclass_fields__})
        self.refused = list(spend.get("refused_calls") or [])
        self.overspends = list(spend.get("overspends") or [])
        self.underestimates = list(spend.get("price_table_underestimates") or [])
        self.unreconciled = list(spend.get("unreconciled_reservations") or [])
        pending = spend.get("pending")
        if pending:
            role = pending.get("role") if pending.get("role") in self.tally else "agent"
            self.tally[role].usd += float(pending["reservation_usd"])
            self.tally[role].failed_attempts += 1
            self.unreconciled.append({**pending, "status": "unreconciled_reservation"})
        self.pending = None

    def snapshot(self) -> dict[str, RoleTally]:
        return {role: RoleTally(**asdict(t)) for role, t in self.tally.items()}

    def summary(self) -> dict[str, Any]:
        return {
            "cap_usd": self.cap_usd,
            "spent_usd": round(self.spent_usd, 9),
            "by_role": {role: asdict(t) for role, t in self.tally.items()},
            "pending": self.pending,
            "refused_calls": list(self.refused),
            "overspends": list(self.overspends),
            "price_table_underestimates": list(self.underestimates),
            "unreconciled_reservations": list(self.unreconciled),
            "rate_limit_pauses": list(self.pauses),
        }


def delta(before: dict[str, RoleTally], after: dict[str, RoleTally]) -> dict[str, RoleTally]:
    return {
        role: RoleTally(**{
            name: getattr(after[role], name) - getattr(before[role], name)
            for name in RoleTally.__dataclass_fields__
        })
        for role in after
    }


def _payload_sha256(payload: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _authorize_wire(attempt: _Attempt, kwargs: Mapping[str, Any]) -> None:
    """Last checks before bytes leave: parameters, one dispatch per reservation,
    a request no larger than the one reserved for, and no run halt since."""
    try:
        if attempt.dispatched:
            raise DispatchRejected("a reservation covers exactly one dispatch")
        check_dispatch(kwargs, attempt.config)
        wire = list(kwargs.get("messages") or []) + list(kwargs.get("tools") or [])
        if attempt.authorized_bound is not None and \
                input_token_bound(wire) > attempt.authorized_bound:
            raise DispatchRejected("request is larger than the reserved request")
    except DispatchRejected:
        attempt.rejected = True
        raise
    if attempt.ledger is not None:
        try:
            attempt.ledger.assert_not_halted()
        except BudgetExceeded:
            attempt.rejected = True
            raise


def metered_dispatch(body: Mapping[str, Any], send: Callable[[dict[str, Any]], Any],
                     gate: Callable[[], None] | None = None) -> Any:
    """Dispatch a request built by a harness backend that talks HTTP itself
    (openai_compatible), with the same checks as ``completion_guard``: inside an
    authorized attempt only, parameters re-validated against the authorized
    config (including temperature and the capture fields), usage and raw
    response recorded the moment the response returns.

    ``gate`` (if given) runs last, immediately before the request is sent; an
    exception from it (e.g. ``slh.serving.ServingWindowClosed`` after a serving
    window's deadline) rejects the attempt before dispatch, at no cost."""
    attempt = _attempt.get()
    if attempt is None:
        raise UnmeteredCall("model call attempted outside the budget meter")
    if body.get("temperature") != attempt.config.temperature:
        attempt.rejected = True
        raise DispatchRejected("request temperature differs from the authorized config")
    _authorize_wire(attempt, body)
    if gate is not None:
        try:
            gate()
        except BaseException:
            attempt.rejected = True
            raise
    attempt.dispatched = True
    started = time.perf_counter()
    response = send(dict(body))
    attempt.wire_latency_s = time.perf_counter() - started
    attempt.record(usage_from_response(response), reported_cost_from_response(response))
    attempt.raw.append(raw_response_record(response))
    return response


@contextlib.contextmanager
def completion_guard() -> Iterator[None]:
    """Check and meter every provider completion at the wire; refuse unmetered ones.

    Inside an authorized attempt the wrapper forces provider retries off,
    re-validates the outgoing parameters against the authorized config, and
    records the response's usage and reported cost the moment it returns.
    Outside an attempt it raises.
    """
    import litellm
    from tau2.utils import llm_utils

    targets = [(llm_utils, "completion"), (litellm, "completion"), (litellm, "acompletion")]
    originals = [(obj, name, getattr(obj, name)) for obj, name in targets]

    def wrap(original: Callable[..., Any]) -> Callable[..., Any]:
        def guarded(*args: Any, **kwargs: Any) -> Any:
            attempt = _attempt.get()
            if attempt is None:
                raise UnmeteredCall("model call attempted outside the budget meter")
            kwargs["num_retries"] = 0
            kwargs.pop("max_retries", None)
            if args:
                raise DispatchRejected("positional request arguments are not allowed")
            _authorize_wire(attempt, kwargs)
            attempt.dispatched = True
            started = time.perf_counter()
            response = original(**kwargs)
            attempt.wire_latency_s = time.perf_counter() - started
            attempt.record(usage_from_response(response), reported_cost_from_response(response))
            attempt.raw.append(raw_response_record(response))
            return response

        return guarded

    for obj, name, original in originals:
        setattr(obj, name, wrap(original))
    try:
        yield
    finally:
        for obj, name, original in originals:
            setattr(obj, name, original)
