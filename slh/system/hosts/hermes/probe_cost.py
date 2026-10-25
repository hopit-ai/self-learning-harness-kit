"""Offline all-in episode accounting, with explicit estimation provenance.

Record EVERY attempted model/tool call, including failed attempts and retries.
Retry flags classify those costs; they never add an imaginary extra request.
Prompt totals include cache hits: new prefill = prompt - cached. Rates are
caller-supplied USD per million tokens, not prices inferred from model names.
Tool rereads are diagnostic subsets, never charged twice on top of model usage.

``record()`` is numbers only. ``summarize_episodes()`` reports episode p50/p95;
per-call p50/p95 live in each episode record. Percentiles use nearest rank.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .probe_storage import canonical_bytes


def _nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return number


def _tokens(value: Any, label: str) -> int:
    number = _nonnegative(value, label)
    if not number.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(number)


@dataclass(frozen=True)
class TokenRates:
    prefill: float
    decode: float
    cached: float

    def __post_init__(self) -> None:
        for name in ("prefill", "decode", "cached"):
            _nonnegative(getattr(self, name), name)

    def cost(self, prompt: int, decode: int, cached: int) -> float:
        prompt = _tokens(prompt, "prompt tokens")
        decode = _tokens(decode, "decode tokens")
        cached = _tokens(cached, "cached tokens")
        if cached > prompt:
            raise ValueError("cached tokens cannot exceed prompt tokens")
        return _nonnegative(((prompt - cached) * self.prefill + decode * self.decode
                             + cached * self.cached) / 1_000_000, "token cost")


def estimate_tokens(value: Any) -> int:
    """Heuristic only: ceil(serialized UTF-8 bytes / 4), NOT a token bound."""
    return math.ceil(len(canonical_bytes(value)) / 4)


def _reported(usage: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if usage.get(key) is not None:
            return usage[key]
    return None


def _count(usage: Mapping[str, Any], keys: tuple[str, ...], estimate: int | None,
           text: Any, label: str) -> tuple[int, int]:
    value = _reported(usage, *keys)
    if value is not None:
        return _tokens(value, label), 0
    if estimate is not None:
        return _tokens(estimate, label), 1
    if text is not None:
        return estimate_tokens(text), 1
    raise ValueError(f"missing {label}: provide usage, an estimate, or text")


def call_fingerprint(name: str, arguments: Any) -> str:
    """Canonical name+arguments digest; JSON strings and objects compare equally.

    Argument order/whitespace is ignored; different tools or argument values do
    not match. Invalid JSON argument strings are hashed literally.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("tool name must be nonempty")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            pass
    return hashlib.sha256(canonical_bytes([name, arguments])).hexdigest()


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest rank, including 0/100 endpoints; empty series returns zero."""
    if not math.isfinite(q) or not 0 <= q <= 100:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(_nonnegative(v, "latency") for v in values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)] if ordered else 0.0


class EpisodeCost:
    """One accumulator per episode: base is the frozen 9B, small is the selector.

    ``small_rates`` is required only if a small-model call is recorded. Include
    feature extraction, restore costs and amortized training via add_overhead.
    For exact wall latency, pass elapsed_ms to record; otherwise the sum of
    recorded operation latencies is returned and marked estimated (parallel work
    may overlap). Supply latency for failed calls as well as successful calls.
    """

    def __init__(self, base_rates: TokenRates, *, small_rates: TokenRates | None = None) -> None:
        self._rates = {"base": base_rates, "small": small_rates}
        self._models: dict[str, dict[str, int | float]] = {
            name: {"calls": 0, "prompt": 0, "decode": 0, "cached": 0,
                   "prompt_estimates": 0, "decode_estimates": 0, "cache_estimates": 0,
                   "usd": 0.0} for name in self._rates}
        self._latencies: list[float] = []
        self._missing_latencies = 0
        self._retries = 0
        self._retry_usd = 0.0
        self._tool_calls = 0
        self._pruned: set[str] = set()
        self._recalls = 0
        self._reread_tokens = 0
        self._restores = 0
        self._tool_usd = 0.0
        self._overhead_usd = 0.0
        self._extra_latency = 0.0

    def add_model_call(self, usage: Mapping[str, Any] | None = None, *,
                       model: str = "base", prompt: Any = None, completion: Any = None,
                       estimated_prompt_tokens: int | None = None,
                       estimated_completion_tokens: int | None = None,
                       estimated_cached_tokens: int = 0, latency_ms: float | None = None,
                       retry: bool = False) -> None:
        """Record one attempt, including selector, audit, and retry calls.

        Reads OpenAI/vLLM prompt_tokens / completion_tokens and
        prompt_tokens_details.cached_tokens (or top-level cached_tokens /
        num_cached_tokens / prompt_cache_hit_tokens). Also accepts input_tokens /
        output_tokens / input_tokens_details.cached_tokens. Missing cache usage
        defaults to estimated zero; even reported zero is treated as measured.
        Missing token counts REQUIRE explicit estimates or the corresponding text.
        """
        if model not in self._rates or self._rates[model] is None:
            raise ValueError("model must be base, or small with explicit small_rates")
        if usage is not None and not isinstance(usage, Mapping):
            raise ValueError("usage must be a mapping")
        usage = usage or {}
        prompt_n, prompt_est = _count(usage, ("prompt_tokens", "input_tokens"),
                                     estimated_prompt_tokens, prompt, "prompt tokens")
        decode_n, decode_est = _count(usage, ("completion_tokens", "output_tokens"),
                                     estimated_completion_tokens, completion, "decode tokens")
        cache = None
        for key in ("prompt_tokens_details", "input_tokens_details"):
            details = usage.get(key)
            if isinstance(details, Mapping):
                cache = _reported(details, "cached_tokens")
                if cache is not None:
                    break
        if cache is None:
            cache = _reported(usage, "cached_tokens", "num_cached_tokens", "prompt_cache_hit_tokens")
        cache_est = int(cache is None)
        cached_n = _tokens(estimated_cached_tokens if cache_est else cache, "cached tokens")
        if cached_n > prompt_n:
            raise ValueError("cached tokens cannot exceed prompt tokens")
        latency = None if latency_ms is None else _nonnegative(latency_ms, "latency_ms")
        rates = self._rates[model]
        assert rates is not None
        usd = rates.cost(prompt_n, decode_n, cached_n)
        rec = self._models[model]
        for key, value in (("calls", 1), ("prompt", prompt_n), ("decode", decode_n),
                           ("cached", cached_n), ("prompt_estimates", prompt_est),
                           ("decode_estimates", decode_est), ("cache_estimates", cache_est),
                           ("usd", usd)):
            rec[key] += value
        self._observe_latency(latency)
        if retry:
            self._retries += 1
            self._retry_usd += usd

    def _observe_latency(self, latency: float | None) -> None:
        if latency is None:
            self._missing_latencies += 1
        else:
            self._latencies.append(latency)

    def mark_pruned_call(self, name: str, arguments: Any) -> str:
        """Mark at the pruning boundary, BEFORE observing continuation calls."""
        fingerprint = call_fingerprint(name, arguments)
        self._pruned.add(fingerprint)
        return fingerprint

    def mark_pruned_messages(self, messages: Sequence[Mapping[str, Any]],
                             pruned_call_ids: Sequence[str]) -> None:
        """Bridge from PruneResult.pruned_call_ids using the original snapshot.

        Raw IDs and arguments are used locally and never included in record().
        All IDs must resolve to calls; bad input leaves the accumulator unchanged.
        """
        wanted = set(pruned_call_ids)
        found: dict[str, str] = {}
        for message in messages:
            for call in message.get("tool_calls") or []:
                if call.get("id") in wanted:
                    if call["id"] in found:
                        raise ValueError("duplicate pruned call ID")
                    function = call["function"]
                    found[call["id"]] = call_fingerprint(function["name"], function["arguments"])
        if found.keys() != wanted:
            raise ValueError("pruned call IDs do not all resolve")
        self._pruned.update(found.values())

    def observe_tool_call(self, name: str, arguments: Any, *,
                          result_tokens: int = 0, latency_ms: float | None = None,
                          cost_usd: float = 0.0, retry: bool = False) -> bool:
        """Return whether a later invocation repeats a pruned name+arguments.

        result_tokens attributes fetched tokens diagnostically, not an additional
        model charge. Identical arguments to a *different* tool do not count.
        Repetition is an observable proxy, not proof that pruning caused a call.
        """
        fingerprint = call_fingerprint(name, arguments)
        tokens = _tokens(result_tokens, "result_tokens")
        cost = _nonnegative(cost_usd, "cost_usd")
        latency = None if latency_ms is None else _nonnegative(latency_ms, "latency_ms")
        repeated = fingerprint in self._pruned
        self._tool_calls += 1
        self._tool_usd += cost
        self._observe_latency(latency)
        if repeated:
            self._recalls += 1
            self._reread_tokens += tokens
        if retry:
            self._retries += 1
            self._retry_usd += cost
        return repeated

    def observe_restore(self, *, tokens: int = 0, latency_ms: float | None = None,
                        cost_usd: float = 0.0) -> None:
        """Track DTOC local rereads independently of tool re-invocations."""
        count = _tokens(tokens, "tokens")
        latency = None if latency_ms is None else _nonnegative(latency_ms, "latency_ms")
        cost = _nonnegative(cost_usd, "cost_usd")
        self._restores += 1
        self._reread_tokens += count
        self._tool_usd += cost
        self._observe_latency(latency)

    def add_overhead(self, *, cost_usd: float = 0.0, latency_ms: float = 0.0) -> None:
        """Include e.g. CPU scoring, audits outside model calls, amortized training."""
        cost = _nonnegative(cost_usd, "cost_usd")
        latency = _nonnegative(latency_ms, "latency_ms")
        self._overhead_usd += cost
        self._extra_latency += latency

    def record(self, *, elapsed_ms: float | None = None) -> dict[str, int | float]:
        """Compact numeric record; retry_usd is a SUBSET of total_usd."""
        rec: dict[str, int | float] = {}
        for name, model in self._models.items():
            rec.update({f"{name}_{key}": value for key, value in model.items()})
            rec[f"{name}_prefill"] = model["prompt"] - model["cached"]
        elapsed = (sum(self._latencies) + self._extra_latency if elapsed_ms is None
                   else _nonnegative(elapsed_ms, "elapsed_ms"))
        rec.update({
            "tool_calls": self._tool_calls, "pruned_calls": len(self._pruned),
            "pruned_recalls": self._recalls, "restores": self._restores,
            "reread_tokens": self._reread_tokens, "retries": self._retries,
            "retry_usd": self._retry_usd, "tool_usd": self._tool_usd,
            "overhead_usd": self._overhead_usd,
            "total_usd": sum(m["usd"] for m in self._models.values())
            + self._tool_usd + self._overhead_usd,
            "latency_ms": elapsed, "latency_estimated": int(elapsed_ms is None),
            "latency_missing": self._missing_latencies,
            "call_p50_ms": percentile(self._latencies, 50),
            "call_p95_ms": percentile(self._latencies, 95),
        })
        return rec


def summarize_episodes(records: Sequence[Mapping[str, int | float]]) -> dict[str, int | float]:
    """Aggregate wall-time percentiles across episodes (not across requests)."""
    costs = [_nonnegative(r["total_usd"], "total_usd") for r in records]
    latencies = [_nonnegative(r["latency_ms"], "latency_ms") for r in records]
    return {"episodes": len(records), "total_usd": sum(costs),
            "p50_ms": percentile(latencies, 50), "p95_ms": percentile(latencies, 95),
            "latency_estimated": sum(_tokens(r["latency_estimated"], "latency_estimated")
                                     for r in records)}
