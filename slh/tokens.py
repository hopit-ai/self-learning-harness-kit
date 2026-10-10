"""Token-level capture: levels, grading what an endpoint returned, rollout graph.

Capture levels, lowest to highest:

- ``messages``: text only (any hosted API).
- ``logprobs``: a finite log-probability (<= 0) for every sampled token, but
  no token ids (some hosted providers).
- ``tokens``: prompt token ids, sampled token ids and one aligned log-prob per
  sampled token (a self-served vLLM/SGLang server). This is what RL training
  needs.

A call's level is what its own response *verifiably* contains; an episode's
and a run's level is the minimum over their agent calls. Nothing is claimed
that was not checked: logprobs must align one-to-one with the sampled tokens
(and with ``usage.completion_tokens`` when reported); a response that carries
misaligned or non-finite values is graded ``messages`` and marked not
trainable.

Whether the returned log-probs are the sampler's *processed* distribution
(after temperature / top-p) or the raw model distribution cannot be read off a
response. The server flag ``--logprobs-mode processed_logprobs`` (vLLM) makes
them the processed ones; with temperature 1 and top_p 1 the two coincide. The
mode is recorded as declared in the config and marked ``verified`` only in
that coinciding case.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

LEVELS = ("messages", "logprobs", "tokens")
TOKEN_ID_PREFIX = "token_id:"  # vLLM --return-tokens-as-token-ids


def level_rank(level: str) -> int:
    return LEVELS.index(level)


def min_level(levels: list[str]) -> str:
    return min(levels, key=level_rank) if levels else "messages"


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:  # noqa: BLE001
            return None
    return value


def extract(response: Any) -> dict[str, Any]:
    """Grade one completion response; return its level and token data.

    Reads the OpenAI ``choices[0].logprobs.content[*].logprob`` list, sampled
    ids from vLLM's ``choices[0].token_ids`` (``return_token_ids``) or from
    ``token_id:<n>`` token strings (``--return-tokens-as-token-ids``), and
    prompt ids from ``prompt_token_ids`` (top level or on the choice).
    """
    response = _plain(response)
    choices = _get(response, "choices") or []
    choice = _plain(choices[0]) if choices else None
    logprobs_obj = _plain(_get(choice, "logprobs"))
    content = _get(logprobs_obj, "content") or []
    logprobs = []
    token_strings = []
    for entry in content:
        entry = _plain(entry)
        logprobs.append(_get(entry, "logprob"))
        token_strings.append(_get(entry, "token"))
    sampled = _get(choice, "token_ids")
    if sampled is None and token_strings and all(
            isinstance(t, str) and t.startswith(TOKEN_ID_PREFIX) for t in token_strings):
        try:
            sampled = [int(t[len(TOKEN_ID_PREFIX):]) for t in token_strings]
        except ValueError:
            sampled = None
    prompt = _get(response, "prompt_token_ids") or _get(choice, "prompt_token_ids")
    usage = _plain(_get(response, "usage"))
    completion_tokens = _get(usage, "completion_tokens")
    prompt_tokens = _get(usage, "prompt_tokens")
    problems: list[str] = []

    def ids_ok(ids: Any) -> bool:
        return isinstance(ids, list) and bool(ids) and all(
            isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in ids)

    have_logprobs = bool(logprobs)
    if have_logprobs and not all(isinstance(v, int | float) and not isinstance(v, bool)
                                 and math.isfinite(v) and v <= 0 for v in logprobs):
        problems.append("non_finite_or_positive_logprob")
    if sampled is not None and not ids_ok(sampled):
        problems.append("bad_sampled_ids")
        sampled = None
    if prompt is not None and not ids_ok(prompt):
        problems.append("bad_prompt_ids")
        prompt = None
    if have_logprobs and sampled is not None and len(sampled) != len(logprobs):
        problems.append("logprobs_not_aligned_with_sampled_ids")
    if have_logprobs:
        # Every log-prob must name the token it scores, and the server's own
        # token counts must agree with what came back; otherwise nothing is verified.
        if not all(isinstance(t, str) and t for t in token_strings):
            problems.append("logprob_entry_without_sampled_token")
        if not isinstance(completion_tokens, int) or isinstance(completion_tokens, bool):
            problems.append("usage_completion_tokens_missing")
        elif completion_tokens != len(logprobs):
            problems.append("logprobs_not_aligned_with_completion_tokens")
        if sampled is not None and isinstance(completion_tokens, int) and \
                completion_tokens != len(sampled):
            problems.append("sampled_ids_not_aligned_with_completion_tokens")
        if prompt is not None:
            if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool):
                problems.append("usage_prompt_tokens_missing")
            elif prompt_tokens != len(prompt):
                problems.append("prompt_ids_not_aligned_with_prompt_tokens")
    if (sampled is not None and all(isinstance(t, str) and t.startswith(TOKEN_ID_PREFIX)
                                    for t in token_strings) and token_strings
            and [int(t[len(TOKEN_ID_PREFIX):]) for t in token_strings] != sampled):
        problems.append("token_strings_disagree_with_ids")
    if problems and have_logprobs:
        level = "messages"
    elif have_logprobs and sampled is not None and prompt is not None:
        level = "tokens"
    elif have_logprobs:
        level = "logprobs"
    else:
        level = "messages"
    return {"level": level, "trainable": level != "messages" and not problems,
            "problems": problems, "prompt_token_ids": prompt, "token_ids": sampled,
            "logprobs": logprobs if have_logprobs else None,
            "finish_reason": _get(choice, "finish_reason")}


LOGPROBS_MODES = ("processed_logprobs", "raw_logprobs")


def sampling_untransformed(params: dict[str, Any]) -> list[str]:
    """Reasons the sampling distribution differs from the model's own softmax
    (empty when it does not): only then are raw and processed log-probs the
    probabilities the tokens were actually sampled with."""
    reasons = []
    if params.get("temperature") != 1.0:
        reasons.append("temperature is not 1.0")
    if params.get("top_p", 1.0) != 1.0:
        reasons.append("top_p is not 1.0")
    if "top_k" in params:
        reasons.append("top_k is set")
    for name in ("presence_penalty", "frequency_penalty"):
        if params.get(name, 0) not in (0, 0.0):
            reasons.append(f"{name} is not 0")
    return reasons


def logprobs_mode(declared: str | None, params: dict[str, Any]) -> dict[str, Any]:
    """Raw vs processed cannot be read off a response. It is ``verified`` only
    when a mode was declared and the sampling applies no transformation
    (temperature 1, top_p 1, no top_k, zero penalties), where the two coincide."""
    reasons = sampling_untransformed(params)
    if declared not in LOGPROBS_MODES:
        reasons.append("no logprobs_mode declared (processed_logprobs or raw_logprobs)")
    return {"declared": declared, "verified": not reasons, "reasons": reasons}


class TokenStore:
    """``runs/<id>/tokens/<episode_id>.<execution_id>.jsonl``: one line per agent
    call with its ids, log-probs and its parent in the rollout graph.

    Parent = the earlier call of the same episode execution whose full
    sequence (prompt ids + sampled ids) is the longest exact prefix of this
    call's prompt ids. A retry, or a context that was compacted or rewritten,
    therefore starts a separate branch instead of silently joining one.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._calls: list[dict[str, Any]] = []
        self._path: Path | None = None
        self.last_parent: str | None = None

    def begin(self, episode_id: str, execution_id: str) -> None:
        self._path = self.root / f"{episode_id}.{execution_id}.jsonl"
        self._calls = []

    def end(self) -> None:
        self._path, self._calls = None, []

    def add(self, call_id: str, turn: int | None, attempt: int, data: dict[str, Any]
            ) -> tuple[str | None, int] | None:
        """Store a call's token data; return (relative ref, line index)."""
        if self._path is None or data.get("logprobs") is None:
            return None
        prompt = data.get("prompt_token_ids")
        parent, matched = None, 0
        for previous in self._calls if prompt is not None else []:
            if previous["prompt_token_ids"] is None:
                continue
            full = previous["prompt_token_ids"] + (previous["token_ids"] or [])
            if len(full) <= len(prompt) and prompt[:len(full)] == full and len(full) > matched:
                parent, matched = previous["call_id"], len(full)
        self.last_parent = parent
        self.root.mkdir(parents=True, exist_ok=True)
        record = {"call_id": call_id, "turn": turn, "attempt": attempt,
                  "parent_call_id": parent, "parent_prefix_len": matched,
                  "prompt_token_ids": prompt, "token_ids": data.get("token_ids"),
                  "logprobs": data.get("logprobs"), "level": data["level"],
                  "trainable": data["trainable"], "problems": data["problems"],
                  "finish_reason": data.get("finish_reason")}
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._calls.append(record)
        return f"tokens/{self._path.name}", len(self._calls) - 1
