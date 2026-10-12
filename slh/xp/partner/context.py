"""Context-length rules for the partner runner (phase 0 finding, 2026-10-12).

- **Guard.** Before any 9B request, the request is counted with the 9B tokenizer:
  message texts, tool calls and the tool schemas, plus a per-message allowance for the
  chat template, plus the request's ``max_tokens`` and a margin. Over the server's
  ``max_model_len`` the request is refused before it is sent with
  ``ContextLimitExceeded``, which the runner records as a per-case ``context_limit``
  final row (``errors.classify``), never a halt. A server-side HTTP 400 naming the
  context length gets the same treatment (``openai_compat.ContextLengthError``).
- **Full-context budget (proposed amendment-2 rule).** The full-context arm's text
  holds as many WHOLE training episodes as fit, in training-stream order, within
  ``max_model_len - tokens(B1 system prompt and tools) - tokens(largest test case of
  the task) - max_output_tokens - margin``. The same text serves every test case of
  that run; the number of episodes included and the token count are recorded.
"""

from __future__ import annotations

import functools
import json
from typing import Any

MARGIN = 1024            # tokens kept free beyond the counted request
PER_MESSAGE = 8          # chat-template tokens per message (role markers, separators)
# Decision growth (proposed amendment-2 rule): within one decision (think rounds, read
# turns, guard retries, a writer call), every later request's prompt growth over the
# decision's first request plus its output cap stays within this many tokens; the
# guard shrinks the output cap to the room left and ends the decision (a per-case
# row) when no room is left. The full-context fit reserves it, so no request of a
# decision can exceed max_model_len.
GROWTH = 16384


class DecisionGrowthLimit(Exception):
    """A decision used its whole registered growth (a per-case final row)."""


def max_model_len(endpoint: dict[str, Any] | None) -> int:
    """The served ``--max-model-len`` (from the endpoint's recorded serving args),
    else the pinned default."""
    from slh import serve_spec

    args = list((endpoint or {}).get("serving_args") or [])
    if "--max-model-len" in args:
        return int(args[args.index("--max-model-len") + 1])
    return int(serve_spec.MAX_MODEL_LEN)


@functools.lru_cache(maxsize=65536)
def tokens(text: str) -> int:
    from slh.learner.note_tokens import count_tokens

    return count_tokens(text) if text else 0


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content parts
        return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p)
                         for p in content)
    return "" if content is None else json.dumps(content, sort_keys=True)


def body_tokens(body: dict[str, Any]) -> int:
    """Prompt tokens of an OpenAI-style chat request (an upper estimate of what the
    chat template renders: texts, tool calls and tool schemas plus allowances)."""
    n = 0
    for m in body.get("messages") or []:
        n += PER_MESSAGE + tokens(_content_text(m.get("content")))
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            n += PER_MESSAGE + tokens(str(fn.get("name", ""))) + \
                tokens(fn.get("arguments") if isinstance(fn.get("arguments"), str)
                       else json.dumps(fn.get("arguments"), sort_keys=True))
    if body.get("tools"):
        n += tokens(json.dumps(body["tools"], sort_keys=True))
    return n


def make_guard(limit: int, margin: int = MARGIN, growth: int | None = None):
    """A ``context_guard`` for ``OpenAICompatibleBackend``: raises
    ``ContextLimitExceeded`` when prompt + max_tokens + margin exceeds ``limit``.
    With ``growth``: per thread, ``guard.new_decision()`` marks a decision's start;
    each later request's growth over the first request's prompt plus its output cap is
    held within ``growth`` (the cap shrinks to the room left; none left raises
    ``DecisionGrowthLimit``)."""
    import threading

    from slh.poc9.pipeline import ContextLimitExceeded

    state = threading.local()

    def new_decision() -> None:
        state.base = None
        state.active = True

    def guard(body: dict[str, Any]) -> None:
        prompt = body_tokens(body)
        if growth and getattr(state, "active", False):
            if state.base is None:
                state.base = prompt
            room = int(growth) - (prompt - state.base)
            if room <= 0:
                raise DecisionGrowthLimit(
                    f"decision growth limit: the prompt grew {prompt - state.base} tokens "
                    f"(registered growth {growth}); not sent")
            body["max_tokens"] = min(int(body.get("max_tokens") or room), room)
        out = int(body.get("max_tokens") or 0)
        if prompt + out + margin > limit:
            raise ContextLimitExceeded(
                f"context limit: {prompt} prompt tokens + {out} output tokens + {margin} "
                f"margin exceed max_model_len {limit}; not sent")
    guard.limit, guard.margin, guard.growth = limit, margin, growth
    guard.new_decision = new_decision
    return guard


def new_decision(backend: Any) -> None:
    """Mark a decision's start on ``backend``'s guard (no-op without one)."""
    guard = getattr(backend, "context_guard", None)
    if guard is not None and hasattr(guard, "new_decision"):
        guard.new_decision()


def install(client: Any, guard: Any) -> None:
    """Attach ``guard`` to every per-thread backend an ``AgentClient`` creates."""
    original = client.parts

    def parts() -> tuple[Any, Any, Any]:
        backend, meter, env = original()
        backend.context_guard = guard
        return backend, meter, env

    client.parts = parts


# --------------------------------------------------------------------------- full context
#
# The full-context fit is measured with EXACTLY the guard's accounting, on the real
# request body the agent sends (system message with the note inserted by the agent's own
# template, the case's messages, the tool schemas), captured offline. ``body_tokens`` is
# a sum over messages plus the tools, so a request's tokens split exactly into the
# system message part and the rest; the largest rest over the task's test cases with
# the system part of a candidate text gives the largest request that text can make.


class _Captured(BaseException):
    def __init__(self, body: dict[str, Any]) -> None:
        super().__init__("captured")
        self.body = body


@functools.lru_cache(maxsize=1)
def _capture_client():
    import dataclasses

    import yaml

    from slh.budget import ModelConfig
    from slh.demo1 import decide
    from slh.demo1.context import runner_config
    from slh.openai_compat import OpenAICompatibleBackend
    from slh.xp.partner import common as pc

    class Capture(OpenAICompatibleBackend):
        def complete(self, messages, tools, seed):  # never sends
            raise _Captured(self.request_body(messages, tools, seed))

    raw = yaml.safe_load((pc.REPO / "configs/agent-vllm.yaml").read_text(encoding="utf-8"))
    raw.pop("serving_session", None)
    raw.update(api_base="http://127.0.0.1:9/v1", api_key_env=None, capture="messages")
    config = dataclasses.replace(runner_config(), agent=ModelConfig.from_dict(raw))
    return decide.AgentClient(config, backend_factory=lambda model: Capture(
        model, call_name="agent_response", level="messages"))


def request_body(case: dict[str, Any], text: str = "") -> dict[str, Any]:
    """The first request body the agent sends for ``case`` with ``text`` as its
    notes (``slh.poc9.loader.notes_for``: stripped; empty = B1), captured offline."""
    from slh.demo1 import decide
    from slh.poc9.loader import notes_for

    try:
        decide.next_action(case, notes_for(text), _capture_client(), seed=0)
    except _Captured as c:
        return c.body
    raise RuntimeError("the agent made no request")


def system_part(body: dict[str, Any]) -> int:
    """The system message's share of ``body_tokens`` (it is the first message)."""
    m = body["messages"][0]
    if m.get("role") != "system":
        raise ValueError("the request does not start with a system message")
    return PER_MESSAGE + tokens(_content_text(m.get("content")))


def fit_parameters(test_cases: list[dict[str, Any]], *, limit: int, max_output: int,
                   margin: int = MARGIN, growth: int = GROWTH) -> dict[str, int]:
    """Everything of the guard's check except the note: the largest non-system part
    (case messages and tools) over the task's test cases, the margin, ``max_model_len``
    and the decision growth reserved for every later request of a decision (at least
    the output cap). ``budget`` is the first request's prompt budget: then every
    request of any decision on any test case passes the guard."""
    if growth < max_output:
        raise ValueError("the decision growth must cover the output cap")
    rest = 0
    for q in test_cases:
        body = request_body(q, "")
        rest = max(rest, body_tokens(body) - system_part(body))
    return {"limit": int(limit), "max_output": int(max_output), "margin": int(margin),
            "growth": int(growth), "fixed_tokens": rest,
            "budget": int(limit) - int(growth) - int(margin)}


def prompt_tokens(text: str, probe_case: dict[str, Any], params: dict[str, int]) -> int:
    """The largest request ``text`` can produce on the task's test cases (guard
    accounting): its system part plus the largest rest."""
    return system_part(request_body(probe_case, text)) + params["fixed_tokens"]


def fits(text: str, probe_case: dict[str, Any], params: dict[str, int]) -> bool:
    return prompt_tokens(text, probe_case, params) <= params["budget"]


def render_evidence(header: str, blocks: list[str], k: int, separator: str = "\n\n") -> str:
    return header + separator.join(blocks[:k])


def fit_full_context(header: str, blocks: list[str], probe_case: dict[str, Any],
                     params: dict[str, int]) -> tuple[str, dict[str, int]]:
    """The longest prefix of whole episodes whose largest request passes the guard;
    returns the text and its receipt."""
    lo, hi = 0, len(blocks)
    if fits(render_evidence(header, blocks, hi), probe_case, params):
        lo = hi
    else:
        while lo < hi:  # tokens grow with the number of episodes
            mid = (lo + hi + 1) // 2
            if fits(render_evidence(header, blocks, mid), probe_case, params):
                lo = mid
            else:
                hi = mid - 1
    text = render_evidence(header, blocks, lo)
    return text, {"episodes_total": len(blocks), "episodes_included": lo,
                  "tokens": prompt_tokens(text, probe_case, params),
                  "budget": params["budget"], "fixed_tokens": params["fixed_tokens"],
                  "limit": params["limit"], "max_output": params["max_output"],
                  "margin": params["margin"], "growth": params["growth"]}


def fit_episodes(header: str, blocks: list[str], budget: int,
                 separator: str = "\n\n") -> tuple[str, int, int]:
    """Plain token-count prefix fit (no request rendering); kept for unit tests."""
    def render(k: int) -> str:
        return header + separator.join(blocks[:k])

    lo, hi = 0, len(blocks)
    if tokens(render(hi)) <= budget:
        lo = hi
    else:
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if tokens(render(mid)) <= budget:
                lo = mid
            else:
                hi = mid - 1
    text = render(lo)
    return text, lo, tokens(text)
