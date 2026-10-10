"""Decision runner: one case -> the B1 agent's next action, scored.

The agent is called exactly as the runner calls it in a B1 episode, using the
runner's own functions and objects (no runtime module is edited):
``slh.runner.load_config`` (configs/arm-think-vllm.yaml), ``pin_serving``,
``capture_plan`` (the same capture probe, so the request carries the same
fields), ``make_backend`` (the self-served client with the pinned window gate)
and ``make_meter``, and ``slh.arms.think.ThinkAgent`` built as
``slh.runner._run_pairs`` builds it: the same system prompt, tools (with the
local think tool), temperature 0, non-thinking chat template, no max_tokens,
the trial-0 seed of the run seed, tau2's reply parsing and the think loop.

The case history goes in as the agent's message history; the agent's
``generate_next_message`` is called on the last tool result. If the next
action is a read-only retail tool call, it is executed on the unmodified public
database and the agent continues, at most ``MAX_READS`` times; a write call is
never executed. Nothing is written to any database.

Request limits (deviation D5, window B retry; demo code only, the runner and
its client are unchanged): every request carries ``max_tokens`` =
``MAX_OUTPUT_TOKENS`` (a capped reply is scored as the agent's answer and
recorded ``capped``); a timed-out request is retried at most once
(``DemoBackend``: the second timeout raises a non-transient error, so the
meter stops); a case stops after ``CASE_LIMIT_S`` seconds of wall clock and
is recorded as ``no_decision`` with its accounting. Window A (train and the
writer) ran before this change, without an output cap.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from slh.demo1.context import B1_PROMPT_SHA256, build_agent, runner_config, state_for
from slh.demo1.schema import READ_TOOLS, action_from_message, to_tau2_messages
from slh.demo1.score import score

MAX_READS = 2
SAMPLE_TEMPERATURE = 1.0  # best-of-2's second candidate (B1 itself is greedy)
MAX_OUTPUT_TOKENS = 4096
CASE_LIMIT_S = 300.0
TIMEOUT_RETRIES = 1
PROBE_MAX_TOKENS = 16   # the startup capture probe (reply "OK")
PROBE_LIMIT_S = 60.0
ABANDON_GRACE_S = 5.0   # after a case deadline, before the case is abandoned
POLL_S = 0.2


class CaseTimeLimit(RuntimeError):
    """The case's wall-clock limit passed (non-transient: the meter stops)."""


class RequestTimedOutAgain(RuntimeError):
    """A request timed out after its one retry (non-transient: the meter stops)."""


class ProbeTimeLimit(RuntimeError):
    """The capture probe did not finish within its hard outer deadline."""


class StageDeadlineReached(RuntimeError):
    """No case may start: it could not finish before the serving window's
    request-admission deadline."""


# --------------------------------------------------------------------------- policy


_case = threading.local()  # this thread's current case deadline (epoch seconds)


def _remaining() -> float | None:
    deadline = getattr(_case, "deadline", None)
    return None if deadline is None else deadline - time.time()


def install_sleep_cap() -> None:
    """Every meter sleep (retry backoff, pause) in a demo process is capped by
    the remaining time of the case its thread is running (process-local;
    idempotent)."""
    from slh import budget

    if getattr(budget._sleep, "_demo_capped", False):
        return
    original = budget._sleep

    def capped(seconds: float) -> None:
        left = _remaining()
        original(seconds if left is None else max(0.0, min(seconds, left)))

    capped._demo_capped = True  # type: ignore[attr-defined]
    budget._sleep = capped


def is_timeout(exc: BaseException | None) -> bool:
    """A timeout anywhere in the exception's cause/context chain: the client's
    ``Timeout``, ``TimeoutError`` / ``socket.timeout``, or a URL error whose
    reason is one (``post_json`` reports those as ``APIConnectionError``)."""
    import socket
    import urllib.error

    from slh import openai_compat

    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (openai_compat.Timeout, TimeoutError, socket.timeout)):
            return True
        if isinstance(exc, urllib.error.URLError) and isinstance(
                exc.reason, (TimeoutError, socket.timeout)):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


class RequestState:
    """One logical request (all of its meter attempts): its timeout count."""

    def __init__(self) -> None:
        self.timeouts = 0


def _demo_meter_class() -> type:
    from slh.budget import Meter

    class DemoMeter(Meter):
        """The runner's meter with one fresh ``RequestState`` per logical
        request (one ``call``), handed to the backend explicitly, and no long
        pauses (``max_pauses`` 0)."""

        current: RequestState | None = None

        def call(self, role: str, config: Any, request_payload: Any, invoke: Any) -> Any:
            self.current = RequestState()
            try:
                return super().call(role, config, request_payload, invoke)
            finally:
                self.current = None

    return DemoMeter


_PRICES: dict[str, Any] = {}
_METER_CLASS: list[type] = []


def demo_meter(config: Any) -> Any:
    from slh.budget import load_prices

    key = str(config.prices_path)
    if key not in _PRICES:  # read once per process (the file is hash-pinned)
        _PRICES[key] = load_prices(config.prices_path)
    if not _METER_CLASS:
        _METER_CLASS.append(_demo_meter_class())
    params = {**config.rate_limit_params(), "max_pauses": 0}
    return _METER_CLASS[0](config.cap_usd, _PRICES[key], **params)


def _demo_backend_class() -> type:
    from slh import openai_compat

    class DemoBackend(openai_compat.OpenAICompatibleBackend):
        """The self-served client with the demo's request limits. The request
        body, the window gate, metering and tau2 parsing are the parent's; only
        the HTTP timeout (bounded by the case deadline), the timeout-retry bound
        (per logical request, from the meter's ``RequestState``), the deadline
        checks and the handling of a capped reply differ."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.deadline: float | None = None  # epoch seconds
            self.capped_calls = 0
            self.request_meter: Any = None

        def _state(self) -> RequestState:
            state = getattr(self.request_meter, "current", None)
            if state is None:
                raise RuntimeError("a demo request needs its meter's request state")
            return state

        def complete(self, messages: list[Any], tools: list[Any] | None,
                     seed: int | None) -> tuple[Any, Any]:
            from slh.budget import metered_dispatch
            from slh.tokens import extract

            state = self._state()
            now = time.time()
            if self.deadline is not None and now >= self.deadline:
                raise CaseTimeLimit(f"case wall-clock limit of {CASE_LIMIT_S:.0f} s passed")
            timeout = openai_compat.TIMEOUT_S if self.deadline is None else \
                max(0.5, min(openai_compat.TIMEOUT_S, self.deadline - now))
            body = self.request_body(messages, tools, seed)
            self._served = None
            try:
                response = metered_dispatch(
                    body, lambda sent: openai_compat.post_json(
                        self.url(), sent, self.headers(), timeout),
                    gate=self.check_window)
            except Exception as exc:
                if is_timeout(exc):
                    state.timeouts += 1
                    if state.timeouts > TIMEOUT_RETRIES:
                        raise RequestTimedOutAgain(f"timed out again: {exc}") from None
                    if not isinstance(exc, openai_compat.Timeout):
                        raise openai_compat.Timeout(str(exc)) from exc  # retry once
                raise
            if self.deadline is not None and time.time() > self.deadline:
                raise CaseTimeLimit("the reply arrived after the case wall-clock limit")
            self.last_capture = extract(response)
            choice = (response.get("choices") or [{}])[0]
            if choice.get("finish_reason") == "length":
                self.capped_calls += 1
                try:
                    return openai_compat.parse_reply(response), None
                except (ValueError, KeyError, TypeError):
                    # A tool call cut off by the cap: its text is the answer.
                    from tau2.data_model.message import AssistantMessage

                    message = choice.get("message") or {}
                    return AssistantMessage(role="assistant",
                                            content=message.get("content") or "",
                                            cost=0.0), None
            return openai_compat.parse_reply(response), None

    return DemoBackend


def demo_backend(model: Any, level: str = "messages", pin: dict[str, str] | None = None,
                 meter: Any = None) -> Any:
    """A DemoBackend wired as ``slh.runner.make_backend`` wires the agent's,
    bound to ``meter`` (whose request state it reads)."""
    from slh.runner import pinned_gate

    backend = _demo_backend_class()(model, call_name="agent_response", level=level)
    if backend.gate is not None:
        backend.gate = pinned_gate(model, pin)
    backend.request_meter = meter
    return backend


def capture_probe(config: Any, pin: dict[str, str] | None,
                  backend_factory: Callable[[Any], Any] | None = None) -> dict[str, Any]:
    """The runner's capture probe (``slh.runner.capture_plan`` logic for an
    ``openai_compatible`` agent with ``capture: auto``) on the demo path: a
    capped (``PROBE_MAX_TOKENS``) request with a bounded deadline and the demo
    timeout rule. Returns the plan with the probe's accounting."""
    from slh import openai_compat
    from slh.tokens import level_rank

    agent = config.agent
    meter = demo_meter(config)
    plan: dict[str, Any] = {"mode": agent.capture, "backend": agent.backend}
    if agent.backend == "openai_compatible" and agent.capture == "auto":
        model = dataclasses.replace(agent, max_tokens=PROBE_MAX_TOKENS)
        backend = backend_factory(model) if backend_factory else \
            demo_backend(model, "messages", pin, meter)
        backend.request_meter = meter
        deadline = time.time() + PROBE_LIMIT_S
        backend.deadline = deadline
        box: dict[str, Any] = {}

        def run() -> None:
            _case.deadline = deadline
            try:
                box["record"] = openai_compat.probe(backend, meter)
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
                box["error"] = exc
            finally:
                _case.deadline = None

        # Hard outer deadline: the transport may ignore its timeout; the probe
        # runs on a daemon thread and is abandoned (the stage fails) at the limit.
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(max(0.0, deadline - time.time()) + ABANDON_GRACE_S)
        if thread.is_alive():
            raise ProbeTimeLimit(f"capture probe still running after {PROBE_LIMIT_S:.0f} s "
                                 f"+ {ABANDON_GRACE_S:.0f} s; abandoned")
        if "error" in box:
            raise box["error"]
        plan["probe"] = box["record"]
        level = plan["probe"]["detected_level"]
    else:
        level = max(agent.capture_levels(), key=level_rank) if agent.capture != "messages" \
            else "messages"
    plan["request_level"] = level
    t = meter.tally["agent"]
    return {"plan": plan, "level": level, "model_calls": t.calls,
            "failed_attempts": t.failed_attempts, "input_tokens": t.input_tokens,
            "output_tokens": t.output_tokens}


def b1_seed(config: Any, trial: int = 0) -> int:
    from slh.runner import trial_seeds

    return trial_seeds(config.seed, trial + 1)[trial]


class AgentClient:
    """The served agent, as one runner session would call it. One backend and
    meter per thread (the meter is not thread-safe); every thread checks the
    same pinned serving window before every request."""

    def __init__(self, config: Any = None, *, pin: dict[str, str] | None = None,
                 level: str | None = None, temperature: float | None = None,
                 backend_factory: Callable[[Any], Any] | None = None) -> None:
        self.config = config or runner_config()
        self.pin = pin
        self.level = level or "messages"
        changes: dict[str, Any] = {"max_tokens": MAX_OUTPUT_TOKENS}
        if temperature is not None:
            changes["temperature"] = temperature
        self.model = dataclasses.replace(self.config.agent, **changes)
        self._factory = backend_factory
        self._local = threading.local()
        self.probe: dict[str, Any] | None = None  # the capture probe's accounting
        self.stage_deadline: float | None = None  # epoch: no case may run past it

    @classmethod
    def for_window(cls, config: Any = None) -> AgentClient:
        """Pin the active serving window, run the capped capture probe once and
        take the stage deadline (the window's request-admission deadline)."""
        from slh.runner import pin_serving
        from slh.serving import gate_for

        install_sleep_cap()
        config = config or runner_config()
        pin = pin_serving(config)
        probe = capture_probe(config, pin)
        client = cls(config, pin=pin, level=probe["level"])
        client.probe = {"capture": {"agent": probe["plan"]},
                        **{k: probe[k] for k in ("model_calls", "failed_attempts",
                                                 "input_tokens", "output_tokens")}}
        gate = gate_for(config.agent)
        if gate is not None:
            client.stage_deadline = float(gate._window()["request_deadline_epoch"])
        return client

    def with_temperature(self, temperature: float) -> AgentClient:
        other = AgentClient(self.config, pin=self.pin, level=self.level,
                            temperature=temperature, backend_factory=self._factory)
        other.stage_deadline = self.stage_deadline
        return other

    def parts(self) -> tuple[Any, Any, Any]:
        """(backend, meter, environment) of this thread."""
        if not hasattr(self._local, "parts"):
            from tau2.registry import registry

            meter = demo_meter(self.config)
            backend = self._factory(self.model) if self._factory else \
                demo_backend(self.model, self.level, self.pin, meter)
            backend.request_meter = meter
            self._local.parts = (backend, meter, registry.get_env_constructor("retail")())
        return self._local.parts


def _tally(meter: Any) -> tuple[int, int, int, int]:
    t = meter.tally["agent"]
    return t.calls, t.failed_attempts, t.input_tokens, t.output_tokens


def _delta(after: tuple[int, ...], before: tuple[int, ...]) -> dict[str, int]:
    keys = ("model_calls", "failed_attempts", "input_tokens", "output_tokens")
    return {k: a - b for k, a, b in zip(keys, after, before, strict=True)}


def next_action(case: dict[str, Any], notes: list[str], client: AgentClient, *,
                config: Any = None,
                seed: int, expect_prompt_sha256: str | None = None) -> dict[str, Any]:
    """Run the agent on one case; returns the action and its accounting."""
    from tau2.data_model.message import ToolCall

    from slh.capture import sha256_json

    backend, meter, env = client.parts()
    agent = build_agent(config or client.config, state_for(notes), backend, meter, env)
    prompt_sha = sha256_json(agent.system_prompt)
    if expect_prompt_sha256 is not None and prompt_sha != expect_prompt_sha256:
        raise ValueError("the rendered system prompt differs from the expected arm prompt")
    history = to_tau2_messages(case["messages"])
    state = agent.get_init_state(history[:-1])
    agent.set_seed(seed)
    before = _tally(meter)
    capped_before = getattr(backend, "capped_calls", 0)
    started = time.perf_counter()
    message, state = agent.generate_next_message(history[-1], state)
    reads: list[dict[str, Any]] = []
    action = action_from_message(message)
    while action["type"] == "tool_call" and action["name"] in READ_TOOLS and \
            not action.get("extra_calls") and len(reads) < MAX_READS:
        call = ToolCall(id=message.tool_calls[0].id, name=action["name"],
                        arguments=action["arguments"], requestor="assistant")
        result = env.get_response(call)
        reads.append({"name": action["name"], "arguments": action["arguments"],
                      "error": bool(result.error)})
        message, state = agent.generate_next_message(result, state)
        action = action_from_message(message)
    metrics = getattr(agent, "private_episode_metrics", None) or {
        "think_count": 0, "think_cap_hit_count": 0}  # B0 has no think tool
    return {"action": action, "reads": reads, "system_prompt_sha256": prompt_sha,
            "think_count": metrics["think_count"],
            "think_cap_hit": metrics["think_cap_hit_count"],
            "capped": getattr(backend, "capped_calls", 0) > capped_before,
            "wall_s": round(time.perf_counter() - started, 3),
            **_delta(_tally(meter), before)}


# --------------------------------------------------------------------------- best-of-2

SELECTOR_INSTRUCTION = (
    "You are checking the next step of the retail agent in the conversation below. Two "
    "candidate next actions are given. Choose the one that follows the policy and the "
    "user's request correctly. Reply with exactly one letter: A or B.")


def render_action(action: dict[str, Any]) -> str:
    if action["type"] == "tool_call":
        return f"tool call {action['name']}({json.dumps(action['arguments'], sort_keys=True)})"
    if action["type"] == "message":
        return f"message to the user: {action['content']}"
    return "no action"


def render_transcript(messages: list[dict[str, Any]]) -> str:
    lines = []
    for m in messages:
        if m["role"] == "tool":
            lines.append(f"[tool result] {m['content']}")
            continue
        if m.get("content"):
            lines.append(f"[{m['role']}] {m['content']}")
        for c in m.get("tool_calls") or []:
            lines.append(f"[{m['role']} tool call] {c['name']}"
                         f"({json.dumps(c['arguments'], sort_keys=True)})")
    return "\n".join(lines)


def selector_text(case: dict[str, Any], first: dict[str, Any], second: dict[str, Any]) -> str:
    return (f"{SELECTOR_INSTRUCTION}\n\nConversation:\n{render_transcript(case['messages'])}"
            f"\n\nCandidate A: {render_action(first)}\nCandidate B: {render_action(second)}")


def parse_choice(text: str | None) -> str | None:
    cleaned = (text or "").strip().strip(".*`'\" ").upper()
    return cleaned if cleaned in ("A", "B") else None


def select(case: dict[str, Any], candidates: list[dict[str, Any]], client: AgentClient, *,
           seed: int) -> dict[str, Any]:
    """Same-model selector over two candidate actions. Identical candidates need
    no call. Candidate order is fixed per case (hash parity), not by which one
    was greedy. An unparseable reply selects A (disclosed)."""
    import hashlib

    from tau2.data_model.message import SystemMessage, UserMessage

    from slh.agent import metered_complete

    if json.dumps(candidates[0]["action"], sort_keys=True) == \
            json.dumps(candidates[1]["action"], sort_keys=True):
        return {"chosen": 0, "selector_called": False, "selector_reply": None,
                "model_calls": 0, "input_tokens": 0, "output_tokens": 0, "failed_attempts": 0}
    flip = int(hashlib.sha256(case["case_id"].encode()).hexdigest(), 16) % 2 == 1
    order = [1, 0] if flip else [0, 1]
    backend, meter, env = client.parts()
    agent = build_agent(client.config, state_for([]), backend, meter, env)
    before = _tally(meter)
    capped_before = getattr(backend, "capped_calls", 0)
    reply = metered_complete(meter, "agent", backend, [
        SystemMessage(role="system", content=agent.system_prompt),
        UserMessage(role="user", content=selector_text(
            case, candidates[order[0]]["action"], candidates[order[1]]["action"]))],
        None, seed)
    letter = parse_choice(reply.content)
    chosen = order[0] if letter in (None, "A") else order[1]
    return {"chosen": chosen, "selector_called": True, "selector_reply": reply.content,
            "capped": getattr(backend, "capped_calls", 0) > capped_before,
            "selector_parsed": letter is not None, "flip": flip,
            **_delta(_tally(meter), before)}


# --------------------------------------------------------------------------- arms


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    notes: tuple[str, ...] = ()
    kind: str = "notes"  # notes | best_of_2 | code_rule | b0


GREEDY_KEYS = ("action", "reads", "system_prompt_sha256", "think_count", "think_cap_hit",
               "capped", "wall_s", "model_calls", "failed_attempts", "input_tokens",
               "output_tokens")


def _set_deadline(clients: list[AgentClient], deadline: float | None) -> None:
    for c in clients:
        c.parts()[0].deadline = deadline
    _case.deadline = deadline


def run_case(arm: Arm, case: dict[str, Any], client: AgentClient,
             sampler: AgentClient | None = None,
             greedy: dict[str, Any] | None = None,
             deadline: float | None = None) -> dict[str, Any]:
    """One arm on one case, within the case wall-clock limit (``deadline``,
    epoch seconds; default now + ``CASE_LIMIT_S``)."""
    clients = [c for c in (client, sampler) if c is not None]
    if arm.kind != "code_rule":
        _set_deadline(clients, deadline if deadline is not None
                      else time.time() + CASE_LIMIT_S)
    try:
        return _run_case(arm, case, client, sampler, greedy)
    finally:
        if arm.kind != "code_rule":
            _set_deadline(clients, None)


def _run_case(arm: Arm, case: dict[str, Any], client: AgentClient,
              sampler: AgentClient | None = None,
              greedy: dict[str, Any] | None = None) -> dict[str, Any]:
    """One arm on one case -> a scored result row. ``greedy``: best-of-2 reuses
    the B1 arm's own (greedy, temperature 0) result row for this case as its
    first candidate; its calls and tokens are charged to best-of-2 as well."""
    row: dict[str, Any] = {"arm": arm.name, "case_id": case["case_id"], "task": case["task"],
                           "split": case["split"], "family": case["family"],
                           "kind": case["kind"], "role": case["role"]}
    seed = b1_seed(client.config)
    if arm.kind == "code_rule":
        from slh.demo1 import code_rule  # separately reviewed module (variant task only)

        result = {"action": code_rule.decide(case), "model_calls": 0, "input_tokens": 0,
                  "output_tokens": 0, "failed_attempts": 0}
    elif arm.kind == "best_of_2":
        if sampler is None:
            raise ValueError("best-of-2 needs a sampling client")
        if greedy is not None and greedy.get("system_prompt_sha256") == B1_PROMPT_SHA256 \
                and not greedy.get("error"):
            greedy = {k: greedy.get(k, False if k == "capped" else None)
                      for k in GREEDY_KEYS}
        else:
            greedy = next_action(case, [], client, seed=seed,
                                 expect_prompt_sha256=B1_PROMPT_SHA256)
        sample = next_action(case, [], sampler, seed=b1_seed(client.config, 1),
                             expect_prompt_sha256=B1_PROMPT_SHA256)
        choice = select(case, [greedy, sample], client, seed=seed)
        picked = (greedy, sample)[choice["chosen"]]
        result = {"action": picked["action"], "candidates": [greedy, sample],
                  "selection": choice,
                  "capped": bool(greedy.get("capped") or sample.get("capped")
                                 or choice.get("capped")),
                  **{k: greedy[k] + sample[k] + choice[k] for k in (
                      "model_calls", "input_tokens", "output_tokens", "failed_attempts")}}
    elif arm.kind == "b0":
        from slh.demo1.context import B0_PROMPT_SHA256, b0_config

        result = next_action(case, [], client, seed=seed, config=b0_config(),
                             expect_prompt_sha256=B0_PROMPT_SHA256)
    else:
        expect = B1_PROMPT_SHA256 if not arm.notes else None
        result = next_action(case, list(arm.notes), client, seed=seed,
                             expect_prompt_sha256=expect)
    return {**row, **result, **score(case, result["action"])}


COST_KEYS = ("model_calls", "failed_attempts", "input_tokens", "output_tokens")

# Stage-wide (process-wide) abandoned cases whose requests are still in flight:
# shared by every arm, calibration and the sealed run, so active plus abandoned
# requests never exceed the worker limit.
_ABANDONED: list[threading.Thread] = []
_ABANDONED_LOCK = threading.Lock()


def abandoned_in_flight() -> int:
    with _ABANDONED_LOCK:
        _ABANDONED[:] = [t for t in _ABANDONED if t.is_alive()]
        return len(_ABANDONED)


def _abandon(thread: threading.Thread | None) -> None:
    if thread is not None:
        with _ABANDONED_LOCK:
            _ABANDONED.append(thread)


class _CaseRun:
    """One case on its own daemon thread (a worker never outlives the process,
    so an abandoned case cannot keep the window open)."""

    def __init__(self, case: dict[str, Any], deadline: float) -> None:
        self.case = case
        self.deadline = deadline
        self.done = threading.Event()
        self.row: dict[str, Any] | None = None
        self.fatal: BaseException | None = None
        self.abandoned = False
        self.meters: list[Any] = []
        self.before: list[tuple[int, int, int, int]] = []
        self.thread: threading.Thread | None = None


def _no_decision_row(arm: Arm, case: dict[str, Any], error: str, **flags: Any
                     ) -> dict[str, Any]:
    return {"arm": arm.name, "case_id": case["case_id"], "task": case["task"],
            "split": case["split"], "family": case["family"], "kind": case["kind"],
            "role": case["role"], "error": error, "case_time_limit": False,
            "timeout_after_retry": False, **flags,
            "action": {"type": "none", "reason": "error"},
            **score(case, {"type": "none", "reason": "error"})}


def run_arm(arm: Arm, cases: list[dict[str, Any]], client: AgentClient, out: Path, *,
            workers: int = 4, sampler: AgentClient | None = None,
            greedy_rows: dict[str, dict[str, Any]] | None = None,
            stage_deadline: float | None = None) -> list[dict[str, Any]]:
    """Run an arm on cases, appending one row per case to ``out`` (resumable:
    cases already in ``out`` for this arm are skipped).

    Hard wall-clock bounds: each case runs on its own daemon thread with a
    deadline of ``CASE_LIMIT_S``; a case still running ``ABANDON_GRACE_S`` after
    it is abandoned and recorded ``no_decision`` with the accounting its meters
    show (its later result is discarded). A case starts only if it can finish
    before the stage deadline (default: the client's, the serving window's
    request-admission deadline); when none can, dispatch stops, running cases
    finish or are abandoned, and ``StageDeadlineReached`` is raised. A closed
    serving window stops dispatch the same way and is re-raised."""
    from slh.serving import ServingWindowClosed

    install_sleep_cap()
    stage_deadline = stage_deadline if stage_deadline is not None else client.stage_deadline
    done: dict[str, dict[str, Any]] = {}
    if out.is_file():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["arm"] == arm.name:
                    done[r["case_id"]] = r
    pending = [c for c in cases if c["case_id"] not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    clients = [c for c in (client, sampler) if c is not None]
    reused_rows = greedy_rows or {}
    rows: dict[str, dict[str, Any]] = {}
    running: list[_CaseRun] = []
    fatal: BaseException | None = None
    stopped = False
    wake = threading.Event()

    def target(run: _CaseRun) -> None:
        _case.deadline = run.deadline
        try:
            run.meters = [c.parts()[1] for c in clients]
            run.before = [_tally(m) for m in run.meters]
            row = run_case(arm, run.case, client, sampler,
                           reused_rows.get(run.case["case_id"]), deadline=run.deadline)
        except ServingWindowClosed as exc:
            run.fatal = exc
            row = None
        except Exception as exc:  # noqa: BLE001 - recorded as no decision
            row = _no_decision_row(arm, run.case, f"{type(exc).__name__}: {exc}",
                                   case_time_limit=isinstance(exc, CaseTimeLimit),
                                   timeout_after_retry=isinstance(exc, RequestTimedOutAgain))
            _charge(row, run)
            _charge_reused(row, arm, reused_rows.get(run.case["case_id"]))
        finally:
            _case.deadline = None
        run.row = row
        run.done.set()
        wake.set()

    def write(row: dict[str, Any]) -> None:
        rows[row["case_id"]] = row
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    while pending or running:
        now = time.time()
        if pending and not stopped and stage_deadline is not None and \
                now + CASE_LIMIT_S > stage_deadline:
            stopped = True
        # Outstanding abandoned requests count against the worker limit: with
        # as many of them as workers, no new case starts (wait for them).
        while pending and not stopped and \
                len(running) + abandoned_in_flight() < max(1, workers):
            run = _CaseRun(pending.pop(0), time.time() + CASE_LIMIT_S)
            run.thread = threading.Thread(target=target, args=(run,), daemon=True)
            run.thread.start()
            running.append(run)
        if not running and (stopped or not pending):
            break
        wake.wait(POLL_S)
        wake.clear()
        for run in list(running):
            if run.done.is_set():
                running.remove(run)
                if run.fatal is not None:
                    fatal = fatal or run.fatal
                    stopped = True
                    continue
                write(run.row)
            elif time.time() > run.deadline + ABANDON_GRACE_S:
                run.abandoned = True
                running.remove(run)
                _abandon(run.thread)
                row = _no_decision_row(arm, run.case, "CaseTimeLimit: abandoned at the case "
                                       "wall-clock limit", case_time_limit=True,
                                       abandoned=True)
                _charge(row, run)
                _charge_reused(row, arm, reused_rows.get(run.case["case_id"]))
                write(row)
    if stopped and pending and fatal is None:
        # The stage deadline was reached: the cases that cannot start are
        # recorded no_decision (flagged; the report treats them as incomplete).
        for case in pending:
            write(_no_decision_row(arm, case, "StageDeadlineReached: not started before "
                                   "the request-admission deadline", stage_deadline=True))
    if fatal is not None:
        raise fatal
    if stopped and pending:
        raise StageDeadlineReached(f"{len(pending)} {arm.name} case(s) not started: they "
                                   "could not finish before the request-admission deadline "
                                   "(recorded no_decision, flagged stage_deadline)")
    by_id = {**done, **rows}
    return [by_id[c["case_id"]] for c in cases if c["case_id"] in by_id]


def _charge(row: dict[str, Any], run: _CaseRun) -> None:
    """Calls already consumed by the case stay charged to the arm (read from
    the case thread's meters; an in-flight request is not yet counted)."""
    for meter, prior in zip(run.meters, run.before, strict=False):
        for k, v in _delta(_tally(meter), prior).items():
            row[k] = row.get(k, 0) + v


def _charge_reused(row: dict[str, Any], arm: Arm, reused: dict[str, Any] | None) -> None:
    """Best-of-2 reuses the B1 greedy call without a new request: its accounting
    stays charged when a later call (sample, selector) fails."""
    if arm.kind == "best_of_2" and reused is not None and \
            reused.get("system_prompt_sha256") == B1_PROMPT_SHA256 and not reused.get("error"):
        for k in COST_KEYS:
            row[k] = row.get(k, 0) + int(reused.get(k) or 0)
        row["reused_greedy_case"] = True
