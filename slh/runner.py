"""Run tasks through tau2's orchestrator and evaluator, resume, and replay.

Run layout (``runs/<run_id>/``):

- ``manifest.json``: identity, code and data provenance, split, models, harness
  state version, seeds, budget cap, status and spend (rewritten after every
  episode).
- ``episodes.jsonl``: one line per (task, trial). Never contains rewards, tau2
  grading fields (recursive key check) or task scenario/grading text outside
  the conversation (string scan); errors are stored as type, code and hash.
- ``outcomes.jsonl``: tau2's reward and breakdown per episode, with ``valid``
  false for infrastructure or grading failures.
- ``snapshot/``: the resolved config, every referenced YAML, the price table,
  and the uncommitted diff plus untracked-file hashes when the tree is dirty.
- ``calls.jsonl``: one record per agent model-call attempt (raw output before
  parsing, provider, response id, usage, charge, latency). Leak-checked.
- ``blobs/<sha256>.json``: content-addressed agent system prompts and tool
  schemas, referenced from each episode line.
- ``private/``: NOT FOR THE LEARNER. Full error text, tau2 and provider logs,
  quarantined torn lines, user-simulator and judge call records
  (``private/calls.jsonl``), the user simulator's prompts (``private/blobs/``).
  May contain task scenario and grading text.
- ``think_public.jsonl`` (think arm only): the agent's own think-call text,
  one line per thought, projected from ``calls.jsonl`` for the learner as
  fallible evidence (``write_think_projection``); ``private/think.jsonl`` is
  unchanged.
- ``tokens/``: agent-side token ids and log-probs per call, when the endpoint
  returned them (``slh.tokens``), referenced from ``calls.jsonl``.
- Parallel runs only: ``ledger.jsonl`` (the shared spend ledger), ``workers/``
  (each worker's own files, merged into the top-level files) and lock files.

``docs/trace-format.md`` describes every file.
"""

from __future__ import annotations

import ast
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import logging
import multiprocessing
import os
import random
import re
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from slh.agent import HarnessAgent, HarnessUser, LiteLLMBackend, build_guards, request_payload
from slh.arms.think import THINK_GIVE_UP_TEXT, THINK_INSTRUCTION, ThinkAgent
from slh.budget import (
    BudgetExceeded,
    Meter,
    ModelConfig,
    Overspend,
    RoleTally,
    UnmeteredCall,
    Usage,
    completion_guard,
    delta,
    input_token_bound,
    load_prices,
)
from slh.capture import BlobStore, CallRecorder
from slh.execution import (
    IdentityMismatch,
    base_arm_fingerprint,
    current_execution_fingerprint,
    run_files_digest,
    trial_indices,
    verify_run_identity,
    worker_slots,
)
from slh.ledger import SharedLedger
from slh.serving import ServingGate, ServingWindowClosed, gate_for
from slh.state import HarnessState
from slh.stub import StubAgentBackend, StubUserBackend
from slh.tokens import TokenStore, level_rank, min_level
from slh.trace import (
    GradingLeak,
    append_jsonl,
    check_agent_prompt_clean,
    check_no_grading_fields,
    check_no_task_text,
    dumps,
    error_record,
    messages_to_records,
    read_jsonl_tolerant,
    records_to_messages,
    task_leak_windows,
)
from slh.worker import worker_entry

TAU2_COMMIT_PINNED = "672227c6b6676edc20d57ea53b7000262aae77b9"
STUB_JUDGE = ModelConfig(backend="stub", model="stub-judge")
REPO_ROOT = Path(__file__).resolve().parent.parent
SPLIT_NAMES = ("stream", "confirm", "dev", "test", "transfer")
# Only domains whose model calls all go through litellm completions, which the
# meter guards. telecom (user tools, environment interface) and
# banking_knowledge (direct embedding SDK calls) need their own metering first.
SUPPORTED_DOMAINS = ("retail", "airline")
JUDGE_MODES = ("official", "stub")
CONVERSATION_PATHS = ("$.messages", "$.guard_events")
TRACE_FORMAT = "slh-trace.v1"
MAX_WORKERS = 4
PARALLEL_CHECKPOINT_S = 5.0  # coordinator refreshes the manifest's spend this often
WORKER_GRACE_S = 30.0  # after an interrupt, how long workers get to record and exit
RATE_LIMIT_KEYS = {"pause_after", "pause_s", "max_pauses"}
ARMS = ("baseline", "think")
PRIVATE_README = (
    "NOT FOR THE LEARNER.\n\nThis directory holds raw error text, tau2 and provider logs and "
    "quarantined lines. They may contain task scenarios and grading text. No learning code "
    "may read from here.\n"
)


class RunAborted(RuntimeError):
    """Internal: stop the run loop with a status."""

    def __init__(self, status: str, reason: dict[str, str]) -> None:
        super().__init__(status)
        self.status = status
        self.reason = reason


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class RunConfig:
    path: Path
    agent: ModelConfig
    user: ModelConfig
    judge_mode: str
    stub_judge_met: bool
    cap_usd: float
    prices_path: Path
    seed: int
    max_steps: int
    max_errors: int
    enforce_communication_protocol: bool
    timeout_s: float | None
    sources: dict[str, Path] = field(default_factory=dict)
    # Optional ``rate_limit:`` section (pause_after, pause_s, max_pauses);
    # None uses the meter's defaults and leaves the resolved hash unchanged.
    rate_limit: dict[str, Any] | None = None
    # Fixed treatment arm (docs/arms/think-INTEGRATION.md); not a learned state edit.
    arm: str = "baseline"
    think_instruction: str | None = None
    think_cap_message: str | None = None

    def rate_limit_params(self) -> dict[str, Any]:
        return dict(self.rate_limit or {})

    @property
    def evidence_eligible(self) -> bool:
        return (self.agent.backend != "stub" and self.user.backend != "stub"
                and self.judge_mode == "official")

    @property
    def fully_stub(self) -> bool:
        return (self.agent.backend == "stub" and self.user.backend == "stub"
                and self.judge_mode == "stub")


    def resolved(self) -> dict[str, Any]:
        return {"agent": self.agent.to_dict(), "user": self.user.to_dict(),
                "judge_mode": self.judge_mode,
                "stub_judge_met": self.stub_judge_met if self.judge_mode == "stub" else None,
                "cap_usd": self.cap_usd, "seed": self.seed,
                "orchestrator": {"max_steps": self.max_steps, "max_errors": self.max_errors,
                                 "enforce_communication_protocol":
                                     self.enforce_communication_protocol,
                                 "timeout_s": self.timeout_s},
                "prices_sha256": _sha256_file(self.prices_path),
                **({"rate_limit": self.rate_limit} if self.rate_limit else {}),
                # Only non-baseline arms add keys: baseline config hashes are unchanged.
                **({"arm": self.arm, "think_instruction": self.think_instruction,
                    "think_cap_message": self.think_cap_message}
                   if self.arm != "baseline" else {})}

    def resolved_sha256(self) -> str:
        return _sha256_bytes(json.dumps(self.resolved(), sort_keys=True).encode())


def check_state_allowed(state: HarnessState, config: RunConfig) -> None:
    """A rehearsal state (or one derived from it) may only run fully stubbed; a
    state bound to a base arm runs on that base arm only."""
    if state.rehearsal and not config.fully_stub:
        raise ValueError(f"harness state {state.version_id} comes from a learner rehearsal; "
                         "it may only be used with stub agent, user and judge")
    if state.base is None and config.arm == "think" and (
            state.notes or state.skills or state.guards):
        raise ValueError(f"harness state {state.version_id} is a legacy learned state without a "
                         "base-arm binding; B1 (think) runs accept only bound states. Migrate "
                         "it explicitly: python -m slh bind-state")
    if state.base is not None and state.base != base_arm_fingerprint(config):
        raise ValueError(f"harness state {state.version_id} is bound to base arm "
                         f"{state.base['arm']} ({state.base['config_sha256'][:12]}), not "
                         f"this config's {config.arm} "
                         f"({base_arm_fingerprint(config)['config_sha256'][:12]})")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _model_section(value: Any, base: Path, name: str, sources: dict[str, Path]) -> ModelConfig:
    if isinstance(value, str):
        path = (base / value).resolve()
        sources[f"{name}.yaml"] = path
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("model section must be a mapping or a path to one")
    return ModelConfig.from_dict(value)


def load_config(path: str | Path) -> RunConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent
    sources: dict[str, Path] = {"config.yaml": path}
    orchestrator = raw.get("orchestrator") or {}
    budget = raw.get("budget") or {}
    if "cap_usd" not in budget:
        raise ValueError(f"{path}: budget.cap_usd is required")
    timeout = orchestrator.get("timeout_s")
    grader = raw.get("grader") or {}
    judge_mode = grader.get("judge", "official")
    if judge_mode not in JUDGE_MODES:
        raise ValueError(f"{path}: grader.judge must be one of {JUDGE_MODES}")
    agent = _model_section(raw["agent"], base, "agent", sources)
    user = _model_section(raw["user"], base, "user", sources)
    if judge_mode == "stub" and (agent.backend != "stub" or user.backend != "stub"):
        raise ValueError(f"{path}: judge: stub is only allowed when agent and user are stubs")
    prices_path = (base / raw.get("prices", "prices.yaml")).resolve()
    sources["prices.yaml"] = prices_path
    rate_limit = raw.get("rate_limit")
    if rate_limit is not None:
        if not isinstance(rate_limit, dict) or set(rate_limit) - RATE_LIMIT_KEYS:
            raise ValueError(f"{path}: rate_limit may only set {sorted(RATE_LIMIT_KEYS)}")
        Meter(0.0, {}, **rate_limit)  # validates the values
    arm = raw.get("arm", "baseline")
    think_instruction = raw.get("think_instruction")
    if arm not in ARMS:
        raise ValueError(f"{path}: arm must be one of {ARMS}")
    if arm == "think" and think_instruction != THINK_INSTRUCTION:
        raise ValueError(f"{path}: think_instruction must match the frozen arm text")
    if arm == "baseline" and think_instruction is not None:
        raise ValueError(f"{path}: think_instruction requires arm: think")
    think_cap_message = raw.get("think_cap_message")
    if arm == "think" and think_cap_message != THINK_GIVE_UP_TEXT:
        raise ValueError(f"{path}: think_cap_message must match the frozen arm text")
    if arm == "baseline" and think_cap_message is not None:
        raise ValueError(f"{path}: think_cap_message requires arm: think")
    return RunConfig(
        path=path, agent=agent, user=user, judge_mode=judge_mode,
        stub_judge_met=bool(grader.get("stub_met", True)),
        cap_usd=float(budget["cap_usd"]), prices_path=prices_path,
        seed=int(raw.get("seed", 300)),
        max_steps=int(orchestrator.get("max_steps", 200)),
        max_errors=int(orchestrator.get("max_errors", 10)),
        enforce_communication_protocol=bool(
            orchestrator.get("enforce_communication_protocol", False)),
        timeout_s=None if timeout is None else float(timeout),
        sources=sources, rate_limit=rate_limit, arm=arm, think_instruction=think_instruction,
        think_cap_message=think_cap_message,
    )


# Provider key needed per model-id prefix (names only; values are never logged).
API_KEY_PREFIXES = (("openrouter/", "OPENROUTER_API_KEY"), ("openai/", "OPENAI_API_KEY"),
                    ("gpt-", "OPENAI_API_KEY"), ("deepseek/", "DEEPSEEK_API_KEY"))


def api_key_for(model: str) -> str:
    for prefix, name in API_KEY_PREFIXES:
        if model.startswith(prefix):
            return name
    raise ValueError(f"no known API key for model {model!r}")


def required_api_keys(config: RunConfig) -> dict[str, list[str]]:
    """Key names each paid model needs. At a $0 cap no paid call can be authorized."""
    if config.cap_usd == 0:
        return {}
    models = [m.model for m in (config.agent, config.user) if m.backend == "litellm"]
    if config.judge_mode == "official":
        models.append(judge_config().model)
    keys: dict[str, list[str]] = {}
    for model in models:
        keys.setdefault(api_key_for(model), []).append(model)
    for model in (config.agent, config.user):
        if model.backend == "openai_compatible" and model.api_key_env:
            keys.setdefault(model.api_key_env, []).append(model.model)
    return keys


def check_api_keys(config: RunConfig) -> None:
    missing = [name for name in required_api_keys(config) if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing API key(s) {', '.join(missing)}; set them before a paid "
                           "run (nothing was spent)")


class ServingSpecMismatch(RuntimeError):
    """The active serving window does not run the code's pinned serving spec."""


class PinnedServingGate(ServingGate):
    """The serving gate of one runner session: every request and episode start
    is refused unless the active window is the one pinned at session start
    (same window id, same spec hash). A window switch mid-session aborts the
    session resumably; the next session pins the new window."""

    def __init__(self, session_dir: str | Path, model: str, pin: dict[str, str]) -> None:
        super().__init__(session_dir, model)
        self.pin = dict(pin)

    def _window(self) -> dict[str, Any]:
        window = super()._window()
        if (window.get("window_id"), window.get("spec_sha256")) != (
                self.pin["window_id"], self.pin["spec_sha256"]):
            raise ServingWindowClosed(
                f"serving window changed mid-session (pinned {self.pin['window_id']}, "
                f"active {window.get('window_id')}); resume the run to pin the new window")
        return window


def pin_serving(config: RunConfig) -> dict[str, str] | None:
    """Pin the active serving window for this session (self-served agents only).
    Raises ServingWindowClosed without a serving window, and refuses a window
    whose spec differs from the code's pinned serving spec."""
    from slh import serve_spec

    gate = gate_for(config.agent)
    if gate is None:
        return None
    window = gate._window()
    if window.get("spec_sha256") != serve_spec.spec_sha256():
        raise ServingSpecMismatch(f"serving window {window.get('window_id')} runs spec "
                           f"{window.get('spec_sha256')}, not the pinned "
                           f"{serve_spec.spec_sha256()}; refusing")
    return {"window_id": window["window_id"], "spec_sha256": window["spec_sha256"]}


def pinned_gate(config: ModelConfig, pin: dict[str, str] | None) -> ServingGate | None:
    if config.serving_session is None:
        return None
    if pin is None:
        raise ServingWindowClosed("self-served session has no pinned serving window")
    return PinnedServingGate(config.serving_session, config.model, pin)


def make_backend(config: ModelConfig, role: str, level: str = "messages",
                 pin: dict[str, str] | None = None):
    """A participant backend. Only the agent ever requests token-level capture.
    A self-served backend checks the session's pinned window before every request."""
    level = level if role == "agent" else "messages"
    if config.backend == "stub":
        return StubAgentBackend(config) if role == "agent" else StubUserBackend(config)
    if config.backend == "openai_compatible":
        from slh.openai_compat import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend(config, call_name=f"{role}_response", level=level)
        if backend.gate is not None:
            backend.gate = pinned_gate(config, pin)
        return backend
    return LiteLLMBackend(config, call_name=f"{role}_response", level=level)


def capture_plan(manifest: dict[str, Any], config: RunConfig, meter: Meter,
                 recorder: CallRecorder | None = None,
                 pin: dict[str, str] | None = None) -> str:
    """The agent's requested capture level for this run, recorded in the
    manifest once (resume reuses it). ``capture: auto`` on a self-served
    endpoint runs one metered probe; hosted providers request log-probs only
    when the pinned provider advertises them. Every call is still graded on
    its own response (slh.tokens)."""
    recorded = (manifest.get("capture") or {}).get("agent")
    if recorded:
        return recorded["request_level"]
    agent = config.agent
    plan: dict[str, Any] = {"mode": agent.capture, "backend": agent.backend}
    if agent.backend == "openai_compatible" and agent.capture == "auto":
        from slh.openai_compat import probe

        backend = make_backend(agent, "agent", pin=pin)
        if recorder is not None:
            recorder.requested_levels = {"agent": max(agent.capture_levels(), key=level_rank)}
            meter.on_attempt = recorder
            recorder.begin("probe", f"probe-{uuid.uuid4().hex}", frozenset())
        try:
            plan["probe"] = probe(backend, meter)
        finally:
            if recorder is not None:
                recorder.end()
        level = plan["probe"]["detected_level"]
    else:
        level = max(agent.capture_levels(), key=level_rank) if agent.capture != "messages" \
            else "messages"
    plan["request_level"] = level
    manifest.setdefault("capture", {})["agent"] = plan
    return level


def _run_capture_level(out: Path, manifest: dict[str, Any]) -> None:
    """Run level = the minimum over its valid episodes' levels."""
    valid = {o["episode_id"] for o in read_jsonl_tolerant(out / "outcomes.jsonl")[0]
             if o.get("valid") is True}
    levels = [e.get("capture_level", "messages")
              for e in read_jsonl_tolerant(out / "episodes.jsonl")[0]
              if e["episode_id"] in valid]
    manifest.setdefault("capture", {})["run_level"] = min_level(levels)


# --------------------------------------------------------------------------- tasks


def resolve_task_ids(domain: str, split: str, splits_dir: Path = REPO_ROOT / "splits"
                     ) -> tuple[str, list[str], str | None]:
    """Return (split_name, task_ids, split_file_sha256).

    ``split`` is one split name, several comma-separated split names (their
    tasks in that order; the split name becomes e.g. ``stream+dev+confirm``),
    or comma-separated task ids. A multi-split run is not stream evidence:
    lesson and export eligibility require ``split_name == "stream"``.
    """
    path = splits_dir / f"{domain}.json"
    parts = [part.strip() for part in split.split(",") if part.strip()]
    if len(parts) > 1 and all(part in SPLIT_NAMES for part in parts):
        if len(set(parts)) != len(parts):
            raise ValueError("a split name is repeated")
        ids: list[str] = []
        digest = None
        for part in parts:
            _, part_ids, digest = resolve_task_ids(domain, part, splits_dir)
            if set(part_ids) & set(ids):
                raise ValueError(f"split {part!r} shares task ids with an earlier split")
            ids += part_ids
        return "+".join(parts), ids, digest
    if split in SPLIT_NAMES:
        if not path.exists():
            raise ValueError(f"split {split!r} requested but {path} does not exist")
        body = path.read_bytes()
        manifest = json.loads(body)
        ids = manifest.get(split)
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise ValueError(f"{path}: no task list for split {split!r}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: duplicate task ids in {split!r}")
        return split, list(ids), hashlib.sha256(body).hexdigest()
    ids = [part.strip() for part in split.split(",") if part.strip()]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("expected a split name or comma-separated unique task ids")
    return "ids", ids, None


def task_split_map(domain: str, split_name: str, task_ids: list[str],
                   splits_dir: Path) -> dict[str, str] | None:
    """Each task's split for named-split runs (None for explicit task ids)."""
    if split_name == "ids":
        return None
    out: dict[str, str] = {}
    for name in split_name.split("+"):
        for task in resolve_task_ids(domain, name, splits_dir)[1]:
            out[task] = name
    return {task: out[task] for task in task_ids}


def load_tasks(domain: str, task_ids: list[str]) -> list[Any]:
    from tau2.runner.helpers import get_tasks

    by_id = {t.id: t for t in get_tasks(domain, task_ids=task_ids)}
    return [by_id[i] for i in task_ids]


def trial_seeds(base_seed: int, trials: int) -> list[int]:
    """Same derivation as tau2's batch runner: seed a RNG, draw one per trial."""
    rng = random.Random(base_seed)
    return [rng.randint(0, 1_000_000) for _ in range(trials)]


def episode_id_for(task_id: str, trial: int) -> str:
    return f"ep-{task_id}-{trial}"


# --------------------------------------------------------------------------- provenance


def _git(path: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(path), *args], check=True,
                          capture_output=True, text=True).stdout


def _git_sha(path: Path) -> tuple[str | None, bool | None]:
    try:
        return (_git(path, "rev-parse", "HEAD").strip(),
                bool(_git(path, "status", "--porcelain").strip()))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def tau2_root() -> Path:
    import tau2

    return Path(tau2.__file__).resolve().parents[2]


def tau2_commit() -> str | None:
    return _git_sha(tau2_root())[0]


def check_tau2_pinned() -> str:
    """The vendored tau2 must be the pinned commit, clean, and reading its own data."""
    from tau2.utils import DATA_DIR

    root = tau2_root()
    commit, dirty = _git_sha(root)
    if commit != TAU2_COMMIT_PINNED:
        raise RuntimeError(f"tau2 checkout is at {commit}, pinned {TAU2_COMMIT_PINNED}")
    if dirty is not False:
        raise RuntimeError(f"tau2 checkout at {root} has uncommitted or untracked changes")
    expected = (root / "data").resolve()
    env = os.environ.get("TAU2_DATA_DIR")
    if Path(DATA_DIR).resolve() != expected or (env and Path(env).resolve() != expected):
        raise RuntimeError(f"tau2 data directory must be {expected} "
                           f"(TAU2_DATA_DIR={env!r}, in use {DATA_DIR})")
    return commit


def data_hashes(domain: str) -> dict[str, str]:
    """sha256 of the data files the run reads: tasks, db, policy, user guidelines."""
    from tau2.utils import DATA_DIR

    data = Path(DATA_DIR) / "tau2"
    files = [data / "domains" / domain / name
             for name in ("tasks.json", "split_tasks.json", "db.json", "policy.md")]
    files += sorted((data / "user_simulator").glob("simulation_guidelines*.md"))
    return {p.relative_to(data).as_posix(): _sha256_file(p) for p in files if p.exists()}


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def judge_config() -> ModelConfig:
    """tau2's NL-assertion judge exactly as tau2 calls it (only temperature is sent).

    No max_tokens is sent, so the reservation uses the model's max_output_tokens.
    """
    from tau2.config import DEFAULT_LLM_NL_ASSERTIONS, DEFAULT_LLM_NL_ASSERTIONS_ARGS

    return ModelConfig(backend="litellm", model=DEFAULT_LLM_NL_ASSERTIONS,
                       temperature=DEFAULT_LLM_NL_ASSERTIONS_ARGS.get("temperature"))


class JudgeCoverageError(ValueError):
    """The judge reply does not give exactly one verdict per task assertion."""


def judge_assertions(messages: list[Any]) -> list[str]:
    """The assertion list tau2 put into its judge prompt."""
    prompt = messages[-1].content or ""
    match = re.search(r"expectedOutcomes:\s*(\[.*\])\s*$", prompt, re.DOTALL)
    if match is None:
        raise JudgeCoverageError("judge prompt carries no assertion list")
    assertions = ast.literal_eval(match.group(1))
    if not isinstance(assertions, list) or not all(isinstance(a, str) for a in assertions):
        raise JudgeCoverageError("judge prompt assertion list is malformed")
    return assertions


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split()).strip(" .\"'")


def check_judge_coverage(reply_content: str | None, assertions: list[str]) -> None:
    """Exactly one boolean verdict per assertion, matching its text, in order or not.

    tau2's evaluator treats a missing ``results`` list as all-met, so an empty
    or partial reply would otherwise manufacture a pass.
    """
    try:
        data = json.loads(reply_content or "")
    except ValueError:
        raise JudgeCoverageError("judge reply is not JSON") from None
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list) or len(results) != len(assertions):
        raise JudgeCoverageError("judge reply does not have one verdict per assertion")
    remaining = [_norm(a) for a in assertions]
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("metExpectation"), bool):
            raise JudgeCoverageError("judge verdict is missing a boolean metExpectation")
        outcome = _norm(result.get("expectedOutcome", ""))
        if outcome not in remaining:
            raise JudgeCoverageError("judge verdict does not match a task assertion")
        remaining.remove(outcome)


def stub_judge_reply(messages: list[Any], met: bool) -> Any:
    """Deterministic stand-in for the judge's reply: one verdict per assertion."""
    from tau2.data_model.message import AssistantMessage

    assertions = judge_assertions(messages)
    results = [{"expectedOutcome": a, "reasoning": "stub judge: fixed verdict",
                "metExpectation": met} for a in assertions]
    return AssistantMessage(role="assistant", content=json.dumps({"results": results}),
                            cost=0.0)


@contextlib.contextmanager
def metered_grader(meter: Meter, mode: str = "official", stub_met: bool = True
                   ) -> Iterator[None]:
    """Route tau2's natural-language-assertion judge call through the meter.

    tau2's evaluator runs unchanged and the official judge request is unchanged
    (the guard only turns provider retries off; the meter retries instead). In
    ``stub`` mode the call is answered by a deterministic stub at zero price.
    """
    import tau2.evaluator.evaluator_nl_assertions as nl

    if mode not in JUDGE_MODES:
        raise ValueError(f"unknown judge mode {mode!r}")
    original = nl.generate
    official = judge_config()

    def generate(model, messages, tools=None, **kwargs):
        payload = request_payload(messages, tools)
        assertions = judge_assertions(messages)
        if mode == "stub":
            def invoke():
                reply = stub_judge_reply(messages, stub_met)
                check_judge_coverage(reply.content, assertions)
                return reply, Usage(input_token_bound(payload), len(reply.content) // 4)

            return meter.call("judge", STUB_JUDGE, payload, invoke)

        def invoke_official():
            reply = original(model=model, messages=messages, tools=tools, **kwargs)
            # Usage was charged at the wire before this check can fail.
            check_judge_coverage(reply.content, assertions)
            return reply, None

        return meter.call("judge", replace(official, model=model), payload, invoke_official)

    nl.generate = generate
    try:
        yield
    finally:
        nl.generate = original


def evaluate(simulation: Any, task: Any, domain: str) -> Any:
    """tau2's official evaluation, exactly as its run_simulation calls it."""
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.modes import CommunicationMode

    return evaluate_simulation(simulation=simulation, task=task,
                               evaluation_type=EvaluationType.ALL, solo_mode=False,
                               domain=domain, mode=CommunicationMode.HALF_DUPLEX)


def breakdown_dict(reward_info: Any) -> dict[str, float]:
    items = (reward_info.reward_breakdown or {}).items()
    return {getattr(k, "value", str(k)): float(v) for k, v in items}


# --------------------------------------------------------------------------- private dir and logs


def private_dir(out: Path) -> Path:
    private = out / "private"
    private.mkdir(parents=True, exist_ok=True)
    readme = private / "README.txt"
    if not readme.exists():
        readme.write_text(PRIVATE_README, encoding="utf-8")
    return private


def _console_filter(record: Any) -> bool:
    return not record["name"].startswith(("tau2", "litellm"))


@contextlib.contextmanager
def isolated_logging(private: Path, log_name: str) -> Iterator[None]:
    """Send all tau2 and provider logging to private/; disable tau2's LLM call logs."""
    from loguru import logger
    from tau2.utils import llm_utils

    logger.remove()
    logger.add(private / log_name, level="DEBUG")
    logger.add(sys.stderr, level="WARNING", filter=_console_filter)
    handler = logging.FileHandler(private / f"provider-{log_name}", encoding="utf-8")
    saved = []
    for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "litellm", "httpx", "openai"):
        std = logging.getLogger(name)
        saved.append((std, list(std.handlers), std.propagate))
        std.handlers, std.propagate = [handler], False
    dir_token = llm_utils.llm_log_dir.set(None)
    try:
        yield
    finally:
        llm_utils.llm_log_dir.reset(dir_token)
        for std, handlers, propagate in saved:
            std.handlers, std.propagate = handlers, propagate
        handler.close()
        logger.remove()
        logger.add(sys.stderr, level="WARNING", filter=_console_filter)


def _record_error(private: Path, episode_id: str, stage: str, exc: BaseException,
                  code: str) -> dict[str, str]:
    public = error_record(exc, code)
    append_jsonl(private / "errors.jsonl", {
        "time_utc": _now(), "episode_id": episode_id, "stage": stage, **public,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    })
    return public


# --------------------------------------------------------------------------- snapshot


# Paths whose content the harness executes or reads during a run.
CODE_PATHS = ("slh", "configs", "splits", "tests", "pyproject.toml")


def code_identity() -> dict[str, Any]:
    """Harness revision plus a hash of every uncommitted change to executed paths."""
    sha = _git(REPO_ROOT, "rev-parse", "HEAD").strip()
    diff = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "HEAD", "--binary", "--",
                           *CODE_PATHS], check=True, capture_output=True).stdout
    untracked = _git(REPO_ROOT, "ls-files", "--others", "--exclude-standard", "--",
                     *CODE_PATHS).splitlines()
    files = {p: _sha256_file(REPO_ROOT / p) for p in sorted(untracked)
             if (REPO_ROOT / p).is_file()}
    return {"repo_git_sha": sha, "code_diff_sha256": _sha256_bytes(diff),
            "untracked_code_sha256": _sha256_bytes(json.dumps(files, sort_keys=True).encode()),
            "untracked_code_files": sorted(files)}


def _write_snapshot(out: Path, config: RunConfig) -> dict[str, str]:
    """Copy configs and prices; record uncommitted state so the run can be rebuilt.

    Untracked files under the executed code paths are copied in full into
    snapshot/untracked/; other untracked files are recorded by hash only.
    """
    snap = out / "snapshot"
    snap.mkdir(exist_ok=True)
    for name, source in config.sources.items():
        (snap / name).write_bytes(source.read_bytes())
    (snap / "resolved.json").write_text(
        json.dumps(config.resolved(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _, dirty = _git_sha(REPO_ROOT)
    if dirty:
        diff = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "HEAD", "--binary"],
                              check=True, capture_output=True).stdout
        (snap / "repo.diff").write_bytes(diff)
        untracked = _git(REPO_ROOT, "ls-files", "--others", "--exclude-standard").splitlines()
        hashes = {}
        for rel in sorted(untracked):
            source = REPO_ROOT / rel
            if not source.is_file():
                continue
            hashes[rel] = _sha256_file(source)
            if rel.split("/", 1)[0] in CODE_PATHS:
                target = snap / "untracked" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
        (snap / "untracked.json").write_text(json.dumps(hashes, indent=2) + "\n",
                                             encoding="utf-8")
    return {p.relative_to(snap).as_posix(): _sha256_file(p)
            for p in sorted(snap.rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------- run


def _usage_fields(tally: dict[str, RoleTally]) -> dict[str, int]:
    return {"agent_in": tally["agent"].input_tokens, "agent_out": tally["agent"].output_tokens,
            "agent_cached": tally["agent"].cached_tokens, "user_in": tally["user"].input_tokens,
            "user_out": tally["user"].output_tokens}


def _write_manifest(out: Path, manifest: dict[str, Any]) -> None:
    _write_json_atomic(out / "manifest.json", manifest)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write("".join(dumps(r) for r in records))
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def make_meter(config: RunConfig, ledger: SharedLedger | None = None) -> Meter:
    return Meter(config.cap_usd, load_prices(config.prices_path), ledger=ledger,
                 **config.rate_limit_params())


def _serving_identity(config: RunConfig) -> dict[str, Any]:
    """For a self-served agent: the serving window this session ran against."""
    gate = gate_for(config.agent)
    return {} if gate is None else {"serving": gate.identity()}


def _canonical_pairs(manifest: dict[str, Any], tasks: list[Any]) -> list[tuple[int, int, Any]]:
    """(trial, seed, task) in the sequential run order: trial-major, manifest task order.

    A trial extension (``trial_offset`` k) numbers its trials k, k+1, ...: its
    episode ids never collide with the parent run's, and seeds come from
    tau2's derivation at those indices."""
    offset = int(manifest.get("trial_offset", 0))
    return [(offset + index, seed, task) for index, seed in enumerate(manifest["seeds"])
            for task in tasks]


def run(
    domain: str, split: str, config_path: str | Path, trials: int, out: str | Path,
    state_dir: str | Path | None = None, splits_dir: Path = REPO_ROOT / "splits",
    cap_usd: float | None = None, workers: int = 1, worker_init: Any = None,
    trial_offset: int = 0, parent_run: dict[str, Any] | None = None,
    protocol: str | Path | None = None,
) -> dict[str, Any]:
    """Start a new run and write the run layout. Returns the final manifest.

    ``trial_offset`` k runs trials k..k+trials-1 (seeds from tau2's per-trial
    derivation at those indices); ``extend_trials`` uses it with ``parent_run``
    to add trials to a finished run as a new, linked run directory.
    ``protocol``: the registered protocol text this run belongs to (e.g.
    docs/protocol-amendment-1.md); its path and SHA-256 go into the manifest
    next to the base fingerprint and the state version.

    ``cap_usd`` overrides the config's budget cap for this run; the effective
    cap is what the manifest records and what resume enforces. ``workers`` > 1
    runs the (task, trial) pairs in that many worker processes under one
    shared cap (see ``_execute_parallel``). ``worker_init`` is a picklable
    callable each worker runs first (tests use it to install offline fakes).
    """
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"domain {domain!r} not supported; choose from {SUPPORTED_DOMAINS}")
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if isinstance(trial_offset, bool) or not isinstance(trial_offset, int) or trial_offset < 0:
        raise ValueError("trial_offset must be a nonnegative integer")
    _check_workers(workers)
    protocol_record = None if protocol is None else {
        "path": str(Path(protocol).resolve()), "sha256": _sha256_file(Path(protocol))}
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty run directory {out} "
                              "(use --resume to continue it)")
    config = load_config(config_path)
    config_cap = config.cap_usd
    if config.agent.serving_session is not None and cap_usd is None:
        # A self-served run's config cap (configs/a0-vllm.yaml: $20) is never
        # used implicitly: every such run states its approved cap.
        raise ValueError("a self-served (serving_session) run needs an explicit --cap")
    if cap_usd is not None:
        config = replace(config, cap_usd=float(cap_usd))
    check_api_keys(config)
    commit = check_tau2_pinned()
    split_name, task_ids, split_sha = resolve_task_ids(domain, split, splits_dir)
    split_verified = None
    if Path(splits_dir).resolve() == (REPO_ROOT / "splits").resolve():
        # The frozen split files and the pinned task file are rebuilt from the
        # vendored sources and must match, for named splits and explicit ids alike.
        from slh.splits import verify

        verify(tau2_root(), splits_dir)
        split_verified = True
    tasks = load_tasks(domain, task_ids)
    state = HarnessState.load(state_dir) if state_dir else HarnessState.empty()
    check_state_allowed(state, config)
    build_guards(state.guards)
    # Worker slots of the serving session are taken before anything is written:
    # a refusal leaves no run directory behind.
    with worker_slots(config.agent.serving_session, workers):
        repo_sha, repo_dirty = _git_sha(REPO_ROOT)

        out.mkdir(parents=True, exist_ok=True)
        private_dir(out)
        snapshot = _write_snapshot(out, config)
        manifest: dict[str, Any] = {
            "run_id": out.name, "created_utc": _now(), "status": "running",
            # A rehearsal-derived harness state makes any run ineligible as evidence.
            "evidence_eligible": config.evidence_eligible and not state.rehearsal,
            "state_rehearsal": state.rehearsal,
            "repo_git_sha": repo_sha, "repo_dirty": repo_dirty, "tau2_commit": commit,
            "data_sha256": data_hashes(domain),
            "domain": domain, "split_name": split_name, "split_file_sha256": split_sha,
            "task_splits": task_split_map(domain, split_name, task_ids, splits_dir),
            "split_verified": split_verified, "task_ids": task_ids, "trials": trials,
            "agent_model": config.agent.to_dict(), "user_model": config.user.to_dict(),
            "judge_mode": config.judge_mode,
            "arm": config.arm,
            "grader": {"evaluation_type": "all", "judge_mode": config.judge_mode,
                       "nl_assertions_judge": (STUB_JUDGE if config.judge_mode == "stub"
                                               else judge_config()).to_dict(),
                       "stub_judge_met": (config.stub_judge_met if config.judge_mode == "stub"
                                          else None)},
            "harness_version_id": state.version_id,
            "state_dir": None if state_dir is None else str(Path(state_dir).resolve()),
            "seeds": trial_seeds(config.seed, trial_offset + trials)[trial_offset:],
            "base_seed": config.seed, "trial_offset": trial_offset,
            "trial_indices": list(range(trial_offset, trial_offset + trials)),
            "parent_run": parent_run,
            # (base arm, state version) is the harness identity: B0 and B1 share
            # the empty state's version id.
            "base_arm": base_arm_fingerprint(config), "state_base_arm": state.base,
            "execution_fingerprint": current_execution_fingerprint(config, domain),
        "protocol": protocol_record,
            "budget_cap_usd": config.cap_usd, "config_cap_usd": config_cap,
            "cap_override": cap_usd is not None,
            "orchestrator": config.resolved()["orchestrator"],
            "config_path": str(config.path), "config_resolved_sha256": config.resolved_sha256(),
            "snapshot_sha256": snapshot,
            "code_identity": code_identity(),
            "trace_format": TRACE_FORMAT,
            "workers": workers,
            "sessions": [],
        }
        _write_manifest(out, manifest)
        for name in ("episodes.jsonl", "outcomes.jsonl", "calls.jsonl"):
            (out / name).touch()
        if workers > 1:
            return _execute_parallel(out, manifest, config, tasks, state, workers, worker_init)
        return _execute(out, manifest, config, tasks, state, done=set())


def extend_trials(parent: str | Path, out: str | Path, trials: int, *,
                  cap_usd: float | None = None, workers: int = 1, worker_init: Any = None,
                  splits_dir: Path = REPO_ROOT / "splits",
                  protocol: str | Path | None = None) -> dict[str, Any]:
    """Run ``trials`` more trials of a finished run as a NEW run directory.

    The parent is only read. Before anything is written, its execution
    identity is ENFORCED against the current code, config and state
    (``slh.execution.verify_run_identity``): code surface, base arm, rendered
    prompt and tool schemas, serving spec, tau2 commit and data, seeds. The
    extension runs the same tasks (same split selector and split hash) with
    trial indices continuing after the parent's, seeds from tau2's per-trial
    derivation, and records ``parent_run`` (including the parent's file
    hashes) so ``python -m slh merge-trials`` can verify the pair.
    """
    parent = Path(parent)
    manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    config = load_config(manifest["config_path"])
    state_dir = manifest.get("state_dir")
    state = HarnessState.load(state_dir) if state_dir else HarnessState.empty()
    verify_run_identity(parent, config, state)
    indices = trial_indices(manifest)
    split_name = manifest["split_name"]
    selector = (",".join(manifest["task_ids"]) if split_name == "ids"
                else split_name.replace("+", ","))
    _, task_ids, split_sha = resolve_task_ids(manifest["domain"], selector, splits_dir)
    if task_ids != manifest["task_ids"] or split_sha != manifest.get("split_file_sha256"):
        raise IdentityMismatch("the split now resolves to other tasks or another split file "
                               "than the parent's; refusing to extend")
    link = {"run_id": manifest["run_id"], "path": str(parent.resolve()),
            "trial_indices": indices, "files_sha256": run_files_digest(parent),
            "manifest_sha256": _sha256_file(parent / "manifest.json")}
    return run(manifest["domain"], selector, manifest["config_path"], trials, out,
               state_dir=state_dir, splits_dir=splits_dir, cap_usd=cap_usd, workers=workers,
               worker_init=worker_init, trial_offset=max(indices) + 1, parent_run=link,
               protocol=protocol)


def _check_workers(workers: Any) -> None:
    if isinstance(workers, bool) or not isinstance(workers, int) or not (
            1 <= workers <= MAX_WORKERS):
        raise ValueError(f"workers must be an integer from 1 to {MAX_WORKERS}")


def _check_resumable(out: Path) -> tuple[dict[str, Any], RunConfig, HarnessState]:
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    config = load_config(manifest["config_path"])
    if manifest.get("cap_override"):
        config = replace(config, cap_usd=float(manifest["budget_cap_usd"]))
    if config.resolved_sha256() != manifest["config_resolved_sha256"]:
        raise RuntimeError("config or prices changed since the run started; refusing to resume")
    check_api_keys(config)
    if check_tau2_pinned() != manifest["tau2_commit"]:
        raise RuntimeError("tau2 commit differs from the run's; refusing to resume")
    if data_hashes(manifest["domain"]) != manifest["data_sha256"]:
        raise RuntimeError("tau2 data files changed since the run started; refusing to resume")
    state_dir = manifest.get("state_dir")
    state = HarnessState.load(state_dir) if state_dir else HarnessState.empty()
    if state.version_id != manifest["harness_version_id"]:
        raise RuntimeError("harness state differs from the run's; refusing to resume")
    check_state_allowed(state, config)
    if code_identity() != manifest["code_identity"]:
        raise RuntimeError("harness code (revision or uncommitted changes) differs from the "
                           "run's first session; refusing to resume")
    return manifest, config, state


def resume(out: str | Path, workers: int | None = None,
           worker_init: Any = None) -> dict[str, Any]:
    """Continue an interrupted run: skip episodes that have both lines.

    A parallel run resumes in parallel: only the missing pairs are re-split
    across ``workers`` (default: the run's worker count). A sequential run
    resumes sequentially.
    """
    out = Path(out)
    manifest, config, state = _check_resumable(out)
    parallel = manifest.get("workers", 1) > 1
    if workers is not None:
        _check_workers(workers)
        if not parallel or workers < 2:
            raise ValueError("--workers on resume applies to parallel runs only, "
                             "with 2 or more workers")
    tasks = load_tasks(manifest["domain"], manifest["task_ids"])
    count = (workers or manifest["workers"]) if parallel else 1
    with worker_slots(config.agent.serving_session, count):
        if parallel:
            return _execute_parallel(out, manifest, config, tasks, state, count, worker_init)
        done = _repair(out)
        return _execute(out, manifest, config, tasks, state, done=done)


def _repair(out: Path, quarantine: Path | None = None, label: str | None = None) -> set[str]:
    """Quarantine torn tails and unpaired lines in private/; return finished ids.

    Nothing is deleted: every removed line is first appended to the quarantine.
    """
    quarantine = quarantine or private_dir(out) / "quarantine.jsonl"
    where = {"worker": label} if label else {}
    kept: dict[str, list[dict[str, Any]]] = {}
    for name in ("episodes.jsonl", "outcomes.jsonl"):
        records, torn = read_jsonl_tolerant(out / name)
        if torn is not None:
            append_jsonl(quarantine, {"time_utc": _now(), "file": name, **where,
                                      "torn_line": torn})
        kept[name] = records
    done = ({r["episode_id"] for r in kept["episodes.jsonl"]}
            & {r["episode_id"] for r in kept["outcomes.jsonl"]})
    for name, records in kept.items():
        for record in records:
            if record["episode_id"] not in done:
                append_jsonl(quarantine, {"time_utc": _now(), "file": name, **where,
                                          "unpaired_record": record})
        _write_jsonl_atomic(out / name, [r for r in records if r["episode_id"] in done])
    return done


def _drop_torn_tail(path: Path, quarantine: Path, label: str) -> list[dict[str, Any]]:
    """Complete records of an append-only log; a torn tail is quarantined and cut."""
    records, torn = read_jsonl_tolerant(path)
    if torn is not None:
        append_jsonl(quarantine, {"time_utc": _now(), "file": path.name, "worker": label,
                                  "torn_line": torn})
        _write_jsonl_atomic(path, records)
    return records


@dataclass
class _Sink:
    """Where one execution (the sequential run or one worker) writes."""

    episodes: Path
    outcomes: Path
    private: Path  # errors.jsonl, episode_blobs.jsonl, logs
    blobs: BlobStore
    private_blobs: BlobStore


def _sink(out: Path, public: Path, private: Path) -> _Sink:
    return _Sink(episodes=public / "episodes.jsonl", outcomes=public / "outcomes.jsonl",
                 private=private, blobs=BlobStore(out / "blobs"),
                 private_blobs=BlobStore(private_dir(out) / "blobs"))


def _run_pairs(domain: str, config: RunConfig, state: HarnessState,
               pairs: list[tuple[int, int, Any]], meter: Meter, sink: _Sink,
               recorder: CallRecorder, done: set[str], on_episode: Any,
               agent_level: str = "messages", serving_pin: dict[str, str] | None = None
               ) -> tuple[str | None, dict[str, str] | None]:
    """Run each (trial, seed, task) not in ``done``; return (abort status, reason).

    Raises GradingLeak and UnmeteredCall (fatal); the caller records them.
    """
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.registry import registry

    guards = build_guards(state.guards)
    # Self-served window: deadlines (slh.serving) and the session's pinned window.
    serving = pinned_gate(config.agent, serving_pin)
    agent_backend = make_backend(config.agent, "agent", agent_level, pin=serving_pin)
    user_backend = make_backend(config.user, "user", pin=serving_pin)
    base_arm = base_arm_fingerprint(config)
    recorder.requested_levels = {"agent": agent_level}
    meter.on_attempt = recorder
    try:
        with isolated_logging(sink.private, "tau2.log"), completion_guard(), \
                metered_grader(meter, config.judge_mode, config.stub_judge_met):
            for trial, seed, task in pairs:
                episode_id = episode_id_for(task.id, trial)
                if episode_id in done:
                    continue
                if serving is not None:
                    try:
                        serving.check_episode_start()
                    except ServingWindowClosed as exc:
                        raise RunAborted("aborted", _record_error(
                            sink.private, episode_id, "episode_start", exc,
                            "serving_window_closed")) from None
                execution_id = uuid.uuid4().hex
                windows, _ = task_leak_windows(task)
                recorder.begin(episode_id, execution_id, windows)
                before = meter.snapshot()
                environment = registry.get_env_constructor(domain)()
                agent_type = ThinkAgent if config.arm == "think" else HarnessAgent
                arm_kwargs = ({"think_instruction": config.think_instruction,
                               "think_cap_message": config.think_cap_message}
                              if config.arm == "think" else {})
                agent = agent_type(tools=environment.get_tools(),
                                   domain_policy=environment.get_policy(),
                                   backend=agent_backend, meter=meter, state=state,
                                   guards=guards, **arm_kwargs)
                check_agent_prompt_clean(agent.system_prompt, task,
                                         policy=environment.get_policy())
                # Exactly what the agent request carries (tau2's generate sends
                # each tool's openai_schema), stored once by content hash.
                tool_schemas = [tool.openai_schema for tool in agent.tools]
                check_no_task_text(tool_schemas, windows)
                prompt_sha = sink.blobs.put(agent.system_prompt)
                tools_sha = sink.blobs.put(tool_schemas)
                try:
                    user_tools = environment.get_user_tools(include=task.user_tools) or None
                except Exception:  # noqa: BLE001 - same fallback as tau2's build_user
                    user_tools = None
                user = HarnessUser(backend=user_backend, meter=meter,
                                   instructions=str(task.user_scenario), tools=user_tools)
                # The simulator's prompt holds the scenario: private store only.
                user_prompt_sha = sink.private_blobs.put(user.system_prompt)
                append_jsonl(sink.private / "episode_blobs.jsonl", {
                    "episode_id": episode_id, "execution_id": execution_id,
                    "user_system_prompt_sha256": user_prompt_sha})
                orchestrator = Orchestrator(
                    domain=domain, agent=agent, user=user, environment=environment,
                    task=task, max_steps=config.max_steps, max_errors=config.max_errors,
                    seed=seed, solo_mode=False, simulation_id=episode_id,
                    validate_communication=config.enforce_communication_protocol,
                    timeout=config.timeout_s)
                started = time.perf_counter()
                simulation, error, abort = None, None, None
                try:
                    simulation = orchestrator.run()
                except (GradingLeak, UnmeteredCall):
                    raise
                except ServingWindowClosed as exc:
                    # The serving window closed mid-episode: nothing is written for
                    # this episode, so resume re-runs it in a later window. Its
                    # spend (user simulator calls) is already in the meter.
                    raise RunAborted("aborted", _record_error(
                        sink.private, episode_id, "simulation", exc,
                        "serving_window_closed")) from None
                except BudgetExceeded as exc:
                    error = _record_error(sink.private, episode_id, "simulation", exc,
                                          "budget_refused")
                    abort = RunAborted("aborted", error)
                except Overspend as exc:
                    error = _record_error(sink.private, episode_id, "simulation", exc,
                                          "overspend")
                    abort = RunAborted("aborted_overspend", error)
                except Exception as exc:  # noqa: BLE001 - becomes an invalid episode
                    error = _record_error(sink.private, episode_id, "simulation", exc,
                                          "simulation_crash")
                finally:
                    # Think telemetry is private: never in episodes, outcomes or evidence.
                    if isinstance(agent, ThinkAgent):
                        append_jsonl(sink.private / "think.jsonl", {
                            "episode_id": episode_id, "execution_id": execution_id,
                            **agent.private_episode_metrics})
                wall = time.perf_counter() - started
                middle = meter.snapshot()
                reward, breakdown, grade_error = 0.0, {}, None
                if simulation is not None:
                    simulation.policy = environment.get_policy()
                    try:
                        info = evaluate(simulation, task, domain)
                        reward, breakdown = float(info.reward), breakdown_dict(info)
                    except (GradingLeak, UnmeteredCall):
                        raise
                    except BudgetExceeded as exc:
                        grade_error = _record_error(sink.private, episode_id, "grading", exc,
                                                    "grading_budget_refused")
                        abort = RunAborted("aborted", grade_error)
                    except Overspend as exc:
                        grade_error = _record_error(sink.private, episode_id, "grading", exc,
                                                    "overspend")
                        abort = RunAborted("aborted_overspend", grade_error)
                    except JudgeCoverageError as exc:
                        grade_error = _record_error(sink.private, episode_id, "grading", exc,
                                                    "grading_failure_judge_coverage")
                    except Exception as exc:  # noqa: BLE001
                        grade_error = _record_error(sink.private, episode_id, "grading", exc,
                                                    "grading_failed")
                level = recorder.episode_level()
                recorder.end()
                after = meter.snapshot()
                spent = delta(before, middle)
                judged = delta(middle, after)["judge"]
                trajectory = (simulation.messages if simulation is not None
                              else getattr(orchestrator, "trajectory", []))
                episode: dict[str, Any] = {
                    "episode_id": episode_id, "task_id": task.id, "trial": trial,
                    "seed": seed, "messages": messages_to_records(trajectory),
                    "termination_reason": (simulation.termination_reason.value
                                           if simulation is not None else "error"),
                    "usage": _usage_fields(spent),
                    "usd": round(spent["agent"].usd + spent["user"].usd, 9),
                    "wall_s": round(wall, 4), "harness_version": state.version_id,
                    "execution_id": execution_id,
                    "agent_system_prompt_sha256": prompt_sha, "tools_sha256": tools_sha,
                    "capture_level": level, "base_arm": base_arm,
                }
                if serving_pin is not None:
                    episode["serving_window_id"] = serving_pin["window_id"]
                if error:
                    episode["error"] = error
                if agent.guard_events:
                    episode["guard_events"] = [vars(e) for e in agent.guard_events]
                outcome: dict[str, Any] = {
                    "episode_id": episode_id, "task_id": task.id, "reward": reward,
                    "reward_breakdown": breakdown, "grader_usd": round(judged.usd, 9),
                    "valid": not (error or grade_error)}
                if judged.calls or judged.failed_attempts:
                    outcome["judge"] = config.judge_mode
                if error or grade_error:
                    outcome["error"] = error or grade_error
                _check_and_write(sink.episodes, sink.outcomes, episode, outcome, windows)
                done.add(episode_id)
                on_episode(outcome)
                if abort is not None:
                    raise abort
    except RunAborted as stop:
        return stop.status, stop.reason
    finally:
        recorder.end()
    return None, None


def _execute(out: Path, manifest: dict[str, Any], config: RunConfig, tasks: list[Any],
             state: HarnessState, done: set[str]) -> dict[str, Any]:
    """The sequential run: one process, the manifest carries the meter's spend."""
    private = private_dir(out)
    meter = make_meter(config)
    if manifest.get("spend"):
        meter.restore(manifest["spend"])
    repo_sha, repo_dirty = _git_sha(REPO_ROOT)
    manifest["sessions"].append({"started_utc": _now(), "repo_git_sha": repo_sha,
                                 "repo_dirty": repo_dirty, "already_done": len(done),
                                 **_serving_identity(config)})
    manifest["status"] = "running"
    manifest.pop("abort_reason", None)
    # Persist restored spend now, including any unreconciled reservation.
    manifest["spend"] = meter.summary()
    _write_manifest(out, manifest)
    invalid = sum(1 for o in read_jsonl_tolerant(out / "outcomes.jsonl")[0]
                  if o.get("valid") is False)

    def checkpoint() -> None:
        manifest.update(spend=meter.summary(), invalid_episodes=invalid,
                        episodes_written=len(done))
        _write_manifest(out, manifest)

    def on_episode(outcome: dict[str, Any]) -> None:
        nonlocal invalid
        invalid += not outcome["valid"]
        checkpoint()

    # Persist cumulative spend after every metered attempt, so an interrupted
    # run resumes with its true spend against the original cap.
    meter.on_change = lambda _meter: checkpoint()
    recorder = CallRecorder(out / "calls.jsonl", private / "calls.jsonl",
                            token_store=TokenStore(out / "tokens"))
    try:
        try:
            pin = pin_serving(config)
            manifest["sessions"][-1]["serving_pin"] = pin
            level = capture_plan(manifest, config, meter, recorder, pin=pin)
        except ServingWindowClosed as exc:
            status, reason = "aborted", _record_error(private, "probe", "capture_probe", exc,
                                                      "serving_window_closed")
        else:
            checkpoint()
            status, reason = _run_pairs(manifest["domain"], config, state,
                                        _canonical_pairs(manifest, tasks), meter,
                                        _sink(out, out, private), recorder, done, on_episode,
                                        agent_level=level, serving_pin=pin)
    except (GradingLeak, UnmeteredCall, ServingSpecMismatch) as exc:
        manifest.update(status="failed",
                        abort_reason=_record_error(private, "-", "run", exc, "fatal"),
                        finished_utc=_now(), spend=meter.summary())
        _write_manifest(out, manifest)
        raise
    if status is None:
        status = "complete_with_errors" if invalid else "complete"
    else:
        manifest["abort_reason"] = reason
    manifest.update(status=status, finished_utc=_now(), not_run=[
        episode_id_for(t.id, trial) for trial, _, t in _canonical_pairs(manifest, tasks)
        if episode_id_for(t.id, trial) not in done])
    _run_capture_level(out, manifest)
    if config.arm == "think":
        manifest["think_projection"] = write_think_projection(out, tasks)
    checkpoint()
    return manifest


THINK_WINDOW = 12  # normalized characters: shorter than the 30 used for metadata
THINK_MIN_SECRET = 6  # whole-string check for shorter hidden fragments


def _think_norm(text: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(text).lower()).split())


def _think_windows(text: str) -> set[str]:
    return {text[i:i + THINK_WINDOW] for i in range(len(text) - THINK_WINDOW + 1)}


def _hidden_task_strings(task: Any) -> list[str]:
    """Everything of the task the agent never receives: scenario, persona,
    assertions, communicate_info and every gold action argument (no length
    floor, unlike ``task_secret_strings``)."""
    from slh.trace import task_secret_strings

    strings = list(task_secret_strings(task))
    scenario = getattr(task, "user_scenario", None)
    instructions = getattr(scenario, "instructions", None)
    if isinstance(instructions, str):
        strings.append(instructions)
    elif instructions is not None:
        strings += [str(getattr(instructions, n, "") or "") for n in
                    ("reason_for_call", "known_info", "unknown_info", "task_instructions")]
    strings.append(str(getattr(scenario, "persona", "") or ""))
    criteria = getattr(task, "evaluation_criteria", None)
    if criteria is not None:
        strings += list(criteria.nl_assertions or []) + list(criteria.communicate_info or [])
        for action in criteria.actions or []:
            for value in (action.arguments or {}).values():
                strings.append(value if isinstance(value, str) else json.dumps(value))
    return [s for s in strings if s]


def _embedded_json_keys_ok(text: str) -> bool:
    """False if any JSON object/array embedded in the text has a grading key."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        try:
            check_no_grading_fields(value)
        except GradingLeak:
            return False
    return True


def _thought_withheld(text: str, secret_windows: set[str], secret_short: set[str]) -> str | None:
    """Reason to withhold a thought, or None. When in doubt, withhold."""
    from slh.trace import FORBIDDEN_KEYS

    lowered = text.lower()
    if any(key in lowered for key in FORBIDDEN_KEYS if "_" in key):
        return "grading_key_name"
    if not _embedded_json_keys_ok(text):
        return "embedded_grading_json"
    normalized = _think_norm(text)
    if not _think_windows(normalized).isdisjoint(secret_windows):
        return "hidden_task_text"
    padded = f" {normalized} "
    if any(f" {short} " in padded for short in secret_short):
        return "hidden_task_text"
    return None


def write_think_projection(out: Path, tasks: list[Any]) -> dict[str, Any]:
    """``think_public.jsonl``: the agent's own think-call text, for the learner.

    Built from the run's public ``calls.jsonl`` (never from ``private/``; the
    run directory and its input files must not be symlinks): every ``think``
    tool call in an agent response that parsed, of the attempt whose execution
    id is the episode's final one (calls without an execution id, and records
    of interrupted attempts superseded by resume, are dropped). One line per
    thought: episode id, execution id, turn, index, text, ``sanitized: true``.

    Sanitization (stricter than episode records; when in doubt the thought is
    withheld and counted, never emitted): no grading key names; no embedded
    JSON carrying a grading key; no hidden task text -- scenario, persona,
    assertions, communicate_info and gold action arguments -- matched on
    normalized (lowercase, alphanumeric) 12-character windows, and as whole
    strings for hidden fragments of 6-11 characters. Text the episode itself
    already shows publicly (its conversation, tool calls and results, and the
    domain policy) is not hidden and is not counted. Identifiers are aliased
    learner-side, as for messages. The projection is fallible evidence: what
    the agent thought, not a verified diagnosis.
    """
    from tau2.registry import registry

    for path in (out, out / "episodes.jsonl", out / "calls.jsonl"):
        if path.is_symlink():
            raise ValueError(f"think projection refuses symlinked input {path}")
    episodes = read_jsonl_tolerant(out / "episodes.jsonl")[0]
    final = {e["execution_id"]: e for e in episodes if e.get("execution_id")}
    by_task = {task.id: task for task in tasks}
    policy_windows: set[str] | None = None
    secrets: dict[str, tuple[set[str], set[str]]] = {}
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"file": "think_public.jsonl", "thoughts": 0,
                               "episodes_with_thoughts": 0, "withheld": {}, "unparseable": 0}

    def secrets_for(episode: dict[str, Any]) -> tuple[set[str], set[str]]:
        nonlocal policy_windows
        eid = episode["episode_id"]
        if eid not in secrets:
            if policy_windows is None:
                domain = json.loads((out / "manifest.json").read_text())["domain"] \
                    if (out / "manifest.json").exists() else "retail"
                policy = registry.get_env_constructor(domain)().get_policy()
                policy_windows = _think_windows(_think_norm(policy))
            public = _think_norm(json.dumps(episode.get("messages") or [],
                                            ensure_ascii=False))
            visible = _think_windows(public) | policy_windows
            hidden = [_think_norm(h) for h in _hidden_task_strings(by_task[episode["task_id"]])]
            windows = set().union(*(_think_windows(h) for h in hidden)) - visible
            short = {h for h in hidden if THINK_MIN_SECRET <= len(h) < THINK_WINDOW
                     and f" {h} " not in f" {public} "}
            secrets[eid] = (windows, short)
        return secrets[eid]

    for call in read_jsonl_tolerant(out / "calls.jsonl")[0]:
        execution = call.get("execution_id")
        episode = final.get(execution) if execution else None
        if call.get("role") != "agent" or call.get("parse_ok") is not True or episode is None:
            continue
        windows, short = secrets_for(episode)
        for index, tool_call in enumerate(call.get("raw_tool_calls") or []):
            function = (tool_call or {}).get("function") or {}
            if function.get("name") != "think":
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except ValueError:
                summary["unparseable"] += 1
                continue
            text = arguments.get("thought") if isinstance(arguments, dict) else None
            if not isinstance(text, str) or not text.strip():
                continue
            reason = _thought_withheld(text, windows, short)
            if reason is not None:
                summary["withheld"][reason] = summary["withheld"].get(reason, 0) + 1
                continue
            record = {"episode_id": episode["episode_id"], "execution_id": execution,
                      "turn": call.get("turn") if isinstance(call.get("turn"), int) else None,
                      "index": index, "content": text, "sanitized": True}
            check_no_grading_fields(record)
            rows.append(record)
    summary["thoughts"] = len(rows)
    summary["episodes_with_thoughts"] = len({r["episode_id"] for r in rows})
    summary["withheld_total"] = sum(summary["withheld"].values())
    _write_jsonl_atomic(out / "think_public.jsonl", rows)
    return summary


def _check_and_write(episodes_path: Path, outcomes_path: Path, episode: dict[str, Any],
                     outcome: dict[str, Any], windows: frozenset[str]) -> None:
    """Leak checks on metadata, then append the episode and its outcome (fsynced).

    The conversation itself is not scanned: it legitimately carries what the
    user says. Errors, metadata and outcomes must never carry task text.
    """
    check_no_grading_fields(episode)
    check_no_task_text(episode, windows, skip_prefixes=CONVERSATION_PATHS)
    check_no_task_text(outcome, windows)
    append_jsonl(episodes_path, episode)
    append_jsonl(outcomes_path, outcome)


# --------------------------------------------------------------------------- parallel


def _try_lock(path: Path) -> Any:
    """An exclusive, non-blocking fcntl lock held until the handle is closed (or
    the process dies). Returns the open handle, or None if another holds it."""
    handle = path.open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _worker_dirs(out: Path) -> list[Path]:
    def key(path: Path) -> tuple[int, int]:
        session, worker = path.name[1:].split("-w")
        return int(session), int(worker)

    root = out / "workers"
    return sorted((p for p in root.glob("s*-w*") if p.is_dir()), key=key) if root.exists() \
        else []


def _check_no_live_workers(out: Path) -> None:
    for wdir in _worker_dirs(out):
        handle = _try_lock(wdir / "worker.lock")
        if handle is None:
            raise RuntimeError(f"worker {wdir.name} of this run is still running; "
                               "refusing to start another session")
        handle.close()


def _merge(out: Path, manifest: dict[str, Any], tasks: list[Any]) -> set[str]:
    """Rebuild the run's top-level files from every worker's files.

    Deterministic: episodes and outcomes in the sequential run order
    (trial-major, manifest task order), whatever the scheduling; calls, errors
    and private records in (session, worker, line) order. Torn tails and
    unpaired lines of each worker are quarantined first, as on resume.
    """
    private = private_dir(out)
    quarantine = private / "quarantine.jsonl"
    order = {episode_id_for(t.id, trial): i
             for i, (trial, _, t) in enumerate(_canonical_pairs(manifest, tasks))}
    episodes: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    logs: dict[str, list[dict[str, Any]]] = {"calls": [], "private_calls": [], "errors": [],
                                             "episode_blobs": [], "think": []}
    if (out / "coordinator" / "calls.jsonl").exists():  # the capture probe, if any
        logs["calls"] += _drop_torn_tail(out / "coordinator" / "calls.jsonl", quarantine,
                                         "coordinator")
    for wdir in _worker_dirs(out):
        _repair(wdir, quarantine, wdir.name)
        for name, target in (("episodes.jsonl", episodes), ("outcomes.jsonl", outcomes)):
            for record in read_jsonl_tolerant(wdir / name)[0]:
                if record["episode_id"] in target:
                    raise RuntimeError(f"episode {record['episode_id']} was written by two "
                                       "workers; refusing to merge")
                target[record["episode_id"]] = record
        pdir = private / "workers" / wdir.name
        for key, path in (("calls", wdir / "calls.jsonl"),
                          ("private_calls", pdir / "calls.jsonl"),
                          ("errors", pdir / "errors.jsonl"),
                          ("episode_blobs", pdir / "episode_blobs.jsonl"),
                          ("think", pdir / "think.jsonl")):
            if path.exists():
                logs[key] += _drop_torn_tail(path, quarantine, wdir.name)
    ids = sorted(episodes, key=lambda e: order.get(e, len(order)))
    _write_jsonl_atomic(out / "episodes.jsonl", [episodes[e] for e in ids])
    _write_jsonl_atomic(out / "outcomes.jsonl", [outcomes[e] for e in ids])
    _write_jsonl_atomic(out / "calls.jsonl", logs["calls"])
    _write_jsonl_atomic(private / "calls.jsonl", logs["private_calls"])
    _write_jsonl_atomic(private / "errors.jsonl", logs["errors"])
    _write_jsonl_atomic(private / "episode_blobs.jsonl", logs["episode_blobs"])
    if logs["think"]:
        _write_jsonl_atomic(private / "think.jsonl", logs["think"])
    return set(ids)


def _worker_status(wdir: Path) -> dict[str, Any] | None:
    path = wdir / "status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _parallel_spend(out: Path, ledger: SharedLedger) -> dict[str, Any]:
    """Run spend for the manifest: dollars from the shared ledger (authoritative,
    includes unsettled reservations), call and token counts from the workers."""
    snap = ledger.snapshot()
    by_role = {role: RoleTally() for role in ("agent", "user", "judge")}
    extra: dict[str, list[Any]] = {"refused_calls": [], "overspends": [],
                                   "price_table_underestimates": [], "rate_limit_pauses": []}
    for wdir in _worker_dirs(out):
        spend = (_worker_status(wdir) or {}).get("spend") or {}
        for role, values in (spend.get("by_role") or {}).items():
            tally = by_role.setdefault(role, RoleTally())
            for name in ("calls", "failed_attempts", "input_tokens", "output_tokens",
                         "cached_tokens"):
                setattr(tally, name, getattr(tally, name) + int(values.get(name, 0)))
        for name in extra:
            extra[name] += [{**e, "worker": wdir.name} for e in spend.get(name) or []]
    for role, tally in by_role.items():
        tally.usd = snap["charged_by_role"].get(role, 0.0)
    return {
        "cap_usd": snap["cap_usd"],
        "spent_usd": round(snap["committed_usd"], 9),
        "by_role": {role: vars(t) for role, t in by_role.items()},
        "pending": None if not snap["pending"] else snap["pending"],
        **extra,
        "unreconciled_reservations": snap["unreconciled"],
        "ledger": {"charged_usd": snap["charged_usd"], "pending_usd": snap["pending_usd"],
                   "halted": snap["halted"], "refusals": snap["refusals"]},
    }


def _execute_parallel(out: Path, manifest: dict[str, Any], config: RunConfig,
                      tasks: list[Any], state: HarnessState, workers: int,
                      worker_init: Any) -> dict[str, Any]:
    """Coordinate one session of a parallel run.

    1. Refuse if another coordinator or any earlier worker is still alive
       (fcntl locks), then charge in full every ledger reservation left
       unsettled by a dead worker.
    2. Merge earlier sessions' worker files; split the missing (task, trial)
       pairs round-robin, in the sequential order, into ``workers`` disjoint
       subsets, record the assignment, and start one spawned process each.
    3. Each worker writes its own files under workers/<sN-wK>/ (private parts
       under private/workers/<sN-wK>/) and authorizes every paid attempt
       through the shared ledger, so the cap holds across workers.
    4. When all workers have exited, reconcile again, merge deterministically
       into the run's top-level files and set the run status.
    """
    run_lock = _try_lock(out / "run.lock")
    if run_lock is None:
        raise RuntimeError("another process is running this run; refusing")
    try:
        _check_no_live_workers(out)
        private = private_dir(out)
        session = len(manifest["sessions"])
        resuming = session > 0
        # Never start a resumed run from a fresh ledger: that would reset its spend.
        ledger = SharedLedger(out / "ledger.jsonl", config.cap_usd, owner=f"s{session}-coord",
                              session=session, quarantine=private / "quarantine.jsonl",
                              create=not resuming)
        if resuming and ledger.snapshot() and ledger.sessions == 0:
            raise RuntimeError("the spend ledger holds no session record although the run "
                               "has earlier sessions; refusing to resume with unknown spend")
        ledger.reconcile()
        recorded = (manifest.get("spend") or {}).get("spent_usd") or 0.0
        ledger.open_session(float(recorded))
        pin = None
        needs_probe = not (manifest.get("capture") or {}).get("agent")
        probe_meter = make_meter(config, ledger)
        coord = out / "coordinator"
        if needs_probe and config.agent.backend == "openai_compatible" and \
                config.agent.capture == "auto":
            coord.mkdir(exist_ok=True)  # the probe's call record goes here
        try:
            # Pin this session's serving window before any request (workers enforce it).
            pin = pin_serving(config)
            if needs_probe:
                capture_plan(manifest, config, probe_meter,
                             CallRecorder(coord / "calls.jsonl", private / "calls.jsonl",
                                          session=session, worker=f"s{session}-coord"),
                             pin=pin)
        except ServingWindowClosed as exc:
            done = _merge(out, manifest, tasks)
            manifest.update(
                status="aborted", finished_utc=_now(), episodes_written=len(done),
                invalid_episodes=sum(1 for o in read_jsonl_tolerant(
                    out / "outcomes.jsonl")[0] if o.get("valid") is False),
                abort_reason=_record_error(private, "probe", "capture_probe", exc,
                                           "serving_window_closed"),
                spend=_parallel_spend(out, ledger),
                not_run=[episode_id_for(t.id, trial)
                         for trial, _, t in _canonical_pairs(manifest, tasks)
                         if episode_id_for(t.id, trial) not in done])
            _write_manifest(out, manifest)
            return manifest
        if needs_probe:
            if "probe" in manifest["capture"]["agent"] and "probe_spend" not in manifest[
                    "capture"]:
                manifest["capture"]["probe_spend"] = probe_meter.summary()
            _write_manifest(out, manifest)
        done = _merge(out, manifest, tasks)
        missing = [episode_id_for(t.id, trial)
                   for trial, _, t in _canonical_pairs(manifest, tasks)
                   if episode_id_for(t.id, trial) not in done]
        subsets = [s for s in (missing[k::workers] for k in range(workers)) if s]
        entries = [{"worker": f"s{session}-w{k}", "episodes": subset, "status": "starting",
                    "exitcode": None} for k, subset in enumerate(subsets)]
        repo_sha, repo_dirty = _git_sha(REPO_ROOT)
        manifest["sessions"].append({"started_utc": _now(), "repo_git_sha": repo_sha,
                                     "repo_dirty": repo_dirty, "already_done": len(done),
                                     "workers": entries, "serving_pin": pin,
                                     **_serving_identity(config)})
        manifest.update(status="running", workers=workers,
                        assignment={e["worker"]: e["episodes"] for e in entries})
        manifest.pop("abort_reason", None)
        manifest["spend"] = _parallel_spend(out, ledger)
        _write_manifest(out, manifest)

        context = multiprocessing.get_context("spawn")
        processes: list[Any] = []
        interrupted: BaseException | None = None
        started_all = False
        try:
            for k, entry in enumerate(entries):
                process = context.Process(target=worker_entry, name=entry["worker"],
                                          args=(str(out), session, k, entry["episodes"],
                                                worker_init))
                process.start()
                processes.append(process)
            started_all = True
            last = time.monotonic()
            reconciled: set[str] = set()
            while any(p.is_alive() for p in processes):
                for process in processes:
                    process.join(timeout=0.5)
                    if not process.is_alive() and process.name not in reconciled:
                        # A dead worker's in-flight reservation is charged in full
                        # now, so live workers waiting for room are not blocked.
                        reconciled.add(process.name)
                        ledger.reconcile({process.name})
                if time.monotonic() - last > PARALLEL_CHECKPOINT_S:
                    last = time.monotonic()
                    manifest["spend"] = _parallel_spend(out, ledger)
                    _write_manifest(out, manifest)
        except BaseException as exc:  # noqa: BLE001 - finalize, then re-raise
            interrupted = exc
            if not started_all:
                # A worker failed to start: stop the ones already running at once.
                ledger.halt("worker_start_failed")
            for process in processes:
                process.join(timeout=WORKER_GRACE_S if started_all else 0)
                if process.is_alive():
                    process.terminate()
                    process.join()
        # Every worker of this session has exited: whatever it left unsettled
        # is charged in full.
        ledger.reconcile()
        done = _merge(out, manifest, tasks)
        statuses = []
        for index, entry in enumerate(entries):
            if index >= len(processes):
                entry.update(status="not_started", exitcode=None, episodes_written=0,
                             abort_reason=None)
                statuses.append(entry)
                continue
            process = processes[index]
            status = _worker_status(out / "workers" / entry["worker"]) or {}
            entry.update(exitcode=process.exitcode,
                         status=status.get("status") if status.get("status") not in
                         (None, "running") else "crashed",
                         episodes_written=status.get("episodes_written", 0),
                         abort_reason=status.get("abort_reason"))
            statuses.append(entry)
        outcomes = read_jsonl_tolerant(out / "outcomes.jsonl")[0]
        invalid = sum(1 for o in outcomes if o.get("valid") is False)
        failed = [e for e in statuses if e["status"] == "failed"]
        crashed = [e for e in statuses if e["status"] in ("crashed", "interrupted")]
        if interrupted is not None:
            status = "interrupted"
            reason = None if started_all else {
                "type": type(interrupted).__name__, "code": "worker_start_failed",
                "sha256": _sha256_bytes(str(interrupted).encode())}
        elif failed:
            status, reason = "failed", failed[0]["abort_reason"]
        elif any(e["status"] == "aborted_overspend" for e in statuses):
            status = "aborted_overspend"
            reason = next(e["abort_reason"] for e in statuses
                          if e["status"] == "aborted_overspend")
        elif any(e["status"] == "aborted" for e in statuses):
            status = "aborted"
            reason = next(e["abort_reason"] for e in statuses if e["status"] == "aborted")
        elif crashed:
            status = "aborted"
            reason = {"type": "WorkerExit", "code": "worker_crash",
                      "sha256": _sha256_bytes(crashed[0]["worker"].encode())}
        else:
            status, reason = ("complete_with_errors" if invalid else "complete"), None
        if reason:
            manifest["abort_reason"] = reason
        manifest.update(status=status, finished_utc=_now(), invalid_episodes=invalid,
                        episodes_written=len(done), spend=_parallel_spend(out, ledger),
                        not_run=[episode_id_for(t.id, trial)
                                 for trial, _, t in _canonical_pairs(manifest, tasks)
                                 if episode_id_for(t.id, trial) not in done])
        _run_capture_level(out, manifest)
        if config.arm == "think":
            manifest["think_projection"] = write_think_projection(out, tasks)
        _write_manifest(out, manifest)
        if interrupted is not None:
            raise interrupted
        if failed:
            raise (GradingLeak if failed[0]["abort_reason"].get("type") == "GradingLeak"
                   else UnmeteredCall if failed[0]["abort_reason"].get("type") ==
                   "UnmeteredCall" else RuntimeError)(
                f"worker {failed[0]['worker']} failed: {failed[0]['abort_reason'].get('code')}"
                " (details in private/)")
        return manifest
    finally:
        run_lock.close()


def _worker_main(out_str: str, session: int, index: int, episode_ids: list[str],
                 worker_init: Any) -> None:
    """One worker process: run the assigned episodes under the shared ledger."""
    if worker_init is not None:
        worker_init()
    out = Path(out_str)
    wid = f"s{session}-w{index}"
    wdir = out / "workers" / wid
    wdir.mkdir(parents=True, exist_ok=False)
    lock = _try_lock(wdir / "worker.lock")
    if lock is None:
        raise SystemExit(3)
    pdir = private_dir(out) / "workers" / wid
    pdir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {"worker": wid, "pid": os.getpid(), "status": "running",
                              "started_utc": _now(), "episodes": episode_ids,
                              "episodes_written": 0, "invalid_episodes": 0}
    for name in ("episodes.jsonl", "outcomes.jsonl", "calls.jsonl"):
        (wdir / name).touch()

    def write_status() -> None:
        _write_json_atomic(wdir / "status.json", status)

    write_status()
    try:
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        config = load_config(manifest["config_path"])
        if manifest.get("cap_override"):
            config = replace(config, cap_usd=float(manifest["budget_cap_usd"]))
        if config.resolved_sha256() != manifest["config_resolved_sha256"]:
            raise RuntimeError("config or prices changed since the run started")
        state_dir = manifest.get("state_dir")
        state = HarnessState.load(state_dir) if state_dir else HarnessState.empty()
        if state.version_id != manifest["harness_version_id"]:
            raise RuntimeError("harness state differs from the run's")
        check_state_allowed(state, config)
        wanted = set(episode_ids)
        tasks = load_tasks(manifest["domain"], manifest["task_ids"])
        pairs = [p for p in _canonical_pairs(manifest, tasks)
                 if episode_id_for(p[2].id, p[0]) in wanted]
        ledger = SharedLedger(out / "ledger.jsonl", config.cap_usd, owner=wid, create=False,
                              session=session,
                              quarantine=private_dir(out) / "quarantine.jsonl")
        meter = make_meter(config, ledger)
    except Exception as exc:  # noqa: BLE001 - recorded as a failed worker setup
        status.update(status="aborted", finished_utc=_now(),
                      abort_reason=_record_error(pdir, "-", "worker_setup", exc,
                                                 "worker_setup_failed"))
        write_status()
        raise SystemExit(1) from None

    done: set[str] = set()

    def checkpoint() -> None:
        status.update(spend=meter.summary(), episodes_written=len(done))
        write_status()

    def on_episode(outcome: dict[str, Any]) -> None:
        status["invalid_episodes"] += not outcome["valid"]
        checkpoint()

    meter.on_change = lambda _meter: checkpoint()
    recorder = CallRecorder(wdir / "calls.jsonl", pdir / "calls.jsonl", session=session,
                            worker=wid, token_store=TokenStore(out / "tokens"))
    level = ((manifest.get("capture") or {}).get("agent") or {}).get("request_level",
                                                                      "messages")
    # The coordinator pinned this session's serving window; workers enforce it.
    pin = (manifest["sessions"][session] if len(manifest["sessions"]) > session
           else {}).get("serving_pin")
    try:
        result, reason = _run_pairs(manifest["domain"], config, state, pairs, meter,
                                    _sink(out, wdir, pdir), recorder, done, on_episode,
                                    agent_level=level, serving_pin=pin)
    except (GradingLeak, UnmeteredCall) as exc:
        ledger.halt("fatal")
        status.update(status="failed", finished_utc=_now(),
                      abort_reason=_record_error(pdir, "-", "run", exc, "fatal"))
        checkpoint()
        raise SystemExit(2) from None
    except BaseException as exc:
        status.update(status="interrupted", finished_utc=_now(),
                      abort_reason={"type": type(exc).__name__, "code": "interrupted",
                                    "sha256": _sha256_bytes(str(exc).encode())})
        checkpoint()
        raise
    if result is None:
        result = "complete_with_errors" if status["invalid_episodes"] else "complete"
    status.update(status=result, abort_reason=reason, finished_utc=_now())
    checkpoint()
    lock.close()


# --------------------------------------------------------------------------- replay


@dataclass
class ReplayResult:
    episode_id: str
    task_id: str
    stored_reward: float
    replayed_reward: float | None
    stored_breakdown: dict[str, float]
    replayed_breakdown: dict[str, float] | None
    note: str = ""
    official_judge_called: bool = False

    @property
    def match(self) -> bool:
        return (self.replayed_reward is not None
                and abs(self.replayed_reward - self.stored_reward) < 1e-9
                and self.replayed_breakdown == self.stored_breakdown)

    @property
    def kind(self) -> str:
        """match; judge_nondeterministic (only the official judge's component
        differs); or mismatch (a deterministic difference, or not replayed)."""
        if self.match:
            return "match"
        if self.replayed_breakdown is not None and self.official_judge_called:
            keys = set(self.stored_breakdown) | set(self.replayed_breakdown)
            differing = {k for k in keys
                         if self.stored_breakdown.get(k) != self.replayed_breakdown.get(k)}
            if differing == {"NL_ASSERTION"}:
                return "judge_nondeterministic"
        return "mismatch"


def replay_judge_mode(run_dir: str | Path) -> str:
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    return manifest.get("judge_mode", "official")


def replay(run_dir: str | Path, episode_id: str | None = None,
           cap_usd: float | None = None) -> list[ReplayResult]:
    """Re-grade stored episodes with tau2's evaluator against their tasks.

    Uses the run's recorded judge mode and snapshotted price table, and refuses
    if tau2's code or data differ from what the run used.
    """
    from tau2.data_model.simulation import SimulationRun, TerminationReason

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if check_tau2_pinned() != manifest.get("tau2_commit"):
        raise RuntimeError("tau2 checkout differs from the run's")
    if data_hashes(manifest["domain"]) != manifest.get("data_sha256"):
        raise RuntimeError("tau2 data files differ from the ones the run used")
    episodes, _ = read_jsonl_tolerant(run_dir / "episodes.jsonl")
    outcomes = {o["episode_id"]: o
                for o in read_jsonl_tolerant(run_dir / "outcomes.jsonl")[0]}
    episodes = [e for e in episodes if e["episode_id"] in outcomes]
    if episode_id is not None:
        episodes = [e for e in episodes if e["episode_id"] == episode_id]
        if not episodes:
            raise KeyError(f"no episode {episode_id!r} in {run_dir}")
    domain = manifest["domain"]
    tasks = {t.id: t for t in load_tasks(domain, sorted({e["task_id"] for e in episodes}))}
    cap = manifest.get("budget_cap_usd", 0.0) if cap_usd is None else cap_usd
    meter = Meter(cap, load_prices(run_dir / "snapshot" / "prices.yaml"))
    mode = manifest.get("judge_mode", "official")
    stub_met = (manifest.get("grader") or {}).get("stub_judge_met")
    results = []
    with isolated_logging(private_dir(run_dir), "replay-tau2.log"), completion_guard(), \
            metered_grader(meter, mode, True if stub_met is None else bool(stub_met)):
        for episode in episodes:
            outcome = outcomes[episode["episode_id"]]
            result = ReplayResult(episode["episode_id"], episode["task_id"],
                                  float(outcome["reward"]), None,
                                  outcome.get("reward_breakdown") or {}, None)
            if episode["termination_reason"] == "error" or outcome.get("valid") is False:
                # Invalid episodes are not benchmark results; there is nothing to grade.
                result.replayed_reward = result.stored_reward
                result.replayed_breakdown = result.stored_breakdown
                result.note = "invalid episode: not re-graded"
                results.append(result)
                continue
            simulation = SimulationRun(
                id=episode["episode_id"], task_id=episode["task_id"], start_time="",
                end_time="", duration=float(episode["wall_s"]),
                termination_reason=TerminationReason(episode["termination_reason"]),
                messages=records_to_messages(episode["messages"]), seed=episode.get("seed"))
            before = meter.snapshot()
            try:
                info = evaluate(simulation, tasks[episode["task_id"]], domain)
                result.replayed_reward = float(info.reward)
                result.replayed_breakdown = breakdown_dict(info)
            except BudgetExceeded:
                result.note = "not replayed: judge call refused by the budget cap"
            judged = delta(before, meter.snapshot())["judge"]
            if judged.calls or judged.failed_attempts:
                result.note = result.note or f"judge: {mode}"
                result.official_judge_called = mode == "official"
            results.append(result)
    return results
