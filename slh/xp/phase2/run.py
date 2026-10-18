"""Phase-2 conversations: arms x runs x scenarios through tau2, resumable.

    python -m slh.xp.phase2.run dry-run --package PKG --endpoint endpoint.yaml --out runs/p2-dry
    python -m slh.xp.phase2.run smoke   --package PKG --endpoint endpoint.yaml --out runs/p2-smoke
    python -m slh.xp.phase2.run full    --package PKG --endpoint endpoint.yaml --out runs/p2
    python -m slh.xp.phase2.run status  --out runs/p2
    python -m slh.xp.phase2.run check-servers --endpoint endpoint.yaml

The same code runs wherever the two models are served: the endpoint file names
an OpenAI-compatible base URL for the 9B agent and one for the user simulator
(two local vLLM servers, or a self-served agent session plus a hosted
simulator config on our side). Nothing is graded where the run happens: every
conversation runs as a tau2 task without evaluation criteria, and the export is
scored offline by replay (``slh.xp.phase2.check``).

Modes. ``dry-run``: no GPU and no server; validates the package (file hashes,
code hash, benchmark data), the endpoint file, every arm's rendered prompt,
the job plan and its counts, then runs the smoke scenarios end to end with
deterministic offline stand-ins for both models (``slh.stub``) and exports.
``smoke``: the SMOKE-ONLY scenarios (never test scenarios), every arm, run 0.
``smoke`` and ``full`` first check that both servers answer with the configured
names and send one real chat completion to each, which must come back as a
plain (non-thinking) answer (``check-servers`` runs that step alone).
``full``: the test split, every arm on its runs. Each mode has its own run
directory; a directory started under another identity is refused.

Jobs. A job is one conversation, identified by (scenario, arm, run). Jobs are
dispatched in (scenario, run) blocks that hold every arm of that block in a
seeded order (blocks in a seeded order too), so drift hits every arm alike.
The simulator seed is ``phase2.simulator_seed(plan_id, scenario, run)``: the
same in every arm of a block (paired). Arm states come from the package's
bundles (``slh.xp.phase2.bundles``): B1 is the empty state; the other arms
load their run's notes, bound to this run's base arm.

Resume by identity. Every conversation is written by the existing episode loop
(``slh.runner._run_pairs``: leak checks on every record, metered calls,
serving-window gate) into the worker's own directory,
``workers/<sN-wK>/arms/<arm>/{episodes,outcomes}.jsonl``. On every start the
coordinator reads all of them: a job is done once it has a valid record; an
invalid one (crash, model error) is rerun, at most ``MAX_ATTEMPTS`` times in
all; records written while the run halted on a systematic error do not count
as attempts. Rerun the same command after any interruption.

Deadlines (hang fix, 2026-10-17). The agent's output is capped
(``AGENT_MAX_TOKENS``, endpoint ``agent.max_tokens``); each conversation has a
deadline (the plan's ``conversation_time_limit_s``) that the meter and every
attempt's network timeout (``min(ATTEMPT_TIMEOUT_S, remaining)``) obey; an
identical deterministic request that timed out is sent at most twice; a
conversation that runs out of time ends INVALID (an exception, or tau2's TIMEOUT
termination). All recorded in the run identity (section "deadline" below).

Halts. The spend ledger (``slh.ledger``) holds the cap across workers and is
the cross-process halt latch: a budget refusal, a systematic error (connection,
authentication, credit, configuration: ``slh.xp.partner.errors.classify``),
``CONSECUTIVE_INVALID`` invalid conversations in a row in one worker, or a
closed serving window stops every worker at its next conversation. Exit 3 =
halted (``halt.json`` names the cause), 4 = refused, 0 = complete.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from slh.xp import phase2 as P
from slh.xp.partner import common as pc
from slh.xp.phase2 import bundles as B
from slh.xp.phase2 import package as PK

RUN_SCHEMA = "xp-phase2-run-v1"
DOMAIN = "retail"
MODES = ("dry", "smoke", "full")
SPLIT_OF_MODE = {"dry": "smoke", "smoke": "smoke", "full": "test"}
SMOKE_RUNS = (0,)
MAX_WORKERS = 128
MAX_ATTEMPTS = 2  # an invalid conversation is rerun once; then its last record is final
CONSECUTIVE_INVALID = 3
PASSES = MAX_ATTEMPTS  # coordinator rounds per session (reruns of invalid jobs)
EXIT_OK, EXIT_HALTED, EXIT_REFUSED = 0, 3, 4
# Durable spend high-water mark beside the ledger: one fsynced line after every
# reservation and settlement; resume refuses a ledger that shows less.
SPEND_MARK = "spend.jsonl"
ENDPOINT_VOLATILE = ("workers",)  # scheduling only, not part of the run identity
B1_AGENT = "configs/agent-vllm.yaml"
PRICES = "configs/prices.yaml"
# Agent output cap (hang fix 2026-10-17): the B1 agent config sends no max_tokens, and
# a degenerate deterministic generation then ran until the request timed out, again
# on every retry. Phase 2 caps it (endpoint ``agent.max_tokens``; recorded in the
# run identity). Terminations that count as invalid in phase 2 (``records``).
AGENT_MAX_TOKENS = 4096
INVALID_TERMINATIONS = ("timeout",)


class Refused(ValueError):
    """The inputs, the code or the run directory do not match: nothing was run."""


class Halted(RuntimeError):
    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__(f"{record.get('label')}: {record.get('detail', '')}")
        self.record = record


# --------------------------------------------------------------------------- jobs


@dataclasses.dataclass(frozen=True)
class Job:
    scenario_id: str
    arm: str
    kind: str
    run: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.scenario_id}|{self.arm}|{self.run}"

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _rank(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def subset(plan: P.Phase2Plan, rows: list[dict[str, Any]], n: int) -> set[str]:
    """A diagnostic arm's scenarios: the first ``n`` generated scenarios per type in
    a seeded order (by plan id), fixed before anything runs."""
    out: set[str] = set()
    for t in P.DECISION_TYPES:
        ids = sorted((r["scenario_id"] for r in rows
                      if r["source"] == "generated" and r["decision_type"] == t),
                     key=lambda s: _rank(plan.plan_id, "subset", s))
        out.update(ids[:n])
    return out


def plan_jobs(plan: P.Phase2Plan, rows: list[dict[str, Any]], split: str
              ) -> list[list[Job]]:
    """Blocks of jobs in dispatch order. Test split: every arm on its runs (diagnostic
    arms on their subset; the official shard on the arms and runs the plan names).
    Smoke split: every arm on run 0 of every smoke scenario."""
    blocks: dict[tuple[str, int], list[Job]] = {}
    official_runs = plan.notes.get("official_runs")
    # G1 (docs/xp/g1-plan.md): scenarios listed here run on each arm's FIRST run only
    # (the non-trap set: one seed; the trap set: every run). Absent in phase 2.
    one_run = set(plan.notes.get("one_run_scenarios") or ())
    for arm in plan.arms:
        chosen = subset(plan, rows, arm.subset_per_type) \
            if split == "test" and arm.subset_per_type is not None else None
        for row in rows:
            sid = row["scenario_id"]
            if split == "smoke":
                runs = [r for r in arm.runs if r in SMOKE_RUNS] or list(arm.runs[:1])
            elif row["source"] == "official_test":
                if not arm.official:
                    continue
                runs = list(arm.runs if official_runs is None else arm.runs[:official_runs])
            else:
                if chosen is not None and sid not in chosen:
                    continue
                runs = list(arm.runs[:1]) if sid in one_run else list(arm.runs)
            for run in runs:
                for _ in range(arm.trials_per_run):
                    blocks.setdefault((sid, run), []).append(
                        Job(sid, arm.name, arm.kind, run, P.simulator_seed(plan.plan_id, sid,
                                                                           run)))
    out = []
    for (sid, run), jobs in sorted(blocks.items(),
                                   key=lambda kv: _rank(plan.plan_id, "blocks", *kv[0])):
        jobs.sort(key=lambda j: _rank(plan.plan_id, "arms", sid, run, j.arm))
        if len({j.key for j in jobs}) != len(jobs):
            raise Refused(f"block ({sid}, {run}): an arm repeats (trials_per_run > 1 is not "
                          "supported in phase 2)")
        out.append(jobs)
    return out


def smoke_test_overlap(pkg: Any) -> list[str]:
    """Smoke scenarios whose simulator instructions equal a test scenario's (by
    content fingerprint, whatever the ids)."""
    test = {P.content_fingerprint(r["user_scenario"]) for r in pkg.scenarios("test")}
    return sorted(r["scenario_id"] for r in pkg.scenarios("smoke")
                  if P.content_fingerprint(r["user_scenario"]) in test)


def flat(blocks: list[list[Job]]) -> list[Job]:
    return [j for b in blocks for j in b]


# --------------------------------------------------------------------------- endpoint


def load_endpoint(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    problems = endpoint_problems(data)
    if problems:
        raise Refused(f"{path}: {'; '.join(problems)}")
    return data


def endpoint_problems(ep: dict[str, Any]) -> list[str]:
    out = []
    for role in ("agent", "user"):
        part = ep.get(role)
        if not isinstance(part, dict):
            out.append(f"{role}: missing")
            continue
        if part.get("stub"):
            continue
        if role == "user" and part.get("config"):
            continue  # a hosted simulator config file (our side)
        if not part.get("model"):
            out.append(f"{role}: model (the served model name) is required")
        if not (part.get("base_url") or (role == "agent" and part.get("serving_session"))):
            out.append(f"{role}: base_url is required")
        url = part.get("base_url")
        if url and not str(url).startswith(("http://", "https://")):
            out.append(f"{role}: base_url must start with http:// or https://")
        out += [f"{role}: {p}" for p in pricing_problems(part)]
    workers = ep.get("workers", 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or \
            not 1 <= workers <= MAX_WORKERS:
        out.append(f"workers must be an integer from 1 to {MAX_WORKERS}")
    return out


PRICE_FIELDS = ("input_per_mtok", "output_per_mtok", "cached_input_per_mtok")


def pricing_problems(part: dict[str, Any]) -> list[str]:
    """A served model needs its price declared: ``self_hosted: true`` (our own or the
    partner's server: $0 per token, GPU time metered elsewhere) or ``prices``
    (USD per million tokens). A model in the price table needs neither; an
    undeclared unknown model is refused (never priced at $0 by default)."""
    out = []
    sh = part.get("self_hosted")
    if sh is not None and not isinstance(sh, bool):
        out.append("self_hosted must be true or false")
    prices = part.get("prices")
    if prices is not None:
        if sh:
            out.append("self_hosted and prices are exclusive")
        if not isinstance(prices, dict) or any(
                not isinstance(prices.get(k), (int, float)) or isinstance(prices.get(k), bool)
                or not 0 <= float(prices[k]) < 1e6
                for k in ("input_per_mtok", "output_per_mtok")):
            out.append("prices needs input_per_mtok and output_per_mtok (USD per million "
                       "tokens, finite, >= 0)")
        elif set(prices) - set(PRICE_FIELDS):
            out.append(f"prices: unknown fields {sorted(set(prices) - set(PRICE_FIELDS))}")
        elif "cached_input_per_mtok" in prices:
            c = prices["cached_input_per_mtok"]
            if not isinstance(c, (int, float)) or isinstance(c, bool) or not 0 <= float(c) < 1e6:
                out.append("prices: cached_input_per_mtok must be finite and >= 0")
            elif float(c) > float(prices["input_per_mtok"]):
                out.append("prices: cached_input_per_mtok is above input_per_mtok (a cached "
                           "token never costs more than an uncached one; refused)")
    return out


def cached_price_problem(model: str, entry: dict[str, Any]) -> str | None:
    """A price entry whose cached-input price exceeds its input price is refused
    (reservations and admission bound input at the input price)."""
    try:
        inp = float(entry.get("input_per_mtok", 0.0))
        cached = float(entry.get("cached_input_per_mtok", inp))
    except (TypeError, ValueError):
        return f"{model!r}: unreadable price entry"
    if cached > inp:
        return (f"{model!r}: cached_input_per_mtok {cached} is above input_per_mtok {inp}; "
                "refused")
    return None


def resolve_prices(table: dict[str, Any], role: str, part: dict[str, Any],
                   section: dict[str, Any]) -> dict[str, Any]:
    """The price entry the meter will use for ``section``'s model (Refused when the
    model is neither in the table nor declared)."""
    entry = _resolve_prices(table, role, part, section)
    problem = cached_price_problem(section["model"], entry)
    if problem:
        raise Refused(f"{role}: {problem}")
    return entry


def _resolve_prices(table: dict[str, Any], role: str, part: dict[str, Any],
                    section: dict[str, Any]) -> dict[str, Any]:
    model = section["model"]
    if part.get("prices"):
        entry = {k: float(part["prices"][k]) for k in PRICE_FIELDS if k in part["prices"]}
        entry.setdefault("cached_input_per_mtok", entry["input_per_mtok"])
        if model in table and {k: float(table[model].get(k, entry[k])) for k in entry} != entry:
            raise Refused(f"{role}: declared prices for {model!r} differ from the price table")
        return {**table.get(model, {}), **entry}
    if part.get("self_hosted"):
        if model in table and any(float(table[model].get(k, 0.0)) for k in PRICE_FIELDS):
            raise Refused(f"{role}: {model!r} is priced in the table; a self-hosted model "
                          "needs its own served name")
        return dict(table[model]) if model in table else {k: 0.0 for k in PRICE_FIELDS}
    if model in table:
        return dict(table[model])
    raise Refused(f"{role}: model {model!r} has no price: declare self_hosted: true (a server "
                  "you run, $0 per token) or prices (USD per million tokens)")


def stub_endpoint(workers: int = 2) -> dict[str, Any]:
    return {"agent": {"stub": True}, "user": {"stub": True}, "workers": workers}


def endpoint_sha256(ep: dict[str, Any]) -> str:
    return pc.sha_text(pc.canonical({k: v for k, v in ep.items()
                                     if k not in ENDPOINT_VOLATILE}))


def _agent_section(part: dict[str, Any], root: Path) -> dict[str, Any]:
    if part.get("stub"):
        return {"backend": "stub", "model": "stub-agent", "temperature": 0.0}
    # Same request semantics as the self-served B1 agent (non-thinking, temperature
    # 0, per-conversation seed) except an explicit output cap (AGENT_MAX_TOKENS, the
    # hang fix); only where it is served differs.
    raw = yaml.safe_load((root / B1_AGENT).read_text(encoding="utf-8"))
    for k in ("serving_session", "capture", "logprobs_mode", "api_base", "api_key_env"):
        raw.pop(k, None)
    raw["model"] = part.get("model", raw["model"])
    if part.get("serving_session"):
        raw["serving_session"] = part["serving_session"]
    else:
        raw["api_base"] = part["base_url"]
        if part.get("api_key_env"):
            raw["api_key_env"] = part["api_key_env"]
    if part.get("weights"):
        raw["checkpoint"] = part["weights"]
    raw["capture"] = "messages"
    raw["max_tokens"] = agent_max_tokens(part)
    return raw


def agent_max_tokens(part: dict[str, Any]) -> int:
    """The agent's output cap: endpoint ``agent.max_tokens`` (default 4096)."""
    value = part.get("max_tokens", AGENT_MAX_TOKENS)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Refused(f"agent.max_tokens must be a positive integer, not {value!r}")
    return value


def _user_section(part: dict[str, Any], root: Path) -> dict[str, Any]:
    if part.get("stub"):
        return {"backend": "stub", "model": "stub-user", "temperature": 0.0}
    if part.get("config"):
        path = Path(part["config"])
        return yaml.safe_load((path if path.is_absolute() else root / path).read_text(
            encoding="utf-8"))
    # A locally served simulator: the pinned simulator settings (temperature 0,
    # per-conversation seed, max_tokens 4096, no reasoning budget); the server is
    # started in non-thinking mode, so no thinking switch is sent per request.
    section = {"backend": "openai_compatible", "model": part["model"],
               "api_base": part["base_url"], "temperature": 0.0, "retries": 6,
               "max_tokens": int(part.get("max_tokens", 4096)), "reasoning_reserve_tokens": 0}
    if part.get("api_key_env"):
        section["api_key_env"] = part["api_key_env"]
    if part.get("weights"):
        section["checkpoint"] = part["weights"]
    section.update(part.get("extra") or {})
    return section


def config_files(ep: dict[str, Any], *, plan: P.Phase2Plan, cap_usd: float,
                 root: Path = pc.REPO) -> dict[str, str]:
    """The run's config files (name -> text): run.yaml, agent.yaml, user.yaml,
    prices.yaml. Every served model's price is resolved explicitly
    (``resolve_prices``): $0 per token only for a model declared ``self_hosted``."""
    from slh.arms.think import THINK_GIVE_UP_TEXT, THINK_INSTRUCTION

    agent, user = _agent_section(ep["agent"], root), _user_section(ep["user"], root)
    stub = agent["backend"] == "stub" and user["backend"] == "stub"
    prices = yaml.safe_load((root / PRICES).read_text(encoding="utf-8"))
    for role, section in (("agent", agent), ("user", user)):
        if section["backend"] == "stub":
            continue
        prices["models"][section["model"]] = resolve_prices(prices["models"], role,
                                                            ep[role], section)
    run = {"agent": "agent.yaml", "user": "user.yaml", "arm": "think",
           "think_instruction": THINK_INSTRUCTION, "think_cap_message": THINK_GIVE_UP_TEXT,
           "budget": {"cap_usd": float(cap_usd)}, "prices": "prices.yaml", "seed": 300,
           "grader": {"judge": "stub" if stub else "official"},
           "orchestrator": {"max_steps": plan.max_steps, "max_errors": 10,
                            "enforce_communication_protocol": False,
                            "timeout_s": float(plan.conversation_time_limit_s)}}

    def dump(value: Any) -> str:
        return yaml.safe_dump(value, sort_keys=True, allow_unicode=False)

    return {"run.yaml": dump(run), "agent.yaml": dump(agent), "user.yaml": dump(user),
            "prices.yaml": dump(prices)}


def required_keys(config: Any) -> list[str]:
    """API key variables the two participants need (names only; never the values).
    The judge is not called here (grading is offline)."""
    from slh.runner import api_key_for

    names = []
    for model in (config.agent, config.user):
        if model.backend == "litellm":
            names.append(api_key_for(model.model))
        elif model.backend == "openai_compatible" and model.api_key_env:
            names.append(model.api_key_env)
    return sorted(set(names))


# --------------------------------------------------------------------------- states


class _Offline:
    def __init__(self, model: Any) -> None:
        self.config = model

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("offline rendering never calls a model")


def arm_state(kind: str, run: int, bundles: dict[str, Any], config: Any) -> Any:
    from slh.execution import base_arm_fingerprint
    from slh.state import HarnessState

    notes = B.notes(bundles, kind, run)
    if not notes:
        return HarnessState.empty()
    return HarnessState(notes=[{"content": n} for n in notes]).bound_to(
        base_arm_fingerprint(config))


def prompt_sha256(config: Any, state: Any) -> str:
    """sha256 of the system prompt the agent of this arm instance is sent (the hash
    the episode loop records), rendered offline."""
    from slh.budget import Meter
    from slh.capture import sha256_json
    from slh.demo1.context import build_agent

    return sha256_json(build_agent(config, state, _Offline(config.agent), Meter(0.0, {}))
                       .system_prompt)


def arm_prompts(jobs: list[Job], bundles: dict[str, Any], config: Any) -> dict[str, Any]:
    out = {}
    for arm, kind, run in sorted({(j.arm, j.kind, j.run) for j in jobs}):
        state = arm_state(kind, run, bundles, config)
        out[f"{arm}|{run}"] = {"arm": arm, "kind": kind, "run": run,
                               "state_version": state.version_id,
                               "notes": len(state.notes),
                               "prompt_sha256": prompt_sha256(config, state)}
    return out


# --------------------------------------------------------------------------- records


def _jsonl(path: Path) -> list[dict[str, Any]]:
    from slh.trace import read_jsonl_tolerant

    return read_jsonl_tolerant(path)[0] if path.is_file() else []


def _worker_dirs(out: Path) -> list[Path]:
    from slh.runner import _worker_dirs

    return _worker_dirs(out)


def records(out: Path) -> list[dict[str, Any]]:
    """Every finished conversation record (episode and outcome lines both present),
    oldest session first. ``halted``: written while the run halted on a systematic
    error (not an attempt)."""
    rows = []
    for wdir in _worker_dirs(Path(out)):
        halted = {h["execution_id"] for h in _jsonl(wdir / "halts.jsonl")
                  if h.get("execution_id")}
        session = int(wdir.name[1:].split("-w")[0])
        for adir in sorted((wdir / "arms").glob("*")) if (wdir / "arms").is_dir() else []:
            outcomes = {o["episode_id"]: o for o in _jsonl(adir / "outcomes.jsonl")}
            for order, ep in enumerate(_jsonl(adir / "episodes.jsonl")):
                outcome = outcomes.get(ep["episode_id"])
                if outcome is None:
                    continue
                rows.append({"key": f"{ep['task_id']}|{adir.name}|{ep['trial']}",
                             "scenario_id": ep["task_id"], "arm": adir.name,
                             "run": ep["trial"], "worker": wdir.name, "session": session,
                             "order": order, "episode": ep, "outcome": outcome,
                             "valid": (outcome.get("valid") is True
                                       and ep.get("termination_reason")
                                       not in INVALID_TERMINATIONS),
                             "halted": ep.get("execution_id") in halted})
    return rows


def resolve(jobs: list[Job], rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per job key: status ``valid`` | ``invalid_final`` | ``pending``, the final
    record (first valid one; else the last invalid one once attempts are used up)
    and the attempt count. Records of keys outside ``jobs`` are reported as extra."""
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["key"], []).append(r)
    out: dict[str, dict[str, Any]] = {}
    for job in jobs:
        recs = by.get(job.key, [])
        valid = [r for r in recs if r["valid"]]
        attempts = [r for r in recs if not r["halted"]]
        if valid:
            out[job.key] = {"status": "valid", "final": valid[0], "attempts": len(attempts)}
        elif len(attempts) >= MAX_ATTEMPTS:
            out[job.key] = {"status": "invalid_final", "final": attempts[-1],
                            "attempts": len(attempts)}
        else:
            out[job.key] = {"status": "pending", "final": None, "attempts": len(attempts)}
    keys = {j.key for j in jobs}
    extra = sorted(k for k in by if k not in keys)
    if extra:
        out["__extra__"] = {"status": "extra", "keys": extra}
    return out


def coverage(resolved: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for k, v in resolved.items():
        if k != "__extra__":
            counts[v["status"]] = counts.get(v["status"], 0) + 1
    extra = resolved.get("__extra__", {}).get("keys", [])
    return {"jobs": sum(counts.values()), **{s: counts.get(s, 0) for s in
            ("valid", "invalid_final", "pending")}, "extra": len(extra),
            "complete": counts.get("pending", 0) == 0 and not extra}


# --------------------------------------------------------------------------- public text

# Public records (status.json, halt.json, worker status and halts, the ledger's halt
# reason, env.jsonl, terminal output, exports) never carry a provider's raw error
# message: only the exception type, an error code/class and a short hash of the
# message. Raw messages stay in the run's private/ directory, which is never
# exported. ``redact_text`` is a second, defensive layer over every public write.

_SECRET_ENV_NAME = re.compile(r"KEY|TOKEN|SECRET", re.IGNORECASE)
_SECRET_PATTERNS = (
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password|"
                r"authorization)(\"?\s*[=:]\s*\"?)(?!\[REDACTED\])[^\s\"',;}]+"),
     r"\1\2[REDACTED]"),
    (re.compile(r"\b(?:sk|pk|rk|ak|tk)-[A-Za-z0-9_-]{8,}"), "[REDACTED]"),
    (re.compile(r"\b(?:hf|ghp|gho|ghs|ghu|github_pat|xox[abprs])_[A-Za-z0-9_-]{12,}"),
     "[REDACTED]"),
    (re.compile(r"\b[0-9a-fA-F]{32,}\b"), "[REDACTED]"),
    # long base64-like runs with digits and both letter cases
    (re.compile(r"(?<![A-Za-z0-9+/_=-])(?=[A-Za-z0-9+/_=-]*\d)(?=[A-Za-z0-9+/_=-]*[a-z])"
                r"(?=[A-Za-z0-9+/_=-]*[A-Z])[A-Za-z0-9+/_=-]{32,}"), "[REDACTED]"),
)
# identifiers and hashes we write ourselves (never provider text)
_ID_KEY = re.compile(r"(?:^id$|_id$|_ids$|sha256$|^state_version$|^weights$|^checkpoint$|"
                     r"revision)")


def _secret_env_values() -> list[str]:
    vals = {v for k, v in os.environ.items() if _SECRET_ENV_NAME.search(k) and v
            and len(v) >= 8}
    return sorted(vals, key=len, reverse=True)


def redact_text(text: Any) -> str:
    """``text`` with every secret-shaped part masked: values of environment
    variables whose names contain KEY, TOKEN or SECRET, sk-... keys, Bearer
    tokens, key=value credentials and long hex or base64 runs."""
    out = str(text)
    for value in _secret_env_values():
        out = out.replace(value, "[REDACTED]")
    for pattern, repl in _SECRET_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def redact_public(value: Any, key: str | None = None) -> Any:
    """A copy of ``value`` safe for a public record: every string redacted
    (``redact_text``); our own ids and hashes only lose known secret values."""
    if isinstance(value, dict):
        return {k: redact_public(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_public(v, key) for v in value]
    if isinstance(value, str):
        if key is not None and _ID_KEY.search(key):
            for secret in _secret_env_values():
                value = value.replace(secret, "[REDACTED]")
            return value
        return redact_text(value)
    return value


_HTTP_STATUS = re.compile(r"\b(?:HTTP(?: Error)?|status)[ :]*([1-5]\d\d)\b", re.IGNORECASE)


def public_error(type_name: str, message: str, code: str | None = None) -> str:
    """The public form of an error: type, code/class (or HTTP status) and a short
    hash of the raw message; never the message itself."""
    digest = hashlib.sha256(str(message).encode("utf-8", "replace")).hexdigest()[:12]
    if code is None:
        m = _HTTP_STATUS.search(str(message))
        code = f"http_{m.group(1)}" if m else None
    name = re.sub(r"[^A-Za-z0-9_.]", "", str(type_name))[:80] or "Error"
    return f"{name}{f' [{code}]' if code else ''} (message hash {digest})"


def public_exception(exc: BaseException) -> str:
    return public_error(type(exc).__name__, str(exc))


# --------------------------------------------------------------------------- deadline
#
# Per-conversation deadline for phase-2 conversations (the pilot hang fix, 2026-10-17).
#
# The pilot showed a conversation that never ended: tau2's orchestrator checks its
# wall-clock limit only BETWEEN steps, while one agent request could block for
# ``openai_compat.TIMEOUT_S`` (600 s) and the meter retried a timed-out request up
# to ``retries`` times (and paused after a streak), all with the same deterministic
# body (temperature 0, fixed seed), which times out the same way again.
#
# What this module adds, for the phase-2 driver only (``slh.xp.phase2.run``):
#
# * ``conversation_deadline(limit_s)``: the deadline of ONE conversation (monotonic clock);
# * the meter learns the deadline (``bind_deadline(meter)``): every metered call, of any role,
#   is refused before its reservation once the deadline has passed
#   (``ConversationTimeLimit``; nothing is sent, nothing is charged);
# * every attempt, not only the first, is refused before its reservation once the deadline
#   has passed (the meter's ``_authorize`` is wrapped per instance); the meter's retry
#   backoff and rate-limit pauses and the ledger's reservation waits are refused when
#   they would reach the deadline (``slh.budget._sleep`` / ``slh.ledger._sleep`` wrapped);
#   an attempt whose deadline passes between authorization and the wire is marked
#   rejected before dispatch, so it is charged nothing;
# * every attempt's network timeout is ``min(ATTEMPT_TIMEOUT_S, remaining)``: the
#   self-served agent's client (``slh.openai_compat.post_json``) and litellm's
#   completion (the hosted simulator) are wrapped at their module attributes, below
#   the meter's wire guard (``budget.completion_guard`` wraps whatever is installed,
#   so its checks still run first and the request it authorized is the one sent);
# * an IDENTICAL deterministic request (same body, temperature 0) that times out is
#   retried at most once: its second timeout raises ``RepeatedTimeout``, which the
#   meter does not retry (not a transient error name).
#
# Both errors end the conversation as an exception, so the episode loop records it
# as an INVALID conversation (``simulation_crash``); the partner error classifier
# reads both messages ("timed out") as per-case outcomes, never as a systematic halt.
# An orchestrator TIMEOUT termination (the step-level check) is likewise invalid in
# phase 2 (``run.records``). Invalid conversations are rerun once and then final,
# and scored under the existing rules (``check``: never a success; harm from replay).
#
# Why runtime wrapping and not an edit of ``slh/openai_compat.py``, ``slh/budget.py``
# or ``slh/runner.py``: those modules are in the pinned execution surface
# (``slh.execution.RUNTIME_MODULES``) of the registered phase-1 runs; editing them
# would change every pinned surface hash. This module is outside the surface; the
# wrappers are installed only inside phase-2 worker processes.

ATTEMPT_TIMEOUT_S = 180.0
# An identical deterministic request that timed out is sent at most this many times.
MAX_SENDS_AFTER_TIMEOUT = 2


class ConversationTimeLimit(Exception):
    """The conversation's deadline passed (message says 'timed out': a case outcome)."""


class RepeatedTimeout(Exception):
    """The same deterministic request timed out twice (never retried again)."""


class _DeadlineState:
    def __init__(self) -> None:
        self.deadline: float | None = None
        self.limit_s: float | None = None
        self.timeouts: dict[str, int] = {}
        self.events: list[dict[str, Any]] = []


_DEADLINE = _DeadlineState()
_DEADLINE_INSTALLED: dict[str, Any] = {}
_deadline_clock = time.monotonic  # replaced in tests


def deadline_remaining() -> float | None:
    """Seconds left in the current conversation (None outside one)."""
    if _DEADLINE.deadline is None:
        return None
    return _DEADLINE.deadline - _deadline_clock()


def attempt_timeout(default: float | None) -> float | None:
    """The network timeout of the next attempt: min(ATTEMPT_TIMEOUT_S, remaining,
    default). Raises ConversationTimeLimit when no time is left."""
    rem = deadline_remaining()
    if rem is None:
        return default
    if rem <= 0:
        raise ConversationTimeLimit(f"conversation timed out (limit {_DEADLINE.limit_s:.0f} s)")
    out = min(ATTEMPT_TIMEOUT_S, rem)
    return out if default is None else min(out, float(default))


@contextlib.contextmanager
def conversation_deadline(limit_s: float | None) -> Iterator[None]:
    """The deadline of one conversation; ``None`` = no deadline."""
    _DEADLINE.deadline = None if limit_s is None else _deadline_clock() + float(limit_s)
    _DEADLINE.limit_s = None if limit_s is None else float(limit_s)
    _DEADLINE.timeouts = {}
    _DEADLINE.events = []
    try:
        yield
    finally:
        _DEADLINE.deadline = None
        _DEADLINE.limit_s = None
        _DEADLINE.timeouts = {}


def deadline_events() -> list[dict[str, Any]]:
    """Timeouts seen in the current (or last) conversation: kind and body hash only."""
    return list(_DEADLINE.events)


def _body_sha(body: Any) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def _deterministic(body: dict[str, Any]) -> bool:
    return body.get("temperature") in (0, 0.0)


def _on_timeout(body: dict[str, Any], where: str) -> None:
    """Record a timed-out attempt; raise when it cannot be retried."""
    sha = _body_sha(body)
    n = _DEADLINE.timeouts.get(sha, 0) + 1
    _DEADLINE.timeouts[sha] = n
    _DEADLINE.events.append({"where": where, "body_sha256": sha[:16], "sends": n})
    rem = deadline_remaining()
    if rem is not None and rem <= 0:
        raise ConversationTimeLimit(f"conversation timed out (limit {_DEADLINE.limit_s:.0f} s)")
    if _deterministic(body) and n >= MAX_SENDS_AFTER_TIMEOUT:
        raise RepeatedTimeout(f"identical deterministic request timed out {n} times ({where})")


def _expired(what: str) -> ConversationTimeLimit:
    return ConversationTimeLimit(
        f"conversation timed out {what} (limit {_DEADLINE.limit_s:.0f} s)")


def _attempt_timeout_at_wire(default: float | None) -> float | None:
    """``attempt_timeout`` below the meter's wire guard. The guard has already
    marked the attempt dispatched; when the deadline passed between authorization
    and here, nothing is sent, so the attempt is re-marked rejected before
    dispatch and the meter charges it nothing (not its full reservation)."""
    try:
        return attempt_timeout(default)
    except ConversationTimeLimit:
        from slh import budget

        attempt = budget._attempt.get()
        if attempt is not None and not attempt.responses:
            attempt.rejected = True
            attempt.dispatched = False
        raise


def _wrap_sleep(original: Any, what: str) -> Any:
    """A meter backoff/pause or ledger reservation wait inside a conversation:
    refused (``ConversationTimeLimit``) when it would reach the deadline, so no
    retry or reservation follows an expired wait."""

    def sleep(seconds: float) -> None:
        rem = deadline_remaining()
        if rem is not None and seconds >= rem:
            _DEADLINE.events.append({"where": what, "sleep_s": round(float(seconds), 3),
                                     "remaining_s": round(max(rem, 0.0), 3)})
            raise _expired(f"before a {what}")
        original(seconds)

    sleep.__wrapped__ = original  # type: ignore[attr-defined]
    return sleep


def _wrap_post_json(original: Any) -> Any:
    from slh import openai_compat as oc

    def post_json(url: str, body: dict[str, Any], headers: dict[str, str],
                  timeout: float) -> dict[str, Any]:
        t = _attempt_timeout_at_wire(timeout)
        try:
            return original(url, body, headers, t)
        except (oc.Timeout, TimeoutError):
            if _DEADLINE.deadline is not None:
                _on_timeout(body, "agent_http")
            raise

    post_json.__wrapped__ = original  # type: ignore[attr-defined]
    return post_json


# G1 halt fix (2026-10-18). A provider that drops the connection before sending any
# response (httpx ``RemoteProtocolError`` "Server disconnected without sending a
# response") reaches us as litellm's GENERIC ``APIError``: not a transient name, so
# the meter did not retry it, and the partner classifier read it as
# ``unrecognised:APIError`` -> a run-wide halt. It is re-raised as litellm's
# ``APIConnectionError``: a transient name (``slh.budget.TRANSIENT_ERRORS``), so the
# meter retries it under the usual rules (each failed attempt charged its full
# reservation, every retry re-authorized against the ledger cap, backoff bounded by
# the deadline); only when retries run out does it end the conversation, and then it
# is a systematic connection halt (``APIConnectionError`` is a halt type), as before.
_DISCONNECT = re.compile(r"server disconnected without sending a response", re.IGNORECASE)
# Second G1 halt (2026-10-18 06:41Z): a DNS lookup failure while connecting
# ("[Errno 8] nodename nor servname provided, or not known", httpcore ``ConnectError``)
# also reaches us as the generic ``APIError``; every s1 worker failed within two
# seconds, i.e. the network dropped, not a per-conversation fault. A connect-phase
# failure (no request delivered, no response) is retried the same way. TLS or
# certificate failures are configuration: they are never matched here.
_UNREACHABLE = re.compile(r"nodename nor servname|name or service not known|"
                          r"temporary failure in name resolution|getaddrinfo failed|"
                          r"network is unreachable|no route to host|"
                          r"connection (?:refused|reset)", re.IGNORECASE)
_TLS = re.compile(r"ssl|certificate|tls", re.IGNORECASE)


def _disconnected(exc: BaseException) -> str | None:
    """Why the attempt got no response ("disconnected" | "unreachable"), else None."""
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        text = str(e)
        if type(e).__name__ == "RemoteProtocolError" or _DISCONNECT.search(text):
            return "disconnected"
        if _TLS.search(text):
            return None
        if type(e).__name__ in ("ConnectError", "gaierror") or _UNREACHABLE.search(text):
            return "unreachable"
        e = e.__cause__ or e.__context__
    return None


def _as_connection_error(exc: Exception, kwargs: dict[str, Any]) -> Exception | None:
    """The transient form of a no-response disconnect or connect failure, else None
    (raise as is)."""
    why = _disconnected(exc) if type(exc).__name__ == "APIError" else None
    if why is None:
        return None
    import litellm

    return litellm.APIConnectionError(
        message="provider disconnected without sending a response" if why == "disconnected"
        else "provider unreachable (connect failed before any response)",
        llm_provider=str(kwargs.get("custom_llm_provider") or "unknown"),
        model=str(kwargs.get("model") or "unknown"))


def _wrap_completion(original: Any) -> Any:
    def completion(*args: Any, **kwargs: Any) -> Any:
        if _DEADLINE.deadline is not None:
            t = _attempt_timeout_at_wire(kwargs.get("timeout"))
            kwargs["timeout"] = t
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            if _DEADLINE.deadline is not None and "timeout" in type(exc).__name__.lower():
                body = {k: v for k, v in kwargs.items() if k != "timeout"}
                _on_timeout(body, "litellm")
            transient = _as_connection_error(exc, kwargs)
            if transient is not None:
                raise transient from exc
            raise

    completion.__wrapped__ = original  # type: ignore[attr-defined]
    return completion


def install_deadline() -> None:
    """Wrap the clients' module attributes once per process (idempotent). Must run
    before the episode loop enters ``budget.completion_guard`` (it wraps what it
    finds, so the guard's wire checks stay outermost)."""
    if _DEADLINE_INSTALLED:
        return
    from slh import openai_compat as oc

    from slh import budget, ledger

    _DEADLINE_INSTALLED["post_json"] = oc.post_json
    oc.post_json = _wrap_post_json(oc.post_json)
    # Retry backoff, rate-limit pauses (slh.budget) and reservation waits
    # (slh.ledger) are bounded by the conversation's remaining time.
    _DEADLINE_INSTALLED["budget._sleep"] = budget._sleep
    budget._sleep = _wrap_sleep(budget._sleep, "retry backoff")
    _DEADLINE_INSTALLED["ledger._sleep"] = ledger._sleep
    ledger._sleep = _wrap_sleep(ledger._sleep, "ledger reservation wait")
    try:
        import litellm
        from tau2.utils import llm_utils
    except ImportError:  # pragma: no cover - tau2 is a hard dependency of runs
        return
    _DEADLINE_INSTALLED["litellm.completion"] = litellm.completion
    litellm.completion = _wrap_completion(litellm.completion)
    _DEADLINE_INSTALLED["llm_utils.completion"] = llm_utils.completion
    llm_utils.completion = _wrap_completion(llm_utils.completion)


def uninstall_deadline() -> None:
    """Tests: restore the original attributes."""
    if not _DEADLINE_INSTALLED:
        return
    from slh import openai_compat as oc

    from slh import budget, ledger

    oc.post_json = _DEADLINE_INSTALLED.pop("post_json")
    budget._sleep = _DEADLINE_INSTALLED.pop("budget._sleep")
    ledger._sleep = _DEADLINE_INSTALLED.pop("ledger._sleep")
    if "litellm.completion" in _DEADLINE_INSTALLED:
        import litellm
        from tau2.utils import llm_utils

        litellm.completion = _DEADLINE_INSTALLED.pop("litellm.completion")
        llm_utils.completion = _DEADLINE_INSTALLED.pop("llm_utils.completion")
    _DEADLINE_INSTALLED.clear()


def bind_deadline(meter: Any) -> Any:
    """The meter learns the deadline: each metered call is refused before its
    reservation once the conversation's time is up (instance attribute; the class
    is unchanged)."""
    if getattr(meter, "_p2_deadline_bound", False):
        return meter
    original = meter.call

    def call(role: str, config: Any, request_payload: Any, invoke: Any) -> Any:
        rem = deadline_remaining()
        if rem is not None and rem <= 0:
            raise ConversationTimeLimit(
                f"conversation timed out before a {role} call (limit {_DEADLINE.limit_s:.0f} s)")
        return original(role, config, request_payload, invoke)

    authorize = meter._authorize

    def _authorize(role: str, config: Any, reservation: float, number: int) -> Any:
        # Every attempt (first, retry, after a pause) is refused before its
        # reservation once the time is up: nothing reserved, sent or charged.
        rem = deadline_remaining()
        if rem is not None and rem <= 0:
            raise _expired(f"before {role} attempt {number}")
        return authorize(role, config, reservation, number)

    meter.call = call
    meter._authorize = _authorize
    meter._p2_deadline_bound = True
    return meter


# --------------------------------------------------------------------------- worker


def _quiet_logging() -> None:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING",
               filter=lambda record: not record["name"].startswith(("tau2", "litellm")))


def _worker_entry(*args: Any) -> None:
    """A spawned worker process: dies with its coordinator (parent watchdog)."""
    from slh.worker import start_parent_watchdog

    start_parent_watchdog()
    _quiet_logging()
    _worker_main(*args)


def _error_text(private: Path, episode_id: str) -> str:
    """The raw error text (private: used to classify, never published)."""
    rows = [r for r in _jsonl(private / "errors.jsonl") if r.get("episode_id") == episode_id]
    if not rows:
        return "unknown error"
    return f"{rows[-1].get('type')}: {rows[-1].get('message', '')}"


def _error_public(private: Path, episode_id: str) -> str:
    """The public form of the episode's last error (type, code, message hash)."""
    rows = [r for r in _jsonl(private / "errors.jsonl") if r.get("episode_id") == episode_id]
    if not rows:
        return "unknown error"
    r = rows[-1]
    return public_error(str(r.get("type")), str(r.get("message", "")), r.get("code"))


def _worker_main(out_str: str, package_str: str, session: int, index: int,
                 jobs: list[dict[str, Any]], pin: dict[str, str] | None,
                 init: Any = None) -> None:
    """Run ``jobs`` in order under the shared ledger; stop at the first halt."""
    from tau2.data_model.tasks import Task

    from slh import runner
    from slh.budget import BudgetExceeded
    from slh.capture import CallRecorder
    from slh.poc9.pipeline import credit_failure
    from slh.xp.partner.errors import classify

    if init is not None:
        init()
    out = Path(out_str)
    wid = f"s{session}-w{index}"
    wdir = out / "workers" / wid
    wdir.mkdir(parents=True, exist_ok=False)
    lock = runner._try_lock(wdir / "worker.lock")
    if lock is None:
        raise SystemExit(EXIT_REFUSED)
    pdir = runner.private_dir(out) / "workers" / wid
    pdir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {"worker": wid, "pid": os.getpid(), "status": "running",
                              "jobs": len(jobs), "written": 0, "invalid": 0}

    def write_status() -> None:
        runner._write_json_atomic(wdir / "status.json", redact_public(status))

    write_status()
    identity = pc.read_json(out / "identity.json")
    config = runner.load_config(out / "config" / "run.yaml")
    if config.resolved_sha256() != identity["config_resolved_sha256"]:
        raise SystemExit(EXIT_REFUSED)
    # Selected by ARM NAME (guard review 2026-10-18): two arms of one kind (B1 and
    # B1+guard, identical prompts) differ only here.
    guard_arms = set((identity.get("identity_guard") or {}).get("arms") or [])
    if guard_arms:
        from slh.guards import identity as IG

        IG.install()  # composition: rebinds slh.runner.build_guards in this process
    package = PK.Package(Path(package_str), check_code=False)  # coordinator checked the code
    if package.registration_sha256 != identity["package_registration_sha256"]:
        raise SystemExit(EXIT_REFUSED)
    rows = {r["scenario_id"]: r for split in PK.SPLITS for r in package.scenarios(split)}
    ledger = marked_ledger(out / "ledger.jsonl", config.cap_usd, owner=wid, session=session,
                          create=False, quarantine=runner.private_dir(out) / "quarantine.jsonl",
                          spend_mark=out / SPEND_MARK)
    meter = runner.make_meter(config, ledger)
    meter.on_change = lambda _m: write_status()
    # Hang fix: the conversation deadline reaches the meter and every attempt's
    # network timeout (the deadline section above); installed before the episode loop.
    install_deadline()
    bind_deadline(meter)
    limit_s = float(config.timeout_s) if config.timeout_s is not None else None
    recorder = CallRecorder(wdir / "calls.jsonl", pdir / "calls.jsonl", session=session,
                            worker=wid)
    tasks: dict[str, Any] = {}
    states: dict[tuple[str, int], Any] = {}
    sinks: dict[str, Any] = {}
    consecutive = 0

    def halt(label: str, detail: str, execution_id: str | None, job: Job,
             raw: str | None = None) -> None:
        """``detail`` is public (sanitized); ``raw`` goes to the private log only."""
        record = redact_public({"label": label, "detail": detail[:500],
                                "execution_id": execution_id, "job": job.key, "worker": wid,
                                "time_epoch": time.time()})
        if raw is not None:
            pc.append_jsonl(pdir / "halts.jsonl", [{**record, "raw": raw}])
        pc.append_jsonl(wdir / "halts.jsonl", [record])
        ledger.halt(redact_text(f"{record['label']}: {record['detail'][:200]}"))
        status.update(status="halted", halt=record)

    for spec in jobs:
        job = Job(**spec)
        try:
            ledger.assert_not_halted()
        except BudgetExceeded:
            status["status"] = "stopped_by_halt"
            break
        if job.scenario_id not in tasks:
            tasks[job.scenario_id] = Task.model_validate(PK.to_tau2_task(rows[job.scenario_id]))
        if (job.kind, job.run) not in states:
            states[(job.kind, job.run)] = arm_state(job.kind, job.run, package.bundles, config)
        state = states[(job.kind, job.run)]
        runner.check_state_allowed(state, config)
        if job.arm not in sinks:
            adir = wdir / "arms" / job.arm
            adir.mkdir(parents=True, exist_ok=True)
            for name in ("episodes.jsonl", "outcomes.jsonl"):
                (adir / name).touch()
            apriv = pdir / "arms" / job.arm
            apriv.mkdir(parents=True, exist_ok=True)
            sinks[job.arm] = runner._Sink(
                episodes=adir / "episodes.jsonl", outcomes=adir / "outcomes.jsonl",
                private=apriv, blobs=runner.BlobStore(out / "blobs"),
                private_blobs=runner.BlobStore(runner.private_dir(out) / "blobs"))
        sink = sinks[job.arm]
        box: dict[str, Any] = {}
        try:
            with conversation_deadline(limit_s), (
                    IG.active(job.arm in guard_arms) if guard_arms
                    else contextlib.nullcontext()):
                result, reason = runner._run_pairs(
                    DOMAIN, config, state, [(job.run, job.seed, tasks[job.scenario_id])],
                    meter, sink, recorder, set(),
                    lambda o, box=box: box.__setitem__("outcome", o),
                    agent_level="messages", serving_pin=pin)
        except Exception as exc:  # noqa: BLE001 - leaks and unmetered calls are fatal
            halt("fatal", public_exception(exc), None, job, raw=f"{type(exc).__name__}: {exc}")
            write_status()
            raise SystemExit(2) from None
        outcome = box.get("outcome")
        execution_id = None
        if outcome is not None:
            status["written"] += 1
            last = _jsonl(sink.episodes)
            execution_id = last[-1].get("execution_id") if last else None
        if outcome is not None and not outcome.get("valid"):
            status["invalid"] += 1
            eid = runner.episode_id_for(job.scenario_id, job.run)
            text = _error_text(sink.private, eid)  # raw: classification only
            public = _error_public(sink.private, eid)
            kind, label = classify(text)
            if credit_failure(text):
                kind, label = "halt", "credit"
            if kind == "halt":
                halt(label, public, execution_id, job, raw=text)
                break
            consecutive += 1
            if consecutive >= CONSECUTIVE_INVALID:
                halt("consecutive_invalid", f"{consecutive} invalid conversations in a row "
                     f"(last: {public})", execution_id, job, raw=text)
                break
        elif outcome is not None:
            consecutive = 0
        if result is not None:
            halt(result, json.dumps(reason or {}), None, job)
            break
        write_status()
    if status["status"] == "running":
        status["status"] = "done"
    write_status()
    lock.close()


# --------------------------------------------------------------------------- coordinator


def env_report(ep: dict[str, Any], *, mode: str) -> dict[str, Any]:
    def cmd(*args: str) -> str | None:
        try:
            return subprocess.run(list(args), capture_output=True, text=True, timeout=15,
                                  check=True).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    def version(name: str) -> str | None:
        from importlib import metadata

        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    served = {}
    if mode != "dry":
        for role in ("agent", "user"):
            url = (ep.get(role) or {}).get("base_url")
            if url:
                got = probe_models(url, (ep[role] or {}).get("api_key_env"))
                served[role] = {k: v for k, v in got.items() if k != "raw_error"}
    return {"mode": mode, "time_epoch": time.time(), "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu": cmd("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                       "--format=csv,noheader"),
            "packages": {n: version(n) for n in ("vllm", "tau2", "litellm", "torch")},
            "endpoint": redact_public({k: v for k, v in ep.items()
                                       if k not in ENDPOINT_VOLATILE}),
            "served": redact_public(served)}


def probe_models(base_url: str, api_key_env: str | None = None) -> dict[str, Any]:
    """GET <base_url>/models (no generation, nothing billed). The token, if any,
    is read from the named variable and never recorded."""
    import urllib.request

    req = urllib.request.Request(base_url.rstrip("/") + "/models")
    if api_key_env and os.environ.get(api_key_env):
        req.add_header("Authorization", f"Bearer {os.environ[api_key_env]}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        return {"models": sorted(m.get("id") for m in data.get("data") or [])}
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        # public: type, HTTP status, message hash; raw_error: private logs only
        return {"error": public_exception(exc), "raw_error": f"{type(exc).__name__}: {exc}"}


def _log_raw(log: Any, role: str, step: str, got: dict[str, Any]) -> None:
    if log is not None and got.get("raw_error"):
        log({"role": role, "step": step, "raw_error": got["raw_error"],
             "time_epoch": time.time()})


def check_servers(ep: dict[str, Any], *, log: Any = None) -> list[str]:
    """Both local servers answer and serve the configured model names. Problems
    carry only sanitized errors; ``log`` (private) receives the raw ones."""
    out = []
    for role in ("agent", "user"):
        part = ep.get(role) or {}
        if not part.get("base_url"):
            continue
        got = probe_models(part["base_url"], part.get("api_key_env"))
        _log_raw(log, role, "models", got)
        if "error" in got:
            out.append(f"{role} server at {part['base_url']} does not answer ({got['error']})")
        elif part["model"] not in got["models"]:
            out.append(f"{role} server serves {got['models']}, not {part['model']!r}")
    return out


PROBE_PROMPT = "Reply with one short sentence: what is 2 + 3?"
THINK_MARKERS = ("<think>", "</think>")


PROBE_MAX_TOKENS = 64


def probe_body(model: str) -> dict[str, Any]:
    return {"model": model, "temperature": 0.0, "max_tokens": PROBE_MAX_TOKENS,
            "messages": [{"role": "user", "content": PROBE_PROMPT}]}


def probe_chat(base_url: str, model: str, api_key_env: str | None = None) -> dict[str, Any]:
    """One real chat completion (a few tokens, temperature 0, no thinking switch sent:
    the server's default mode is what the run gets). Non-thinking means a non-empty
    answer, no reasoning field and no think markers in the text. ``usage`` is the
    response's token usage (None when the server sends none)."""
    import urllib.request

    body = json.dumps(probe_body(model)).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    if api_key_env and os.environ.get(api_key_env):
        req.add_header("Authorization", f"Bearer {os.environ[api_key_env]}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        msg = (data.get("choices") or [{}])[0].get("message") or {}
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        return {"error": public_exception(exc), "raw_error": f"{type(exc).__name__}: {exc}",
                "usage": None}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    thinking = bool(str(reasoning).strip()) or any(t in content for t in THINK_MARKERS)
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
    return {"content": content[:200], "reasoning_chars": len(str(reasoning)),
            "thinking": thinking, "ok": bool(content.strip()) and not thinking,
            "usage": usage}


def probe_bound_usd(price: Any, input_tokens: int) -> float:
    """Upper bound for a probe: every input token at the larger of the input and
    cached-input prices (equal to the meter's input-price bound, since a cached
    price above the input price is refused at config time)."""
    return (input_tokens * max(price.input, price.cached_input)
            + PROBE_MAX_TOKENS * price.output) / 1_000_000


def metered_prober(out: Path, cap_usd: float, *, probe: Any = None) -> Any:
    """A chat probe of a hosted endpoint, charged through the run's spend ledger
    under its cap: the full bound (every input token at the larger of the input
    and cached prices, PROBE_MAX_TOKENS output) is reserved before the request,
    then settled at the usage-priced cost (the full reservation when the
    response reports no usage or the request fails). A cost above the
    reservation halts the ledger session (overspend)."""
    from slh.budget import (
        UNDERESTIMATE_TOLERANCE,
        BudgetExceeded,
        input_token_bound,
        load_prices,
        reported_cost_from_response,
    )

    out = Path(out)
    probe = probe or probe_chat

    def run_probe(role: str, part: dict[str, Any]) -> dict[str, Any]:
        prices = load_prices(out / "config" / "prices.yaml")
        price = prices.get(part["model"])
        if price is None:
            return {"error": f"no resolved price for {part['model']!r}; not probed",
                    "usage": None}
        problem = accounting_problem(out)
        if problem is not None:
            return {"error": f"not probed: {problem}", "usage": None}
        fresh = not (out / "ledger.jsonl").is_file()
        ledger = marked_ledger(out / "ledger.jsonl", float(cap_usd), owner="probe",
                              session=_session(out), create=True,
                              spend_mark=out / SPEND_MARK)
        if fresh:
            ledger.open_session(_opening_spend(out))
        bound = probe_bound_usd(price, input_token_bound(probe_body(part["model"])))
        try:
            rid = ledger.reserve(role, part["model"], bound, 1) if bound > 0 else None
        except BudgetExceeded as exc:
            return {"error": f"probe refused by the budget: {exc}", "usage": None}
        got: dict[str, Any] = {"error": "probe interrupted", "usage": None}
        try:
            got = probe(part["base_url"], part["model"], part.get("api_key_env"))
        finally:
            usage = got.get("usage") if isinstance(got, dict) else None
            cost, table_cost = bound, None
            if usage and isinstance(usage.get("prompt_tokens"), int) and isinstance(
                    usage.get("completion_tokens"), int):
                cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                table_cost = price.usd(usage["prompt_tokens"], usage["completion_tokens"],
                                       int(cached))
                cost = table_cost
            # Same rule as the main meter (slh.budget): charge max(table, reported)
            # and halt on a provider-reported cost above the table's.
            reported = reported_cost_from_response({"usage": usage}) if usage else None
            underestimate = (reported is not None and table_cost is not None and reported
                             > table_cost * (1 + UNDERESTIMATE_TOLERANCE) + 1e-12)
            if reported is not None:
                cost = max(cost, reported)
            if rid is not None or cost > 0:
                ledger.settle(rid, cost, role, part["model"])
            if underestimate:
                ledger.halt("price_table_underestimate")
                got = {**got, "error": f"probe: provider-reported cost ${reported:.6f} "
                                       f"exceeds the price table's ${table_cost:.6f} "
                                       "(price_table_underestimate)"}
            elif not math.isfinite(cost) or cost > bound + 1e-12:
                ledger.halt("overspend")
                got = {**got, "error": f"probe cost ${cost:.6f} exceeds its ${bound:.6f} "
                                       "reservation (overspend)"}
        return {**got, "charged_usd": cost}

    return run_probe


def check_chat(ep: dict[str, Any], *, metered: Any = None,
               skipped: list[str] | None = None, log: Any = None) -> list[str]:
    """Smoke step: one real chat completion per served model; the reply must be a
    plain (non-thinking) answer. Only an endpoint declared ``self_hosted`` is
    probed unmetered; any other (hosted, billed per token) is probed through
    ``metered`` (``metered_prober``: the run's ledger and cap) or, without it,
    not at all (named in ``skipped``)."""
    out = []
    for role in ("agent", "user"):
        part = ep.get(role) or {}
        if not part.get("base_url"):
            continue
        if part.get("self_hosted") is True:
            got = probe_chat(part["base_url"], part["model"], part.get("api_key_env"))
        elif metered is not None:
            got = metered(role, part)
        else:
            if skipped is not None:
                skipped.append(f"{role}: hosted endpoint (not self_hosted); its chat probe "
                               "is metered and runs only inside smoke/full, under the "
                               "run's cap")
            continue
        _log_raw(log, role, "chat", got)
        if "error" in got:
            out.append(f"{role} server: chat completion failed ({got['error']})")
        elif got["thinking"]:
            out.append(f"{role} server answers in thinking mode (start it with thinking off "
                       "by default)")
        elif not got["ok"]:
            out.append(f"{role} server returned an empty answer")
    return out


def _write_config(out: Path, files: dict[str, str]) -> Path:
    cdir = out / "config"
    if cdir.exists():
        for name, text in files.items():
            path = cdir / name
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                raise Refused(f"{path} differs from the settings now given (endpoint, cap, "
                              "plan or price table changed); keep them or use a new run "
                              "directory")
        return cdir / "run.yaml"
    cdir.mkdir(parents=True)
    for name, text in files.items():
        (cdir / name).write_text(text, encoding="utf-8")
    return cdir / "run.yaml"


AMENDMENTS = "amendments.jsonl"
AMENDMENT_SCHEMA = "xp-phase2-run-amendment-v1"


def amendment_record_sha(row: dict[str, Any]) -> str:
    return pc.sha_text(pc.canonical({k: v for k, v in row.items() if k != "record_sha256"}))


def code_amendment(out: Path) -> dict[str, Any] | None:
    """The code a run directory may resume under after recorded amendments.

    ``<run>/amendments.jsonl`` is append-only, written only by an explicit, owner-approved
    amendment command (``python -m slh.xp.g1 amend``, which does the file-level checks).
    Here each record is verified: self-hash, chained to the previous record, bound to
    this run's identity.json (never rewritten), ``old_code_sha256`` = the previous
    accepted code, ``new_code_sha256`` = the hash of its recorded file map. Returns
    None when the run has no amendment; raises Refused when the chain is damaged."""
    out = Path(out)
    path = out / AMENDMENTS
    if not path.is_file():
        return None
    if not (out / "identity.json").is_file():
        raise Refused(f"{path} exists but the run has no identity.json")
    identity = pc.read_json(out / "identity.json")
    ident_sha = pc.sha_file(out / "identity.json")
    rows = pc.read_jsonl(path)
    if not rows:
        raise Refused(f"{path} is empty")
    code, prev = identity["code_sha256"], None
    for i, row in enumerate(rows):
        where = f"{path} record {i + 1}"
        if row.get("schema") != AMENDMENT_SCHEMA:
            raise Refused(f"{where}: unknown schema {row.get('schema')!r}")
        if row.get("record_sha256") != amendment_record_sha(row):
            raise Refused(f"{where} fails its own hash")
        if row.get("prev_record_sha256") != prev:
            raise Refused(f"{where} is not chained to the previous record")
        if row.get("identity_sha256") != ident_sha:
            raise Refused(f"{where} belongs to another run identity")
        if row.get("old_code_sha256") != code:
            raise Refused(f"{where}: old code {row.get('old_code_sha256')} is not the code "
                          f"accepted before it ({code})")
        if pc.sha_text(pc.canonical(row.get("new_files") or {})) != row.get("new_code_sha256"):
            raise Refused(f"{where}: new_code_sha256 does not match its file map")
        code, prev = row["new_code_sha256"], row["record_sha256"]
    return {"origin_code_sha256": identity["code_sha256"], "code_sha256": code,
            "amendments": len(rows), "last_record_sha256": prev,
            "changed_files": sorted({f for r in rows for f in r.get("changed_files") or ()})}


def _amended_package(package: Path, amended: dict[str, Any], root: Path) -> Any:
    """The package, its code check replaced by the run's amendment chain: the package
    was built under the origin code and this checkout holds the amended code."""
    pkg = PK.Package(package, root=root, check_code=False)
    if pkg.registration.get("code_sha256") != amended["origin_code_sha256"]:
        raise Refused("the package's registered code is not the code this run started with")
    have = pc.code_hashes(root)["code_sha256"]
    if have != amended["code_sha256"]:
        raise Refused(f"the code in this checkout ({have[:12]}) differs from the run's amended "
                      f"code ({amended['code_sha256'][:12]}); only a new recorded amendment "
                      "(python -m slh.xp.g1 amend) can accept it")
    return pkg


def prepare(package: Path, out: Path, *, mode: str, endpoint: dict[str, Any],
            cap_usd: float = 0.0, root: Path = pc.REPO, check_code: bool = True,
            write: bool = True, identity_guard: Any = None,
            code_amendments: Path | None = None) -> dict[str, Any]:
    """Every check that needs no server: package, code, benchmark data, endpoint,
    arm prompts, job plan; and (``write``) the run directory's identity. Raises
    Refused on any mismatch.

    ``identity_guard``: ARM NAMES (e.g. ``("b1_guard",)``; never kinds, so one of
    two arms of the same kind can be guarded) whose conversations run under
    ``slh.guards.identity`` (off by default). When set,
    the identity gains ``identity_guard`` (the guard's descriptor); when off the
    identity is exactly as before, so a run directory is never resumed under the
    other setting.

    Code amendments (``code_amendment``): when ``<out>/amendments.jsonl`` (or that of
    ``code_amendments``, the run whose amendments govern a throwaway rehearsal
    directory) holds a valid chain, the code check accepts exactly the amended code
    and a resumed identity may differ from the stored one in ``code_sha256`` only,
    and only from the origin code to the amended code; identity.json is kept as
    written at the start."""
    from slh import runner

    if mode not in MODES:
        raise Refused(f"unknown mode {mode!r}")
    amended = code_amendment(Path(code_amendments or out))
    try:
        if amended is not None and check_code:
            pkg = _amended_package(package, amended, root)
        else:
            pkg = PK.Package(package, root=root, check_code=check_code)
    except PK.PackageRefused as exc:
        raise Refused(str(exc)) from None
    problems = endpoint_problems(endpoint) + B.problems(pkg.bundles, pkg.plan)
    want_data = pkg.registration.get("data_sha256")
    have_data = runner.data_hashes(DOMAIN)
    if want_data is not None and want_data != have_data:
        problems.append("the benchmark data files differ from the registered ones")
    if problems:
        raise Refused("; ".join(problems))
    split = SPLIT_OF_MODE[mode]
    rows = pkg.scenarios(split)
    blocks = plan_jobs(pkg.plan, rows, split)
    jobs = flat(blocks)
    if split == "smoke":
        test_ids = {r["scenario_id"] for r in pkg.scenarios("test")}
        if any(j.scenario_id in test_ids for j in jobs):
            raise Refused("a smoke job names a test scenario")
        if smoke_test_overlap(pkg):
            raise Refused("a smoke job runs a test scenario under another id (same "
                          "instructions)")
    files = config_files(endpoint, plan=pkg.plan, cap_usd=cap_usd, root=root)
    out = Path(out)
    if write:
        out.mkdir(parents=True, exist_ok=True)
        config_path = _write_config(out, files)
    else:
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="p2-config."))
        config_path = _write_config(tmp / "x", files)
    config = runner.load_config(config_path)
    prompts = arm_prompts(jobs, pkg.bundles, config)
    identity = {"schema": RUN_SCHEMA, "mode": mode, "split": split,
                "plan_id": pkg.plan.plan_id, "plan_sha256": pkg.plan_json["sha256"],
                "package_registration_sha256": pkg.registration_sha256,
                "code_sha256": pc.code_hashes(root)["code_sha256"],
                "endpoint_sha256": endpoint_sha256(endpoint),
                "config_resolved_sha256": config.resolved_sha256(),
                "data_sha256": have_data, "cap_usd": float(cap_usd),
                "jobs": len(jobs), "jobs_sha256": pc.sha_text(pc.canonical(
                    [j.key for j in jobs])),
                "arm_prompts_sha256": pc.sha_text(pc.canonical(prompts)),
                "agent_max_tokens": config.agent.max_tokens,
                "conversation_deadline": {
                    "limit_s": float(pkg.plan.conversation_time_limit_s),
                    "attempt_timeout_s": ATTEMPT_TIMEOUT_S,
                    "max_sends_after_timeout": MAX_SENDS_AFTER_TIMEOUT,
                    "invalid_terminations": list(INVALID_TERMINATIONS)}}
    if identity_guard:
        from slh.guards import identity as IG

        names = sorted({j.arm for j in jobs})
        bad = sorted(set(identity_guard) - set(names))
        if bad:
            raise Refused(f"identity guard: arm(s) {bad} not in this plan ({names})")
        identity["identity_guard"] = IG.descriptor(identity_guard)
    if write:
        path = out / "identity.json"
        if path.is_file():
            old = pc.read_json(path)
            diff = sorted(k for k in set(old) | set(identity) if old.get(k) != identity.get(k))
            if (diff == ["code_sha256"] and amended is not None and code_amendments is None
                    and old["code_sha256"] == amended["origin_code_sha256"]
                    and identity["code_sha256"] == amended["code_sha256"]):
                diff, identity = [], old  # the recorded amendment; identity.json unchanged
            if diff:
                raise Refused(f"{out} was started under another identity ({diff}); keep the "
                              "settings or use a new run directory")
        else:
            pc.write_json(path, identity)
        pc.write_json(out / "arm-prompts.json", prompts)
        pc.write_json(out / "jobs.json", {"blocks": [[j.as_dict() for j in b]
                                                     for b in blocks]})
    return {"package": pkg, "config": config, "blocks": blocks, "jobs": jobs,
            "identity": identity, "prompts": prompts, "files": files}


def _assign(blocks: list[list[Job]], pending: set[str], workers: int) -> list[list[Job]]:
    """Pending jobs, whole blocks kept together, dealt round-robin in dispatch order."""
    todo = [[j for j in b if j.key in pending] for b in blocks]
    todo = [b for b in todo if b]
    n = max(1, min(workers, len(todo)))
    return [[j for b in todo[k::n] for j in b] for k in range(n) if todo[k::n]]


def _session(out: Path) -> int:
    dirs = _worker_dirs(out)
    return 1 + max((int(d.name[1:].split("-w")[0]) for d in dirs), default=-1)


def execution_history(out: Path) -> list[str]:
    """What shows that this run directory already executed something: worker
    directories (public or private) and a coordinator status file."""
    out = Path(out)
    found = [f"workers/{d.name}" for d in _worker_dirs(out)]
    pw = out / "private" / "workers"
    if pw.is_dir():
        found += [f"private/workers/{d.name}" for d in sorted(pw.iterdir()) if d.is_dir()]
    if (out / "status.json").is_file():
        found.append("status.json")
    return found


# --------------------------------------------------------------------------- spend mark
# Kept here, not in slh.ledger: slh/ledger.py is part of the pinned execution
# surface (slh.execution.RUNTIME_MODULES).

SPEND_OPS = ("reserve", "settle", "charge")


def marked_ledger(*args: Any, spend_mark: str | Path, **kwargs: Any) -> Any:
    """A ``SharedLedger`` that, after every reservation and settlement it appends
    (lock held, after the ledger's own fsynced append, so the mark never runs
    ahead of the ledger), appends the run's charged and committed spend and the
    in-flight reservation ids to ``spend_mark`` (fsynced)."""
    from slh.ledger import SharedLedger, _now

    class _Marked(SharedLedger):
        def _append(self, record: dict[str, Any]) -> None:
            super()._append(record)
            if record.get("op") not in SPEND_OPS:
                return
            mark = {"op": record["op"], "id": record.get("id"), "owner": self.owner,
                    "session": self.session, "charged_usd": self.charged_usd,
                    "committed_usd": self.committed_usd,
                    "pending_ids": sorted(self.pending), "time_utc": _now()}
            line = (json.dumps(mark, ensure_ascii=True, allow_nan=False) + "\n").encode()
            with Path(spend_mark).open("ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    return _Marked(*args, **kwargs)


def spend_mark_problem(ledger_path: Path, mark_path: Path) -> str | None:
    """None when the ledger accounts for every spend the high-water file recorded:
    its charged total is at least the highest charged total ever marked (charged
    spend only grows), and every reservation in flight at the last mark is in the
    ledger (settled or pending). A ledger that lost reservation or settlement
    records fails one of the two; so does recorded spend without a mark file."""
    from slh.ledger import EPSILON, read_records

    marks = read_records(mark_path) if mark_path.is_file() else []
    recs = read_records(ledger_path) if ledger_path.is_file() else []
    if not marks:
        if any(r.get("op") in SPEND_OPS for r in recs):
            return (f"the ledger records spend but the spend high-water file "
                    f"{mark_path.name} is missing or empty; refusing to resume with "
                    "unverifiable spend")
        return None
    charged = sum(float(r["usd"]) for r in recs if r.get("op") in ("settle", "charge"))
    high = max(float(m.get("charged_usd", 0.0)) for m in marks)
    if not math.isfinite(charged) or charged + EPSILON < high:
        return (f"the ledger's charged spend ${charged:.6f} is below the recorded "
                f"high-water mark ${high:.6f} (records were lost); refusing to resume")
    reserved = {r.get("id") for r in recs if r.get("op") == "reserve"}
    lost = [i for i in marks[-1].get("pending_ids") or [] if i not in reserved]
    if lost:
        return (f"{len(lost)} reservation(s) in flight at the last spend mark are "
                "missing from the ledger (records were lost); refusing to resume")
    return None


def spend_high_water(mark_path: Path) -> float:
    """The highest charged spend the high-water file recorded (0 without one)."""
    from slh.ledger import read_records

    marks = read_records(mark_path) if Path(mark_path).is_file() else []
    return max((float(m.get("charged_usd", 0.0)) for m in marks), default=0.0)


def accounting_problem(out: Path) -> str | None:
    """None when the spend ledger accounts for the run's execution history and
    recorded spend (see ``_accounting``); otherwise the reason spend is unknown."""
    found = _accounting(out)
    return None if found is None else found[1]


def _accounting(out: Path) -> tuple[str, str] | None:
    """None when the spend ledger accounts for the run's execution history: no
    history and no ledger (a fresh run), or a ledger with a session record for
    every worker session on disk whose recorded spend is not below the durable
    high-water mark (``SPEND_MARK``, appended and fsynced after every
    reservation and settlement). Otherwise ``(halt label, reason)``."""
    from slh.ledger import read_records

    out = Path(out)
    history = execution_history(out)
    path = out / "ledger.jsonl"
    if not path.is_file():
        if history:
            return "accounting_missing", (
                f"execution history exists ({', '.join(history[:5])}"
                f"{', ...' if len(history) > 5 else ''}) but the spend ledger "
                "ledger.jsonl is missing; refusing to run with unknown spend (restore "
                "the ledger, or start a new run directory)")
        if (out / SPEND_MARK).is_file():
            return "accounting_inconsistent", (
                f"the spend high-water file {SPEND_MARK} exists but the spend ledger "
                "ledger.jsonl is missing; refusing to run with unknown spend")
        return None
    recs = read_records(path)
    sessions = {r.get("session") for r in recs if r.get("op") == "session"}
    if history and not sessions:
        return "accounting_missing", ("the spend ledger has no session record; refusing to "
                                      "resume with unknown spend")
    workers = sorted({int(d.name[1:].split("-w")[0]) for d in _worker_dirs(out)})
    missing = [s for s in workers if s not in sessions]
    if missing:
        return "accounting_missing", (
            f"worker session(s) {missing} have no session record in the spend ledger "
            "(the ledger was replaced or truncated); refusing to resume with unknown "
            "spend")
    problem = spend_mark_problem(path, out / SPEND_MARK)
    if problem is not None:
        return "accounting_inconsistent", problem
    return None


def _opening_spend(out: Path) -> float:
    """What a session opens with: the high-water mark, so the session's opening
    spend is max(ledger spend, high-water mark) (``open_session`` charges any gap)."""
    return spend_high_water(Path(out) / SPEND_MARK)


def _check_accounting(out: Path) -> None:
    found = _accounting(out)
    if found is not None:
        raise halt(out, {"label": found[0], "detail": found[1], "time_epoch": time.time()})


def _halt_record(out: Path, session: int) -> dict[str, Any] | None:
    """The session's FIRST halt in time (the cause). The other workers then stop on
    the ledger's halt latch and record that as BudgetExceeded; taking the first worker
    directory instead reported such a follow-on halt (G1, 2026-10-18)."""
    first: dict[str, Any] | None = None
    for wdir in _worker_dirs(out):
        if not wdir.name.startswith(f"s{session}-"):
            continue
        status = _jsonl(wdir / "halts.jsonl")
        if status and (first is None or float(status[0].get("time_epoch") or 0.0)
                       < float(first.get("time_epoch") or 0.0)):
            first = status[0]
    return first


def execute(prep: dict[str, Any], out: Path, *, package: Path, workers: int,
            worker_init: Any = None, in_process: bool | None = None) -> dict[str, Any]:
    """One session: rounds of workers over the pending jobs until none is pending,
    a halt, or ``PASSES`` rounds. Returns the status summary (also status.json)."""
    from slh import runner
    from slh.ledger import SharedLedger, read_records

    out = Path(out)
    config = prep["config"]
    lock = runner._try_lock(out / "run.lock")
    if lock is None:
        raise Refused(f"{out} is in use by another runner")
    try:
        runner._check_no_live_workers(out)
        _check_accounting(out)
        private = runner.private_dir(out)
        fresh = not (out / "ledger.jsonl").is_file()
        session_ledger = marked_ledger(out / "ledger.jsonl", config.cap_usd,
                                      owner="coordinator", session=_session(out),
                                      quarantine=private / "quarantine.jsonl", create=fresh,
                                      spend_mark=out / SPEND_MARK)
        if not fresh and not any(r.get("op") == "session"
                                 for r in read_records(out / "ledger.jsonl")):
            raise Refused("the spend ledger has no session record; refusing to resume with "
                          "unknown spend")
        session_ledger.reconcile()
        pin = runner.pin_serving(config)
        halt = None
        for _round in range(PASSES):
            session = _session(out)
            ledger = marked_ledger(out / "ledger.jsonl", config.cap_usd,
                                  owner=f"s{session}-coord", session=session,
                                  quarantine=private / "quarantine.jsonl", create=False,
                                  spend_mark=out / SPEND_MARK)
            ledger.open_session(_opening_spend(out))
            resolved = resolve(prep["jobs"], records(out))
            pending = {k for k, v in resolved.items() if v["status"] == "pending"}
            if not pending:
                break
            plan = _assign(prep["blocks"], pending, workers)
            args = [(str(out), str(package), session, k, [j.as_dict() for j in jobs], pin,
                     worker_init) for k, jobs in enumerate(plan)]
            if in_process if in_process is not None else len(args) == 1:
                for a in args:
                    _worker_main(*a)
            else:
                ctx = multiprocessing.get_context("spawn")
                procs = [ctx.Process(target=_worker_entry, args=a, name=f"s{session}-w{k}")
                         for k, a in enumerate(args)]
                try:
                    for p in procs:
                        p.start()
                    for p in procs:
                        p.join()
                except BaseException:
                    ledger.halt("coordinator interrupted")
                    for p in procs:
                        if p.pid is None:
                            continue  # never started
                        p.join(timeout=30)
                        if p.is_alive():
                            p.terminate()
                            p.join()
                    raise
                finally:
                    ledger.reconcile()
            halt = _halt_record(out, session)
            if halt is not None:
                break
        resolved = resolve(prep["jobs"], records(out))
        halt = redact_public(halt) if halt is not None else None
        summary = {"identity": prep["identity"], "coverage": coverage(resolved),
                   "halt": halt, "spend": SharedLedger(
                       out / "ledger.jsonl", config.cap_usd, owner="status",
                       session=_session(out), create=False).snapshot(),
                   "time_epoch": time.time()}
        pc.write_json(out / "status.json", summary)
        if halt is not None:
            pc.write_json(out / "halt.json", halt)
        elif (out / "halt.json").is_file():
            (out / "halt.json").unlink()
        return summary
    finally:
        lock.close()


def halt(out: Path, record: dict[str, Any]) -> Halted:
    """The common halt path: persist ``record`` as the run's halt.json (what
    ``status`` reports; removed when a later session finishes without a halt)
    and return the exception to raise."""
    record = redact_public(record)
    pc.write_json(Path(out) / "halt.json", record)
    return Halted(record)


def run(package: Path, out: Path, *, mode: str, endpoint: dict[str, Any],
        cap_usd: float = 0.0, workers: int | None = None, root: Path = pc.REPO,
        check_code: bool = True, worker_init: Any = None, export: bool | None = None,
        in_process: bool | None = None, identity_guard: Any = None,
        code_amendments: Path | None = None) -> dict[str, Any]:
    """Prepare (all checks), check the servers, run the pending jobs, export when
    complete (dry-run: always). ``code_amendments``: see ``prepare``."""
    from slh.runner import load_config

    if mode == "dry":
        endpoint = stub_endpoint(min(2, int(endpoint.get("workers", 2))))
    prep = prepare(package, out, mode=mode, endpoint=endpoint, cap_usd=cap_usd, root=root,
                   check_code=check_code, identity_guard=identity_guard,
                   code_amendments=code_amendments)
    config = load_config(Path(out) / "config" / "run.yaml")
    missing = [k for k in required_keys(config) if not os.environ.get(k)]
    if missing:
        raise Refused(f"missing environment variable(s) {missing} (names only; nothing sent)")
    _check_accounting(Path(out))  # before any probe could open a fresh ledger
    if mode != "dry":
        def log(record: dict[str, Any]) -> None:
            from slh.runner import private_dir

            pc.append_jsonl(private_dir(Path(out)) / "server-checks.jsonl", [record])

        problems = check_servers(endpoint, log=log)
        if problems:
            raise halt(out, {"label": "servers", "detail": "; ".join(problems),
                             "time_epoch": time.time()})
        problems = check_chat(endpoint, metered=metered_prober(Path(out), float(cap_usd)),
                              log=log)
        if problems:
            raise halt(out, {"label": "servers_chat", "detail": "; ".join(problems),
                             "time_epoch": time.time()})
    pc.append_jsonl(Path(out) / "logs" / "env.jsonl", [env_report(endpoint, mode=mode)])
    from slh.runner import ServingSpecMismatch
    from slh.serving import ServingWindowClosed

    try:
        summary = execute(prep, Path(out), package=Path(package),
                          workers=workers or int(endpoint.get("workers", 1)),
                          worker_init=worker_init, in_process=in_process)
    except ServingWindowClosed as exc:
        raise halt(out, {"label": "serving_window_closed", "detail": str(exc),
                         "time_epoch": time.time()}) from None
    except ServingSpecMismatch as exc:
        raise Refused(str(exc)) from None
    if summary["halt"] is not None:
        raise halt(out, summary["halt"])
    do_export = summary["coverage"]["complete"] if export is None else export
    if do_export:
        from slh.xp.phase2 import export as X

        target = Path(out) / "export"
        if target.exists():
            shutil.rmtree(target)  # derived data, rebuilt from the run directory
        summary["export"] = {k: v for k, v in X.export(Path(out), target).items()
                             if k in ("total_bytes", "files")}
    return summary


def status(out: Path) -> dict[str, Any]:
    identity = pc.read_json(Path(out) / "identity.json")
    blocks = [[Job(**j) for j in b] for b in pc.read_json(Path(out) / "jobs.json")["blocks"]]
    resolved = resolve(flat(blocks), records(Path(out)))
    halt = pc.read_json(Path(out) / "halt.json") if (Path(out) / "halt.json").is_file() \
        else None
    return {"mode": identity["mode"], "plan_id": identity["plan_id"],
            "coverage": coverage(resolved), "halt": halt}


# --------------------------------------------------------------------------- servers


AGENT_PORT, USER_PORT = 8001, 8002
USER_NAME = "deepseek-v4-flash"
USER_REPO = "deepseek-ai/DeepSeek-V4-Flash"
USER_MAX_LEN = 65536
# vLLM recipe for DeepSeek-V4-Flash (recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash,
# read 2026-10-16): its own tokenizer mode and reasoning parser, fp8 KV cache with
# block size 256, expert parallel; on H200/B200-class nodes one 4-GPU data-parallel
# group with expert parallel per replica. Non-thinking is the model's default
# (no chat_template_kwargs); the server pins it explicitly so no request can differ.
USER_RECIPE_FLAGS = ("--tokenizer-mode", "deepseek_v4", "--reasoning-parser", "deepseek_v4",
                     "--default-chat-template-kwargs", '{"thinking":false}',
                     "--kv-cache-dtype", "fp8", "--block-size", "256")
LAYOUTS = {
    # simulator on GPUs 0-3 (data parallel 4 + expert parallel, the recipe's layout),
    # the 9B on GPUs 4-7 (4 replicas)
    "split": {"user_gpus": "0,1,2,3", "user_tp": 1, "user_dp": 4, "user_util": 0.92,
              "agent_gpus": "4,5,6,7", "agent_tp": 1, "agent_dp": 4, "agent_util": 0.90,
              "agent_max_num_seqs": 16, "workers": 48},
    # both on all 8 GPUs (as the phase-1 kit ran its teacher; 80 GB GPUs): simulator
    # TP 8 + expert parallel with a large memory share, the 9B as 2 replicas of TP 4
    "shared": {"user_gpus": "0,1,2,3,4,5,6,7", "user_tp": 8, "user_dp": 1, "user_util": 0.70,
               "agent_gpus": "0,1,2,3,4,5,6,7", "agent_tp": 4, "agent_dp": 2,
               "agent_util": 0.16, "agent_max_num_seqs": 32, "workers": 48},
}


def _set_flag(argv: list[str], flag: str, value: Any) -> None:
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv += [flag, str(value)]


def servers(layout: str = "split", *, agent_dir: str = "/path/to/models/Qwen3.5-9B",
            user_dir: str = "/path/to/models/DeepSeek-V4-Flash") -> dict[str, Any]:
    """Recommended ``vllm serve`` command lines for one 8-GPU node and the matching
    endpoint file. The 9B keeps the pinned arguments (slh/serve_spec.py); only port,
    memory share, batch width and parallelism differ. Both servers are ours
    (``self_hosted``: $0 per token)."""
    from slh import serve_spec

    s = LAYOUTS[layout]
    agent = serve_spec.vllm_args(agent_dir)
    _set_flag(agent, "--port", AGENT_PORT)
    _set_flag(agent, "--gpu-memory-utilization", s["agent_util"])
    _set_flag(agent, "--max-num-seqs", s["agent_max_num_seqs"])
    _set_flag(agent, "--tensor-parallel-size", s["agent_tp"])
    if s["agent_dp"] > 1:
        _set_flag(agent, "--data-parallel-size", s["agent_dp"])
    agent += ["--enable-prefix-caching"]
    user = ["vllm", "serve", user_dir, "--served-model-name", USER_NAME,
            "--tensor-parallel-size", str(s["user_tp"])]
    if s["user_dp"] > 1:
        user += ["--data-parallel-size", str(s["user_dp"])]
    user += ["--enable-expert-parallel", *USER_RECIPE_FLAGS,
             "--max-model-len", str(USER_MAX_LEN), "--gpu-memory-utilization",
             str(s["user_util"]), "--enable-prefix-caching", "--generation-config", "vllm",
             "--host", "127.0.0.1", "--port", str(USER_PORT)]
    endpoint = {
        "agent": {"base_url": f"http://127.0.0.1:{AGENT_PORT}/v1",
                  "model": serve_spec.SERVED_MODEL_NAME, "self_hosted": True,
                  "weights": f"{serve_spec.MODEL_REPO}@{serve_spec.MODEL_REVISION}"},
        "user": {"base_url": f"http://127.0.0.1:{USER_PORT}/v1", "model": USER_NAME,
                 "self_hosted": True, "weights": f"{USER_REPO}@<revision you downloaded>"},
        "workers": s["workers"],
        "execution": {"layout": layout, "agent_argv": agent, "user_argv": user,
                      "agent_gpus": s["agent_gpus"], "user_gpus": s["user_gpus"]}}
    return {"layout": layout, "agent": agent, "user": user, "endpoint": endpoint,
            "agent_env": {"CUDA_VISIBLE_DEVICES": s["agent_gpus"]},
            "user_env": {"CUDA_VISIBLE_DEVICES": s["user_gpus"]}}


# --------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.phase2.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dry-run", "smoke", "full"):
        p = sub.add_parser(name)
        p.add_argument("--package", type=Path, default=Path(P.PACKAGE_DIR),
                       help=f"the phase-2 package directory (default {P.PACKAGE_DIR}/; "
                            "phase 1's kit already has package/)")
        p.add_argument("--endpoint", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--workers", type=int, help="conversations in flight (default: the "
                                                   "endpoint file's workers)")
        p.add_argument("--no-export", action="store_true")
        p.add_argument("--identity-guard", default=None, metavar="ARMS",
                       help="comma-separated arm names (e.g. b1_guard) run under the "
                            "deterministic identity-before-write guard "
                            "(slh.guards.identity); off by default, recorded in the "
                            "run identity when on")
    p = sub.add_parser("status")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("check-servers", help="smoke step: both servers answer with the "
                                             "configured names and one real chat "
                                             "completion each is non-thinking")
    p.add_argument("--endpoint", type=Path, required=True)
    p = sub.add_parser("servers", help="print the recommended vllm serve commands and an "
                                       "endpoint file")
    p.add_argument("--layout", choices=sorted(LAYOUTS), default="split")
    p.add_argument("--agent-dir", default="/path/to/models/Qwen3.5-9B")
    p.add_argument("--user-dir", default="/path/to/models/DeepSeek-V4-Flash")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    # No network at the partner: never fetch litellm's price map at import.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    if args.cmd == "status":
        print(json.dumps(status(args.out), indent=1))
        return EXIT_OK
    if args.cmd == "check-servers":
        try:
            endpoint = load_endpoint(args.endpoint)
        except Refused as exc:
            print(json.dumps({"status": "refused", "reason": redact_text(exc)}, indent=1))
            return EXIT_REFUSED
        skipped: list[str] = []
        problems = check_servers(endpoint) or check_chat(endpoint, skipped=skipped)
        print(json.dumps(redact_public({"status": "ok" if not problems else "failed",
                                        "problems": problems,
                                        "chat_probe_skipped": skipped}), indent=1))
        return EXIT_OK if not problems else EXIT_HALTED
    if args.cmd == "servers":
        import shlex

        sv = servers(args.layout, agent_dir=args.agent_dir, user_dir=args.user_dir)
        for label, argv_, env in (("simulator (start it first)", sv["user"], sv["user_env"]),
                                  ("9B agent", sv["agent"], sv["agent_env"])):
            print(f"# {label}\nCUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
                  f"{shlex.join(argv_)}\n")
        print("# endpoint.yaml\n" + json.dumps(sv["endpoint"], indent=1))
        return EXIT_OK
    mode = {"dry-run": "dry", "smoke": "smoke", "full": "full"}[args.cmd]
    try:
        endpoint = load_endpoint(args.endpoint)
        guard = tuple(k.strip() for k in (args.identity_guard or "").split(",")
                      if k.strip()) or None
        summary = run(args.package, args.out, mode=mode, endpoint=endpoint,
                      workers=args.workers, export=False if args.no_export else None,
                      identity_guard=guard)
    except Refused as exc:
        print(json.dumps({"status": "refused", "reason": redact_text(exc)}, indent=1))
        return EXIT_REFUSED
    except Halted as exc:
        print(json.dumps({"status": "halted", **redact_public(exc.record),
                          "resume": "fix the cause, then rerun the same command"}, indent=1,
                         default=str))
        return EXIT_HALTED
    cov = summary["coverage"]
    print(json.dumps({"status": "complete" if cov["complete"] else "incomplete",
                      "coverage": cov, "export": summary.get("export")}, indent=1,
                     default=str))
    return EXIT_OK if cov["complete"] else EXIT_HALTED


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        raise SystemExit(main())
    raise SystemExit(130)
