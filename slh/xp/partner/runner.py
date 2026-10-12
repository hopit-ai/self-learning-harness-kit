"""Phase-1 runner for the partner (mode B): learning, arm instances, decisions, export.

    python -m slh.xp.partner.runner --package PKG --endpoint configs/xp-partner-endpoint.yaml \
        --out runs/xp-phase1 [--dry-run] [--max-cases N] [--no-export]

``PKG`` holds ``plan.json``, ``registration.json``, ``protocol.md`` and ``cases/``
(docs/xp/partner-package.md). Before anything touches a GPU the runner checks
the registration: plan hash, code hash (the endpoint config is partner-editable
and excluded), git commit, protocol and case-file hashes, and the vendored
benchmark (``slh.xp.partner.vendor``); any mismatch refuses the run.

Run identity = registration hashes + mode (``dry``/``real``) + endpoint
fingerprint. A run directory started under another identity is refused (no
mixing fake and real rows, or two endpoints). One runner per directory
(exclusive ``run.lock``). Appends are line-at-a-time and fsynced; a truncated
last line from a crash is repaired on resume and logged (``logs/repairs.jsonl``).

Stages, each resumable (rerun the same command):

1. **learning**: per task x run x writer, ``learners.PartnerLearner`` writes
   ``learning/<task>/<writer>/run-<k>/``; finished runs (``summary.json``) are
   skipped; an interrupted one restarts.
2. **arms**: instances with prompt hashes over the RENDERED prompt
   (``plan.prompt_hash``); equal prompts form one execution unit.
3. **decisions**: every unit x test question; bounded concurrency; per-case
   wall-clock limit and output cap from the plan. Errors are classified
   (``errors.classify``): a halt-class error (spend cap, credit, auth, config,
   connection, unknown) stops the run at once, writes no row for that case and
   records ``halt.json``; rerun after the fix. Only case-level model errors
   (timeout after retry, case time limit, malformed output) become final rows.
4. **export**: ``slh.xp.partner.export`` into ``<out>/export``.

``--dry-run`` uses ``FakeDecider`` and ``learners.FakeChat`` for writers and
critics: the real learning code runs end to end offline. ``EndpointDecider``
is the real client; it is exercised only by a ``--max-cases`` smoke at the partner.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Protocol

from slh.xp.partner import common as pc
from slh.xp.partner import plan as pp
from slh.xp.partner.errors import RunHalted, classify, halt_record

DECISION_SCHEMA = "xp-partner-decision-v1"
ABANDON_GRACE_S = 5.0
STALL_S = 600.0  # every slot held by abandoned calls this long: the run halts
# Every execution setting is part of the run identity: a resumed run refuses any
# change (servers' argv including TP/DP, memory share, length, eager mode and extra
# arguments; the image or vLLM install; concurrency).
ENDPOINT_FINGERPRINT = ("base_url", "model", "api_key_env", "model_revision", "vllm_version",
                        "serving_args", "replicas", "teacher", "concurrency", "execution")
COMPLETE = "complete.json"  # outer completion marker of one learning unit


class Decider(Protocol):
    def decide(self, question: dict[str, Any], unit: dict[str, Any], *, seed: int,
               deadline: float) -> dict[str, Any]: ...


class RegistrationMismatch(ValueError):
    pass


# --------------------------------------------------------------------------- deciders


class FakeDecider:
    """Deterministic actions from (prompt_hash, case_id): a clarifying question,
    a non-question message, an address write with empty arguments, or no action.
    ``delay_s`` simulates slow cases; ``fail`` (an exception) is raised instead."""

    def __init__(self, delay_s: float = 0.0, fail: BaseException | None = None) -> None:
        self.delay_s, self.fail = delay_s, fail

    def decide(self, question: dict[str, Any], unit: dict[str, Any], *, seed: int,
               deadline: float) -> dict[str, Any]:
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail is not None:
            raise self.fail
        pick = int(pc.sha_text(f"{seed}|{unit['prompt_hash']}|{question['case_id']}")[:8], 16)
        action = [
            {"type": "message", "content": "Which of these options do you mean? "
                                           "Could you please clarify?"},
            {"type": "message", "content": "I have updated your order."},
            {"type": "tool_call", "name": "modify_pending_order_address", "arguments": {},
             "extra_calls": 0},
            {"type": "none", "reason": "fake"},
        ][pick % 4]
        calls = unit["n"] * 2 - 1 if unit["class"] == "best_of_n" else 1
        return {"action": action, "input_tokens": 1000 * calls, "output_tokens": 20 * calls,
                "model_calls": calls, "capped": False, "reads": []}

    def for_learning(self, seed: int) -> Any:
        def decide(case: dict[str, Any], text: str) -> dict[str, Any]:
            unit = {"class": "notes", "n": 1, "prompt_hash": pp.prompt_hash("notes", text)}
            return self.decide(case, unit, seed=seed, deadline=time.time() + 300)
        return decide


class EndpointDecider:
    """POC 9's decision call (``slh.demo1.decide.next_action``, B1 config) on the
    partner's OpenAI-compatible endpoint, with the plan's seed and output cap.
    Best-of-n: n candidates (first greedy, then temperature ``SAMPLE_TEMPERATURE``)
    and a pairwise selector, as POC 9. Not run against a live endpoint here."""

    def __init__(self, endpoint: dict[str, Any], *, seed: int, max_output_tokens: int,
                 margin: int | None = None, growth: int | None = None) -> None:
        import yaml

        from slh.budget import ModelConfig
        from slh.demo1 import decide
        from slh.demo1.context import runner_config

        base = runner_config()
        raw = yaml.safe_load((pc.REPO / "configs/agent-vllm.yaml").read_text())
        raw.pop("serving_session", None)
        raw.update(api_base=endpoint["base_url"], model=endpoint["model"],
                   api_key_env=endpoint.get("api_key_env"), capture="messages")
        config = dataclasses.replace(base, agent=ModelConfig.from_dict(raw), seed=int(seed))
        self.decide_mod = decide
        self.client = decide.AgentClient(config)
        self.sampler = self.client.with_temperature(decide.SAMPLE_TEMPERATURE)
        from slh.xp.partner import context

        # Every 9B request is counted first: over max_model_len it is refused before
        # sending (a per-case context_limit row), never sent and never a halt.
        self.context_guard = context.make_guard(
            context.max_model_len(endpoint), margin=context.MARGIN if margin is None else margin,
            growth=growth)
        for c in (self.client, self.sampler):
            c.model = dataclasses.replace(c.model, max_tokens=int(max_output_tokens))
            context.install(c, self.context_guard)

    def decide(self, question: dict[str, Any], unit: dict[str, Any], *, seed: int,
               deadline: float) -> dict[str, Any]:
        from slh.demo1.context import B0_PROMPT_SHA256, B1_PROMPT_SHA256, b0_config
        from slh.poc9.loader import notes_for
        from slh.poc9.pipeline import _guard_backends, credit_gate_check, credit_generation

        d, client = self.decide_mod, self.client
        clients = [client, self.sampler] if unit["class"] == "best_of_n" else [client]
        b1_seed = d.b1_seed(client.config)
        # The halt latch is checked immediately before every HTTP request of this
        # case (each candidate, read turn, selector and retry), not only between cases.
        _guard_backends(clients, credit_generation())
        credit_gate_check()
        d._set_deadline(clients, deadline)
        try:
            if unit["class"] == "b0":
                return d.next_action(question, [], client, seed=b1_seed, config=b0_config(),
                                     expect_prompt_sha256=B0_PROMPT_SHA256)
            if unit["class"] == "notes":
                notes = notes_for(unit["text"])
                return d.next_action(question, notes, client, seed=b1_seed,
                                     expect_prompt_sha256=None if notes else B1_PROMPT_SHA256)
            cands = [d.next_action(question, [], client if i == 0 else self.sampler,
                                   seed=d.b1_seed(client.config, i),
                                   expect_prompt_sha256=B1_PROMPT_SHA256)
                     for i in range(unit["n"])]
            picked, picks = 0, []
            for i in range(1, len(cands)):
                choice = d.select(question, [cands[picked], cands[i]], client, seed=b1_seed)
                picks.append(choice)
                if choice["chosen"] == 1:
                    picked = i
            parts = cands + picks
            return {"action": cands[picked]["action"],
                    "capped": any(p.get("capped") for p in parts),
                    **{k: sum(p.get(k, 0) or 0 for p in parts) for k in d.COST_KEYS}}
        finally:
            d._set_deadline(clients, None)

    def for_learning(self, limit_s: float) -> Any:
        def decide(case: dict[str, Any], text: str) -> dict[str, Any]:
            from slh.poc9 import loader
            from slh.poc9.pipeline import _guard_backends, credit_generation

            _guard_backends([self.client], credit_generation())
            self.decide_mod._set_deadline([self.client], time.time() + limit_s)
            try:
                return loader.decide(case, text, self.client)
            finally:
                self.decide_mod._set_deadline([self.client], None)
        return decide


# --------------------------------------------------------------------------- startup


def load_endpoint(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for key in ("base_url", "model"):
        if not data.get(key):
            raise ValueError(f"{path}: endpoint config needs {key}")
    return data


def endpoint_sha256(endpoint: dict[str, Any]) -> str:
    return pc.sha_text(pc.canonical({k: endpoint.get(k) for k in ENDPOINT_FINGERPRINT}))


def verify_registration(package: Path, plan: dict[str, Any], root: Path) -> dict[str, Any]:
    """Refuse before any GPU work unless this checkout matches our registration."""
    from slh.xp.partner import vendor

    path = package / "registration.json"
    if not path.is_file():
        raise RegistrationMismatch("registration.json missing from the package")
    reg = pc.read_json(path)
    for splits in plan["cases"].values():
        for meta in splits.values():
            if pc.sha_file(package / meta["file"]) != meta["sha256"]:
                raise RegistrationMismatch(f"{meta['file']}: hash differs from the plan")
    if pc.sha_file(package / plan["protocol"]["file"]) != plan["protocol"]["sha256"]:
        raise RegistrationMismatch("protocol hash differs from the plan")
    git = pc.git_state(root)
    commit = git["git_commit"]
    source = (reg.get("source") or {}).get("paths") or {}
    if commit != reg.get("git_commit") and source:
        # A public-kit checkout: HEAD is the kit repository's tag, not our snapshot
        # commit. It counts as the snapshot commit only if every registered code
        # file is present with the registered blob (code_sha256 is checked too).
        missing = pc.source_mismatches(root, source)
        if missing:
            raise RegistrationMismatch(f"code files differ from the registration: "
                                       f"{missing[:5]}")
        commit = reg.get("git_commit")
    hashes = {"git_commit": commit, "git_head": git["git_commit"], "git_dirty": git["git_dirty"],
              "code_sha256": pc.code_hashes(root)["code_sha256"],
              "plan_sha256": plan["sha256"], "protocol_sha256": plan["protocol"]["sha256"],
              "case_files": pp.case_file_hashes(plan)}
    problems = [k for k in ("git_commit", "code_sha256", "plan_sha256", "protocol_sha256",
                            "case_files") if hashes[k] != reg.get(k)]
    if reg.get("plan_id") != plan["plan_id"]:
        problems.append("plan_id")
    if reg.get("export_cap_bytes") != plan["limits"]["export_cap_bytes"]:
        problems.append("export_cap_bytes")
    problems += [f"vendor: {p}" for p in vendor.problems(want=reg.get("vendor") or {})] \
        if reg.get("vendor") else ["vendor: not registered"]
    if problems:
        raise RegistrationMismatch(f"this checkout does not match registration.json: "
                                   f"{problems}")
    return hashes


@contextlib.contextmanager
def run_lock(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    handle = (out / "run.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"{out} is locked by another runner") from None
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def env_report(endpoint: dict[str, Any], *, mode: str) -> dict[str, Any]:
    from slh.xp.partner import vendor

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

    served = None
    if mode == "real":
        import urllib.request

        root = endpoint["base_url"].rstrip("/").removesuffix("/v1")
        try:
            with urllib.request.urlopen(root + "/version", timeout=10) as resp:
                served = json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal
            served = {"error": f"{type(exc).__name__}: {exc}"}
        if not isinstance(served, dict):
            served = {"value": str(served)}
    return {"dry_run": mode == "dry", "mode": mode, "time_epoch": time.time(),
            "gpu": cmd("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                       "--format=csv,noheader"),
            "endpoint": {k: endpoint.get(k) for k in ENDPOINT_FINGERPRINT},
            "endpoint_sha256": endpoint_sha256(endpoint), "served_version": served,
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {n: version(n) for n in ("vllm", "tau2", "litellm", "torch")},
            "vendor": vendor.state()}


# --------------------------------------------------------------------------- stages


DEFAULT_LEARNING_UNITS = 12


def learning_units(endpoint: dict[str, Any], *, dry_run: bool = False) -> int:
    """Learning units run concurrently (scheduling only: not in the run identity).
    The API teacher's client is not thread-safe: one unit at a time there. Bounded
    by the endpoint's concurrency, so the 9B never sees more in-flight requests."""
    teacher = endpoint.get("teacher") or {}
    if teacher.get("mode") == "api":
        return 1
    raw = os.environ.get("XP_LEARNING_UNITS") or endpoint.get("learning_units")
    units = int(raw) if raw else (4 if dry_run else DEFAULT_LEARNING_UNITS)
    cap = int(endpoint.get("concurrency") or units)
    return max(1, min(units, cap))


def context_budgets(package: Path, plan: dict[str, Any],
                    endpoint: dict[str, Any] | None) -> dict[str, dict[str, int]] | None:
    """Per task, the parameters of the full-context fit (amendment-2 rule, when the
    plan registers it), with the guard's exact accounting: ``max_model_len``, output
    cap, margin, and the largest non-system part of a request over the task's test
    questions (only their token counts are used; learning never sees a test case)."""
    from slh.xp.partner import context

    rule = (plan.get("learning") or {}).get("full_context")
    if not rule:
        return None
    if rule.get("rule") != "fit-9b-context":
        raise RegistrationMismatch(f"unknown full-context rule {rule!r}")
    limit = context.max_model_len(endpoint)
    return {task: context.fit_parameters(
                pc.read_jsonl(package / plan["cases"][task]["test"]["file"]), limit=limit,
                max_output=int(plan["limits"]["max_output_tokens"]),
                margin=int(rule.get("margin", context.MARGIN)),
                growth=int(rule.get("decision_growth",
                                    plan["limits"]["max_output_tokens"])))
            for task in plan["tasks"]}


def learning_stage(package: Path, plan: dict[str, Any], out: Path, learner: Any,
                   log: list[dict[str, Any]], *, budgets: dict[str, int] | None = None,
                   units: int = 1) -> None:
    """Every learning unit (task x run x writer) without a valid completion marker.
    ``units`` > 1 runs independent units concurrently on a bounded pool (each on its
    own learner copy: per-unit seed, state and outputs; one shared halt latch), so the
    teacher and the 9B batch their requests. Results do not depend on the schedule."""
    jobs = []
    for task in plan["tasks"]:
        splits = plan["cases"][task]
        pool = pc.read_jsonl(package / splits["train"]["file"])
        dev = pc.read_jsonl(package / splits["validation"]["file"])
        sentinel = pc.read_jsonl(package / splits["sentinel"]["file"]) \
            if "sentinel" in splits else []
        for run in range(plan["runs"]):
            train = pp.run_train(plan, task, run, pool)  # this run's own sample
            for writer in plan["writers"]:
                d = out / "learning" / task / writer / f"run-{run}"
                if unit_complete(d):
                    continue
                if d.exists():  # no valid marker: an interrupted unit is redone whole
                    log.append({"stage": "learning", "event": "redo_incomplete_unit",
                                "task": task, "writer": writer, "run": run})
                kwargs = {"task": task, "writer": writer, "run": run,
                          "critic": plan["critic"].get(writer, "none"), "train": train,
                          "dev": dev, "sentinel": sentinel, "out": d,
                          "seed": int(plan["seed"]) * 1000 + run}
                if budgets is not None:
                    kwargs["context_budget"] = budgets[task]
                jobs.append(kwargs)
    lock = threading.Lock()
    stop = threading.Event()

    def do(kwargs: dict[str, Any]) -> None:
        if stop.is_set():
            return
        unit = learner.unit_copy() if units > 1 and hasattr(learner, "unit_copy") \
            else learner
        started = time.time()
        unit.learn(**kwargs)
        mark_complete(kwargs["out"])
        with lock:
            log.append({"stage": "learning", "task": kwargs["task"],
                        "writer": kwargs["writer"], "run": kwargs["run"],
                        "wall_s": round(time.time() - started, 3)})

    if units <= 1:
        for kwargs in jobs:
            do(kwargs)
        return
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=units) as pool_:
        futures = [pool_.submit(do, kwargs) for kwargs in jobs]
        for fut in futures:
            try:
                fut.result()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                stop.set()  # units not yet started do not start
                errors.append(exc)
    if errors:
        halts = [e for e in errors if isinstance(e, RunHalted)]
        raise (halts or errors)[0]


_SUMMARY_HASH = {"skill.md": "skill_sha256", "generic.md": "generic_sha256",
                 "evidence.md": "evidence_sha256", "search-control.md": "search_control_sha256"}


def _unit_problems(d: Path) -> list[str]:
    """Why a learning unit's artifacts are not a finished unit: the learner's final
    summary (search control and sentinel selection included) and every text file,
    each equal to the hash the summary records; no leftover working directory."""
    summary_path = Path(d) / "summary.json"
    if not summary_path.is_file():
        return ["summary.json missing"]
    try:
        summary = pc.read_json(summary_path)
    except ValueError:
        return ["summary.json unreadable"]
    problems = [k for k in ("selection", "search_control", "budgets", "skill_sha256",
                            "evidence_sha256") if k not in summary]
    if (Path(d) / "search").exists():
        problems.append("search/ working directory left")
    for name, key in _SUMMARY_HASH.items():
        path = Path(d) / name
        if not path.is_file():
            problems.append(f"{name} missing")
            continue
        text = path.read_text(encoding="utf-8")
        want = summary.get(key)
        have = pc.sha_text(text) if text or name == "evidence.md" else None
        if want != have:
            problems.append(f"{name} differs from the summary")
    return problems


def mark_complete(d: Path) -> None:
    """Validate the unit, then write its completion marker atomically (tmp, fsync,
    rename). Written only by the runner, after the learner returned."""
    problems = _unit_problems(d)
    if problems:
        raise RuntimeError(f"{d}: learning unit not complete: {problems}")
    for p in sorted(Path(d).iterdir()):  # unit files are written without fsync
        if p.is_file():
            with p.open("rb") as handle:
                os.fsync(handle.fileno())
    files = {p.name: pc.sha_file(p) for p in sorted(Path(d).iterdir())
             if p.is_file() and p.name != COMPLETE}
    body = json.dumps({"schema": "xp-learning-unit-complete-v1", "files": files},
                      sort_keys=True, indent=1) + "\n"
    tmp = Path(d) / (COMPLETE + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, Path(d) / COMPLETE)
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def unit_complete(d: Path) -> bool:
    """A unit counts as finished only with a marker whose file hashes still hold."""
    marker = Path(d) / COMPLETE
    if not marker.is_file():
        return False
    try:
        files = pc.read_json(marker)["files"]
    except (ValueError, KeyError, TypeError):
        return False
    present = {p.name for p in Path(d).iterdir() if p.is_file() and p.name != COMPLETE}
    if present != set(files):
        return False
    return all(pc.sha_file(Path(d) / n) == h for n, h in files.items()) \
        and not _unit_problems(d)


def arm_stage(plan: dict[str, Any], out: Path) -> dict[tuple[str, str], dict[str, Any]]:
    def text_for(task: str, run: int, writer: str, kind: str) -> str:
        unit = out / "learning" / task / writer / f"run-{run}"
        if not unit_complete(unit):
            raise FileNotFoundError(f"{unit}: learning unit has no valid completion marker")
        path = unit / pp.TEXT_FILES[kind]
        if not path.is_file():
            raise FileNotFoundError(f"{path}: learning stage did not finish")
        return path.read_text(encoding="utf-8")

    insts = pp.instances(plan, text_for)
    units: dict[tuple[str, str], dict[str, Any]] = {}
    for inst in insts:
        key = (inst["task"], inst["prompt_hash"])
        if key not in units:
            units[key] = {"task": inst["task"], "prompt_hash": inst["prompt_hash"],
                          "class": pp.prompt_key(inst["kind"], inst["_text"], inst["n"])["class"],
                          "n": inst["n"] or 1,
                          "text": inst["_text"] if inst["kind"] not in ("b0", "best_of_n")
                          else "", "first": {k: inst[k] for k in ("arm", "run", "writer")}}
    rows = [{k: v for k, v in inst.items() if k != "_text"} for inst in insts]
    (out / "arms.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                                    encoding="utf-8")
    return units


def decision_stage(package: Path, plan: dict[str, Any], out: Path,
                   units: dict[tuple[str, str], dict[str, Any]], decider: Decider, *,
                   concurrency: int, max_cases: int | None, log: list[dict[str, Any]],
                   smoke: dict[str, Any] | None = None) -> dict[str, Any]:
    limit_s = float(plan["limits"]["case_limit_s"])
    questions: dict[str, list[dict[str, Any]]] = {}
    for task in plan["tasks"]:
        rows = pc.read_jsonl(package / plan["cases"][task]["test"]["file"])
        for q in rows:
            pc.assert_no_gold(q, f"question {q['case_id']}")
            if pc.case_hash(q) != q["case_hash"]:
                raise ValueError(f"{q['case_id']}: question differs from its case_hash")
        if smoke:
            keep = set(pp.smoke_ids([q["case_id"] for q in rows], smoke))
            rows = [q for q in rows if q["case_id"] in keep]
        questions[task] = rows[:max_cases] if max_cases else rows
    path = out / "decisions.jsonl"
    done = {(r["task"], r["prompt_hash"], r["case_id"]) for r in pc.read_jsonl(path)}
    pending = [(unit, q) for (task, _), unit in sorted(units.items())
               for q in questions[task]
               if (task, unit["prompt_hash"], q["case_id"]) not in done]
    seed = int(plan["seed"])
    halted = threading.Event()
    halt_info: dict[str, Any] = {}
    # One slot per in-flight model call. A call abandoned at the case limit keeps
    # its slot until it really returns, so in-flight calls never exceed the
    # registered concurrency.
    slots = threading.BoundedSemaphore(max(1, concurrency))
    stall_s = max(STALL_S, 4 * limit_s)

    latch_lock = threading.Lock()

    def latch(error: BaseException | str, unit: dict[str, Any], q: dict[str, Any],
              **extra: Any) -> None:
        """Record the first halt and stop dispatch (callable from any thread)."""
        with latch_lock:
            if halted.is_set():
                return
            if isinstance(error, str):
                halt_info.update({"event": "halt", "class": "halt", "error": error, **extra,
                                  "task": unit["task"], "case_id": q["case_id"]})
            else:
                halt_info.update(halt_record(error, task=unit["task"], case_id=q["case_id"],
                                             prompt_hash=unit["prompt_hash"], **extra))
            from slh.poc9.pipeline import latch_credit

            # Every halt (not only credit) closes the request gate: calls already in
            # flight in other cases send nothing further.
            latch_credit(f"halt: {halt_info.get('label')}: {halt_info.get('error')}")
            halted.set()

    def one(unit: dict[str, Any], q: dict[str, Any]) -> dict[str, Any] | None:
        waited = 0.0
        while not slots.acquire(timeout=0.2):
            if halted.is_set():
                return None
            waited += 0.2
            if waited >= stall_s:
                latch("abandoned calls hold every concurrency slot", unit, q, label="stalled")
                return None
        if halted.is_set():
            slots.release()
            return None
        started = time.perf_counter()
        deadline = time.time() + limit_s
        box: dict[str, Any] = {}

        def call() -> None:
            try:
                box["value"] = decider.decide(q, unit, seed=seed, deadline=deadline)
            except BaseException as exc:  # noqa: BLE001 - classified here and below
                box["error"] = exc
                if classify(exc)[0] == "halt":
                    latch(exc, unit, q)  # before the slot is released: nothing else starts
            finally:
                slots.release()

        thread = threading.Thread(target=call, daemon=True)
        thread.start()
        thread.join(limit_s + ABANDON_GRACE_S)
        timed_out = thread.is_alive()
        wall = round(time.perf_counter() - started, 3)
        if timed_out:
            # Abandoned at the case limit: its row waits until the call really
            # returns, so a late systematic failure (e.g. a 401) halts the run and
            # leaves no final row for this case.
            thread.join(stall_s)
            if thread.is_alive():
                latch("an abandoned call did not return", unit, q, label="stalled")
                return None
        error = box.get("error")
        error_class = None
        if error is not None:
            kind, label = classify(error)
            if kind == "halt":
                return None  # latched in call(); the case runs again after the fix
            error_class = label
        if halted.is_set() and timed_out:
            return None  # the run halted while this call was abandoned: rerun it
        value = {} if timed_out else (box.get("value") or {})
        return {"schema": DECISION_SCHEMA, "task": unit["task"], "case_id": q["case_id"],
                "case_hash": q["case_hash"], "prompt_hash": unit["prompt_hash"],
                **unit["first"],
                "action": value.get("action") or {"type": "none", "reason": "error"},
                "tokens": {"prompt": int(value.get("input_tokens", 0) or 0),
                           "completion": int(value.get("output_tokens", 0) or 0)},
                "model_calls": int(value.get("model_calls", 0) or 0),
                "capped": bool(value.get("capped")), "case_time_limit": timed_out,
                "error": "case_time_limit" if timed_out else
                         (f"{type(error).__name__}: {error}" if error else None),
                "error_class": "case_time_limit" if timed_out else error_class,
                "wall_s": wall}

    started = time.time()
    written = 0
    # Group commit: fsync every few seconds, not per row (slow shared filesystems);
    # a crash loses at most the unsynced rows, which resume simply redoes.
    with pc.GroupAppender(path) as sink, \
            ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        queue = iter(pending)
        running = set()

        def fill() -> None:
            for unit, q in queue:
                running.add(pool.submit(one, unit, q))
                if len(running) >= concurrency * 2 or halted.is_set():
                    break

        fill()
        while running:
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            rows = [r for r in (f.result() for f in finished) if r is not None]
            sink.write(rows)
            written += len(rows)
            if not halted.is_set():
                fill()
    log.append({"stage": "decisions", "pending": len(pending), "written": written,
                "wall_s": round(time.time() - started, 3)})
    if halted.is_set():
        pc.write_json(out / "halt.json", halt_info)
        pc.append_jsonl(out / "logs" / "halts.jsonl", [halt_info])
        raise RunHalted(halt_info.get("label", "halt"), halt_info.get("error", ""))
    return {"pending": len(pending), "written": written}


def write_logs(out: Path, log: list[dict[str, Any]]) -> None:
    rows = pc.read_jsonl(out / "decisions.jsonl")
    costs: dict[str, dict[str, Any]] = {}
    for r in rows:
        c = costs.setdefault(r["task"], {"task": r["task"], "decisions": 0, "model_calls": 0,
                                         "prompt_tokens": 0, "completion_tokens": 0,
                                         "errors": 0, "time_limits": 0})
        c["decisions"] += 1
        c["model_calls"] += r.get("model_calls", 0)
        c["prompt_tokens"] += r["tokens"]["prompt"]
        c["completion_tokens"] += r["tokens"]["completion"]
        c["errors"] += bool(r.get("error"))
        c["time_limits"] += bool(r.get("case_time_limit"))
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "logs" / "costs.jsonl").write_text(
        "".join(json.dumps(c, sort_keys=True) + "\n" for c in costs.values()), encoding="utf-8")
    errors = [{k: r.get(k) for k in ("task", "case_id", "prompt_hash", "error", "error_class")}
              for r in rows if r.get("error")]
    (out / "logs" / "errors.jsonl").write_text(
        "".join(json.dumps(e, sort_keys=True) + "\n" for e in errors), encoding="utf-8")
    pc.append_jsonl(out / "logs" / "stages.jsonl", log)


def _identity(out: Path, hashes: dict[str, Any]) -> None:
    path = out / "hashes.json"
    if path.is_file():
        old = pc.read_json(path)
        volatile = ("git_dirty", "git_head")
        if {k: v for k, v in old.items() if k not in volatile} != \
                {k: v for k, v in hashes.items() if k not in volatile}:
            raise RegistrationMismatch(
                f"{out} was started under another identity (code, plan, cases, mode or "
                "endpoint); use a new run directory")
    pc.write_json(path, hashes)


def _recover(out: Path) -> None:
    repairs = []
    for path in [out / "decisions.jsonl", *sorted((out / "logs").glob("*.jsonl"))]:
        fixed = pc.repair_jsonl(path)
        if fixed:
            repairs.append({**fixed, "time_epoch": time.time()})
    if repairs:
        pc.append_jsonl(out / "logs" / "repairs.jsonl", repairs)
    halt = out / "halt.json"
    if halt.is_file():
        record = pc.read_json(halt)
        pc.append_jsonl(out / "logs" / "resumes.jsonl",
                        [{"event": "resume_after_halt", "label": record.get("label"),
                          "time_epoch": time.time()}])
        halt.unlink()


def build_real(endpoint: dict[str, Any], plan: dict[str, Any]) -> tuple[Any, Any]:
    from slh.poc9.writers import TeacherChat, agent_chat, load_teacher
    from slh.xp.partner import learners

    rule = (plan.get("learning") or {}).get("full_context") or {}
    decider = EndpointDecider(endpoint, seed=int(plan["seed"]),
                              max_output_tokens=int(plan["limits"]["max_output_tokens"]),
                              margin=rule.get("margin"), growth=rule.get("decision_growth"))
    teacher_cfg = endpoint.get("teacher") or {}
    needs_teacher = any(w.startswith("T") for w in plan["writers"]) or \
        "teacher" in plan["critic"].values()
    teacher = None
    if needs_teacher:
        if teacher_cfg.get("mode") == "api":
            teacher = TeacherChat(load_teacher())
        elif teacher_cfg.get("base_url") and teacher_cfg.get("model"):
            teacher = learners.SelfHostedChat(
                teacher_cfg, max_tokens=int(plan["limits"]["max_output_tokens"]))
        else:
            raise RegistrationMismatch("the plan needs a teacher: set teacher.base_url and "
                                       "teacher.model, or teacher.mode: api")
    from slh.budget import MAX_PAUSES
    from slh.poc9.pipeline import _guard_backends, credit_generation

    raw_agent = agent_chat(decider.client)

    def agent(system: str, user: str, purpose: str = "write") -> Any:
        _guard_backends([decider.client], credit_generation())  # gate before each request
        return raw_agent(system, user, purpose=purpose)

    # Transport dispatches per writer call: the meter's retries + 1 + its pauses;
    # the writer budget reserves this worst case before every call.
    agent.max_dispatches = decider.client.model.retries + 1 + MAX_PAUSES
    if teacher is not None and not hasattr(teacher, "max_dispatches"):
        teacher.max_dispatches = teacher.model.retries + 1 + teacher.meter.max_pauses
    if teacher is not None and not getattr(teacher, "per_thread", False):
        teacher.backend = learners.guard_backend(teacher.backend)
    registered = (plan.get("learning") or {}).get("dispatch") or {}
    for role, chat in (("agent", agent), ("teacher", teacher)):
        if chat is not None and chat.max_dispatches > int(registered.get(role, 0)):
            raise RegistrationMismatch(
                f"the {role} chat reserves {chat.max_dispatches} dispatches per call; the "
                f"plan registers {registered.get(role)} (budgets would not cover it)")
    learner = learners.PartnerLearner(
        agent_chat=agent, teacher_chat=teacher,
        decide=decider.for_learning(float(plan["limits"]["case_limit_s"])),
        settings=learners.LearnSettings.from_plan(plan))
    return decider, learner


def build_dry(plan: dict[str, Any], decider: Any = None) -> tuple[Any, Any]:
    from slh.xp.partner import learners

    decider = decider or FakeDecider()
    fake = FakeDecider()
    learner = learners.PartnerLearner(agent_chat=learners.FakeChat(),
                                      teacher_chat=learners.FakeChat(),
                                      decide=fake.for_learning(int(plan["seed"])),
                                      settings=learners.LearnSettings.from_plan(plan))
    return decider, learner


def preflight(package: Path, out: Path, *, endpoint: dict[str, Any], dry_run: bool = False,
              smoke: bool | str = False, root: Path = pc.REPO) -> dict[str, Any]:
    """Verification only, before any server starts: registration (plan, code, cases,
    protocol, benchmark copy), and that ``out`` (if started) has this run identity and
    no live runner. Raises RegistrationMismatch; writes nothing."""
    package, out = Path(package), Path(out)
    plan = pp.load(package / "plan.json")
    hashes = verify_registration(package, plan, root)
    hashes["run_identity"] = {"mode": "dry" if dry_run else "real",
                              "endpoint_sha256": endpoint_sha256(endpoint),
                              "smoke": pp.smoke_setting(smoke)}
    path = out / "hashes.json"
    if path.is_file():
        old = pc.read_json(path)
        volatile = ("git_dirty", "git_head")
        diff = sorted(k for k in set(old) | set(hashes) if k not in volatile
                      and old.get(k) != hashes.get(k))
        if diff:
            raise RegistrationMismatch(f"{out} was started under another identity ({diff}); "
                                       "keep the settings or use a new run directory")
    if (out / "run.lock").is_file():
        with (out / "run.lock").open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except BlockingIOError:
                raise RegistrationMismatch(f"{out} is in use by another runner") from None
    done = sum(unit_complete(d) for d in (out / "learning").glob("*/*/run-*"))
    return {"status": "preflight ok", "plan_id": plan["plan_id"],
            "resume": path.is_file(), "learning_units_complete": done,
            "endpoint_sha256": hashes["run_identity"]["endpoint_sha256"]}


def run_phase1(package: Path, out: Path, *, endpoint: dict[str, Any], dry_run: bool = False,
               decider: Decider | None = None, learner: Any = None,
               max_cases: int | None = None, export: bool = True,
               root: Path = pc.REPO, export_kwargs: dict[str, Any] | None = None,
               smoke: bool | str = False) -> dict[str, Any]:
    """``smoke``: the rehearsal setting (``plan.SMOKE``: the first run, the first
    test cases by id, a short lesson loop); recorded in the run identity, so a
    smoke run never mixes with a full one, and the registered plan is unchanged."""
    from slh.poc9.pipeline import _new_credit_generation

    package, out = Path(package), Path(out)
    plan = pp.load(package / "plan.json")
    hashes = verify_registration(package, plan, root)  # before any GPU work
    mode = "dry" if dry_run else "real"
    smoke_spec = pp.smoke_setting(smoke)
    hashes["run_identity"] = {"mode": mode, "endpoint_sha256": endpoint_sha256(endpoint),
                              "smoke": smoke_spec}
    with run_lock(out):
        _identity(out, hashes)
        _recover(out)
        pc.write_json(out / "plan.json", plan)
        pc.write_json(out / "env.json", env_report(endpoint, mode=mode))
        test_ids = sorted(q["case_id"] for t in plan["tasks"]
                          for q in pc.read_jsonl(package / plan["cases"][t]["test"]["file"]))
        pc.write_json(out / "test-ids.json", test_ids)
        _new_credit_generation()  # this run's own credit latch
        if dry_run:
            d, lr = build_dry(plan, decider)
        elif decider is None or learner is None:
            d, lr = build_real(endpoint, plan)
        else:
            d, lr = decider, learner
        decider, learner = decider or d, learner or lr
        if smoke_spec and getattr(learner, "settings", None) is not None:
            learner.settings.ops = {**learner.settings.ops, **smoke_spec["ops"]}
        run_plan = pp.effective(plan, smoke_spec)
        log: list[dict[str, Any]] = []
        budgets = context_budgets(package, run_plan, endpoint)
        units = learning_units(endpoint, dry_run=dry_run)
        log.append({"stage": "learning_settings", "units": units,
                    **{f"context_budget_{t}": b["budget"] for t, b in (budgets or {}).items()},
                    **{f"context_fixed_{t}": b["fixed_tokens"]
                       for t, b in (budgets or {}).items()}})
        try:
            learning_stage(package, run_plan, out, learner, log, budgets=budgets, units=units)
            units = arm_stage(run_plan, out)
            concurrency = int(endpoint.get("concurrency") or plan["limits"].get("concurrency", 8))
            result = decision_stage(package, plan, out, units, decider,
                                    concurrency=concurrency, max_cases=max_cases, log=log,
                                    smoke=smoke_spec)
        except RunHalted as exc:
            if not (out / "halt.json").is_file():
                record = {"event": "halt", "label": exc.reason, "error": exc.detail[:500],
                          "stage": "learning"}
                pc.write_json(out / "halt.json", record)
                pc.append_jsonl(out / "logs" / "halts.jsonl", [record])
            write_logs(out, log)
            raise
        write_logs(out, log)
        summary = {"units": len(units), "mode": mode, **result}
        if export:
            from slh.xp.partner.export import export as do_export

            target = out / "export"
            if target.exists():
                shutil.rmtree(target)  # derived data; rebuilt from the run directory
            summary["export"] = do_export(out, target, **(export_kwargs or {}))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.partner.runner")
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--endpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-cases", type=int)
    ap.add_argument("--smoke", nargs="?", const="smoke", default=None,
                    choices=sorted(pp.SMOKES),
                    help="rehearsal setting: first run, first test cases, short lesson loop")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--preflight", action="store_true",
                    help="verify only (registration, benchmark, run identity); start nothing")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if args.preflight:
        try:
            print(json.dumps(preflight(args.package, args.out, endpoint=load_endpoint(
                args.endpoint), dry_run=args.dry_run, smoke=args.smoke), indent=1))
        except RegistrationMismatch as exc:
            print(json.dumps({"status": "refused", "reason": str(exc)}, indent=1))
            return 4
        return 0
    try:
        summary = run_phase1(args.package, args.out, endpoint=load_endpoint(args.endpoint),
                             dry_run=args.dry_run, max_cases=args.max_cases, smoke=args.smoke,
                             export=not args.no_export)
    except RunHalted as exc:
        print(json.dumps({"status": "halted", "reason": exc.reason, "detail": exc.detail,
                          "resume": "fix the cause, then rerun the same command"}, indent=1))
        return 3
    except RegistrationMismatch as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, indent=1))
        return 4
    if "export" in summary:
        ex = summary.pop("export")
        summary["export_bytes"] = ex["total_bytes"]
        summary["status"] = "export ok"
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
