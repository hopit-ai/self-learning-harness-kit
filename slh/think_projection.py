"""Strict public think projection, built OFFLINE after a run has finished.

``python -m slh think-project RUN_DIR`` (re)builds ``RUN_DIR/think_public.jsonl``
from the run's public ``episodes.jsonl`` and ``calls.jsonl`` and the pinned
tau2 task data, and records ``think_public.summary.json`` (builder version,
source file hashes, withheld counts, output hash). It never reads ``private/``
and never shapes an episode, so it is NOT an execution (runtime) module: the
execution surface (``slh.execution.RUNTIME_MODULES``) excludes it, and runs
recorded at an earlier code revision (window 1a at 1711fd0) get the strict
projection afterwards without changing their execution identity. The runner's
own end-of-run projection (``slh.runner.write_think_projection``) is the
weaker 1711fd0 version; the learner should only consume a projection whose
summary names this builder (``BUILDER``) and whose output hash matches.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from slh.trace import GradingLeak, check_no_grading_fields, read_jsonl_tolerant

BUILDER = "slh.think_projection/strict-v1"


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(r, ensure_ascii=True, sort_keys=True) + "\n"
                             for r in records))
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _sha_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


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


THINK_MIN_VALUE = 4  # hidden values (gold arguments, list elements, short fields)
THINK_MIN_DIGITS = 6  # digit runs inside hidden text


def _flatten_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [v for item in value.values() for v in _flatten_values(item)]
    if isinstance(value, list | tuple):
        return [v for item in value for v in _flatten_values(item)]
    if value is None or isinstance(value, bool):
        return []
    return [str(value)]


def _hidden_values(task: Any) -> set[str]:
    """Individual hidden VALUES (normalized): every gold action argument value
    and list element, every communicate_info item, and inside every hidden
    text the digit runs (>= 6) and identifier-like tokens (letters and digits,
    >= 4 characters). A thought containing one is withheld unless that exact
    value was already public in the episode before the thought."""
    values: set[str] = set()
    criteria = getattr(task, "evaluation_criteria", None)
    if criteria is not None:
        for action in criteria.actions or []:
            values.update(_flatten_values(action.arguments or {}))
        values.update(str(item) for item in criteria.communicate_info or [])
    for text in _hidden_task_strings(task):
        values.update(re.findall(rf"\d{{{THINK_MIN_DIGITS},}}", text))
        values.update(token for token in re.findall(r"[#\w-]+", text)
                      if len(token) >= THINK_MIN_VALUE and re.search(r"\d", token)
                      and re.search(r"[A-Za-z]", token))
    normalized = {_think_norm(v) for v in values}
    return {v for v in normalized if len(v.replace(" ", "")) >= THINK_MIN_VALUE}


def _public_before(episode: dict[str, Any], turn: int | None) -> str:
    """Normalized public text of the episode before the thought's turn (its
    conversation, tool calls and tool results); nothing if the turn is unknown."""
    if not isinstance(turn, int):
        return ""
    shown = [m for m in episode.get("messages") or [] if isinstance(m.get("turn"), int)
             and m["turn"] < turn]
    return _think_norm(json.dumps(shown, ensure_ascii=False))


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


def _thought_withheld(text: str, secret_windows: set[str], secret_short: set[str],
                      secret_values: frozenset[str] | set[str] = frozenset()) -> str | None:
    """Reason to withhold a thought, or None. When in doubt, withhold."""
    from slh.trace import FORBIDDEN_KEYS

    lowered = text.lower()
    if any(key in lowered for key in FORBIDDEN_KEYS if "_" in key):
        return "grading_key_name"
    if not _embedded_json_keys_ok(text):
        return "embedded_grading_json"
    normalized = _think_norm(text)
    if not _think_windows(f" {normalized} ").isdisjoint(secret_windows):
        return "hidden_task_text"
    padded = f" {normalized} "
    if any(f" {short} " in padded for short in secret_short):
        return "hidden_task_text"
    if any(f" {value} " in padded for value in secret_values):
        return "hidden_value"
    return None


def write_think_projection(out: Path, tasks: list[Any]) -> dict[str, Any]:
    """Build the strict projection (``project``) and write ``think_public.jsonl``."""
    rows, summary = project(out, tasks)
    _write_jsonl_atomic(out / "think_public.jsonl", rows)
    return summary


def _canonical_lines(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=True, sort_keys=True) + "\n"
                   for r in rows).encode("ascii")


def project(out: Path, tasks: list[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    strings for hidden fragments of 6-11 characters; and every individual
    hidden VALUE (each gold action argument value and list element, each
    communicate_info item, digit runs >= 6 and identifier-like tokens in hidden
    text, >= 4 characters). Only text the episode had already shown publicly
    BEFORE the thought's turn (conversation, tool calls and results) and the
    domain policy is exempt; a thought with an unknown turn gets no exemption. Identifiers are aliased
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
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"file": "think_public.jsonl", "thoughts": 0,
                               "episodes_with_thoughts": 0, "withheld": {}, "unparseable": 0}

    hidden_cache: dict[str, tuple[list[str], set[str]]] = {}

    def secrets_for(episode: dict[str, Any], turn: Any
                    ) -> tuple[set[str], set[str], set[str]]:
        """Hidden windows, short strings and values not yet public before ``turn``."""
        nonlocal policy_windows
        if policy_windows is None:
            domain = json.loads((out / "manifest.json").read_text())["domain"] \
                if (out / "manifest.json").exists() else "retail"
            policy = registry.get_env_constructor(domain)().get_policy()
            policy_windows = _think_windows(f" {_think_norm(policy)} ")
        task_id = episode["task_id"]
        if task_id not in hidden_cache:
            task = by_task[task_id]
            hidden_cache[task_id] = ([_think_norm(h) for h in _hidden_task_strings(task)],
                                     _hidden_values(task))
        hidden, values = hidden_cache[task_id]
        public = _public_before(episode, turn)
        padded = f" {public} "
        # Padded on both sides so a value at either end of a text still forms windows.
        visible = _think_windows(padded) | policy_windows
        windows = set().union(*(_think_windows(f" {h} ") for h in hidden)) - visible
        short = {h for h in hidden if THINK_MIN_SECRET <= len(h) < THINK_WINDOW
                 and f" {h} " not in padded}
        return windows, short, {v for v in values if f" {v} " not in padded}

    for call in read_jsonl_tolerant(out / "calls.jsonl")[0]:
        execution = call.get("execution_id")
        episode = final.get(execution) if execution else None
        if call.get("role") != "agent" or call.get("parse_ok") is not True or episode is None:
            continue
        windows, short, values = secrets_for(episode, call.get("turn"))
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
            reason = _thought_withheld(text, windows, short, values)
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
    return rows, summary


def build(run_dir: str | Path) -> dict[str, Any]:
    """Rebuild the strict projection of a finished run (public files + task data)."""
    from slh.runner import load_tasks

    out = Path(run_dir)
    for path in (out, out / "manifest.json", out / "episodes.jsonl", out / "calls.jsonl"):
        if path.is_symlink():
            raise ValueError(f"think projection refuses symlinked input {path}")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") not in ("complete", "complete_with_errors"):
        raise ValueError(f"run {out.name} is not finished ({manifest.get('status')})")
    if manifest.get("arm") != "think":
        raise ValueError(f"run {out.name} is not a think-arm run")
    sources = {name: _sha_file(out / name) for name in ("episodes.jsonl", "calls.jsonl")}
    summary = write_think_projection(out, load_tasks(manifest["domain"], manifest["task_ids"]))
    record = {"builder": BUILDER, "run_id": manifest.get("run_id"), "sources_sha256": sources,
              "output_sha256": _sha_file(out / "think_public.jsonl"), **summary}
    tmp = out / f"think_public.summary.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out / "think_public.summary.json")
    return record


def verify(run_dir: str | Path) -> dict[str, Any]:
    """REFUSE (ValueError) unless the run's ``think_public.jsonl`` is exactly the
    strict projection: the summary exists and names this builder, the run's
    public sources and the output are unchanged since it was built, and the
    file is byte-identical to a fresh strict rebuild from the run's public
    files and the task data (a tampered or hand-edited row fails)."""
    from slh.runner import load_tasks

    out = Path(run_dir)
    for path in (out, out / "think_public.jsonl", out / "think_public.summary.json"):
        if path.is_symlink():
            raise ValueError(f"think projection refuses symlinked input {path}")
    try:
        record = json.loads((out / "think_public.summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError(f"run {out.name}: strict think projection summary is missing or "
                         "damaged; run python -m slh think-project") from None
    sources = {name: _sha_file(out / name) for name in ("episodes.jsonl", "calls.jsonl")}
    if record.get("builder") != BUILDER or record.get("sources_sha256") != sources or \
            record.get("output_sha256") != _sha_file(out / "think_public.jsonl"):
        raise ValueError(f"run {out.name}: think_public.jsonl is not the current strict "
                         "projection; run python -m slh think-project")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rows, _ = project(out, load_tasks(manifest["domain"], manifest["task_ids"]))
    if _canonical_lines(rows) != (out / "think_public.jsonl").read_bytes():
        raise ValueError(f"run {out.name}: think_public.jsonl differs from a fresh strict "
                         "rebuild")
    return record
