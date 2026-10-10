"""Typed allowlists for every exported file (shared by the exporter and the checker).

The exporter keeps only the listed fields and refuses a value of the wrong type;
the checker refuses unknown fields as well. Free-form JSON is allowed only in
learning rows (``row``), conversation transcripts and the environment report;
those are scanned for answer signatures on our side (``slh.xp.partner.check``).
Test-split rows never carry ``kind`` or any answer field.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from slh.xp.partner.common import TASKS

Check = Callable[[Any], bool]
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
REQUIRED_FILES = ("decisions.jsonl.gz", "arms.jsonl.gz", "learning.jsonl.gz",
                  "skills.jsonl.gz", "logs.jsonl.gz", "env.json")
OPTIONAL_FILES = ("conversations.jsonl.gz",)
ARM_KINDS = ("b0", "notes", "learned", "generic", "full_context", "search_control",
             "best_of_n")
TEXT_KINDS = ("learned", "generic", "full_context", "search_control")
LEARNING_SOURCES = ("trace", "patches", "episodes", "search_trace", "acquisition", "summary")


class SchemaError(ValueError):
    pass


def s(v: Any) -> bool:
    return isinstance(v, str)


def opt(check: Check) -> Check:
    return lambda v: v is None or check(v)


def integer(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def boolean(v: Any) -> bool:
    return isinstance(v, bool)


def hex64(v: Any) -> bool:
    return isinstance(v, str) and bool(_HEX64.fullmatch(v))


def hex40(v: Any) -> bool:
    return isinstance(v, str) and bool(_HEX40.fullmatch(v))


def one_of(*values: Any) -> Check:
    return lambda v: v in values


def json_value(v: Any) -> bool:
    if v is None or isinstance(v, str | bool | int | float):
        return True
    if isinstance(v, list):
        return all(json_value(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and json_value(x) for k, x in v.items())
    return False


def json_object(v: Any) -> bool:
    return isinstance(v, dict) and json_value(v)


def scalar(v: Any) -> bool:
    return v is None or isinstance(v, str | bool | int | float)


def flat(v: Any) -> bool:
    return isinstance(v, dict) and all(isinstance(k, str) and scalar(x) for k, x in v.items())


ACTION_FIELDS: dict[str, Check] = {"type": one_of("tool_call", "message", "none"),
                                   "name": s, "arguments": json_object, "extra_calls": integer,
                                   "content": opt(s), "reason": opt(s)}


def action(v: Any) -> bool:
    return isinstance(v, dict) and "type" in v and set(v) <= set(ACTION_FIELDS) and \
        all(ACTION_FIELDS[k](x) for k, x in v.items())


def tokens(v: Any) -> bool:
    return isinstance(v, dict) and set(v) == {"prompt", "completion"} and \
        all(integer(x) for x in v.values())


def str_list(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


TASK = one_of(*TASKS)
# file -> (required fields, optional fields)
FILES: dict[str, tuple[dict[str, Check], dict[str, Check]]] = {
    "decisions.jsonl.gz": (
        {"task": TASK, "case_id": s, "case_hash": hex64, "prompt_hash": hex64, "arm": s,
         "run": opt(integer), "writer": opt(s), "action": action, "tokens": tokens,
         "model_calls": integer, "wall_s": number, "error": opt(s), "capped": boolean,
         "case_time_limit": boolean},
        {"error_class": opt(s), "reported_correct": boolean}),
    "arms.jsonl.gz": (
        {"task": TASK, "run": opt(integer), "writer": opt(s), "arm": s,
         "kind": one_of(*ARM_KINDS), "n": opt(integer), "prompt_hash": hex64,
         "skill_sha256": opt(hex64)}, {}),
    "skills.jsonl.gz": (
        {"task": TASK, "writer": s, "run": integer, "kind": one_of(*TEXT_KINDS), "text": s,
         "sha256": opt(hex64)}, {}),
    "learning.jsonl.gz": (
        {"task": TASK, "writer": s, "run": integer, "source": one_of(*LEARNING_SOURCES),
         "row": json_object}, {}),
    "logs.jsonl.gz": ({"name": s, "row": flat}, {}),
    "conversations.jsonl.gz": (
        {"conversation_id": s, "task": TASK, "arm": s, "run": opt(integer), "writer": opt(s),
         "transcript_included": boolean},
        {"scenario_id": opt(s), "actions": lambda v: isinstance(v, list) and all(
            action(a) for a in v), "reward": opt(number), "db_hash": opt(s),
         "fidelity_flags": str_list, "transcript": lambda v: isinstance(v, list) and
         json_value(v)}),
}
ENV_FIELDS: dict[str, Check] = {
    "dry_run": boolean, "mode": one_of("dry", "real"), "time_epoch": number,
    "gpu": opt(s), "endpoint": json_object, "endpoint_sha256": hex64,
    "served_version": lambda v: v is None or json_object(v), "python": s, "platform": s,
    "packages": flat, "vendor": json_object}
HASH_FIELDS: dict[str, Check] = {
    "git_commit": opt(hex40), "git_head": opt(hex40), "git_dirty": opt(boolean),
    "code_sha256": hex64,
    "plan_sha256": hex64, "protocol_sha256": hex64,
    "case_files": lambda v: isinstance(v, dict) and all(isinstance(k, str) and hex64(x)
                                                        for k, x in v.items()),
    "run_identity": lambda v: isinstance(v, dict) and
    set(v) == {"mode", "endpoint_sha256", "smoke"} and v["mode"] in ("dry", "real")
    and hex64(v["endpoint_sha256"]) and (v["smoke"] is None or json_object(v["smoke"]))}


def pick(name: str, row: dict[str, Any]) -> dict[str, Any]:
    """Exporter side: the listed fields only, each type-checked (raises)."""
    required, optional = FILES[name]
    missing = [k for k in required if k not in row]
    if missing:
        raise SchemaError(f"{name}: missing {missing}")
    out = {k: row[k] for k in (*required, *optional) if k in row}
    _types(name, out, {**required, **optional})
    return out


def strict(name: str, row: Any) -> None:
    """Checker side: exactly the listed fields, typed."""
    if not isinstance(row, dict):
        raise SchemaError(f"{name}: row is not an object")
    required, optional = FILES[name]
    unknown = set(row) - set(required) - set(optional)
    missing = set(required) - set(row)
    if unknown or missing:
        raise SchemaError(f"{name}: unknown {sorted(unknown)} / missing {sorted(missing)}")
    _types(name, row, {**required, **optional})


def _types(name: str, row: dict[str, Any], checks: dict[str, Check]) -> None:
    for k, v in row.items():
        if not checks[k](v):
            raise SchemaError(f"{name}: field {k!r} has a bad value ({type(v).__name__})")


def check_object(name: str, value: Any, fields: dict[str, Check], *,
                 required: bool = True) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{name}: not an object")
    unknown = set(value) - set(fields)
    if unknown:
        raise SchemaError(f"{name}: unknown fields {sorted(unknown)}")
    if required and set(fields) - set(value):
        raise SchemaError(f"{name}: missing {sorted(set(fields) - set(value))}")
    _types(name, value, fields)
    return value


# --------------------------------------------------------------------------- manifest


def _files(v: Any) -> bool:
    return isinstance(v, dict) and all(
        isinstance(k, str) and isinstance(m, dict) and set(m) == {"sha256", "bytes", "rows"}
        and hex64(m["sha256"]) and integer(m["bytes"]) and opt(integer)(m["rows"])
        for k, m in v.items())


MANIFEST_FIELDS: dict[str, Check] = {
    "schema": s, "created_by": s, "plan_id": s,
    "shard": lambda v: isinstance(v, dict) and set(v) == {"tasks"} and str_list(v["tasks"]),
    "files": _files, "hashes": json_object, "cap_bytes": integer,
    "transcripts": lambda v: isinstance(v, dict) and set(v) == {"sample", "seed", "key_arms"}
    and integer(v["sample"]) and integer(v["seed"]) and opt(str_list)(v["key_arms"])}


# --------------------------------------------------------------------------- learning summary


def _selection(v: Any) -> bool:
    keys = {"applied", "reason", "snapshots", "endpoint_sha256", "selected_sha256", "reverted",
            "scores"}
    return isinstance(v, dict) and set(v) == keys and boolean(v["applied"]) and \
        isinstance(v["snapshots"], list) and len(v["snapshots"]) >= 1 and \
        all(opt(hex64)(x) for x in v["snapshots"]) and \
        opt(s)(v["reason"]) and opt(hex64)(v["endpoint_sha256"]) and \
        opt(hex64)(v["selected_sha256"]) and boolean(v["reverted"]) and \
        isinstance(v["scores"], list) and all(
            isinstance(x, dict) and set(x) == {"index", "sha256", "score"} and
            integer(x["index"]) and opt(hex64)(x["sha256"]) and number(x["score"])
            for x in v["scores"])


BUDGET_KEYS = ("calls", "output_tokens", "decisions")


def _budgets(v: Any) -> bool:
    comps = ("acquisition", "learn", "search", "selection_learned", "selection_search",
             "generic")
    if not isinstance(v, dict) or set(v) != {*comps, "total"}:
        return False
    full = {*BUDGET_KEYS, *(f"limit_{k}" for k in BUDGET_KEYS)}
    return all(isinstance(v[c], dict) and set(v[c]) == full and
               all(integer(x) for x in v[c].values()) for c in comps) and \
        isinstance(v["total"], dict) and set(v["total"]) == set(BUDGET_KEYS) and \
        all(integer(x) for x in v["total"].values())


SUMMARY_FIELDS: dict[str, Check] = {
    "task": TASK, "writer": s, "run": integer, "seed": integer,
    "critic": one_of("teacher", "self", "none"), "check": s, "loop": one_of("ops", "note"),
    "train_case_ids_sha256": hex64, "train_cases": integer, "dev": integer,
    "sentinel": integer, "proposals": integer, "accepted": integer, "episodes": integer,
    "stop_reason": opt(s),
    "search_control": lambda v: isinstance(v, dict) and
    set(v) == {"proposals", "accepted", "stop_reason"} and integer(v["proposals"]) and
    integer(v["accepted"]) and opt(s)(v["stop_reason"]),
    "selection": lambda v: isinstance(v, dict) and set(v) == {"learned", "search_control"}
    and all(_selection(x) for x in v.values()),
    "budgets": _budgets, "evidence_cases": integer, "skill_sha256": opt(hex64),
    "generic_sha256": opt(hex64), "evidence_sha256": hex64,
    "search_control_sha256": opt(hex64), "skill_tokens": integer, "generic_ok": boolean,
    "generic_problems": str_list}
