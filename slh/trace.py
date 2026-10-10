"""Trace records: conversion between tau2 messages and stored JSONL, plus leak checks.

Stored episode messages use one flat shape:
``{turn, role, content, tool_calls[{id, name, arguments}], tool_call_id}``.
``turn`` counts user and assistant messages from 0; a tool result shares the
turn of the message that called it.

Grading information must never reach the evidence store or the agent's prompt.
``check_no_grading_fields`` walks a record recursively and raises on any
forbidden key; ``check_agent_prompt_clean`` checks a system prompt against the
task's scenario and grading text.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FORBIDDEN_KEYS = frozenset({
    "evaluation_criteria", "reference_actions", "actions", "initial_state",
    "initialization_data", "initialization_actions", "reward", "reward_info",
    "target_state", "target_db", "target_db_hash", "gold", "gold_actions",
    "communicate_info", "nl_assertions", "user_scenario", "known_info",
    "unknown_info", "task_instructions", "reason_for_call",
})

# Shorter scenario fragments than this are too generic to flag (names, ids).
MIN_LEAK_CHARS = 24


class GradingLeak(RuntimeError):
    """Grading or scenario data was found where it must not be."""


def check_no_grading_fields(record: Any, path: str = "$") -> None:
    if isinstance(record, dict):
        for key, value in record.items():
            if key in FORBIDDEN_KEYS:
                raise GradingLeak(f"forbidden key {key!r} at {path}")
            check_no_grading_fields(value, f"{path}.{key}")
    elif isinstance(record, (list, tuple)):
        for index, value in enumerate(record):
            check_no_grading_fields(value, f"{path}[{index}]")


def task_secret_strings(task: Any) -> list[str]:
    """Scenario and grading text from a tau2 Task that the agent must not see."""
    secrets: list[str] = []
    scenario = getattr(task, "user_scenario", None)
    instructions = getattr(scenario, "instructions", None)
    if isinstance(instructions, str):
        secrets.append(instructions)
    elif instructions is not None:
        for name in ("reason_for_call", "known_info", "unknown_info", "task_instructions"):
            value = getattr(instructions, name, None)
            if value:
                secrets.append(value)
    persona = getattr(scenario, "persona", None)
    if persona:
        secrets.append(str(persona))
    criteria = getattr(task, "evaluation_criteria", None)
    if criteria is not None:
        secrets.extend(criteria.nl_assertions or [])
        secrets.extend(criteria.communicate_info or [])
        for action in criteria.actions or []:
            if action.arguments:
                secrets.append(json.dumps(action.arguments, sort_keys=True))
    return [s.strip() for s in secrets if s and len(s.strip()) >= MIN_LEAK_CHARS]


def check_agent_prompt_clean(system_prompt: str, task: Any, policy: str = "") -> None:
    """Raise if the agent's system prompt carries task scenario or grading data.

    Assertion fragments that also occur in the domain policy (which the agent is
    meant to have, and which assertions often paraphrase) are not counted.
    """
    for key in FORBIDDEN_KEYS:
        if "_" in key and key in system_prompt:
            raise GradingLeak(f"agent prompt mentions grading field {key!r}")
    lowered = system_prompt.lower()
    for secret in task_secret_strings(task):
        if secret.lower() in lowered:
            # Never echo the secret itself into an exception message.
            raise GradingLeak("agent prompt contains task scenario or grading text")
    # Grader-only assertion text must not appear even in fragments.
    _, assertions = task_leak_windows(task)
    assertions = assertions - _windows(_normalize(policy))
    if assertions and not _windows(_normalize(system_prompt)).isdisjoint(assertions):
        raise GradingLeak("agent prompt contains grading assertion text")


LEAK_WINDOW = 30


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _windows(text: str) -> set[str]:
    return {text[i:i + LEAK_WINDOW] for i in range(len(text) - LEAK_WINDOW + 1)}


def leak_windows(strings: Iterable[str]) -> frozenset[str]:
    """All normalized 30-character substrings of the given private strings."""
    out: set[str] = set()
    for text in strings:
        out |= _windows(_normalize(text))
    return frozenset(out)


def task_leak_windows(task: Any) -> tuple[frozenset[str], frozenset[str]]:
    """(all scenario and grading text, grader-only assertion text) as windows."""
    criteria = getattr(task, "evaluation_criteria", None)
    assertions = list((criteria.nl_assertions or []) if criteria is not None else [])
    return leak_windows(task_secret_strings(task)), leak_windows(assertions)


def _strings(value: Any, path: str) -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield f"{path}.{key}", str(key)
            yield from _strings(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _strings(item, f"{path}[{index}]")


def check_no_task_text(record: Any, windows: frozenset[str], *,
                       skip_prefixes: tuple[str, ...] = (), path: str = "$") -> None:
    """Raise if any string in the record contains a 30-char window of task text."""
    if not windows:
        return
    for where, text in _strings(record, path):
        if skip_prefixes and where.startswith(skip_prefixes):
            continue
        normalized = _normalize(text)
        if len(normalized) >= LEAK_WINDOW and not _windows(normalized).isdisjoint(windows):
            raise GradingLeak(f"task scenario or grading text found at {where}")


def error_record(exc: BaseException, code: str) -> dict[str, str]:
    """Public, sanitized description of an exception: no message text."""
    return {"type": type(exc).__name__, "code": code,
            "sha256": hashlib.sha256(str(exc).encode("utf-8", "replace")).hexdigest()}


def _tool_call_record(call: Any) -> dict[str, Any]:
    return {"id": call.id, "name": call.name, "arguments": call.arguments}


def messages_to_records(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """Flatten a tau2 trajectory into stored message records."""
    records: list[dict[str, Any]] = []
    turn = -1
    call_turn: dict[str, int] = {}
    for message in messages:
        role = message.role
        if role == "tool":
            records.append({
                "turn": call_turn.get(message.id, max(turn, 0)),
                "role": "tool",
                "content": message.content,
                "tool_calls": [],
                "tool_call_id": message.id,
            })
            continue
        if role not in ("user", "assistant"):
            raise ValueError(f"unexpected message role in trajectory: {role!r}")
        turn += 1
        calls = [_tool_call_record(c) for c in (message.tool_calls or [])]
        for call in calls:
            call_turn[call["id"]] = turn
        records.append({
            "turn": turn,
            "role": role,
            "content": message.content,
            "tool_calls": calls,
            "tool_call_id": None,
        })
    return records


def records_to_messages(records: Iterable[dict[str, Any]]) -> list[Any]:
    """Rebuild tau2 messages from stored records (for replay)."""
    from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

    messages: list[Any] = []
    requestor_of: dict[str, str] = {}
    for record in records:
        role = record["role"]
        if role == "tool":
            call_id = record["tool_call_id"]
            messages.append(ToolMessage(
                id=call_id, role="tool", content=record.get("content"),
                requestor=requestor_of.get(call_id, "assistant"),
            ))
            continue
        calls = []
        for call in record.get("tool_calls") or []:
            arguments = call["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            requestor = "user" if role == "user" else "assistant"
            requestor_of[call["id"]] = requestor
            calls.append(ToolCall(id=call["id"], name=call["name"],
                                  arguments=arguments, requestor=requestor))
        cls = UserMessage if role == "user" else AssistantMessage
        messages.append(cls(role=role, content=record.get("content"),
                            tool_calls=calls or None, turn_idx=record.get("turn")))
    return messages


def dumps(record: dict[str, Any]) -> str:
    """One JSONL line; ASCII-escaped so U+2028/U+2029 cannot split lines."""
    return json.dumps(record, ensure_ascii=True, sort_keys=False, allow_nan=False) + "\n"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one line and fsync it, so a completed append survives a crash."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(dumps(record))
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Strict reader: every line must be complete JSON."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_jsonl_tolerant(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Read complete lines; return (records, torn_tail).

    Only the final line may be torn (no trailing newline, or not valid JSON);
    it is returned as text and not parsed. A bad line elsewhere raises.
    """
    if not path.exists():
        return [], None
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    tail = lines.pop()  # text after the last newline ("" when the file ends cleanly)
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            if index == len(lines) - 1 and not tail:
                return records, line + "\n"
            raise ValueError(f"{path}: corrupt line {index + 1}") from None
    return records, (tail or None)
