"""BFCL results -> our trace format (``slh-trace.v1`` layout, BFCL profile).

Per cell (one arm x trial run) the exporter writes, under ``<out>/<cell_id>/``:

- ``episodes.jsonl``: one row per instance: ids (``episode_id``, ``instance``, ``root``,
  ``category``, ``arm``, ``trial``, ``cell_id``), ``messages`` (index, role, content,
  ``tool_calls`` [{id, name, arguments (raw string)}], ``tool_call_id``, ``bfcl_turn``:
  the user turn the message belongs to), ``notes_sha256`` and ``tools_sha256`` (into
  ``blobs/``), token counts and latency per step, ``inference_error``;
  **no outcome and no ground truth** (learner-readable, as in docs/trace-format.md);
- ``outcomes.jsonl``: the official verdict reduced to ``success`` (official ``valid``),
  ``error_type``, ``failed_turn`` (when the checker names one), ``harm`` (first failure
  is ``instance_state_mismatch``: the backend was left in a wrong state), ``inaction``
  (``empty_turn_model_response``), ``force_terminated``, ``inference_error``;
- ``calls.jsonl``: one row per model call (episode id, step index, n messages sent,
  finish reason, raw content, raw tool calls, usage, latency);
- ``blobs/<sha256>.json``: system notes and tool lists, canonical JSON;
- ``private/verdicts.jsonl``: the complete official verdicts (they contain ground-truth
  states and calls). **Never shown to a lesson writer**; only our audit code reads it.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from slh.xp.bfcl import data

TRACE_SCHEMA = "slh-trace.v1-bfcl"
_TURN = re.compile(r"\bturn (\d+)\b")
GT_KEYS = frozenset({"possible_answer", "ground_truth", "details", "execution_result",
                     "ground_truth_instance_state", "missing_items"})


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      default=str)


def sha(value: Any) -> str:
    text = value if isinstance(value, str) else canonical(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def episode_id(cell_id: str, instance: str) -> str:
    return "ep-" + sha(f"{cell_id}|{instance}")[:20]


def _tool_calls(msg: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        out.append({"id": tc.get("id"), "name": fn.get("name"), "arguments": fn.get("arguments")})
    return out


def messages(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalized conversation; ``bfcl_turn`` counts user messages (0-based)."""
    out, turn = [], -1
    for k, m in enumerate(raw):
        role = m.get("role")
        if role == "user":
            turn += 1
        row = {"index": k, "role": role, "content": m.get("content"), "bfcl_turn": max(turn, 0)}
        if role == "assistant" and m.get("tool_calls"):
            row["tool_calls"] = _tool_calls(m)
        if role == "tool":
            row["tool_call_id"] = m.get("tool_call_id")
        out.append(row)
    return out


def outcome(verdict: dict[str, Any], result_row: dict[str, Any]) -> dict[str, Any]:
    valid = bool(verdict.get("valid"))
    err = verdict.get("error") or {}
    etype = None if valid else err.get("error_type")
    failed_turn = None
    if not valid:
        if isinstance(err.get("execution_result"), list):
            failed_turn = len(err["execution_result"]) - 1
        else:
            msgs = err.get("error_message") or []
            text = " ".join(msgs) if isinstance(msgs, list) else str(msgs)
            if m := _TURN.search(text):
                failed_turn = int(m.group(1))
    inference_error = isinstance(result_row.get("result"), str)
    return {"success": valid, "error_type": etype, "failed_turn": failed_turn,
            "harm": etype == "multi_turn:instance_state_mismatch",
            "inaction": etype == "multi_turn:empty_turn_model_response",
            "force_terminated": etype == "multi_turn:force_terminated",
            "inference_error": inference_error or etype == "multi_turn:inference_error"}


def assert_no_ground_truth(row: Any, where: str = "row") -> None:
    if isinstance(row, dict):
        for k, v in row.items():
            if k in GT_KEYS:
                raise ValueError(f"{where}: ground-truth field {k!r}")
            assert_no_ground_truth(v, f"{where}.{k}")
    elif isinstance(row, list):
        for k, v in enumerate(row):
            assert_no_ground_truth(v, f"{where}[{k}]")


def export_cell(cell: dict[str, Any], results: list[dict[str, Any]],
                verdicts: dict[str, dict[str, Any]], out: Path) -> dict[str, Any]:
    """Write the trace files of one cell; returns counts."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "blobs").mkdir(exist_ok=True)
    (out / "private").mkdir(exist_ok=True)
    eps, outs, calls, priv = [], [], [], []
    for row in sorted(results, key=lambda r: r["id"]):
        inst = row["id"]
        root, cat = data.parse_instance(inst)
        eid = episode_id(cell["cell_id"], inst)
        raw_msgs = row.get("slh_messages") or []
        tools = row.get("slh_tools") or []
        notes = cell.get("notes") or ""
        blobs = {}
        if notes:
            blobs[sha(notes)] = notes
        if tools:
            blobs[sha(tools)] = tools
        for h, v in blobs.items():
            p = out / "blobs" / f"{h}.json"
            if not p.exists():
                p.write_text(canonical(v), encoding="ascii")
        ep = {"schema": TRACE_SCHEMA, "episode_id": eid, "instance": inst, "root": root,
              "category": cat, "arm": cell["arm"], "trial": cell["trial"],
              "cell_id": cell["cell_id"], "notes_sha256": sha(notes) if notes else None,
              "tools_sha256": sha(tools) if tools else None,
              "messages": messages(raw_msgs),
              "input_token_count": row.get("input_token_count"),
              "output_token_count": row.get("output_token_count"),
              "latency": row.get("latency"),
              "inference_error": isinstance(row.get("result"), str)}
        assert_no_ground_truth(ep, inst)
        eps.append(ep)
        v = verdicts.get(inst)
        if v is not None:
            outs.append({"episode_id": eid, "instance": inst, "cell_id": cell["cell_id"],
                         "arm": cell["arm"], "trial": cell["trial"], **outcome(v, row)})
            priv.append({"episode_id": eid, "instance": inst, "verdict": v})
        for k, c in enumerate(row.get("slh_calls") or []):
            resp = c.get("response") or {}
            choice = (resp.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            calls.append({"episode_id": eid, "step": k, "n_messages": c.get("n_messages"),
                          "finish_reason": choice.get("finish_reason"),
                          "raw_content": msg.get("content"),
                          "raw_tool_calls": _tool_calls(msg),
                          "raw_reasoning_content": msg.get("reasoning_content"),
                          "usage": resp.get("usage"), "latency_s": c.get("latency_s")})
    for name, rows in (("episodes.jsonl", eps), ("outcomes.jsonl", outs),
                       ("calls.jsonl", calls), ("private/verdicts.jsonl", priv)):
        with (out / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(canonical(r) + "\n")
    return {"episodes": len(eps), "outcomes": len(outs), "calls": len(calls)}
