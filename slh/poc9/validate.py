"""Independent gold check for POC 9 cases (no generator import).

Reuses the demo-1 independent auditor: the visible tool results are verified
against the public db, and the address the first user message points to is
re-derived from text (``slh.demo1.validate_independent``). The approval state is
read from the visible dialogue only: a later user "yes" approves; it is a
concrete approval when the agent message right before it names the full derived
address. Expected gold: approved + unique -> the write; not approved + unique ->
confirm; no single address -> clarify (only with a referent-level approval).
"""

from __future__ import annotations

import re
from typing import Any

from slh.demo1 import validate_independent as vi

_YES = re.compile(r"\s*(?:yes\b|confirmed\b|I confirm\b|go ahead\b|proceed\b)", re.IGNORECASE)


def _approval(messages: list[dict[str, Any]], args: dict[str, Any] | None) -> str:
    users = [i for i, m in enumerate(messages) if m["role"] == "user"]
    yes = [i for i in users[1:] if _YES.match(messages[i].get("content") or "")]
    if not yes:
        return "none"
    prev = messages[yes[-1] - 1]
    text = prev.get("content") or ""
    if prev["role"] == "assistant" and args and all(
            vi._has(str(args[k]), text) for k in vi.ADDRESS if args.get(k)):
        return "concrete"
    return "referent"


def derive(case: dict[str, Any], db: dict[str, Any]) -> tuple[dict[str, Any], str]:
    messages = case["messages"]
    if messages[-1].get("role") != "tool":
        raise vi.AuditError("decision must follow visible tool evidence")
    user, orders, _ = vi._evidence(messages, db)
    first = next(m["content"] for m in messages if m["role"] == "user")
    action = vi._address(first, user, orders, db)
    args = action.get("arguments")
    approval = _approval(messages, args)
    if action["type"] == "tool_call" and approval == "none":
        action = {"type": "confirm", "name": action["name"], "arguments": args}
    elif action["type"] == "clarify" and approval != "referent":
        raise vi.AuditError("clarify case without a referent-level approval")
    return action, approval


def validate_case(case: dict[str, Any], db: dict[str, Any]) -> list[str]:
    try:
        expected, approval = derive(case, db)
    except (KeyError, TypeError, ValueError, IndexError, StopIteration) as exc:
        return [f"cannot derive: {exc}"]
    problems = []
    if expected != case["gold"]:
        problems.append(f"gold disagreement: expected {expected}")
    if approval != case["approval"]:
        problems.append(f"approval {approval} != {case['approval']}")
    return problems


def validate_all(cases: list[dict[str, Any]], db: dict[str, Any] | None = None) -> dict[str, Any]:
    if db is None:
        import json

        db = json.loads(vi.DB_PATH.read_text(encoding="utf-8"))
    bad = {c["case_id"]: p for c in cases if (p := validate_case(c, db))}
    return {"ok": not bad, "checked": len(cases), "failures": bad}
