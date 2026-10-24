"""Registered deterministic trace predicates (``genpilot/predicates.json``).

``same-order-write-precedence-v1`` (sequence cell). Grounding: the retail
policy ("Modify items: This action can only be called once ... The agent will
not be able to modify or cancel the order anymore") and the tool contract
(``modify_pending_order_items``: "For a pending order, this function can only
be called once"; ``modify_pending_order_payment`` requires exactly one
payment in the order's history, which an item modification breaks).

Over the episode's agent tool calls in order, for the bound order O:

- ``k`` = the first SUCCESSFUL ``modify_pending_order_items`` call on O
  (success: its tool result is present and does not start with ``Error``);
- the predicate HOLDS iff ``k`` exists, every successful call of each tool in
  ``before`` on O occurs before ``k``, at least one successful call of each
  tool in ``before`` on O exists, and no write call (attempted, successful or
  not) on O occurs after ``k``.

An episode without a successful item modification does not hold (the DB
check also fails there). The predicate is computed only from the visible
tool-call trajectory; it never reads a reward. DB success and
DB-plus-predicate success are reported separately.
"""

from __future__ import annotations

import inspect
from typing import Any

from slh.genpilot.data import sha256_bytes

WRITE_TOOLS = frozenset({
    "cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
    "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
    "return_delivered_order_items"})
PRECEDENCE = "same-order-write-precedence-v1"


def _as_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    return message.model_dump()


def tool_calls(messages: list[Any]) -> list[dict[str, Any]]:
    """Agent tool calls in order: name, arguments, message index, success
    (None when no tool result follows)."""
    rows = [_as_dict(m) for m in messages]
    results: dict[str, str] = {}
    for m in rows:
        if m.get("role") == "tool":
            key = m.get("tool_call_id") or m.get("id")
            if key is not None:
                results.setdefault(str(key), str(m.get("content") or ""))
    calls = []
    for index, m in enumerate(rows):
        if m.get("role") != "assistant":
            continue
        for call in m.get("tool_calls") or []:
            content = results.get(str(call.get("id")))
            calls.append({"index": index, "name": call.get("name"),
                          "arguments": call.get("arguments") or {},
                          "success": None if content is None
                          else not content.lstrip().startswith("Error")})
    return calls


def same_order_precedence(messages: list[Any], binding: dict[str, Any]) -> dict[str, Any]:
    order = binding["order_id"]
    calls = [c for c in tool_calls(messages) if c["name"] in WRITE_TOOLS
             and isinstance(c["arguments"], dict) and c["arguments"].get("order_id") == order]
    position = next((i for i, c in enumerate(calls)
                     if c["name"] == binding["last"] and c["success"]), None)
    if position is None:
        return {"holds": False, "reason": "no successful item modification on the order"}
    after = [c["name"] for c in calls[position + 1:]]
    if after:
        return {"holds": False, "reason": f"write call(s) after the item modification: {after}"}
    for name in binding["before"]:
        done = [i for i, c in enumerate(calls) if c["name"] == name and c["success"]]
        if not done:
            return {"holds": False, "reason": f"no successful {name} before the item "
                                              f"modification"}
    return {"holds": True, "reason": "required writes precede the item modification"}


PREDICATES = {PRECEDENCE: same_order_precedence}


def evaluate(messages: list[Any], binding: dict[str, Any] | None) -> dict[str, Any] | None:
    if binding is None:
        return None
    return PREDICATES[binding["predicate"]](messages, binding)


def registry_entry() -> dict[str, Any]:
    """The registration record: predicate id, its source text hash and grounding."""
    source = "".join(inspect.getsource(f) for f in (tool_calls, same_order_precedence))
    return {"predicate": PRECEDENCE, "module": "slh/genpilot/predicates.py",
            "functions": ["tool_calls", "same_order_precedence"],
            "source_sha256": sha256_bytes(source.encode("utf-8")),
            "write_tools": sorted(WRITE_TOOLS),
            "grounding": [("policy: Modify items can only be called once; afterwards the "
                           "order cannot be modified or cancelled"),
                          ("tool contract: modify_pending_order_items can only be called once "
                           "per pending order"),
                          ("tool contract: modify_pending_order_payment requires exactly one "
                           "payment in the order's history")],
            "semantics": "holds iff a successful item modification on the bound order "
                         "exists, every required earlier write succeeded before it, and no "
                         "write call on the order follows it"}
