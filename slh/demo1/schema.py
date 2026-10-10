"""Shared helpers for ``demo1-case-v1`` (docs/demo1/case-schema.md).

No generator logic lives here, so an independent validator can import it
without importing the generator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "demo1-case-v1"
TASKS = ("address", "variant")
SPLITS = ("train", "calibrate", "sealed")
ROLES = ("act", "act_distractor", "clarify")
ADDRESS_KINDS = ("order_id", "product", "status", "product_status", "city", "payment",
                 "two_products")
VARIANT_KINDS = ("full_spec", "keep_one", "keep_two", "one_price", "two_price",
                 "cheapest_one", "one_only")
KINDS = {"address": ADDRESS_KINDS, "variant": VARIANT_KINDS}
PHRASINGS = 4
ADDRESS_FIELDS = ("address1", "address2", "city", "state", "country", "zip")
WRITE_TOOLS = frozenset({
    "cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
    "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
    "return_delivered_order_items"})
READ_TOOLS = frozenset({
    "calculate", "find_user_id_by_email", "find_user_id_by_name_zip", "get_order_details",
    "get_product_details", "get_item_details", "get_user_details", "list_all_product_types"})
TRANSFER_TOOL = "transfer_to_human_agents"
GREETING = "Hi! How can I help you today?"
CASE_KEYS = frozenset({"schema", "case_id", "task", "kind", "family", "phrasing", "role",
                       "split", "context", "messages", "gold", "request", "resolution",
                       "entities", "generator"})


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def family_id(task: str, kind: str, phrasing: int) -> str:
    return f"{task[0]}-{kind}-{phrasing}"


def gold_tool_for_status(order_status: str) -> str:
    return "exchange_delivered_order_items" if order_status == "delivered" \
        else "modify_pending_order_items"


def check_case(case: dict[str, Any]) -> list[str]:
    """Structural problems of one case (empty list = well formed)."""
    problems = []
    if set(case) != CASE_KEYS:
        problems.append(f"keys differ: {sorted(set(case) ^ CASE_KEYS)}")
        return problems
    if case["schema"] != SCHEMA:
        problems.append("schema")
    if case["task"] not in TASKS or case["kind"] not in KINDS[case["task"]]:
        problems.append("task/kind")
    if case["split"] not in SPLITS or case["role"] not in ROLES:
        problems.append("split/role")
    if case["family"] != family_id(case["task"], case["kind"], case["phrasing"]):
        problems.append("family id")
    messages = case["messages"]
    if not messages or messages[0].get("content") != GREETING:
        problems.append("first message is not the greeting")
    if not messages or messages[-1].get("role") != "tool":
        problems.append("the decision point must follow a tool message")
    pending: str | None = None
    for i, m in enumerate(messages):
        calls = m.get("tool_calls") or []
        if pending is not None and (m.get("role") != "tool" or m.get("tool_call_id") != pending):
            problems.append(f"message {i}: the result of {pending} must follow its call")
        pending = None
        if len(calls) > 1:
            problems.append(f"message {i}: more than one tool call")
        if calls:
            if m.get("role") != "assistant":
                problems.append(f"message {i}: tool call from {m.get('role')}")
            pending = calls[0]["id"]
            if calls[0]["name"] in WRITE_TOOLS:
                problems.append(f"message {i}: a write call before the decision point")
    gold = case["gold"]
    if gold.get("type") == "clarify":
        if gold.get("reason") not in ("no_match", "ambiguous") or case["role"] != "clarify":
            problems.append("clarify gold")
    elif gold.get("type") == "tool_call":
        if gold.get("name") not in WRITE_TOOLS or case["role"] == "clarify":
            problems.append("tool gold")
    else:
        problems.append("gold type")
    return problems


def to_tau2_messages(messages: list[dict[str, Any]]) -> list[Any]:
    """Case messages -> tau2 message objects (agent-history valid)."""
    from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

    out: list[Any] = []
    for m in messages:
        role = m["role"]
        if role == "assistant":
            calls = [ToolCall(id=c["id"], name=c["name"], arguments=dict(c["arguments"]),
                              requestor="assistant") for c in m.get("tool_calls") or []]
            out.append(AssistantMessage(role="assistant", content=m.get("content") or None,
                                        tool_calls=calls or None, cost=0.0))
        elif role == "user":
            out.append(UserMessage(role="user", content=m["content"]))
        elif role == "tool":
            out.append(ToolMessage(id=m["tool_call_id"], role="tool", content=m["content"],
                                   requestor="assistant", error=bool(m.get("error"))))
        else:
            raise ValueError(f"unknown message role {role!r}")
    return out


def action_from_message(message: Any) -> dict[str, Any]:
    """An agent's external ``AssistantMessage`` -> the action format."""
    calls = [c for c in (message.tool_calls or []) if c.name != "think"]
    if calls:
        call = calls[0]
        action = {"type": "tool_call", "name": call.name, "arguments": dict(call.arguments)}
        if len(calls) > 1:
            action["extra_calls"] = len(calls) - 1
        return action
    if message.content:
        return {"type": "message", "content": message.content}
    return {"type": "none", "reason": "empty message"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n"
                            for r in rows), encoding="utf-8")
