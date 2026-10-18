"""Parse a public episode into events and build the agent-visible state.

Only learner-readable content is used: user and agent messages, the agent's tool
calls and the tool results that answered them. ``visible(events, k)`` is what the
agent could have known before event ``k``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Write tools of the retail tool contract (``ToolType.WRITE``); a test checks
# this against the contract source.
WRITE_TOOLS = frozenset({
    "cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
    "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
    "return_delivered_order_items"})
AUTH_TOOLS = frozenset({"find_user_id_by_email", "find_user_id_by_name_zip"})
# Calls that act on, or reveal, one user's account.
ACCOUNT_TOOLS = frozenset({"get_user_details", "get_order_details"}) | WRITE_TOOLS
# Tools whose successful result is the order after the change.
ORDER_RESULT_TOOLS = frozenset({"get_order_details"}) | (WRITE_TOOLS - {"modify_user_address"})


@dataclass(frozen=True)
class Event:
    index: int
    kind: str  # "user" | "agent" | "call"
    message: int  # index of the source message
    text: str = ""
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    calls_in_message: int = 0
    message_has_text: bool = False

    @property
    def error(self) -> bool:
        return self.result is None or self.result.startswith("Error")

    @property
    def is_write(self) -> bool:
        return self.kind == "call" and self.name in WRITE_TOOLS


def _args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def events_from_messages(messages: list[dict[str, Any]]) -> list[Event]:
    """User text, agent text and agent tool calls (with their results), in order."""
    results = {m.get("tool_call_id"): m.get("content") for m in messages
               if m.get("role") == "tool" and m.get("tool_call_id")}
    events: list[Event] = []
    for position, message in enumerate(messages):
        role, text = message.get("role"), message.get("content") or ""
        calls = message.get("tool_calls") or []
        if role == "user" and text.strip():
            events.append(Event(len(events), "user", position, text=text))
        elif role == "assistant":
            if text.strip():
                events.append(Event(len(events), "agent", position, text=text,
                                    calls_in_message=len(calls)))
            for call in calls:
                result = results.get(call.get("id"))
                events.append(Event(len(events), "call", position, name=str(call.get("name")),
                                    args=_args(call.get("arguments")),
                                    result=None if result is None else str(result),
                                    calls_in_message=len(calls),
                                    message_has_text=bool(text.strip())))
    return events


def order_key(order_id: Any) -> str:
    """Order ids compared without the leading '#' and case."""
    return str(order_id).strip().lstrip("#").upper()


def _json(text: str | None) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


@dataclass
class View:
    """What the agent could see before a given event."""

    auth_ids: list[str] = field(default_factory=list)
    users: dict[str, dict[str, Any]] = field(default_factory=dict)
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)  # latest snapshot by key
    looked_up: set[str] = field(default_factory=set)  # orders fetched by get_order_details
    products: dict[str, dict[str, Any]] = field(default_factory=dict)
    variants: dict[str, dict[str, Any]] = field(default_factory=dict)  # item id -> variant
    variant_product: dict[str, str] = field(default_factory=dict)
    user_text: str = ""
    tool_text: str = ""

    @property
    def auth_user(self) -> str | None:
        return self.auth_ids[0] if self.auth_ids else None

    def owned_orders(self) -> set[str]:
        owned = set()
        for uid in self.auth_ids:
            owned |= {order_key(o) for o in (self.users.get(uid) or {}).get("orders") or []}
        return owned

    def payment_methods(self) -> dict[str, dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        for uid in self.auth_ids:
            found.update((self.users.get(uid) or {}).get("payment_methods") or {})
        return found

    def user_mentions(self, token: str) -> bool:
        token = str(token).strip().lstrip("#")
        return bool(token) and re.search(r"(?<![\w])#?" + re.escape(token) + r"(?![\w])",
                                         self.user_text, re.IGNORECASE) is not None

    def tool_mentions(self, token: str) -> bool:
        token = str(token).strip().lstrip("#")
        return bool(token) and re.search(r"(?<![\w])#?" + re.escape(token) + r"(?![\w])",
                                         self.tool_text) is not None


def visible(events: list[Event], before: int) -> View:
    view = View()
    for event in events[:before]:
        if event.kind == "user":
            view.user_text += "\n" + event.text
            continue
        if event.kind != "call" or event.error:
            continue
        view.tool_text += "\n" + (event.result or "")
        data = _json(event.result)
        if event.name in AUTH_TOOLS:
            uid = (event.result or "").strip().strip('"')
            if uid and uid not in view.auth_ids:
                view.auth_ids.append(uid)
        elif event.name in ("get_user_details", "modify_user_address") and isinstance(data, dict):
            view.users[str(data.get("user_id"))] = data
        elif event.name in ORDER_RESULT_TOOLS and isinstance(data, dict) and "order_id" in data:
            view.orders[order_key(data["order_id"])] = data
            if event.name == "get_order_details":
                view.looked_up.add(order_key(data["order_id"]))
        elif event.name == "get_product_details" and isinstance(data, dict):
            pid = str(data.get("product_id"))
            view.products[pid] = data
            for vid, variant in (data.get("variants") or {}).items():
                view.variants[str(vid)] = variant
                view.variant_product[str(vid)] = pid
        elif event.name == "get_item_details" and isinstance(data, dict):
            view.variants.setdefault(str(data.get("item_id")), data)
    return view
