"""Independent gold check for XP cases (no import of the task generators).

For every case:

1. **Evidence**: every visible tool call is replayed, in order, on a fresh
   retail environment; each recorded result must equal the replayed one (so
   the context is true to the public DB, including an earlier write).
2. **Request in words**: the structured request must be stated in the first
   user message (order id, every attribute value, cap, kept attribute, new
   address fields, payment method; address task: the target order id and the
   referent's order id / product(s) / status word / city / payment method), so
   the gold is derivable from what the agent sees. Address task: every order of
   the user must also have been looked up before the decision point.
3. **Re-derivation**: the candidate set is recomputed from the DB with this
   module's own filter (address task: the distinct addresses of the user's
   other orders that the referent describes; a referent that also describes
   the target, or a single candidate equal to the target's current address, is
   refused); the approval state is read from the dialogue (a user
   message starting with "Yes" approves what the assistant message right
   before it names in full: the variant's item id, the full new address).
4. **Single answer**: act needs exactly one candidate and an approval; confirm
   one candidate and no approval; clarify zero or >= 2 candidates.

The expected gold must equal the case gold, and the derived approval its label.
"""

from __future__ import annotations

import json
import re
from typing import Any

ADDRESS = ("address1", "address2", "city", "state", "country", "zip")
_YES = re.compile(r"\s*yes\b", re.IGNORECASE)


class AuditError(ValueError):
    """The case cannot be audited (unsupported or untrue context)."""


def _rx(value: str) -> str:
    parts = re.split(r"[\s_-]+", value.casefold())
    return r"(?<!\w)" + r"[\s_-]+".join(re.escape(p) for p in parts) + r"(?!\w)"


def has(value: str, text: str) -> bool:
    return bool(re.search(_rx(value), text, re.IGNORECASE))


def _money(value: float) -> str:
    return f"${value:,.0f}" if value == int(value) else f"${value:,.2f}"


def replay(messages: list[dict[str, Any]]) -> None:
    from tau2.data_model.message import ToolCall
    from tau2.registry import registry

    env = registry.get_env_constructor("retail")()
    for i, m in enumerate(messages):
        for call in m.get("tool_calls") or []:
            result = messages[i + 1]
            if result.get("role") != "tool" or result.get("tool_call_id") != call["id"]:
                raise AuditError(f"message {i}: unpaired tool call")
            got = env.get_response(ToolCall(id=call["id"], name=call["name"],
                                            arguments=dict(call["arguments"]),
                                            requestor="assistant"))
            if got.content != result["content"] or bool(got.error) != bool(result.get("error")):
                raise AuditError(f"message {i + 1}: {call['name']} result differs from the DB")


def _yes_approvals(messages: list[dict[str, Any]]) -> list[str]:
    """Texts of the assistant messages that a later user "yes" answered."""
    users = [i for i, m in enumerate(messages) if m["role"] == "user"]
    out = []
    for i in users[1:]:
        if _YES.match(messages[i].get("content") or "") and messages[i - 1]["role"] == "assistant":
            out.append(messages[i - 1].get("content") or "")
    return out


def _payment_named(user: dict[str, Any], pid: str, text: str) -> None:
    pm = user["payment_methods"].get(pid)
    if pm is None or pm["source"] == "gift_card":
        raise AuditError("payment method missing or a gift card")
    if pm["source"] == "credit_card":
        same = [p for p in user["payment_methods"].values() if p["source"] == "credit_card"
                and p.get("last_four") == pm["last_four"]]
        if not has(str(pm["last_four"]), text) or len(same) != 1:
            raise AuditError("payment method not named uniquely")
    elif not has("PayPal", text):
        raise AuditError("payment method not named")


def _order_of(case: dict[str, Any], db: dict[str, Any]) -> tuple[dict, dict, dict, str]:
    req = case["request"]
    order = db["orders"].get(req["order_id"])
    user = db["users"].get(req["user_id"])
    if order is None or user is None or order["user_id"] != user["user_id"]:
        raise AuditError("order/user")
    first = next(m["content"] for m in case["messages"] if m["role"] == "user")
    if req["order_id"] not in first:
        raise AuditError("order id not stated")
    items = [it for it in order["items"] if it["item_id"] == req["item_id"]]
    if len(items) != 1 or sum(it["product_id"] == req["product_id"]
                              for it in order["items"]) != 1:
        raise AuditError("item not unique in the order")
    _payment_named(user, req["payment_method_id"], first)
    return order, user, items[0], first


# --------------------------------------------------------------------------- variant


def derive_variant(case: dict[str, Any], db: dict[str, Any]) -> tuple[dict[str, Any], str]:
    req = case["request"]
    order, _, current, first = _order_of(case, db)
    if order["status"] not in ("delivered", "pending"):
        raise AuditError("order status")
    for attr, value in req["set"].items():
        if not (has(attr, first) and has(value, first)):
            raise AuditError(f"constraint {attr}={value} not stated")
    for attr in req["keep"]:
        if not has(attr, first) or attr in req["set"]:
            raise AuditError(f"kept attribute {attr} not stated")
    cap = req["max_price"]
    if cap is not None and _money(cap) not in first:
        raise AuditError("price cap not stated")
    if not req["set"]:
        raise AuditError("no constraint")
    product = db["products"][req["product_id"]]
    cands = sorted(
        iid for iid, v in product["variants"].items()
        if iid != current["item_id"] and v["available"]
        and all(v["options"].get(a) == x for a, x in req["set"].items())
        and all(v["options"].get(a) == current["options"].get(a) for a in req["keep"])
        and (cap is None or v["price"] <= cap + 1e-9))
    approvals = _yes_approvals(case["messages"])
    if not approvals:
        approval = "none"
    elif len(cands) == 1 and has(cands[0], approvals[-1]):
        approval = "concrete"
    else:
        approval = "referent"
    tool = "exchange_delivered_order_items" if order["status"] == "delivered" \
        else "modify_pending_order_items"
    if len(cands) == 1:
        args = {"order_id": req["order_id"], "item_ids": [current["item_id"]],
                "new_item_ids": cands, "payment_method_id": req["payment_method_id"]}
        gold = {"type": "tool_call" if approval != "none" else "confirm", "name": tool,
                "arguments": args}
    else:
        if approval != "referent":
            raise AuditError("clarify case without a referent-level approval")
        gold = {"type": "clarify", "reason": "no_match" if not cands else "ambiguous"}
    return gold, approval


# --------------------------------------------------------------------------- sequence


def derive_sequence(case: dict[str, Any], db: dict[str, Any]) -> tuple[dict[str, Any], str]:
    req = case["request"]
    order, _, current, first = _order_of(case, db)
    if order["status"] != "pending":
        raise AuditError("order not pending")
    product = db["products"][req["product_id"]]
    new = product["variants"].get(req["new_item_id"])
    if new is None or not new["available"] or req["new_item_id"] == current["item_id"]:
        raise AuditError("new item missing, unavailable or unchanged")
    if new["options"] != req["set"]:
        raise AuditError("stated options differ from the new item")
    for attr, value in new["options"].items():
        if not (has(attr, first) and has(value, first)):
            raise AuditError(f"option {attr}={value} not stated")
    same = [k for k, v in product["variants"].items() if v["options"] == new["options"]]
    if same != [req["new_item_id"]]:
        raise AuditError("full option set does not identify one variant")
    addr = {k: str(req["new_address"].get(k, "")) for k in ADDRESS}
    if not all(has(v, first) for v in addr.values() if v):
        raise AuditError("new address not stated")
    if addr == {k: str(order["address"].get(k, "")) for k in ADDRESS}:
        raise AuditError("new address equals the current one")
    a_args = {"order_id": req["order_id"], **addr}
    i_args = {"order_id": req["order_id"], "item_ids": [current["item_id"]],
              "new_item_ids": [req["new_item_id"]],
              "payment_method_id": req["payment_method_id"]}
    approved_a = approved_i = False
    for text in _yes_approvals(case["messages"]):
        approved_a |= all(has(v, text) for v in addr.values() if v)
        approved_i |= has(req["new_item_id"], text)
    a_done = False
    msgs = case["messages"]
    for i, m in enumerate(msgs):
        for call in m.get("tool_calls") or []:
            if call["name"] == "modify_pending_order_address":
                if call["arguments"] != a_args or msgs[i + 1].get("error") or not approved_a:
                    raise AuditError("address write in context is wrong or unapproved")
                a_done = True
            elif call["name"].startswith(("modify_", "exchange_", "return_", "cancel_")):
                raise AuditError(f"unexpected write in context: {call['name']}")
    approval = "+".join(n for n, ok in (("address", approved_a), ("items", approved_i)) if ok) \
        or "none"
    if not a_done:
        gold = {"type": "tool_call" if approved_a else "confirm",
                "name": "modify_pending_order_address", "arguments": a_args}
    else:
        gold = {"type": "tool_call" if approved_i else "confirm",
                "name": "modify_pending_order_items", "arguments": i_args}
    return gold, approval


# --------------------------------------------------------------------------- address

STATUS_WORDS = {"delivered": ("received", "delivered"), "processed": ("processed",),
                "cancelled": ("cancelled",), "pending": ("other pending",)}


def _item_names(order: dict[str, Any]) -> set[str]:
    return {item["name"] for item in order["items"]}


def _paid_with(order: dict[str, Any]) -> str | None:
    return next((p["payment_method_id"] for p in order["payment_history"]
                 if p["transaction_type"] == "payment"), None)


def _describes(order: dict[str, Any], ref: dict[str, Any]) -> bool:
    kind = ref["kind"]
    if kind == "order_id":
        return order["order_id"] == ref["order_id"]
    if kind == "product":
        return ref["product"] in _item_names(order)
    if kind == "status":
        return order["status"] == ref["status"]
    if kind == "product_status":
        return ref["product"] in _item_names(order) and order["status"] == ref["status"]
    if kind == "city":
        return order["address"]["city"] == ref["city"]
    if kind == "payment":
        return _paid_with(order) == ref["payment_method_id"]
    if kind == "two_products":
        return set(ref["products"]) <= _item_names(order)
    raise AuditError(f"unknown referent kind {kind!r}")


def _referent_stated(ref: dict[str, Any], user: dict[str, Any], text: str) -> None:
    kind = ref["kind"]
    if kind == "order_id" and ref["order_id"] not in text:
        raise AuditError("referent order id not stated")
    for name in ([ref["product"]] if "product" in ref else []) + list(ref.get("products", [])):
        if not has(name, text):
            raise AuditError(f"product {name} not stated")
    if "status" in ref and not any(has(w, text) for w in STATUS_WORDS[ref["status"]]):
        raise AuditError(f"status {ref['status']} not stated")
    if kind == "city" and not has(ref["city"], text):
        raise AuditError("city not stated")
    if kind == "payment":
        pms = user["payment_methods"]
        pm = pms.get(ref["payment_method_id"])
        if pm is None:
            raise AuditError("payment method missing")
        same = [p for p in pms.values() if p["source"] == pm["source"]
                and p.get("last_four") == pm.get("last_four")]
        word = {"credit_card": str(pm.get("last_four")), "paypal": "PayPal",
                "gift_card": "gift card"}.get(pm["source"])
        if word is None or len(same) != 1 or not has(word, text):
            raise AuditError("payment method not named uniquely")


def derive_address(case: dict[str, Any], db: dict[str, Any]) -> tuple[dict[str, Any], str]:
    req = case["request"]
    ref = req["referent"]
    user = db["users"].get(req["user_id"])
    target = db["orders"].get(req["order_id"])
    if user is None or target is None or target["user_id"] != user["user_id"] \
            or req["order_id"] not in user["orders"]:
        raise AuditError("order/user")
    if target["status"] != "pending":
        raise AuditError("target order not pending")
    first = next(m["content"] for m in case["messages"] if m["role"] == "user")
    if req["order_id"] not in first:
        raise AuditError("target order id not stated")
    _referent_stated(ref, user, first)
    looked = {c["arguments"].get("order_id") for m in case["messages"]
              for c in m.get("tool_calls") or [] if c["name"] == "get_order_details"}
    if not set(user["orders"]) <= looked:
        raise AuditError("not every order of the user was looked up")
    if ref["kind"] not in ("order_id", "status") and _describes(target, ref):
        raise AuditError("the referent also describes the target order")
    sources = [db["orders"][o] for o in user["orders"]
               if o in db["orders"] and o != req["order_id"] and _describes(db["orders"][o], ref)]
    keys = sorted({tuple(str(o["address"].get(k, "")) for k in ADDRESS) for o in sources})
    current = tuple(str(target["address"].get(k, "")) for k in ADDRESS)
    if keys == [current]:
        raise AuditError("the only candidate is the current address")
    approvals = _yes_approvals(case["messages"])
    if not approvals:
        approval = "none"
    elif len(keys) == 1 and all(has(v, approvals[-1]) for v in keys[0] if v):
        approval = "concrete"
    else:
        approval = "referent"
    if len(keys) == 1:
        args = {"order_id": req["order_id"], **dict(zip(ADDRESS, keys[0], strict=True))}
        gold = {"type": "tool_call" if approval != "none" else "confirm",
                "name": "modify_pending_order_address", "arguments": args}
    else:
        if approval != "referent":
            raise AuditError("clarify case without a referent-level approval")
        gold = {"type": "clarify", "reason": "no_match" if not keys else "ambiguous"}
    return gold, approval


# --------------------------------------------------------------------------- driver


def validate_case(case: dict[str, Any], db: dict[str, Any], *, evidence: bool = True
                  ) -> list[str]:
    derive = {"address": derive_address, "variant": derive_variant,
              "sequence": derive_sequence}.get(case.get("task"))
    if derive is None:
        return [f"unknown task {case.get('task')!r}"]
    try:
        if evidence:
            replay(case["messages"])
        expected, approval = derive(case, db)
    except (KeyError, TypeError, ValueError, IndexError, StopIteration) as exc:
        return [f"cannot derive: {exc}"]
    problems = []
    if json.dumps(expected, sort_keys=True) != json.dumps(case["gold"], sort_keys=True):
        problems.append(f"gold disagreement: expected {expected}")
    if approval != case["approval"]:
        problems.append(f"approval {approval} != {case['approval']}")
    return problems


def validate_all(cases: list[dict[str, Any]], db: dict[str, Any] | None = None, *,
                 evidence: bool = True) -> dict[str, Any]:
    if db is None:
        from slh.xp.tasks.common import load_db

        db = load_db()
    bad = {c["case_id"]: p for c in cases if (p := validate_case(c, db, evidence=evidence))}
    return {"ok": not bad, "checked": len(cases), "failures": bad}
