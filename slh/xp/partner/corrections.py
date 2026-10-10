"""Scripted user corrections for XP training decisions (variant, sequence, address).

Modelled on POC 9's ``slh.poc9.corrections``: ``explain(case, action)`` -> {rule,
text, evidence, reveals} or None, a deterministic function of the case and the
action. A rule fires only when the visible conversation supports it: the gold
is first re-derived from the visible state by the independent checker
(``slh.xp.validate.derive_<task>``) on the VISIBLE database only (``visible_db``: entities whose
lookup results are in the conversation); if that fails or disagrees, no correction.
No correction names an order, item or address the agent was not shown
(``reveals`` is always empty): the user restates their own request or says what
they would know (that they approved, that several / none of the options fit,
that the address goes first). Correct actions and read calls get none.

Training and validation only; never applied to test cases.
"""

from __future__ import annotations

import json
import re
from typing import Any

from slh.demo1.schema import TRANSFER_TOOL, WRITE_TOOLS, canonical

TEXTS = {
    "already_approved": "I already approved that, please go ahead.",
    "go_ahead": "Please go ahead with the change I approved.",
    "wrong_target": "That's not what I asked for. My request was: \"{request}\"",
    "wrong_target_question": "That isn't what I meant. My request was: \"{request}\"",
    "address_first": ("Please change the address first; the item change can only come "
                      "after it."),
    "premature_write": ("Please don't change anything before I confirm. Tell me exactly what "
                        "you will change first."),
    "confirm_details": "Which one exactly? Please tell me the full details first.",
    "ask_confirmation": ("Please tell me exactly what you will change and ask me to confirm "
                         "first."),
    "ambiguous_write": ("Wait, more than one option could match what I described. Please ask "
                        "me which one I mean before changing anything."),
    "ambiguous_unclear": ("More than one option could match what I described. Please ask me "
                          "which one I mean."),
    "no_match_write": ("Wait, I don't think anything matches what I described. Please check "
                       "with me before changing anything."),
    "no_match_unclear": "I don't think anything matches what I described. Please ask me what to do.",
    "no_transfer": "I don't need a human agent, please handle it yourself.",
}
ADDRESS_WRITE = "modify_pending_order_address"
ITEMS_WRITE = "modify_pending_order_items"
_YES = re.compile(r"\s*yes\b", re.IGNORECASE)
_SPECIFIC = re.compile(r"#W\d+|(?<!\d)\d{10}(?!\d)|(?<![\w-])\d{5}(?![\w-])")


def visible_db(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The database as the agent saw it: only users, orders and products whose
    lookup results are in the conversation (the latest result per entity).
    Corrections are derived from this view, never from the full database, so
    an entity the agent never looked up cannot justify a correction."""
    out: dict[str, dict[str, Any]] = {"users": {}, "orders": {}, "products": {}}
    for m in case["messages"]:
        if m.get("role") != "tool" or m.get("error"):
            continue
        try:
            body = json.loads(m.get("content") or "")
        except (TypeError, ValueError):
            continue
        if not isinstance(body, dict):
            continue
        if "order_id" in body:
            out["orders"][body["order_id"]] = body
        elif "product_id" in body and "variants" in body:
            out["products"][body["product_id"]] = body
        elif "user_id" in body and ("orders" in body or "payment_methods" in body):
            out["users"][body["user_id"]] = body
    return out


def texts_for(task: str) -> dict[str, str]:
    if task != "address":
        return TEXTS
    from slh.poc9.corrections import TEXTS as ADDRESS

    return {**TEXTS, **ADDRESS}


def _approval_quote(case: dict[str, Any]) -> dict[str, Any] | None:
    msgs = case["messages"]
    users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    yes = [i for i in users[1:] if _YES.match(msgs[i].get("content") or "")]
    return {"message_index": yes[-1], "quote": msgs[yes[-1]]["content"]} if yes else None


def _request(case: dict[str, Any]) -> str:
    return next(m["content"] for m in case["messages"] if m["role"] == "user")


def explain(case: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    from slh.xp import validate
    from slh.xp.score import is_question, proposal

    task = case.get("task")
    derive = getattr(validate, f"derive_{task}", None)
    if derive is None:
        return None
    seen = visible_db(case)
    if not any(seen.values()):
        return None  # nothing was looked up: no visible justification
    try:
        visible, approval = derive(case, seen)
    except Exception:  # noqa: BLE001 - no visible justification, no correction
        return None
    if json.dumps(visible, sort_keys=True) != json.dumps(case["gold"], sort_keys=True):
        return None
    texts = texts_for(task)
    gtype = visible["type"]
    atype, name = action.get("type"), action.get("name")
    write = atype == "tool_call" and name in WRITE_TOOLS
    question = is_question(case, action)
    content = action.get("content") or ""
    message = atype == "message" and bool(content.strip())
    evidence: dict[str, Any] = {"visible_gold": gtype, "approval": approval}
    rule = None
    address_pending = visible.get("name") == ADDRESS_WRITE and task == "sequence"

    if atype == "tool_call" and name == TRANSFER_TOOL:
        rule = "no_transfer"
    elif address_pending and write and name == ITEMS_WRITE:
        rule = "address_first"
    elif gtype == "tool_call":
        quote = _approval_quote(case)
        if approval == "none" or quote is None:
            return None  # never say "I already approved" without a visible yes
        evidence["approval_quote"] = quote
        exact = write and name == visible["name"] and \
            canonical(action.get("arguments")) == canonical(visible["arguments"])
        if write and not exact:
            rule = "wrong_target"
        elif question:
            prop = proposal(case, content)
            bare = not _SPECIFIC.search(content)
            evidence["proposal"] = {k: prop[k] for k in ("conflict", "matches_gold")}
            rule = "already_approved" if prop["matches_gold"] or (bare and not prop["conflict"]) \
                else "wrong_target_question"
        elif message:
            rule = "go_ahead"
    elif gtype == "confirm":
        if write:
            rule = "premature_write"
        elif question:
            prop = proposal(case, content)
            evidence["proposal"] = {k: prop[k] for k in ("conflict", "matches_gold")}
            if prop["conflict"]:
                rule = "wrong_target_question"
            elif not prop["matches_gold"]:
                rule = "confirm_details"
        elif message:
            rule = "ask_confirmation"
    else:  # clarify
        reason = visible.get("reason")
        if reason in ("ambiguous", "no_match"):
            if write:
                rule = f"{reason}_write"
            elif message and not question:
                rule = f"{reason}_unclear"
    if rule is None:
        return None
    return {"rule": rule, "text": texts[rule].format(request=_request(case)),
            "evidence": evidence, "reveals": []}


def correction_for(case: dict[str, Any], action: dict[str, Any]) -> str | None:
    found = explain(case, action)
    return found["text"] if found else None
