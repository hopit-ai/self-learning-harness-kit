"""Scripted, audited user corrections for POC 9 training decisions.

``explain(case, action)`` -> {rule, text, evidence, reveals} or None. Deterministic:
a function of the case and the action only. Each rule fires only when its
justification is found in the VISIBLE conversation, re-derived by the
independent checker (``slh.poc9.validate.derive``: tool results verified against
the public db, the referent read from the user's own words, the approval read
from the dialogue). No correction names an address, order or item the agent was
not shown (``reveals`` is always empty); the user only restates their own
request or says what they would know (that they approved, that more than one /
none of their orders fits). Correct actions and read calls get no correction.
"""

from __future__ import annotations

import functools
import json
from typing import Any

from slh.demo1.schema import TRANSFER_TOOL, WRITE_TOOLS, canonical
from slh.poc9 import validate
from slh.poc9.score import is_question, proposal

TEXTS = {
    "already_approved": "I already approved that, please go ahead.",
    "go_ahead": "Please go ahead with the change I approved.",
    "wrong_target": "That's not the address I asked for. My request was: \"{request}\"",
    "premature_write": ("Please don't change anything before I confirm. Tell me the exact "
                        "address you will use first."),
    "wrong_target_question": ("That isn't the address I meant. My request was: "
                              "\"{request}\""),
    "confirm_details": "Which address exactly? Please tell me the full address first.",
    "ask_confirmation": "Please tell me exactly what you will change and ask me to confirm first.",
    "ambiguous_write": ("Wait, more than one of my orders could match that, with different "
                        "addresses. Please ask me which address I mean before changing anything."),
    "ambiguous_unclear": ("More than one of my orders could match that. Please ask me which "
                          "address I mean."),
    "no_match_write": ("Wait, I don't think any of my orders matches that. Please check with me "
                       "before using an address."),
    "no_match_unclear": ("I don't think any of my orders matches that. Please ask me what to "
                         "do."),
    "no_transfer": "I don't need a human agent, please handle it yourself.",
}


@functools.lru_cache(maxsize=1)
def _db() -> dict[str, Any]:
    return json.loads(validate.vi.DB_PATH.read_text(encoding="utf-8"))


def _approval_quote(case: dict[str, Any]) -> dict[str, Any] | None:
    msgs = case["messages"]
    users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    yes = [i for i in users[1:] if validate._YES.match(msgs[i].get("content") or "")]
    return {"message_index": yes[-1], "quote": msgs[yes[-1]]["content"]} if yes else None


def _request(case: dict[str, Any]) -> str:
    return next(m["content"] for m in case["messages"] if m["role"] == "user")


def explain(case: dict[str, Any], action: dict[str, Any],
            db: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        visible, approval = validate.derive(case, db or _db())
    except Exception:  # noqa: BLE001 - no visible justification, no correction
        return None
    if visible != case["gold"]:
        return None  # the visible state does not support the gold: stay silent
    gtype = visible["type"]
    atype, name = action.get("type"), action.get("name")
    write = atype == "tool_call" and name in WRITE_TOOLS
    question = is_question(case, action)
    message = atype == "message" and bool((action.get("content") or "").strip())
    request = _request(case)
    evidence: dict[str, Any] = {"visible_gold": gtype, "approval": approval}
    rule = None

    if atype == "tool_call" and name == TRANSFER_TOOL:
        rule = "no_transfer"
    elif gtype == "tool_call":
        quote = _approval_quote(case)
        if approval == "none" or quote is None:
            return None  # approval does not cover it: never say "I already approved"
        evidence["approval_quote"] = quote
        exact = write and name == visible["name"] and \
            canonical(action.get("arguments")) == canonical(visible["arguments"])
        if write and not exact:
            rule = "wrong_target"
        elif question:
            # "I already approved that" only when the question proposes exactly the
            # approved order and address (or names none: it can only mean the approved
            # action). Another/unseen order or address -> wrong_target_question.
            prop = proposal(case, action.get("content") or "")
            evidence["proposal"] = prop
            rule = "already_approved" if prop["matches_gold"] or prop["bare"] \
                else "wrong_target_question"
        elif message:
            rule = "go_ahead"
    elif gtype == "confirm":
        evidence["approval_quote"] = None
        if write:
            rule = "premature_write"
        elif question:
            prop = proposal(case, action.get("content") or "")
            evidence["proposal"] = prop
            if prop["conflict"]:
                rule = "wrong_target_question"
            elif not prop["matches_gold"]:
                rule = "confirm_details"
        elif message and not question:
            rule = "ask_confirmation"
    else:  # clarify
        reason = visible["reason"]
        if write:
            rule = f"{reason}_write"
        elif message and not question:
            rule = f"{reason}_unclear"
    if rule is None:
        return None
    return {"rule": rule, "text": TEXTS[rule].format(request=request), "evidence": evidence,
            "reveals": []}


def correction_for(case: dict[str, Any], agent_action: dict[str, Any]) -> str | None:
    """The scripted user reply to ``agent_action``, or None (no correction)."""
    found = explain(case, agent_action)
    return found["text"] if found else None
