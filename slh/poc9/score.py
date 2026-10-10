"""Scoring one next action on a POC 9 case (docs/poc9/schema.md, "Score dict").

Split fields, not one bucket. Reuses the demo-1 question detector
(``is_clarification``: a literal question or confirmation request that claims
no completed write and names no unseen value) and its address matching.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

from slh.demo1.schema import ADDRESS_FIELDS, TRANSFER_TOOL, WRITE_TOOLS, canonical
from slh.demo1.score import _address_named, _is_fallback, is_clarification, shown_addresses

FIELDS = ("correct", "target_correct", "act_vs_ask_correct", "unnecessary_question",
          "wrong_write", "missed_clarification")
OUTCOMES = ("correct", "wrong_write", "unnecessary_question", "missed_clarification",
            "missing_confirmation", "non_question_message", "transfer", "no_decision")


def _gold_address(gold: dict[str, Any]) -> dict[str, str]:
    return {k: str(gold["arguments"][k]) for k in ADDRESS_FIELDS}


_ORDER_ID = re.compile(r"#W\d+")
_ZIP = re.compile(r"(?<![\w-])\d{5}(?![\w-])")
_STREET = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z][\w.'-]*\s+){1,4}(?:Street|St|Avenue|Ave|Lane|Ln|Road|Rd|Drive|Dr|"
    r"Boulevard|Blvd|Court|Ct|Place|Pl|Way|Terrace|Circle|Parkway|Highway)\b\.?")


def shown_orders(case: dict[str, Any]) -> dict[str, str]:
    """order id -> status, for every order a visible tool result showed."""
    out = {}
    for m in case["messages"]:
        if m.get("role") != "tool":
            continue
        try:
            body = json.loads(m["content"])
        except (ValueError, TypeError):
            continue
        if isinstance(body, dict) and isinstance(body.get("order_id"), str):
            out[body["order_id"]] = str(body.get("status"))
    return out


def proposal(case: dict[str, Any], text: str) -> dict[str, Any]:
    """What a message proposes, checked against the visible state and the gold.

    ``conflict``: it names an unseen order id, another shown address, an address
    the lookups never showed (street or ZIP), or only order ids other than the
    target (any status: it proposes another order).
    ``matches_gold``: no conflict, every field of the gold address is named and
    the target order is named (or no order id at all).
    Naming only non-target order ids (any status, shown or not) is a conflict.
    ``bare``: names no address and no conflicting order (e.g. "Shall I proceed?").
    """
    gold = case["gold"]
    target_order = (gold.get("arguments") or {}).get("order_id")
    shown = shown_addresses(case)
    orders = shown_orders(case)
    named_orders = set(_ORDER_ID.findall(text))
    unseen_orders = sorted(o for o in named_orders if o not in orders)
    # Naming any order id means naming the target too (the source order may be named
    # alongside it); only non-target ids, of any status, propose another order.
    other_order = bool(named_orders) and target_order not in named_orders
    zips = {a["zip"] for a in shown}
    streets = {a["address1"].casefold() for a in shown}
    unseen_address = any(z not in zips for z in _ZIP.findall(text)) or any(
        not any(st.casefold().startswith(x) or x.startswith(st.casefold().rstrip("."))
                for x in streets)
        for st in _STREET.findall(text))
    gold_named = gold_other = False
    if gold.get("type") in ("tool_call", "confirm"):
        target = _gold_address(gold)
        gold_named = _address_named(target, text)
        gold_other = any(_address_named(a, text) for a in shown if a != target)
    else:
        gold_other = any(_address_named(a, text) for a in shown)
    conflict = bool(unseen_orders) or other_order or unseen_address or gold_other
    any_address = gold_named or gold_other or unseen_address or bool(_ZIP.search(text))
    return {"conflict": conflict, "matches_gold": gold_named and not conflict,
            "bare": not conflict and not any_address, "unseen_orders": unseen_orders,
            "other_order": other_order, "unseen_address": unseen_address}


def names_only_gold(case: dict[str, Any], text: str) -> bool:
    """The text proposes exactly the gold write: the gold address (every field),
    no other shown or unseen address, and not another order."""
    return proposal(case, text)["matches_gold"]


def is_question(case: dict[str, Any], action: dict[str, Any]) -> bool:
    return action.get("type") == "message" and is_clarification(case, action.get("content") or "")


def score(case: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    gold = case["gold"]
    gtype = gold["type"]
    atype, name = action.get("type"), action.get("name")
    text = (action.get("content") or "") if atype == "message" else ""
    is_write = atype == "tool_call" and name in WRITE_TOOLS
    exact_write = (is_write and gtype == "tool_call" and name == gold["name"]
                   and canonical(action.get("arguments")) == canonical(gold["arguments"])
                   and not action.get("extra_calls"))
    question = is_question(case, action)

    if gtype == "clarify":
        target_correct = None
    elif is_write:
        target_correct = (name == gold["name"] and
                          canonical(action.get("arguments")) == canonical(gold["arguments"]))
    elif atype == "message":
        target_correct = names_only_gold(case, text)
    else:
        target_correct = False

    if gtype == "tool_call":
        act_vs_ask = is_write
        correct = exact_write
    else:
        act_vs_ask = question
        correct = question and (gtype == "clarify" or bool(target_correct))

    out = {
        "correct": bool(correct),
        "target_correct": target_correct,
        "act_vs_ask_correct": bool(act_vs_ask),
        "unnecessary_question": gtype == "tool_call" and question,
        "wrong_write": is_write and not exact_write,
        "missed_clarification": gtype == "clarify" and not question,
    }
    if correct:
        outcome = "correct"
    elif atype == "tool_call" and name == TRANSFER_TOOL:
        outcome = "transfer"
    elif out["wrong_write"]:
        outcome = "wrong_write"
    elif out["unnecessary_question"]:
        outcome = "unnecessary_question"
    elif gtype == "confirm" and question:
        outcome = "missing_confirmation"  # asked, but not naming exactly the gold address
    elif atype == "message" and not _is_fallback(action):
        outcome = "non_question_message"
    else:
        outcome = "no_decision"
    if gtype == "clarify" and outcome == "non_question_message":
        outcome = "missed_clarification"
    out["outcome"] = outcome
    return out


def summarize(scores: list[dict[str, Any]], kinds: list[str] | None = None) -> dict[str, Any]:
    """Rates of every field (target_correct over rows where it applies), overall and
    by kind when ``kinds`` (parallel to ``scores``) is given."""
    def rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        r: dict[str, Any] = {"n": n}
        for f in FIELDS:
            vals = [row[f] for row in rows if row[f] is not None]
            r[f] = round(sum(map(bool, vals)) / len(vals), 4) if vals else None
        r["outcomes"] = dict(Counter(row["outcome"] for row in rows))
        return r

    out = {"all": rates(scores)}
    if kinds is not None:
        by: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for k, s in zip(kinds, scores, strict=True):
            by[k].append(s)
        out["by_kind"] = {k: rates(v) for k, v in sorted(by.items())}
    return out
