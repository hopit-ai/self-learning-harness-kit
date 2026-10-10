"""Scoring one next action against a case's gold (docs/demo1/case-schema.md)."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from slh.demo1.schema import ADDRESS_FIELDS, TRANSFER_TOOL, WRITE_TOOLS, canonical

OUTCOMES = ("correct", "wrong_write", "unnecessary_question", "missed_clarification",
            "non_clarifying_message", "transfer", "no_decision")
# Fallback texts the B1 agent emits without a model decision (slh.agent / think arm).
FALLBACK_TEXTS = frozenset({
    "I'm unable to choose the next action right now.",
    ("I'm sorry, I can't complete that action right now. Is there anything else I can help "
    "with?")})


def _is_fallback(action: dict[str, Any]) -> bool:
    return action.get("type") == "message" and \
        (action.get("content") or "").strip() in FALLBACK_TEXTS


_QUESTION = re.compile(
    r"\?|\b(?:which|please (?:confirm|specify|clarify|choose|select|provide|let me know)|"
    r"could you|can you (?:confirm|tell|clarify|specify)|let me know|would you like|"
    r"do you want|did you mean)\b", re.IGNORECASE)
_WRITE_CLAIM = re.compile(
    r"\b(?:i(?:'ve| have)? (?:changed|updated|modified|exchanged|switched|placed|processed)|"
    r"(?:has|have) been (?:changed|updated|modified|exchanged|switched|processed)|"
    r"successfully|is now (?:set|updated|changed)|done[,.!])", re.IGNORECASE)
_ITEM_ID = re.compile(r"(?<!\d)\d{10}(?!\d)")


def _token_in(value: str, text: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(value)}(?![\w])", text, re.IGNORECASE) is not None


def _tool_bodies(case: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for m in case["messages"]:
        if m.get("role") != "tool":
            continue
        try:
            body = json.loads(m["content"])
        except (ValueError, TypeError):
            continue
        if isinstance(body, dict):
            out.append(body)
    return out


def shown_addresses(case: dict[str, Any]) -> list[dict[str, str]]:
    found = []
    for body in _tool_bodies(case):
        if isinstance(body.get("address"), dict):
            a = {k: str(body["address"].get(k, "")) for k in ADDRESS_FIELDS}
            if a not in found:
                found.append(a)
    return found


def shown_item_ids(case: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for body in _tool_bodies(case):
        if isinstance(body.get("variants"), dict):
            ids.update(body["variants"])
        for item in body.get("items") or []:
            if isinstance(item, dict) and item.get("item_id"):
                ids.add(str(item["item_id"]))
    return ids


def _address_named(address: dict[str, str], text: str) -> bool:
    """Every non-empty field of the address appears as a whole token."""
    return all(_token_in(v, text) for v in address.values() if v)


def unsupported_values(case: dict[str, Any], text: str) -> list[str]:
    """Item ids in the text that no lookup showed (address task: none checked
    beyond ids, since free text cannot be parsed into an address reliably)."""
    shown = shown_item_ids(case)
    return sorted(v for v in set(_ITEM_ID.findall(text)) if v not in shown)


def is_clarification(case: dict[str, Any], text: str) -> bool:
    """A genuine question to the user: asks or requests a choice, claims no
    completed write and names no value the lookups did not show."""
    text = (text or "").strip()
    return bool(text) and text not in FALLBACK_TEXTS and bool(_QUESTION.search(text)) and \
        not _WRITE_CLAIM.search(text) and not unsupported_values(case, text)


def reconfirms_gold(case: dict[str, Any], text: str) -> bool:
    """Secondary score: a re-confirmation question naming the full gold value
    (all address fields / the new item id) and no other candidate."""
    gold = case["gold"]
    if gold.get("type") != "tool_call" or not is_clarification(case, text):
        return False
    args = gold["arguments"]
    if case["task"] == "address":
        target = {k: str(args[k]) for k in ADDRESS_FIELDS}
        others = [a for a in shown_addresses(case) if a != target]
        return _address_named(target, text) and not any(_address_named(a, text)
                                                        for a in others)
    new = args["new_item_ids"][0]
    others = shown_item_ids(case) - {new}
    return _token_in(new, text) and not any(_token_in(v, text) for v in others)


def score(case: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    gold = case["gold"]
    kind = action.get("type")
    name = action.get("name")
    text = (action.get("content") or "") if kind == "message" else ""
    if kind == "tool_call" and name == TRANSFER_TOOL:
        outcome = "transfer"
    elif gold["type"] == "tool_call":
        if kind == "tool_call" and name == gold["name"] and \
                canonical(action.get("arguments")) == canonical(gold["arguments"]) and \
                not action.get("extra_calls"):
            outcome = "correct"
        elif kind == "tool_call" and name in WRITE_TOOLS:
            outcome = "wrong_write"
        elif kind == "message" and not _is_fallback(action):
            outcome = "unnecessary_question"
        else:
            outcome = "no_decision"
    else:
        if kind == "message" and is_clarification(case, text):
            outcome = "correct"
        elif kind == "message" and not _is_fallback(action):
            outcome = "non_clarifying_message"
        elif kind == "tool_call" and name in WRITE_TOOLS:
            outcome = "missed_clarification"
        else:
            outcome = "no_decision"
    decision_correct = outcome == "correct" or (
        outcome == "unnecessary_question" and reconfirms_gold(case, text))
    return {"outcome": outcome, "correct": outcome == "correct",
            "decision_correct": decision_correct}


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts over rows that carry ``outcome`` and ``decision_correct``."""
    n = len(scored)
    counts = Counter(r["outcome"] for r in scored)
    correct = counts.get("correct", 0)
    return {"n": n, "correct": correct,
            "accuracy": round(correct / n, 4) if n else None,
            "decision_correct": sum(1 for r in scored if r["decision_correct"]),
            "outcomes": {o: counts.get(o, 0) for o in OUTCOMES}}
