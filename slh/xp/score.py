"""Scoring one next action on an XP case; generalised from POC 9's score.py.

Same split fields and the same outcome taxonomy (``correct``, ``wrong_write``,
``unnecessary_question``, ``missed_clarification``, ``missing_confirmation``,
``non_question_message``, ``transfer``, ``no_decision``) with the same decision
order. What changed:

* The proposal check (does a confirmation name exactly the gold write?) covers
  two target types: an **address** (every field of the gold address named) and
  an **item change** (the gold new item id named, or every option value of the
  gold variant named while no other non-allowed variant has all its values
  named). Values the plan legitimately mentions are allowed: the order's
  current address and the item being replaced, plus, in the sequence task, the
  other planned change (a joint confirmation may list both).
* A write claim in a question is tolerated only when a write already completed
  in the visible context (e.g. "The address has been updated. Shall I now
  change the item?").
* ``sequence_violation`` (sequence task only, else None): the action writes the
  items while the address change is still to do, or proposes the item change
  without the address when the address comes first.

``score_run`` joins a remote runner's actions to the answer key by case_id and
verifies each row's case_hash.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from slh.demo1.schema import ADDRESS_FIELDS, TRANSFER_TOOL, WRITE_TOOLS, canonical

FIELDS = ("correct", "target_correct", "act_vs_ask_correct", "unnecessary_question",
          "wrong_write", "missed_clarification")
OUTCOMES = ("correct", "wrong_write", "unnecessary_question", "missed_clarification",
            "missing_confirmation", "non_question_message", "transfer", "no_decision")
ADDRESS_WRITE = "modify_pending_order_address"
ITEMS_WRITE = "modify_pending_order_items"
FALLBACK_TEXTS = frozenset({
    "I'm unable to choose the next action right now.",
    ("I'm sorry, I can't complete that action right now. Is there anything else I can help "
     "with?")})

_QUESTION = re.compile(
    r"\?|\b(?:which|please (?:confirm|specify|clarify|choose|select|provide|let me know)|"
    r"could you|can you (?:confirm|tell|clarify|specify)|let me know|would you like|"
    r"do you want|did you mean)\b", re.IGNORECASE)
_WRITE_CLAIM = re.compile(
    r"\b(?:i(?:'ve| have)? (?:changed|updated|modified|exchanged|switched|placed|processed)|"
    r"(?:has|have) been (?:changed|updated|modified|exchanged|switched|processed)|"
    r"successfully|is now (?:set|updated|changed)|done[,.!])", re.IGNORECASE)
_ITEM_ID = re.compile(r"(?<!\d)\d{10}(?!\d)")
_ORDER_ID = re.compile(r"#W\d+")
_ZIP = re.compile(r"(?<![\w-])\d{5}(?![\w-])")
_STREET = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z][\w.'-]*\s+){1,4}(?:Street|St|Avenue|Ave|Lane|Ln|Road|Rd|Drive|Dr|"
    r"Boulevard|Blvd|Court|Ct|Place|Pl|Way|Terrace|Circle|Parkway|Highway)\b\.?")


def _is_fallback(action: dict[str, Any]) -> bool:
    return action.get("type") == "message" and \
        (action.get("content") or "").strip() in FALLBACK_TEXTS


def _token_in(value: str, text: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(value)}(?![\w])", text, re.IGNORECASE) is not None


def _named(address: dict[str, str], text: str) -> bool:
    return all(_token_in(v, text) for v in address.values() if v)


def _addr(a: dict[str, Any]) -> dict[str, str]:
    return {k: str(a.get(k, "")) for k in ADDRESS_FIELDS}


# --------------------------------------------------------------------------- visible state


def _bodies(case: dict[str, Any]) -> list[dict[str, Any]]:
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
    found: list[dict[str, str]] = []
    for body in _bodies(case):
        if isinstance(body.get("address"), dict) and (a := _addr(body["address"])) not in found:
            found.append(a)
    return found


def shown_orders(case: dict[str, Any]) -> set[str]:
    return {b["order_id"] for b in _bodies(case) if isinstance(b.get("order_id"), str)}


def shown_variants(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for body in _bodies(case):
        if isinstance(body.get("variants"), dict):
            out.update(body["variants"])
    return out


def shown_item_ids(case: dict[str, Any]) -> set[str]:
    ids = set(shown_variants(case))
    for body in _bodies(case):
        for item in body.get("items") or []:
            if isinstance(item, dict) and item.get("item_id"):
                ids.add(str(item["item_id"]))
    return ids


def write_done(case: dict[str, Any]) -> bool:
    msgs = case["messages"]
    return any(c["name"] in WRITE_TOOLS and not msgs[i + 1].get("error")
               for i, m in enumerate(msgs) for c in m.get("tool_calls") or [])


def is_clarification(case: dict[str, Any], text: str) -> bool:
    """A question or confirmation request: claims no write (unless one really
    happened) and names no item id the lookups did not show."""
    text = (text or "").strip()
    if not text or text in FALLBACK_TEXTS or not _QUESTION.search(text):
        return False
    if _WRITE_CLAIM.search(text) and not write_done(case):
        return False
    shown = shown_item_ids(case)
    return all(v in shown for v in _ITEM_ID.findall(text))


# --------------------------------------------------------------------------- proposals


def _plan(case: dict[str, Any]) -> dict[str, Any]:
    """Values a proposal may name: the planned new address(es), the current
    address of the target order, the replaced and the planned new item ids."""
    req, gold = case.get("request") or {}, case["gold"]
    args = gold.get("arguments") or {}
    order_id = args.get("order_id") or req.get("order_id")
    addresses = []
    for body in _bodies(case):
        if body.get("order_id") == order_id and isinstance(body.get("address"), dict):
            addresses.append(_addr(body["address"]))  # current (and post-write) address
            break
    if gold.get("name") == ADDRESS_WRITE:
        addresses.append(_addr(args))
    if isinstance(req.get("new_address"), dict):
        addresses.append(_addr(req["new_address"]))
    items = set(args.get("item_ids") or []) | set(args.get("new_item_ids") or [])
    items |= {v for v in (req.get("item_id"), req.get("new_item_id")) if v}
    return {"order_id": order_id, "addresses": addresses, "items": items}


def _variant_named(options: dict[str, str], text: str) -> bool:
    return all(_token_in(str(v), text) for v in options.values())


def proposal(case: dict[str, Any], text: str) -> dict[str, Any]:
    """What a message proposes. ``conflict``: an unseen or other order id, an
    item id or address outside the plan, or an unseen street/ZIP.
    ``matches_gold``: no conflict and the gold write's value is named."""
    gold = case["gold"]
    plan = _plan(case)
    named_orders = set(_ORDER_ID.findall(text))
    unseen_orders = sorted(o for o in named_orders if o not in shown_orders(case))
    other_order = bool(named_orders) and plan["order_id"] not in named_orders
    shown_items = shown_item_ids(case)
    ids = set(_ITEM_ID.findall(text))
    other_items = sorted(i for i in ids if i not in plan["items"])
    shown = shown_addresses(case)
    other_address = any(_named(a, text) for a in shown if a not in plan["addresses"])
    known = shown + plan["addresses"]
    zips = {a["zip"] for a in known}
    streets = {a["address1"].casefold() for a in known}
    unseen_address = any(z not in zips for z in _ZIP.findall(text)) or any(
        not any(st.casefold().startswith(x) or x.startswith(st.casefold().rstrip("."))
                for x in streets)
        for st in _STREET.findall(text))
    conflict = bool(unseen_orders or other_order or other_items or other_address
                    or unseen_address)
    gold_named = False
    if gold.get("type") in ("tool_call", "confirm"):
        args = gold["arguments"]
        if gold["name"] == ADDRESS_WRITE:
            gold_named = _named(_addr(args), text)
        else:
            new = args["new_item_ids"][0]
            variants = shown_variants(case)
            gold_named = _token_in(new, text)
            if not gold_named and new in variants:
                rivals = [v for k, v in variants.items()
                          if k not in plan["items"] and v.get("available")]
                gold_named = _variant_named(variants[new]["options"], text) and not any(
                    _variant_named(v["options"], text) for v in rivals)
    return {"conflict": conflict, "matches_gold": gold_named and not conflict,
            "unseen_orders": unseen_orders, "other_order": other_order,
            "other_items": other_items, "other_address": other_address,
            "unseen_address": unseen_address, "unseen_items": sorted(ids - shown_items)}


def is_question(case: dict[str, Any], action: dict[str, Any]) -> bool:
    return action.get("type") == "message" and is_clarification(case, action.get("content") or "")


def _sequence_violation(case: dict[str, Any], action: dict[str, Any]) -> bool | None:
    if case.get("task") != "sequence":
        return None
    address_pending = case["gold"].get("name") == ADDRESS_WRITE
    if not address_pending:
        return False
    if action.get("type") == "tool_call":
        return action.get("name") == ITEMS_WRITE
    if action.get("type") == "message":
        text = action.get("content") or ""
        req = case.get("request") or {}
        names_item = bool(req.get("new_item_id")) and _token_in(req["new_item_id"], text)
        names_addr = isinstance(req.get("new_address"), dict) and \
            _named(_addr(req["new_address"]), text)
        return names_item and not names_addr
    return False


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
        target_correct = proposal(case, text)["matches_gold"]
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
        outcome = "missing_confirmation"  # asked, but not naming exactly the gold write
    elif atype == "message" and not _is_fallback(action):
        outcome = "non_question_message"
    else:
        outcome = "no_decision"
    if gtype == "clarify" and outcome == "non_question_message":
        outcome = "missed_clarification"
    out["outcome"] = outcome
    out["sequence_violation"] = _sequence_violation(case, action)
    return out


def summarize(scores: list[dict[str, Any]], kinds: list[str] | None = None) -> dict[str, Any]:
    """Rates of every field (target_correct over rows where it applies), overall and
    by kind when ``kinds`` (parallel to ``scores``) is given."""
    def rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
        r: dict[str, Any] = {"n": len(rows)}
        for f in (*FIELDS, "sequence_violation"):
            vals = [row[f] for row in rows if row.get(f) is not None]
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


# --------------------------------------------------------------------------- runs


def score_run(key: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Join ``{"case_id", "case_hash", "action"}`` rows to the answer key
    (``common.load_answer_key``). A hash mismatch, an unknown or duplicated
    case_id raises; key cases without an action are reported as ``missing``."""
    seen: set[str] = set()
    scored = []
    for row in rows:
        cid = row["case_id"]
        if cid not in key:
            raise ValueError(f"{cid}: not in the answer key")
        if cid in seen:
            raise ValueError(f"{cid}: more than one action")
        if row.get("case_hash") != key[cid]["case_hash"]:
            raise ValueError(f"{cid}: case_hash differs from the answer key")
        seen.add(cid)
        case = key[cid]
        scored.append({"case_id": cid, "kind": case["kind"], "family": case["family"],
                       "group": case["group"], **score(case, row["action"])})
    return {"rows": scored, "missing": sorted(set(key) - seen),
            "summary": summarize(scored, [r["kind"] for r in scored])}


def score_files(key_path: Path, actions_path: Path) -> dict[str, Any]:
    from slh.xp.tasks.common import load_answer_key, read_jsonl

    return score_run(load_answer_key(Path(key_path)), read_jsonl(Path(actions_path)))
