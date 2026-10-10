"""Visible-text VARIANT baseline; no case metadata or database access.

Coverage: one item in one pending/delivered order, literal inventory option
values (case/space/hyphen insensitive), labelled options when values overlap,
inclusive/exclusive dollar ceilings, cheapest, and keep-all-other-options.
Payment must be named in the conversation and confirmation must be explicit.
Uses the first user request and final confirmation in the demo's fixed dialogue;
it does not resolve arbitrary multi-turn edits or payment-source nicknames.
Negation, alternatives, revisions, relative prices, unit conversion and arbitrary
paraphrases are outside this rule's coverage and cause a clarification when
detected. This is a decision baseline, not a general conversation policy agent.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Any


class Unclear(ValueError):
    """Visible evidence cannot support a single safe decision."""


def _norm(text: str) -> str:
    return re.sub(r"[\s_-]+", " ", text.casefold()).strip()


def _phrase(value: str) -> str:
    return r"(?<!\w)" + r"[\s_-]+".join(map(re.escape, _norm(value).split())) + r"(?!\w)"


def _records(messages: list[dict]) -> list[tuple[str, dict]]:
    calls = {}
    records = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            calls[call["id"]] = call
        if message.get("role") == "tool" and not message.get("error"):
            call = calls.get(message.get("tool_call_id"))
            if not call:
                continue
            try:
                data = json.loads(message["content"])
            except (ValueError, TypeError):
                continue
            if isinstance(data, dict):
                records.append((call["name"], data))
    return records


def extract_constraints(text: str, product: dict) -> dict[str, Any]:
    """Extract a literal option lexicon from visible variants, then parse text.

    Raises Unclear for ambiguous syntax; never consults ``case['request']``.
    """
    ceiling = None
    strict = False
    amount = r"(\d+(?:,\d{3})*(?:\.\d{1,2})?)(?!\d|\.\d|,\d)"
    price_pattern = (
        r"(?:at most|no more than|up to|not more than|maximum(?: price)?(?: of)?|"
        r"max(?: price)?(?: of)?|under|below|less than|within|budget(?: of)?|"
        r"(?:price )?(?:<=|≤))\s*\$?\s*" + amount
    )
    # "no more than" is supported; unrelated negation remains unsupported.
    suffix = r"\$\s*" + amount + r"\s*(?:at most|or less|maximum|max)\b"
    prices = list(re.finditer(price_pattern + "|" + suffix, text, re.IGNORECASE))
    if len(prices) > 1:
        raise Unclear("Please give one price ceiling.")
    if prices:
        ceiling = Decimal((prices[0][1] or prices[0][2]).replace(",", ""))
        strict = bool(re.match(r"under|below|less than", prices[0][0], re.IGNORECASE))
        text = text[: prices[0].start()] + " " + text[prices[0].end() :]
    elif "$" in text or re.search(
        r"\b(?:budget|price ceiling|at most|under)\b", text, re.IGNORECASE
    ):
        raise Unclear("Please specify a dollar price ceiling.")
    # A trailing "instead" marks an ordinary replacement, not alternatives.
    text = re.sub(r"\binstead\b[.!?]?\s*$", " ", text, flags=re.IGNORECASE)
    # This exception introduces changed options, not an excluded option value.
    # Strip only the marker so "same except not black" remains unsupported.
    exception_marker = (
        r"\b(?:same|identical)(?:\s+(?:item|one|product|variant|version))?\s+except\b"
    )
    preserves_rest = bool(re.search(exception_marker, text, re.IGNORECASE))
    negation_text = re.sub(exception_marker, " ", text, flags=re.IGNORECASE)
    for variant in product["variants"].values():
        for value in variant["options"].values():
            if re.search(r"\bnot\b", value, re.IGNORECASE):
                negation_text = re.sub(_phrase(value), " ", negation_text, flags=re.IGNORECASE)
    if re.search(
        r"\b(?:not|except|instead|rather|either|or|avoid)\b", negation_text, re.IGNORECASE
    ):
        raise Unclear("Please specify the desired options without alternatives or negation.")
    keep = preserves_rest or bool(
        re.search(
            r"\b(?:keep|leave|retain)\b.*\b(?:other|rest|everything else|all else)\b|"
            r"\b(?:same|unchanged)\b.*\b(?:other|rest)\b|"
            r"\b(?:other|everything else|all else|rest)\b.*\b(?:same|unchanged)\b|"
            r"\b(?:only|just)\s+chang(?:e[ds]?|ing)\b|"
            r"\bchang(?:e[ds]?|ing)\s+(?:only|just)\b|"
            r"\bidentical\s+(?:item|one|product|variant|version)\b",
            text,
            re.IGNORECASE,
        )
    )
    cheapest = bool(
        re.search(
            r"\b(?:cheapest|least expensive|lowest[ -](?:price|priced|cost))\b", text, re.IGNORECASE
        )
    )
    # Product names and identifiers are not option requests.
    text = re.sub(_phrase(product["name"]), " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\S+@\S+\b|#W\w+|\b\d{7,}\b", " ", text)
    vocabulary: dict[str, set[str]] = {}
    for variant in product["variants"].values():
        for key, value in variant["options"].items():
            vocabulary.setdefault(key, set()).add(value)
    chosen = {}
    # Explicit key=value, key: value, key value, or value key disambiguates
    # e.g. a black frame versus black lenses.
    for key, values in vocabulary.items():
        for value in sorted(values, key=len, reverse=True):
            pattern = (
                _phrase(key)
                + r"\s*(?:[:=]|(?:is|to|of)\b)?\s*"
                + _phrase(value)
                + "|"
                + _phrase(value)
                + r"\s+"
                + _phrase(key)
            )
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if key in chosen and chosen[key] != value:
                    raise Unclear("Please give one value per option.")
                chosen[key] = value
            # Consume before testing shorter values of this option.
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    remaining = text
    by_value: dict[str, list[tuple[str, str]]] = {}
    for key, values in vocabulary.items():
        for value in values:
            by_value.setdefault(_norm(value), []).append((key, value))
    # Longest first prevents gold from matching rose gold, wired from wireless,
    # and Wi-Fi from matching Wi-Fi + Cellular.
    for normalized, options in sorted(by_value.items(), key=lambda pair: -len(pair[0])):
        matches = list(re.finditer(_phrase(normalized), remaining, re.IGNORECASE))
        if not matches:
            continue
        if len(options) != 1:
            raise Unclear(f"Which option should have the value {normalized}?")
        key, value = options[0]
        if key in chosen and chosen[key] != value:
            raise Unclear("Please give one value per option.")
        chosen[key] = value
        remaining = re.sub(_phrase(normalized), " ", remaining, flags=re.IGNORECASE)
    if not chosen:
        raise Unclear("Which variant options would you like?")
    # Unknown labelled values must not disappear into the surrounding prose.
    if "=" in remaining or any(
        re.search(_phrase(key) + r"\s*[:=]\s*\S+", remaining, re.IGNORECASE) for key in vocabulary
    ):
        raise Unclear("Please specify an option value from the visible variants.")
    return {
        "options": chosen,
        "keep": keep,
        "ceiling": ceiling,
        "strict": strict,
        "cheapest": cheapest,
    }


def decide(case: dict) -> dict:
    """Return a schema action, including a clarification message on uncertainty.

    Only ``messages`` is read. A clarification uses ``type=message`` as required
    by the action interface; ``type=clarify`` is reserved for gold labels.
    """
    try:
        messages = case["messages"]
        user_messages = [m.get("content", "") for m in messages if m["role"] == "user"]
        if not user_messages:
            raise Unclear("Which order item and options would you like to change?")
        user_text = user_messages[0]
        if len(user_messages) < 2 or not re.match(
            r"\s*(?:yes\b|confirmed\b|I confirm\b|go ahead\b|proceed\b)",
            user_messages[-1],
            re.IGNORECASE,
        ):
            raise Unclear("Please confirm the item change before I proceed.")
        confirmation_syntax = re.sub(
            r"\b(?:don't|do not)\s+need\s+to\s+(?:confirm|ask|check)\b",
            " ",
            user_messages[-1],
            flags=re.IGNORECASE,
        )
        if re.search(
            r"\b(?:but|actually|instead|rather|except|not|don't)\b",
            confirmation_syntax,
            re.IGNORECASE,
        ):
            raise Unclear("Please clarify and confirm the revised item change.")
        records = _records(messages)
        products = {p["product_id"]: p for name, p in records if name == "get_product_details"}
        orders = {o["order_id"]: o for name, o in records if name == "get_order_details"}
        visible_text = " ".join(
            m.get("content", "") or "" for m in messages if m["role"] in ("user", "assistant")
        )
        targets = []
        named_orders = set(re.findall(r"#W\w+", user_text))
        for order in orders.values():
            if named_orders and order["order_id"] not in named_orders:
                continue
            for item in order["items"]:
                if item["product_id"] in products:
                    targets.append((order, item, products[item["product_id"]]))
        named_items = [
            target for target in targets if re.search(_phrase(target[1]["item_id"]), visible_text)
        ]
        if named_items:
            targets = named_items
        if len(targets) != 1:
            raise Unclear("Which order item would you like to change?")
        order, item, product = targets[0]
        if order["status"] not in ("pending", "delivered"):
            raise Unclear("Please clarify the order to change; this order cannot be changed.")
        constraints = extract_constraints(user_text, product)
        matches = []
        for variant in product["variants"].values():
            if not variant["available"] or variant["item_id"] == item["item_id"]:
                continue
            options = variant["options"]
            if any(options.get(k) != v for k, v in constraints["options"].items()):
                continue
            if constraints["keep"] and any(
                options.get(k) != v
                for k, v in item["options"].items()
                if k not in constraints["options"]
            ):
                continue
            price = Decimal(str(variant["price"]))
            cap = constraints["ceiling"]
            if cap is not None and (price > cap or (constraints["strict"] and price == cap)):
                continue
            matches.append(variant)
        if constraints["cheapest"] and matches:
            minimum = min(Decimal(str(v["price"])) for v in matches)
            matches = [v for v in matches if Decimal(str(v["price"])) == minimum]
        if len(matches) != 1:
            raise Unclear(
                "No available variant matches; which options should I change?"
                if not matches
                else "Several variants match; which one would you like?"
            )
        profiles = [
            p
            for name, p in records
            if name == "get_user_details" and p["user_id"] == order["user_id"]
        ]
        methods = profiles[-1]["payment_methods"] if profiles else {}
        named_methods = [mid for mid in methods if re.search(_phrase(mid), visible_text)]
        if len(named_methods) != 1:
            raise Unclear("Which payment method should cover the price difference?")
        method = methods[named_methods[0]]
        difference = Decimal(str(matches[0]["price"])) - Decimal(str(item["price"]))
        if method.get("source") == "gift_card" and Decimal(str(method["balance"])) < difference:
            raise Unclear("Which payment method can cover the price difference?")
        return {
            "type": "tool_call",
            "name": "exchange_delivered_order_items"
            if order["status"] == "delivered"
            else "modify_pending_order_items",
            "arguments": {
                "order_id": order["order_id"],
                "item_ids": [item["item_id"]],
                "new_item_ids": [matches[0]["item_id"]],
                "payment_method_id": named_methods[0],
            },
        }
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, ArithmeticError) as exc:
        return {
            "type": "message",
            "content": str(exc)
            if isinstance(exc, Unclear)
            else "Please clarify the item, its options, and the payment method.",
        }
