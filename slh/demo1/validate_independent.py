"""Offline, independent audit of demo case labels from text and the public DB.

No imports from generators, the baseline, or structured request/resolution fields.
The grammar supports literal inventory values, labelled overlapping options,
price ceilings, cheapest/keep-other-options, and address references by order,
product(s), status, city or payment source. Unsupported/contradictory wording is
an audit failure, not evidence that a gold label is correct. This is a bounded
English grammar, not a semantic oracle. Sealed files require --sealed; this
module never changes permissions or writes files.
The first user message is the request and the last is its confirmation, as in
the fixed demo dialogue. Relative references and arbitrary multi-turn edits
are outside this audit grammar.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal
from pathlib import Path

DB_PATH = (
    Path(__file__).resolve().parents[2]
    / "vendor"
    / "tau2-bench"
    / "data"
    / "tau2"
    / "domains"
    / "retail"
    / "db.json"
)
ADDRESS = ("address1", "address2", "city", "state", "country", "zip")


class AuditError(ValueError):
    """The conversation is unsupported or lacks trustworthy evidence."""


def _rx(value: str) -> str:
    parts = re.split(r"[\s_-]+", value.casefold())
    return r"(?<!\w)" + r"[\s_-]+".join(re.escape(p) for p in parts) + r"(?!\w)"


def _has(value: str, text: str) -> bool:
    return bool(re.search(_rx(value), text, re.IGNORECASE))


def _agrees(shown, actual) -> bool:
    """Allow serialization's optional null defaults; compare all public fields."""
    if isinstance(actual, dict):
        return (
            isinstance(shown, dict)
            and all(k in shown and _agrees(shown[k], v) for k, v in actual.items())
            and all(k in actual or v is None for k, v in shown.items())
        )
    if isinstance(actual, list):
        return (
            isinstance(shown, list)
            and len(shown) == len(actual)
            and all(_agrees(a, b) for a, b in zip(shown, actual))
        )
    return (
        type(shown) is type(actual)
        and shown == actual
        or (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and isinstance(shown, (int, float))
            and not isinstance(shown, bool)
            and shown == actual
        )
    )


def _evidence(messages: list[dict], db: dict) -> tuple[dict, dict, dict]:
    """Authenticate and verify every paired public read result against the DB."""
    profiles, orders, products = {}, {}, {}
    pending = None
    authenticated = None
    for message in messages:
        if pending is not None:
            if message.get("role") != "tool" or message.get("tool_call_id") != pending["id"]:
                raise AuditError("tool result does not immediately follow its call")
            name, args = pending["name"], pending["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            content = message.get("content", "")
            if name == "find_user_id_by_email":
                found = [
                    u["user_id"]
                    for u in db["users"].values()
                    if u["email"].casefold() == args["email"].casefold()
                ]
                if len(found) != 1 or message.get("error"):
                    raise AuditError("authentication failed")
                try:
                    result = json.loads(content)
                except ValueError:
                    result = content
                if result != found[0]:
                    raise AuditError("authentication result disagrees with public DB")
                authenticated = result
            elif name in ("get_user_details", "get_order_details", "get_product_details"):
                collection, key, destination = {
                    "get_user_details": ("users", "user_id", profiles),
                    "get_order_details": ("orders", "order_id", orders),
                    "get_product_details": ("products", "product_id", products),
                }[name]
                actual = db[collection].get(args[key])
                if actual is None:
                    if not content.startswith("Error:"):
                        raise AuditError("missing entity must have an error tool result")
                else:
                    if authenticated is None:
                        raise AuditError("details shown before authentication")
                    if collection == "users" and args[key] != authenticated:
                        raise AuditError("profile belongs to another user")
                    if collection == "orders" and actual["user_id"] != authenticated:
                        raise AuditError("order belongs to another user")
                    if message.get("error") or not _agrees(json.loads(content), actual):
                        raise AuditError(f"visible {name} result disagrees with public DB")
                    destination[args[key]] = actual
            else:
                raise AuditError(f"unsupported evidence tool: {name}")
            pending = None
            continue
        if message.get("role") == "tool":
            raise AuditError("unpaired tool result")
        calls = message.get("tool_calls") or []
        if calls:
            if message.get("role") != "assistant" or len(calls) != 1:
                raise AuditError("invalid tool call sequence")
            pending = calls[0]
    if pending is not None or not profiles:
        raise AuditError("missing tool result or authenticated profile")
    return next(iter(profiles.values())), orders, products


def _variant_words(text: str, product: dict) -> tuple[dict, bool, Decimal | None, bool, bool]:
    """Independent parser: labelled assignments, then longest literal mentions."""
    money = r"(\d+(?:,\d{3})*(?:\.\d{1,2})?)(?!\d|\.\d|,\d)"
    price = re.compile(
        r"\b(at most|no more than|not more than|up to|under|below|less than|within|"
        r"max(?:imum)?(?: price)?(?: of)?|budget(?: of)?)\s*\$?\s*"
        + money
        + r"|(?:<=|≤)\s*\$?\s*"
        + money
        + r"|\$\s*"
        + money
        + r"\s*(?:at most|or less|maximum|max)\b",
        re.IGNORECASE,
    )
    ceilings = list(price.finditer(text))
    if len(ceilings) > 1:
        raise AuditError("multiple price ceilings")
    cap, exclusive = None, False
    if ceilings:
        match = ceilings[0]
        cap = Decimal((match[2] or match[3] or match[4]).replace(",", ""))
        exclusive = (match[1] or "").casefold() in ("under", "below", "less than")
        text = text[: match.start()] + " " + text[match.end() :]
    text = re.sub(r"\binstead\b[.!?]?\s*$", " ", text, flags=re.IGNORECASE)
    # "Same except ..." names the changes while preserving other attributes;
    # consume just that exception marker, leaving any actual negation to fail.
    same_except = r"\b(?:same|identical)(?:\s+(?:item|one|product|variant|version))?\s+except\b"
    keep_except = bool(re.search(same_except, text, re.IGNORECASE))
    syntax = re.sub(same_except, " ", text, flags=re.IGNORECASE)
    literal_negations = {
        v
        for variant in product["variants"].values()
        for v in variant["options"].values()
        if _has("not", v)
    }
    for value in literal_negations:
        syntax = re.sub(_rx(value), " ", syntax, flags=re.IGNORECASE)
    if "$" in text or re.search(
        r"\b(?:not|except|either|or|instead|rather|avoid|budget|price ceiling|at most|under)\b",
        syntax,
        re.IGNORECASE,
    ):
        raise AuditError("unsupported price, negation, revision or alternative")
    keep = keep_except or bool(
        re.search(
            r"\b(?:keep|leave|retain)\b.*\b(?:other|rest|everything else|all else)\b|"
            r"\b(?:other|rest|everything else|all else)\b.*\b(?:same|unchanged)\b|"
            r"\b(?:same|unchanged)\b.*\b(?:other|rest)\b|"
            r"\b(?:only|just)\s+chang(?:e[ds]?|ing)\b|"
            r"\bchang(?:e[ds]?|ing)\s+(?:only|just)\b|"
            r"\bidentical\s+(?:item|one|product|variant|version)\b",
            text,
            re.IGNORECASE,
        )
    )
    cheap = bool(
        re.search(
            r"\b(?:cheapest|least expensive|lowest[ -](?:price|priced|cost))\b", text, re.IGNORECASE
        )
    )
    text = re.sub(_rx(product["name"]), " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\S+@\S+\b|#W\w+|\b\d{7,}\b", " ", text)
    lexicon = {}
    for variant in product["variants"].values():
        for key, val in variant["options"].items():
            lexicon.setdefault(key, set()).add(val)
    assignments = {}
    for key, values in lexicon.items():
        for val in sorted(values, key=len, reverse=True):
            pattern = (
                _rx(key)
                + r"\s*(?:[:=]|(?:is|to|of)\b)?\s*"
                + _rx(val)
                + "|"
                + _rx(val)
                + r"\s+"
                + _rx(key)
            )
            if re.search(pattern, text, re.IGNORECASE):
                if key in assignments and assignments[key] != val:
                    raise AuditError("conflicting option values")
                assignments[key] = val
                text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    mentions = []
    for key, values in lexicon.items():
        for val in values:
            mentions.extend(
                (m.start(), m.end(), key, val) for m in re.finditer(_rx(val), text, re.IGNORECASE)
            )
    occupied = set()
    for start, end, key, val in sorted(mentions, key=lambda m: (-(m[1] - m[0]), m[0], m[2])):
        span = set(range(start, end))
        if span & occupied:
            continue
        possible = {(k, v) for a, b, k, v in mentions if a == start and b == end}
        if len(possible) != 1:
            raise AuditError("unlabelled value belongs to multiple options")
        if key in assignments and assignments[key] != val:
            raise AuditError("conflicting option values")
        assignments[key] = val
        occupied.update(span)
    if not assignments:
        raise AuditError("no recognizable variant constraints")
    leftover = "".join(" " if i in occupied else c for i, c in enumerate(text))
    if "=" in leftover or any(
        re.search(_rx(key) + r"\s*[:=]\s*\S+", leftover, re.IGNORECASE) for key in lexicon
    ):
        raise AuditError("unknown labelled option value")
    return assignments, keep, cap, exclusive, cheap


def _variant(text: str, speech: str, user: dict, orders: dict, products: dict) -> dict:
    ids = set(re.findall(r"#W\w+", text))
    targets = [
        (o, i, products[i["product_id"]])
        for o in orders.values()
        for i in o["items"]
        if i["product_id"] in products and (not ids or o["order_id"] in ids)
    ]
    explicit = [t for t in targets if _has(t[1]["item_id"], speech)]
    if explicit:
        targets = explicit
    if len(targets) != 1:
        raise AuditError("cannot identify one target item from visible evidence")
    order, current, product = targets[0]
    if order["status"] not in ("pending", "delivered"):
        raise AuditError("target order status forbids item change")
    opts, keep, cap, exclusive, cheap = _variant_words(text, product)
    candidates = []
    for variant in product["variants"].values():
        if not variant["available"] or variant["item_id"] == current["item_id"]:
            continue
        requirements = dict(current["options"]) if keep else {}
        requirements.update(opts)
        if not all(variant["options"].get(k) == v for k, v in requirements.items()):
            continue
        cost = Decimal(str(variant["price"]))
        if cap is None or cost < cap or (cost == cap and not exclusive):
            candidates.append(variant)
    if cheap and candidates:
        low = min(Decimal(str(v["price"])) for v in candidates)
        candidates = [v for v in candidates if Decimal(str(v["price"])) == low]
    if len(candidates) != 1:
        return {"type": "clarify", "reason": "ambiguous" if candidates else "no_match"}
    payments = [mid for mid in user["payment_methods"] if _has(mid, speech)]
    if len(payments) != 1:
        raise AuditError("no unique visible payment method")
    method = user["payment_methods"][payments[0]]
    diff = Decimal(str(candidates[0]["price"])) - Decimal(str(current["price"]))
    if method["source"] == "gift_card" and Decimal(str(method["balance"])) < diff:
        raise AuditError("gift card cannot cover the price difference")
    return {
        "type": "tool_call",
        "name": "exchange_delivered_order_items"
        if order["status"] == "delivered"
        else "modify_pending_order_items",
        "arguments": {
            "order_id": order["order_id"],
            "item_ids": [current["item_id"]],
            "new_item_ids": [candidates[0]["item_id"]],
            "payment_method_id": payments[0],
        },
    }


def _address_reference(text: str, target_id: str) -> str:
    """Separate an address's source description from the order being changed.

    Prefer explicit on/of/from/used-for phrases. A phrase describing the
    target ("change the address on order #W...") is not a source. Keep the
    older where/same-place forms as a fallback, removing only a descriptor
    attached to the target id, never the source's "other pending order".
    """
    starts = list(re.finditer(
        r"\b(?:address|one)\s+(?:on|of|from|(?:that\s+was\s+)?used\s+for)\s+",
        text, re.IGNORECASE,
    ))
    for start in reversed(starts):
        source = text[start.end():]
        # Public templates may put the target after the source description.
        source = re.split(
            r"\s+for\s+(?:(?:my|the|pending)\s+)*order\s+" + _rx(target_id),
            source, maxsplit=1, flags=re.IGNORECASE,
        )[0]
        # "the Boston address from my other order" carries its constraint
        # before "address"; retain the whole reference in that form.
        if not _has(target_id, source) and not re.match(
            r"(?:(?:my|your|the)\s+)?other\s+orders?\b", source, re.IGNORECASE,
        ):
            return source
    return re.sub(
        r"\b(?:(?:my|the)\s+)?(?:pending\s+)?order\s+" + _rx(target_id),
        " ", text, flags=re.IGNORECASE,
    )


def _address(text: str, user: dict, orders: dict, db: dict) -> dict:
    pending = [o for o in orders.values() if o["status"] == "pending" and _has(o["order_id"], text)]
    if len(pending) != 1:
        # An explicit source id may itself be pending; the first mentioned id
        # identifies the target in the demo's imperative shipping request.
        first = re.search(r"#W\w+", text)
        pending = [o for o in pending if first and o["order_id"] == first[0]]
    if len(pending) != 1:
        raise AuditError("cannot identify one pending target order")
    target = pending[0]
    if set(user["orders"]) - set(orders):
        raise AuditError("address decision lacks complete visible order history")
    reference = _address_reference(text, target["order_id"])
    reference = re.sub(_rx(target["order_id"]), " ", reference, flags=re.IGNORECASE)
    ids = set(re.findall(r"#W\w+", reference))
    names = [p["name"] for p in db["products"].values() if _has(p["name"], reference)]
    cities = {
        o["address"]["city"] for o in db["orders"].values() if _has(o["address"]["city"], reference)
    }
    statuses = []
    for status, pattern in {
        "delivered": r"\b(?:delivered|received|arrived)\b",
        "cancelled": r"\b(?:cancelled|canceled)\b",
        "processed": r"\b(?:processed|processing)\b",
        "pending": r"\bpending\b",
    }.items():
        if re.search(pattern, reference, re.IGNORECASE):
            statuses.append(status)
    methods = {mid for mid in user["payment_methods"] if _has(mid, reference)}
    for source, pattern in {
        "paypal": r"\bpaypal\b",
        "gift_card": r"\bgift[ -]card\b",
        "credit_card": r"\b(?:credit[ -]card|card\s+ending)\b",
    }.items():
        if re.search(pattern, reference, re.IGNORECASE):
            matches = {
                mid for mid, method in user["payment_methods"].items() if method["source"] == source
            }
            if source == "credit_card":
                digits = set(re.findall(r"(?<!\d)\d{4}(?!\d)", reference))
                brands = {
                    method.get("brand")
                    for method in user["payment_methods"].values()
                    if method.get("brand") and _has(method["brand"], reference)
                }
                matches = {
                    mid
                    for mid in matches
                    if (not digits or user["payment_methods"][mid].get("last_four") in digits)
                    and (not brands or user["payment_methods"][mid].get("brand") in brands)
                }
            if len(matches) != 1:
                raise AuditError("payment-source reference is not unique")
            methods.update(matches)
    if not (ids or names or cities or statuses or methods):
        raise AuditError("unrecognized address reference")
    if len(ids) > 1 or len(cities) > 1 or len(statuses) > 1 or len(methods) > 1:
        raise AuditError("multiple address reference constraints are unsupported")
    addresses = {}
    for oid in user["orders"]:
        if oid == target["order_id"]:
            continue
        order = orders[oid]
        if ids and oid not in ids:
            continue
        if names and not set(names) <= {i["name"] for i in order["items"]}:
            continue
        if cities and order["address"]["city"] not in cities:
            continue
        if statuses and order["status"] not in statuses:
            continue
        first_payment = next(
            (p for p in order["payment_history"] if p["transaction_type"] == "payment"), None
        )
        if methods and (first_payment is None or first_payment["payment_method_id"] not in methods):
            continue
        address = {key: order["address"][key] for key in ADDRESS}
        addresses[tuple(address[k] for k in ADDRESS)] = address
    if len(addresses) != 1:
        return {"type": "clarify", "reason": "ambiguous" if addresses else "no_match"}
    return {
        "type": "tool_call",
        "name": "modify_pending_order_address",
        "arguments": {"order_id": target["order_id"], **next(iter(addresses.values()))},
    }


def derive_gold(case: dict, db: dict) -> dict:
    """Re-derive an action without inspecting gold or any generator metadata."""
    messages = case["messages"]
    if not messages or messages[-1].get("role") != "tool":
        raise AuditError("decision must follow visible tool evidence")
    user_messages = [m.get("content", "") for m in messages if m["role"] == "user"]
    if not user_messages:
        raise AuditError("missing user request")
    text = user_messages[0]
    speech = " ".join(
        m.get("content", "") or "" for m in messages if m["role"] in ("user", "assistant")
    )
    if len(user_messages) < 2 or not re.match(
        r"\s*(?:yes\b|confirmed\b|I confirm\b|go ahead\b|proceed\b)",
        user_messages[-1],
        re.IGNORECASE,
    ):
        raise AuditError("missing explicit user confirmation")
    confirmation_syntax = re.sub(
        r"\b(?:don't|do not)\s+need\s+to\s+(?:confirm|ask|check)\b",
        " ",
        user_messages[-1],
        flags=re.IGNORECASE,
    )
    if re.search(
        r"\b(?:but|actually|instead|rather|except|not|don't)\b", confirmation_syntax, re.IGNORECASE
    ):
        raise AuditError("confirmation changes or negates the original request")
    user, orders, products = _evidence(messages, db)
    if case["task"] == "variant":
        return _variant(text, speech, user, orders, products)
    if case["task"] == "address":
        return _address(text, user, orders, db)
    raise AuditError("unknown decision task")


def validate_case(case: dict, db: dict) -> list[str]:
    """Return audit failures; an empty list means the visible gold agrees."""
    try:
        expected = derive_gold(case, db)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, ArithmeticError) as exc:
        return [f"cannot independently derive gold: {exc}"]
    if expected != case.get("gold"):
        return ["gold disagreement: expected " + json.dumps(expected, sort_keys=True)]
    return []


def _guard_path(path: Path, allow_sealed: bool) -> None:
    # Check both spelling and resolved destination so symlinks cannot bypass it.
    if not allow_sealed and any("sealed" in p.parts for p in (path.absolute(), path.resolve())):
        raise AuditError("refusing to read sealed/ without --sealed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument(
        "--sealed", action="store_true", help="explicitly allow reading sealed cases"
    )
    args = parser.parse_args(argv)
    try:
        _guard_path(args.cases, args.sealed)
        db = json.loads(DB_PATH.read_text(encoding="utf-8"))
        failures, count = 0, 0
        seen = set()
        with args.cases.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                count += 1
                try:
                    case = json.loads(line)
                    if not isinstance(case, dict):
                        raise AuditError("case must be an object")
                    if case.get("split") == "sealed" and not args.sealed:
                        raise AuditError("sealed-labelled case requires --sealed")
                    case_id = case.get("case_id", f"line-{line_number}")
                    problems = validate_case(case, db)
                    if case_id in seen:
                        problems.append("duplicate case id")
                    seen.add(case_id)
                except (ValueError, KeyError, TypeError) as exc:
                    case_id, problems = f"line-{line_number}", [str(exc)]
                if problems:
                    failures += 1
                    print(
                        json.dumps(
                            {"line": line_number, "case_id": case_id, "errors": problems},
                            sort_keys=True,
                        )
                    )
        print(json.dumps({"cases": count, "failed": failures}, sort_keys=True))
        return 1 if failures else 0
    except (OSError, ValueError) as exc:
        print(f"audit refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
