"""Deterministic case generator for demo 1 (``demo1-gen-v1``).

Inputs: the public retail ``db.json`` and the tau2 retail tools (for the exact
tool result strings). No model call, no official task, no reference action.

Per task: 7 request kinds x 4 phrasings = 28 families, 3 mirrored cases each
(``act``, ``act_distractor``, ``clarify``). For each family and role the
generator enumerates every database configuration that fits, keeps the users
that no other split of the task uses (preferring users not used at all), and
draws with a seeded RNG (``sha256(seed|family|role)``). The split of families
(``demo1-split-v1``) is fixed before any configuration is drawn.

Sealed cases go to ``sealed/demo1/cases/`` (mode 000 after writing, the
repository's sealing convention); ``demo1/manifest.json`` records the SHA-256 of
every case file, so a re-generation is checked without opening the sealed tree.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Iterator
from itertools import combinations
from pathlib import Path
from typing import Any

from slh.demo1 import schema
from slh.demo1.schema import (
    ADDRESS_FIELDS,
    GREETING,
    KINDS,
    PHRASINGS,
    REPO,
    ROLES,
    SCHEMA,
    canonical,
    family_id,
    gold_tool_for_status,
)

GEN_VERSION = "demo1-gen-v1"
SEED = 4401
SPLIT_SALT = "demo1-split-v1"
MAX_ADDRESS_ORDERS = 5  # keeps the conversation short (every order is looked up)
STATUSES = ("delivered", "processed", "cancelled", "pending")
DB_PATH = REPO / "vendor/tau2-bench/data/tau2/domains/retail/db.json"
OUT = REPO / "demo1"
CASES_DIR = OUT / "cases"
SEALED_DIR = REPO / "sealed/demo1"
SEALED_CASES = SEALED_DIR / "cases"
MANIFEST = OUT / "manifest.json"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_db(path: Path = DB_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def db_sha256(path: Path = DB_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- split


def split_plan(task: str) -> dict[str, str]:
    """family id -> split (docs/demo1/case-schema.md, "Families and the frozen split")."""
    kinds = sorted(KINDS[task], key=lambda k: (_sha(f"{SPLIT_SALT}|{task}|{k}"), k))
    plan: dict[str, str] = {}
    ordered: dict[str, list[str]] = {}
    for kind in kinds:
        fams = [family_id(task, kind, p) for p in range(PHRASINGS)]
        ordered[kind] = sorted(fams, key=lambda f: (_sha(f"{SPLIT_SALT}|{task}|{f}"), f))
        for fam in fams:
            plan[fam] = "sealed"
    for kind in kinds[:4]:
        plan[ordered[kind][0]] = "train"
    for kind in kinds[4:]:
        plan[ordered[kind][0]] = "calibrate"
    plan[ordered[kinds[0]][1]] = "calibrate"
    return plan


# --------------------------------------------------------------------------- db helpers


def addr_key(address: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(address.get(f, "")) for f in ADDRESS_FIELDS)


def addr_args(address: dict[str, Any]) -> dict[str, str]:
    return {f: str(address.get(f, "")) for f in ADDRESS_FIELDS}


def first_payment(order: dict[str, Any]) -> str | None:
    for p in order["payment_history"]:
        if p["transaction_type"] == "payment":
            return p["payment_method_id"]
    return None


def pm_desc(pm: dict[str, Any], *, you: bool = False) -> str:
    owner = "your" if you else "my"
    if pm["source"] == "credit_card":
        return f"{owner} {str(pm.get('brand', 'credit')).title()} card ending in {pm['last_four']}"
    if pm["source"] == "paypal":
        return f"{owner} PayPal account"
    return f"{owner} gift card"


def names(order: dict[str, Any]) -> set[str]:
    return {item["name"] for item in order["items"]}


# --------------------------------------------------------------------------- address


def address_sources(db: dict[str, Any], user: dict[str, Any], target: dict[str, Any],
                    referent: dict[str, Any]) -> list[dict[str, Any]]:
    """Orders of ``user`` other than ``target`` that the referent names."""
    out = []
    for oid in user["orders"]:
        order = db["orders"].get(oid)
        if order is None or oid == target["order_id"]:
            continue
        if address_matches(order, referent):
            out.append(order)
    return out


def address_matches(order: dict[str, Any], referent: dict[str, Any]) -> bool:
    kind = referent["kind"]
    if kind == "order_id":
        return order["order_id"] == referent["order_id"]
    if kind == "product":
        return referent["product"] in names(order)
    if kind == "status":
        return order["status"] == referent["status"]
    if kind == "product_status":
        return referent["product"] in names(order) and order["status"] == referent["status"]
    if kind == "city":
        return order["address"]["city"] == referent["city"]
    if kind == "payment":
        return first_payment(order) == referent["payment_method_id"]
    if kind == "two_products":
        return set(referent["products"]) <= names(order)
    raise ValueError(f"unknown referent kind {kind!r}")


def resolve_address(db: dict[str, Any], user: dict[str, Any], target: dict[str, Any],
                    referent: dict[str, Any]) -> dict[str, Any]:
    sources = address_sources(db, user, target, referent)
    keys = sorted({addr_key(o["address"]) for o in sources})
    outcome = "unique" if len(keys) == 1 else "no_match" if not keys else "ambiguous"
    others = [db["orders"][o] for o in user["orders"]
              if o in db["orders"] and o != target["order_id"]]
    near = 0
    if outcome == "unique":
        near = sum(1 for o in others if o not in sources and addr_key(o["address"]) != keys[0])
        near += len(sources) - 1
    return {"outcome": outcome, "candidates": [dict(zip(ADDRESS_FIELDS, k)) for k in keys],
            "near_misses": near}


def _fake_order_id(db: dict[str, Any], salt: str) -> str:
    n = int(_sha(salt)[:12], 16)
    while True:
        oid = f"#W{n % 9_000_000 + 1_000_000}"
        if oid not in db["orders"]:
            return oid
        n += 7919


def address_referents(db: dict[str, Any], user: dict[str, Any], target: dict[str, Any],
                      kind: str, all_products: list[str], all_cities: list[str]
                      ) -> Iterator[dict[str, Any]]:
    others = [db["orders"][o] for o in user["orders"]
              if o in db["orders"] and o != target["order_id"]]
    owned = set().union(*(names(o) for o in [target, *others]))
    other_names = sorted(set().union(*(names(o) for o in others))) if others else []
    if kind == "order_id":
        for o in others:
            yield {"kind": kind, "order_id": o["order_id"]}
        yield {"kind": kind, "order_id": _fake_order_id(db, f"{user['user_id']}|{target['order_id']}")}
    elif kind == "product":
        for name in other_names:
            yield {"kind": kind, "product": name}
        for name in [p for p in all_products if p not in owned][:3]:
            yield {"kind": kind, "product": name}
    elif kind == "status":
        for status in STATUSES:
            yield {"kind": kind, "status": status}
    elif kind == "product_status":
        for name in other_names:
            for status in STATUSES:
                yield {"kind": kind, "product": name, "status": status}
    elif kind == "city":
        cities = sorted({o["address"]["city"] for o in others})
        known = cities + [target["address"]["city"], user["address"]["city"]]
        for city in cities:
            yield {"kind": kind, "city": city}
        for city in [c for c in all_cities if c not in known][:2]:
            yield {"kind": kind, "city": city}
    elif kind == "payment":
        descs = Counter(pm_desc(pm) for pm in user["payment_methods"].values())
        for pid, pm in sorted(user["payment_methods"].items()):
            if descs[pm_desc(pm)] == 1:
                yield {"kind": kind, "payment_method_id": pid}
    elif kind == "two_products":
        for a, b in combinations(other_names, 2):
            yield {"kind": kind, "products": [a, b]}


def _target_matches(target: dict[str, Any], referent: dict[str, Any]) -> bool:
    if referent["kind"] in ("order_id", "status"):
        return False  # the target is excluded by id; "other pending order" for status
    return address_matches(target, referent)


def address_configs(db: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    all_products = sorted({p["name"] for p in db["products"].values()})
    all_cities = sorted({u["address"]["city"] for u in db["users"].values()})
    out = []
    for uid in sorted(db["users"]):
        user = db["users"][uid]
        orders = [db["orders"][o] for o in user["orders"] if o in db["orders"]]
        if not 2 <= len(orders) <= MAX_ADDRESS_ORDERS:
            continue
        for target in orders:
            if target["status"] != "pending":
                continue
            for referent in address_referents(db, user, target, kind, all_products, all_cities):
                if _target_matches(target, referent):
                    continue
                res = resolve_address(db, user, target, referent)
                if res["outcome"] == "unique" and \
                        addr_key(res["candidates"][0]) == addr_key(target["address"]):
                    continue
                out.append({"user_id": uid, "target_order_id": target["order_id"],
                            "referent": referent, "resolution": res})
    return out


_STATUS_ME = {"delivered": "the order I already received",
              "processed": "my order that is still being processed",
              "cancelled": "the order I cancelled", "pending": "my other pending order"}
_STATUS_YOU = {"delivered": "the order you already received",
               "processed": "your order that is still being processed",
               "cancelled": "the order you cancelled", "pending": "your other pending order"}
_STATUS_ADJ = {"delivered": "delivered", "processed": "processed", "cancelled": "cancelled",
               "pending": "other pending"}

ADDRESS_TEMPLATES: dict[str, tuple[tuple[str, ...], str]] = {
    "order_id": ((
        ("Hi, I need to change the shipping address of my pending order {target}. Please send "
        "it to the address that order {order} was shipped to."),
        "Could you ship my order {target} to the same address as my order {order}?",
        ("My pending order {target} is going to the wrong place. Use the address from order "
        "{order} instead."),
        "Please update order {target} so it goes to wherever order {order} was sent."),
        "the address of order {order}"),
    "product": ((
        ("Hi, please change the shipping address of my pending order {target} to the address "
        "my {product} order went to."),
        "Can you send order {target} to the same place as the order with my {product}?",
        "I want order {target} shipped where I had my {product} sent.",
        ("For order {target}, please use the address from the order where I bought the "
        "{product}.")),
        "the address of your {product} order"),
    "status": ((
        "Hi, please ship my pending order {target} to the address of {status_me}.",
        "Could order {target} go to the same address as {status_me}?",
        "Change the address on order {target} to the one used for {status_me}.",
        "Please send order {target} to the address on {status_me}."),
        "the address of {status_you}"),
    "product_status": ((
        ("Hi, please change the address of pending order {target} to the address of my "
        "{status_adj} {product} order."),
        "Can order {target} go to the same address as my {status_adj} {product} order?",
        "Use the address from my {status_adj} {product} order for order {target}, please.",
        "I'd like order {target} sent where my {status_adj} {product} order was going."),
        "the address of your {status_adj} {product} order"),
    "city": ((
        ("Hi, please ship my pending order {target} to my {city} address, the one I used on "
        "another order."),
        "Can you send order {target} to the address in {city} from one of my other orders?",
        ("Order {target} should go to {city} instead. Use the {city} address from my other "
        "order."),
        "Please change order {target} to the {city} address I have used before."),
        "the {city} address from your other orders"),
    "payment": ((
        ("Hi, please send my pending order {target} to the address of the order I paid with "
        "{payment_me}."),
        ("Could order {target} go to the same address as the order I paid for with "
        "{payment_me}?"),
        "Use the address of the order I bought with {payment_me} for order {target}.",
        "I want order {target} shipped to the address of my order paid by {payment_me}."),
        "the address of the order you paid with {payment_you}"),
    "two_products": ((
        ("Hi, please change the address of my pending order {target} to the address of the "
        "order that had both the {p1} and the {p2}."),
        "Can you ship order {target} to the address of my order with the {p1} and the {p2}?",
        "Order {target} should go where my {p1} and {p2} order went.",
        ("For order {target}, use the address of the order where I got the {p1} together with "
        "the {p2}.")),
        "the address of your order with both the {p1} and the {p2}"),
}

YES = ("Yes, please go ahead. No need to check back with me.",
       "Yes. Go ahead and do it, you don't need to confirm with me again.",
       "Yes, proceed. No need to ask me again.",
       "Yes please. Just do it without checking back with me.")


def _address_fields(db: dict[str, Any], user: dict[str, Any], config: dict[str, Any]
                    ) -> dict[str, str]:
    ref = config["referent"]
    f: dict[str, str] = {"target": config["target_order_id"]}
    if ref["kind"] == "order_id":
        f["order"] = ref["order_id"]
    if "product" in ref:
        f["product"] = ref["product"]
    if "status" in ref:
        f |= {"status_me": _STATUS_ME[ref["status"]], "status_you": _STATUS_YOU[ref["status"]],
              "status_adj": _STATUS_ADJ[ref["status"]]}
    if ref["kind"] == "city":
        f["city"] = ref["city"]
    if ref["kind"] == "payment":
        pm = user["payment_methods"][ref["payment_method_id"]]
        f |= {"payment_me": pm_desc(pm), "payment_you": pm_desc(pm, you=True)}
    if ref["kind"] == "two_products":
        f |= {"p1": ref["products"][0], "p2": ref["products"][1]}
    return f


# --------------------------------------------------------------------------- variant


def variant_candidates(product: dict[str, Any], current: dict[str, Any],
                       request: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """(matching available variants, near misses: variants failing exactly one check)."""
    out, near = [], 0
    for item_id in sorted(product["variants"]):
        v = product["variants"][item_id]
        if item_id == current["item_id"]:
            continue
        failed = 0 if v["available"] else 1
        for k, val in request["set"].items():
            failed += v["options"].get(k) != val
        if request["keep_others"]:
            for k, val in current["options"].items():
                if k not in request["set"]:
                    failed += v["options"].get(k) != val
        if request["max_price"] is not None:
            failed += v["price"] > request["max_price"] + 1e-9
        if failed == 0:
            out.append(v)
        elif failed == 1:
            near += 1
    return out, near


def resolve_variant(product: dict[str, Any], current: dict[str, Any],
                    request: dict[str, Any]) -> dict[str, Any]:
    cands, near = variant_candidates(product, current, request)
    if not cands:
        return {"outcome": "no_match", "candidates": [], "near_misses": near}
    if request["select"] == "cheapest":
        low = min(v["price"] for v in cands)
        cands = [v for v in cands if abs(v["price"] - low) < 1e-9]
    ids = sorted(v["item_id"] for v in cands)
    return {"outcome": "unique" if len(ids) == 1 else "ambiguous", "candidates": ids,
            "near_misses": near}


def _cap(price: float) -> float:
    return float(5 * math.ceil(price / 5 - 1e-9))


def variant_requests(product: dict[str, Any], current: dict[str, Any], kind: str
                     ) -> Iterator[dict[str, Any]]:
    variants = [product["variants"][k] for k in sorted(product["variants"])]
    attrs = sorted(current["options"])
    values = {a: sorted({v["options"][a] for v in variants if a in v["options"]})
              for a in attrs}

    def req(set_: dict[str, str], keep: bool = False, cap: float | None = None,
            select: str = "only") -> dict[str, Any]:
        return {"set": dict(sorted(set_.items())), "keep_others": keep, "max_price": cap,
                "select": select}

    if kind == "full_spec":
        for v in variants:
            if v["item_id"] != current["item_id"]:
                yield req(dict(v["options"]))
    elif kind == "keep_one":
        for a in attrs:
            for x in values[a]:
                if x != current["options"][a]:
                    yield req({a: x}, keep=True)
    elif kind == "keep_two":
        for a, b in combinations(attrs, 2):
            for x in values[a]:
                for y in values[b]:
                    if x != current["options"][a] and y != current["options"][b]:
                        yield req({a: x, b: y}, keep=True)
    elif kind in ("one_price", "two_price"):
        groups = [(a,) for a in attrs] if kind == "one_price" else list(combinations(attrs, 2))
        for group in groups:
            combos = sorted({tuple(v["options"][a] for a in group) for v in variants})
            for combo in combos:
                set_ = dict(zip(group, combo))
                prices = sorted({_cap(v["price"]) for v in variants
                                 if all(v["options"][a] == set_[a] for a in group)})
                for cap in prices:
                    yield req(set_, cap=cap)
    elif kind == "cheapest_one":
        for a in attrs:
            for x in values[a]:
                yield req({a: x}, select="cheapest")
    elif kind == "one_only":
        for a in attrs:
            for x in values[a]:
                yield req({a: x})


def _payment_for(user: dict[str, Any], order: dict[str, Any]) -> str | None:
    original = first_payment(order)
    pms = user["payment_methods"]
    if original in pms and pms[original]["source"] != "gift_card":
        return original
    for pid in sorted(pms):
        if pms[pid]["source"] != "gift_card":
            return pid
    return None


def variant_configs(db: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    out = []
    for oid in sorted(db["orders"]):
        order = db["orders"][oid]
        if order["status"] not in ("delivered", "pending"):
            continue
        user = db["users"][order["user_id"]]
        pay = _payment_for(user, order)
        if pay is None:
            continue
        counts = Counter(item["product_id"] for item in order["items"])
        for item in order["items"]:
            if counts[item["product_id"]] != 1:
                continue
            product = db["products"][item["product_id"]]
            for request in variant_requests(product, item, kind):
                res = resolve_variant(product, item, request)
                full = {"user_id": user["user_id"], "order_id": oid,
                        "order_status": order["status"], "item_id": item["item_id"],
                        "product_id": item["product_id"], **request,
                        "payment_method_id": pay}
                out.append({"user_id": user["user_id"], "request": full, "resolution": res})
    return out


def _spec(set_: dict[str, str]) -> str:
    return ", ".join(f"{k} {v}" for k, v in set_.items())


def _money(value: float) -> str:
    return f"${value:,.0f}" if value == int(value) else f"${value:,.2f}"


VARIANT_CONSTRAINTS: dict[str, tuple[str, ...]] = {
    "full_spec": ("the version with {spec}", "the one with {spec}",
                  "the variant that has {spec}", "exactly this version: {spec}"),
    "keep_one": ("the same one but with {spec}, keeping everything else the same",
                 "the same item with {spec} and all other options unchanged",
                 "one that only changes to {spec} and keeps the rest as it is",
                 "the identical item, only changed to {spec}"),
    "keep_two": ("the same one but with {spec}, keeping everything else the same",
                 "the same item with {spec} and all other options unchanged",
                 "one that changes only to {spec} and keeps the rest as it is",
                 "the identical item, only changed to {spec}"),
    "one_price": ("one with {spec} that costs at most {cap}", "a {spec} one, {cap} at most",
                  "one with {spec}, no more than {cap}", "one with {spec} for {cap} or less"),
    "two_price": ("one with {spec} that costs at most {cap}", "one with {spec}, {cap} at most",
                  "one with {spec}, no more than {cap}", "one with {spec} for {cap} or less"),
    "cheapest_one": ("the cheapest available one with {spec}",
                     "the lowest-priced one with {spec} that is in stock",
                     "whichever available one with {spec} is cheapest",
                     "the least expensive available version with {spec}"),
    "one_only": ("one with {spec}", "the one with {spec}", "a version with {spec}",
                 "the {spec} one"),
}
VARIANT_OPENINGS = {
    "delivered": ("Hi, I'd like to exchange the {product} from my order {order} for {c}.",
                  "Can I exchange the {product} in order {order}? I want {c}.",
                  "I received order {order}. Please exchange the {product} for {c}.",
                  "Hello, the {product} from order {order} isn't right. I want {c} instead."),
    "pending": ("Hi, I'd like to change the {product} in my pending order {order} to {c}.",
                "Can you modify the {product} in order {order}? I want {c}.",
                "My order {order} hasn't shipped yet. Please change the {product} to {c}.",
                "Hello, for order {order}, please switch the {product} to {c}."),
}


# --------------------------------------------------------------------------- conversations


class Conversation:
    def __init__(self, env: Any, salt: str) -> None:
        self.env = env
        self.salt = salt
        self.messages: list[dict[str, Any]] = [{"role": "assistant", "content": GREETING}]

    def say(self, role: str, text: str) -> None:
        self.messages.append({"role": role, "content": text})

    def call(self, text: str, name: str, arguments: dict[str, Any]) -> str:
        from tau2.data_model.message import ToolCall

        call_id = f"chatcmpl-tool-{_sha(f'{self.salt}|{len(self.messages)}')[:16]}"
        self.messages.append({"role": "assistant", "content": text,
                              "tool_calls": [{"id": call_id, "name": name,
                                              "arguments": arguments}]})
        result = self.env.get_response(ToolCall(id=call_id, name=name, arguments=arguments,
                                                requestor="assistant"))
        self.messages.append({"role": "tool", "content": result.content,
                              "tool_call_id": call_id, "error": bool(result.error)})
        return result.content

    def authenticate(self, user: dict[str, Any]) -> None:
        self.say("assistant", "I'd be happy to help. First, I need to verify your identity. "
                              "Could you please provide your email address?")
        self.say("user", f"Sure, it's {user['email']}.")
        self.call("Let me look up your account.", "find_user_id_by_email",
                  {"email": user["email"]})
        self.call("Let me get your account details.", "get_user_details",
                  {"user_id": user["user_id"]})


def _quiet() -> None:
    try:
        import sys

        from loguru import logger

        logger.remove()
        logger.add(sys.stderr, level="ERROR")
    except ImportError:  # pragma: no cover
        pass


def _env() -> Any:
    from tau2.registry import registry

    _quiet()
    return registry.get_env_constructor("retail")()


def address_case(db: dict[str, Any], env: Any, config: dict[str, Any], kind: str,
                 phrasing: int, salt: str) -> dict[str, Any]:
    user = db["users"][config["user_id"]]
    fields = _address_fields(db, user, config)
    openings, restate = ADDRESS_TEMPLATES[kind]
    conv = Conversation(env, salt)
    conv.say("user", openings[phrasing].format(**fields))
    conv.authenticate(user)
    first = user["name"]["first_name"]
    conv.say("assistant",
             f"Thank you, {first}. To confirm: I will update the shipping address of pending "
             f"order {config['target_order_id']} to {restate.format(**fields)}. I will check "
             "your orders to find it. Do you want me to proceed (yes/no)?")
    conv.say("user", YES[phrasing])
    shown = [o for o in user["orders"]]
    for i, oid in enumerate(shown):
        conv.call("Let me check your orders." if i == 0 else f"Let me check order {oid}.",
                  "get_order_details", {"order_id": oid})
    ref = config["referent"]
    if ref["kind"] == "order_id" and ref["order_id"] not in user["orders"]:
        shown.append(ref["order_id"])
        conv.call(f"Let me check order {ref['order_id']}.", "get_order_details",
                  {"order_id": ref["order_id"]})
    res = config["resolution"]
    if res["outcome"] == "unique":
        gold = {"type": "tool_call", "name": "modify_pending_order_address",
                "arguments": {"order_id": config["target_order_id"],
                              **addr_args(res["candidates"][0])}}
    else:
        gold = {"type": "clarify", "reason": res["outcome"]}
    request = {"user_id": user["user_id"], "target_order_id": config["target_order_id"],
               "referent": ref}
    return {"messages": conv.messages, "gold": gold, "request": request,
            "resolution": res, "entities": {"user_id": user["user_id"], "order_ids": shown}}


def variant_case(db: dict[str, Any], env: Any, config: dict[str, Any], kind: str,
                 phrasing: int, salt: str) -> dict[str, Any]:
    request = config["request"]
    user = db["users"][config["user_id"]]
    product = db["products"][request["product_id"]]
    cap = request["max_price"]
    constraint = VARIANT_CONSTRAINTS[kind][phrasing].format(
        spec=_spec(request["set"]), cap=_money(cap) if cap is not None else "")
    status = request["order_status"]
    conv = Conversation(env, salt)
    conv.say("user", VARIANT_OPENINGS[status][phrasing].format(
        product=product["name"].lower(), order=request["order_id"], c=constraint))
    conv.authenticate(user)
    conv.call("Let me check the order.", "get_order_details", {"order_id": request["order_id"]})
    pm = user["payment_methods"][request["payment_method_id"]]
    verb = "exchange" if status == "delivered" else "modify"
    reminder = ("" if status == "delivered" else
                "Items in a pending order can be modified only once, so please also confirm "
                "this is the only item you want to change in this order. ")
    conv.say("assistant",
             f"Thank you, {user['name']['first_name']}. To confirm: I will {verb} the "
             f"{product['name']} (item {request['item_id']}) in order {request['order_id']} "
             f"for {constraint}. Any price difference will be charged or refunded to "
             f"{pm_desc(pm, you=True)} ({request['payment_method_id']}). I will check the "
             f"available options to find it. {reminder}Do you want me to proceed (yes/no)?")
    yes = YES[phrasing]
    conv.say("user", yes if status == "delivered" else "Yes, that's the only item. " + yes)
    conv.call("Let me check the available options.", "get_product_details",
              {"product_id": request["product_id"]})
    res = config["resolution"]
    if res["outcome"] == "unique":
        gold = {"type": "tool_call", "name": gold_tool_for_status(status),
                "arguments": {"order_id": request["order_id"], "item_ids": [request["item_id"]],
                              "new_item_ids": [res["candidates"][0]],
                              "payment_method_id": request["payment_method_id"]}}
    else:
        gold = {"type": "clarify", "reason": res["outcome"]}
    return {"messages": conv.messages, "gold": gold, "request": request, "resolution": res,
            "entities": {"user_id": user["user_id"], "order_ids": [request["order_id"]]}}


# --------------------------------------------------------------------------- sampling


def _role_pools(configs: list[dict[str, Any]], role: str, phrasing: int
                ) -> list[list[dict[str, Any]]]:
    """Configuration pools for one role, in order of preference. Clarify: even
    phrasings seek ``no_match``, odd ones ``ambiguous``; the other reason is the
    fallback. Act: no near miss preferred; act_distractor: at least one."""
    def outcome(o: str) -> list[dict[str, Any]]:
        return [c for c in configs if c["resolution"]["outcome"] == o]

    if role == "clarify":
        order = ("no_match", "ambiguous") if phrasing % 2 == 0 else ("ambiguous", "no_match")
        return [outcome(o) for o in order]
    unique = outcome("unique")
    if role == "act_distractor":
        return [[c for c in unique if c["resolution"]["near_misses"] >= 1]]
    return [[c for c in unique if c["resolution"]["near_misses"] == 0], unique]


def _draw(pool: list[dict[str, Any]], rng: random.Random, split: str,
          user_split: dict[str, str]) -> dict[str, Any]:
    allowed = [c for c in pool if user_split.get(c["user_id"], split) == split]
    fresh = [c for c in allowed if c["user_id"] not in user_split]
    choices = fresh or allowed
    if not choices:
        raise ValueError("no configuration left for this family and role")
    users = sorted({c["user_id"] for c in choices})
    uid = rng.choice(users)
    mine = [c for c in choices if c["user_id"] == uid]
    return mine[rng.randrange(len(mine))]


def case_id(task: str, family: str, role: str, seed: int) -> str:
    prefix = "d1a" if task == "address" else "d1v"
    return f"{prefix}-{_sha(f'{GEN_VERSION}|{seed}|{task}|{family}|{role}')[:10]}"


def generate_task(task: str, db: dict[str, Any], *, seed: int = SEED, env: Any = None,
                  context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    env = env or _env()
    plan = split_plan(task)
    configs_of = address_configs if task == "address" else variant_configs
    build = address_case if task == "address" else variant_case
    ctx = {"file": "demo1/b1-context.json",
           "system_prompt_sha256": (context or {}).get("system_prompt_sha256"),
           "tools_sha256": (context or {}).get("tools_sha256")}
    dbsha = db_sha256()
    user_split: dict[str, str] = {}
    cases = []
    all_configs = {kind: configs_of(db, kind) for kind in KINDS[task]}

    def scarcity(kind: str) -> tuple[int, str]:
        unique = {c["user_id"] for c in all_configs[kind]
                  if c["resolution"]["outcome"] == "unique"}
        return len(unique), kind

    # Scarcest kinds draw first, so the user pool lasts (deterministic order).
    for kind in sorted(KINDS[task], key=scarcity):
        configs = all_configs[kind]
        for phrasing in range(PHRASINGS):
            fam = family_id(task, kind, phrasing)
            split = plan[fam]
            for role in ("clarify", "act_distractor", "act"):
                rng = random.Random(int(_sha(f"{seed}|{fam}|{role}")[:16], 16))
                config = None
                pools = _role_pools(configs, role, phrasing)
                if task == "variant":
                    # Balance delivered (exchange) and pending (modify) orders.
                    want = ("delivered", "pending")[(phrasing + ROLES.index(role)) % 2]
                    pools = [p for pool in pools for p in (
                        [c for c in pool if c["request"]["order_status"] == want], pool)]
                for pool in pools:
                    try:
                        config = _draw(pool, rng, split, user_split)
                        break
                    except ValueError:
                        continue
                if config is None:
                    raise ValueError(f"{fam} {role}: no configuration left")
                user_split[config["user_id"]] = split
                cid = case_id(task, fam, role, seed)
                body = build(db, env, config, kind, phrasing, cid)
                cases.append({"schema": SCHEMA, "case_id": cid, "task": task, "kind": kind,
                              "family": fam, "phrasing": phrasing, "role": role,
                              "split": split, "context": ctx,
                              "generator": {"version": GEN_VERSION, "seed": seed,
                                            "db_sha256": dbsha}, **body})
    order = {r: i for i, r in enumerate(ROLES)}
    cases.sort(key=lambda c: (KINDS[task].index(c["kind"]), c["phrasing"], order[c["role"]]))
    for c in cases:
        problems = schema.check_case(c)
        if problems:
            raise ValueError(f"{c['case_id']}: {problems}")
    return cases


# --------------------------------------------------------------------------- files


def _file_sha(rows: list[dict[str, Any]]) -> str:
    return _sha("".join(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n" for r in rows))


def summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    def count(key: Any) -> dict[str, int]:
        return dict(sorted(Counter(key(c) for c in cases).items()))

    return {"cases": len(cases),
            "by_split": count(lambda c: c["split"]),
            "by_split_role": count(lambda c: f"{c['split']}/{c['role']}"),
            "by_gold": count(lambda c: f"{c['split']}/" + (
                c["gold"]["name"] if c["gold"]["type"] == "tool_call"
                else "clarify:" + c["gold"]["reason"])),
            "by_kind_split": count(lambda c: f"{c['kind']}/{c['split']}")}


def build_all(*, seed: int = SEED, db_path: Path = DB_PATH) -> dict[str, Any]:
    """Generate both tasks in memory (no file written)."""
    from slh.demo1.context import b1_context

    ctx = b1_context()
    db = load_db(db_path)
    env = _env()
    out = {"context": ctx, "tasks": {}}
    for task in schema.TASKS:
        out["tasks"][task] = generate_task(task, db, seed=seed, env=env, context=ctx)
    users = {t: {s: {c["entities"]["user_id"] for c in cs if c["split"] == s}
                 for s in schema.SPLITS} for t, cs in out["tasks"].items()}
    for t, by in users.items():
        for a, b in combinations(schema.SPLITS, 2):
            if by[a] & by[b]:
                raise ValueError(f"{t}: users shared between {a} and {b}")
    return out


def manifest_for(built: dict[str, Any], seed: int = SEED) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for task, cases in built["tasks"].items():
        for split in schema.SPLITS:
            rows = [c for c in cases if c["split"] == split]
            rel = (f"sealed/demo1/cases/sealed-{task}.jsonl" if split == "sealed"
                   else f"demo1/cases/{split}-{task}.jsonl")
            files[rel] = {"sha256": _file_sha(rows), "cases": len(rows),
                          "case_ids": [c["case_id"] for c in rows]}
    ctx_text = json.dumps(built["context"], sort_keys=True, indent=1, ensure_ascii=True) + "\n"
    return {"schema": "demo1-manifest-v1", "generator": GEN_VERSION, "seed": seed,
            "db_sha256": db_sha256(), "split_rule": SPLIT_SALT,
            "b1_context": {"file": "demo1/b1-context.json", "sha256": _sha(ctx_text),
                           "system_prompt_sha256": built["context"]["system_prompt_sha256"],
                           "tools_sha256": built["context"]["tools_sha256"]},
            "split_plan": {t: split_plan(t) for t in schema.TASKS},
            "files": files,
            "summary": {t: summary(cs) for t, cs in built["tasks"].items()}}


def write_all(built: dict[str, Any], *, seed: int = SEED, out: Path = OUT,
              sealed_dir: Path = SEALED_DIR) -> dict[str, Any]:
    """Write public files, then the sealed files, then lock the sealed tree."""
    from slh import sealing

    if sealed_dir.exists():
        raise FileExistsError(f"{sealed_dir} exists: the sealed cases are written once "
                              "(use `verify` to check a re-generation)")
    ctx_text = json.dumps(built["context"], sort_keys=True, indent=1, ensure_ascii=True) + "\n"
    (out / "cases").mkdir(parents=True, exist_ok=True)
    (out / "b1-context.json").write_text(ctx_text, encoding="utf-8")
    for task, cases in built["tasks"].items():
        for split in ("train", "calibrate"):
            schema.write_jsonl(out / "cases" / f"{split}-{task}.jsonl",
                               [c for c in cases if c["split"] == split])
    manifest = manifest_for(built, seed)
    with sealing.writer():
        (sealed_dir / "cases").mkdir(parents=True)
        for task, cases in built["tasks"].items():
            schema.write_jsonl(sealed_dir / "cases" / f"sealed-{task}.jsonl",
                               [c for c in cases if c["split"] == "sealed"])
        hashes = sealing.tree_hashes(sealed_dir / "cases")
        (sealed_dir / sealing.MARKER).write_text(json.dumps({
            "label": "demo1-sealed", "generator": GEN_VERSION, "seed": seed,
            "files_sha256": hashes,
            "readers": "slh.demo1.cases.sealed_access only (sealed run, answer validators)",
        }, sort_keys=True, indent=1) + "\n", encoding="utf-8")
        sealing.lock_tree(sealed_dir / "cases")
    for task in schema.TASKS:
        rel = f"sealed/demo1/cases/sealed-{task}.jsonl"
        if hashes[f"sealed-{task}.jsonl"] != manifest["files"][rel]["sha256"]:
            raise ValueError(f"{rel}: written bytes differ from the manifest hash")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=1) + "\n",
                                       encoding="utf-8")
    return manifest


def verify(out: Path = OUT, seed: int = SEED) -> dict[str, Any]:
    """Re-generate in memory and compare with the manifest and the public files
    (the sealed files are compared by hash only; the sealed tree is not opened)."""
    built = build_all(seed=seed)
    fresh = manifest_for(built, seed)
    recorded = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    problems = []
    if canonical(fresh) != canonical(recorded):
        problems.append("the re-generated manifest differs from demo1/manifest.json")
    for rel, entry in recorded["files"].items():
        if rel.startswith("sealed/"):
            continue
        actual = _sha((REPO / rel).read_text(encoding="utf-8"))
        if actual != entry["sha256"]:
            problems.append(f"{rel}: file hash differs")
    ctx = (out / "b1-context.json").read_text(encoding="utf-8")
    if _sha(ctx) != recorded["b1_context"]["sha256"]:
        problems.append("b1-context.json hash differs")
    return {"ok": not problems, "problems": problems}
