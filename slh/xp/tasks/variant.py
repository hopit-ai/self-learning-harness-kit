"""XP variant task: pick the exact replacement variant from attribute constraints.

The user wants an item of a delivered order exchanged (or of a pending order
modified) for a variant described by attributes ("color black, connectivity
wired, at most $50") and names the payment method for any price difference.
The agent must pick the one available variant (other than the current item)
that satisfies every constraint, or ask when none or several do.

Kinds (gold analogous to the POC 9 address task):

* ``resolvable``       referent-level approval, one match -> the write
* ``already_approved`` concrete approval naming the variant -> the write
* ``ambiguous``        referent-level approval, >= 2 matches -> clarify
* ``no_match``         referent-level approval, 0 matches -> clarify
* ``not_yet_approved`` no approval, one match -> confirm naming that variant

Request patterns (family = pattern x wording style): ``one`` (one attribute),
``two`` (two attributes), ``one_cap`` / ``two_cap`` (plus a price ceiling),
``one_keep`` (one attribute changed, one named attribute kept as now).

    python -m slh.xp.tasks.cli summary --task variant
"""

from __future__ import annotations

import math
import random
from collections import Counter
from itertools import combinations
from typing import Any

from slh.xp.tasks import common as cm

TASK = "variant"
KINDS = ("resolvable", "already_approved", "ambiguous", "no_match", "not_yet_approved")
PATTERNS = ("one", "two", "one_cap", "two_cap", "one_keep")
OUTCOME = {"resolvable": "unique", "already_approved": "unique", "not_yet_approved": "unique",
           "ambiguous": "ambiguous", "no_match": "no_match"}
APPROVAL = {"resolvable": "referent", "ambiguous": "referent", "no_match": "referent",
            "already_approved": "concrete", "not_yet_approved": "none"}
GOLD_TYPE = {"resolvable": "tool_call", "already_approved": "tool_call",
             "not_yet_approved": "confirm", "ambiguous": "clarify", "no_match": "clarify"}
EPS = 1e-9

# --------------------------------------------------------------------------- wording

CONSTRAINTS: dict[str, tuple[str, str, str]] = {
    "one": ("one with {spec}", "a version with {spec}", "the {spec} version"),
    "two": ("one with {spec}", "a version that has {spec}", "the version with {spec}"),
    "one_cap": ("one with {spec} that costs at most {cap}", "one with {spec}, no more than {cap}",
                "a version with {spec} for {cap} or less"),
    "two_cap": ("one with {spec} that costs at most {cap}", "one with {spec}, no more than {cap}",
                "a version with {spec} for {cap} or less"),
    "one_keep": ("one with {spec} and the same {keep} as my current one",
                 "one with {spec}, keeping the {keep} I have now",
                 "a version with {spec} but the same {keep} as before"),
}
OPENINGS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "delivered": (
        (("Hi, I'd like to exchange the {product} from my order {order} for {c}. Please use "
         "{pay} for any price difference."),
         ("Can I exchange the {product} in order {order}? I want {c}. Any price difference "
         "should go to {pay}."),
         ("I received order {order}. Please exchange the {product} for {c}, and use {pay} for "
         "the difference.")),
        (("Hello, the {product} from order {order} isn't working out for me. I'd like {c} "
         "instead; put any difference on {pay}."),
         ("The {product} I got in order {order} is not what I wanted. Could I swap it for {c}? "
         "Use {pay} for any difference."),
         ("I'm not happy with the {product} from order {order}. Please swap it for {c} and "
         "settle the difference with {pay}.")),
        ("Exchange request for order {order}: the {product}, I want {c}. Difference to {pay}.",
         "Order {order}: exchange the {product} for {c}. Pay or refund via {pay}.",
         "Need an exchange on order {order}. The {product} should become {c}. Use {pay}."),
        (("My order {order} arrived, but I think I picked the wrong {product}. Could you "
         "exchange it for {c}? Any difference can go on {pay}."),
         ("So order {order} came and the {product} just isn't right for me. I'd rather have "
         "{c}. Please use {pay} for the price difference."),
         ("I opened order {order} and realised I want a different {product}: {c}. Please "
         "exchange it and use {pay} for any difference.")),
    ),
    "pending": (
        (("Hi, I'd like to change the {product} in my pending order {order} to {c}. Please use "
         "{pay} for any price difference."),
         ("Can you modify the {product} in order {order}? I want {c}. Any price difference "
         "should go to {pay}."),
         ("My order {order} hasn't shipped yet. Please change the {product} to {c}, and use "
         "{pay} for the difference.")),
        (("Hello, I made a mistake with the {product} in order {order}, which hasn't shipped. "
         "I'd like {c} instead; put any difference on {pay}."),
         ("I ordered the wrong {product} in order {order}. Could you switch it to {c} before it "
         "ships? Use {pay} for any difference."),
         ("Order {order} is still pending and the {product} isn't what I want. Please switch it "
         "to {c} and settle the difference with {pay}.")),
        (("Modify request for pending order {order}: the {product}, I want {c}. Difference to "
         "{pay}."),
         "Order {order}: change the {product} to {c}. Pay or refund via {pay}.",
         "Need a change on pending order {order}. The {product} should become {c}. Use {pay}."),
        (("I placed order {order} recently and I think I picked the wrong {product}. Could you "
         "change it to {c} before it ships? Any difference can go on {pay}."),
         ("So order {order} hasn't gone out yet, and I'd rather have a different {product}: "
         "{c}. Please use {pay} for the price difference."),
         ("Before order {order} ships, I want a different {product}: {c}. Please change it and "
         "use {pay} for any difference.")),
    ),
}


# --------------------------------------------------------------------------- resolution


def matches(variant: dict[str, Any], current: dict[str, Any], request: dict[str, Any]) -> int:
    """Number of failed checks (0 = the variant satisfies the request)."""
    failed = 0 if variant["available"] else 1
    failed += variant["item_id"] == current["item_id"]
    for k, val in request["set"].items():
        failed += variant["options"].get(k) != val
    for k in request["keep"]:
        failed += variant["options"].get(k) != current["options"][k]
    if request["max_price"] is not None:
        failed += variant["price"] > request["max_price"] + EPS
    return failed


def resolve(product: dict[str, Any], current: dict[str, Any],
            request: dict[str, Any]) -> dict[str, Any]:
    fails = {iid: matches(product["variants"][iid], current, request)
             for iid in sorted(product["variants"])}
    ids = [i for i, f in fails.items() if f == 0]
    outcome = "unique" if len(ids) == 1 else "no_match" if not ids else "ambiguous"
    return {"outcome": outcome, "candidates": ids,
            "near_misses": sum(1 for i, f in fails.items()
                               if f == 1 and i != current["item_id"])}


def _cap(price: float) -> float:
    return float(5 * math.ceil(price / 5 - EPS))


def requests(product: dict[str, Any], current: dict[str, Any], pattern: str
             ) -> list[dict[str, Any]]:
    """Every request of ``pattern`` for this item that the current item does not
    already satisfy (so it is a real change)."""
    variants = [product["variants"][k] for k in sorted(product["variants"])]
    attrs = sorted(current["options"])

    def req(set_: dict[str, str], keep: tuple[str, ...] = (), cap: float | None = None
            ) -> dict[str, Any]:
        return {"set": dict(sorted(set_.items())), "keep": list(keep), "max_price": cap}

    out: list[dict[str, Any]] = []
    width = 1 if pattern in ("one", "one_cap", "one_keep") else 2
    for group in combinations(attrs, width):
        combos = sorted({tuple(v["options"][a] for a in group) for v in variants})
        for combo in combos:
            set_ = dict(zip(group, combo, strict=True))
            if all(current["options"][a] == set_[a] for a in group):
                continue  # not a change
            if pattern in ("one", "two"):
                out.append(req(set_))
            elif pattern == "one_keep":
                for keep in attrs:
                    if keep not in set_:
                        out.append(req(set_, keep=(keep,)))
            else:
                prices = sorted({v["price"] for v in variants
                                 if all(v["options"][a] == set_[a] for a in group)})
                caps = sorted({_cap(p) for p in prices} | (
                    {float(5 * math.floor(prices[0] / 5) - 5)} if prices[0] > 10 else set()))
                out.extend(req(set_, cap=c) for c in caps)
    return out


def eligible_items(db: dict[str, Any]) -> list[dict[str, Any]]:
    """(order, item) pairs a variant case can be built on: delivered or pending
    order, the item's product appears once in the order, >= 3 variants, and a
    payment method the user can name uniquely."""
    out = []
    for oid in sorted(db["orders"]):
        order = db["orders"][oid]
        if order["status"] not in ("delivered", "pending"):
            continue
        user = db["users"][order["user_id"]]
        pay = cm.payment_for(user, order)
        if pay is None:
            continue
        counts = Counter(item["product_id"] for item in order["items"])
        for item in order["items"]:
            if counts[item["product_id"]] == 1 and \
                    len(db["products"][item["product_id"]]["variants"]) >= 3:
                out.append({"user_id": user["user_id"], "order_id": oid,
                            "order_status": order["status"], "item": item,
                            "payment_method_id": pay})
    return out


_ELIGIBLE: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}


def pick(db: dict[str, Any], slot: dict[str, Any], rng: random.Random,
         user_split: dict[str, str], used_keys: set[Any]) -> dict[str, Any]:
    """One configuration for the slot's kind and pattern (deterministic)."""
    if id(db) not in _ELIGIBLE or _ELIGIBLE[id(db)][0] is not db:
        _ELIGIBLE[id(db)] = (db, eligible_items(db))  # keeps db alive: the id stays valid
    items = _ELIGIBLE[id(db)][1][:]
    rng.shuffle(items)
    want = ("delivered", "pending")[int(cm.sha(f"{slot['split']}|{slot['index']}|status"), 16) % 2]
    items.sort(key=lambda e: e["order_status"] != want)  # stable: wanted status first
    outcome = OUTCOME[slot["kind"]]
    for pool in cm.tiers(items, lambda e: e["user_id"],
                         lambda e: (e["order_id"], e["item"]["item_id"]),
                         slot["split"], user_split, used_keys):
        for e in pool:
            product = db["products"][e["item"]["product_id"]]
            fits = []
            for r in requests(product, e["item"], slot["pattern"]):
                res = resolve(product, e["item"], r)
                if res["outcome"] == outcome:
                    fits.append((r, res))
            if not fits:
                continue
            r, res = fits[rng.randrange(len(fits))]
            request = {"user_id": e["user_id"], "order_id": e["order_id"],
                       "order_status": e["order_status"], "item_id": e["item"]["item_id"],
                       "product_id": e["item"]["product_id"], "pattern": slot["pattern"],
                       **r, "payment_method_id": e["payment_method_id"]}
            return {"user_id": e["user_id"], "key": (e["order_id"], e["item"]["item_id"]),
                    "request": request, "resolution": res}
    raise ValueError(f"no variant configuration left for {slot}")


# --------------------------------------------------------------------------- conversations


def constraint_text(request: dict[str, Any], variant: int) -> str:
    cap = request["max_price"]
    return CONSTRAINTS[request["pattern"]][variant].format(
        spec=cm.spec_text(request["set"]), cap=cm.money(cap) if cap is not None else "",
        keep=" and ".join(request["keep"]))


def gold_args(request: dict[str, Any], new_item_id: str) -> dict[str, Any]:
    return {"order_id": request["order_id"], "item_ids": [request["item_id"]],
            "new_item_ids": [new_item_id], "payment_method_id": request["payment_method_id"]}


def build(db: dict[str, Any], kind: str, config: dict[str, Any], style: int, variant: int,
          salt: str) -> dict[str, Any]:
    """Messages and gold for one phrasing of one group."""
    request, res = config["request"], config["resolution"]
    user = db["users"][request["user_id"]]
    product = db["products"][request["product_id"]]
    status = request["order_status"]
    pm = user["payment_methods"][request["payment_method_id"]]
    c = constraint_text(request, variant)
    conv = cm.Conversation(cm.new_env(), salt)
    conv.say("user", OPENINGS[status][style][variant].format(
        product=product["name"].lower(), order=request["order_id"], c=c, pay=cm.pm_desc(pm)))
    conv.authenticate(user)
    first = user["name"]["first_name"]
    order = request["order_id"]
    verb = "exchange" if status == "delivered" else "modify"
    reminder = ("Please also confirm you have listed every item you want to exchange. "
                if status == "delivered" else
                "Items in a pending order can be modified only once, so please also confirm "
                "this is the only item you want to change in this order. ")
    only = "Yes, that's the only item. "
    said = c.replace("my current one", "your current one").replace("I have now", "you have now")
    pay = f"{cm.pm_desc(pm, you=True)} ({request['payment_method_id']})"
    write = cm.write_tool_for_status(status)
    conv.call(f"Thank you, {first}. Let me check the order.", "get_order_details",
              {"order_id": order})
    if APPROVAL[kind] == "referent":
        conv.say("assistant",
                 f"To confirm: I will {verb} the {product['name']} (item {request['item_id']}) "
                 f"in order {order} for {said}. Any price difference will be charged or refunded "
                 f"to {pay}. I will check the available options to find it. {reminder}"
                 "Do you want me to proceed (yes/no)?")
        conv.say("user", only + cm.YES[variant])
        conv.call("Let me check the available options.", "get_product_details",
                  {"product_id": request["product_id"]})
        if res["outcome"] == "unique":
            gold = {"type": "tool_call", "name": write,
                    "arguments": gold_args(request, res["candidates"][0])}
        else:
            gold = {"type": "clarify", "reason": kind}
    else:
        conv.call("Let me check the available options.", "get_product_details",
                  {"product_id": request["product_id"]})
        new = res["candidates"][0]
        args = gold_args(request, new)
        if kind == "already_approved":
            v = product["variants"][new]
            conv.say("assistant",
                     f"I found it: item {new} ({cm.spec_text(v['options'])}, "
                     f"{cm.money(v['price'])}). To confirm: I will {verb} the {product['name']} "
                     f"(item {request['item_id']}) in order {order} for item {new}. Any price "
                     f"difference will be charged or refunded to {pay}. {reminder}"
                     "Do you want me to proceed (yes/no)?")
            conv.say("user", only + cm.YES[variant])
            conv.call(f"Thank you. Let me check order {order} once more before I {verb} it.",
                      "get_order_details", {"order_id": order})
            gold = {"type": "tool_call", "name": write, "arguments": args}
        else:  # not_yet_approved: decision right after the lookups
            gold = {"type": "confirm", "name": write, "arguments": args}
    return {"messages": conv.messages, "gold": gold, "request": request, "resolution": res,
            "entities": {"user_id": user["user_id"], "order_ids": [order],
                         "product_id": request["product_id"]}}


def check_case(case: dict[str, Any]) -> list[str]:
    problems = cm.base_check(case, task=TASK, kinds=KINDS, approval=APPROVAL,
                             gold_type=GOLD_TYPE)
    if problems:
        return problems
    req, res, gold = case["request"], case["resolution"], case["gold"]
    if req["pattern"] != case["pattern"] or res["outcome"] != OUTCOME[case["kind"]]:
        problems.append("pattern/outcome")
    if gold["type"] in ("tool_call", "confirm"):
        if gold["name"] != cm.write_tool_for_status(req["order_status"]):
            problems.append("write tool for status")
        if gold["arguments"]["new_item_ids"] != res["candidates"][:1]:
            problems.append("gold variant")
    for m in case["messages"]:
        for call in m.get("tool_calls") or []:
            if call["name"] in cm.WRITE_TOOLS:
                problems.append("write before the decision point")
    return problems


def generate(db: dict[str, Any] | None = None, *, seed: int = cm.SEED,
             sizes: dict[str, int] | None = None,
             user_split: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """All cases in memory (train, validation, test; groups contiguous)."""
    return cm.generate_cases(TASK, kinds=KINDS, patterns=PATTERNS, approval=APPROVAL,
                             pick=pick, build=build, check=check_case, db=db or cm.load_db(),
                             sizes=sizes or cm.group_sizes(), seed=seed,
                             user_split=user_split)
