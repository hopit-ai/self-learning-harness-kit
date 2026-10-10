"""XP sequence task: an address change and an item change on one pending order.

The benchmark's tools make the order matter: ``modify_pending_order_items`` may
be called once and moves the order to ``pending (item modified)``, after which
the address can no longer be changed. So when a user wants both, the address
goes first, whatever order the user names them in. Every write also needs an
explicit yes after the agent listed its details.

The user states the new address in full and the new item by its full option
set (so the variant is unique), plus the payment method for the difference.

Kinds (next action at the decision point; A = address write, I = item write):

* ``both_approved``                 both listed and approved, nothing done -> write A
* ``address_done``                  both approved, A done -> write I
* ``none_approved``                 nothing approved yet -> confirm A (may also list I)
* ``items_approved_only``           only I approved, A still asked for -> confirm A
* ``address_done_items_unapproved`` A approved and done, I not approved -> confirm I

Patterns (family = pattern x wording style): ``address_first`` and
``items_first`` (which change the user names first). In ``items_first`` groups of
``both_approved`` / ``address_done`` the agent's own confirmation also lists the
item change first.

    python -m slh.xp.tasks.cli summary --task sequence
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from slh.xp.tasks import common as cm

TASK = "sequence"
KINDS = ("both_approved", "address_done", "none_approved", "items_approved_only",
         "address_done_items_unapproved")
PATTERNS = ("address_first", "items_first")
APPROVAL = {"both_approved": "address+items", "address_done": "address+items",
            "none_approved": "none", "items_approved_only": "items",
            "address_done_items_unapproved": "address"}
GOLD_TYPE = {"both_approved": "tool_call", "address_done": "tool_call",
             "none_approved": "confirm", "items_approved_only": "confirm",
             "address_done_items_unapproved": "confirm"}
GOLD_WRITE = {"both_approved": cm.ADDRESS_WRITE, "address_done": cm.ITEMS_WRITE,
              "none_approved": cm.ADDRESS_WRITE, "items_approved_only": cm.ADDRESS_WRITE,
              "address_done_items_unapproved": cm.ITEMS_WRITE}

# --------------------------------------------------------------------------- wording
# Fields: {order} {product} {spec} {addr} {pay}

OPENINGS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "address_first": (
        (("Hi, for my pending order {order}, I need to change the shipping address to {addr}. "
         "I also want to change the {product} to the one with {spec}; use {pay} for any price "
         "difference."),
         ("Hello. Two changes on order {order}, please: first, ship it to {addr}. Second, switch "
         "the {product} to the version with {spec}, with any difference on {pay}."),
         ("I'd like to update order {order}. The new shipping address is {addr}, and the "
         "{product} should be the one with {spec}. Please use {pay} for the difference.")),
        (("I'm moving, so order {order} needs to go to {addr} instead. While you're at it, I'd "
         "like the {product} changed to the one with {spec}. Charge or refund {pay}."),
         ("My address changed: please send order {order} to {addr}. And I'd also prefer the "
         "{product} with {spec}; any difference goes on {pay}."),
         ("Order {order} has to be shipped to my new place at {addr}. I also changed my mind "
         "about the {product}: I want the one with {spec}. Use {pay} for any difference.")),
        (("Order {order}: new address {addr}. Also swap the {product} for the one with {spec}. "
         "Difference to {pay}."),
         "For order {order}: address -> {addr}; {product} -> the one with {spec}. Use {pay}.",
         ("Two updates for {order}. Address: {addr}. Item: the {product} with {spec}. Payment "
         "for any difference: {pay}.")),
        (("Hi there! My order {order} hasn't shipped yet, and I just realised it's going to my "
         "old address. Could it go to {addr}? Oh, and I'd like the {product} with {spec} "
         "instead of what I picked; {pay} is fine for any difference."),
         ("Hey, quick one about order {order}: it should ship to {addr}, not where it's set "
         "now. Also, could the {product} be the one with {spec}? Put any difference on {pay}."),
         ("So I need two things on order {order}. It has to go to {addr}, and I'd rather have "
         "the {product} with {spec}. You can use {pay} for the difference.")),
    ),
    "items_first": (
        (("Hi, for my pending order {order}, I want to change the {product} to the one with "
         "{spec}; use {pay} for any price difference. I also need the shipping address changed "
         "to {addr}."),
         ("Hello. Two changes on order {order}, please: first, switch the {product} to the "
         "version with {spec}, with any difference on {pay}. Second, ship it to {addr}."),
         ("I'd like to update order {order}. The {product} should be the one with {spec}, and "
         "the new shipping address is {addr}. Please use {pay} for the difference.")),
        (("I changed my mind about the {product} in order {order}: I want the one with {spec}. "
         "Charge or refund {pay}. And since I'm moving, it needs to go to {addr}."),
         ("I'd prefer the {product} with {spec} in order {order}; any difference goes on "
         "{pay}. My address also changed, so please send it to {addr}."),
         ("Please swap the {product} in order {order} for the one with {spec}, using {pay} for "
         "any difference. Then the order has to ship to my new place at {addr}.")),
        (("Order {order}: swap the {product} for the one with {spec}. Difference to {pay}. "
         "Also new address {addr}."),
         "For order {order}: {product} -> the one with {spec}; address -> {addr}. Use {pay}.",
         ("Two updates for {order}. Item: the {product} with {spec}. Address: {addr}. Payment "
         "for any difference: {pay}.")),
        (("Hi there! I picked the wrong {product} in order {order}; I want the one with {spec} "
         "({pay} is fine for any difference). Oh, and it's going to my old address, could it "
         "go to {addr}?"),
         ("Hey, quick one about order {order}: could the {product} be the one with {spec}? Put "
         "any difference on {pay}. Also it should ship to {addr}, not where it's set now."),
         ("So I need two things on order {order}. I'd rather have the {product} with {spec}, "
         "and it has to go to {addr}. You can use {pay} for the difference.")),
    ),
}


# --------------------------------------------------------------------------- configurations


def eligible_orders(db: dict[str, Any]) -> list[dict[str, Any]]:
    """Pending orders with a nameable payment method and >= 1 item (product once
    in the order) that has another available variant."""
    out = []
    for oid in sorted(db["orders"]):
        order = db["orders"][oid]
        if order["status"] != "pending":
            continue
        user = db["users"][order["user_id"]]
        pay = cm.payment_for(user, order)
        if pay is None:
            continue
        counts = Counter(item["product_id"] for item in order["items"])
        items = []
        for item in order["items"]:
            product = db["products"][item["product_id"]]
            alts = sorted(k for k, v in product["variants"].items()
                          if v["available"] and k != item["item_id"])
            if counts[item["product_id"]] == 1 and alts:
                items.append((item, alts))
        if items:
            out.append({"user_id": user["user_id"], "order_id": oid, "items": items,
                        "payment_method_id": pay})
    return out


_ELIGIBLE: dict[int, tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]] = {}


def _addresses(db: dict[str, Any]) -> list[tuple[str, dict[str, str]]]:
    """(owner user id, home address) per distinct address."""
    seen, out = set(), []
    for uid in sorted(db["users"]):
        a = cm.addr_args(db["users"][uid]["address"])
        if cm.canonical(a) not in seen:
            seen.add(cm.canonical(a))
            out.append((uid, a))
    return out


def pick(db: dict[str, Any], slot: dict[str, Any], rng: random.Random,
         user_split: dict[str, str], used_keys: set[Any]) -> dict[str, Any]:
    if id(db) not in _ELIGIBLE or _ELIGIBLE[id(db)][0] is not db:
        _ELIGIBLE[id(db)] = (db, eligible_orders(db), _addresses(db))
    _, orders, addresses = _ELIGIBLE[id(db)]
    orders = orders[:]
    rng.shuffle(orders)
    for pool in cm.tiers(orders, lambda e: e["user_id"], lambda e: e["order_id"],
                         slot["split"], user_split, used_keys):
        if not pool:
            continue
        e = pool[0]
        item, alts = e["items"][rng.randrange(len(e["items"]))]
        new = alts[rng.randrange(len(alts))]
        current = db["orders"][e["order_id"]]["address"]
        own = cm.addr_args(db["users"][e["user_id"]]["address"])
        # The new address is another user's home address; that owner is kept in
        # this split (reserved in user_split), so an answer address never also
        # belongs to a user of another split, in this or a later-generated task.
        choices = [(uid, a) for uid, a in addresses if a["city"] != current["city"]
                   and a["zip"] != current["zip"] and a != own
                   and user_split.get(uid, slot["split"]) == slot["split"]]
        owner, address = choices[rng.randrange(len(choices))]
        user_split.setdefault(owner, slot["split"])
        product = db["products"][item["product_id"]]
        request = {"user_id": e["user_id"], "order_id": e["order_id"],
                   "item_id": item["item_id"], "product_id": item["product_id"],
                   "new_item_id": new, "set": dict(product["variants"][new]["options"]),
                   "new_address": address, "payment_method_id": e["payment_method_id"],
                   "pattern": slot["pattern"]}
        return {"user_id": e["user_id"], "key": e["order_id"], "request": request}
    raise ValueError(f"no sequence configuration left for {slot}")


# --------------------------------------------------------------------------- conversations


def address_args(request: dict[str, Any]) -> dict[str, str]:
    return {"order_id": request["order_id"], **cm.addr_args(request["new_address"])}


def items_args(request: dict[str, Any]) -> dict[str, Any]:
    return {"order_id": request["order_id"], "item_ids": [request["item_id"]],
            "new_item_ids": [request["new_item_id"]],
            "payment_method_id": request["payment_method_id"]}


def build(db: dict[str, Any], kind: str, config: dict[str, Any], style: int, variant: int,
          salt: str) -> dict[str, Any]:
    request = config["request"]
    user = db["users"][request["user_id"]]
    product = db["products"][request["product_id"]]
    pm = user["payment_methods"][request["payment_method_id"]]
    order = request["order_id"]
    addr = cm.address_text(request["new_address"])
    conv = cm.Conversation(cm.new_env(), salt)
    conv.say("user", OPENINGS[request["pattern"]][style][variant].format(
        order=order, product=product["name"].lower(), spec=cm.spec_text(request["set"]),
        addr=addr, pay=cm.pm_desc(pm)))
    conv.authenticate(user)
    conv.call(f"Thank you, {user['name']['first_name']}. Let me check the order.",
              "get_order_details", {"order_id": order})
    conv.call(f"Let me check the available options for the {product['name']}.",
              "get_product_details", {"product_id": request["product_id"]})
    a_desc = f"update the shipping address of order {order} to {addr}"
    i_desc = (f"change the {product['name']} (item {request['item_id']}) to item "
              f"{request['new_item_id']} ({cm.spec_text(request['set'])}), with any price "
              f"difference charged or refunded to {cm.pm_desc(pm, you=True)} "
              f"({request['payment_method_id']})")
    reminder = ("Items in a pending order can be modified only once, so please also confirm "
                "this is the only item you want to change. ")
    ask = "Do you want me to proceed (yes/no)?"
    if kind in ("both_approved", "address_done"):
        first, second = (a_desc, i_desc) if request["pattern"] == "address_first" \
            else (i_desc, a_desc)
        conv.say("assistant", f"To confirm, I will {first}, and {second}. {reminder}{ask}")
        conv.say("user", "Yes, that's the only item. " + cm.YES[variant])
        if kind == "both_approved":
            conv.call(f"Thank you. Let me check order {order} once more before making the "
                      "changes.", "get_order_details", {"order_id": order})
        else:
            conv.call("Thank you. I will update the shipping address first.",
                      cm.ADDRESS_WRITE, address_args(request))
    elif kind == "items_approved_only":
        conv.say("assistant", f"To confirm, I will {i_desc}. {reminder}{ask}")
        conv.say("user", f"Yes, that's the only item. Please also make sure the shipping address "
                         f"gets changed to {addr}.")
        conv.call(f"Thank you. Let me check order {order} once more.", "get_order_details",
                  {"order_id": order})
    elif kind == "address_done_items_unapproved":
        conv.say("assistant", f"Let's start with the address. To confirm, I will {a_desc}. {ask}")
        conv.say("user", "Yes, please update the address.")
        conv.call("Thank you. Let me update the shipping address.", cm.ADDRESS_WRITE,
                  address_args(request))
    # none_approved: the decision follows the lookups directly
    write = GOLD_WRITE[kind]
    args = address_args(request) if write == cm.ADDRESS_WRITE else items_args(request)
    gold = {"type": GOLD_TYPE[kind], "name": write, "arguments": args}
    current = db["orders"][order]["address"]
    return {"messages": conv.messages, "gold": gold, "request": request,
            "resolution": {"address_done": kind.startswith("address_done"),
                           "approved": APPROVAL[kind],
                           "current_address": cm.addr_args(current)},
            "entities": {"user_id": user["user_id"], "order_ids": [order],
                         "product_id": request["product_id"]}}


def check_case(case: dict[str, Any]) -> list[str]:
    problems = cm.base_check(case, task=TASK, kinds=KINDS, approval=APPROVAL,
                             gold_type=GOLD_TYPE)
    if problems:
        return problems
    if case["gold"]["name"] != GOLD_WRITE[case["kind"]]:
        problems.append("gold write")
    if case["request"]["pattern"] != case["pattern"]:
        problems.append("pattern")
    writes = [c["name"] for m in case["messages"] for c in m.get("tool_calls") or []
              if c["name"] in cm.WRITE_TOOLS]
    want = [cm.ADDRESS_WRITE] if case["kind"].startswith("address_done") else []
    if writes != want:
        problems.append(f"writes in context {writes} != {want}")
    return problems


def generate(db: dict[str, Any] | None = None, *, seed: int = cm.SEED,
             sizes: dict[str, int] | None = None,
             user_split: dict[str, str] | None = None) -> list[dict[str, Any]]:
    return cm.generate_cases(TASK, kinds=KINDS, patterns=PATTERNS, approval=APPROVAL,
                             pick=pick, build=build, check=check_case, db=db or cm.load_db(),
                             sizes=sizes or cm.group_sizes(), seed=seed,
                             user_split=user_split)
