"""XP address task: POC 9's address-resolution decision, rescaled to the XP conventions.

The user asks to ship a pending order to "the address of" another of their
orders, described by a referent (an order id, a product, a status, a product and
status, a city, a payment method, or two products). After the lookups the agent
must write the one address the referent names, or ask when it names none or
several. Same kinds and gold rules as POC 9 (``slh/poc9/cases.py``; copied, not
imported or edited; the configuration enumeration is demo 1's
``address_configs``, as in POC 9):

* ``resolvable``       referent-level approval, one address -> the write
* ``already_approved`` concrete approval naming the address -> the write
* ``ambiguous``        referent-level approval, >= 2 addresses -> clarify
* ``no_match``         referent-level approval, no address -> clarify
* ``not_yet_approved`` no approval, one address -> confirm naming that address

What differs from POC 9 (the XP conventions, ``common.py``): a group is one
configuration (user, pending target order, referent) and one kind, written in
the 3 phrasings of one wording style (POC 9: 3 kinds per user); the request
family is referent kind x wording style, split train / selection / test; the
validation, sentinel and test sizes are parameters; export carries no gold.

Pattern and kind are not crossed blindly: some referent kinds have few or no
configurations of an outcome (an order id never names two addresses; few users
have two orders with the same product and different addresses). Each group's
kind follows the usual seeded cycle; its family is the least-used family of the
split, in a seeded rotation, whose referent kind still has an unused
configuration of the needed outcome. So kinds stay exactly balanced and
families as balanced as the database allows.

Separation from POC 9:

* Seed: ``common.SEED`` (7301), not POC 9's 9101, and a different group/case
  id scheme.
* Hard exclusion: every (user, target order) pair of POC 9's train and
  validation cases (loaded with ``slh.poc9.cases.load``) is never used, in any
  split (so no POC 9 learning configuration, whatever its referent, comes back).
* POC 9's sealed test cases are not read here. Their (user, target) pairs are
  known only as group-id hashes in the public ``poc9/manifest.json``; pairs
  whose hash matches a POC 9 test group are avoided whenever another
  configuration fits (they are used only when the pool is exhausted, mainly for
  ``ambiguous``). ``poc9_overlap`` reports how many groups still touch a POC 9
  test pair.

TODO(poc9-test-overlap): overlap with POC 9's sealed test set is possible but
unlikely beyond what ``poc9_overlap`` reports, and it must be checked on our
side later, at configuration level (user, target order, referent), inside a
logged ``slh.poc9.cases.sealed_access`` window, before the XP test set is
registered (docs/xp/plan-v1.md, "Address task").

    python -m slh.xp.tasks.cli summary --task address
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from functools import lru_cache
from itertools import combinations
from typing import Any

from slh.xp.tasks import common as cm

TASK = "address"
KINDS = ("resolvable", "ambiguous", "no_match", "already_approved", "not_yet_approved")
PATTERNS = ("order_id", "product", "status", "product_status", "city", "payment",
            "two_products")
OUTCOME = {"resolvable": "unique", "already_approved": "unique", "not_yet_approved": "unique",
           "ambiguous": "ambiguous", "no_match": "no_match"}
APPROVAL = {"resolvable": "referent", "ambiguous": "referent", "no_match": "referent",
            "already_approved": "concrete", "not_yet_approved": "none"}
GOLD_TYPE = {"resolvable": "tool_call", "already_approved": "tool_call",
             "not_yet_approved": "confirm", "ambiguous": "clarify", "no_match": "clarify"}
POC9_GEN_VERSION = "poc9-gen-v1"
POC9_SEED = 9101

# --------------------------------------------------------------------------- wording
# A user opening = FRAMES[style][phrasing] with {target} and {ref}; {ref} is the
# referent's noun phrase REFS[pattern][phrasing]. RESTATE is the agent's wording.

FRAMES: tuple[tuple[str, str, str], ...] = (
    (("Hi, I need to change the shipping address of my pending order {target}. Please send "
      "it to {ref}."),
     "Could you change the shipping address of my order {target} to {ref}?",
     "Please update my pending order {target} so it ships to {ref}."),
    (("Hello, I think my pending order {target} is going to the wrong place. It should go "
      "to {ref}."),
     ("I made a mistake with the shipping address on order {target}, which hasn't shipped "
      "yet. Please use {ref} instead."),
     "My order {target} is still pending and the address is wrong. Can you change it to {ref}?"),
    ("Address change for pending order {target}: use {ref}.",
     "Order {target}: ship to {ref}.",
     "Need order {target} redirected to {ref}."),
    (("Hi there! Quick one: my order {target} hasn't shipped yet, and I'd like it to go to "
      "{ref}, if that's possible."),
     "Hey, so I won't be at the address on order {target}. Could you send it to {ref} instead?",
     ("So I realised order {target} is set to ship to the wrong place. I'd rather it go to "
      "{ref}. Can you sort that out?")),
)
REFS: dict[str, tuple[str, str, str]] = {
    "order_id": ("the address that order {order} was shipped to",
                 "the address from my order {order}",
                 "the same address as my order {order}"),
    "product": ("the address my {product} order went to",
                "the address from the order where I bought the {product}",
                "the same address as the order with my {product}"),
    "status": ("the address of {status_me}",
               "the address used for {status_me}",
               "the same address as {status_me}"),
    "product_status": ("the address of my {status_adj} {product} order",
                       "the address used for my {status_adj} {product} order",
                       "the same address as my {status_adj} {product} order"),
    "city": ("my {city} address, the one I used on another order",
             "the {city} address from one of my other orders",
             "the {city} address I have used before"),
    "payment": ("the address of the order I paid with {payment_me}",
                "the address of the order I bought with {payment_me}",
                "the same address as my order paid by {payment_me}"),
    "two_products": ("the address of the order that had both the {p1} and the {p2}",
                     "the address of my order with the {p1} and the {p2}",
                     "the address where my {p1} and {p2} order went"),
}
RESTATE = {  # copied from demo 1's ADDRESS_TEMPLATES (the agent's restatement)
    "order_id": "the address of order {order}",
    "product": "the address of your {product} order",
    "status": "the address of {status_you}",
    "product_status": "the address of your {status_adj} {product} order",
    "city": "the {city} address from your other orders",
    "payment": "the address of the order you paid with {payment_you}",
    "two_products": "the address of your order with both the {p1} and the {p2}",
}
_STATUS_ME = {"delivered": "the order I already received",
              "processed": "my order that is still being processed",
              "cancelled": "the order I cancelled", "pending": "my other pending order"}
_STATUS_YOU = {"delivered": "the order you already received",
               "processed": "your order that is still being processed",
               "cancelled": "the order you cancelled", "pending": "your other pending order"}
_STATUS_ADJ = {"delivered": "delivered", "processed": "processed", "cancelled": "cancelled",
               "pending": "other pending"}


def fields(user: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    """Template fields of a configuration (copied from demo 1's _address_fields)."""
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
        f |= {"payment_me": cm.pm_desc(pm), "payment_you": cm.pm_desc(pm, you=True)}
    if ref["kind"] == "two_products":
        f |= {"p1": ref["products"][0], "p2": ref["products"][1]}
    return f


# --------------------------------------------------------------------------- POC 9 separation


def poc9_group_id(user_id: str, target_order_id: str) -> str:
    """POC 9's group id for a (user, target) pair (slh/poc9/cases.py plan_groups)."""
    text = f"{POC9_GEN_VERSION}|{POC9_SEED}|{user_id}|{target_order_id}"
    return f"p9g-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]}"


@lru_cache(maxsize=1)
def poc9_pairs() -> dict[str, Any]:
    """``learning``: (user, target) pairs of POC 9's train and validation cases
    (read with slh.poc9.cases.load, hash-checked); ``test_groups``: the group
    ids of POC 9's sealed test groups from the public manifest (the sealed file
    is not opened)."""
    from slh.poc9 import cases as p9

    learning = {(c["request"]["user_id"], c["request"]["target_order_id"])
                for split in ("train", "validation") for c in p9.load(split)}
    test_groups = frozenset(g for g, s in p9.manifest()["group_split"].items() if s == "test")
    return {"learning": frozenset(learning), "test_groups": test_groups}


def poc9_overlap(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Groups per split whose (user, target) pair was a POC 9 learning pair
    (must be 0) or a POC 9 test pair (by manifest group hash)."""
    p9 = poc9_pairs()
    out: dict[str, Any] = {"learning": Counter(), "test": Counter()}
    seen = set()
    for c in cases:
        if c["group"] in seen:
            continue
        seen.add(c["group"])
        pair = (c["request"]["user_id"], c["request"]["order_id"])
        out["learning"][c["split"]] += pair in p9["learning"]
        out["test"][c["split"]] += poc9_group_id(*pair) in p9["test_groups"]
    return {k: dict(sorted(v.items())) for k, v in out.items()}


# --------------------------------------------------------------------------- configurations


def config_key(config: dict[str, Any]) -> tuple[str, str, str]:
    return (config["user_id"], config["target_order_id"], cm.canonical(config["referent"]))


_POOL: dict[int, tuple[dict[str, Any], dict[tuple[str, str], list[dict[str, Any]]]]] = {}


def pool(db: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """(pattern, outcome) -> demo 1 address configurations, POC 9 learning pairs
    removed, each flagged ``poc9_test`` when its pair hashes to a POC 9 test group."""
    if id(db) in _POOL and _POOL[id(db)][0] is db:
        return _POOL[id(db)][1]
    from slh.demo1 import generate as d1

    p9 = poc9_pairs()
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for pattern in PATTERNS:
        for c in d1.address_configs(db, pattern):
            pair = (c["user_id"], c["target_order_id"])
            if pair in p9["learning"]:
                continue
            by.setdefault((pattern, c["resolution"]["outcome"]), []).append(
                {**c, "poc9_test": poc9_group_id(*pair) in p9["test_groups"]})
    _POOL[id(db)] = (db, by)  # keeps db alive: the id stays valid
    return by


def scarcity(db: dict[str, Any]) -> Counter[str]:
    """user -> number of ``ambiguous`` configurations (the scarce outcome)."""
    return Counter(c["user_id"] for (_, outcome), rows in pool(db).items()
                   if outcome == "ambiguous" for c in rows)


def _draw(rows: list[dict[str, Any]], rng: random.Random, split: str,
          user_split: dict[str, str], used_keys: set[Any], *, avoid_poc9_test: bool,
          scarce: Counter[str]) -> dict[str, Any] | None:
    """An unused configuration whose user no other split owns; users not used
    anywhere first. Outside the test split, users with fewer ``ambiguous``
    configurations come first (a user drawn into a small split takes all of
    their configurations with them, and test needs about 67 ambiguous groups).
    None when there is none."""
    rows = rows[:]
    rng.shuffle(rows)
    if split != "test":
        rows.sort(key=lambda c: scarce[c["user_id"]])  # stable: shuffled within a level
    allowed = [c for c in rows if user_split.get(c["user_id"], split) == split
               and config_key(c) not in used_keys
               and not (avoid_poc9_test and c["poc9_test"])]
    fresh = [c for c in allowed if c["user_id"] not in user_split]
    return (fresh or allowed or [None])[0]


def plan_and_pick(db: dict[str, Any], sizes: dict[str, int], seed: int,
                  user_split: dict[str, str] | None = None
                  ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(slot, configuration) per group, splits in order train -> validation ->
    sentinel -> test (so a smaller test set is a prefix of a larger one)."""
    by = pool(db)
    scarce = scarcity(db)
    fsplit = cm.family_split(TASK, PATTERNS)
    user_split = dict(user_split or {})  # optional pre-assignment (users across tasks)
    used_keys: set[Any] = set()
    out = []
    for split in cm.SPLITS:
        fams = sorted(((p, s) for p in PATTERNS for s in range(cm.STYLES)
                       if fsplit[cm.family_id(TASK, p, s)] == cm.FAMILY_SPLIT[split]),
                      key=lambda f: cm.sha(f"{seed}|{TASK}|{split}|fam|{cm.family_id(TASK, *f)}"))
        korder = sorted(KINDS, key=lambda k: cm.sha(f"{seed}|{TASK}|{split}|kind|{k}"))
        used: Counter[tuple[str, int]] = Counter()
        for i in range(sizes.get(split, 0)):
            kind = korder[i % len(korder)]
            start = (i % len(korder) + i // len(korder)) % len(fams)
            order = sorted(range(len(fams)),
                           key=lambda j: (used[fams[j]], (j - start) % len(fams)))
            config = fam = None
            for avoid in (True, False):
                for j in order:
                    pattern = fams[j][0]
                    r = cm.rng(seed, TASK, split, i, pattern, "group")
                    config = _draw(by.get((pattern, OUTCOME[kind]), []), r, split, user_split,
                                   used_keys, avoid_poc9_test=avoid, scarce=scarce)
                    if config is not None:
                        fam = fams[j]
                        break
                if config is not None:
                    break
            if config is None or fam is None:
                raise ValueError(f"address pool exhausted: {split} group {i} ({kind})")
            used[fam] += 1
            user_split.setdefault(config["user_id"], split)
            used_keys.add(config_key(config))
            slot = {"split": split, "index": i, "kind": kind, "pattern": fam[0], "style": fam[1],
                    "family": cm.family_id(TASK, *fam)}
            out.append((slot, config))
    return out


# --------------------------------------------------------------------------- conversations


class Conversation(cm.Conversation):
    def lookup_missing(self, text: str, order_id: str) -> None:
        """A lookup of an order id that does not exist: the tool error stays in
        the context (as in POC 9's foreign-order-id cases)."""
        from tau2.data_model.message import ToolCall

        call_id = f"chatcmpl-tool-{cm.sha(f'{self.salt}|{len(self.messages)}')[:16]}"
        args = {"order_id": order_id}
        self.messages.append({"role": "assistant", "content": text,
                              "tool_calls": [{"id": call_id, "name": "get_order_details",
                                              "arguments": args}]})
        result = self.env.get_response(ToolCall(id=call_id, name="get_order_details",
                                                arguments=args, requestor="assistant"))
        if not result.error:
            raise ValueError(f"{order_id} was expected not to exist")
        self.messages.append({"role": "tool", "content": result.content,
                              "tool_call_id": call_id, "error": True})


def _lookups(conv: Conversation, user: dict[str, Any], ref: dict[str, Any],
             opening: str) -> list[str]:
    """Look up every order of the user (and a mentioned foreign id)."""
    shown = list(user["orders"])
    for i, oid in enumerate(shown):
        conv.call(opening if i == 0 else f"Let me check order {oid}.",
                  "get_order_details", {"order_id": oid})
    if ref["kind"] == "order_id" and ref["order_id"] not in user["orders"]:
        shown.append(ref["order_id"])
        conv.lookup_missing(f"Let me check order {ref['order_id']}.", ref["order_id"])
    return shown


def build(db: dict[str, Any], kind: str, config: dict[str, Any], style: int, variant: int,
          salt: str) -> dict[str, Any]:
    """Messages and gold for one phrasing of one group (POC 9's flows)."""
    user = db["users"][config["user_id"]]
    target = config["target_order_id"]
    ref = config["referent"]
    f = fields(user, config)
    res = config["resolution"]
    conv = Conversation(cm.new_env(), salt)
    conv.say("user", FRAMES[style][variant].format(target=target,
                                                    ref=REFS[ref["kind"]][variant].format(**f)))
    conv.authenticate(user)
    first = user["name"]["first_name"]
    restate = RESTATE[ref["kind"]].format(**f)
    args = ({"order_id": target, **cm.addr_args(res["candidates"][0])}
            if res["outcome"] == "unique" else None)
    if APPROVAL[kind] == "referent":
        conv.say("assistant",
                 f"Thank you, {first}. To confirm: I will update the shipping address of pending "
                 f"order {target} to {restate}. I will check your orders to find it. Do you want "
                 "me to proceed (yes/no)?")
        conv.say("user", cm.YES[variant])
        shown = _lookups(conv, user, ref, "Let me check your orders.")
        gold = ({"type": "tool_call", "name": cm.ADDRESS_WRITE, "arguments": args}
                if args else {"type": "clarify", "reason": kind})
    elif kind == "already_approved":
        shown = _lookups(conv, user, ref,
                         f"Thank you, {first}. Let me check your orders to find that address.")
        new = cm.address_text(args)
        conv.say("assistant",
                 f"I found it: {restate} is {new}. To confirm: I will update the shipping address "
                 f"of pending order {target} to {new}. Do you want me to proceed (yes/no)?")
        conv.say("user", cm.YES[variant])
        conv.call(f"Thank you. Let me check order {target} once more before updating it.",
                  "get_order_details", {"order_id": target})
        gold = {"type": "tool_call", "name": cm.ADDRESS_WRITE, "arguments": args}
    else:  # not_yet_approved: the agent went straight to the lookups
        shown = _lookups(conv, user, ref,
                         f"Thank you, {first}. Let me check your orders to find that address.")
        gold = {"type": "confirm", "name": cm.ADDRESS_WRITE, "arguments": args}
    request = {"user_id": user["user_id"], "order_id": target, "referent": ref,
               "pattern": ref["kind"]}
    resolution = {k: res[k] for k in ("outcome", "candidates", "near_misses")}
    return {"messages": conv.messages, "gold": gold, "request": request,
            "resolution": resolution,
            "entities": {"user_id": user["user_id"], "order_ids": shown}}


def _expected_errors_masked(case: dict[str, Any]) -> dict[str, Any]:
    """The case with the foreign-order-id lookup's error flag cleared (the
    only tool error a case may carry), for the shared structural check."""
    ref = (case.get("request") or {}).get("referent") or {}
    msgs = []
    pending_foreign = None
    for m in case["messages"]:
        if m.get("role") == "tool" and m.get("error") and m.get("tool_call_id") == pending_foreign:
            m = {**m, "error": False}
        pending_foreign = None
        for c in m.get("tool_calls") or []:
            if ref.get("kind") == "order_id" and c["name"] == "get_order_details" \
                    and c["arguments"].get("order_id") == ref.get("order_id"):
                pending_foreign = c["id"]
        msgs.append(m)
    return {**case, "messages": msgs}


def check_case(case: dict[str, Any]) -> list[str]:
    problems = cm.base_check(_expected_errors_masked(case), task=TASK, kinds=KINDS,
                             approval=APPROVAL, gold_type=GOLD_TYPE)
    if problems:
        return problems
    req, res, gold = case["request"], case["resolution"], case["gold"]
    if req["pattern"] != case["pattern"] or req["referent"]["kind"] != case["pattern"]:
        problems.append("pattern")
    if res["outcome"] != OUTCOME[case["kind"]]:
        problems.append("outcome")
    if gold["type"] in ("tool_call", "confirm"):
        if gold["name"] != cm.ADDRESS_WRITE:
            problems.append("gold tool")
        if gold["arguments"] != {"order_id": req["order_id"],
                                 **cm.addr_args(res["candidates"][0])}:
            problems.append("gold address")
    for m in case["messages"]:
        for call in m.get("tool_calls") or []:
            if call["name"] in cm.WRITE_TOOLS:
                problems.append("write before the decision point")
    return problems


def generate(db: dict[str, Any] | None = None, *, seed: int = cm.SEED,
             sizes: dict[str, int] | None = None,
             user_split: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """All cases in memory (train, validation, sentinel, test; groups contiguous).
    ``user_split``: optional user -> split pre-assignment (users across tasks)."""
    db = db or cm.load_db()
    sizes = sizes or cm.group_sizes()
    ctx = cm.context()
    gen = {"version": cm.GEN_VERSION, "seed": seed, "db_sha256": cm.db_sha256(),
           "poc9_separation": "poc9 train+validation (user, target) pairs excluded; "
                              "poc9 test pairs avoided by manifest group hash"}
    out = []
    for slot, config in plan_and_pick(db, sizes, seed, user_split):
        group = (f"xpag-"
                 f"{cm.sha(f'{cm.GEN_VERSION}|{seed}|{TASK}|{slot['split']}|{slot['index']}')[:8]}")
        for v in range(cm.VARIANTS):
            cid = f"xpa-{cm.sha(f'{cm.GEN_VERSION}|{seed}|{group}|{v}')[:10]}"
            body = build(db, slot["kind"], config, slot["style"], v, cid)
            out.append({"schema": cm.SCHEMA, "task": TASK, "case_id": cid, "kind": slot["kind"],
                        "group": group, "family": slot["family"], "pattern": slot["pattern"],
                        "style": slot["style"], "variant": v, "split": slot["split"],
                        "approval": APPROVAL[slot["kind"]], "context": ctx, "generator": gen,
                        **body})
    for c in out:
        if problems := check_case(c):
            raise ValueError(f"{c['case_id']}: {problems}")
    users = {s: {c["entities"]["user_id"] for c in out if c["split"] == s} for s in cm.SPLITS}
    for a, b in combinations(cm.SPLITS, 2):
        if users[a] & users[b]:
            raise ValueError(f"users shared between {a} and {b}")
    return out
