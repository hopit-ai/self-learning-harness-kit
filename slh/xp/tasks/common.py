"""Shared pieces of the XP decision tasks (``xp-case-v1``).

Copied (not imported, not edited) from POC 9 / demo 1 where a helper had to
change; read-only constants still come from ``slh.demo1.schema``.

Design (both tasks):

* A **group** is one database configuration (one user, one order, one request,
  one kind, one gold) written in the 3 phrasings of one wording style. Groups
  never cross splits; users never cross splits either.
* A **request family** is ``(pattern, style)``: the request pattern (variant:
  the attribute constraint form; sequence: which change the user names first;
  address: the referent kind) and one of 4 wording styles. Per pattern the 4
  styles go 1 / 1 / 2 to train / selection / test under a fixed salt, so
  families are disjoint across train, selection and test while every pattern
  appears in every split. The selection families serve two splits:
  ``validation`` (development scores during learning) and ``sentinel`` (a
  separate set for choosing among candidates). Validation and sentinel share
  families but never groups or users.
* Kinds cycle in a seeded order inside each split, so kinds stay balanced.
* Splits are generated train -> validation -> sentinel -> test with per-group
  seeded RNGs, so train, validation and sentinel do not depend on the test
  size and a smaller test set is a prefix of a larger one (same seed and the
  same train/validation/sentinel sizes).
* The **export** writes questions without gold (``case_hash`` = SHA-256 of the
  question payload, which carries no gold) and a separate answer key.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable, Iterable
from itertools import combinations
from pathlib import Path
from typing import Any

from slh.demo1.schema import ADDRESS_FIELDS, GREETING, REPO, WRITE_TOOLS

SCHEMA = "xp-case-v1"
QUESTION_SCHEMA = "xp-question-v1"
GEN_VERSION = "xp-gen-v1"
SEED = 7301
SPLIT_SALT = "xp-family-split-v1"
SPLITS = ("train", "validation", "sentinel", "test")
DEFAULT_CASES = {"train": 48, "validation": 60, "sentinel": 30, "test": 1000}
VARIANTS = 3  # phrasings per group
STYLES = 4  # wording styles per pattern (families = pattern x style)
STYLE_SPLITS = ("train", "validation", "test", "test")  # family split; "validation" = selection
FAMILY_SPLIT = {"train": "train", "validation": "validation", "sentinel": "validation",
                "test": "test"}  # split -> the family split its groups draw from
DB_PATH = REPO / "vendor/tau2-bench/data/tau2/domains/retail/db.json"
CASE_KEYS = frozenset({"schema", "task", "case_id", "kind", "group", "family", "pattern",
                       "style", "variant", "split", "approval", "context", "messages", "gold",
                       "request", "resolution", "entities", "generator"})
QUESTION_KEYS = ("schema", "task", "case_id", "context", "messages")
ADDRESS_WRITE = "modify_pending_order_address"
ITEMS_WRITE = "modify_pending_order_items"
EXCHANGE_WRITE = "exchange_delivered_order_items"
YES = ("Yes, please go ahead. No need to check back with me.",
       "Yes. Go ahead and do it, you don't need to confirm with me again.",
       "Yes, proceed. No need to ask me again.")


# --------------------------------------------------------------------------- basics


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rng(*parts: Any) -> random.Random:
    return random.Random(int(sha("|".join(map(str, parts)))[:16], 16))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n" for r in rows)


def load_db(path: Path | None = None) -> dict[str, Any]:
    """The retail database (default: the active one, ``slh.xp.retaildb.use``; the
    original ``db.json`` unless a builder selected the extension)."""
    return json.loads(Path(path or db_path()).read_text(encoding="utf-8"))


def db_path() -> Path:
    from slh.xp import retaildb

    return retaildb.path(retaildb.active())


def db_sha256(path: Path | None = None) -> str:
    return hashlib.sha256(Path(path or db_path()).read_bytes()).hexdigest()


def context() -> dict[str, Any]:
    """The B1 decision context the cases are written for (hashes only)."""
    from slh.demo1.context import B1_PROMPT_SHA256, B1_TOOLS_SHA256

    return {"file": "demo1/b1-context.json", "system_prompt_sha256": B1_PROMPT_SHA256,
            "tools_sha256": B1_TOOLS_SHA256}


def group_sizes(test_cases: int = DEFAULT_CASES["test"], *,
                train_cases: int = DEFAULT_CASES["train"],
                validation_cases: int = DEFAULT_CASES["validation"],
                sentinel_cases: int = DEFAULT_CASES["sentinel"]) -> dict[str, int]:
    """Cases -> groups (3 phrasings each, rounded down): 48/60/30/1000 -> 16/20/10/333."""
    return {"train": train_cases // VARIANTS, "validation": validation_cases // VARIANTS,
            "sentinel": sentinel_cases // VARIANTS, "test": test_cases // VARIANTS}


# --------------------------------------------------------------------------- db helpers


def addr_args(address: dict[str, Any]) -> dict[str, str]:
    return {f: str(address.get(f, "")) for f in ADDRESS_FIELDS}


def address_text(address: dict[str, Any]) -> str:
    a = addr_args(address)
    street = ", ".join(v for v in (a["address1"], a["address2"]) if v)
    return f"{street}, {a['city']}, {a['state']} {a['zip']}, {a['country']}"


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


def payment_for(user: dict[str, Any], order: dict[str, Any]) -> str | None:
    """The order's original payment method unless it is a gift card (a balance
    could block the write); else the user's first non-gift-card method, and only
    if its description is unique among the user's methods (so the user's words
    name it)."""
    pms = user["payment_methods"]
    descs = Counter(pm_desc(pm) for pm in pms.values())
    original = first_payment(order)
    order_ids = [original] if original in pms else []
    for pid in [*order_ids, *sorted(pms)]:
        pm = pms[pid]
        if pm["source"] != "gift_card" and descs[pm_desc(pm)] == 1:
            return pid
    return None


def spec_text(options: dict[str, str]) -> str:
    return ", ".join(f"{k} {v}" for k, v in options.items())


def money(value: float) -> str:
    return f"${value:,.0f}" if value == int(value) else f"${value:,.2f}"


def write_tool_for_status(status: str) -> str:
    return EXCHANGE_WRITE if status == "delivered" else ITEMS_WRITE


# --------------------------------------------------------------------------- conversation


class Conversation:
    """Case messages with real tau2 tool results (copied from demo 1)."""

    def __init__(self, env: Any, salt: str) -> None:
        self.env = env
        self.salt = salt
        self.messages: list[dict[str, Any]] = [{"role": "assistant", "content": GREETING}]

    def say(self, role: str, text: str) -> None:
        self.messages.append({"role": role, "content": text})

    def call(self, text: str, name: str, arguments: dict[str, Any]) -> str:
        from tau2.data_model.message import ToolCall

        call_id = f"chatcmpl-tool-{sha(f'{self.salt}|{len(self.messages)}')[:16]}"
        self.messages.append({"role": "assistant", "content": text,
                              "tool_calls": [{"id": call_id, "name": name,
                                              "arguments": arguments}]})
        result = self.env.get_response(ToolCall(id=call_id, name=name, arguments=arguments,
                                                requestor="assistant"))
        if result.error:
            raise ValueError(f"{name} failed while building a case: {result.content}")
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


def new_env() -> Any:
    """A fresh retail environment on the active database (its own DB copy; writes do
    not leak). For the original database: tau2's registered ``retail`` constructor."""
    from slh.xp import retaildb

    _quiet()
    return retaildb.env()


# --------------------------------------------------------------------------- planning


def family_id(task: str, pattern: str, style: int) -> str:
    return f"{task[0]}-{pattern}-s{style}"


def family_split(task: str, patterns: Iterable[str]) -> dict[str, str]:
    """family -> family split (train / validation / test; "validation" families
    also serve the sentinel split). Fixed salt (not the seed): test families
    stay fixed."""
    out = {}
    for pattern in patterns:
        styles = sorted(range(STYLES), key=lambda s: sha(f"{SPLIT_SALT}|{task}|{pattern}|{s}"))
        for split, style in zip(STYLE_SPLITS, styles, strict=True):
            out[family_id(task, pattern, style)] = split
    return out


def plan(task: str, kinds: tuple[str, ...], patterns: tuple[str, ...],
         sizes: dict[str, int], seed: int) -> list[dict[str, Any]]:
    """One slot per group: split, index, kind, family, pattern, style."""
    fsplit = family_split(task, patterns)
    slots = []
    for split in SPLITS:
        fams = sorted(((p, s) for p in patterns for s in range(STYLES)
                       if fsplit[family_id(task, p, s)] == FAMILY_SPLIT[split]),
                      key=lambda f: sha(f"{seed}|{task}|{split}|fam|{family_id(task, *f)}"))
        korder = sorted(kinds, key=lambda k: sha(f"{seed}|{task}|{split}|kind|{k}"))
        for i in range(sizes.get(split, 0)):
            # Kind cycles fastest; each round shifts the family, so every kind
            # meets every family of the split and families stay balanced.
            pattern, style = fams[(i % len(korder) + i // len(korder)) % len(fams)]
            slots.append({"split": split, "index": i, "kind": korder[i % len(korder)],
                          "family": family_id(task, pattern, style), "pattern": pattern,
                          "style": style})
    return slots


def tiers(rows: list[Any], user_of: Callable[[Any], str], key_of: Callable[[Any], Any],
          split: str, user_split: dict[str, str], used_keys: set[Any]) -> list[list[Any]]:
    """Candidate pools in preference order: unused user and key; unused key;
    any row whose user is not taken by another split."""
    allowed = [r for r in rows if user_split.get(user_of(r), split) == split]
    fresh_key = [r for r in allowed if key_of(r) not in used_keys]
    fresh_user = [r for r in fresh_key if user_of(r) not in user_split]
    return [fresh_user, fresh_key, allowed]


def generate_cases(task: str, *, kinds: tuple[str, ...], patterns: tuple[str, ...],
                   approval: dict[str, str], pick: Callable[..., dict[str, Any]],
                   build: Callable[..., dict[str, Any]],
                   check: Callable[[dict[str, Any]], list[str]], db: dict[str, Any],
                   sizes: dict[str, int], seed: int,
                   user_split: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """The generic driver: plan slots, draw one configuration per group, build
    its three phrasings, check every case, check user disjointness.

    ``user_split`` (optional) pre-assigns users to splits (user -> split), e.g.
    carried over from another task so users never cross splits across tasks."""
    ctx = context()
    gen = {"version": GEN_VERSION, "seed": seed, "db_sha256": db_sha256()}
    user_split = dict(user_split or {})
    used_keys: set[Any] = set()
    out = []
    for slot in plan(task, kinds, patterns, sizes, seed):
        r = rng(seed, task, slot["split"], slot["index"], "group")
        config = pick(db, slot, r, user_split, used_keys)
        user_split.setdefault(config["user_id"], slot["split"])
        used_keys.add(config["key"])
        group = (f"xp{task[0]}g-"
                 f"{sha(f'{GEN_VERSION}|{seed}|{task}|{slot['split']}|{slot['index']}')[:8]}")
        for v in range(VARIANTS):
            cid = f"xp{task[0]}-{sha(f'{GEN_VERSION}|{seed}|{group}|{v}')[:10]}"
            body = build(db, slot["kind"], config, slot["style"], v, cid)
            out.append({"schema": SCHEMA, "task": task, "case_id": cid, "kind": slot["kind"],
                        "group": group, "family": slot["family"], "pattern": slot["pattern"],
                        "style": slot["style"], "variant": v, "split": slot["split"],
                        "approval": approval[slot["kind"]], "context": ctx, "generator": gen,
                        **body})
    for c in out:
        if problems := check(c):
            raise ValueError(f"{c['case_id']}: {problems}")
    users = {s: {c["entities"]["user_id"] for c in out if c["split"] == s} for s in SPLITS}
    for a, b in combinations(SPLITS, 2):
        if users[a] & users[b]:
            raise ValueError(f"users shared between {a} and {b}")
    return out


def base_check(case: dict[str, Any], *, task: str, kinds: tuple[str, ...],
               approval: dict[str, str], gold_type: dict[str, str]) -> list[str]:
    """Structure shared by both tasks (empty list = well formed)."""
    if set(case) != CASE_KEYS:
        return [f"keys differ: {sorted(set(case) ^ CASE_KEYS)}"]
    problems = []
    if case["schema"] != SCHEMA or case["task"] != task or case["kind"] not in kinds \
            or case["split"] not in SPLITS:
        problems.append("schema/task/kind/split")
    if case["approval"] != approval[case["kind"]]:
        problems.append("approval")
    if case["family"] != family_id(task, case["pattern"], case["style"]):
        problems.append("family id")
    msgs = case["messages"]
    if not msgs or msgs[0].get("content") != GREETING or msgs[-1].get("role") != "tool":
        problems.append("greeting / decision point")
    pending = None
    for i, m in enumerate(msgs):
        if pending is not None and (m.get("role") != "tool" or m.get("tool_call_id") != pending):
            problems.append(f"message {i}: tool result must follow its call")
        pending = None
        calls = m.get("tool_calls") or []
        if len(calls) > 1 or (calls and m.get("role") != "assistant"):
            problems.append(f"message {i}: tool call shape")
        if calls:
            pending = calls[0]["id"]
        if m.get("role") == "tool" and m.get("error"):
            problems.append(f"message {i}: tool error in context")
    gold = case["gold"]
    if gold.get("type") != gold_type[case["kind"]]:
        problems.append("gold type")
    if gold["type"] == "clarify" and gold.get("reason") != case["kind"]:
        problems.append("clarify reason")
    if gold["type"] in ("tool_call", "confirm") and gold.get("name") not in WRITE_TOOLS:
        problems.append("gold tool")
    return problems


def summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    def count(key: Callable[[dict[str, Any]], str]) -> dict[str, int]:
        return dict(sorted(Counter(key(c) for c in cases).items()))

    return {"cases": len(cases), "by_split": count(lambda c: c["split"]),
            "groups": {s: len({c["group"] for c in cases if c["split"] == s}) for s in SPLITS},
            "by_split_kind": count(lambda c: f"{c['split']}/{c['kind']}"),
            "by_split_family": count(lambda c: f"{c['split']}/{c['family']}")}


# --------------------------------------------------------------------------- export


def question_row(case: dict[str, Any]) -> dict[str, Any]:
    """The runner-facing row: no gold, kind, group, request or resolution."""
    row = {k: case[k] for k in QUESTION_KEYS}
    row["schema"] = QUESTION_SCHEMA
    return {**row, "case_hash": case_hash(case)}


def case_hash(case: dict[str, Any]) -> str:
    """SHA-256 of the question payload only (never of the gold: a hash over the
    gold could be brute-forced over the small answer space)."""
    return sha(canonical({k: case[k] for k in QUESTION_KEYS if k != "schema"}))


def _inside_repo(path: Path) -> bool:
    try:
        Path(path).resolve().relative_to(REPO.resolve())
        return True
    except ValueError:
        return False


def export(cases: list[dict[str, Any]], out_dir: Path, name: str, *,
           allow_repo: bool = False) -> dict[str, Any]:
    """Write ``<name>.questions.jsonl`` (no gold), ``<name>.answer-key.jsonl``
    (full cases + case_hash) and ``<name>.manifest.json``. Test-split cases are
    never written inside the repository."""
    out_dir = Path(out_dir)
    if any(c["split"] == "test" for c in cases) and _inside_repo(out_dir) and not allow_repo:
        raise PermissionError(f"refusing to write test cases inside the repository: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = jsonl_text(question_row(c) for c in cases)
    key = jsonl_text({**c, "case_hash": case_hash(c)} for c in cases)
    qpath, kpath = out_dir / f"{name}.questions.jsonl", out_dir / f"{name}.answer-key.jsonl"
    qpath.write_text(questions, encoding="utf-8")
    kpath.write_text(key, encoding="utf-8")
    man = {"schema": "xp-export-v1", "name": name, "cases": len(cases),
           "questions": {"file": qpath.name, "sha256": sha(questions)},
           "answer_key": {"file": kpath.name, "sha256": sha(key)},
           "generator": cases[0]["generator"] if cases else None, "summary": summary(cases)}
    (out_dir / f"{name}.manifest.json").write_text(
        json.dumps(man, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    return man


def load_answer_key(path: Path) -> dict[str, dict[str, Any]]:
    """case_id -> full case; every row's case_hash is re-computed and checked."""
    out = {}
    for row in read_jsonl(path):
        recorded = row.pop("case_hash")
        if case_hash(row) != recorded:
            raise ValueError(f"{row['case_id']}: answer-key row differs from its case_hash")
        if row["case_id"] in out:
            raise ValueError(f"{row['case_id']}: duplicate in the answer key")
        out[row["case_id"]] = {**row, "case_hash": recorded}
    return out
