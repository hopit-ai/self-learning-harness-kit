"""POC 9 address-decision cases (``poc9-case-v1``, docs/poc9/schema.md).

Seeded and deterministic, from the public retail ``db.json`` and the tau2 retail
tools only (no model call). Reuses the demo-1 generator: its address
configurations (user, pending target order, referent, resolution), its wording
templates and its conversation builder.

A group is one user and one pending target order; its three variants are three
different kinds over that user (minimal pairs). Groups (and users) never cross
splits. Kind triples are assigned greedily so kinds stay balanced (test: exactly
24 per kind). Test cases go to ``sealed/poc9/cases/test.jsonl`` (mode 000);
``poc9/manifest.json`` records every file's SHA-256.

    python -m slh.poc9.cases generate   # writes once
    python -m slh.poc9.cases verify     # re-generate in memory, compare hashes
    python -m slh.poc9.cases validate   # independent gold check (train+validation)
    python -m slh.poc9.cases reseal --reason TEXT   # idempotent re-lock + access log
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from itertools import combinations
from pathlib import Path
from typing import Any

from slh.demo1 import generate as d1
from slh.demo1.schema import ADDRESS_FIELDS, GREETING, REPO, canonical, read_jsonl, write_jsonl

SCHEMA = "poc9-case-v1"
GEN_VERSION = "poc9-gen-v1"
SEED = 9101
KINDS = ("resolvable", "ambiguous", "no_match", "already_approved", "not_yet_approved")
SPLITS = ("train", "validation", "test")
GROUPS = {"train": 16, "validation": 4, "test": 40}
VARIANTS = 3
OUTCOME = {"resolvable": "unique", "already_approved": "unique", "not_yet_approved": "unique",
           "ambiguous": "ambiguous", "no_match": "no_match"}
APPROVAL = {"resolvable": "referent", "ambiguous": "referent", "no_match": "referent",
            "already_approved": "concrete", "not_yet_approved": "none"}
WRITE = "modify_pending_order_address"
OUT = REPO / "poc9"
SEALED_DIR = REPO / "sealed/poc9"
ACCESS_LOG = OUT / "sealed-access.jsonl"
CASE_KEYS = frozenset({"schema", "case_id", "kind", "group", "variant", "split", "stream_index",
                       "referent_kind", "phrasing", "approval", "context", "messages", "gold",
                       "request", "resolution", "entities", "generator"})


class SealedRefused(PermissionError):
    """The sealed test split was requested outside ``sealed_access``."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rng(*parts: Any) -> random.Random:
    return random.Random(int(_sha("|".join(map(str, parts)))[:16], 16))


def address_text(address: dict[str, Any]) -> str:
    a = {k: str(address.get(k, "")) for k in ADDRESS_FIELDS}
    street = ", ".join(v for v in (a["address1"], a["address2"]) if v)
    return f"{street}, {a['city']}, {a['state']} {a['zip']}, {a['country']}"


# --------------------------------------------------------------------------- planning


def _triples() -> list[tuple[str, ...]]:
    return list(combinations(KINDS, VARIANTS))


def kind_plan(seed: int = SEED) -> dict[str, list[tuple[str, ...]]]:
    """split -> one kind triple per group, greedy-balanced (deterministic)."""
    order = sorted(_triples(), key=lambda t: _sha(f"{seed}|triple|{t}"))
    plan = {}
    for split in SPLITS:
        counts: Counter[str] = Counter()
        rows = []
        for _ in range(GROUPS[split]):
            best = min(order, key=lambda t: (max(counts[k] + (k in t) for k in KINDS),
                                             sum(counts[k] for k in t), order.index(t)))
            counts.update(best)
            rows.append(best)
            order = order[1:] + order[:1]  # rotate so ties spread over triples
        plan[split] = rows
    return plan


def _configs(db: dict[str, Any]) -> dict[tuple[str, str], dict[str, list[dict[str, Any]]]]:
    """(user, target) -> outcome -> demo-1 configurations (all referent kinds)."""
    by: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list))
    for rkind in d1.schema.ADDRESS_KINDS:
        for c in d1.address_configs(db, rkind):
            by[(c["user_id"], c["target_order_id"])][c["resolution"]["outcome"]].append(
                {**c, "referent_kind": rkind})
    return by


def _pick_variants(options: dict[str, list[dict[str, Any]]], triple: tuple[str, ...],
                   rng: random.Random) -> list[dict[str, Any]]:
    """One configuration per kind, distinct referents where possible."""
    chosen: list[dict[str, Any]] = []
    for kind in triple:
        pool = options[OUTCOME[kind]]
        fresh = [c for c in pool if all(canonical(c["referent"]) != canonical(o["referent"])
                                        for o in chosen)]
        pool = fresh or pool
        # Prefer a referent kind not used yet in the group, then any.
        used = {o["referent_kind"] for o in chosen}
        pool = [c for c in pool if c["referent_kind"] not in used] or pool
        chosen.append(pool[rng.randrange(len(pool))])
    return chosen


def plan_groups(db: dict[str, Any], seed: int = SEED) -> list[dict[str, Any]]:
    """Every group: split, triple, user, target, the three configurations."""
    configs = _configs(db)
    plan = kind_plan(seed)
    slots = [(split, i, t) for split in SPLITS for i, t in enumerate(plan[split])]
    # Scarcest first: triples needing an ambiguous referent draw first.
    slots.sort(key=lambda s: ("ambiguous" not in s[2], SPLITS.index(s[0]), s[1]))
    used_users: set[str] = set()
    groups = []
    for split, i, triple in slots:
        needs = {OUTCOME[k] for k in triple}
        fits = sorted(k for k, v in configs.items()
                      if needs <= set(v) and k[0] not in used_users)
        if not fits:
            raise ValueError(f"{split} group {i} {triple}: no user left")
        rng = _rng(seed, split, i, "group")
        users = sorted({u for u, _ in fits})
        uid = rng.choice(users)
        key = rng.choice([k for k in fits if k[0] == uid])
        used_users.add(uid)
        groups.append({"split": split, "index": i, "triple": triple, "user_id": uid,
                       "target_order_id": key[1],
                       "group": f"p9g-{_sha(f'{GEN_VERSION}|{seed}|{uid}|{key[1]}')[:8]}",
                       "configs": _pick_variants(configs[key], triple, rng)})
    groups.sort(key=lambda g: (SPLITS.index(g["split"]), g["index"]))
    return groups


# --------------------------------------------------------------------------- conversations


def _lookups(conv: d1.Conversation, user: dict[str, Any], ref: dict[str, Any],
             opening: str) -> list[str]:
    """Look up every order (and a mentioned foreign id); ``opening`` is the
    first call's text, so no two assistant messages are adjacent."""
    shown = list(user["orders"])
    for i, oid in enumerate(shown):
        conv.call(opening if i == 0 else f"Let me check order {oid}.",
                  "get_order_details", {"order_id": oid})
    if ref["kind"] == "order_id" and ref["order_id"] not in user["orders"]:
        shown.append(ref["order_id"])
        conv.call(f"Let me check order {ref['order_id']}.", "get_order_details",
                  {"order_id": ref["order_id"]})
    return shown


def _write_args(config: dict[str, Any]) -> dict[str, Any]:
    return {"order_id": config["target_order_id"],
            **d1.addr_args(config["resolution"]["candidates"][0])}


def build_case(db: dict[str, Any], env: Any, kind: str, config: dict[str, Any], phrasing: int,
               salt: str) -> dict[str, Any]:
    """Messages and gold for one variant."""
    rkind = config["referent_kind"]
    if APPROVAL[kind] == "referent":
        body = d1.address_case(db, env, config, rkind, phrasing, salt)
        if kind != "resolvable":
            assert body["gold"] == {"type": "clarify", "reason": kind}, (kind, body["gold"])
        return body
    user = db["users"][config["user_id"]]
    fields = d1._address_fields(db, user, config)
    openings, restate = d1.ADDRESS_TEMPLATES[rkind]
    target = config["target_order_id"]
    conv = d1.Conversation(env, salt)
    conv.say("user", openings[phrasing].format(**fields))
    conv.authenticate(user)
    first = user["name"]["first_name"]
    args = _write_args(config)
    if kind == "already_approved":
        shown = _lookups(conv, user, config["referent"],
                         f"Thank you, {first}. Let me check your orders to find that address.")
        found = restate.format(**fields)
        conv.say("assistant",
                 f"I found it: {found} is {address_text(args)}. To confirm: I will update the "
                 f"shipping address of pending order {target} to {address_text(args)}. Do you "
                 "want me to proceed (yes/no)?")
        conv.say("user", d1.YES[phrasing])
        conv.call(f"Thank you. Let me check order {target} once more before updating it.",
                  "get_order_details", {"order_id": target})
        gold = {"type": "tool_call", "name": WRITE, "arguments": args}
    else:  # not_yet_approved: the agent went straight to the lookups
        shown = _lookups(conv, user, config["referent"],
                         f"Thank you, {first}. Let me check your orders to find that address.")
        gold = {"type": "confirm", "name": WRITE, "arguments": args}
    request = {"user_id": user["user_id"], "target_order_id": target,
               "referent": config["referent"]}
    return {"messages": conv.messages, "gold": gold, "request": request,
            "resolution": config["resolution"],
            "entities": {"user_id": user["user_id"], "order_ids": shown}}


def check_case(case: dict[str, Any]) -> list[str]:
    """Structural problems (empty = well formed)."""
    problems = []
    if set(case) != CASE_KEYS:
        return [f"keys differ: {sorted(set(case) ^ CASE_KEYS)}"]
    if case["schema"] != SCHEMA or case["kind"] not in KINDS or case["split"] not in SPLITS:
        problems.append("schema/kind/split")
    if case["approval"] != APPROVAL[case["kind"]]:
        problems.append("approval")
    msgs = case["messages"]
    if not msgs or msgs[0].get("content") != GREETING or msgs[-1].get("role") != "tool":
        problems.append("greeting / decision point")
    for m in msgs:
        for c in m.get("tool_calls") or []:
            if c["name"] in d1.schema.WRITE_TOOLS:
                problems.append("write before the decision point")
    gold = case["gold"]
    want = {"resolvable": "tool_call", "already_approved": "tool_call",
            "not_yet_approved": "confirm", "ambiguous": "clarify", "no_match": "clarify"}
    if gold.get("type") != want[case["kind"]]:
        problems.append("gold type")
    if gold["type"] == "clarify" and gold.get("reason") != case["kind"]:
        problems.append("clarify reason")
    if gold["type"] in ("tool_call", "confirm") and gold.get("name") != WRITE:
        problems.append("gold tool")
    return problems


def generate(db: dict[str, Any] | None = None, *, seed: int = SEED, env: Any = None,
             context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """All 180 cases in memory (train in stream order, then validation, test)."""
    db = db or d1.load_db()
    env = env or d1._env()
    if context is None:
        from slh.demo1.context import b1_context
        context = b1_context()
    ctx = {"file": "demo1/b1-context.json",
           "system_prompt_sha256": context.get("system_prompt_sha256"),
           "tools_sha256": context.get("tools_sha256")}
    gen = {"version": GEN_VERSION, "seed": seed, "db_sha256": d1.db_sha256()}
    out: list[dict[str, Any]] = []
    for g in plan_groups(db, seed):
        rng = _rng(seed, g["group"], "phrasing")
        for v, (kind, config) in enumerate(zip(g["triple"], g["configs"], strict=True)):
            cid = f"p9-{_sha(f'{GEN_VERSION}|{seed}|{g['group']}|{v}')[:10]}"
            phrasing = rng.randrange(d1.PHRASINGS)
            body = build_case(db, env, kind, config, phrasing, cid)
            out.append({"schema": SCHEMA, "case_id": cid, "kind": kind, "group": g["group"],
                        "variant": v, "split": g["split"], "stream_index": -1,
                        "referent_kind": config["referent_kind"], "phrasing": phrasing,
                        "approval": APPROVAL[kind], "context": ctx, "generator": gen, **body})
    by_split = {s: [c for c in out if c["split"] == s] for s in SPLITS}
    stream = by_split["train"][:]
    _rng(seed, "stream").shuffle(stream)
    by_split["train"] = stream
    final = []
    for s in SPLITS:
        for i, c in enumerate(by_split[s]):
            c["stream_index"] = i
            final.append(c)
    for c in final:
        if problems := check_case(c):
            raise ValueError(f"{c['case_id']}: {problems}")
    users = {s: {c["entities"]["user_id"] for c in final if c["split"] == s} for s in SPLITS}
    for a, b in combinations(SPLITS, 2):
        if users[a] & users[b]:
            raise ValueError(f"users shared between {a} and {b}")
    return final


# --------------------------------------------------------------------------- files


def _file_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n" for r in rows)


def _rel(split: str) -> str:
    return "sealed/poc9/cases/test.jsonl" if split == "test" else f"poc9/cases/{split}.jsonl"


def summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    def count(key: Any) -> dict[str, int]:
        return dict(sorted(Counter(key(c) for c in cases).items()))

    return {"cases": len(cases), "by_split": count(lambda c: c["split"]),
            "by_split_kind": count(lambda c: f"{c['split']}/{c['kind']}"),
            "by_split_referent": count(lambda c: f"{c['split']}/{c['referent_kind']}"),
            "groups": {s: len({c["group"] for c in cases if c["split"] == s}) for s in SPLITS}}


def manifest_for(cases: list[dict[str, Any]], seed: int = SEED) -> dict[str, Any]:
    files = {}
    for split in SPLITS:
        rows = [c for c in cases if c["split"] == split]
        files[_rel(split)] = {"sha256": _sha(_file_text(rows)), "cases": len(rows),
                              "case_ids": [c["case_id"] for c in rows]}
    groups = {c["group"]: c["split"] for c in cases}
    ctx = cases[0]["context"]
    return {"schema": "poc9-manifest-v1", "generator": GEN_VERSION, "seed": seed,
            "db_sha256": d1.db_sha256(), "b1_context": ctx,
            "group_split": dict(sorted(groups.items())), "files": files,
            "summary": summary(cases)}


def write_all(cases: list[dict[str, Any]], *, seed: int = SEED, out: Path = OUT,
              sealed_dir: Path = SEALED_DIR) -> dict[str, Any]:
    from slh import sealing

    if sealed_dir.exists():
        raise FileExistsError(f"{sealed_dir} exists: the sealed cases are written once")
    man = manifest_for(cases, seed)
    for split in ("train", "validation"):
        write_jsonl(out / "cases" / f"{split}.jsonl", [c for c in cases if c["split"] == split])
    with sealing.writer():
        (sealed_dir / "cases").mkdir(parents=True)
        write_jsonl(sealed_dir / "cases" / "test.jsonl",
                    [c for c in cases if c["split"] == "test"])
        hashes = sealing.tree_hashes(sealed_dir / "cases")
        (sealed_dir / sealing.MARKER).write_text(json.dumps({
            "label": "poc9-sealed", "generator": GEN_VERSION, "seed": seed,
            "files_sha256": hashes,
            "readers": "slh.poc9.cases.sealed_access only (final run, answer validation)",
        }, sort_keys=True, indent=1) + "\n", encoding="utf-8")
        sealing.lock_tree(sealed_dir / "cases")
    if hashes["test.jsonl"] != man["files"][_rel("test")]["sha256"]:
        raise ValueError("sealed test bytes differ from the manifest hash")
    (out / "manifest.json").write_text(json.dumps(man, sort_keys=True, indent=1) + "\n",
                                       encoding="utf-8")
    return man


def verify(out: Path = OUT, seed: int = SEED) -> dict[str, Any]:
    """Re-generate in memory; compare with the manifest and the public files."""
    fresh = manifest_for(generate(seed=seed), seed)
    recorded = manifest(out)
    problems = []
    if canonical(fresh) != canonical(recorded):
        problems.append("re-generated manifest differs from poc9/manifest.json")
    for rel, entry in recorded["files"].items():
        if not rel.startswith("sealed/") and \
                _sha((REPO / rel).read_text(encoding="utf-8")) != entry["sha256"]:
            problems.append(f"{rel}: file hash differs")
    return {"ok": not problems, "problems": problems}


# --------------------------------------------------------------------------- loading


def manifest(out: Path = OUT) -> dict[str, Any]:
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def _checked(path: Path, rel: str, out: Path) -> list[dict[str, Any]]:
    if _sha(path.read_text(encoding="utf-8")) != manifest(out)["files"][rel]["sha256"]:
        raise ValueError(f"{rel} differs from its manifest hash")
    return read_jsonl(path)


def load(split: str, *, out: Path = OUT) -> list[dict[str, Any]]:
    """Train (stream order) or validation cases, hash-checked."""
    if split == "test":
        raise SealedRefused("test cases are read only inside slh.poc9.cases.sealed_access")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return _checked(out / "cases" / f"{split}.jsonl", _rel(split), out)


def _log_access(record: dict[str, Any], out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / ACCESS_LOG.name).open("a", encoding="utf-8") as log:
        log.write(json.dumps({**record, "time_utc": dt.datetime.now(dt.UTC).isoformat()},
                             sort_keys=True) + "\n")


def _lock(cases_dir: Path) -> None:
    """Mode 000 again; idempotent (re-opens first so a partly open tree locks fully)."""
    from slh import sealing

    if not cases_dir.exists():
        return
    with sealing.writer():
        sealing.open_tree(cases_dir)
        sealing.lock_tree(cases_dir)


def reseal(reason: str, *, sealed_dir: Path = SEALED_DIR, out: Path = OUT) -> dict[str, Any]:
    """Re-lock ``sealed/poc9`` (mode 000) and log it. Idempotent; safe to call from
    any cleanup path (``python -m slh.poc9.cases reseal --reason ...``)."""
    cases_dir = sealed_dir / "cases"
    was_open = cases_dir.exists() and (cases_dir.stat().st_mode & 0o777) != 0
    _lock(cases_dir)
    mode = cases_dir.stat().st_mode & 0o777 if cases_dir.exists() else None
    record = {"purpose": "reseal", "reason": reason, "was_open": was_open, "mode": mode}
    _log_access(record, out)
    return record


@contextlib.contextmanager
def sealed_access(purpose: str, *, sealed_dir: Path = SEALED_DIR, out: Path = OUT
                  ) -> Iterator[list[dict[str, Any]]]:
    """Yield the 120 test cases and log the access. The tree is open only while
    the file is read (then locked again before the caller sees a case); a SIGTERM
    during that window is turned into an exception so the lock still runs."""
    import signal
    import threading

    from slh import sealing

    if purpose not in ("final_run", "final_run_topup", "final_run_b0", "answer_validation",
                       "post_results_demo"):
        raise ValueError(f"unregistered sealed-access purpose {purpose!r}")
    cases_dir = sealed_dir / "cases"
    previous = None
    main_thread = threading.current_thread() is threading.main_thread()

    def _term(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"signal {signum} during sealed access")

    if main_thread:
        previous = signal.signal(signal.SIGTERM, _term)
    try:
        try:
            with sealing.writer():
                sealing.open_tree(cases_dir)
                rows = _checked(cases_dir / "test.jsonl", _rel("test"), out)
        finally:
            _lock(cases_dir)
            if main_thread:
                signal.signal(signal.SIGTERM, previous)
        yield rows
    finally:
        _log_access({"purpose": purpose}, out)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "verify"
    if cmd == "generate":
        man = write_all(generate())
        print(json.dumps(man["summary"], indent=1))
        return 0
    if cmd == "verify":
        res = verify()
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 1
    if cmd == "validate":
        from slh.poc9.validate import validate_all

        rows = [c for s in ("train", "validation") for c in load(s)]
        if "--sealed" in argv:
            with sealed_access("answer_validation") as test:
                rows += test
        res = validate_all(rows)
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 1
    if cmd == "reseal":
        reason = argv[argv.index("--reason") + 1] if "--reason" in argv else "manual"
        print(json.dumps(reseal(reason), sort_keys=True))
        return 0
    raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
