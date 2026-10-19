"""Training-only decision prefixes, the ban list, and the randomized inclusion units.

    python -m slh.xp.fgate.design bans --from FILE [--from FILE ...] --out BANS.json
    python -m slh.xp.fgate.design capacity --bans BANS.json [--tiny]

**Prefixes.** The phase-1 decision generators (``slh.xp.tasks``: address, variant,
sequence) draw fresh configurations from a fixed, non-secret seed (``SEED_LABEL``; this
material is training-only). Users are carried across tasks (a user stays in one role
everywhere). Generator splits map to roles: ``train`` -> ``fit`` (C training and the F'
screen), ``validation`` -> ``calibrate`` and ``sentinel`` -> ``evaluate`` (C only). No
test split is drawn. Wording families follow the phase-1 family split (fit families are
disjoint from calibrate/evaluate families).

**Bans.** No user or configuration of any test, calibration or smoke split may enter:
the owner builds ``BANS.json`` on our side from the key and scenario files of every
earlier split (phase-1 test keys, phase-2 test/calibration/smoke scenarios, the G1 test,
any later test) with ``bans``, which extracts every user id and order id found in any
string of those files. Banned users are pre-assigned to a role no split uses, so the
generators cannot draw them; after generation every case is checked again (no banned
user, no banned order id anywhere in its messages). Configurations are owned by users
(orders belong to users), so a user ban covers every configuration of that user; the
owner of every banned order is banned as well (a file may name an order but not its user). A
generator that runs out of eligible users stops with ``CapacityShort`` (no silent
fallback). Answer addresses owned by a banned user are counted and reported.

**F' units.** Each fit prefix gets one empty-library anchor (B1) and ``draws`` lesson
vectors in antithetic pairs (v, 1 - v; each lesson independently with p = 0.5 in v),
so every lesson is included in exactly half of a prefix's randomized draws. No vector
repeats within a prefix and none is empty (the anchor is the only B1 unit).
**C collection units.** B1 on every calibrate and evaluate prefix (the fit prefixes'
B1 decisions are the F' anchors).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from slh.xp import retaildb as RDB
from slh.xp.partner import common as pc

SEED_LABEL = "slh-xp-c-fgate-training-v1"
ROLES = {"train": "fit", "validation": "calibrate", "sentinel": "evaluate"}
TASK_ORDER = ("address", "variant", "sequence")
BANS_SCHEMA = "xp-bans-v1"
BANNED = "__banned__"
# Groups (configurations; 3 phrasings each) per task and generator split.
# The address pool is scarce (the "ambiguous" outcome has 135 configurations from 41
# users in the pinned database), so address gets fewer groups. Evaluate is sized so that
# the approved-and-correct write kinds (2 of 5 kinds per task) give >= 149 independent
# groups (docs/xp/partner-c-fgate-plan.md, section 2.6).
FULL_SIZES = {"address": {"train": 60, "validation": 10, "sentinel": 40},
              "variant": {"train": 170, "validation": 30, "sentinel": 200},
              "sequence": {"train": 170, "validation": 30, "sentinel": 200}}
TINY_SIZES = {t: {"train": 4, "validation": 2, "sentinel": 3} for t in TASK_ORDER}
DRAWS = 16         # randomized vectors per fit prefix (8 antithetic pairs) + 1 anchor;
                   # owner decision 2026-10-18 (was 8): power at +2/+3 points
TINY_DRAWS = 2

_USER = re.compile(r"\b[a-z]+_[a-z]+_\d{4}\b")
_ORDER = re.compile(r"#W\d{7}")


class CapacityShort(RuntimeError):
    pass


def seed() -> int:
    return int(pc.sha_text(SEED_LABEL)[:12], 16)


# --------------------------------------------------------------------------- bans


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _rows(path: Path) -> list[Any]:
    """Rows of a JSON, JSONL or gzipped JSONL file."""
    path = Path(path)
    if path.suffix == ".gz":
        return pc.read_jsonl(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        whole = json.loads(text)
    except ValueError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return whole if isinstance(whole, list) else [whole]


def make_bans(files: list[Path]) -> dict[str, Any]:
    """User ids and order ids found in any string of the given key/scenario files."""
    users: set[str] = set()
    orders: set[str] = set()
    sources = []
    for f in files:
        rows = _rows(Path(f))
        for row in rows:
            for s in _strings(row):
                users.update(_USER.findall(s))
                orders.update(_ORDER.findall(s))
        sources.append({"file": Path(f).name, "sha256": pc.sha_file(Path(f)), "rows": len(rows)})
    body = {"schema": BANS_SCHEMA, "users": sorted(users), "orders": sorted(orders),
            "sources": sources}
    return {**body, "sha256": pc.sha_text(pc.canonical(body))}


def load_bans(path: Path | None) -> dict[str, Any]:
    if path is None:
        body = {"schema": BANS_SCHEMA, "users": [], "orders": [], "sources": []}
        return {**body, "sha256": pc.sha_text(pc.canonical(body)), "empty": True}
    bans = pc.read_json(path)
    body = {k: v for k, v in bans.items() if k != "sha256"}
    if bans.get("schema") != BANS_SCHEMA or pc.sha_text(pc.canonical(body)) != bans.get("sha256"):
        raise ValueError(f"{path}: not a valid {BANS_SCHEMA} file")
    return bans


# --------------------------------------------------------------------------- prefixes


def _owners() -> dict[str, str]:
    from slh.xp.tasks import common as cm

    db = cm.load_db()
    return {cm.canonical(cm.addr_args(u["address"])): uid for uid, u in db["users"].items()}


@contextlib.contextmanager
def _kinds(mod: Any, drop: tuple[str, ...]):
    """Generate without some kinds (a recorded capacity fallback, never silent)."""
    if not drop:
        yield
        return
    original = mod.KINDS
    unknown = set(drop) - set(original)
    if unknown:
        raise ValueError(f"unknown kinds {sorted(unknown)}")
    mod.KINDS = tuple(k for k in original if k not in drop)
    try:
        yield
    finally:
        mod.KINDS = original


def generate(sizes: dict[str, dict[str, int]], bans: dict[str, Any], *,
             gen_seed: int | None = None,
             drop_kinds: dict[str, tuple[str, ...]] | None = None,
             retail_db: str = "original") -> dict[str, Any]:
    """Cases per task (full cases with gold, ``role`` added) and a capacity report.
    ``drop_kinds`` (task -> kinds) is an explicit, recorded capacity fallback.
    ``retail_db`` names the pinned retail database the prefixes are drawn from and
    built on (``slh.xp.retaildb``: ``original``, the library default, or a synthetic
    extension, ``ext-v2`` being the CLI default); it is recorded in the report and in every case's generator
    block (``db_sha256``). Bans apply unchanged (they name original users and orders)."""
    from slh.xp import retaildb

    with retaildb.use(retail_db):
        out = _generate(sizes, bans, gen_seed=gen_seed, drop_kinds=drop_kinds)
    out["report"]["retail_db"] = retaildb.identity(retail_db)
    return out


def _generate(sizes: dict[str, dict[str, int]], bans: dict[str, Any], *,
              gen_seed: int | None = None,
              drop_kinds: dict[str, tuple[str, ...]] | None = None,
              address_families: dict[tuple[str, int], tuple[str, int]] | None = None,
              address_exclude: frozenset[tuple[str, str]] | None = None
              ) -> dict[str, Any]:
    """``address_families``: pinned address slot families (``tasks.address.plan_and_pick``;
    the difficulty check's common slot specification). ``address_exclude``: excluded
    (kind, pattern) address slot families (the difficulty check's reduced specification)."""
    from slh.xp.tasks import common as cm
    from slh.xp.tasks.cli import task_module

    s = seed() if gen_seed is None else gen_seed
    banned_orders = set(bans["orders"])
    # A banned order bans its owner too (a scenario file may name an order without its
    # user id): the generators then never draw that user, so no banned order can surface.
    order_owner = {o: r.get("user_id") for o, r in cm.load_db()["orders"].items()}
    via_orders = {order_owner[o] for o in banned_orders if order_owner.get(o)}
    banned_users = set(bans["users"]) | via_orders
    user_split: dict[str, str] = {u: BANNED for u in banned_users}
    owners = _owners()
    out: list[dict[str, Any]] = []
    report: dict[str, Any] = {"seed_label": SEED_LABEL, "bans_sha256": bans["sha256"],
                              "banned_users": len(banned_users),
                              "banned_users_via_orders": len(via_orders - set(bans["users"])),
                              "tasks": {},
                              "answer_address_owned_by_banned": 0,
                              "drop_kinds": {t: list(k) for t, k in (drop_kinds or {}).items()}}
    for task in TASK_ORDER:
        if task not in sizes:
            continue
        sz = {**sizes[task], "test": 0}
        mod = task_module(task)
        try:
            with _kinds(mod, tuple((drop_kinds or {}).get(task, ()))):
                pin = ({"families": address_families, "exclude": address_exclude}
                       if task == "address" and (address_families is not None
                                                 or address_exclude) else {})
                cases = mod.generate(seed=s, sizes=sz, user_split=user_split, **pin)
        except ValueError as exc:  # generators raise on an exhausted pool
            raise CapacityShort(f"{task}: {exc} (sizes {sizes[task]}, "
                                f"{len(banned_users)} banned users)") from exc
        for c in cases:
            user = c["entities"]["user_id"]
            text = json.dumps(c["messages"])
            if user in banned_users or set(_ORDER.findall(text)) & banned_orders:
                raise CapacityShort(f"{c['case_id']}: a banned user or order reached a case")
            user_split.setdefault(user, c["split"])
            new = (c.get("request") or {}).get("new_address")
            if isinstance(new, dict):
                owner = owners.get(cm.canonical(cm.addr_args(new)))
                if owner in banned_users:
                    report["answer_address_owned_by_banned"] += 1
                elif owner:
                    user_split.setdefault(owner, c["split"])
            out.append({**c, "role": ROLES[c["split"]]})
        report["tasks"][task] = {
            role: {"groups": len({c["group"] for c in cases if ROLES[c["split"]] == role}),
                   "cases": sum(1 for c in cases if ROLES[c["split"]] == role)}
            for role in ROLES.values()}
    roles_users = {r: {c["entities"]["user_id"] for c in out if c["role"] == r}
                   for r in ROLES.values()}
    for a in ROLES.values():
        for b in ROLES.values():
            if a < b and roles_users[a] & roles_users[b]:
                raise CapacityShort(f"users shared between {a} and {b}")
    return {"cases": out, "report": report}


# --------------------------------------------------------------------------- difficulty


# Synthetic-difficulty check (hand-off 1b part A; docs/xp/retail-db-extension.md section 7,
# docs/xp/partner-c-fgate-plan.md section 3.6): B1 on MATCHED decision prefixes drawn from
# original users only and from synthetic users only, same database file, one common slot
# specification (seed, task, split, slot index -> kind, family = pattern x style, and the
# 3 phrasings), sentinel split (the validation families; never the test families). Per
# arm: address 40, variant 70, sequence 70 groups = 540 prefixes; 1,080 B1 units in all.
DIFFICULTY_ROLE = "difficulty"
DIFFICULTY_SIZES = {"address": 40, "variant": 70, "sequence": 70}
TINY_DIFFICULTY_SIZES = {t: 2 for t in TASK_ORDER}
DIFFICULTY_PHRASINGS = 3
# Pre-specified (plan section 3.6, revised 2026-10-19 after the power simulation,
# ``fgate.difficulty_power``): the primary question is a difficulty SHIFT, two-sided,
# powered for a large one (|synthetic - original| >= 10 points success); ``no large
# shift`` = 90% CI inside +-10 points. +-5-point equivalence is reported only when the
# 90% CI happens to lie inside +-5 (it is not powered at 180 pairs).
DIFFICULTY_LARGE_SHIFT = 0.10
DIFFICULTY_EQUIVALENCE = 0.05
DIFFICULTY_ARMS = ("original", "synthetic")
# Own seed (slots, and so case and group ids, differ from part A's); both arms share it,
# so they get the same slots; ids then carry the arm (``-do`` / ``-ds``).
DIFFICULTY_SEED_LABEL = "slh-xp-h1b-difficulty-v1"
SLOT_FIELDS = ("task", "split", "kind", "family", "pattern", "style", "variant")
# Owner decision 2026-10-19 (disclosed limitation, docs/xp/retail-db-extension.md section 7):
# the address ``ambiguous``/``city`` slot family is excluded from BOTH arms. The synthetic
# ext-v2 users have only 3 such configurations, all used by part A, so the synthetic arm
# could never match an original-arm slot of that family. Both arms draw from the same
# reduced slot specification (the original arm too), so the pairs stay exactly matched;
# the check says nothing about ambiguous city referents.
DIFFICULTY_ADDRESS_EXCLUDE: frozenset[tuple[str, str]] = frozenset({("ambiguous", "city")})


def difficulty_seed() -> int:
    return int(pc.sha_text(DIFFICULTY_SEED_LABEL)[:12], 16)


def _slot_spec(cases: list[dict[str, Any]]) -> dict[str, list[tuple[Any, ...]]]:
    """pair -> sorted (kind, family, pattern, style, variant, ...) of its cases."""
    spec: dict[str, list[tuple[Any, ...]]] = {}
    for c in cases:
        spec.setdefault(c["pair"], []).append(tuple(c[f] for f in SLOT_FIELDS))
    return {k: sorted(v) for k, v in sorted(spec.items())}


def difficulty(sizes: dict[str, int], *, exclude_users: set[str], retail_db: str,
               bans: dict[str, Any] | None = None,
               gen_seed: int | None = None) -> dict[str, Any]:
    """The two matched arms (cases with ``role`` = ``difficulty``, ``difficulty_arm`` and
    ``pair`` = task:slot). ``exclude_users``: the part-A users (never reused here).

    ``bans`` (the release's ban list: historical users, and orders whose owners are then
    banned too, as in ``generate``) apply to the **synthetic** arm, so a later release can
    never reuse a previously exposed synthetic user. Only the deliberately diagnostic
    **original** arm is exempt: the earlier-test ban list covers 405 of the 500 original
    users, so it draws from every original user not in part A; its units are a diagnostic
    only: B1 with an empty library, never a lesson, never an F' or C
    training/calibration/evaluation example, never a proposal (``fgate.score``,
    ``partner_c.build`` skip the role).

    **One common slot specification.** The original arm is drawn first (the scarcer
    population); its address slot families (the address picker may fall back to another
    family when a referent kind is exhausted) are then pinned for the synthetic arm, which
    has no fallback. Both arms draw from the same reduced slot specification: the address
    slot families in ``DIFFICULTY_ADDRESS_EXCLUDE`` (``ambiguous``/``city``) are excluded
    from both (owner decision 2026-10-19, a disclosed limitation). Every pair must then match exactly on task, split, kind, family,
    pattern, style and the phrasings; any mismatch is refused (``CapacityShort``)."""
    from slh.xp import retaildb

    if retail_db == retaildb.ORIGINAL:
        raise ValueError("the difficulty check needs an extension database (synthetic users)")
    bans = bans or load_bans(None)
    s = difficulty_seed() if gen_seed is None else gen_seed
    original_users = set(retaildb.load(retaildb.ORIGINAL)["users"])
    split_sizes = {t: {"train": 0, "validation": 0, "sentinel": n} for t, n in sizes.items()}
    with retaildb.use(retail_db):
        from slh.xp.tasks import common as cm

        everyone = set(cm.load_db()["users"])
        out: list[dict[str, Any]] = []
        report: dict[str, Any] = {"sizes": sizes, "phrasings": DIFFICULTY_PHRASINGS,
                                  "large_shift": DIFFICULTY_LARGE_SHIFT,
                                  "equivalence_reported": DIFFICULTY_EQUIVALENCE,
                                  "seed_label": DIFFICULTY_SEED_LABEL,
                                  "address_excluded_slot_families": sorted(
                                      "/".join(x) for x in DIFFICULTY_ADDRESS_EXCLUDE),
                                  "bans_sha256": bans["sha256"], "arms": {}}
        pinned: dict[tuple[str, int], tuple[str, int]] | None = None
        for arm in DIFFICULTY_ARMS:
            if arm == "original":  # diagnostic arm: exempt from the historical bans
                users = sorted((everyone - original_users) | set(exclude_users))
                orders: list[str] = []
            else:
                users = sorted(original_users | set(exclude_users) | set(bans["users"]))
                orders = list(bans["orders"])
            body = {"schema": BANS_SCHEMA, "users": users, "orders": orders, "sources": []}
            arm_bans = {**body, "sha256": pc.sha_text(pc.canonical(body))}
            res = _generate(split_sizes, arm_bans, gen_seed=s, address_families=pinned,
                            address_exclude=DIFFICULTY_ADDRESS_EXCLUDE)
            if arm == "original":
                pinned = {(c["split"], int(c["group_index"])): (c["pattern"], c["style"])
                          for c in _with_index(res["cases"]) if c["task"] == "address"}
            for c in _with_index(res["cases"]):
                tag = f"-d{arm[0]}"
                idx = c.pop("group_index")
                out.append({**c, "case_id": c["case_id"] + tag, "group": c["group"] + tag,
                            "role": DIFFICULTY_ROLE, "difficulty_arm": arm,
                            "pair": f"{c['task']}:{idx}"})
            arm_cases = [c for c in out if c["difficulty_arm"] == arm]
            report["arms"][arm] = {
                "groups": len({c["pair"] for c in arm_cases}), "cases": len(arm_cases),
                "users": len({c["entities"]["user_id"] for c in arm_cases}),
                "banned_users": res["report"]["banned_users"]}
    arm_users = {a: {c["entities"]["user_id"] for c in out if c["difficulty_arm"] == a}
                 for a in DIFFICULTY_ARMS}
    if arm_users["original"] - original_users or arm_users["synthetic"] & original_users:
        raise CapacityShort("a difficulty arm drew a user of the other population")
    if (arm_users["original"] | arm_users["synthetic"]) & set(exclude_users):
        raise CapacityShort("a difficulty arm reused a part-A user")
    if any(c["task"] == "address" and (c["kind"], c["pattern"]) in DIFFICULTY_ADDRESS_EXCLUDE
           for c in out):
        raise CapacityShort("an excluded address slot family reached a difficulty arm")
    spec = {a: _slot_spec([c for c in out if c["difficulty_arm"] == a])
            for a in DIFFICULTY_ARMS}
    expected = {f"{t}:{i}" for t, n in sizes.items() for i in range(n)}
    for a in DIFFICULTY_ARMS:
        if set(spec[a]) != expected:
            raise CapacityShort(f"the {a} arm does not cover the registered pairs")
    if bad := sorted(p for p in expected if spec["original"][p] != spec["synthetic"][p]):
        raise CapacityShort(f"{len(bad)} pairs differ between the arms (e.g. {bad[0]}): "
                            "no common slot specification")
    report["pairs"] = len(expected)
    report["pairs_matched_exactly"] = len(expected)
    report["slot_spec_sha256"] = pc.sha_text(pc.canonical(
        {p: [list(x) for x in v] for p, v in spec["original"].items()}))
    return {"cases": out, "report": report}


def _with_index(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cases with ``group_index``: the slot index of their group within its task, in
    generation order (generators emit each task's groups in slot order)."""
    idx: dict[str, dict[str, int]] = {}
    out = []
    for c in cases:
        g = idx.setdefault(c["task"], {})
        out.append({**c, "group_index": g.setdefault(c["group"], len(g))})
    return out


def parse_drop_kinds(values: list[str]) -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for v in values:
        task, _, kind = v.partition(":")
        if task not in TASK_ORDER or not kind:
            raise SystemExit(f"--drop-kind {v!r}: expected TASK:KIND")
        out.setdefault(task, []).append(kind)
    return {t: tuple(k) for t, k in out.items()}


def context_of(case: dict[str, Any]) -> str:
    """Decision context: task x gold type (act = tool_call, confirm, clarify)."""
    gtype = case["gold"]["type"]
    return f"{case['task']}:{'act' if gtype == 'tool_call' else gtype}"


# --------------------------------------------------------------------------- units


def inclusion_vectors(case_id: str, n_lessons: int, draws: int, *, salt: str = "fgate-incl-v1"
                      ) -> list[tuple[int, ...]]:
    """Anchor (all zero) first, then ``draws`` vectors in antithetic pairs."""
    from slh.xp.tasks import common as cm

    if draws % 2:
        raise ValueError("draws must be even (antithetic pairs)")
    if n_lessons < 2:
        raise ValueError("the pool needs at least two lessons")
    r = cm.rng(salt, case_id)
    zero = tuple([0] * n_lessons)
    ones = tuple([1] * n_lessons)
    seen = {zero}
    out = [zero]
    attempts = 0
    while len(out) < draws + 1:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("could not draw distinct inclusion vectors")
        v = tuple(int(r.random() < 0.5) for _ in range(n_lessons))
        w = tuple(1 - x for x in v)
        if v in (zero, ones) or v in seen or w in seen:
            continue
        seen.update((v, w))
        out.extend((v, w))
    return out


def bits(x: tuple[int, ...] | list[int]) -> str:
    return "".join(str(int(b)) for b in x)


def units(cases: list[dict[str, Any]], pool: dict[str, Any], *, draws: int
          ) -> dict[str, Any]:
    """Partner-facing units (no gold) and our unit key."""
    from slh.xp.fgate import pool as fp
    from slh.xp.partner import plan as pp

    lessons = pool["lessons"]
    prompts: dict[str, dict[str, Any]] = {}

    def prompt(x: tuple[int, ...]) -> dict[str, str]:
        text = fp.render(lessons, x)
        sha = pc.sha_text(text)
        if sha not in prompts:
            prompts[sha] = {"notes_sha256": sha, "text": text,
                            "prompt_hash": pp.prompt_hash("notes", text)}
        return prompts[sha]

    rows, key = [], []
    for c in sorted(cases, key=lambda c: (TASK_ORDER.index(c["task"]), c["role"], c["case_id"])):
        h = pc.case_hash(c)
        if c["role"] == "fit":
            vectors = [("fgate", x) for x in inclusion_vectors(c["case_id"], len(lessons), draws)]
        elif c["role"] == DIFFICULTY_ROLE:
            vectors = [("difficulty", tuple([0] * len(lessons)))]
        else:
            vectors = [("c-collect", tuple([0] * len(lessons)))]
        for track, x in vectors:
            p = prompt(x)
            uid = "u-" + pc.sha_text(f"{c['case_id']}|{track}|{bits(x)}")[:16]
            rows.append({"unit_id": uid, "track": track, "case_id": c["case_id"],
                         "case_hash": h, "x": bits(x), "anchor": not any(x),
                         "notes_sha256": p["notes_sha256"], "prompt_hash": p["prompt_hash"]})
            key.append({"unit_id": uid, "track": track, "case_id": c["case_id"], "x": bits(x),
                        "task": c["task"], "role": c["role"], "kind": c["kind"],
                        "group": c["group"], "family": c["family"],
                        "context": context_of(c),
                        **({"arm": c["difficulty_arm"], "pair": c["pair"],
                            "variant": c["variant"]}
                           if c["role"] == DIFFICULTY_ROLE else {})})
    return {"units": rows, "prompts": sorted(prompts.values(), key=lambda p: p["notes_sha256"]),
            "key": key}


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.fgate.design")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bans")
    b.add_argument("--from", dest="files", type=Path, action="append", required=True)
    b.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("capacity")
    c.add_argument("--bans", type=Path)
    c.add_argument("--tiny", action="store_true")
    c.add_argument("--retail-db", default=RDB.GENERATED_DEFAULT,
                   help="pinned retail database (slh.xp.retaildb): original | ext-v1 | "
                        f"ext-v2 (default {RDB.GENERATED_DEFAULT})")
    c.add_argument("--sizes", type=Path, help="JSON task -> split -> groups")
    c.add_argument("--drop-kind", action="append", default=[], metavar="TASK:KIND")
    args = ap.parse_args(argv)
    if args.cmd == "bans":
        bans = make_bans(args.files)
        pc.write_json(args.out, bans)
        print(json.dumps({"users": len(bans["users"]), "orders": len(bans["orders"]),
                          "sha256": bans["sha256"]}))
        return 0
    try:
        sizes = (pc.read_json(args.sizes) if args.sizes
                 else TINY_SIZES if args.tiny else FULL_SIZES)
        res = generate(sizes, load_bans(args.bans), drop_kinds=parse_drop_kinds(args.drop_kind),
                       retail_db=args.retail_db)
    except CapacityShort as exc:
        print(json.dumps({"ok": False, "short": str(exc)}))
        return 1
    print(json.dumps({"ok": True, **res["report"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
