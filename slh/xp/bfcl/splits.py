"""Frozen BFCL multi-turn splits: discovery / gate / sealed test over the 200 roots.

    python -m slh.xp.bfcl.splits audit                 # print the audit only
    python -m slh.xp.bfcl.splits build --out splits/bfcl.json
    python -m slh.xp.bfcl.splits verify [--splits splits/bfcl.json]

Procedure (pre-declared, deterministic; seed label ``SEED_LABEL``):

1. **Sibling audit.** ``multi_turn_<cat>_<i>`` must belong to root ``i`` for every
   augmented category: same ``involved_classes``, and its question words are closer
   (Jaccard) to ``base_i`` than to any other base entry. A misaligned sibling stops the
   build (``SplitRefused``).
2. **Clusters.** Roots are joined (union-find) when they share an identical base
   ``initial_config``, when their question word sets have Jaccard >= ``Q_JACCARD``, or
   when their ground-truth call sets have Jaccard >= ``GT_JACCARD`` (procedure overlap;
   the answers are used for this grouping only and never for learning). A cluster and all
   siblings of its roots go to one partition.
3. **Strata.** A cluster's stratum is the domain (sorted ``involved_classes``) of its
   lowest root. Quotas per stratum: ``n * share`` floored, the remaining slots of each
   role handed to the strata with the largest remainders (ties in seeded order), so the
   totals are exactly ``SIZES`` (50 / 30 / 120).
4. **Assignment.** Within each stratum, clusters in (size desc, seeded) order go to the
   role with the largest remaining quota (ties: test, gate, discovery). A repair pass
   moves singleton clusters between roles of one stratum only if a multi-root cluster
   overshot. The totals are checked; any deviation is reported in ``audit``.
5. **Freeze.** The manifest records the vendor pins, the SHA-256 of every source file,
   the rules and thresholds, the audit, the assignment of every root and its instance
   ids, and ``sha256`` over the canonical content. ``verify`` recomputes everything from
   the vendored data and refuses any difference.

The **test** split never feeds learning: ``guard_learning`` refuses any instance id that
is not in the discovery split (lesson evidence) or the gate split (selection), and the
hand-off builders call it.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from slh.xp.bfcl import data, vendor
from slh.xp.partner.common import REPO

SCHEMA = "xp-bfcl-splits-v1"
SEED_LABEL = "slh-bfcl-splits-v1"
ROLES = ("discovery", "gate", "test")
SIZES = {"discovery": 50, "gate": 30, "test": 120}
Q_JACCARD = 0.40
GT_JACCARD = 0.60
DEFAULT_PATH = REPO / "splits/bfcl.json"
_WORD = re.compile(r"[a-z]+")


class SplitRefused(ValueError):
    pass


def _rng(*parts: Any) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16],
                             16))


def _words(entry: dict[str, Any]) -> set[str]:
    text = " ".join(m.get("content") or "" for turn in entry["question"] for m in turn)
    return set(_WORD.findall(text.lower()))


def _jac(a: set[Any], b: set[Any]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# --------------------------------------------------------------------------- audit


def sibling_audit(package: Path | None = None) -> dict[str, Any]:
    base = data.load_category("base", package)
    wb = [_words(e) for e in base]
    out: dict[str, Any] = {}
    for cat in data.CATEGORIES[1:]:
        sib = data.load_category(cat, package)
        if [e["id"] for e in sib] != [data.instance_id(i, cat) for i in range(data.N_ROOTS)]:
            raise SplitRefused(f"{cat}: ids are not 0..199 in order")
        bad, own_min, same_cfg = [], 1.0, 0
        for i, e in enumerate(sib):
            if e["involved_classes"] != base[i]["involved_classes"]:
                bad.append({"root": i, "reason": "involved_classes differ"})
                continue
            w = _words(e)
            own = _jac(w, wb[i])
            best = max(_jac(w, wb[j]) for j in range(data.N_ROOTS) if j != i)
            own_min = min(own_min, own)
            if own <= best:
                bad.append({"root": i, "reason": f"closer to another root ({own:.2f} <= {best:.2f})"})
            same_cfg += _canon(e["initial_config"]) == _canon(base[i]["initial_config"])
        out[cat] = {"misaligned": bad, "min_own_jaccard": round(own_min, 3),
                    "same_initial_config": same_cfg}
    return out


def clusters(package: Path | None = None) -> tuple[list[list[int]], dict[str, Any]]:
    base = data.load_category("base", package)
    gt = data.load_category("base", package, answers=True)
    if [g["id"] for g in gt] != [e["id"] for e in base]:
        raise SplitRefused("possible answers are not aligned with base entries")
    parent = list(range(data.N_ROOTS))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    links: dict[str, list[list[int]]] = {"initial_config": [], "question": [], "ground_truth": []}
    by_cfg: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(base):
        by_cfg[_canon(e["initial_config"])].append(i)
    for members in by_cfg.values():
        if len(members) > 1:
            links["initial_config"].append(members)
    words = [_words(e) for e in base]
    calls = [{c for turn in g["ground_truth"] for c in turn} for g in gt]
    for i, j in itertools.combinations(range(data.N_ROOTS), 2):
        if _jac(words[i], words[j]) >= Q_JACCARD:
            links["question"].append([i, j])
        if _jac(calls[i], calls[j]) >= GT_JACCARD:
            links["ground_truth"].append([i, j])
    for group in itertools.chain.from_iterable(links.values()):
        for k in group[1:]:
            a, b = find(group[0]), find(k)
            if a != b:
                parent[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(data.N_ROOTS):
        groups[find(i)].append(i)
    cl = sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
    doms = [data.domain_of(e) for e in base]
    mixed = [g for g in cl if len({doms[i] for i in g}) > 1]
    info = {"links": links, "multi_root_clusters": [g for g in cl if len(g) > 1],
            "mixed_domain_clusters": mixed,
            "thresholds": {"question_jaccard": Q_JACCARD, "ground_truth_jaccard": GT_JACCARD}}
    return cl, info


# --------------------------------------------------------------------------- build


def _quotas(strata: dict[str, int], rng: random.Random) -> dict[str, dict[str, int]]:
    total = sum(strata.values())
    if total != sum(SIZES.values()):
        raise SplitRefused(f"{total} roots, sizes add to {sum(SIZES.values())}")
    names = sorted(strata)
    order = names[:]
    rng.shuffle(order)
    rank = {s: k for k, s in enumerate(order)}
    q = {s: {r: 0 for r in ROLES} for s in names}
    for role in ("discovery", "gate"):
        share = SIZES[role] / total
        exact = {s: strata[s] * share for s in names}
        for s in names:
            q[s][role] = int(exact[s])
        left = SIZES[role] - sum(q[s][role] for s in names)
        cand = sorted(names, key=lambda s: (-(exact[s] - int(exact[s])), rank[s]))
        for s in cand:
            if left == 0:
                break
            if q[s]["discovery"] + q[s]["gate"] + 1 <= strata[s]:
                q[s][role] += 1
                left -= 1
        if left:
            raise SplitRefused(f"cannot place {left} {role} slots")
    for s in names:
        q[s]["test"] = strata[s] - q[s]["discovery"] - q[s]["gate"]
    return q


def build(package: Path | None = None) -> dict[str, Any]:
    package = data.package_dir(package)
    probs = vendor.file_problems(package)
    if probs:
        raise SplitRefused("vendored BFCL differs from the pins: " + "; ".join(probs))
    sib = sibling_audit(package)
    if any(v["misaligned"] for v in sib.values()):
        raise SplitRefused("misaligned siblings: " + _canon({k: v["misaligned"]
                                                               for k, v in sib.items()}))
    roots = data.roots(package)
    cl, info = clusters(package)
    strat_of_cluster = [tuple(roots[g[0]]["domain"]) for g in cl]
    strata: dict[str, int] = defaultdict(int)
    for g, s in zip(cl, strat_of_cluster, strict=True):
        strata["+".join(s)] += len(g)
    rng = _rng(SEED_LABEL, "quotas")
    quotas = _quotas(dict(strata), rng)
    assign: dict[int, str] = {}
    deviations = []
    for s in sorted(strata):
        members = [g for g, st in zip(cl, strat_of_cluster, strict=True) if "+".join(st) == s]
        r = _rng(SEED_LABEL, "assign", s)
        r.shuffle(members)
        members.sort(key=lambda g: -len(g))  # stable: seeded order within a size
        left = dict(quotas[s])
        for g in members:
            role = max(("test", "gate", "discovery"), key=lambda ro: left[ro])
            fits = [ro for ro in ("test", "gate", "discovery") if left[ro] >= len(g)]
            if fits:
                role = max(fits, key=lambda ro: left[ro])
            for i in g:
                assign[i] = role
            left[role] -= len(g)
        if any(v != 0 for v in left.values()):
            deviations.append({"stratum": s, "left": left})
    counts = {ro: sum(1 for v in assign.values() if v == ro) for ro in ROLES}
    if counts != SIZES:
        deviations.append({"totals": counts})
    split_rows = []
    cid = {i: k for k, g in enumerate(cl) for i in g}
    for row in roots:
        i = row["root"]
        split_rows.append({**row, "role": assign[i], "cluster": cid[i]})
    by_role = {ro: sorted(r["root"] for r in split_rows if r["role"] == ro) for ro in ROLES}
    families = {ro: dict(sorted(_count(r["family"] for r in split_rows if r["role"] == ro)
                                .items())) for ro in ROLES}
    body = {
        "schema": SCHEMA,
        "vendor": {k: v for k, v in vendor.expected().items() if k != "files"},
        "sources": {rel: vendor.FILES[rel] for rel in sorted(vendor.FILES)
                    if rel.startswith("data/")},
        "rules": {"seed_label": SEED_LABEL, "sizes": SIZES, "roles": list(ROLES),
                  "question_jaccard": Q_JACCARD, "ground_truth_jaccard": GT_JACCARD,
                  "stratum": "sorted involved_classes of the cluster's lowest root",
                  "primary_categories": list(data.PRIMARY),
                  "sibling_categories": list(data.CATEGORIES[1:]),
                  "test_never_feeds_learning": True},
        "audit": {"siblings": sib, "clusters": info, "strata": dict(sorted(strata.items())),
                  "quotas": quotas, "deviations": deviations, "counts": counts,
                  "families": families},
        "roles": by_role,
        "roots": split_rows,
    }
    return {**body, "sha256": hashlib.sha256(_canon(body).encode()).hexdigest()}


def _count(items: Any) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for x in items:
        out[x] += 1
    return out


# --------------------------------------------------------------------------- use


def load(path: Path = DEFAULT_PATH, *, package: Path | None = None, recompute: bool = True
         ) -> dict[str, Any]:
    man = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in man.items() if k != "sha256"}
    if man.get("schema") != SCHEMA:
        raise SplitRefused(f"{path}: unknown schema")
    if hashlib.sha256(_canon(body).encode()).hexdigest() != man.get("sha256"):
        raise SplitRefused(f"{path}: content differs from its sha256")
    if recompute:
        fresh = build(package)
        if fresh["sha256"] != man["sha256"]:
            raise SplitRefused(f"{path}: differs from a rebuild from the vendored data")
    return man


def instances(man: dict[str, Any], role: str, categories: tuple[str, ...] = data.PRIMARY
              ) -> list[str]:
    if role not in ROLES:
        raise ValueError(role)
    return [data.instance_id(i, c) for i in man["roles"][role] for c in categories]


def role_of(man: dict[str, Any], instance: str) -> str:
    root, _ = data.parse_instance(instance)
    for ro in ROLES:
        if root in man["roles"][ro]:
            return ro
    raise KeyError(instance)


def guard_learning(man: dict[str, Any], ids: list[str], *, allowed: tuple[str, ...]) -> None:
    """Refuse any instance outside ``allowed`` roles (``test`` is never allowed)."""
    if "test" in allowed:
        raise SplitRefused("the sealed test split never feeds learning or selection")
    bad = sorted({i for i in ids if role_of(man, i) not in allowed})
    if bad:
        raise SplitRefused(f"{len(bad)} instance(s) outside {allowed}: {bad[:5]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.bfcl.splits")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit")
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, default=DEFAULT_PATH)
    v = sub.add_parser("verify")
    v.add_argument("--splits", type=Path, default=DEFAULT_PATH)
    args = ap.parse_args(argv)
    if args.cmd == "audit":
        man = build()
        print(json.dumps({"audit": {k: man["audit"][k] for k in
                                    ("counts", "families", "deviations", "strata")},
                          "multi_root_clusters": man["audit"]["clusters"]["multi_root_clusters"],
                          "siblings": man["audit"]["siblings"]}, indent=1))
        return 0
    if args.cmd == "build":
        man = build()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"out": str(args.out), "sha256": man["sha256"],
                          "counts": man["audit"]["counts"]}))
        return 0
    man = load(args.splits)
    print(json.dumps({"ok": True, "sha256": man["sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
