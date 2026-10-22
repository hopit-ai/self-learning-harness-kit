"""Our side: arms, frozen libraries and the three hand-off packages of the BFCL track.

    python -m slh.xp.bfcl.plan h1 --out PKG-H1
    python -m slh.xp.bfcl.plan h2 --pool-evidence POOL-U.json --pool-search POOL-S.json --out PKG-H2
    python -m slh.xp.bfcl.plan freeze --arm ARM --text FILE|--library LIB.json --out LIB-ARM.json
    python -m slh.xp.bfcl.plan freeze-selected --arm FPRIME|SEARCH --pool POOL.json --report FPRIME-X.json --out LIB-ARM.json
    python -m slh.xp.bfcl.plan h3 --libraries LIBDIR --out PKG-H3 [--arms B1,SEARCH,...]
    python -m slh.xp.bfcl.plan h3-official --out PKG-H3-OFFICIAL   # descriptive B1, T=0.001

A package is one directory with ``plan.json``: the cells to run (one cell = one arm x
trial or one F' draw; every instance at most once per cell, because the official backend
instances are keyed by model name and entry id inside one process), the notes texts by
SHA-256, the decoding settings, the splits hash, the vendor pins and the SHA-256 of every
``slh/xp/bfcl`` file plus ``slh/xp/partner/common.py``. ``plan.json`` hashes itself.

Split discipline (``splits.guard_learning``): H1 runs only discovery instances, H2 only
gate instances; H3 (the sealed test) is built only from **frozen** libraries, every one
of which records that its evidence came from discovery (and selection from gate) only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from slh.xp.bfcl import data, splits, vendor
from slh.xp.partner.common import REPO

PLAN_SCHEMA = "xp-bfcl-plan-v1"
LIBRARY_SCHEMA = "xp-bfcl-library-v1"
POOL_SCHEMA = "xp-bfcl-pool-v1"
SERVED_MODEL = "qwen3.5-9b-vllm"
# Official BFCL decoding is ``bfcl generate --temperature`` default 0.001. Owner decision
# 2026-10-18 (docs/xp/bfcl-plan.md, decision 1): every arm, discovery, F' and confirmation
# use "sampled" decoding (temperature 0.7) for informative trials, a disclosed deviation
# (deviation B-D1); "official" is kept for one descriptive B1 trial on test (``h3-official``).
DECODING = {"official": {"temperature": 0.001, "sampling": {}},
            "sampled": {"temperature": 0.7, "sampling": {"top_p": 0.8}}}
DEFAULT_DECODING = "sampled"
# Protocol detail B-P1 (2026-10-19, after hand-off 1a): every request carries
# max_tokens = min(4096, context_len - prompt_tokens - 2) (1000 when the prompt already
# fills the context), BFCL's own rule for locally served models (vendored
# bfcl_eval/model_handler/local_inference/base_oss_handler.py:328-336). context_len is the
# served --max-model-len (the endpoint's serving_args, else this default); prompt tokens
# are counted with the pinned 9B tokenizer. Stopping on the cap is an episode result.
GENERATION = {"max_output_tokens": 4096, "over_context_tokens": 1000, "context_len": 131072,
              "rule": "min(max_output_tokens, context_len - prompt_tokens - 2); "
                      "over_context_tokens if context_len < prompt_tokens + 2",
              "source": "bfcl_eval/model_handler/local_inference/base_oss_handler.py:328-336",
              "prompt_tokens": "slh.xp.partner.context.body_tokens (pinned 9B tokenizer)"}
TRIALS_DISCOVERY = 2
TRIALS_TEST = 4
FEASIBILITY_ROOTS = 10           # x 2 categories = 20 instances, run first in H1
FPRIME_DRAWS = 32                # randomized vectors per gate instance (16 antithetic pairs)
POOL_MAX = 8                     # candidate lessons per pool entering F'

# The seven confirmation arms (design review of 2026-10-18): what fills the notes.
ARMS = {
    "B1": "standard harness: the official conversation, no system message",
    "B1P": "frozen meaning-preserving prompt rewrite: one fixed neutral system note",
    "PERSON": "person-written procedure library (frozen)",
    "UNGATED": "evidence-learned library, every candidate of pool U (frozen)",
    "FPRIME": "F'-selected subset L* of pool U (frozen; = B1 when L* is empty)",
    "SEARCH": "matched-budget no-evidence search: F'-selected subset of pool S (frozen)",
    "MEMORY": "published memory baseline: ExpeL-style, train-only, frozen insights + "
              "retrieved discovery successes",
}
# B1P: restates only what the official protocol already gives the model (functions are
# available; requests arrive in turns); no procedure. Frozen 2026-10-18.
NEUTRAL_NOTE = ("You can call the functions provided in this conversation. The user will "
                "make one or more requests, one turn at a time. Handle each request using "
                "the functions as needed, then reply.")
# PERSON arm (owner decision 2026-10-18, decision 4): written by the project team from the
# public BFCL documentation and function documentation only, frozen before any BFCL run
# (docs/xp/bfcl-person/PROVENANCE.md). build_h3 refuses any other PERSON text.
PERSON_LIBRARY = REPO / "docs" / "xp" / "bfcl-person" / "library.txt"
PERSON_LIBRARY_SHA256 = "9b0f5e924adbc8ce4a010cfffec8462c3036b68b643cb5c76a20266ab9598b4f"
CODE_GLOBS = ("slh/xp/bfcl/*.py", "slh/xp/partner/common.py", "slh/__init__.py",
              "slh/xp/__init__.py")


class PlanRefused(ValueError):
    pass


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seal(body: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in body.items() if k != "sha256"}
    return {**body, "sha256": sha_text(_canon(body))}


def check_seal(obj: dict[str, Any], what: str) -> dict[str, Any]:
    if seal(obj)["sha256"] != obj.get("sha256"):
        raise PlanRefused(f"{what}: content differs from its sha256")
    return obj


def code_hashes(root: Path = REPO) -> dict[str, str]:
    """SHA-256 of the registered code files the kit ships (our-side modules such as
    ``learn``, ``expel``, ``fprime`` and ``confirm`` are not in the kit)."""
    from slh.xp.partner.common import kit_paths

    kit = kit_paths(Path(root))
    out = {}
    for pattern in CODE_GLOBS:
        for p in sorted(Path(root).glob(pattern)):
            if p.is_file() and (kit is None or p.relative_to(root).as_posix() in kit):
                out[p.relative_to(root).as_posix()] = vendor.sha_file(p)
    return dict(sorted(out.items()))


def _cell(cell_id: str, arm: str, trial: int, purpose: str, units: list[dict[str, Any]]
          ) -> dict[str, Any]:
    ids = [u["instance"] for u in units]
    if len(ids) != len(set(ids)):
        raise PlanRefused(f"{cell_id}: an instance occurs twice in one cell")
    return {"cell_id": cell_id, "arm": arm, "trial": trial, "purpose": purpose,
            "units": units}


def _package(handoff: str, man: dict[str, Any], cells: list[dict[str, Any]],
             notes: dict[str, str], *, decoding: str, extra: dict[str, Any] | None = None
             ) -> dict[str, Any]:
    if decoding not in DECODING:
        raise PlanRefused(f"decoding is one of {sorted(DECODING)}")
    return seal({"schema": PLAN_SCHEMA, "handoff": handoff, "splits_sha256": man["sha256"],
                 "vendor": {k: v for k, v in vendor.expected().items() if k != "files"},
                 "served_model": SERVED_MODEL, "decoding": {"name": decoding,
                                                            **DECODING[decoding]},
                 "generation": dict(GENERATION),
                 "code": code_hashes(), "notes": dict(sorted(notes.items())),
                 "cells": cells, **(extra or {})})


def _units(ids: list[str], notes_sha: str | None = None) -> list[dict[str, Any]]:
    return [{"instance": i, "notes_sha256": notes_sha} for i in ids]


# --------------------------------------------------------------------------- H1


def build_h1(man: dict[str, Any], *, decoding: str = DEFAULT_DECODING) -> dict[str, Any]:
    disc = splits.instances(man, "discovery")
    splits.guard_learning(man, disc, allowed=("discovery",))
    feas_roots = man["roles"]["discovery"][:FEASIBILITY_ROOTS]
    feas = [data.instance_id(r, c) for r in feas_roots for c in data.PRIMARY]
    cells = [_cell("h1-feas-b1-t0", "B1", 0, "feasibility", _units(feas))]
    rest = [i for i in disc if i not in set(feas)]
    cells.append(_cell("h1-disc-b1-t0", "B1", 0, "discovery", _units(rest)))
    for t in range(1, TRIALS_DISCOVERY):
        cells.append(_cell(f"h1-disc-b1-t{t}", "B1", t, "discovery", _units(disc)))
    return _package("h1", man, cells, {}, decoding=decoding,
                    extra={"order": [c["cell_id"] for c in cells],
                           "note": "cell h1-feas-b1-t0 first; it is also discovery trial 0 "
                                   "of its 20 instances"})


# --------------------------------------------------------------------------- pools


def make_pool(name: str, lessons: list[str], *, provenance: dict[str, Any]) -> dict[str, Any]:
    if name not in ("U", "S"):
        raise PlanRefused("pool is U (evidence) or S (no-evidence search)")
    if not 2 <= len(lessons) <= POOL_MAX:
        raise PlanRefused(f"a pool holds 2..{POOL_MAX} lessons, got {len(lessons)}")
    rows = [{"id": f"{name}{k + 1:02d}", "text": t, "sha256": sha_text(t), "intent": "bfcl",
             "tools": ["any"]} for k, t in enumerate(lessons)]
    return seal({"schema": POOL_SCHEMA, "pool": name, "lessons": rows,
                 "provenance": provenance})


def verify_pool(pool: dict[str, Any]) -> dict[str, Any]:
    if pool.get("schema") != POOL_SCHEMA:
        raise PlanRefused("pool: unknown schema")
    check_seal(pool, "pool")
    for les in pool["lessons"]:
        if sha_text(les["text"]) != les["sha256"]:
            raise PlanRefused(f"pool: {les['id']} text differs from its sha256")
    return pool


# --------------------------------------------------------------------------- H2


def draws_for(n_lessons: int, draws: int = FPRIME_DRAWS) -> int:
    """Distinct antithetic draws available for a pool (no empty or full vector)."""
    return min(draws, 2 * (2 ** (n_lessons - 1) - 1))


def build_h2(man: dict[str, Any], pools: list[dict[str, Any]], *, decoding: str = DEFAULT_DECODING,
             draws: int = FPRIME_DRAWS) -> dict[str, Any]:
    from slh.xp.bfcl import fprime

    gate = splits.instances(man, "gate")
    splits.guard_learning(man, gate, allowed=("gate",))
    for p in pools:
        verify_pool(p)
        prov = p.get("provenance") or {}
        if prov.get("evidence_roles") not in (["discovery"], []):
            raise PlanRefused(f"pool {p['pool']}: evidence must come from discovery only")
    notes: dict[str, str] = {}
    cells = [_cell("h2-gate-anchor-t0", "B1", 0, "fprime-anchor", _units(gate))]
    per_pool = {}
    for p in pools:
        n_draws = per_pool[p["pool"]] = draws_for(len(p["lessons"]), draws)
        units = fprime.design(p, gate, draws=n_draws)
        for u in units:
            if u["notes"]:
                notes[sha_text(u["notes"])] = u["notes"]
        for d in range(1, n_draws + 1):
            rows = [{"instance": u["instance"], "x": u["x"],
                     "notes_sha256": sha_text(u["notes"]) if u["notes"] else None}
                    for u in units if u["draw"] == d]
            cells.append(_cell(f"h2-gate-{p['pool']}-d{d:02d}", f"FPRIME-{p['pool']}", 0,
                               "fprime-draw", rows))
    # Run order: the anchors first, then the pools' draws interleaved (U d01, S d01, U d02,
    # ...), so a run stopped at its cap leaves both pools with the same number of draws.
    draw_cells = [c["cell_id"] for c in cells[1:]]
    order = [cells[0]["cell_id"]] + sorted(
        draw_cells, key=lambda c: (int(c.rsplit("-d", 1)[1]), c.split("-")[2] != "U"))
    return _package("h2", man, cells, notes, decoding=decoding,
                    extra={"pools": {p["pool"]: p for p in pools}, "draws": per_pool,
                           "order": order})


# --------------------------------------------------------------------------- libraries


def freeze_library(arm: str, *, text: str | None = None, memory: dict[str, Any] | None = None,
                   provenance: dict[str, Any]) -> dict[str, Any]:
    """A frozen library for one confirmation arm (static text, or ExpeL memory)."""
    if arm not in ARMS or arm == "B1":
        raise PlanRefused(f"no library for arm {arm!r}")
    if arm == "B1P" and text not in (None, NEUTRAL_NOTE):
        raise PlanRefused("B1P is the frozen neutral note")
    if arm == "MEMORY":
        if memory is None:
            raise PlanRefused("MEMORY needs its frozen memory")
        kind, body = "retrieval", {"memory": memory}
    else:
        kind, body = "static", {"text": NEUTRAL_NOTE if arm == "B1P" else (text or "")}
    roles = provenance.get("evidence_roles", [])
    if "test" in roles or "test" in provenance.get("selection_roles", []):
        raise PlanRefused("a library may never use the test split")
    return seal({"schema": LIBRARY_SCHEMA, "arm": arm, "kind": kind, **body,
                 "provenance": provenance})


def freeze_selected(arm: str, pool: dict[str, Any], report: dict[str, Any], *,
                    report_sha256: str) -> dict[str, Any]:
    """The FPRIME (pool U) or SEARCH (pool S) library from the primary F' report: the
    advanced set rendered exactly as in F' (the rule of ``slh.xp.fgate.pool.render``,
    canonical order; not imported, so the kit's part B/C import closure is unchanged).
    An empty advanced set gives an empty text: the arm is B1 (plan s. 5, s. 6.1)."""
    want = {"FPRIME": "U", "SEARCH": "S"}
    if arm not in want:
        raise PlanRefused("freeze-selected is for FPRIME (pool U) or SEARCH (pool S)")
    verify_pool(pool)
    if pool["pool"] != want[arm]:
        raise PlanRefused(f"{arm} is selected from pool {want[arm]}, not {pool['pool']}")
    if report.get("schema") != "xp-bfcl-fprime-report-v1" or \
            report.get("pool_sha256") != pool["sha256"]:
        raise PlanRefused("report is not the primary F' report of this pool")
    ids = [les["id"] for les in pool["lessons"]]
    adv = list(report.get("advanced") or [])
    if not set(adv) <= set(ids):
        raise PlanRefused(f"advanced {adv} not in pool {ids}")
    chosen = [les["text"] for les in pool["lessons"] if les["id"] in adv]
    text = "## Lessons\n" + "\n".join(chosen) if chosen else ""
    prov = {"evidence_roles": (pool.get("provenance") or {}).get("evidence_roles", []),
            "selection_roles": ["gate"], "pool": pool["pool"], "pool_sha256": pool["sha256"],
            "fprime_report_sha256": report_sha256, "advanced": adv,
            "rule": "plan s. 5: the arm is the advanced set of the primary (official-score) "
                    "F' report; an empty set means the arm is B1"}
    return freeze_library(arm, text=text, provenance=prov)


def notes_for(library: dict[str, Any] | None, entry: dict[str, Any] | None = None) -> str:
    if library is None:
        return ""
    check_seal(library, f"library {library.get('arm')}")
    if library["kind"] == "static":
        return library["text"]
    from slh.xp.bfcl import expel

    return expel.render(library["memory"], entry)


# --------------------------------------------------------------------------- H3


# Proposed deviation B-D9 (2026-10-22, PROPOSED, not approved; docs/xp/bfcl-plan.md s. 0):
# L* is empty, so FPRIME is B1 and the registered family P1-P3 has no informative member
# about evidence-learned lessons; the proposed H3 runs B1 and SEARCH (primary: SEARCH - B1)
# plus UNGATED (descriptive), and the official-temperature B1 trial (``h3-official``).
H3_ARMS_BD9 = ("B1", "SEARCH", "UNGATED")


def build_h3(man: dict[str, Any], libraries: dict[str, dict[str, Any]], *,
             decoding: str = DEFAULT_DECODING, trials: int = TRIALS_TEST,
             package: Path | None = None, arms: tuple[str, ...] | None = None
             ) -> dict[str, Any]:
    """Every library is frozen and checked; ``arms`` (default: all seven, as registered)
    selects the arms that get cells. Run order is trial-major in the order of ``arms``
    (B1 t0, SEARCH t0, ..., B1 t1, ...), so a run stopped at its cap leaves the first arms
    with paired, equal numbers of trials."""
    test = splits.instances(man, "test")
    missing = sorted(set(ARMS) - {"B1"} - set(libraries))
    if missing:
        raise PlanRefused(f"frozen libraries missing for {missing}")
    run_arms = tuple(ARMS) if arms is None else tuple(arms)
    if "B1" not in run_arms or len(set(run_arms)) != len(run_arms) or \
            not set(run_arms) <= set(ARMS):
        raise PlanRefused(f"arms {run_arms}: distinct arms from {sorted(ARMS)}, B1 included")
    entries = {}
    for cat in data.PRIMARY:
        for e in data.load_category(cat, package):
            entries[e["id"]] = e
    notes: dict[str, str] = {}
    cells = []
    frozen = {}
    for arm in ARMS:
        lib = libraries.get(arm)
        if lib is not None:
            check_seal(lib, f"library {arm}")
            if lib["arm"] != arm:
                raise PlanRefused(f"library for {arm} is labelled {lib['arm']}")
            frozen[arm] = lib["sha256"]
        same_as_b1 = arm != "B1" and lib is not None and lib["kind"] == "static" \
            and not lib["text"]
        if same_as_b1:  # an empty F'/search selection: the arm is B1 itself (reported)
            continue
        if arm not in run_arms:
            continue
        if arm == "PERSON" and sha_text(lib["text"]) != PERSON_LIBRARY_SHA256:
            raise PlanRefused("PERSON text differs from the frozen person-written library")
        for t in range(trials):
            units = []
            for inst in test:
                text = notes_for(lib, entries[inst]) if lib else ""
                h = sha_text(text) if text else None
                if text:
                    notes[h] = text
                units.append({"instance": inst, "notes_sha256": h})
            cells.append(_cell(f"h3-{arm.lower()}-t{t}", arm, t, "confirmation", units))
    empty = sorted(a for a, lib in libraries.items()
                   if lib["kind"] == "static" and not lib["text"] and a != "B1P")
    bad = sorted(set(run_arms) & set(empty))
    if arms is not None and bad:
        raise PlanRefused(f"arms {bad} are empty selections (= B1); they are never run")
    rank = {a: k for k, a in enumerate(run_arms)}
    order = [c["cell_id"] for c in sorted(cells, key=lambda c: (c["trial"], rank[c["arm"]]))]
    extra: dict[str, Any] = {"libraries": frozen, "empty_arms": empty, "order": order}
    if arms is not None:
        extra["design"] = {"arms_run": list(run_arms),
                           "arms_not_run": sorted(set(ARMS) - set(run_arms) - set(empty)),
                           "deviation": "B-D9" if set(run_arms) != set(ARMS) - set(empty)
                           else None}
    return _package("h3", man, cells, notes, decoding=decoding, extra=extra)


def build_b1_official(man: dict[str, Any]) -> dict[str, Any]:
    """One descriptive B1 trial on test at the official temperature (comparability row)."""
    test = splits.instances(man, "test")
    cells = [_cell("h3o-b1-t0", "B1", 0, "comparability-official-decoding", _units(test))]
    return _package("h3-official", man, cells, {}, decoding="official",
                    extra={"note": "descriptive only; not part of the primary family"})


# --------------------------------------------------------------------------- CLI


def write_package(pkg: dict[str, Any], out: Path) -> None:
    out = Path(out)
    if out.resolve().is_relative_to(REPO.resolve()) and not out.resolve().is_relative_to(
            (REPO / "runs").resolve()):
        raise PlanRefused("packages go outside the repository (or under runs/)")
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(pkg, indent=1, sort_keys=True) + "\n",
                                   encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.bfcl.plan")
    ap.add_argument("--splits", type=Path, default=splits.DEFAULT_PATH)
    ap.add_argument("--decoding", default=DEFAULT_DECODING, choices=sorted(DECODING))
    sub = ap.add_subparsers(dest="cmd", required=True)
    h1 = sub.add_parser("h1")
    h1.add_argument("--out", type=Path, required=True)
    h2 = sub.add_parser("h2")
    h2.add_argument("--pool-evidence", type=Path, required=True)
    h2.add_argument("--pool-search", type=Path, required=True)
    h2.add_argument("--out", type=Path, required=True)
    fr = sub.add_parser("freeze")
    fr.add_argument("--arm", required=True, choices=sorted(set(ARMS) - {"B1"}))
    fr.add_argument("--text", type=Path, help="static library text (UTF-8)")
    fr.add_argument("--memory", type=Path, help="ExpeL memory JSON (MEMORY arm)")
    fr.add_argument("--provenance", type=Path, required=True)
    fr.add_argument("--out", type=Path, required=True)
    h3 = sub.add_parser("h3")
    h3.add_argument("--libraries", type=Path, required=True,
                    help="directory with LIB-<ARM>.json for every arm except B1")
    h3.add_argument("--out", type=Path, required=True)
    h3.add_argument("--arms", help="comma-separated arms to run (default: all, as "
                    "registered); e.g. B1,SEARCH,UNGATED (proposed deviation B-D9)")
    fs = sub.add_parser("freeze-selected", help="FPRIME/SEARCH library from the F' report")
    fs.add_argument("--arm", required=True, choices=["FPRIME", "SEARCH"])
    fs.add_argument("--pool", type=Path, required=True)
    fs.add_argument("--report", type=Path, required=True)
    fs.add_argument("--out", type=Path, required=True)
    h3o = sub.add_parser("h3-official", help="one B1 trial on test at temperature 0.001")
    h3o.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    man = splits.load(args.splits)
    if args.cmd == "h1":
        pkg = build_h1(man, decoding=args.decoding)
    elif args.cmd == "h2":
        pools = [json.loads(p.read_text()) for p in (args.pool_evidence, args.pool_search)]
        pkg = build_h2(man, pools, decoding=args.decoding)
    elif args.cmd == "freeze":
        prov = json.loads(args.provenance.read_text())
        lib = freeze_library(args.arm, text=args.text.read_text() if args.text else None,
                             memory=json.loads(args.memory.read_text()) if args.memory else None,
                             provenance=prov)
        args.out.write_text(json.dumps(lib, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"arm": lib["arm"], "sha256": lib["sha256"]}))
        return 0
    elif args.cmd == "freeze-selected":
        raw = args.report.read_bytes()
        lib = freeze_selected(args.arm, json.loads(args.pool.read_text()), json.loads(raw),
                              report_sha256=hashlib.sha256(raw).hexdigest())
        args.out.write_text(json.dumps(lib, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"arm": lib["arm"], "sha256": lib["sha256"],
                          "advanced": lib["provenance"]["advanced"],
                          "empty": not lib["text"]}))
        return 0
    elif args.cmd == "h3-official":
        pkg = build_b1_official(man)
    else:
        libs = {}
        for p in sorted(args.libraries.glob("LIB-*.json")):
            lib = json.loads(p.read_text())
            libs[lib["arm"]] = lib
        pkg = build_h3(man, libs, decoding=args.decoding,
                       arms=tuple(args.arms.split(",")) if args.arms else None)
    write_package(pkg, args.out)
    print(json.dumps({"handoff": pkg["handoff"], "cells": len(pkg["cells"]),
                      "episodes": sum(len(c["units"]) for c in pkg["cells"]),
                      "sha256": pkg["sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
