"""Our side: build the hand-off 1 package (F' screen units + C proposal collection).

    python -m slh.xp.fgate.build --pool POOL.json --bans BANS.json \
        --out ~/.slh-xp/handoff/h1-<id> --keys ~/.slh-xp/keys/h1-<id> [--tiny] \
        [--sizes SIZES.json] [--drop-kind address:ambiguous] [--draws 16] \
        [--retail-db ext-v2] [--no-difficulty]

Writes the package (``plan.json`` sealed with its own SHA-256; ``questions.jsonl`` with
no gold; ``prompts.jsonl``; ``units.jsonl``; ``pool.json``; ``README-h1.md``) and, to
the keys directory only, the full cases with gold (``cases.answer-key.jsonl``), the
unit key (``units-key.jsonl``: role, kind, context, group per unit) and the build
report. On an extension database (default ``ext-v2``) the package also carries the
synthetic-difficulty check (``fgate.design.difficulty``: B1 on matched original-only and
synthetic-only prefixes, track ``difficulty``, never training data).
Refuses: an output or keys directory inside the repository or not empty, a
missing ban list (except ``--tiny``), a question carrying an answer field.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from slh.xp import retaildb as RDB
from slh.xp.fgate import design as D
from slh.xp.fgate import pool as fp
from slh.xp.fgate import units as U
from slh.xp.partner import common as pc

LIMITS = {"case_limit_s": 300, "concurrency": 32, "max_output_tokens": 4096}
DECIDER_SEED = 7401  # the registered phase-1 plan seed (B1 decision seed source)
EXPORT_CAP = 100_000_000
CODE_PREFIXES = ("slh/",)
# Partner GPU cap for hand-off 1 (owner decision 2026-10-18, with 16 draws per prefix).
H1_CAP_H100_HOURS = 6.5


def _inside(path: Path, root: Path = pc.REPO) -> bool:
    try:
        Path(path).expanduser().resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _fresh(path: Path, what: str, *, allow_repo: bool) -> Path:
    path = Path(path).expanduser()
    if _inside(path) and not allow_repo:
        raise SystemExit(f"{what} {path} lies inside the repository")
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"{what} {path} is not empty")
    path.mkdir(parents=True, exist_ok=True)
    return path


def code_registration(root: Path = pc.REPO) -> dict[str, str]:
    files = pc.code_hashes(root)["files"]
    kit = pc.kit_paths(root)
    return {k: v for k, v in files.items() if k.startswith(CODE_PREFIXES) and k.endswith(".py")
            and (kit is None or k in kit)}


def seal(body: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in body.items() if k != "sha256"}
    return {**body, "sha256": pc.sha_text(pc.canonical(body))}


def readme(plan: dict[str, Any]) -> str:
    n = plan["counts"]
    return f"""# Hand-off 1: decision units ({plan["plan_id"]})

{n["units"]} decision units ({n["fgate_units"]} lesson-screen units, {n["collect_units"]}
collection units, {n.get("difficulty_units", 0)} difficulty-check units) on {n["questions"]}
decision prefixes. One model only: the frozen 9B
agent served with vLLM exactly as in the plain phase-1 runbook (README-plain.md,
section 2: the agent server; no teacher server is needed). No network beyond localhost.

```bash
P="--package package-h1 --endpoint endpoint.yaml"
python -m slh.xp.fgate.units run --package package-h1 --out runs/h1-dry --dry-run   # no GPU; "export ok"
python -m slh.xp.fgate.units run $P --out runs/h1-smoke --smoke                      # small slice
python -m slh.xp.fgate.units run $P --out runs/h1                                    # full
python -m slh.xp.fgate.units export --run runs/h1 --out export-h1
```

Rerun the same command after an interruption (finished units are kept). Exit 3 = halted
(cause in `halt.json`), exit 4 = refused (package, code or settings differ). Send back
`export-h1/` (and the smoke export) unchanged. Estimated GPU time: {plan["estimate"]["h100_hours"]}
H100-hours ({plan["estimate"]["basis"]}).
"""


def estimate(units: int) -> dict[str, Any]:
    lo, hi = units / (3.0 * 3600), units / (1.0 * 3600)
    return {"h100_hours": f"{lo:.1f}-{hi:.1f}",
            "basis": "1-3 decisions per second per H100 (phase-1 assumption 3/s with prefix "
                     "caching; lesson-varied system prompts cache less)",
            "node_hours_8xh100": f"{lo / 8:.2f}-{hi / 8:.2f}",
            "cap_h100_hours": H1_CAP_H100_HOURS, "within_cap": hi <= H1_CAP_H100_HOURS}


def build(*, pool: dict[str, Any], bans: dict[str, Any], out: Path, keys: Path,
          plan_id: str, tiny: bool = False, sizes: dict[str, Any] | None = None,
          draws: int | None = None, drop_kinds: dict[str, tuple[str, ...]] | None = None,
          allow_repo: bool = False, retail_db: str = "original",
          difficulty: bool = True) -> dict[str, Any]:
    """``difficulty``: add the synthetic-difficulty check units (extension databases
    only; sizes ``D.DIFFICULTY_SIZES``, tiny ``D.TINY_DIFFICULTY_SIZES``)."""
    fp.verify_pool(pool)
    if bans.get("empty") and not tiny:
        raise SystemExit("a ban list is required (python -m slh.xp.fgate.design bans ...)")
    sizes = sizes or (D.TINY_SIZES if tiny else D.FULL_SIZES)
    draws = draws or (D.TINY_DRAWS if tiny else D.DRAWS)
    gen = D.generate(sizes, bans, drop_kinds=drop_kinds, retail_db=retail_db)
    cases = gen["cases"]
    diff_report = None
    if difficulty and retail_db != RDB.ORIGINAL:
        part_a_users = {c["entities"]["user_id"] for c in cases} | {
            x for c in cases for x in D._USER.findall(json.dumps(c["messages"]))}
        dsizes = D.TINY_DIFFICULTY_SIZES if tiny else D.DIFFICULTY_SIZES
        # the release's ban list applies to the synthetic arm (only the diagnostic
        # original arm is exempt; fgate.design.difficulty)
        diff = D.difficulty(dsizes, exclude_users=part_a_users, retail_db=retail_db,
                            bans=bans)
        if {c["case_id"] for c in diff["cases"]} & {c["case_id"] for c in cases}:
            raise SystemExit("a difficulty-check case id repeats a part-A case id")
        cases = cases + diff["cases"]
        diff_report = diff["report"]
    u = D.units(cases, pool, draws=draws)
    out = _fresh(out, "--out", allow_repo=allow_repo)
    keys = _fresh(keys, "--keys", allow_repo=allow_repo)
    questions = [pc.question_row(c) for c in cases]
    files: dict[str, bytes] = {
        "questions.jsonl": pc.jsonl_bytes(questions)[0],
        "prompts.jsonl": pc.jsonl_bytes(u["prompts"])[0],
        "units.jsonl": pc.jsonl_bytes(u["units"])[0],
        "pool.json": (json.dumps(pool, indent=1, sort_keys=True) + "\n").encode()}
    for name, data in files.items():
        (out / name).write_bytes(data)
    n_fgate = sum(1 for x in u["units"] if x["track"] == "fgate")
    n_diff = sum(1 for x in u["units"] if x["track"] == "difficulty")
    body = {"schema": U.PLAN_SCHEMA, "plan_id": plan_id, "tiny": tiny,
            "seed": DECIDER_SEED, "limits": LIMITS, "export_cap_bytes": EXPORT_CAP,
            "draws": draws, "pool_sha256": pool["sha256"], "bans_sha256": bans["sha256"],
            "generator": {"seed_label": D.SEED_LABEL, "sizes": sizes,
                          "drop_kinds": gen["report"]["drop_kinds"]},
            # The pinned retail database the prefixes were built on (slh.xp.retaildb).
            "retail_db": gen["report"]["retail_db"],
            # Synthetic-difficulty check (counts only; arms and pairs are in the keys).
            "difficulty": None if diff_report is None else {
                **{k: diff_report[k] for k in ("sizes", "phrasings", "large_shift",
                                               "equivalence_reported", "slot_spec_sha256")},
                "units": n_diff},
            "files": {n: {"sha256": pc.sha_bytes(d), "bytes": len(d)} for n, d in files.items()},
            "code": code_registration(),
            "counts": {"questions": len(questions), "units": len(u["units"]),
                       "fgate_units": n_fgate, "difficulty_units": n_diff,
                       "collect_units": len(u["units"]) - n_fgate - n_diff,
                       "prompts": len(u["prompts"]), "lessons": len(pool["lessons"])},
            "estimate": estimate(len(u["units"]))}
    plan = seal(body)
    pc.write_json(out / "plan.json", plan)
    (out / "README-h1.md").write_text(readme(plan), encoding="utf-8")
    key_rows = pc.answer_key_rows(cases)
    for row, case in zip(key_rows, cases, strict=True):
        row["role"] = case["role"]
    (keys / "cases.answer-key.jsonl").write_bytes(pc.jsonl_bytes(key_rows)[0])
    (keys / "units-key.jsonl").write_bytes(pc.jsonl_bytes(u["key"])[0])
    report = {"plan_id": plan_id, "plan_sha256": plan["sha256"], **gen["report"],
              "difficulty": diff_report, "counts": plan["counts"],
              "estimate": plan["estimate"]}
    pc.write_json(keys / "build-report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.fgate.build")
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--bans", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keys", type=Path, required=True)
    ap.add_argument("--plan-id", default="xp-h1-c-fgate")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--sizes", type=Path, help="JSON: task -> {train, validation, sentinel}")
    ap.add_argument("--draws", type=int)
    ap.add_argument("--retail-db", default=RDB.GENERATED_DEFAULT,
                    help="pinned retail database (slh.xp.retaildb): original | ext-v1 | "
                         f"ext-v2 (default {RDB.GENERATED_DEFAULT})")
    ap.add_argument("--no-difficulty", action="store_true",
                    help="omit the synthetic-difficulty check units")
    ap.add_argument("--drop-kind", action="append", default=[],
                    help="task:kind, an explicit capacity fallback (recorded)")
    args = ap.parse_args(argv)
    drop: dict[str, tuple[str, ...]] = {}
    for item in args.drop_kind:
        task, kind = item.split(":", 1)
        drop[task] = (*drop.get(task, ()), kind)
    try:
        report = build(pool=pc.read_json(args.pool), bans=D.load_bans(args.bans), out=args.out,
                       keys=args.keys, plan_id=args.plan_id, tiny=args.tiny,
                       sizes=pc.read_json(args.sizes) if args.sizes else None,
                       draws=args.draws, drop_kinds=drop, retail_db=args.retail_db,
                       difficulty=not args.no_difficulty)
    except D.CapacityShort as exc:
        print(f"capacity short: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
