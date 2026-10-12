"""The phase-0 -> phase-1 gate (proposed amendment 2): pre-registered conditions on
partner-side evidence only (no answer keys, no scores), and a registered,
deterministic rule that derives phase-1 settings from the phase-0 measurements.

    python scripts/xp/xp_boot.py --module gate evaluate --record R --package P --out G
    python scripts/xp/xp_boot.py --module gate verify --gate G --record R --package P \
        [--expect CONCURRENCY=32,LEARNING_UNITS=12,TEACHER_EAGER=0]

``R`` is ``$WORK/run_all/phase0.json`` (phase 0's job, run folder, archive and the
archive's SHA-256). ``evaluate`` writes ``gate.json``: every condition with its value
and bound, the decision (GO only if all hold), the derived settings and the phase-1
estimate, bound to the phase-0 archive's hash and self-hashed. ``verify`` (run by
every phase-1 job before anything starts) recomputes the gate from the same evidence
and refuses unless it equals the stored one, the decision is GO, the archive still has
the bound hash, and the job's settings are the derived ones. Exit 0 / 2 (NO-GO or
mismatch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

from slh.xp.partner import common as pc

RULE = "xp-gate-v1"
# Registered bounds (amendment 2, proposed).
GPU_HOURS_CAP = 1600
STARTUP_BOUND_S = {"teacher": 7200, "agent": 3600}
CONTEXT_LIMIT_SHARE = {"full_context": 0.05, "other": 0.02}
# Paired arms (learned vs its search control, per writer) in the slice. Owner (pending
# confirmation): a catastrophic check only, NO-GO when the error CASES differ by more
# than PAIRED_MAX_CASE_DIFF; the strict 2-point rule stays in the final analysis.
# PAIRED_MODE "points" restores the 2-point rule here.
PAIRED_MODE = "cases"
PAIRED_MAX_CASE_DIFF = 5
PAIRED_ERROR_DIFF = 0.02
ESTIMATE_MARGIN = 0.8  # the derived-setting estimate must stay within 0.8 x the cap
TIME_LIMIT_CLASSES = frozenset({"case_time_limit", "CaseTimeLimit", "RequestTimedOutAgain",
                                "Timeout", "TimeoutError", "ReadTimeout"})
MAX_MODEL_LEN = 131072
CONCURRENCY_BOUNDS = (8, 64)
UNITS_BOUNDS = (4, 16)
TIME_LIMIT_BOUNDS_H = (8, 48)
UTIL_SUM_MAX = 0.95
# Extra teacher flags accepted by the gate (registered; amendment 2 section 9): text
# only, MoE experts split over the GPUs, the engram tables kept on the GPUs.
TEACHER_EXTRA_ALLOWED = ("--language-model-only", "--enable-expert-parallel",
                         '--engram-config={"cpu_offload":false}')


def teacher_extra_ok(token: str) -> bool:
    if token in TEACHER_EXTRA_ALLOWED:
        return True
    if token.startswith("--engram-config="):  # the same JSON, spelled differently
        try:
            return json.loads(token.split("=", 1)[1]) == {"cpu_offload": False}
        except ValueError:
            return False
    return False


def _json(path: Path) -> Any:
    try:
        text = Path(path).read_text(encoding="utf-8")
        return json.loads(text[text.index("{"):])
    except (OSError, ValueError):
        return None


def _sha_file(path: Path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _arg(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) \
        else None


def _cond(name: str, ok: bool, value: Any, bound: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "value": value, "bound": bound}


def _export_valid(export: Path, package: Path, plan: dict[str, Any]) -> tuple[bool, str]:
    """The slice export against the schema and its own manifest (no keys)."""
    from slh.xp.partner import check

    if not (export / "manifest.json").is_file():
        return False, "no export manifest"
    try:
        check.read_shard(export, pc.read_json(package / "registration.json"), plan)
    except (check.CheckFailed, ValueError, KeyError) as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:200]}"
    return True, "ok"


def _extract(archive: Path, sha: str | None, dest: Path) -> Path:
    """The bound phase-0 archive, hash-checked, into a fresh directory; its job folder."""
    import tarfile

    if not sha or _sha_file(archive) != sha:
        raise ValueError("the phase-0 archive does not have the bound SHA-256")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")
    hits = sorted(dest.rglob("phase0.json"))
    if not hits:
        raise ValueError("the phase-0 archive holds no phase0.json")
    return hits[0].parent


def evaluate(record: dict[str, Any], package: Path) -> dict[str, Any]:
    """The gate, recomputed ONLY from the bound phase-0 archive's contents (extracted to
    a fresh directory after its hash is checked), never from a mutable run folder."""
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="xp-gate-"))
    try:
        run = _extract(Path(record.get("archive") or ""), record.get("archive_sha256"),
                       tmp)
        return _evaluate(record, run, package)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _evaluate(record: dict[str, Any], run: Path, package: Path) -> dict[str, Any]:
    from slh import serve_spec
    from slh.xp.partner import estimate as est
    from slh.xp.partner import plan as pp

    plan = pp.load(package / "plan.json")
    conds: list[dict[str, Any]] = []
    truncated = sorted(str(p.relative_to(run.parent)) for p in run.parent.rglob(
        "TRUNCATED.txt"))
    conds.append(_cond("phase-0 archive complete", not truncated, truncated, "no truncation"))
    status = _json(run / "phase0.json") or {}
    rows = status.get("rows") or []
    failed = [r["row"] for r in rows if r.get("status") != "pass"]
    want_rows = {"facts", "vendor", "bootstrap", "preflight", "dry-run", "serve",
                 "probe-agent", "measure", "slice"}
    if plan.get("writers") and any(w.startswith("T") for w in plan["writers"]):
        want_rows.add("probe-teacher")
    missing = sorted(want_rows - {r["row"] for r in rows})
    conds.append(_cond("every phase-0 row passed", not failed and not missing,
                       {"failed": failed, "missing": missing}, "all pass"))
    starts = {}
    for name in ("teacher", "agent"):
        s = _json(run / "serve" / f"{name}.startup.json") or {}
        starts[name] = {"status": s.get("status"), "seconds": s.get("seconds")}
    teacher_needed = any(w.startswith("T") for w in plan["writers"])
    ok = all((starts[n]["status"] == "ready" and (starts[n]["seconds"] or 0)
              <= STARTUP_BOUND_S[n]) for n in (("teacher", "agent") if teacher_needed
                                               else ("agent",)))
    conds.append(_cond("both servers started within bounds", ok, starts, STARTUP_BOUND_S))
    halts = [h for h in ("measure/halt.json", "slice/halt.json") if (run / h).is_file()]
    conds.append(_cond("no halt", not halts, halts, "none"))
    # the slice: every arm, valid rows, context limits, paired error rates
    slice_dir = run / "slice"
    decisions = pc.read_jsonl(slice_dir / "decisions.jsonl")
    arms = pc.read_jsonl(slice_dir / "arms.jsonl")
    cases = int(pp.SMOKES["slice"]["cases"])
    # Completeness: every registered task x arm instance of the slice plan, each with
    # exactly the registered set of distinct case ids (no duplicates), rows valid.
    from slh.xp.partner import check

    slice_plan = pp.effective(plan, pp.SMOKES["slice"])
    want_ids = {t: set(pp.smoke_ids([q["case_id"] for q in pc.read_jsonl(
        package / plan["cases"][t]["test"]["file"])], pp.SMOKES["slice"]))
        for t in plan["tasks"]}
    rows_by: dict[tuple[str, str], list[str]] = {}
    invalid = 0
    for r in decisions:
        rows_by.setdefault((r["task"], r["prompt_hash"]), []).append(r["case_id"])
        try:
            check.verify_failure_row(r, "slice")
        except check.CheckFailed:
            invalid += 1
    problems = instance_problems(slice_plan, arms, _run_texts(slice_dir, slice_plan))
    keyed = [a for a in arms if all(f in a for f in ("task", "run", "writer", "arm"))]
    for a in {pp.slot_key(a): a for a in keyed}.values():
        key = pp.slot_key(a)
        ids = rows_by.get((a["task"], a.get("prompt_hash")), [])
        if len(ids) != len(set(ids)):
            problems.append(f"duplicate case rows for {key}")
        elif set(ids) != want_ids.get(a["task"], set()):
            problems.append(f"{key}: {len(set(ids))} of "
                            f"{len(want_ids.get(a['task'], ()))} cases")
    if invalid:
        problems.append(f"{invalid} invalid rows")
    names = {a["arm"] for a in arms}
    conds.append(_cond("every arm instance produced exactly its slice cases",
                       bool(arms) and not problems, {"problems": problems[:20]},
                       f"every registered instance; the first {cases} test cases per task, "
                       "once each"))
    # A decision row serves every arm instance with the same rendered prompt (equal
    # prompts run once): per-arm statistics count it for each of them.
    uses: dict[tuple[str, str], set[tuple[str, Any]]] = {}
    for a in arms:
        uses.setdefault((a["task"], a["prompt_hash"]), set()).add((a["arm"], a.get("writer")))
    per: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for r in decisions:
        for key in uses.get((r["task"], r["prompt_hash"]), ()):
            per.setdefault(key, []).append(r)
    share: dict[str, dict[str, Any]] = {}
    for a in sorted(names):
        rs = [r for (arm, _), v in per.items() if arm == a for r in v]
        n = len(rs)
        cl = sum(1 for r in rs if r.get("error_class") == "context_limit")
        bound = CONTEXT_LIMIT_SHARE["full_context" if a == "full_context" else "other"]
        share[a] = {"rows": n, "context_limit": cl, "share": round(cl / n, 4) if n else None,
                    "bound": bound}
    conds.append(_cond("context-limit rows within the per-arm share",
                       all(v["share"] is not None and v["share"] <= v["bound"]
                           for v in share.values()) and bool(share), share,
                       CONTEXT_LIMIT_SHARE))
    pairs = {}
    for w in plan["writers"]:
        counts, rates = [], []
        for arm in ("learned", "search_control"):
            rs = per.get((arm, w), [])
            n_err = sum(1 for r in rs if r.get("error_class"))
            counts.append(n_err if rs else None)
            rates.append(n_err / len(rs) if rs else None)
        pairs[w] = {"learned_errors": counts[0], "search_control_errors": counts[1],
                    "case_difference": None if None in counts else abs(counts[0] - counts[1]),
                    "point_difference": None if None in rates
                    else round(abs(rates[0] - rates[1]), 4)}
    if PAIRED_MODE == "cases":
        ok = all(p["case_difference"] is not None and
                 p["case_difference"] <= PAIRED_MAX_CASE_DIFF for p in pairs.values())
        bound: Any = {"max_error_case_difference": PAIRED_MAX_CASE_DIFF}
    else:
        ok = all(p["point_difference"] is not None and
                 p["point_difference"] <= PAIRED_ERROR_DIFF for p in pairs.values())
        bound = {"max_error_rate_difference": PAIRED_ERROR_DIFF}
    conds.append(_cond("paired arms: no catastrophic error-rate difference" if
                       PAIRED_MODE == "cases" else
                       "paired-arm error-rate difference within bound", ok, pairs, bound))
    # the estimate, with the DERIVED settings (slower settings scale the measured
    # throughput down; faster ones are not credited), within ESTIMATE_MARGIN x the cap
    measure = _json(run / "measure.json") or {}
    startup = sum((starts[n]["seconds"] or 0) for n in starts)
    estimate = est.measured_estimate(plan, run / "measure", float(measure.get("wall_s") or 0),
                                     slice_dir, startup_s=startup) \
        if measure.get("wall_s") else {"estimate": "no measurement"}
    ep = _json(run / "endpoint.yaml") or {}
    derived = derive(run, decisions, ep)
    adjusted = adjust(estimate, derived, measured_units=int(measure.get("units") or 12),
                      measured_concurrency=int(ep.get("concurrency") or 32), startup_s=startup)
    derived["TIME_LIMIT_H"] = time_limit(adjusted)
    gpu_hi = (adjusted.get("gpu_h") or [None, None])[1]
    conds.append(_cond("measured phase-1 estimate (derived settings) within the cap margin",
                       gpu_hi is not None and gpu_hi <= ESTIMATE_MARGIN * GPU_HOURS_CAP,
                       gpu_hi, ESTIMATE_MARGIN * GPU_HOURS_CAP))
    # serving settings
    from slh.xp.partner import context

    args = ep.get("serving_args") or []
    teacher_args = (ep.get("execution") or {}).get("teacher_argv") or []

    def flag(argv: list[str], name: str, default: Any = None) -> Any:
        return argv[argv.index(name) + 1] if name in argv else default

    agent_tp = int(flag(args, "--tensor-parallel-size", 1))
    agent_dp = int(flag(args, "--data-parallel-size", 1))
    util = float(flag(args, "--gpu-memory-utilization", 0)) + \
        float(flag(teacher_args, "--gpu-memory-utilization", 0) if teacher_args else 0)
    serving = {"max_model_len": context.max_model_len(ep), "agent_tp_x_dp": agent_tp * agent_dp,
               "teacher_tp": int(flag(teacher_args, "--tensor-parallel-size", 0) or 0),
               "memory_share": round(util, 3), "concurrency": ep.get("concurrency"),
               "model_revision": ep.get("model_revision"),
               "vllm_version": ep.get("vllm_version"),
               "teacher_extra_args": (ep.get("execution") or {}).get("teacher_extra_args")}
    extras = serving["teacher_extra_args"]
    extras_ok = not teacher_args or (isinstance(extras, list)
                                     and all(teacher_extra_ok(str(t)) for t in extras))
    ok = (serving["max_model_len"] == MAX_MODEL_LEN and serving["agent_tp_x_dp"] <= 8
          and (not teacher_needed or serving["teacher_tp"] in (1, 2, 4, 8))
          and util <= UTIL_SUM_MAX
          and CONCURRENCY_BOUNDS[0] <= int(ep.get("concurrency") or 0) <= CONCURRENCY_BOUNDS[1]
          and ep.get("model_revision") == serve_spec.MODEL_REVISION
          and ep.get("vllm_version") == serve_spec.VLLM_VERSION and extras_ok)
    conds.append(_cond("serving settings within registered bounds", ok, serving, {
        "max_model_len": MAX_MODEL_LEN, "agent_tp_x_dp": "<= 8",
        "memory_share": f"<= {UTIL_SUM_MAX}", "concurrency": CONCURRENCY_BOUNDS,
        "model_revision": serve_spec.MODEL_REVISION, "vllm_version": serve_spec.VLLM_VERSION,
        "teacher_extra_args": f"each one of {list(TEACHER_EXTRA_ALLOWED)}"}))
    valid, why = _export_valid(slice_dir / "export", package, plan)
    conds.append(_cond("slice export valid against the schema", valid, why, "valid"))
    decision = "GO" if all(c["ok"] for c in conds) else "NO-GO"
    body = {"schema": "xp-gate-v1", "rule": RULE, "plan_id": plan["plan_id"],
            "plan_sha256": plan["sha256"],
            "phase0": {"job": record.get("job"), "archive_sha256": record.get("archive_sha256")},
            "conditions": conds, "decision": decision,
            "reasons": [c["name"] for c in conds if not c["ok"]],
            "derived": derived, "estimate": estimate, "estimate_derived": adjusted,
            "serving_bound": serving_bound(ep)}
    return {**body, "sha256": pc.sha_text(pc.canonical(body))}


def derive(run: Path, decisions: list[dict[str, Any]], endpoint: dict[str, Any]
           ) -> dict[str, Any]:
    """The registered derivation rule (deterministic, clamped to registered bounds);
    the time limit follows from the estimate under these settings (``time_limit``)."""
    mode = (_json(run / "serve" / "teacher-mode.json") or {}).get("mode")
    teacher_eager = 0 if mode == "graphs" else 1
    rows = len(decisions)
    limits = sum(1 for r in decisions if r.get("case_time_limit")
                 or r.get("error_class") in TIME_LIMIT_CLASSES)
    conc = int(endpoint.get("concurrency") or 32)
    concurrency = int(_clamp(conc if not rows or limits / rows <= 0.01 else conc // 2,
                             *CONCURRENCY_BOUNDS))
    errors = 0
    for s in Path(run, "measure", "learning").glob("*/*/run-*/summary.json"):
        errors += sum((_json(s) or {}).get("case_errors", {}).values())
    units = int(_clamp(12 if errors == 0 else 8, *UNITS_BOUNDS))
    return {"CONCURRENCY": concurrency, "LEARNING_UNITS": units,
            "TEACHER_EAGER": teacher_eager, "TIME_LIMIT_H": TIME_LIMIT_BOUNDS_H[1],
            "GPU_HOURS_CAP": GPU_HOURS_CAP,
            "basis": {"teacher_mode": mode, "slice_rows": rows, "slice_time_limits": limits,
                      "measure_case_errors": errors}}


def adjust(estimate: dict[str, Any], derived: dict[str, Any], *, measured_units: int,
           measured_concurrency: int, startup_s: float) -> dict[str, Any]:
    """The measured estimate under the derived settings: learning slower by
    measured/derived effective pool (min(units, concurrency): the runner bounds the
    learning pool by the endpoint's concurrency), the test stage by measured/derived concurrency (only ever
    slower: a larger setting is not credited)."""
    if not estimate.get("learning_h") or estimate["learning_h"][0] is None or \
            not (estimate.get("test") or {}).get("hours"):
        return {**estimate, "adjusted": False}
    # the learning pool actually running: units bounded by the endpoint's concurrency
    measured_pool = max(1, min(measured_units, measured_concurrency))
    derived_pool = max(1, min(derived["LEARNING_UNITS"], derived["CONCURRENCY"]))
    lf = max(1.0, measured_pool / derived_pool)
    tf = max(1.0, measured_concurrency / max(1, derived["CONCURRENCY"]))
    learn = [round(x * lf, 1) for x in estimate["learning_h"]]
    test_h = round(estimate["test"]["hours"] * tf, 1)
    wall = [learn[0] + test_h, learn[1] + test_h]
    jobs = [max(1, math.ceil(w / 47)) for w in wall]
    gpu = [round(8 * (w + j * startup_s / 3600)) for w, j in zip(wall, jobs, strict=True)]
    return {"adjusted": True, "learning_factor": lf, "test_factor": tf,
            "pool": {"measured": measured_pool, "derived": derived_pool},
            "learning_h": learn, "test_h": test_h, "wall_h": [round(w, 1) for w in wall],
            "jobs": jobs, "gpu_h": gpu}


def time_limit(adjusted: dict[str, Any]) -> int:
    wall = (adjusted.get("wall_h") or [None, None])[1]
    jobs = (adjusted.get("jobs") or [1, 1])[1]
    if wall is None:
        return TIME_LIMIT_BOUNDS_H[1]
    return int(_clamp(math.ceil(wall / max(1, jobs)) + 1, *TIME_LIMIT_BOUNDS_H))


def instance_problems(plan: dict[str, Any], arms: list[dict[str, Any]],
                      texts: dict[tuple[Any, ...], str]) -> list[str]:
    """Exact registered instance identity: every slot of ``plan`` exactly once, none
    other; kind and n as registered; the prompt hash (and skill hash) those of the
    registered text (writer arms: the learning stage's text, ``texts[(task, run,
    writer, kind)]``; notes: the registered notes; b0 / best_of_n: no text)."""
    from slh.xp.partner import plan as pp

    want = {pp.slot_key(s_): s_ for s_ in pp.instance_slots(plan)}
    plan_arms = {a["name"]: a for a in plan["arms"]}
    seen: set[tuple[Any, ...]] = set()
    problems = []
    for a in arms:
        try:
            k = pp.slot_key(a)
        except KeyError as exc:
            problems.append(f"arm row without {exc}")
            continue
        if k in seen:
            problems.append(f"duplicate instance {k}")
            continue
        seen.add(k)
        spec = want.get(k)
        if spec is None:
            problems.append(f"unregistered instance {k}")
            continue
        if a.get("kind") != spec["kind"] or a.get("n") != spec["n"]:
            problems.append(f"instance {k}: kind/n {a.get('kind')!r}/{a.get('n')!r} "
                            f"!= registered {spec['kind']!r}/{spec['n']!r}")
            continue
        if spec["kind"] in pp.WRITER_TEXT_KINDS:
            text = texts.get((a["task"], a["run"], a["writer"], spec["kind"]))
            if text is None:
                problems.append(f"instance {k}: no learning-stage text")
                continue
        else:
            text = pp.arm_text(plan_arms[a["arm"]], a["task"]) if spec["kind"] == "notes" \
                else ""
        if a.get("prompt_hash") != pp.prompt_hash(spec["kind"], text, spec["n"]) or \
                a.get("skill_sha256") != (pc.sha_text(text) if text else None):
            problems.append(f"instance {k}: prompt hash is not that of its registered text")
    for k in sorted(set(want) - seen, key=str):
        problems.append(f"missing instance {k}")
    return problems


def _run_texts(out: Path, plan: dict[str, Any]) -> dict[tuple[Any, ...], str]:
    """The learning stage's texts in a run folder (runner layout)."""
    from slh.xp.partner import plan as pp

    texts = {}
    for task, run, writer in pp.learning_slots(plan):
        for kind, fname in pp.TEXT_FILES.items():
            path = out / "learning" / task / writer / f"run-{run}" / fname
            if path.is_file():
                texts[(task, run, writer, kind)] = path.read_text(encoding="utf-8")
    return texts


SERVING_EXECUTION_KEYS = ("serve_mode", "image", "vllm_bin", "vllm_bin_sha256", "pyxis_flags",
                          "serve_env", "teacher_argv", "teacher_extra_args")
SERVING_TOP_KEYS = ("serving_args", "replicas", "model_revision", "vllm_version")


def serving_bound(ep: dict[str, Any]) -> dict[str, Any]:
    """Every serving setting phase 0 ran with (its endpoint file, in the archive):
    the 9B's argv, the teacher's argv and extra flags as given, the launch and the
    serving environment. Phase 1 must pass exactly these (``serving_problems``)."""
    ex = ep.get("execution") or {}
    return {**{k: ep.get(k) for k in SERVING_TOP_KEYS},
            **{k: ex.get(k) for k in SERVING_EXECUTION_KEYS}}


def serving_problems(gate: dict[str, Any], endpoint: dict[str, Any]) -> list[str]:
    """Phase 1's resolved endpoint (before any server starts) against the gate: every
    serving setting equal to the bound phase-0 value, the teacher's extra flags within
    the allowlist (never --enforce-eager: eager mode is TEACHER_EAGER's alone), the
    teacher's eager flag as derived, and the derived concurrency."""
    problems = []
    bound = gate.get("serving_bound")
    if not isinstance(bound, dict):
        return ["the gate records no serving settings"]
    derived = gate.get("derived") or {}
    now = serving_bound(endpoint)
    for key in (*SERVING_TOP_KEYS, *SERVING_EXECUTION_KEYS):
        if now.get(key) != bound.get(key):
            problems.append(f"{key} differs from phase 0's (gate-bound) value")
    extras = now.get("teacher_extra_args") or []
    if "--enforce-eager" in extras:
        problems.append("--enforce-eager among the teacher's extra flags (eager mode is "
                        "set by TEACHER_EAGER only)")
    bad = [a for a in extras if not teacher_extra_ok(str(a))]
    if bad:
        problems.append(f"teacher extra flags outside the registered allowlist: {bad}")
    argv = now.get("teacher_argv") or []
    if argv and ("--enforce-eager" in argv) != (int(derived.get("TEACHER_EAGER", -1)) == 1):
        problems.append("the teacher's eager mode differs from the derived TEACHER_EAGER="
                        f"{derived.get('TEACHER_EAGER')}")
    if int(endpoint.get("concurrency") or -1) != int(derived.get("CONCURRENCY", -2)):
        problems.append(f"concurrency {endpoint.get('concurrency')} is not the derived "
                        f"{derived.get('CONCURRENCY')}")
    return problems


def verify(gate: dict[str, Any], record: dict[str, Any], package: Path,
           expect: dict[str, str] | None = None) -> list[str]:
    """Problems with a stored gate (empty = it binds): self-hash, GO, the archive's
    hash, a recomputation equal to the stored gate, and the job's settings."""
    problems = []
    body = {k: v for k, v in gate.items() if k != "sha256"}
    if pc.sha_text(pc.canonical(body)) != gate.get("sha256"):
        problems.append("gate.json does not hash to its own sha256 (edited)")
    if gate.get("decision") != "GO":
        problems.append(f"gate decision is {gate.get('decision')!r}")
    sha = _sha_file(Path(record.get("archive") or ""))
    if not sha or sha != record.get("archive_sha256") or \
            (gate.get("phase0") or {}).get("archive_sha256") != sha:
        problems.append("the gate is not bound to this phase-0 archive")
    try:
        fresh = evaluate(record, package)
    except ValueError as exc:
        return [*problems, f"cannot recompute from the bound archive: {exc}"]
    if fresh["sha256"] != gate.get("sha256"):
        problems.append("recomputing the gate from the phase-0 evidence gives another gate")
    derived = gate.get("derived") or {}
    for key, value in (expect or {}).items():
        if key == "JOB_LIMIT_S":  # the job's actual Slurm limit within the derived one
            try:
                limit = int(value)
            except (TypeError, ValueError):
                limit = -1
            if limit <= 0:  # fail closed: UNLIMITED, unreadable or scontrol failed
                problems.append("the job's Slurm time limit is unknown or UNLIMITED "
                                f"({value!r}); submit with --time within the derived "
                                f"{derived.get('TIME_LIMIT_H')} h")
            elif limit > int(derived.get("TIME_LIMIT_H", 0)) * 3600 + 60:
                problems.append(f"the job's time limit {limit} s exceeds the derived "
                                f"{derived.get('TIME_LIMIT_H')} h")
        elif str(derived.get(key)) != str(value):
            problems.append(f"{key}={value} is not the derived {derived.get(key)!r}")
    return problems


def completion_problems(export: Path, package: Path) -> list[str]:
    """Phase 1 counts as complete only for an export of the FULL registered plan,
    checked by the same key-free rules as our checker (``check.learning_identity``,
    ``check.arm_identity``, ``check.decision_identity``): schema and manifest valid, a
    real run with no smoke setting, every learning run summarised and verified, every
    registered arm instance exactly as registered, every decision row a registered
    test case with its registered case hash on a registered instance's prompt, once,
    and every instance on every test case of its task. (No answer keys: scoring stays
    with us.)"""
    from slh.xp.partner import check
    from slh.xp.partner import plan as pp

    plan = pp.load(package / "plan.json")
    try:
        shard = check.read_shard(export, pc.read_json(package / "registration.json"), plan)
    except (check.CheckFailed, ValueError, KeyError) as exc:
        return [f"export invalid: {type(exc).__name__}: {str(exc)[:200]}"]
    ident = shard["hashes"]["run_identity"]
    if ident.get("smoke") is not None or ident.get("mode") != "real":
        return ["not a real run of the full plan (smoke or dry run)"]
    rows = shard["rows"]
    tasks = set(plan["tasks"])
    case_hashes = {t: {q["case_id"]: q["case_hash"] for q in pc.read_jsonl(
        package / plan["cases"][t]["test"]["file"])} for t in plan["tasks"]}
    problems: list[str] = []
    try:
        texts, gaps = check.learning_identity(
            plan, rows.get("skills.jsonl.gz", []), rows.get("learning.jsonl.gz", []), tasks,
            package, check.verified_endpoint([shard]))
        slots, more = check.arm_identity(plan, rows.get("arms.jsonl.gz", []), texts, tasks)
        gaps += more
        by_prompt: dict[tuple[str, str], set[tuple[Any, ...]]] = {}
        for k, a in slots.items():
            by_prompt.setdefault((a["task"], a["prompt_hash"]), set()).add(k)
        seen: set[tuple[str, str, str]] = set()
        for r in rows.get("decisions.jsonl.gz", []):
            check.decision_identity(r, case_hashes, by_prompt, seen)
            check.verify_failure_row(r, f"{r['task']}/{r['case_id']}")
    except (check.CheckFailed, ValueError, KeyError, TypeError) as exc:
        return [f"identity: {type(exc).__name__}: {str(exc)[:300]}"]
    problems += [f"{g['problem']}: {g.get('slot') or g.get('task')}" for g in gaps][:20]
    covered: dict[tuple[str, str], set[str]] = {}
    for t, ph, cid in seen:
        covered.setdefault((t, ph), set()).add(cid)
    short = 0
    for a in slots.values():
        if covered.get((a["task"], a["prompt_hash"]), set()) != set(case_hashes[a["task"]]):
            short += 1
    if short:
        problems.append(f"{short} arm instances without exactly every test case")
    return problems


def stream_problem(archive: Path) -> str | None:
    """The whole gzip stream read to EOF (CRC and footer verified by gzip) and the tar
    stream walked block by block to its end-of-archive marker (two zero blocks). A
    collection killed before its footer, cut bytes or a missing marker: a problem."""
    import gzip
    import tarfile
    import zlib

    zero = b"\0" * 512

    def skip(f: Any, n: int) -> bool:
        while n > 0:
            got = len(f.read(min(n, 1 << 20)))
            if not got:
                return False
            n -= got
        return True

    try:
        with gzip.open(archive, "rb") as f:
            while True:
                head = f.read(512)
                if len(head) < 512:
                    return "tar stream ends without its end-of-archive marker"
                if head == zero:
                    if f.read(512) != zero:
                        return "tar end-of-archive marker incomplete"
                    while f.read(1 << 20):  # to EOF: gzip checks CRC and length
                        pass
                    return None
                info = tarfile.TarInfo.frombuf(head, "utf-8", "surrogateescape")
                if not skip(f, -(-info.size // 512) * 512):
                    return f"tar member {info.name!r} cut short"
    except (OSError, EOFError, zlib.error, tarfile.TarError) as exc:
        return f"export archive stream invalid: {type(exc).__name__}: {exc}"


def archive_problems(archive: Path, package: Path) -> tuple[list[str], str | None]:
    """The export ARCHIVE itself: its gzip and tar streams whole (``stream_problem``),
    then unpacked into a fresh folder: complete (no TRUNCATED.txt, readable), one export, valid for the full registered plan. Returns
    the problems and the archive's SHA-256 as validated."""
    import tarfile

    if not archive.is_file():
        return ["no export archive"], None
    sha = _sha_file(archive)
    bad = stream_problem(archive)
    if bad:
        return [bad], sha
    with tempfile.TemporaryDirectory(prefix="xp-complete-") as tmp:
        try:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(tmp, filter="data")
        except (tarfile.TarError, OSError, EOFError) as exc:
            return [f"export archive unreadable: {type(exc).__name__}: {exc}"], sha
        if any(Path(tmp).rglob("TRUNCATED.txt")):
            return ["export archive truncated (TRUNCATED.txt)"], sha
        manifests = sorted(Path(tmp).rglob("manifest.json"))
        if len(manifests) != 1:
            return [f"export archive holds {len(manifests)} export manifests"], sha
        problems = completion_problems(manifests[0].parent, package)
    if _sha_file(archive) != sha:
        problems.append("the export archive changed while it was validated")
    return problems, sha


def _complete(archive: Path, dest: Path, package: Path, out: Path) -> int:
    problems, sha = archive_problems(archive, package)
    if problems:
        print(json.dumps({"complete": False, "problems": problems}))
        return 2
    archive.replace(dest)  # the validated bytes (same file system), then the marker
    if _sha_file(dest) != sha:
        print(json.dumps({"complete": False, "problems": ["archive changed on move"]}))
        return 2
    marker = {"schema": "xp-phase1-complete-v1", "archive": str(dest),
              "archive_sha256": sha}
    tmp = Path(str(out) + ".tmp")
    tmp.write_text(json.dumps(marker, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)  # written last, atomically
    print(json.dumps({"complete": True, **marker}))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="slh.xp.partner.gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("complete")
    p.add_argument("--archive", type=Path, required=True,
                   help="the freshly collected export archive (validated as unpacked)")
    p.add_argument("--dest", type=Path, required=True,
                   help="where the validated archive is moved before the marker")
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("serving")
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--endpoint", type=Path, required=True)
    p.add_argument("--package", type=Path)  # passed by the bootstrap; unused
    for name in ("evaluate", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--record", type=Path, required=True)
        p.add_argument("--package", type=Path, required=True)
        p.add_argument("--gate", type=Path)
        p.add_argument("--out", type=Path)
        p.add_argument("--expect", default="")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if args.cmd == "complete":
        return _complete(args.archive, args.dest, args.package, args.out)
    if args.cmd == "serving":
        gate, ep = _json(args.gate), _json(args.endpoint)
        if not gate or not ep:
            print(json.dumps({"status": "refused", "reason": "no gate.json or endpoint"}))
            return 2
        body = {k: v for k, v in gate.items() if k != "sha256"}
        problems = [] if pc.sha_text(pc.canonical(body)) == gate.get("sha256") else \
            ["gate.json does not hash to its own sha256 (edited)"]
        problems += serving_problems(gate, ep)
        print(json.dumps({"status": "refused" if problems else "ok", "problems": problems}))
        return 2 if problems else 0
    record = _json(args.record)
    if not record:
        print(json.dumps({"status": "refused", "reason": "no phase-0 record"}))
        return 2
    if args.cmd == "evaluate":
        gate = evaluate(record, args.package)
        out = args.out or args.record.with_name("gate.json")
        tmp = Path(str(out) + ".tmp")
        tmp.write_text(json.dumps(gate, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(out)
        print(json.dumps({"decision": gate["decision"], "reasons": gate["reasons"],
                          "derived": {k: v for k, v in gate["derived"].items()
                                      if k != "basis"}, "gate": str(out)}))
        return 0 if gate["decision"] == "GO" else 2
    gate = _json(args.gate) if args.gate else None
    if not gate:
        print(json.dumps({"status": "refused", "reason": "no gate.json"}))
        return 2
    expect = dict(kv.split("=", 1) for kv in args.expect.split(",") if "=" in kv)
    problems = verify(gate, record, args.package, expect)
    print(json.dumps({"status": "ok" if not problems else "refused", "problems": problems,
                      "gate_sha256": gate.get("sha256")}))
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
