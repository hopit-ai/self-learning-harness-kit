"""Phase-1 estimates from phase-0 measurements (partner side: the gate; our side: the
reader). Standard library plus the plan module; no answer keys, no scores.

- ``estimate``: from a short smoke (phase 0 v1): per-unit learning walls scaled to the
  full loop and the plan's runs, divided by an assumed parallel speed-up.
- ``measured_estimate``: from phase 0 v2: learning extrapolated from the MEASURED
  progress of the plan's own learning units run in parallel for a capped wall time
  (``measure`` setting), the test stage from the MEASURED decision rate of the slice
  (every arm, 20 cases per task).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from slh.xp.partner import common as pc

GPUS = 8
MAX_JOB_H = 48
# Reference teacher speeds (XD1 teacher, partner's node, phase 0 r4/r5), reported with
# every estimate for readers; the estimate itself uses only this run's measurements.
TEACHER_REFERENCE = {
    "tok_s_per_stream": {"eager": 5.8, "graphs": 150.0},
    "realistic_call_s": {"eager": 183.0, "graphs": 9.5},
    "learning_stage": {"graphs": "7 of 9 units in 55 min (r5)",
                       "eager": "9 units in about 8 h (r4)"},
    "startup_s": {"graphs": 390},
}


def _json(path: Path) -> Any:
    try:
        text = Path(path).read_text(encoding="utf-8")
        return json.loads(text[text.index("{"):] if not text.lstrip().startswith(("{", "["))
                          else text)
    except (OSError, ValueError):
        return None


def workload(plan: dict[str, Any] | None, writers: tuple[str, ...] | None = None
             ) -> dict[str, Any] | None:
    """Test-stage rows and calls from the plan (best-of-n: n candidates + n-1 selector
    calls per row). Equal prompts across arms run once; that saving is not counted
    (conservative)."""
    if plan is None:
        return None
    from slh.xp.partner import plan as pp

    plan = dict(plan)
    if writers is not None:
        plan["writers"] = [w for w in plan["writers"] if w in writers]
    rows = calls = 0
    for slot in pp.instance_slots(plan):
        n = plan["cases"][slot["task"]]["test"]["rows"]
        rows += n
        calls += n * ((2 * (slot.get("n") or 1) - 1) if slot["kind"] == "best_of_n" else 1)
    return {"test_decision_rows": rows, "test_calls": calls}


def learning_units(run: Path) -> list[dict[str, Any]]:
    """Per learning unit of the smoke: wall seconds (logs/stages.jsonl), loop, steps."""
    walls = {}
    for r in pc.read_jsonl(run / "logs" / "stages.jsonl") if \
            (run / "logs" / "stages.jsonl").is_file() else []:
        if r.get("stage") == "learning" and "wall_s" in r:
            walls[(r["task"], r["writer"], int(r.get("run", 0)))] = float(r["wall_s"])
    out = []
    for (task, writer, k), wall in sorted(walls.items()):
        s = _json(run / "learning" / task / writer / f"run-{k}" / "summary.json") or {}
        out.append({"task": task, "writer": writer, "run": k, "wall_s": wall,
                    "loop": s.get("loop") or ("ops" if writer.endswith("+") else "note"),
                    "steps": max(1, int(s.get("proposals") or 1)),
                    "search_steps": max(1, int((s.get("search_control") or {})
                                                .get("proposals") or 1)),
                    "stop_reason": s.get("stop_reason")})
    return out


# Parallel learning: units run LEARNING_UNITS (12) at a time; the teacher and the 9B
# batch their requests. Measured: the teacher at 4 concurrent requests without errors;
# its knee is not measured. Conservative assumption: throughput scales up to 4x and no
# further; optimistic: 8x.
SPEEDUP = {"conservative": 4, "optimistic": 8}


def estimate(units: list[dict[str, Any]], plan: dict[str, Any] | None, *,
             test_rate: float | None, startup_s: float, writers: tuple[str, ...] | None = None
             ) -> dict[str, Any]:
    """Learning and test stages estimated separately (the smoke's wall time includes
    its whole learning stage and must not be divided by its 15 decisions).

    Learning: each smoke unit's wall time, scaled to the full loop and to the plan's
    runs. Ops writers (T+, S+) ran the smoke loop (``SMOKE`` ops: max_steps 2,
    patience 1); the full loop runs between ``patience`` (no accepted edit) and
    ``max_steps`` steps, so a unit scales by [patience, max_steps] / smoke steps (the
    whole unit, acquisition included: conservative). Note-loop writers (T) process the
    whole training stream in the smoke too: factor 1.
    Test: the plan's calls at the 9B's measured rate on real decision prompts."""
    from slh.xp.learn.loop import LoopConfig
    from slh.xp.partner import plan as pp

    if plan is None or not units:
        return {"estimate": "no plan or no learning units in the archive"}
    ops = {**LoopConfig().to_dict(), **((plan.get("learning") or {}).get("ops") or {})}
    full_lo, full_hi = int(ops["patience"]), int(ops["max_steps"])
    runs = int(plan["runs"])
    keep = set(writers or plan["writers"])
    per_writer: dict[str, dict[str, float]] = {}
    for u in units:
        if u["writer"] not in keep:
            continue
        if u["loop"] == "ops":
            smoke_steps = max(u["steps"], u["search_steps"])
            lo, hi = u["wall_s"] * full_lo / smoke_steps, u["wall_s"] * full_hi / smoke_steps
        else:
            lo = hi = u["wall_s"]
        w = per_writer.setdefault(u["writer"], {"units": 0, "lo_h": 0.0, "hi_h": 0.0})
        w["units"] += runs  # one smoke unit per task stands for the plan's runs
        w["lo_h"] += runs * lo / 3600
        w["hi_h"] += runs * hi / 3600
    seq_lo = sum(w["lo_h"] for w in per_writer.values())
    seq_hi = sum(w["hi_h"] for w in per_writer.values())
    work = workload(plan, tuple(keep))
    test_h = work["test_calls"] / test_rate / 3600 if test_rate else None
    out: dict[str, Any] = {
        "writers": sorted(keep), "full_loop_steps": [full_lo, full_hi],
        "smoke_ops": dict(pp.SMOKE.get("ops") or {}),
        "learning_sequential_h": [round(seq_lo, 1), round(seq_hi, 1)],
        "per_writer": {k: {"units": v["units"], "sequential_h": [round(v["lo_h"], 1),
                                                                round(v["hi_h"], 1)]}
                       for k, v in sorted(per_writer.items())},
        "test": {**(work or {}), "rate_calls_per_s": round(test_rate, 2) if test_rate
                 else None, "hours": round(test_h, 1) if test_h else None}}
    for name, speed in SPEEDUP.items():
        lo_h, hi_h = seq_lo / speed, seq_hi / speed
        wall = [lo_h + (test_h or 0), hi_h + (test_h or 0)]
        jobs = [max(1, math.ceil(w / (MAX_JOB_H - 1))) for w in wall]
        gpu = [GPUS * (w + j * startup_s / 3600) for w, j in zip(wall, jobs, strict=True)]
        out[name] = {"speedup": speed, "learning_h": [round(lo_h, 1), round(hi_h, 1)],
                     "wall_h": [round(x, 1) for x in wall], "jobs": jobs,
                     "gpu_h": [round(x) for x in gpu]}
    return out


def test_rate(job: Path, concurrency: int) -> tuple[float | None, str]:
    """The 9B's rate on REAL decision prompts: in-flight requests / mean latency of
    the smoke's test decisions (its prompts are the real ones); else the probe."""
    run = job / "smoke"
    rows = pc.read_jsonl(run / "decisions.jsonl") if (run / "decisions.jsonl").is_file() \
        else []
    walls = sorted(float(r["wall_s"]) / max(1, int(r.get("model_calls") or 1))
                   for r in rows if r.get("wall_s") and not r.get("error"))
    if walls:
        p75 = walls[min(len(walls) - 1, int(0.75 * len(walls)))]
        return concurrency / p75, (f"{concurrency} in flight / p75 latency {p75:.2f} s of "
                                   f"{len(walls)} real test decisions")
    return None, "no real test decisions in the archive"




# --------------------------------------------------------------------------- phase 0 v2


def unit_progress(unit: Path, writer: str, *, full_steps: tuple[int, int],
                  note_patches: int) -> tuple[float, float]:
    """(optimistic, conservative) fraction of a learning unit's full work done: 1 when
    complete; else ops steps done over 2 x [patience, max_steps] (main loop and search
    control), or note patches done over the main and search loops' patches."""
    if (unit / "complete.json").is_file():
        return 1.0, 1.0
    if writer.endswith("+"):
        done = sum(1 for f in ("trace.jsonl", "search/trace.jsonl")
                   for r in pc.read_jsonl(unit / f) if r.get("type") == "step")
        return min(1.0, done / (2 * full_steps[0])), min(1.0, done / (2 * full_steps[1]))
    done = len(pc.read_jsonl(unit / "patches.jsonl")) + \
        len(pc.read_jsonl(unit / "search_patches.jsonl"))
    frac = min(1.0, done / max(1, 2 * note_patches))
    return frac, frac


def metrics_summary(path: Path) -> dict[str, Any] | None:
    """Throughput from a vLLM metrics sample file (xpsite.py sample-metrics): tokens and
    requests per second, mean running requests, prefix-cache hit rate."""
    rows = pc.read_jsonl(path) if Path(path).is_file() else []
    rows = [r for r in rows if isinstance(r.get("t"), int | float)]
    if len(rows) < 2:
        return None
    a, b = rows[0], rows[-1]
    dt = max(1e-9, b["t"] - a["t"])

    def rate(key: str) -> float | None:
        if a.get(key) is None or b.get(key) is None:
            return None
        return round((b[key] - a[key]) / dt, 3)

    running = [r["running"] for r in rows if r.get("running") is not None]
    hits = (b.get("prefix_hits") or 0) - (a.get("prefix_hits") or 0)
    queries = (b.get("prefix_queries") or 0) - (a.get("prefix_queries") or 0)
    return {"seconds": round(dt, 1), "prompt_tokens_per_s": rate("prompt_tokens"),
            "generation_tokens_per_s": rate("generation_tokens"),
            "requests_per_s": rate("requests"),
            "mean_running": round(sum(running) / len(running), 2) if running else None,
            "max_running": max(running) if running else None,
            "prefix_cache_hit_rate": round(hits / queries, 4) if queries else None}


def measured_estimate(plan: dict[str, Any], measure: Path, measure_wall_s: float,
                      slice_run: Path, *, startup_s: float,
                      writers: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Phase-1 estimate from phase-0 v2 measurements (no assumed speed-up).

    Learning: the ``measure`` run ran the plan's first ``runs`` learning runs (every
    task x writer) on the parallel pool for ``measure_wall_s``; each unit's progress is
    its fraction of full work (``unit_progress``); the full plan has ``plan runs /
    measure runs`` times as many units of the same mix, so its learning wall is
    ``wall x units_full / progress``. Optimistic and conservative bounds come from the
    ops loop's [patience, max_steps]. The tail (fewer units left than pool slots) is not
    modelled; the conservative bound covers it only partly.
    Test: the slice's measured decision rate (rows per second at the run's
    concurrency) and calls per row, applied to the plan's test rows."""
    from slh.xp.learn.loop import LoopConfig
    from slh.xp.partner import plan as pp

    keep = tuple(writers or plan["writers"])
    ops = {**LoopConfig().to_dict(), **((plan.get("learning") or {}).get("ops") or {})}
    full_steps = (int(ops["patience"]), int(ops["max_steps"]))
    sample = len(plan["learning"]["train_samples"][plan["tasks"][0]]["runs"]["0"])
    note_patches = math.ceil(sample / int((plan.get("learning") or {}).get("note_batch", 4)))
    m_runs = int(pp.SMOKES["measure"]["runs"])
    units, prog_lo, prog_hi = 0, 0.0, 0.0
    for task in plan["tasks"]:
        for writer in keep:
            for run in range(min(m_runs, int(plan["runs"]))):
                d = Path(measure) / "learning" / task / writer / f"run-{run}"
                units += 1
                if d.is_dir():
                    hi, lo = unit_progress(d, writer, full_steps=full_steps,
                                           note_patches=note_patches)
                    prog_hi += hi
                    prog_lo += lo
    scale = int(plan["runs"]) / max(1, min(m_runs, int(plan["runs"])))
    full_units = units * scale
    learn = [None, None] if not prog_hi else [
        round(measure_wall_s * full_units / prog_hi / 3600, 1),
        round(measure_wall_s * full_units / max(prog_lo, 1e-9) / 3600, 1)]
    stages = pc.read_jsonl(Path(slice_run) / "logs" / "stages.jsonl")
    dec = [r for r in stages if r.get("stage") == "decisions" and r.get("written")]
    rows = pc.read_jsonl(Path(slice_run) / "decisions.jsonl")
    calls_per_row = (sum(int(r.get("model_calls") or 1) for r in rows) / len(rows)) \
        if rows else 1.0
    rate = None
    if dec:
        written = sum(r["written"] for r in dec)
        wall = sum(r["wall_s"] for r in dec)
        rate = written / wall if wall else None
    work = workload(plan, keep)
    test_h = round(work["test_decision_rows"] / rate / 3600, 1) if rate else None
    out: dict[str, Any] = {
        "basis": "measured", "writers": list(keep), "measure_wall_s": round(measure_wall_s),
        "units_measured": units, "progress": [round(prog_lo, 2), round(prog_hi, 2)],
        "full_units": full_units, "learning_h": learn,
        "test": {**work, "rows_per_s": round(rate, 3) if rate else None,
                 "calls_per_row": round(calls_per_row, 3), "hours": test_h}}
    if learn[0] is not None and test_h is not None:
        wall = [learn[0] + test_h, learn[1] + test_h]
        jobs = [max(1, math.ceil(w / (MAX_JOB_H - 1))) for w in wall]
        out["wall_h"] = [round(w, 1) for w in wall]
        out["jobs"] = jobs
        out["gpu_h"] = [round(GPUS * (w + j * startup_s / 3600))
                        for w, j in zip(wall, jobs, strict=True)]
    out["teacher_reference"] = TEACHER_REFERENCE  # context only, not used above
    return out
