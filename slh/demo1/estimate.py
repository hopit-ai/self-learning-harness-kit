"""Cost and time estimate for the demo-1 window (offline; reads recorded timing).

Tokens: the pinned agent tokenizer on the rendered B1 prompt, the tool schemas
and each case's messages (the chat template adds a small overhead that is not
counted; a 10% allowance covers it). Calls: the registered plan (amendment 8):
B1 on train, the writer (procedure and generic note, each with at most one
repair), B1 and the learned arm on calibration, and all arms on sealed.
Latency: a least-squares fit ``latency = a + b * output_tokens + c * input_tokens``
over every agent call of the B1 stream run on the same serving stack
(``runs/b1-stream/calls.jsonl``, 4 concurrent workers), applied to the planned
calls at the same concurrency. GPU price: the window reservation that ``serve
start`` makes (``slh.window.reservation_usd``); the agent costs $0 per token.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from slh.demo1.schema import REPO

TIMING_SOURCE = REPO / "runs/b1-stream/calls.jsonl"
WORKERS = 4
TEMPLATE_ALLOWANCE = 1.10
STARTUP_MIN = 10.0  # serve start to first request (registered timing audit)
DRAIN_MIN = 10.0    # stop and verified zero containers
EPISODE_MARGIN_MIN = 5
# Combined demo cap (raised from 9.50 after the window-B runaway generation,
# deviation D5): every booked demo start since the first counts against it.
CAP_USD = 14.21
DEMO_BASELINE_USD = 20.846515  # ledger commitment before the first demo start
# Booked by the ledger before the next window-B attempt: three failed window-A
# starts (3 x 1.32708), window A (0.398421), the stalled window B (3.029399)
# and the unscheduled window-B attempt 2 (3.2427, deviation D6).
BOOKED_BEFORE_RETRY_USD = 10.65176
# Registered window lengths: the conservative need plus room for think/read
# continuations and retries, which the call count leaves out.
WINDOWS_MIN = {"learn": 30, "test": 75}
DECISION_OUTPUT_TOKENS = 120   # median B1 agent completion (b1-stream)
WRITER_OUTPUT_TOKENS = 250
SELECTOR_OUTPUT_TOKENS = 5
CONSERVATIVE_FACTOR = 2.0


def prefetch_reservation_usd(prices: Path = REPO / "configs/prices.yaml") -> float:
    """Exactly what `serve prefetch` reserves (same plan code as slh.serving.prefetch)."""
    import time

    from slh.serving import PREFETCH_TIMEOUT_S, build_allowance_usd, load_rate, plan_window

    plan = plan_window(PREFETCH_TIMEOUT_S, load_rate(prices, "prefetch"), time.time(),
                       kind="prefetch", stop_margin_s=0, request_margin_s=0,
                       episode_margin_s=0, startup_timeout_s=0, idle_s=0,
                       fixed_charge_usd=build_allowance_usd(prices))
    return float(plan["reservation_usd"])


def latency_model(path: Path = TIMING_SOURCE) -> dict[str, Any]:
    import numpy as np

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    calls = [r for r in rows if r.get("role") == "agent" and r.get("usage")
             and r.get("latency_s") and r["usage"].get("completion_tokens") is not None]
    out_t = np.array([r["usage"]["completion_tokens"] for r in calls], dtype=float)
    in_t = np.array([r["usage"]["prompt_tokens"] for r in calls], dtype=float)
    lat = np.array([r["latency_s"] for r in calls], dtype=float)
    coef = np.linalg.lstsq(np.c_[np.ones(len(calls)), out_t, in_t], lat, rcond=None)[0]
    return {"source": str(path.relative_to(REPO)), "calls": len(calls),
            "a_s": round(float(coef[0]), 4), "b_s_per_output_token": round(float(coef[1]), 6),
            "c_s_per_input_token": round(float(coef[2]), 8),
            "median_latency_s": round(float(statistics.median(lat)), 3),
            "median_completion_tokens": float(statistics.median(out_t)),
            "median_prompt_tokens": float(statistics.median(in_t)), "workers": WORKERS}


def _count(text: str) -> int:
    from slh.learner.note_tokens import count_tokens

    return count_tokens(text)


def case_tokens(case: dict[str, Any]) -> int:
    parts = []
    for m in case["messages"]:
        parts.append(m.get("content") or "")
        for c in m.get("tool_calls") or []:
            parts.append(c["name"] + json.dumps(c["arguments"]))
    return sum(_count(p) for p in parts)


def estimate(cases: list[dict[str, Any]], context: dict[str, Any],
             model: dict[str, Any] | None = None,
             procedure_tokens: int = 200, full_context_tokens: int | None = None,
             b0_context: dict[str, Any] | None = None
             ) -> dict[str, Any]:
    from slh.window import reservation_usd

    model = model or latency_model()
    base = _count(context["system_prompt"]) + _count(json.dumps(context["tools"]))
    per_case = {c["case_id"]: case_tokens(c) for c in cases}
    by_split = {s: [c for c in cases if c["split"] == s] for s in ("train", "calibrate", "sealed")}
    mean = {s: statistics.mean(per_case[c["case_id"]] for c in v) for s, v in by_split.items()}
    if full_context_tokens is None:
        full_context_tokens = 450 * 12  # about 450 tokens per train case of evidence
    prompt = {s: (base + mean[s]) * TEMPLATE_ALLOWANCE for s in mean}

    def latency(inp: float, out: float) -> float:
        return model["a_s"] + model["b_s_per_output_token"] * out + \
            model["c_s_per_input_token"] * inp

    n = {s: len(v) for s, v in by_split.items()}
    stages: dict[str, dict[str, float]] = {}

    def add(stage: str, calls: float, inp: float, out: float) -> None:
        s = stages.setdefault(stage, {"calls": 0.0, "input_tokens": 0.0, "output_tokens": 0.0,
                                      "busy_s": 0.0})
        s["calls"] += calls
        s["input_tokens"] += calls * inp
        s["output_tokens"] += calls * out
        s["busy_s"] += calls * latency(inp, out)

    add("learn", n["train"], prompt["train"], DECISION_OUTPUT_TOKENS)
    writer_in = base + full_context_tokens + 300
    add("learn", 8, writer_in, WRITER_OUTPUT_TOKENS)
    add("calibrate", n["calibrate"], prompt["calibrate"], DECISION_OUTPUT_TOKENS)
    add("calibrate", n["calibrate"], prompt["calibrate"] + procedure_tokens,
        DECISION_OUTPUT_TOKENS)
    sp = prompt["sealed"]
    for extra in (0, procedure_tokens, full_context_tokens, procedure_tokens, procedure_tokens):
        add("sealed", n["sealed"], sp + extra, DECISION_OUTPUT_TOKENS)  # b1 learned fc hu gen
    add("sealed", n["sealed"], sp, DECISION_OUTPUT_TOKENS)  # best-of-2 sample
    add("sealed", n["sealed"], sp + 200, SELECTOR_OUTPUT_TOKENS)  # selector (upper bound)
    if b0_context is not None:  # B0 (descriptive): the reference prompt, no think tool
        b0_base = _count(b0_context["system_prompt"]) + _count(json.dumps(b0_context["tools"]))
        add("sealed", n["sealed"], (b0_base + mean["sealed"]) * TEMPLATE_ALLOWANCE,
            DECISION_OUTPUT_TOKENS)
    for s in stages.values():
        s["minutes"] = s["busy_s"] / WORKERS / 60.0
        s["minutes_conservative"] = s["minutes"] * CONSERVATIVE_FACTOR
    windows: dict[str, Any] = {}
    from slh.serving import load_rate

    # The reservation rate: the most expensive allowed GPU (GPU fallback,
    # docs/surface-migration-2026-10-09-gpu.md); $2.70296/h, was $2.55416/h.
    gpu_per_hour = load_rate(REPO / "configs/prices.yaml", "serve").per_hour
    for name, parts in (("learn", ("learn",)), ("test", ("calibrate", "sealed"))):
        expected = sum(stages[p]["minutes"] for p in parts)
        cons = sum(stages[p]["minutes_conservative"] for p in parts)
        need = STARTUP_MIN + cons + DRAIN_MIN
        window = WINDOWS_MIN[name] if need <= WINDOWS_MIN[name] else None
        windows[name] = {
            "model_min_expected": round(expected, 1), "model_min_conservative": round(cons, 1),
            "needed_min_conservative": round(need, 1), "window_min": window,
            "gpu_reservation_usd": None if window is None else round(
                reservation_usd(window, episode_margin_min=EPISODE_MARGIN_MIN), 4),
            "expected_gpu_usd": round((STARTUP_MIN + expected + DRAIN_MIN) / 60.0
                                      * gpu_per_hour, 2)}
    combined = sum(w["gpu_reservation_usd"] or float("inf") for w in windows.values())
    # The GPU fallback changes the serving-spec hash, so `serve start` needs a
    # fresh passed prefetch on the new hash first (CPU sandbox, booked too).
    prefetch = prefetch_reservation_usd()
    test_res = reservation_usd(WINDOWS_MIN["test"], episode_margin_min=EPISODE_MARGIN_MIN)
    retry_total = BOOKED_BEFORE_RETRY_USD + prefetch + test_res
    totals = {k: round(sum(s[k] for s in stages.values())) for k in
              ("calls", "input_tokens", "output_tokens")}
    return {
        "schema": "demo1-estimate-v1", "latency_model": model,
        "tokens": {"b1_prompt_and_tools": base,
                   "case_history_mean": {s: round(v) for s, v in mean.items()},
                   "case_history_max": max(per_case.values()),
                   "procedure_assumed": procedure_tokens,
                   "full_context_assumed": full_context_tokens,
                   "template_allowance": TEMPLATE_ALLOWANCE},
        "stages": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in stages.items()},
        "totals": totals,
        "startup_min": STARTUP_MIN, "drain_min": DRAIN_MIN,
        "windows": windows, "combined_gpu_reservation_usd": round(combined, 4),
        "token_cost_usd": 0.0, "cap_usd": CAP_USD,
        "booked_before_window_b_retry_usd": BOOKED_BEFORE_RETRY_USD,
        "prefetch_reservation_usd": round(prefetch, 6),
        "window_b_retry_total_usd": round(retry_total, 6),
        # The smallest combined cap (whole cents, rounded up) that admits the
        # prefetch and window B after everything already booked.
        "min_cap_needed_usd": math.ceil(retry_total * 100 - 1e-9) / 100,
        "within_cap": combined <= CAP_USD + 1e-9 and retry_total <= CAP_USD + 1e-9,
        "request_limits": {"max_output_tokens": 4096, "case_limit_s": 300,
                           "timeout_retries": 1,
                           "worst_case_case_min": "5 (the case limit); a capped request "
                                                  "takes about 2 min at the fitted rate"},
        "excludes": ("calls assume one model call per decision; think and read "
                     "continuations, retries, the capture probe per window and selector "
                     "calls beyond one per case are not in the estimate; the serving "
                     "deadline is the hard limit")}


def run(out: Path = REPO / "demo1/estimate.json") -> dict[str, Any]:
    from slh.demo1 import cases as case_files

    context = json.loads((case_files.OUT / "b1-context.json").read_text(encoding="utf-8"))
    man = case_files.manifest()
    cases = []
    for task in ("address", "variant"):
        for split in ("train", "calibrate"):
            cases += case_files.load(task, split)
    # Sealed cases are not opened: their sizes are taken as the calibration mean
    # (same generator and families), scaled to their count.
    sealed_n = sum(e["cases"] for rel, e in man["files"].items() if rel.startswith("sealed/"))
    cal = [c for c in cases if c["split"] == "calibrate"]
    for i in range(sealed_n):
        twin = dict(cal[i % len(cal)])
        cases.append({**twin, "case_id": f"sealed-proxy-{i}", "split": "sealed"})
    # Full-context note size: the train evidence with each agent action taken as
    # a message (the longest form is the gold tool call; either is close).
    from slh.demo1 import learn

    sizes = []
    for task in ("address", "variant"):
        train = [c for c in cases if c["task"] == task and c["split"] == "train"]
        rows = [{"case_id": c["case_id"], "correct": True,
                 "action": c["gold"] if c["gold"]["type"] == "tool_call"
                 else {"type": "message", "content": "Which one do you mean?"}}
                for c in train]
        sizes.append(_count(learn.full_context_note(task, learn.evidence_text(train, rows))))
    from slh.demo1.context import b0_context

    result = estimate(cases, context, full_context_tokens=max(sizes), b0_context=b0_context())
    result["tokens"]["full_context_measured"] = sizes
    result["sealed_size_note"] = ("sealed case sizes are proxied by the calibration cases "
                                  "(the sealed tree is not opened for an estimate)")
    out.write_text(json.dumps(result, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    return result

