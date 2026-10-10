"""Offline POC9 calls, tokens, latency and corrected sandbox prices.

The shared prices file is identity-pinned and is never changed here. GPU totals
include six CPU cores and 40 GiB of memory at their actual per-second rates.
Teacher token prices are supplied by the operator; zero is only a placeholder.
"""
from __future__ import annotations

import math
from typing import Any

from slh.demo1 import estimate as demo

CPU_CORE_S = 0.00003942
MEMORY_GIB_S = 0.00000667
CPU_CORES = 6
MEMORY_GIB = 40
GPU_PER_HOUR = {"L40S": 1.952, "A100-40GB": 2.10}
WINDOW_MIN = 180
CAP_USD = 14.70
TEACHER_PLACEHOLDER_USD = 0.0
# Writer-path bounds (review 4): 12 updates x 6 + 6 generic; 12 x 9 + 6 generic.
ACTOR_WRITER_CALLS = 12 * (2 + 4) + 6
TEACHER_WRITER_CALLS = 12 * (3 + 3 + 3) + 6


def hourly_rate(gpu: str = "A100-40GB") -> float:
    return GPU_PER_HOUR[gpu] + 3600 * (CPU_CORES * CPU_CORE_S + MEMORY_GIB * MEMORY_GIB_S)


def reservation(window_min: float = WINDOW_MIN, *, gpu: str = "A100-40GB",
                teacher_cap_usd: float = 0.0, fixed_usd: float = 0.0) -> float:
    if not 0 < window_min <= WINDOW_MIN:
        raise ValueError("one window must be positive and at most 180 minutes")
    if any(not math.isfinite(x) or x < 0 for x in (teacher_cap_usd, fixed_usd)):
        raise ValueError("teacher and fixed charges must be finite and nonnegative")
    return math.ceil((hourly_rate(gpu) * window_min / 60 + teacher_cap_usd + fixed_usd)
                     * 1_000_000) / 1_000_000


def measured_model() -> dict[str, Any]:
    # Demo1's fit uses the recorded B1 stream, at the same four slots.
    return demo.latency_model()


def call_seconds(inp: float, out: float, model: dict[str, Any]) -> float:
    return max(0.001, model["a_s"] + model["b_s_per_output_token"] * out
               + model["c_s_per_input_token"] * inp)


def estimate(*, model: dict[str, Any] | None = None, window_min: int = WINDOW_MIN,
             teacher_cap_usd: float = TEACHER_PLACEHOLDER_USD,
             cap_usd: float = CAP_USD, attempts_mean: float = 2.0,
             prompt_tokens: float | None = None, skill_tokens: int = 400,
             history_tokens: int = 48 * 450, fixed_usd: float = 0.0) -> dict[str, Any]:
    """No sealed reads. Writer/gate bounds include critique, repair and one shorten.

    Learning is serial within each writer; only sealed calls get the measured
    four-worker divisor. TS drafts and rewrites each patch. A conservative 1.5x
    allowance covers variable generation; hard request/stage limits still rule.
    """
    model = model or measured_model()
    prompt = prompt_tokens or model["median_prompt_tokens"]
    if not math.isfinite(attempts_mean) or attempts_mean < 1:
        raise ValueError("best-of-N needs at least one attempt")
    stages: dict[str, dict[str, float]] = {}

    def add(stage: str, calls: float, inp: float, out: float, slots: int = 1) -> None:
        part = stages.setdefault(stage, {"calls": 0., "input_tokens": 0.,
                                         "output_tokens": 0., "slot_s": 0., "wall_s": 0.})
        busy = calls * call_seconds(inp, out, model)
        for key, value in (("calls", calls), ("input_tokens", calls * inp),
                           ("output_tokens", calls * out), ("slot_s", busy),
                           ("wall_s", busy / slots)):
            part[key] += value

    add("probe", 1, 50, 16)
    # Training and validation checks for each of three independent trajectories.
    add("learning", 48, prompt, demo.DECISION_OUTPUT_TOKENS, demo.WORKERS)
    add("learning", 12 + 3 * 12 * 12, prompt + skill_tokens,
        demo.DECISION_OUTPUT_TOKENS, demo.WORKERS)
    # Per update: draft, critique, length-aware repair and one bounded shorten
    # repair, with a second rewrite/fidelity pass for TS (78 actor-served calls,
    # generic notes included). The teacher side also carries TS equivalence
    # checks: up to 114 calls before the per-writer ceiling. Teacher spend stays
    # under its hard cap (reservation), whatever the call count.
    add("agent_writers", ACTOR_WRITER_CALLS, prompt + history_tokens / 2, 500)
    add("teacher", TEACHER_WRITER_CALLS, history_tokens / 2 + 1000, 500)
    # B1, nine learned snapshots, human, three generic notes, full context.
    add("sealed", 120, prompt, demo.DECISION_OUTPUT_TOKENS, demo.WORKERS)
    add("sealed", 120 * 13, prompt + skill_tokens, demo.DECISION_OUTPUT_TOKENS,
        demo.WORKERS)
    add("sealed", 120, prompt + history_tokens, demo.DECISION_OUTPUT_TOKENS, demo.WORKERS)
    add("best_of_n", 3 * 120 * attempts_mean, prompt, demo.DECISION_OUTPUT_TOKENS, demo.WORKERS)
    # Pairwise tournament: one same-agent selector per extra candidate, budgeted
    # even if identical candidates let the dispatcher skip a selector request.
    add("best_of_n", 3 * 120 * (attempts_mean - 1), prompt + 300,
        demo.SELECTOR_OUTPUT_TOKENS, demo.WORKERS)
    # Teacher throughput is unknown: reserve this time; do not infer it from 9B.
    teacher_time_min = 10.0
    model_min = sum(v["wall_s"] for k, v in stages.items() if k != "teacher") / 60
    needed = demo.STARTUP_MIN + demo.DRAIN_MIN + 1.5 * model_min + teacher_time_min
    cost = reservation(window_min, teacher_cap_usd=teacher_cap_usd, fixed_usd=fixed_usd)
    return {"schema": "poc9-estimate-v1", "latency_model": model,
            "stages": stages, "totals": {k: sum(s[k] for s in stages.values())
                for k in ("calls", "input_tokens", "output_tokens")},
            "model_min_expected": model_min, "conservative_factor": 1.5, "needed_min_conservative": needed,
            "teacher_time_placeholder_min": teacher_time_min,
            "teacher_cost_placeholder": teacher_cap_usd == 0,
            "teacher_cap_usd": teacher_cap_usd, "window_min": window_min,
            "best_of_n_mean_attempts": attempts_mean,
            "rates_per_hour": {g: hourly_rate(g) for g in GPU_PER_HOUR},
            "cpu_core_per_second": CPU_CORE_S, "memory_gib_per_second": MEMORY_GIB_S,
            "cpu_cores": CPU_CORES, "memory_gib": MEMORY_GIB,
            "reservation_usd": cost, "cap_usd": cap_usd,
            "within_cap": math.isfinite(cap_usd) and cost <= cap_usd,
            "fits_window": needed <= window_min,
            "assumptions": "one call per decision; continuations and retries may increase "
                           "calls. Best-of-N schedule is refrozen from measured learning "
                           "costs before sealed access. Teacher latency/cost are placeholders; "
                           "the serving deadline and teacher budget are hard limits."}


if __name__ == "__main__":
    import json

    print(json.dumps(estimate(), indent=2, sort_keys=True))
