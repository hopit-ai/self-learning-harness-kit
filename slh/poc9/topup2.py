"""POC9 second top-up window: the "Model only" B0 reference arm (deviation D5).

Added after the results were seen, as a descriptive reference row only: the
same 120 sealed decisions and scoring with the phase-1 B0 setup (tau2's
reference agent, ``configs/a0-vllm.yaml``: no think tool, no think
instruction, no skill), recorded prompt hash 3384b6e5... (B1 is 386fcf13...).
It changes no learned artifact, no existing arm, row, freeze or contrast.

Output goes to ``<run>/topup2/`` (one attempt); its own frozen plan is
``artifacts/topup2-freeze.json``; the base freeze and the first top-up freeze
are re-verified unchanged before anything runs.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from slh.demo1 import decide
from slh.demo1.context import B1_PROMPT_SHA256, SURFACE_SHA256
from slh.poc9 import arms, estimate, pipeline, topup

TOPUP2_DIR = "topup2"
FREEZE2 = "artifacts/topup2-freeze.json"
PURPOSE = "final_run_b0"
ARM = "b0"
NOTE = ("D5: Model-only reference row (phase-1 B0: reference agent, no think tool or "
        "instruction, no skill), added after the results were seen; descriptive only, "
        "changes no learned artifact or comparison")


def b0_record() -> dict[str, Any]:
    """The phase-1 B0 prompt/tools (checked against their recorded hashes and
    B1's model settings by demo1), verified to be a different prompt from B1
    without the think tool or the think instruction."""
    from slh.demo1.context import B0_CONFIG, b0_context, runner_config
    from slh.poc9 import writers

    ctx = b0_context()
    names = [t["function"]["name"] for t in ctx["tools"]]
    instruction = (runner_config().think_instruction or "").strip()
    if ctx["system_prompt_sha256"] == B1_PROMPT_SHA256:
        raise ValueError("b0 renders the B1 prompt")
    if "think" in names:
        raise ValueError("b0 must not carry the think tool")
    if not instruction or instruction in ctx["system_prompt"] or \
            instruction.splitlines()[0] in ctx["system_prompt"]:
        raise ValueError("b0 must not carry the think instruction")
    return {"config": B0_CONFIG, "system_prompt_sha256": ctx["system_prompt_sha256"],
            "tools_sha256": ctx["tools_sha256"], "tools": names,
            "prompt_tokens": writers.count_tokens(ctx["system_prompt"]) +
            writers.count_tokens(json.dumps(ctx["tools"], sort_keys=True))}


def prior_spend(run: Path = pipeline.RUN_DIR) -> dict[str, float]:
    """Recorded spend including the first top-up window (through verified stop)."""
    run = Path(run)
    parts = {k: v for k, v in topup.prior_spend(run).items() if k != "total"}
    costs = pipeline._read(run / topup.TOPUP_DIR / "costs.json")
    gpu = costs.get("gpu_usd_through_verified_stop_estimated")
    if gpu is None:
        raise ValueError("first top-up: no verified-stop GPU cost recorded")
    parts[f"{run.name}/{topup.TOPUP_DIR}"] = float(gpu) + float(costs.get("teacher_usd", 0))
    parts["total"] = sum(parts.values())
    return parts


def build_plan(run: Path = pipeline.RUN_DIR) -> dict[str, Any]:
    run = Path(run)
    frozen = pipeline.verify_freeze(run)
    plan1 = topup.load_plan(run, frozen)
    if plan1 is None:
        raise ValueError("the second top-up needs the first top-up freeze")
    if ARM in {a["name"] for a in frozen["arms"]}:
        raise ValueError("b0 is already a frozen arm")
    record = b0_record()
    taken = {k[1] for k in topup._complete_keys(run, frozen)} | \
        {e["system_prompt_sha256"] for e in plan1["arms"]}
    if record["system_prompt_sha256"] in taken:
        raise ValueError("b0 would repeat a complete arm's prompt")
    entry = {"name": ARM, "kind": "b0", "text": "", "writer": None, "snapshot": None,
             "mode": "new", "case_ids": list(frozen["test_case_ids"]), **record}
    body = {"schema": "poc9-topup2-freeze-v1", "base_freeze_sha256": frozen["sha256"],
            "topup_freeze_sha256": plan1["sha256"],
            "topup_freeze_file_sha256": pipeline._file_sha(run / topup.FREEZE),
            "b1_prompt_sha256": B1_PROMPT_SHA256, "arms": [entry], "deviation": "D5",
            "note": NOTE}
    body["sha256"] = arms.sha(arms._canonical(body))
    return body


def write_plan(run: Path = pipeline.RUN_DIR) -> dict[str, Any]:
    run = Path(run)
    if (run / TOPUP2_DIR).exists():
        raise FileExistsError("the second top-up already started; its freeze is fixed")
    plan = build_plan(run)
    pipeline._write(run / FREEZE2, plan)
    return plan


def load_plan(run: Path, frozen: dict[str, Any]) -> dict[str, Any] | None:
    """The recorded b0 freeze, re-verified with the base and first top-up freezes."""
    run = Path(run)
    path = run / FREEZE2
    if not path.is_file():
        return None
    plan = pipeline._read(path)
    if arms.sha(arms._canonical({k: v for k, v in plan.items() if k != "sha256"})) != \
            plan.get("sha256"):
        raise ValueError("second top-up freeze changed")
    if plan["base_freeze_sha256"] != frozen["sha256"]:
        raise ValueError("second top-up freeze belongs to a different base freeze")
    plan1 = topup.load_plan(run, frozen)
    if plan1 is None or plan1["sha256"] != plan["topup_freeze_sha256"] or \
            pipeline._file_sha(run / topup.FREEZE) != plan["topup_freeze_file_sha256"]:
        raise ValueError("the first top-up freeze changed")
    record = b0_record()
    if any(plan["arms"][0][k] != v for k, v in record.items()):
        raise ValueError("the b0 prompt or tools no longer reproduce")
    return plan


def rows(run: Path) -> list[dict[str, Any]]:
    return [r for r in pipeline._rows(Path(run) / TOPUP2_DIR / "results.jsonl")
            if r["split"] == "test" and r["arm"] == ARM]


def estimate_window(run: Path, plan: dict[str, Any], *, cumulative_cap_usd: float,
                    window_min: int | None = None,
                    margin_factor: float = topup.MARGIN_FACTOR) -> dict[str, Any]:
    """Measured estimate; ``margin_factor`` is explicit and recorded (D5: 1.10)."""
    prior = prior_spend(run)
    est = topup.estimate_topup(run, plan, cumulative_cap_usd=cumulative_cap_usd,
                               prior_spend_usd=prior["total"], window_min=window_min,
                               margin_factor=margin_factor)
    return {**est, "schema": "poc9-topup2-estimate-v1"}


def registration_template(*, root: Path = pipeline.REPO, run: Path = pipeline.RUN_DIR,
                          cumulative_cap_usd: float = 17.25,
                          window_min: int | None = None,
                          margin_factor: float = topup.MARGIN_FACTOR) -> dict[str, Any]:
    from slh.demo1.preflight import request_limits

    run = Path(run)
    frozen = pipeline.verify_freeze(run)
    plan = load_plan(run, frozen)
    if plan is None:
        raise FileNotFoundError("write the second top-up freeze first (topup2-freeze)")
    prior = prior_spend(run)
    est = estimate_window(run, plan, cumulative_cap_usd=cumulative_cap_usd,
                          window_min=window_min, margin_factor=margin_factor)
    return {"margin_factor": margin_factor,"event": "poc9_amendment", "amendment_id": "learning-v2-amendment-9",
            "mode": "topup2", "owner_approved": False, "surface_sha256": SURFACE_SHA256,
            "hashes": pipeline.bound_hashes(root), "window_min": est["window_min"],
            "cap_usd": est["window_cap_usd"], "cumulative_cap_usd": cumulative_cap_usd,
            "prior_spend": prior, "teacher_cap_usd": 0.0,
            "request_limits": request_limits(), "estimate": est,
            "base_freeze_sha256": frozen["sha256"],
            "topup_freeze_sha256": plan["topup_freeze_sha256"],
            "topup2_freeze_sha256": plan["sha256"], "arms": [ARM], "deviations": ["D5"]}


def preflight_run(registered: dict[str, Any], run: Path, *,
                  allow_window_record: bool = False) -> dict[str, Any]:
    run = Path(run)
    if registered.get("mode") != "topup2":
        raise ValueError("registration is not a second top-up (b0) registration")
    frozen = pipeline.verify_freeze(run)
    if frozen["sha256"] != registered.get("base_freeze_sha256"):
        raise ValueError("base freeze differs from the registered one")
    plan = load_plan(run, frozen)
    if plan is None or plan["sha256"] != registered.get("topup2_freeze_sha256") or \
            plan["topup_freeze_sha256"] != registered.get("topup_freeze_sha256"):
        raise ValueError("top-up freezes missing or differ from the registered ones")
    out = run / TOPUP2_DIR
    if out.exists() and not (allow_window_record and
                             {p.name for p in out.iterdir()} == {"window.json"}):
        raise FileExistsError("second top-up output exists; one attempt only, no reruns")
    prior = prior_spend(run)
    if prior != registered.get("prior_spend"):
        raise ValueError("recorded prior spend differs from the registered one")
    margin = registered.get("margin_factor")
    if not isinstance(margin, int | float) or isinstance(margin, bool) or \
            margin != registered.get("estimate", {}).get("margin_factor"):
        raise ValueError("registration must record the margin factor its estimate used")
    est = estimate_window(run, plan, cumulative_cap_usd=registered["cumulative_cap_usd"],
                          window_min=registered["window_min"], margin_factor=float(margin))
    if est != registered.get("estimate") or not est["within_cap"] or not est["fits_window"]:
        raise ValueError("registered b0 estimate does not fit the window/cumulative cap")
    return {"window_min": est["window_min"], "cap_usd": est["window_cap_usd"],
            "teacher_cap_usd": 0.0, "true_reservation_usd": est["reservation_usd"],
            "cumulative_cap_usd": est["cumulative_cap_usd"],
            "prior_spend_usd": prior["total"], "surface_sha256": SURFACE_SHA256}


def stage(run: Path = pipeline.RUN_DIR, client: Any = None) -> dict[str, Any]:
    """Probe, then ONLY b0 on the 120 sealed cases, into <run>/topup2."""
    from slh.poc9 import cases

    run = Path(run)
    frozen = pipeline.verify_freeze(run)
    plan = load_plan(run, frozen)
    if plan is None:
        raise FileNotFoundError("no second top-up freeze")
    out = run / TOPUP2_DIR
    if (out / "stage-started.json").exists() or (out / "results.jsonl").exists() or \
            (out / "raw").exists():
        raise FileExistsError("the second top-up stage was already attempted")
    pipeline._write(out / "stage-started.json", {"started_epoch": time.time(),
                                                 "surface_sha256": SURFACE_SHA256,
                                                 "topup2_freeze_sha256": plan["sha256"]})
    client = client or decide.AgentClient.for_window()
    pipeline._write(out / "probe.json", client.probe)
    registered = pipeline._read(out / "registration.json") \
        if (out / "registration.json").is_file() else {}
    need_s = 60 * registered.get("estimate", {}).get("arms_min", {}).get(ARM, 0)
    remaining = client.stage_deadline - time.time() if client.stage_deadline else math.inf
    if need_s + decide.CASE_LIMIT_S > remaining:
        raise decide.StageDeadlineReached("b0 cannot fit the remaining window; "
                                          "sealed test was not opened")
    entry = plan["arms"][0]
    failures: list[str] = []
    started = time.time()
    with pipeline._dispatcher_adapter(frozen["best_of_n"]):
        try:
            with cases.sealed_access(PURPOSE) as test:
                if [c["case_id"] for c in test] != frozen["test_case_ids"]:
                    raise ValueError("sealed membership/order differs from frozen manifest")
                wanted = set(entry["case_ids"])
                try:
                    pipeline.execute(arms.Arm(ARM, kind="b0"),
                                     [c for c in test if c["case_id"] in wanted], client, out)
                except pipeline.ArmFailed as exc:
                    failures.append(str(exc))
                    pipeline._write(out / "arm-failures.json", failures)
        finally:
            done = rows(run)
            window = pipeline._read(out / "window.json") if (out / "window.json").is_file() \
                else {}
            elapsed = time.time() - window.get("started_epoch", started)
            gpu = elapsed / 3600 * estimate.hourly_rate()
            prior = registered.get("prior_spend", {}).get("total")
            pipeline._write(out / "costs.json", {
                "schema": "poc9-topup2-costs-v1", "teacher_usd": 0.0,
                "gpu_elapsed_s_before_cleanup": elapsed,
                "gpu_usd_before_cleanup_estimated": gpu,
                "arm_slot_s": {ARM: pipeline._slot_seconds(done)},
                "sealed_s": time.time() - started, "prior_spend_usd": prior,
                "cumulative_usd_before_cleanup_estimated": None if prior is None
                else prior + gpu, "shutdown_cost_pending": True})
    result = pipeline.report(run)
    if failures:
        raise pipeline.ArmFailed("b0 failed on every started case: " + "; ".join(failures))
    return result
