"""POC9 top-up window on the SAME frozen run (deviations D3 and D4).

The first paid window (attempt 3, ``poc9/runs/w5p``) stopped at the watchdog
with some sealed arms complete. This module, offline unless ``stage`` is
called with a live client:

- verifies the owner's D3 identities: arms whose rendered system prompt is
  byte-identical (same skill text, same prompt hash on every recorded row)
  count as each other, and fails closed on any difference;
- builds a top-up freeze (``artifacts/topup-freeze.json``, the base freeze is
  never rewritten) listing ONLY the arms still to run, in priority order, and
  refuses any arm whose dispatch key (kind, prompt hash, best-of-N schedule)
  equals an already-complete arm;
- D4 (proposed; binding only once the owner approves the top-up registration):
  the frozen full-context text (raw transcripts + raw feedback JSON) does not
  fit the served context, so the arm shows the SAME evidence text every writer
  saw (``writers.render_evidence`` over all 48 episodes, each 4-episode batch
  reproducing the hash recorded at learning time), never truncated;
- estimates the window from durations measured in the run;
- runs the listed arms into ``<run>/topup/`` (one attempt; base rows untouched).
"""
from __future__ import annotations

import datetime as dt
import functools
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from slh import serving
from slh.demo1 import decide
from slh.demo1.context import B1_PROMPT_SHA256, SURFACE_SHA256
from slh.poc9 import arms, estimate, pipeline

TOPUP_DIR = "topup"
FREEZE = "artifacts/topup-freeze.json"
# Most important first: a window cut short still yields the earlier arms. The
# self-written skill (S) is the central claim, so its generic control precedes T's.
PRIORITY = ("full_context", "generic_S", "generic_T", "best_of_n_T")
SKIPPED = {"generic_TS": "D3: writer TS learned nothing (snapshots 12/24/48 are empty, i.e. "
                         "B1), so its generic control tests no learned artifact; owner decision"}
B1_SOURCE = "learned_TS@48"  # first B1-prompt arm completed in the frozen arm order
IDENTITIES = {"b1": B1_SOURCE, "learned_TS@24": B1_SOURCE, "learned_T@12": B1_SOURCE,
              "removal_T": B1_SOURCE, "removal_S": B1_SOURCE, "removal_TS": B1_SOURCE,
              "learned_S@24": "learned_S@48", "learned_T@48": "learned_T@24"}
B1_REPLICATES = ("learned_TS@12",)
FULL_CONTEXT_RULE = ("D4: full-context text = 'Training histories and deterministic feedback:' "
                     "+ writers.render_evidence over all 48 training episodes in stream order "
                     "(the exact evidence blocks every writer saw; no episode dropped, nothing "
                     "truncated). The frozen raw-transcript rendering did not fit max_model_len.")
MARGIN_FACTOR = 1.15
# Serving stops admitting requests stop + request margins (5 + 2 = 7 min) before
# the window ends; no case starts within CASE_LIMIT_S of that deadline.
ADMISSION_MARGIN_MIN = (serving.DEFAULT_STOP_MARGIN_S + serving.DEFAULT_REQUEST_MARGIN_S) / 60
LONG_CONTEXT_FACTOR = 2.0  # extra-prefill allowance beyond the fitted per-token term


def _raw(run: Path, name: str) -> list[dict[str, Any]]:
    return pipeline._rows(run / "raw" / f"{name}.jsonl")


@functools.lru_cache(maxsize=256)
def prompt_sha(text: str) -> str:
    from slh.poc9 import loader

    return loader.render(text or None)["system_prompt_sha256"]


def _complete(rows: list[dict[str, Any]], expected: list[str]) -> bool:
    return pipeline._exact(rows, expected)


def _check_hashes(name: str, rows: list[dict[str, Any]], rendered: str) -> int:
    """Every successful row MUST carry the rendered prompt hash; an error row
    may lack one (no request completed) but never carries another."""
    for r in rows:
        recorded = r.get("system_prompt_sha256")
        if recorded is None and not r.get("error"):
            raise ValueError(f"D3: a successful {name} row ({r['case_id']}) has no prompt hash")
        if recorded is not None and recorded != rendered:
            raise ValueError(f"D3: {name} row {r['case_id']} carries prompt hash {recorded}, "
                             f"not the rendered {rendered}")
    if not any(r.get("system_prompt_sha256") for r in rows):
        raise ValueError(f"D3: no {name} row carries a prompt hash")
    return sum(bool(r.get("system_prompt_sha256")) for r in rows)


def verify_identities(run: Path, frozen: dict[str, Any]) -> list[dict[str, Any]]:
    """D3, fail closed: same kind, same text and skill hash, rendered prompt hash
    equal to EVERY hash recorded on the source's rows, source coverage exact, and
    the target never ran. B1 identities also match the B1 constant and the
    acquisition rows."""
    by = {a["name"]: a for a in frozen["arms"]}
    expected = frozen["test_case_ids"]
    acquisition = _raw(run, "acquisition")
    if not acquisition or any(r.get("system_prompt_sha256") != B1_PROMPT_SHA256
                              for r in acquisition):
        raise ValueError("D3: acquisition rows do not all carry the B1 prompt hash")
    table = []
    pairs = [(t, s, "identity") for t, s in IDENTITIES.items()] + \
        [(t, B1_SOURCE, "replicate") for t in B1_REPLICATES]
    for target, source, role in pairs:
        t, s = by.get(target), by.get(source)
        if t is None or s is None:
            raise ValueError(f"D3: {target} or {source} is not a frozen arm")
        if t["kind"] != "notes" or s["kind"] != "notes" or s["reuse_b1"]:
            raise ValueError(f"D3: {target} <- {source} are not single-pass note arms")
        if t["text"] != s["text"] or t["skill_sha256"] != s["skill_sha256"]:
            raise ValueError(f"D3: {target} and {source} differ in skill text")
        rendered = prompt_sha(t["text"])
        rows = _raw(run, source)
        if not _complete(rows, expected):
            raise ValueError(f"D3: source {source} is not complete")
        hashed = _check_hashes(source, rows, rendered)
        if not t["text"] and rendered != B1_PROMPT_SHA256:
            raise ValueError(f"D3: empty-skill {target} does not render the B1 prompt")
        if role == "identity" and _raw(run, target):
            raise ValueError(f"D3: identity target {target} has rows of its own")
        if role == "replicate":
            own = _raw(run, target)
            if not _complete(own, expected):
                raise ValueError(f"D3: replicate {target} is not complete")
            _check_hashes(target, own, rendered)
        table.append({"target": target, "source": source, "role": role,
                      "skill_sha256": t["skill_sha256"], "system_prompt_sha256": rendered,
                      "source_rows": len(rows), "source_rows_with_prompt_hash": hashed})
    return table


def full_context_text(run: Path) -> tuple[str, dict[str, Any]]:
    """D4 text from the frozen run's own acquisition rows and the train cases;
    every 4-episode batch must reproduce the evidence hash its writers logged."""
    from slh.poc9 import cases, corrections, score, writers

    train = sorted(cases.load("train"), key=lambda c: c["stream_index"])
    raw = {r["case_id"]: r for r in _raw(run, "acquisition")}
    if len(train) != 48 or set(raw) != {c["case_id"] for c in train}:
        raise ValueError("D4: acquisition rows do not match the 48 train cases")
    episodes = {c["stream_index"] + 1: {
        "index": c["stream_index"] + 1, "case": c, "action": raw[c["case_id"]]["action"],
        "reads": raw[c["case_id"]].get("reads") or [],
        "score": score.score(c, raw[c["case_id"]]["action"]),
        "correction": corrections.explain(c, raw[c["case_id"]]["action"])} for c in train}
    checked = 0
    for writer in arms.WRITERS:
        for patch in pipeline._rows(run / "learning" / writer / "patches.jsonl"):
            text = writers.render_evidence([episodes[i] for i in patch["episodes"]])
            if writers.sha256_text(text) != patch["evidence_sha256"]:
                raise ValueError(f"D4: {writer} batch {patch['episodes']} evidence differs "
                                 "from what the writer saw")
            checked += 1
    if not checked:
        raise ValueError("D4: no recorded writer evidence to check against")
    text = "Training histories and deterministic feedback:\n\n" + \
        writers.render_evidence([episodes[i] for i in sorted(episodes)])
    return text, {"writer_batches_checked": checked, "tokens": writers.count_tokens(text)}


def _key(kind: str, text: str, schedule: Any = None) -> tuple[str, str, str | None]:
    return (kind, B1_PROMPT_SHA256 if kind == "best_of_n" else prompt_sha(text),
            arms.sha(arms._canonical(schedule)) if kind == "best_of_n" else None)


def _complete_keys(run: Path, frozen: dict[str, Any]) -> dict[tuple, str]:
    expected = frozen["test_case_ids"]
    return {_key(a["kind"], a["text"], frozen["best_of_n"].get(a["writer"])): a["name"]
            for a in frozen["arms"]
            if not a["reuse_b1"] and _complete(_raw(run, a["name"]), expected)}


def _fill_split(name: str, base_rows: list[dict[str, Any]], expected: list[str]
                ) -> tuple[list[str], list[str]]:
    """(redo, kept): only never-started cases (stage deadline, no model call)
    are redone; every started case, including a case-limit timeout, is kept."""
    if not base_rows:
        raise ValueError(f"{name}: fill mode needs the base rows")
    redo = [r["case_id"] for r in base_rows if r.get("stage_deadline")]
    if any(r.get("model_calls") for r in base_rows if r.get("stage_deadline")):
        raise ValueError(f"{name}: a deadline row recorded model calls")
    kept = [r["case_id"] for r in base_rows if not r.get("stage_deadline")]
    if sorted(redo + kept) != sorted(expected):
        raise ValueError(f"{name}: kept and redone cases do not cover the test once")
    return [c for c in expected if c in set(redo)], kept


def check_eligible(run: Path, frozen: dict[str, Any], plan: dict[str, Any],
                   entry: dict[str, Any]) -> None:
    """Re-validate, from the CURRENT base rows, that ``entry`` may still be
    dispatched exactly as frozen; refuse otherwise (call right before dispatch)."""
    run = Path(run)
    name, expected = entry["name"], frozen["test_case_ids"]
    covered = {i["target"] for i in plan["identities"]} | set(B1_REPLICATES)
    if name in covered or name in plan["skipped"]:
        raise ValueError(f"{name} is covered by D3 or skipped; never dispatched")
    if (run / TOPUP_DIR / "raw" / f"{name}.jsonl").exists():
        raise FileExistsError(f"{name} already attempted in the top-up")
    base_rows = _raw(run, name)
    if entry["mode"] == "fill":
        redo, kept = _fill_split(name, base_rows, expected)
        if redo != entry["case_ids"] or len(kept) != entry["kept_case_ids"]:
            raise ValueError(f"{name}: base rows changed since the top-up freeze")
    elif entry.get("amendment") == "D4":
        if len(base_rows) != entry["base_rows_superseded"] or \
                (base_rows and not (pipeline.arm_failed(base_rows) or
                                    pipeline.all_errors(base_rows))):
            raise ValueError(f"{name}: base rows changed since the top-up freeze")
    elif base_rows:
        raise ValueError(f"{name} now has base rows; no outcome-based reruns")
    a = next(x for x in frozen["arms"] if x["name"] == name)
    key = _key(a["kind"], entry["text"], frozen["best_of_n"].get(a["writer"]))
    clash = _complete_keys(run, frozen).get(key)
    if clash is not None and clash != name:
        raise ValueError(f"{name} would repeat complete arm {clash}")


def build_plan(run: Path = pipeline.RUN_DIR, *, names: tuple[str, ...] | list[str] = PRIORITY,
               dropped: dict[str, str] | None = None) -> dict[str, Any]:
    """Offline top-up freeze for the arms ``names`` (priority order kept)."""
    from slh.poc9 import cases, writers

    run = Path(run)
    frozen = pipeline.verify_freeze(run)
    by = {a["name"]: a for a in frozen["arms"]}
    expected = frozen["test_case_ids"]
    identities = verify_identities(run, frozen)
    covered = {i["target"] for i in identities} | set(B1_REPLICATES)
    dropped = dict(dropped or {})
    skipped = {**SKIPPED, **dropped}
    for name in names:
        if name in covered or name in skipped:
            raise ValueError(f"{name} is covered by D3 or skipped; never dispatched")
    order = [n for n in PRIORITY if n in names]
    if len(order) != len(set(names)):
        raise ValueError(f"top-up arms must come from {PRIORITY}")
    complete_keys = _complete_keys(run, frozen)
    longest = max(writers.count_tokens(json.dumps(c["messages"], sort_keys=True))
                  for split in ("train", "validation") for c in cases.load(split))
    entries = []
    for name in order:
        if name in covered or name in skipped:
            raise ValueError(f"{name} is covered by D3 or skipped; never dispatched")
        a = dict(by[name])
        base_rows = _raw(run, name)
        entry: dict[str, Any] = {"name": name, "kind": a["kind"], "writer": a["writer"],
                                 "snapshot": a["snapshot"]}
        if name == "full_context":
            text, info = full_context_text(run)
            if base_rows and not (pipeline.arm_failed(base_rows) or
                                  pipeline.all_errors(base_rows)):
                raise ValueError("full_context base rows are not a failed arm")
            entry.update(text=text, mode="replace", case_ids=list(expected), amendment="D4",
                         rule=FULL_CONTEXT_RULE, frozen_skill_sha256=a["skill_sha256"],
                         frozen_tokens=frozen.get("full_context_tokens"),
                         base_rows_superseded=len(base_rows), **info)
        elif a["kind"] == "best_of_n":
            redo, kept = _fill_split(name, base_rows, expected)
            entry.update(text="", mode="fill", case_ids=redo,
                         kept_case_ids=len(kept),
                         schedule_sha256=arms.sha(arms._canonical(frozen["best_of_n"][a["writer"]])))
        else:
            if base_rows:
                raise ValueError(f"{name} already has rows; no outcome-based reruns")
            entry.update(text=a["text"], mode="replace", case_ids=list(expected))
        text = entry["text"]
        entry["skill_sha256"] = arms.sha(text) if text else None
        entry["system_prompt_sha256"] = B1_PROMPT_SHA256 if not text else prompt_sha(text)
        key = _key(a["kind"], text, frozen["best_of_n"].get(a["writer"]))
        if key in complete_keys:
            raise ValueError(f"{name} would repeat complete arm {complete_keys[key]}")
        complete_keys[key] = name
        if text:
            entry["prompt_tokens"] = pipeline.prompt_tokens(text)
            entry["context_need_train_max"] = entry["prompt_tokens"] + longest + \
                decide.MAX_OUTPUT_TOKENS + pipeline.CONTEXT_MARGIN_TOKENS
            if entry["context_need_train_max"] > pipeline.MAX_MODEL_LEN:
                raise ValueError(f"{name} cannot fit max_model_len {pipeline.MAX_MODEL_LEN}")
        entries.append(entry)
    body = {"schema": "poc9-topup-freeze-v1", "base_freeze_sha256": frozen["sha256"],
            "identities": identities, "skipped": skipped, "arms": entries,
            "max_model_len": pipeline.MAX_MODEL_LEN,
            "note": "D3 identities and skipped arms decided before any sealed score was read"}
    body["sha256"] = arms.sha(arms._canonical(body))
    return body


def write_plan(run: Path = pipeline.RUN_DIR, **kwargs: Any) -> dict[str, Any]:
    run = Path(run)
    if (run / TOPUP_DIR).exists():
        raise FileExistsError("the top-up window already started; its freeze is fixed")
    plan = build_plan(run, **kwargs)
    pipeline._write(run / FREEZE, plan)
    return plan


def load_plan(run: Path, frozen: dict[str, Any]) -> dict[str, Any] | None:
    """The recorded top-up freeze, re-verified (sha, base freeze, D3 table, D4 text)."""
    path = Path(run) / FREEZE
    if not path.is_file():
        return None
    plan = pipeline._read(path)
    if arms.sha(arms._canonical({k: v for k, v in plan.items() if k != "sha256"})) != \
            plan.get("sha256"):
        raise ValueError("top-up freeze changed")
    if plan["base_freeze_sha256"] != frozen["sha256"]:
        raise ValueError("top-up freeze belongs to a different base freeze")
    if verify_identities(Path(run), frozen) != plan["identities"]:
        raise ValueError("D3 identities no longer verify as recorded")
    for entry in plan["arms"]:
        if entry["name"] == "full_context" and \
                full_context_text(Path(run))[0] != entry["text"]:
            raise ValueError("D4 full-context text no longer reproduces")
    return plan


def effective_rows(run: Path, plan: dict[str, Any], by: dict[str, list[dict[str, Any]]]
                   ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Base rows + top-up rows ('fill' keeps the started base cases) + D3 copies.

    Every top-up arm's frozen premise is re-checked against the CURRENT base
    rows. On any mismatch nothing is substituted and the arm is returned in
    ``errors``; the report then holds it incomplete (scores stay concealed)."""
    out = dict(by)
    errors: dict[str, str] = {}
    new = [r for r in pipeline._rows(Path(run) / TOPUP_DIR / "results.jsonl")
           if r["split"] == "test"]
    for entry in plan["arms"]:
        name = entry["name"]
        rows = [{**r, "topup": True, **({"amendment": entry["amendment"]}
                                         if entry.get("amendment") else {})}
                for r in new if r["arm"] == name]
        base = by[name]
        try:
            if entry["mode"] == "fill":
                started = [r for r in base if not r.get("stage_deadline")]
                redo = [r["case_id"] for r in base if r.get("stage_deadline")]
                if sorted(redo) != sorted(entry["case_ids"]) or \
                        len(started) != entry["kept_case_ids"] or \
                        any(r.get("model_calls") for r in base if r.get("stage_deadline")) or \
                        len({r["case_id"] for r in base}) != len(base):
                    raise ValueError(f"{name}: base rows no longer match the frozen fill split "
                                     f"({len(started)} started / {len(redo)} never started, "
                                     f"frozen {entry['kept_case_ids']} / "
                                     f"{len(entry['case_ids'])})")
            elif entry.get("amendment") == "D4":
                if len(base) != entry["base_rows_superseded"] or (base and not (
                        pipeline.arm_failed(base) or pipeline.all_errors(base))):
                    raise ValueError(f"{name}: base rows are not the recorded failure")
            elif base:
                raise ValueError(f"{name} has base rows; top-up rows cannot replace them")
            if any(r["case_id"] not in set(entry["case_ids"]) for r in rows):
                raise ValueError(f"{name}: top-up rows outside the frozen case list")
        except ValueError as exc:
            errors[name] = str(exc)
            out[name] = []  # no rows: no coverage and no contrast from a stale premise
            continue
        redo = set(entry["case_ids"])
        kept = [r for r in base if r["case_id"] not in redo] if entry["mode"] == "fill" else []
        out[name] = kept + rows
    for ident in plan["identities"]:
        if ident["role"] != "identity":
            continue
        target, source = ident["target"], ident["source"]
        if by.get(target):
            raise ValueError(f"D3 target {target} has rows of its own")
        extra = {"reused_b1": True, "identical_to_b1_by_construction": True} \
            if target.startswith("removal_") else {}
        out[target] = [{**r, "arm": target, "identity_source": source, "deviation": "D3",
                        "incremental_model_calls": 0, **extra} for r in out[source]]
    return out, errors


# ------------------------------------------------------------------ measured estimate

def prior_spend(run: Path = pipeline.RUN_DIR) -> dict[str, float]:
    """Recorded POC9 spend: this run and every archived attempt (GPU through
    verified stop at the true sandbox rate, plus teacher)."""
    run = Path(run)
    parts = {}
    for d in [run, *sorted(run.parent.glob(f"{run.name}.attempt*"))]:
        costs = pipeline._read(d / "costs.json")
        gpu = costs.get("gpu_usd_through_verified_stop_estimated")
        if gpu is None:
            raise ValueError(f"{d.name}: no verified-stop GPU cost recorded")
        parts[d.name] = float(gpu) + float(costs.get("teacher_usd", 0))
    parts["total"] = sum(parts.values())
    return parts


def measured(run: Path = pipeline.RUN_DIR) -> dict[str, Any]:
    """Durations measured in the base window (no sealed score is read)."""
    run = Path(run)
    frozen = pipeline.verify_freeze(run)
    window = pipeline._read(run / "window.json")
    probe = pipeline._read(run / "probe.json")
    probe_epoch = dt.datetime.fromisoformat(
        probe["capture"]["agent"]["probe"]["time_utc"]).timestamp()
    costs = pipeline._read(run / "costs.json")
    expected = frozen["test_case_ids"]
    by = {a["name"]: a for a in frozen["arms"]}
    per_case: dict[str, float] = {}
    busy = 0.
    for name in frozen["arm_order"]:
        started = [r for r in _raw(run, name) if not r.get("stage_deadline")]
        slot = sum(float(r.get("wall_s") or 0) for r in started)
        busy += slot
        if started and not pipeline.arm_failed(started):
            per_case[name] = slot / len(started)
    notes = [per_case[n] for n in per_case if by[n]["kind"] == "notes" and by[n]["text"]
             and _complete(_raw(run, n), expected)]
    best = {n: per_case[n] for n in per_case if by[n]["kind"] == "best_of_n"}
    return {"serve_start_min": (probe_epoch - window["started_epoch"]) / 60,
            "notes_per_case_slot_s_max": max(notes),
            "notes_per_case_slot_s_mean": statistics.mean(notes),
            "best_of_n_per_case_slot_s": best,
            "elapsed_over_busy": costs["sealed_s"] / (busy / pipeline.WORKERS),
            "sealed_s": costs["sealed_s"], "busy_slot_s": busy}


def estimate_topup(run: Path, plan: dict[str, Any], *, cumulative_cap_usd: float,
                   prior_spend_usd: float, window_min: int | None = None,
                   latency: dict[str, Any] | None = None,
                   margin_factor: float = MARGIN_FACTOR) -> dict[str, Any]:
    """Window = measured serve start + MARGIN_FACTOR x measured arm time + the
    case-start rule (no case starts within CASE_LIMIT_S of admission) + the
    episode margin. Reservation at the true sandbox rate (prices.yaml unedited)."""
    from slh.serving import build_allowance_usd

    if not math.isfinite(margin_factor) or not 1.0 <= margin_factor <= 2.0:
        raise ValueError("margin factor must be between 1.0 and 2.0")
    m = measured(run)
    latency = latency or estimate.measured_model()
    b1_tokens = pipeline.prompt_tokens("")
    arms_min = {}
    for e in plan["arms"]:
        n = len(e["case_ids"])
        if e["kind"] == "best_of_n":
            per = m["best_of_n_per_case_slot_s"].get(e["name"]) or \
                max(m["best_of_n_per_case_slot_s"].values())
        else:
            per = m["notes_per_case_slot_s_max"]
            extra = max(0, e.get("prompt_tokens", b1_tokens) - b1_tokens - 500)
            per += LONG_CONTEXT_FACTOR * latency["c_s_per_input_token"] * extra
        arms_min[e["name"]] = n * per / pipeline.WORKERS * m["elapsed_over_busy"] / 60
    model_min = sum(arms_min.values())
    overhead = m["serve_start_min"] + decide.CASE_LIMIT_S / 60 + ADMISSION_MARGIN_MIN
    needed = overhead + margin_factor * model_min
    window = window_min or math.ceil(needed)
    fixed = build_allowance_usd(pipeline.REPO / "configs/prices.yaml")
    reservation = estimate.reservation(window, fixed_usd=fixed)
    window_cap = round(cumulative_cap_usd - prior_spend_usd, 6)
    return {"schema": "poc9-topup-estimate-v1", "measured": m, "arms_min": arms_min,
            "model_min": model_min, "margin_factor": margin_factor,
            "overhead_min": overhead, "needed_min": needed, "window_min": window,
            "rate_per_hour": estimate.hourly_rate(), "fixed_usd": fixed,
            "reservation_usd": reservation, "prior_spend_usd": prior_spend_usd,
            "cumulative_cap_usd": cumulative_cap_usd, "window_cap_usd": window_cap,
            "cumulative_reserved_usd": prior_spend_usd + reservation,
            "fits_window": needed <= window <= estimate.WINDOW_MIN,
            "within_cap": reservation <= window_cap,
            "max_window_min_for_cap": math.floor((window_cap - fixed) /
                                                 estimate.hourly_rate() * 60)}


def fit_names(run: Path, *, cumulative_cap_usd: float, prior_spend_usd: float,
              names: tuple[str, ...] = PRIORITY) -> tuple[list[str], dict[str, str]]:
    """Drop arms from the end of the priority order (best_of_n_T first) until the
    measured reservation fits the cumulative cap."""
    names = list(names)
    dropped: dict[str, str] = {}
    while names:
        plan = build_plan(run, names=names)
        est = estimate_topup(run, plan, cumulative_cap_usd=cumulative_cap_usd,
                             prior_spend_usd=prior_spend_usd)
        if est["within_cap"] and est["fits_window"]:
            return names, dropped
        last = names.pop()
        dropped[last] = (f"dropped from the top-up: measured window does not fit the "
                         f"cumulative cap ${cumulative_cap_usd:.2f}")
    return names, dropped


# ------------------------------------------------------------------ registration / preflight

def registration_template(*, root: Path = pipeline.REPO, run: Path = pipeline.RUN_DIR,
                          cumulative_cap_usd: float = 16.50,
                          window_min: int | None = None) -> dict[str, Any]:
    from slh.demo1.preflight import request_limits

    run = Path(run)
    frozen = pipeline.verify_freeze(run)
    plan = load_plan(run, frozen)
    if plan is None:
        raise FileNotFoundError("write the top-up freeze first (topup-freeze)")
    prior = prior_spend(run)
    est = estimate_topup(run, plan, cumulative_cap_usd=cumulative_cap_usd,
                         prior_spend_usd=prior["total"], window_min=window_min)
    return {"event": "poc9_amendment", "amendment_id": "learning-v2-amendment-9",
            "mode": "topup", "owner_approved": False, "surface_sha256": SURFACE_SHA256,
            "hashes": pipeline.bound_hashes(root), "window_min": est["window_min"],
            "cap_usd": est["window_cap_usd"], "cumulative_cap_usd": cumulative_cap_usd,
            "prior_spend": prior, "teacher_cap_usd": 0.0,
            "request_limits": request_limits(), "estimate": est,
            "base_freeze_sha256": frozen["sha256"], "topup_freeze_sha256": plan["sha256"],
            "arms": [e["name"] for e in plan["arms"]], "skipped": plan["skipped"],
            "deviations": ["D3", "D4"] if any(e.get("amendment") == "D4"
                                              for e in plan["arms"]) else ["D3"]}


def preflight_run(registered: dict[str, Any], run: Path, *,
                  allow_window_record: bool = False) -> dict[str, Any]:
    """Top-up checks on the existing run (common checks are pipeline.preflight's)."""
    run = Path(run)
    if registered.get("mode") != "topup":
        raise ValueError("registration is not a top-up registration")
    frozen = pipeline.verify_freeze(run)
    if frozen["sha256"] != registered.get("base_freeze_sha256"):
        raise ValueError("base freeze differs from the registered one")
    plan = load_plan(run, frozen)
    if plan is None or plan["sha256"] != registered.get("topup_freeze_sha256"):
        raise ValueError("top-up freeze missing or differs from the registered one")
    for entry in plan["arms"]:
        check_eligible(run, frozen, plan, entry)
    out = run / TOPUP_DIR
    if out.exists() and not (allow_window_record and
                             {p.name for p in out.iterdir()} == {"window.json"}):
        raise FileExistsError("top-up output exists; one attempt only, no reruns")
    prior = prior_spend(run)
    if prior != registered.get("prior_spend"):
        raise ValueError("recorded prior spend differs from the registered one")
    est = estimate_topup(run, plan, cumulative_cap_usd=registered["cumulative_cap_usd"],
                         prior_spend_usd=prior["total"], window_min=registered["window_min"])
    if est != registered.get("estimate") or not est["within_cap"] or not est["fits_window"]:
        raise ValueError("registered top-up estimate does not fit the window/cumulative cap")
    return {"window_min": est["window_min"], "cap_usd": est["window_cap_usd"],
            "teacher_cap_usd": 0.0, "true_reservation_usd": est["reservation_usd"],
            "cumulative_cap_usd": est["cumulative_cap_usd"],
            "prior_spend_usd": prior["total"], "surface_sha256": SURFACE_SHA256}


# ------------------------------------------------------------------ the top-up stage

def stage(run: Path = pipeline.RUN_DIR, client: Any = None) -> dict[str, Any]:
    """Probe, then ONLY the frozen top-up arms in priority order, into <run>/topup."""
    from slh.poc9 import cases

    run = Path(run)
    frozen = pipeline.verify_freeze(run)
    plan = load_plan(run, frozen)
    if plan is None:
        raise FileNotFoundError("no top-up freeze")
    out = run / TOPUP_DIR
    if (out / "stage-started.json").exists() or (out / "results.jsonl").exists():
        raise FileExistsError("the top-up stage was already attempted")
    pipeline._write(out / "stage-started.json", {"started_epoch": time.time(),
                                                 "surface_sha256": SURFACE_SHA256,
                                                 "topup_freeze_sha256": plan["sha256"]})
    client = client or decide.AgentClient.for_window()
    pipeline._write(out / "probe.json", client.probe)
    registered = pipeline._read(out / "registration.json") \
        if (out / "registration.json").is_file() else {}
    arm_min = registered.get("estimate", {}).get("arms_min", {})
    first = plan["arms"][0]["name"] if plan["arms"] else None
    remaining = client.stage_deadline - time.time() if client.stage_deadline else math.inf
    if first and 60 * arm_min.get(first, 0) + decide.CASE_LIMIT_S > remaining:
        raise decide.StageDeadlineReached("the first top-up arm cannot fit the remaining window;"
                                          " sealed test was not opened")
    for entry in plan["arms"]:
        check_eligible(run, frozen, plan, entry)
    sampler = client.with_temperature(decide.SAMPLE_TEMPERATURE)
    failures: list[str] = []
    started = time.time()
    with pipeline._dispatcher_adapter(frozen["best_of_n"]):
        try:
            with cases.sealed_access("final_run_topup") as test:
                if [c["case_id"] for c in test] != frozen["test_case_ids"]:
                    raise ValueError("sealed membership/order differs from frozen manifest")
                for entry in plan["arms"]:
                    check_eligible(run, frozen, plan, entry)  # immediately before dispatch
                    arm = arms.Arm(entry["name"], entry["text"], entry["writer"],
                                   entry["snapshot"], entry["kind"])
                    wanted = set(entry["case_ids"])
                    subset = [c for c in test if c["case_id"] in wanted]
                    try:
                        pipeline.execute(arm, subset, client, out, sampler=sampler
                                         if arm.kind == "best_of_n" else None)
                    except pipeline.ArmFailed as exc:
                        failures.append(str(exc))
                        pipeline._write(out / "arm-failures.json", failures)
        finally:
            rows = pipeline._rows(out / "results.jsonl")
            window = pipeline._read(out / "window.json") if (out / "window.json").is_file() \
                else {}
            elapsed = time.time() - window.get("started_epoch", started)
            gpu = elapsed / 3600 * estimate.hourly_rate()
            prior = registered.get("prior_spend", {}).get("total")
            pipeline._write(out / "costs.json", {
                "schema": "poc9-topup-costs-v1", "teacher_usd": 0.0,
                "gpu_elapsed_s_before_cleanup": elapsed,
                "gpu_usd_before_cleanup_estimated": gpu,
                "arm_slot_s": {e["name"]: pipeline._slot_seconds(
                    [r for r in rows if r["arm"] == e["name"]]) for e in plan["arms"]},
                "sealed_s": time.time() - started, "prior_spend_usd": prior,
                "cumulative_usd_before_cleanup_estimated": None if prior is None
                else prior + gpu, "shutdown_cost_pending": True})
    result = pipeline.report(run)
    if failures:
        raise pipeline.ArmFailed("top-up arm(s) failed on every started case: " +
                                 "; ".join(failures))
    return result
