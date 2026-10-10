"""One bounded POC9 window and a free, descriptive paired-group report.

Use ``python -m slh.poc9.pipeline``. Model execution is only the ``stage``
command; preflight, caps/estimate and report are offline. The dispatcher is
from demo1, including its process-wide abandoned-request slot accounting.
"""
from __future__ import annotations

import contextlib
import json
import math
import statistics
import threading
import time
from pathlib import Path
from typing import Any

from slh.demo1 import decide
from slh.demo1.context import B1_PROMPT_SHA256, SURFACE_SHA256
from slh.demo1.schema import REPO
from slh.poc9 import arms, estimate
from slh.serve_spec import MAX_MODEL_LEN  # served --max-model-len; over it is HTTP 400
from slh.serving import ServingWindowClosed

RUN_DIR = REPO / "poc9/runs/w5p"
REGISTRATION = REPO / "poc9/registration.json"
WORKERS = 4
POWER_NOTE = ("Descriptive POC: 40 independent groups, three variants each; optimistic "
              "detectable gains are about 13–20 percentage points and clustering can "
              "raise them. One stream and one learning run per writer do not establish "
              "repeatability; bootstrap intervals are not a significance or power claim.")
_ADAPTER_LOCK = threading.Lock()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=1) + "\n", encoding="utf-8")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] \
        if path.is_file() else []


def _append(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _file_sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContextLimitExceeded(ValueError):
    """An arm prompt cannot fit the served context; never dispatched (no request)."""


class ArmFailed(RuntimeError):
    """Every started case of an arm ended in an error: a systematic failure."""


class CreditHalt(ServingWindowClosed):
    """Account-wide credit failure. A ServingWindowClosed, so demo1's dispatcher
    stops starting cases and re-raises it; no later arm of the window runs."""


CREDIT_HALT = "credit-halt.json"
# Process-wide request gate, one latch per stage GENERATION: a worker reads and
# sets only the latch of the generation it started in (an abandoned case from
# an older stage cannot halt a new one). Checked before EVERY model request:
# each candidate, selector and read turn, and last before the HTTP send.
_CREDIT_LOCK = threading.Lock()
_CREDIT_GEN = [0]
_CREDIT_LATCHED: dict[int, str] = {}


def credit_generation() -> int:
    return _CREDIT_GEN[0]


def credit_gate_check(gen: int | None = None) -> None:
    gen = _CREDIT_GEN[0] if gen is None else gen
    first = _CREDIT_LATCHED.get(gen)
    if first is not None:
        raise CreditHalt(f"credit stop latched; request not sent (first: {first})")


def latch_credit(message: str, gen: int | None = None) -> None:
    with _CREDIT_LOCK:
        _CREDIT_LATCHED.setdefault(_CREDIT_GEN[0] if gen is None else gen, message)


def _new_credit_generation() -> int:
    with _CREDIT_LOCK:
        _CREDIT_GEN[0] += 1
        return _CREDIT_GEN[0]


def _guard_backends(clients: list[Any], gen: int) -> None:
    """Bind this thread's backends to ``gen`` and wrap them once: the credit
    gate runs AFTER the serving check, as the last step before the HTTP send,
    and a credit error latches the gate before it unwinds."""
    for c in clients:
        backend = c.parts()[0]
        with contextlib.suppress(AttributeError, TypeError):
            backend._credit_gen = gen
        if getattr(backend, "_credit_gated", False):
            continue
        window = getattr(backend, "check_window", None)
        complete = getattr(backend, "complete", None)
        if window is not None:
            def gated(*a: Any, _window: Any = window, _b: Any = backend, **k: Any) -> Any:
                result = _window(*a, **k)  # may pause (serving check)
                credit_gate_check(getattr(_b, "_credit_gen", gen))
                return result
            backend.check_window = gated
        if complete is not None:
            def latched(*a: Any, _complete: Any = complete, _b: Any = backend, **k: Any) -> Any:
                try:
                    return _complete(*a, **k)
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    if credit_failure(message):
                        latch_credit(message, getattr(_b, "_credit_gen", gen))
                    raise
            backend.complete = latched
        with contextlib.suppress(AttributeError, TypeError):
            backend._credit_gated = True


def credit_failure(message: str) -> bool:
    """The credit stop's signature (slh.provider): HTTP 402 with credit wording,
    or any HTTP 402 (never a per-case outcome on the window's endpoint)."""
    from slh.provider import _statuses, classify_message

    return classify_message(message) == "provider_credit" or 402 in _statuses(message)


# Allowance for chat-template tokens and up to MAX_READS read results.
CONTEXT_MARGIN_TOKENS = 2048
_PROMPT_TOKENS: dict[str, int] = {}


def prompt_tokens(text: str) -> int:
    """Tokens of the rendered B1 system prompt with ``text`` as the note, plus tools."""
    key = arms.sha(text)
    if key not in _PROMPT_TOKENS:
        from slh.poc9 import loader, writers

        rendered = loader.render(text or None)
        system = rendered["system_prompt"]
        _PROMPT_TOKENS[key] = writers.count_tokens(
            system if isinstance(system, str) else json.dumps(system, sort_keys=True)) + \
            writers.count_tokens(json.dumps(rendered["tools"], sort_keys=True))
    return _PROMPT_TOKENS[key]


def context_need(text: str, messages: list[dict[str, Any]]) -> int:
    """Conservative request size: prompt + JSON case messages + output cap + margin."""
    from slh.poc9 import writers

    return prompt_tokens(text) + writers.count_tokens(json.dumps(messages, sort_keys=True)) + \
        decide.MAX_OUTPUT_TOKENS + CONTEXT_MARGIN_TOKENS


def arm_failed(rows: list[dict[str, Any]]) -> bool:
    """All started cases errored and not only by the case wall-clock limit."""
    started = [r for r in rows if not r.get("stage_deadline")]
    return bool(started) and all(r.get("error") for r in started) and \
        any(not r.get("case_time_limit") for r in started)


def _transport_case(case: dict[str, Any]) -> dict[str, Any]:
    # Extra metadata is for demo1's dispatcher/error rows, never added to messages.
    return {**case, "task": "address", "family": case["group"],
            "role": "act" if case["gold"]["type"] == "tool_call" else "clarify"}


def _decision(arm: arms.Arm, case: dict[str, Any], client: Any, sampler: Any = None,
              greedy: Any = None, deadline: float | None = None, *,
              schedules: dict[str, Any]) -> dict[str, Any]:
    from slh.poc9.score import score

    started = time.perf_counter()
    clients = [c for c in (client, sampler) if c is not None]
    gen = credit_generation()  # the generation this case started in
    decide._set_deadline(clients, deadline or time.time() + decide.CASE_LIMIT_S)
    try:
        _guard_backends(clients, gen)
        credit_gate_check(gen)
        seed = decide.b1_seed(client.config)
        if arm.notes and arm.kind != "best_of_n":
            need = context_need(arm.text, case["messages"])
            if need > MAX_MODEL_LEN:
                raise ContextLimitExceeded(
                    f"context length: about {need} tokens (prompt, case, {decide.MAX_OUTPUT_TOKENS}"
                    f" output, margin) exceed max_model_len {MAX_MODEL_LEN}; not dispatched")
        if arm.kind == "b0":
            # Model only: tau2's reference agent (phase-1 B0), no think tool or
            # instruction, no skill; same client, output cap, limits and seed.
            from slh.demo1.context import B0_PROMPT_SHA256, b0_config

            credit_gate_check(gen)
            result = decide.next_action(case, [], client, seed=seed, config=b0_config(),
                                        expect_prompt_sha256=B0_PROMPT_SHA256)
        elif arm.kind != "best_of_n":
            credit_gate_check(gen)
            result = decide.next_action(case, list(arm.notes), client, seed=seed,
                                        expect_prompt_sha256=B1_PROMPT_SHA256
                                        if not arm.notes else None)
        else:
            n = schedules[arm.writer]["attempts"][case["case_id"]]
            candidates = []
            for i in range(n):
                credit_gate_check(gen)
                candidates.append(decide.next_action(case, [], client if i == 0 else sampler,
                                  seed=decide.b1_seed(client.config, i),
                                  expect_prompt_sha256=B1_PROMPT_SHA256))
            picked = 0
            selections = []
            for i in range(1, n):
                credit_gate_check(gen)
                choice = decide.select(case, [candidates[picked], candidates[i]], client,
                                       seed=seed)
                selections.append(choice)
                if choice["chosen"] == 1:
                    picked = i
            costs = candidates + selections
            result = {"action": candidates[picked]["action"], "candidates": candidates,
                      "selections": selections, "attempts": n, "chosen": picked,
                      "selector_failure_rule": "unparseable reply selects hash-ordered A; "
                                               "transport failure records no decision",
                      "system_prompt_sha256": B1_PROMPT_SHA256,
                      "capped": any(c.get("capped") for c in costs),
                      **{k: sum(c.get(k, 0) or 0 for c in costs) for k in decide.COST_KEYS}}
        return {"arm": arm.name, "case_id": case["case_id"], "task": "address",
                "family": case["group"], "role": case["role"], "kind": case["kind"],
                "split": case["split"], **result,
                "wall_s": time.perf_counter() - started, **score(case, result["action"])}
    except ServingWindowClosed:
        raise
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if credit_failure(message):
            latch_credit(message, gen)
            raise CreditHalt(f"credit stop: {message}") from exc
        raise
    finally:
        decide._set_deadline(clients, None)


@contextlib.contextmanager
def _dispatcher_adapter(schedules: dict[str, Any], *, new_generation: bool = True):
    """Scoped process-local callbacks; reuse demo1's unchanged bounded loop.

    Its error rows also need the POC scorer (including confirm gold). Only one
    adapter may run per process. No source or frozen execution surface changes.
    """
    from slh.poc9.score import score

    if not _ADAPTER_LOCK.acquire(blocking=False):
        raise RuntimeError("a POC9 stage is already dispatching in this process")
    if new_generation:  # stage_window starts its own before any writer call
        _new_credit_generation()  # a recorded halt (credit-halt.json) still blocks execute
    original_case, original_score = decide.run_case, decide.score
    try:
        def run_case(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return _decision(*args, **kwargs, schedules=schedules)

        decide.run_case, decide.score = run_case, score
        yield
    finally:
        decide.run_case, decide.score = original_case, original_score
        _ADAPTER_LOCK.release()


def result_row(case: dict[str, Any], raw: dict[str, Any], arm: arms.Arm
               ) -> dict[str, Any]:
    from slh.poc9 import corrections, score

    row = {"schema": "poc9-result-v1", "arm": arm.name, "writer": arm.writer,
           "snapshot": arm.snapshot, "skill_sha256": arms.sha(arm.text) if arm.text else None,
           **{k: case[k] for k in ("case_id", "group", "variant", "kind", "split")},
           "action": raw["action"], "score": score.score(case, raw["action"]),
           "correction": corrections.explain(case, raw["action"])
                         if case["split"] == "train" else None,
           "surface_sha256": SURFACE_SHA256,
           "prompt_sha256": raw.get("system_prompt_sha256"),
           "tokens": {"prompt": raw.get("input_tokens", 0),
                      "completion": raw.get("output_tokens", 0)},
           "error": raw.get("error"), "reads": raw.get("reads", []),
           "context_limit": any(word in str(raw.get("error", "")).lower() for word in
                                ("context length", "context window", "max_model_len")),
           **{k: raw.get(k, 0) for k in (*decide.COST_KEYS, "wall_s")},
           **{k: raw[k] for k in ("capped", "case_time_limit", "timeout_after_retry",
                                  "abandoned", "stage_deadline", "attempts", "selection",
                                  "selections", "candidates", "reads") if k in raw}}
    if arm.reuse_b1:
        row.update(reused_b1=True, identical_to_b1_by_construction=True,
                   source_arm="b1", source_case_id=case["case_id"],
                   incremental_model_calls=0)
    return row


def execute(arm: arms.Arm, cases: list[dict[str, Any]], client: Any, run: Path, *,
            sampler: Any = None) -> list[dict[str, Any]]:
    raw_path = run / "raw" / f"{arm.name}.jsonl"
    if raw_path.exists():
        raise FileExistsError(f"{arm.name} already attempted; no outcome-based reruns")
    if (run / CREDIT_HALT).exists():
        raise CreditHalt(f"credit stop recorded in {run / CREDIT_HALT}; no further dispatch")
    credit_gate_check()
    if arm.notes and arm.kind != "best_of_n":
        prompt_tokens(arm.text)  # render once on this thread, not in the workers
    # Convert even on a deadline/fatal error so completed pairs remain reviewable.
    converted = []
    try:
        decide.run_arm(arm, [_transport_case(c) for c in cases], client, raw_path,
                       workers=WORKERS, sampler=sampler,
                       stage_deadline=client.stage_deadline)
    except CreditHalt as exc:
        _write(run / CREDIT_HALT, {"arm": arm.name, "time_epoch": time.time(),
                                   "error": str(exc), "owner": "credit-stop"})
        raise
    finally:
        by_id = {c["case_id"]: c for c in cases}
        converted = [result_row(by_id[r["case_id"]], r, arm) for r in _rows(raw_path)]
        _append(run / "results.jsonl", converted)
    if arm_failed(converted):
        first = next(r["error"] for r in converted if r.get("error"))
        raise ArmFailed(f"{arm.name}: every started case failed (first error: {first})")
    return converted


def full_context_history(train: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    """Render each training case in stream order with ITS OWN feedback row.

    Acquisition rows arrive in completion order (parallel workers), so they are
    joined by case_id, never zipped positionally."""
    by_id = {r["case_id"]: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {c["case_id"] for c in train}:
        raise ValueError("acquisition rows do not match the training cases one-to-one")
    return "Training histories and deterministic feedback:\n\n" + "\n\n".join(
        f"Episode {c['stream_index'] + 1}:\n{decide.render_transcript(c['messages'])}\n"
        + json.dumps({k: by_id[c["case_id"]][k]
                      for k in ("action", "score", "correction", "error", "reads")},
                     sort_keys=True) for c in train)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from slh.poc9.score import FIELDS, summarize

    result = summarize([r["score"] for r in rows], [r["kind"] for r in rows])
    result["denominators"] = {f: sum(r["score"][f] is not None for r in rows) for f in FIELDS}
    result["accuracy"] = result["all"]["correct"]
    return result


def _group_accuracy(rows: list[dict[str, Any]]) -> float:
    grouped: dict[str, list[int]] = {}
    for r in rows:
        grouped.setdefault(r["group"], []).append(int(r["score"]["correct"]))
    return statistics.mean(statistics.mean(v) for v in grouped.values())


def contrast(a: list[dict[str, Any]], b: list[dict[str, Any]], expected: list[str], *,
             samples: int = 10000) -> dict[str, Any]:
    def complete(rows: list[dict[str, Any]]) -> bool:
        return _exact(rows, expected)

    if not expected or not complete(a) or not complete(b):
        return {"status": "incomplete", "difference_points": None}
    bm = {r["case_id"]: r for r in b}
    if any(r["group"] != bm[r["case_id"]]["group"] for r in a):
        raise ValueError("paired case groups differ")
    # Follow demo1's seeded family bootstrap with one averaged row per group.
    # All variants remain together; floats retain equal group weights.
    def group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[int]] = {}
        for r in rows:
            grouped.setdefault(r["group"], []).append(int(r["score"]["correct"]))
        return [{"case_id": g, "family": g, "correct": statistics.mean(v)}
                for g, v in sorted(grouped.items())]

    # demo1 casts correctness to int; use its group-resampling algorithm directly
    # on floats here to retain group means, including unequal-size failure audits.
    import random

    ga, gb = group_rows(a), {r["case_id"]: r["correct"] for r in group_rows(b)}
    diffs = [r["correct"] - gb[r["case_id"]] for r in ga]
    rng = random.Random(0)
    draws = sorted(100 * statistics.mean(rng.choices(diffs, k=len(diffs)))
                   for _ in range(samples))
    return {"status": "complete", "difference_points": 100 * statistics.mean(diffs),
            "low": draws[int(.025 * samples)], "high": draws[int(.975 * samples) - 1],
            "samples": samples, "seed": 0, "unit": "group", "confidence": .95,
            "wins": sum(r["score"]["correct"] and not bm[r["case_id"]]["score"]["correct"]
                        for r in a),
            "losses": sum(not r["score"]["correct"] and bm[r["case_id"]]["score"]["correct"]
                          for r in a)}


def verify_freeze(run: Path) -> dict[str, Any]:
    frozen = _read(run / "artifacts/freeze.json")
    arms.verify(frozen)
    for name, expected in frozen.get("artifact_files_sha256", {}).items():
        if _file_sha(run / name) != expected:
            raise ValueError(f"frozen artifact changed: {name}")
    return frozen


SCORE_KEYS = ("all", "by_kind", "accuracy", "group_accuracy")


def all_errors(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(r.get("error") for r in rows)


def _exact(rows: list[dict[str, Any]], expected: list[str]) -> bool:
    """Every expected case once, none cut or over context, and the arm did not
    fail (all started cases errors) nor consist of errors only."""
    return sorted(r["case_id"] for r in rows) == sorted(expected) and \
        not any(r.get("stage_deadline") or r.get("context_limit") for r in rows) and \
        not arm_failed(rows) and not all_errors(rows)


def conceal(output: dict[str, Any]) -> dict[str, Any]:
    """Drop every score-bearing value until coverage is complete (D3 mapping
    and top-up included). Coverage, errors and accounting stay visible."""
    for entry in [*output["arms"].values(), *output.get("reference", {}).values()]:
        if "b1_minus_b0" in entry:
            entry["b1_minus_b0"] = {"status": "concealed"}
        for key in SCORE_KEYS:
            if key in entry:
                entry[key] = None
        entry["scores"] = "concealed"
    for writer, entry in list(output["writers"].items()):
        if "learned_48_minus_b1" in entry:
            output["writers"][writer] = {"status": "concealed until coverage is complete",
                                         "curve_status": entry["curve_status"],
                                         "best_of_n_budget": entry["best_of_n_budget"]}
    output["concealed"] = True
    return output


def report(run: Path = RUN_DIR) -> dict[str, Any]:
    """Descriptive report; with an approved top-up plan (artifacts/topup-freeze.json)
    the D3 identities, skipped arms and top-up rows are applied, all re-verified."""
    from slh.poc9 import topup

    frozen = verify_freeze(run)
    defined = arms.verify(frozen)
    rows = [r for r in _rows(run / "results.jsonl") if r["split"] == "test"]
    expected = frozen["test_case_ids"]
    by = {a.name: [r for r in rows if r["arm"] == a.name] for a in defined}
    plan = topup.load_plan(run, frozen)
    skipped: dict[str, str] = {}
    plan_errors: dict[str, str] = {}
    if plan is not None:
        by, plan_errors = topup.effective_rows(run, plan, by)
        skipped = plan["skipped"]
    costs = _read(run / "costs.json") if (run / "costs.json").exists() else {}
    output: dict[str, Any] = {"schema": "poc9-report-v1", "status": "descriptive",
        "power_note": POWER_NOTE, "arms": {}, "writers": {}, "costs": costs,
        "rejected_patches": frozen.get("rejected_patches", {}),
        "writer_status": frozen.get("writer_status",
                                    {w: {"status": "active"} for w in arms.WRITERS}),
        "freeze_sha256": frozen["sha256"], "removal_note": frozen["removal_note"],
        "human_note": "policy-only drafted procedure approved by owner; "
                      "not an independently human-authored control"}
    if plan is not None:
        top_costs = run / topup.TOPUP_DIR / "costs.json"
        output["topup"] = {"freeze_sha256": plan["sha256"], "identities": plan["identities"],
                           "skipped": skipped, "arms": [{k: e[k] for k in e if k != "text"}
                                                        for e in plan["arms"]],
                           "costs": _read(top_costs) if top_costs.is_file() else {},
                           "errors": plan_errors}
    for arm in defined:
        rs = by[arm.name]
        entry = output["arms"][arm.name] = {**_summary(rs),
            "coverage": {"rows": len(rs), "expected": len(expected),
                         "exact": _exact(rs, expected)},
            "errors": sum(bool(r.get("error")) for r in rs),
            "context_limit_errors": sum(bool(r.get("context_limit")) for r in rs),
            "all_started_cases_failed": arm_failed(rs),
            "identical_to_b1_by_construction": arm.reuse_b1,
            "group_accuracy": _group_accuracy(rs) if rs else None,
            "gpu_time_share": costs.get("arm_gpu_time_share", {}).get(arm.name, 0),
            "gpu_usd_estimated": costs.get("arm_gpu_time_share", {}).get(arm.name, 0) *
                costs.get("gpu_usd_through_verified_stop_estimated",
                          costs.get("gpu_usd_before_cleanup_estimated", 0)),
            "accounting": {k: sum(r.get(k, 0) or 0 for r in rs)
                           for k in (*decide.COST_KEYS, "wall_s")}}
        if arm.name in skipped:
            entry["coverage"]["skipped"] = skipped[arm.name]
        if arm.name in plan_errors:
            # Stale top-up premise: fail closed, never substitute or count it.
            entry["coverage"]["exact"] = False
            entry["topup_error"] = plan_errors[arm.name]
        sources = {r.get("identity_source") for r in rs} - {None}
        if sources:
            entry["identity_source"] = sorted(sources)
    output["failed_arms"] = sorted(n for n, v in output["arms"].items()
                                   if v["all_started_cases_failed"])
    for writer, status in output["writer_status"].items():
        if status["status"] != "active":
            output["writers"][writer] = {**status, "arms": "dropped before acquisition; "
                                         "no learned, generic, best-of-N or removal arms"}
            continue
        endpoint = by[f"learned_{writer}@48"]
        primary = contrast(endpoint, by["b1"], expected)
        points = (0, *arms.SNAPSHOTS)
        names = ["b1", *(f"learned_{writer}@{n}" for n in arms.SNAPSHOTS)]
        complete = all(output["arms"][name]["coverage"]["exact"] and
                       not any(r.get("stage_deadline") for r in by[name]) for name in names)
        values = [_group_accuracy(by[name]) if by[name] else None for name in names]
        slope = None
        if complete:
            mx, my = statistics.mean(points), statistics.mean(values)
            slope = 100 * sum((x - mx) * (y - my) for x, y in zip(points, values)) / \
                sum((x - mx) ** 2 for x in points)
        comparisons = {control: contrast(endpoint, by[control], expected) for control in (
            f"generic_{writer}", "full_context", f"best_of_n_{writer}", "human",
            f"removal_{writer}")}
        generic_delta = comparisons[f"generic_{writer}"]["difference_points"]
        signal = (complete and slope > 0 and values[-1] > values[0] and
                  values[-1] > values[1] and generic_delta is not None and generic_delta > 0)
        output["writers"][writer] = {"learned_48_minus_b1": primary,
            "curve": dict(zip(map(str, points), values)),
            "slope_points_per_episode": slope, "curve_status": "complete" if complete
                                                                else "incomplete",
            "regressions": [{"from": x, "to": y} for x, y, a, b in
                            zip(points, points[1:], values, values[1:])
                            if a is not None and b is not None and b < a],
            "descriptive_learning_signal": bool(signal), "controls": comparisons,
            "best_of_n_budget": frozen["best_of_n"][writer]}
    if plan_errors:
        output["errors"] = plan_errors
    from slh.poc9 import topup2

    plan2 = topup2.load_plan(run, frozen)
    if plan2 is not None:
        # D5: descriptive reference row; never part of coverage or learning fields.
        b0 = topup2.rows(run)
        output["reference"] = {"b0": {**_summary(b0),
            "coverage": {"rows": len(b0), "expected": len(expected),
                         "exact": _exact(b0, expected)},
            "errors": sum(bool(r.get("error")) for r in b0),
            "all_started_cases_failed": arm_failed(b0),
            "group_accuracy": _group_accuracy(b0) if b0 else None,
            "system_prompt_sha256": plan2["arms"][0]["system_prompt_sha256"],
            "freeze_sha256": plan2["sha256"], "note": plan2["note"],
            "b1_minus_b0": contrast(by["b1"], b0, expected),
            "accounting": {k: sum(r.get(k, 0) or 0 for r in b0)
                           for k in (*decide.COST_KEYS, "wall_s")}}}
    required = [v for n, v in output["arms"].items() if n not in skipped or n in plan_errors]
    output["coverage_status"] = "complete" if all(v["coverage"]["exact"] for v in required) \
        else "incomplete"
    if output["coverage_status"] != "complete":
        conceal(output)
    _write(run / "report.json", output)
    return output


def _bounded_call(fn: Any, client: Any) -> Any:
    """Serial writer/teacher call sharing demo1's abandoned worker slots.

    A stuck transport stops learning; it cannot hide behind a retry/repair.
    The process watchdog is the final bound, including teacher transports.
    """
    while decide.abandoned_in_flight() >= WORKERS:
        if client.stage_deadline is not None and time.time() + decide.CASE_LIMIT_S > \
                client.stage_deadline:
            raise decide.StageDeadlineReached("writer cannot finish before stage deadline")
        time.sleep(decide.POLL_S)
    if client.stage_deadline is not None and time.time() + decide.CASE_LIMIT_S > \
            client.stage_deadline:
        raise decide.StageDeadlineReached("writer cannot finish before stage deadline")
    deadline = time.time() + decide.CASE_LIMIT_S
    box: dict[str, Any] = {}
    gen = credit_generation()
    credit_gate_check(gen)

    def target() -> None:
        decide._set_deadline([client], deadline)
        try:
            _guard_backends([client], gen)  # writer backends latch and gate too
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - returned to caller, including interrupts
            box["error"] = exc
        finally:
            decide._set_deadline([client], None)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(decide.CASE_LIMIT_S + decide.ABANDON_GRACE_S)
    if thread.is_alive():
        decide._abandon(thread)
        raise decide.CaseTimeLimit("writer call abandoned; stop the stage")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _chat(chat: Any, client: Any, log: list[dict[str, Any]], *, role: str,
          ceiling: dict[str, int]) -> Any:
    def call(system: str, user: str, purpose: str = "write") -> Any:
        from slh.poc9 import writers

        if ceiling["calls"] >= writers.WRITER_CALL_CEILING or \
                ceiling["output_tokens"] >= writers.WRITER_OUTPUT_TOKEN_CEILING:
            raise RuntimeError("equal per-writer call/token ceiling reached")
        ceiling["calls"] += 1
        started = time.perf_counter()
        gen = credit_generation()
        value = _bounded_call(lambda: chat(system, user, purpose=purpose), client)
        ceiling["output_tokens"] += value[1].get("output_tokens", 0)
        log.append({"role": role, "purpose": purpose, "wall_s": time.perf_counter() - started,
                    "error": value[2], **value[1]})
        if value[2] and credit_failure(str(value[2])):  # _ask returns errors, not raises
            latch_credit(str(value[2]), gen)
        credit_gate_check(gen)  # a credit error on this call halts the stage
        return value
    return call


def _slot_seconds(rows: list[dict[str, Any]]) -> float:
    return sum(float(r.get("wall_s", 0) or 0) for r in rows)


def _matched_schedules(common: list[dict[str, Any]], logs: dict[str, list[dict[str, Any]]],
                       validation_rows: dict[str, list[dict[str, Any]]],
                       teacher_usd: dict[str, float], test_ids: list[str],
                       latency: dict[str, Any],
                       active: list[str] | tuple[str, ...] = arms.WRITERS) -> dict[str, Any]:
    gpu_slot_usd = estimate.hourly_rate() / 3600 / WORKERS
    # Include all continuations in the measured empty-skill training decisions.
    attempt_s = statistics.mean(float(r.get("wall_s", 0) or 0) for r in common)
    if attempt_s <= 0:
        attempt_s = estimate.call_seconds(latency["median_prompt_tokens"], 120, latency)
    selector_s = estimate.call_seconds(latency["median_prompt_tokens"] + 300, 5, latency)
    acquisition = _slot_seconds(common) * gpu_slot_usd
    schedules = {}
    for w in active:
        # Writing is serial: the full sandbox is allocated to that call,
        # including GPU idle time while the teacher answers.
        writer_gpu = _slot_seconds(logs[w])
        validation_gpu = _slot_seconds(validation_rows[w])
        learning = acquisition + writer_gpu * gpu_slot_usd * WORKERS + \
            validation_gpu * gpu_slot_usd + teacher_usd[w]
        # The endpoint inference is a declared pre-test proxy (same cases,
        # extra 400-token skill). Actual differences are disclosed in report.
        endpoint_s = attempt_s + latency["c_s_per_input_token"] * 400
        budget = learning + len(test_ids) * endpoint_s * gpu_slot_usd
        schedule = arms.budget_schedule(test_ids, matched_usd=budget,
            attempt_usd=attempt_s * gpu_slot_usd, selector_usd=selector_s * gpu_slot_usd)
        schedule.update(learning_usd=learning, acquisition_usd=acquisition,
                        teacher_usd=teacher_usd[w], endpoint_proxy_usd=len(test_ids) *
                        endpoint_s * gpu_slot_usd, proxy="demo1 measured latency + common "
                        "training wall time; four-slot decisions, serial writer/teacher waiting",
                        sampling="first candidate greedy; extras temperature 1, paired seeds",
                        planned_match_relative_error=abs(schedule["scheduled_usd"] - budget) /
                                                     budget)
        schedule["matched"] = schedule["planned_match_relative_error"] <= .05
        schedules[w] = schedule
    return schedules


def stage_window(run: Path = RUN_DIR, client: Any = None, *, teacher: Any = None,
                 teacher_config: dict[str, Any] | None = None,
                 latency: dict[str, Any] | None = None) -> dict[str, Any]:
    """probe -> common B1 evidence -> T/S/TS -> hash freeze -> randomized test.

    Generic controls are written before evidence exposure. The builder-B
    learning callback receives cached common B1 decisions, so writers consume
    identical evidence. Validation is a gate only, never writer evidence.
    Inject client/teacher in tests; this function never creates a sandbox.
    """
    from slh.poc9 import cases, learn, writers

    run = Path(run)
    if (run / "stage-started.json").exists() or (run / "results.jsonl").exists():
        raise FileExistsError("this one-window stage was already attempted")
    _write(run / "stage-started.json", {"started_epoch": time.time(),
                                       "surface_sha256": SURFACE_SHA256})
    # This stage's credit generation, before any writer call; kept by the adapter.
    _new_credit_generation()
    latency = latency or estimate.measured_model()
    client = client or decide.AgentClient.for_window()  # capped, bounded probe once
    _write(run / "probe.json", client.probe)
    teacher = teacher or writers.TeacherChat(teacher_config or writers.load_teacher())
    if hasattr(teacher, "log_to"):
        teacher.log_to(run / "teacher-calls.jsonl")
    logs = {w: [] for w in arms.WRITERS}
    ceilings = {w: {"calls": 0, "output_tokens": 0} for w in arms.WRITERS}
    objects = {w: writers.make_writer(w,
        agent=_chat(writers.agent_chat(client), client, logs[w], role="agent",
                    ceiling=ceilings[w]),
        teacher=_chat(teacher, client, logs[w], role="teacher", ceiling=ceilings[w]))
        for w in arms.WRITERS}
    generic: dict[str, str] = {}
    writer_status: dict[str, dict[str, Any]] = {}
    generic_teacher_start = float(teacher.spent_usd)
    for w in arms.WRITERS:
        patch = writers.generic_note(objects[w])
        _write(run / "artifacts" / f"generic-{w}.json", patch.to_dict())
        if not patch.ok or not patch.text:
            # A writer-specific required control failed: drop only that writer's
            # arms (learning, snapshots, generic, best-of-N, removal). B1 and the
            # shared controls still run; failures there abort the window.
            writer_status[w] = {"status": "control_failed", "control": f"generic_{w}",
                                "reason": "; ".join(patch.problems) or "no generic note text"}
            continue
        generic[w] = patch.text
        writer_status[w] = {"status": "active"}
    active = [w for w in arms.WRITERS if w in generic]
    _write(run / "artifacts/writer-status.json", writer_status)
    if not active:
        raise ValueError("every writer's required generic control failed; nothing to compare")
    generic_teacher_usd = float(teacher.spent_usd) - generic_teacher_start
    generic_logs = {w: list(logs[w]) for w in arms.WRITERS}
    for w in arms.WRITERS:
        logs[w].clear()
    train, validation = cases.load("train"), cases.load("validation")
    if len(train) != 48 or len(validation) != 12:
        raise ValueError("POC9 requires 48 train and 12 validation cases")
    manifest = cases.manifest()
    test_ids = next(v["case_ids"] for k, v in manifest["files"].items()
                    if k.endswith("/test.jsonl"))
    if len(test_ids) != 120:
        raise ValueError("POC9 requires 120 committed test cases")
    snapshots, summaries, rejected = {}, {}, {}
    validation_rows = {w: [] for w in arms.WRITERS}
    teacher_usd = {}
    # One scoped adapter covers acquisition, validation and all sealed arms.
    schedules: dict[str, Any] = {}
    started = time.time()
    with _dispatcher_adapter(schedules, new_generation=False):
        common = execute(arms.Arm("acquisition", snapshot=0), train, client, run)
        raw_common = {r["case_id"]: r for r in _rows(run / "raw/acquisition.jsonl")}
        base_validation = execute(arms.Arm("validation_b1"), validation, client, run)
        for w in active:
            writer = objects[w]
            propose = writer.propose_patch
            cache = {arms.sha(""): base_validation}
            candidate_index = [0]
            teacher_start = float(teacher.spent_usd)
            validation_rows[w].extend(base_validation)

            def propose_with_validation(current: str, evidence: Any, *, w: str = w,
                                        propose: Any = propose, cache: Any = cache,
                                        candidate_index: Any = candidate_index) -> Any:
                patch = propose(current, evidence)
                if patch.ok and patch.changed:
                    candidate_index[0] += 1
                    key = arms.sha(patch.text)
                    if key not in cache:
                        cache[key] = execute(arms.Arm(f"validation_{w}_{candidate_index[0]}",
                                                     patch.text), validation, client, run)
                        validation_rows[w].extend(cache[key])
                    before, after = cache[arms.sha(current)], cache[key]
                    good = sum(r["score"]["correct"] for r in after) >= \
                        sum(r["score"]["correct"] for r in before)
                    safe = sum(r["score"]["wrong_write"] for r in after) <= \
                        sum(r["score"]["wrong_write"] for r in before)
                    if not good or not safe:
                        patch.ok = False
                        patch.problems.append("validation gate: accuracy fell or wrong writes rose")
                return patch

            writer.propose_patch = propose_with_validation
            critic = _chat(teacher, client, logs[w], role="teacher", ceiling=ceilings[w])
            summaries[w] = learn.run_learning(writer, cases=train,
                decide=lambda case, skill: raw_common[case["case_id"]], critic=critic,
                out=run / "learning" / w)
            teacher_usd[w] = float(teacher.spent_usd) - teacher_start
            summaries[w]["teacher_usd"] = teacher_usd[w]
            snapshots[w] = {n: (run / "learning" / w / "snapshots" / f"skill-{n}.md")
                            .read_text(encoding="utf-8") for n in (0, *arms.SNAPSHOTS)}
            rejected[w] = [r for r in _rows(run / "learning" / w / "patches.jsonl")
                           if not r["accepted"] and not r["no_change"]]
            _write(run / "learning" / w / "pipeline-calls.json", logs[w])
        schedules.update(_matched_schedules(common, logs, validation_rows, teacher_usd,
                                            test_ids, latency, active))
        history = full_context_history(train, common)
        defined = arms.definitions(snapshots, generic, history, active=active)
        # Use the derived schedules for a second fit check, before sealed access.
        final_plan = estimate.estimate(model=latency,
            attempts_mean=statistics.mean(s["mean_attempts_frozen"] for s in schedules.values()),
            history_tokens=writers.count_tokens(history))
        test_s = 1.5 * sum(final_plan["stages"][s]["wall_s"] for s in ("sealed", "best_of_n"))
        remaining_s = client.stage_deadline - time.time() if client.stage_deadline else math.inf
        frozen = arms.freeze(defined, schedules)
        frozen.update(test_case_ids=test_ids, manifest_sha256=arms.sha(arms._canonical(manifest)),
                      rejected_patches=rejected, learning_summaries=summaries,
                      active_writers=active, writer_status=writer_status,
                      writer_ceilings={"calls_including_generic": writers.WRITER_CALL_CEILING,
                                       "output_tokens_including_generic":
                                           writers.WRITER_OUTPUT_TOKEN_CEILING},
                      full_context_tokens=writers.count_tokens(history),
                      # Arms over this never dispatch (context_limit rows, no request).
                      context_prompt_tokens={a.name: prompt_tokens(a.text) for a in defined
                                             if a.notes and a.kind != "best_of_n"},
                      max_model_len=MAX_MODEL_LEN,
                      sealed_estimated_s=test_s,
                      artifact_files_sha256={p.relative_to(run).as_posix(): _file_sha(p)
                          for p in sorted(run.rglob("*")) if p.is_file()
                          and p.name != "results.jsonl" and p.name != "freeze.json"})
        frozen["sha256"] = arms.sha(arms._canonical({k: v for k, v in frozen.items()
                                                   if k != "sha256"}))
        _write(run / "artifacts/freeze.json", frozen)
        verify_freeze(run)
        if test_s + decide.CASE_LIMIT_S > remaining_s:
            raise decide.StageDeadlineReached("frozen complete arm plan cannot fit the remaining "
                                              "window; sealed test was not opened")
        by_name = {a.name: a for a in defined}
        sampler = client.with_temperature(decide.SAMPLE_TEMPERATURE)
        sealed_started = time.time()
        failures: list[str] = []
        try:
            with cases.sealed_access("final_run") as test:
                if [c["case_id"] for c in test] != test_ids:
                    raise ValueError("sealed membership/order differs from frozen manifest")
                b1_rows = []
                for name in frozen["arm_order"]:
                    try:
                        rows = execute(by_name[name], test, client, run, sampler=sampler
                                       if by_name[name].kind == "best_of_n" else None)
                    except ArmFailed as exc:
                        # Loud, recorded, and the window goes on to the next arm.
                        failures.append(str(exc))
                        _write(run / "arm-failures.json", failures)
                        rows = [r for r in _rows(run / "results.jsonl") if r["arm"] == name]
                    if name == "b1":
                        b1_rows = rows
                by_case = {c["case_id"]: c for c in test}
                for a in defined:
                    if a.reuse_b1:
                        reused = [result_row(by_case[r["case_id"]],
                            {**r, "system_prompt_sha256": r["prompt_sha256"]}, a)
                                  for r in b1_rows]
                        _append(run / "results.jsonl", reused)
        finally:
            all_rows = _rows(run / "results.jsonl")
            real_rows = [r for r in all_rows if not r.get("reused_b1")]
            total_slot_s = _slot_seconds(real_rows) + sum(_slot_seconds(logs[w]) +
                _slot_seconds(generic_logs[w]) for w in arms.WRITERS)
            # Allocate the complete elapsed GPU window (including startup/idle)
            # proportionally to measured busy-slot time; costs are estimates,
            # pending provider lifetime reconciliation after verified stop.
            window_record = _read(run / "window.json") if (run / "window.json").is_file() else {}
            admission = window_record.get("started_epoch", started)
            elapsed = time.time() - admission
            costs = {"schema": "poc9-costs-v1", "teacher_usd": float(teacher.spent_usd),
                "generic_teacher_usd": generic_teacher_usd, "teacher_by_writer_usd": teacher_usd,
                "gpu_elapsed_s_before_cleanup": elapsed,
                "gpu_usd_before_cleanup_estimated": elapsed / 3600 * estimate.hourly_rate(),
                "gpu_reservation_usd": estimate.reservation(window_record.get("window_min", 180)),
                "time_share_method": "busy wall seconds normalized across measured calls; "
                                     "shared acquisition counted once in actual spend, "
                                     "charged to each standalone budget comparison",
                "arm_gpu_time_share": {a.name: _slot_seconds([r for r in real_rows
                    if r["arm"] == a.name]) / total_slot_s if total_slot_s else 0 for a in defined},
                "learning_gpu_time_share": (_slot_seconds([r for r in real_rows
                    if r["split"] != "test"]) + sum(_slot_seconds(logs[w])
                    for w in arms.WRITERS)) / total_slot_s if total_slot_s else 0,
                "generic_gpu_time_share": sum(_slot_seconds(generic_logs[w])
                    for w in arms.WRITERS) / total_slot_s if total_slot_s else 0,
                "learning_s": sealed_started - started, "sealed_s": time.time() - sealed_started,
                "planned_best_of_n": schedules, "shutdown_cost_pending": True}
            for w in schedules:
                actual_rows = [r for r in real_rows if r["arm"] == f"best_of_n_{w}"]
                learned_rows = [r for r in real_rows if r["arm"] == f"learned_{w}@48"]
                share_rate = estimate.hourly_rate() / 3600 / WORKERS
                best_cost = _slot_seconds(actual_rows) * share_rate
                learned_cost = schedules[w]["learning_usd"] + _slot_seconds(learned_rows) * share_rate
                complete = len(actual_rows) == len(learned_rows) == len(test_ids)
                costs.setdefault("actual_budget_comparisons", {})[w] = {
                    "best_of_n_usd": best_cost, "learned_amortized_usd": learned_cost,
                    "relative_difference": (best_cost - learned_cost) / learned_cost
                                           if learned_cost else None,
                    "status": "incomplete" if not complete else "matched" if
                        abs(best_cost - learned_cost) <= .05 * learned_cost else "unmatched",
                    "note": "measured workload time estimates before shutdown; actual differences "
                            "descriptive; unmatched supports no efficiency claim"}
            _write(run / "costs.json", costs)
    result = report(run)
    if failures:
        raise ArmFailed("arm(s) failed on every started case: " + "; ".join(failures))
    return result


def bound_hashes(root: Path = REPO) -> dict[str, str]:
    """Use demo1's code-binding pattern without its two-window registration."""
    from slh.demo1.preflight import bound_files

    names = bound_files(root) + [p.relative_to(root).as_posix()
        for p in (root / "slh/poc9").glob("*.py")] + [
        "scripts/w5p.sh", "docs/poc9/schema.md", "docs/poc9/pricing-correction.md",
        "docs/protocol-amendment-9.md", "poc9/manifest.json", "configs/poc9-teacher.yaml",
        "demo1/human-procedure-address.md"]
    return {n: _file_sha(root / n) for n in sorted(set(names))}


def registration_template(*, root: Path = REPO, window_min: int = 180,
                          cap_usd: float = 14.70, teacher_cap_usd: float = 1.50,
                          attempts_mean: float = 2.0) -> dict[str, Any]:
    from slh.demo1.preflight import request_limits
    from slh.poc9 import writers
    from slh.serving import build_allowance_usd

    fixed = build_allowance_usd(root / "configs/prices.yaml")
    plan = estimate.estimate(window_min=window_min, cap_usd=cap_usd,
        teacher_cap_usd=teacher_cap_usd, fixed_usd=fixed, attempts_mean=attempts_mean)
    return {"event": "poc9_amendment", "amendment_id": "learning-v2-amendment-9",
            "owner_approved": False, "surface_sha256": SURFACE_SHA256,
            "hashes": bound_hashes(root), "window_min": window_min,
            "cap_usd": cap_usd, "teacher_cap_usd": teacher_cap_usd,
            "request_limits": request_limits(), "estimate": plan,
            "writer_call_ceiling": writers.WRITER_CALL_CEILING,
            "writer_output_token_ceiling": writers.WRITER_OUTPUT_TOKEN_CEILING}


def preflight(run: Path = RUN_DIR, *, registration: Path = REGISTRATION,
              root: Path = REPO, allow_window_record: bool = False,
              topup: bool = False, topup2: bool = False) -> dict[str, Any]:
    """Free checks, including corrected cap, seal, APIs and frozen surface.

    ``topup``: the registration must be a top-up one for the SAME frozen run
    (slh.poc9.topup.preflight_run: freezes, D3, prior spend, cumulative cap).

    A reviewable registration JSON binds this separate POC. It is not an
    amendment-8 registration and does not mutate the learner registry.
    """
    import stat

    from slh.demo1.context import b1_context
    from slh.demo1.preflight import request_limits
    from slh.execution import current_surface
    from slh.poc9 import cases, learn, loader, writers

    registered = _read(registration)
    if registered.get("event") != "poc9_amendment" or \
            registered.get("amendment_id") != "learning-v2-amendment-9" or \
            registered.get("owner_approved") is not True:
        raise ValueError("POC9 registration must record the owner's approved cap and hashes")
    if current_surface(root)["sha256"] != SURFACE_SHA256 or \
            registered.get("surface_sha256") != SURFACE_SHA256:
        raise ValueError("execution surface drifted")
    if registered.get("hashes") != bound_hashes(root):
        raise ValueError("registered code, prices note, teacher or case manifest hashes differ")
    if _read(root / "demo1/b1-context.json") != b1_context():
        raise ValueError("B1 rendered prompt/tools differ")
    if registered.get("request_limits") != request_limits():
        raise ValueError("registered request limits differ")
    if (registered.get("mode") == "topup") != topup or \
            (registered.get("mode") == "topup2") != topup2:
        raise ValueError("registration mode and window mode differ")
    if topup2:
        from slh.poc9 import topup2 as top2

        if not cases.verify()["ok"]:
            raise ValueError("POC9 cases do not regenerate to their committed manifest")
        if stat.S_IMODE((cases.SEALED_DIR / "cases").stat().st_mode) != 0:
            raise ValueError("test case directory is not sealed mode 000")
        return top2.preflight_run(registered, run, allow_window_record=allow_window_record)
    if topup:
        from slh.poc9 import topup as top

        if not cases.verify()["ok"]:
            raise ValueError("POC9 cases do not regenerate to their committed manifest")
        if stat.S_IMODE((cases.SEALED_DIR / "cases").stat().st_mode) != 0:
            raise ValueError("test case directory is not sealed mode 000")
        return top.preflight_run(registered, run, allow_window_record=allow_window_record)
    if run.exists() and not (allow_window_record and
                             {p.name for p in run.iterdir()} == {"window.json"}):
        raise FileExistsError("window output exists; one attempt only, no reruns")
    if not cases.verify()["ok"]:
        raise ValueError("POC9 cases do not regenerate to their committed manifest")
    if stat.S_IMODE((cases.SEALED_DIR / "cases").stat().st_mode) != 0:
        raise ValueError("test case directory is not sealed mode 000")
    for split, expected in (("train", 48), ("validation", 12)):
        if len(cases.load(split)) != expected:
            raise ValueError(f"{split} case count differs")
    if not all(callable(f) for f in (learn.run_learning, writers.make_writer,
                                    writers.generic_note, loader.decide)):
        raise ValueError("builder-B interfaces unavailable")
    teacher = writers.load_teacher(root / "configs/poc9-teacher.yaml")
    teacher_cap = float(teacher["cap_usd"])
    if not math.isfinite(teacher_cap) or teacher_cap < 0 or \
            teacher_cap != registered["teacher_cap_usd"]:
        raise ValueError("teacher cap differs from registered window allowance")
    if teacher.get("max_tokens") != decide.MAX_OUTPUT_TOKENS:
        raise ValueError("teacher and actor must have the same 4096-token ceiling")
    if registered.get("writer_call_ceiling") != writers.WRITER_CALL_CEILING or \
            registered.get("writer_output_token_ceiling") != writers.WRITER_OUTPUT_TOKEN_CEILING:
        raise ValueError("registered equal writer ceilings differ")
    from slh.serving import build_allowance_usd

    plan = estimate.estimate(window_min=registered["window_min"],
        teacher_cap_usd=teacher_cap, cap_usd=registered["cap_usd"],
        fixed_usd=build_allowance_usd(root / "configs/prices.yaml"),
        attempts_mean=registered["estimate"]["best_of_n_mean_attempts"])
    if registered["estimate"] != plan or not plan["within_cap"] or not plan["fits_window"]:
        raise ValueError("corrected registered estimate does not fit the window/cap")
    return {"window_min": plan["window_min"], "cap_usd": plan["cap_usd"],
            "teacher_cap_usd": teacher_cap, "true_reservation_usd": plan["reservation_usd"],
            "surface_sha256": SURFACE_SHA256}


DEVIATIONS = REPO / "docs/poc9/deviations.md"
_DEVIATIONS_HEADER = ("# POC9 deviations\n\nOne line per deviation from the registered "
                      "protocol, appended by `python -m slh.poc9.pipeline archive-run`.\n\n")


def archive_run(reason: str, note: str = "", *, run: Path = RUN_DIR,
                deviations: Path = DEVIATIONS) -> Path:
    """Rename an aborted run dir to ``<run>.attempt<N>-<reason>`` and log a deviation.

    Never deletes anything; the next window then starts from an absent run dir."""
    import datetime as dt
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", reason.lower()).strip("-")
    if not slug:
        raise ValueError("archive-run needs a non-empty --reason")
    if not run.is_dir():
        raise FileNotFoundError(f"no run directory to archive at {run}")
    taken = [int(m.group(1)) for p in run.parent.iterdir()
             if (m := re.fullmatch(re.escape(run.name) + r"\.attempt(\d+)-.*", p.name))]
    target = run.with_name(f"{run.name}.attempt{max(taken, default=0) + 1}-{slug}")
    run.rename(target)
    text = deviations.read_text(encoding="utf-8") if deviations.is_file() else _DEVIATIONS_HEADER
    number = len(re.findall(r"^- D\d+ ", text, flags=re.MULTILINE)) + 1
    try:
        where = f"{run.relative_to(REPO)} -> {target.relative_to(REPO)}"
    except ValueError:
        where = f"{run} -> {target}"
    line = (f"- D{number} ({dt.datetime.now(dt.UTC).date().isoformat()}): "
            f"{note.strip() or reason.strip()} Archived {where}.\n")
    deviations.parent.mkdir(parents=True, exist_ok=True)
    deviations.write_text(text + line, encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="python -m slh.poc9.pipeline")
    cmd = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "stage", "report", "finalize-costs", "topup-stage",
                 "topup2-stage", "topup2-freeze"):
        sub = cmd.add_parser(name)
        sub.add_argument("--run", type=Path, default=RUN_DIR)
        if name in ("preflight", "stage", "topup-stage", "topup2-stage"):
            sub.add_argument("--registration", type=Path, default=REGISTRATION)
        if name in ("preflight", "finalize-costs"):
            sub.add_argument("--topup", action="store_true")
            sub.add_argument("--topup2", action="store_true")
    sub = cmd.add_parser("topup-freeze")
    sub.add_argument("--run", type=Path, default=RUN_DIR)
    sub.add_argument("--arms", default="full_context,generic_T,generic_S,best_of_n_T")
    sub.add_argument("--fit-cap-usd", type=float, default=None,
                     help="drop arms from the end of the priority order until this "
                          "cumulative cap fits (best_of_n_T first)")
    cmd.add_parser("admission-deadline")
    sub = cmd.add_parser("archive-run")
    sub.add_argument("--run", type=Path, default=RUN_DIR)
    sub.add_argument("--reason", required=True)
    sub.add_argument("--note", default="")
    for name in ("estimate", "registration-template"):
        sub = cmd.add_parser(name)
        sub.add_argument("--window-min", type=int, default=None)
        sub.add_argument("--cap-usd", type=float, default=14.70)
        sub.add_argument("--teacher-cap-usd", type=float, default=0.0 if name == "estimate" else 1.50)
        sub.add_argument("--attempts-mean", type=float, default=2.0)
        sub.add_argument("--topup", action="store_true",
                         help="estimate/register the top-up window of --run (D3/D4)")
        sub.add_argument("--topup2", action="store_true",
                         help="estimate/register the second top-up (b0, D5) of --run")
        sub.add_argument("--margin", type=float, default=1.15,
                         help="top-up 2: factor on measured arm time (recorded)")
        sub.add_argument("--run", type=Path, default=RUN_DIR)
        sub.add_argument("--cumulative-cap-usd", type=float, default=None,
                         help="cumulative POC cap (default 16.50 top-up, 17.25 top-up 2)")
    args = parser.parse_args(argv)
    if getattr(args, "cumulative_cap_usd", 0) is None:
        args.cumulative_cap_usd = 17.25 if args.topup2 else 16.50
    try:
        if args.command == "preflight":
            result = preflight(args.run, registration=args.registration, topup=args.topup,
                               topup2=args.topup2)
            for k, v in result.items():
                print(f"{k.upper()}={v}")
            print("preflight ok")
        elif args.command == "stage":
            preflight(args.run, registration=args.registration, allow_window_record=True)
            registration = _read(args.registration)
            if bound_hashes() != registration["hashes"]:
                raise ValueError("inputs changed after preflight")
            _write(args.run / "registration.json", registration)
            # Client creation includes the probe; TeacherChat construction itself
            # is local. Only this command can dispatch model requests.
            print(json.dumps(stage_window(args.run), sort_keys=True, indent=1))
        elif args.command == "topup-stage":
            from slh.poc9 import topup

            preflight(args.run, registration=args.registration, allow_window_record=True,
                      topup=True)
            registration = _read(args.registration)
            if bound_hashes() != registration["hashes"]:
                raise ValueError("inputs changed after preflight")
            _write(args.run / topup.TOPUP_DIR / "registration.json", registration)
            print(json.dumps(topup.stage(args.run), sort_keys=True, indent=1))
        elif args.command == "topup2-stage":
            from slh.poc9 import topup2

            preflight(args.run, registration=args.registration, allow_window_record=True,
                      topup2=True)
            registration = _read(args.registration)
            if bound_hashes() != registration["hashes"]:
                raise ValueError("inputs changed after preflight")
            _write(args.run / topup2.TOPUP2_DIR / "registration.json", registration)
            print(json.dumps(topup2.stage(args.run), sort_keys=True, indent=1))
        elif args.command == "topup2-freeze":
            from slh.poc9 import topup2

            plan = topup2.write_plan(args.run)
            print(json.dumps({k: v for k, v in plan.items() if k != "arms"} |
                             {"arms": [{k: v for k, v in e.items() if k != "case_ids"}
                                       for e in plan["arms"]]}, sort_keys=True, indent=1))
        elif args.command == "topup-freeze":
            from slh.poc9 import topup

            names = [n for n in args.arms.split(",") if n]
            # Unselected top-up arms are recorded as skipped, with the reason.
            dropped = {n: "not selected for the top-up (owner decision)" for n in
                       topup.PRIORITY if n not in names}
            if "full_context" in dropped:
                dropped["full_context"] = ("amendment 9: the frozen full-context text does not "
                                           "fit max_model_len and is never truncated; D4 declined")
            if args.fit_cap_usd is not None:
                names, unfit = topup.fit_names(
                    args.run, cumulative_cap_usd=args.fit_cap_usd,
                    prior_spend_usd=topup.prior_spend(args.run)["total"], names=tuple(names))
                dropped.update(unfit)
            plan = topup.write_plan(args.run, names=names, dropped=dropped)
            print(json.dumps({k: v for k, v in plan.items() if k != "arms"} |
                             {"arms": [{k: v for k, v in e.items() if k not in
                                        ("text", "case_ids")} for e in plan["arms"]]},
                             sort_keys=True, indent=1))
        elif args.command == "archive-run":
            print(archive_run(args.reason, args.note, run=args.run))
        elif args.command == "admission-deadline":
            from slh.demo1.cli import main as demo_main

            return demo_main(["admission-deadline"])
        elif args.command == "report":
            print(json.dumps(report(args.run), sort_keys=True, indent=1))
        elif args.command == "finalize-costs":
            if args.topup or args.topup2:
                args.run = args.run / ("topup2" if args.topup2 else "topup")
            costs = _read(args.run / "costs.json") if (args.run / "costs.json").is_file() else {
                "schema": "poc9-costs-v1", "status": "incomplete",
                "teacher_usd": sum(float(r.get("charged_usd", 0))
                                   for r in _rows(args.run / "teacher-calls.jsonl")),
                "teacher_cost_note": "recorded attempts only; reconcile any in-flight calls",
                "gpu_cost_note": "fallback-rate elapsed upper estimate, including failed starts"}
            window = _read(args.run / "window.json")
            seconds = time.time() - window["started_epoch"]
            costs.update(gpu_elapsed_s_through_verified_stop=seconds,
                         gpu_usd_through_verified_stop_estimated=seconds / 3600 *
                         estimate.hourly_rate(), shutdown_cost_pending=False,
                         gpu_cost_note="fallback-rate wall estimate; reconcile provider lifetime")
            if args.topup or args.topup2:
                registered = args.run / "registration.json"
                prior = _read(registered).get("prior_spend", {}).get("total") \
                    if registered.is_file() else None
                costs.update(prior_spend_usd=prior,
                             cumulative_usd_through_verified_stop_estimated=None if prior is None
                             else prior + costs["gpu_usd_through_verified_stop_estimated"])
            _write(args.run / "costs.json", costs)
        elif args.topup2:
            from slh.poc9 import topup2

            if args.command == "registration-template":
                result = topup2.registration_template(run=args.run,
                    cumulative_cap_usd=args.cumulative_cap_usd, window_min=args.window_min,
                    margin_factor=args.margin)
            else:
                plan = topup2.load_plan(args.run, verify_freeze(args.run))
                if plan is None:
                    raise FileNotFoundError("write the second top-up freeze first "
                                            "(topup2-freeze)")
                result = topup2.estimate_window(args.run, plan,
                    cumulative_cap_usd=args.cumulative_cap_usd, window_min=args.window_min,
                    margin_factor=args.margin)
            print(json.dumps(result, sort_keys=True, indent=1))
        elif args.topup:
            from slh.poc9 import topup

            if args.command == "registration-template":
                result = topup.registration_template(run=args.run,
                    cumulative_cap_usd=args.cumulative_cap_usd, window_min=args.window_min)
            else:
                frozen = verify_freeze(args.run)
                plan = topup.load_plan(args.run, frozen)
                if plan is None:
                    raise FileNotFoundError("write the top-up freeze first (topup-freeze)")
                result = topup.estimate_topup(args.run, plan,
                    cumulative_cap_usd=args.cumulative_cap_usd,
                    prior_spend_usd=topup.prior_spend(args.run)["total"],
                    window_min=args.window_min)
            print(json.dumps(result, sort_keys=True, indent=1))
        else:
            fn = registration_template if args.command == "registration-template" else estimate.estimate
            result = fn(window_min=args.window_min or 180, cap_usd=args.cap_usd,
                        teacher_cap_usd=args.teacher_cap_usd, attempts_mean=args.attempts_mean)
            print(json.dumps(result, sort_keys=True, indent=1))
        return 0
    except CreditHalt as exc:
        print(f"POC9 CREDIT STOP: {exc}", file=sys.stderr)
        return 3
    except (OSError, ValueError, AssertionError, RuntimeError) as exc:
        print(f"POC9 refused: {exc}", file=sys.stderr)
        return 124 if isinstance(exc, (decide.StageDeadlineReached, decide.CaseTimeLimit)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
