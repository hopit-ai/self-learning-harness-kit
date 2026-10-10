"""Demo-1 stages and the offline report (draft amendment 8). Two GPU windows:

Window A, ``scripts/w4d.sh learn`` -> stage ``learn``:
  B1 on the 12 train cases of each task; the agent writes one procedure per
  task from its own train answers and the correct answers; the same writer
  writes the same-length generic note without the evidence. The stage ends with
  ``learn-record.json``: the SHA-256 of every artifact file and text that passes
  to window B.

Between the windows (GPU off): the owner accepts or rejects each procedure by
the hash of its text (recomputed, never trusted from the file), and registers
the learn record's artifact hashes (``python -m slh.demo1 artifacts-row``).

Window B, ``scripts/w4d.sh test`` -> stage ``test`` (one process, no gaps):
  calibration once (B1 and B1 + the accepted procedure on the 12 calibration
  cases of each task; a procedure below B1 there is rejected), the freeze of
  every arm text by hash, then all arms on the 60 sealed cases of each task.

``report`` is offline and free.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from slh.demo1 import cases as case_files
from slh.demo1 import learn
from slh.demo1.decide import SAMPLE_TEMPERATURE, AgentClient, Arm, run_arm
from slh.demo1.schema import KINDS, ROLES, TASKS
from slh.demo1.score import summarize

AMENDMENT_ID = "learning-v2-amendment-8"
AMENDMENT_DOC = "docs/protocol-amendment-8.md"
RUN_DIR = case_files.OUT / "runs" / "w4d"
WORKERS = 4
PASS_POINTS = 10.0
SEALED_PER_TASK = 60
REQUIRED_ARMS = {
    "address": ("b1", "learned", "full_context", "human", "generic", "best_of_2"),
    "variant": ("b1", "learned", "full_context", "human", "generic", "best_of_2",
                "code_rule"),
}
# B0 (the reference agent, no think tool) is descriptive only: never required,
# never in the pass mark.
ARM_ORDER = ("b1", "b0", "learned", "full_context", "human", "generic", "best_of_2", "code_rule")
BOOTSTRAP = {"samples": 10000, "seed": 0, "unit": "family", "confidence": 0.95}
POWER_NOTE = ("20 sealed families per task: the registered pass is descriptive. Ten points "
              "is six net cases; at 80% power the detectable paired gain is about 16 to 26 "
              "points (20-50% discordant pairs), more with family correlation.")
COST_KEYS = ("model_calls", "failed_attempts", "input_tokens", "output_tokens")
ARTIFACTS = tuple(f"{kind}-{task}.json" for task in TASKS
                  for kind in ("learned", "generic", "full-context"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=1) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path, arm: str | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return [r for r in rows if arm is None or r["arm"] == arm]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _log_probe(run: Path, stage: str, client: AgentClient) -> None:
    if client.probe is None:
        return
    run.mkdir(parents=True, exist_ok=True)
    with (run / "probes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, **client.probe}, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- window A


def stage_learn(run: Path, client: AgentClient, *, out: Path = case_files.OUT) -> dict[str, Any]:
    art = run / "artifacts"
    if (run / "learn-record.json").exists() or (art / "learned-address.json").exists():
        raise FileExistsError("the learning stage already ran (it runs once)")
    _log_probe(run, "learn", client)
    summary: dict[str, Any] = {}
    for task in TASKS:
        train = case_files.load(task, "train", out=out)
        rows = run_arm(Arm("b1"), train, client, run / "train-results.jsonl", workers=WORKERS)
        proc = learn.write_procedure(task, train, rows, client)
        _write_json(art / f"learned-{task}.json", proc)
        generic = {"task": task, "artifact": "generic_note", "ok": False, "text": None,
                   "sha256": None, "problems": ["no learned procedure to match in length"],
                   "model_calls": 0, "input_tokens": 0, "output_tokens": 0}
        if proc["ok"]:
            generic = learn.write_generic(task, proc["tokens"], train, client)
        _write_json(art / f"generic-{task}.json", generic)
        fc_text = learn.full_context_note(task, learn.evidence_text(train, rows))
        _write_json(art / f"full-context-{task}.json", {
            "task": task, "artifact": "full_context_note", "ok": True, "text": fc_text,
            "sha256": _sha(fc_text)})
        summary[task] = {"train_b1": summarize(rows), "procedure_ok": proc["ok"],
                         "generic_ok": generic["ok"]}
    record = learn_record(run)
    record["summary"] = summary
    _write_json(run / "learn-record.json", record)
    return record


def learn_record(run: Path) -> dict[str, Any]:
    """Hashes of everything window A passes to window B (files and texts)."""
    art = run / "artifacts"
    files = {name: _file_sha(art / name) for name in ARTIFACTS}
    files["train-results.jsonl"] = _file_sha(run / "train-results.jsonl")
    texts = {}
    for name in ARTIFACTS:
        body = _read_json(art / name)
        texts[name] = _sha(body["text"]) if body.get("ok") and body.get("text") else None
    return {"schema": "demo1-learn-record-v1", "amendment": AMENDMENT_ID,
            "artifact_files_sha256": files, "artifact_texts_sha256": texts}


def artifacts_row(run: Path) -> dict[str, Any]:
    """The registry row the owner appends between the windows."""
    record = _read_json(run / "learn-record.json")
    current = learn_record(run)
    for key in ("artifact_files_sha256", "artifact_texts_sha256"):
        if record[key] != current[key]:
            raise ValueError("window-A artifacts changed after the learn record was written")
    return {"event": "demo1_learn_artifacts", "amendment_id": AMENDMENT_ID,
            "learn_record_sha256": _file_sha(run / "learn-record.json"),
            "artifact_files_sha256": current["artifact_files_sha256"],
            "artifact_texts_sha256": current["artifact_texts_sha256"],
            "owner_verdicts": owner_status(run)}


# --------------------------------------------------------------------------- owner


def verdict_path(run: Path) -> Path:
    return run / "owner-verdicts.jsonl"


def accepted_text(run: Path, task: str) -> str | None:
    """The procedure text whose RECOMPUTED hash equals the owner's accept verdict."""
    proc = _read_json(run / "artifacts" / f"learned-{task}.json")
    verdicts = {r["task"]: r for r in learn.owner_verdicts(verdict_path(run))}
    v = verdicts.get(task)
    if not proc.get("ok") or not proc.get("text") or v is None or v["verdict"] != "accept":
        return None
    return proc["text"] if _sha(proc["text"]) == v["procedure_sha256"] else None


def owner_status(run: Path) -> dict[str, Any]:
    """Per task: accept / reject / waiting / no_procedure / not_written /
    text_hash_mismatch (the stored text no longer hashes to its recorded hash)
    / verdict_hash_mismatch (the verdict names another text)."""
    verdicts = {r["task"]: r for r in learn.owner_verdicts(verdict_path(run))}
    out: dict[str, Any] = {}
    for task in TASKS:
        path = run / "artifacts" / f"learned-{task}.json"
        if not path.is_file():
            out[task] = "not_written"
            continue
        proc = _read_json(path)
        if not proc["ok"]:
            out[task] = "no_procedure"
            continue
        actual = _sha(proc["text"])
        if actual != proc["sha256"]:
            out[task] = "text_hash_mismatch"
        elif task not in verdicts:
            out[task] = "waiting"
        elif verdicts[task]["procedure_sha256"] != actual:
            out[task] = "verdict_hash_mismatch"
        else:
            out[task] = verdicts[task]["verdict"]
    return out


# --------------------------------------------------------------------------- window B


def calibrate(run: Path, client: AgentClient, *, out: Path = case_files.OUT,
              human_root: Path = learn.HUMAN_DIR) -> dict[str, Any]:
    art = run / "artifacts"
    if (art / "freeze.json").exists():
        raise FileExistsError("calibration ran already (it is used once)")
    status = owner_status(run)
    if any(s not in ("accept", "reject", "no_procedure") for s in status.values()):
        raise ValueError(f"owner review is not complete or a hash differs: {status}")
    record = _read_json(run / "learn-record.json")
    current = learn_record(run)
    if record["artifact_files_sha256"] != current["artifact_files_sha256"]:
        raise ValueError("window-A artifacts changed after the learn record")
    freeze: dict[str, Any] = {"amendment": AMENDMENT_ID, "tasks": {},
                              "learn_record_sha256": _file_sha(run / "learn-record.json")}
    for task in TASKS:
        cal = case_files.load(task, "calibrate", out=out)
        b1 = run_arm(Arm("b1"), cal, client, run / "calibrate-results.jsonl", workers=WORKERS)
        entry: dict[str, Any] = {"owner": status[task], "b1": summarize(b1)}
        text = accepted_text(run, task)
        if text is not None:
            learned = run_arm(Arm("learned", (text,)), cal, client,
                              run / "calibrate-results.jsonl", workers=WORKERS)
            entry["learned"] = summarize(learned)
            frozen = entry["learned"]["correct"] >= entry["b1"]["correct"]
            entry["decision"] = "frozen" if frozen else "rejected_at_calibration"
        else:
            entry["decision"] = {"reject": "rejected_by_owner",
                                 "no_procedure": "no_valid_procedure"}[status[task]]
        entry["b1_few_errors"] = entry["b1"]["correct"] >= 11
        texts = {"full_context": _read_json(art / f"full-context-{task}.json")["text"],
                 "human": learn.human_procedure(task, human_root)}
        if entry["decision"] == "frozen":
            texts["learned"] = text
        generic = _read_json(art / f"generic-{task}.json")
        if generic["ok"]:
            texts["generic"] = generic["text"]
        entry["texts"] = texts
        entry["texts_sha256"] = {k: _sha(v) for k, v in texts.items()}
        freeze["tasks"][task] = entry
    _write_json(art / "freeze.json", freeze)
    return freeze


def sealed_arms(task: str, freeze: dict[str, Any]) -> list[Arm]:
    texts = freeze["tasks"][task]["texts"]
    arms = [Arm("b1"), Arm("b0", kind="b0")]
    for name in ("learned", "full_context", "human", "generic"):
        if name in texts:
            arms.append(Arm(name, (texts[name],)))
    arms.append(Arm("best_of_2", kind="best_of_2"))
    if task == "variant":
        arms.append(Arm("code_rule", kind="code_rule"))
    return arms


def sealed_run(run: Path, client: AgentClient, *, out: Path = case_files.OUT,
               sealed_dir: Path = case_files.SEALED_DIR) -> dict[str, Any]:
    freeze = _read_json(run / "artifacts" / "freeze.json")
    for task in TASKS:
        for name, sha in freeze["tasks"][task]["texts_sha256"].items():
            if _sha(freeze["tasks"][task]["texts"][name]) != sha:
                raise ValueError(f"{task}/{name}: frozen text changed")
    sampler = client.with_temperature(SAMPLE_TEMPERATURE)
    results = run / "sealed-results.jsonl"
    with case_files.sealed_access("final_run", sealed_dir=sealed_dir, out=out) as sealed:
        for task in TASKS:
            cases = sealed[task]
            greedy: dict[str, dict[str, Any]] = {}
            for arm in sealed_arms(task, freeze):
                rows = run_arm(arm, cases, client, results, workers=WORKERS, sampler=sampler,
                               greedy_rows=greedy if arm.kind == "best_of_2" else None)
                if arm.name == "b1":
                    greedy = {r["case_id"]: r for r in rows}
    return {"rows": len(_rows(results))}


def stage_test(run: Path, client: AgentClient, *, out: Path = case_files.OUT,
               human_root: Path = learn.HUMAN_DIR,
               sealed_dir: Path = case_files.SEALED_DIR) -> dict[str, Any]:
    """Window B in one process: calibration once, freeze, sealed run."""
    _log_probe(run, "test", client)
    freeze = calibrate(run, client, out=out, human_root=human_root)
    done = sealed_run(run, client, out=out, sealed_dir=sealed_dir)
    return {"decisions": {t: freeze["tasks"][t]["decision"] for t in TASKS}, **done}


# --------------------------------------------------------------------------- report


def _paired(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, int]:
    bm = {r["case_id"]: r["correct"] for r in b}
    wins = sum(1 for r in a if r["correct"] and not bm.get(r["case_id"]))
    losses = sum(1 for r in a if not r["correct"] and bm.get(r["case_id"]))
    return {"wins": wins, "losses": losses}


def family_bootstrap(a: list[dict[str, Any]], b: list[dict[str, Any]], *,
                     samples: int = BOOTSTRAP["samples"], seed: int = BOOTSTRAP["seed"]
                     ) -> dict[str, float]:
    """Difference in accuracy (a - b, points), resampling sealed families."""
    bm = {r["case_id"]: r["correct"] for r in b}
    fams: dict[str, list[tuple[int, int]]] = {}
    for r in a:
        fams.setdefault(r["family"], []).append((int(r["correct"]), int(bm[r["case_id"]])))
    keys = sorted(fams)
    rng = random.Random(seed)
    diffs = []
    for _ in range(samples):
        pick = [fams[keys[rng.randrange(len(keys))]] for _ in keys]
        pairs = [p for fam in pick for p in fam]
        diffs.append(100.0 * sum(x - y for x, y in pairs) / len(pairs))
    diffs.sort()
    alpha = (1 - BOOTSTRAP["confidence"]) / 2
    point = 100.0 * sum(x - y for fam in fams.values() for x, y in fam) / sum(
        len(f) for f in fams.values())
    return {"diff_points": round(point, 2),
            "low": round(diffs[int(alpha * samples)], 2),
            "high": round(diffs[int((1 - alpha) * samples) - 1], 2)}


def _cost(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {k: sum(int(r.get(k) or 0) for r in rows) for k in COST_KEYS}


def _add(*costs: dict[str, Any]) -> dict[str, int]:
    return {k: sum(int(c.get(k) or 0) for c in costs) for k in COST_KEYS}


def _breakdown(rows: list[dict[str, Any]], key: str, values: tuple[str, ...]
               ) -> dict[str, dict[str, int]]:
    return {v: {"n": sum(1 for r in rows if r[key] == v),
                "correct": sum(1 for r in rows if r[key] == v and r["correct"])}
            for v in values}


def coverage(rows: list[dict[str, Any]], expected_ids: list[str]) -> dict[str, Any]:
    ids = [r["case_id"] for r in rows]
    return {"rows": len(ids), "unique": len(set(ids)), "expected": len(expected_ids),
            "exact": sorted(ids) == sorted(expected_ids) and len(ids) == len(set(ids))}


def report(run: Path, *, out: Path = case_files.OUT) -> dict[str, Any]:
    freeze = _read_json(run / "artifacts" / "freeze.json")
    sealed = _rows(run / "sealed-results.jsonl")
    train = _rows(run / "train-results.jsonl")
    probes = [json.loads(x) for x in (run / "probes.jsonl").read_text().splitlines()
              if x.strip()] if (run / "probes.jsonl").is_file() else []
    manifest = case_files.manifest(out)
    result: dict[str, Any] = {"amendment": AMENDMENT_ID, "tasks": {}, "power_note": POWER_NOTE,
                              "capture_probes": _add(*probes),
                              "estimate_note": "counted calls include think and read "
                                               "continuations, retries and failed calls; the "
                                               "serving deadline is the hard limit"}
    for task in TASKS:
        expected_ids = manifest["files"][f"sealed/demo1/cases/sealed-{task}.jsonl"]["case_ids"]
        rows = [r for r in sealed if r["task"] == task]
        by = {a: [r for r in rows if r["arm"] == a] for a in ARM_ORDER
              if any(r["arm"] == a for r in rows)}
        cov = {a: coverage(rs, expected_ids) for a, rs in by.items()}
        art = run / "artifacts"
        proc = _read_json(art / f"learned-{task}.json")
        generic = _read_json(art / f"generic-{task}.json")
        train_cost = _cost([r for r in train if r["task"] == task])
        learning = _add(train_cost, proc)
        arm_out = {a: {**summarize(rs), "cost": _cost(rs),
                       "errors": sum(1 for r in rs if r.get("error")),
                       "by_kind": _breakdown(rs, "kind", KINDS[task]),
                       "by_role": _breakdown(rs, "role", ROLES), "coverage": cov[a]}
                   for a, rs in by.items()}
        entry: dict[str, Any] = {
            "freeze": {k: v for k, v in freeze["tasks"][task].items() if k != "texts"},
            "arms": arm_out,
            "learning_cost": {"total": learning,
                              "per_sealed_case": {k: round(v / SEALED_PER_TASK, 2)
                                                  for k, v in learning.items()},
                              "generic_writer": _add(generic)}}
        if "learned" in by:
            learned_cost = arm_out["learned"]["cost"]
            entry["learning_cost"]["learned_arm_per_sealed_case_amortized"] = {
                k: round((learned_cost[k] + learning[k]) / SEALED_PER_TASK, 2)
                for k in COST_KEYS}
        missing = [a for a in REQUIRED_ARMS[task] if a not in by and a != "learned"]
        # Eligibility depends on the required arms only; B0 (descriptive) coverage
        # is reported separately and never changes the status.
        bad_cov = [a for a in by if a in REQUIRED_ARMS[task] and (
            not cov[a]["exact"] or any(r.get("stage_deadline") for r in by[a]))]
        if "b0" in cov:
            entry["descriptive_b0_coverage"] = cov["b0"]
        decision = freeze["tasks"][task].get("decision")
        if "learned" not in by and decision != "frozen":
            entry["status"] = "learning_failed"
        elif "learned" not in by or missing or bad_cov:
            entry["status"] = "incomplete"
        else:
            entry["status"] = "evaluated"
        entry["missing_arms"] = missing
        entry["coverage_failures"] = bad_cov
        if entry["status"] != "evaluated":
            entry["primary_pass"] = False
            entry["secondary_pass"] = False
            entry["reason"] = decision if entry["status"] == "learning_failed" \
                else "incomplete: a registered arm is missing or does not cover the 60 cases"
            result["tasks"][task] = entry
            continue
        acc = {a: 100.0 * arm_out[a]["correct"] / SEALED_PER_TASK for a in arm_out}
        if "b0" in by and cov["b0"]["exact"]:
            entry["descriptive_b0"] = {
                "b1_vs_b0": {**_paired(by["b1"], by["b0"]),
                             **family_bootstrap(by["b1"], by["b0"])},
                "learned_vs_b0": {**_paired(by["learned"], by["b0"]),
                                  **family_bootstrap(by["learned"], by["b0"])}}
        entry["accuracy_points"] = {a: round(v, 2) for a, v in acc.items()}
        for other in ("b1", "full_context", "best_of_2", "generic", "human"):
            entry[f"learned_vs_{other}"] = {**_paired(by["learned"], by[other]),
                                            **family_bootstrap(by["learned"], by[other])}
        primary = {"gain_vs_b1_points": round(acc["learned"] - acc["b1"], 2),
                   "gain_at_least_10": acc["learned"] - acc["b1"] >= PASS_POINTS - 1e-9,
                   "above_generic_note": acc["learned"] > acc["generic"],
                   "not_below_full_context": acc["learned"] >= acc["full_context"],
                   "not_below_best_of_2": acc["learned"] >= acc["best_of_2"]}
        entry["primary"] = primary
        entry["primary_pass"] = all(v for k, v in primary.items() if k != "gain_vs_b1_points")
        entry["secondary"] = {"above_full_context": acc["learned"] > acc["full_context"],
                              "above_best_of_2": acc["learned"] > acc["best_of_2"]}
        entry["secondary_pass"] = entry["primary_pass"] and all(entry["secondary"].values())
        notes = []
        if acc["learned"] == acc["full_context"]:
            notes.append("learning not better than retaining the evidence (tie with "
                         "full context)")
        if acc["learned"] == acc["best_of_2"]:
            notes.append("learning not better than extra compute (tie with best-of-2)")
        b1_based = [v for a, v in acc.items() if a not in ("code_rule", "b0")]
        if "code_rule" in acc and acc["code_rule"] >= max(b1_based):
            notes.append("the code rule is at least as good as every B1-based arm")
        if acc["human"] <= acc["b1"]:
            notes.append("the person-written procedure does not help: stop the procedure "
                         "track for this task")
        elif acc["learned"] <= acc["b1"]:
            notes.append("the person-written procedure helps and the learned one does not: "
                         "the learning step failed")
        if freeze["tasks"][task].get("b1_few_errors"):
            notes.append("B1 made at most one calibration error: no weak point")
        entry["readings"] = notes
        result["tasks"][task] = entry
    return result
