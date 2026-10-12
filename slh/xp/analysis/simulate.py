"""Simulated phase-1 deliveries with known injected effects (tests and rehearsal).

    python -m slh.xp.analysis.simulate --out DIR [--runs 12] [--test-cases 90]
        [--effect T+=6 --effect S+=1 ...] [--run-sd 4] [--seed 0] [--human]

Writes ``DIR/package`` (plan.json, registration.json, cases), ``DIR/keys``
(answer keys and membership.json from a synthetic secret: fake keys, never ours)
and ``DIR/export`` (a dry-run delivery in the partner export format that passes
``slh.xp.partner.check`` with ``--allow-dry-run``). Then::

    python -m slh.xp.analysis.report --export DIR/export --plan DIR/package/plan.json \
        --registration DIR/package/registration.json --membership DIR/keys/membership.json \
        --keys DIR/keys --out DIR/results --allow-dry-run

Outcome model (points, per arm instance and test case): accuracy
``p = base_task + group_effect + arm_effect + run_deviation``; run deviations
apply to learned skills only (between-run SD ``run_sd`` per task, a share ``rho``
common to the tasks of one run). Correctness draws share a per-case uniform with
probability ``common`` (cases that are hard for one arm are hard for all), and
wrong writes have the same marginal rate in every arm (wrong writes unchanged).
Every action is a real action the XP scorer scores as intended (checked).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from slh.xp.partner import common as pc
from slh.xp.partner import plan as pp
from slh.xp.partner import schemas as sc

TASKS = ("address", "variant", "sequence")
WRITERS = ("T+", "S+", "T")
CRITIC = {"T+": "teacher", "S+": "self", "T": "teacher"}
SECRET = b"synthetic-delivery-not-a-real-secret"[:32].ljust(32, b"0")
PLAN_ID = "xp-sim"
LEARNING = {"ops": {"max_steps": 2, "patience": 1, "batch_size": 4}, "note_batch": 4}
SAMPLE = 12
HUMAN_TEXT = "Check the order's state and ask before any change the customer has not approved."


@dataclasses.dataclass
class Effects:
    """Injected effects in accuracy points (B1 = base)."""

    base: dict[str, float] = dataclasses.field(
        default_factory=lambda: {"address": 55.0, "variant": 50.0, "sequence": 60.0})
    shared: dict[str, float] = dataclasses.field(
        default_factory=lambda: {"b0": -8.0, "b1": 0.0, "human": 2.0, "best_of_n": 2.0})
    writer: dict[str, dict[str, float]] = dataclasses.field(default_factory=lambda: {
        "T+": {"learned": 6.0, "search_control": 0.0, "generic": 1.0, "full_context": 3.0},
        "S+": {"learned": 1.0, "search_control": 0.0, "generic": 0.5, "full_context": 3.0},
        "T": {"learned": 4.0, "search_control": 0.0, "generic": 1.0, "full_context": 3.0}})
    run_sd: float = 4.0
    rho: float = 0.5
    group_sd: float = 8.0
    wrong_write: float = 6.0  # marginal rate in every arm (points)
    common: float = 0.6

    @classmethod
    def null(cls) -> Effects:
        e = cls()
        for w in e.writer:
            e.writer[w] = {k: 0.0 for k in e.writer[w]}
        return e


# --------------------------------------------------------------------------- outcomes


def _run_devs(effects: Effects, tasks: list[str], writers: list[str], runs: int,
              rng: np.random.Generator) -> dict[tuple[str, str], np.ndarray]:
    """(writer, task) -> (runs,) learned-skill deviations in points."""
    out = {}
    for w in writers:
        shared = rng.normal(0, effects.run_sd * np.sqrt(effects.rho), runs)
        for t in tasks:
            own = rng.normal(0, effects.run_sd * np.sqrt(1 - effects.rho), runs)
            out[(w, t)] = shared + own
    return out


def draw(p: np.ndarray, effects: Effects, rng: np.random.Generator, u_case: np.ndarray
         ) -> tuple[np.ndarray, np.ndarray]:
    """Correct and wrong-write indicators for (instances, cases) probabilities."""
    p = np.clip(p, 0.03, 0.92)
    shared = rng.random(p.shape) < effects.common
    v = np.where(shared, u_case[None, :], rng.random(p.shape))
    correct = v < p
    q = effects.wrong_write / 100.0
    ww = ~correct & (rng.random(p.shape) < q / (1 - p))
    return correct, ww


def instance_effect(effects: Effects, inst: dict[str, Any],
                    devs: dict[tuple[str, str], np.ndarray]) -> float:
    if inst["writer"] is None:
        return effects.shared.get(inst["arm"], 0.0)
    eff = effects.writer.get(inst["writer"], {}).get(inst["kind"], 0.0)
    if inst["kind"] == "learned":
        eff += float(devs[(inst["writer"], inst["task"])][inst["run"]])
    return eff


def simulate_table(effects: Effects, *, runs: int = 12, groups: int = 30, per_group: int = 3,
                   tasks: tuple[str, ...] = TASKS, writer: str = "T+", seed: int = 0
                   ) -> dict[str, dict[str, np.ndarray]]:
    """Endpoint and search-control outcomes of one writer without files (fast;
    for Monte Carlo checks): task -> {"learned", "control" (R, n) correct,
    "learned_ww", "control_ww", "group" (n,)}."""
    rng = np.random.default_rng(seed)
    devs = _run_devs(effects, list(tasks), [writer], runs, rng)
    out = {}
    for t in tasks:
        n = groups * per_group
        group = np.repeat(np.arange(groups), per_group)
        b = rng.normal(0, effects.group_sd, groups)[group]
        u = rng.random(n)
        eff = effects.writer[writer]
        p_l = (effects.base[t] + b[None, :] + eff["learned"] + devs[(writer, t)][:, None]) / 100
        p_c = np.broadcast_to((effects.base[t] + b + eff["search_control"]) / 100, (runs, n))
        c_l, w_l = draw(p_l, effects, rng, u)
        c_c, w_c = draw(np.array(p_c), effects, rng, u)
        out[t] = {"learned": c_l, "control": c_c, "learned_ww": w_l, "control_ww": w_c,
                  "group": group}
    return out


# --------------------------------------------------------------------------- actions


def actions_for(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Real actions for the three simulated outcomes, checked with the scorer."""
    from slh.demo1.schema import ADDRESS_FIELDS
    from slh.xp.score import ADDRESS_WRITE, score

    gold = case["gold"]
    gtype = gold["type"]
    if gtype == "tool_call":
        right = {"type": "tool_call", "name": gold["name"], "arguments": gold["arguments"]}
        other = {"type": "message", "content": "Could you tell me a little more about it?"}
    elif gtype == "clarify":
        right = {"type": "message", "content": "Which one would you like?"}
        other = {"type": "message", "content": "Okay, noted."}
    else:
        args = gold["arguments"]
        if gold["name"] == ADDRESS_WRITE:
            named = ", ".join(str(args[k]) for k in ADDRESS_FIELDS if args.get(k))
        else:
            named = f"item {args['new_item_ids'][0]}"
        right = {"type": "message", "content": f"Shall I go ahead with the change to {named}?"}
        other = {"type": "message", "content": "Okay, noted."}
    wrong = {"type": "tool_call", "name": "modify_pending_order_address",
             "arguments": {"order_id": "#W0000000"}}
    acts = {"correct": right, "wrong_write": wrong, "other": other}
    for name, act in acts.items():
        s = score(case, act)
        ok = {"correct": s["correct"], "wrong_write": s["wrong_write"],
              "other": not s["correct"] and not s["wrong_write"]}[name]
        if not ok:
            raise ValueError(f"{case['case_id']}: simulated {name} action scores {s}")
    return acts


# --------------------------------------------------------------------------- learning rows


def _h(text: str | None) -> str | None:
    return pc.sha_text(text) if text else None


def _budgets(plan: dict[str, Any], selection: int) -> dict[str, Any]:
    reg = plan["learning"]["budgets"]
    used = {"acquisition": {"calls": 0, "output_tokens": 0, "decisions": SAMPLE},
            "learn": {"calls": 9, "output_tokens": 4000, "decisions": SAMPLE + 36},
            "search": {"calls": 9, "output_tokens": 3800, "decisions": SAMPLE + 36},
            "selection_learned": {"calls": 0, "output_tokens": 0, "decisions": selection},
            "selection_search": {"calls": 0, "output_tokens": 0, "decisions": selection},
            "generic": {"calls": 3, "output_tokens": 900, "decisions": 0}}
    out = {}
    for comp, u in used.items():
        lim = reg[comp]
        u = {k: min(v, int(lim[k])) for k, v in u.items()}
        out[comp] = {**u, **{f"limit_{k}": int(lim[k]) for k in sc.BUDGET_KEYS}}
    out["total"] = {k: sum(out[c][k] for c in used) for k in sc.BUDGET_KEYS}
    return out


def _selection(snaps: list[str | None], sentinel: int, revert: bool, rng: random.Random,
               base: float) -> dict[str, Any]:
    k = sentinel
    lvl = [round(base * k) / k, min(k, round(base * k) + 1) / k, min(k, round(base * k) + 2) / k]
    if revert:  # endpoint no better than its predecessor: back to the best (index 1)
        lvl = [lvl[0], lvl[2], lvl[1]]
    scores = [{"index": i, "sha256": snaps[i], "score": round(lvl[i], 4)}
              for i in range(len(snaps))]
    chosen = 2 if not revert else 1
    return {"applied": True, "reason": None, "snapshots": snaps, "endpoint_sha256": snaps[-1],
            "selected_sha256": snaps[chosen], "reverted": chosen != 2, "scores": scores}


def learning_unit(plan: dict[str, Any], task: str, run: int, writer: str,
                  rng: random.Random, dev_gain: float) -> tuple[list[dict], dict[str, str]]:
    """Learning rows (summary, traces, acquisition) and the four exported texts
    of one learning run, consistent with ``check.verify_learning``."""
    ids = plan["learning"]["train_samples"][task]["runs"][str(run)]
    sentinel = int(plan["cases"][task]["sentinel"]["rows"])
    dev = int(plan["cases"][task]["validation"]["rows"])
    tag = f"{task} {writer} run {run}"
    texts_l = [f"Lesson draft for {tag}: read the order state first.",
               f"Lessons for {tag}: read the order state first; ask when two options fit."]
    texts_s = [f"Search note for {tag}: be careful.",
               f"Search notes for {tag}: be careful; confirm before writing."]
    hl = [None, _h(texts_l[0]), _h(texts_l[1])]
    hs = [None, _h(texts_s[0]), _h(texts_s[1])]
    rev_l, rev_s = rng.random() < 0.2, rng.random() < 0.2
    base = 0.5
    sel_l = _selection(hl, sentinel, rev_l, rng, base)
    sel_s = _selection(hs, sentinel, rev_s, rng, base)
    skill = texts_l[0] if rev_l else texts_l[1]
    control = texts_s[0] if rev_s else texts_s[1]
    evidence = f"Training histories and deterministic feedback for {tag} (synthetic)."
    generic = f"General note for {tag}: follow the policy."
    rows: list[tuple[str, dict[str, Any]]] = []
    rows += [("acquisition", {"case_id": c, "stream_position": i + 1,
                              "score": {"correct": rng.random() < 0.5}})
             for i, c in enumerate(ids)]
    if writer.endswith("+"):
        loop = "ops"

        def trace(states: list[str | None], gain: float) -> list[dict[str, Any]]:
            d0 = round(round(0.5 * dev) / dev, 4)
            d1 = round(min(1.0, d0 + (1 + round(gain * 20)) / dev), 4)
            d2 = round(min(1.0, d1 + (1 + round(gain * 10)) / dev), 4)
            out = [{"type": "start", "dev_score": d0, "train": SAMPLE, "dev": dev}]
            for step, (before, after, state) in enumerate(
                    ((d0, d1, states[1]), (d1, d2, states[2])), 1):
                out.append({"type": "step", "step": step, "batch": ids[4 * step - 4:4 * step],
                            "accepted": True, "verdicts": [{"ok": True}],
                            "calls": [{"role": "writer"}], "dev_before": before,
                            "dev_after": after, "library_sha256": state})
            out.append({"type": "step", "step": 3, "batch": ids[8:12], "accepted": False,
                        "verdicts": [{"ok": True}], "calls": [{"role": "writer"}],
                        "dev_before": d2, "dev_after": round(max(0.0, d2 - 1 / dev), 4),
                        "library_sha256": states[2]})
            out.append({"type": "stop", "dev_score": d2, "steps": 3, "accepted": 2})
            return out

        rows += [("trace", r) for r in trace(hl, dev_gain)]
        rows += [("search_trace", r) for r in trace(hs, 0.0)]
        episodes = SAMPLE + 3 * dev
    else:
        loop = "note"

        def patches(states: list[str | None], search: bool) -> list[dict[str, Any]]:
            out = []
            for i, (acc, after) in enumerate(((True, states[1]), (True, states[2]),
                                              (False, states[2])), 1):
                row = {"after_episode": 4 * i, "accepted": acc, "stage": "checks",
                       "patch": {"changed": True}, "writer_budget": {"calls": 2 * i},
                       "skill_sha256_after": after}
                if search:
                    row["episodes"] = list(range(4 * i - 3, 4 * i + 1))
                out.append(row)
            return out

        rows += [("patches", r) for r in patches(hl, False)]
        rows += [("search_trace", r) for r in patches(hs, True)]
        p_ok = 0.5 + dev_gain / 2
        rows += [("episodes", {"case_id": c, "stream_position": i + 1,
                               "score": {"correct": rng.random() < (0.45 if i < 4 else p_ok)}})
                 for i, c in enumerate(ids)]
        episodes = SAMPLE
    selection = 3 * sentinel
    summary = {"task": task, "writer": writer, "run": run, "seed": 1000 + run,
               "critic": CRITIC.get(writer, "none"),
               "check": {"teacher": "teacher-checked", "self": "self-checked",
                         "none": "unchecked"}[CRITIC.get(writer, "none")],
               "loop": loop, "train_case_ids_sha256": pc.sha_text(pc.canonical(ids)),
               "train_cases": len(ids), "dev": dev, "sentinel": sentinel, "proposals": 3,
               "accepted": 2, "episodes": episodes, "stop_reason": "patience",
               "search_control": {"proposals": 3, "accepted": 2, "stop_reason": "patience"},
               "selection": {"learned": sel_l, "search_control": sel_s},
               "budgets": _budgets(plan, selection), "evidence_cases": len(ids),
               "skill_sha256": _h(skill), "generic_sha256": _h(generic),
               "evidence_sha256": _h(evidence), "search_control_sha256": _h(control),
               "skill_tokens": len(skill.split()), "generic_ok": True, "generic_problems": []}
    rows.append(("summary", summary))
    out = [{"task": task, "writer": writer, "run": run, "source": s, "row": r} for s, r in rows]
    return out, {"learned": skill, "search_control": control, "full_context": evidence,
                 "generic": generic}


# --------------------------------------------------------------------------- delivery


def write_delivery(out: Path, *, effects: Effects | None = None, runs: int = 12,
                   test_cases: int = 90, tasks: tuple[str, ...] = TASKS,
                   writers: tuple[str, ...] = WRITERS, human: bool = False, seed: int = 0,
                   gpus: int = 8, gpu_hours: float = 210.0) -> dict[str, Path]:
    """Package, fake keys and a checker-valid dry-run export with known effects."""
    effects = effects or Effects()
    out = Path(out)
    root = out / "root"
    root.mkdir(parents=True, exist_ok=True)
    gen = pp.generation(list(tasks), test_cases=test_cases,
                        pool_cases={t: 60 for t in tasks}, validation_cases=12,
                        sentinel_cases=6)
    pkg = pp.write_package(out / "package", out / "keys", secret=SECRET, plan_id=PLAN_ID,
                           tasks=list(tasks), gen=gen, writers=list(writers), runs=runs,
                           arms=pp.default_arms(HUMAN_TEXT if human else ""),
                           protocol_text="Synthetic delivery protocol.\n",
                           critic={w: CRITIC.get(w, "none") for w in writers},
                           learning=LEARNING, sample_cases=SAMPLE, root=root)
    plan, reg = pkg["plan"], pkg["registration"]
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    devs = _run_devs(effects, list(tasks), list(writers), runs, rng)

    learning, skills, texts = [], [], {}
    for task in tasks:
        for run in range(runs):
            for w in writers:
                gain = (effects.writer[w]["learned"] - effects.writer[w]["search_control"]
                        + float(devs[(w, task)][run])) / 100
                rows, t = learning_unit(plan, task, run, w, prng, max(0.0, gain))
                learning += rows
                for kind, text in t.items():
                    texts[(task, run, w, kind)] = text
                    skills.append({"task": task, "writer": w, "run": run, "kind": kind,
                                   "text": text, "sha256": _h(text)})
    insts = pp.instances(plan, lambda t, r, w, k: texts[(t, r, w, k)])
    arms_rows = [sc.pick("arms.jsonl.gz", i) for i in insts]

    decisions, costs, stages = [], [], []
    for task in tasks:
        key = pc.load_answer_key(pkg["keys"][task])
        cases = [key[c] for c in sorted(key)]
        acts = [actions_for(c) for c in cases]
        groups = sorted({c["group"] for c in cases})
        gidx = {g: i for i, g in enumerate(groups)}
        b = rng.normal(0, effects.group_sd, len(groups))
        bcase = np.array([b[gidx[c["group"]]] for c in cases])
        u = rng.random(len(cases))
        units: dict[str, dict[str, Any]] = {}
        for inst in insts:
            if inst["task"] == task:
                units.setdefault(inst["prompt_hash"], inst)
        reps = list(units.values())
        p = np.array([(effects.base[task] + bcase + instance_effect(effects, i, devs)) / 100
                      for i in reps])
        correct, ww = draw(p, effects, rng, u)
        n_calls = tok_p = tok_c = 0
        for j, inst in enumerate(reps):
            for c_i, case in enumerate(cases):
                outcome = "correct" if correct[j, c_i] else \
                    "wrong_write" if ww[j, c_i] else "other"
                calls = 1 + int(prng.random() < 0.1)
                tokens = {"prompt": 2800 + int(prng.random() * 600),
                          "completion": 40 + int(prng.random() * 60)}
                n_calls, tok_p, tok_c = (n_calls + calls, tok_p + tokens["prompt"],
                                         tok_c + tokens["completion"])
                decisions.append({"task": task, "case_id": case["case_id"],
                                  "case_hash": case["case_hash"],
                                  "prompt_hash": inst["prompt_hash"], "arm": inst["arm"],
                                  "run": inst["run"], "writer": inst["writer"],
                                  "action": acts[c_i][outcome], "tokens": tokens,
                                  "model_calls": calls,
                                  "wall_s": round(1.5 + prng.random() * 2, 3), "error": None,
                                  "capped": False, "case_time_limit": False})
        costs.append({"name": "costs.jsonl", "row": {
            "task": task, "decisions": len(reps) * len(cases), "model_calls": n_calls,
            "prompt_tokens": tok_p, "completion_tokens": tok_c, "errors": 0,
            "time_limits": 0}})
        for run in range(runs):
            for w in writers:
                stages.append({"name": "stages.jsonl", "row": {
                    "stage": "learning", "task": task, "writer": w, "run": run,
                    "wall_s": round(600 + prng.random() * 900, 1)}})
    epoch = 1_800_000_000.0
    ledger = [{"name": "gpu-hours.jsonl", "row": {"event": "start", "job": "1", "epoch": epoch,
                                                  "gpus": gpus, "limit_s": 86400}},
              {"name": "gpu-hours.jsonl", "row": {
                  "event": "end", "job": "1", "epoch": epoch + gpu_hours * 3600 / gpus,
                  "gpus": gpus, "limit_s": 86400}}]
    files = {"decisions.jsonl.gz": decisions, "arms.jsonl.gz": arms_rows,
             "learning.jsonl.gz": learning, "skills.jsonl.gz": skills,
             "logs.jsonl.gz": costs + stages + ledger}
    from slh.xp.partner.runner import ENDPOINT_FINGERPRINT, endpoint_sha256

    endpoint = {k: None for k in ENDPOINT_FINGERPRINT}
    endpoint.update(model="synthetic", base_url="http://127.0.0.1:9/v1")
    hashes = {"git_commit": reg["git_commit"], "git_head": None, "git_dirty": None,
              "code_sha256": reg["code_sha256"], "plan_sha256": reg["plan_sha256"],
              "protocol_sha256": reg["protocol_sha256"], "case_files": reg["case_files"],
              "run_identity": {"mode": "dry", "endpoint_sha256": endpoint_sha256(endpoint),
                               "smoke": None}}
    env = {"dry_run": True, "mode": "dry", "time_epoch": epoch, "gpu": None,
           "endpoint": endpoint, "endpoint_sha256": hashes["run_identity"]["endpoint_sha256"],
           "served_version": None, "python": "synthetic", "platform": "synthetic",
           "packages": {}, "vendor": {}}
    export = out / "export"
    export.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name, rows in files.items():
        for r in rows:
            sc.strict(name, r)
        raw, n = pc.jsonl_bytes(rows)
        data = pc.gz_bytes(raw)
        (export / name).write_bytes(data)
        meta[name] = {"sha256": pc.sha_bytes(data), "bytes": len(data), "rows": n}
    data = (json.dumps(env, sort_keys=True, indent=1) + "\n").encode()
    (export / "env.json").write_bytes(data)
    meta["env.json"] = {"sha256": pc.sha_bytes(data), "bytes": len(data), "rows": None}
    manifest = {"schema": "xp-partner-export-v2", "created_by": "slh.xp.analysis.simulate",
                "plan_id": plan["plan_id"], "shard": {"tasks": sorted(tasks)},
                "files": dict(sorted(meta.items())), "hashes": hashes,
                "cap_bytes": int(reg["export_cap_bytes"]),
                "transcripts": {"sample": 0, "seed": 0, "key_arms": None}}
    (export / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=1) + "\n")
    truth = {"effects": dataclasses.asdict(effects), "runs": runs, "test_cases": test_cases,
             "learned_run_deviation": {f"{w}|{t}": [round(float(x), 3) for x in v]
                                       for (w, t), v in devs.items()}}
    pc.write_json(out / "truth.json", truth)
    return {"export": export, "plan": out / "package" / "plan.json",
            "registration": out / "package" / "registration.json",
            "membership": out / "keys" / "membership.json", "keys": out / "keys",
            "truth": out / "truth.json"}


def rewrite(export: Path, name: str, rows: list[dict[str, Any]]) -> None:
    """Replace one export file and its manifest entry (incomplete-delivery tests)."""
    raw, n = pc.jsonl_bytes(rows)
    data = pc.gz_bytes(raw)
    (Path(export) / name).write_bytes(data)
    man = pc.read_json(Path(export) / "manifest.json")
    man["files"][name] = {"sha256": pc.sha_bytes(data), "bytes": len(data), "rows": n}
    (Path(export) / "manifest.json").write_text(json.dumps(man, sort_keys=True, indent=1) + "\n")


def main(argv: list[str] | None = None) -> int:
    import sys

    ap = argparse.ArgumentParser(prog="python -m slh.xp.analysis.simulate")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--test-cases", type=int, default=90)
    ap.add_argument("--effect", action="append", default=[],
                    help="WRITER=POINTS: learned-skill gain over its search control")
    ap.add_argument("--run-sd", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--human", action="store_true", help="include a person-written arm")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    effects = Effects(run_sd=args.run_sd)
    for item in args.effect:
        w, _, v = item.partition("=")
        effects.writer[w]["learned"] = effects.writer[w]["search_control"] + float(v)
    paths = write_delivery(args.out, effects=effects, runs=args.runs,
                           test_cases=args.test_cases, human=args.human, seed=args.seed)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
