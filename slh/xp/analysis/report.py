"""Results day in one command: check, analyse as registered, report.

    python -m slh.xp.analysis.report --export DIR [--export DIR ...] --plan plan.json \
        --registration registration.json --membership membership.json --keys KEYS_DIR \
        --out OUT_DIR [--allow-incomplete] [--allow-dry-run] [--ledger gpu-hours.jsonl] \
        [--deviations deviations.json]

1. Our checker (``slh.xp.partner.check``) verifies and scores the delivery against
   our answer keys; its scores are the only scores used. Status ``failed``: refused.
   Status ``incomplete``: refused unless ``--allow-incomplete``, which writes a
   clearly labelled DESCRIPTIVE report (no confirmatory verdict, no gate).
2. Descriptive tables: per task and pooled (equal task weights), per arm accuracy,
   wrong-write, unnecessary-question and missed-clarification rates and outcome
   counts; per-run endpoint scores; learning curves (dev and sentinel) from the
   learning rows; costs and GPU-hours from the logs.
3. The registered analysis (docs/xp/plan-v1.md v1.2, docs/xp/power.md;
   ``registered`` below): primary T+ endpoint vs its search control, secondary
   S+ (self-written, advisory self-critic) vs its search control tested only if the primary succeeds;
   crossed run x group parametric bootstrap (``stats.crossed``); superiority =
   lower 95% bound > 0; wrong-write non-inferiority = upper 95% bound < 2 points.
4. The phase-2 gate as registered (``gate``).
5. ``OUT/results.json``, ``OUT/results.md``, ``OUT/results.html``.

Exit code: 0 analysed (complete delivery), 1 descriptive report of an incomplete
delivery, 2 refused.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from slh.xp.analysis import stats
from slh.xp.partner import check as ck
from slh.xp.partner import common as pc

SCHEMA = "xp-phase1-results-v1"
RATES = ("correct", "wrong_write", "unnecessary_question", "missed_clarification")
# PROPOSED AMENDMENT (not in plan v1.2; docs/xp/analysis.md): when the case-level
# error rate (errors and time limits) of a confirmatory endpoint and its search
# control differ by more than this many points (pooled over tasks and runs), the
# contrast is read as descriptive only, since unequal failures alone shift accuracy.
ERROR_RATE_MARGIN = 2.0

# --------------------------------------------------------------------------- registered

REGISTERED = {
    "source": "plan v1.2 (docs/xp/plan-v1.md, Registered design) and docs/xp/power.md",
    "primary": {"writer": "T+", "endpoint": "learned", "control": "search_control",
                "metric": "accuracy on the sealed test, pooled over tasks (equal weights)"},
    "secondary": {"writer": "S+", "endpoint": "learned", "control": "search_control",
                  "critic": "self (advisory)", "tested_only_if": "the primary succeeds"},
    "superiority": "lower bound of the two-sided 95% interval above 0 points",
    "minimum_worthwhile_gain": 5.0,
    "wrong_write_margin": 2.0,
    "unit": "per-case paired difference (same case, same run), averaged within scenario "
            "group, crossed with learning run",
    "interval": f"crossed random-effects model (run x group, task fixed), parametric "
                f"bootstrap-t, {stats.DRAWS} draws, seed {stats.SEED}",
    "gate": "phase 2 only if the primary succeeds and wrong-write non-inferiority holds "
            "(primary and, if tested, secondary writer)",
}

AMBIGUITIES = [
    ("Gate scope of wrong-write non-inferiority: plan v1.2 defines it 'for the primary and, if "
    "tested, the secondary writer' and gates on 'wrong-write non-inferiority holds'. "
    "Conservative reading used: the gate needs both when the secondary is tested. The verdict "
    "under the narrower reading (primary writer only) is reported alongside."),
    ("Parametric bootstrap form: not specified. Bootstrap-t (studentised draws) is used, which "
    "carries the uncertainty of the estimated variance components; the percentile and the "
    "closed-form Satterthwaite intervals are reported as sensitivity."),
    ("Older gate wording (plan 'Phases and gates' table: endpoint vs its generic note and "
    "best-of-N, lower bound above a registered minimum gain, wrong writes vs B1) is "
    "superseded by v1.2 ('where they differ, v1.2 wins'): superiority over the search "
    "control at 0, with the 5-point minimum worthwhile gain reported, not gated."),
    ("'Cluster by request family where groups share a template' and the address-task "
    "user-cluster analysis are not part of the registered interval; both are reported as "
    "descriptive sensitivity checks (non-parametric run x cluster bootstrap)."),
    ("Wrong-write non-inferiority uses the same crossed model on the per-case wrong-write "
    "difference (endpoint minus search control) with a two-sided 95% upper bound."),
]

CAVEATS = [
    ("Single-use test set: the sealed test questions are public in the partner kit and their "
    "answers stay with us; this analysis is their one registered use. Any later look at the "
    "same cases is exploratory."),
    ("Honest-partner model: the partner ran all GPU work. Our checker verifies hashes, the "
    "registered plan, learning provenance and answer-key leakage, but cannot prove how the "
    "runs were executed beyond what the export records."),
    ("Synthetic identities: every customer, order and request is generated from the public "
    "benchmark database; no real person is involved, and results speak to these generated "
    "decision cases only."),
    ("Learning runs share training cases (overlapping 48-case samples from a 150- or 400-case "
    "pool), which makes runs less independent than the power simulation assumes; the measured "
    "overlap is reported."),
    ("The teacher was self-hosted at the partner (FP8), a recorded deviation from POC 9's "
    "hosted endpoint."),
    ("Per-task contrasts, all control arms and the T writer are descriptive; only the pooled "
    "primary contrast (and, behind its gate, the secondary) is confirmatory."),
]

ARM_LABELS = {"b0": "B0 (model only)", "b1": "B1 (base prompt; = removal)",
              "human": "Person-written procedure", "learned": "Learned skill (endpoint)",
              "generic": "Same-length generic note", "full_context": "Full context (evidence)",
              "search_control": "No-evidence search control", "best_of_n": "Best-of-N"}


class Refused(RuntimeError):
    def __init__(self, reason: str, report: dict[str, Any]) -> None:
        super().__init__(reason)
        self.report = report


# --------------------------------------------------------------------------- helpers


def _f(x: Any, nd: int | None = None) -> Any:
    """JSON-safe values; rounds only when ``nd`` is given (results.json keeps the
    unrounded numbers the gate decided on; rounding is for display)."""
    if isinstance(x, float):
        return None if not math.isfinite(x) else (x if nd is None else round(x, nd))
    if isinstance(x, dict):
        return {k: _f(v, nd) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_f(v, nd) for v in x]
    if isinstance(x, np.floating):
        return _f(float(x), nd)
    if isinstance(x, np.integer):
        return int(x)
    return x


_ABS = re.compile(r"(?:[A-Za-z]:)?(?:/(?:Users|home|private|tmp|var|mnt|scratch|work|opt|srv)"
                  r"|~)/[^\s'\"),;]*")


def scrub_paths(text: str, known: list[str] | None = None) -> str:
    for i, p in enumerate(sorted(known or [], key=len, reverse=True)):
        if p:
            text = text.replace(p, f"<input {i + 1}>")
    return _ABS.sub("<path>", text)


def key_paths(keys_dir: Path) -> dict[str, Path]:
    return {p.name.split("-test.")[0]: p
            for p in sorted(Path(keys_dir).glob("*-test.answer-key.jsonl"))}


# --------------------------------------------------------------------------- data


class Data:
    """Scored cases per arm instance, aligned on our membership order."""

    def __init__(self, check: dict[str, Any], plan: dict[str, Any],
                 members: dict[str, list[str]], keys: dict[str, dict[str, dict[str, Any]]]):
        self.plan = plan
        self.tasks = [t for t in plan["tasks"]]
        self.runs = list(range(int(plan["runs"])))
        self.writers = list(plan["writers"])
        self.arms = {a["name"]: a for a in plan["arms"]}
        self.members = members
        self.index = {t: {c: i for i, c in enumerate(ids)} for t, ids in members.items()}
        self.group = {}
        self.family = {}
        self.user = {}
        for t, ids in members.items():
            key = keys[t]
            self.group[t] = np.array([key[c]["group"] for c in ids])
            self.family[t] = np.array([str(key[c].get("family")) for c in ids])
            self.user[t] = np.array([str((key[c].get("entities") or {}).get("user_id"))
                                     for c in ids])
        self.inst: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.error_rate: dict[tuple[Any, ...], float] = {}
        self.vec: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}
        for e in check.get("instances", []):
            k = (e["task"], e["run"], e["writer"], e["arm"])
            self.inst[k] = e
            if e.get("n"):
                self.error_rate[k] = 100.0 * e.get("errors", 0) / e["n"]
            if not e.get("cases"):
                continue
            n = len(members[e["task"]])
            v = {f: np.full(n, np.nan) for f in RATES}
            outcome = np.full(n, "", dtype=object)
            for c in e["cases"]:
                i = self.index[e["task"]].get(c["case_id"])
                if i is None:
                    continue
                for f in RATES:
                    v[f][i] = float(bool(c.get(f)))
                outcome[i] = c.get("outcome", "")
            v["outcome"] = outcome
            self.vec[k] = v

    def vector(self, task: str, run: int, writer: str | None, arm: str, metric: str
               ) -> np.ndarray | None:
        spec = self.arms.get(arm)
        if spec is None:
            return None
        if spec["scope"] == "shared":
            k = (task, None, None, arm)
        else:
            k = (task, run, writer, arm)
        v = self.vec.get(k)
        return None if v is None else v[metric]

    def has_arm(self, arm: str) -> bool:
        return arm in self.arms

    def pooled_error_rate(self, writer: str | None, arm: str) -> float | None:
        """Mean case-level error rate (points) over tasks (equal weights) and runs."""
        spec = self.arms.get(arm)
        if spec is None:
            return None
        per_task = []
        for t in self.tasks:
            keys = [(t, None, None, arm)] if spec["scope"] == "shared" else \
                [(t, r, writer, arm) for r in self.runs]
            vals = [self.error_rate[k] for k in keys if k in self.error_rate]
            if vals:
                per_task.append(statistics.mean(vals))
        return statistics.mean(per_task) if per_task else None


# --------------------------------------------------------------------------- contrasts


def _group_matrix(diffs: list[np.ndarray], groups: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray] | None:
    """Runs x groups means of the per-case differences; only groups with at least
    one scored case in every run (complete deliveries: all of them)."""
    if not diffs:
        return None
    m = np.vstack(diffs)
    labels, inv = np.unique(groups, return_inverse=True)
    ok = ~np.isnan(m)
    sums = np.zeros((m.shape[0], len(labels)))
    cnts = np.zeros((m.shape[0], len(labels)))
    np.add.at(sums, (slice(None), inv), np.where(ok, m, 0.0))
    np.add.at(cnts, (slice(None), inv), ok.astype(float))
    keep = (cnts > 0).all(axis=0)
    if keep.sum() == 0:
        return None
    return sums[:, keep] / cnts[:, keep], labels[keep]


def contrast_mats(data: Data, writer: str | None, endpoint: str, ctrl_writer: str | None,
                  control: str, metric: str) -> dict[str, Any]:
    """task -> runs x groups differences (points) and the runs used."""
    per_task: dict[str, Any] = {}
    for t in data.tasks:
        diffs, runs = [], []
        for r in data.runs:
            a = data.vector(t, r, writer, endpoint, metric)
            b = data.vector(t, r, ctrl_writer, control, metric)
            if a is None or b is None:
                continue
            diffs.append(100.0 * (a - b))
            runs.append(r)
        gm = _group_matrix(diffs, data.group[t])
        if gm is not None:
            per_task[t] = {"mat": gm[0], "labels": gm[1], "runs": runs}
    return per_task


def analyse_contrast(data: Data, name: str, writer: str | None, endpoint: str,
                     ctrl_writer: str | None, control: str, *, role: str,
                     sensitivity: bool = False) -> dict[str, Any] | None:
    out: dict[str, Any] = {"name": name, "role": role, "writer": writer,
                           "endpoint": endpoint, "control": control,
                           "control_writer": ctrl_writer}
    for metric, label in (("correct", "accuracy"), ("wrong_write", "wrong_write")):
        per = contrast_mats(data, writer, endpoint, ctrl_writer, control, metric)
        if not per:
            return None
        common = sorted(set.intersection(*(set(p["runs"]) for p in per.values())))
        complete = len(per) == len(data.tasks) and all(
            p["runs"] == common for p in per.values())
        mats = {}
        for t, p in per.items():
            rows = [p["runs"].index(r) for r in common]
            mats[t] = p["mat"][rows]
        if not common:
            return None
        res = stats.crossed(mats)
        res["runs_used"] = common
        res["tasks_used"] = list(mats)
        res["complete"] = complete
        if sensitivity and metric == "correct":
            res["sensitivity"] = _sensitivity(data, per, common)
        out[label] = res
    return out


def _sensitivity(data: Data, per: dict[str, Any], runs: list[int]) -> dict[str, Any]:
    out = {}
    for cluster in ("group", "family", "user"):
        sums, counts = {}, {}
        for t, p in per.items():
            rows = [p["runs"].index(r) for r in runs]
            m = p["mat"][rows]  # runs x groups
            g_lab = p["labels"]
            first = {}
            labels = getattr(data, cluster)[t]
            for gi, g in enumerate(data.group[t]):
                first.setdefault(g, labels[gi])
            cl = np.array([first[g] for g in g_lab])
            uniq, inv = np.unique(cl, return_inverse=True)
            s = np.zeros((m.shape[0], len(uniq)))
            n = np.zeros((m.shape[0], len(uniq)))
            np.add.at(s, (slice(None), inv), m)
            np.add.at(n, (slice(None), inv), 1.0)
            sums[t], counts[t] = s, n
        out[cluster] = stats.pigeonhole(sums, counts)
    return out


# --------------------------------------------------------------------------- gate


def gate(status: str, mode: str, primary: dict[str, Any] | None,
         secondary: dict[str, Any] | None, *, minimum: float = 5.0, margin: float = 2.0,
         blocked: str | None = None) -> dict[str, Any]:
    """Phase 1 -> 2 as registered (plan v1.2), conservative reading of the
    non-inferiority scope (see AMBIGUITIES)."""

    def sup(c: dict[str, Any] | None) -> dict[str, Any]:
        iv = (c or {}).get("accuracy", {}).get("interval")
        est = (c or {}).get("accuracy", {}).get("estimate")
        return {"estimate": est, "interval": iv,
                "superiority": bool(iv is not None and iv[0] > 0),
                "reaches_minimum_worthwhile_gain": bool(est is not None and est >= minimum)}

    def ni(c: dict[str, Any] | None) -> dict[str, Any]:
        iv = (c or {}).get("wrong_write", {}).get("interval")
        est = (c or {}).get("wrong_write", {}).get("estimate")
        return {"estimate": est, "interval": iv, "margin": margin,
                "holds": bool(iv is not None and iv[1] < margin)}

    out: dict[str, Any] = {"rule": REGISTERED["gate"], "status": status, "mode": mode}
    if status != "ok" or primary is None or blocked:
        out.update(phase2="NOT EVALUABLE", evaluable=False,
                   reason="the delivery is incomplete; no confirmatory reading"
                   if status != "ok" else blocked if blocked
                   else "the primary contrast could not be computed")
        return out
    p_sup, p_ni = sup(primary), ni(primary)
    tested = p_sup["superiority"]
    s_sup = sup(secondary) if secondary is not None else None
    s_ni = ni(secondary) if secondary is not None else None
    out.update(evaluable=True, primary=p_sup, primary_wrong_write=p_ni,
               secondary={"tested": tested and secondary is not None,
                          "gate_reason": "tested because the primary succeeded" if tested
                          else "not tested: the primary did not succeed (reported as "
                               "descriptive)",
                          **(s_sup or {}), "wrong_write": s_ni})
    sec_ni_ok = True
    if tested and s_ni is not None:
        sec_ni_ok = s_ni["holds"]
    go = p_sup["superiority"] and p_ni["holds"] and sec_ni_ok
    out["phase2"] = "GO" if go else "NO-GO"
    out["phase2_narrow_reading"] = "GO" if p_sup["superiority"] and p_ni["holds"] else "NO-GO"
    reasons = []
    if not p_sup["superiority"]:
        reasons.append("primary: the lower 95% bound is not above 0")
    if not p_ni["holds"]:
        reasons.append(f"primary: wrong-write upper bound not below {margin:g} points")
    if tested and s_ni is not None and not s_ni["holds"]:
        reasons.append(f"secondary: wrong-write upper bound not below {margin:g} points")
    out["reasons"] = reasons
    if mode != "real":
        out["label"] = "REHEARSAL: dry-run (synthetic) delivery; not a result"
    return out


# --------------------------------------------------------------------------- descriptive


def arm_table(data: Data) -> list[dict[str, Any]]:
    """Per task and pooled, per (writer, arm): mean over runs of instance rates."""
    rows = []
    specs = []
    for name, a in data.arms.items():
        if a["scope"] == "shared":
            specs.append((None, name))
        else:
            specs += [(w, name) for w in data.writers]
    for writer, arm in specs:
        per_task = {}
        for t in data.tasks:
            runs = [None] if writer is None else data.runs
            rates = defaultdict(list)
            outcomes: Counter = Counter()
            n = 0
            for r in runs:
                v = data.vec.get((t, r, writer, arm))
                if v is None:
                    continue
                for f in RATES:
                    vals = v[f][~np.isnan(v[f])]
                    if vals.size:
                        rates[f].append(100.0 * vals.mean())
                outcomes.update(x for x in v["outcome"] if x)
                if (t, r, writer, arm) in data.error_rate:
                    rates["error_rate"].append(data.error_rate[(t, r, writer, arm)])
                n += 1
            if not n:
                continue
            per_task[t] = {**{f: statistics.mean(rates[f]) if rates[f] else None
                              for f in RATES},
                           "error_rate": statistics.mean(rates["error_rate"])
                           if rates["error_rate"] else None,
                           "run_sd": statistics.stdev(rates["correct"])
                           if len(rates["correct"]) > 1 else None,
                           "instances": n, "outcomes": dict(outcomes)}
        if not per_task:
            rows.append({"writer": writer, "arm": arm, "per_task": {}, "pooled": None,
                         "missing": True})
            continue
        pooled = None
        if len(per_task) == len(data.tasks):
            pooled = {f: statistics.mean(per_task[t][f] for t in data.tasks)
                      for f in (*RATES, "error_rate")
                      if all(per_task[t][f] is not None for t in data.tasks)}
            pooled["outcomes"] = dict(sum((Counter(p["outcomes"]) for p in per_task.values()),
                                          Counter()))
        rows.append({"writer": writer, "arm": arm, "per_task": per_task, "pooled": pooled,
                     "missing": len(per_task) < len(data.tasks)})
    return rows


def endpoint_runs(data: Data, learning: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = {(r["task"], r["run"], r["writer"]): r["row"] for r in learning
                 if r["source"] == "summary"}
    out = []
    for w in data.writers:
        for t in data.tasks:
            for r in data.runs:
                a = data.vector(t, r, w, "learned", "correct")
                b = data.vector(t, r, w, "search_control", "correct")
                s = summaries.get((t, r, w)) or {}
                sel = (s.get("selection") or {})
                out.append({
                    "writer": w, "task": t, "run": r,
                    "learned": None if a is None else 100 * float(np.nanmean(a)),
                    "search_control": None if b is None else 100 * float(np.nanmean(b)),
                    "difference": None if a is None or b is None
                    else 100 * float(np.nanmean(a - b)),
                    "learned_reverted": (sel.get("learned") or {}).get("reverted"),
                    "control_reverted": (sel.get("search_control") or {}).get("reverted"),
                    "accepted": s.get("accepted"), "proposals": s.get("proposals")})
    return out


def _carry(seq: list[float | None]) -> list[float]:
    out, last = [], None
    for x in seq:
        last = x if x is not None else last
        out.append(last)
    return out


def learning_curves(learning: list[dict[str, Any]], tasks: list[str], writers: list[str]
                    ) -> dict[str, Any]:
    """Dev curves (ops loop: dev accuracy of the current library after each
    proposal), training-stream accuracy per batch (note loop), sentinel scores."""
    by: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list))
    for r in learning:
        by[(r["task"], r["writer"], r["run"])][r["source"]].append(r["row"])
    curves: dict[str, Any] = {}
    for (t, w, run), src in sorted(by.items(), key=str):
        s = (src.get("summary") or [{}])[0]
        entry: dict[str, Any] = {"loop": s.get("loop")}
        for source, label in (("trace", "learned"), ("search_trace", "search_control")):
            rows = src.get(source) or []
            if s.get("loop") == "ops":
                start = next((x.get("dev_score") for x in rows if x.get("type") == "start"),
                             None)
                cur, seq = start, [start]
                for x in rows:
                    if x.get("type") == "step":
                        if x.get("accepted") and x.get("dev_after") is not None:
                            cur = x["dev_after"]
                        seq.append(cur)
                entry[label] = {"kind": "dev", "values": [None if v is None else 100 * v
                                                         for v in _carry(seq)]}
        if s.get("loop") == "note":
            eps = sorted(src.get("episodes") or [], key=lambda x: x.get("stream_position", 0))
            vals = [float(bool((e.get("score") or {}).get("correct"))) for e in eps]
            batch = 4
            entry["learned"] = {"kind": "train stream",
                                "values": [100 * statistics.mean(vals[i:i + batch])
                                           for i in range(0, len(vals), batch)]}
        sel = s.get("selection") or {}
        entry["sentinel"] = {k: {"scores": [100 * x["score"] for x in sorted(
            (v or {}).get("scores") or [], key=lambda x: x["index"])],
            "reverted": (v or {}).get("reverted")} for k, v in sel.items()}
        curves.setdefault(f"{w}|{t}", {})[str(run)] = entry
    mean: dict[str, Any] = {}
    for key, runs in curves.items():
        out = {}
        for label in ("learned", "search_control"):
            seqs = [e[label]["values"] for e in runs.values() if label in e
                    and e[label]["values"] and e[label]["values"][0] is not None]
            if not seqs:
                continue
            n = max(len(x) for x in seqs)
            padded = np.array([[*x, *([x[-1]] * (n - len(x)))] for x in seqs], dtype=float)
            out[label] = {"kind": next(e[label]["kind"] for e in runs.values() if label in e),
                          "mean": [float(v) for v in np.nanmean(padded, axis=0)],
                          "runs": [[float(v) for v in row] for row in padded]}
        out["sentinel_reverted"] = {
            arm: sum(1 for e in runs.values()
                     if (e["sentinel"].get(arm) or {}).get("reverted"))
            for arm in ("learned", "search_control")}
        out["runs"] = len(runs)
        mean[key] = out
    return mean


def ledger_hours(rows: list[dict[str, Any]]) -> float | None:
    """GPU-hours from ledger rows (same rule as the Slurm ledger: start/end pairs,
    charge rows, open jobs charged their reservation)."""
    if not rows:
        return None
    used, starts, charges = 0.0, {}, {}
    for r in rows:
        ev = r.get("event")
        if ev == "charge":
            charges[r["job"]] = r
        elif ev == "start":
            starts[r["job"]] = r
        elif ev == "end" and r.get("job") in starts:
            s = starts.pop(r["job"])
            used += max(0.0, r["epoch"] - s["epoch"]) / 3600 * s["gpus"]
    used += sum(max(0.0, r["epoch"] - r.get("start_epoch", r["epoch"])) / 3600 * r["gpus"]
                for r in charges.values())
    used += sum(s["gpus"] * s.get("limit_s", 0) / 3600 for s in starts.values())
    return used


def costs(logs: list[dict[str, Any]], learning: list[dict[str, Any]],
          decisions_wall: float, ledger: list[dict[str, Any]] | None) -> dict[str, Any]:
    cost_rows = [r["row"] for r in logs if r["name"] == "costs.jsonl"]
    dec = {k: sum(int(r.get(k) or 0) for r in cost_rows)
           for k in ("decisions", "model_calls", "prompt_tokens", "completion_tokens",
                     "errors", "time_limits")}
    stage_wall = sum(float(r["row"].get("wall_s") or 0) for r in logs
                     if r["name"] == "stages.jsonl" and r["row"].get("stage") == "learning")
    budget = Counter()
    for r in learning:
        if r["source"] == "summary":
            for k, v in ((r["row"].get("budgets") or {}).get("total") or {}).items():
                budget[k] += int(v)
    in_logs = [r["row"] for r in logs if r["name"] in ("gpu-hours.jsonl", "ledger.jsonl")]
    rows = ledger if ledger is not None else in_logs
    hours = ledger_hours(rows)
    return {"decisions": dec, "decision_agent_hours": decisions_wall / 3600,
            "learning_wall_hours": stage_wall / 3600, "learning_budget_used": dict(budget),
            "gpu_hours": hours,
            "gpu_hours_source": None if hours is None else
            ("ledger file" if ledger is not None else "export logs"),
            "by_task": cost_rows}


# --------------------------------------------------------------------------- run


CONTRASTS_DESCRIPTIVE = ("search_control", "b1", "b0", "generic", "full_context", "best_of_n",
                         "human")


def analyse(exports: list[Path], *, plan: dict[str, Any], registration: dict[str, Any],
            membership: dict[str, Any], keys_dir: Path, allow_incomplete: bool = False,
            allow_dry: bool = False, ledger: list[dict[str, Any]] | None = None,
            deviations: list[str] | None = None,
            package: Path | None = None) -> dict[str, Any]:
    known = [str(Path(e)) for e in exports] + [str(Path(e).resolve()) for e in exports] + \
        [str(keys_dir), str(Path(keys_dir).resolve())]
    keys = key_paths(keys_dir)
    report = ck.check(list(exports), registration, keys, plan=plan, membership=membership,
                      reveal_incomplete=allow_incomplete, allow_dry=allow_dry, case_rows=True,
                      package=package)
    status = report["status"]
    check_info = {"status": status, "mode": report.get("mode"),
                  "reason": scrub_paths(report.get("reason") or "", known) or None,
                  "decision_rows": report.get("decision_rows"),
                  "coverage_gaps": len(report.get("coverage_gaps") or []),
                  "gap_kinds": dict(Counter(g["problem"] for g in
                                            report.get("coverage_gaps") or [])),
                  "shards": [{"tasks": s["tasks"], "bytes": s["bytes"]}
                             for s in report.get("shards") or []]}
    if status == "failed":
        raise Refused(f"the checker failed the delivery: {check_info['reason']}",
                      {"schema": SCHEMA, "status": "refused", "check": check_info})
    if status == "incomplete" and not allow_incomplete:
        raise Refused("the delivery is incomplete (coverage gaps); rerun with "
                      "--allow-incomplete for a descriptive report",
                      {"schema": SCHEMA, "status": "refused", "check": check_info})
    members = ck.load_membership(membership, registration, plan)
    key_rows = ck.load_keys(keys, members)
    data = Data(report, plan, members, key_rows)
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in exports:
        for name in ("learning.jsonl.gz", "logs.jsonl.gz", "decisions.jsonl.gz"):
            rows[name] += pc.read_jsonl(Path(e) / name)
    wall = sum(float(r.get("wall_s") or 0) for r in rows["decisions.jsonl.gz"])
    learning, logs = rows["learning.jsonl.gz"], rows["logs.jsonl.gz"]

    pairs = {w: {"endpoint": data.pooled_error_rate(w, "learned"),
                 "search_control": data.pooled_error_rate(w, "search_control")}
             for w in ("T+", "S+") if w in data.writers}
    unequal = [f"{w}: endpoint {v['endpoint']:.2f}% vs search control "
               f"{v['search_control']:.2f}% case-level errors"
               for w, v in pairs.items() if v["endpoint"] is not None
               and v["search_control"] is not None
               and abs(v["endpoint"] - v["search_control"]) > ERROR_RATE_MARGIN]
    blocked = (f"error rates of a confirmatory pair differ by more than "
               f"{ERROR_RATE_MARGIN:g} points ({'; '.join(unequal)}); descriptive only "
               "(proposed amendment)") if unequal else None
    confirm = status == "ok" and blocked is None
    why = "incomplete delivery" if status != "ok" else "unequal error rates" if blocked \
        else None
    primary = secondary = None
    if "T+" in data.writers:
        primary = analyse_contrast(data, "T+ endpoint vs its search control", "T+", "learned",
                                   "T+", "search_control",
                                   role="confirmatory (primary)" if confirm
                                   else f"descriptive ({why})", sensitivity=True)
    if "S+" in data.writers:
        secondary = analyse_contrast(data, "S+ (self-written, advisory self-critic) endpoint vs its search control",
                                     "S+", "learned", "S+", "search_control",
                                     role="confirmatory (secondary, gated)" if confirm
                                     else f"descriptive ({why})")
    g = gate(status, report.get("mode") or "?", primary, secondary,
             minimum=REGISTERED["minimum_worthwhile_gain"],
             margin=REGISTERED["wrong_write_margin"], blocked=blocked)
    if secondary is not None and confirm and not g.get("secondary", {}).get("tested"):
        secondary["role"] = "not tested (primary failed); descriptive"
        secondary["tested"] = False

    descriptive = []
    for w in data.writers:
        for ctrl in CONTRASTS_DESCRIPTIVE:
            if not data.has_arm(ctrl):
                continue
            if ctrl == "search_control" and w in ("T+", "S+"):
                continue
            cw = w if data.arms[ctrl]["scope"] == "writer" else None
            c = analyse_contrast(data, f"{w} endpoint vs {ARM_LABELS.get(ctrl, ctrl)}", w,
                                 "learned", cw, ctrl, role="descriptive")
            if c is not None:
                descriptive.append(c)
    if data.has_arm("b1") and data.has_arm("b0"):
        c = analyse_contrast(data, "B1 vs B0", None, "b1", None, "b0", role="descriptive")
        if c is not None:
            descriptive.append(c)

    missing_arms = []
    if not data.has_arm("human"):
        missing_arms.append("Person-written procedure: not in the registered plan (left out "
                            "pending an owner decision).")
    missing_arms.append("Removal: equals B1 by design (not run separately).")
    for row in arm_table(data):
        if row["missing"]:
            missing_arms.append(f"{row['writer'] or 'shared'} / {row['arm']}: missing for "
                                f"some tasks or runs in this delivery.")
    dev_list = list(deviations or [])
    if report.get("mode") != "real":
        dev_list.insert(0, "Dry-run (synthetic) delivery: a rehearsal of the analysis, "
                           "not a result.")
    if blocked:
        dev_list.insert(0, f"DESCRIPTIVE report only: {blocked}.")
    if status != "ok":
        dev_list.insert(0, f"Incomplete delivery ({check_info['coverage_gaps']} coverage "
                           "gaps): DESCRIPTIVE report only; no confirmatory verdict.")
    samples = (plan.get("learning") or {}).get("train_samples") or {}
    overlap = {t: samples[t].get("overlap") for t in data.tasks if t in samples}
    groups = {t: len(np.unique(data.group[t])) for t in data.tasks}
    out = {
        "schema": SCHEMA,
        "status": "analysed" if confirm else "descriptive",
        "label": None if confirm else f"DESCRIPTIVE: {why}",
        "confirmatory": confirm,
        "error_rates": {"margin_points": ERROR_RATE_MARGIN, "status": "proposed amendment",
                        "pairs": pairs},
        "rehearsal": report.get("mode") != "real",
        "check": check_info,
        "design": {"plan_id": plan["plan_id"], "tasks": data.tasks, "writers": data.writers,
                   "runs": len(data.runs), "arms": list(data.arms),
                   "test_cases": {t: len(ids) for t, ids in members.items()},
                   "groups": groups, "critic": plan.get("critic")},
        "registered": REGISTERED,
        "primary": primary, "secondary": secondary, "gate": g,
        "descriptive_contrasts": descriptive,
        "arms": arm_table(data),
        "endpoints": endpoint_runs(data, learning),
        "learning_curves": learning_curves(learning, data.tasks, data.writers),
        "costs": costs(logs, learning, wall, ledger),
        "training_overlap": overlap,
        "missing_arms": missing_arms,
        "deviations": dev_list,
        "ambiguities": AMBIGUITIES,
        "caveats": CAVEATS,
    }
    return _f(out)


def write_outputs(results: dict[str, Any], out: Path) -> dict[str, Path]:
    from slh.xp.analysis import render

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pc.write_json(out / "results.json", results)
    (out / "results.md").write_text(render.markdown(results), encoding="utf-8")
    (out / "results.html").write_text(render.html(results), encoding="utf-8")
    return {"json": out / "results.json", "md": out / "results.md",
            "html": out / "results.html"}


def write_refusal(body: dict[str, Any], out: Path) -> None:
    """Replace every output (json, md, html) so no earlier verdict survives."""
    from slh.xp.analysis import render

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pc.write_json(out / "results.json", body)
    (out / "results.md").write_text(render.refusal_markdown(body), encoding="utf-8")
    (out / "results.html").write_text(render.refusal_html(body), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.analysis.report")
    ap.add_argument("--export", type=Path, action="append", required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--registration", type=Path, required=True)
    ap.add_argument("--membership", type=Path, required=True)
    ap.add_argument("--keys", type=Path, required=True,
                    help="directory with <task>-test.answer-key.jsonl files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write a labelled descriptive report for an incomplete delivery")
    ap.add_argument("--allow-dry-run", action="store_true",
                    help="accept a dry-run (synthetic) delivery; labelled as a rehearsal")
    ap.add_argument("--ledger", type=Path, help="GPU-hour ledger (gpu-hours.jsonl)")
    ap.add_argument("--deviations", type=Path,
                    help="JSON list of deviations logged before unblinding")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    ledger = pc.read_jsonl(args.ledger) if args.ledger else None
    deviations = pc.read_json(args.deviations) if args.deviations else None
    try:
        results = analyse(args.export, plan=pc.read_json(args.plan),
                          registration=pc.read_json(args.registration),
                          membership=pc.read_json(args.membership), keys_dir=args.keys,
                          allow_incomplete=args.allow_incomplete,
                          allow_dry=args.allow_dry_run, ledger=ledger, deviations=deviations,
                          package=args.plan.parent)
    except Refused as exc:
        write_refusal({**exc.report, "reason": str(exc)}, args.out)
        print(f"refused: {exc}")
        return 2
    paths = write_outputs(results, args.out)
    g = results["gate"]
    print(f"status: {results['status']}; phase 2: {g.get('phase2')}"
          + (f" ({g['label']})" if g.get("label") else ""))
    for p in paths.values():
        print(f"wrote {p.name}")
    return 0 if results["status"] == "analysed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
