"""Our side of the S1 headroom probe (docs/xp/s1-headroom-probe-design.md,
docs/xp/s1-headroom-prereg-additions.md): the analysis-side parts the kit leaves to us,
FROZEN before any probe outcome is read (its file hash is recorded in
docs/xp/mh1-return-runbook.md and checked by tests/test_mh1_return.py).

Order (each step refuses out of order):

1. ``calibrate``: reads ONLY decisions of pool C (prereg s. 7) and their branches.
   Threshold / k / fixed-rule choice on calibration-fit, ONE evaluation on
   calibration-audit (the plan's preassigned ``c_part``), plus 20 family-grouped
   fit/audit partitions inside C as sensitivity (seeds 20261025 ... 20261044; the
   first, primary, partition IS the plan's preassigned split). Writes ``calibration.json``;
   commit and push it before step 2.
2. ``report --calibration-sha256 SHA``: the design's ``probe.analyze`` (primary), an
   independent recomputation of the oracles, the matched-real-minus-placebo headroom per
   decision type with a cluster-bootstrap interval, HP1-HP3 classifications, the null
   ladder rung per type, the review-skip (type 2) label statistics, the all-in cost
   ledger, and the committed calibration points assessed on the untouched pool P.
3. ``self-audit``: the pre-registered 9-item checklist (prereg s. 7), each item pass /
   fail / not testable, with input hashes and a separate recomputation.

    PY=".venv/bin/python -I slh/xp/ourside/s1_probe_ourside.py"
    $PY calibrate  --export E --package package-mh1 --out W/calibration.json
    $PY freeze-record --calibration W/calibration.json --out docs/xp/s1-calibration-frozen.json
    #   git commit + push that record, then:
    $PY report     --export E --package package-mh1 --calibration W/calibration.json \
                   --calibration-sha256 SHA --out W/probe-report.json
    $PY self-audit --export E --package package-mh1 --calibration W/calibration.json \
                   --report W/probe-report.json --out W/self-audit.json

Frozen definitions (all pre-registered here, before outcomes):

* Outcome order "not worse" = the kit's oracle rule: success not lower, harm not higher,
  critical not new, scored not lost (``probe.analyze``).
* Type 1 chunk labels (prereg s. 3.1, conservative): on replicate 0, an eligible chunk is
  KEEP-POSITIVE if ANY executed real deletion option that removes it (kinds stub,
  reversible, native_ids; placebos and Hermes' native records excluded) is worse than
  keep or not strictly cheaper (branch cost, no chooser request); DELETE-SAFE otherwise;
  never removed = missing (counted, excluded).
* Type 3 positives: a decision is positive when some real alternative (retry, repair,
  repair_fixed, stop) is, on replicate 0, not worse than native AND strictly better in
  success or strictly cheaper. A policy MISSES a positive when its choice is not one of
  those alternatives.
* Type 2 (review) calibration: not estimable in this probe (no future-value labels;
  learning-history clones are deferred; prereg s. 5.2 forbids empty-fork proxies).
* Operating points. Type 1: ``keep`` (incumbent), controls ``last_k{1,2,5,10}``,
  selectors ``P9@t`` (the prompted 9B's drop set when its confidence >= t) and ``CL@t``
  (``probe_arms.FeatureScorer`` fitted on pool F chunk labels; drops eligible chunks with
  keep-probability < t; scored only when its drop set equals an executed option's).
  Type 3 (per stratum natural / injected): ``native`` (incumbent), controls ``always_retry``,
  ``always_repair``, ``always_repair_fixed`` (native when no fix is known), ``always_stop``;
  selector ``P9@t`` (its choice when confidence >= t). t on {0.00, 0.01, ..., 1.00}.
  Selector points always pay the chooser request (also when they fall back).
* Evaluation of a point on a decision set: its choice's replicate-1 row against the
  incumbent's replicate-1 row (labels come from replicate 0). Quality bounds are exact
  one-sided 95 % split bounds (``slh.system.stats``): success diff lower >= -0.02, harm
  diff upper <= +0.01, non-scored diff upper <= +0.01; p95 seconds <= 1.05 x the
  incumbent's; no candidate-only critical event; every choice executed (coverage 1).
* Family-miss target: a positive-bearing family (group = the unit's user from the probe
  plan; the case when no plan is given) is missed if any positive in it is missed; the
  exact one-sided 95 % Clopper-Pearson upper bound on the family miss rate must be
  <= 0.05. Zero positive-bearing families: not estimable (cannot certify).
* Selection on calibration-fit: controls need the quality bounds; selectors need the
  quality bounds AND the family-miss target. Minimum mean cost, the incumbent included
  (a point must be strictly cheaper than the incumbent); ties: fewer risky actions, then
  the point id. None qualifies: the incumbent. A secondary run with point-estimate
  quality bars is reported, descriptive only (at this probe's calibration size the
  one-sided bounds cannot clear a 1-point harm bar; the frozen choice is the primary's). Audit once on
  calibration-audit with the same criteria; a failure returns the incumbent (no retuning).
* Headroom per type (design s. 2.2): per decision, assessed matched-real oracle minus
  assessed placebo oracle (success; cost share of the keep/native base episode), cluster
  bootstrap 10,000 resamples, seed 20261025, clusters = cases. Classified supported
  (lower bound >= bar), ruled out (upper bound < bar), else inconclusive (prereg s. 2);
  the owner's HP bars (design s. 3) are reported next to it.
* Ladder per type: R0 exposure (design s. 4 counts), R1 fidelity (<= 5 % diverged), R2
  noise (the headroom interval excludes 0 for success or cost), R3 the HP bar; the first
  failing rung is the conclusion; R4+ not tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a script under ``python -I``: only OUR root
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SCHEMA = "slh-s1probe-ourside-v1"
GRID = tuple(round(i / 100, 2) for i in range(101))
PARTITION_SEEDS = tuple(range(20261025, 20261045))  # 20 partitions; the first = the plan's
MISS_TARGET = 0.05
# THE SWITCH (owner, before any probe outcome): which quality bars choose the frozen
# calibration points. "bounds" = the pre-registered rule (prereg s. 7: one-sided 95 %
# bounds clear the bars); "points" = the point estimates clear them. The other mode is
# always reported as the clearly labelled secondary. Flipping it is a pre-registration
# change: commit it (new module hash in the runbook) BEFORE ``calibrate`` runs.
PRIMARY_QUALITY_MODE = "bounds"
QUALITY = {"success_lower": -0.02, "harm_upper": 0.01, "nonscored_upper": 0.01,
           "p95_ratio": 1.05}
ALPHA = 0.05  # one-sided bounds
DELETE_KINDS = frozenset({"stub", "reversible", "native_ids"})
REAL3 = ("retry", "repair", "repair_fixed", "stop")
HP_BARS = {"type1_cost": 0.10, "type1_cost_ci_lower_owner": 0.05, "type3_success": 0.03,
           "type3_cost": 0.10, "type2_F": 0.5, "type2_saving": 0.5, "type2_loss": 0.2,
           "type2_min_surviving": 5}
EXPOSURE = {"type1": 200, "type3": 100, "type2": 150}


class Refused(RuntimeError):
    pass


def _sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode() if isinstance(b, str) else b).hexdigest()


def module_sha256() -> str:
    return _sha(Path(__file__).read_bytes())


# ============================================================================ data


@dataclass
class Data:
    base: list[dict[str, Any]]
    decs: list[dict[str, Any]]
    brs: list[dict[str, Any]]
    revs: list[dict[str, Any]]
    review_prompted: list[dict[str, Any]]
    users: dict[str, str]
    plan_units: list[dict[str, Any]]
    files: dict[str, Any]

    def __post_init__(self) -> None:
        self.rows: dict[tuple[str, str, int], dict[str, Any]] = {}
        for r in self.brs:
            if not r.get("valid"):
                continue
            for n in r.get("options") or []:
                self.rows.setdefault((r["decision"], n, int(r["rep"])), r)
        self.base_cost: dict[str, float] = {}
        from slh.system.hosts.hermes import probe as PR

        for r in self.base:
            t = r.get("tokens") or [0, 0]
            self.base_cost[r["tag"]] = PR.COST_RATES["prefill"] * float(t[0] or 0) + \
                PR.COST_RATES["decode"] * float(t[1] or 0)

    def row(self, d: Mapping[str, Any], name: str, rep: int) -> dict[str, Any] | None:
        return self.rows.get((d["id"], name, rep))

    def group(self, d: Mapping[str, Any]) -> str:
        return self.users.get(d["case"], d["case"])


def load(exports: Sequence[Path], package: Path | None = None, *,
         pools: Sequence[str] | None = None) -> Data:
    """The probe parts of the exports. ``pools``: keep only decisions of these pools
    (``calibrate`` passes ("C",): no branch of another pool is ever indexed)."""
    from slh.system.hosts.hermes import probe as PR
    from slh.system.hosts.hermes.mh1 import _read_gz_jsonl

    base, decs, brs, revs, rp = [], [], [], [], []
    files: dict[str, Any] = {}
    for e in exports:
        e = Path(e)
        files[str(e)] = _sha(Path(e / "MANIFEST.json").read_bytes()) \
            if (e / "MANIFEST.json").exists() else None
        base += _read_gz_jsonl(e / "probe" / "base.jsonl.gz")
        decs += _read_gz_jsonl(e / "probe" / "decisions.jsonl.gz")
        brs += _read_gz_jsonl(e / "probe" / "branches.jsonl.gz")
        revs += _read_gz_jsonl(e / "probe" / "review-rows.jsonl.gz")
        rp += _read_gz_jsonl(e / "probe" / "review-prompted.jsonl.gz")
    if pools is not None:
        decs = [d for d in decs if (d.get("split") or {}).get("pool") in pools]
        keep = {d["id"] for d in decs}
        brs = [r for r in brs if r["decision"] in keep]
        revs, rp = [], []
    users: dict[str, str] = {}
    units: list[dict[str, Any]] = []
    if package is not None:
        plan = PR.load_plan(Path(package))
        if plan:
            units = list((plan.get("units") or {}).get("full") or [])
            for u in units + list((plan.get("units") or {}).get("smoke") or []):
                users.setdefault(u["case"], str(u["user"]))
    return Data(base, decs, brs, revs, rp, users, units, files)


# ============================================================================ outcomes


def not_worse(r: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """The kit's oracle order (``probe.analyze``), restated."""
    return (bool(r["success"]) >= bool(b["success"]) and bool(r["harm"]) <= bool(b["harm"])
            and (r["critical"] is False or b["critical"] is not False)
            and (r["status"] == "scored" or b["status"] != "scored"))


def branch_cost(r: Mapping[str, Any] | None) -> float | None:
    from slh.system.hosts.hermes import probe as PR

    return None if r is None else PR._cost(r)


def chooser_cost(d: Mapping[str, Any]) -> float:
    """The prompted 9B's own request in cost units (prefill 1, decode 3); 0 if none."""
    from slh.system.hosts.hermes import probe as PR

    t = (d.get("prompted") or {}).get("tokens") or [None, None]
    if t[0] is None:
        return 0.0
    return PR.COST_RATES["prefill"] * float(t[0]) + PR.COST_RATES["decode"] * float(t[1] or 0)


def ref_of(d: Mapping[str, Any]) -> str:
    return "keep" if d["type"] == 1 else "native"


def removed_idx(d: Mapping[str, Any], name: str) -> frozenset[int]:
    return frozenset(int(e["i"]) for e in (d.get("options") or {}).get(name) or []
                     if e.get("kind") in DELETE_KINDS)


def real_delete_options(d: Mapping[str, Any]) -> list[str]:
    from slh.system.hosts.hermes import probe as PR

    return [n for n in (d.get("options") or {}) if n != "keep" and n not in PR.PLACEBOS
            and n not in PR.NATIVE_ARMS and removed_idx(d, n)]


def chunk_labels(data: Data, d: Mapping[str, Any]) -> dict[int, bool]:
    """Type 1, replicate 0: chunk -> True (keep-positive) / False (delete-safe)."""
    b = data.row(d, "keep", 0)
    cb = branch_cost(b)
    out: dict[int, bool] = {}
    if b is None:
        return out
    for n in real_delete_options(d):
        r = data.row(d, n, 0)
        if r is None:
            continue
        ca = branch_cost(r)
        bad = not not_worse(r, b) or ca is None or cb is None or not ca < cb
        for i in removed_idx(d, n):
            out[i] = out.get(i, False) or bad
    return out


def beneficial3(data: Data, d: Mapping[str, Any]) -> set[str]:
    """Type 3, replicate 0: the real alternatives that beat native."""
    b = data.row(d, "native", 0)
    out: set[str] = set()
    if b is None:
        return out
    cb = branch_cost(b)
    for n in REAL3:
        r = data.row(d, n, 0)
        if r is None or not not_worse(r, b):
            continue
        ca = branch_cost(r)
        if bool(r["success"]) > bool(b["success"]) or (ca is not None and cb is not None
                                                       and ca < cb):
            out.add(n)
    return out


# ============================================================================ points


@dataclass(frozen=True)
class Point:
    id: str
    dtype: int
    role: str  # incumbent / control / selector
    choose: Callable[[Mapping[str, Any]], Any]  # -> option name, or frozenset (CL drop set)
    chooser: bool = False


def _conf(d: Mapping[str, Any]) -> float | None:
    p = d.get("prompted") or {}
    if p.get("abstain"):
        return None
    try:
        c = float(p.get("confidence"))
    except (TypeError, ValueError):
        return None
    return c if math.isfinite(c) else None


def points_type1(cl: Any = None) -> list[Point]:
    pts = [Point("keep", 1, "incumbent", lambda d: "keep")]
    for k in (1, 2, 5, 10):
        pts.append(Point(f"last_k{k}", 1, "control",
                         lambda d, k=k: f"last_k{k}" if f"last_k{k}" in d["options"]
                         else "keep"))

    def p9(t: float) -> Callable[[Mapping[str, Any]], str]:
        def f(d: Mapping[str, Any]) -> str:
            c = _conf(d)
            return "prompted" if (c is not None and c >= t and removed_idx(d, "prompted")
                                  ) else "keep"
        return f
    pts += [Point(f"P9@{t:.2f}", 1, "selector", p9(t), chooser=True) for t in GRID]
    if cl is not None:
        def clp(t: float) -> Callable[[Mapping[str, Any]], frozenset[int]]:
            def f(d: Mapping[str, Any]) -> frozenset[int]:
                return frozenset(i for i, p in cl(d).items() if p < t)
            return f
        pts += [Point(f"CL@{t:.2f}", 1, "selector", clp(t)) for t in GRID]
    return pts


def points_type3() -> list[Point]:
    pts = [Point("native", 3, "incumbent", lambda d: "native")]
    for n in REAL3:
        pts.append(Point(f"always_{n}", 3, "control",
                         lambda d, n=n: n if n in d["options"] else "native"))

    def p9(t: float) -> Callable[[Mapping[str, Any]], str]:
        def f(d: Mapping[str, Any]) -> str:
            c, ch = _conf(d), (d.get("prompted") or {}).get("choice")
            return ch if (c is not None and c >= t and ch in REAL3 and ch in d["options"]
                          ) else "native"
        return f
    return pts + [Point(f"P9@{t:.2f}", 3, "selector", p9(t), chooser=True) for t in GRID]


def fit_cl(data: Data, decs: Sequence[Mapping[str, Any]]
           ) -> tuple[Callable[[Mapping[str, Any]], dict[int, float]] | None, dict[str, Any]]:
    """CL: ``probe_arms.FeatureScorer`` (L2 logistic, the kit's own) fitted on pool F's
    type-1 chunk labels; label 1 = keep. Deviation (stated): the kit's scorer uses its
    four stored features (age, size, later-overlap, tool), not the prereg's ten; no class
    balancing. Returns (decision -> {chunk: keep probability}, report)."""
    from slh.system.hosts.hermes import probe_arms as PA

    def feats(d: Mapping[str, Any]) -> dict[int, Any]:
        out = {}
        for it in (d.get("inputs") or {}).get("items") or []:
            f = it.get("features")
            if it.get("eligible") and isinstance(f, Mapping) and "age" in f:
                out[int(it["i"])] = PA.ToolResultFeatures(
                    int(f["age"]), int(f["size"]), str(it.get("tool") or ""),
                    float(f.get("overlap") or 0.0))
        return out
    X, y = [], []
    for d in decs:
        if d["type"] != 1 or (d.get("split") or {}).get("pool") != "F":
            continue
        lab, fs = chunk_labels(data, d), feats(d)
        for i, v in lab.items():
            if i in fs:
                X.append(fs[i])
                y.append(int(v))
    rep = {"fit_chunks": len(X), "keep_labels": sum(y),
           "solver": "probe_arms.FeatureScorer (gradient descent, 500 iterations, lr 0.2, "
                     "l2 0.01)"}
    if not X:
        return None, {**rep, "status": "not estimable (no pool-F chunk labels or features)"}
    sc = PA.FeatureScorer().fit(X, y)
    rep["weights"] = [round(w, 6) for w in sc.weights]

    def score(d: Mapping[str, Any]) -> dict[int, float]:
        fs = feats(d)
        ks = sorted(fs)
        try:
            return dict(zip(ks, sc.predict_proba([fs[k] for k in ks]), strict=True))
        except ValueError:  # a tool unseen at fit: keep-probability 1 (never dropped)
            return {}
    return score, {**rep, "status": "fitted"}


# ============================================================================ evaluation


def _resolve(data: Data, d: Mapping[str, Any], choice: Any) -> str | None:
    """An option name for a choice; a CL drop set maps to an executed option with
    exactly that drop set (stub deletions), ``keep`` for an empty set, else None."""
    if not isinstance(choice, frozenset):
        return choice
    if not choice:
        return "keep"
    for n in sorted(real_delete_options(d)):
        if removed_idx(d, n) == choice and all(
                e.get("kind") == "stub" for e in d["options"][n]):
            return n
    return None


def evaluate(data: Data, pt: Point, decs: Sequence[Mapping[str, Any]],
             labels: Mapping[str, Any]) -> dict[str, Any]:
    """A point on a decision set (replicate 1 vs the incumbent's replicate 1; positives
    from replicate 0)."""
    from slh.system import stats as ST

    sp = sm = hp = hm = np_ = nm = n = 0
    crit_new = risky = bad = executed = 0
    costs: list[float] = []
    sec_a: list[float] = []
    sec_b: list[float] = []
    fam_pos: set[str] = set()
    fam_miss: set[str] = set()
    pos = miss = 0
    for d in decs:
        ref = ref_of(d)
        choice = pt.choose(d)
        name = _resolve(data, d, choice)
        dropped = choice if isinstance(choice, frozenset) else (
            removed_idx(d, name) if d["type"] == 1 and name else frozenset())
        g = data.group(d)
        if d["type"] == 1:
            lab = labels.get(d["id"]) or {}
            for i, v in lab.items():
                if v:
                    pos += 1
                    fam_pos.add(g)
                    if i in dropped:
                        miss += 1
                        fam_miss.add(g)
        else:
            ben = labels.get(d["id"]) or set()
            if ben:
                pos += 1
                fam_pos.add(g)
                if name not in ben:
                    miss += 1
                    fam_miss.add(g)
        b = data.row(d, ref, 1)
        a = data.row(d, name, 1) if name else None
        if name and name != ref:
            risky += 1
        if b is None or a is None:
            continue
        executed += 1
        n += 1
        x, y = bool(a["success"]), bool(b["success"])
        sp += x > y
        sm += x < y
        x, y = bool(a["harm"]), bool(b["harm"])
        hp += x > y
        hm += x < y
        x, y = a["status"] != "scored", b["status"] != "scored"
        np_ += x > y
        nm += x < y
        crit_new += a["critical"] is not False and b["critical"] is False
        if name != ref and not not_worse(a, b):
            bad += 1
        ca = branch_cost(a)
        if ca is not None:
            costs.append(ca + (chooser_cost(d) if pt.chooser else 0.0))
        if a.get("seconds") is not None and b.get("seconds") is not None:
            sec_a.append(float(a["seconds"]))
            sec_b.append(float(b["seconds"]))
    from slh.system.hosts.hermes import probe_cost as PC

    nf = len(fam_pos)
    out: dict[str, Any] = {
        "point": pt.id, "decisions": len(decs), "executed": executed,
        "coverage": executed / len(decs) if decs else None, "risky": risky,
        "bad_among_risky": bad, "bad_rate_risky": bad / risky if risky else None,
        "positives": pos, "missed_positives": miss,
        "miss_rate_raw": miss / pos if pos else None,
        "families_positive": nf, "families_missed": len(fam_miss),
        "family_miss_rate": len(fam_miss) / nf if nf else None,
        "family_miss_cp_upper": ST.cp_upper(len(fam_miss), nf, ALPHA) if nf else None,
        "success": {"plus": sp, "minus": sm, "n": n,
                    "diff": (sp - sm) / n if n else None,
                    "lower": ST.diff_lower(sp, sm, n, ALPHA) if n else None},
        "harm": {"plus": hp, "minus": hm, "n": n, "diff": (hp - hm) / n if n else None,
                 "upper": ST.diff_upper(hp, hm, n, ALPHA) if n else None},
        "nonscored": {"plus": np_, "minus": nm, "n": n,
                      "upper": ST.diff_upper(np_, nm, n, ALPHA) if n else None},
        "candidate_only_critical": crit_new,
        "mean_cost": sum(costs) / len(costs) if costs else None,
        "p95_seconds": [PC.percentile(sec_a, 95) if sec_a else None,
                        PC.percentile(sec_b, 95) if sec_b else None]}
    return out


def qualifies(ev: Mapping[str, Any], role: str, mode: str = "bounds"
              ) -> tuple[bool, list[str]]:
    """``mode`` "bounds" (PRIMARY, prereg s. 7: the one-sided 95 % bounds clear the bars) or
    "points" (SECONDARY, descriptive: the point estimates clear them; reported because at
    this probe's calibration size the bounds cannot clear a 1-point harm bar)."""
    why = []
    if role == "incumbent":
        return True, []
    if not ev["decisions"] or ev["coverage"] != 1:
        why.append("not every choice executed")
    s, h, ns = ev["success"], ev["harm"], ev["nonscored"]
    if mode == "points":
        sv, hv = s["diff"], h["diff"]
        nv = (ns["plus"] - ns["minus"]) / ns["n"] if ns["n"] else None
    else:
        sv, hv, nv = s["lower"], h["upper"], ns["upper"]
    if sv is None or sv < QUALITY["success_lower"]:
        why.append("success bar")
    if hv is None or hv > QUALITY["harm_upper"]:
        why.append("harm bar")
    if nv is None or nv > QUALITY["nonscored_upper"]:
        why.append("non-scored bar")
    pa, pb = ev["p95_seconds"]
    if pa is None or pb is None or (pb > 0 and pa > QUALITY["p95_ratio"] * pb):
        why.append("p95 latency")
    if ev["candidate_only_critical"]:
        why.append("candidate-only critical event (HOLD)")
    if role == "selector":
        if not ev["families_positive"]:
            why.append("family miss not estimable (no positive-bearing family)")
        elif ev["family_miss_cp_upper"] > MISS_TARGET:
            why.append("family-miss bound")
    return not why, why


def select(data: Data, pts: Sequence[Point], fit: Sequence[Mapping[str, Any]],
           labels: Mapping[str, Any], role: str, mode: str = "bounds") -> dict[str, Any]:
    """The pre-registered choice among ``role`` points (controls or selectors) on
    calibration-fit: minimum mean cost among qualifying points; ties: fewer risky
    actions, then id; none: the incumbent."""
    inc = next(p for p in pts if p.role == "incumbent")
    iev = evaluate(data, inc, fit, labels)
    # the incumbent is always an operating point (zero risk): a qualifying point must be
    # strictly cheaper than it, and wins only then (ties go to the incumbent)
    best = ((iev["mean_cost"], 0, ""), inc) if iev["mean_cost"] is not None else None
    table: list[dict[str, Any]] = []
    for p in (q for q in pts if q.role == role):
        ev = evaluate(data, p, fit, labels)
        ok, why = qualifies(ev, role, mode)
        table.append({"point": p.id, "qualifies": ok, "why_not": why,
                      "mean_cost": ev["mean_cost"], "risky": ev["risky"],
                      "family_miss_cp_upper": ev["family_miss_cp_upper"]})
        if ok and ev["mean_cost"] is not None:
            k = (ev["mean_cost"], ev["risky"], p.id)
            if best is None or k < best[0]:
                best = (k, p)
    chosen = best[1] if best else inc
    return {"chosen": chosen.id, "fallback": chosen is inc, "grid": table}


def _cparts(data: Data, decs: Sequence[Mapping[str, Any]], seed: int, primary: bool
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if primary:  # the plan's preassigned split (probe.split_of: c_part by user hash)
        fit = [d for d in decs if (d.get("split") or {}).get("c_part") == "fit"]
        aud = [d for d in decs if (d.get("split") or {}).get("c_part") == "audit"]
        return fit, aud

    def u(g: str) -> float:
        return int(_sha(f"{seed}|c-part|{g}")[:15], 16) / 16 ** 15
    fit = [d for d in decs if u(data.group(d)) < 0.5]
    aud = [d for d in decs if u(data.group(d)) >= 0.5]
    return fit, aud


def _calibrate_type(data: Data, pts: Sequence[Point], decs: Sequence[Mapping[str, Any]],
                    labels: Mapping[str, Any], mode: str = "bounds") -> dict[str, Any]:
    out: dict[str, Any] = {"decisions_C": len(decs), "quality_mode": mode}
    parts = []
    by = {p.id: p for p in pts}
    for j, seed in enumerate(PARTITION_SEEDS):
        fit, aud = _cparts(data, decs, seed, primary=j == 0)
        rec: dict[str, Any] = {"seed": seed, "primary": j == 0,
                               "fit": {"decisions": len(fit),
                                       "families": len({data.group(d) for d in fit})},
                               "audit": {"decisions": len(aud),
                                         "families": len({data.group(d) for d in aud})}}
        for role in ("control", "selector"):
            s = select(data, pts, fit, labels, role, mode)
            ev = evaluate(data, by[s["chosen"]], aud, labels)
            ok, why = qualifies(ev, by[s["chosen"]].role, mode)
            inc = next(p for p in pts if p.role == "incumbent")
            rec[role] = {"chosen_on_fit": s["chosen"], "fallback": s["fallback"],
                         "audit_pass": ok, "audit_why_not": why,
                         "frozen": s["chosen"] if ok else inc.id,
                         "audit": {k: ev[k] for k in (
                             "families_positive", "families_missed", "family_miss_rate",
                             "family_miss_cp_upper", "miss_rate_raw", "positives",
                             "mean_cost", "coverage", "bad_rate_risky")} | {
                             "success_diff": ev["success"]["diff"],
                             "harm_diff": ev["harm"]["diff"]}}
            if j == 0:
                rec[role]["fit_grid"] = s["grid"]
        parts.append(rec)
    out["primary"] = {r: {k: parts[0][r][k] for k in ("chosen_on_fit", "fallback",
                                                       "audit_pass", "audit_why_not",
                                                       "frozen")}
                      for r in ("control", "selector")}
    out["partitions"] = parts
    held = [p["selector"]["audit"]["family_miss_rate"] for p in parts
            if p["selector"]["audit"]["family_miss_rate"] is not None]
    held.sort()
    out["selector_heldout_family_miss"] = {
        "estimable_partitions": len(held),
        "median": held[len(held) // 2] if held else None, "max": max(held) if held else None,
        "p95": held[min(len(held) - 1, math.ceil(0.95 * len(held)) - 1)] if held else None,
        "share_missing_target": (sum(1 for x in held if x > MISS_TARGET) / len(held))
        if held else None,
        "note": "overlapping sensitivity partitions, not independent experiments; only the "
                "first (the plan's preassigned split) chose the frozen point"}
    return out


def _secondary_mode() -> str:
    return "points" if PRIMARY_QUALITY_MODE == "bounds" else "bounds"


def _secondary(data: Data, pts: Sequence[Point], decs: Sequence[Mapping[str, Any]],
               labels: Mapping[str, Any]) -> dict[str, Any]:
    """SECONDARY (descriptive, never frozen): the same calibration with the other quality
    mode (point-estimate bars while the primary is the pre-registered bounds)."""
    r = _calibrate_type(data, pts, decs, labels, mode=_secondary_mode())
    return {"label": f"SECONDARY ({_secondary_mode()} quality bars): descriptive only, "
                     "not the frozen choice",
            "quality_mode": _secondary_mode(), "primary": r["primary"],
            "selector_heldout_family_miss": r["selector_heldout_family_miss"],
            "frozen": False}


def calibrate(exports: Sequence[Path], package: Path | None) -> dict[str, Any]:
    """Step 1: pool C only (decisions of other pools are dropped at load, before any
    branch is indexed; CL is fitted on pool F's labels, loaded separately)."""
    data = load(exports, package, pools=("C",))
    fdata = load(exports, package, pools=("F",))
    cl, cl_rep = fit_cl(fdata, fdata.decs)
    res: dict[str, Any] = {"schema": SCHEMA, "module_sha256": module_sha256(),
                           "inputs": data.files, "grid": [GRID[0], GRID[-1], len(GRID)],
                           "partition_seeds": [PARTITION_SEEDS[0], PARTITION_SEEDS[-1]],
                           "miss_target": MISS_TARGET, "quality": QUALITY, "cl": cl_rep,
                           "decision_ids": sorted(d["id"] for d in data.decs),
                           "groups": "user (probe plan)" if data.users else "case"}
    t1 = [d for d in data.decs if d["type"] == 1]
    lab1 = {d["id"]: chunk_labels(data, d) for d in t1}
    elig = sum(len((d.get("inputs") or {}).get("eligible") or []) for d in t1)
    res["primary_quality_mode"] = PRIMARY_QUALITY_MODE
    res["type1"] = _calibrate_type(data, points_type1(cl), t1, lab1, PRIMARY_QUALITY_MODE)
    res["type1"]["secondary_point_bars"] = _secondary(data, points_type1(cl), t1, lab1)
    res["type1"]["labels"] = {"eligible_chunks": elig,
                              "labelled": sum(len(v) for v in lab1.values()),
                              "keep_positive": sum(sum(v.values()) for v in lab1.values())}
    for stratum, inj in (("natural", False), ("injected", True)):
        t3 = [d for d in data.decs if d["type"] == 3 and bool(d.get("injected")) == inj]
        lab3 = {d["id"]: beneficial3(data, d) for d in t3}
        res[f"type3_{stratum}"] = _calibrate_type(data, points_type3(), t3, lab3,
                                                  PRIMARY_QUALITY_MODE)
        res[f"type3_{stratum}"]["secondary_point_bars"] = _secondary(data, points_type3(),
                                                                     t3, lab3)
    res["type2"] = {"status": "not estimable: no future-value labels in this probe "
                              "(learning-history clones deferred; prereg s. 5.2)"}
    body = json.dumps(res, sort_keys=True, default=str)
    return {**res, "sha256": _sha(body)}


# ============================================================================ report


def _boot(vals: Sequence[float], cl: Sequence[str]) -> list[float | None]:
    from slh.system.hosts.hermes import probe as PR

    m, lo, hi = PR._boot_mean(list(vals), list(cl))
    return [m, lo, hi]


def classify(ci: Sequence[float | None], bar: float) -> str:
    _, lo, hi = ci
    if lo is None or hi is None:
        return "not estimable"
    if lo >= bar:
        return "supported"
    if hi < bar:
        return "ruled out"
    return "inconclusive"


def _pick(data: Data, d: Mapping[str, Any], names: Sequence[str], rep: int, ref: str
          ) -> str | None:
    """Cheapest not-worse option on ``rep`` (prompted pays its chooser); ties by order; the
    reference is always eligible."""
    b = data.row(d, ref, rep)
    if b is None:
        return None
    best = None
    for order, n in enumerate(names):
        r = data.row(d, n, rep)
        if r is None or not not_worse(r, b):
            continue
        c = branch_cost(r)
        c = None if c is None else c + (chooser_cost(d) if n == "prompted" else 0.0)
        k = (c if c is not None else float("inf"), n != ref, order)
        if best is None or k < best[0]:
            best = (k, n)
    return best[1] if best else None


def _sdiff(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> int | None:
    """Success difference, complete case (an unscored success is skipped), exactly as the
    kit's ``probe.analyze`` pairs it, so the headroom matches the registered code."""
    if a is None or b is None or a.get("success") is None or b.get("success") is None:
        return None
    return int(bool(a["success"])) - int(bool(b["success"]))


def oracle_rows(data: Data, decs: Sequence[Mapping[str, Any]], real: Sequence[str]
                ) -> list[dict[str, Any]]:
    """Per decision, independently of ``probe.analyze``: the real, placebo and size-matched
    real oracles' choice (replicate 0) and their assessed success and cost share
    (replicate 1 vs the reference's replicate 1)."""
    from slh.system.hosts.hermes import probe as PR

    out = []
    for d in decs:
        ref = ref_of(d)
        r1 = data.row(d, ref, 1)
        whole = data.base_cost.get(d["base_tag"]) or 0.0
        have = [n for n in real if data.row(d, n, 0) is not None]
        have.sort(key=lambda n: PR._u("matched", d["id"], n))
        sets = {"real": [ref, *real], "placebo": [ref, *PR.PLACEBOS],
                "matched": [ref, *have[:len(PR.PLACEBOS)]]}
        row: dict[str, Any] = {"id": d["id"], "case": d["case"]}
        for k, names in sets.items():
            ch = _pick(data, d, names, 0, ref)
            a = data.row(d, ch, 1) if ch else None
            e = data.row(d, ch, 0) if ch else None
            s = _sdiff(a, r1)
            ca, cb = branch_cost(a), branch_cost(r1)
            if ca is not None and ch == "prompted":
                ca += chooser_cost(d)
            sv = (cb - ca) / whole if (ca is not None and cb is not None and whole) else None
            se = _sdiff(e, r1)
            row[k] = {"choice": ch, "success": s, "saving": sv, "success_envelope": se,
                      "harm": None if a is None or r1 is None else
                      int(bool(a["harm"])) - int(bool(r1["harm"])),
                      "critical_new": None if a is None or r1 is None else
                      int(a["critical"] is not False and r1["critical"] is False)}
        out.append(row)
    return out


def headroom(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for f, key in (("success", "success"), ("cost_share", "saving")):
        vals, cl = [], []
        for r in rows:
            a, b = r["matched"][key], r["placebo"][key]
            if a is not None and b is not None:
                vals.append(float(a - b))
                cl.append(r["case"])
        res[f] = {"n": len(vals), "clusters": len(set(cl)), "mean_ci95": _boot(vals, cl)}
    return res


def _means(rows: Sequence[Mapping[str, Any]], k: str) -> dict[str, Any]:
    s = [r[k]["success"] for r in rows if r[k]["success"] is not None]
    v = [r[k]["saving"] for r in rows if r[k]["saving"] is not None]
    return {"success": round(sum(s) / len(s), 4) if s else None,
            "saving": round(sum(v) / len(v), 4) if v else None,
            "envelope_success": _mean([r[k]["success_envelope"] for r in rows
                                       if r[k]["success_envelope"] is not None])}


def _mean(xs: Sequence[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def _noise(data: Data, decs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = dis = crit = 0
    for d in decs:
        a, b = data.row(d, ref_of(d), 1), data.row(d, ref_of(d), 0)
        if a is None or b is None:
            continue
        n += 1
        dis += bool(a["success"]) != bool(b["success"])
        crit += (a["critical"] is False) != (b["critical"] is False)
    return {"n": n, "success_discordant": dis, "critical_discordant": crit,
            "rate": dis / n if n else None}


def _fixed_effect(data: Data, decs: Sequence[Mapping[str, Any]], name: str
                  ) -> dict[str, dict[str, float]]:
    """Per decision: ``name`` replicate 1 minus the reference's replicate 1 (success and
    cost share); ``native`` gives 0."""
    out = {}
    for d in decs:
        ref = ref_of(d)
        r1 = data.row(d, ref, 1)
        a = data.row(d, name, 1) if name in (d.get("options") or {}) else r1
        if a is None or r1 is None:
            continue
        whole = data.base_cost.get(d["base_tag"]) or 0.0
        ca, cb = branch_cost(a), branch_cost(r1)
        out[d["id"]] = {"success": int(bool(a["success"])) - int(bool(r1["success"])),
                        "saving": (cb - ca) / whole if (ca is not None and cb is not None
                                                        and whole) else None}
    return out


def _type_report(data: Data, decs: Sequence[Mapping[str, Any]], real: Sequence[str],
                 dtype: str, best_fixed: str | None) -> dict[str, Any]:
    from slh.system import stats as ST

    rows = oracle_rows(data, decs, real)
    hr = headroom(rows)
    noise = _noise(data, decs)
    out: dict[str, Any] = {"decisions": len(decs), "noise_ref_rep1_vs_rep0": noise,
                           "oracle_means": {k: _means(rows, k) for k in
                                            ("real", "placebo", "matched")},
                           "headroom_matched_minus_placebo": hr}
    m = [r["matched"] for r in rows if r["matched"]["success"] is not None]
    sp = sum(1 for r in m if r["success"] > 0)
    sm = sum(1 for r in m if r["success"] < 0)
    hp = sum(1 for r in m if (r["harm"] or 0) > 0)
    hm = sum(1 for r in m if (r["harm"] or 0) < 0)
    out["matched_oracle_quality"] = {
        "success_diff": (sp - sm) / len(m) if m else None,
        "success_loss_bound": ST.diff_lower(sp, sm, len(m), ALPHA) if m else None,
        "harm_diff": (hp - hm) / len(m) if m else None,
        "critical_new": sum(r["critical_new"] or 0 for r in m),
        "noise_critical_discordant": noise["critical_discordant"]}
    q = out["matched_oracle_quality"]
    q_ok = (q["success_diff"] is not None and q["success_diff"] >= -0.02
            and q["success_loss_bound"] >= -0.05 and q["harm_diff"] <= 0.01
            and q["critical_new"] <= noise["critical_discordant"])
    if dtype == "type1":
        ci = hr["cost_share"]["mean_ci95"]
        owner = (ci[0] is not None and ci[1] is not None and ci[0] >= HP_BARS["type1_cost"]
                 and ci[1] > HP_BARS["type1_cost_ci_lower_owner"] and q_ok)
        out["HP1"] = {"owner_bar": "supported" if owner else "not met",
                      "prereg_class": classify(ci, HP_BARS["type1_cost"])
                      if q_ok else "not supported (quality bars)",
                      "quality_ok": q_ok}
        out["hp_class"] = out["HP1"]["prereg_class"]
    else:
        fx = _fixed_effect(data, decs, (best_fixed or "always_native").removeprefix("always_"))
        vs, vc, cl, clc = [], [], [], []
        for r in rows:
            f = fx.get(r["id"])
            a, b = r["matched"]["success"], r["placebo"]["success"]
            if f is not None and a is not None and b is not None:
                vs.append(float(a - b - f["success"]))
                cl.append(r["case"])
            a, b = r["matched"]["saving"], r["placebo"]["saving"]
            if f is not None and a is not None and b is not None and f["saving"] is not None:
                vc.append(float(a - b - f["saving"]))
                clc.append(r["case"])
        cs, cc = _boot(vs, cl), _boot(vc, clc)
        succ_cls = classify(cs, HP_BARS["type3_success"])
        cost_cls = classify(cc, HP_BARS["type3_cost"]) if (
            cs[1] is not None and cs[1] >= -0.02) else "not supported (success not equal)"
        out["HP3"] = {"best_fixed_rule": best_fixed, "success_margin_ci95": cs,
                      "cost_margin_ci95": cc, "success_class": succ_cls,
                      "cost_class": cost_cls}
        out["hp_class"] = "supported" if "supported" in (succ_cls, cost_cls) else (
            "ruled out" if succ_cls == "ruled out" and cost_cls in (
                "ruled out", "not supported (success not equal)") else "inconclusive")
    s_ci, c_ci = hr["success"]["mean_ci95"], hr["cost_share"]["mean_ci95"]
    out["distinguishable_from_noise"] = any(
        ci[1] is not None and (ci[1] > 0 or ci[2] < 0) for ci in (s_ci, c_ci))
    return out


def review_stats(data: Data) -> dict[str, Any]:
    """Type 2 (HP2) from the review rows: F, kinds, survival, audit flags, and the one
    four pre-declared skip rules: 'no task write' (skip when ``ep_task_write`` is False),
    'episode success' (skip when ``ep_success`` is True), 'short episode' (skip when
    ``ep_turns`` <= the median ``ep_turns`` of all C review rows: a pre-fork input, never an
    outcome), and the prompted 9B says skip (its request charged). A fact that is missing
    (None) never skips. The episode facts exist from kit v2.1 on; a v2 export lacks them
    and those three rules are then reported as not testable."""
    from slh.system.hosts.hermes import probe as PR

    def cost(t: Sequence[Any]) -> float:
        return PR.COST_RATES["prefill"] * float(t[0] or 0) + \
            PR.COST_RATES["decode"] * float(t[1] or 0)
    c = [r for r in data.revs if r["arm"] == "C"]
    tot = sum(cost(r["fork_tokens"]) for r in c)
    surv = [r for r in c if r.get("survives")]
    out: dict[str, Any] = {
        "forks_C": len(c), "forks_B": sum(1 for r in data.revs if r["arm"] == "B"),
        "surviving": len(surv), "fork_cost_units": tot,
        "F_share_not_surviving": (sum(cost(r["fork_tokens"]) for r in c
                                      if not r.get("survives")) / tot) if tot else None,
        "kinds": {k: sum(1 for r in c if r["kind"] == k)
                  for k in ("empty", "duplicate", "proposals")},
        "survival_by_kind": {k: sum(1 for r in surv if r["kind"] == k)
                             for k in ("empty", "duplicate", "proposals")},
        "fates": {f: sum(1 for r in c if r.get("c_fate") == f)
                  for f in sorted({str(r.get("c_fate")) for r in c})},
        "audit_flagged": sum(1 for r in c if r.get("audit"))}

    def rule(skip: Sequence[Mapping[str, Any]], pool: Sequence[Mapping[str, Any]],
             extra: float = 0.0) -> dict[str, Any]:
        jt = sum(cost(r["fork_tokens"]) for r in pool)
        js = [r for r in pool if r.get("survives")]
        sk = sum(cost(r["fork_tokens"]) for r in skip)
        return {"forks": len(pool), "skipped": len(skip),
                "gross_saving": sk / jt if jt else None,
                "net_saving": (sk - extra) / jt if jt else None,
                "surviving": len(js),
                "surviving_skipped": sum(1 for r in skip if r.get("survives")),
                "loss": (sum(1 for r in skip if r.get("survives")) / len(js)) if js else None,
                "skipped_audit_flagged": sum(1 for r in skip if r.get("audit"))}
    has_facts = any("ep_status" in r for r in c)
    out["rules_not_testable"] = [] if has_facts else [
        "no task write", "episode success", "short episode"]
    if has_facts:
        turns = sorted(int(r["ep_turns"]) for r in c if r.get("ep_turns") is not None)
        med = turns[(len(turns) - 1) // 2] if turns else None
        out["short_episode_threshold_turns"] = med
        out["rule_no_task_write"] = rule([r for r in c if r.get("ep_task_write") is False], c)
        out["rule_episode_success"] = rule([r for r in c if r.get("ep_success") is True], c)
        out["rule_short_episode"] = rule([r for r in c if med is not None and r.get(
            "ep_turns") is not None and int(r["ep_turns"]) <= med], c)
    rp = {(str(r.get("batch")), r.get("unit")): r for r in data.review_prompted}
    joined = [(r, rp.get((str(r["batch"]), r["unit"]))) for r in c]
    joined = [(r, p) for r, p in joined if p is not None]
    if joined:
        out["rule_prompted_skip"] = rule(
            [r for r, p in joined if p.get("choice") is False], [r for r, _ in joined],
            extra=sum(cost(p.get("tokens") or [0, 0]) for _, p in joined))
    under = len(surv) < HP_BARS["type2_min_surviving"]
    f_ok = (out["F_share_not_surviving"] or 0) >= HP_BARS["type2_F"]
    names = [k for k in ("rule_no_task_write", "rule_episode_success", "rule_short_episode",
                         "rule_prompted_skip") if k in out]

    def met(r: Mapping[str, Any]) -> bool:
        return (r.get("net_saving") or 0) >= HP_BARS["type2_saving"] and \
            r.get("loss") is not None and r["loss"] <= HP_BARS["type2_loss"]
    rules_met = [k for k in names if met(out[k])]
    rule_ok = bool(rules_met)
    out["HP2"] = {"F_met": f_ok, "rule_met": rule_ok, "rules_met": rules_met,
                  "rules_tested": names, "underpowered": under,
                  "class": "underpowered" if under else (
                      "supported" if f_ok and rule_ok else "not met")}
    return out


def cost_ledger(data: Data) -> dict[str, Any]:
    """The probe's whole expense in cost units (prefill 1, decode 3, cache hit 0.1): base
    episodes, every branch, every prompted request; uncostable and unreported-cache
    branches counted."""
    br_cost, unc, nocache = 0.0, 0, 0
    per_arm: dict[str, list[float]] = {}
    for r in data.brs:
        c = branch_cost(r)
        if c is None:
            unc += 1
            continue
        br_cost += c
        if any(x[2] is None for x in r.get("live_calls") or []):
            nocache += 1
        for n in r.get("options") or []:
            per_arm.setdefault(n, []).append(c)
    chooser = sum(chooser_cost(d) for d in data.decs)
    rv = sum(float((p.get("tokens") or [0, 0])[0] or 0) + 3 * float(
        (p.get("tokens") or [0, 0])[1] or 0) for p in data.review_prompted)
    base = sum(data.base_cost.values())
    return {"base_episodes": base, "branches": br_cost, "prompted_decisions": chooser,
            "prompted_reviews": rv, "total": base + br_cost + chooser + rv,
            "branches_uncostable": unc, "branches_cache_unreported": nocache,
            "branches_total": len(data.brs),
            "mean_branch_cost_by_option": {k: round(sum(v) / len(v), 1)
                                           for k, v in sorted(per_arm.items())}}


def assess_p(data: Data, calib: Mapping[str, Any], cl: Any) -> dict[str, Any]:
    """The committed primary points on the untouched pool P (prereg s. 7)."""
    out = {}
    p_decs = [d for d in data.decs if (d.get("split") or {}).get("pool") == "P"]
    for key, dtype, pts, sel in (
            [("type1", 1, points_type1(cl), lambda d: d["type"] == 1)] + [
                (f"type3_{s}", 3, points_type3(),
                 lambda d, inj=inj: d["type"] == 3 and bool(d.get("injected")) == inj)
                for s, inj in (("natural", False), ("injected", True))]):
        decs = [d for d in p_decs if sel(d)]
        by = {p.id: p for p in pts}
        if dtype == 1:
            labels: dict[str, Any] = {d["id"]: chunk_labels(data, d) for d in decs}
        else:
            labels = {d["id"]: beneficial3(data, d) for d in decs}
        res = {}
        for role in ("control", "selector"):
            pid = calib[key]["primary"][role]["frozen"]
            if pid not in by:
                res[role] = {"point": pid, "status": "not available (CL not fitted)"}
                continue
            res[role] = evaluate(data, by[pid], decs, labels)
        res["incumbent"] = evaluate(data, next(p for p in pts if p.role == "incumbent"),
                                    decs, labels)
        out[key] = res
    return out


def report(exports: Sequence[Path], package: Path | None, calib_path: Path,
           calib_sha: str) -> dict[str, Any]:
    """Step 2 (refuses unless the calibration file is the committed one)."""
    from slh.system.hosts.hermes import probe as PR

    raw = Path(calib_path).read_bytes()
    if _sha(raw) != calib_sha:
        raise Refused("the calibration file's sha256 differs from --calibration-sha256: "
                      "calibrate, commit, then report with the committed file's hash")
    calib = json.loads(raw)
    body = {k: v for k, v in calib.items() if k != "sha256"}
    if _sha(json.dumps(body, sort_keys=True, default=str)) != calib.get("sha256"):
        raise Refused("calibration.json does not hash to its own sha256")
    if calib.get("module_sha256") != module_sha256():
        raise Refused("calibration.json was made by another version of this module")
    data = load(exports, package)
    design = PR.analyze(exports)
    fid = design["fidelity"]
    out: dict[str, Any] = {"schema": SCHEMA, "module_sha256": module_sha256(),
                           "calibration_sha256": calib["sha256"], "inputs": data.files,
                           "design_report": design}
    t1 = [d for d in data.decs if d["type"] == 1]
    real1 = ["native_default_ids", "group1", "group2", "native_low_ids", "heuristic",
             *[f"last_k{k}" for k in PR.LAST_KS], "reversible", "prompted"]
    out["type1"] = _type_report(data, t1, real1, "type1", None)
    for s, inj in (("natural", False), ("injected", True)):
        t3 = [d for d in data.decs if d["type"] == 3 and bool(d.get("injected")) == inj]
        bf = calib[f"type3_{s}"]["primary"]["control"]["frozen"]
        out[f"type3_{s}"] = _type_report(data, t3, list(REAL3), "type3",
                                         None if bf == "native" else bf)
    out["type2"] = review_stats(data)
    out["cost_ledger"] = cost_ledger(data)
    fdata = load(exports, package, pools=("F",))
    cl, _ = fit_cl(fdata, fdata.decs)
    out["calibrated_on_P"] = assess_p(data, calib, cl)
    n3 = len([d for d in data.decs if d["type"] == 3])
    fid_ok = fid["rate"] is not None and fid["rate"] <= PR.FIDELITY_MAX_DIVERGED

    def ladder(n: int, need: int, t: Mapping[str, Any] | None, cls: str) -> dict[str, Any]:
        rungs = [("R0 exposure", n >= need, f"{n} (needs {need})"),
                 ("R1 replay fidelity", fid_ok, f"diverged rate {fid['rate']}"),
                 ("R2 serving noise", bool(t and t["distinguishable_from_noise"]),
                  "headroom interval excludes 0 (success or cost)"),
                 ("R3 oracle headroom", cls == "supported", cls)]
        fail = next((r for r in rungs if not r[1]), None)
        return {"rungs": [{"rung": r[0], "passed": r[1], "detail": r[2]} for r in rungs],
                "stops_at": fail[0] if fail else "R4+ (not tested here)"}
    out["ladder"] = {
        "type1": ladder(len(t1), EXPOSURE["type1"], out["type1"], out["type1"]["hp_class"]),
        "type3": ladder(n3, EXPOSURE["type3"], out["type3_injected"],
                        out["type3_natural"]["hp_class"]
                        if out["type3_natural"]["hp_class"] == "supported"
                        else out["type3_injected"]["hp_class"]),
        "type2": {"rungs": [{"rung": "R0 exposure",
                             "passed": out["type2"]["forks_C"] >= EXPOSURE["type2"],
                             "detail": f"{out['type2']['forks_C']} C forks"},
                            {"rung": "R3 HP2", "passed": out["type2"]["HP2"]["class"] ==
                             "supported", "detail": out["type2"]["HP2"]["class"]}]}}
    t2 = out["ladder"]["type2"]["rungs"]
    out["ladder"]["type2"]["stops_at"] = next((r["rung"] for r in t2 if not r["passed"]),
                                              "R4+ (not tested here)")
    return out


# ============================================================================ self-audit


def self_audit(exports: Sequence[Path], package: Path | None, calib_path: Path,
               report_path: Path) -> dict[str, Any]:
    """The pre-registered self-audit (prereg s. 7, items 1-9) after the first report, with
    a separate recomputation; each item pass / fail / not testable."""
    from slh.system.hosts.hermes import probe as PR

    calib = json.loads(Path(calib_path).read_bytes())
    rep = json.loads(Path(report_path).read_bytes())
    data = load(exports, package)
    inputs = {"exports": data.files, "calibration": _sha(Path(calib_path).read_bytes()),
              "report": _sha(Path(report_path).read_bytes()), "module": module_sha256()}
    items: list[dict[str, Any]] = []

    def item(i: int, title: str, status: str, evidence: Any) -> None:
        items.append({"item": i, "title": title, "status": status, "evidence": evidence})

    led = cost_ledger(data)
    valid_unc = sum(1 for r in data.brs if r.get("valid") and branch_cost(r) is None)
    pr_missing = sum(1 for d in data.decs if d.get("prompted") and
                     ((d["prompted"].get("tokens") or [None])[0] is None))
    item(1, "missing costs", "fail" if valid_unc or pr_missing else "pass",
         {"valid_branches_uncostable": valid_unc, "prompted_without_tokens": pr_missing,
          "cache_unreported_branches": led["branches_cache_unreported"],
          "ledger_total": led["total"], "not testable": "training attempts and "
          "requalification (none in this probe); GPU-second shadow prices (stated deviation)"})
    design = PR.analyze(exports)
    mism = []
    for key, ref, arms in (("type1", "keep", ["group1", "heuristic", "last_k1", "prompted"]),
                           ):
        for a in arms:
            got = (design.get(key) or {}).get(a, {}).get("success", {})
            xs, ys = [], []
            for d in (x for x in data.decs if x["type"] == 1):
                ra, rb = data.row(d, a, 0), data.row(d, ref, 0)
                if ra is not None and rb is not None:
                    xs.append(ra["success"])
                    ys.append(rb["success"])
            up = sum(1 for x, y in zip(xs, ys, strict=True) if x and not y)
            dn = sum(1 for x, y in zip(xs, ys, strict=True) if y and not x)
            if got and (got.get("n_plus"), got.get("n_minus")) != (up, dn):
                mism.append(a)
    item(2, "wrong endpoint", "fail" if mism else "pass",
         {"success_recomputed_from_branch_labels_mismatch": mism,
          "endpoints": "headroom = episode success and all-in cost share; type 2 fates are "
                       "intermediate events, never success"})
    cal_ids = set(calib.get("decision_ids") or [])
    pool = {d["id"]: (d.get("split") or {}).get("pool") for d in data.decs}
    leak = sorted(i for i in cal_ids if pool.get(i) != "C")
    groups = {p: {data.group(d) for d in data.decs if pool[d["id"]] == p} for p in "FCP"}
    overlap = {f"{a}{b}": len(groups[a] & groups[b]) for a, b in ("FC", "FP", "CP")}
    evalx = None
    if package is not None:
        from slh.system.hosts.hermes import mh1 as M

        ev = set(M.Package.load(Path(package), check_code=False).eval_units)
        evalx = len(ev & {d["case"] for d in data.decs})
    ok3 = not leak and not any(overlap.values()) and not evalx
    item(3, "threshold leakage", "pass" if ok3 else "fail",
         {"calibration_ids_outside_C": leak[:10], "group_overlap": overlap,
          "probe_cases_in_mh1_eval": evalx,
          "late_share": {p: _mean([float(bool((d.get("split") or {}).get("late")))
                                   for d in data.decs if pool[d["id"]] == p])
                         for p in "FCP"}})
    asked = [d for d in data.decs if d["type"] == 3 and d.get("prompted")
             and d["prompted"].get("reversed_choice") is not None]
    flips = sum(1 for d in asked if d["prompted"]["reversed_choice"] !=
                (d["prompted"].get("choice") or "abstain"))
    item(4, "channel and wording confounds", "pass" if asked else "not testable",
         {"wording_audit_decisions": len(asked), "flips": flips,
          "abstentions": sum(1 for d in data.decs if (d.get("prompted") or {}).get("abstain")),
          "placebos": "native tool-output re-serialisation (ids intact), not user prose"})
    infl = {k: {"envelope_success": rep[k]["oracle_means"]["real"]["envelope_success"],
                "assessed_success": rep[k]["oracle_means"]["real"]["success"]}
            for k in ("type1", "type3_natural", "type3_injected") if k in rep}
    item(5, "selection inflation", "pass", {
        "oracle_envelope_vs_assessed": infl,
        "rule": "headroom is matched-real minus placebo oracle on the assessment repeat; the "
                "comparator was chosen on calibration-fit only; no arm was discarded"})
    noise = {k: rep[k]["noise_ref_rep1_vs_rep0"] for k in
             ("type1", "type3_natural", "type3_injected") if k in rep}
    item(6, "execution and caching imbalance", "pass" if noise else "not testable", {
        "identity_replay_noise": noise,
        "cache_unreported_branches": led["branches_cache_unreported"],
        "not testable": "server concurrency and cache namespaces at the partner"})
    intended = {u["case"] for u in data.plan_units}
    based = {r["case"] for r in data.base}
    no_ref = sum(1 for d in data.decs if data.row(d, ref_of(d), 0) is None
                 or data.row(d, ref_of(d), 1) is None)
    mine = rep.get("type1", {}).get("oracle_means", {})
    theirs = design.get("type1", {})
    dif = []
    for k, o in (("real", "oracle_real"), ("placebo", "oracle_placebo"),
                 ("matched", "oracle_real_matched")):
        a = (mine.get(k) or {}).get("success")
        b = (((theirs.get(o) or {}).get("assessed") or {}).get("success") or {}).get("diff")
        if a is not None and b is not None and abs(a - b) > 1e-4:
            dif.append(k)
    ok7 = not dif
    item(7, "denominators and missingness", "pass" if ok7 else "fail", {
        "intended_parents": len(intended), "base_rows": len(data.base),
        "intended_without_base_row": len(intended - based) if intended else None,
        "decisions_missing_a_reference_repeat": no_ref,
        "invalid_branches": sum(1 for r in data.brs if not r.get("valid")),
        "clusters": len({d["case"] for d in data.decs}),
        "oracle_recompute_mismatch": dif})
    c_rev = [r for r in data.revs if r["arm"] == "C"]
    item(8, "learning contamination", "not testable" if not c_rev else "pass", {
        "review_rows": len(data.revs),
        "audit_flag_share": (sum(1 for r in c_rev if r.get("audit")) / len(c_rev))
        if c_rev else None,
        "not testable": "store cloning, future horizons, shadow-review writes (deferred)"})
    trims, over, err = [], [], []
    for e in exports:
        p = Path(e) / "probe"
        if (p / "export-trims.json.gz").exists():
            import gzip

            trims.append(json.loads(gzip.decompress((p / "export-trims.json.gz")
                                                    .read_bytes()))["steps"][-1])
        over += [str(e)] if (p / "over_budget.json.gz").exists() else []
        err += [str(e)] if (p / "error.json.gz").exists() else []
    item(9, "scope and reporting", "fail" if over else "pass", {
        "final_trim_step": trims, "over_budget": over, "probe_error_files": err,
        "rule": "critical dialogues are never trimmed (export); no row is dropped"})
    body = {"schema": SCHEMA, "inputs": inputs, "items": items}
    return {**body, "sha256": _sha(json.dumps(body, sort_keys=True, default=str)),
            "suspended_claims": [i["title"] for i in items if i["status"] == "fail"]}


# ============================================================================ freeze record


def freeze_record(calib_path: Path) -> dict[str, Any]:
    """The small committed record of step 1 (no case ids, no outcomes): the calibration
    file's hash, this module's hash and the frozen primary points per type. Commit and push
    it before ``report`` runs."""
    raw = Path(calib_path).read_bytes()
    calib = json.loads(raw)
    return {"event": "s1probe_calibration_frozen", "calibration_sha256": _sha(raw),
            "calibration_body_sha256": calib.get("sha256"),
            "module_sha256": calib.get("module_sha256"), "inputs": calib.get("inputs"),
            "primary": {k: calib[k]["primary"] for k in ("type1", "type3_natural",
                                                        "type3_injected") if k in calib},
            "cl": (calib.get("cl") or {}).get("status")}


# ============================================================================ CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -I slh/xp/ourside/s1_probe_ourside.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    fr = sub.add_parser("freeze-record")
    fr.add_argument("--calibration", type=Path, required=True)
    fr.add_argument("--out", type=Path, required=True)
    for name in ("calibrate", "report", "self-audit"):
        p = sub.add_parser(name)
        p.add_argument("--export", type=Path, nargs="+", required=True)
        p.add_argument("--package", type=Path)
        p.add_argument("--out", type=Path, required=True)
        if name != "calibrate":
            p.add_argument("--calibration", type=Path, required=True)
        if name == "report":
            p.add_argument("--calibration-sha256", required=True)
        if name == "self-audit":
            p.add_argument("--report", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        print(json.dumps({"ok": False, "error": f"{a.out} exists: outputs are written once"}))
        return 4
    try:
        if a.cmd == "freeze-record":
            out = freeze_record(a.calibration)
        elif a.cmd == "calibrate":
            out = calibrate(a.export, a.package)
        elif a.cmd == "report":
            out = report(a.export, a.package, a.calibration, a.calibration_sha256)
        else:
            out = self_audit(a.export, a.package, a.calibration, a.report)
    except Refused as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 4
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"ok": True, "out": str(a.out),
                      "sha256": _sha(a.out.read_bytes())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
