"""Phase-1 pre-registration row for plan xp-p1-2026-10-10 (docs/xp/registration-phase1.md).

    python -m slh.xp.partner.phase1_registration [--public DIR] --append   # once
    python -m slh.xp.partner.phase1_registration [--public DIR]            # check
    python -m slh.xp.partner.phase1_registration --append-amendment-1       # once

Amendment 1 (docs/xp/registration-phase1-amendment-1.md, row ``xp_phase1_amendment``):
the amended plan is derived from the registered public ``plan.json`` plus the
person-written arm built from the committed texts (no secret needed; its self-hash
must equal the owner-approved ``a212195b...``); analysis code and texts are hashed
from the recorded commit (``git show``), so later commits do not change the check.

Every hash is recomputed from the built public tree (no keys, no secret): file
SHA-256s, the plan's internal hash, and the test membership (sorted test case ids
from the question files, hashed as ``slh.xp.partner.plan.membership`` does). The
row is appended only through ``slh.learner.registry.Registry`` (hash chain), at
most once; the check verifies the chain and that the registered row equals a fresh
recomputation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from slh.learner.registry import Registry
from slh.xp.partner import common as pc

REPO = Path(__file__).resolve().parents[3]
PLAN_ID = "xp-p1-2026-10-10"
EVENT = "xp_phase1_registration"
REGISTRATION_ID = f"{PLAN_ID}-phase1"
PUBLIC = Path.home() / ".slh-xp/public" / PLAN_ID
DOC = "docs/xp/registration-phase1.md"
TASKS = ("address", "variant", "sequence")
PHASE0_TAG = "xp-p1-phase0-v1"
PHASE0_TAG_COMMIT = "640c2766d9fc166845ffe65b6b0946bb5a06a705"
OWNER_APPROVAL = "owner approved plan v1.2 decisions on 2026-10-10"
PHASE1_CODE = ("Phase-1 code will differ from the phase-0 build (later fixes). It will be "
               "recorded by a registry amendment, which must keep plan_sha256, "
               "plan_file_sha256, protocol_sha256 and case_files identical to this row.")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def membership(public: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for task in TASKS:
        rows = (public / f"package/cases/{task}-test.questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        ids = sorted(json.loads(r)["case_id"] for r in rows if r.strip())
        if len(set(ids)) != len(ids):
            raise ValueError(f"{task}: duplicate test case ids")
        out[task] = {"count": len(ids), "sha256": pc.sha_text(pc.canonical(ids))}
    return out


def build_event(public: Path = PUBLIC, root: Path = REPO) -> dict[str, Any]:
    pkg = public / "package"
    plan = json.loads((pkg / "plan.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in plan.items() if k != "sha256"}
    if pc.sha_text(pc.canonical(body)) != plan.get("sha256"):
        raise ValueError("plan.json does not hash to its own sha256")
    reg = json.loads((pkg / "registration.json").read_text(encoding="utf-8"))
    cases = {f"cases/{p.name}": _sha(p) for p in sorted((pkg / "cases").glob("*.jsonl"))}
    event = {
        "event": EVENT, "registration_id": REGISTRATION_ID, "plan_id": plan["plan_id"],
        "doc_path": DOC, "doc_sha256": _sha(root / DOC),
        "plan_sha256": plan["sha256"], "plan_file_sha256": _sha(pkg / "plan.json"),
        "protocol_sha256": _sha(pkg / "protocol.md"),
        "registration_file_sha256": _sha(pkg / "registration.json"),
        "manifest_sha256": _sha(public / "MANIFEST.json"),
        "case_files": cases, "test_membership": membership(public),
        "phase0_tag": PHASE0_TAG, "phase0_tag_commit": PHASE0_TAG_COMMIT,
        "phase0_snapshot_commit": reg["git_commit"],
        "source_commit": reg["source"]["commit"], "code_sha256": reg["code_sha256"],
        "phase1_code_note": PHASE1_CODE, "owner_approval": OWNER_APPROVAL,
    }
    # The package must agree with itself before anything is registered.
    if plan["plan_id"] != PLAN_ID or reg["plan_id"] != PLAN_ID:
        raise ValueError("public tree belongs to another plan")
    if reg["plan_sha256"] != plan["sha256"] or \
            reg["protocol_sha256"] != event["protocol_sha256"] or \
            plan["protocol"]["sha256"] != event["protocol_sha256"]:
        raise ValueError("plan/protocol hashes differ from registration.json")
    if {k: v for k, v in reg["case_files"].items()} != cases:
        raise ValueError("case files differ from registration.json")
    if reg["test_membership"] != event["test_membership"]:
        raise ValueError("test membership differs from registration.json")
    return event


def registered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("event") == EVENT and r.get("plan_id") == PLAN_ID]


def check(registry_dir: Path = REPO / "learner_runs", public: Path = PUBLIC,
          root: Path = REPO) -> dict[str, Any]:
    rows = Registry(registry_dir).rows()  # verifies the hash chain and both heads
    found = registered(rows)
    if len(found) != 1:
        raise ValueError(f"expected one {EVENT} row, found {len(found)}")
    row = found[0]
    fresh = build_event(public, root)
    stored = {k: v for k, v in row.items() if k != "prev_sha256"}
    if stored != fresh:
        diff = sorted(k for k in set(stored) | set(fresh) if stored.get(k) != fresh.get(k))
        raise ValueError(f"registered row differs from the public tree: {diff}")
    index = rows.index(row)
    line = (registry_dir / "REGISTRY.jsonl").read_bytes().splitlines(keepends=True)[index]
    return {"ok": True, "rows": len(rows), "line": index + 1,
            "row_sha256": hashlib.sha256(line).hexdigest(),
            "prev_sha256": row["prev_sha256"]}


def append(registry_dir: Path = REPO / "learner_runs", public: Path = PUBLIC,
           root: Path = REPO) -> dict[str, Any]:
    registry = Registry(registry_dir)
    if registered(registry.rows()):
        raise FileExistsError(f"{EVENT} already registered; append refused")
    registry.append(build_event(public, root))
    return check(registry_dir, public, root)


# --------------------------------------------------------------------------- amendment 1

AMEND_EVENT = "xp_phase1_amendment"
AMEND_ID = f"{PLAN_ID}-phase1-amendment-1"
AMEND_DOC = "docs/xp/registration-phase1-amendment-1.md"
AMENDED_PLAN_SHA256 = "a212195bd56489f49d37cb540992a770904196adec97a01e28795761ae410c9f"
PERSON_DIR = "docs/xp/person-procedures"
AMEND_APPROVAL = "owner approved amendment 1 on 2026-10-11 (approve all)"
AMEND_DECISIONS = {
    "error_rate_guard": "if a confirmatory pair's case-level error rates differ by more than "
                        "ERROR_RATE_MARGIN points, the report is descriptive only and the "
                        "gate is not evaluable",
    "wrong_write_ni": "gates on both writers when the secondary is tested; the narrow "
                      "reading (primary writer only) is reported alongside",
    "interval": "crossed run x group parametric bootstrap-t, 10,000 draws; percentile and "
                "Satterthwaite intervals are sensitivity only",
    "gates": "the v1.2 gate supersedes the older phase table in the frozen protocol",
    "person_arm": "arm 'human' added by this amendment; descriptive control outside the "
                  "primary and secondary contrasts, the non-inferiority test and the gate",
}
PHASE1_CODE_A2 = ("The phase-1 code snapshot (serving settings from phase 0) will be recorded "
                  "by amendment 2 before tag xp-p1-v1, with plan, case files, test "
                  "membership, protocol and analysis-code hashes unchanged.")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                          check=True).stdout


def _show(root: Path, commit: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=root,
                          capture_output=True, check=True).stdout


def analysis_paths(root: Path, commit: str) -> list[str]:
    names = _git(root, "ls-tree", "--name-only", commit, "slh/xp/analysis/").split()
    return sorted([n for n in names if n.endswith(".py")] + ["slh/xp/partner/check.py"])


def _constant(source: bytes, name: str) -> float:
    m = re.search(rf"^{name}\s*=\s*([0-9_.]+)", source.decode("utf-8"), re.MULTILINE)
    if not m:
        raise ValueError(f"{name} not found")
    return float(m.group(1).replace("_", ""))


def amended_plan(public: Path, root: Path, commit: str) -> dict[str, Any]:
    from slh.xp.partner import plan as pp

    plan = json.loads((public / "package/plan.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in plan.items() if k != "sha256"}
    texts = {t: _show(root, commit, f"{PERSON_DIR}/{t}.txt").decode("utf-8").strip()
             for t in plan["tasks"]}
    body["arms"] = body["arms"] + [pp.person_arm(texts)]
    new = {**body, "sha256": pc.sha_text(pc.canonical(body))}
    pp.verify(new)
    return new


def build_amendment(public: Path = PUBLIC, root: Path = REPO,
                    commit: str | None = None) -> dict[str, Any]:
    from slh.xp.partner import plan as pp

    commit = commit or _git(root, "rev-parse", "HEAD").strip()
    base = build_event(public, root)
    new = amended_plan(public, root, commit)
    if new["sha256"] != AMENDED_PLAN_SHA256:
        raise ValueError(f"amended plan hashes to {new['sha256']}, not the approved one")
    paths = analysis_paths(root, commit)
    code = {p: hashlib.sha256(_show(root, commit, p)).hexdigest() for p in paths}
    stats_src = _show(root, commit, "slh/xp/analysis/stats.py")
    report_src = _show(root, commit, "slh/xp/analysis/report.py")
    arm = new["arms"][-1]
    person = {t: {"file": f"{PERSON_DIR}/{t}.txt",
                  "file_sha256": hashlib.sha256(_show(root, commit,
                                                      f"{PERSON_DIR}/{t}.txt")).hexdigest(),
                  "text_sha256": pc.sha_text(arm["texts"][t]),
                  "prompt_hash": pp.prompt_hash("notes", arm["texts"][t])}
              for t in new["tasks"]}
    return {
        "event": AMEND_EVENT, "amendment_id": AMEND_ID, "amends": REGISTRATION_ID,
        "plan_id": PLAN_ID, "doc_path": AMEND_DOC, "doc_sha256": _sha(root / AMEND_DOC),
        "code_commit": commit, "analysis_code_sha256": code,
        "bootstrap_seed": int(_constant(stats_src, "SEED")),
        "bootstrap_draws": int(_constant(stats_src, "DRAWS")),
        "error_rate_margin_points": _constant(report_src, "ERROR_RATE_MARGIN"),
        "decisions": AMEND_DECISIONS,
        "previous_plan_sha256": base["plan_sha256"],
        "plan_sha256": new["sha256"],
        "plan_file_sha256": pc.sha_text(json.dumps(new, sort_keys=True, indent=1) + "\n"),
        "plan_derivation": "registered public plan.json with arm 'human' appended "
                           "(slh.xp.partner.plan.person_arm), self-hash recomputed",
        "person_arm": {"name": arm["name"], "kind": arm["kind"], "scope": arm["scope"],
                       "texts": person},
        "protocol_sha256": base["protocol_sha256"], "case_files": base["case_files"],
        "test_membership": base["test_membership"],
        "owner_approval": AMEND_APPROVAL, "phase1_code_note": PHASE1_CODE_A2,
    }


def check_amendment(registry_dir: Path = REPO / "learner_runs", public: Path = PUBLIC,
                    root: Path = REPO) -> dict[str, Any]:
    rows = Registry(registry_dir).rows()
    found = [r for r in rows if r.get("event") == AMEND_EVENT
             and r.get("amendment_id") == AMEND_ID]
    if len(found) != 1:
        raise ValueError(f"expected one {AMEND_EVENT} row, found {len(found)}")
    row = found[0]
    base = registered(rows)
    if len(base) != 1 or rows.index(base[0]) > rows.index(row):
        raise ValueError("the amendment must follow exactly one registration row")
    fresh = build_amendment(public, root, row["code_commit"])
    stored = {k: v for k, v in row.items() if k != "prev_sha256"}
    if stored != fresh:
        diff = sorted(k for k in set(stored) | set(fresh) if stored.get(k) != fresh.get(k))
        raise ValueError(f"registered amendment differs from a recomputation: {diff}")
    for key in ("protocol_sha256", "case_files", "test_membership"):
        if row[key] != base[0][key]:
            raise ValueError(f"amendment changed {key}")
    index = rows.index(row)
    line = (registry_dir / "REGISTRY.jsonl").read_bytes().splitlines(keepends=True)[index]
    return {"ok": True, "rows": len(rows), "line": index + 1,
            "row_sha256": hashlib.sha256(line).hexdigest(),
            "prev_sha256": row["prev_sha256"]}


def append_amendment(registry_dir: Path = REPO / "learner_runs", public: Path = PUBLIC,
                     root: Path = REPO) -> dict[str, Any]:
    registry = Registry(registry_dir)
    rows = registry.rows()
    if not registered(rows):
        raise ValueError("no phase-1 registration row to amend")
    if any(r.get("event") == AMEND_EVENT and r.get("amendment_id") == AMEND_ID for r in rows):
        raise FileExistsError(f"{AMEND_ID} already registered; append refused")
    registry.append(build_amendment(public, root))
    return check_amendment(registry_dir, public, root)


# --------------------------------------------------------------------------- amendment 2

AMEND2_ID = f"{PLAN_ID}-phase1-amendment-2"
AMEND2_DOC = "docs/xp/registration-phase1-amendment-2.md"
AMEND2_COMMIT = "a68f8991c75aae48ce0445c6e98b593b152103a6"
AMEND2_PLAN_SHA256 = "c6ebe7bb08764a7bbd828900d9814af5ca10ebc67ee1983579dd9bf197c9c5c0"
AMEND2_SNAPSHOT = "74c8b7c76b1fad4bcfead00c304f939faa37a592"
PHASE1_TAG = "xp-p1-v1"
PUBLIC_V1 = Path.home() / ".slh-xp/public" / PHASE1_TAG
GATE_CODE = ("slh/xp/partner/gate.py", "slh/xp/partner/estimate.py",
             "slh/xp/partner/context.py", "scripts/xp/slurm/run_all.sh",
             "scripts/xp/slurm/submit_phase1.sh", "scripts/xp/slurm/lib.sh")
GATE_CONSTANTS = ("RULE", "GPU_HOURS_CAP", "STARTUP_BOUND_S", "CONTEXT_LIMIT_SHARE",
                  "PAIRED_MODE", "PAIRED_MAX_CASE_DIFF", "PAIRED_ERROR_DIFF",
                  "ESTIMATE_MARGIN", "TIME_LIMIT_CLASSES", "MAX_MODEL_LEN",
                  "CONCURRENCY_BOUNDS", "UNITS_BOUNDS", "TIME_LIMIT_BOUNDS_H",
                  "UTIL_SUM_MAX", "TEACHER_EXTRA_ALLOWED")
TEACHER_DEFAULTS = ("TEACHER_REPO", "TEACHER_REVISION", "TEACHER_NAME", "TEACHER_TP",
                    "TEACHER_GPU_UTIL", "TEACHER_EXPERT_PARALLEL", "TEACHER_MAX_LEN",
                    "TEACHER_EAGER", "TEACHER_EXTRA_ARGS", "AGENT_TP", "AGENT_DP",
                    "AGENT_GPU_UTIL", "LEARNING_UNITS")
AMEND2_APPROVALS = [
    "2026-10-11: teacher switch XD1 (docs/xp/deviations.md)",
    "2026-10-12: advisory critic for every writer",
    "2026-10-12: keep writer T with all 12 runs",
    "2026-10-12: GPU-hour cap 1,600 with the 0.8 x cap admission rule",
    "2026-10-12: full-context whole-episode fit rule with decision-growth reserve",
    "2026-10-12: per-case context-limit handling",
    "2026-10-12: parallel learning",
    "2026-10-12: serving defaults (CUDA graphs with eager fallback, expert parallel, "
    "engram cpu_offload off, language-model-only, teacher memory share 0.68)",
    "2026-10-12: one-command flow with its gate conditions, relaxed slice paired check "
    "(more than 5 error cases -> NO-GO; strict 2-point rule stays in the final analysis), "
    "the derivation rule xp-gate-v1 and gate-bound serving settings",
]
AMEND2_NOTE = ("No phase-1 result exists; phase-0 runs are site checks whose smoke scores "
               "nobody has read. Analysis code is not hash-identical to amendment 1 "
               "(labels, package path, refactor and added checks in check.py; stats.py "
               "unchanged; scoring of test decisions unchanged); the hashes here bind "
               "phase 1.")


def _literals(source: bytes, names: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for node in ast.parse(source).body:
        target = node.targets[0] if isinstance(node, ast.Assign) else \
            node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(target, ast.Name) and target.id in names and node.value is not None:
            value = node.value
            if isinstance(value, ast.Call) and getattr(value.func, "id", "") == "frozenset":
                out[target.id] = sorted(ast.literal_eval(value.args[0]))
            else:
                v = ast.literal_eval(value)
                out[target.id] = list(v) if isinstance(v, tuple) else v
    missing = set(names) - set(out)
    if missing:
        raise ValueError(f"constants not found: {sorted(missing)}")
    return out


def gate_conditions(source: bytes) -> list[str]:
    """The condition names in the order ``_evaluate`` appends them (both spellings of a
    mode-dependent name)."""
    fn = next(n for n in ast.parse(source).body
              if isinstance(n, ast.FunctionDef) and n.name == "_evaluate")
    names = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_cond":
            arg = node.args[0]
            parts = [arg.body, arg.orelse] if isinstance(arg, ast.IfExp) else [arg]
            names.append(" | ".join(ast.literal_eval(a) for a in parts))
    return sorted(set(names), key=lambda n: source.decode().index(n.split(" | ")[0]))


DERIVATION_FUNCTIONS = ("derive", "adjust", "time_limit")
DERIVATION_RULE = ("TEACHER_EAGER 0 if the CUDA-graph teacher started within its bound, else 1; "
                   "CONCURRENCY = phase 0's, halved if more than 1% of slice rows hit a "
                   "time-limit class, clamped to CONCURRENCY_BOUNDS; LEARNING_UNITS 12, or 8 "
                   "if the measure run recorded case-level errors, clamped to UNITS_BOUNDS; "
                   "time limit = ceil(rescaled conservative wall / jobs) + 1 h, clamped to "
                   "TIME_LIMIT_BOUNDS_H; GPU_HOURS_CAP registered")


def derivation_sources(source: bytes) -> dict[str, str]:
    """SHA-256 of each derivation function's exact source text in gate.py."""
    text = source.decode("utf-8")
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in DERIVATION_FUNCTIONS:
            out[node.name] = pc.sha_text(ast.get_source_segment(text, node))
    if set(out) != set(DERIVATION_FUNCTIONS):
        raise ValueError("derivation functions not found")
    return out


def shell_defaults(source: bytes, names: tuple[str, ...]) -> dict[str, str]:
    text = source.decode("utf-8")
    out = {}
    for n in names:
        m = re.search(rf'^{n}="\$\{{{n}:-(.*?)\}}"', text, re.M) or \
            re.search(rf"^\s*{n}='(.*?)'", text, re.M)
        if not m:
            raise ValueError(f"default for {n} not found")
        out[n] = m.group(1)
    return out


def public_hashes(public: Path) -> dict[str, Any]:
    pkg = public / "package"
    plan = json.loads((pkg / "plan.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in plan.items() if k != "sha256"}
    if pc.sha_text(pc.canonical(body)) != plan.get("sha256"):
        raise ValueError("plan.json does not hash to its own sha256")
    reg = json.loads((pkg / "registration.json").read_text(encoding="utf-8"))
    cases = {f"cases/{p.name}": _sha(p) for p in sorted((pkg / "cases").glob("*.jsonl"))}
    out = {"manifest_sha256": _sha(public / "MANIFEST.json"),
           "plan_sha256": plan["sha256"], "plan_file_sha256": _sha(pkg / "plan.json"),
           "registration_file_sha256": _sha(pkg / "registration.json"),
           "protocol_sha256": _sha(pkg / "protocol.md"), "case_files": cases,
           "test_membership": membership(public),
           "snapshot_commit": reg["git_commit"], "source_commit": reg["source"]["commit"],
           "code_sha256": reg["code_sha256"]}
    if reg["plan_sha256"] != plan["sha256"] or reg["protocol_sha256"] != \
            out["protocol_sha256"] or plan["protocol"]["sha256"] != out["protocol_sha256"] \
            or reg["case_files"] != cases or reg["test_membership"] != out["test_membership"]:
        raise ValueError("public tree disagrees with its own registration.json")
    return out, plan


def _plan_changes(old: dict[str, Any], new: dict[str, Any], path: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new)):
            if k == "sha256" and not path:
                continue
            if old.get(k) != new.get(k):
                out.update(_plan_changes(old.get(k), new.get(k), f"{path}.{k}".lstrip(".")))
        return out
    return {path: new}


def build_amendment2(public: Path = PUBLIC_V1, root: Path = REPO,
                     rows: list[dict[str, Any]] | None = None,
                     commit: str = AMEND2_COMMIT) -> dict[str, Any]:
    rows = Registry(root / "learner_runs").rows() if rows is None else rows
    a1 = [r for r in rows if r.get("event") == AMEND_EVENT and r.get("amendment_id") == AMEND_ID]
    if len(a1) != 1:
        raise ValueError("amendment 2 needs exactly one amendment-1 row")
    a1 = a1[0]
    pub, plan = public_hashes(public)
    if pub["plan_sha256"] != AMEND2_PLAN_SHA256 or pub["snapshot_commit"] != AMEND2_SNAPSHOT \
            or pub["source_commit"] != commit:
        raise ValueError("public tree is not the amendment-2 build")
    for key in ("protocol_sha256", "case_files", "test_membership"):
        if pub[key] != a1[key]:
            raise ValueError(f"{key} differs from amendment 1")
    old = amended_plan(PUBLIC, root, a1["code_commit"])
    changes = _plan_changes(old, plan)
    paths = analysis_paths(root, commit)
    analysis = {p: hashlib.sha256(_show(root, commit, p)).hexdigest() for p in paths}
    changed = sorted(p for p in set(analysis) | set(a1["analysis_code_sha256"])
                     if analysis.get(p) != a1["analysis_code_sha256"].get(p))
    stats_src = _show(root, commit, "slh/xp/analysis/stats.py")
    report_src = _show(root, commit, "slh/xp/analysis/report.py")
    gate_src = _show(root, commit, "slh/xp/partner/gate.py")
    return {
        "event": AMEND_EVENT, "amendment_id": AMEND2_ID, "amends": REGISTRATION_ID,
        "follows": AMEND_ID, "plan_id": PLAN_ID,
        "doc_path": AMEND2_DOC, "doc_sha256": _sha(root / AMEND2_DOC),
        "source_docs_sha256": {d: hashlib.sha256(_show(root, commit, d)).hexdigest()
                               for d in ("docs/xp/amendment-2.md", "docs/xp/deviations.md",
                                         "docs/xp/conventions.md")},
        "code_commit": commit, "phase1_tag": PHASE1_TAG, "public": pub,
        "previous_plan_sha256": a1["plan_sha256"], "plan_changes": changes,
        "analysis_code_sha256": analysis, "analysis_changed_since_amendment_1": changed,
        "bootstrap_seed": int(_constant(stats_src, "SEED")),
        "bootstrap_draws": int(_constant(stats_src, "DRAWS")),
        "error_rate_margin_points": _constant(report_src, "ERROR_RATE_MARGIN"),
        "gate_code_sha256": {p: hashlib.sha256(_show(root, commit, p)).hexdigest()
                             for p in GATE_CODE},
        "gate_constants": _literals(gate_src, GATE_CONSTANTS),
        "gate_conditions": gate_conditions(gate_src),
        "derivation_rule": DERIVATION_RULE,
        "derivation_source_sha256": derivation_sources(gate_src),
        "serving_defaults": shell_defaults(_show(root, commit, "scripts/xp/slurm/lib.sh"),
                                           TEACHER_DEFAULTS),
        "owner_approvals": AMEND2_APPROVALS, "note": AMEND2_NOTE,
    }


def _find(rows: list[dict[str, Any]], amendment_id: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("event") == AMEND_EVENT
            and r.get("amendment_id") == amendment_id]


def check_amendment2(registry_dir: Path = REPO / "learner_runs", public: Path = PUBLIC_V1,
                     root: Path = REPO) -> dict[str, Any]:
    rows = Registry(registry_dir).rows()
    found = _find(rows, AMEND2_ID)
    if len(found) != 1:
        raise ValueError(f"expected one {AMEND2_ID} row, found {len(found)}")
    row = found[0]
    a1 = _find(rows, AMEND_ID)
    if len(a1) != 1 or rows.index(a1[0]) > rows.index(row):
        raise ValueError("amendment 2 must follow exactly one amendment-1 row")
    fresh = build_amendment2(public, root, rows, row["code_commit"])
    stored = {k: v for k, v in row.items() if k != "prev_sha256"}
    if stored != fresh:
        diff = sorted(k for k in set(stored) | set(fresh) if stored.get(k) != fresh.get(k))
        raise ValueError(f"registered amendment 2 differs from a recomputation: {diff}")
    index = rows.index(row)
    line = (registry_dir / "REGISTRY.jsonl").read_bytes().splitlines(keepends=True)[index]
    return {"ok": True, "rows": len(rows), "line": index + 1,
            "row_sha256": hashlib.sha256(line).hexdigest(),
            "prev_sha256": row["prev_sha256"]}


def append_amendment2(registry_dir: Path = REPO / "learner_runs", public: Path = PUBLIC_V1,
                      root: Path = REPO) -> dict[str, Any]:
    registry = Registry(registry_dir)
    rows = registry.rows()
    if _find(rows, AMEND2_ID):
        raise FileExistsError(f"{AMEND2_ID} already registered; append refused")
    registry.append(build_amendment2(public, root, rows))
    return check_amendment2(registry_dir, public, root)


# --------------------------------------------------------------------------- amendment 3

AMEND3_ID = f"{PLAN_ID}-phase1-amendment-3"
AMEND3_DOC = "docs/xp/registration-phase1-amendment-3.md"
AMEND3_APPROVAL = "owner approved amendment 3 on 2026-10-13, before any phase-1 result was read"
AMEND3_RULE = ("For each confirmatory contrast, when the wrong-write interval is degenerate "
               "(zero width) or rests on fewer than EXACT_MIN_DISCORDANT discordant pairs, "
               "non-inferiority is judged on 100 x the one-sided 97.5% Clopper-Pearson upper "
               "bound of (endpoint-only wrong-write pairs / scored pairs), pooled over tasks "
               "and runs; it holds only if that bound is below the 2-point margin. Otherwise "
               "the registered crossed bootstrap-t interval is used unchanged.")
AMEND3_STATUS = ("No phase-1 result has been received or read. The published tag xp-p1-v1 "
                 "is unchanged (its analysis copy has the amendment-2 hashes); the analysis "
                 "runs on our side only.")


def build_amendment3(public: Path = PUBLIC_V1, root: Path = REPO,
                     rows: list[dict[str, Any]] | None = None,
                     commit: str | None = None) -> dict[str, Any]:
    rows = Registry(root / "learner_runs").rows() if rows is None else rows
    a2 = _find(rows, AMEND2_ID)
    if len(a2) != 1:
        raise ValueError("amendment 3 needs exactly one amendment-2 row")
    a2 = a2[0]
    commit = commit or _git(root, "rev-parse", "HEAD").strip()
    pub, _ = public_hashes(public)
    if pub != a2["public"]:
        raise ValueError("the public tree differs from the one amendment 2 registered")
    shipped = {p: _sha(public / p) for p in a2["analysis_code_sha256"]
               if p.startswith("slh/xp/analysis/")}
    if any(shipped[p] != a2["analysis_code_sha256"][p] for p in shipped):
        raise ValueError("the published tag's analysis copy changed")
    paths = analysis_paths(root, commit)
    analysis = {p: hashlib.sha256(_show(root, commit, p)).hexdigest() for p in paths}
    changed = sorted(p for p in set(analysis) | set(a2["analysis_code_sha256"])
                     if analysis.get(p) != a2["analysis_code_sha256"].get(p))
    stats_src = _show(root, commit, "slh/xp/analysis/stats.py")
    report_src = _show(root, commit, "slh/xp/analysis/report.py")
    test = "tests/test_xp_analysis_exact.py"
    return {
        "event": AMEND_EVENT, "amendment_id": AMEND3_ID, "amends": REGISTRATION_ID,
        "follows": AMEND2_ID, "plan_id": PLAN_ID,
        "doc_path": AMEND3_DOC, "doc_sha256": _sha(root / AMEND3_DOC),
        "code_commit": commit, "analysis_code_sha256": analysis,
        "analysis_changed_since_amendment_2": changed,
        "test_sha256": {test: hashlib.sha256(_show(root, commit, test)).hexdigest()},
        "rule": AMEND3_RULE,
        "exact_min_discordant": int(_constant(stats_src, "EXACT_MIN_DISCORDANT")),
        "exact_alpha_one_sided": _constant(stats_src, "ALPHA") / 2,
        "wrong_write_margin_points": 2.0,
        "bootstrap_seed": int(_constant(stats_src, "SEED")),
        "bootstrap_draws": int(_constant(stats_src, "DRAWS")),
        "error_rate_margin_points": _constant(report_src, "ERROR_RATE_MARGIN"),
        "plan_sha256": pub["plan_sha256"], "plan_file_sha256": pub["plan_file_sha256"],
        "protocol_sha256": pub["protocol_sha256"], "case_files": pub["case_files"],
        "test_membership": pub["test_membership"], "phase1_tag": PHASE1_TAG,
        "owner_approval": AMEND3_APPROVAL, "status": AMEND3_STATUS,
    }


def check_amendment3(registry_dir: Path = REPO / "learner_runs", public: Path = PUBLIC_V1,
                     root: Path = REPO) -> dict[str, Any]:
    rows = Registry(registry_dir).rows()
    found = _find(rows, AMEND3_ID)
    if len(found) != 1:
        raise ValueError(f"expected one {AMEND3_ID} row, found {len(found)}")
    row = found[0]
    a2 = _find(rows, AMEND2_ID)
    if len(a2) != 1 or rows.index(a2[0]) > rows.index(row):
        raise ValueError("amendment 3 must follow exactly one amendment-2 row")
    fresh = build_amendment3(public, root, rows, row["code_commit"])
    stored = {k: v for k, v in row.items() if k != "prev_sha256"}
    if stored != fresh:
        diff = sorted(k for k in set(stored) | set(fresh) if stored.get(k) != fresh.get(k))
        raise ValueError(f"registered amendment 3 differs from a recomputation: {diff}")
    index = rows.index(row)
    line = (registry_dir / "REGISTRY.jsonl").read_bytes().splitlines(keepends=True)[index]
    return {"ok": True, "rows": len(rows), "line": index + 1,
            "row_sha256": hashlib.sha256(line).hexdigest(),
            "prev_sha256": row["prev_sha256"]}


def append_amendment3(registry_dir: Path = REPO / "learner_runs", public: Path = PUBLIC_V1,
                      root: Path = REPO) -> dict[str, Any]:
    registry = Registry(registry_dir)
    rows = registry.rows()
    if _find(rows, AMEND3_ID):
        raise FileExistsError(f"{AMEND3_ID} already registered; append refused")
    registry.append(build_amendment3(public, root, rows))
    return check_amendment3(registry_dir, public, root)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", type=Path, default=PUBLIC)
    ap.add_argument("--registry-dir", type=Path, default=REPO / "learner_runs")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--append-amendment-1", action="store_true")
    ap.add_argument("--append-amendment-2", action="store_true")
    ap.add_argument("--append-amendment-3", action="store_true")
    ap.add_argument("--public-v1", type=Path, default=PUBLIC_V1)
    args = ap.parse_args(argv)
    if args.append_amendment_3:
        out = {"amendment_3": append_amendment3(args.registry_dir, args.public_v1)}
    elif args.append_amendment_2:
        out = {"amendment_2": append_amendment2(args.registry_dir, args.public_v1)}
    elif args.append_amendment_1:
        out = {"amendment_1": append_amendment(args.registry_dir, args.public)}
    elif args.append:
        out = {"registration": append(args.registry_dir, args.public)}
    else:
        out = {"registration": check(args.registry_dir, args.public),
               "amendment_1": check_amendment(args.registry_dir, args.public),
               "amendment_2": check_amendment2(args.registry_dir, args.public_v1),
               "amendment_3": check_amendment3(args.registry_dir, args.public_v1)}
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
