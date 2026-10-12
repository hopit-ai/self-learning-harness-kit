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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", type=Path, default=PUBLIC)
    ap.add_argument("--registry-dir", type=Path, default=REPO / "learner_runs")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--append-amendment-1", action="store_true")
    args = ap.parse_args(argv)
    if args.append_amendment_1:
        out = {"amendment_1": append_amendment(args.registry_dir, args.public)}
    elif args.append:
        out = {"registration": append(args.registry_dir, args.public)}
    else:
        out = {"registration": check(args.registry_dir, args.public),
               "amendment_1": check_amendment(args.registry_dir, args.public)}
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
