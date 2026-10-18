"""Arm texts for phase 2, frozen from the phase-1 export (docs/xp/phase2-design.md, s. 1).

An arm puts one bundle per learning run into B1's notes slot: run r's address,
variant and sequence texts as three notes in that fixed order (empty texts are
dropped, so an all-empty bundle is B1's prompt). The texts are the phase-1
endpoint skills **chosen by the phase-1 sentinel rule** (``skill.md`` and
``search-control.md`` of each learning unit; the selection record is in the
unit's ``summary.json``). Nothing here reads a decision row, a test score or an
answer key: only ``skills.jsonl.gz`` and the ``summary`` rows of
``learning.jsonl.gz``, after each export file's hash is checked against its
manifest.

``bundles.json`` (``xp-phase2-bundles-v1``)::

    {"schema", "selection": "phase1-sentinel", "writer": "T+",
     "sources": [{"manifest_sha256", "plan_id", "shard"}],
     "kinds": {"learned": {"<run>": {"texts": [a, v, s], "sha256": [...]}},
               "search_control": {...}},
     "human": {"texts": [a, v, s], "sha256": [...]} | null}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from slh.xp import phase2 as P
from slh.xp.partner import common as pc

BUNDLE_SCHEMA = "xp-phase2-bundles-v1"
TASK_ORDER = ("address", "variant", "sequence")
ARM_KIND_TEXT = {"learned": "learned", "search_control": "search_control"}
SUMMARY_SHA = {"learned": "skill_sha256", "search_control": "search_control_sha256"}
PERSON_DIR = "docs/xp/person-procedures"


class BundleRefused(ValueError):
    pass


def _sha_or_none(text: str) -> str | None:
    return pc.sha_text(text) if text else None


def _read_export(export: Path) -> tuple[dict[str, Any], list[dict[str, Any]],
                                        list[dict[str, Any]]]:
    manifest_path = Path(export) / "manifest.json"
    if not manifest_path.is_file():
        raise BundleRefused(f"{export}: manifest.json missing")
    manifest = pc.read_json(manifest_path)
    for name in ("skills.jsonl.gz", "learning.jsonl.gz"):
        meta = (manifest.get("files") or {}).get(name)
        path = Path(export) / name
        if meta is None or not path.is_file() or pc.sha_file(path) != meta["sha256"]:
            raise BundleRefused(f"{export}/{name}: missing or differs from the manifest")
    skills = pc.read_jsonl(Path(export) / "skills.jsonl.gz")
    summaries = [r for r in pc.read_jsonl(Path(export) / "learning.jsonl.gz")
                 if r.get("source") == "summary"]
    return manifest, skills, summaries


def from_phase1_exports(exports: list[Path], *, runs: int, writer: str = "T+",
                        human: list[str] | None = None) -> dict[str, Any]:
    """Bundles for runs 0..runs-1 from one or more phase-1 export shards."""
    texts: dict[tuple[str, str, int], str] = {}
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    sources = []
    for export in exports:
        manifest, skills, sums = _read_export(Path(export))
        sources.append({"manifest_sha256": pc.sha_file(Path(export) / "manifest.json"),
                        "plan_id": manifest.get("plan_id"), "shard": manifest.get("shard")})
        for r in skills:
            if r.get("writer") != writer or r.get("kind") not in ARM_KIND_TEXT.values():
                continue
            key = (r["kind"], r["task"], int(r["run"]))
            if key in texts:
                raise BundleRefused(f"duplicate skill row {key}")
            if r.get("sha256") != _sha_or_none(r.get("text") or ""):
                raise BundleRefused(f"skill row {key}: text differs from its sha256")
            texts[key] = r.get("text") or ""
        for r in sums:
            if r.get("writer") != writer:
                continue
            key = (r["task"], int(r["run"]))
            if key in summaries:
                raise BundleRefused(f"duplicate summary row {key}")
            summaries[key] = r["row"]
    kinds: dict[str, dict[str, Any]] = {}
    for kind, text_kind in ARM_KIND_TEXT.items():
        per_run = {}
        for run in range(runs):
            bundle, hashes = [], []
            for task in TASK_ORDER:
                summary = summaries.get((task, run))
                if summary is None:
                    raise BundleRefused(f"no learning summary for {task} run {run} ({writer})")
                selection = (summary.get("selection") or {}).get(kind)
                if not isinstance(selection, dict) or "selected_sha256" not in selection:
                    raise BundleRefused(f"{task} run {run}: no {kind} selection record "
                                        "(the sentinel rule did not run)")
                text = texts.get((text_kind, task, run), "")
                sha = _sha_or_none(text)
                if sha != summary.get(SUMMARY_SHA[kind]) or sha != selection["selected_sha256"]:
                    raise BundleRefused(f"{task} run {run}: the {kind} text is not the one the "
                                        "sentinel rule selected")
                bundle.append(text)
                hashes.append(sha)
            per_run[str(run)] = {"texts": bundle, "sha256": hashes}
        kinds[kind] = per_run
    return {"schema": BUNDLE_SCHEMA, "selection": P.SELECTION_RULE, "writer": writer,
            "sources": sources, "kinds": kinds,
            "human": None if human is None else {"texts": list(human),
                                                 "sha256": [_sha_or_none(t) for t in human]}}


def person_procedures(root: Path = pc.REPO) -> list[str]:
    """The approved person-written procedures, in task order."""
    return [(Path(root) / PERSON_DIR / f"{t}.txt").read_text(encoding="utf-8")
            for t in TASK_ORDER]


def problems(bundles: dict[str, Any], plan: P.Phase2Plan) -> list[str]:
    out = []
    if bundles.get("schema") != BUNDLE_SCHEMA:
        return [f"bundles: unknown schema {bundles.get('schema')!r}"]
    if bundles.get("selection") != P.SELECTION_RULE:
        out.append(f"bundles: texts must be chosen by {P.SELECTION_RULE!r}")
    for kind, per_run in (bundles.get("kinds") or {}).items():
        for run, b in per_run.items():
            if len(b["texts"]) != len(TASK_ORDER) or \
                    [_sha_or_none(t) for t in b["texts"]] != b["sha256"]:
                out.append(f"bundles: {kind} run {run} texts differ from their hashes")
    for arm in plan.arms:
        if arm.kind in ARM_KIND_TEXT:
            have = (bundles.get("kinds") or {}).get(arm.kind) or {}
            missing = [r for r in arm.runs if str(r) not in have]
            if missing:
                out.append(f"bundles: no {arm.kind} texts for runs {missing}")
        elif arm.kind == "human":
            h = bundles.get("human")
            if not h or [_sha_or_none(t) for t in h["texts"]] != h["sha256"]:
                out.append("bundles: the person-written arm needs its procedure texts")
        elif arm.kind != "b1":
            out.append(f"bundles: arm kind {arm.kind!r} is not run in phase 2")
    return out


def notes(bundles: dict[str, Any], kind: str, run: int) -> list[str]:
    """The notes an arm instance loads (empty texts dropped), in task order."""
    if kind == "b1":
        return []
    if kind == "human":
        texts = bundles["human"]["texts"]
    elif kind in ARM_KIND_TEXT:
        texts = bundles["kinds"][kind][str(run)]["texts"]
    else:
        raise BundleRefused(f"arm kind {kind!r} has no phase-2 texts")
    return [t.strip() for t in texts if t and t.strip()]
