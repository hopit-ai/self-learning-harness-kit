"""Compact export of a partner run directory (docs/xp/partner-package.md, section 3).

    python -m slh.xp.partner.export --run RUN_DIR --out EXPORT_DIR
        [--cap-bytes N] [--transcript-sample 500] [--transcript-seed 0]
        [--key-arm learned --key-arm b1] [--task variant]

Run directory (written by ``slh.xp.partner.runner``)::

    plan.json, hashes.json, env.json, test-ids.json
    arms.jsonl                     arm instances
    decisions.jsonl                one row per executed decision (deduped by prompt hash)
    learning/<task>/<writer>/run-<k>/{trace,patches,episodes}.jsonl, summary.json,
                                     skill.md, generic.md, evidence.md, search-control.md
    logs/*.jsonl                   flat cost, timing, error and repair rows
    conversations.jsonl            phase 2 only

Every exported row passes the typed allowlist of its file (``schemas``): only
listed fields leave, a wrong type refuses the export. Answer keys
(``common.GOLD_KEYS``) are removed from free-form learning rows; a test case id
in learning, skill or conversation rows refuses the export (learning never sees
test). The cap is the plan's registered ``export_cap_bytes`` (100,000,000 bytes);
a caller may lower it, never raise it. Sizes are computed BEFORE writing; above
the cap nothing is written and ``ExportTooLarge`` lists what to trim. ``--task``
writes one shard (its manifest names the shard's tasks); the checker merges shards.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from slh.xp.partner import common as pc
from slh.xp.partner import schemas as sc

EXPORT_SCHEMA = "xp-partner-export-v2"
LEARNING_FILES = {"trace.jsonl": "trace", "patches.jsonl": "patches",
                  "episodes.jsonl": "episodes", "search_trace.jsonl": "search_trace",
                  "acquisition.jsonl": "acquisition"}
TEXT_FILES = {"skill.md": "learned", "generic.md": "generic", "evidence.md": "full_context",
              "search-control.md": "search_control"}
_ID_TOKEN = re.compile(r"[A-Za-z0-9]+-[0-9a-f]{6,}")


class ExportTooLarge(RuntimeError):
    def __init__(self, report: dict[str, Any]) -> None:
        super().__init__(f"export would be {report['total_bytes']} bytes, over the cap of "
                         f"{report['cap_bytes']}; nothing written. Trim: "
                         + "; ".join(report["suggestions"]))
        self.report = report


class ExportRefused(ValueError):
    pass


def _learning_dirs(run: Path) -> list[tuple[str, str, int, Path]]:
    out = []
    for path in sorted((run / "learning").glob("*/*/run-*")):
        if path.is_dir():
            task, writer = path.parent.parent.name, path.parent.name
            out.append((task, writer, int(path.name.split("-", 1)[1]), path))
    return out


def _sample_ids(rows: list[dict[str, Any]], n: int, seed: int, key_arms: set[str] | None
                ) -> set[str]:
    ids = sorted({r["conversation_id"] for r in rows})
    rng = random.Random(seed)
    chosen = set(rng.sample(ids, min(n, len(ids))))
    for r in rows:
        if (key_arms is None or r.get("arm") in key_arms) and \
                r.get("reward") is not None and r["reward"] < 1:
            chosen.add(r["conversation_id"])
    return chosen


def _no_test_ids(name: str, row: Any, test_ids: set[str]) -> None:
    hit = set(_ID_TOKEN.findall(pc.canonical(row))) & test_ids
    if hit:
        raise ExportRefused(f"{name}: a row names test case {min(hit)} "
                            "(learning, skills and conversations never see test cases)")


def collect(run: Path, *, tasks: set[str], transcript_sample: int = 500,
            transcript_seed: int = 0, key_arms: set[str] | None = None
            ) -> dict[str, list[dict[str, Any]]]:
    """File name -> rows (allowlisted, typed, scrubbed)."""
    run = Path(run)
    ids_path = run / "test-ids.json"
    if not ids_path.is_file():
        raise ExportRefused("test-ids.json missing from the run directory")
    test_ids = set(pc.read_json(ids_path))
    files: dict[str, list[dict[str, Any]]] = {}
    files["decisions.jsonl.gz"] = sorted(
        (sc.pick("decisions.jsonl.gz", r) for r in pc.read_jsonl(run / "decisions.jsonl")
         if r.get("task") in tasks),
        key=lambda r: (r["task"], r["prompt_hash"], r["case_id"]))
    files["arms.jsonl.gz"] = [sc.pick("arms.jsonl.gz", r)
                              for r in pc.read_jsonl(run / "arms.jsonl") if r["task"] in tasks]
    learning, skills = [], []
    for task, writer, k, path in _learning_dirs(run):
        if task not in tasks:
            continue
        tag = {"task": task, "writer": writer, "run": k}
        rows = [(source, r) for fname, source in LEARNING_FILES.items()
                for r in pc.read_jsonl(path / fname)]
        if (path / "summary.json").is_file():
            rows.append(("summary", pc.read_json(path / "summary.json")))
        for source, r in rows:
            row = sc.pick("learning.jsonl.gz", {**tag, "source": source, "row": pc.scrub(r)})
            _no_test_ids("learning", row, test_ids)
            learning.append(row)
        for fname, kind in TEXT_FILES.items():
            if (path / fname).is_file():
                text = (path / fname).read_text(encoding="utf-8")
                row = sc.pick("skills.jsonl.gz", {**tag, "kind": kind, "text": text,
                                                  "sha256": pc.sha_text(text) if text else None})
                _no_test_ids("skills", row, test_ids)
                skills.append(row)
    files["learning.jsonl.gz"], files["skills.jsonl.gz"] = learning, skills
    logs = []
    for path in sorted((run / "logs").glob("*.jsonl")):
        logs += [sc.pick("logs.jsonl.gz", {"name": path.name, "row": r})
                 for r in pc.read_jsonl(path)]
    files["logs.jsonl.gz"] = logs
    convs = [r for r in pc.read_jsonl(run / "conversations.jsonl") if r.get("task") in tasks]
    if convs:
        chosen = _sample_ids(convs, transcript_sample, transcript_seed, key_arms)
        out = []
        for r in convs:
            included = r["conversation_id"] in chosen and "transcript" in r
            r = {**pc.scrub(r), "transcript_included": included}
            if not included:
                r.pop("transcript", None)
            row = sc.pick("conversations.jsonl.gz", r)
            _no_test_ids("conversations", row, test_ids)
            out.append(row)
        files["conversations.jsonl.gz"] = out
    for name, rows in files.items():
        for i, row in enumerate(rows):
            pc.assert_no_gold(row, f"{name}[{i}]")
    return files


def _suggestions(sizes: dict[str, int], files: dict[str, list[dict[str, Any]]],
                 cap: int, transcript_sample: int) -> list[str]:
    out = []
    if sizes.get("conversations.jsonl.gz", 0) > cap // 4:
        out.append(f"conversations are {sizes['conversations.jsonl.gz']} bytes: lower "
                   f"--transcript-sample (now {transcript_sample}) or narrow --key-arm")
    per_task: dict[str, int] = {}
    for row in files["decisions.jsonl.gz"]:
        per_task[row["task"]] = per_task.get(row["task"], 0) + 1
    if len(per_task) > 1:
        out.append("export one shard per task with --task (decision rows per task: "
                   + ", ".join(f"{t}={n}" for t, n in sorted(per_task.items())) + ")")
    if sizes:
        biggest = max(sizes, key=sizes.get)
        out.append(f"largest file: {biggest} ({sizes[biggest]} bytes)")
    return out


def export(run: Path, out: Path, *, cap_bytes: int | None = None,
           tasks: list[str] | None = None, transcript_sample: int = 500,
           transcript_seed: int = 0, key_arms: list[str] | None = None) -> dict[str, Any]:
    run, out = Path(run), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; export into a fresh directory")
    if not (run / "plan.json").is_file():
        raise ExportRefused("plan.json missing from the run directory")
    from slh.xp.partner import plan as pp

    plan = pp.verify(pc.read_json(run / "plan.json"))
    registered = int(plan["limits"]["export_cap_bytes"])
    if cap_bytes is None:
        cap_bytes = registered
    if cap_bytes > registered:
        raise ExportRefused(f"cap {cap_bytes} exceeds the registered cap {registered}")
    shard = sorted(tasks) if tasks else sorted(plan["tasks"])
    if set(shard) - set(plan["tasks"]):
        raise ExportRefused(f"unknown task in {shard}")
    files = collect(run, tasks=set(shard), transcript_sample=transcript_sample,
                    transcript_seed=transcript_seed, key_arms=set(key_arms) if key_arms else None)
    blobs: dict[str, tuple[bytes, int | None]] = {}
    for name, rows in files.items():
        raw, n = pc.jsonl_bytes(rows)
        blobs[name] = (pc.gz_bytes(raw), n)
    env = sc.check_object("env.json", pc.read_json(run / "env.json"), sc.ENV_FIELDS)
    pc.assert_no_gold(env, "env.json")
    blobs["env.json"] = ((json.dumps(env, sort_keys=True, indent=1) + "\n").encode(), None)
    hashes = sc.check_object("hashes.json", pc.read_json(run / "hashes.json"), sc.HASH_FIELDS)
    meta = {name: {"sha256": pc.sha_bytes(data), "bytes": len(data), "rows": n}
            for name, (data, n) in sorted(blobs.items())}
    manifest = {"schema": EXPORT_SCHEMA, "created_by": "slh.xp.partner.export",
                "plan_id": plan["plan_id"], "shard": {"tasks": shard}, "files": meta,
                "hashes": hashes, "cap_bytes": cap_bytes,
                "transcripts": {"sample": transcript_sample, "seed": transcript_seed,
                                "key_arms": sorted(key_arms) if key_arms else None}}
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=1) + "\n").encode()
    total = sum(f["bytes"] for f in meta.values()) + len(manifest_bytes)
    if total > cap_bytes:
        sizes = {k: v["bytes"] for k, v in meta.items()}
        raise ExportTooLarge({"total_bytes": total, "cap_bytes": cap_bytes, "files": sizes,
                              "suggestions": _suggestions(sizes, files, cap_bytes,
                                                          transcript_sample)})
    out.mkdir(parents=True, exist_ok=True)
    for name, (data, _) in blobs.items():
        (out / name).write_bytes(data)
    (out / "manifest.json").write_bytes(manifest_bytes)
    return {**manifest, "total_bytes": total}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.partner.export")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cap-bytes", type=int, help="at most the registered cap")
    ap.add_argument("--task", action="append")
    ap.add_argument("--transcript-sample", type=int, default=500)
    ap.add_argument("--transcript-seed", type=int, default=0)
    ap.add_argument("--key-arm", action="append")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        man = export(args.run, args.out, cap_bytes=args.cap_bytes, tasks=args.task,
                     transcript_sample=args.transcript_sample,
                     transcript_seed=args.transcript_seed, key_arms=args.key_arm)
    except ExportTooLarge as exc:
        print(json.dumps({"status": "refused", **exc.report}, indent=1))
        return 2
    print(json.dumps({"status": "export ok", "total_bytes": man["total_bytes"],
                      "shard": man["shard"],
                      "files": {k: v["rows"] for k, v in man["files"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
