"""Export of a phase-2 run directory: deterministic gz shards with a hashed manifest.

    python -m slh.xp.phase2.export --run runs/p2 --out runs/p2/export [--shard-bytes N]

One row per job (scenario, arm, run): its final record as the run resolved it
(first valid conversation; else the last invalid one once its attempts are used
up; pending jobs are listed in the manifest's coverage and have no row), with
the full message list, usage, timing, the prompt and tool hashes and the
attempt history (validity and error codes only). No reward or grading field
leaves: nothing was graded at the run, and every row is checked for answer
fields. Files::

    conversations-00000.jsonl.gz ...   rows sorted by (scenario, run, arm)
    blobs.jsonl.gz                     every system prompt and tool list a row names
    run.jsonl.gz                       identity, arm prompts, job plan, env, halts,
                                       spend summary, worker statuses
    openings.jsonl.gz                  (pinned-opening runs only) the stored simulator
                                       openings and which job generated / replayed each
    consent-verdicts.jsonl.gz          (consent-chain runs only) every public verdict row
                                       of the step-4 guard chain, all attempts; each
                                       conversation row also carries its own
                                       (``consent_verdicts``)
    manifest.json                      sha256, bytes and rows of every file

Bytes are a pure function of the run directory (sorted rows, canonical JSON,
gzip with mtime 0), so a re-export is byte-identical. Every file stays below
the shard cap (default 95,000,000 bytes; never above 100,000,000): the row
count per conversation shard halves until each shard fits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from slh.xp.partner import common as pc
from slh.xp.phase2 import openings as OP
from slh.xp.phase2 import run as R

EXPORT_SCHEMA = "xp-phase2-export-v1"
ROW_SCHEMA = "xp-phase2-conversation-v1"
MAX_SHARD_BYTES = 100_000_000
DEFAULT_SHARD_BYTES = 95_000_000
START_ROWS = 4096
FORBIDDEN_KEYS = frozenset({"reward", "reward_breakdown", "expected_writes",
                            "expected_db_sha256", "evaluation_criteria", "nl_assertions"})
EPISODE_FIELDS = ("episode_id", "seed", "messages", "termination_reason", "usage", "usd",
                  "wall_s", "harness_version", "execution_id", "agent_system_prompt_sha256",
                  "tools_sha256", "capture_level", "base_arm", "serving_window_id",
                  "guard_events", "error")


class ExportRefused(ValueError):
    pass


def _forbidden(value: Any, path: str = "") -> str | None:
    if isinstance(value, dict):
        for k, v in value.items():
            if k in FORBIDDEN_KEYS:
                return f"{path}.{k}"
            hit = _forbidden(v, f"{path}.{k}")
            if hit:
                return hit
    elif isinstance(value, list):
        for i, v in enumerate(value):
            hit = _forbidden(v, f"{path}[{i}]")
            if hit:
                return hit
    return None


def conversation_rows(run: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identity = pc.read_json(run / "identity.json")
    blocks = [[R.Job(**j) for j in b] for b in pc.read_json(run / "jobs.json")["blocks"]]
    jobs = R.flat(blocks)
    recs = R.records(run)
    resolved = R.resolve(jobs, recs)
    history: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        history.setdefault(r["key"], []).append({
            "worker": r["worker"], "valid": r["valid"], "halted": r["halted"],
            "error": (r["outcome"].get("error") or {}).get("code")})
    rows = []
    for job in jobs:
        res = resolved[job.key]
        if res["final"] is None:
            continue
        ep = res["final"]["episode"]
        row = {"schema": ROW_SCHEMA, "key": job.key, "scenario_id": job.scenario_id,
               "arm": job.arm, "kind": job.kind, "run": job.run, "status": res["status"],
               "attempts": res["attempts"], "history": history.get(job.key, []),
               "valid": res["final"]["valid"], "worker": res["final"]["worker"],
               **{k: ep[k] for k in EPISODE_FIELDS if k in ep}}
        if "consent_verdicts" in res["final"]:  # consent-chain runs only
            row["consent_verdicts"] = res["final"]["consent_verdicts"]
        if row.get("seed") != job.seed:
            raise ExportRefused(f"{job.key}: recorded seed {row.get('seed')} is not the "
                                f"planned {job.seed}")
        hit = _forbidden(row)
        if hit:
            raise ExportRefused(f"{job.key}: grading field {hit} in an export row")
        pc.assert_no_gold(row, job.key)
        rows.append(row)
    rows.sort(key=lambda r: (r["scenario_id"], r["run"], r["arm"]))
    return rows, {"identity": identity, "coverage": R.coverage(resolved),
                  "extra_keys": resolved.get("__extra__", {}).get("keys", [])}


def _blobs(run: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from slh.capture import BlobStore

    store = BlobStore(run / "blobs")
    shas = sorted({r[k] for r in rows for k in ("agent_system_prompt_sha256", "tools_sha256")
                   if r.get(k)})
    return [{"sha256": s, "value": store.get(s)} for s in shas]


def _run_rows(run: Path, meta: dict[str, Any]) -> list[dict[str, Any]]:
    from slh.ledger import SharedLedger

    out = [{"name": "identity", "value": meta["identity"]},
           {"name": "coverage", "value": meta["coverage"]},
           {"name": "extra_keys", "value": meta["extra_keys"]},
           {"name": "arm_prompts", "value": pc.read_json(run / "arm-prompts.json")},
           {"name": "jobs", "value": pc.read_json(run / "jobs.json")}]
    # Records that may carry provider text are redacted again here (defence in
    # depth: they are written sanitized); identity, plan and config are ours.
    for wdir in R._worker_dirs(run):
        status = wdir / "status.json"
        if status.is_file():
            out.append({"name": f"worker:{wdir.name}",
                        "value": R.redact_public(pc.read_json(status))})
        for h in R._jsonl(wdir / "halts.jsonl"):
            out.append({"name": f"halt:{wdir.name}", "value": R.redact_public(h)})
    for i, e in enumerate(R._jsonl(run / "logs" / "env.jsonl")):
        out.append({"name": f"env:{i}", "value": R.redact_public(e)})
    if (run / "ledger.jsonl").is_file():
        snap = SharedLedger(run / "ledger.jsonl", 1e9, owner="export", session=-1,
                            create=False).snapshot()
        out.append({"name": "spend", "value": R.redact_public(snap)})
    for i, f in enumerate(sorted((run / "config").glob("*.yaml"))):
        out.append({"name": f"config:{f.name}", "value": f.read_text(encoding="utf-8")})
    return out


def _gz(rows: list[dict[str, Any]]) -> tuple[bytes, int]:
    raw, n = pc.jsonl_bytes(rows)
    return pc.gz_bytes(raw), n


def shards(rows: list[dict[str, Any]], cap: int) -> list[tuple[bytes, int]]:
    per = START_ROWS
    while True:
        parts = [_gz(rows[i:i + per]) for i in range(0, max(len(rows), 1), per)]
        if all(len(b) <= cap for b, _ in parts):
            return parts
        if per == 1:
            raise ExportRefused("one conversation is larger than the shard cap")
        per //= 2


def export(run: Path, out: Path, *, shard_bytes: int = DEFAULT_SHARD_BYTES) -> dict[str, Any]:
    run, out = Path(run), Path(out)
    if shard_bytes > MAX_SHARD_BYTES:
        raise ExportRefused(f"shard cap {shard_bytes} is above {MAX_SHARD_BYTES}")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; export into a fresh directory")
    rows, meta = conversation_rows(run)
    blobs: dict[str, tuple[bytes, int]] = {}
    for i, part in enumerate(shards(rows, shard_bytes)):
        blobs[f"conversations-{i:05d}.jsonl.gz"] = part
    blobs["blobs.jsonl.gz"] = _gz(_blobs(run, rows))
    blobs["run.jsonl.gz"] = _gz(_run_rows(run, meta))
    if meta["identity"].get("opening_pin"):
        # Pinned simulator openings (slh.xp.phase2.openings): cache entries and uses.
        rows_o = OP.export_rows(run)
        for i, o in enumerate(rows_o):
            pc.assert_no_gold(o, f"openings[{i}]")
        blobs[OP.EXPORT_FILE] = _gz(rows_o)
    if meta["identity"].get("consent_chain"):
        # stable sort: rows of one attempt keep their logged order
        rows_v = sorted(R.consent_verdicts(run), key=lambda v: (
            str(v.get("key")), str(v.get("worker")), str(v.get("attempt_log"))))
        for i, v in enumerate(rows_v):
            pc.assert_no_gold(v, f"consent_verdicts[{i}]")
            hit = _forbidden(v)
            if hit:
                raise ExportRefused(f"consent_verdicts[{i}]: grading field {hit}")
        blobs[R.CONSENT_EXPORT] = _gz(rows_v)
    for name, (data, _) in blobs.items():
        if len(data) > shard_bytes:
            raise ExportRefused(f"{name} is {len(data)} bytes, above the shard cap")
    files = {name: {"sha256": pc.sha_bytes(data), "bytes": len(data), "rows": n}
             for name, (data, n) in sorted(blobs.items())}
    manifest = {"schema": EXPORT_SCHEMA, "created_by": "slh.xp.phase2.export",
                "identity": meta["identity"], "coverage": meta["coverage"],
                "shard_bytes": shard_bytes, "files": files}
    body = (json.dumps(manifest, sort_keys=True, indent=1) + "\n").encode()
    out.mkdir(parents=True, exist_ok=True)
    for name, (data, _) in blobs.items():
        (out / name).write_bytes(data)
    (out / "manifest.json").write_bytes(body)
    return {**manifest, "total_bytes": sum(f["bytes"] for f in files.values()) + len(body)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.phase2.export")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        man = export(args.run, args.out, shard_bytes=args.shard_bytes)
    except (ExportRefused, FileExistsError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, indent=1))
        return 2
    print(json.dumps({"status": "export ok", "total_bytes": man["total_bytes"],
                      "coverage": man["coverage"],
                      "files": {k: v["bytes"] for k, v in man["files"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
