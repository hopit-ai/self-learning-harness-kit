"""Partner side: run, score and export BFCL multi-turn cells on the locally served 9B.

    python -m slh.xp.bfcl.runner check                                      # offline readiness
    python -m slh.xp.bfcl.runner run    --package PKG --out runs/h1-dry --dry-run
    python -m slh.xp.bfcl.runner run    --package PKG --endpoint endpoint.yaml --out runs/h1-smoke --smoke
    python -m slh.xp.bfcl.runner run    --package PKG --endpoint endpoint.yaml --out runs/h1
    python -m slh.xp.bfcl.runner score  --package PKG --out runs/h1
    python -m slh.xp.bfcl.runner export --package PKG --run runs/h1 --out export-h1

``run`` starts one child process per cell (``--parallel-cells``, default 4), each running
the cell's instances on ``--threads`` threads (default 16) through the official BFCL
driver and loop (``slh.xp.bfcl.handler``). One process per cell is required: the
official backends keep their state in module globals keyed by model name and entry id.
Results are appended one line per finished instance; a rerun resumes. **Resume** holds an
exclusive lock on the cell folder, cuts a torn last line (a cell killed mid-write) after
copying it to ``results.jsonl.torn``, then appends; a malformed line before the last one
refuses (exit 4) rather than guessing. ``score`` and ``export`` skip a torn last line and
refuse the same malformed lines. A connection,
authentication or model-not-found error halts the cell without a row (``halt.json``,
exit 3); any other error becomes the official "Error during inference" result (a failed
episode, as upstream).

**Feasibility barrier.** ``run`` first runs the plan's feasibility cells alone (to
completion), then evaluates the pre-declared feasibility go / no-go on their results
(``feasibility_gate``: every feasibility instance finished, no halt, unparsed tool-call
steps < 5% of model steps, inference errors < 2% of episodes) and writes
``feasibility-gate.json``. Only on GO does it start the discovery cells; on NO-GO it
starts nothing more and exits 5 (send the run's export and logs; do not rerun). A rerun
recomputes the same verdict from the same rows.

**Interruption.** Ctrl-C or SIGTERM stops every cell process (terminate, then kill after
``STOP_GRACE_S``) before ``run`` / ``score`` exits (130); finished instances are kept and
the same command resumes.

``score`` scores every finished cell with the official checker in a fresh child process
per cell (CPU only). ``export`` writes our trace format (``slh.xp.bfcl.traces``) for
every cell, plus ``manifest.json`` (plan hash, cells, counts), ``env.json`` and
``profile.json`` (tokens, steps, latency, wall time per cell: the throughput profile for
the GPU-hour estimates) and, under ``run/``, the run's own records unchanged
(``identity.json``, ``feasibility-gate.json``, each cell's ``status.json`` / ``halt.json`` /
torn-tail backup: the recovery evidence of a partial or failed run).

``--dry-run`` replaces the model with a fixed local fake (it always answers in text, so
every episode ends without calls and fails); it exercises the official loop, the
backends, the scorer and the export with no server and no GPU. ``--smoke`` runs the first
two instances of each cell. Startup refuses (exit 4) when ``plan.json`` differs from its
hash, when a registered code file differs, or when the run directory belongs to another
run identity.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from slh.xp.bfcl import data, traces
from slh.xp.bfcl import plan as P
from slh.xp.partner.common import REPO

EXIT_HALT, EXIT_REFUSED, EXIT_NOGO, EXIT_INTERRUPTED = 3, 4, 5, 130
# Pre-declared feasibility go / no-go (docs/xp/bfcl-plan.md, section 8: the conditions
# measurable on the feasibility cell; the success band is read on discovery, our side).
FEASIBILITY_GATE = {"max_unparsed_step_share": 0.05, "max_inference_error_share": 0.02}
STOP_GRACE_S = 10.0
HALT_MARKERS = ("APIConnectionError", "Connection refused", "AuthenticationError",
                "NotFoundError", "PermissionDeniedError", "ConnectError",
                # BFCL records ``f"Error during inference: {str(e)}"``: the OpenAI client's
                # str() carries no class name, so the exact messages are matched too
                # (APIConnectionError / APITimeoutError after its retries, and the
                # token / route / gateway statuses); vLLM's 400 / 500 stay episode results.
                "Error during inference: Connection error.",
                "Error during inference: Request timed out.",
                "Error during inference: Error code: 401",
                "Error during inference: Error code: 403",
                "Error during inference: Error code: 404",
                "Error during inference: Error code: 502",
                "Error during inference: Error code: 503",
                "Error during inference: Error code: 504")
SMOKE_UNITS = 2
EXPORT_CAP_BYTES = 2_000_000_000  # uncompressed; the partner sends it as .tar.gz
ENDPOINT_FIELDS = ("base_url", "model", "api_key_env", "model_revision", "vllm_version",
                   "serving_args", "replicas")


class Refused(RuntimeError):
    pass


class Halt(RuntimeError):
    pass


# --------------------------------------------------------------------------- package


def load_package(package: Path) -> dict[str, Any]:
    plan = json.loads((Path(package) / "plan.json").read_text(encoding="utf-8"))
    if plan.get("schema") != P.PLAN_SCHEMA:
        raise Refused("plan.json: unknown schema")
    try:
        P.check_seal(plan, "plan.json")
    except P.PlanRefused as exc:
        raise Refused(str(exc)) from exc
    bad = [rel for rel, h in plan["code"].items()
           if not (REPO / rel).is_file() or P.vendor.sha_file(REPO / rel) != h]
    if bad:
        raise Refused(f"registered code differs: {bad[:5]}")
    return plan


def load_endpoint(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    import yaml

    ep = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if ep.get("model") != P.SERVED_MODEL:
        raise Refused(f"endpoint model must be {P.SERVED_MODEL!r}")
    if not ep.get("base_url"):
        raise Refused("endpoint base_url is empty")
    from slh.xp.partner.common import serving_sampling_problems

    problems = serving_sampling_problems(ep.get("serving_args"))
    if problems:  # decoding is sent per request (temperature 0.7, top-p 0.8)
        raise Refused("; ".join(problems))
    return ep


def identity(plan: dict[str, Any], *, mode: str, endpoint: dict[str, Any]) -> dict[str, Any]:
    ep = {k: endpoint.get(k) for k in ENDPOINT_FIELDS}
    return {"plan_sha256": plan["sha256"], "mode": mode,
            "endpoint_sha256": None if mode in ("dry", "oracle") else P.sha_text(P._canon(ep))}


def claim(out: Path, ident: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "identity.json"
    if path.is_file():
        if json.loads(path.read_text()) != ident:
            raise Refused(f"{out} belongs to another run identity; use a new --out")
    else:
        path.write_text(json.dumps(ident, sort_keys=True))


def generation_of(plan: dict[str, Any], endpoint: dict[str, Any] | None) -> dict[str, Any]:
    """The output-length rule of the plan (B-P1) with ``context_len`` the served
    ``--max-model-len`` when the endpoint records it (as BFCL reads the model's own
    context length). A plan built before B-P1 gets the same rule (handler defaults)."""
    from slh.xp.bfcl import handler as H

    gen = dict(plan.get("generation") or {"max_output_tokens": H.MAX_OUTPUT_TOKENS,
                                           "context_len": H.CONTEXT_LEN})
    raw = (endpoint or {}).get("serving_args") or []
    if isinstance(raw, str):
        import shlex

        raw = shlex.split(raw)
    args = [str(a) for a in raw]
    for k, a in enumerate(args):
        if a == "--max-model-len" and k + 1 < len(args):
            gen["context_len"] = int(args[k + 1])
        elif a.startswith("--max-model-len="):
            gen["context_len"] = int(a.split("=", 1)[1])
    return gen


def units_of(cell: dict[str, Any], smoke: bool) -> list[dict[str, Any]]:
    return cell["units"][:SMOKE_UNITS] if smoke else cell["units"]


# --------------------------------------------------------------------------- fake model


class _FakeCompletions:
    def create(self, **kwargs: Any) -> Any:
        from openai.types.chat import ChatCompletion

        n = sum(len(str(m)) for m in kwargs.get("messages", []))
        return ChatCompletion.model_validate({
            "id": "dry", "object": "chat.completion", "created": 0, "model": kwargs["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": "I have handled the request."}}],
            "usage": {"prompt_tokens": n // 4, "completion_tokens": 6,
                      "total_tokens": n // 4 + 6}})


class FakeClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


def _call_of(text: str, params: dict[str, list[str]]) -> dict[str, Any]:
    """``f(1, b='x')`` (a possible-answer call string) -> name and JSON arguments;
    positional arguments are named in the function document's parameter order."""
    import ast

    node = ast.parse(text.strip(), mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"unsupported answer call: {text!r}")
    order = params.get(node.func.id, [])
    if len(node.args) > len(order):
        raise ValueError(f"cannot name the positional arguments of {text!r}")
    args = {order[k]: ast.literal_eval(a) for k, a in enumerate(node.args)}
    args.update({kw.arg: ast.literal_eval(kw.value) for kw in node.keywords})
    return {"name": node.func.id, "arguments": json.dumps(args)}


class _OracleCompletions:
    """Replays the official answer of the current turn as tool calls, then replies.
    Checks the adapter end to end (request, parsing, execution, scoring) with no model:
    every replayed episode must score as a success."""

    def __init__(self, answers: dict[str, Any], entries: dict[str, Any]) -> None:
        self.answers = answers
        self.params = {i: {f["name"]: list((f.get("parameters") or {}).get("properties", {}))
                           for f in e.get("function", [])} for i, e in entries.items()}
        self.local = threading.local()

    def create(self, **kwargs: Any) -> Any:
        from openai.types.chat import ChatCompletion

        msgs = kwargs.get("messages", [])

        def role(m: Any) -> str:
            return m["role"] if isinstance(m, dict) else m.role

        turn = sum(1 for m in msgs if role(m) == "user") - 1
        gt = self.answers[self.local.instance]["ground_truth"]
        params = self.params[self.local.instance]
        message: dict[str, Any] = {"role": "assistant", "content": "Done."}
        if role(msgs[-1]) == "user" and 0 <= turn < len(gt) and gt[turn]:
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_{turn}_{k}", "type": "function", "function": _call_of(c, params)}
                for k, c in enumerate(gt[turn])]}
        return ChatCompletion.model_validate({
            "id": "oracle", "object": "chat.completion", "created": 0,
            "model": kwargs["model"], "choices": [{"index": 0, "message": message,
                                                   "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


class OracleClient:
    def __init__(self, answers: dict[str, Any], entries: dict[str, Any]) -> None:
        self.completions = _OracleCompletions(answers, entries)
        self.chat = type("Chat", (), {"completions": self.completions})()


# --------------------------------------------------------------------------- one cell


def _append(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(row, sort_keys=True, default=str) + "\n"
    with lock, path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


class Malformed(Refused):
    pass


def read_results(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    """The completed rows of ``results.jsonl`` and its torn tail (the bytes after the last
    newline: an interrupted append; ``b""`` when the file ends with a newline). Every
    newline-terminated line must be a JSON row with an ``id``; anything else refuses
    (``Malformed``) instead of guessing which rows are good."""
    if not path.is_file():
        return [], b""
    raw = path.read_bytes()
    cut = raw.rfind(b"\n") + 1
    rows = []
    for k, line in enumerate(raw[:cut].split(b"\n")[:-1], start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            row["id"]
        except (ValueError, KeyError, TypeError):
            raise Malformed(f"{path}: line {k} is not a complete result row (only the last "
                            "line can be torn); do not edit the file, pack the run folder "
                            "and tell us") from None
        rows.append(row)
    return rows, raw[cut:]


def repair_tail(path: Path) -> dict[str, Any] | None:
    """Make ``results.jsonl`` safe to append to: a torn last line (no newline: the cell
    was killed mid-write) is copied to ``results.jsonl.torn`` (one backup per repair,
    numbered) and cut off, so the resumed row starts on its own line; completed rows are
    kept byte for byte. A last line that is a whole row without its newline gets the
    newline instead. Malformed non-final lines refuse (``read_results``). Call it with
    the cell lock held."""
    tail = read_results(path)[1]
    if not tail:
        return None
    try:
        row = json.loads(tail)
        whole = isinstance(row, dict) and "id" in row
    except ValueError:
        whole = False
    if whole:
        with path.open("ab") as f:
            f.write(b"\n")
            f.flush()
            os.fsync(f.fileno())
        return {"kept": row["id"], "bytes": len(tail)}
    k = 0
    while (backup := path.with_name(path.name + (".torn" if k == 0 else f".torn.{k}"))
           ).exists():
        k += 1
    backup.write_bytes(tail)
    size = path.stat().st_size
    with path.open("r+b") as f:
        f.truncate(size - len(tail))
        f.flush()
        os.fsync(f.fileno())
    return {"cut_bytes": len(tail), "backup": backup.name}


def _done(path: Path) -> set[str]:
    return {r["id"] for r in read_results(path)[0]}


class _CellLock:
    """An exclusive lock on the cell folder for the whole run of the cell: a second
    process on the same cell (two ``run`` commands on one ``--out``) refuses."""

    def __init__(self, cell_dir: Path) -> None:
        self.path = cell_dir / ".cell.lock"
        self.f: Any = None

    def __enter__(self) -> _CellLock:  # noqa: PYI034
        import fcntl

        self.f = self.path.open("a")
        try:
            fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.f.close()
            raise Refused(f"{self.path.parent} is being run by another process") from None
        return self

    def __exit__(self, *exc: object) -> None:
        import fcntl

        fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
        self.f.close()


def _bfcl_env(cell_dir: Path) -> None:
    os.environ["BFCL_PROJECT_ROOT"] = str(cell_dir / "bfcl_root")
    (cell_dir / "bfcl_root").mkdir(parents=True, exist_ok=True)


def run_cell(plan: dict[str, Any], cell_id: str, cell_dir: Path, *, endpoint: dict[str, Any],
             dry: bool, smoke: bool, threads: int, oracle: bool = False) -> int:
    from slh.xp.bfcl import handler as H

    cell = next(c for c in plan["cells"] if c["cell_id"] == cell_id)
    cell_dir.mkdir(parents=True, exist_ok=True)
    _bfcl_env(cell_dir)
    api_key = os.environ.get(endpoint.get("api_key_env") or "", "") or "EMPTY"
    units = units_of(cell, smoke)
    entries: dict[str, dict[str, Any]] = {}
    answers: dict[str, dict[str, Any]] = {}
    for cat in sorted({data.parse_instance(u["instance"])[1] for u in units}):
        prompts, ans = H.load_entries(cat)
        entries.update(prompts)
        answers.update(ans)
    client = (OracleClient(answers, entries) if oracle else FakeClient()) if dry else None
    h = H.build_handler(served_model=plan["served_model"], registry_name=f"slh-{cell_id}",
                        base_url=endpoint.get("base_url") or "http://127.0.0.1:9/v1",
                        api_key=api_key, temperature=plan["decoding"]["temperature"],
                        sampling=plan["decoding"].get("sampling"),
                        client=client, generation=generation_of(plan, endpoint))
    h.slh_notes_by_id = {u["instance"]: plan["notes"][u["notes_sha256"]]
                         for u in units if u.get("notes_sha256")}
    results = cell_dir / "results.jsonl"
    with _CellLock(cell_dir):
        repaired = repair_tail(results)
        if repaired:
            print(json.dumps({"torn_tail": repaired}), file=sys.stderr)
        return _run_units(cell_id, cell_dir, results, units, h, entries, client,
                          threads, repaired)


def _run_units(cell_id: str, cell_dir: Path, results: Path, units: list[dict[str, Any]],
               h: Any, entries: dict[str, Any], client: Any, threads: int,
               repaired: dict[str, Any] | None = None) -> int:
    from slh.xp.bfcl import handler as H

    done = _done(results)
    todo = [u for u in units if u["instance"] not in done]
    lock = threading.Lock()
    halted: list[str] = []
    t0 = time.time()

    def one(u: dict[str, Any]) -> None:
        if halted:
            return
        start = time.time()
        if isinstance(client, OracleClient):
            client.completions.local.instance = u["instance"]
        row = H.run_entry(h, entries[u["instance"]])
        if isinstance(row.get("result"), str) and any(m in row["result"] for m in HALT_MARKERS):
            halted.append(row["result"][:500])
            return
        row["slh_wall_s"] = round(time.time() - start, 3)
        row["slh_cell_id"] = cell_id
        _append(results, row, lock)

    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        for f in as_completed([pool.submit(one, u) for u in todo]):
            f.result()
    status = {"cell_id": cell_id, "units": len(units), "done": len(_done(results)),
              "wall_s": round(time.time() - t0, 1), "halted": bool(halted)}
    if repaired:
        status["torn_tail"] = repaired
    (cell_dir / "status.json").write_text(json.dumps(status, sort_keys=True))
    if halted:
        (cell_dir / "halt.json").write_text(json.dumps({"reason": halted[0]}))
        return EXIT_HALT
    return 0


def _json_key(k: Any) -> str:
    """The text ``json`` writes for a mapping key (str, int, float, bool, None)."""
    if isinstance(k, str):
        return k
    if isinstance(k, bool) or k is None:
        return {True: "true", False: "false", None: "null"}[k]
    if isinstance(k, int):
        return int.__repr__(k)
    if isinstance(k, float):
        return json.dumps(k)
    return str(k)  # json itself refuses other key types


def presort(obj: Any) -> Any:
    """``obj`` with every mapping rebuilt in sorted key order: natural order where the keys
    are mutually orderable (what ``sort_keys`` does), else by the JSON key text, then the
    key's type name. Nothing is dropped or converted."""
    if isinstance(obj, dict):
        keys = list(obj)
        try:
            keys = sorted(keys)
        except TypeError:
            keys = sorted(keys, key=lambda k: (_json_key(k), type(k).__name__))
        return {k: presort(obj[k]) for k in keys}
    if isinstance(obj, (list, tuple)):
        return [presort(v) for v in obj]
    return obj


def dumps_sorted(obj: Any, **kw: Any) -> str:
    """``json.dumps(obj, sort_keys=True, **kw)``, or, when that raises ``TypeError`` (a
    mapping with keys of mixed types, e.g. an official state-mismatch verdict carrying a
    backend's ``orders`` keyed by int and str ids), the same object in ``presort`` key
    order. Byte-identical whenever the plain call succeeds (deviation B-D8)."""
    try:
        return json.dumps(obj, sort_keys=True, **kw)
    except TypeError:
        return json.dumps(presort(obj), sort_keys=False, **kw)


def score_cell(plan: dict[str, Any], cell_id: str, cell_dir: Path) -> int:
    from slh.xp.bfcl import handler as H

    _bfcl_env(cell_dir)
    h = H.build_handler(served_model=plan["served_model"], registry_name=f"slh-{cell_id}",
                        base_url="http://127.0.0.1:9/v1", temperature=0.001,
                        client=FakeClient())
    rows = read_results(cell_dir / "results.jsonl")[0]  # a torn last line is not a row
    cats = sorted({data.parse_instance(r["id"])[1] for r in rows})
    prompts: dict[str, Any] = {}
    answers: dict[str, Any] = {}
    for cat in cats:
        p, a = H.load_entries(cat)
        prompts.update(p)
        answers.update(a)
    out = cell_dir / "verdicts.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: r["id"]):
            v = H.score_entry(h, r, prompts[r["id"]], answers[r["id"]],
                              model_name=h.registry_dir_name)
            f.write(dumps_sorted({"id": r["id"], "verdict": v}, default=str) + "\n")
    return 0


# --------------------------------------------------------------------------- orchestration


def episode_profile(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Episodes, model steps, unparsed tool-call steps (text that holds a tool call the
    parser did not return) and inference errors (the official string result)."""
    unparsed = inference_errors = steps = 0
    for r in rows:
        inference_errors += isinstance(r.get("result"), str)
        for c in r.get("slh_calls") or []:
            steps += 1
            msg = ((c.get("response") or {}).get("choices") or [{}])[0].get("message") or {}
            text = str(msg.get("content") or "")
            if not msg.get("tool_calls") and ("<tool_call>" in text or "<function=" in text):
                unparsed += 1
    return {"episodes": len(rows), "model_steps": steps, "unparsed_tool_call_steps": unparsed,
            "inference_errors": inference_errors}


def _rows(path: Path) -> list[dict[str, Any]]:
    return read_results(path)[0]


def feasibility_gate(plan: dict[str, Any], out: Path, *, smoke: bool) -> dict[str, Any]:
    """The pre-declared go / no-go on the feasibility cells' results (no scorer, no keys:
    plumbing conditions only). Written to ``out/feasibility-gate.json``."""
    cells = [c for c in plan["cells"] if c.get("purpose") == "feasibility"]
    reasons: list[str] = []
    per_cell = {}
    for cell in cells:
        cdir = out / "cells" / cell["cell_id"]
        want = {u["instance"] for u in units_of(cell, smoke)}
        rows = [r for r in _rows(cdir / "results.jsonl") if r.get("id") in want]
        prof = episode_profile(rows)
        unparsed = prof["unparsed_tool_call_steps"] / max(1, prof["model_steps"])
        errors = prof["inference_errors"] / max(1, prof["episodes"])
        per_cell[cell["cell_id"]] = {**prof, "expected": len(want),
                                     "unparsed_step_share": round(unparsed, 4),
                                     "inference_error_share": round(errors, 4)}
        if {r["id"] for r in rows} != want:
            reasons.append(f"{cell['cell_id']}: {len(want - {r['id'] for r in rows})} of "
                           f"{len(want)} instances not finished")
        if (cdir / "halt.json").is_file():
            reasons.append(f"{cell['cell_id']}: halted (halt.json)")
        if unparsed >= FEASIBILITY_GATE["max_unparsed_step_share"]:
            reasons.append(f"{cell['cell_id']}: unparsed tool-call steps {unparsed:.1%} "
                           f">= {FEASIBILITY_GATE['max_unparsed_step_share']:.0%}")
        if errors >= FEASIBILITY_GATE["max_inference_error_share"]:
            reasons.append(f"{cell['cell_id']}: inference errors {errors:.1%} >= "
                           f"{FEASIBILITY_GATE['max_inference_error_share']:.0%}")
    gate = {"go": not reasons, "reasons": reasons, "rule": FEASIBILITY_GATE,
            "cells": per_cell, "smoke": smoke, "plan_sha256": plan["sha256"]}
    (out / "feasibility-gate.json").write_text(json.dumps(gate, indent=1, sort_keys=True))
    return gate


def _stop(procs: list[tuple[str, subprocess.Popen]]) -> None:
    for _, p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.time() + STOP_GRACE_S
    for _, p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def _dispatch(cells: list[str], package: Path, out: Path, cmd: str, *, extra: list[str],
              parallel: int, smoke: bool) -> int:
    """Run ``cells`` (``parallel`` at a time) to completion; on Ctrl-C / SIGTERM stop every
    child before returning."""
    procs: list[tuple[str, subprocess.Popen]] = []
    worst = 0
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(cells)
    try:
        while pending or procs:
            while pending and len(procs) < max(1, parallel):
                cid = pending.pop(0)
                if cmd == "score" and not (out / "cells" / cid / "results.jsonl").is_file():
                    continue
                argv = [sys.executable, "-m", "slh.xp.bfcl.runner", f"_{cmd}", "--package",
                        str(package), "--out", str(out), "--cell", cid, *extra]
                if smoke:
                    argv.append("--smoke")
                with (log_dir / f"{cmd}-{cid}.log").open("a") as log:
                    # own session: the parent alone decides when a cell stops
                    procs.append((cid, subprocess.Popen(argv, stdout=log,
                                                        stderr=subprocess.STDOUT, cwd=REPO,
                                                        start_new_session=True)))
            time.sleep(0.5)
            for cid, p in list(procs):
                rc = p.poll()
                if rc is None:
                    continue
                procs.remove((cid, p))
                worst = max(worst, rc)
                if rc == EXIT_HALT:
                    pending.clear()  # stop dispatching; running cells finish
    except KeyboardInterrupt:
        _stop(procs)
        print(f"interrupted: stopped {len(procs)} cell process(es); finished instances are "
              "kept, rerun the same command to resume", file=sys.stderr)
        return EXIT_INTERRUPTED
    return worst


def _children(plan: dict[str, Any], package: Path, out: Path, cmd: str, *, extra: list[str],
              parallel: int, smoke: bool) -> int:
    cells = [c["cell_id"] for c in plan["cells"]]
    order = plan.get("order") or cells
    cells = sorted(cells, key=lambda c: order.index(c) if c in order else len(order))
    if cmd != "run":
        return _dispatch(cells, package, out, cmd, extra=extra, parallel=parallel, smoke=smoke)
    purpose = {c["cell_id"]: c.get("purpose") for c in plan["cells"]}
    first = [c for c in cells if purpose[c] == "feasibility"]
    rest = [c for c in cells if purpose[c] != "feasibility"]
    rc = _dispatch(first, package, out, cmd, extra=extra, parallel=parallel, smoke=smoke)
    if rc:
        return rc  # halted, refused or interrupted: no discovery cell starts
    if first and rest:
        gate = feasibility_gate(plan, out, smoke=smoke)
        print(json.dumps({"feasibility": "GO" if gate["go"] else "NO-GO",
                          "reasons": gate["reasons"]}), file=sys.stderr)
        if not gate["go"]:
            return EXIT_NOGO
    return _dispatch(rest, package, out, cmd, extra=extra, parallel=parallel, smoke=smoke) \
        if rest else 0


def read_outcomes(export: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for path in sorted(Path(export).glob("cells/*/outcomes.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                out[(o["cell_id"], o["instance"])] = o
    return out


def export(plan: dict[str, Any], run: Path, out: Path, *, cap_bytes: int = EXPORT_CAP_BYTES
           ) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    counts, profile, torn = {}, {}, {}
    records = _copy_run_records(plan, run, out / "run")
    for cell in plan["cells"]:
        cdir = run / "cells" / cell["cell_id"]
        res = cdir / "results.jsonl"
        if not res.is_file():
            continue
        rows, tail = read_results(res)  # a torn last line (killed mid-write) is no row
        if tail:
            torn[cell["cell_id"]] = len(tail)
        verdicts = {}
        if (cdir / "verdicts.jsonl").is_file():
            for line in (cdir / "verdicts.jsonl").read_text().splitlines():
                if line.strip():
                    v = json.loads(line)
                    verdicts[v["id"]] = v["verdict"]
        notes_of = {u["instance"]: plan["notes"].get(u["notes_sha256"] or "", "")
                    for u in cell["units"]}
        per_cell = {**cell, "notes": None}
        # one notes text per instance: export each instance with its own notes
        sub = []
        for r in rows:
            sub.append((r, notes_of.get(r["id"], "")))
        by_notes: dict[str, list[dict[str, Any]]] = {}
        for r, n in sub:
            by_notes.setdefault(n, []).append(r)
        c = {"episodes": 0, "outcomes": 0, "calls": 0}
        for n, rs in by_notes.items():
            got = traces.export_cell({**per_cell, "notes": n}, rs, verdicts,
                                     out / "cells" / cell["cell_id"] / "part" /
                                     (P.sha_text(n)[:12] if n else "b1"))
            c = {k: c[k] + got[k] for k in c}
        counts[cell["cell_id"]] = c
        steps = [sum(len(t) for t in r.get("input_token_count") or []) for r in rows]
        prompt_tok = [sum(sum(t) for t in r.get("input_token_count") or []) for r in rows]
        out_tok = [sum(sum(t) for t in r.get("output_token_count") or []) for r in rows]
        wall = [r.get("slh_wall_s") or 0 for r in rows]
        status = json.loads((cdir / "status.json").read_text()) \
            if (cdir / "status.json").is_file() else {}
        prof = episode_profile(rows)
        profile[cell["cell_id"]] = {
            "unparsed_tool_call_steps": prof["unparsed_tool_call_steps"],
            "inference_errors": prof["inference_errors"], "model_steps": prof["model_steps"],
            "episodes": len(rows), "steps": sum(steps), "prompt_tokens": sum(prompt_tok),
            "completion_tokens": sum(out_tok), "episode_wall_s_sum": round(sum(wall), 1),
            "cell_wall_s": status.get("wall_s")}
    _merge_parts(out)
    env = {"python": sys.version, "platform": platform.platform(),
           "packages": _versions(("bfcl-eval", "openai", "numpy", "pandas", "pydantic"))}
    (out / "env.json").write_text(json.dumps(env, indent=1, sort_keys=True))
    (out / "profile.json").write_text(json.dumps(profile, indent=1, sort_keys=True))
    gate = run / "feasibility-gate.json"
    if gate.is_file():
        (out / "feasibility-gate.json").write_bytes(gate.read_bytes())
    man = {"schema": "xp-bfcl-export-v1", "plan_sha256": plan["sha256"],
           "handoff": plan["handoff"], "counts": counts,
           "identity": json.loads((run / "identity.json").read_text())
           if (run / "identity.json").is_file() else None,
           "run_records": records,
           "halted_cells": sorted(p.parent.name for p in run.glob("cells/*/halt.json")),
           "torn_tails": torn}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if size > cap_bytes:
        raise Refused(f"export is {size} bytes, over the cap {cap_bytes}")
    return man


RUN_RECORDS = ("status.json", "halt.json")


def _copy_run_records(plan: dict[str, Any], run: Path, dest: Path) -> list[str]:
    """The run's own records, unchanged, under ``EXPORT/run/``: ``identity.json``,
    ``feasibility-gate.json`` and per cell ``status.json``, ``halt.json`` and any
    ``results.jsonl.torn*`` backup (also for a cell that halted before its first row), so
    a partial or failed run comes back with its recovery evidence."""
    import shutil

    got = []
    for name in ("identity.json", "feasibility-gate.json"):
        if (run / name).is_file():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run / name, dest / name)
            got.append(name)
    for cell in plan["cells"]:
        cdir = run / "cells" / cell["cell_id"]
        names = [n for n in RUN_RECORDS if (cdir / n).is_file()]
        names += sorted(p.name for p in cdir.glob("results.jsonl.torn*"))
        for n in names:
            (dest / "cells" / cell["cell_id"]).mkdir(parents=True, exist_ok=True)
            shutil.copy2(cdir / n, dest / "cells" / cell["cell_id"] / n)
            got.append(f"cells/{cell['cell_id']}/{n}")
    return got


def _merge_parts(out: Path) -> None:
    """One file set per cell: concatenate the per-notes parts (blobs are shared)."""
    import shutil

    for cdir in sorted((out / "cells").glob("*")):
        parts = sorted((cdir / "part").glob("*")) if (cdir / "part").is_dir() else []
        if not parts:
            continue
        (cdir / "blobs").mkdir(exist_ok=True)
        (cdir / "private").mkdir(exist_ok=True)
        for name in ("episodes.jsonl", "outcomes.jsonl", "calls.jsonl",
                     "private/verdicts.jsonl"):
            with (cdir / name).open("w", encoding="utf-8") as f:
                for part in parts:
                    if (part / name).is_file():
                        f.write((part / name).read_text(encoding="utf-8"))
        for part in parts:
            for b in (part / "blobs").glob("*.json"):
                if not (cdir / "blobs" / b.name).exists():
                    shutil.copy(b, cdir / "blobs" / b.name)
        shutil.rmtree(cdir / "part")


def _versions(names: tuple[str, ...]) -> dict[str, str | None]:
    from importlib import metadata

    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


# --------------------------------------------------------------------------- check


def check() -> dict[str, Any]:
    """Offline readiness: pinned package installed, official modules import, no network."""
    import socket

    from slh.xp.bfcl import vendor

    problems = vendor.installed_problems()
    real = socket.socket.connect

    def refuse(*a: Any, **k: Any) -> None:
        raise OSError("network disabled during check")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    try:
        for mod in ("bfcl_eval._llm_response_generation", "bfcl_eval.eval_checker.eval_runner",
                    "bfcl_eval.model_handler.api_inference.openai_completion"):
            try:
                __import__(mod)
            except Exception as exc:  # noqa: BLE001 - report every import failure
                problems.append(f"import {mod}: {type(exc).__name__}: {exc}"[:300])
    finally:
        socket.socket.connect = real  # type: ignore[method-assign]
    return {"ok": not problems, "problems": problems,
            "packages": _versions(("bfcl-eval", "openai", "numpy"))}


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.bfcl.runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    for name in ("run", "score", "_run", "_score"):
        s = sub.add_parser(name)
        s.add_argument("--package", type=Path, required=True)
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--endpoint", type=Path)
        s.add_argument("--dry-run", action="store_true")
        s.add_argument("--smoke", action="store_true")
        s.add_argument("--oracle", action="store_true",
                       help="with --dry-run: replay the official answers (adapter check)")
        s.add_argument("--threads", type=int, default=16)
        s.add_argument("--parallel-cells", type=int, default=4)
        s.add_argument("--cell")
    e = sub.add_parser("export")
    e.add_argument("--package", type=Path, required=True)
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd in ("run", "score"):
        import signal

        def _term(*_a: Any) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, _term)  # stop the cell processes as on Ctrl-C
    try:
        if args.cmd == "check":
            rep = check()
            print(json.dumps(rep, indent=1))
            return 0 if rep["ok"] else 1
        plan = load_package(args.package)
        if args.cmd == "export":
            print(json.dumps(export(plan, args.run, args.out)["counts"]))
            return 0
        if args.cmd in ("run", "_run") and not args.dry_run and args.endpoint is None:
            raise Refused("--endpoint is required (or --dry-run)")
        endpoint = load_endpoint(args.endpoint)
        if args.oracle and not args.dry_run:
            raise Refused("--oracle needs --dry-run")
        mode = ("oracle" if args.oracle else "dry") if args.dry_run else \
            ("smoke" if args.smoke else "full")
        if args.cmd == "run":
            claim(args.out, identity(plan, mode=mode, endpoint=endpoint))
            extra = ["--threads", str(args.threads)]
            if args.endpoint:
                extra += ["--endpoint", str(args.endpoint)]
            if args.dry_run:
                extra.append("--dry-run")
            if args.oracle:
                extra.append("--oracle")
            return _children(plan, args.package, args.out, "run", extra=extra,
                             parallel=args.parallel_cells, smoke=args.smoke)
        if args.cmd == "score":
            return _children(plan, args.package, args.out, "score", extra=[],
                             parallel=args.parallel_cells, smoke=False)
        cell_dir = args.out / "cells" / args.cell
        if args.cmd == "_run":
            return run_cell(plan, args.cell, cell_dir, endpoint=endpoint, dry=args.dry_run,
                            smoke=args.smoke, threads=args.threads, oracle=args.oracle)
        return score_cell(plan, args.cell, cell_dir)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
