"""Partner side, hand-off 1: run decision units on the locally served 9B, then export.

    python -m slh.xp.fgate.units run    --package PKG --endpoint endpoint.yaml --out runs/h1-dry --dry-run
    python -m slh.xp.fgate.units run    --package PKG --endpoint endpoint.yaml --out runs/h1-smoke --smoke
    python -m slh.xp.fgate.units run    --package PKG --endpoint endpoint.yaml --out runs/h1
    python -m slh.xp.fgate.units export --run runs/h1 --out export-h1

One unit = one decision prefix (a question row: no gold) x one notes text (an F' lesson
selection, or empty = B1). The decision call is the phase-1 kit's
(``slh.xp.partner.runner.EndpointDecider``: B1 context, notes in the notes slot, the
plan's seed and output cap, the context guard). ``--dry-run`` uses the phase-1 kit's
fake decider (no server, no GPU). ``--smoke`` runs a fixed small slice (all units of the
first ``SMOKE_PREFIXES`` prefixes of each task and role).

Startup refuses (exit 4) when the plan's own hash, any package file or any registered
code file differs from the registration, or when the run directory belongs to another
run identity (package, mode, endpoint settings). Rows are appended one line at a time;
a rerun resumes (finished units are skipped). Errors follow the phase-1 classes: a
``halt`` (configuration, authentication, connection, anything unrecognised) stops the
run without a row (``halt.json``, exit 3; fix and rerun); per-case outcomes (time limit,
context limit, malformed output) are final rows.

**Interruption.** Ctrl-C (SIGINT) or SIGTERM stops the run promptly (about a second):
queued units are cancelled, no new decision starts, and decisions still in flight are
abandoned without a row (a row is only ever written for a decision that finished before
the interruption). The run exits 130 and a rerun resumes the missing units. At a hard
deadline stop the model server first (nothing more can be decided), then Ctrl-C.

The export (at most ``export_cap_bytes``) holds ``decisions.jsonl.gz`` (allowlisted
fields only), ``env.json`` and ``manifest.json``. Nothing in it is scored here: the
answer keys stay with us.

**Read turns (from 2026-10-23; deviation XD2, docs/xp/deviations.md).** A decision's
live read calls run on the database the plan's prefixes were built on
(``plan["retail_db"]["name"]``, passed to ``EndpointDecider``); startup refuses (exit 4)
a decider whose database differs from the plan's, and each row exports ``reads`` (name
and error flag of every read turn). Packages registered before (``xp-h1b-v1``) keep their
registered code: there the reads ran on the original database and rows carry no
``reads`` (``read_export`` accepts both row shapes).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from slh.xp.partner import common as pc

PLAN_SCHEMA = "xp-h1-plan-v1"
ROW_SCHEMA = "xp-h1-decision-v1"
EXPORT_SCHEMA = "xp-h1-export-v1"
SMOKE_PREFIXES = 2
ROW_FIELDS = ("schema", "unit_id", "track", "case_id", "case_hash", "prompt_hash", "x",
              "action", "tokens", "model_calls", "capped", "case_time_limit", "error_class",
              "wall_s", "reads")
# Rows of packages registered before XD2 (2026-10-23) carry no ``reads``.
LEGACY_ROW_FIELDS = tuple(f for f in ROW_FIELDS if f != "reads")
READ_FIELDS = ("name", "error")
ACTION_FIELDS = ("type", "name", "arguments", "content", "extra_calls", "reason")
ENDPOINT_FINGERPRINT = ("base_url", "model", "api_key_env", "model_revision", "vllm_version",
                        "serving_args", "replicas", "concurrency")
ABANDON_GRACE_S = 5.0
POLL_S = 0.1  # how often the runner and the workers look at the stop flag


class Interrupted(KeyboardInterrupt):
    """Exit 130: Ctrl-C / SIGTERM; the run directory is resumable."""


class Refused(RuntimeError):
    """Exit 4: package, code or identity differs from the registration."""


# --------------------------------------------------------------------------- package


def load_package(package: Path) -> dict[str, Any]:
    package = Path(package)
    plan = pc.read_json(package / "plan.json")
    body = {k: v for k, v in plan.items() if k != "sha256"}
    if plan.get("schema") != PLAN_SCHEMA or pc.sha_text(pc.canonical(body)) != plan.get("sha256"):
        raise Refused("plan.json differs from its own sha256")
    for name, meta in plan["files"].items():
        path = package / name
        if not path.is_file() or pc.sha_file(path) != meta["sha256"]:
            raise Refused(f"package file {name} differs from the plan")
    problems = code_mismatches(plan.get("code") or {})
    if problems:
        raise Refused(f"registered code differs: {problems[:5]}")
    questions = {q["case_id"]: q for q in pc.read_jsonl(package / "questions.jsonl")}
    for q in questions.values():
        pc.assert_no_gold(q, f"question {q['case_id']}")
        if pc.case_hash(q) != q["case_hash"]:
            raise Refused(f"{q['case_id']}: question differs from its case_hash")
    prompts = {p["notes_sha256"]: p for p in pc.read_jsonl(package / "prompts.jsonl")}
    for p in prompts.values():
        if pc.sha_text(p["text"]) != p["notes_sha256"]:
            raise Refused("a prompt text differs from its sha256")
    units = pc.read_jsonl(package / "units.jsonl")
    for u in units:
        if u["case_id"] not in questions or u["notes_sha256"] not in prompts:
            raise Refused(f"{u['unit_id']}: unknown question or prompt")
    return {"plan": plan, "questions": questions, "prompts": prompts, "units": units}


def code_mismatches(code: dict[str, str], root: Path = pc.REPO) -> list[str]:
    out = []
    for rel, sha in sorted(code.items()):
        p = Path(root) / rel
        if not p.is_file():
            out.append(f"missing {rel}")
        elif pc.sha_file(p) != sha:
            out.append(f"changed {rel}")
    return out


def plan_retail_db(plan: dict[str, Any]) -> str:
    """The pinned retail database the plan's prefixes were built on (``original`` for a
    plan that names none)."""
    from slh.xp import retaildb

    rec = plan.get("retail_db") or {}
    name = rec.get("name") or retaildb.ORIGINAL
    if rec.get("sha256") and rec["sha256"] != retaildb.spec(name)["sha256"]:
        raise Refused(f"plan retail_db {name}: sha256 differs from the pinned database")
    return name


def decider_db_problems(plan: dict[str, Any], decider: Any) -> list[str]:
    """A decider that runs read turns must run them on the plan's database. A decider
    without ``retail_db`` (fake deciders: no read turns) is not checked."""
    have = getattr(decider, "retail_db", None)
    want = plan_retail_db(plan)
    if have is not None and have != want:
        return [(f"decider reads on retail database {have!r}, the plan's prefixes were "
                 f"built on {want!r}")]
    return []


def retail_db_problems(pkg: dict[str, Any], *, probes: int = 3) -> list[str]:
    """Preflight: the plan's database file is here and matches its pin, and the decider's
    read environment (``decide.read_environment``) answers the package's own prefix reads
    (``get_user_details`` / ``get_order_details``) exactly as recorded in the prefixes.
    On an extension database the probes are reads of SYNTHETIC ids (absent from the
    original database), which only the right database can answer; a package on an
    extension with no such read is refused."""
    from slh.demo1.decide import read_environment
    from slh.xp import retaildb

    try:
        name = plan_retail_db(pkg["plan"])
        if name != retaildb.ORIGINAL:
            retaildb.verify(name)
        env = read_environment(name)
    except Exception as exc:  # noqa: BLE001 - reported
        return [f"retail_db: {exc}"[:300]]
    from tau2.data_model.message import ToolCall

    known: dict[str, set[str]] = {"get_user_details": set(), "get_order_details": set()}
    if name != retaildb.ORIGINAL:
        orig = retaildb.load(retaildb.ORIGINAL)
        known = {"get_user_details": set(orig["users"]),
                 "get_order_details": set(orig["orders"])}
    arg = {"get_user_details": "user_id", "get_order_details": "order_id"}
    problems: list[str] = []
    done = 0
    for q in pkg["questions"].values():
        msgs = q.get("messages") or []
        for k, m in enumerate(msgs[:-1]):
            call = (m.get("tool_calls") or [None])[0]
            if not call or call["name"] not in arg:
                continue
            result = msgs[k + 1]
            if result.get("role") != "tool" or "not found" in (result.get("content") or ""):
                continue
            if str((call.get("arguments") or {}).get(arg[call["name"]])) in known[call["name"]]:
                continue  # an original id: both databases answer it alike
            got = env.get_response(ToolCall(id=call["id"], name=call["name"],
                                            arguments=dict(call["arguments"]),
                                            requestor="assistant"))
            if got.error or got.content != result["content"]:
                problems.append(f"retail_db {name}: {call['name']} in {q['case_id']} is "
                                "answered differently from the prefix (decider reads on "
                                "another database)")
            done += 1
            break
        if done >= probes or problems:
            break
    if not done and not problems and name != retaildb.ORIGINAL:
        problems.append(f"retail_db {name}: no prefix reads a synthetic id; the read "
                        "environment cannot be probed")
    return problems


def smoke_units(units: list[dict[str, Any]], questions: dict[str, dict[str, Any]],
                roles: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """All units of the first SMOKE_PREFIXES prefixes per (task, track), in file order."""
    keep: dict[tuple[str, str], list[str]] = {}
    for u in units:
        k = (questions[u["case_id"]]["task"], u["track"])
        ids = keep.setdefault(k, [])
        if u["case_id"] not in ids and len(ids) < SMOKE_PREFIXES:
            ids.append(u["case_id"])
    chosen = {cid for ids in keep.values() for cid in ids}
    return [u for u in units if u["case_id"] in chosen]


# --------------------------------------------------------------------------- run


def endpoint_sha256(endpoint: dict[str, Any]) -> str:
    return pc.sha_text(pc.canonical({k: endpoint.get(k) for k in ENDPOINT_FINGERPRINT}))


def identity(pkg: dict[str, Any], *, mode: str, endpoint: dict[str, Any]) -> dict[str, Any]:
    return {"plan_sha256": pkg["plan"]["sha256"], "mode": mode,
            "endpoint_sha256": None if mode == "dry" else endpoint_sha256(endpoint)}


def _claim(out: Path, ident: dict[str, Any]) -> None:
    path = out / "identity.json"
    if path.is_file():
        if pc.read_json(path) != ident:
            raise Refused(f"{out} belongs to another run identity; use a new --out")
    else:
        pc.write_json(path, ident)


def _clean(action: dict[str, Any]) -> dict[str, Any]:
    return {k: action[k] for k in ACTION_FIELDS if k in action}


def run(package: Path, out: Path, *, endpoint: dict[str, Any] | None, dry_run: bool = False,
        smoke: bool = False, decider: Any = None, concurrency: int | None = None,
        stop: threading.Event | None = None) -> dict[str, Any]:
    """``stop`` (set by SIGINT/SIGTERM while running in the main thread, or by the
    caller) ends the run promptly: see the module docstring, **Interruption**."""
    from slh.xp.partner.errors import RunHalted, classify, halt_record

    pkg = load_package(package)
    plan = pkg["plan"]
    mode = "dry" if dry_run else "real"
    endpoint = endpoint or {}
    if not dry_run:
        # Part A decodes with the phase-1 decision config (configs/agent-vllm.yaml: greedy,
        # temperature 0, per-decision seed), sent with every request; the server must not
        # add sampling defaults of its own.
        problems = pc.serving_sampling_problems(endpoint.get("serving_args"))
        if problems:
            raise Refused("; ".join(problems))
    if decider is None:
        if dry_run:
            from slh.xp.partner.runner import FakeDecider

            decider = FakeDecider()
        else:
            from slh.xp.partner.runner import EndpointDecider

            decider = EndpointDecider(endpoint, seed=int(plan["seed"]),
                                      max_output_tokens=int(plan["limits"]["max_output_tokens"]),
                                      retail_db=plan_retail_db(plan))
    problems = decider_db_problems(plan, decider)
    if problems:
        raise Refused("; ".join(problems))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ident = {**identity(pkg, mode=mode, endpoint=endpoint), "smoke": bool(smoke)}
    _claim(out, ident)
    units = smoke_units(pkg["units"], pkg["questions"]) if smoke else pkg["units"]
    path = out / "decisions.jsonl"
    pc.repair_jsonl(path)
    done = {r["unit_id"] for r in pc.read_jsonl(path)}
    pending = [u for u in units if u["unit_id"] not in done]
    limit_s = float(plan["limits"]["case_limit_s"])
    conc = int(concurrency or endpoint.get("concurrency") or plan["limits"]["concurrency"])
    seed = int(plan["seed"])
    halted = threading.Event()
    stop = stop if stop is not None else threading.Event()
    halt_info: dict[str, Any] = {}
    lock = threading.Lock()

    def latch(error: BaseException | str, unit: dict[str, Any]) -> None:
        with lock:
            if halted.is_set():
                return
            if isinstance(error, str):
                halt_info.update({"event": "halt", "class": "halt", "error": error,
                                  "unit_id": unit["unit_id"]})
            else:
                halt_info.update(halt_record(error, unit_id=unit["unit_id"]))
            try:
                from slh.poc9.pipeline import latch_credit

                latch_credit(f"halt: {halt_info.get('error')}")
            except ImportError:  # the latch is best effort outside the kit
                halt_info["latch"] = "unavailable"
            halted.set()

    def one(unit: dict[str, Any]) -> dict[str, Any] | None:
        if halted.is_set() or stop.is_set():
            return None
        q = pkg["questions"][unit["case_id"]]
        text = pkg["prompts"][unit["notes_sha256"]]["text"]
        call_unit = {"class": "notes", "n": 1, "text": text, "prompt_hash": unit["prompt_hash"],
                     "task": q["task"]}
        box: dict[str, Any] = {}
        started = time.perf_counter()

        def call() -> None:
            try:
                box["value"] = decider.decide(q, call_unit, seed=seed,
                                              deadline=time.time() + limit_s)
            except BaseException as exc:  # noqa: BLE001 - classified below
                box["error"] = exc
                if classify(exc)[0] == "halt":
                    latch(exc, unit)

        t = threading.Thread(target=call, daemon=True)
        t.start()
        give_up = time.monotonic() + limit_s + ABANDON_GRACE_S
        while t.is_alive() and not stop.is_set():  # wake on Ctrl-C: abandon, no row
            t.join(max(0.0, min(POLL_S, give_up - time.monotonic())))
            if time.monotonic() >= give_up:
                break
        if stop.is_set():
            return None
        timed_out = t.is_alive()
        error = box.get("error")
        error_class = "case_time_limit" if timed_out else None
        if error is not None:
            kind, label = classify(error)
            if kind == "halt":
                return None
            error_class = label
        if halted.is_set() and timed_out:
            return None
        value = {} if timed_out else (box.get("value") or {})
        return {"schema": ROW_SCHEMA, "unit_id": unit["unit_id"], "track": unit["track"],
                "case_id": unit["case_id"], "case_hash": unit["case_hash"],
                "prompt_hash": unit["prompt_hash"], "x": unit["x"],
                "action": _clean(value.get("action") or {"type": "none", "reason": "error"}),
                "tokens": {"prompt": int(value.get("input_tokens", 0) or 0),
                           "completion": int(value.get("output_tokens", 0) or 0)},
                "model_calls": int(value.get("model_calls", 0) or 0),
                "capped": bool(value.get("capped")), "case_time_limit": timed_out,
                "error_class": error_class, "wall_s": round(time.perf_counter() - started, 3),
                "reads": [{k: (bool(r.get(k)) if k == "error" else r.get(k))
                           for k in READ_FIELDS} for r in (value.get("reads") or [])]}

    started = time.time()
    written = 0
    previous = _catch_signals(stop)
    ex = ThreadPoolExecutor(max_workers=max(1, conc))
    try:
        with pc.GroupAppender(path) as sink:
            queue = iter(pending)
            running: set[Any] = set()

            def fill() -> None:
                for u in queue:
                    if stop.is_set():
                        break
                    running.add(ex.submit(one, u))
                    if len(running) >= conc * 2 or halted.is_set():
                        break

            fill()
            while running and not stop.is_set():
                finished, running = wait(running, timeout=POLL_S,
                                         return_when=FIRST_COMPLETED)
                if stop.is_set():  # nothing decided after the interruption is written
                    break
                rows = [r for r in (f.result() for f in finished) if r is not None]
                sink.write(rows)
                written += len(rows)
                if not halted.is_set():
                    fill()
            if stop.is_set():
                for f in running:
                    f.cancel()
    finally:
        # Queued units are cancelled; in-flight workers notice the stop flag within
        # POLL_S and return without a row (their decision threads are daemons).
        ex.shutdown(wait=not stop.is_set(), cancel_futures=True)
        _restore_signals(previous)
    log = {"event": "run", "pending": len(pending), "written": written, "mode": mode,
           "smoke": smoke, "wall_s": round(time.time() - started, 3)}
    if stop.is_set():
        log["interrupted"] = True
    pc.append_jsonl(out / "logs.jsonl", [log])
    if stop.is_set():
        raise Interrupted(f"interrupted after {written} new rows; rerun to resume")
    if halted.is_set():
        pc.write_json(out / "halt.json", halt_info)
        raise RunHalted(halt_info.get("label", "halt"), halt_info.get("error", ""))
    pc.write_json(out / "env.json", env_report(endpoint, mode=mode))
    total = {r["unit_id"] for r in pc.read_jsonl(path)}
    complete = all(u["unit_id"] in total for u in units)
    return {**log, "units": len(units), "complete": complete}


def _catch_signals(stop: threading.Event) -> dict[int, Any] | None:
    """SIGINT/SIGTERM set ``stop`` instead of raising inside the wait (main thread only)."""
    import signal

    if threading.current_thread() is not threading.main_thread():
        return None

    def handler(signum: int, frame: Any) -> None:
        stop.set()

    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, handler)
    return previous


def _restore_signals(previous: dict[int, Any] | None) -> None:
    import signal

    for sig, h in (previous or {}).items():
        signal.signal(sig, h)


def env_report(endpoint: dict[str, Any], *, mode: str) -> dict[str, Any]:
    from importlib import metadata

    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {"mode": mode, "python": platform.python_version(), "platform": platform.platform(),
            "endpoint": {k: endpoint.get(k) for k in ENDPOINT_FINGERPRINT},
            "endpoint_sha256": endpoint_sha256(endpoint) if endpoint else None,
            "packages": {n: version(n) for n in ("vllm", "torch", "transformers", "tau2")},
            "gpus": os.environ.get("CUDA_VISIBLE_DEVICES")}


# --------------------------------------------------------------------------- export


def export(run_dir: Path, out: Path, *, cap_bytes: int | None = None) -> dict[str, Any]:
    run_dir, out = Path(run_dir), Path(out)
    ident = pc.read_json(run_dir / "identity.json")
    rows = []
    for r in pc.read_jsonl(run_dir / "decisions.jsonl"):
        if set(r) - set(ROW_FIELDS):
            raise ValueError(f"{r.get('unit_id')}: unlisted fields {sorted(set(r) - set(ROW_FIELDS))}")
        if set(ROW_FIELDS) - set(r):
            raise ValueError(f"{r.get('unit_id')}: missing fields "
                             f"{sorted(set(ROW_FIELDS) - set(r))} (a run started before XD2?)")
        pc.assert_no_gold(r, f"row {r['unit_id']}")
        rows.append({k: r[k] for k in ROW_FIELDS})
    data, n = pc.jsonl_bytes(sorted(rows, key=lambda r: r["unit_id"]))
    blob = pc.gz_bytes(data)
    env = pc.read_json(run_dir / "env.json") if (run_dir / "env.json").is_file() else {}
    env_bytes = (json.dumps(env, sort_keys=True, indent=1) + "\n").encode()
    cap = int(cap_bytes or 100_000_000)
    if len(blob) + len(env_bytes) > cap:
        raise ValueError(f"export would be {len(blob) + len(env_bytes)} bytes > cap {cap}")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"{out} is not empty")
    (out / "decisions.jsonl.gz").write_bytes(blob)
    (out / "env.json").write_bytes(env_bytes)
    manifest = {"schema": EXPORT_SCHEMA, "identity": ident,
                "files": {"decisions.jsonl.gz": {"sha256": pc.sha_bytes(blob),
                                                 "bytes": len(blob), "rows": n},
                          "env.json": {"sha256": pc.sha_bytes(env_bytes),
                                       "bytes": len(env_bytes)}},
                "cap_bytes": cap}
    pc.write_json(out / "manifest.json", manifest)
    return manifest


def read_export(export_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Our side: the manifest and rows, every file checked against the manifest."""
    export_dir = Path(export_dir)
    man = pc.read_json(export_dir / "manifest.json")
    if man.get("schema") != EXPORT_SCHEMA:
        raise ValueError("unknown export schema")
    names = {p.name for p in export_dir.iterdir()}
    if names != {"manifest.json", *man["files"]}:
        raise ValueError(f"export files differ from the manifest: {sorted(names)}")
    for name, meta in man["files"].items():
        if pc.sha_file(export_dir / name) != meta["sha256"]:
            raise ValueError(f"{name} differs from the manifest")
    rows = pc.read_jsonl(export_dir / "decisions.jsonl.gz")
    shapes = {frozenset(r) for r in rows}
    if len(shapes) > 1 or (shapes and next(iter(shapes)) not in
                           (frozenset(ROW_FIELDS), frozenset(LEGACY_ROW_FIELDS))):
        bad = next(r for r in rows if set(r) not in (set(ROW_FIELDS), set(LEGACY_ROW_FIELDS))
                   or frozenset(r) != frozenset(rows[0]))
        raise ValueError(f"{bad.get('unit_id')}: row fields differ from the allowlist")
    return man, rows


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    from slh.xp.partner.errors import RunHalted

    ap = argparse.ArgumentParser(prog="python -m slh.xp.fgate.units")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--package", type=Path, required=True)
    r.add_argument("--endpoint", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--concurrency", type=int)
    e = sub.add_parser("export")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "export":
        print(json.dumps(export(args.run, args.out), indent=1))
        print("export ok")
        return 0
    endpoint = None
    if not args.dry_run:
        if args.endpoint is None:
            ap.error("--endpoint is required unless --dry-run")
        from slh.xp.partner.runner import load_endpoint

        endpoint = load_endpoint(args.endpoint)
    try:
        res = run(args.package, args.out, endpoint=endpoint, dry_run=args.dry_run,
                  smoke=args.smoke, concurrency=args.concurrency)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 4
    except RunHalted as exc:
        print(f"halted: {exc} (see {args.out}/halt.json); fix the cause and rerun",
              file=sys.stderr)
        return 3
    except Interrupted as exc:
        print(f"stopped: {exc} (the same command re-runs only the missing units)",
              file=sys.stderr)
        return 130
    print(json.dumps(res, indent=1))
    if args.dry_run and res["complete"]:
        export(args.out, args.out / "export")
        print("export ok")
    return 0 if res["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
