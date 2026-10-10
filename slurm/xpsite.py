#!/usr/bin/env python3
"""Site helper for the XP Slurm jobs (standard library only).

Runs with the kit's Python 3.12 (``$WORK/venv/bin/python``; the jobs pass it as
SITE_PY and nothing falls back to a system python3); it refuses an older Python. It
imports no third-party package and never imports ``slh``: the pinned serving values
come from the JSON that the trusted bootstrap writes (``scripts/xp/xp_boot.py
--serve-spec``, file in XP_SERVE_SPEC).

    xpsite.py facts --out F               site facts (GPUs, Slurm, step, disk, sockets)
    xpsite.py remaining                   seconds left in this job (-1 if unknown)
    xpsite.py agent-argv [...]            the 9B `vllm serve` argv, one per line
    xpsite.py teacher-argv [...]          the teacher `vllm serve` argv, one per line
    xpsite.py wait --name N --port P --pid PID --timeout S --out F
    xpsite.py probe --name N --port P --model M --requests K --concurrency C --out F
    xpsite.py endpoint --out F [...]      the runner's endpoint file (JSON is YAML)
    xpsite.py ledger start|end|check ...  GPU-hour ledger across resubmissions
    xpsite.py budget --file L --cap H ... this job's budget: min(Slurm time left,
                                          (cap - used) / GPUs); exit 2 if too small
    xpsite.py check-version --startup F   the served vLLM version equals the pin
    xpsite.py collect --out A.tar.gz ...  small return archive (log tails capped)

Exit codes: 0 ok; 1 failed (a server died, a check failed); 2 timeout or refused.
"""
# ruff: noqa: UP031

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

if sys.version_info < (3, 12):  # noqa: UP036 - a runtime guard at the partner site
    sys.exit("xpsite.py needs Python 3.12 or newer (use $WORK/venv/bin/python as SITE_PY)")
KIT = Path(__file__).resolve().parents[1]
SOCKET_LIMIT = 107  # sun_path bytes on Linux (108 with the terminating NUL)
LOG_TAIL_BYTES = 2_000_000
ARCHIVE_CAP_BYTES = 100_000_000


def now():
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def run(argv, timeout=30):
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout, text=True, check=False)
        return {"argv": argv, "rc": p.returncode, "out": p.stdout[-20000:]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"argv": argv, "rc": None, "out": f"{type(exc).__name__}: {exc}"}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(str(tmp), str(path))


# --------------------------------------------------------------------------- Slurm time

def parse_slurm_duration(text):
    """Slurm durations as squeue %L / scontrol print them (25.05 included):
    'MM:SS', 'HH:MM:SS', 'D-HH', 'D-HH:MM', 'D-HH:MM:SS', 'MM'. None for
    'UNLIMITED', 'NOT_SET', 'INVALID', 'N/A' or anything unreadable."""
    text = (text or "").strip()
    if not text or text.upper() in ("UNLIMITED", "NOT_SET", "INVALID", "N/A", "INFINITE"):
        return None
    m = re.fullmatch(r"(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?", text)
    if not m:
        return None
    days, a, b, c = m.groups()
    days = int(days or 0)
    if days and b is None:            # D-HH
        h, mi, s = int(a), 0, 0
    elif days and c is None:          # D-HH:MM
        h, mi, s = int(a), int(b), 0
    elif c is not None:               # [D-]HH:MM:SS
        h, mi, s = int(a), int(b), int(c)
    elif b is not None:               # MM:SS
        h, mi, s = 0, int(a), int(b)
    else:                             # MM
        h, mi, s = 0, int(a), 0
    return ((days * 24 + h) * 60 + mi) * 60 + s


def parse_scontrol(text):
    """Key=Value pairs of `scontrol show job/step` (25.05: one record, wrapped
    lines; values without spaces except a few, which we do not need)."""
    out = {}
    for token in re.split(r"\s+", text or ""):
        if "=" in token:
            k, v = token.split("=", 1)
            out.setdefault(k, v)
    return out


def remaining_seconds(env=None):
    """Seconds left in this job: SLURM_JOB_END_TIME (epoch), then scontrol's
    EndTime, then squeue %L. -1 when none is readable (e.g. no time limit)."""
    env = os.environ if env is None else env
    end = env.get("SLURM_JOB_END_TIME", "")
    if end.isdigit():
        return max(0, int(end) - int(time.time()))
    job = env.get("SLURM_JOB_ID")
    if not job:
        return -1
    rec = parse_scontrol(run(["scontrol", "show", "job", job])["out"])
    if rec.get("EndTime") and rec["EndTime"] not in ("Unknown", "None"):
        try:
            t = dt.datetime.strptime(rec["EndTime"], "%Y-%m-%dT%H:%M:%S")  # noqa: DTZ007 - local
            return max(0, int(time.mktime(t.timetuple()) - time.time()))
        except ValueError:
            pass
    lines = run(["squeue", "-h", "-j", job, "-o", "%L"])["out"].strip().splitlines()
    left = parse_slurm_duration(lines[-1]) if lines else None
    return -1 if left is None else left


def in_step(env=None):
    """True inside an srun step (a dispatcher's step or a pyxis container step):
    a nested srun would wait for GPUs this step already holds."""
    env = os.environ if env is None else env
    step = env.get("SLURM_STEP_ID", "")
    return bool(step) and step not in ("batch", "extern", "4294967294", "4294967291")


# --------------------------------------------------------------------------- facts

def socket_ok(base, suffix):
    return len(os.fsencode(str(base))) + suffix <= SOCKET_LIMIT


def facts(args):
    env = os.environ
    work = Path(env.get("WORK") or ".")
    tmp = Path(env.get("TMPDIR") or "/tmp")
    usage = {}
    for name, p in (("work", work), ("tmpdir", tmp), ("dev_shm", Path("/dev/shm"))):
        try:
            u = shutil.disk_usage(str(p))
            usage[name] = {"path": str(p), "free_gib": round(u.free / 2**30, 1),
                           "total_gib": round(u.total / 2**30, 1)}
        except OSError as exc:
            usage[name] = {"path": str(p), "error": str(exc)}
    data = {
        "time_utc": now(), "host": socket.gethostname(), "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "slurm_env": {k: v for k, v in sorted(env.items())
                      if k.startswith("SLURM_") and "TOKEN" not in k},
        "in_step": in_step(env), "remaining_s": remaining_seconds(env),
        "slurm_version": run(["srun", "--version"])["out"].strip(),
        "scontrol_job": run(["scontrol", "show", "job", env.get("SLURM_JOB_ID", "")])["out"]
        if env.get("SLURM_JOB_ID") else None,
        "squeue_steps": run(["squeue", "-s", "-j", env.get("SLURM_JOB_ID", "")])["out"]
        if env.get("SLURM_JOB_ID") else None,
        "pyxis": "container-image" in run(["srun", "--help"])["out"],
        "enroot": shutil.which("enroot") is not None,
        "gpus": run(["nvidia-smi", ("--query-gpu=index,name,memory.total,memory.used,"
                     "driver_version"), "--format=csv,noheader"])["out"].strip().splitlines(),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "disk": usage,
        "sockets": {"TMPDIR": socket_ok(tmp, 32),
                    "VLLM_RPC_BASE_PATH": socket_ok(env.get("VLLM_RPC_BASE_PATH") or tmp, 37)},
        # Libraries only (slh is imported only through scripts/xp/xp_boot.py).
        "harness_python": run([env.get("HARNESS_PY", "python3"), "-I", "-B", "-c",
                               "import sys, tau2, yaml; print(sys.version.split()[0])"],
                              timeout=120)["out"].strip()[-300:],
        "login_python": run(["python3", "--version"])["out"].strip(),
    }
    write_json(args.out, data)
    print(json.dumps({"gpus": len(data["gpus"]), "in_step": data["in_step"],
                      "remaining_s": data["remaining_s"], "pyxis": data["pyxis"]}))
    return 0


# --------------------------------------------------------------------------- argv

MODEL_PLACEHOLDER = "<MODEL>"  # = scripts/xp/xp_boot.py


def _serve_spec():
    """The pinned serving values, as scripts/xp/xp_boot.py --serve-spec printed them
    after its checks (file in XP_SERVE_SPEC). This helper never imports slh."""
    path = os.environ.get("XP_SERVE_SPEC")
    if not path or not Path(path).is_file():
        sys.exit("XP_SERVE_SPEC must name the JSON from scripts/xp/xp_boot.py --serve-spec")
    return json.loads(Path(path).read_text())


def _set(argv, flag, value):
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv += [flag, str(value)]
    return argv


def agent_argv(model, port, tp, dp, util, max_num_seqs):
    """The pinned argv (slh/serve_spec.py vllm_args) with only the site's
    infrastructure settings changed: port, memory share, batch width, TP x DP."""
    spec = _serve_spec()
    argv = [model if a == MODEL_PLACEHOLDER else a for a in spec["vllm_args"]]
    _set(argv, "--port", port)
    _set(argv, "--gpu-memory-utilization", util)
    _set(argv, "--max-num-seqs", max_num_seqs)
    _set(argv, "--tensor-parallel-size", tp)
    if int(dp) > 1:
        _set(argv, "--data-parallel-size", dp)
    return argv


def teacher_argv(model, name, port, tp, util, max_len, eager, extra):
    argv = ["vllm", "serve", model, "--served-model-name", name,
            "--tensor-parallel-size", str(tp), "--max-model-len", str(max_len),
            "--gpu-memory-utilization", str(util), "--generation-config", "vllm",
            "--reasoning-parser", "qwen3",
            "--default-chat-template-kwargs", json.dumps({"enable_thinking": False}),
            "--host", "127.0.0.1", "--port", str(port), "--uvicorn-log-level", "warning"]
    if eager:
        argv.append("--enforce-eager")  # reference incident: 27 min graph capture
    return argv + [a for a in (extra or "").split() if a]


def cmd_agent_argv(args):
    print("\n".join(agent_argv(args.model, args.port, args.tp, args.dp, args.util,
                               args.max_num_seqs)))
    return 0


def cmd_endpoint_argv(args):
    """The served argv, read back from the endpoint file (what is served is what the
    run identity records)."""
    text = Path(args.endpoint).read_text()
    data = json.loads(text[text.index("{"):])
    argv = data["serving_args"] if args.which == "agent" else \
        (data.get("execution") or {}).get("teacher_argv")
    if not argv:
        sys.exit("no %s argv in %s" % (args.which, args.endpoint))
    print("\n".join(argv))
    return 0


def cmd_teacher_argv(args):
    print("\n".join(teacher_argv(args.model, args.name, args.port, args.tp, args.util,
                                 args.max_len, args.eager == "1", args.extra)))
    return 0


# --------------------------------------------------------------------------- serving

def _get(url, key=None, timeout=10):
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def _alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    try:  # a zombie child counts as dead
        with open("/proc/%d/stat" % int(pid)) as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def wait(args):
    """Health probe: /health then /v1/models (with the key), until ready, the
    process dies, or the startup timeout passes. Writes the startup record."""
    start, base = time.time(), f"http://127.0.0.1:{args.port}"
    key = os.environ.get(args.key_env) if args.key_env else None
    record = {"name": args.name, "port": args.port, "timeout_s": args.timeout,
              "started_utc": now(), "status": None}
    last, beat = None, start
    while True:
        elapsed = time.time() - start
        if args.pid and not _alive(args.pid):
            record.update(status="died", seconds=round(elapsed, 1), last_error=last)
            break
        try:
            code, _ = _get(base + "/health", timeout=10)
            if code == 200:
                code, body = _get(base + "/v1/models", key=key, timeout=30)
                ver = None
                try:
                    ver = json.loads(_get(base + "/version", timeout=10)[1])
                except Exception:  # noqa: BLE001, S110 - the version is optional
                    pass
                record.update(status="ready", seconds=round(elapsed, 1),
                              models=[m.get("id") for m in json.loads(body).get("data", [])],
                              version=ver)
                break
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = f"{type(exc).__name__}: {str(exc)[:200]}"
        if elapsed > args.timeout:
            record.update(status="timeout", seconds=round(elapsed, 1), last_error=last)
            break
        if time.time() - beat > 300:
            print("[xpsite] %s still starting after %d s" % (args.name, elapsed), flush=True)
            beat = time.time()
        time.sleep(args.interval)
    record["gpus_after"] = run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                                "--format=csv,noheader"])["out"].strip().splitlines()
    write_json(args.out, record)
    print(json.dumps({k: record.get(k) for k in ("name", "status", "seconds")}), flush=True)
    return {"ready": 0, "died": 1}.get(record["status"], 2)


def probe(args):
    """Small latency/throughput probe: K chat requests, C in flight."""
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    key = os.environ.get(args.key_env) if args.key_env else None
    body = {"model": args.model, "temperature": 0, "max_tokens": args.max_tokens, "seed": 1,
            "messages": [{"role": "system", "content": "You are a retail support agent."},
                         {"role": "user", "content": "In one sentence: what do you ask a "
                                                     "customer before changing an address?"}],
            "chat_template_kwargs": {"enable_thinking": False}}
    data = json.dumps(body).encode()
    results, lock = [], threading.Lock()

    def one(_):
        t = time.time()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if key:
            req.add_header("Authorization", "Bearer " + key)
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                out = json.loads(r.read().decode())
            row = {"ok": True, "s": round(time.time() - t, 3),
                   "out_tokens": (out.get("usage") or {}).get("completion_tokens")}
        except Exception as exc:  # noqa: BLE001 - recorded
            row = {"ok": False, "s": round(time.time() - t, 3),
                   "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        with lock:
            results.append(row)

    start = time.time()
    todo = list(range(args.requests))
    threads = []
    while todo or threads:
        threads = [th for th in threads if th.is_alive()]
        while todo and len(threads) < args.concurrency:
            th = threading.Thread(target=one, args=(todo.pop(),), daemon=True)
            th.start()
            threads.append(th)
        time.sleep(0.05)
    wall = time.time() - start
    ok = [r for r in results if r["ok"]]
    record = {"name": args.name, "requests": args.requests, "concurrency": args.concurrency,
              "ok": len(ok), "wall_s": round(wall, 2),
              "requests_per_s": round(len(ok) / wall, 3) if wall else None,
              "latency_s": sorted(r["s"] for r in ok),
              "errors": [r["error"] for r in results if not r["ok"]][:5]}
    write_json(args.out, record)
    print(json.dumps({k: record[k] for k in ("name", "ok", "requests_per_s")}))
    return 0 if len(ok) == args.requests else 1


SERVE_ENV_PREFIXES = ("VLLM_", "NCCL_", "TORCH_NCCL_", "PYTORCH_", "HF_HUB_", "CUDA_LAUNCH_")
SERVE_ENV_NAMES = ("SERVE_CPUS", "XP_GPUS")
SERVE_ENV_VOLATILE = frozenset({"VLLM_API_KEY", "VLLM_RPC_BASE_PATH"})  # per job


def serve_env(env=None):
    """Every environment setting the servers see that can change their behaviour
    (vLLM, NCCL, torch and hub variables, CPUs and GPUs per step); per-job values
    (the bearer token, the socket directory) excluded."""
    env = os.environ if env is None else env
    return {k: v for k, v in sorted(env.items())
            if (k.startswith(SERVE_ENV_PREFIXES) or k in SERVE_ENV_NAMES)
            and k not in SERVE_ENV_VOLATILE}


def _models(path):
    """Repositories and revisions from the download receipt (no timings)."""
    if not path or not Path(path).is_file():
        return None
    data = json.loads(Path(path).read_text())
    return {k: {"repo": v.get("repo"), "revision": v.get("revision")}
            for k, v in sorted(data.items()) if isinstance(v, dict)}


def endpoint(args):
    """The runner's endpoint file. Deterministic for a given site setting (the
    run identity includes it, so a resubmitted job must write the same file)."""
    spec = _serve_spec()
    agent_dir = os.path.realpath(args.agent_model_dir)  # resolves to snapshots/<revision>
    teacher_dir = os.path.realpath(args.teacher_model_dir) if args.teacher == "1" else None
    data = {
        "base_url": f"http://127.0.0.1:{args.agent_port}/v1",
        "model": spec["served_model_name"], "api_key_env": args.key_env,
        "model_revision": spec["model_revision"],
        "vllm_version": spec["vllm_version"],  # the served version is checked against it
        "serving_args": agent_argv(agent_dir, args.agent_port, args.agent_tp,
                                   args.agent_dp, args.agent_util, args.agent_max_num_seqs),
        "replicas": int(args.agent_dp), "concurrency": int(args.concurrency),
        "teacher": None,
        # Every other execution setting (run identity: a resume refuses any change).
        "execution": {
            "serve_mode": args.serve_mode, "image": args.image or None,
            "vllm_bin": args.vllm_bin or None, "serve_launch": args.serve_launch,
            "pyxis_flags": args.pyxis_flags, "agent_model_dir": agent_dir,
            "teacher_model_dir": teacher_dir,
            "models": _models(args.download),
            "serve_env": serve_env(),
            "teacher_argv": teacher_argv(teacher_dir, args.teacher_name,
                                         args.teacher_port, args.teacher_tp,
                                         args.teacher_util, args.teacher_max_len,
                                         args.teacher_eager == "1", args.teacher_extra)
            if args.teacher == "1" else None},
    }
    if args.teacher == "1":
        data["teacher"] = {
            "mode": "self_hosted", "base_url": f"http://127.0.0.1:{args.teacher_port}/v1",
            "model": args.teacher_name, "api_key_env": args.key_env,
            "extra": {"chat_template_kwargs": {"enable_thinking": False}},
            "weights": "{}@{}".format(args.teacher_repo, args.teacher_revision or "unpinned")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("# written by slurm/xpsite.py endpoint (JSON is YAML)\n"
                              + json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(args.out)
    return 0


# --------------------------------------------------------------------------- ledger

END_MARGIN_S = 30  # teardown after the end row (temp cleanup, Slurm epilogue)


@contextlib.contextmanager
def _locked(path):
    """Exclusive lock on <ledger>.lock: admission and rows are atomic across jobs
    (the ledger lives on shared storage; flock is honoured there by Lustre/NFSv4)."""
    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _append(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def ledger(args):
    """GPU-hours across resubmissions of one phase. ``start`` and ``end`` rows (end:
    the job's last act, after archiving, plus END_MARGIN_S); ``check`` refuses once
    the cap is reached. A job with a start and no end is charged its reservation."""
    path = Path(args.file)
    with _locked(path):
        used, _ = ledger_used(path, args.job)
        if args.action == "check":
            print(json.dumps({"gpu_hours_used": round(used, 2), "cap": args.cap}))
            return 2 if args.cap and used >= args.cap else 0
        epoch = time.time() + (END_MARGIN_S if args.action == "end" else 0)
        _append(path, {"event": args.action, "job": args.job, "epoch": epoch, "utc": now(),
                       "gpus": args.gpus, "limit_s": args.limit_s})
    print(json.dumps({"gpu_hours_used_before": round(used, 2)}))
    return 0


def admit(args):
    """Atomic admission (under the ledger lock): this job's budget is measured from
    its START (the whole allocation is charged): min(Slurm limit, (cap - used) x 3600 /
    GPUs, max), where ``used`` counts other jobs' open reservations in full. On
    admission the job's start row reserves the whole budget at once; its end row
    later releases what it did not use. Refuses (exit 2) when less than ``min``
    seconds remain or another job of this ledger still runs."""
    start = float(args.start or os.environ.get("SLURM_JOB_START_TIME") or time.time())
    with _locked(args.file):
        used, open_jobs = ledger_used(args.file, args.job)
        alive = [j for j in open_jobs if _job_alive(j)]
        elapsed = max(0, int(time.time() - start))
        remaining_h = max(0.0, args.cap - used)
        limits = [int(remaining_h * 3600 / max(1, args.gpus))]
        if args.slurm_left > 0:
            limits.append(args.slurm_left + elapsed)
        if args.max_s > 0:
            limits.append(args.max_s)
        budget_s = max(0, min(limits))
        left = budget_s - elapsed
        ok = left >= args.min and not alive
        record = {"gpu_hours_used": round(used, 3), "cap": args.cap,
                  "gpu_hours_remaining": round(remaining_h, 3), "gpus": args.gpus,
                  "start_epoch": int(start), "elapsed_s": elapsed, "budget_s": budget_s,
                  "left_s": left, "deadline_epoch": int(start) + budget_s,
                  "min_s": args.min, "other_running_jobs": alive,
                  "status": "ok" if ok else "refused"}
        if ok:
            _append(args.file, {"event": "start", "job": args.job, "epoch": start,
                                "utc": now(), "gpus": args.gpus, "limit_s": budget_s})
    print(json.dumps(record))
    return 0 if ok else 2


def ledger_used(path, job=None):
    """(GPU-hours used, [jobs started and not ended]) from a ledger file."""
    rows = []
    if Path(path).is_file():
        rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    used, starts = 0.0, {}
    for r in rows:
        if r["event"] == "start":
            starts[r["job"]] = r
        elif r["event"] == "end" and r["job"] in starts:
            s = starts.pop(r["job"])
            used += max(0.0, r["epoch"] - s["epoch"]) / 3600 * s["gpus"]
    open_jobs = [s for s in starts.values() if s["job"] != job]
    used += sum(s["gpus"] * s.get("limit_s", 0) / 3600 for s in open_jobs)
    return used, [s["job"] for s in open_jobs]


def _job_alive(job):
    out = run(["squeue", "-h", "-j", str(job), "-o", "%i"])
    return out["rc"] == 0 and str(job) in out["out"]


def budget(args):
    """This allocation's budget in seconds: min(Slurm time left, (cap - used) x 3600 /
    GPUs), so GPUs x budget never exceeds the remaining cap (archive time included:
    the job stops its runner ``reserve`` seconds before the budget deadline). Refuses
    (exit 2) below ``min`` seconds, or while another phase-1 job of this ledger runs."""
    used, open_jobs = ledger_used(args.file, args.job)
    alive = [j for j in open_jobs if _job_alive(j)]
    remaining_h = max(0.0, args.cap - used)
    cap_s = int(remaining_h * 3600 / max(1, args.gpus))
    limits = [cap_s] + ([args.slurm_left] if args.slurm_left > 0 else []) + \
        ([args.max_s] if args.max_s > 0 else [])
    budget_s = max(0, min(limits))
    record = {"gpu_hours_used": round(used, 3), "cap": args.cap,
              "gpu_hours_remaining": round(remaining_h, 3), "gpus": args.gpus,
              "slurm_left_s": args.slurm_left, "budget_s": budget_s,
              "deadline_epoch": int(time.time()) + budget_s, "reserve_s": args.reserve,
              "min_s": args.min, "other_running_jobs": alive}
    ok = budget_s >= args.min and not alive
    record["status"] = "ok" if ok else "refused"
    print(json.dumps(record))
    return 0 if ok else 2


def check_version(args):
    rec = json.loads(Path(args.startup).read_text())
    ver = rec.get("version") or {}
    got = ver.get("version") if isinstance(ver, dict) else None
    want = _serve_spec()["vllm_version"]
    print(json.dumps({"served": got, "pinned": want}))
    return 0 if got == want else 2


# --------------------------------------------------------------------------- collect

def collect(args):
    """A small archive to send back: whole small files, the last 2 MB of logs,
    the export directory as is; refuses above 100 MB."""
    out = Path(args.out)
    total = 0
    members = []
    for spec in args.include:
        root = Path(spec)
        if not root.exists():
            continue
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for p in files:
            if p.name.endswith((".lock", ".tmp")) or "__pycache__" in p.parts:
                continue
            arc = os.path.join(args.name, root.name,
                               os.path.relpath(str(p), str(root))) if root.is_dir() \
                else os.path.join(args.name, p.name)
            size = p.stat().st_size
            tail = p.suffix in (".log", ".out", ".err") and size > LOG_TAIL_BYTES
            members.append((p, os.path.normpath(arc), tail))
            total += min(size, LOG_TAIL_BYTES) if tail else size
    if total > args.cap:
        print("refused: %d bytes > %d" % (total, args.cap), file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    import io
    with tarfile.open(str(out), "w:gz") as tar:
        for p, arc, tail in members:
            if args.deadline and time.time() > args.deadline:
                skipped = len(members) - members.index((p, arc, tail))
                note = ("archive cut at the budget deadline; %d files left out\n"
                        % skipped).encode()
                info = tarfile.TarInfo(os.path.join(args.name, "TRUNCATED.txt"))
                info.size, info.mtime, info.mode = len(note), int(time.time()), 0o644
                tar.addfile(info, io.BytesIO(note))
                break
            if tail:
                with p.open("rb") as f:
                    f.seek(-LOG_TAIL_BYTES, 2)
                    data = b"[... truncated to the last 2 MB ...]\n" + f.read()
                info = tarfile.TarInfo(arc)
                info.size, info.mtime, info.mode = len(data), int(p.stat().st_mtime), 0o644
                tar.addfile(info, io.BytesIO(data))
            else:
                tar.add(str(p), arcname=arc, recursive=False)
    print(json.dumps({"archive": str(out), "files": len(members),
                      "bytes": out.stat().st_size}))
    return 0


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(prog="xpsite.py")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("facts")
    p.add_argument("--out", required=True)
    sub.add_parser("remaining")
    sub.add_parser("in-step")
    p = sub.add_parser("agent-argv")
    for k, d in (("model", None), ("port", "8001"), ("tp", "1"), ("dp", "1"),
                 ("util", "0.92"), ("max-num-seqs", "8")):
        p.add_argument("--" + k, default=d, required=d is None)
    p = sub.add_parser("teacher-argv")
    for k, d in (("model", None), ("name", None), ("port", "8002"), ("tp", "8"),
                 ("util", "0.74"), ("max-len", "131072"), ("eager", "1"), ("extra", "")):
        p.add_argument("--" + k, default=d, required=d is None)
    p = sub.add_parser("endpoint-argv")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--which", choices=("agent", "teacher"), required=True)
    p = sub.add_parser("wait")
    p.add_argument("--name", required=True)
    p.add_argument("--port", required=True)
    p.add_argument("--pid")
    p.add_argument("--key-env")
    p.add_argument("--timeout", type=float, required=True)
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--name", required=True)
    p.add_argument("--port", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--key-env")
    p.add_argument("--requests", type=int, default=16)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--out", required=True)
    p = sub.add_parser("endpoint")
    for k, d in (("out", None), ("agent-port", "8001"), ("agent-tp", "1"), ("agent-dp", "1"),
                 ("agent-util", "0.92"), ("agent-max-num-seqs", "8"), ("concurrency", "8"),
                 ("key-env", None), ("teacher", "0"),
                 ("teacher-port", "8002"), ("teacher-name", "teacher"),
                 ("teacher-repo", ""), ("teacher-revision", ""), ("teacher-tp", "8"),
                 ("teacher-util", "0.74"), ("teacher-max-len", "131072"),
                 ("teacher-eager", "1"), ("teacher-extra", ""), ("serve-mode", "container"),
                 ("image", ""), ("vllm-bin", ""), ("agent-model-dir", "."),
                 ("teacher-model-dir", "."), ("serve-launch", "auto"), ("pyxis-flags", ""),
                 ("download", "")):
        p.add_argument("--" + k, default=d, required=k == "out")
    p = sub.add_parser("ledger")
    p.add_argument("action", choices=("start", "end", "check"))
    p.add_argument("--file", required=True)
    p.add_argument("--job", default=os.environ.get("SLURM_JOB_ID", "local"))
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--limit-s", type=int, default=0)
    p.add_argument("--cap", type=float, default=0.0)
    p = sub.add_parser("budget")
    p.add_argument("--file", required=True)
    p.add_argument("--cap", type=float, required=True)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--job", default=os.environ.get("SLURM_JOB_ID", "local"))
    p.add_argument("--slurm-left", type=int, default=-1)
    p.add_argument("--max-s", type=int, default=0)
    p.add_argument("--reserve", type=int, default=1200)
    p.add_argument("--min", type=int, default=7200)
    p = sub.add_parser("admit")
    p.add_argument("--file", required=True)
    p.add_argument("--cap", type=float, required=True)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--job", default=os.environ.get("SLURM_JOB_ID", "local"))
    p.add_argument("--start", type=float)
    p.add_argument("--slurm-left", type=int, default=-1)
    p.add_argument("--max-s", type=int, default=0)
    p.add_argument("--min", type=int, default=7200)
    p = sub.add_parser("check-version")
    p.add_argument("--startup", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--out", required=True)
    p.add_argument("--deadline", type=float, default=0.0,
                   help="epoch: stop adding files at this time (partial archive)")
    p.add_argument("--name", default="xp-return")
    p.add_argument("--cap", type=int, default=ARCHIVE_CAP_BYTES)
    p.add_argument("include", nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "facts":
        return facts(args)
    if args.cmd == "remaining":
        print(remaining_seconds())
        return 0
    if args.cmd == "in-step":
        print("1" if in_step() else "0")
        return 0
    return {"agent-argv": cmd_agent_argv, "endpoint-argv": cmd_endpoint_argv, "teacher-argv": cmd_teacher_argv, "wait": wait,
            "probe": probe, "endpoint": endpoint, "ledger": ledger, "budget": budget, "admit": admit,
            "check-version": check_version,
            "collect": collect}[args.cmd](args) if args.cmd else (ap.print_help() or 2)


if __name__ == "__main__":
    sys.exit(main())
