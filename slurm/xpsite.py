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
    for flag in SPEED_ARGS:
        if flag not in argv:
            argv.append(flag)
    return argv


# Prefill speed for repeated long prompts (phase 0: teacher prefill ~1,200 tok/s with
# writers resending long evidence): cache shared prefixes and chunk long prefills so
# concurrent learning units batch. Output-neutral serving settings.
SPEED_ARGS = ["--enable-prefix-caching", "--enable-chunked-prefill"]


def teacher_argv(model, name, port, tp, util, max_len, eager, extra, expert_parallel=False,
                 max_batched_tokens=0):
    argv = ["vllm", "serve", model, "--served-model-name", name,
            "--tensor-parallel-size", str(tp), "--max-model-len", str(max_len),
            "--gpu-memory-utilization", str(util), "--generation-config", "vllm",
            "--reasoning-parser", "qwen3",
            "--default-chat-template-kwargs", json.dumps({"enable_thinking": False}),
            "--host", "127.0.0.1", "--port", str(port), "--uvicorn-log-level", "warning",
            *SPEED_ARGS]
    if int(max_batched_tokens or 0) > 0:
        argv += ["--max-num-batched-tokens", str(int(max_batched_tokens))]
    if eager:
        argv.append("--enforce-eager")  # reference incident: 27 min graph capture
    if expert_parallel:
        argv.append("--enable-expert-parallel")  # MoE experts split across the GPUs
    tokens = list(extra) if isinstance(extra, (list, tuple)) else (extra or "").split()
    extra = [a for a in tokens if a and not
             (expert_parallel and a == "--enable-expert-parallel")]
    return argv + extra


def extra_args(text, json_text=""):
    """Extra serving flags: ``json_text`` (a JSON array of strings, e.g.
    TEACHER_EXTRA_ARGS_JSON) when given, replacing ``text``; else ``text`` split on
    whitespace without a shell (so ``--engram-config={"cpu_offload":false}`` is one
    token as written). Returned verbatim; anything else refuses."""
    if (json_text or "").strip():
        try:
            value = json.loads(json_text)
        except ValueError as exc:
            raise SystemExit("extra args JSON is not valid JSON: %s" % exc)
        if not isinstance(value, list) or not all(isinstance(a, str) and a.strip()
                                                  for a in value):
            raise SystemExit("extra args JSON must be an array of non-empty strings")
        return value
    return (text or "").split()


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
                                 args.max_len, args.eager == "1",
                                 extra_args(args.extra, args.extra_json),
                                 args.expert_parallel == "1", args.max_batched_tokens)))
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


# Allowlist by prefix: every variable family that can change numerics, threading,
# kernels or communication of the servers.
SERVE_ENV_PREFIXES = (
    "VLLM_", "TORCH", "PYTORCH_", "CUDA_", "CUBLAS", "CUDNN_", "NVIDIA_", "NCCL_", "NVTE_",
    "TRITON_", "XFORMERS_", "FLASHINFER_", "FLASH_ATTENTION", "OMP_", "KMP_", "GOMP_",
    "MKL_", "OPENBLAS_", "BLIS_", "VECLIB_", "NUMEXPR_", "TOKENIZERS_", "TRANSFORMERS_",
    "HF_", "HUGGINGFACE_", "SAFETENSORS_", "RAY_", "UCX_", "FI_", "GLOO_", "MASTER_",
    "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONHASHSEED", "XLA_")
SERVE_ENV_NAMES = ("SERVE_CPUS", "XP_GPUS")
# Per-job values only: the bearer token, the socket directory; and anything secret.
SERVE_ENV_VOLATILE = frozenset({"VLLM_API_KEY", "VLLM_RPC_BASE_PATH", "RAY_TMPDIR"})
# Locations and diagnostics, not behaviour: cache, temp, log, debug and trace paths
# (phase 0: VLLM_CACHE_ROOT and TORCH_NCCL_DEBUG_INFO_TEMP_FILE named the job, which
# would have refused every resume in a new job).
_LOCATION = re.compile(r"CACHE|TMP|TEMP|_LOG|LOGGING|DEBUG|TRACE|DUMP|PROFIL", re.IGNORECASE)
_SECRET = re.compile(r"(^|_)(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|KEY|CREDENTIALS?)(_|$)",
                     re.IGNORECASE)


def _file_sha256(path):
    """SHA-256 of a launcher (VLLM_BIN may be a wrapper script): part of the identity."""
    import hashlib
    try:
        p = Path(shutil.which(path) or path) if path else None
        return hashlib.sha256(p.read_bytes()).hexdigest() if p and p.is_file() else None
    except OSError:
        return None


def serve_env(env=None):
    """Every environment setting the servers see that can change their behaviour:
    numerics and precision (TORCH_ALLOW_TF32*, CUBLAS*, NVIDIA_TF32_OVERRIDE ...),
    threading (OMP_, MKL_, ...), kernels and caches, communication, CPUs and GPUs per
    step. Excluded only: per-job values (token, socket directory) and secrets."""
    env = os.environ if env is None else env
    return {k: v for k, v in sorted(env.items())
            if (k.startswith(SERVE_ENV_PREFIXES) or k in SERVE_ENV_NAMES)
            and k not in SERVE_ENV_VOLATILE and not _SECRET.search(k)
            and not _LOCATION.search(k)}


def _models(path):
    """Repositories and revisions from the download receipt (no timings)."""
    if not path or not Path(path).is_file():
        return None
    data = json.loads(Path(path).read_text())
    return {k: {"repo": v.get("repo"), "revision": v.get("revision")}
            for k, v in sorted(data.items()) if isinstance(v, dict)}


METRICS = {"prompt_tokens": "vllm:prompt_tokens_total",
           "generation_tokens": "vllm:generation_tokens_total",
           "requests": "vllm:request_success_total",
           "running": "vllm:num_requests_running",
           "prefix_hits": "vllm:prefix_cache_hits_total",
           "prefix_queries": "vllm:prefix_cache_queries_total"}


def parse_metrics(text):
    """Sum every label variant of the vLLM metrics we use (Prometheus text format)."""
    out = {}
    for line in (text or "").splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        for key, metric in METRICS.items():
            if name in (metric, metric.replace("_total", "")):
                try:
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out


def sample_metrics(args):
    """Append one row per interval with the server's counters (until killed or
    ``until``): the phase-0 throughput, concurrency and prefix-cache measurements."""
    url = "http://127.0.0.1:%s/metrics" % args.port
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    while not args.until or time.time() < args.until:
        try:
            row = {"t": time.time(), "name": args.name, **parse_metrics(_get(url)[1])}
        except Exception as exc:  # noqa: BLE001 - recorded, the sampler keeps going
            row = {"t": time.time(), "name": args.name, "error": str(exc)[:200]}
        with open(args.out, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        time.sleep(args.interval)
    return 0


def endpoint(args):
    """The runner's endpoint file. Deterministic for a given site setting (the
    run identity includes it, so a resubmitted job must write the same file)."""
    spec = _serve_spec()
    agent_dir = os.path.realpath(args.agent_model_dir)  # resolves to snapshots/<revision>
    teacher_dir = os.path.realpath(args.teacher_model_dir) if args.teacher == "1" else None
    teacher_extra = extra_args(args.teacher_extra, args.teacher_extra_json)
    data = {
        "base_url": f"http://127.0.0.1:{args.agent_port}/v1",
        "model": spec["served_model_name"], "api_key_env": args.key_env,
        "model_revision": spec["model_revision"],
        "vllm_version": spec["vllm_version"],  # the served version is checked against it
        "serving_args": agent_argv(agent_dir, args.agent_port, args.agent_tp,
                                   args.agent_dp, args.agent_util, args.agent_max_num_seqs),
        "replicas": int(args.agent_dp), "concurrency": int(args.concurrency),
        # Scheduling only (not in the run identity): learning units run concurrently.
        "learning_units": int(args.learning_units),
        "teacher": None,
        # Every other execution setting (run identity: a resume refuses any change).
        "execution": {
            "serve_mode": args.serve_mode, "image": args.image or None,
            "vllm_bin": args.vllm_bin or None, "serve_launch": args.serve_launch,
            "pyxis_flags": args.pyxis_flags, "agent_model_dir": agent_dir,
            "teacher_model_dir": teacher_dir,
            "models": _models(args.download),
            "serve_env": serve_env(),
            # the extra flags exactly as given (TEACHER_EXTRA_ARGS[_JSON]), verbatim
            "teacher_extra_args": teacher_extra if args.teacher == "1" else None,
            "teacher_argv": teacher_argv(teacher_dir, args.teacher_name,
                                         args.teacher_port, args.teacher_tp,
                                         args.teacher_util, args.teacher_max_len,
                                         args.teacher_eager == "1", teacher_extra,
                                         args.teacher_expert_parallel == "1",
                                         args.teacher_max_batched_tokens)
            if args.teacher == "1" else None,
            "vllm_bin_sha256": _file_sha256(args.vllm_bin),
            # The phase gate this run was started under (run_all.sh): identity.
            "gate_sha256": args.gate_sha or None},
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
        epoch = time.time() + (END_MARGIN_S if args.action in ("end", "charge") else 0)
        if args.action == "charge" and args.until:  # provisional: covers cleanup ahead
            epoch = float(args.until) + END_MARGIN_S
        row = {"event": args.action, "job": args.job, "epoch": epoch, "utc": now(),
               "gpus": args.gpus, "limit_s": args.limit_s}
        if args.action == "charge":  # refused or failed admission: setup and archive
            row["start_epoch"] = float(args.start or os.environ.get("SLURM_JOB_START_TIME")
                                       or epoch)
        _append(path, row)
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
    used, starts, charges = 0.0, {}, {}
    for r in rows:
        if r["event"] == "charge":  # never admitted: the LAST charge row per job counts
            charges[r["job"]] = r
        elif r["event"] == "start":
            starts[r["job"]] = r
        elif r["event"] == "end" and r["job"] in starts:
            s = starts.pop(r["job"])
            used += max(0.0, r["epoch"] - s["epoch"]) / 3600 * s["gpus"]
    used += sum(max(0.0, r["epoch"] - r["start_epoch"]) / 3600 * r["gpus"]
                for r in charges.values())
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

class _Deadline(Exception):
    pass


class _Timed:
    """File reader that stops at the deadline: the rest of the member is padded
    with zeros (fast), so the tar stays valid, and the cut is recorded."""

    def __init__(self, handle, size, deadline):
        self.handle, self.left, self.deadline, self.cut = handle, size, deadline, False

    def read(self, n=-1):
        n = self.left if n is None or n < 0 else min(n, self.left)
        if n <= 0:
            return b""
        if self.cut or (self.deadline and time.time() > self.deadline):
            self.cut = True
            data = b"\0" * n
        else:
            data = self.handle.read(n)
            data += b"\0" * (n - len(data))  # a file that shrank meanwhile
        self.left -= len(data)
        return data


class _Prefixed:
    """A fixed prefix, then a _Timed reader."""

    def __init__(self, head, timed):
        self.head, self.timed = head, timed

    def read(self, n=-1):
        if self.head:
            if n is None or n < 0 or n >= len(self.head):
                head, self.head = self.head, b""
                rest = self.timed.read(-1 if n is None or n < 0 else n - len(head))
                return head + rest
            head, self.head = self.head[:n], self.head[n:]
            return head
        return self.timed.read(n)


def run_bounded(args):
    """Run CMD in its own process group; at ``seconds``, and again after it ends, stop
    the whole group: TERM, ``grace`` seconds, KILL; return only when the group is
    empty (exit 124 on timeout, else CMD's code)."""
    import signal

    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    proc = subprocess.Popen(cmd, start_new_session=True)
    pgid = proc.pid
    timed_out = False
    try:
        rc = proc.wait(timeout=max(1, args.seconds))
    except subprocess.TimeoutExpired:
        timed_out = True
        rc = 124

    def group_alive():
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    if group_alive():
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)
        end = time.time() + args.grace
        while group_alive() and time.time() < end:
            proc.poll()
            time.sleep(0.2)
        if group_alive():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
        end = time.time() + 10
        while group_alive() and time.time() < end:
            proc.poll()
            time.sleep(0.2)
    proc.poll()
    if group_alive():
        print("run-bounded: process group %d survived KILL" % pgid, file=sys.stderr)
        return 125
    return 124 if timed_out else rc


def run_locked(args):
    """Run CMD while holding an exclusive lock on ``lock`` (single writer for the
    phase-1 submission: the gate job and run_all.sh never submit twice)."""
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    Path(args.lock).parent.mkdir(parents=True, exist_ok=True)
    with open(args.lock, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            return subprocess.call(cmd)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def job_limit(args):
    """This job's Slurm time limit in seconds (scontrol TimeLimit), -1 if unknown."""
    job = os.environ.get("SLURM_JOB_ID")
    rec = parse_scontrol(run(["scontrol", "show", "job", job])["out"]) if job else {}
    left = parse_slurm_duration(rec.get("TimeLimit", ""))
    print(-1 if left is None else left)
    return 0


def collect(args):
    """A small archive to send back: whole small files, the last 2 MB of logs, the
    export directory as is; refuses above ``cap``. With ``deadline`` (epoch) the
    deadline holds during enumeration AND copying: files not reached are left out,
    a file being copied is cut (zero-padded), and TRUNCATED.txt says what happened."""
    import io

    out = Path(args.out)
    deadline = args.deadline or 0.0
    total, members, notes = 0, [], []

    def late():
        return deadline and time.time() > deadline

    try:
        for spec in args.include:
            root = Path(spec)
            if not root.exists():
                continue
            walker = [root] if root.is_file() else (p for p in root.rglob("*"))
            for p in walker:
                if late():
                    raise _Deadline("enumeration")
                if not p.is_file() or p.name.endswith((".lock", ".tmp")) or \
                        "__pycache__" in p.parts:
                    continue
                arc = os.path.join(args.name, root.name,
                                   os.path.relpath(str(p), str(root))) if root.is_dir() \
                    else os.path.join(args.name, p.name)
                size = p.stat().st_size
                tail = p.suffix in (".log", ".out", ".err") and size > LOG_TAIL_BYTES
                members.append((p, os.path.normpath(arc), tail))
                total += min(size, LOG_TAIL_BYTES) if tail else size
    except _Deadline:
        notes.append("enumeration stopped at the deadline (%d files listed)" % len(members))
    if total > args.cap:
        print("refused: %d bytes > %d" % (total, args.cap), file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    members.sort(key=lambda m: m[1])
    done = 0
    with tarfile.open(str(out), "w:gz") as tar:
        for p, arc, tail in members:
            if late():
                notes.append("%d of %d files left out at the deadline"
                             % (len(members) - done, len(members)))
                break
            try:
                with p.open("rb") as f:
                    if tail:
                        f.seek(-LOG_TAIL_BYTES, 2)
                        head = b"[... truncated to the last 2 MB ...]\n"
                        timed = _Timed(f, LOG_TAIL_BYTES, deadline)  # same deadline
                        info = tarfile.TarInfo(arc)
                        info.size, info.mtime, info.mode = len(head) + LOG_TAIL_BYTES, \
                            int(p.stat().st_mtime), 0o644
                        tar.addfile(info, _Prefixed(head, timed))
                        if timed.cut:
                            notes.append("%s cut at the deadline (zero-padded)" % arc)
                            done += 1
                            break
                    else:
                        info = tar.gettarinfo(str(p), arcname=arc)
                        timed = _Timed(f, info.size, deadline)
                        tar.addfile(info, timed)
                        if timed.cut:
                            notes.append("%s cut at the deadline (zero-padded)" % arc)
                            done += 1
                            break
            except OSError as exc:
                notes.append("%s unreadable: %s" % (arc, exc))
            done += 1
        if notes:
            note = ("archive incomplete:\n" + "\n".join(notes) + "\n").encode()
            info = tarfile.TarInfo(os.path.join(args.name, "TRUNCATED.txt"))
            info.size, info.mtime, info.mode = len(note), int(time.time()), 0o644
            tar.addfile(info, io.BytesIO(note))
    print(json.dumps({"archive": str(out), "files": done, "bytes": out.stat().st_size,
                      "complete": not notes}))
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
                 ("util", "0.68"), ("max-len", "131072"), ("eager", "1"), ("extra", ""),
                 ("extra-json", ""),
                 ("expert-parallel", "0"), ("max-batched-tokens", "0")):
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
                 ("teacher-eager", "1"), ("teacher-extra", ""), ("teacher-extra-json", ""),
                 ("serve-mode", "container"),
                 ("image", ""), ("vllm-bin", ""), ("agent-model-dir", "."),
                 ("teacher-model-dir", "."), ("serve-launch", "auto"), ("pyxis-flags", ""),
                 ("download", ""), ("teacher-expert-parallel", "0"), ("learning-units", "12"),
                 ("gate-sha", ""),
                 ("teacher-max-batched-tokens", "0")):
        p.add_argument("--" + k, default=d, required=k == "out")
    p = sub.add_parser("sample-metrics")
    p.add_argument("--name", required=True)
    p.add_argument("--port", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--interval", type=float, default=30.0)
    p.add_argument("--until", type=float, default=0.0)
    p = sub.add_parser("ledger")
    p.add_argument("action", choices=("start", "end", "check", "charge"))
    p.add_argument("--start", type=float, help="charge: the allocation's start epoch")
    p.add_argument("--until", type=float,
                   help="charge: provisional end epoch (a later charge row replaces it)")
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
    p = sub.add_parser("run-locked")
    p.add_argument("--lock", required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    sub.add_parser("job-limit")
    p = sub.add_parser("run-bounded")
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--grace", type=float, default=20.0)
    p.add_argument("command", nargs=argparse.REMAINDER)
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
            "probe": probe, "endpoint": endpoint, "ledger": ledger, "budget": budget, "admit": admit, "run-bounded": run_bounded, "run-locked": run_locked,
            "job-limit": job_limit,
            "sample-metrics": sample_metrics,
            "check-version": check_version,
            "collect": collect}[args.cmd](args) if args.cmd else (ap.print_help() or 2)


if __name__ == "__main__":
    sys.exit(main())
