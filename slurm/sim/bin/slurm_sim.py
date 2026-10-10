#!/usr/bin/env python3
"""Mock Slurm 25.05 + nvidia-smi + vLLM for rehearsing the XP job scripts on a
machine without Slurm or GPUs. Process-only: no cgroups, no scheduling, no GPUs.

Put this directory first on PATH and set SIM_SLURM_STATE to an empty directory.
- sbatch: runs the script synchronously (stdout to --output with %x/%j), prints the
  job id with --parsable; sets SLURM_JOB_ID, SLURM_JOB_END_TIME, ... (no step id).
- srun: refuses (hangs like a real step waiting for GPUs, or exits 1 with
  SIM_SRUN_BUSY=fail) when called from inside a step that holds the GPUs unless
  --overlap is given; otherwise runs the command with SLURM_STEP_ID set; pyxis
  --container-* flags are accepted and recorded, the command runs on the host.
- scontrol show job / squeue -s / squeue -o %L / sacct -P: 25.05 output formats
  (taken from a partner's capture, names replaced).
- nvidia-smi --query-gpu=...: 8 x H100 80GB HBM3.
- vllm serve ...: a fake OpenAI-compatible server (fake_vllm.py) on the --port.
Every call is appended to $SIM_SLURM_STATE/calls.jsonl.
"""
# Runs on any python3 >= 3.8 at partner sites; %-formatting kept on purpose.
# ruff: noqa: UP031
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = Path(os.environ.get("SIM_SLURM_STATE") or "/nonexistent")
VERSION = "slurm 25.05.3"
SRUN_VALUED = {"--ntasks", "-n", "--nodes", "-N", "--gres", "--gpus", "--mem", "-c",
               "--cpus-per-task", "--time", "-t", "--container-image", "--container-mounts",
               "--container-workdir", "--container-name", "--job-name", "-J", "--output",
               "--error", "--account", "-A", "--qos", "--partition", "-p", "--export"}


def record(cmd, argv, **extra):
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "calls.jsonl").open("a") as f:
        f.write(json.dumps({"cmd": cmd, "argv": argv, "step": os.environ.get("SLURM_STEP_ID"),
                            "time": time.time(), **extra}) + "\n")


def job_state():
    try:
        return json.loads((STATE / "job.json").read_text())
    except (OSError, ValueError):
        return {}


def fmt_duration(s):
    s = max(0, int(s))
    d, rest = divmod(s, 86400)
    h, rest = divmod(rest, 3600)
    m, sec = divmod(rest, 60)
    return ("%d-%02d:%02d:%02d" % (d, h, m, sec)) if d else ("%02d:%02d:%02d" % (h, m, sec)
                                                             if h else "%02d:%02d" % (m, sec))


def parse_time(text):
    text = text.strip()
    m = re.fullmatch(r"(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?", text)
    d, a, b, c = m.groups()
    d = int(d or 0)
    if c is not None:
        return ((d * 24 + int(a)) * 60 + int(b)) * 60 + int(c)
    if b is not None:
        return (d * 24 * 60 + int(a) * 60 + int(b)) * 60 if d else int(a) * 60 + int(b)
    return (d * 24 + int(a)) * 3600 if d else int(a) * 60


def sbatch(argv):
    opts, script, rest = {}, None, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if script is None and a.startswith("-"):
            if "=" in a:
                k, v = a.split("=", 1)
                opts[k] = v
            elif a in ("--parsable", "--exclusive", "--no-requeue", "--wait"):
                opts[a] = True
            else:
                opts[a] = argv[i + 1]
                i += 1
        elif script is None:
            script = a
        else:
            rest.append(a)
        i += 1
    text = Path(script).read_text()
    for line in text.splitlines():
        m = re.match(r"#SBATCH\s+(--[\w-]+)(?:[= ](\S+))?", line)
        if m and m.group(1) not in opts:
            opts[m.group(1)] = m.group(2) if m.group(2) is not None else True
    counter = STATE / "next-job"
    job = int(counter.read_text()) if counter.is_file() else 404801
    STATE.mkdir(parents=True, exist_ok=True)
    counter.write_text(str(job + 1))
    limit = parse_time(opts.get("--time") or opts.get("-t") or "01:00:00")
    name = opts.get("--job-name") or opts.get("-J") or Path(script).name
    start = int(time.time())
    (STATE / "job.json").write_text(json.dumps({
        "job": job, "name": name, "start": start, "limit": limit,
        "gres": opts.get("--gres"), "state": "RUNNING"}))
    out = (opts.get("--output") or "slurm-%j.out").replace("%j", str(job)).replace("%x", name)
    err = (opts.get("--error") or out).replace("%j", str(job)).replace("%x", name)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(SLURM_JOB_ID=str(job), SLURM_JOBID=str(job), SLURM_JOB_NAME=name,
               SLURM_JOB_NUM_NODES="1", SLURM_NNODES="1", SLURM_GPUS_ON_NODE="8",
               SLURM_JOB_GPUS="0,1,2,3,4,5,6,7", SLURM_SUBMIT_DIR=os.getcwd(),
               SLURM_JOB_START_TIME=str(start), SLURM_JOB_END_TIME=str(start + limit),
               SLURM_CLUSTER_NAME="sim", SLURMD_NODENAME="sim-node-001",
               CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7")
    env.update(os.environ.get("SIM_SBATCH_ENV") and
               dict(kv.split("=", 1) for kv in os.environ["SIM_SBATCH_ENV"].split(",")) or {})
    record("sbatch", argv, job=job, opts={k: v for k, v in opts.items()})
    if opts.get("--parsable"):
        print(job, flush=True)
    else:
        print("Submitted batch job %d" % job, flush=True)
    with open(out, "ab") as fo, open(err, "ab") if err != out else open(os.devnull) as fe:
        rc = subprocess.call(["bash", script, *rest], env=env, stdout=fo,
                             stderr=fe if err != out else subprocess.STDOUT)
    st = job_state()
    st.update(state="COMPLETED" if rc == 0 else "FAILED", rc=rc, end=int(time.time()))
    (STATE / "job.json").write_text(json.dumps(st))
    return 0


def srun(argv):
    opts, i = {}, 0
    while i < len(argv) and argv[i].startswith("-"):
        a = argv[i]
        if a in ("--version",):
            print(VERSION)
            return 0
        if a in ("--help", "-h"):
            print("Usage: srun [OPTIONS(0)...] executable(0) [args(0)...]\n"
                  "      --container-image=[USER@][REGISTRY#]IMAGE[:TAG]|PATH\n"
                  "      --container-mounts=SRC:DST[:FLAGS][,SRC:DST...]\n"
                  "      --no-container-mount-home\n      --overlap")
            return 0
        if "=" in a:
            k, v = a.split("=", 1)
            opts[k] = v
        elif a in SRUN_VALUED:
            opts[a] = argv[i + 1]
            i += 1
        else:
            opts[a] = True
        i += 1
    cmd = argv[i:]
    record("srun", argv, opts=opts)
    if os.environ.get("SLURM_STEP_ID") and "--overlap" not in opts:
        # Reference incident: the enclosing step already holds all 8 GPUs.
        sys.stderr.write("srun: Job {} step creation temporarily disabled, retrying "
                         "(Requested nodes are busy)\n".format(os.environ.get("SLURM_JOB_ID")))
        sys.stderr.flush()
        if os.environ.get("SIM_SRUN_BUSY") == "fail":
            return 1
        while True:
            time.sleep(3600)
    counter = STATE / "next-step"
    step = int(counter.read_text()) if counter.is_file() else 0
    counter.write_text(str(step + 1))
    env = dict(os.environ, SLURM_STEP_ID=str(step), SLURM_STEPID=str(step))
    if "--container-image" in opts:
        env["SIM_CONTAINER_IMAGE"] = opts["--container-image"]
    try:
        os.execvpe(cmd[0], cmd, env)
    except OSError as exc:
        sys.stderr.write(f"srun: error: {cmd[0]}: {exc}\n")
        return 2


def scontrol(argv):
    record("scontrol", argv)
    st = job_state()
    if argv[:2] == ["show", "job"] and st:
        start = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st["start"]))
        end = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st["start"] + st["limit"]))
        print("""JobId={job} JobName={name}
   UserId=<USER>(737239) GroupId=<USER>(737239) MCS_label=N/A
   Priority=15401 Nice=0 Account=midpri QOS=midpri
   JobState={state} Reason=None Dependency=(null)
   Requeue=0 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
   RunTime={run} TimeLimit={limit} TimeMin=N/A
   SubmitTime={start} EligibleTime={start}
   StartTime={start} EndTime={end} Deadline=N/A
   Partition=h100 AllocNode:Sid=<IP>:2117900
   NodeList=sim-node-001   BatchHost=sim-node-001
   NumNodes=1 NumCPUs=192 NumTasks=1 CPUs/Task=1
   ReqTRES=cpu=1,mem=8G,node=1,billing=1,gres/gpu=8
   AllocTRES=cpu=192,mem=1800G,node=1,billing=192,gres/gpu=8
   OverSubscribe=NO Contiguous=0 Licenses=(null)""".format(
            job=st["job"], name=st["name"], state=st.get("state", "RUNNING"),
            run=fmt_duration(time.time() - st["start"]), limit=fmt_duration(st["limit"]),
            start=start, end=end))
        return 0
    if argv[:1] == ["update"] and os.environ.get("SIM_SCONTROL_UPDATE") == "fail":
        print("slurm_update error: Access/permission denied", file=sys.stderr)
        return 1
    if argv[:1] == ["update"] and st:
        kv = dict(a.split("=", 1) for a in argv[1:] if "=" in a)
        if kv.get("JobId") == str(st["job"]) and "TimeLimit" in kv:
            st["limit"] = parse_time(kv["TimeLimit"])
            (STATE / "job.json").write_text(json.dumps(st))
            return 0
        print("slurm_update error: Invalid job id specified", file=sys.stderr)
        return 1
    if argv[:2] == ["show", "config"]:
        print("SLURM_VERSION           = 25.05.3\nProctrackType           = proctrack/cgroup\n"
              "KillWait                = 40 sec\nOverTimeLimit           = 0 min")
        return 0
    print("slurm_load_jobs error: Invalid job id specified", file=sys.stderr)
    return 1


def squeue(argv):
    record("squeue", argv)
    st = job_state()
    if not st:
        return 0
    if "-o" in argv and argv[argv.index("-o") + 1] == "%L":
        print(fmt_duration(st["start"] + st["limit"] - time.time()))
        return 0
    if "-s" in argv:
        run = fmt_duration(time.time() - st["start"])
        print("         STEPID     NAME PARTITION     USER      TIME NODELIST")
        print("   %d.batch    batch      h100 <USER>   %s sim-node-001" % (st["job"], run))
        print("  %d.extern   extern      h100 <USER>   %s sim-node-001" % (st["job"], run))
        return 0
    print("JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)")
    print("%d h100 %s <USER> R %s 1 sim-node-001" % (st["job"], st["name"][:8],
                                                    fmt_duration(time.time() - st["start"])))
    return 0


def sacct(argv):
    record("sacct", argv)
    st = job_state()
    if st:
        print("%d|%s|0:0|%s|Unknown|%s|%s|billing=192,cpu=192,gres/gpu=8,mem=1800G,node=1|"
              % (st["job"], st.get("state", "RUNNING"),
                 time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st["start"])),
                 fmt_duration(time.time() - st["start"]), fmt_duration(st["limit"])))
    return 0


def nvidia_smi(argv):
    record("nvidia-smi", argv)
    q = next((a.split("=", 1)[1] for a in argv if a.startswith("--query-gpu=")), None)
    if q is None:
        print("NVIDIA-SMI 570.86.15   Driver Version: 570.86.15   CUDA Version: 12.8 (sim)")
        return 0
    values = {"index": None, "name": "NVIDIA H100 80GB HBM3", "memory.total": "81559 MiB",
              "memory.used": "1 MiB", "driver_version": "570.86.15", "utilization.gpu": "0 %",
              "uuid": None}
    for gpu in range(8):
        row = []
        for field in q.split(","):
            v = values.get(field.strip(), "N/A")
            row.append(str(gpu) if field.strip() == "index" else
                       ("GPU-sim-%04d" % gpu) if field.strip() == "uuid" else v)
        print(", ".join(row))
    return 0


def vllm(argv):
    record("vllm", argv, container=os.environ.get("SIM_CONTAINER_IMAGE"))
    if argv[:1] == ["--version"]:
        print("0.30.0")
        return 0
    os.execv(sys.executable, [sys.executable, str(HERE.parent / "fake_vllm.py"), *argv])


def main(name, argv):
    fn = {"sbatch": sbatch, "srun": srun, "scontrol": scontrol, "squeue": squeue,
          "sacct": sacct, "nvidia-smi": nvidia_smi, "vllm": vllm,
          "sinfo": lambda a: print(VERSION) or 0}.get(name)
    if fn is None:
        sys.stderr.write(f"slurm_sim: unknown command {name} ({shlex.join(argv)})\n")
        return 2
    return fn(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
