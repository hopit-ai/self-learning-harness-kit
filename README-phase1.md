# Runbook: plan xp-p1-2026-10-10 (phase 0, then phase 1)

Repository `hopit-ai/self-learning-harness-kit`, tag `xp-p1-v1`: one command runs phase 0 (v2), the gate
and phase 1. One 8xH100 node under Slurm. You send back small archives; no model weights,
at most 100 MB per delivery.

## What runs

- **The 9B agent**: `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`,
  served by vLLM 0.30.0 with the pinned arguments in `slh/serve_spec.py`
  (only port, memory share, batch width and tensor/data parallel sizes are set by the job).
- **The teacher** (writers T+, S+, T; advisory teacher critic): the model named by the
  `TEACHER_*` settings (repository, revision, served name, layout; all recorded in the run
  identity), served by the same vLLM on all 8 GPUs next to the 9B.
- **The runner** (`python -m slh.xp.partner.runner`): learning runs, then every arm on
  every test question, then an export. Resumable; it refuses to start when the code,
  `package/` or the benchmark copy differ from `package/registration.json`
  (code snapshot `74c8b7c76b1fad4bcfead00c304f939faa37a592`, plan `c6ebe7bb08764a7b...`).

Cases per task (train pool, validation, sentinel, test questions):

| Task | train | validation | sentinel | test |
|---|---|---|---|---|
| address | 150 | 60 | 30 | 999 |
| variant | 399 | 60 | 30 | 999 |
| sequence | 399 | 60 | 30 | 999 |

## 0. What you need

- Shared storage visible from login and compute nodes with room for the teacher's
  weights plus about 100 GB (9B about 20 GB, caches, runs). Call it `WORK`.
- On the login node: git, network to GitHub and PyPI, and a **Python 3.12** (any way you
  like, e.g. `micromamba create -p $WORK/py312 python=3.12`). Everything, the job helpers
  in `slurm/` included, runs with that Python (`$WORK/venv/bin/python`); nothing falls
  back to the system `python3`.
- On compute nodes: network to Hugging Face (model download job only), and either
  **pyxis** (the default: the pinned `vllm/vllm-openai:v0.30.0` image) or
  a separate Python environment with `vllm==0.30.0`.
- Add your own `--account/--qos/--partition` to every `sbatch` line. All jobs use
  `--no-requeue`.

## 1. Setup (login node, about 20 minutes)

```bash
export WORK=/your/large/shared/storage/xp          # not home, not /tmp
export KIT=$WORK/kit TMPDIR=$WORK/tmp && mkdir -p $TMPDIR
git clone --branch xp-p1-v1 --depth 1 https://github.com/hopit-ai/self-learning-harness-kit.git $KIT
python3.12 -m venv $WORK/venv                        # or your 3.12's venv
cd $KIT && PYTHON=$WORK/venv/bin/python scripts/xp/fetch_vendor.sh   # must print "ok": true
$WORK/venv/bin/pip install -e $KIT/vendor/tau2-bench 'pyyaml>=6' 'jinja2>=3.1'   'pydantic>=2' 'huggingface_hub>=0.34'
cd $KIT && $WORK/venv/bin/python -c "import slh, tau2, yaml; print('ok')"
```

The kit itself is not pip-installed (the snapshot holds only the modules the experiment
imports): every script runs from `$KIT`, where `slh` is importable.

Optional, recommended with pyxis: import the image once (compute or login node):
`enroot import -o $WORK/vllm.sqsh 'docker://vllm/vllm-openai:v0.30.0'`, then
`export CONTAINER_IMAGE=$WORK/vllm.sqsh`. Without pyxis: create a vLLM environment
(`python3.12 -m venv $WORK/vllm-env && $WORK/vllm-env/bin/pip install vllm==0.30.0`)
and `export SERVE_MODE=venv VLLM_BIN=$WORK/vllm-env/bin/vllm`.

## 2. Models (zero-GPU job, Hugging Face from a compute node)

```bash
sbatch [--account ...] $KIT/slurm/download.sbatch    # writes $WORK/models/download.json
```

The teacher is whatever `TEACHER_REPO` / `TEACHER_REVISION` name (defaults in
`slurm/lib.sh`); the receipt records the revision either way.

## 3. Rehearse the job scripts without GPUs (optional, 2-5 minutes, login node)

```bash
bash $KIT/slurm/selftest.sh       # mock Slurm 25.05 + fake vLLM; must end "selftest ok"
```

## 4. One command: phase 0, the gate, phase 1

```bash
bash $KIT/slurm/run_all.sh [--account ...]     # run it again at any time: it resumes
```

1. **Phase 0** (one 8-GPU job, limit 8 h; expected **3.5-5 h**, about
   **28-40 GPU-hours**; no training of model weights). Rows, in order
   (`$WORK/phase0/job-<id>/phase0.json`): `facts`, `vendor`, `bootstrap`, `preflight`,
   `dry-run` (fake model; phase 0 stops here if it fails, before any server), `serve`
   (the teacher with CUDA graphs first, eager if graphs fail or take longer than
   `TEACHER_GRAPHS_STARTUP_S`; then the 9B), `probe-agent`, `probe-teacher`, `measure`
   (the plan's first two learning runs on the parallel pool, stopped at
   `MEASURE_LEARN_S`), `slice` (one short learning run, then the first 20 test cases per
   task through every arm, exported). Both servers' metrics are sampled throughout.
2. **The gate** (a CPU-only job after phase 0): pre-registered conditions on phase 0's
   evidence only (every row passed; servers started within bounds; no halt; every arm
   produced rows in the slice; context-limit rows within a fixed share per arm; paired
   error rates close; the measured phase-1 estimate within the GPU-hour cap; serving
   settings within bounds; the slice export valid). It writes `$WORK/run_all/gate.json`.
   **GO**: it submits phase 1 with settings derived from phase 0 by a registered rule
   (concurrency, learning units, time limit, CUDA graphs or eager). **NO-GO**: it
   archives the reasons (`$WORK/returns/xp-gate-<id>.tar.gz`) and stops; tell us.
3. **Phase 1** (estimate **about 35-100 h (2-3 jobs of 48 h)**, **about 280-790 GPU-hours**, hard cap
   `GPU_HOURS_CAP` (1600) over all phase-1 jobs; learning units run
   12 at a time). Every phase-1 job first re-verifies the gate (GO, bound to
   the phase-0 archive's hash, equal to a recomputation, the job's settings as derived),
   then a preflight (registration, code, cases, benchmark, run identity); no server starts
   before both pass. Each job lowers its own time limit to its GPU-hour budget, stops its
   runner 20 minutes before it so it always archives; `run_all.sh` resubmits to resume.
   Exit 3 = halted (cause in `halt.json`): fix and run `run_all.sh` again. Exit 4 =
   refused: tell us.

`run_all.sh` never reruns a passed phase 0 (unless `--force-phase0`) and never starts
phase 1 without a GO `gate.json` bound to that phase-0 archive. When it reports phase 1
complete, send back: `$WORK/returns/xp-phase0-<id>.tar.gz`, `$WORK/run_all/gate.json`,
`$WORK/returns/xp-phase1-export.tar.gz` (the delivery, at most 100 MB) and the last
`xp-phase1-<id>-logs.tar.gz`. If the export does not fit, export one shard per task:
`$WORK/venv/bin/python -m slh.xp.partner.export --run $WORK/runs/xp-phase1 --out shard-<task> --task <task>`.

## Settings (environment variables; defaults in `slurm/lib.sh`)

| Variable | Default | Meaning |
|---|---|---|
| `SERVE_MODE` | `container` | `container` (pyxis `--container-image`) or `venv` (`VLLM_BIN`) |
| `SERVE_LAUNCH` | `auto` | `srun` (overlapping steps), `direct` (child processes); `auto` = direct when the job script already runs inside a step |
| `CONTAINER_IMAGE` | `vllm/vllm-openai:v0.30.0` | image or `.sqsh` path |
| `PYXIS_FLAGS` | `--no-container-mount-home --container-mounts=$WORK:$WORK` | extra pyxis flags |
| `TEACHER_REPO`, `TEACHER_REVISION`, `TEACHER_NAME` | see `slurm/lib.sh` | the teacher |
| `TEACHER_TP`, `TEACHER_GPU_UTIL` | 8, 0.68 | teacher layout |
| `TEACHER_EXPERT_PARALLEL` | 1 | split the teacher's MoE experts across the GPUs |
| `TEACHER_EXTRA_ARGS` | `--language-model-only --engram-config={"cpu_offload":false}` | extra teacher flags, split on whitespace without a shell (write a JSON value as one token, no spaces) |
| `TEACHER_EXTRA_ARGS_JSON` | unset | the same as a JSON array of strings, e.g. `'["--language-model-only","--enable-expert-parallel","--engram-config={\"cpu_offload\":false}"]'`; replaces `TEACHER_EXTRA_ARGS` when set. Both are recorded verbatim in the run identity; the gate accepts only the registered flags (`--language-model-only`, `--enable-expert-parallel`, `--engram-config={"cpu_offload":false}`) |
| `TEACHER_MAX_BATCHED_TOKENS` | 0 | prefill chunk size (0 = vLLM's default) |
| `LEARNING_UNITS` | 12 | learning units run concurrently (scheduling only) |
| `XP_FSYNC_EVERY_S` | 2 | group commit of decision rows (seconds between fsyncs) |
| `OUT` | `$WORK/runs/xp-phase1` | run folder; keep it on shared storage (see below) |
| `AGENT_TP`, `AGENT_DP`, `AGENT_GPU_UTIL` | 4, 2, 0.16 | 9B layout (shares the GPUs) |
| `TEACHER_STARTUP_TIMEOUT_S`, `AGENT_STARTUP_TIMEOUT_S` | 7200, 3600 | health-probe deadlines |
| `TEACHER_EAGER` | 0 | CUDA graphs (startup ~390 s, ~150 tok/s per stream; eager ~5.8). Phase 0 falls back to eager when graphs fail or exceed `TEACHER_GRAPHS_STARTUP_S`; phase 1: derived by the gate from phase 0 |
| `TEACHER_GRAPHS_STARTUP_S` | 2400 | phase 0: bound for the CUDA-graph teacher start |
| `MEASURE_LEARN_S` | 5400 | phase 0: learning-throughput window |
| `CONCURRENCY` | 32 | decisions in flight |
| `GPU_HOURS_CAP`, `ARCHIVE_RESERVE_S` | 1600, 1200 | phase-1 budget and archive margin |

Keep every setting fixed across phase-1 resubmissions: serving arguments (TP, DP, memory
share, lengths, eager mode, expert parallel, extra arguments), the image or vLLM install
(and the launcher script's SHA-256), the serving environment variables and `CONCURRENCY`
are all part of the run identity, and the preflight refuses a run directory started under
other settings. `LEARNING_UNITS` may change (scheduling only).

Both servers run with prefix caching and chunked prefill (writers resend long evidence;
concurrent learning units batch). Decision rows are fsynced every `XP_FSYNC_EVERY_S`
seconds (group commit), so a slow shared filesystem (one earlier site measured 1.3 s per
fsync) does not throttle the run; after a crash, the few unsynced rows are simply redone.
Keep `OUT` on shared storage: a run folder on node-local disk works, but a resubmitted job
landing on another node cannot resume it.

The runner starts through `scripts/xp/xp_boot.py`: no bytecode is written, cached bytecode
is read only from a fresh empty directory, and it refuses compiled or path-hook files in
the kit (`.pyc` outside `__pycache__`, `.so`, `.pth`, `sitecustomize.py`) before checking
the registered code and importing it.

## Site notes (lessons from an earlier kit on this kind of node)

- The job script itself runs on the batch host; the servers start as `srun --overlap`
  steps. If your dispatcher already wraps the job in `srun` (a step that holds all 8 GPUs),
  a nested `srun` waits forever: the scripts detect `SLURM_STEP_ID` and start the servers
  as child processes instead (then `vllm` must be on `PATH` in that environment).
- Large models can take a long time to start (one earlier teacher spent 27 minutes
  capturing CUDA graphs and then hit vLLM's 300 s execute deadline): the startup timeouts
  are generous, the teacher runs eager and vLLM's deadlines are raised tenfold.
- The login node's `/tmp` may be small: set `TMPDIR=$WORK/tmp` there. Inside jobs the
  scripts use a short node-local `/tmp/xp-<job>` (Unix socket paths are limited to 107
  bytes) and keep every large cache under `WORK`.
- Login-node Python may lack libraries: only `$WORK/venv` needs them; `slurm/xpsite.py`
  needs no third-party package but does need that Python 3.12.
- Nothing in the scripts names a path on our side; everything is under `WORK` and `KIT`.

## What you may change, and what you must not

You may change infrastructure that does not change what is generated or scored: time
limits, Slurm directives, the settings above, cache and temp locations, the container or
vLLM environment route, a crash in our scripts (fix minimally and tell us). Record each
change: `git -C $KIT diff xp-p1-v1 > $WORK/local.diff` and send it with the archive.

You must not change the models or their revisions, `package/` (plan, cases, protocol),
anything under `slh/`, `configs/` or `scripts/xp/` (the runner refuses), sampling settings,
or any output file. A failed row with its logs is more useful to us than a row made to pass.

## Questions for you

1. Pyxis with the pinned vLLM image, or a separate vLLM 0.30.0 environment?
2. Is the default teacher (`slurm/lib.sh`) still the one you serve, and at which revision?
3. Can one job run 48 h, and how many GPU-hours per day can you give phase 1?
4. Does your dispatcher start job scripts inside an `srun` step?
