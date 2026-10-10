# Runbook: plan xp-p1-2026-10-10 (phase 0, then phase 1)

Repository `hopit-ai/self-learning-harness-kit`, tags `xp-p1-phase0-v1` (phase 0) and `xp-p1-v1`
(phase 1). One 8xH100 node under Slurm. You send back small archives; no model weights,
at most 100 MB per delivery.

## What runs

- **The 9B agent**: `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`,
  served by vLLM 0.30.0 with the pinned arguments in `slh/serve_spec.py`
  (only port, memory share, batch width and tensor/data parallel sizes are set by the job).
- **The teacher** (writers T+, S+, T; teacher critic): a Qwen3.5-397B-A17B **FP8**
  checkpoint served by the same vLLM on all 8 GPUs (tensor parallel 8), next to the 9B.
- **The runner** (`python -m slh.xp.partner.runner`): learning runs, then every arm on
  every test question, then an export. Resumable; it refuses to start when the code,
  `package/` or the benchmark copy differ from `package/registration.json`
  (code snapshot `5470c549fb7732b5eca7b45f1de198665f9427e7`, plan `8c56c6c008863a75...`).

Cases per task (train pool, validation, sentinel, test questions):

| Task | train | validation | sentinel | test |
|---|---|---|---|---|
| address | 150 | 60 | 30 | 999 |
| variant | 399 | 60 | 30 | 999 |
| sequence | 399 | 60 | 30 | 999 |

## 0. What you need

- Shared storage visible from login and compute nodes with about **600 GB free**
  (teacher about 400 GB, 9B about 20 GB, caches, runs). Call it `WORK`.
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
git clone --branch xp-p1-phase0-v1 --depth 1 https://github.com/hopit-ai/self-learning-harness-kit.git $KIT
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

Set `TEACHER_REPO` / `TEACHER_REVISION` first if you already hold an FP8 checkpoint of
Qwen3.5-397B-A17B; the receipt records the revision either way.

## 3. Rehearse the job scripts without GPUs (optional, 2-5 minutes, login node)

```bash
bash $KIT/slurm/selftest.sh       # mock Slurm 25.05 + fake vLLM; must end "selftest ok"
```

## 4. Phase 0: site check (one job, no training)

```bash
sbatch [--account ...] $KIT/slurm/phase0.sbatch
```

Rows, in order (`$WORK/phase0/job-<id>/phase0.json`): `facts` (GPUs, Slurm, disk, socket
paths), `vendor` (benchmark copy), `dry-run` (fake model, real learning code; must print
"export ok"), `serve` (teacher then 9B with health probes; startup seconds), `probe-agent`
and `probe-teacher` (request rates), `smoke` (the real runner on your servers: smoke
setting, 5 test cases per task). Expected **1.5-3 h** wall (limit 6 h), about **15-25
GPU-hours**. Send back `$WORK/returns/xp-phase0-<id>.tar.gz` (small) whatever happened.

## 5. Phase 1 (after we confirm phase 0 and name the phase-1 tag)

```bash
git -C $KIT fetch --depth 1 origin tag xp-p1-v1 && git -C $KIT checkout xp-p1-v1
bash $KIT/slurm/submit.sh [--account ...]            # resubmit the same line after any stop
```

Estimate **12-24 h** wall, **110-230 GPU-hours**, hard cap `GPU_HOURS_CAP` (240) over all
phase-1 jobs. `submit.sh` asks Slurm for a time limit that fits what is left of the cap;
the job checks again at start-up, trims its own limit, and stops its runner 20 minutes
before the budget ends so it always archives; resubmitting resumes. Before any server
starts, a verification-only preflight checks the registration, the code, the cases, the
benchmark copy and the run identity (exit 4 on a mismatch, no GPU time spent on serving). Exit 3 = halted (cause in `halt.json`): fix and resubmit.
Exit 4 = refused (code, package or benchmark differ from the registration): tell us.
Send back `$WORK/returns/xp-phase1-export.tar.gz` (the delivery, at most 100 MB) and the
last `xp-phase1-<id>-logs.tar.gz`. If the export does not fit, export one shard per task:
`$WORK/venv/bin/python -m slh.xp.partner.export --run $WORK/runs/xp-phase1 --out shard-<task> --task <task>`.

## Settings (environment variables; defaults in `slurm/lib.sh`)

| Variable | Default | Meaning |
|---|---|---|
| `SERVE_MODE` | `container` | `container` (pyxis `--container-image`) or `venv` (`VLLM_BIN`) |
| `SERVE_LAUNCH` | `auto` | `srun` (overlapping steps), `direct` (child processes); `auto` = direct when the job script already runs inside a step |
| `CONTAINER_IMAGE` | `vllm/vllm-openai:v0.30.0` | image or `.sqsh` path |
| `PYXIS_FLAGS` | `--no-container-mount-home --container-mounts=$WORK:$WORK` | extra pyxis flags |
| `TEACHER_TP`, `TEACHER_GPU_UTIL` | 8, 0.74 | teacher layout |
| `AGENT_TP`, `AGENT_DP`, `AGENT_GPU_UTIL` | 4, 2, 0.16 | 9B layout (shares the GPUs) |
| `TEACHER_STARTUP_TIMEOUT_S`, `AGENT_STARTUP_TIMEOUT_S` | 7200, 3600 | health-probe deadlines |
| `TEACHER_EAGER` | 1 | no CUDA-graph capture for the teacher |
| `CONCURRENCY` | 32 | decisions in flight |
| `GPU_HOURS_CAP`, `ARCHIVE_RESERVE_S` | 240, 1200 | phase-1 budget and archive margin |

Keep every setting fixed across phase-1 resubmissions: serving arguments (TP, DP, memory
share, lengths, eager mode, extra arguments), the image or vLLM install and `CONCURRENCY`
are all part of the run identity, and the preflight refuses a run directory started under
other settings.

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
change: `git -C $KIT diff xp-p1-phase0-v1 > $WORK/local.diff` and send it with the archive.

You must not change the models or their revisions, `package/` (plan, cases, protocol),
anything under `slh/`, `configs/` or `scripts/xp/` (the runner refuses), sampling settings,
or any output file. A failed row with its logs is more useful to us than a row made to pass.

## Questions for you

1. Pyxis with the pinned vLLM image, or a separate vLLM 0.30.0 environment?
2. Can you get a Qwen3.5-397B-A17B FP8 checkpoint (which revision), and does it fit next to
   the 9B on 8xH100 80 GB with the default layout (phase 0 measures it)?
3. Can one job run 48 h, and how many GPU-hours per day can you give phase 1?
4. Does your dispatcher start job scripts inside an `srun` step?
