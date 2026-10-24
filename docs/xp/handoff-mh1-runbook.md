# Hand-off M-H1 runbook: Hermes Agent on the retail benchmark, three arms, two customer models

This file ships in the kit (`docs/xp/handoff-mh1-runbook.md`). The kit's `README.md` names
the tag (a new tag of `hopit-ai/self-learning-harness-kit`; earlier tags are unchanged) and
the SHA-256 of the package and of every offline asset; `MANIFEST.json` lists the same.

Everything runs offline on one node with self-served models. No API calls, no network
(each episode runs in its own sandbox with no network at all; the models are reached
through a local socket). What runs:

- **Agent:** Hermes Agent (open source, MIT; commit `cb757fac`) driving `Qwen/Qwen3.5-9B`
  (the same 9B weights and revision as in earlier hand-offs).
- **Customer simulator**, two blocks run as separate commands: block `c9b` uses the same 9B
  as the customer; block `c27b` uses `Qwen/Qwen3.8-27B-FP8` at revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- **Three arms per block:** plain Hermes; Hermes learning skills and memories on its own;
  Hermes learning through our gate. Then all three are evaluated on 3,000 held-out
  conversations.

| Step | GPUs | Expected time | Python |
|---|---|---|---|
| 0. Setup and checks | none | 30–45 min | kit env (3.12) + Hermes env (3.14, from the assets) |
| 1. Servers | 8 (4 for the 9B, 4 for the 27B) | start-up | your vLLM 0.30.0 env |
| 2. Smoke runs (both blocks) | as above | 15–30 min | kit env |
| 3. Full runs (both blocks at the same time) | as above | about 1 day (planning figure: 70–190 H100-hours in total) | kit env |
| 4. Export and pack | none | minutes | kit env |

**There is no time or GPU-hour limit: run both blocks to completion.** The smoke run prints
a projection of the full run's hours per block (`projection_full_block_hours`, from the
smoke's mean conversation time with all workers busy). If a full
run looks badly off (about 3× its projection, or repeated crashes), pause it (Ctrl-C; the
same command resumes) and tell us.

## Stop rules (in this order of precedence)

1. **Exit 4** (refused: the package, code, endpoint file or run folder differ from what we
   registered, or the sandbox self-test failed; or an unexpected error, traceback in
   `halt.json`): stop, pack, tell us. Do not work around it.
2. **Exit 5** (a sandbox incident): stop, pack, tell us.
3. **Exit 3** (halted, reason in `halt.json` in the run folder; usually a server stopped
   answering) or an interruption: fix the cause and rerun the same command; it resumes
   (finished episodes are kept; nothing produced while a server was down is recorded).
4. **A run looks badly off** (about 3× the projection, or repeated crashes): pause and tell
   us.

Do not edit the package, the code or the run folders.

## 0. Setup (CPU only, no network on the compute node)

**What you receive.** The kit at the new tag (code only; copy it to the node as it is),
the task data in three pieces `package-mh1.tar.gz.part0`, `.part1`, `.part2` (at most
10 MB each) with `package-mh1.sha256`, handed to you directly by the owner (not on the
release; please keep it private and do not publish it), and three release assets in one
directory `assets-mh1/` next to the kit:

```
cpython-3.14.6-x86_64-unknown-linux-gnu-install_only.tar.gz
wheelhouse-hermes-linux-x86_64-py314.tar.gz
hermes-assets.sha256
```

You also need, from earlier hand-offs: the 9B weights (`$M9B`), your vLLM 0.30.0 env, the
kit environment and its assets (`assets/` of hand-off 2/3, unchanged). New: the 27B FP8
weights at the pinned revision (`$M27B`), downloaded the way you downloaded the 9B.

**Hermes source** (on the login node, which can reach GitHub; about 400 MB):

```bash
git clone https://github.com/NousResearch/hermes-agent $WORK/hermes-agent
git -C $WORK/hermes-agent checkout cb757fac59c92b49e9b8f2910e5cd7ab75efffdd
```

**Every shell** starts with:

```bash
export LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export M9B=/path/to/models/Qwen3.5-9B M27B=/path/to/models/Qwen3.8-27B-FP8
export VLLM=/path/to/phase1-vllm-env/bin/vllm
export HERMES_CLONE=$WORK/hermes-agent HERMES_VENV=$WORK/hermes-env/venv
export SLH_EPISODE_TMP=/tmp/mh1     # short node-local path (Unix sockets: max 107 bytes)
mkdir -p $SLH_EPISODE_TMP
PY=/path/to/kit-env/bin/python      # the kit environment of earlier hand-offs
cd kit                              # every command below runs here
```

**First: the task data** (verifies the file against the checksum pinned in the kit, then extracts it
to `package-mh1/`; ends with `"ok": true`):

Put the three pieces and `package-mh1.sha256` next to the kit, then (one line to rejoin
and check):

```bash
(cd .. && cat package-mh1.tar.gz.part0 package-mh1.tar.gz.part1 package-mh1.tar.gz.part2 > package-mh1.tar.gz && sha256sum -c package-mh1.sha256)
$PY -m slh.system.hosts.hermes.mh1 unpack-package --tarball ../package-mh1.tar.gz --out package-mh1
```

**Then: the sandbox check.** Each episode runs under bubblewrap with its own (empty)
network and a read-only view of everything except its own folder. This needs `bwrap`
(package `bubblewrap`) and unprivileged user namespaces on the compute nodes. Run this on
a compute node before setting anything else up (the `sandbox` block must show `"backend": "bwrap"`
and a self-test with `"ok": true`; the Hermes and endpoint parts may still be missing):

```bash
$PY -m slh.system.hosts.hermes.mh1 preflight --package package-mh1 2>&1 | tee preflight-sandbox.log
```

This one check exits 4 because Hermes is not set up yet; that is expected here (the stop
rules apply from the next preflight on). If the sandbox block is not ok, stop here and
tell us (the kit has no unsandboxed mode).

**Hermes environment** (ends with `{"ok": true, ...}`):

```bash
scripts/xp/setup_hermes_offline.sh ../assets-mh1 $HERMES_CLONE $WORK/hermes-env 2>&1 | tee setup-hermes.log
```

**Preflight** (sandbox self-test with live probes, package checksums, Hermes pin; must end
with `"ok": true` once the servers are up; before that it lists the endpoints as missing):

```bash
$PY -m slh.system.hosts.hermes.mh1 preflight --package package-mh1 \
  --hermes-clone $HERMES_CLONE --hermes-venv $HERMES_VENV 2>&1 | tee preflight-cpu.log
```

**Dry run** (CPU only, about 5–15 minutes: a stub model and a scripted customer through the
real sandbox, Hermes and every stage; the results mean nothing, only that it works):

```bash
$PY -m slh.system.hosts.hermes.mh1 dry-run --package package-mh1 --out runs/mh1-dry \
  --hermes-clone $HERMES_CLONE --hermes-venv $HERMES_VENV --work-root $SLH_EPISODE_TMP \
  2>&1 | tee dry-run.log            # ends with "ok": true
$PY -m slh.system.hosts.hermes.mh1 export --run runs/mh1-dry --out export-mh1-dry
```

## 1. Servers (8 GPUs)

9B on GPUs 0–3 (agent for both blocks and the `c9b` customer), the same arguments as in
earlier hand-offs plus data-parallel 4:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 $VLLM serve $M9B \
  --served-model-name qwen3.5-9b-vllm --dtype bfloat16 --max-model-len 131072 \
  --max-num-seqs 128 --gpu-memory-utilization 0.85 --language-model-only \
  --generation-config vllm --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  --host 127.0.0.1 --port 8011 --uvicorn-log-level warning \
  --tensor-parallel-size 1 --data-parallel-size 4 \
  --enable-prefix-caching --enable-chunked-prefill 2>&1 | tee server-9b.log
```

27B FP8 on GPUs 4–7 (customer of block `c27b`; the checkpoint's own FP8 quantization):

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 $VLLM serve $M27B \
  --served-model-name qwen3.8-27b-fp8-vllm --max-model-len 32768 \
  --max-num-seqs 128 --gpu-memory-utilization 0.85 --language-model-only \
  --generation-config vllm --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --host 127.0.0.1 --port 8012 --uvicorn-log-level warning \
  --tensor-parallel-size 1 --data-parallel-size 4 \
  --enable-prefix-caching --enable-chunked-prefill 2>&1 | tee server-27b.log
```

Then write `endpoint-mh1.yaml`; `serving_args` is the exact command you started, one list
item per argument, with your real program and model paths (your wrapper first if you use
one):

```yaml
agent:
  base_url: http://127.0.0.1:8011/v1
  model: qwen3.5-9b-vllm
  model_revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a
  vllm_version: 0.30.0
  serving_args: [/path/to/vllm, serve, /path/to/models/Qwen3.5-9B, ...]
customers:
  c9b:
    base_url: http://127.0.0.1:8011/v1
    model: qwen3.5-9b-vllm
    model_revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a
    vllm_version: 0.30.0
    serving_args: [/path/to/vllm, serve, /path/to/models/Qwen3.5-9B, ...]
  c27b:
    base_url: http://127.0.0.1:8012/v1
    model: qwen3.8-27b-fp8-vllm
    model_revision: 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
    vllm_version: 0.30.0
    serving_args: [/path/to/vllm, serve, /path/to/models/Qwen3.8-27B-FP8, ...]
```

Wait until both answer (`curl -s http://127.0.0.1:8011/v1/models`, port 8012 likewise),
then rerun the preflight with the endpoint file for each block (both must end
`"ok": true`):

```bash
for B in c9b c27b; do
  $PY -m slh.system.hosts.hermes.mh1 preflight --package package-mh1 --block $B \
    --endpoint endpoint-mh1.yaml --hermes-clone $HERMES_CLONE --hermes-venv $HERMES_VENV \
    2>&1 | tee preflight-$B.log
done
```

## 2. Smoke runs, then 3. full runs

Run the two smoke runs one after the other (the first one unpacks the retail database
once). The two full runs are independent; run them at the same time in two shells
(`B=c9b` in one, `B=c27b` in the other). `--workers` is the number of conversations in flight per block (64;
each needs about 0.3 GB of host memory plus one Hermes process).

```bash
COMMON="--package package-mh1 --endpoint endpoint-mh1.yaml --hermes-clone $HERMES_CLONE \
  --hermes-venv $HERMES_VENV --work-root $SLH_EPISODE_TMP --workers 64"
$PY -m slh.system.hosts.hermes.mh1 run --block $B --smoke --out runs/mh1-$B-smoke $COMMON 2>&1 | tee run-$B-smoke.log
$PY -m slh.system.hosts.hermes.mh1 run --block $B --out runs/mh1-$B $COMMON 2>&1 | tee -a run-$B.log
```

**The smoke run is a gate.** It first sends one tiny request to each model (the replies
must carry no thinking, and the customer's must be a valid structured consent act), then
runs a small version of everything, and writes `runs/mh1-$B-smoke/smoke.json` with the
episode counts per status, the consent-act validity rate and the projection. It ends with
`"ok": true` only if it passed; otherwise exit 4 with the reasons (stop, pack, tell us).
**The full run of a block refuses to start unless that block's smoke run passed** with
the same package, code and endpoint file.

The full run goes through four stages in order and prints progress: learning through the
gate (3 completed cycles), learning without the gate (3 batches on the same evidence
conversations), the held-out evaluation (9,000 conversations), and a smaller trajectory
evaluation. Ctrl-C (or a SIGTERM from your
scheduler) stops it within seconds (conversations in flight are killed and not recorded);
the same command resumes. A run that ends prints `"ok": true` and
writes `done.json` in its folder.

## 4. Export and send back (CPU)

Stop the servers, then:

```bash
for R in mh1-c9b mh1-c27b mh1-c9b-smoke mh1-c27b-smoke; do
  $PY -m slh.system.hosts.hermes.mh1 export --run runs/$R --out export-$R
done
scripts/xp/pack_export.sh handoff-mh1 export-mh1-c9b export-mh1-c27b \
  export-mh1-c9b-smoke export-mh1-c27b-smoke export-mh1-dry \
  setup-hermes.log preflight-sandbox.log preflight-cpu.log preflight-c9b.log preflight-c27b.log dry-run.log \
  run-c9b-smoke.log run-c27b-smoke.log run-c9b.log run-c27b.log \
  server-9b.log server-27b.log endpoint-mh1.yaml
```

Expected size: 10–30 MB in 10 MB parts. Export works on an unfinished run too (it marks
the export incomplete), so after an exit 4 or 5 export and pack the same way. If
`pack_export.sh` refuses (over 100 MB), tell us before sending; do not drop files.

Send every `handoff-mh1.tar.gz.part-*` and `handoff-mh1.sha256`, plus a short note:
wall-clock time per block, GPU type and count, any halts, stops or exit codes. Keep the
`runs/` folders for 30 days.

## What happens next (our side)

We verify the checksums and the run identity, then run the pre-registered analysis.
