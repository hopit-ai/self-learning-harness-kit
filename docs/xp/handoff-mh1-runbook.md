# Hand-off M-H1 runbook: Hermes Agent on the retail benchmark, three arms, two customer models

**Version 2.6** (kit tag `mh1-2026-10-25-v2.6`, plan `mh1-2026-10-25-v2.6`; the follow-up
tag `mh1-2026-10-25-v2.6.1` adds only the text-report tool of section 4, the run is
unchanged). Fixed in 2.6:
the exit 5 in the `c27b` learning of your 2.5 run. With 64 conversations in one process,
Python's memory clean-up walked every conversation's copy of the retail database and froze
the process for tens of seconds, so our tool-call check answered Hermes later than its 30 s
limit; Hermes then refused that call and skipped our check for a minute, and the run stopped
(nothing ran without approval). Now all conversations share one copy of the database and each keeps only
the records it changes, Hermes waits up to 300 s for the check, a refusal Hermes makes after
such a timeout is recorded instead of stopping the run, and the smoke run first tests the
check under the full worker count (it fails if one answer takes over 10 s). Everything else
is as in 2.5. New in 2.5: each block can run on several nodes (section 3b); with one node
nothing changes. It replaces versions 2.5 and earlier. Fixed in 2.4: the cause of the exit 5 in the 2.1/2.2 smoke runs: when the
model edits an existing memory note in the after-conversation review, Hermes takes a lock
on a small file next to the memory file; the sandbox keeps that folder read-only, the edit
failed, and Hermes answered every other call of that turn with an error without passing
them through our check. Version 2.4 creates those lock files up front as the only writable
files in that folder (they must stay empty; anything else is still an incident). Kept from
2.3: Hermes' own refusals (a malformed call, an unknown tool name) are recorded instead of
stopping the run (the smoke report lists them by kind), and a stopped conversation is
ALWAYS kept in the run folder (`incident-episodes/`). Kept from 2.2 and 2.1: Hermes'
argument repair before our guard; the locked Hermes install; `/home` symlinks. New task
data pieces (the same tasks). Start fresh in a NEW directory. The three
release assets are unchanged (reuse `assets-mh1/`).

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
- **A small extra measurement, block `c9b` only** (in the same command, after everything
  above): plain Hermes conversations that are recorded and then continued a few different
  ways from the same point (how much of the earlier context the agent needs, and how it
  should react to a failed lookup). Nothing for you to do; it adds a few hours to the
  `c9b` run.

| Step | GPUs | Expected time | Python |
|---|---|---|---|
| 0. Setup and checks | none | 30–45 min | kit env (3.12) + Hermes env (3.14, from the assets) |
| 1. Servers | 8 (4 for the 9B, 4 for the 27B) | start-up | your vLLM 0.30.0 env |
| 2. Smoke runs (both blocks) | as above | 15–30 min | kit env |
| 3. Full runs (both blocks at the same time) | as above | about 1 day (planning figure: 76–242 H100-hours in total) | kit env |
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
2. **Exit 5** (a sandbox incident or a mediation failure): stop, pack, tell us. The run
   folder keeps a copy of every episode that stopped the run, with our side's records of
   every tool decision (`runs/<run>/incident-episodes/`, at most 50;
   the episode work folder itself is often under a node's `/tmp`, which may be cleaned):
   pack it separately, `scripts/xp/pack_export.sh handoff-mh1-incident
   runs/<run>/incident-episodes`, in the same directory as the main pack, and send it with
   the rest (both packs together must stay under 20 MB; the script checks the total).
3. **Exit 3** (halted, reason in `halt.json` in the run folder; usually a server stopped
   answering) or an interruption: fix the cause and rerun the same command; it resumes
   (finished episodes are kept; nothing produced while a server was down is recorded).
4. **A run looks badly off** (about 3× the projection, or repeated crashes): pause and tell
   us.

Do not edit the package, the code or the run folders.

## 0. Setup (CPU only, no network on the compute node)

**What you receive.** The kit at the new tag (code only; copy it to the node as it is),
the version 2.1 task data in three pieces `package-mh1.tar.gz.part0`, `.part1`, `.part2` (at
most 10 MB each) with `package-mh1.sha256`, handed to you privately by the owner (not on
the release; please keep it private and do not publish it), and three release assets in
one directory `assets-mh1/` next to the kit (the same three files as for version 1):

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

**Hermes environment** (ends with `{"ok": true, ...}`). It installs the locked package
set exactly as listed (no re-resolution), then runs `pip check`: it prints Hermes' own
version overrides and accepts only the mismatch they explain (today one line,
browser-harness asking for pillow 12.3.0 while Hermes pins Pillow 13.0.0); any other
mismatch stops it (exit 4: tell us):

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
conversations), the held-out evaluation (9,000 conversations, up to 18,000 when some learned changes wait for our
review: they are evaluated too), and a smaller trajectory evaluation. Block `c9b` then runs
a fifth stage, `probe` (about 400 recorded conversations and about 12,000 shorter
continuations; it prints `probe base: ...` and `probe branches: ...`). It runs only if the
`c9b` smoke run's probe part reproduced its recorded conversations exactly; otherwise it
is skipped and says so (that needs nothing from you). If only the probe fails, the run
still ends with `"ok": true` and the failure is in `runs/mh1-c9b/probe/error.json`. Ctrl-C (or a SIGTERM from your
scheduler) stops it within seconds (conversations in flight are killed and not recorded);
the same command resumes. A run that ends prints `"ok": true` and
writes `done.json` in its folder. Some learned changes may wait for a review on our side;
that is expected and needs nothing from you: the run continues and the export carries
them.

## 3b. Several nodes (optional, version 2.5)

You can run each block's full run on N nodes (N = 1 to 4; above 4 the return could pass
20 MB, so the kit refuses). Learning stays on one node
(it is sequential); once it is done, the held-out evaluation, the trajectory evaluation and
the probe are split into N shards by a fixed hash of each conversation, one node each. Every
node starts its own servers (section 1) and needs the same package, kit, Hermes clone and
endpoint file, and a shared filesystem for the `runs/` folders. **N = 1 is exactly the
one-node run above**: leave `--shards` out. The smoke runs always run on one node.

Per block (`B=c9b` or `B=c27b`), N nodes:

```bash
# 1) the learning node (one job): learning, then it writes runs/mh1-$B/learned/
$PY -m slh.system.hosts.hermes.mh1 run --block $B --shards $N --out runs/mh1-$B $COMMON
# 2) N eval shards (I = 0 .. N-1, one node each), after the learning job ended "ok": true
$PY -m slh.system.hosts.hermes.mh1 run --block $B --shards $N --shard $I \
  --learned runs/mh1-$B --out runs/mh1-$B-s$I $COMMON
```

With Slurm (one script `node.sh` that starts the servers of section 1, waits for them, then
runs the command it is given):

```bash
L=$(sbatch --parsable -N1 --gres=gpu:8 node.sh learn $B $N)
sbatch -N1 --gres=gpu:8 --array=0-$((N-1)) --dependency=afterok:$L node.sh shard $B $N
# in node.sh: shard I = $SLURM_ARRAY_TASK_ID
```

Launch ONE Slurm step per node (one `sbatch` job per node, as above, each running its own
commands inside; or one `srun` per node from inside an allocation), not many steps from the
login node: many concurrent steps can exhaust the login node's ports.

If the learning job stops (exit 3) and you resubmit it, the pending eval array still waits
on the OLD job id and will never start: cancel it and submit the array again with
`--dependency=afterok:<new learning job id>`. To resume one eval shard, resubmit only that
array index (`--array=2`) with the same N; never rerun learning to resume a shard.

Each eval shard copies the learning node's states into its own folder, checks them against
their hashes, and refuses if they belong to another run. Every job resumes with the same
command (Ctrl-C or a scheduler stop, as above). Export every folder: `runs/mh1-$B` (the
learning node) and `runs/mh1-$B-s0` .. `runs/mh1-$B-s<N-1>`, and pack them **all into the
one `handoff-mh1` pack** of section 4 (list every export folder); the 20 MB limit is one
total for everything returned, whatever N (it holds by construction for N up to 4).
Byte shares per block: the learning export about 2 MB (learning, not trimmed) and at most
0.4 MB of probe rows; each eval shard 1/N of the remaining 3 MB, a probe share sized for
the largest shard (about 0.6 MB at N = 4), and one sealed file of a fixed size (about
0.5 MB at N = 4). `pack_export.sh` keeps all log tails together under 1 MB, so list every
node's logs.

**Biggest saving first: put the two blocks on separate nodes.** Today both blocks share one
node (the 9B on GPUs 0-3 serves both agents and the `c9b` customer, the 27B on GPUs 4-7).
Learning does not get faster with more eval nodes, but each block's learning gets twice
the GPUs on a node of its own (`c9b`: the 9B on all 8 GPUs, `--data-parallel-size 8`;
`c27b`: the 9B and the 27B on 4 GPUs each, as in section 1). The two blocks are
independent: run them at the same time, each with its own learning node and, optionally,
its own N eval nodes (at most 2N nodes at once).

Expected wall-clock at 10 conversations per GPU-minute (the smoke run measures your real
rate; at 5 double these, at 20 halve them), from the first full-run job to the last, with
both blocks running at the same time (range: no learned change waits for review .. every
cycle runs its full confirmation; excluded: queueing, server start, smoke runs, packing):

| layout | nodes at most | both blocks done |
|---|---|---|
| one node for both blocks (as before) | 1 | 11.4 – 30.5 h |
| one node per block, N = 1 | 2 | 5.7 – 15.3 h |
| one learning node per block, then N = 2 eval nodes per block | 4 | 3.9 – 12.5 h |
| N = 4 per block | 8 | 2.9 – 11.1 h |

Per block with its own nodes (learning on one 8-GPU node, then N eval nodes):

| block | N = 1 | N = 2 | N = 4 |
|---|---|---|---|
| `c9b` (with the probe) | 5.7 – 15.3 h | 3.9 – 12.5 h | 2.9 – 11.1 h |
| `c27b` | 4.3 – 13.9 h | 3.2 – 11.8 h | 2.6 – 10.8 h |

Learning alone is 2.0 – 9.7 h per block on its own node; N = 2 to 4 gives most of what
sharding can add. If a shard stops with exit 3, fix the cause and rerun that shard's
command (only that array index); never rerun learning to resume a shard. For exit 4 or 5,
stop every job of the block, export and pack what exists, and tell us.

## 4. Report back as TEXT (version 2.6.1; no files are sent)

Do not pack or send any file. Stop the servers, export every run folder (CPU), then make
two short text reports with the kit's report tool (tag `mh1-2026-10-25-v2.6.1`: the v2.6
kit plus this tool; the run and its data package are exactly v2.6's, so a run made with
the v2.6 kit is reported the same way):

```bash
for R in mh1-c9b mh1-c27b mh1-c9b-smoke mh1-c27b-smoke; do
  $PY -m slh.system.hosts.hermes.mh1 export --run runs/$R --out export-$R
done
# multi-node (section 3b): export the learning folder and every shard folder too, and
# give all of a block's exports, comma separated, to --block below
$PY -m slh.system.hosts.hermes.mh1_text part1 --package package-mh1 \
  --block c9b=export-mh1-c9b --block c27b=export-mh1-c27b \
  --smoke c9b=export-mh1-c9b-smoke --smoke c27b=export-mh1-c27b-smoke > report-part1.txt
```

1. Paste `report-part1.txt` into a message to us, exactly as printed (under one page; it
   ends with a `CHECK part1 ...` line that lets us detect copying errors). It holds the run
   identity, completion and timing, our integrity checks, the smoke summary and the
   learned changes that wait for our review, but no results.
2. **Wait for our reply.** We review the held changes and send back one decisions line,
   for example `c9b:k1=reject c27b:k2=approve` (or `none`), and the CHECK value of your
   part 1.
3. Then run, with exactly that line and value:

```bash
$PY -m slh.system.hosts.hermes.mh1_text part2 --package package-mh1 \
  --block c9b=export-mh1-c9b --block c27b=export-mh1-c27b \
  --decisions "<the line we sent>" --part1-check <the value we sent> > report-part2.txt
```

4. Paste `report-part2.txt` (under two pages, ends with `CHECK part2 ...`). It holds the
   pre-registered analysis and the probe summary; it takes a few minutes to compute.

After an exit 4 or 5, export what exists and make part 1 the same way (it reports the
stop); paste it and wait for us. Add a short note: wall-clock time per block, GPU type and
count, any halts, stops or exit codes. **Keep the `runs/` and `export-*` folders for at
least 60 days**: we may ask for one more number from them later.

### 4b. Part 3: the probe from the run folders (version 2.6.2)

The `c9b` probe ran, but its rows did not fit the byte share an export file allows, so the
exports left them out (the run folders still hold them). With text there is no such limit.
With the kit tag `mh1-2026-10-25-v2.6.3` (the same run code as v2.6; v2.6.2 crashed on some rows), on the node(s) that
hold the `runs/` folders (CPU only, a few minutes, no servers needed):

```bash
# one node per block:  --run c9b=runs/mh1-c9b
# several nodes:       every run folder of the block, the learning one first
$PY -m slh.system.hosts.hermes.mh1_text probe --package package-mh1 \
  --run c9b=runs/mh1-c9b,runs/mh1-c9b-s0,runs/mh1-c9b-s1 \
  --smoke-run c9b=runs/mh1-c9b-smoke > report-part3.txt
```

Paste `report-part3.txt` (about one page, ends with `CHECK part3 ...`). It needs no
decision from us and reads no evaluation result. The decisions line may be `none` (or empty)
when no change waits for review.

## What happens next (our side)

We check each pasted part's CHECK line and its identity line, decide the held changes from
part 1 alone, record those decisions before asking for part 2, and read the pre-registered
analysis from part 2.
