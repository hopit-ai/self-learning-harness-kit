# Hand-off 1a runbook (one session, parts B and C)

This file ships in the kit (`docs/xp/handoff1a-runbook.md`) and is the runbook for
hand-off 1a. The kit's `README.md` names the tag (`xp-h1a-v1` of
`hopit-ai/self-learning-harness-kit`, a new tag; earlier tags are unchanged) and the SHA-256
of the part B package and of every offline asset; `MANIFEST.json` lists the same under
`release`. Part A (decision screen) is not in this kit: it follows later as hand-off 1b.

Everything runs offline on two GPUs of one node with the same 9B agent model as before. No
API calls, no user simulator, no teacher model, no network (the preflight in section 0
fails if anything would need it). Two parts, run in this order:

| Part | What | GPUs | GPU-hour cap (H100) | Wall-clock limit (server start to server stop) | Python environment |
|---|---|---|---:|---:|---|
| B | function-calling benchmark: oracle check, feasibility barrier, discovery baseline | 2 | 1 | **30 min**, server start-up included | `~/bfcl-venv` |
| C | household text games: offline dependency build (CPU), then feasibility on 48 games | the same 2 | 1 | **30 min** after part B's run ends (or from a new server start, start-up included) | `~/alfworld-venv` |

Total cap 2 GPU-hours (expected about 0.5–1.5). **The wall-clock limit counts from the
moment a vLLM server is running for that part, so server start-up, the smoke run and the
full run all count**; CPU-only steps (setup, dry runs, scoring, exports, packing) do not, as
long as no server is running.

**Decoding (fixed by us, nothing to set).** Parts B and C sample with temperature 0.7 and
top-p 0.8; each runner sends these settings with every request. The server must not add
sampling defaults: keep `--generation-config vllm` in the server line and never add
`--override-generation-config`; the runners refuse an endpoint file without the first or
with the second (exit 4).

## Stop rules (in this order of precedence)

1. **Limit reached** (a part's wall-clock limit or GPU-hour cap): stop it (Ctrl-C), stop
   the server, export what has finished and pack it (section "What to send back"). **Do
   not rerun that part, not even to finish it, unless we ask.** This rule overrides every
   "rerun" below.
2. **Exit 4** (refused: package, code or endpoint differs from what we registered, or a
   run folder belongs to another run) or **exit 5** (part B feasibility NO-GO): stop that
   part, pack it, tell us. Do not work around it.
3. **Exit 3** (halted, reason in `halt.json` under the run folder; usually the server is
   down) or an accidental interruption: fix the cause and rerun the same command, if the
   part is still within its limit. Finished work is kept; the rerun's time counts against
   the same limit.
4. **Setup fails** for one part: pack its logs (partial commands below) and carry on with
   the other part. The parts are independent.

Do not edit packages, code or run folders.

## 0. Setup (CPU only, about 20–30 minutes, no network)

**What you receive.** The kit at tag `xp-h1a-v1` (copy it to the node as it is; it holds
`package-bfcl-h1/` for part B; part C needs no package) and seven release assets, all in
one directory `assets/` next to the kit:

```
kit/        the tag: code, this runbook, package-bfcl-h1/, MANIFEST.json
assets/     offline-assets.json  tau2-bench-<commit>.tar.gz  wheelhouse-linux-x86_64-py312.tar.gz
            wheelhouse-bfcl-linux-x86_64-py312.tar.gz  wheelhouse-bfcl.sha256
            alfworld-offline-linux-x86_64-py312.tar.gz  alfworld-offline.sha256
```

The first three set up the kit environment, which runs only the preflight. You also need,
from phase 1: the 9B weights (`Qwen/Qwen3.5-9B` at the pinned revision, a directory with
`config.json` and `tokenizer.json`; call it `$M9B`) and your vLLM 0.30.0 environment (we
ship no vLLM; reuse the phase-1 one, the preflight checks its version).

**Every shell** (the server shell and the run shells) starts with:

```bash
export LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export M9B=/path/to/models/Qwen3.5-9B
export SLH_AGENT_TOKENIZER=$M9B/tokenizer.json   # the pinned 9B tokenizer (hash-checked)
export VLLM=/path/to/phase1-vllm-env/bin/vllm    # the phase-1 vLLM 0.30.0
cd kit                                           # every command below runs here
```

**Checksums** of the two archives with their own checksum files (`offline-assets.json`
covers the other three; the setup script checks them):

```bash
(cd ../assets && sha256sum -c wheelhouse-bfcl.sha256 alfworld-offline.sha256)
```

**Game environment for part C first** (needs `python3.12`, `gcc`, `g++`, `make`; CPU
only, about 5–15 minutes; **start it in a second shell now and let it run** while you do
the rest of the setup and part B). It writes `alfworld-build/` (build log, installed
versions and licences, check output) and ends with `"ok": true`:

```bash
scripts/xp/setup_alfworld_offline.sh ../assets/alfworld-offline-linux-x86_64-py312.tar.gz ~/alfworld-venv ~/alfworld-data
```

If it stops with `MISSING: <tool>` load a compiler module and rerun; if it stops with
`SOURCE BUILD FAILED`, skip part C and send `alfworld-build/` (we fix and resend).

**Kit environment** (the preflight only). Either a new one:

```bash
python3.12 -m venv .venv
PYTHON=.venv/bin/python scripts/xp/setup_offline.sh ../assets 2>&1 | tee setup-kit.log   # ends "ok": true
PY=.venv/bin/python; LOCK="--lock .offline/wheelhouse/requirements.lock"
```

or your phase-1 kit environment, reused as it is (only the pinned benchmark is added):

```bash
PYTHON=/path/to/phase1-env/bin/python scripts/xp/setup_offline.sh ../assets --reuse-env 2>&1 | tee setup-kit.log
PY=/path/to/phase1-env/bin/python; LOCK=""
```

**Benchmark environment for part B** (ends with `"ok": true`):

```bash
scripts/xp/setup_bfcl_offline.sh ../assets/wheelhouse-bfcl-linux-x86_64-py312.tar.gz ~/bfcl-venv 2>&1 | tee setup-bfcl.log
```

**Disconnected preflight** (every network connection refused while it runs; checks the
run-shell variables above, the pinned benchmark and packages, the local price table, the
9B tokenizer, the part B package against the code, the 9B weights directory and your
vLLM version). The weights check compares `$M9B` with the pinned revision: config, index
and tokenizer files byte for byte, all four shards present with their sizes, and the
index naming exactly those shards; add `--hash-weights` once (reads about 19 GB, a few
minutes) to hash the shards too. It must end with `"ok": true`; otherwise it lists what is
missing:

```bash
$PY -m slh.xp.partner.offline preflight-h1 --assets ../assets --wheelhouse .offline/wheelhouse $LOCK \
  --package-bfcl package-bfcl-h1 --vllm $VLLM --model-dir $M9B
~/bfcl-venv/bin/python -m slh.xp.bfcl.runner check          # part B environment, "ok": true
```

Run the preflight again with `--endpoint endpoint-bfcl.yaml` once you have written that
file (part B, step 1).

## Part B: function-calling benchmark (2 GPUs, ≤ 30 min, ≤ 1 H100-hour)

CPU only, before the server (seconds; every outcome must be a success, 200 of 200):

```bash
BPY=~/bfcl-venv/bin/python
$BPY -m slh.xp.bfcl.runner run    --package package-bfcl-h1 --out runs/bfcl-h1-oracle --dry-run --oracle
$BPY -m slh.xp.bfcl.runner score  --package package-bfcl-h1 --out runs/bfcl-h1-oracle
$BPY -m slh.xp.bfcl.runner export --package package-bfcl-h1 --run runs/bfcl-h1-oracle --out export-bfcl-h1-oracle
```

1. Start the 9B server on two GPUs: the agent line of `README-plain.md` §2 (same command
   and settings as phase 1; no teacher), as `$VLLM serve ...`, changing only
   `CUDA_VISIBLE_DEVICES=0,1`, `--port 8011`, `--tensor-parallel-size 1`,
   `--data-parallel-size 2`, `--max-num-seqs 128`, `--gpu-memory-utilization 0.85`,
   logging to `server-BC.log` (`... 2>&1 | tee server-BC.log`). **The 30 minutes start
   now.** Write `endpoint-bfcl.yaml` as the `endpoint.yaml` of `README-plain.md` §2 with
   `base_url` on port 8011, `serving_args` the exact command you started, `replicas: 2`,
   and the teacher block left out; rerun the preflight with
   `--endpoint endpoint-bfcl.yaml`.
2. Run:
   ```bash
   $BPY -m slh.xp.bfcl.runner run --package package-bfcl-h1 --endpoint endpoint-bfcl.yaml --out runs/bfcl-h1-smoke --smoke
   $BPY -m slh.xp.bfcl.runner run --package package-bfcl-h1 --endpoint endpoint-bfcl.yaml --out runs/bfcl-h1 --parallel-cells 3 --threads 32
   ```
   The full run first runs the feasibility cell (20 instances) **alone**, then checks the
   pre-declared feasibility go / no-go (every feasibility instance finished, no halt,
   unparsed tool-call steps < 5%, inference errors < 2%) and writes
   `runs/bfcl-h1/feasibility-gate.json`. Only on GO does it start the two discovery cells.
   On NO-GO it starts nothing more and exits 5: score and export what ran, send it, do not
   rerun. The smoke run applies the same barrier to its 2 feasibility instances.
3. Ctrl-C (or a SIGTERM from your scheduler) stops every cell process within about 10
   seconds; finished instances are kept.
4. Leave the server running for part C if part C's setup ended `"ok": true`; otherwise
   stop it. Then, CPU only:
   ```bash
   $BPY -m slh.xp.bfcl.runner score  --package package-bfcl-h1 --out runs/bfcl-h1
   $BPY -m slh.xp.bfcl.runner export --package package-bfcl-h1 --run runs/bfcl-h1 --out export-bfcl-h1
   $BPY -m slh.xp.bfcl.runner score  --package package-bfcl-h1 --out runs/bfcl-h1-smoke
   $BPY -m slh.xp.bfcl.runner export --package package-bfcl-h1 --run runs/bfcl-h1-smoke --out export-bfcl-h1-smoke
   ```

Expected: 200 episodes (20 feasibility + 180 discovery, two trials), about 10–25 minutes.

## Part C: household text games (same 2-GPU server, ≤ 30 min, ≤ 1 H100-hour)

Only if the game environment setup in section 0 ended with `"ok": true`. The dry run is
CPU only and can run any time before. The 30 minutes start when part B's run ends (server
kept running); if you stopped the server, they start when you start it again (same line
and `endpoint-bfcl.yaml` as part B), so start-up counts.

```bash
APY=~/alfworld-venv/bin/python
export ALFWORLD_DATA=~/alfworld-data
# dry run, no model, no GPU: three fixed commands in each of the 48 games; must show 48 games and 0 env_error
$APY -m slh.xp.alfworld.feasibility run --out runs/alfworld-dry --dry-run
$APY -m slh.xp.alfworld.feasibility export --run runs/alfworld-dry --out export-alfworld-dry
$APY -m slh.xp.alfworld.feasibility run --endpoint endpoint-bfcl.yaml --out runs/alfworld-smoke --smoke
$APY -m slh.xp.alfworld.feasibility run --endpoint endpoint-bfcl.yaml --out runs/alfworld-feas --workers 16
```

Stop the server, then
`$APY -m slh.xp.alfworld.feasibility export --run runs/alfworld-feas --out export-alfworld-feas`.
Expected: 48 games, at most 50 commands each, about 5–15 minutes.

## What to send back

`scripts/xp/pack_export.sh NAME PATH...` packs the paths that exist (a missing path is
listed in the archive and on screen, not an error), refuses an archive over 100 MB and
splits it into parts of at most 10,000,000 bytes. After a normal session:

```bash
scripts/xp/pack_export.sh handoff1a-B export-bfcl-h1 export-bfcl-h1-smoke export-bfcl-h1-oracle \
  runs/bfcl-h1/feasibility-gate.json runs/bfcl-h1/logs setup-kit.log setup-bfcl.log server-BC.log
scripts/xp/pack_export.sh handoff1a-C alfworld-build export-alfworld-dry export-alfworld-feas
```

**After a failure or a stop** the same commands work (whatever exists is packed). If an
export command itself fails, pack the run folders' small files and logs instead:

```bash
# setup or preflight failed (nothing ran)
scripts/xp/pack_export.sh handoff1a-setup setup-kit.log setup-bfcl.log alfworld-build
# part B stopped, NO-GO or failed
scripts/xp/pack_export.sh handoff1a-B-partial export-bfcl-h1 export-bfcl-h1-smoke runs/bfcl-h1/logs \
  runs/bfcl-h1/feasibility-gate.json runs/bfcl-h1-smoke/logs runs/bfcl-h1-smoke/feasibility-gate.json \
  server-BC.log
# part C stopped or failed
scripts/xp/pack_export.sh handoff1a-C-partial alfworld-build export-alfworld-dry export-alfworld-feas \
  runs/alfworld-smoke runs/alfworld-feas/identity.json
```

Send every `handoff1a-*.tar.gz.part-*` file and every `handoff1a-*.sha256` file, plus a
short note: wall-clock time of each part (server start to stop), GPU type, any halts,
stops or exit codes, and `endpoint-bfcl.yaml` (it holds no secrets). Expected sizes: B
about 5–30 MB; C under 5 MB. If `pack_export.sh` refuses (over 100 MB), tell us before
sending. Keep the `runs/` folders for 30 days.

## What happens next (our side)

We check part B against the pre-declared go / no-go (oracle passes, unparsed tool calls
< 5%, inference errors < 2%, discovery success 20–80%, at least a quarter of failures
procedural) and write the lesson pools; read part C against its go / no-go (success
25–80%, at least a quarter of failures procedural, no environment errors). Hand-off 1b
(part A: decision screen and proposal collection) follows as its own tag.
