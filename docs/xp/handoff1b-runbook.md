# Hand-off 1b runbook (one session, part A only)

This file ships in the kit (`docs/xp/handoff1b-runbook.md`) and is the runbook for
hand-off 1b. The kit's `README.md` names the tag (`xp-h1b-v1` of
`hopit-ai/self-learning-harness-kit`, a new tag; earlier tags, including `xp-h1a-v1`, are
unchanged) and the SHA-256 of the part A package and of every release asset;
`MANIFEST.json` lists the same under `release`. Parts B and C were hand-off 1a; they are
not run again here.

Part A is the decision screen and proposal collection: the frozen 9B agent answers one
decision per unit (decision prefixes for the lesson screen with 16 draws per prefix, the
proposal-collection units, and a small set of difficulty-check units). Everything runs
offline on one 8-GPU node with the same 9B agent model as before. No API calls, no user
simulator, no teacher model, no network (the preflight in section 0 fails if anything
would need it).

| Part | What | GPUs | GPU-hour cap (H100) | Wall-clock limit (server start to server stop) | Python environment |
|---|---|---|---:|---:|---|
| A | decision screen and proposal collection | all 8 | **6.5** | **49 min** (6.5 GPU-h / 8 GPUs = 48 min 45 s), server start-up included | kit environment |

Expected: about 2–6.4 H100-hours (15–48 min of the node), the exact unit count and
estimate are in `README.md` (package-h1 counts) and `package-h1/README-h1.md`. **The
wall-clock limit counts from the moment you start the vLLM server, so server start-up,
the smoke run and the full run all count**; CPU-only steps (setup, preflight, dry run,
exports, packing) do not, as long as no server is running.

**Decoding (fixed by us, nothing to set).** Part A decodes exactly as phase 1: **greedy**
(temperature 0) with one seed per decision, the phase-1 decision configuration
(`configs/agent-vllm.yaml`); the runner sends these settings with every request. The
server must not add sampling defaults: keep `--generation-config vllm` in the server line
and never add `--override-generation-config`; the runner refuses an endpoint file without
the first or with the second (exit 4).

## Stop rules (in this order of precedence)

1. **Limit reached** (49 minutes since the server started, or 6.5 H100-hours): **first
   stop the vLLM server** (the clock stops and nothing more can be decided), **then Ctrl-C
   the runner**. The runner stops within about a second: queued units are cancelled, no
   new decision starts, decisions in flight are dropped without a row (exit 130, "stopped:
   ..."); every row already written is a finished decision. Export what has finished and
   pack it (section "What to send back"). **Do not rerun, not even to finish, unless we
   ask.** If we ask, the same run command resumes later and runs only the missing units.
   This rule overrides every "rerun" below.
2. **Exit 4** (refused: package, code or endpoint differs from what we registered, or a
   run folder belongs to another run): stop, pack, tell us. Do not work around it.
3. **Exit 3** (halted, reason in `runs/h1/halt.json`; usually the server is down) or an
   accidental interruption: fix the cause and rerun the same command if you are still
   within the limit. Finished units are kept; the rerun's time counts against the same
   limit.
4. **Setup or preflight fails**: do not start the server; pack the setup log and the
   preflight output (partial commands below) and tell us.

Do not edit the package, the code or run folders.

## 0. Setup (CPU only, about 15 minutes, no network)

**What you receive.** The kit at tag `xp-h1b-v1` (copy it to the node as it is; it holds
`package-h1/`) and five release assets, all in one directory `assets/` next to the kit:

```
kit/        the tag: code, this runbook, package-h1/, MANIFEST.json
assets/     offline-assets.json  tau2-bench-<commit>.tar.gz  wheelhouse-linux-x86_64-py312.tar.gz
            db-ext-v2.json  db-ext-v2.sha256
```

`db-ext-v2.json` is the extended retail database the decision prefixes were built on
(public synthetic users and orders added to the benchmark's own database). It is for
reference only: nothing in this session reads it. Check it once:

```bash
(cd ../assets && sha256sum -c db-ext-v2.sha256)
```

You also need, from phase 1: the 9B weights (`Qwen/Qwen3.5-9B` at the pinned revision, a
directory with `config.json` and `tokenizer.json`; call it `$M9B`) and your vLLM 0.30.0
environment (we ship no vLLM; reuse the phase-1 one, the preflight checks its version).

**Every shell** (the server shell and the run shell) starts with:

```bash
export LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export M9B=/path/to/models/Qwen3.5-9B
export SLH_AGENT_TOKENIZER=$M9B/tokenizer.json   # the pinned 9B tokenizer (hash-checked)
export VLLM=/path/to/phase1-vllm-env/bin/vllm    # the phase-1 vLLM 0.30.0
cd kit                                           # every command below runs here
```

**Kit environment.** Either a new one (`offline-assets.json` covers the benchmark archive
and the wheelhouse; the setup script checks them):

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

**Disconnected preflight** (every network connection refused while it runs; checks the
run-shell variables above, the pinned benchmark, the local price table, the 9B tokenizer,
the package against the code, the 9B weights directory and your vLLM version). The
weights check compares `$M9B` with the pinned revision (config, index and tokenizer files
byte for byte, all four shards present with their sizes); add `--hash-weights` once
(reads about 19 GB) to hash the shards too. It must end with `"ok": true`; otherwise it
lists what is missing:

```bash
$PY -m slh.xp.partner.offline preflight-h1 --assets ../assets --wheelhouse .offline/wheelhouse $LOCK \
  --package-h1 package-h1 --vllm $VLLM --model-dir $M9B 2>&1 | tee preflight-A.log
```

**Dry run** (CPU only, no model, before the server; must end with "export ok"):

```bash
$PY -m slh.xp.fgate.units run --package package-h1 --out runs/h1-dry --dry-run
```

## Part A: decision screen and proposal collection (all 8 GPUs, ≤ 49 min)

1. Start only the 9B server: the agent line of `README-plain.md` §2 (same command and
   settings as phase 1; no teacher), as `$VLLM serve ...`, logging to `server-A.log`
   (`... 2>&1 | tee server-A.log`). **The 49 minutes start now.** Write `endpoint.yaml`
   as there (the teacher block may be left out) and rerun the preflight with
   `--endpoint endpoint.yaml` (appending to `preflight-A.log`).
2. Run (smoke first, then the full run):
   ```bash
   $PY -m slh.xp.fgate.units run --package package-h1 --endpoint endpoint.yaml --out runs/h1-smoke --smoke
   $PY -m slh.xp.fgate.units run --package package-h1 --endpoint endpoint.yaml --out runs/h1
   ```
   Note the clock when the full run starts. If it is not done by minute 49 since the server
   started, stop rule 1 applies (server first, then Ctrl-C the runner).
3. Stop the server (the clock stops). Then, CPU only:
   ```bash
   $PY -m slh.xp.fgate.units export --run runs/h1 --out export-h1
   $PY -m slh.xp.fgate.units export --run runs/h1-smoke --out export-h1-smoke
   ```

## What to send back

`scripts/xp/pack_export.sh NAME PATH...` packs the paths that exist (a missing path is
listed in the archive and on screen, not an error), refuses an archive over 100 MB and
splits it into parts of at most 10,000,000 bytes each, with a `.sha256` file. After a
normal session:

```bash
scripts/xp/pack_export.sh handoff1b-A export-h1 export-h1-smoke runs/h1-dry/export \
  setup-kit.log preflight-A.log server-A.log
```

**After a failure or a stop** the same command works (whatever exists is packed). If an
export command itself fails, or setup failed, pack these instead:

```bash
# setup or preflight failed (nothing ran)
scripts/xp/pack_export.sh handoff1b-setup setup-kit.log preflight-A.log
# part A stopped or failed (smoke or full): exports if any, else halt/identity files and the server log
scripts/xp/pack_export.sh handoff1b-A-partial export-h1 export-h1-smoke runs/h1/halt.json \
  runs/h1/identity.json runs/h1-smoke/halt.json runs/h1-smoke/identity.json \
  preflight-A.log server-A.log
```

Send every `handoff1b-*.tar.gz.part-*` file and every `handoff1b-*.sha256` file, plus a
short note: wall-clock time from server start to server stop, GPU type and count, any
halts, stops or exit codes, and `endpoint.yaml` (it holds no secrets). Expected size well
under 100 MB. If `pack_export.sh` refuses (over 100 MB), tell us before sending. Keep the
`runs/` folders for 30 days.

## What happens next (our side)

We score part A against the keys we hold: the F' lesson screen, the proposal pool for the
consent checker, and the difficulty check (do the decision prefixes on the extended
database behave like those on the benchmark's own users). Hand-off 2 (training) follows.

Known limitation of the difficulty check (disclosed): one address slot family (ambiguous
references by city) is left out of both of its arms, because the extended database has too
few synthetic users of that shape; the check keeps its 180 matched pairs. Nothing changes on
your side.
