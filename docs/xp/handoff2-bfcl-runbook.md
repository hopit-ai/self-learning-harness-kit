# Hand-off 2 runbook: function-calling benchmark, lesson-inclusion screen (part D)

This file ships in the kit (`docs/xp/handoff2-bfcl-runbook.md`) and is the runbook for
hand-off 2 of the function-calling benchmark. The kit's `README.md` names the tag (a new
tag of `hopit-ai/self-learning-harness-kit`; earlier tags are unchanged) and the SHA-256 of
the package and of every offline asset; `MANIFEST.json` lists the same under `release`.

Everything runs offline on two GPUs of one node with the same 9B agent model as before. No
API calls, no user simulator, no teacher model, no network. One part:

| Part | What | GPUs | GPU-hour cap (H100) | Wall-clock limit (server start to server stop) | Python environment |
|---|---|---|---:|---:|---|
| D | function-calling benchmark: oracle check, then 2,820 episodes on the 30 gate scenarios (base and long context) | 2 | **12** | **6 hours**, server start-up included | `~/bfcl-venv` |

Expected: about 1–2.5 hours on two H100s (2,820 episodes, i.e. 2–4 H100-hours; hand-off 1
ran about 700 episodes per hour on one smaller GPU at low concurrency). **The wall-clock limit counts from the moment the vLLM server is
running, so start-up, the smoke run and the full run all count**; CPU-only steps (setup,
oracle, scoring, export, packing) do not, as long as no server is running.

**What runs.** `package-bfcl-h2/plan.json` holds 47 cells over the same 60 instances (30
gate scenarios x {base, long context}): one cell with no lessons (the anchor, the standard
conversation) and 46 cells in which each instance gets a fixed, randomly drawn subset of a
short lesson list as one system message (32 subsets of the first list of 8 lessons, 14 of
the second list of 4 lessons). The
subsets and texts are fixed in the package; nothing is chosen at run time. The runner runs
the anchor first, then the two lists' cells alternately (while the second list has cells
left), so a run stopped at its limit leaves both lists with as many cells as possible.

**Decoding and output length (fixed by us, nothing to set).** Temperature 0.7, top-p 0.8,
non-thinking, sent with every request; `max_tokens = min(4096, max-model-len - prompt
tokens - 2)` per request (BFCL's own rule for locally served models), prompt tokens counted
with the pinned 9B tokenizer (`SLH_AGENT_TOKENIZER`). A reply cut at that length is a normal
result of its episode, not an error. The server must not add sampling defaults: keep
`--generation-config vllm` and never add `--override-generation-config` (the runner refuses
an endpoint file without the first or with the second, exit 4).

## Stop rules (in this order of precedence)

1. **Limit reached** (6 hours wall clock or 12 H100-hours): stop the run (Ctrl-C), stop the
   server, score, export and pack what has finished ("What to send back"). **Do not rerun
   to finish it unless we ask.** This rule overrides every "rerun" below.
2. **Exit 4** (refused: package, code or endpoint differs from what we registered, or a run
   folder belongs to another run): stop, pack, tell us. Do not work around it.
3. **Exit 3** (halted, reason in `halt.json` under the run folder; usually the server is
   down) or an accidental interruption: fix the cause and rerun the same command, if still
   within the limits. Finished episodes are kept; the rerun's time counts against the same
   limits.

Do not edit packages, code or run folders.

## 0. Setup (CPU only, about 15–25 minutes, no network)

**What you receive.** The kit at the new tag (copy it to the node as it is; it holds
`package-bfcl-h2/`) and five release assets, all in one directory `assets/` next to the kit:

```
kit/        the tag: code, this runbook, package-bfcl-h2/, MANIFEST.json
assets/     offline-assets.json  tau2-bench-<commit>.tar.gz  wheelhouse-linux-x86_64-py312.tar.gz
            wheelhouse-bfcl-linux-x86_64-py312.tar.gz  wheelhouse-bfcl.sha256
```

You also need the 9B weights (`Qwen/Qwen3.5-9B` at the pinned revision; `$M9B`) and your
vLLM 0.30.0 environment from phase 1 (we ship no vLLM).

**Every shell** starts with:

```bash
export LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export M9B=/path/to/models/Qwen3.5-9B
export SLH_AGENT_TOKENIZER=$M9B/tokenizer.json   # the pinned 9B tokenizer (hash-checked)
export VLLM=/path/to/phase1-vllm-env/bin/vllm    # the phase-1 vLLM 0.30.0
cd kit                                           # every command below runs here
```

**Checksum** of the benchmark wheelhouse (`offline-assets.json` covers the other three; the
setup script checks them):

```bash
(cd ../assets && sha256sum -c wheelhouse-bfcl.sha256)
```

**Kit environment** (the preflight only): a new one, or your earlier kit environment reused
as it is:

```bash
python3.12 -m venv .venv
PYTHON=.venv/bin/python scripts/xp/setup_offline.sh ../assets 2>&1 | tee setup-kit.log   # ends "ok": true
PY=.venv/bin/python; LOCK="--lock .offline/wheelhouse/requirements.lock"
# or: PYTHON=/path/to/earlier-env/bin/python scripts/xp/setup_offline.sh ../assets --reuse-env 2>&1 | tee setup-kit.log
#     PY=/path/to/earlier-env/bin/python; LOCK=""
```

**Benchmark environment** (ends with `"ok": true`; if `~/bfcl-venv` exists from an earlier
hand-off, remove it first or use a new path, so it is built from this kit):

```bash
scripts/xp/setup_bfcl_offline.sh ../assets/wheelhouse-bfcl-linux-x86_64-py312.tar.gz ~/bfcl-venv 2>&1 | tee setup-bfcl.log
```

**Disconnected preflight** (every network connection refused while it runs). It must end
with `"ok": true`; otherwise it lists what is missing:

```bash
$PY -m slh.xp.partner.offline preflight-h1 --assets ../assets --wheelhouse .offline/wheelhouse $LOCK \
  --package-bfcl package-bfcl-h2 --vllm $VLLM --model-dir $M9B
~/bfcl-venv/bin/python -m slh.xp.bfcl.runner check          # "ok": true
```

Run the preflight again with `--endpoint endpoint-bfcl.yaml` once you have written that
file (step 1).

## Part D (2 GPUs, ≤ 6 hours, ≤ 12 H100-hours)

CPU only, before the server (a few minutes; every outcome must be a success):

```bash
BPY=~/bfcl-venv/bin/python
$BPY -m slh.xp.bfcl.runner run    --package package-bfcl-h2 --out runs/bfcl-h2-oracle --dry-run --oracle --smoke
$BPY -m slh.xp.bfcl.runner score  --package package-bfcl-h2 --out runs/bfcl-h2-oracle
$BPY -m slh.xp.bfcl.runner export --package package-bfcl-h2 --run runs/bfcl-h2-oracle --out export-bfcl-h2-oracle
```

1. Start the 9B server on two GPUs exactly as in hand-off 1 (the agent line of
   `README-plain.md` §2 as `$VLLM serve ...`, changing only `CUDA_VISIBLE_DEVICES=0,1`,
   `--port 8011`, `--tensor-parallel-size 1`, `--data-parallel-size 2`,
   `--max-num-seqs 128`, `--gpu-memory-utilization 0.85`), logging to `server-D.log`
   (`... 2>&1 | tee server-D.log`). **The 6 hours start now.** Write `endpoint-bfcl.yaml`
   as the `endpoint.yaml` of `README-plain.md` §2 with `base_url` on port 8011,
   `serving_args` the exact command you started, `replicas: 2`, and the teacher block left
   out; rerun the preflight with `--endpoint endpoint-bfcl.yaml`.
2. Run:
   ```bash
   $BPY -m slh.xp.bfcl.runner run --package package-bfcl-h2 --endpoint endpoint-bfcl.yaml --out runs/bfcl-h2-smoke --smoke
   $BPY -m slh.xp.bfcl.runner run --package package-bfcl-h2 --endpoint endpoint-bfcl.yaml --out runs/bfcl-h2 --parallel-cells 4 --threads 32
   ```
   There is no feasibility barrier in this hand-off (it passed in hand-off 1).
3. Ctrl-C (or a SIGTERM from your scheduler) stops every cell process within about 10
   seconds; finished episodes are kept and the same command resumes. If a cell was killed
   while writing a result, the rerun moves that half-written last line to
   `results.jsonl.torn` and carries on (score and export skip it); a damaged line anywhere
   else stops the cell with exit 4 (stop rule 2: do not edit the file, pack and tell us).
4. Stop the server, then, CPU only:
   ```bash
   $BPY -m slh.xp.bfcl.runner score  --package package-bfcl-h2 --out runs/bfcl-h2
   $BPY -m slh.xp.bfcl.runner export --package package-bfcl-h2 --run runs/bfcl-h2 --out export-bfcl-h2
   $BPY -m slh.xp.bfcl.runner score  --package package-bfcl-h2 --out runs/bfcl-h2-smoke
   $BPY -m slh.xp.bfcl.runner export --package package-bfcl-h2 --run runs/bfcl-h2-smoke --out export-bfcl-h2-smoke
   ```

## What to send back

`scripts/xp/pack_export.sh NAME PATH...` packs the paths that exist, refuses an archive
over 100 MB and splits it into parts of at most 10,000,000 bytes:

```bash
scripts/xp/pack_export.sh handoff2-D export-bfcl-h2 export-bfcl-h2-smoke export-bfcl-h2-oracle \
  runs/bfcl-h2/logs setup-kit.log setup-bfcl.log server-D.log
```

Each export also carries the run's own records under `run/` (`identity.json`, each cell's
`status.json`, `halt.json` if it halted, and any `results.jsonl.torn` backup).

If an export or score command fails (or the run stopped early and export refuses), pack
the run folders' records and finished raw results instead (everything under `runs/` except
the large `bfcl_root` folders; it is about 30 MB compressed for a full run):

```bash
scripts/xp/pack_export.sh handoff2-D-partial export-bfcl-h2 export-bfcl-h2-smoke export-bfcl-h2-oracle \
  runs/bfcl-h2/logs runs/bfcl-h2-smoke/logs \
  $(find runs/bfcl-h2 runs/bfcl-h2-smoke -name bfcl_root -prune -o -type f \( -name '*.json' -o -name 'results.jsonl*' -o -name verdicts.jsonl \) -print) \
  setup-kit.log setup-bfcl.log server-D.log
```

Send every `handoff2-*.tar.gz.part-*` and `handoff2-*.sha256` file, plus a short note:
wall-clock time (server start to stop), GPU type and count, any halts, stops or exit codes,
and `endpoint-bfcl.yaml` (it holds no secrets). Expected size: about 20–45 MB compressed (about 95 MB of exports before compression);
if `pack_export.sh` refuses (over 100 MB), tell us before sending (we will send a smaller
export command). Keep the `runs/` folders for 30 days.

## What happens next (our side)

We fit the pre-registered lesson-inclusion analysis on these episodes, choose (or reject)
the lessons of each list, freeze every library, and build hand-off 3 (the confirmation
run on scenarios that no step so far has used).
