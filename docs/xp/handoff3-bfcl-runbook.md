# Hand-off 3 runbook: function-calling benchmark, confirmation on the held-out scenarios (part E)

This file ships in the kit (`docs/xp/handoff3-bfcl-runbook.md`) and is the runbook for
hand-off 3 of the function-calling benchmark. The kit's `README.md` names the tag (a new
tag of `hopit-ai/self-learning-harness-kit`; earlier tags are unchanged) and the SHA-256 of
the packages and of every offline asset; `MANIFEST.json` lists the same under `release`.

Everything runs offline on two GPUs of one node with the same 9B agent model as in
hand-off 2, with the same setup, server and commands. No API calls, no user simulator, no
teacher model, no network. One part:

| Part | What | GPUs | GPU-hour cap (H100) | Wall-clock limit (server start to server stop) | Python environment |
|---|---|---|---:|---:|---|
| E | function-calling benchmark: oracle check, then 5,760 + 240 episodes on the 120 held-out scenarios (base and long context) | 2 | **20** | **4 hours**, server start-up included | `~/bfcl-venv` |

Expected: about 1.5 hours on two H100s (6,000 episodes, about 3 H100-hours; hand-off 2 ran
about 5,000 episodes per hour on two H100s). **The wall-clock limit counts from the moment
the vLLM server is running, so start-up, the smoke run and both full runs all count**;
CPU-only steps (setup, oracle, scoring, export, packing) do not, as long as no server is
running.

**What runs.** Two packages, both over the same 240 instances (120 held-out scenarios x
{base, long context}) that no earlier hand-off has used:

- `package-bfcl-h3/plan.json`: 24 cells = 6 settings x 4 repeats. The settings differ only in
  the system message: none (the standard conversation), one fixed neutral note, three
  fixed lesson lists, and one fixed list with two worked examples (fixed per scenario in the package).
  All texts are fixed in the package; nothing is chosen at run time. The runner starts the cells in
  order (every setting's first repeat, then every setting's second repeat, and so on), up
  to `--parallel-cells` at a time; it does not wait for one repeat to finish before
  starting the next, so a run stopped at its limit can leave settings with different
  numbers of finished episodes. That is fine: finished episodes are kept, and we analyse
  only complete coverage as registered (we tell you if anything needs to be resumed).
- `package-bfcl-h3-official/plan.json`: 1 cell, the standard conversation once more, at
  temperature 0.001 (the benchmark's own setting). It runs after the first package and is
  the first thing to skip if time runs short.

**Decoding and output length (fixed by us, nothing to set).** `package-bfcl-h3`:
temperature 0.7, top-p 0.8; `package-bfcl-h3-official`: temperature 0.001. Both
non-thinking and sent with every request; `max_tokens = min(4096, max-model-len - prompt
tokens - 2)` per request, prompt tokens counted with the pinned 9B tokenizer
(`SLH_AGENT_TOKENIZER`). A reply cut at that length is a normal result of its episode, not
an error. The server must not add sampling defaults: keep `--generation-config vllm` and
never add `--override-generation-config` (the runner refuses an endpoint file without the
first or with the second, exit 4).

## Stop rules (in this order of precedence)

1. **Limit reached** (4 hours wall clock or 20 H100-hours): stop the run (Ctrl-C), stop the
   server, score, export and pack what has finished ("What to send back"). **Do not rerun
   to finish it unless we ask.** This rule overrides every "rerun" below. If the limit is
   reached before `package-bfcl-h3-official` starts, skip it.
2. **Exit 4** (refused: package, code or endpoint differs from what we registered, or a run
   folder belongs to another run): stop, pack, tell us. Do not work around it.
3. **Exit 3** (halted, reason in `halt.json` under the run folder; usually the server is
   down) or an accidental interruption: fix the cause and rerun the same command, if still
   within the limits. Finished episodes are kept; the rerun's time counts against the same
   limits.
4. **Score fails** on any cell (as in hand-off 2): do not work around it; pack with the
   partial command below and tell us.

Do not edit packages, code or run folders.

## 0. Setup (CPU only, about 15–25 minutes, no network)

**What you receive.** The kit at the new tag (copy it to the node as it is; it holds
`package-bfcl-h3/` and `package-bfcl-h3-official/`) and five release assets, all in one directory `assets/` next to the kit:

```
kit/        the tag: code, this runbook, package-bfcl-h3/, package-bfcl-h3-official/, MANIFEST.json
assets/     offline-assets.json  tau2-bench-<commit>.tar.gz  wheelhouse-linux-x86_64-py312.tar.gz
            wheelhouse-bfcl-linux-x86_64-py312.tar.gz  wheelhouse-bfcl.sha256
```

**The assets are the same as hand-off 2's**: the same five files with the same SHA-256
(compare with the kit's `README.md` / `MANIFEST.json`). If you still have hand-off 2's
`assets/` folder, reuse it as it is; nothing new to download.

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
  --package-bfcl package-bfcl-h3 --vllm $VLLM --model-dir $M9B
~/bfcl-venv/bin/python -m slh.xp.bfcl.runner check          # "ok": true
```

Run the preflight again with `--endpoint endpoint-bfcl.yaml` once you have written that
file (step 1).

## Part E (2 GPUs, ≤ 4 hours, ≤ 20 H100-hours)

CPU only, before the server (a few minutes; every outcome must be a success):

```bash
BPY=~/bfcl-venv/bin/python
for P in package-bfcl-h3 package-bfcl-h3-official; do
  $BPY -m slh.xp.bfcl.runner run    --package $P --out runs/$P-oracle --dry-run --oracle --smoke
  $BPY -m slh.xp.bfcl.runner score  --package $P --out runs/$P-oracle
  $BPY -m slh.xp.bfcl.runner export --package $P --run runs/$P-oracle --out export-$P-oracle
done
```

1. Start the 9B server on two GPUs (the same command as in hand-off 2; if you launch
   vLLM through your own wrapper script, as then, keep doing so with the same arguments).
   **The 4 hours start now.**
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 $VLLM serve $M9B \
     --served-model-name qwen3.5-9b-vllm --dtype bfloat16 --max-model-len 131072 \
     --max-num-seqs 128 --gpu-memory-utilization 0.85 --language-model-only \
     --generation-config vllm --enable-auto-tool-choice --tool-call-parser qwen3_coder \
     --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
     --return-tokens-as-token-ids --logprobs-mode processed_logprobs \
     --host 127.0.0.1 --port 8011 --uvicorn-log-level warning \
     --tensor-parallel-size 1 --data-parallel-size 2 \
     --enable-prefix-caching --enable-chunked-prefill 2>&1 | tee server-E.log
   ```
   Then write `endpoint-bfcl.yaml` (hand-off 2's file can be reused unchanged if the
   command is identical). `serving_args` is the exact command you started, one list item
   per argument, with your real program and model paths (your wrapper first if you use
   one); no teacher block:
   ```yaml
   base_url: http://127.0.0.1:8011/v1
   model: qwen3.5-9b-vllm
   api_key_env: null
   model_revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a
   vllm_version: 0.30.0
   serving_args: [/path/to/vllm, serve, /path/to/models/Qwen3.5-9B,
     --served-model-name, qwen3.5-9b-vllm, --dtype, bfloat16, --max-model-len, "131072",
     --max-num-seqs, "128", --gpu-memory-utilization, "0.85", --language-model-only,
     --generation-config, vllm, --enable-auto-tool-choice, --tool-call-parser, qwen3_coder,
     --reasoning-parser, qwen3, --default-chat-template-kwargs, '{"enable_thinking": false}',
     --return-tokens-as-token-ids, --logprobs-mode, processed_logprobs,
     --host, 127.0.0.1, --port, "8011", --uvicorn-log-level, warning,
     --tensor-parallel-size, "1", --data-parallel-size, "2",
     --enable-prefix-caching, --enable-chunked-prefill]
   replicas: 2
   concurrency: 32
   ```
   Wait until the server answers (`curl -s http://127.0.0.1:8011/v1/models`), then rerun the
   preflight with `--endpoint endpoint-bfcl.yaml` (it must end with `"ok": true`).
2. Run, in this order:
   ```bash
   $BPY -m slh.xp.bfcl.runner run --package package-bfcl-h3 --endpoint endpoint-bfcl.yaml --out runs/bfcl-h3-smoke --smoke
   $BPY -m slh.xp.bfcl.runner run --package package-bfcl-h3 --endpoint endpoint-bfcl.yaml --out runs/bfcl-h3 --parallel-cells 4 --threads 32
   $BPY -m slh.xp.bfcl.runner run --package package-bfcl-h3-official --endpoint endpoint-bfcl.yaml --out runs/bfcl-h3-official --parallel-cells 1 --threads 32
   ```
3. Ctrl-C (or a SIGTERM from your scheduler) stops every cell process within about 10
   seconds; finished episodes are kept and the same command resumes (torn last lines are
   handled as in hand-off 2).
4. Stop the server, then, CPU only:
   ```bash
   for R in bfcl-h3:package-bfcl-h3 bfcl-h3-smoke:package-bfcl-h3 bfcl-h3-official:package-bfcl-h3-official; do
     $BPY -m slh.xp.bfcl.runner score  --package ${R#*:} --out runs/${R%%:*}
     $BPY -m slh.xp.bfcl.runner export --package ${R#*:} --run runs/${R%%:*} --out export-${R%%:*}
   done
   ```

## What to send back

```bash
scripts/xp/pack_export.sh handoff3-E export-bfcl-h3 export-bfcl-h3-smoke export-bfcl-h3-official \
  export-package-bfcl-h3-oracle export-package-bfcl-h3-official-oracle \
  runs/bfcl-h3/logs runs/bfcl-h3-official/logs setup-kit.log setup-bfcl.log server-E.log
```

If a score or export command fails (or the run stopped early and export refuses), pack the
run folders' records and finished raw results instead (everything under `runs/` except the
large `bfcl_root` folders):

```bash
scripts/xp/pack_export.sh handoff3-E-partial export-bfcl-h3 export-bfcl-h3-smoke export-bfcl-h3-official \
  export-package-bfcl-h3-oracle export-package-bfcl-h3-official-oracle \
  runs/bfcl-h3/logs runs/bfcl-h3-smoke/logs runs/bfcl-h3-official/logs \
  $(find runs/bfcl-h3 runs/bfcl-h3-smoke runs/bfcl-h3-official -name bfcl_root -prune -o -type f \( -name '*.json' -o -name 'results.jsonl*' -o -name verdicts.jsonl \) -print) \
  setup-kit.log setup-bfcl.log server-E.log
```

Expected size: about 45–60 MB compressed for the normal pack (hand-off 2 was 20 MB for 2,820
episodes); the partial pack may come close to the 100 MB limit. If `pack_export.sh` refuses
(over 100 MB), tell us before sending; do not drop files.

Send every `handoff3-*.tar.gz.part-*` and `handoff3-*.sha256` file, plus a short note:
wall-clock time (server start to stop), GPU type and count, any halts, stops or exit codes,
and `endpoint-bfcl.yaml`. Keep the `runs/` folders for 30 days.

## What happens next (our side)

We run the pre-registered confirmation analysis on these episodes. This is the last
hand-off of the function-calling benchmark.
