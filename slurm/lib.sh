# XP Slurm jobs: settings and helpers (sourced by phase0.sbatch and phase1.sbatch).
#
# Every setting below can be overridden from the environment (sbatch --export=ALL,
# the default, passes the submitting shell's values). Nothing here names a local
# path: everything lives under WORK, which you choose.
#
# Launch route (reference-kit lessons, see README-phase1.md "Site notes"):
#   - the job script runs on the batch host (no enclosing srun needed);
#   - the vLLM servers start as overlapping job steps (srun --overlap), in the
#     pinned container when SERVE_MODE=container (pyxis --container-image), or
#     from a vLLM install on PATH / VLLM_BIN when SERVE_MODE=venv;
#   - if this script already runs INSIDE a step (your dispatcher used srun, with or
#     without a container), a nested srun would wait forever for the GPUs that step
#     holds: SERVE_LAUNCH=auto then starts the servers as plain child processes.

: "${WORK:?set WORK to a directory on large shared storage (login and compute nodes)}"
KIT="${KIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PACKAGE="${PACKAGE:-$KIT/package}"
HARNESS_PY="${HARNESS_PY:-$WORK/venv/bin/python}"   # Python >= 3.12 with the kit's dependencies
SITE_PY="${SITE_PY:-$HARNESS_PY}"                    # the same Python 3.12; no system fallback
if ! "$SITE_PY" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; then
  echo "[xp] $SITE_PY is not a Python >= 3.12 (set HARNESS_PY to \$WORK/venv/bin/python)" >&2
  return 4 2>/dev/null || exit 4
fi

# Models (downloaded by slurm/download.sbatch; Hugging Face is reached from compute nodes).
MODELS="${MODELS:-$WORK/models}"
AGENT_MODEL_DIR="${AGENT_MODEL_DIR:-$MODELS/agent}"
TEACHER_ENABLE="${TEACHER_ENABLE:-1}"          # the registered plan has teacher writers/critic
# The teacher is whatever these settings name (recorded in the run identity). Default:
# the owner-approved teacher of deviation XD1 (docs/xp/deviations.md).
TEACHER_REPO="${TEACHER_REPO:-Qwen/Qwen3.8-Flash-Next-FP8}"
TEACHER_REVISION="${TEACHER_REVISION:-236dfdf285828023ca3bcd3f37366c58a3469b13}"
TEACHER_MODEL_DIR="${TEACHER_MODEL_DIR:-$MODELS/teacher}"
TEACHER_NAME="${TEACHER_NAME:-qwen3.8-flash-next-fp8}"

# Serving.
SERVE_MODE="${SERVE_MODE:-container}"          # container | venv
SERVE_LAUNCH="${SERVE_LAUNCH:-auto}"           # auto | srun | direct
CONTAINER_IMAGE="${CONTAINER_IMAGE:-vllm/vllm-openai:v0.30.0}"  # or a .sqsh from `enroot import`
PYXIS_FLAGS="${PYXIS_FLAGS:---no-container-mount-home --container-mounts=$WORK:$WORK}"
VLLM_BIN="${VLLM_BIN:-vllm}"                   # venv mode: the vllm executable (0.30.0)
AGENT_PORT="${AGENT_PORT:-8001}"
AGENT_TP="${AGENT_TP:-4}"
AGENT_DP="${AGENT_DP:-2}"
AGENT_GPU_UTIL="${AGENT_GPU_UTIL:-0.16}"       # shares the GPUs with the teacher (FP8, TP 8)
AGENT_MAX_NUM_SEQS="${AGENT_MAX_NUM_SEQS:-32}"
TEACHER_PORT="${TEACHER_PORT:-8002}"
TEACHER_TP="${TEACHER_TP:-8}"
TEACHER_GPU_UTIL="${TEACHER_GPU_UTIL:-0.68}"      # 0.68 + the 9B's 0.16 fit (engram tables on GPU)
TEACHER_EXPERT_PARALLEL="${TEACHER_EXPERT_PARALLEL:-1}"  # MoE experts split over the 8 GPUs
TEACHER_MAX_BATCHED_TOKENS="${TEACHER_MAX_BATCHED_TOKENS:-0}"  # 0 = vLLM's default chunk
LEARNING_UNITS="${LEARNING_UNITS:-12}"  # learning units run concurrently (scheduling only)
TEACHER_MAX_LEN="${TEACHER_MAX_LEN:-131072}"
# CUDA graphs by default: the XD1 teacher records its graphs in seconds (startup ~390 s
# in total on the partner's node) and serves ~150 tok/s per stream against ~5.8 eager
# (a realistic teacher call 9.5 s vs 183 s). Eager-by-default came from an older
# teacher that took 27 min to record graphs. Phase 0 still falls back to eager when
# graphs fail or exceed TEACHER_GRAPHS_STARTUP_S; phase 1 takes the gate's value.
TEACHER_EAGER="${TEACHER_EAGER:-0}"
# Extra teacher flags, split on whitespace without a shell (a JSON value stays one
# token as written). Defaults for the XD1 teacher: text only, and the engram tables
# kept on the GPUs (cpu_offload false: ~5 GiB per GPU less, which helps the 9B fit). TEACHER_EXTRA_ARGS_JSON (a JSON array of strings) replaces it when
# set. Recorded verbatim in the run identity; the gate allowlists the flags.
if [[ -z "${TEACHER_EXTRA_ARGS:-}" ]]; then
  TEACHER_EXTRA_ARGS='--language-model-only --engram-config={"cpu_offload":false}'
fi
TEACHER_EXTRA_ARGS_JSON="${TEACHER_EXTRA_ARGS_JSON:-}"
# SERVE_MODE=venv: VLLM_BIN may be a launcher script, e.g. one that puts the
# environment's own libstdc++ first before exec'ing vllm (a site whose system
# libstdc++ is older than the wheels need):
#   #!/bin/bash
#   export LD_LIBRARY_PATH=/path/to/vllm-env/lib:${LD_LIBRARY_PATH:-}
#   exec /path/to/vllm-env/bin/vllm "$@"
# Its SHA-256 is recorded in the run identity.
CONCURRENCY="${CONCURRENCY:-32}"
export XP_GPUS="${XP_GPUS:-8}" SERVE_CPUS="${SERVE_CPUS:-16}"  # per server step
# Startup timeouts: generous on purpose. The reference node needed 27 min of graph
# capture plus 5 min of loading for one 27B engine set; a 397B FP8 load from shared
# storage can be slower. Raise them rather than cutting anything else.
TEACHER_STARTUP_TIMEOUT_S="${TEACHER_STARTUP_TIMEOUT_S:-7200}"
AGENT_STARTUP_TIMEOUT_S="${AGENT_STARTUP_TIMEOUT_S:-3600}"
# vLLM / torch deadlines x10 (the reference failed on vLLM's 300 s execute deadline).
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3000}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600000}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-3000}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-6000}"

# Caches and temporary files: large ones on WORK; sockets need a SHORT TMPDIR
# (Unix socket paths are limited to 107 bytes), so the job uses node-local /tmp.
export HF_HOME="${HF_HOME:-$WORK/hf-home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORK/cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$WORK/cache/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORK/cache/triton}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # jobs serve local snapshots only
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export TMPDIR="${XP_TMPDIR:-/tmp/xp-$SLURM_JOB_ID}"
  export VLLM_RPC_BASE_PATH="$TMPDIR"
  mkdir -p "$TMPDIR"
fi

XP_PIDS=()

xp_log() { echo "[xp $(date -u +%H:%M:%S)] $*" >&2; }

# Process trees. A TERM to the job script must reach grandchildren too (the bootstrap
# under a subshell, srun steps, servers), and a child that ignores TERM must still go:
# xp_stop_tree snapshots the whole tree first (so reparented grandchildren stay
# tracked), sends TERM, re-collects children born meanwhile, escalates to KILL after
# GRACE seconds and returns only when every tracked process is gone (or 1 if some
# survived KILL for 10 s more).
xp_tree() {
  local p
  echo "$1"
  for p in $(pgrep -P "$1" 2>/dev/null); do xp_tree "$p"; done
}
_xp_alive() {  # alive (not zombie) pids among the arguments
  local p st
  for p in "$@"; do
    st="$(ps -o stat= -p "$p" 2>/dev/null)" || continue
    [[ -n "$st" && "$st" != Z* ]] && echo "$p"
  done
}
xp_stop_tree() {  # xp_stop_tree GRACE PID...
  local grace="$1"; shift
  local pids="" p alive t start=$SECONDS
  for p in "$@"; do pids="$pids $(xp_tree "$p")"; done
  pids="$(printf '%s\n' $pids | sort -un | tr '\n' ' ')"  # one entry per pid
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null
  while :; do
    # shellcheck disable=SC2086
    alive="$(_xp_alive $pids)"
    for p in $alive; do pids="$pids $(xp_tree "$p")"; done
    # shellcheck disable=SC2086
    pids="$(printf '%s\n' $pids | sort -un | tr '\n' ' ')"
    # shellcheck disable=SC2086
    alive="$(_xp_alive $pids)"
    [[ -z "$alive" ]] && return 0
    t=$(( SECONDS - start ))  # grace measured in elapsed seconds, not loop turns
    # shellcheck disable=SC2086
    if [[ $t -ge $grace ]]; then kill -KILL $alive 2>/dev/null; fi
    [[ $t -ge $(( grace + 10 )) ]] && return 1
    sleep 1
  done
}

# xp_bounded SECONDS CMD... : run an executable CMD in its own process group, stop the
# whole group (TERM, grace, KILL) at SECONDS and after it ends, and return only when
# the group is gone (slurm/xpsite.py run-bounded; independent of any bash wait).
xp_bounded() {
  local limit="$1"; shift
  [[ "$limit" -lt 1 ]] && limit=1
  "$SITE_PY" "$KIT/slurm/xpsite.py" run-bounded --seconds "$limit" \
    --grace "${XP_STOP_GRACE_S:-20}" -- "$@"
}
xp_site() { "$SITE_PY" "$KIT/slurm/xpsite.py" "$@"; }

# Decide how a server starts: srun step (optionally in the container) or child process.
xp_launch_prefix() {
  local launch="$SERVE_LAUNCH"
  if [[ "$launch" == "auto" ]]; then
    if [[ -z "${SLURM_JOB_ID:-}" || "$(xp_site in-step)" == "1" ]]; then launch=direct
    else launch=srun; fi
  fi
  if [[ "$launch" == "srun" ]]; then
    local p=(srun --overlap --ntasks=1 --nodes=1 --gres="gpu:$XP_GPUS" --mem=0
             --cpus-per-task="$SERVE_CPUS" --kill-on-bad-exit=1)
    if [[ "$SERVE_MODE" == "container" ]]; then
      # shellcheck disable=SC2206
      p+=(--container-image="$CONTAINER_IMAGE" $PYXIS_FLAGS)
    fi
    printf '%s\n' "${p[@]}"
  fi
  echo "launch=$launch" >> "$RUN_DIR/serve/launch.txt"
}

# xp_serve NAME PORT ARGV... : start a server in the background, log to RUN_DIR/serve.
xp_serve() {
  local name="$1" port="$2"; shift 2
  local argv=("$@")
  if [[ "$SERVE_MODE" == "venv" ]]; then argv[0]="$VLLM_BIN"; fi
  mkdir -p "$RUN_DIR/serve"
  local prefix=() line
  while IFS= read -r line; do prefix+=("$line"); done < <(xp_launch_prefix)
  printf '%s\n' ${prefix[@]+"${prefix[@]}"} "${argv[@]}" > "$RUN_DIR/serve/$name.argv"
  xp_log "starting $name on port $port (${prefix[0]:-direct})"
  VLLM_API_KEY="$XP_VLLM_API_KEY" ${prefix[@]+"${prefix[@]}"} "${argv[@]}" \
    > "$RUN_DIR/serve/$name.log" 2>&1 &
  local pid=$!
  XP_PIDS+=("$pid")
  echo "$pid" > "$RUN_DIR/serve/$name.pid"
}

# xp_wait NAME PORT TIMEOUT : health probe; non-zero if it died or timed out. With
# XP_DEADLINE (epoch) set, the timeout never runs past it (the job's GPU budget).
xp_wait() {
  local name="$1" port="$2" timeout="$3"
  if [[ -n "${XP_DEADLINE:-}" ]]; then
    local left=$((XP_DEADLINE - $(date +%s)))
    [[ $left -lt $timeout ]] && timeout=$left
    [[ $timeout -lt 1 ]] && { xp_log "no budget left to start $name"; return 2; }
  fi
  xp_site wait --name "$name" --port "$port" --pid "$(cat "$RUN_DIR/serve/$name.pid")" \
    --key-env XP_VLLM_API_KEY --timeout "$timeout" --out "$RUN_DIR/serve/$name.startup.json"
}

xp_stop_servers() {
  [[ ${#XP_PIDS[@]} -gt 0 ]] || return 0
  xp_stop_tree "${XP_STOP_GRACE_S:-20}" "${XP_PIDS[@]}" || xp_log "a server survived KILL"
  XP_PIDS=()
}

# The pinned serving values, read by the trusted bootstrap under bytecode isolation
# (the helpers never import slh): XP_SERVE_SPEC names the JSON.
xp_spec() {
  local out="${1:-$RUN_DIR/serve-spec.json}" prefix
  prefix="$(mktemp -d "${TMPDIR:-/tmp}/xp-pyc.XXXXXX")"
  if ! (cd "$KIT" && PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$prefix" \
        "$HARNESS_PY" scripts/xp/xp_boot.py --package "$PACKAGE" --serve-spec) > "$out.tmp"; then
    cat "$out.tmp" >&2; rm -f "$out.tmp"; return 4
  fi
  mv "$out.tmp" "$out"
  export XP_SERVE_SPEC="$out"
}

# xp_gate evaluate|verify ARGS... : the registered phase gate (slh/xp/partner/gate.py)
# through the trusted bootstrap.
xp_gate() {
  local prefix
  prefix="$(mktemp -d "${TMPDIR:-/tmp}/xp-pyc.XXXXXX")"
  (cd "$KIT" && PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$prefix" \
    "$HARNESS_PY" scripts/xp/xp_boot.py --module gate "$@" --package "$PACKAGE")
}

# xp_phase1_state NAME [ID] : print "active <id>", "inactive" or "unknown".
# A job is looked up by its deterministic name (squeue --name, then sacct --name over
# the last 60 days) and by the recorded id (sacct -j). Any failed scheduler query, or a
# recorded id the scheduler has no record of, is "unknown", never "inactive".
XP_LIVE_STATES='PENDING|RUNNING|REQUEUED|REQUEUE_HOLD|REQUEUE_FED|CONFIGURING|COMPLETING|SUSPENDED|RESIZING|SIGNALING|STAGE_OUT|PREEMPTED'
xp_phase1_state() {
  local name="$1" id="${2:-}" out
  out="$(squeue -h --name "$name" -o %i 2>/dev/null)" || { echo unknown; return 0; }
  out="$(echo "$out" | grep -oE '^[0-9]+' | head -n 1)"
  [[ -n "$out" ]] && { echo "active $out"; return 0; }
  out="$(sacct -n -X -P --name "$name" -S now-60days -o JobID,State 2>/dev/null)" \
    || { echo unknown; return 0; }
  out="$(echo "$out" | grep -E "^[0-9]+\|($XP_LIVE_STATES)" | head -n 1 | cut -d'|' -f1)"
  [[ -n "$out" ]] && { echo "active $out"; return 0; }
  if [[ -n "$id" ]]; then
    out="$(sacct -n -X -P -j "$id" -o JobID,State 2>/dev/null)" || { echo unknown; return 0; }
    [[ -n "$out" ]] || { echo unknown; return 0; }   # no record yet: not known to be gone
    echo "$out" | grep -qE "^$id\|($XP_LIVE_STATES)" && { echo "active $id"; return 0; }
  fi
  echo inactive
}

# xp_phase1_complete : the completion marker exists and matches the export archive.
xp_phase1_complete() {
  local marker="$WORK/run_all/phase1.complete.json" archive="$WORK/returns/xp-phase1-export.tar.gz"
  [[ -f "$marker" && -f "$archive" ]] || return 1
  "$SITE_PY" - "$marker" "$archive" <<'PY'
import hashlib, json, sys
m = json.load(open(sys.argv[1]))
sys.exit(0 if m.get("archive_sha256") ==
         hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest() else 1)
PY
}

# xp_gate_env GATE : print the derived settings as KEY=VALUE lines.
xp_gate_env() {
  "$SITE_PY" - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))["derived"]
for k in ("CONCURRENCY", "LEARNING_UNITS", "TEACHER_EAGER", "GPU_HOURS_CAP"):
    print("%s=%s" % (k, d[k]))
print("MAX_JOB_S=%d" % (int(d["TIME_LIMIT_H"]) * 3600))
PY
}

# The runner's endpoint file, written BEFORE any server starts (the preflight checks
# the run identity with it). Deterministic: every execution setting, no measurement.
xp_endpoint() {
  xp_spec || return 4   # always fresh, through the bootstrap (never an inherited file)
  local rev="$TEACHER_REVISION"
  if [[ -z "$rev" && -f "$MODELS/download.json" ]]; then
    rev="$("$SITE_PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("teacher",{}).get("revision",""))' "$MODELS/download.json")"
  fi
  ENDPOINT="$RUN_DIR/endpoint.yaml"
  xp_site endpoint --out "$ENDPOINT" --agent-port "$AGENT_PORT" --agent-tp "$AGENT_TP" \
    --agent-dp "$AGENT_DP" --agent-util "$AGENT_GPU_UTIL" \
    --agent-max-num-seqs "$AGENT_MAX_NUM_SEQS" --concurrency "$CONCURRENCY" \
    --key-env XP_VLLM_API_KEY --teacher "$TEACHER_ENABLE" \
    --teacher-port "$TEACHER_PORT" --teacher-name "$TEACHER_NAME" \
    --teacher-repo "$TEACHER_REPO" --teacher-revision "$rev" --teacher-tp "$TEACHER_TP" \
    --teacher-util "$TEACHER_GPU_UTIL" --teacher-max-len "$TEACHER_MAX_LEN" \
    --teacher-eager "$TEACHER_EAGER" --teacher-extra="$TEACHER_EXTRA_ARGS" \
    --teacher-extra-json="$TEACHER_EXTRA_ARGS_JSON" \
    --teacher-expert-parallel "$TEACHER_EXPERT_PARALLEL" \
    --teacher-max-batched-tokens "$TEACHER_MAX_BATCHED_TOKENS" --learning-units "$LEARNING_UNITS" \
    --serve-mode "$SERVE_MODE" --image "$([[ $SERVE_MODE == container ]] && echo "$CONTAINER_IMAGE")" \
    --vllm-bin "$([[ $SERVE_MODE == venv ]] && echo "$VLLM_BIN")" \
    --agent-model-dir "$AGENT_MODEL_DIR" --teacher-model-dir "$TEACHER_MODEL_DIR" \
    --serve-launch "$SERVE_LAUNCH" --pyxis-flags="$PYXIS_FLAGS" \
    --download "$MODELS/download.json" \
    --gate-sha "$([[ -n "${XP_GATE:-}" && -f "${XP_GATE:-}" ]] && "$SITE_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$XP_GATE")" \
    > /dev/null || return 4
  export ENDPOINT
}

# xp_preflight [runner flags] : verification only, before any server starts
# (registration, code, cases, benchmark copy, run identity, no live runner).
xp_preflight() {
  (cd "$KIT" && PYTHON="$HARNESS_PY" PACKAGE="$PACKAGE" ENDPOINT="$ENDPOINT" OUT="$OUT" \
    scripts/xp/run_phase1.sh --preflight "$@")
}

# Start the teacher (if enabled) then the 9B with the argv of the endpoint file.
xp_token() {
  # A per-job bearer token: the servers listen on 127.0.0.1 only, the token keeps
  # other users of a shared node out. Its NAME (not value) is in the endpoint file.
  export XP_VLLM_API_KEY
  [[ -n "${XP_VLLM_API_KEY:-}" ]] || \
    XP_VLLM_API_KEY="$("$SITE_PY" -c 'import secrets; print(secrets.token_hex(24))')"
}

# xp_start_teacher [TIMEOUT]: the teacher with exactly the endpoint file's argv.
xp_start_teacher() {
  local timeout="${1:-$TEACHER_STARTUP_TIMEOUT_S}" argv=() line
  [[ -d "$TEACHER_MODEL_DIR" ]] || { xp_log "missing $TEACHER_MODEL_DIR"; return 4; }
  while IFS= read -r line; do argv+=("$line"); done < <(xp_site endpoint-argv --endpoint "$ENDPOINT" --which teacher)
  xp_serve teacher "$TEACHER_PORT" "${argv[@]}"
  xp_wait teacher "$TEACHER_PORT" "$timeout"
}

xp_start_agent() {
  local argv=() line
  while IFS= read -r line; do argv+=("$line"); done < <(xp_site endpoint-argv --endpoint "$ENDPOINT" --which agent)
  xp_serve agent "$AGENT_PORT" "${argv[@]}"
  xp_wait agent "$AGENT_PORT" "$AGENT_STARTUP_TIMEOUT_S" || return 5
  # The served vLLM must be the pinned version recorded in the endpoint file.
  xp_site check-version --startup "$RUN_DIR/serve/agent.startup.json" || {
    xp_log "served vLLM version differs from the pin"; return 4; }
}

# Start the teacher (if enabled) then the 9B with the argv of the endpoint file.
xp_start_all() {
  [[ -f "${ENDPOINT:-}" ]] || { xp_log "no endpoint file (xp_endpoint first)"; return 4; }
  xp_token
  [[ -d "$AGENT_MODEL_DIR" ]] || { xp_log "missing $AGENT_MODEL_DIR (run download.sbatch)"; return 4; }
  if [[ "$TEACHER_ENABLE" == "1" ]]; then
    # The teacher first: it takes the larger memory share; the 9B starts on what is left.
    xp_start_teacher || return 5
  fi
  xp_start_agent
}

# Phase 0: the teacher with CUDA graphs first (TEACHER_GRAPHS_STARTUP_S bound), else
# eager; records serve/teacher-mode.json and leaves ENDPOINT for the mode that runs.
xp_start_all_auto() {
  xp_token
  [[ -d "$AGENT_MODEL_DIR" ]] || { xp_log "missing $AGENT_MODEL_DIR (run download.sbatch)"; return 4; }
  local mode=eager graphs_status=skipped graphs_s=null eager_s=null
  if [[ "$TEACHER_ENABLE" == "1" ]]; then
    TEACHER_EAGER=0; xp_endpoint || return 4
    if xp_start_teacher "${TEACHER_GRAPHS_STARTUP_S:-2400}"; then
      mode=graphs; graphs_status=ready
    else
      graphs_status=failed
      cp "$RUN_DIR/serve/teacher.startup.json" "$RUN_DIR/serve/teacher-graphs.startup.json" 2>/dev/null
      cp "$RUN_DIR/serve/teacher.log" "$RUN_DIR/serve/teacher-graphs.log" 2>/dev/null
      xp_stop_servers
      TEACHER_EAGER=1; xp_endpoint || return 4
      xp_start_teacher || { xp_log "teacher failed in both modes"; return 5; }
    fi
    graphs_s="$("$SITE_PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("seconds"))' "$RUN_DIR/serve/teacher.startup.json")"
    if [[ "$mode" == eager ]]; then
      eager_s="$graphs_s"; graphs_s=null
      [[ -f "$RUN_DIR/serve/teacher-graphs.startup.json" ]] && graphs_s="$("$SITE_PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("seconds"))' "$RUN_DIR/serve/teacher-graphs.startup.json")"
    fi
  fi
  printf '{"mode": "%s", "graphs_status": "%s", "graphs_startup_s": %s, "eager_startup_s": %s, "graphs_bound_s": %s}\n' \
    "$mode" "$graphs_status" "${graphs_s:-null}" "${eager_s:-null}" "${TEACHER_GRAPHS_STARTUP_S:-2400}" \
    > "$RUN_DIR/serve/teacher-mode.json"
  export TEACHER_EAGER
  xp_start_agent
}

xp_gpu_snapshot() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader > "$RUN_DIR/gpu-$1.csv" 2>&1 || true
}
