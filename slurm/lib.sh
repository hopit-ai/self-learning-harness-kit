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
TEACHER_REPO="${TEACHER_REPO:-Qwen/Qwen3.5-397B-A17B-FP8}"
TEACHER_REVISION="${TEACHER_REVISION:-}"       # filled from the download receipt if empty
TEACHER_MODEL_DIR="${TEACHER_MODEL_DIR:-$MODELS/teacher}"
TEACHER_NAME="${TEACHER_NAME:-qwen3.5-397b-a17b-fp8}"

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
TEACHER_GPU_UTIL="${TEACHER_GPU_UTIL:-0.74}"
TEACHER_MAX_LEN="${TEACHER_MAX_LEN:-131072}"
TEACHER_EAGER="${TEACHER_EAGER:-1}"            # skip CUDA-graph capture (27 min in the reference)
TEACHER_EXTRA_ARGS="${TEACHER_EXTRA_ARGS:---language-model-only}"
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
  local pid
  [[ ${#XP_PIDS[@]} -gt 0 ]] || return 0
  for pid in "${XP_PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep "${XP_STOP_GRACE_S:-20}"
  for pid in "${XP_PIDS[@]}"; do
    [[ -n "$pid" ]] && kill -KILL "$pid" 2>/dev/null || true
  done
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
    --serve-mode "$SERVE_MODE" --image "$([[ $SERVE_MODE == container ]] && echo "$CONTAINER_IMAGE")" \
    --vllm-bin "$([[ $SERVE_MODE == venv ]] && echo "$VLLM_BIN")" \
    --agent-model-dir "$AGENT_MODEL_DIR" --teacher-model-dir "$TEACHER_MODEL_DIR" \
    --serve-launch "$SERVE_LAUNCH" --pyxis-flags="$PYXIS_FLAGS" \
    --download "$MODELS/download.json" > /dev/null || return 4
  export ENDPOINT
}

# xp_preflight [runner flags] : verification only, before any server starts
# (registration, code, cases, benchmark copy, run identity, no live runner).
xp_preflight() {
  (cd "$KIT" && PYTHON="$HARNESS_PY" PACKAGE="$PACKAGE" ENDPOINT="$ENDPOINT" OUT="$OUT" \
    scripts/xp/run_phase1.sh --preflight "$@")
}

# Start the teacher (if enabled) then the 9B with the argv of the endpoint file.
xp_start_all() {
  [[ -f "${ENDPOINT:-}" ]] || { xp_log "no endpoint file (xp_endpoint first)"; return 4; }
  # A per-job bearer token: the servers listen on 127.0.0.1 only, the token keeps
  # other users of a shared node out. Its NAME (not value) is in the endpoint file.
  export XP_VLLM_API_KEY
  XP_VLLM_API_KEY="$("$SITE_PY" -c 'import secrets; print(secrets.token_hex(24))')"
  [[ -d "$AGENT_MODEL_DIR" ]] || { xp_log "missing $AGENT_MODEL_DIR (run download.sbatch)"; return 4; }
  local argv=() line
  if [[ "$TEACHER_ENABLE" == "1" ]]; then
    [[ -d "$TEACHER_MODEL_DIR" ]] || { xp_log "missing $TEACHER_MODEL_DIR"; return 4; }
    # Exactly the argv recorded in the endpoint file (the run identity).
    while IFS= read -r line; do argv+=("$line"); done < <(xp_site endpoint-argv --endpoint "$ENDPOINT" --which teacher)
    xp_serve teacher "$TEACHER_PORT" "${argv[@]}"
    # The teacher first: it takes the larger memory share; the 9B starts on what is left.
    xp_wait teacher "$TEACHER_PORT" "$TEACHER_STARTUP_TIMEOUT_S" || return 5
  fi
  argv=()
  while IFS= read -r line; do argv+=("$line"); done < <(xp_site endpoint-argv --endpoint "$ENDPOINT" --which agent)
  xp_serve agent "$AGENT_PORT" "${argv[@]}"
  xp_wait agent "$AGENT_PORT" "$AGENT_STARTUP_TIMEOUT_S" || return 5
  # The served vLLM must be the pinned version recorded in the endpoint file.
  xp_site check-version --startup "$RUN_DIR/serve/agent.startup.json" || {
    xp_log "served vLLM version differs from the pin"; return 4; }
}

xp_gpu_snapshot() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader > "$RUN_DIR/gpu-$1.csv" 2>&1 || true
}
