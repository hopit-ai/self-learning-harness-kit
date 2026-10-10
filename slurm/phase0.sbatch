#!/bin/bash
#SBATCH --job-name=xp-phase0
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=%x-%j.out
#
# XP phase 0: site check. No training, no registered result: it proves the shipped
# code runs on this node and measures what phase 1 needs (server startup times,
# memory, request rate). Rows run in order; each is recorded in phase0.json.
#
#   export WORK=/your/large/shared/storage/xp KIT=$WORK/kit
#   sbatch [--account=.. --qos=.. --partition=..] "$KIT/slurm/phase0.sbatch"
#
# Rows: facts, vendor, preflight (verification only: registration, code, cases,
# benchmark copy), dry-run (fake model, must print "export ok"; phase 0 stops here if
# it does not, before any server starts), serve (teacher then 9B, startup seconds),
# probe (request rate), smoke (the real runner on the real servers: smoke setting,
# 5 test cases per task), archive. One small archive comes back:
# $WORK/returns/xp-phase0-<job>.tar.gz.
set -uo pipefail
if [[ -z "${KIT:-}" ]]; then
  for c in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/.." "$(dirname "$0")/.."; do
    if [[ -f "$c/slurm/lib.sh" ]]; then KIT="$(cd "$c" && pwd)"; break; fi
  done
fi
[[ -n "${KIT:-}" && -f "$KIT/slurm/lib.sh" ]] || { echo "set KIT to the kit checkout" >&2; exit 4; }
# shellcheck source=lib.sh
source "$KIT/slurm/lib.sh" || exit $?

JOB="${SLURM_JOB_ID:-local}"
RUN_DIR="${RUN_DIR:-$WORK/phase0/job-$JOB}"
mkdir -p "$RUN_DIR/serve" "$WORK/returns"
STATUS="$RUN_DIR/phase0.json"
ROWS=()
row() {  # row NAME STATUS SECONDS [NOTE]
  ROWS+=("{\"row\": \"$1\", \"status\": \"$2\", \"seconds\": $3, \"note\": \"${4:-}\"}")
  local IFS=,
  printf '{"job": "%s", "rows": [%s]}\n' "$JOB" "${ROWS[*]}" > "$STATUS"
  xp_log "row $1: $2 (${3}s) ${4:-}"
}
timed() {  # timed NAME CMD... : run, record, return its exit code
  local name="$1"; shift
  local t0=$SECONDS rc
  "$@" > "$RUN_DIR/$name.log" 2>&1; rc=$?
  if [[ $rc == 0 ]]; then row "$name" pass $((SECONDS - t0)); else row "$name" fail $((SECONDS - t0)) "exit $rc"; fi
  return $rc
}
finish() {
  xp_stop_servers
  xp_gpu_snapshot end
  xp_site collect --name "xp-phase0-$JOB" --out "$WORK/returns/xp-phase0-$JOB.tar.gz" \
    "$RUN_DIR" "${SLURM_SUBMIT_DIR:-.}/xp-phase0-$JOB.out" > "$RUN_DIR/archive.json" 2>&1 || true
  cat "$RUN_DIR/archive.json" >&2
  case "${TMPDIR:-}" in /tmp/xp-*) rm -rf "$TMPDIR" ;; esac
}
trap finish EXIT
trap 'xp_log "signal: stopping"; exit 143' TERM INT

xp_log "phase 0, job $JOB, kit $KIT, work $WORK"
timed facts xp_site facts --out "$RUN_DIR/facts.json"
xp_gpu_snapshot start
timed vendor bash -c "cd '$KIT' && PYTHON='$HARNESS_PY' scripts/xp/fetch_vendor.sh --verify" \
  || { row stop fail 0 "benchmark checkout missing or wrong: run scripts/xp/fetch_vendor.sh"; exit 4; }
OUT="$RUN_DIR/smoke"
timed bootstrap xp_endpoint \
  || { row stop fail 0 "bootstrap refused: see bootstrap.log; no server started"; exit 4; }
timed preflight xp_preflight --smoke \
  || { row stop fail 0 "preflight refused: see preflight.log; no server started"; exit 4; }
if ! timed dry-run bash -c "cd '$KIT' && PYTHON='$HARNESS_PY' PACKAGE='$PACKAGE' \
    OUT='$RUN_DIR/dry' scripts/xp/run_phase1.sh --dry-run --smoke" \
    || ! grep -q '"status": "export ok"' "$RUN_DIR/dry-run.log"; then
  row stop fail 0 "dry run failed: see dry-run.log; no server started"; exit 4
fi

T0=$SECONDS
if xp_start_all > "$RUN_DIR/serve.log" 2>&1; then
  row serve pass $((SECONDS - T0))
  xp_gpu_snapshot served
  timed probe-agent xp_site probe --name agent --port "$AGENT_PORT" --model qwen3.5-9b-vllm \
    --key-env XP_VLLM_API_KEY --requests 64 --concurrency "$CONCURRENCY" \
    --out "$RUN_DIR/probe-agent.json" || true
  if [[ "$TEACHER_ENABLE" == "1" ]]; then
    timed probe-teacher xp_site probe --name teacher --port "$TEACHER_PORT" \
      --model "$TEACHER_NAME" --key-env XP_VLLM_API_KEY --requests 8 --concurrency 4 \
      --max-tokens 256 --out "$RUN_DIR/probe-teacher.json" || true
  fi
  timed smoke bash -c "cd '$KIT' && PYTHON='$HARNESS_PY' PACKAGE='$PACKAGE' \
    ENDPOINT='$ENDPOINT' OUT='$OUT' scripts/xp/run_phase1.sh --smoke --max-cases 5" \
    || true
else
  row serve fail $((SECONDS - T0)) "see serve.log and serve/*.log"
fi
xp_log "phase 0 done: $STATUS"
