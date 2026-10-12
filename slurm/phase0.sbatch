#!/bin/bash
#SBATCH --job-name=xp-phase0
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=%x-%j.out
#
# XP phase 0 (v2): site check and measurements for phase 1. No registered result.
# Rows run in order; each is recorded in phase0.json:
#   facts, vendor, bootstrap, preflight (verification only), dry-run (fake model; phase
#   0 stops here if it fails, before any server starts),
#   serve (teacher with CUDA graphs first, eager if graphs fail or exceed
#     TEACHER_GRAPHS_STARTUP_S; then the 9B; serve/teacher-mode.json),
#   probe-agent, probe-teacher,
#   measure (the plan's first 2 learning runs, every task x writer, the full loop, on
#     the parallel pool, stopped at MEASURE_LEARN_S: learning throughput),
#   slice (1 run, short loop, the first 20 test cases per task through EVERY arm,
#     exported: per-arm latency, tokens, context limits, full-context receipts).
# Throughout, both servers' vLLM metrics are sampled (metrics/*.jsonl: tokens/s,
# requests/s, running requests, prefix-cache hits). Every step is bounded by the job's
# deadline; the archive is always written: $WORK/returns/xp-phase0-<job>.tar.gz, and
# $WORK/run_all/phase0.json records it (job, archive, SHA-256) for the gate.
#
#   export WORK=/your/large/shared/storage/xp KIT=$WORK/kit
#   bash "$KIT/slurm/run_all.sh" [--account=.. --qos=.. --partition=..]   # the usual way
#   sbatch [--account=..] "$KIT/slurm/phase0.sbatch"                       # phase 0 alone
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
MEASURE_LEARN_S="${MEASURE_LEARN_S:-5400}"     # learning-throughput window
SLICE_RESERVE_S="${SLICE_RESERVE_S:-9000}"     # time kept for the slice after measuring
ARCHIVE_RESERVE_S="${ARCHIVE_RESERVE_S:-900}"
mkdir -p "$RUN_DIR/serve" "$RUN_DIR/metrics" "$WORK/returns" "$WORK/run_all"
STATUS="$RUN_DIR/phase0.json"
ROWS=()
SAMPLERS=()
row() {  # row NAME STATUS SECONDS [NOTE]
  ROWS+=("{\"row\": \"$1\", \"status\": \"$2\", \"seconds\": $3, \"note\": \"${4:-}\"}")
  local IFS=,
  printf '{"job": "%s", "version": 2, "rows": [%s]}\n' "$JOB" "${ROWS[*]}" > "$STATUS"
  xp_log "row $1: $2 (${3}s) ${4:-}"
}
timed() {  # timed NAME CMD... : run, record, return its exit code
  local name="$1"; shift
  local t0=$SECONDS rc
  "$@" > "$RUN_DIR/$name.log" 2>&1; rc=$?
  if [[ $rc == 0 ]]; then row "$name" pass $((SECONDS - t0)); else row "$name" fail $((SECONDS - t0)) "exit $rc"; fi
  return $rc
}
left() {  # seconds left before the work deadline (never below 60)
  local l=$(( DEADLINE - $(date +%s) )); [[ $l -lt 60 ]] && l=60; echo "$l"
}
window() {  # window NAME start|end : measurement windows for the metrics
  printf '{"window": "%s", "event": "%s", "t": %s}\n' "$1" "$2" "$(date +%s)" >> "$RUN_DIR/windows.jsonl"
}
finish() {
  trap - TERM INT
  [[ ${#SAMPLERS[@]} -gt 0 ]] && xp_stop_tree 2 "${SAMPLERS[@]}"
  xp_stop_servers
  xp_gpu_snapshot end
  local archive="$WORK/returns/xp-phase0-$JOB.tar.gz" hard=$(( $(date +%s) + 600 ))
  xp_bounded 600 "$SITE_PY" "$KIT/slurm/xpsite.py" collect --deadline "$hard" \
    --name "xp-phase0-$JOB" --out "$archive" "$RUN_DIR" \
    "${SLURM_SUBMIT_DIR:-.}/xp-phase0-$JOB.out" > "$RUN_DIR/archive.json" 2>&1 || true
  cat "$RUN_DIR/archive.json" >&2
  # The record the gate binds to (run_all.sh): job, run folder, archive and its hash.
  "$SITE_PY" - "$JOB" "$RUN_DIR" "$archive" "$WORK/run_all/phase0.json" <<'PY' || true
import hashlib, json, os, sys
job, run_dir, archive, out = sys.argv[1:5]
rows = json.load(open(os.path.join(run_dir, "phase0.json"))).get("rows", [])
sha = hashlib.sha256(open(archive, "rb").read()).hexdigest() if os.path.isfile(archive) else None
rec = {"job": job, "run_dir": run_dir, "archive": archive, "archive_sha256": sha,
       "rows_passed": bool(rows) and all(r["status"] == "pass" for r in rows)}
tmp = out + ".tmp"
open(tmp, "w").write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
os.replace(tmp, out)
PY
  case "${TMPDIR:-}" in /tmp/xp-*) rm -rf "$TMPDIR" ;; esac
}
on_term() {
  xp_log "stopping (signal or deadline)"
  # shellcheck disable=SC2046
  xp_stop_tree "${XP_STOP_GRACE_S:-20}" $(pgrep -P $$ 2>/dev/null) || true
  exit 143
}
trap finish EXIT
trap on_term TERM INT

LEFT="$(xp_site remaining)"; [[ "$LEFT" -gt 0 ]] || LEFT=$(( 8 * 3600 ))
DEADLINE=$(( $(date +%s) + LEFT - ARCHIVE_RESERVE_S ))
xp_log "phase 0 (v2), job $JOB, kit $KIT, work $WORK, $(( LEFT / 60 )) min left"
timed facts xp_site facts --out "$RUN_DIR/facts.json"
xp_gpu_snapshot start
timed vendor bash -c "cd '$KIT' && PYTHON='$HARNESS_PY' scripts/xp/fetch_vendor.sh --verify" \
  || { row stop fail 0 "benchmark checkout missing or wrong: run scripts/xp/fetch_vendor.sh"; exit 4; }
OUT="$RUN_DIR/slice"
timed bootstrap xp_endpoint \
  || { row stop fail 0 "bootstrap refused: see bootstrap.log; no server started"; exit 4; }
timed preflight xp_preflight --smoke slice \
  || { row stop fail 0 "preflight refused: see preflight.log; no server started"; exit 4; }
if ! timed dry-run bash -c "cd '$KIT' && PYTHON='$HARNESS_PY' PACKAGE='$PACKAGE' \
    OUT='$RUN_DIR/dry' scripts/xp/run_phase1.sh --dry-run --smoke" \
    || ! grep -q '"status": "export ok"' "$RUN_DIR/dry-run.log"; then
  row stop fail 0 "dry run failed: see dry-run.log; no server started"; exit 4
fi

T0=$SECONDS
if ! xp_start_all_auto > "$RUN_DIR/serve.log" 2>&1; then
  row serve fail $((SECONDS - T0)) "see serve.log and serve/*.log"; exit 5
fi
row serve pass $((SECONDS - T0))
xp_gpu_snapshot served
for name in teacher agent; do
  port="$AGENT_PORT"; [[ $name == teacher ]] && port="$TEACHER_PORT"
  [[ $name == teacher && "$TEACHER_ENABLE" != 1 ]] && continue
  "$SITE_PY" "$KIT/slurm/xpsite.py" sample-metrics --name "$name" --port "$port" \
    --interval "${METRICS_INTERVAL_S:-30}" --until "$DEADLINE" \
    --out "$RUN_DIR/metrics/$name.jsonl" > /dev/null 2>&1 &
  SAMPLERS+=($!)
done
timed probe-agent xp_site probe --name agent --port "$AGENT_PORT" --model qwen3.5-9b-vllm \
  --key-env XP_VLLM_API_KEY --requests 64 --concurrency "$CONCURRENCY" \
  --out "$RUN_DIR/probe-agent.json" || true
if [[ "$TEACHER_ENABLE" == "1" ]]; then
  timed probe-teacher xp_site probe --name teacher --port "$TEACHER_PORT" \
    --model "$TEACHER_NAME" --key-env XP_VLLM_API_KEY --requests 8 --concurrency 4 \
    --max-tokens 256 --out "$RUN_DIR/probe-teacher.json" || true
fi

# measure: the parallel pool on the plan's own learning units, stopped at its cap.
CAP=$(( $(left) - SLICE_RESERVE_S )); [[ $CAP -gt $MEASURE_LEARN_S ]] && CAP=$MEASURE_LEARN_S
[[ $CAP -lt ${MEASURE_MIN_S:-300} ]] && CAP=${MEASURE_MIN_S:-300}
window measure start
T0=$SECONDS
xp_bounded "$CAP" bash -c "cd '$KIT' && PYTHON='$HARNESS_PY' PACKAGE='$PACKAGE' \
  ENDPOINT='$ENDPOINT' OUT='$RUN_DIR/measure' XP_LEARNING_UNITS='$LEARNING_UNITS' \
  scripts/xp/run_phase1.sh --smoke measure --no-export" > "$RUN_DIR/measure.log" 2>&1
RC=$?
window measure end
printf '{"cap_s": %s, "wall_s": %s, "exit": %s, "units": %s}\n' "$CAP" "$((SECONDS - T0))" \
  "$RC" "$LEARNING_UNITS" > "$RUN_DIR/measure.json"
case "$RC" in
  0|124) row measure pass $((SECONDS - T0)) "exit $RC (124 = stopped at the cap, as intended)" ;;
  *) row measure fail $((SECONDS - T0)) "exit $RC" ;;
esac

# slice: every arm on the first 20 test cases per task, exported.
window slice start
T0=$SECONDS
xp_bounded "$(left)" bash -c "cd '$KIT' && PYTHON='$HARNESS_PY' PACKAGE='$PACKAGE' \
  ENDPOINT='$ENDPOINT' OUT='$OUT' XP_LEARNING_UNITS='$LEARNING_UNITS' \
  scripts/xp/run_phase1.sh --smoke slice" > "$RUN_DIR/slice.log" 2>&1
RC=$?
window slice end
if [[ $RC == 0 ]] && grep -q '"status": "export ok"' "$RUN_DIR/slice.log"; then
  row slice pass $((SECONDS - T0))
else
  row slice fail $((SECONDS - T0)) "exit $RC"
fi
xp_log "phase 0 done: $STATUS"
