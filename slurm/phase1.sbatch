#!/bin/bash
#SBATCH --job-name=xp-phase1
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=%x-%j.out
#
# XP phase 1 (decisions): serve the teacher and the 9B, run the registered plan,
# export. Resumable: submit the SAME command again after any stop; finished work in
# $OUT is kept (the runner refuses a run directory started under other settings).
#
#   export WORK=/your/large/shared/storage/xp KIT=$WORK/kit
#   sbatch [--account=.. --qos=.. --partition=..] "$KIT/slurm/phase1.sbatch"
#
# Budgets and halts:
#   GPU_HOURS_CAP (default 240): GPU-hours over all phase-1 jobs (ledger
#     $WORK/phase1/gpu-hours.jsonl). First thing in the job, under the ledger lock,
#     the job is admitted with budget = min(Slurm limit, (cap - used) x 3600 / GPUs)
#     counted from the job's START, and its start row reserves all of it (concurrent
#     jobs cannot both take it). The job then lowers its own Slurm time limit to the
#     budget (or stops), and one watchdog stops preflight, serving or the runner
#     ARCHIVE_RESERVE_S (1200 s) before the budget ends; the archive is cut at the
#     budget too. The end row, the job's last act, charges start..end and releases
#     the rest; a job killed without one stays charged its whole reservation. Below
#     MIN_USEFUL_S (7200 s) or while another phase-1 job of this ledger runs, the job
#     refuses (exit 2). slurm/submit.sh requests a fitting time limit up front.
#   Before any server starts, a verification-only preflight (registration, code,
#     cases, benchmark copy, run identity) refuses with exit 4.
#   Runner exit 3 = halted (spend cap, authentication, connection, configuration):
#     fix the cause (see halt.json in the archive) and resubmit. Exit 4 = refused
#     (checkout, package or benchmark differ from the registration): tell us.
# Archives: $WORK/returns/xp-phase1-<job>-logs.tar.gz every time; when the export is
# complete also $WORK/returns/xp-phase1-export.tar (the delivery, at most 100 MB).
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
OUT="${OUT:-$WORK/runs/xp-phase1}"
RUN_DIR="${RUN_DIR:-$WORK/phase1/job-$JOB}"
LEDGER="${LEDGER:-$WORK/phase1/gpu-hours.jsonl}"
GPU_HOURS_CAP="${GPU_HOURS_CAP:-240}"
ARCHIVE_RESERVE_S="${ARCHIVE_RESERVE_S:-1200}"
MIN_USEFUL_S="${MIN_USEFUL_S:-7200}"   # refuse a budget below startup + a useful slice
RUNNER_ARGS="${RUNNER_ARGS:-}"   # e.g. "--max-cases 20" for a first smoke on this setup
mkdir -p "$RUN_DIR/serve" "$WORK/returns"
RC=""
START_EPOCH="${SLURM_JOB_START_TIME:-$(date +%s)}"
OUT_RUN="$OUT"; [[ " $RUNNER_ARGS " == *" --dry-run "* ]] && OUT_RUN="$OUT-dry-run"  # run_phase1.sh's rule

finish() {
  trap - TERM INT
  [[ -n "${WATCH_PID:-}" ]] && kill "$WATCH_PID" 2>/dev/null
  xp_stop_servers
  xp_gpu_snapshot end
  printf '{"job": "%s", "runner_exit": "%s", "out": "%s"}\n' "$JOB" "${RC:-none}" "$OUT_RUN" \
    > "$RUN_DIR/phase1.json"
  local inc=("$RUN_DIR" "$LEDGER" "${SLURM_SUBMIT_DIR:-.}/xp-phase1-$JOB.out")
  local f
  for f in halt.json hashes.json env.json logs summary.json; do
    [[ -e "$OUT_RUN/$f" ]] && inc+=("$OUT_RUN/$f")
  done
  # The archive is bounded by the budget too (a partial archive says so).
  xp_site collect --deadline "${HARD_DEADLINE:-0}" --name "xp-phase1-$JOB" \
    --out "$WORK/returns/xp-phase1-$JOB-logs.tar.gz" "${inc[@]}" >&2 || true
  if [[ "$RC" == "0" && -f "$OUT_RUN/export/manifest.json" ]]; then
    xp_site collect --deadline "${HARD_DEADLINE:-0}" --name xp-phase1-export \
      --cap 101000000 --out "$WORK/returns/xp-phase1-export.tar.gz" "$OUT_RUN/export" >&2 || true
  fi
  # Last act: the end row (charges start..now + margin; releases the unused reservation).
  [[ -n "${ADMITTED:-}" ]] && xp_site ledger end --file "$LEDGER" --gpus "$XP_GPUS" > /dev/null 2>&1
  case "${TMPDIR:-}" in /tmp/xp-*) rm -rf "$TMPDIR" ;; esac
}
on_term() {
  xp_log "stopping (signal or budget deadline)"
  RC="${RC:-deadline}"
  pkill -TERM -P $$ 2>/dev/null   # foreground steps, servers and the runner
  exit 143
}
trap finish EXIT
trap on_term TERM INT

xp_log "phase 1, job $JOB, kit $KIT, out $OUT"

# 1. Admission, first of all: atomic under the ledger lock, the whole budget reserved
#    from the job's start (every second of the allocation is charged).
LEFT="$(xp_site remaining)"
if ! xp_site admit --file "$LEDGER" --cap "$GPU_HOURS_CAP" --gpus "$XP_GPUS" \
    --start "$START_EPOCH" --slurm-left "$LEFT" --min "$MIN_USEFUL_S" > "$RUN_DIR/budget.json"; then
  xp_log "not starting: $(cat "$RUN_DIR/budget.json")"; RC=budget; exit 2
fi
ADMITTED=1
BUDGET_S="$("$SITE_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["budget_s"])' "$RUN_DIR/budget.json")"
DEADLINE_EPOCH=$(( START_EPOCH + BUDGET_S ))
export HARD_DEADLINE=$(( DEADLINE_EPOCH - 60 ))                 # archive done by then
export XP_DEADLINE=$(( DEADLINE_EPOCH - ARCHIVE_RESERVE_S ))     # work stops here

# 2. The allocation itself must end within the budget: lower the time limit, or stop.
if [[ "$JOB" != local ]]; then
  if ! scontrol update JobId="$JOB" TimeLimit="$(( BUDGET_S / 60 ))" > "$RUN_DIR/trim.log" 2>&1; then
    xp_log "could not lower the time limit to the budget ($(cat "$RUN_DIR/trim.log")): stopping"
    RC=trim; exit 2
  fi
  xp_log "time limit lowered to $(( BUDGET_S / 60 )) min"
fi

# 3. One watchdog for everything that follows (preflight, serving, runner): at the
#    work deadline it stops this script; finish() then archives before HARD_DEADLINE.
( sleep $(( XP_DEADLINE - $(date +%s) > 1 ? XP_DEADLINE - $(date +%s) : 1 ))
  echo "[xp] budget deadline" >&2; kill -TERM $$ 2>/dev/null ) &
WATCH_PID=$!
xp_site facts --out "$RUN_DIR/facts.json" > /dev/null

# 4. Verification only, before any server starts (bootstrap, registration, identity).
if ! xp_endpoint > "$RUN_DIR/endpoint.log" 2>&1; then
  xp_log "bootstrap refused (no server started): $(tail -5 "$RUN_DIR/endpoint.log")"; RC=4; exit 4
fi
# shellcheck disable=SC2086
xp_preflight $RUNNER_ARGS > "$RUN_DIR/preflight.log" 2>&1 &
if ! wait $!; then
  xp_log "preflight refused (no server started): $(tail -5 "$RUN_DIR/preflight.log")"; RC=4; exit 4
fi
xp_gpu_snapshot start

# 5. Servers (start-up waits bounded by XP_DEADLINE).
if ! xp_start_all > "$RUN_DIR/serve.log" 2>&1; then
  xp_log "servers did not start: see $RUN_DIR/serve.log and serve/*.log"; RC="${RC:-serve}"; exit 5
fi
xp_gpu_snapshot served

# 6. The runner.
# shellcheck disable=SC2086
(cd "$KIT" && PYTHON="$HARNESS_PY" PACKAGE="$PACKAGE" ENDPOINT="$ENDPOINT" OUT="$OUT" \
  exec scripts/xp/run_phase1.sh $RUNNER_ARGS) > "$RUN_DIR/runner.log" 2>&1 &
RUNNER_PID=$!
wait "$RUNNER_PID"; RC=$?
case "$RC" in
  0) xp_log "export ok: $OUT_RUN/export (also $WORK/returns/xp-phase1-export.tar.gz)" ;;
  3) xp_log "halted: see $OUT_RUN/halt.json; fix the cause and resubmit the same command" ;;
  4) xp_log "refused: the checkout, package or benchmark differ from the registration" ;;
  *) xp_log "runner stopped (exit $RC): resubmit the same command to resume" ;;
esac
exit "$RC"
