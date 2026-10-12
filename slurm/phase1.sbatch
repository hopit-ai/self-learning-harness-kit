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
#   GPU_HOURS_CAP (default 1600: 2 x the phase-0 estimate): GPU-hours over all phase-1 jobs (ledger
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
GPU_HOURS_CAP="${GPU_HOURS_CAP:-1600}"
ARCHIVE_RESERVE_S="${ARCHIVE_RESERVE_S:-1200}"
MIN_USEFUL_S="${MIN_USEFUL_S:-7200}"   # refuse a budget below startup + a useful slice
RUNNER_ARGS="${RUNNER_ARGS:-}"   # e.g. "--max-cases 20" for a first smoke on this setup
mkdir -p "$RUN_DIR/serve" "$WORK/returns"
RC=""
REFUSED_CLEANUP_S=$(( 300 + ${XP_STOP_GRACE_S:-20} + 60 ))  # archive bound + stop + margin
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
  # The archive is bounded by the budget too: the deadline holds while listing and
  # copying (a partial archive says so in TRUNCATED.txt), and a backstop kills it.
  local hard="${HARD_DEADLINE:-$(( $(date +%s) + 300 ))}"
  xp_bounded $(( hard - $(date +%s) )) "$SITE_PY" "$KIT/slurm/xpsite.py" collect \
    --deadline "$hard" --name "xp-phase1-$JOB" \
    --out "$WORK/returns/xp-phase1-$JOB-logs.tar.gz" "${inc[@]}" >&2 || true
  # Phase 1 complete only for a valid export ARCHIVE of the FULL registered plan: this
  # job collects into its own part file, the gate's `complete` unpacks that archive,
  # validates it (not truncated, schema, manifest, every registered instance and
  # case), moves it into place and writes the marker last. A truncated, failed or
  # older archive never gets a marker.
  local part="$WORK/returns/xp-phase1-export.$JOB.part.tar.gz"
  rm -f "$part"
  if [[ "$RC" == "0" && -f "$OUT_RUN/export/manifest.json" ]]; then
    xp_bounded $(( hard - $(date +%s) )) "$SITE_PY" "$KIT/slurm/xpsite.py" collect --deadline "$hard" \
      --name xp-phase1-export --cap 101000000 --out "$part" "$OUT_RUN/export" >&2 || true
  fi
  if [[ "$RC" == "0" && -f "$part" ]]; then
    mkdir -p "$WORK/run_all"
    xp_gate complete --archive "$part" --dest "$WORK/returns/xp-phase1-export.tar.gz" \
      --out "$WORK/run_all/phase1.complete.json" > "$RUN_DIR/complete.log" 2>&1 || true
  fi
  # not complete: kept under another name for the logs, never as the export
  [[ -f "$part" ]] && mv -f "$part" "$WORK/returns/xp-phase1-export.$JOB.incomplete.tar.gz"
  # Last act: every allocation is charged. Admitted: the end row (start..now + margin,
  # releasing the unused reservation). Refused or failed admission: the final charge
  # row replaces the provisional one written at refusal (start..now + margin).
  if [[ -n "${ADMITTED:-}" ]]; then
    xp_site ledger end --file "$LEDGER" --gpus "$XP_GPUS" > /dev/null 2>&1
  else
    xp_site ledger charge --file "$LEDGER" --gpus "$XP_GPUS" --start "$START_EPOCH" > /dev/null 2>&1
  fi
  case "${TMPDIR:-}" in /tmp/xp-*) rm -rf "$TMPDIR" ;; esac
}
on_term() {
  xp_log "stopping (signal or budget deadline)"
  RC="${RC:-deadline}"
  # The whole tree, TERM then KILL after the grace, before archiving.
  # shellcheck disable=SC2046
  xp_stop_tree "${XP_STOP_GRACE_S:-20}" $(pgrep -P $$ 2>/dev/null) || xp_log "a process survived KILL"
  exit 143
}
trap finish EXIT
trap on_term TERM INT

JOB_LIMIT_S="$(xp_site job-limit)"   # the allocation's limit as submitted (before trimming)
xp_log "phase 1, job $JOB, kit $KIT, out $OUT"

# 1. Admission, first of all: atomic under the ledger lock, the whole budget reserved
#    from the job's start (every second of the allocation is charged).
LEFT="$(xp_site remaining)"
if ! xp_site admit --file "$LEDGER" --cap "$GPU_HOURS_CAP" --gpus "$XP_GPUS" \
    --start "$START_EPOCH" --slurm-left "$LEFT" --min "$MIN_USEFUL_S" > "$RUN_DIR/budget.json"; then
  # Charged FIRST (provisional: start..now + the bounded cleanup and archive), so a
  # kill during cleanup still leaves the allocation charged; finish() updates it.
  xp_site ledger charge --file "$LEDGER" --gpus "$XP_GPUS" --start "$START_EPOCH" \
    --until $(( $(date +%s) + REFUSED_CLEANUP_S )) > /dev/null 2>&1
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
xp_site facts --out "$RUN_DIR/facts.json" > /dev/null &
wait $! || true   # background child: the watchdog TERM is not deferred

# 3b. The phase gate (run_all.sh): a GO gate.json bound to the phase-0 archive, equal to
#     a recomputation from phase 0's evidence, with this job's settings as derived.
if [[ "${XP_REQUIRE_GATE:-0}" == 1 || -n "${XP_GATE:-}" ]]; then
  GATE_FILE="${XP_GATE:-$WORK/run_all/gate.json}"
  xp_gate verify --record "$WORK/run_all/phase0.json" --gate "$GATE_FILE" \
    --expect "CONCURRENCY=$CONCURRENCY,LEARNING_UNITS=$LEARNING_UNITS,TEACHER_EAGER=$TEACHER_EAGER,GPU_HOURS_CAP=$GPU_HOURS_CAP,JOB_LIMIT_S=$JOB_LIMIT_S" \
    > "$RUN_DIR/gate-verify.log" 2>&1 &
  if ! wait $!; then
    xp_log "gate refused (no server started): $(tail -3 "$RUN_DIR/gate-verify.log")"; RC=gate; exit 4
  fi
  export XP_GATE="$GATE_FILE"
fi

# 4. Verification only, before any server starts (bootstrap, registration, identity),
#    as background children so the watchdog's TERM is handled at once.
( xp_endpoint ) > "$RUN_DIR/endpoint.log" 2>&1 &
if ! wait $!; then
  xp_log "bootstrap refused (no server started): $(tail -5 "$RUN_DIR/endpoint.log")"; RC=4; exit 4
fi
export ENDPOINT="$RUN_DIR/endpoint.yaml" XP_SERVE_SPEC="$RUN_DIR/serve-spec.json"
[[ -f "$ENDPOINT" && -f "$XP_SERVE_SPEC" ]] || { xp_log "no endpoint file"; RC=4; exit 4; }
# 4b. Under a gate: every serving argument this job will pass equals phase 0's
#     (gate-bound), teacher extras within the allowlist, eager mode as derived.
if [[ -n "${XP_GATE:-}" ]]; then
  xp_gate serving --gate "$XP_GATE" --endpoint "$ENDPOINT" > "$RUN_DIR/gate-serving.log" 2>&1 &
  if ! wait $!; then
    xp_log "serving settings refused (no server started): $(tail -3 "$RUN_DIR/gate-serving.log")"; RC=gate; exit 4
  fi
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
