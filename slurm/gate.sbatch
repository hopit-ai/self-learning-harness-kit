#!/bin/bash
#SBATCH --job-name=xp-gate
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=%x-%j.out
#
# The phase-0 -> phase-1 gate (CPU only; run_all.sh submits it after phase 0 with
# --dependency=afterany). It evaluates the pre-registered conditions on phase 0's
# evidence (no answer keys, no scores; slh/xp/partner/gate.py) and writes
# $WORK/run_all/gate.json. GO: it submits phase 1 through submit.sh with the settings
# derived by the registered rule. NO-GO: it archives phase 0's record, gate.json and
# the reasons ($WORK/returns/xp-gate-<job>.tar.gz) and stops.
set -uo pipefail
if [[ -z "${KIT:-}" ]]; then
  for c in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/.." "$(dirname "$0")/.."; do
    if [[ -f "$c/slurm/lib.sh" ]]; then KIT="$(cd "$c" && pwd)"; break; fi
  done
fi
source "$KIT/slurm/lib.sh" || exit $?
JOB="${SLURM_JOB_ID:-local}"
STATE="$WORK/run_all"
REC="$STATE/phase0.json"
GATE="$STATE/gate.json"
mkdir -p "$STATE" "$WORK/returns"
echo "$JOB" > "$STATE/gate.job"
if [[ ! -f "$REC" ]]; then
  xp_log "no phase-0 record ($REC): phase 0 did not finish"; exit 2
fi
xp_gate evaluate --record "$REC" --out "$GATE" > "$STATE/gate.log" 2>&1
RC=$?
cat "$STATE/gate.log" >&2
if [[ $RC != 0 ]]; then
  xp_log "NO-GO: phase 1 is not started (reasons in $GATE)"
  xp_site collect --name "xp-gate-$JOB" --out "$WORK/returns/xp-gate-$JOB.tar.gz" \
    "$REC" "$GATE" "$STATE/gate.log" >&2 || true
  exit 3
fi
xp_log "GO: submitting phase 1 (single writer, under the state lock)"
xp_site run-locked --lock "$STATE/phase1.lock" -- bash "$KIT/slurm/submit_phase1.sh"
exit $?
