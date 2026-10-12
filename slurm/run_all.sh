#!/bin/bash
# One command for the whole package: phase 0 (v2), the gate, phase 1.
#
#   export WORK=/your/large/shared/storage/xp KIT=$WORK/kit
#   bash $KIT/slurm/run_all.sh [--force-phase0] [--account=.. --qos=.. --partition=..]
#
# Idempotent: run it again at any time; it resumes at the first incomplete stage.
#   1. phase 0: submitted unless $WORK/run_all/phase0.json records a finished phase 0
#      (or its job is still queued/running). --force-phase0 starts a new phase 0 (the
#      old record, gate and job ids are moved to $WORK/run_all/old-<time>/).
#   2. gate: a CPU-only job after phase 0 (--dependency=afterany) unless gate.json is
#      already bound to the recorded phase-0 archive. It evaluates the pre-registered
#      conditions; on GO it submits phase 1 itself.
#   3. phase 1: only with a GO gate.json bound to that phase-0 archive's hash, with the
#      derived settings; resubmitted (resume) when no phase-1 job runs and the export
#      is not complete. Every phase-1 job re-verifies the gate before anything starts.
# Exit 0 = submitted or done; 3 = NO-GO (reasons printed); 2 = refused.
set -uo pipefail
: "${WORK:?set WORK}"
KIT="${KIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export KIT
FORCE=0; ARGS=()
for a in "$@"; do
  case "$a" in --force-phase0) FORCE=1 ;; *) ARGS+=("$a") ;; esac
done
for a in ${ARGS[@]+"${ARGS[@]}"}; do
  case "$a" in --job-name|--job-name=*|-J|-J*)
    echo "[run_all] refusing: '$a'; the job names are fixed (phase 1: xp-p1-<gate hash>) so reruns find the jobs" >&2
    exit 2 ;;
  esac
done
export XP_SBATCH_ARGS="${ARGS[*]:-}" XP_REQUIRE_GATE=1
source "$KIT/slurm/lib.sh" || exit $?
STATE="$WORK/run_all"; mkdir -p "$STATE"
export XP_RUN_ALL_ID="${XP_RUN_ALL_ID:-$(date +%s)-$$-$RANDOM}"
REC="$STATE/phase0.json"; GATE="$STATE/gate.json"
say() { echo "[run_all] $*" >&2; }
alive() { [[ -n "${1:-}" ]] && squeue -h -j "$1" -o %i 2>/dev/null | grep -qx "$1"; }
jobof() { cat "$STATE/$1.job" 2>/dev/null || true; }
submit() {  # submit NAME SCRIPT [extra sbatch args]: prints the job id
  local name="$1" script="$2"; shift 2
  local id
  # shellcheck disable=SC2086
  id="$(sbatch --parsable "$@" ${XP_SBATCH_ARGS:-} "$script" | tail -n 1 | cut -d';' -f1)"
  echo "$id" > "$STATE/$name.job"; echo "$id"
}

if [[ $FORCE == 1 ]]; then
  old="$STATE/old-$(date -u +%Y%m%dT%H%M%S)"; mkdir -p "$old"
  for f in phase0.json gate.json phase0.job gate.job phase1.job phase1.submit gate.log; do
    [[ -e "$STATE/$f" ]] && mv "$STATE/$f" "$old/"
  done
  say "forced: a new phase 0 (previous state in $old)"
fi

# 1. phase 0
P0="$(jobof phase0)"
if [[ -f "$REC" ]]; then
  say "phase 0 finished (record $REC)"
elif alive "$P0"; then
  say "phase 0 running as job $P0"
else
  P0="$(submit phase0 "$KIT/slurm/phase0.sbatch")"; say "phase 0 submitted: job $P0"
fi

# 2. gate
bound() {  # gate.json bound to the recorded phase-0 archive?
  [[ -f "$GATE" && -f "$REC" ]] && "$SITE_PY" - "$GATE" "$REC" <<'PY'
import json, sys
g, r = (json.load(open(p)) for p in sys.argv[1:3])
sys.exit(0 if g.get("phase0", {}).get("archive_sha256") == r.get("archive_sha256")
         and r.get("archive_sha256") else 1)
PY
}
G="$(jobof gate)"
if bound; then
  say "gate decided: $("$SITE_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$GATE")"
elif alive "$G"; then
  say "gate running as job $G"; exit 0
else
  dep=(); alive "$P0" && dep=(--dependency="afterany:$P0")
  [[ -f "$GATE" ]] && mv "$GATE" "$STATE/gate.unbound-$(date -u +%s).json"
  G="$(submit gate "$KIT/slurm/gate.sbatch" ${dep[@]+"${dep[@]}"})"; say "gate submitted: job $G"
  bound || exit 0   # (with a real scheduler the gate runs later and submits phase 1)
fi

# 3. phase 1
DECISION="$("$SITE_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$GATE")"
if [[ "$DECISION" != GO ]]; then
  say "NO-GO: $("$SITE_PY" -c 'import json,sys; print("; ".join(json.load(open(sys.argv[1]))["reasons"]))' "$GATE")"
  exit 3
fi
if xp_phase1_complete; then
  say "phase 1 complete: send $WORK/returns/xp-phase1-export.tar.gz, the phase-0 archive and $GATE"
  exit 0
fi
if [[ -f "$STATE/phase1.by" && "$(cat "$STATE/phase1.by")" == "$XP_RUN_ALL_ID" ]]; then
  say "phase 1 was submitted by the gate moments ago (job $(jobof phase1))"; exit 0
fi
xp_site run-locked --lock "$STATE/phase1.lock" -- bash "$KIT/slurm/submit_phase1.sh"
exit $?
