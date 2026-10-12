#!/bin/bash
# Submit (or resume) phase 1 once, under the run_all state lock (single writer):
#   xpsite.py run-locked --lock $WORK/run_all/phase1.lock -- bash submit_phase1.sh
# Refuses unless gate.json is GO and bound to the recorded phase-0 archive; does
# nothing when phase 1 is complete (completion marker valid for the export archive)
# or a phase-1 job of this gate is still queued or running (by its deterministic name
# xp-p1-<gate hash 12> and the recorded id; squeue, then sacct); else submits through
# submit.sh with the derived settings and that name, and records the job id
# atomically. A failed scheduler query refuses (exit 4, retry later).
# Exit 0 submitted or nothing to do; 2 refused; 3 NO-GO; 4 scheduler unknown.
set -uo pipefail
: "${WORK:?set WORK}"
KIT="${KIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$KIT/slurm/lib.sh" || exit $?
STATE="$WORK/run_all"; REC="$STATE/phase0.json"; GATE="$STATE/gate.json"
say() { echo "[phase1] $*" >&2; }
[[ -f "$GATE" && -f "$REC" ]] || { say "no gate.json or phase-0 record"; exit 2; }
"$SITE_PY" - "$GATE" "$REC" <<'PY' || { say "gate.json is not GO and bound to the phase-0 archive"; exit 3; }
import json, sys
g, r = (json.load(open(p)) for p in sys.argv[1:3])
sys.exit(0 if g.get("decision") == "GO" and r.get("archive_sha256") and
         g.get("phase0", {}).get("archive_sha256") == r["archive_sha256"] else 1)
PY
if xp_phase1_complete; then say "phase 1 is complete"; exit 0; fi
P1="$(cat "$STATE/phase1.job" 2>/dev/null || true)"
# One deterministic job name per gate: a job submitted earlier whose id was never
# recorded (the submitter died after sbatch) is still found by its name.
NAME="xp-p1-$("$SITE_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"][:12])' "$GATE")"
record() {
  printf '%s\n' "$1" > "$STATE/phase1.job.tmp" && mv "$STATE/phase1.job.tmp" "$STATE/phase1.job"
  # which run_all.sh invocation's chain submitted it (the gate inherits its id)
  printf '%s\n' "${XP_RUN_ALL_ID:-}" > "$STATE/phase1.by.tmp" && mv "$STATE/phase1.by.tmp" "$STATE/phase1.by"
}
read -r state live < <(xp_phase1_state "$NAME" "$P1")
case "$state" in
  active)
    [[ "$live" == "$P1" ]] || record "$live"
    say "phase 1 job $live ($NAME) is queued or running"; exit 0 ;;
  inactive) ;;
  *) say "cannot tell from the scheduler whether phase 1 ($NAME, recorded job ${P1:-none}) is queued: not submitting; retry when squeue and sacct answer"
     exit 4 ;;
esac
# the deterministic name is the only job name: a user-supplied one is refused
for a in ${XP_SBATCH_ARGS:-}; do
  case "$a" in --job-name|--job-name=*|-J|-J*)
    say "refusing: '$a' in the sbatch flags; phase 1's job name is fixed ($NAME) so a rerun finds the job"
    exit 2 ;;
  esac
done
while IFS= read -r kv; do export "${kv?}"; done < <(xp_gate_env "$GATE")
export XP_GATE="$GATE" XP_REQUIRE_GATE=1
# shellcheck disable=SC2086
out="$(bash "$KIT/slurm/submit.sh" --parsable ${XP_SBATCH_ARGS:-} --job-name="$NAME" 2>&1)"; rc=$?
echo "$out" >&2
[[ $rc == 0 ]] || { say "not submitted (exit $rc)"; exit "$rc"; }
id="$(echo "$out" | tail -n 1 | grep -oE '^[0-9]+' || true)"
record "$id"
say "submitted: job $id ($NAME)"
