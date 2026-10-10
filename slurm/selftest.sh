#!/bin/bash
# Rehearse the Slurm job scripts on a machine without Slurm or GPUs (login node or
# laptop): a mock Slurm 25.05 (slurm/sim/bin: sbatch, srun, scontrol, squeue, sacct,
# nvidia-smi) and a fake vLLM server stand in for the cluster. The REAL runner, package
# and checks run. Nothing is downloaded; no GPU is used.
#
#   bash $KIT/slurm/selftest.sh [--phase1] [--in-step] [--keep]
#
#   --phase1   also run phase1.sbatch (smoke setting, 3 test cases per task)
#   --in-step  pretend the job script already runs inside an srun step (the
#              dispatcher case): the servers must start as child processes
# Needs: HARNESS_PY (default $WORK/venv/bin/python) with the kit installed, and the
# benchmark checkout ($KIT/vendor/tau2-bench, from scripts/xp/fetch_vendor.sh).
# Ends with "selftest ok" or exits non-zero naming the failed row.
set -euo pipefail
KIT="${KIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HARNESS_PY="${HARNESS_PY:-${WORK:-.}/venv/bin/python}"
# The runner reads the 9B's tokenizer from the Hugging Face cache (download.sbatch).
HF_HOME="${HF_HOME:-${WORK:+$WORK/hf-home}}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
PHASE1=0; IN_STEP=0; KEEP=0
for a in "$@"; do
  case "$a" in --phase1) PHASE1=1 ;; --in-step) IN_STEP=1 ;; --keep) KEEP=1 ;;
    *) echo "unknown option $a" >&2; exit 2 ;; esac
done
BASE="$(mktemp -d "${TMPDIR:-/tmp}/xp-selftest.XXXXXX")"
cleanup() { if [[ $KEEP == 0 && ${OK:-0} == 1 ]]; then rm -rf "$BASE"; else echo "kept: $BASE" >&2; fi; }
trap cleanup EXIT
export KIT HARNESS_PY WORK="$BASE/work" SIM_SLURM_STATE="$BASE/slurm-state"
export PATH="$KIT/slurm/sim/bin:$PATH" SERVE_MODE=venv VLLM_BIN="$KIT/slurm/sim/bin/vllm"
export MODELS="$WORK/models" XP_STOP_GRACE_S=1 XP_TMPDIR="$BASE/t"
export TEACHER_STARTUP_TIMEOUT_S="${TEACHER_STARTUP_TIMEOUT_S:-120}" AGENT_STARTUP_TIMEOUT_S="${AGENT_STARTUP_TIMEOUT_S:-120}"
mkdir -p "$MODELS/agent" "$MODELS/teacher" "$WORK"
[[ $IN_STEP == 1 ]] && export SIM_SBATCH_ENV="SLURM_STEP_ID=0" SIM_SRUN_BUSY=fail
cd "$BASE"

echo "[selftest] phase 0 under mock Slurm" >&2
sbatch --parsable "$KIT/slurm/phase0.sbatch" > "$BASE/job0"
JOB="$(cat "$BASE/job0")"
STATUS="$WORK/phase0/job-$JOB/phase0.json"
[[ -f "$STATUS" ]] || { echo "no $STATUS; see $BASE/xp-phase0-$JOB.out" >&2; exit 1; }
"$HARNESS_PY" - "$STATUS" "$PHASE1" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))["rows"]
need = ["facts", "vendor", "dry-run", "serve", "probe-agent", "probe-teacher", "smoke"]
got = {r["row"]: r["status"] for r in rows}
bad = [n for n in need if got.get(n) != "pass"] + [r["row"] for r in rows if r["status"] != "pass"]
for r in rows:
    print("  %-14s %s %ss %s" % (r["row"], r["status"], r["seconds"], r["note"]), file=sys.stderr)
sys.exit("failed rows: %s" % sorted(set(bad)) if bad else 0)
PY
ARCHIVE="$WORK/returns/xp-phase0-$JOB.tar.gz"
[[ -s "$ARCHIVE" ]] || { echo "no archive $ARCHIVE" >&2; exit 1; }
echo "[selftest] phase-0 archive: $(wc -c < "$ARCHIVE") bytes" >&2

if [[ $PHASE1 == 1 ]]; then
  echo "[selftest] phase 1 under mock Slurm (smoke, 3 cases per task)" >&2
  RUNNER_ARGS="--smoke --max-cases 3" sbatch --parsable "$KIT/slurm/phase1.sbatch" > "$BASE/job1" || true
  JOB1="$(cat "$BASE/job1")"
  grep -q '"runner_exit": "0"' "$WORK/phase1/job-$JOB1/phase1.json" \
    || { echo "phase 1 did not finish; see $BASE/xp-phase1-$JOB1.out" >&2; exit 1; }
  [[ -s "$WORK/returns/xp-phase1-export.tar.gz" ]] || { echo "no phase-1 export archive" >&2; exit 1; }
fi
OK=1
echo "selftest ok"
