#!/bin/bash
# Submit phase 1 with a time limit that fits the remaining GPU-hour budget.
#
#   export WORK=... KIT=$WORK/kit
#   bash $KIT/slurm/submit.sh [--account=.. --qos=.. --partition=..]
#
# Reads the phase-1 ledger ($WORK/phase1/gpu-hours.jsonl), computes
# min(MAX_JOB_S, (GPU_HOURS_CAP - used) x 3600 / 8) and runs
# `sbatch --time=<that> <your flags> $KIT/slurm/phase1.sbatch`. Refuses (exit 2) when
# the remainder is below MIN_USEFUL_S or another phase-1 job is still running. The job
# checks the budget again at start-up and trims its own limit.
set -euo pipefail
: "${WORK:?set WORK}"
KIT="${KIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SITE_PY="${SITE_PY:-${HARNESS_PY:-$WORK/venv/bin/python}}"
LEDGER="${LEDGER:-$WORK/phase1/gpu-hours.jsonl}"
OUTPUT="$("$SITE_PY" "$KIT/slurm/xpsite.py" budget --file "$LEDGER" \
  --cap "${GPU_HOURS_CAP:-1600}" --gpus "${XP_GPUS:-8}" --max-s "${MAX_JOB_S:-172800}" \
  --min "${MIN_USEFUL_S:-7200}" --job submit)" || { echo "not submitting: $OUTPUT" >&2; exit 2; }
BUDGET_S="$("$SITE_PY" -c 'import json,sys; print(json.loads(sys.argv[1])["budget_s"])' "$OUTPUT")"
MINUTES=$(( BUDGET_S / 60 ))
echo "budget: $OUTPUT" >&2
exec sbatch --time="$MINUTES" "$@" "$KIT/slurm/phase1.sbatch"
