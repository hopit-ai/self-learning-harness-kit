#!/usr/bin/env bash
# XP phase 1 (decisions) for the partner, mode B (docs/xp/partner-package.md).
#
#   scripts/xp/run_phase1.sh --dry-run          # fake client, no GPU; must print "export ok"
#   scripts/xp/run_phase1.sh                    # real run on the endpoint in $ENDPOINT
#   scripts/xp/run_phase1.sh --max-cases 20     # smoke: 20 test cases per task
#
# First: scripts/xp/fetch_vendor.sh. The runner refuses to start unless the checkout,
# the vendored benchmark and the package match registration.json.
# Rerun the same command after any interruption: finished rows are kept; exit 3 =
# halted (spend cap, auth, connection...): fix the cause and rerun; exit 4 = refused.
# Environment: PACKAGE (dir with plan.json, protocol.md, cases/), ENDPOINT (yaml),
# OUT (run dir), PYTHON (interpreter, Python >= 3.12).
# The runner starts through the trusted bootstrap scripts/xp/xp_boot.py: no bytecode is
# written, cached bytecode is read only from a fresh empty directory, and the kit is
# checked for planted import artifacts and registered code before slh is imported.
# --preflight: verification only (registration, benchmark, run identity).
set -euo pipefail
# A relative interpreter or path given by the caller is resolved against the
# caller's directory now, before any cd (a bare command name stays a PATH lookup).
case "${PYTHON:-}" in /*|"") ;; */*) PYTHON="$PWD/$PYTHON" ;; esac
cd "$(dirname "$0")/../.."
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
PACKAGE="${PACKAGE:-../xp-package}"
ENDPOINT="${ENDPOINT:-configs/xp-partner-endpoint.yaml}"
OUT="${OUT:-runs/xp-phase1}"
if [[ " $* " == *" --dry-run "* ]]; then
  OUT="${OUT}-dry-run"
fi
export PYTHONDONTWRITEBYTECODE=1
PYTHONPYCACHEPREFIX="$(mktemp -d "${TMPDIR:-/tmp}/xp-pyc.XXXXXX")"
export PYTHONPYCACHEPREFIX
exec "$PY" scripts/xp/xp_boot.py --package "$PACKAGE" --endpoint "$ENDPOINT" --out "$OUT" "$@"
