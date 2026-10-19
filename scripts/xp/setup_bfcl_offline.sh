#!/usr/bin/env bash
# Partner side (no network): the BFCL environment, separate from the model server's.
#
#   scripts/xp/setup_bfcl_offline.sh wheelhouse-bfcl-linux-x86_64-py312.tar.gz [VENV]
#
# Creates VENV (default ~/bfcl-venv) with Python 3.12, installs the pinned benchmark
# wheel and its locked dependencies from the wheelhouse only, then runs the offline
# readiness check (every network connection refused while it runs). Exit 0 = ready.
set -euo pipefail
TARBALL="${1:?usage: $0 WHEELHOUSE_TARBALL [VENV]}"
# A relative interpreter or path given by the caller is resolved against the
# caller's directory now, before any cd (a bare command name stays a PATH lookup).
case "$TARBALL" in /*) ;; *) TARBALL="$PWD/$TARBALL" ;; esac
VENV="${2:-$HOME/bfcl-venv}"
case "$VENV" in /*) ;; *) VENV="$PWD/$VENV" ;; esac
cd "$(dirname "$0")/../.."
TMP="$(mktemp -d)"
tar xzf "$TARBALL" -C "$TMP"
WH="$TMP/wheelhouse-bfcl"
python3.12 -m venv "$VENV"
"$VENV/bin/pip" install -q --no-index --find-links "$WH" -r "$WH/bfcl-requirements.lock"
"$VENV/bin/pip" install -q --no-index --no-deps "$WH"/bfcl_eval-*.whl
exec "$VENV/bin/python" -m slh.xp.bfcl.runner check
