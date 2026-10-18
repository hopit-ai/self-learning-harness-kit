#!/usr/bin/env bash
# Disconnected setup (no outgoing network needed), from the kit directory:
#
#   scripts/xp/setup_offline.sh ASSETS_DIR [--reuse-env]
#
# ASSETS_DIR holds the kit release's assets: tau2-bench-<commit>.tar.gz,
# wheelhouse-linux-x86_64-py312.tar.gz and offline-assets.json. Steps:
#   1. verify every asset against offline-assets.json;
#   2. unpack the pinned benchmark into vendor/tau2-bench (kept if already there)
#      and verify it (commit, tree, data hashes);
#   3. install from the wheelhouse only (pip --no-index --find-links). With
#      --reuse-env, an existing environment (e.g. the phase-1 one with vLLM) keeps
#      its packages: only the pinned benchmark is added (--no-deps) if missing,
#      and the check then decides (package versions are not held to the lock);
#   4. run the disconnected setup check (python -m slh.xp.partner.offline check).
# Use the Python of the environment you run with: PYTHON=/path/to/python (default:
# python3). Exit 0 = ready.
set -euo pipefail
cd "$(dirname "$0")/../.."
ASSETS="${1:?usage: setup_offline.sh ASSETS_DIR [--reuse-env]}"
REUSE="${2:-}"
PY="${PYTHON:-python3}"
export LITELLM_LOCAL_MODEL_COST_MAP=True   # never fetch litellm's price map
export PIP_NO_INDEX=1
COMMIT="$("$PY" -c 'from slh.xp.partner import vendor; print(vendor.COMMIT)')"
WH=".offline/wheelhouse"
"$PY" - "$ASSETS" <<'PYEOF'
import sys
from pathlib import Path
from slh.xp.partner import offline
problems = offline.asset_problems(Path(sys.argv[1]))
if problems:
    sys.exit("assets: " + "; ".join(problems))
PYEOF
if [ ! -e vendor/tau2-bench ]; then
  mkdir -p vendor
  tar xzf "$ASSETS/tau2-bench-$COMMIT.tar.gz" -C vendor
fi
"$PY" -m slh.xp.partner.vendor >/dev/null || { echo "vendor/tau2-bench differs from the pin" >&2; exit 1; }
mkdir -p .offline
[ -d "$WH" ] || tar xzf "$ASSETS/wheelhouse-linux-x86_64-py312.tar.gz" -C .offline
# The verifier parses requirements with `packaging` (its only non-stdlib module). A
# fresh environment has none: bootstrap the locked version from the wheelhouse (pure
# Python, no dependencies, no network). An environment that already has it
# (--reuse-env) is left as it is.
if ! "$PY" -c 'import packaging.requirements, packaging.version' 2>/dev/null; then
  PKG_PIN="$(cat "$WH/requirements.lock" "$WH/build.lock" 2>/dev/null \
             | sed -n 's/^\(packaging==[^ ;\\]*\).*/\1/p' | head -1 || true)"
  [ -n "$PKG_PIN" ] || { echo "the wheelhouse locks do not pin packaging" >&2; exit 1; }
  "$PY" -m pip install -q --no-index --find-links "$WH" --no-deps "$PKG_PIN" || {
    echo "cannot bootstrap $PKG_PIN from the wheelhouse" >&2; exit 1; }
fi
# Before installing anything: the wheelhouse holds the build backend's (hatchling +
# editables) whole closure and every pinned runtime requirement's (no network).
"$PY" -m slh.xp.partner.offline verify-wheelhouse --wheelhouse "$WH" \
  --lock "$WH/requirements.lock" >/dev/null || {
  echo "the wheelhouse is incomplete (see: $PY -m slh.xp.partner.offline verify-wheelhouse --wheelhouse $WH --lock $WH/requirements.lock)" >&2; exit 1; }
if [ "$REUSE" = "--reuse-env" ]; then
  # Keep the existing environment's packages (never install the lock over them);
  # only add the pinned benchmark (no dependencies) if it is not installed yet.
  "$PY" -c 'import tau2' 2>/dev/null || \
    "$PY" -m pip install --no-index --find-links "$WH" --no-deps -e vendor/tau2-bench
  exec "$PY" -m slh.xp.partner.offline check --assets "$ASSETS" --wheelhouse "$WH"
fi
"$PY" -m pip install --no-index --find-links "$WH" -r "$WH/requirements.lock"
"$PY" -m pip install --no-index --find-links "$WH" --no-deps -e vendor/tau2-bench
exec "$PY" -m slh.xp.partner.offline check --assets "$ASSETS" --lock "$WH/requirements.lock" \
  --wheelhouse "$WH"
