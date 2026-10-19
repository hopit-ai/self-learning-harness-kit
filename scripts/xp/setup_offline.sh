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
#      its packages: only the pinned benchmark is (re)bound to THIS kit's
#      vendor/tau2-bench (--no-deps, offline) when it is missing or bound to another
#      checkout (e.g. the phase-1 kit), an installed slh likewise, and the check then
#      decides (package versions are not held to the lock);
#   4. run the disconnected setup check (python -m slh.xp.partner.offline check).
# Use the Python of the environment you run with: PYTHON=/path/to/python (default:
# python3). Exit 0 = ready.
set -euo pipefail
# A relative interpreter or path given by the caller is resolved against the
# caller's directory now, before any cd (a bare command name stays a PATH lookup).
ASSETS="${1:?usage: setup_offline.sh ASSETS_DIR [--reuse-env]}"
case "$ASSETS" in /*) ;; *) ASSETS="$PWD/$ASSETS" ;; esac
REUSE="${2:-}"
PY="${PYTHON:-python3}"
case "$PY" in /*|"") ;; */*) PY="$PWD/$PY" ;; esac
cd "$(dirname "$0")/../.."
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
KIT="$(pwd -P)"
# Where MODULE resolves in this environment from outside the kit directory (the
# installed binding, not the kit's own source tree); empty when it is not installed.
resolved() {
  (cd / && "$PY" -I -c 'import importlib.util as u, os, sys
s = u.find_spec(sys.argv[1])
print(os.path.realpath(s.origin) if s and s.origin else "")' "$1" 2>/dev/null) || true
}
# tau2 must be the editable install of THIS kit's vendor/tau2-bench. A reused
# environment (e.g. phase 1's) may hold an editable tau2 bound to an older kit
# checkout: rebind it here, offline, without dependencies.
bind_tau2() {
  case "$(resolved tau2)" in
    "$KIT/vendor/tau2-bench/"*) return 0 ;;
  esac
  echo "binding tau2 to $KIT/vendor/tau2-bench (offline, --no-deps)" >&2
  "$PY" -m pip install -q --no-index --find-links "$WH" --no-deps -e vendor/tau2-bench || {
    echo "cannot install vendor/tau2-bench from the wheelhouse; fix: $PY -m pip install --no-index --find-links $WH --no-deps -e vendor/tau2-bench" >&2; exit 1; }
  case "$(resolved tau2)" in
    "$KIT/vendor/tau2-bench/"*) ;;
    *) echo "tau2 still resolves to '$(resolved tau2)', not $KIT/vendor/tau2-bench; fix: $PY -m pip uninstall -y tau2 && rerun this script" >&2; exit 1 ;;
  esac
}
# The kit runs from its own directory and is not installed. An slh installed in the
# environment (e.g. an editable install of an older kit) would be imported instead
# outside the kit directory: rebind it to this kit when it points elsewhere.
bind_slh() {
  local at; at="$(resolved slh)"
  case "$at" in
    ""|"$KIT/slh/"*) return 0 ;;
  esac
  echo "binding slh (installed at $at) to $KIT (offline, --no-deps)" >&2
  # --no-build-isolation: the environment's own setuptools builds the editable binding.
  "$PY" -m pip install -q --no-index --find-links "$WH" --no-deps --no-build-isolation -e . || {
    echo "ERROR: cannot rebind slh (installed at $at) to $KIT: the offline editable install failed (see the pip output above; it needs setuptools in this environment). Fix: $PY -m pip uninstall -y slh (the kit runs from its directory) and rerun this script" >&2
    exit 1; }
  case "$(resolved slh)" in
    ""|"$KIT/slh/"*) ;;
    *) echo "slh resolves to '$(resolved slh)', not this kit; fix: $PY -m pip uninstall -y slh (the kit runs from its directory) and rerun this script" >&2; exit 1 ;;
  esac
}
if [ "$REUSE" = "--reuse-env" ]; then
  # Keep the existing environment's packages (never install the lock over them);
  # only (re)bind the pinned benchmark and slh to this kit (no dependencies).
  bind_tau2
  bind_slh
  exec "$PY" -m slh.xp.partner.offline check --assets "$ASSETS" --wheelhouse "$WH"
fi
"$PY" -m pip install --no-index --find-links "$WH" -r "$WH/requirements.lock"
bind_tau2
bind_slh
exec "$PY" -m slh.xp.partner.offline check --assets "$ASSETS" --lock "$WH/requirements.lock" \
  --wheelhouse "$WH"
