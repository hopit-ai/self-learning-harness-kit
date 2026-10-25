#!/usr/bin/env bash
# Partner side (no network): the Hermes Agent environment for the Hermes run.
#
#   scripts/xp/setup_hermes_offline.sh ASSETS_DIR HERMES_CLONE [PREFIX]
#
# ASSETS_DIR holds cpython-3.14.*-x86_64-unknown-linux-gnu-install_only.tar.gz,
# wheelhouse-hermes-linux-x86_64-py314.tar.gz and hermes-assets.sha256.
# HERMES_CLONE is the Hermes Agent repository checked out at the pinned commit.
# PREFIX (default $PWD/hermes-env) receives python3.14/ and venv/ .
# Ends with a line {"ok": true, ...} when Hermes imports from the pinned clone.
set -euo pipefail
ASSETS="${1:?usage: $0 ASSETS_DIR HERMES_CLONE [PREFIX]}"
CLONE="${2:?usage: $0 ASSETS_DIR HERMES_CLONE [PREFIX]}"
PREFIX="${3:-$PWD/hermes-env}"
PIN=cb757fac59c92b49e9b8f2910e5cd7ab75efffdd
# real (symlink-free) paths: the venv records them, and the sandbox binds real paths
# (e.g. /home -> /storage/home on some clusters)
ASSETS="$(cd "$ASSETS" && pwd -P)"
CLONE="$(cd "$CLONE" && pwd -P)"
mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd -P)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
(cd "$ASSETS" && sha256sum -c hermes-assets.sha256)
[ "$(git -C "$CLONE" rev-parse HEAD)" = "$PIN" ] || { echo "Hermes clone is not at $PIN"; exit 4; }
[ -z "$(git -C "$CLONE" status --porcelain --untracked-files=no)" ] || {
  echo "Hermes clone has local changes; use a clean checkout"; exit 4; }
mkdir -p "$PREFIX"
if [ ! -x "$PREFIX/python3.14/bin/python3.14" ]; then
  rm -rf "$PREFIX/python3.14" "$PREFIX/python"
  tar xzf "$ASSETS"/cpython-3.14.*-x86_64-unknown-linux-gnu-install_only.tar.gz -C "$PREFIX"
  mv "$PREFIX/python" "$PREFIX/python3.14"
fi
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
tar xzf "$ASSETS/wheelhouse-hermes-linux-x86_64-py314.tar.gz" -C "$TMP"
WH="$TMP/wheelhouse-hermes"
rm -rf "$PREFIX/venv"
"$PREFIX/python3.14/bin/python3.14" -m venv "$PREFIX/venv"
# The lock is uv's FULL resolution of Hermes' pinned pyproject with its own [tool.uv]
# override-dependencies (e.g. Pillow 13.0.0 over browser-harness's pillow==12.3.0). pip
# cannot apply those overrides, so it must not re-resolve: install exactly the locked set
# (--no-deps), then pip check, accepting only mismatches that a Hermes override explains.
"$PREFIX/venv/bin/pip" install -q --no-index --no-deps --find-links "$WH" \
  -r "$WH/hermes-requirements.lock"
"$PREFIX/venv/bin/python" "$HERE/hermes_pip_check.py" "$CLONE/pyproject.toml" || {
  echo "pip check found a dependency mismatch that no Hermes override explains"; exit 4; }
# the package itself is not built or installed: a .pth points at the pinned clone
SP="$("$PREFIX/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$CLONE" > "$SP/hermes_agent_pinned_clone.pth"
"$PREFIX/venv/bin/python" -I -B - "$CLONE" <<'EOF'
import json, os, sys
import run_agent
ok = os.path.realpath(run_agent.__file__).startswith(os.path.realpath(sys.argv[1]))
print(json.dumps({"ok": ok, "python": sys.version.split()[0], "run_agent": run_agent.__file__}))
sys.exit(0 if ok else 1)
EOF
echo "HERMES_VENV=$PREFIX/venv  HERMES_CLONE=$CLONE"
