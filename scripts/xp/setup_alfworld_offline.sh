#!/usr/bin/env bash
# Partner side (no network): build and install the ALFWorld text environment, unpack the
# games, check, and write a build record to send back.
#
#   scripts/xp/setup_alfworld_offline.sh alfworld-offline-linux-x86_64-py312.tar.gz [VENV] [DATA]
#
# VENV defaults to ~/alfworld-venv (its own Python 3.12 environment: not the model
# server's, not the BFCL one), DATA to ~/alfworld-data. Needs on the node: python3.12,
# gcc, g++ and make (CMake comes from the shipped wheel). CPU only, about 5-15 minutes.
#
# Steps: install the locked wheels from the archive only; build jericho and
# fast-downward-textworld from their unmodified source archives into wheels; install them
# and the alfworld and textworld wheels; unpack the game archive into DATA; run
# `python -m slh.xp.alfworld.check`. Everything is logged to alfworld-build/ (build log,
# installed versions and licences, check output), which is sent back. Exit 0 = ready.
set -euo pipefail
cd "$(dirname "$0")/../.."
TARBALL="${1:?usage: $0 ARCHIVE [VENV] [DATA]}"
VENV="${2:-$HOME/alfworld-venv}"
DATA="${3:-$HOME/alfworld-data}"
REC="$PWD/alfworld-build"
mkdir -p "$REC"
exec > >(tee "$REC/build.log") 2>&1
echo "start $(date -u +%FT%TZ) on $(uname -srm)"
for tool in python3.12 gcc g++ make; do
  command -v "$tool" >/dev/null || { echo "MISSING: $tool (load a compiler module or install it, then rerun)"; exit 1; }
  echo "$tool: $("$tool" --version 2>&1 | head -1)"
done
TMP="$(mktemp -d)"
tar xzf "$TARBALL" -C "$TMP"
A="$TMP/alfworld-offline"
( cd "$A" && sha256sum -c SHA256SUMS --quiet ) && echo "archive checksums ok"
python3.12 -m venv "$VENV"
export PATH="$VENV/bin:$PATH"      # the source builds call `python` and `cmake` from PATH
pip install -q --no-index --find-links "$A/wheels" -r "$A/alfworld-requirements.lock"
mkdir -p "$TMP/built"
# --no-build-isolation: build with the tools installed above (offline); MAKEFLAGS parallel.
MAKEFLAGS="-j$(nproc)" pip wheel -v --no-index --no-deps --no-build-isolation \
  -w "$TMP/built" "$A/sdist/jericho-3.3.1.tar.gz" "$A/sdist/fast_downward_textworld-20.6.4.tar.gz" \
  > "$REC/source-builds.log" 2>&1 || { echo "SOURCE BUILD FAILED: see alfworld-build/source-builds.log"; exit 1; }
ls -l "$TMP/built"
pip install -q --no-index --no-deps "$TMP"/built/*.whl \
  "$A"/wheels/alfworld-0.4.2-*.whl "$A"/wheels/textworld-1.7.0-*.whl
pip check || echo "WARNING: pip check reported the problems above"
mkdir -p "$DATA"
python - "$A/data/json_2.1.2_tw-pddl.zip" "$DATA" <<'EOF'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])
EOF
python - "$REC/packages.json" <<'EOF'
import json, sys
from importlib import metadata
rows = []
for d in metadata.distributions():
    m = d.metadata
    lic = m.get("License-Expression") or (m.get("License") or "").splitlines()[0:1]
    rows.append({"name": m["Name"], "version": d.version,
                 "license": lic if isinstance(lic, str) else (lic[0] if lic else ""),
                 "classifiers": [c for c in m.get_all("Classifier") or [] if "License" in c]})
json.dump(sorted(rows, key=lambda r: r["name"].lower()), open(sys.argv[1], "w"), indent=1)
EOF
cp "$A/LICENSES.md" "$A/alfworld-requirements.lock" "$REC/"
set +e
python -m slh.xp.alfworld.check --data "$DATA" | tee "$REC/check.json"
rc=${PIPESTATUS[0]}
echo "end $(date -u +%FT%TZ) check exit $rc"
exit "$rc"
