#!/usr/bin/env bash
# Fetch the vendored benchmark at the pinned upstream commit and verify it
# (partner side; vendor/ is git-ignored, so the git bundle does not carry it).
#
#   scripts/xp/fetch_vendor.sh            # clone + checkout + verify
#   scripts/xp/fetch_vendor.sh --verify   # verify an existing checkout only
#
# Pins (slh/xp/partner/vendor.py): upstream URL, commit, tree hash and the
# SHA-256 of the retail db.json and policy.md. Exit 1 on any mismatch.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
URL="https://github.com/sierra-research/tau2-bench"
COMMIT="672227c6b6676edc20d57ea53b7000262aae77b9"
DEST="vendor/tau2-bench"
if [[ "${1:-}" != "--verify" ]]; then
  if [[ -e "$DEST" ]]; then
    echo "$DEST exists; verifying it instead of fetching" >&2
  else
    mkdir -p "$DEST"
    git -C "$DEST" init -q
    git -C "$DEST" remote add origin "$URL"
    git -C "$DEST" fetch -q --depth 1 origin "$COMMIT"
    git -C "$DEST" checkout -q --detach FETCH_HEAD
  fi
fi
exec "$PY" -m slh.xp.partner.vendor
