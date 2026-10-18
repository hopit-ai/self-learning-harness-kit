#!/usr/bin/env bash
# Partner side: pack result folders (or files) for sending back.
#
#   scripts/xp/pack_export.sh NAME PATH [PATH ...]
#
# Packs the PATHs that exist (a missing one is listed in NAME.missing.txt inside the
# archive and on screen, never an error: after a failed setup or smoke run, pack the logs
# and run folders that do exist). Refuses only when none exists. Writes NAME.tar.gz,
# refuses it above 100 MB (then tell us; do not drop files), splits it into parts of at
# most 10 MB = 10,000,000 bytes (NAME.tar.gz.part-aa, -ab, ...) and writes NAME.sha256
# with the checksum of the whole archive and of every part (bare file names). Send the
# parts and NAME.sha256. We rejoin with:
#   cat NAME.tar.gz.part-* > NAME.tar.gz && sha256sum -c NAME.sha256
set -euo pipefail
NAME="${1:?usage: $0 NAME PATH [PATH ...]}"
shift
[ "$#" -ge 1 ] || { echo "usage: $0 NAME PATH [PATH ...]"; exit 2; }
HAVE=()
MISSING=()
for p in "$@"; do
  if [ -e "$p" ]; then HAVE+=("$p"); else MISSING+=("$p"); fi
done
if [ "${#HAVE[@]}" -eq 0 ]; then
  echo "REFUSED: none of $* exists; nothing to pack"; exit 1
fi
LIST=()
if [ "${#MISSING[@]}" -gt 0 ]; then
  printf '%s\n' "${MISSING[@]}" > "$NAME.missing.txt"
  LIST=("$NAME.missing.txt")
  echo "not found (recorded in $NAME.missing.txt, packed anyway): ${MISSING[*]}"
fi
tar czf "$NAME.tar.gz" "${HAVE[@]}" ${LIST[@]+"${LIST[@]}"}
SIZE=$(wc -c < "$NAME.tar.gz" | tr -d ' ')
if [ "$SIZE" -gt 100000000 ]; then
  echo "REFUSED: $NAME.tar.gz is $SIZE bytes (over 100 MB); tell us before sending"; exit 1
fi
rm -f "$NAME.tar.gz.part-"*
split -b 10000000 -a 2 "$NAME.tar.gz" "$NAME.tar.gz.part-"
if command -v sha256sum >/dev/null; then SUM=sha256sum; else SUM="shasum -a 256"; fi
$SUM "$NAME.tar.gz" "$NAME.tar.gz.part-"* > "$NAME.sha256"
echo "$NAME: $SIZE bytes in $(ls "$NAME.tar.gz.part-"* | wc -l | tr -d ' ') part(s); send the parts and $NAME.sha256"
