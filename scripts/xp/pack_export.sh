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
#
# The Hermes run (NAME handoff-mh1*; owner 2026-10-25): the whole hand-back must stay
# UNDER 20,000,000 bytes, so the pack is refused when the archive, or all returned parts
# plus the NAME.sha256 file, reach 20,000,000 bytes or more, and a
# *.log file larger than 200,000 bytes is packed as its LAST 200,000 bytes
# (NAME.logtails/<file>, listed in NAME.logtails.txt; server logs can be very large and
# only their end is needed). PACK_MAX_BYTES may only LOWER the cap (tests).
set -euo pipefail
NAME="${1:?usage: $0 NAME PATH [PATH ...]}"
shift
[ "$#" -ge 1 ] || { echo "usage: $0 NAME PATH [PATH ...]"; exit 2; }
CAP=100000001
LOGTAIL=""
case "$NAME" in
  handoff-mh1*) CAP=20000000; LOGTAIL=200000 ;;
esac
if [ -n "${PACK_MAX_BYTES:-}" ] && [ "$PACK_MAX_BYTES" -lt "$CAP" ]; then CAP=$PACK_MAX_BYTES; fi
HAVE=()
MISSING=()
TAILED=()
rm -rf "$NAME.logtails"
for p in "$@"; do
  if [ ! -e "$p" ]; then MISSING+=("$p"); continue; fi
  if [ -n "$LOGTAIL" ] && [ -f "$p" ] && [ "${p%.log}" != "$p" ] \
      && [ "$(wc -c < "$p" | tr -d ' ')" -gt "$LOGTAIL" ]; then
    mkdir -p "$NAME.logtails/$(dirname "$p")"
    tail -c "$LOGTAIL" "$p" > "$NAME.logtails/$p"
    TAILED+=("$p")
    HAVE+=("$NAME.logtails/$p")
  else
    HAVE+=("$p")
  fi
done
if [ "${#HAVE[@]}" -eq 0 ]; then
  echo "REFUSED: none of $* exists; nothing to pack"; exit 1
fi
LIST=()
if [ "${#MISSING[@]}" -gt 0 ]; then
  printf '%s\n' "${MISSING[@]}" > "$NAME.missing.txt"
  LIST+=("$NAME.missing.txt")
  echo "not found (recorded in $NAME.missing.txt, packed anyway): ${MISSING[*]}"
fi
if [ "${#TAILED[@]}" -gt 0 ]; then
  printf '%s\n' "${TAILED[@]}" > "$NAME.logtails.txt"
  LIST+=("$NAME.logtails.txt")
  echo "packed as their last $LOGTAIL bytes: ${TAILED[*]}"
fi
tar czf "$NAME.tar.gz" "${HAVE[@]}" ${LIST[@]+"${LIST[@]}"}
SIZE=$(wc -c < "$NAME.tar.gz" | tr -d ' ')
if [ "$SIZE" -ge "$CAP" ]; then
  echo "REFUSED: $NAME.tar.gz is $SIZE bytes (it must stay under $CAP bytes); do not send"
  echo "it: tell us the size of each folder you packed (du -sh) and we will say what to do"
  rm -f "$NAME.tar.gz"
  exit 1
fi
rm -f "$NAME.tar.gz.part-"*
split -b 10000000 -a 2 "$NAME.tar.gz" "$NAME.tar.gz.part-"
if command -v sha256sum >/dev/null; then SUM=sha256sum; else SUM="shasum -a 256"; fi
$SUM "$NAME.tar.gz" "$NAME.tar.gz.part-"* > "$NAME.sha256"
# what is sent back is every part plus the checksum file: all of it counts against the cap
TOTAL=$(cat "$NAME.tar.gz.part-"* "$NAME.sha256" | wc -c | tr -d ' ')
if [ "$TOTAL" -ge "$CAP" ]; then
  echo "REFUSED: the parts and $NAME.sha256 are $TOTAL bytes (they must stay under $CAP"
  echo "bytes); do not send them: tell us the size of each folder you packed (du -sh)"
  rm -f "$NAME.tar.gz" "$NAME.tar.gz.part-"* "$NAME.sha256"
  exit 1
fi
echo "$NAME: $SIZE bytes in $(ls "$NAME.tar.gz.part-"* | wc -l | tr -d ' ') part(s); send the parts and $NAME.sha256"
