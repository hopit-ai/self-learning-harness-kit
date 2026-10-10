#!/bin/bash
#SBATCH --job-name=xp-download
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=%x-%j.out
#
# Zero-GPU compute job: download both models at their revisions into $WORK/models
# (Hugging Face is reachable from compute nodes, not from the login node). Writes
# $WORK/models/download.json (revisions resolved, file count, bytes). No token is
# needed for public repositories; if one is, set HF_TOKEN in the environment.
#
#   export WORK=... KIT=$WORK/kit
#   sbatch [--account=.. --qos=.. --partition=..] "$KIT/slurm/download.sbatch"
set -euo pipefail
if [[ -z "${KIT:-}" ]]; then
  for c in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/.." "$(dirname "$0")/.."; do
    if [[ -f "$c/slurm/lib.sh" ]]; then KIT="$(cd "$c" && pwd)"; break; fi
  done
fi
source "$KIT/slurm/lib.sh" || exit $?
export HF_HUB_OFFLINE=0
mkdir -p "$MODELS"
xp_spec "$MODELS/serve-spec.json"   # the pinned values, via the trusted bootstrap
AGENT_REPO="$("$SITE_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_repo"])' "$XP_SERVE_SPEC")"
AGENT_REVISION="$("$SITE_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_revision"])' "$XP_SERVE_SPEC")"
"$HARNESS_PY" - "$MODELS" "$AGENT_REPO" "$AGENT_REVISION" "$TEACHER_ENABLE" "$TEACHER_REPO" \
  "${TEACHER_REVISION:-}" <<'PY'
import json, os, sys, time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

models, arepo, arev, teach, trepo, trev = sys.argv[1:7]
models = Path(models)
models.mkdir(parents=True, exist_ok=True)
out = {"time_epoch": time.time()}
jobs = [("agent", arepo, arev)] + ([("teacher", trepo, trev or None)] if teach == "1" else [])
for name, repo, rev in jobs:
    rev = rev or HfApi().model_info(repo).sha  # unpinned: resolve once and record
    t = time.time()
    path = Path(snapshot_download(repo, revision=rev, max_workers=8))
    link = models / name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(path)
    files = [p for p in path.rglob("*") if p.is_file()]
    out[name] = {"repo": repo, "revision": rev, "files": len(files),
                 "bytes": sum(p.stat().st_size for p in files),
                 "seconds": round(time.time() - t, 1)}
    print(json.dumps({name: out[name]}), flush=True)
(models / "download.json").write_text(json.dumps(out, indent=1) + "\n")
PY
echo "download ok: $MODELS/download.json"
