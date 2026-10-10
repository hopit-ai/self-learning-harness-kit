"""The vendored benchmark: pinned upstream commit, tree and data hashes.

``vendor/`` is git-ignored, so the git bundle does not carry it. The partner
fetches it with ``scripts/xp/fetch_vendor.sh`` (pinned commit, then a hash
check); the runner refuses to start unless ``state()`` equals ``expected()``.
No network here: ``state`` reads the local checkout only.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from slh.xp.partner.common import REPO, sha_file

UPSTREAM = "https://github.com/sierra-research/tau2-bench"
COMMIT = "672227c6b6676edc20d57ea53b7000262aae77b9"  # v1.0.1 (pyproject.toml)
TREE = "50f44a22272c23aa12a51dc2000543bf51874bb9"
DATA = {
    "data/tau2/domains/retail/db.json":
        "413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765",
    "data/tau2/domains/retail/policy.md":
        "2c9652afbce57d6e087768d37cda64d31c53d50b3e3225cfdb791bac66466467",
}
PATH = REPO / "vendor/tau2-bench"


def expected() -> dict[str, Any]:
    return {"upstream": UPSTREAM, "commit": COMMIT, "tree": TREE, "data": dict(DATA)}


def state(path: Path = PATH) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                                  text=True, timeout=20, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    dirty = git("status", "--porcelain", "--untracked-files=no")
    return {"upstream": UPSTREAM, "commit": git("rev-parse", "HEAD"),
            "tree": git("rev-parse", "HEAD^{tree}"),
            "data": {rel: sha_file(path / rel) if (path / rel).is_file() else None
                     for rel in DATA},
            "dirty": None if dirty is None else bool(dirty)}


def problems(path: Path = PATH, want: dict[str, Any] | None = None) -> list[str]:
    want = want or expected()
    got = state(path)
    out = [f"{k}: {got[k]} != {want[k]}" for k in ("commit", "tree") if got[k] != want[k]]
    out += [f"{rel}: hash differs" for rel, h in want["data"].items() if got["data"].get(rel) != h]
    if got["dirty"]:
        out.append("vendor checkout has local modifications")
    return out


def main() -> int:
    import json

    found = problems()
    print(json.dumps({"ok": not found, "problems": found, "state": state()}, indent=1))
    return 0 if not found else 1


if __name__ == "__main__":
    raise SystemExit(main())
