"""ALFWorld (text-only) readiness check: dependencies and the game data, offline.

    python -m slh.xp.alfworld.check [--data $ALFWORLD_DATA]

Conditional second domain of the BFCL track (docs/xp/bfcl-plan.md s. 9; hand-off 1 part C,
docs/xp/handoff1-runbook.md). Checks, without any network connection and without
starting a game:

- the packages import at the pinned versions: ``alfworld`` 0.4.2 (MIT, pure Python),
  ``textworld`` 1.7.0 (MIT; manylinux wheel), ``jericho`` 3.3.1 (GPL-2.0-or-later; sdist
  only, C build with make) and ``fast-downward-textworld`` 20.6.4 (GPL-3.0; sdist only,
  C++ build with CMake; import name ``fast_downward``, checked in the sdist's ``setup.py``
  on 2026-10-18). The two GPL packages are built on the partner node from the shipped,
  unmodified source archives (``scripts/xp/setup_alfworld_offline.sh``);
- the game data (``json_2.1.2_tw-pddl.zip`` of the ALFWorld release, unpacked into
  ``$ALFWORLD_DATA``): ``json_2.1.1/{train,valid_train,valid_seen,valid_unseen}`` with
  3,553 / 200 / 140 / 134 ``game.tw-pddl`` files (counted on the release archive
  2026-10-18; every game is marked solvable).

Exit 0 = ready, 1 = not ready (problems listed).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

PINS = {"alfworld": "0.4.2", "textworld": "1.7.0", "jericho": "3.3.1",
        "fast-downward-textworld": "20.6.4"}
SPLITS = {"train": 3553, "valid_train": 200, "valid_seen": 140, "valid_unseen": 134}


def games(root: Path) -> int:
    return sum(1 for _ in root.rglob("game.tw-pddl")) if root.is_dir() else 0


def check(data: Path | None) -> dict:
    from importlib import metadata

    problems, versions = [], {}
    for mod, dist in (("alfworld", "alfworld"), ("textworld", "textworld"),
                      ("jericho", "jericho"), ("fast_downward", "fast-downward-textworld")):
        try:
            importlib.import_module(mod)
            versions[dist] = metadata.version(dist)
        except Exception as exc:  # noqa: BLE001 - report every failure
            problems.append(f"import {mod}: {type(exc).__name__}: {exc}"[:200])
    for dist, want in PINS.items():
        if versions.get(dist) not in (None, want):
            problems.append(f"{dist} {versions[dist]} != pinned {want}")
    counts = {}
    if data is None:
        problems.append("no data directory (set ALFWORLD_DATA or --data)")
    else:
        for split, want in SPLITS.items():
            counts[split] = games(data / "json_2.1.1" / split)
            if counts[split] != want:
                problems.append(f"{split}: {counts[split]} games, expected {want}")
    return {"ok": not problems, "problems": problems, "versions": versions, "games": counts}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.alfworld.check")
    ap.add_argument("--data", type=Path, default=os.environ.get("ALFWORLD_DATA"))
    args = ap.parse_args(argv)
    rep = check(Path(args.data) if args.data else None)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
