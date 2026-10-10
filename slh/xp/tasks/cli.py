"""Generator CLI for the XP decision tasks.

    python -m slh.xp.tasks.cli summary  --task variant [--test-cases 1000] [--seed N]
                                        [--validation-cases 60] [--sentinel-cases 30]
    python -m slh.xp.tasks.cli validate --task address [--test-cases 30]
    python -m slh.xp.tasks.cli export   --task variant --out DIR [--splits train,validation]
    python -m slh.xp.tasks.cli score    --key DIR/x.answer-key.jsonl --actions actions.jsonl

Tasks: ``address`` (POC 9's decision, rescaled), ``variant``, ``sequence``.
Splits: train, validation (development scores), sentinel (separate selection
set), test. ``summary`` and ``validate`` work in memory. ``export`` writes, per split,
``<task>-<split>.questions.jsonl`` (no gold, with case_hash) and
``<task>-<split>.answer-key.jsonl``; it refuses to write a test split inside
the repository. ``score`` reads action rows ``{"case_id", "case_hash",
"action"}`` and prints the summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from slh.xp.tasks import common as cm


def task_module(task: str) -> Any:
    if task == "address":
        from slh.xp.tasks import address as mod
    elif task == "variant":
        from slh.xp.tasks import variant as mod
    elif task == "sequence":
        from slh.xp.tasks import sequence as mod
    else:
        raise SystemExit(f"unknown task {task!r}")
    return mod


def build(task: str, *, seed: int = cm.SEED, test_cases: int = cm.DEFAULT_CASES["test"],
          splits: tuple[str, ...] = cm.SPLITS,
          validation_cases: int = cm.DEFAULT_CASES["validation"],
          sentinel_cases: int = cm.DEFAULT_CASES["sentinel"]) -> list[dict[str, Any]]:
    # Train, validation and sentinel are always drawn first (in memory), so a
    # test split is the same whichever splits are requested; test is skipped
    # when unused.
    sizes = cm.group_sizes(test_cases, validation_cases=validation_cases,
                           sentinel_cases=sentinel_cases)
    if "test" not in splits:
        sizes["test"] = 0
    cases = task_module(task).generate(seed=seed, sizes=sizes)
    return [c for c in cases if c["split"] in splits]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.tasks.cli")
    ap.add_argument("command", choices=("summary", "validate", "export", "score"))
    ap.add_argument("--task", choices=("address", "variant", "sequence"))
    ap.add_argument("--seed", type=int, default=cm.SEED)
    ap.add_argument("--test-cases", type=int, default=cm.DEFAULT_CASES["test"])
    ap.add_argument("--validation-cases", type=int, default=cm.DEFAULT_CASES["validation"])
    ap.add_argument("--sentinel-cases", type=int, default=cm.DEFAULT_CASES["sentinel"])
    ap.add_argument("--splits", default=",".join(cm.SPLITS))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--key", type=Path)
    ap.add_argument("--actions", type=Path)
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    if args.command == "score":
        from slh.xp.score import score_files

        res = score_files(args.key, args.actions)
        print(json.dumps({"missing": len(res["missing"]), **res["summary"]}, indent=1))
        return 0
    if not args.task:
        ap.error("--task is required")
    splits = tuple(s for s in args.splits.split(",") if s)
    if set(splits) - set(cm.SPLITS):
        ap.error(f"unknown split in {splits}")
    cases = build(args.task, seed=args.seed, test_cases=args.test_cases, splits=splits,
                  validation_cases=args.validation_cases, sentinel_cases=args.sentinel_cases)
    if args.command == "summary":
        print(json.dumps(cm.summary(cases), indent=1))
        return 0
    if args.command == "validate":
        from slh.xp.validate import validate_all

        res = validate_all(cases)
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 1
    if args.out is None:
        ap.error("export needs --out")
    out = {}
    for split in splits:
        rows = [c for c in cases if c["split"] == split]
        out[split] = cm.export(rows, args.out, f"{args.task}-{split}")["summary"]["cases"]
    print(json.dumps({"out": str(args.out), "cases": out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
