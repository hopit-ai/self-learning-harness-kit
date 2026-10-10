"""``python -m slh.demo1``: demo-1 commands.

Offline and free: build, verify, estimate, preflight, registration,
artifacts-row, owner-show, owner-verdict, owner-status, report, validate-all.
GPU (inside ``scripts/w4d.sh`` only): stage learn (window A) | stage test
(window B).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from slh.demo1 import pipeline


def _print(value: object) -> None:
    print(json.dumps(value, sort_keys=True, indent=1))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m slh.demo1", description=__doc__)
    cmd = parser.add_subparsers(dest="command", required=True)
    cmd.add_parser("build", help="generate both tasks, write demo1/, seal sealed/demo1/")
    cmd.add_parser("verify", help="re-generate and compare with demo1/manifest.json")
    cmd.add_parser("estimate", help="write demo1/estimate.json (calls, tokens, windows, $)")
    pre = cmd.add_parser("preflight", help="free checks before a window")
    pre.add_argument("--mode", required=True, choices=["learn", "test"])
    reg = cmd.add_parser("registration", help="the registry row that registers amendment 8 "
                                              "(print, or --append through the registry)")
    for sub in (reg,):
        sub.add_argument("--append", action="store_true")
        sub.add_argument("--expect-prefix-sha256", default=None)
        sub.add_argument("--registry-dir", type=Path, default=None)
    reg.add_argument("--supersedes", default=None,
                     help="row hash of the registration this row replaces (required "
                          "once amendment 8 is registered)")
    reg.add_argument("--reason", default=None, help="why it is superseded")
    cmd.add_parser("admission-deadline", help="print the active serving window's "
                                              "request-admission deadline (epoch seconds)")
    cmd.add_parser("validate-all", help="independent validator on all 168 cases "
                                        "(sealed via answer_validation access)")
    for name in ("owner-show", "owner-status", "report", "artifacts-row"):
        sub = cmd.add_parser(name)
        sub.add_argument("--run", type=Path, default=pipeline.RUN_DIR)
        if name == "artifacts-row":
            sub.add_argument("--append", action="store_true")
            sub.add_argument("--expect-prefix-sha256", default=None)
            sub.add_argument("--registry-dir", type=Path, default=None)
        if name == "owner-show":
            sub.add_argument("--task", required=True, choices=["address", "variant"])
    verdict = cmd.add_parser("owner-verdict", help="accept or reject one procedure (no edit)")
    verdict.add_argument("--run", type=Path, default=pipeline.RUN_DIR)
    verdict.add_argument("--task", required=True, choices=["address", "variant"])
    verdict.add_argument("--verdict", required=True, choices=["accept", "reject"])
    verdict.add_argument("--sha256", required=True, help="the procedure hash shown")
    verdict.add_argument("--note", default="")
    stage = cmd.add_parser("stage", help="GPU stage (scripts/w4d.sh only)")
    stage.add_argument("name", choices=["learn", "test"])
    stage.add_argument("--run", type=Path, default=pipeline.RUN_DIR)
    args = parser.parse_args(argv)

    if args.command == "build":
        from slh.demo1 import generate

        _print(generate.write_all(generate.build_all())["summary"])
        return 0
    if args.command == "verify":
        from slh.demo1 import generate

        result = generate.verify()
        _print(result)
        return 0 if result["ok"] else 1
    if args.command == "estimate":
        from slh.demo1 import estimate

        result = estimate.run()
        _print({k: result[k] for k in ("totals", "windows", "combined_gpu_reservation_usd",
                                       "within_cap", "excludes")})
        return 0 if result["within_cap"] else 2
    if args.command == "preflight":
        from slh.demo1.preflight import preflight

        try:
            result = preflight(args.mode)
        except (AssertionError, ImportError, OSError, ValueError, KeyError) as exc:
            print(f"preflight refused: {exc}")
            return 1
        for key, value in result.items():
            print(f"{key.upper()}={value}")
        print("preflight ok")
        return 0
    if args.command in ("registration", "artifacts-row"):
        from slh.demo1 import preflight as pf

        directory = args.registry_dir or pf.REGISTRY_DIR
        flags = ""
        if args.command == "registration":
            try:
                latest = pf.latest_registration(directory)
            except ValueError as exc:
                print(f"refused: {exc}")
                return 1
            supersedes = args.supersedes
            if latest is not None and not args.append:  # print: show what to confirm
                supersedes = supersedes or latest[1]
                flags = (f" --supersedes {supersedes} --reason "
                         f"{json.dumps(args.reason or '<reason>')}")
            row = pf.registration_row(supersedes=supersedes, reason=args.reason) \
                if supersedes is not None else pf.registration_row()
        else:
            row = {k: v for k, v in pipeline.artifacts_row(args.run).items()}
        if not args.append:
            _print({"row": row, "registry_prefix_sha256": pf.prefix_sha256(directory),
                    "append_with": f"python -m slh.demo1 {args.command} --append "
                                   f"--expect-prefix-sha256 <registry_prefix_sha256>{flags}"})
            return 0
        if not args.expect_prefix_sha256:
            print("refused: --append needs --expect-prefix-sha256")
            return 1
        try:
            done = pf.append_row(directory, row,
                                 expect_prefix_sha256=args.expect_prefix_sha256)
        except ValueError as exc:
            print(f"refused: {exc}")
            return 1
        print(f"appended {row['event']}; ROW_SHA256={done['row_sha256']}")
        return 0
    if args.command == "admission-deadline":
        from slh.demo1.context import runner_config
        from slh.serving import ServingWindowClosed, gate_for

        try:
            window = gate_for(runner_config().agent)._window()
        except ServingWindowClosed as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1
        print(int(window["request_deadline_epoch"]))
        return 0
    if args.command == "validate-all":
        from slh.demo1.preflight import validate_all

        result = validate_all()
        _print(result)
        return 0 if result["agree"] == result["cases"] else 1
    if args.command == "owner-show":
        from slh.demo1 import learn

        proc = json.loads((args.run / "artifacts" / f"learned-{args.task}.json").read_text())
        text_sha = learn.sha256_text(proc["text"]) if proc.get("text") else None
        print(f"task: {args.task}\nvalid: {proc['ok'] and text_sha == proc['sha256']}  "
              f"tokens: {proc['tokens']}  repaired: {proc['repaired']}\n"
              f"sha256 of the text: {text_sha}\n")
        print(proc["text"] if proc["ok"] else f"no valid procedure: {proc['problems']}")
        return 0
    if args.command == "owner-verdict":
        from slh.demo1 import learn

        proc = json.loads((args.run / "artifacts" / f"learned-{args.task}.json").read_text())
        if not proc["ok"] or learn.sha256_text(proc["text"]) != args.sha256 or \
                proc["sha256"] != args.sha256:
            print("refused: the hash does not match the written procedure text")
            return 1
        _print(learn.owner_record(pipeline.verdict_path(args.run), task=args.task,
                                  procedure_sha256=args.sha256, verdict=args.verdict,
                                  note=args.note))
        return 0
    if args.command == "owner-status":
        status = pipeline.owner_status(args.run)
        _print(status)
        return 0 if all(s in ("accept", "reject", "no_procedure") for s in status.values()) \
            else 3
    if args.command == "report":
        _print(pipeline.report(args.run))
        return 0
    if args.command == "stage":
        from slh.demo1.decide import AgentClient
        from slh.demo1.generate import _quiet

        _quiet()
        client = AgentClient.for_window()
        fn = {"learn": pipeline.stage_learn, "test": pipeline.stage_test}[args.name]
        _print(fn(args.run, client))
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
