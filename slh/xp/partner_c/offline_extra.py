"""Offline install of the training extras for hand-off 2 (no network at the partner).

Hand-off 1 needs nothing beyond the phase-1 plain kit (vLLM serves the 9B; our code is
plain Python on numpy and the pinned benchmark). Hand-off 2 adds the training stack.
The partner's vLLM environment already carries torch and transformers; ``peft`` (and
its ``accelerate`` dependency) and ``safetensors`` are the additions.

Our side, on a networked Linux x86_64 / CPython 3.12 machine that has the pinned
versions installed (never this laptop)::

    python -m slh.xp.partner_c.offline_extra lock > train-requirements.lock
    pip download --only-binary=:all: --platform manylinux_2_28_x86_64 \
        --python-version 3.12 --no-deps -r train-requirements.lock -d wheelhouse-train/
    python -m slh.xp.partner.offline verify-wheelhouse --wheelhouse wheelhouse-train/ \
        --lock train-requirements.lock

Partner side::

    pip install --no-index --find-links wheelhouse-train/ -r train-requirements.lock
    python -m slh.xp.partner_c.offline_extra check     # exit 0 = ready

``check`` opens no connection: it imports each package and compares versions with the
minimums below (the exact pins are recorded in the registration and in every job.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata

TRAIN_ROOTS = ("peft", "safetensors")
MINIMUM = {"torch": "2.4", "transformers": "4.48", "peft": "0.13", "safetensors": "0.4"}


def _tuple(v: str) -> tuple[int, ...]:
    out = []
    for part in v.split("+")[0].split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def check() -> dict[str, object]:
    import socket

    real = socket.socket.connect

    def refuse(*_a: object, **_k: object) -> None:
        raise OSError("network disabled during the offline check")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    problems, found = [], {}
    try:
        for name, minimum in MINIMUM.items():
            try:
                v = metadata.version(name)
            except metadata.PackageNotFoundError:
                problems.append(f"{name} is not installed")
                continue
            found[name] = v
            if _tuple(v) < _tuple(minimum):
                problems.append(f"{name} {v} < {minimum}")
            try:
                __import__(name)
            except Exception as exc:  # noqa: BLE001 - reported
                problems.append(f"{name} does not import: {exc}")
    finally:
        socket.socket.connect = real  # type: ignore[method-assign]
    return {"ok": not problems, "versions": found, "problems": problems}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.partner_c.offline_extra")
    ap.add_argument("cmd", choices=("lock", "check"))
    args = ap.parse_args(argv)
    if args.cmd == "lock":
        from slh.xp.partner import offline

        print("\n".join(offline.lock(roots=TRAIN_ROOTS)))
        return 0
    res = check()
    print(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
