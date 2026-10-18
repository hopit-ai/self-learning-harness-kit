"""The vendored BFCL (Berkeley Function Calling Leaderboard) evaluator and data: pins.

Pinned 2026-10-18 (read only; nothing executed):

- upstream ``https://github.com/ShishirPatil/gorilla``, subdirectory
  ``berkeley-function-call-leaderboard``, commit ``6ea5797`` (2026-03-23,
  "[BFCL] Request to add MiniCPM-SALA ...", the head of ``main`` on the pin date);
- PyPI ``bfcl-eval==2026.3.23`` (uploaded 2026-03-23, the latest release on the pin
  date). **Verified**: the wheel's ``bfcl_eval/`` tree is byte-identical to the
  commit's ``berkeley-function-call-leaderboard/bfcl_eval/`` (``diff -r``, 2026-10-18);
- licence: Apache License 2.0 (repository ``LICENSE``; wheel metadata
  ``License: Apache 2.0``). The data ships inside the package (``bfcl_eval/data``).

``vendor/`` is git-ignored. Our side and the partner fetch the checkout with
``scripts/xp/fetch_bfcl.sh`` (sparse, the subdirectory only) or install the pinned wheel;
``problems()`` checks a checkout (commit, subtree, file hashes) and ``installed_problems()``
checks an installed ``bfcl_eval`` package (file hashes only; no import, no network).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from slh.xp.partner.common import REPO

UPSTREAM = "https://github.com/ShishirPatil/gorilla"
SUBDIR = "berkeley-function-call-leaderboard"
COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
SUBTREE = "112d6789512dabeb9969b4fb4c97348d1047198a"  # git rev-parse COMMIT:SUBDIR
LICENSE = "Apache-2.0"
PYPI = {"name": "bfcl-eval", "version": "2026.3.23",
        "wheel": "bfcl_eval-2026.3.23-py3-none-any.whl",
        "wheel_sha256": "3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a",
        "sdist_sha256": "4a3869673721fa59be93d8f55ca92d69ab5797058aed792149c6adafda064bc1"}
PATH = REPO / "vendor/gorilla"
PACKAGE = PATH / SUBDIR / "bfcl_eval"

# SHA-256 of the files the track depends on, relative to the ``bfcl_eval`` package.
FILES = {
    "data/BFCL_v4_multi_turn_base.json":
        "1a21a995d06fd6f20ba55de7bced30ef953ec35e998f502ec2ecf4d66ef1c43a",
    "data/BFCL_v4_multi_turn_long_context.json":
        "78c3268c5cc8e97c0f4ec6c811b3b9a2bba14323b1830b7a874b06d822749324",
    "data/BFCL_v4_multi_turn_miss_func.json":
        "87d28ce10e37d864b72de85d5732eef2a867b241d6c1c99b4ae682c9e3ea921c",
    "data/BFCL_v4_multi_turn_miss_param.json":
        "f0c66dda3795f5f53e3e1c0cc8ba0246b6761c8f58bdba8317203bf451ab8838",
    "data/possible_answer/BFCL_v4_multi_turn_base.json":
        "1fee67823b317571649177dd89d63969feaae4e810cc7448ee55ba797fb7c8fc",
    "data/possible_answer/BFCL_v4_multi_turn_long_context.json":
        "e82aa0e839c39d23c64a05834f7ae024d7c9738ec21738ecee78dc876e3d0d18",
    "data/possible_answer/BFCL_v4_multi_turn_miss_func.json":
        "69e679b806d1c871b05393a4b95583bb973248e5b8d96c2d7f4ca05e29fc32e6",
    "data/possible_answer/BFCL_v4_multi_turn_miss_param.json":
        "59c442901779e2c31c33abcd566d032e03736e5ad8069de2fe05489873046ecf",
    "data/multi_turn_func_doc/gorilla_file_system.json":
        "c4c1b741c71e2a17c97a5dc9c4a91d89978c4eaece56494d14272d5df6c650e9",
    "data/multi_turn_func_doc/math_api.json":
        "83fa31708c89442bdcf12ac4dfbe3be8663ec9183a1b1524a9fda98164bc7e0b",
    "data/multi_turn_func_doc/message_api.json":
        "4d58ea933a5d2b280d7a52366617fb47fecd333d7b0e08e724db6fa12fb5f847",
    "data/multi_turn_func_doc/posting_api.json":
        "87f9fc404a06e4107d7c366f1399c596440e84c16cb119faf374b2f532d86e8b",
    "data/multi_turn_func_doc/ticket_api.json":
        "31324e0380782664fe82ba05bd23517808ad55692b45a5fe7c3d4d395fe4f0f8",
    "data/multi_turn_func_doc/trading_bot.json":
        "1a7933fd8f0cb8fbec38aeae05cb8c24132747cad4904fe1c13ac3d01fc22c2d",
    "data/multi_turn_func_doc/travel_booking.json":
        "f17b950c13adddf41d0848077df58788252e4c2e7cad5cfa71c8c4bf04f57b26",
    "data/multi_turn_func_doc/vehicle_control.json":
        "0c8a66292844874ef7b168f343bc394d8615d2d9e1f4387999a9ee23011eac78",
    "eval_checker/multi_turn_eval/multi_turn_checker.py":
        "b3b787515eda3bf0e365e39634a8fc7f831763f010f716d35d86f96fd5aab1b5",
    "eval_checker/multi_turn_eval/multi_turn_utils.py":
        "9fdbd779ec715959c310445b54b431e91abaa8b45a302abb6fd8db8d61b62ba2",
    "eval_checker/multi_turn_eval/func_source_code/long_context.py":
        "5c9a5710115609c779e5b2ccb43f6ad58271ed96f2ea7a797edb0d65c385aa38",
    "eval_checker/multi_turn_eval/func_source_code/gorilla_file_system.py":
        "84173750930b09aaa31b90547c9bd79ce75714682608b35e78a9c1ea259faf2b",
    "eval_checker/multi_turn_eval/func_source_code/math_api.py":
        "abde32cc25e71b2ff3acf4d028a3c005b98409c174aba4abddbfe1c018afb9a8",
    "eval_checker/multi_turn_eval/func_source_code/message_api.py":
        "cf72af1635654147af17df1891a7fb354549e78a61930139f741a31f4be21121",
    "eval_checker/multi_turn_eval/func_source_code/posting_api.py":
        "32b887aaa37c7167ad9c84ddfa7bbb855c75d3c1edab19ba868aba29acc4cfa1",
    "eval_checker/multi_turn_eval/func_source_code/ticket_api.py":
        "6a9375aa6425410c07a151aef27f3e44e9ed1023bb5cc5c25f14f4a9b3c99116",
    "eval_checker/multi_turn_eval/func_source_code/trading_bot.py":
        "cc2e3949035b561e9d07418b081561aa07b211672200e2e511aec074bf79fcb2",
    "eval_checker/multi_turn_eval/func_source_code/travel_booking.py":
        "e698e22adccf0d6d41b297f2d5e44faebd828cef4752df6a4ffb1570c0c35296",
    "eval_checker/multi_turn_eval/func_source_code/vehicle_control.py":
        "f5f7a2f28968728927a98ae12ec68a32d70781cfb74c2d7b14801772362b12d1",
    "eval_checker/eval_runner.py":
        "b1033684908819ccb312d4d0e2c563359d69247412f00206013b2292b0e3ce81",
    "model_handler/base_handler.py":
        "5ac4bdd139a7fe15cb9ee6cd75735ec27f9b25478ff3d82d2ea5208d832969e4",
    "model_handler/api_inference/openai_completion.py":
        "f0bc8734a3173df79e52fa6d757ac6b2585f7582e2f19ac7cddf5170269dd3f7",
    "_llm_response_generation.py":
        "a733ac4ad9111cab5f91dcb3cc2fab70a13ef1261d9b6211924c766ca8e81729",
    "constants/default_prompts.py":
        "4cc033be99ae2b10bbde30e7751ddcac5468cb45841b6dc7b3a97b92adf38205",
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected() -> dict[str, Any]:
    return {"upstream": UPSTREAM, "subdir": SUBDIR, "commit": COMMIT, "subtree": SUBTREE,
            "license": LICENSE, "pypi": dict(PYPI), "files": dict(FILES)}


def file_problems(package: Path) -> list[str]:
    out = []
    for rel, want in FILES.items():
        p = package / rel
        if not p.is_file():
            out.append(f"{rel}: missing")
        elif sha_file(p) != want:
            out.append(f"{rel}: hash differs")
    return out


def state(path: Path = PATH) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                                  text=True, timeout=20, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", SUBDIR)
    return {"commit": git("rev-parse", "HEAD"), "subtree": git("rev-parse", f"HEAD:{SUBDIR}"),
            "dirty": None if dirty is None else bool(dirty)}


def problems(path: Path = PATH) -> list[str]:
    """A checkout at ``path`` (repository root of the sparse clone)."""
    got = state(path)
    out = [f"{k}: {got[k]} != {v}" for k, v in (("commit", COMMIT), ("subtree", SUBTREE))
           if got[k] != v]
    if got["dirty"]:
        out.append("vendor checkout has local modifications")
    return out + file_problems(path / SUBDIR / "bfcl_eval")


def installed_package() -> Path | None:
    """The installed ``bfcl_eval`` directory, located without importing it."""
    import importlib.util

    spec = importlib.util.find_spec("bfcl_eval")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def installed_problems() -> list[str]:
    pkg = installed_package()
    if pkg is None:
        return ["bfcl_eval is not installed"]
    return file_problems(pkg)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m slh.xp.bfcl.vendor")
    ap.add_argument("--installed", action="store_true",
                    help="check the installed bfcl_eval package instead of vendor/gorilla")
    args = ap.parse_args(argv)
    probs = installed_problems() if args.installed else problems()
    print(json.dumps({"expected": {k: v for k, v in expected().items() if k != "files"},
                      "problems": probs}, indent=1))
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
