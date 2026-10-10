"""Loading demo-1 cases. Train and calibration cases are public files; sealed
cases are read only inside ``sealed_access`` (docs/demo1/case-schema.md)."""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from slh.demo1.schema import REPO, SPLITS, TASKS, read_jsonl

OUT = REPO / "demo1"
SEALED_DIR = REPO / "sealed/demo1"
ACCESS_LOG = OUT / "sealed-access.jsonl"


class SealedRefused(PermissionError):
    """A sealed case file was requested outside ``sealed_access``."""


def manifest(out: Path = OUT) -> dict[str, Any]:
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def _check_hash(path: Path, rel: str, out: Path) -> None:
    expected = manifest(out)["files"][rel]["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"{rel} differs from its manifest hash")


def load(task: str, split: str, *, out: Path = OUT) -> list[dict[str, Any]]:
    """Train or calibration cases of one task (hash-checked against the manifest)."""
    if task not in TASKS or split not in SPLITS:
        raise ValueError(f"unknown task/split {task}/{split}")
    if split == "sealed":
        raise SealedRefused("sealed cases are read only inside slh.demo1.cases.sealed_access")
    rel = f"demo1/cases/{split}-{task}.jsonl"
    path = out / "cases" / f"{split}-{task}.jsonl"
    _check_hash(path, rel, out)
    return read_jsonl(path)


@contextlib.contextmanager
def sealed_access(purpose: str, *, sealed_dir: Path = SEALED_DIR, out: Path = OUT
                  ) -> Iterator[dict[str, list[dict[str, Any]]]]:
    """Open the sealed tree, yield {task: cases} (hash-checked), lock it again
    (mode 000) and log the access. The learning step never calls this."""
    from slh import sealing

    if purpose not in ("final_run", "answer_validation"):
        raise ValueError(f"unregistered sealed-access purpose {purpose!r}")
    cases_dir = sealed_dir / "cases"
    with sealing.writer():
        sealing.open_tree(cases_dir)
        try:
            loaded = {}
            for task in TASKS:
                path = cases_dir / f"sealed-{task}.jsonl"
                _check_hash(path, f"sealed/demo1/cases/sealed-{task}.jsonl", out)
                loaded[task] = read_jsonl(path)
            yield loaded
        finally:
            sealing.lock_tree(cases_dir)
            out.mkdir(parents=True, exist_ok=True)
            with (out / ACCESS_LOG.name).open("a", encoding="utf-8") as log:
                log.write(json.dumps({"purpose": purpose,
                                      "time_utc": dt.datetime.now(dt.UTC).isoformat()},
                                     sort_keys=True) + "\n")
