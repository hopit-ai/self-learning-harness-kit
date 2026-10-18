"""Shared helpers for the partner hand-off: hashing, gz shards, question rows,
answer keys and the code hash (docs/xp/partner-package.md).

Stdlib only at import time; scorers are imported lazily.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
QUESTION_SCHEMA = "xp-question-v1"
QUESTION_FIELDS = ("task", "case_id", "context", "messages")  # what case_hash covers
XP_TASKS = ("variant", "sequence")
TASKS = ("address", *XP_TASKS)
# Keys that carry (or reveal) the answer. Removed recursively from every exported
# row; a row that still has one after scrubbing is refused.
GOLD_KEYS = frozenset({"gold", "resolution", "request", "answer_key"})
# Files whose hash binds the code the partner runs (registration.code_sha256).
CODE_GLOBS = ("slh/**/*.py", "scripts/**/*.sh", "scripts/**/*.py", "configs/**/*.yaml",
              "pyproject.toml")
# Files the partner edits (recorded in env.json, never part of code_sha256).
PARTNER_EDITABLE = frozenset({"configs/xp-partner-endpoint.yaml"})


class GoldLeak(ValueError):
    """A row bound for the partner or the export still carries an answer field."""


# --------------------------------------------------------------------------- hashing


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_text(text: str) -> str:
    return sha_bytes(text.encode("utf-8"))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def code_hashes(root: Path = REPO) -> dict[str, Any]:
    """SHA-256 per bound file and one ``code_sha256`` over them (sorted paths)."""
    root = Path(root)
    files: dict[str, str] = {}
    for pattern in CODE_GLOBS:
        for path in root.glob(pattern):
            rel = path.relative_to(root).as_posix()
            if path.is_file() and "__pycache__" not in path.parts and \
                    rel not in PARTNER_EDITABLE:
                files[rel] = sha_file(path)
    files = dict(sorted(files.items()))
    return {"files": files, "code_sha256": sha_text(canonical(files))}


def kit_paths(root: Path = REPO) -> frozenset[str] | None:
    """The paths the partner kit ships from ``root`` (``snapshot.allowlist``), or None
    inside a kit (the snapshot builder is never shipped: every file present ships).

    Hand-off package builders register only shipped code, so a package built from our
    repository and one built from the kit tree register the same files, and the
    partner's code check never asks for an our-side module the kit leaves out."""
    try:
        from slh.xp.partner import snapshot
    except ImportError:
        return None
    if not (Path(root) / "slh/xp/partner/snapshot.py").is_file():
        return None
    return frozenset(snapshot.allowlist(Path(root)))


# vLLM server flags that would put sampling defaults under the requests' own parameters:
# ``--generation-config`` other than ``vllm`` (vLLM's default ``auto`` applies the model
# repository's generation_config.json to every parameter a request leaves out) and any
# ``--override-generation-config``. Every hand-off sends its decoding per request (part A:
# the phase-1 decision config, temperature 0; parts B and C: temperature 0.7, top-p 0.8);
# the server must add nothing to it.
SAMPLING_FLAGS = ("--generation-config", "--override-generation-config")


def serving_sampling_problems(serving_args: Any) -> list[str]:
    """Problems with the recorded ``vllm serve`` argv (``endpoint.serving_args``): it
    must be a list naming ``--generation-config vllm`` and no
    ``--override-generation-config``, so the server never supplies sampling defaults."""
    if not isinstance(serving_args, list) or not all(isinstance(a, str) for a in serving_args):
        return ["endpoint serving_args must be the exact vllm serve argv (a list of strings)"]
    out, gen = [], []
    for k, arg in enumerate(serving_args):
        flag, _, inline = arg.partition("=")
        if flag not in SAMPLING_FLAGS:
            continue
        value = inline if inline else (serving_args[k + 1] if k + 1 < len(serving_args)
                                       else "")
        if flag == "--override-generation-config":
            out.append(f"serving_args sets --override-generation-config {value!r}: remove it "
                       "(sampling is sent per request)")
        else:
            gen.append(value)
    if gen != ["vllm"]:
        out.append("serving_args must contain --generation-config vllm exactly once (no "
                   f"sampling defaults from the model repository); found {gen or 'none'}")
    return out


def git_blob_id(path: Path) -> str:
    """The git blob id of a file's bytes (``git hash-object``), computed without git."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def source_mismatches(root: Path, paths: dict[str, str]) -> list[str]:
    """Registered code paths (path -> git blob id) whose file is missing or differs.

    The public kit is not our commit: its repository holds the snapshot's files
    next to the package and runbook, so the checkout is verified by content (every
    registered blob present and equal) when HEAD is not the snapshot commit."""
    root = Path(root)
    return sorted(p for p, blob in paths.items()
                  if not (root / p).is_file() or git_blob_id(root / p) != blob)


def git_state(root: Path = REPO) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                                  timeout=20, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": run("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status)}


# --------------------------------------------------------------------------- rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> tuple[bytes, int]:
    buf = io.StringIO()
    n = 0
    for row in rows:
        buf.write(json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":")))
        buf.write("\n")
        n += 1
    return buf.getvalue().encode("utf-8"), n


def gz_bytes(data: bytes) -> bytes:
    """Deterministic gzip (no file name, mtime 0)."""
    out = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=out, compresslevel=9, mtime=0) as gz:
        gz.write(data)
    return out.getvalue()


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, sync: bool = True) -> None:
    """Crash-safe append: one complete line per write, flushed (and fsynced once per
    call unless ``sync`` is False: files whose unit is redone whole after a crash)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        for row in rows:
            handle.write((json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n").encode())
        handle.flush()
        if sync:
            os.fsync(handle.fileno())


class GroupAppender:
    """Group commit for append-only JSONL on slow shared filesystems (Lustre fsync can
    cost over a second): lines are written and flushed at once, fsynced every
    ``every_s`` seconds or ``every_rows`` rows and on close. A crash can lose at most
    the rows since the last fsync (and leave a truncated last line, repaired on
    resume); resume recomputes what is pending, so those rows are simply redone.
    Defaults from XP_FSYNC_EVERY_S (2) and XP_FSYNC_ROWS (256)."""

    def __init__(self, path: Path, *, every_s: float | None = None,
                 every_rows: int | None = None, clock: Any = None) -> None:
        import time as _time

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.every_s = float(os.environ.get("XP_FSYNC_EVERY_S", "2") if every_s is None
                             else every_s)
        self.every_rows = int(os.environ.get("XP_FSYNC_ROWS", "256") if every_rows is None
                              else every_rows)
        self.clock = clock or _time.monotonic
        self.handle = self.path.open("ab")
        self.unsynced, self.last = 0, self.clock()
        self.syncs = 0

    def write(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.handle.write(
                (json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n").encode())
            self.unsynced += 1
        self.handle.flush()
        if self.unsynced and (self.unsynced >= self.every_rows
                              or self.clock() - self.last >= self.every_s):
            self.sync()

    def sync(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.unsynced, self.last = 0, self.clock()
        self.syncs += 1

    def close(self) -> None:
        if not self.handle.closed:
            self.sync()
            self.handle.close()

    def __enter__(self) -> GroupAppender:  # noqa: PYI034
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def repair_jsonl(path: Path) -> dict[str, Any] | None:
    """Drop a truncated LAST line left by a crash (no trailing newline, or not
    parseable); returns a repair record, or None if the file was intact. A bad
    line anywhere else is corruption and raises."""
    path = Path(path)
    if not path.is_file():
        return None
    data = path.read_bytes()
    if not data:
        return None
    lines = data.split(b"\n")
    tail = lines[-1]  # b"" when the file ends with a newline
    body = lines[:-1]
    bad_last = False
    if body:
        try:
            json.loads(body[-1])
        except ValueError:
            bad_last = True
    for i, line in enumerate(body[:-1] if body else []):
        try:
            json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}: corrupt line {i + 1} (not the last one)") from exc
    if not tail and not bad_last:
        return None
    keep = body[:-1] if bad_last else body
    kept = b"".join(line + b"\n" for line in keep)
    dropped = data[len(kept):]
    with path.open("r+b") as handle:
        handle.truncate(len(kept))
        handle.flush()
        os.fsync(handle.fileno())
    return {"event": "repair_truncated_line", "file": path.name, "dropped_bytes": len(dropped),
            "kept_lines": len(keep)}


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=1) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scrub(value: Any) -> Any:
    """A copy with every GOLD_KEYS key removed at any depth."""
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if k not in GOLD_KEYS}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def gold_paths(value: Any, path: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for k, v in value.items():
            if k in GOLD_KEYS:
                yield f"{path}.{k}"
            yield from gold_paths(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from gold_paths(v, f"{path}[{i}]")


def assert_no_gold(row: Any, where: str = "row") -> None:
    found = next(gold_paths(row), None)
    if found is not None:
        raise GoldLeak(f"{where}: answer field {found}")


# --------------------------------------------------------------------------- cases


def task_of(case: dict[str, Any]) -> str:
    """POC 9 address cases have no ``task`` field; XP cases do."""
    return case.get("task") or "address"


def case_hash(case: dict[str, Any]) -> str:
    """SHA-256 of the question payload only (task, case_id, context, messages).

    For XP cases this equals ``slh.xp.tasks.common.case_hash``; POC 9 address
    cases get ``task = "address"``. Never a hash over gold."""
    payload = {"task": task_of(case), **{k: case[k] for k in QUESTION_FIELDS if k != "task"}}
    return sha_text(canonical(payload))


def question_row(case: dict[str, Any]) -> dict[str, Any]:
    row = {"schema": QUESTION_SCHEMA, "task": task_of(case),
           **{k: case[k] for k in ("case_id", "context", "messages")},
           "case_hash": case_hash(case)}
    assert_no_gold(row, f"question {case['case_id']}")
    return row


def answer_key_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**c, "task": task_of(c), "case_hash": case_hash(c)} for c in cases]


def load_answer_key(path: Path) -> dict[str, dict[str, Any]]:
    """case_id -> full case (with gold); each row's case_hash is re-computed."""
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(Path(path)):
        recorded = row.get("case_hash")
        if recorded is None or case_hash(row) != recorded:
            raise ValueError(f"{row.get('case_id')}: answer-key row differs from its case_hash")
        if row["case_id"] in out:
            raise ValueError(f"{row['case_id']}: duplicate in the answer key")
        out[row["case_id"]] = row
    return out


def scorer(task: str, case: dict[str, Any] | None = None) -> tuple[Any, Any]:
    """(score, summarize): POC 9's for POC 9 address cases (``poc9-case-v1``),
    XP's for XP cases of any task."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    if case is not None and str(case.get("schema", "")).startswith("poc9-"):
        from slh.poc9.score import score, summarize
    else:
        from slh.xp.score import score, summarize
    return score, summarize


ADDRESS_ARGS = ("address1", "address2", "city", "state", "country", "zip")


def gold_atoms(case: dict[str, Any]) -> dict[str, Any] | None:
    """Distinctive pieces of a write/confirm gold: the exact call (name + args),
    its order id, the values that pin the answer (address fields, item ids) and
    the (address1, zip) pair of an address change. None for clarify golds."""
    gold = case.get("gold") or {}
    args = gold.get("arguments")
    if not isinstance(args, dict) or not args.get("order_id"):
        return None
    values: set[str] = set()
    for k, v in args.items():
        if k == "order_id":
            continue
        for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, str) and len(x.strip()) >= 4:
                values.add(x.strip())
    address = (args["address1"], str(args["zip"])) if args.get("address1") and \
        args.get("zip") else None
    return {"call": canonical({"name": gold.get("name"), "arguments": args}),
            "order_id": args["order_id"], "values": values, "address": address}


def entry(module: str, main: Any) -> int:
    """``python -m slh.xp.partner.<module>``: run through the trusted bootstrap
    (scripts/xp/xp_boot.py: no bytecode written, a fresh empty bytecode cache, no
    import artifacts, every registered code file checked) unless already inside it.
    The process re-executes itself; arguments are passed unchanged."""
    import sys
    import tempfile

    boot = REPO / "scripts" / "xp" / "xp_boot.py"
    if os.environ.get("XP_BOOT") == "1" or not boot.is_file():
        return int(main() or 0)
    env = {**os.environ, "XP_BOOT": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPYCACHEPREFIX": tempfile.mkdtemp(prefix="xp-pyc.")}
    extra = [] if module == "runner" else ["--module", module]
    os.chdir(REPO)
    os.execve(sys.executable, [sys.executable, str(boot), *extra, *sys.argv[1:]], env)
    return 4  # not reached

