"""Free checks before each demo-1 GPU window (``scripts/w4d.sh learn|test``).

Every check raises AssertionError on failure; nothing calls a model, the GPU
provider or the network. The sealed tree is checked by mode and by the
manifest hashes, never opened (``validate_all`` is the one registered
exception: it opens it for answer validation only).

The registration binds the executable demo code (every ``slh/demo1/*.py``,
``scripts/w4d.sh``, ``demo1/estimate.json``): these are outside the frozen
execution surface, so their hashes are registered and compared here.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from slh.demo1 import cases as case_files
from slh.demo1.context import SURFACE_SHA256, SURFACE_SHA256_BEFORE_MIGRATION
from slh.demo1.estimate import WINDOWS_MIN
from slh.demo1.schema import REPO, TASKS

COMBINED_CAP_USD = 14.21  # raised from 9.50 (D5), then 12.80 -> 14.21 (owner, 2026-10-09)
DEMO_BASELINE_USD = 20.846515  # ledger commitment before the first demo start
SURFACE_MIGRATION_DOC = "docs/surface-migration-2026-10-09-gpu.md"  # was ...-10-08.md


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound_files(root: Path = REPO) -> list[str]:
    return sorted([p.relative_to(root).as_posix() for p in (root / "slh/demo1").glob("*.py")]
                  + ["scripts/w4d.sh", "demo1/estimate.json", "demo1/b0-context.json"])


def code_hashes(root: Path = REPO) -> dict[str, str]:
    return {rel: _sha_file(root / rel) for rel in bound_files(root)}


REGISTRY_DIR = REPO / "learner_runs"
# Demo-1 rows use their own event names. A ``protocol_amendment`` row would
# become the learner loop's latest gate amendment (``Registry.amendment``),
# which every later learner round must then match; amendment 8 does not
# change the learner protocol, so it must not take that place.
REGISTRATION_EVENT = "demo1_amendment"
ARTIFACTS_EVENT = "demo1_learn_artifacts"


def registry_rows(directory: Path) -> list[dict[str, Any]]:
    """All rows through the registry's own reader (hash chain, both heads and
    the sealed amendment offsets verified; refuses a damaged registry)."""
    from slh.learner.registry import Registry

    if not (Path(directory) / "REGISTRY.jsonl").is_file():
        return []
    return Registry(Path(directory)).rows()


def prefix_sha256(directory: Path) -> str:
    return _sha_file(Path(directory) / "REGISTRY.jsonl")


def _line_sha256s(raw: bytes) -> list[str]:
    """SHA-256 of every registry line (with its newline), in row order: the
    row hash ``append_row`` reports and ``supersedes`` refers to."""
    return [hashlib.sha256(line).hexdigest() for line in raw.splitlines(keepends=True)]


def registration_chain(rows: list[dict[str, Any]], shas: list[str],
                       amendment_id: str) -> list[tuple[dict[str, Any], str]]:
    """The amendment's registration rows with their row hashes, oldest first.
    The first has no ``supersedes``; each later row must name the row hash of
    the one before it and give a reason. Anything else refuses."""
    chain = [(row, sha) for row, sha in zip(rows, shas, strict=True)
             if row.get("event") == REGISTRATION_EVENT
             and row.get("amendment_id") == amendment_id]
    for index, (row, _) in enumerate(chain):
        if index == 0:
            if "supersedes" in row:
                raise ValueError("the first registration row supersedes nothing")
            continue
        if row.get("supersedes") != chain[index - 1][1]:
            raise ValueError("a superseding registration row does not name the row "
                             "hash of the registration before it")
        if not str(row.get("reason") or "").strip():
            raise ValueError("a superseding registration row has no reason")
    return chain


def latest_registration(directory: Path) -> tuple[dict[str, Any], str] | None:
    """The amendment's current registration (the last row of a verified
    supersession chain) and its row hash; None while unregistered."""
    from slh.demo1 import pipeline

    rows = registry_rows(directory)
    if not rows:
        return None
    raw = (Path(directory) / "REGISTRY.jsonl").read_bytes()
    chain = registration_chain(rows, _line_sha256s(raw), pipeline.AMENDMENT_ID)
    return chain[-1] if chain else None


def append_row(directory: Path, event: dict[str, Any], *, expect_prefix_sha256: str
               ) -> dict[str, Any]:
    """Append one demo-1 row through the registry's own append path (hash chain,
    ``REGISTRY.head.json`` and the sealed head updated as for every row), under
    the registry lock, after verifying the whole file. Refuses unless the
    registry's current bytes hash to ``expect_prefix_sha256``. The row records
    that prefix. Returns the row and the SHA-256 of its appended line.

    A registration row for an amendment that is already registered is
    accepted only as a superseding row: ``supersedes`` must be the row hash
    of the current (latest) registration and ``reason`` must be given."""
    import fcntl

    from slh.learner.registry import Registry

    if not isinstance(expect_prefix_sha256, str) or len(expect_prefix_sha256) != 64:
        raise ValueError("the expected registry prefix hash is mandatory")
    registry = Registry(Path(directory))
    with registry.path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        rows = registry._rows(handle)  # verifies the chain, heads and seals
        prefix = registry._body(handle)
        actual = hashlib.sha256(prefix).hexdigest()
        if actual != expect_prefix_sha256:
            raise ValueError(f"registry prefix is {actual}, not the expected "
                             f"{expect_prefix_sha256}")
        if event["event"] == REGISTRATION_EVENT:
            chain = registration_chain(rows, _line_sha256s(prefix),
                                       event.get("amendment_id"))
            if not chain and "supersedes" in event:
                raise ValueError("nothing is registered for "
                                 f"{event.get('amendment_id')}; there is nothing to supersede")
            if chain and event.get("supersedes") != chain[-1][1]:
                raise ValueError(f"{event.get('amendment_id')} is already registered; a "
                                 "superseding row must name the latest registration row "
                                 f"hash {chain[-1][1]} (--supersedes)")
            if chain and not str(event.get("reason") or "").strip():
                raise ValueError("a superseding registration row needs a reason (--reason)")
        elif any(r.get("event") == event["event"] and
                 r.get("amendment_id") == event.get("amendment_id") for r in rows):
            raise ValueError(f"a {event['event']} row for {event.get('amendment_id')} "
                             "is already registered")
        if event["event"] == ARTIFACTS_EVENT and not any(
                r.get("event") == REGISTRATION_EVENT and
                r.get("amendment_id") == event.get("amendment_id") for r in rows):
            raise ValueError("register amendment 8 before the window-A artifacts")
        row = {**event, "prefix_sha256": actual, "prefix_bytes": len(prefix),
               "prefix_records": prefix.count(b"\n")}
        registry._append(handle, row)
        body = registry._body(handle)
    line = body[len(prefix):]
    return {"row": row, "row_sha256": hashlib.sha256(line).hexdigest()}


def registered(directory: Path, event: str) -> list[dict[str, Any]]:
    from slh.demo1 import pipeline

    return [r for r in registry_rows(directory) if r.get("event") == event
            and r.get("amendment_id") == pipeline.AMENDMENT_ID]


def registration_row(root: Path = REPO, *, supersedes: str | None = None,
                     reason: str | None = None) -> dict[str, Any]:
    """The row that registers amendment 8 (``python -m slh.demo1 registration
    [--append --expect-prefix-sha256 SHA [--supersedes ROW_SHA --reason TEXT]]``;
    the owner appends it). A superseding row (for example after a surface
    migration) names the row hash of the registration it replaces."""
    from slh.demo1 import learn, pipeline

    extra = {} if supersedes is None else {"supersedes": supersedes, "reason": reason or ""}
    return {**extra, "event": REGISTRATION_EVENT, "amendment_id": pipeline.AMENDMENT_ID,
            "surface_sha256": SURFACE_SHA256,
            "surface_migration": {"doc_path": SURFACE_MIGRATION_DOC,
                                  "doc_sha256": _sha_file(root / SURFACE_MIGRATION_DOC),
                                  "from": SURFACE_SHA256_BEFORE_MIGRATION,
                                  "to": SURFACE_SHA256},
            "doc_path": pipeline.AMENDMENT_DOC,
            "doc_sha256": _sha_file(root / pipeline.AMENDMENT_DOC),
            "demo1_manifest_sha256": _sha_file(root / "demo1/manifest.json"),
            "human_procedure_sha256": {t: _sha_file(learn.human_path(t, root / "demo1"))
                                       for t in TASKS},
            "code_sha256": code_hashes(root),
            "windows_min": dict(WINDOWS_MIN), "combined_cap_usd": COMBINED_CAP_USD,
            "demo_baseline_usd": DEMO_BASELINE_USD,
            "request_limits": request_limits(),
            "effective_from_round": None}


def request_limits() -> dict[str, Any]:
    from slh.demo1 import decide

    return {"max_output_tokens": decide.MAX_OUTPUT_TOKENS, "case_limit_s": decide.CASE_LIMIT_S,
            "timeout_retries": decide.TIMEOUT_RETRIES}


def _common(root: Path, registry: Path) -> None:
    from slh.demo1 import generate, learn
    from slh.demo1.context import b1_context
    from slh.execution import current_surface

    assert current_surface(root)["sha256"] == SURFACE_SHA256, "execution surface drifted"
    ctx = b1_context()
    recorded = json.loads((root / "demo1/b1-context.json").read_text(encoding="utf-8"))
    assert recorded == ctx, "demo1/b1-context.json differs from the rendered B1 context"
    from slh.demo1.context import b0_context

    b0 = json.loads((root / "demo1/b0-context.json").read_text(encoding="utf-8"))
    assert b0 == b0_context(), "demo1/b0-context.json differs from the rendered B0 context"
    checked = generate.verify()
    assert checked["ok"], f"case files: {checked['problems']}"
    cases_dir = case_files.SEALED_DIR / "cases"
    assert stat.S_IMODE(os.stat(cases_dir).st_mode) == 0, "the sealed case tree is not mode 000"
    latest = latest_registration(registry)  # the last row of the supersession chain
    assert latest is not None, "amendment 8 is not registered (draft only)"
    expected = registration_row(root)
    for key in ("surface_sha256", "surface_migration", "doc_sha256", "demo1_manifest_sha256",
                "human_procedure_sha256", "code_sha256", "windows_min", "combined_cap_usd",
                "demo_baseline_usd", "request_limits"):
        assert latest[0].get(key) == expected[key], f"registered {key} differs"
    for task in TASKS:
        learn.human_procedure(task, root / "demo1")
    assert learn.count_tokens("WHEN x THEN y") > 0, "pinned tokenizer"
    from slh.demo1 import code_rule

    assert callable(getattr(code_rule, "decide", None)), "slh.demo1.code_rule.decide missing"
    est = json.loads((root / "demo1/estimate.json").read_text(encoding="utf-8"))
    for name, minutes in WINDOWS_MIN.items():
        w = est["windows"][name]["window_min"]
        assert w is not None and w <= minutes, f"the {name} estimate does not fit {minutes}m"
    assert est["within_cap"] and est["combined_gpu_reservation_usd"] <= COMBINED_CAP_USD, \
        "estimate over the combined cap"


def preflight(mode: str, *, root: Path = REPO, run: Path | None = None,
              registry: Path | None = None) -> dict[str, Any]:
    from slh.demo1 import pipeline

    assert mode in WINDOWS_MIN, "mode is learn or test"
    run = run or pipeline.RUN_DIR
    registry = registry or REGISTRY_DIR
    _common(root, registry)
    if mode == "learn":
        assert not run.exists(), f"{run} exists: window A runs once"
    else:
        assert (run / "learn-record.json").is_file(), "window A did not finish"
        assert (run / "window-learn.json").is_file(), "window A's spend baseline is missing"
        assert not (run / "artifacts" / "freeze.json").exists(), "window B already ran"
        assert not (run / "sealed-results.jsonl").exists(), "sealed results already exist"
        # Calibration runs from scratch: a stalled attempt's rows are moved aside
        # (calibrate-results.attempt1-stalled.jsonl; kept, disclosed, never used).
        assert not (run / "calibrate-results.jsonl").exists(), \
            "calibrate-results.jsonl exists: move a stalled attempt aside first"
        status = pipeline.owner_status(run)
        assert all(s in ("accept", "reject", "no_procedure") for s in status.values()), \
            f"owner review incomplete or a hash differs: {status}"
        row = pipeline.artifacts_row(run)
        found = registered(registry, ARTIFACTS_EVENT)
        assert len(found) == 1, "the window-A artifact hashes are not registered"
        for key in ("learn_record_sha256", "artifact_files_sha256", "artifact_texts_sha256",
                    "owner_verdicts"):
            assert found[0].get(key) == row[key], f"registered {key} differs"
    return {"mode": mode, "window_min": WINDOWS_MIN[mode],
            "combined_cap_usd": f"{COMBINED_CAP_USD:.2f}"}


def validate_all(out: Path = case_files.OUT) -> dict[str, Any]:
    """The independent validator on all 168 cases (public and sealed). Prints
    counts only; the sealed tree is opened through ``answer_validation``."""
    from slh.demo1 import validate_independent
    from slh.demo1.generate import load_db

    db = load_db()
    result: dict[str, Any] = {"cases": 0, "agree": 0, "by_split": {}, "failures": []}

    def check(task: str, split: str, cases: list[dict[str, Any]]) -> None:
        key = f"{task}/{split}"
        result["by_split"][key] = {"cases": len(cases), "agree": 0}
        for case in cases:
            problems = validate_independent.validate_case(case, db)
            result["cases"] += 1
            if problems:
                result["failures"].append({"case_id": case["case_id"], "split": split,
                                           "problems": problems})
            else:
                result["agree"] += 1
                result["by_split"][key]["agree"] += 1

    for task in TASKS:
        for split in ("train", "calibrate"):
            check(task, split, case_files.load(task, split, out=out))
    with case_files.sealed_access("answer_validation", out=out) as sealed:
        for task in TASKS:
            check(task, "sealed", sealed[task])
    return result
