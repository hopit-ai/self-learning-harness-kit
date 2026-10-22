"""Pinned simulator openings: one opening per (scenario, seed), replayed to every arm.

Why (docs/xp/g1-audit-2026-10-19.md, docs/xp/pairing-fix.md). Arms of a block share
the simulator seed, but the served simulator is not deterministic: in G1 only 96 of
374 non-trap pairs got the same opening, which added noise and an estimated -1.3 of
the -2.14 point difference. With the pin on, the FIRST simulator reply of a
conversation (its opening, generated before any agent-specific turn) is generated
once, stored content-addressed in the run directory, and replayed byte-identically
to every other arm of the pair (and to every rerun or resumed attempt). Every later
simulator turn stays live: it answers the agent's own replies, which differ between
arms, so nothing after the opening can be cached and replayed.

Key (content address). sha256 over the canonical JSON of the simulator request as
the meter sizes it (system prompt with the scenario, the fixed agent greeting, the
user tools), the per-conversation seed and the simulator model config. Arms of a
pair send the same request, so they share one key; anything that changes the
request (another scenario, seed, simulator config, or an agent greeting that
differs) gets another key and is never replayed across. A key is a hash of the
request, never its text; the stored entry holds the opening message only (the same
text every conversation record carries) plus the job that first generated it.

Store. ``<run>/openings/<key>.json``, written once (atomic link; the first writer
wins, a later writer of the same key reads and uses the stored entry). Each entry
carries ``message_sha256`` (the digest of the stored opening), verified on every load:
a changed or corrupted entry is refused, never replayed. Every use of a key (generated
or replayed) is recorded twice: a marker ``<run>/openings/used/<key>`` (shared by all
workers) and a row in ``<worker dir>/openings.jsonl`` (job, key, source). A key that was
ever used is NEVER regenerated: if its entry is missing the conversation stops
(``OpeningLost``) instead of giving another arm of the pair a different opening, and the
coordinator refuses to start or resume (``verify_store``) while any used key lacks a
valid entry. Both are exported (``openings.jsonl.gz``) and ``match_rate`` measures, from
the exported conversations alone, how many pairs actually received the same opening;
the step-4 analysis is inconclusive below its registered agreement
(``slh.xp.consent.analysis.MIN_OPENING_AGREEMENT``).

Composition (as ``slh.guards.identity``): ``install()`` rebinds
``slh.agent.metered_complete`` in this process; the wrapper acts only on the
simulator's first turn while ``active(...)`` and passes everything else through.
Nothing in the pinned execution surface is edited; a run with the pin records
``descriptor()`` in its identity (absent when off).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

NAME = "simulator-opening-pin"
VERSION = 2  # 2: response digests, use markers, no regeneration of a used key
SCHEMA = "xp-phase2-opening-v1"
DIRNAME = "openings"
USES = "openings.jsonl"
USED_DIR = "used"
EXPORT_FILE = "openings.jsonl.gz"

_INSTALLED: dict[str, Any] = {}
_ACTIVE: dict[str, Any] = {"cache": None, "job": None, "log": None}


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def descriptor() -> dict[str, Any]:
    """What a pinned run records in its identity (absent when the pin is off)."""
    return {"name": NAME, "version": VERSION, "source_sha256": source_sha256(),
            "scope": "first simulator turn only; later turns live",
            "key": "sha256(simulator request, seed, simulator config)",
            "store": f"{DIRNAME}/<key>.json", "digest": "message_sha256, verified on load",
            "used": f"{DIRNAME}/{USED_DIR}/<key>; a used key is never regenerated"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      default=str)


class OpeningLost(RuntimeError):
    """A key some arm already used has no valid stored entry (the pair would get a
    different opening): never regenerated."""


def message_sha256(message: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(message).encode()).hexdigest()


def opening_key(config: Any, seed: int | None, payload: Any) -> str:
    cfg = config.to_dict() if hasattr(config, "to_dict") else config
    return hashlib.sha256(_canonical({"v": VERSION, "config": cfg, "seed": seed,
                                      "request": payload}).encode()).hexdigest()


def _message_dict(reply: Any) -> dict[str, Any]:
    calls = []
    for c in getattr(reply, "tool_calls", None) or []:
        calls.append({"id": c.id, "name": c.name, "arguments": c.arguments})
    return {"content": getattr(reply, "content", None), "tool_calls": calls or None}


def _message(stored: dict[str, Any]) -> Any:
    from tau2.data_model.message import AssistantMessage, ToolCall

    calls = stored.get("tool_calls")
    return AssistantMessage(
        role="assistant", content=stored.get("content"), cost=0.0,
        tool_calls=[ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"])
                    for c in calls] if calls else None)


class OpeningCache:
    """Content-addressed store of simulator openings in one run directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self.path(key)
        if not p.is_file():
            return None
        entry = json.loads(p.read_text(encoding="utf-8"))
        if entry.get("key") != key or entry.get("schema") != SCHEMA:
            raise ValueError(f"opening cache entry {p.name} is not the entry of its key")
        if entry.get("message_sha256") != message_sha256(entry.get("message") or {}):
            raise OpeningLost(f"opening cache entry {p.name}: stored opening does not "
                              "match its digest")
        return entry

    def used_path(self, key: str) -> Path:
        return self.root / USED_DIR / key

    def record_use(self, key: str) -> None:
        """Mark ``key`` used (idempotent, shared by every worker)."""
        d = self.root / USED_DIR
        d.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileExistsError):
            fd = os.open(self.used_path(key), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)

    def was_used(self, key: str) -> bool:
        return self.used_path(key).is_file()

    def used_keys(self) -> list[str]:
        d = self.root / USED_DIR
        return sorted(p.name for p in d.iterdir()) if d.is_dir() else []

    def put(self, entry: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Write once: (stored entry, True if this call wrote it). If the key is
        already stored, the stored entry is returned and nothing is written."""
        self.root.mkdir(parents=True, exist_ok=True)
        final = self.path(entry["key"])
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True, indent=1) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(tmp, final)  # fails if present: the first writer wins
            except FileExistsError:
                return self.get(entry["key"]) or entry, False
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
        return entry, True

    def entries(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        return [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(self.root.glob("*.json"))]


def verify_store(run: Path) -> list[str]:
    """Problems of a run's opening store (empty = fine): every entry matches its key
    and digest, and every key that any arm used (markers and the workers' use logs) has
    its entry. The coordinator refuses to start or resume otherwise."""
    run = Path(run)
    cache = OpeningCache(run / DIRNAME)
    problems = []
    used = set(cache.used_keys())
    workers = run / "workers"
    if workers.is_dir():
        for wdir in sorted(p for p in workers.iterdir() if p.is_dir()):
            path = wdir / USES
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    with contextlib.suppress(json.JSONDecodeError):
                        if line.strip() and json.loads(line).get("key"):
                            used.add(json.loads(line)["key"])
    for p in sorted(cache.root.glob("*.json")) if cache.root.is_dir() else []:
        try:
            cache.get(p.stem)
        except (ValueError, OpeningLost) as exc:
            problems.append(str(exc))
    for key in sorted(used):
        if not cache.path(key).is_file():
            problems.append(f"opening {key[:16]} was used by an arm but its entry is "
                            "missing (it is never regenerated)")
    return problems


def _is_opening(messages: list[Any]) -> bool:
    """The simulator sees roles flipped: its own past turns are assistant messages."""
    from tau2.data_model.message import AssistantMessage

    return not any(isinstance(m, AssistantMessage) for m in messages)


def _wrap(original: Any) -> Any:
    def metered_complete(meter: Any, role: str, backend: Any, messages: list[Any],
                         tools: Any, seed: int | None) -> Any:
        cache = _ACTIVE["cache"]
        if cache is None or role != "user" or not _is_opening(messages):
            return original(meter, role, backend, messages, tools, seed)
        from slh.agent import request_payload

        key = opening_key(backend.config, seed, request_payload(messages, tools))
        job = dict(_ACTIVE["job"] or {})
        entry = cache.get(key)
        source = "replayed"
        if entry is None:
            if cache.was_used(key):
                raise OpeningLost(f"opening {key[:16]} of {job.get('scenario_id')} was "
                                  "used by another arm and its entry is gone; it is never "
                                  "regenerated")
            reply = original(meter, role, backend, messages, tools, seed)
            message = _message_dict(reply)
            entry, created = cache.put({
                "schema": SCHEMA, "key": key, "seed": seed,
                "scenario_id": job.get("scenario_id"), "run": job.get("run"),
                "generated_by": job.get("key"), "message": message,
                "message_sha256": message_sha256(message)})
            if created:
                source, result = "generated", reply
            else:  # another writer stored this key first: use the stored opening
                result = _message(entry["message"])
        else:
            result = _message(entry["message"])
        cache.record_use(key)
        log = _ACTIVE["log"]
        if log is not None:
            log({"job": job.get("key"), "scenario_id": job.get("scenario_id"),
                 "run": job.get("run"), "arm": job.get("arm"), "key": key,
                 "source": source, "message_sha256": entry["message_sha256"]})
        return result

    metered_complete.__wrapped__ = original  # type: ignore[attr-defined]
    return metered_complete


def install() -> None:
    """Rebind ``slh.agent.metered_complete`` once per process (idempotent);
    installed but inactive, the wrapper is the original call."""
    if _INSTALLED:
        return
    from slh import agent

    _INSTALLED["metered_complete"] = agent.metered_complete
    agent.metered_complete = _wrap(agent.metered_complete)


def uninstall() -> None:
    """Tests: restore the original binding and switch off."""
    if not _INSTALLED:
        return
    from slh import agent

    agent.metered_complete = _INSTALLED.pop("metered_complete")
    _ACTIVE.update(cache=None, job=None, log=None)


@contextlib.contextmanager
def active(cache: OpeningCache | None, job: dict[str, Any] | None = None,
           log: Any = None) -> Iterator[None]:
    """Pin openings for one conversation (``cache`` None: off)."""
    if cache is not None:
        install()
    previous = dict(_ACTIVE)
    _ACTIVE.update(cache=cache, job=job, log=log)
    try:
        yield
    finally:
        _ACTIVE.update(previous)


# --------------------------------------------------------------------------- analysis


def opening_of(messages: list[dict[str, Any]]) -> str | None:
    """Canonical bytes (as text) of a conversation's first user message."""
    for m in messages or []:
        if m.get("role") == "user":
            return _canonical({"content": m.get("content"),
                               "tool_calls": [{k: c.get(k) for k in ("name", "arguments")}
                                              for c in (m.get("tool_calls") or [])]})
    return None


def match_rate(rows: list[dict[str, Any]], blobs: dict[str, Any] | None = None
               ) -> dict[str, Any]:
    """Pairs = (scenario, run) groups with at least two arms; a pair matches when
    every arm's first user message is byte-identical (content and tool calls)."""
    groups: dict[tuple[str, int], list[str | None]] = {}
    for r in rows:
        msgs = r.get("messages") or []
        if isinstance(msgs, str):
            msgs = (blobs or {}).get(msgs) or []
        groups.setdefault((r["scenario_id"], int(r["run"])), []).append(opening_of(msgs))
    pairs = {k: v for k, v in groups.items() if len(v) >= 2}
    matched = sorted(k for k, v in pairs.items() if None not in v and len(set(v)) == 1)
    unmatched = sorted(k for k in pairs if k not in set(matched))
    return {"pairs": len(pairs), "matched": len(matched),
            "rate": round(len(matched) / len(pairs), 4) if pairs else None,
            "unmatched": [f"{s}|{r}" for s, r in unmatched][:50]}


def export_rows(run: Path) -> list[dict[str, Any]]:
    """Cache entries and per-worker uses of a run directory, sorted (deterministic)."""
    run = Path(run)
    rows = [{"type": "entry", **e} for e in OpeningCache(run / DIRNAME).entries()]
    uses = []
    workers = run / "workers"
    if workers.is_dir():
        for wdir in sorted(p for p in workers.iterdir() if p.is_dir()):
            path = wdir / USES
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            uses.append({"type": "use", "worker": wdir.name,
                                         **json.loads(line)})
                        except json.JSONDecodeError:
                            continue  # torn tail of an interrupted worker
    return rows + sorted(uses, key=lambda u: (u["worker"], str(u.get("job")),
                                              str(u.get("key"))))
