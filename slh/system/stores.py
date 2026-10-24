"""Durable stores: audit log (hash chain, adoption pointers inside it), trace store, unit
registry and the harness-state version store, under one single-writer lock.

Durability: every append is one ``write`` of a complete line, flushed and fsynced. At open,
a torn final line (a writer died mid-append) is copied to ``<name>.torn`` and the file is
truncated to its last complete line; a corrupt line anywhere else refuses to open. Rows are
cached in memory after the first read, so appends and queries are O(1)/O(rows) per call
instead of re-reading the file.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from slh.state import HarnessState
from slh.system.interfaces import Outcome, Unit
from slh.trace import check_no_grading_fields


class StoreError(RuntimeError):
    pass


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def now_utc() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds")


class SystemLock:
    """Exclusive single-writer lock on the system directory (held for the harness's life)."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "system.lock"
        self._fh = self.path.open("a")
        try:
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            self._fh.close()
            raise StoreError(f"another writer holds {self.path}") from e
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(f"{os.getpid()}\n")
        self._fh.flush()

    def release(self) -> None:
        if not self._fh.closed:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()


class JsonlLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.repaired: bytes | None = None
        self._rows = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        data = self.path.read_bytes()
        end = data.rfind(b"\n") + 1
        rows, good = [], 0
        lines = data[:end].split(b"\n")[:-1] if end else []
        for i, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
                good += len(line) + 1
            except json.JSONDecodeError as e:
                if i == len(lines) - 1 and end == len(data):
                    break  # a complete but corrupt final line: treat as torn
                raise StoreError(f"{self.path}: corrupt line {i}") from e
        if good < len(data):
            torn = data[good:]
            with (self.path.parent / (self.path.name + ".torn")).open("ab") as fh:
                fh.write(torn + b"\n")
                fh.flush()
                os.fsync(fh.fileno())
            with self.path.open("r+b") as fh:
                fh.truncate(good)
                fh.flush()
                os.fsync(fh.fileno())
            self.repaired = torn
        return rows

    def append(self, record: dict[str, Any]) -> None:
        self.append_many([record])

    def append_many(self, records: list[dict[str, Any]]) -> None:
        """One write and one fsync for a batch of complete lines."""
        if not records:
            return
        data = b"".join((_canon(r) + "\n").encode() for r in records)
        with self.path.open("ab") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        self._rows.extend(records)

    def rows(self) -> list[dict[str, Any]]:
        return self._rows


class AuditLog:
    """Hash-chained record of every decision with its evidence. Adoption pointers live in it
    (records with a ``pointer`` field), so the chain covers them."""

    def __init__(self, root: Path) -> None:
        self.log = JsonlLog(Path(root) / "audit.jsonl")
        self.path = self.log.path

    def append(self, kind: str, /, **payload: Any) -> dict[str, Any]:
        rows = self.log.rows()
        prev = rows[-1]["sha256"] if rows else "0" * 64
        body = {"seq": len(rows), "kind": kind, "ts": now_utc(), "prev_sha256": prev,
                **payload}
        body["sha256"] = hashlib.sha256(_canon(body).encode()).hexdigest()
        self.log.append(body)
        return body

    def rows(self, kind: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.log.rows() if kind is None or r["kind"] == kind]

    def last(self, *kinds: str) -> dict[str, Any] | None:
        for r in reversed(self.log.rows()):
            if r["kind"] in kinds:
                return r
        return None

    def verify(self) -> None:
        prev = "0" * 64
        rows =[json.loads(x) for x in self.path.read_bytes().splitlines() if x.strip()] \
            if self.path.exists() else []
        mem = self.log.rows()
        if len(rows) != len(mem):
            raise ValueError("audit log changed under the writer")
        for i, r in enumerate(rows):
            body = {k: v for k, v in r.items() if k != "sha256"}
            if r.get("seq") != i or r.get("prev_sha256") != prev:
                raise ValueError(f"audit chain broken at {i}")
            if hashlib.sha256(_canon(body).encode()).hexdigest() != r.get("sha256"):
                raise ValueError(f"audit record {i} altered")
            if _canon(r) != _canon(mem[i]):  # the copy every decision reads
                raise ValueError(f"audit record {i} differs from the writer's copy")
            prev = r["sha256"]


class TraceStore:
    """Every outcome, tagged with role and batch. ``evidence`` is the only view a proposer
    gets; screen, confirm and monitor rows are gate data and never evidence."""

    def __init__(self, root: Path) -> None:
        self.log = JsonlLog(Path(root) / "traces.jsonl")

    def add(self, batch: int, outcomes: Iterable[Outcome], *, stage: str,
            k: int | None = None) -> None:
        recs = []
        for o in outcomes:
            if o.trace is not None:
                check_no_grading_fields(o.trace)  # the runner's own leak check, reused
            recs.append({"batch": batch, "stage": stage, "k": k, **dataclasses.asdict(o)})
        self.log.append_many(recs)

    def rows(self, role: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.log.rows() if role is None or r["role"] == role]

    @staticmethod
    def outcome(r: dict[str, Any]) -> Outcome:
        return Outcome(**{k: v for k, v in r.items() if k not in ("batch", "stage", "k")})

    def evidence(self, batch: int) -> list[Outcome]:
        return [self.outcome(r) for r in self.rows("evidence") if r["batch"] == batch]


class UnitError(RuntimeError):
    pass


class UnitRegistry:
    """Persistent canonical allocation of units and clusters to roles.

    * A unit id or cluster belongs to exactly one role, for ever (across restarts).
    * evidence / screen / monitor units are single-use (fresh); their clusters may recur
      within the same role only.
    * confirm units come from the permanent confirm partition: they may recur ACROSS
      confirmations (each confirmation is a draw from that finite pool), never within one,
      and one confirmation holds at most one unit per cluster.
    """

    def __init__(self, root: Path) -> None:
        self.log = JsonlLog(Path(root) / "units.jsonl")
        self.unit_role: dict[str, str] = {}
        self.cluster_role: dict[str, str] = {}
        for r in self.log.rows():
            self.unit_role[r["unit_id"]] = r["role"]
            self.cluster_role[r["cluster"]] = r["role"]

    def admit(self, units: list[Unit], role: str, n: int, batch: int,
              k: int | None = None) -> None:
        if len(units) != n:
            raise UnitError(f"{role}: adapter returned {len(units)} units, {n} requested")
        ids = [u.unit_id for u in units]
        if len(set(ids)) != len(ids):
            raise UnitError(f"{role}: duplicate unit ids in one draw")
        if role == "confirm" and len({u.cluster for u in units}) != len(units):
            raise UnitError("confirm: more than one unit per cluster in one confirmation")
        for u in units:
            if u.role != role:
                raise UnitError(f"unit {u.unit_id} has role {u.role}, requested {role}")
            prev = self.unit_role.get(u.unit_id)
            if prev is not None and (role != "confirm" or prev != "confirm"):
                raise UnitError(f"unit {u.unit_id} already used ({prev}); {role} needs fresh")
            crole = self.cluster_role.get(u.cluster)
            if crole is not None and crole != role:
                raise UnitError(f"cluster {u.cluster} belongs to {crole}, not {role}")
        self.log.append_many([{"unit_id": u.unit_id, "cluster": u.cluster, "role": role,
                               "batch": batch, "k": k} for u in units])
        for u in units:
            self.unit_role[u.unit_id] = role
            self.cluster_role[u.cluster] = role


class VersionStore:
    """Content-addressed ``HarnessState`` versions (``slh.state``; never overwritten)."""

    def __init__(self, root: Path) -> None:
        self.dir = Path(root) / "versions"

    def save(self, state: HarnessState) -> str:
        state.save(self.dir)
        return state.version_id

    def load(self, version: str) -> HarnessState:
        return HarnessState.load(self.dir, version)
