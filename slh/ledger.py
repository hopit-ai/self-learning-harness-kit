"""Cross-process spend ledger: one budget cap shared by parallel workers.

Every paid attempt of every worker goes through one append-only file,
``runs/<id>/ledger.jsonl``, guarded by an exclusive ``fcntl`` lock on
``ledger.lock``:

- ``reserve``: under the lock, the ledger re-reads every record appended since
  this process last looked, computes the run's committed spend (everything
  charged plus every reservation not yet settled, by any worker) and appends
  the reservation, fsynced, only if committed + reservation stays within the
  cap. The append returns before the attempt is dispatched, so a worker killed
  mid-call leaves a durable unsettled reservation.
- ``settle``: after the attempt, the charged cost replaces the reservation.
- An unsettled reservation keeps counting as spent, at its full amount, for
  every other worker. ``reconcile`` (run by the coordinator once the owning
  worker is known to be dead, and by resume) settles it at its full amount as
  an ``unreconciled_reservation``; nothing is ever refunded.
- A reservation that fits against what is already *charged* but not against
  the other workers' in-flight reservations waits (polling, up to
  ``wait_s``) for them to settle, since settled costs are usually far below
  their reservations. Only a reservation that cannot fit even against the
  charged total (the cap is exhausted, exactly as in a sequential run), or one
  that waited ``wait_s`` in vain, is refused.
- ``halt``: the first refusal or overspend of any worker halts the session, so
  every other worker is refused at its next call and at its next dispatch
  (``assert_not_halted`` runs immediately before bytes are sent). A refusal is
  one record that is itself a halt, so a crash cannot persist one without the
  other.

  Accepted residual race: the dispatch check releases the ledger lock before
  the HTTP call (holding a file lock across a network call would serialize
  every worker and turn a slow provider into a run-wide stall). So a worker
  can pass the check, another worker can halt, and the first request is still
  sent. That request was already reserved, so the cap still holds: it is
  settled at its real cost (at most its reservation), and every later
  reservation and dispatch of the session is refused. The halting worker's
  reason (owner and reason) is in the halt or refuse record.
- ``session``: the coordinator opens every session with a record. Resume
  refuses a run whose ledger is missing or has no session record (a fresh
  ledger would silently reset spend), and never creates one.

A torn final line can only come from a writer that died while holding the
lock. A torn ``reserve`` was never returned to its caller, so nothing was
dispatched against it; a torn ``settle`` leaves its reservation unsettled,
which is charged in full. The torn bytes are copied to the quarantine file
before the ledger is truncated back to its last complete line.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import math
import os
import random
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from slh.budget import ROLES, BudgetExceeded

EPSILON = 1e-12
WAIT_S = 900.0  # longest wait for other workers' reservations to settle
POLL_S = 0.2
_sleep = time.sleep


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


class SharedLedger:
    def __init__(self, path: str | Path, cap_usd: float, owner: str, session: int,
                 quarantine: str | Path | None = None, wait_s: float = WAIT_S,
                 create: bool = True) -> None:
        if not math.isfinite(cap_usd) or cap_usd < 0:
            raise ValueError("budget cap must be a finite nonnegative number")
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.cap_usd = float(cap_usd)
        self.owner = owner
        self.session = session
        self.quarantine = None if quarantine is None else Path(quarantine)
        self._offset = 0
        self._counter = 0
        self.pending: dict[str, dict[str, Any]] = {}
        self.charged_by_role: dict[str, float] = dict.fromkeys(ROLES, 0.0)
        self.charged_usd = 0.0
        self.halted: dict[str, Any] | None = None
        self.unreconciled: list[dict[str, Any]] = []
        self.refusals = 0
        self.wait_s = wait_s
        self.waits = 0
        self.sessions = 0
        if create:
            self.path.touch(exist_ok=True)
        elif not self.path.is_file():
            raise RuntimeError(f"spend ledger {self.path} is missing; refusing to continue "
                               "without the run's accounting history")

    # ------------------------------------------------------------------ locking and reading

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        with self.lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                self._refresh()
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _refresh(self) -> None:
        """Apply records appended since the last read (lock held)."""
        with self.path.open("rb") as handle:
            handle.seek(self._offset)
            data = handle.read()
        if not data:
            return
        end = data.rfind(b"\n") + 1
        if end < len(data):
            self._repair_tail(self._offset + end, data[end:])
        for line in data[:end].split(b"\n"):
            if line.strip():
                self._apply(json.loads(line))
        self._offset += end

    def _repair_tail(self, keep: int, torn: bytes) -> None:
        if self.quarantine is not None:
            self.quarantine.parent.mkdir(parents=True, exist_ok=True)
            with self.quarantine.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"time_utc": _now(), "file": self.path.name,
                                         "torn_line": torn.decode("utf-8", "replace")}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        with self.path.open("r+b") as handle:
            handle.truncate(keep)
            handle.flush()
            os.fsync(handle.fileno())

    def _apply(self, record: dict[str, Any]) -> None:
        op = record.get("op")
        if op == "reserve":
            self.pending[record["id"]] = record
        elif op in ("settle", "charge"):
            usd = float(record["usd"])
            self.pending.pop(record.get("id"), None)
            self.charged_usd += usd
            role = record.get("role") if record.get("role") in ROLES else "agent"
            self.charged_by_role[role] += usd
            if record.get("status") == "unreconciled_reservation":
                self.unreconciled.append(record)
        elif op == "halt":
            if record.get("session") == self.session and self.halted is None:
                self.halted = record
        elif op == "refuse":
            # A refusal is a halt of its session (one atomic record).
            self.refusals += 1
            if record.get("session") == self.session and self.halted is None:
                self.halted = record
        elif op == "session":
            self.sessions += 1

    def _append(self, record: dict[str, Any]) -> None:
        line = (json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n").encode()
        with self.path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self._offset += len(line)
        self._apply(record)

    # ------------------------------------------------------------------ queries

    @property
    def pending_usd(self) -> float:
        return sum(float(r["usd"]) for r in self.pending.values())

    @property
    def committed_usd(self) -> float:
        return self.charged_usd + self.pending_usd

    def snapshot(self) -> dict[str, Any]:
        with self._locked():
            return {"cap_usd": self.cap_usd, "charged_usd": round(self.charged_usd, 12),
                    "pending_usd": round(self.pending_usd, 12),
                    "committed_usd": round(self.committed_usd, 12),
                    "charged_by_role": dict(self.charged_by_role),
                    "pending": sorted(self.pending.values(), key=lambda r: r["id"]),
                    "unreconciled": list(self.unreconciled),
                    "halted": self.halted, "refusals": self.refusals}

    # ------------------------------------------------------------------ operations

    def assert_not_halted(self) -> None:
        """Raise BudgetExceeded if any worker halted this session (lock held while
        reading the ledger)."""
        with self._locked():
            self._refuse_if_halted()

    def open_session(self, manifest_spent_usd: float) -> dict[str, Any]:
        """Coordinator: mark a session start. If the manifest recorded more spend
        than the ledger holds, charge the difference (take the max of the two)."""
        with self._locked():
            gap = float(manifest_spent_usd) - self.committed_usd
            if math.isfinite(gap) and gap > EPSILON:
                self._append({"op": "charge", "id": None, "owner": self.owner,
                              "session": self.session, "role": "agent", "model": None,
                              "usd": gap, "status": "manifest_reconciliation",
                              "time_utc": _now()})
            record = {"op": "session", "owner": self.owner, "session": self.session,
                      "manifest_spent_usd": manifest_spent_usd,
                      "ledger_committed_usd": self.committed_usd, "time_utc": _now()}
            self._append(record)
            return record

    def _refuse_if_halted(self) -> None:
        if self.halted is not None:
            raise BudgetExceeded(f"run halted by {self.halted.get('owner')}: "
                                 f"{self.halted.get('reason')}")

    def check(self) -> float:
        """Authorize a zero-cost attempt: refused only when the run is halted or
        committed spend is already past the cap. Returns committed spend."""
        with self._locked():
            self._refuse_if_halted()
            committed = self.committed_usd
            if not math.isfinite(committed) or committed > self.cap_usd + EPSILON:
                raise BudgetExceeded(f"committed ${committed:.6f} is past the "
                                     f"${self.cap_usd:.6f} cap")
            return committed

    def reserve(self, role: str, model: str, usd: float, attempt: int) -> str:
        """Append a reservation once committed spend plus ``usd`` fits the cap."""
        if not math.isfinite(usd) or usd < 0:
            raise BudgetExceeded("non-finite reservation; refusing")
        deadline = time.monotonic() + self.wait_s
        while True:
            with self._locked():
                self._refuse_if_halted()
                committed = self.committed_usd
                charged = self.charged_usd
                exhausted = (not math.isfinite(committed)
                             or charged + usd > self.cap_usd + EPSILON)
                if not exhausted and committed + usd <= self.cap_usd + EPSILON:
                    self._counter += 1
                    rid = f"{self.owner}-{os.getpid()}-{self._counter}-{uuid.uuid4().hex[:8]}"
                    self._append({"op": "reserve", "id": rid, "owner": self.owner,
                                  "session": self.session, "role": role, "model": model,
                                  "usd": usd, "attempt": attempt, "time_utc": _now()})
                    return rid
                if exhausted or time.monotonic() >= deadline:
                    reason = "budget_refused" if exhausted else "budget_wait_timeout"
                    base = {"owner": self.owner, "session": self.session, "time_utc": _now()}
                    self._append({"op": "refuse", **base, "role": role, "model": model,
                                  "usd": usd, "charged_usd": charged,
                                  "committed_usd": committed, "reason": reason,
                                  "halts_session": True})
                    raise BudgetExceeded(
                        f"{role} call to {model!r} reserves up to ${usd:.6f}; committed "
                        f"${committed:.6f} (charged ${charged:.6f}) of ${self.cap_usd:.6f} "
                        f"cap across all workers ({reason})")
            # Fits once other workers' in-flight reservations settle: wait for them.
            self.waits += 1
            _sleep(POLL_S * random.uniform(0.5, 1.5))

    def settle(self, rid: str | None, usd: float, role: str, model: str) -> float:
        """Replace a reservation by its charged cost (or record an unreserved
        charge). Returns the run's charged total."""
        with self._locked():
            if rid is not None and rid not in self.pending:
                # Already reconciled (charged in full): charge again, conservatively.
                op, status = "charge", "late_settle"
            else:
                op, status = ("settle", "charged") if rid is not None else ("charge", "charged")
            self._append({"op": op, "id": rid, "owner": self.owner, "session": self.session,
                          "role": role, "model": model, "usd": usd, "status": status,
                          "time_utc": _now()})
            return self.charged_usd

    def halt(self, reason: str) -> None:
        with self._locked():
            if self.halted is None:
                self._append({"op": "halt", "owner": self.owner, "session": self.session,
                              "reason": reason, "time_utc": _now()})

    def reconcile(self, owners: set[str] | None = None) -> list[dict[str, Any]]:
        """Charge in full every unsettled reservation (of ``owners``, or all).

        Only call this for workers known to be dead: a live worker's
        reservation is still in flight.
        """
        settled = []
        with self._locked():
            for rid, record in sorted(self.pending.items()):
                if owners is not None and record.get("owner") not in owners:
                    continue
                entry = {"op": "settle", "id": rid, "owner": record.get("owner"),
                         "session": record.get("session"), "role": record.get("role"),
                         "model": record.get("model"), "usd": float(record["usd"]),
                         "attempt": record.get("attempt"),
                         "status": "unreconciled_reservation", "reconciled_by": self.owner,
                         "time_utc": _now()}
                self._append(entry)
                settled.append(entry)
        return settled


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Every complete ledger record, in order (for audits and tests)."""
    records = []
    for line in Path(path).read_bytes().split(b"\n"):
        if line.strip():
            with contextlib.suppress(ValueError):
                records.append(json.loads(line))
    return records


def max_committed(records: list[dict[str, Any]]) -> float:
    """Replay a ledger and return the highest committed spend it ever reached."""
    pending: dict[str, float] = {}
    charged = peak = 0.0
    for record in records:
        if record.get("op") == "reserve":
            pending[record["id"]] = float(record["usd"])
        elif record.get("op") in ("settle", "charge"):
            pending.pop(record.get("id"), None)
            charged += float(record["usd"])
        peak = max(peak, charged + sum(pending.values()))
    return peak
