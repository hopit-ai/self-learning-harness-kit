"""Validated lesson records with an append-only JSONL audit trail."""

from __future__ import annotations

import copy
import fcntl
import json
import math
import os
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

LAYERS = frozenset({"L1", "L2", "L3", "L4"})
STATUSES = frozenset({"candidate", "active", "disabled", "archived"})


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _round(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("rounds and counters must be nonnegative integers")


def _stats() -> dict[str, int]:
    return {"retrieved": 0, "helped": 0, "hurt": 0,
            "first_seen_round": 0, "last_changed_round": 0}


def require_evidence_eligible(manifest: Mapping[str, Any]) -> None:
    """Refuse runs that may not feed learning: stub participants or stub judge,
    non-stream splits, or runs that did not finish cleanly."""
    if manifest.get("evidence_eligible") is not True:
        raise ValueError("run is not evidence-eligible (stub backend or stub judge)")
    if manifest.get("state_rehearsal") is not False and "state_rehearsal" in manifest:
        raise ValueError("run used a harness state from a learner rehearsal; not evidence")
    if manifest.get("split_name") != "stream":
        raise ValueError("lesson evidence must come from a stream-split run")
    if manifest.get("status") not in ("complete", "complete_with_errors"):
        raise ValueError("lesson evidence must come from a finished run")


def public_run_file(run_dir: str | Path, name: str) -> Path:
    """A learner-readable file of a run, checked BEFORE it is opened: not a
    symlink anywhere below the run directory, inside the run directory once
    resolved, and not under its private/ directory."""
    root = Path(run_dir).resolve()
    path = Path(run_dir) / name
    current = Path(run_dir)
    for part in Path(name).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"refusing to read {name!r}: symlinked run files are not allowed")
    resolved = path.resolve()
    private = root / "private"
    if root not in resolved.parents or resolved == private or private in resolved.parents:
        raise ValueError(f"refusing to read {name!r}: outside the learner-readable run files")
    return path


def valid_episode_ids(run_dir: Path) -> set[str]:
    """Episodes of a run that have an episode line and a valid outcome."""
    def ids(name: str, valid_only: bool) -> set[str]:
        path = public_run_file(run_dir, name)
        found = set()
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue  # a torn final line is not evidence
            if not valid_only or row.get("valid") is True:
                found.add(row["episode_id"])
        return found

    return ids("episodes.jsonl", False) & ids("outcomes.jsonl", True)


def verify_run_sources(run_manifest: str | Path, episode_ids: tuple[str, ...]) -> None:
    """Every source episode must come from an evidence-eligible stream run and
    have a valid outcome there."""
    path = Path(run_manifest)
    require_evidence_eligible(json.loads(path.read_text(encoding="utf-8")))
    valid = valid_episode_ids(path.parent)
    for episode in episode_ids:
        if episode not in valid:
            raise ValueError(f"source episode is not verified stream evidence: {episode} "
                             "(no valid outcome in the source run)")


@dataclass(frozen=True)
class Provenance:
    source_episode_ids: tuple[str, ...]
    run_manifest: str  # path to the source run's manifest.json (required)
    split: str = "stream"

    def __post_init__(self) -> None:
        if not isinstance(self.source_episode_ids, (tuple, list)):
            raise ValueError("source episodes must be a sequence of ids")
        object.__setattr__(self, "source_episode_ids", tuple(self.source_episode_ids))
        if not self.source_episode_ids or len(set(self.source_episode_ids)) != len(
            self.source_episode_ids
        ):
            raise ValueError("provenance requires unique source episode ids")
        for episode in self.source_episode_ids:
            _text(episode, "source episode id")
        if self.split != "stream":
            raise ValueError("lesson provenance must come only from the stream split")
        _text(self.run_manifest, "run manifest path")
        verify_run_sources(self.run_manifest, self.source_episode_ids)


@dataclass(frozen=True)
class LessonRecord:
    id: str
    content: str
    layer: str
    form: str
    scope: str
    provenance: Provenance
    stats: dict[str, int] = field(default_factory=_stats)
    gate: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    status: str = "candidate"
    status_history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _text(self.content, "content")
        if self.layer not in LAYERS:
            raise ValueError("invalid lesson layer")
        if self.form not in {"fact", "procedure", "structural", "tacit"}:
            raise ValueError("invalid lesson form")
        if self.scope not in {"instance", "family", "global"}:
            raise ValueError("invalid lesson scope")
        if not isinstance(self.provenance, Provenance):
            raise ValueError("provenance must be a Provenance record")
        if self.status not in STATUSES:
            raise ValueError("invalid lesson status")
        if not isinstance(self.stats, dict) or not _stats().keys() <= self.stats.keys():
            raise ValueError("missing shared lesson statistics")
        for value in self.stats.values():
            _round(value)
        for name in ("gate", "history", "status_history"):
            entries = getattr(self, name)
            if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
                raise ValueError(f"{name} must be a list of records")
            rounds = [e.get("round") for e in entries]
            for value in rounds:
                _round(value)
            if rounds != sorted(rounds):
                raise ValueError(f"{name} rounds cannot go backwards")
        for entry in self.gate:
            _round(entry.get("round"))
            if entry.get("layer") not in LAYERS:
                raise ValueError("invalid gate layer")
            ci = entry.get("ci")
            values = [entry.get("dev_delta")]
            if not isinstance(ci, (list, tuple)) or len(ci) != 2:
                raise ValueError("gate confidence interval must have two endpoints")
            values.extend(ci)
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in values) or ci[0] > ci[1]:
                raise ValueError("invalid gate measurements")
        for index, entry in enumerate(self.history):
            _round(entry.get("round"))
            _text(entry.get("reason"), "layer history reason")
            if entry.get("to") not in LAYERS or entry.get("from") not in LAYERS | {None}:
                raise ValueError("invalid layer history")
            expected = self.history[index - 1]["to"] if index else None
            if entry.get("from") != expected:
                raise ValueError("discontinuous layer history")
        if self.history and self.history[-1]["to"] != self.layer:
            raise ValueError("layer history does not match current layer")
        for index, entry in enumerate(self.status_history):
            _round(entry.get("round"))
            _text(entry.get("reason"), "status history reason")
            _text(entry.get("created_utc"), "status history timestamp")
            expected = self.status_history[index - 1]["to"] if index else None
            if entry.get("from") != expected or entry.get("to") not in STATUSES:
                raise ValueError("discontinuous status history")
        if self.status_history and self.status_history[-1]["to"] != self.status:
            raise ValueError("status history does not match current status")
        # Isolate mutable caller-owned dictionaries and histories.
        for name in ("stats", "gate", "history", "status_history"):
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))

    @property
    def source_traces(self) -> tuple[str, ...]:
        """Shared-contract alias: a source trace is an episode id."""
        return self.provenance.source_episode_ids

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["source_traces"] = list(self.source_traces)
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> LessonRecord:
        data = copy.deepcopy(row)
        provenance = Provenance(**data.pop("provenance"))
        if data.pop("source_traces", list(provenance.source_episode_ids)) != list(
            provenance.source_episode_ids
        ):
            raise ValueError("source trace/provenance mismatch")
        return cls(provenance=provenance, **data)


class LessonStore:
    """Append snapshots; creation content and provenance remain immutable.

    Supply episode_tasks checked against the frozen retail split, or run_dirs
    whose manifests and recorded episode task ids are checked against it.
    episode_splits is only an additional restriction, never sufficient evidence.
    A torn final JSONL line is reported by warnings and ignored for queries;
    writes refuse to bury it as an invalid interior line. No bytes are deleted.
    """

    def __init__(self, path: str | Path, *, episode_splits: Mapping[str, str] | None = None,
                 episode_tasks: Mapping[str, str] | None = None,
                 run_dirs: list[str | Path] | None = None,
                 retail_split_path: str | Path | None = None) -> None:
        self.path = Path(path)
        self.episode_splits = None if episode_splits is None else dict(episode_splits)
        self.episode_tasks = dict(episode_tasks or {})
        self.run_dirs = [Path(directory) for directory in run_dirs or []]
        self.retail_split_path = Path(retail_split_path) if retail_split_path else (
            Path(__file__).resolve().parents[1] / "splits/retail.json")
        self.warnings: list[str] = []
        self._truncated = False

    def _source_index(self) -> tuple[set[str], dict[str, str]]:
        split = json.loads(self.retail_split_path.read_text(encoding="utf-8"))
        if split.get("domain") != "retail":
            raise ValueError("lesson evidence requires the frozen retail split")
        stream = set(split["stream"])
        index = dict(self.episode_tasks)
        for directory in self.run_dirs:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("domain") != "retail" or manifest.get(
                "split_name", manifest.get("split")) != "stream":
                raise ValueError("run manifest is not verified stream evidence")
            require_evidence_eligible(manifest)
            valid = valid_episode_ids(directory)
            tasks = manifest.get("task_ids")
            if not isinstance(tasks, list) or not set(tasks) <= stream:
                raise ValueError("run manifest tasks are not verified stream evidence")
            for line in (directory / "episodes.jsonl").read_text(encoding="utf-8").split("\n"):
                if not line.strip():
                    continue
                episode = json.loads(line)
                eid, task_id = episode["episode_id"], episode["task_id"]
                if eid not in valid:
                    continue  # invalid (infrastructure/grading failure) episodes are not evidence
                if task_id not in tasks or (eid in index and index[eid] != task_id):
                    raise ValueError("episode task does not match verified stream manifest")
                index[eid] = task_id
        return stream, index

    def _validate_sources(self, lesson: LessonRecord) -> None:
        stream, index = self._source_index()
        for episode_id in lesson.source_traces:
            if index.get(episode_id) not in stream or (
                self.episode_splits is not None and self.episode_splits.get(episode_id) != "stream"
            ):
                raise ValueError(f"source episode is not verified stream evidence: {episode_id}")

    @staticmethod
    def _transition(previous: LessonRecord, lesson: LessonRecord) -> None:
        for name in ("id", "content", "form", "scope", "provenance"):
            if getattr(previous, name) != getattr(lesson, name):
                raise ValueError("lesson snapshots must preserve content-at-creation and provenance")
        for name in ("gate", "history", "status_history"):
            before, after = getattr(previous, name), getattr(lesson, name)
            if after[:len(before)] != before or len(after) < len(before):
                raise ValueError(f"{name} must be append-only")
        if lesson.stats["first_seen_round"] != previous.stats["first_seen_round"]:
            raise ValueError("first_seen_round is immutable")
        if not previous.stats.keys() <= lesson.stats.keys() or any(
            lesson.stats[key] < value for key, value in previous.stats.items()
        ):
            raise ValueError("lesson counters cannot decrease")

    def _read(self, handle: TextIO) -> dict[str, LessonRecord]:
        handle.seek(0)
        self.warnings = []
        self._truncated = False
        latest: dict[str, LessonRecord] = {}
        for line_number, line in enumerate(handle, 1):
            try:
                try:
                    row = json.loads(line)
                except ValueError:
                    if not line.endswith("\n"):
                        message = f"truncated final lesson store line {line_number}; ignored, bytes preserved"
                        self.warnings.append(message)
                        self._truncated = True
                        warnings.warn(message, RuntimeWarning, stacklevel=2)
                        break
                    raise
                lesson = LessonRecord.from_dict(row)
                self._validate_sources(lesson)
                if not lesson.history or not lesson.status_history:
                    raise ValueError("saved lessons require layer and status histories")
                previous = latest.get(lesson.id)
                if previous is not None:
                    self._transition(previous, lesson)
                elif len(lesson.status_history) != 1 or lesson.status != "candidate":
                    raise ValueError("first snapshot must start status history")
                latest[lesson.id] = lesson
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"invalid lesson store line {line_number}: {error}") from error
        return latest

    def _append(self, handle: TextIO, lesson: LessonRecord) -> None:
        if self._truncated:
            raise ValueError("cannot append after a truncated final line; existing bytes preserved")
        handle.seek(0, 2)
        # A complete final JSON object without its newline is still readable.
        if handle.tell():
            handle.seek(0)
            tail = handle.read()
            handle.seek(0, 2)
            if not tail.endswith("\n"):
                handle.write("\n")
        handle.write(json.dumps(lesson.to_dict(), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    @staticmethod
    def _event(previous: str | None, status: str, reason: str, round: int) -> dict[str, Any]:
        return {"from": previous, "to": status, "reason": reason, "round": round,
                "created_utc": datetime.now(UTC).isoformat()}

    def add(self, lesson: LessonRecord) -> LessonRecord:
        # Revalidate even if a nested dictionary was modified after construction.
        lesson = LessonRecord.from_dict(lesson.to_dict())
        self._validate_sources(lesson)
        if lesson.status != "candidate":
            raise ValueError("new lessons must start as candidate")
        if lesson.status_history:
            raise ValueError("status history is created by the store")
        first_round = lesson.stats["first_seen_round"]
        lesson = replace(lesson,
                         history=lesson.history or [{"round": first_round, "from": None,
                                                    "to": lesson.layer, "reason": "new"}],
                         status_history=[self._event(None, lesson.status, "new", first_round)])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8", errors="surrogateescape", newline="") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            if lesson.id in self._read(handle):
                raise ValueError(f"lesson id already exists: {lesson.id}")
            self._append(handle, lesson)
        return lesson

    def query(self, *, status: str | None = None, layer: str | None = None) -> list[LessonRecord]:
        if status is not None and status not in STATUSES:
            raise ValueError("invalid lesson status")
        if layer is not None and layer not in LAYERS:
            raise ValueError("invalid lesson layer")
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8", errors="surrogateescape", newline="") as handle:
            fcntl.flock(handle, fcntl.LOCK_SH)
            latest = self._read(handle)
        return [lesson for _, lesson in sorted(latest.items())
                if (status is None or lesson.status == status)
                and (layer is None or lesson.layer == layer)]

    def update_status(self, lesson_id: str, status: str, *, reason: str,
                      round: int = 0) -> LessonRecord:
        return self.update(lesson_id, status=status, reason=reason, round=round)

    def update(self, lesson_id: str, *, stats: Mapping[str, int] | None = None,
               gate: dict[str, Any] | None = None, layer: str | None = None,
               status: str | None = None, reason: str = "", round: int = 0) -> LessonRecord:
        """Update absolute counters and append gate, promotion or status events.

        Existing history entries, creation content and provenance cannot be
        edited. Stats is a partial mapping; counters may only increase.
        """
        if status is not None and status not in STATUSES:
            raise ValueError("invalid lesson status")
        if status is not None or layer is not None:
            _text(reason, "update reason")
        _round(round)
        if not self.path.exists():
            raise KeyError(lesson_id)
        with self.path.open("a+", encoding="utf-8", errors="surrogateescape", newline="") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            previous = self._read(handle)[lesson_id]
            last_round = max(previous.stats["last_changed_round"],
                             *(entry["round"] for name in ("gate", "history", "status_history")
                               for entry in getattr(previous, name)))
            if round < last_round:
                raise ValueError("update round cannot go backwards")
            status_history, history = previous.status_history, previous.history
            if status is not None and status != previous.status:
                status_history = [*status_history, self._event(previous.status, status, reason, round)]
            if layer is not None and layer != previous.layer:
                history = [*history, {"from": previous.layer, "to": layer,
                                     "reason": reason, "round": round}]
            if gate is not None and gate.get("round") != round:
                raise ValueError("gate round must match update round")
            updated = replace(previous, stats={**previous.stats, **(stats or {})},
                              gate=[*previous.gate, gate] if gate is not None else previous.gate,
                              layer=previous.layer if layer is None else layer, history=history,
                              status=previous.status if status is None else status,
                              status_history=status_history)
            if updated == previous:
                return previous
            updated = replace(updated, stats={**updated.stats, "last_changed_round": round})
            self._transition(previous, updated)
            self._append(handle, updated)
        return updated

    def archive(self, lesson_id: str, *, reason: str, round: int = 0) -> LessonRecord:
        return self.update_status(lesson_id, "archived", reason=reason, round=round)
