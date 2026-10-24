"""Shared types and the plug-in interfaces: ``AgentAdapter`` and ``Proposer``.

An adapter wraps one frozen agent on one surface (tau2 retail decisions, tau2 episodes, BFCL
multi-turn, later a Hopit agent). It owns drawing units, running a unit under a harness
configuration with its guard chain mediating every tool call, and scoring each run.

The harness trusts none of it blindly (docs/system/v1-design.md s. 8): every outcome must
match one request exactly (unit, role, tag, configuration version) and echo the SHA-256 of the
state the adapter actually applied and of the guard chain it ran; units are admitted through
the harness's persistent unit registry; guard source hashes are computed by the harness; and
the guard chain is probed with harness-owned canaries every cycle.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from slh.state import HarnessState

ROLES = ("evidence", "screen", "confirm", "monitor")
STATUSES = ("scored", "unscored", "error")
KINDS = ("note", "checker", "weights", "skill", "memory")
# Host learned-state kinds (docs/system/host-integration-spike.md s. 4): full post-states of
# one target each (a skill by name, or the MEMORY target), so two candidates with the same
# ``target_key`` overwrite each other and are never applied together (conflict exclusion).
MEMORY_TARGETS = ("MEMORY",)  # USER.md stays disabled on benchmarks (each task = new user)
MEMORY_DELIMITER = "\n§\n"  # Hermes' MEMORY.md entry delimiter (tools/memory_tool_store.py)


@dataclass(frozen=True)
class Unit:
    """One evaluation unit. ``cluster``: the independence cluster (for tau2 retail the
    configuration group / user); clusters never cross roles."""

    unit_id: str
    role: str
    context: str
    cluster: str
    payload: Mapping[str, Any] = field(default_factory=dict)  # adapter-private, no gold


@dataclass(frozen=True)
class RunRequest:
    unit: Unit
    config: HarnessState
    tag: str  # "incumbent", "root", "candidate", or "x=0101" for screen draws

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.unit.unit_id, self.unit.role, self.tag, self.config.version_id)


@dataclass(frozen=True)
class Outcome:
    unit_id: str
    role: str
    context: str
    tag: str
    config_version: str
    applied_state_sha256: str  # echo: content_sha256 of the state the adapter applied
    guard_chain_sha256: str  # echo: the guard chain identity the adapter ran
    mediated: bool  # every tool call of this run passed through the guard chain
    status: str  # scored | unscored | error
    success: bool | None
    harm: bool | None
    # Critical harm (owner 2026-10-23): a write or action the user never authorised, i.e. a
    # consent or identity violation by the guards' ground-truth labeller. Tri-state:
    # True / False / None (no label: no trace, or the labeller could not decide). The label is
    # read off the tool-call trace, so it is independent of task scoring: an errored run that
    # made no unauthorised write is labelled False. Unlabelled runs fail closed.
    critical_harm: bool | None = None
    cost_usd: float = 0.0
    incident: bool = False  # safety incident flagged by the adapter or a human
    trace: Mapping[str, Any] | None = None  # learner-safe trace (no grading fields)

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.unit_id, self.role, self.tag, self.config_version)

    @property
    def scored(self) -> bool:
        return self.status == "scored"

    def worst(self) -> tuple[int, int]:
        """(success, harm), a non-scored run counted as the worst case (0, 1)."""
        return (int(bool(self.success)), int(bool(self.harm))) if self.scored else (0, 1)

    def critical_worst(self) -> int:
        """1 unless the run is labelled not critical (fail closed: unlabelled counts as 1)."""
        return 0 if self.critical_harm is False else 1

    def critical_best(self) -> int:
        """1 only if the run is labelled critical (unlabelled counts as 0)."""
        return 1 if self.critical_harm is True else 0

    def best(self) -> tuple[int, int]:
        """(success, harm), a non-scored run counted as the best case (1, 0)."""
        return (int(bool(self.success)), int(bool(self.harm))) if self.scored else (1, 0)


@dataclass(frozen=True)
class AdapterInfo:
    """Identity the harness pins at start and re-checks every cycle."""

    agent_id: str
    model_identity: str  # model name + weights / serving spec hash
    scorer_identity: str  # scorer name + source hash
    sampling_identity: str  # unit generators, salts, pool definitions
    surface: str
    guard_modules: tuple[str, ...]  # import names; the harness hashes their sources itself
    contexts: tuple[str, ...]
    paid: bool = False  # runs cost real money: the harness requires an owner-approved item cap


class AdapterBudgetStop(RuntimeError):
    """Raised by ``AgentAdapter.run`` when finishing the dispatch would exceed ``budget_usd``
    (the harness's reservation). The harness charges the reservation in full and latches a
    system halt: the estimate that sized it was wrong."""


@runtime_checkable
class AgentAdapter(Protocol):
    def info(self) -> AdapterInfo: ...

    def sample_units(self, batch: int, role: str, n: int, *, k: int | None = None,
                     level: str = "decision") -> list[Unit]:
        """``n`` units for ``role``. Confirm units come from the adapter's permanent confirm
        partition (``k``: the confirmation index, which seeds the draw). ``level``:
        ``decision`` or ``conversation`` (full episodes; production adoption needs one)."""

    def run(self, requests: Sequence[RunRequest], *,
            budget_usd: float | None = None) -> list[Outcome]:
        """Run every request with the guard chain mediating tool calls; one outcome each.
        ``budget_usd``: the reservation covering this dispatch, a HARD stop: a paid adapter
        must raise ``AdapterBudgetStop`` rather than spend beyond it."""

    def estimate_cost_usd(self, n_runs: int, *, level: str = "decision") -> float:
        """Worst-case cost of ``n_runs`` runs at ``level`` (decision | conversation), GPU
        window share included."""

    def policy_conflict(self, text: str) -> str | None:
        """Why a note text is inadmissible on this surface (policy screen), else None."""

    def guard_probe(self, tool: str, arguments: Mapping[str, Any],
                    observations: Sequence[tuple[str, Mapping[str, Any], str]]) -> bool:
        """Conformance: one proposed call through the DEPLOYED guard chain after the given
        (tool, arguments, result) observations; True = allowed."""


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


@dataclass(frozen=True)
class Candidate:
    """One proposed change, typed and content-addressed.

    ``note``: payload {"text"}, applied as one note. ``checker``: payload {"name",
    "artifact_sha256", ...}, applied as an ADDITIONAL guard spec (the floor is never removed).
    ``weights``: payload {"artifact_sha256", "base"}; declared, applied from M6 on.
    ``skill``: payload {"name", "description", "body"[, "files": {relpath: sha256}]} or
    {"name", "tombstone": True}: the FULL post-state of one skill (``HarnessState.skills``,
    rendered by ``render_skill``), or its deletion. A ``skill_ref`` note (empty content)
    records the candidate id, so provenance and quarantine (``loop._ids``) see it.
    ``memory``: payload {"target": "MEMORY", "entries": [str, ...]}: the full post-state of
    the target, kept as one ``memory`` note (content = entries joined by the delimiter)."""

    cid: str
    kind: str
    payload: Mapping[str, Any]
    tools: tuple[str, ...] = ("any",)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown candidate kind {self.kind!r}")
        if self.kind == "skill":
            name = self.payload.get("name")
            if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name):
                raise ValueError(f"skill candidate {self.cid}: unsafe or missing name")
            if not self.payload.get("tombstone") and not all(
                    isinstance(self.payload.get(k), str) for k in ("description", "body")):
                raise ValueError(f"skill candidate {self.cid}: description and body required")
        if self.kind == "memory":
            entries = self.payload.get("entries")
            if self.payload.get("target") not in MEMORY_TARGETS or not isinstance(
                    entries, list) or not all(isinstance(e, str) for e in entries):
                raise ValueError(f"memory candidate {self.cid}: target MEMORY and a list of "
                                 "string entries required")

    @property
    def text(self) -> str:
        """The admissible text the policy screen reads: the note, the skill's description and
        body, or the memory entries."""
        if self.kind == "skill":
            return "\n".join(str(self.payload.get(k, "")) for k in ("description", "body"))
        if self.kind == "memory":
            return "\n".join(str(e) for e in self.payload.get("entries", []))
        return str(self.payload.get("text", ""))

    @property
    def target_key(self) -> str | None:
        """Conflict key of a full post-state kind (None: additive kinds never conflict)."""
        if self.kind == "skill":
            return "skill:" + str(self.payload.get("name", "")).casefold()
        if self.kind == "memory":
            return "memory:" + str(self.payload.get("target", ""))
        return None

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(_canon({"kind": self.kind, "payload": dict(self.payload)})
                              .encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"cid": self.cid, "kind": self.kind, "payload": dict(self.payload),
                "tools": list(self.tools), "provenance": dict(self.provenance),
                "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Candidate:
        c = cls(d["cid"], d["kind"], dict(d["payload"]), tuple(d["tools"]),
                dict(d.get("provenance", {})))
        if d.get("artifact_sha256") not in (None, c.artifact_sha256):
            raise ValueError(f"candidate {c.cid}: artifact hash mismatch")
        return c

    def apply(self, state: HarnessState) -> HarnessState:
        notes, guards = state.notes, state.guards
        if self.kind in ("skill", "memory"):
            return self._apply_post_state(state)
        if self.kind == "note":
            if any(n.get("id") == self.cid for n in notes):
                return state
            notes = notes + [{"id": self.cid, "content": self.text, "kind": "note",
                              "artifact_sha256": self.artifact_sha256}]
        elif self.kind == "checker":
            if any(g.get("id") == self.cid for g in guards):
                return state
            guards = guards + [{**{k: v for k, v in self.payload.items()},
                                "id": self.cid, "kind": "checker",
                                "artifact_sha256": self.artifact_sha256}]
        else:
            raise NotImplementedError("weights candidates are applied from M6 on")
        return HarnessState(notes, state.skills, guards, rehearsal=state.rehearsal,
                            base=state.base)


    def _apply_post_state(self, state: HarnessState) -> HarnessState:
        """``skill`` / ``memory``: replace the target's state with this full post-state."""
        notes, skills = state.notes, state.skills
        if any(n.get("id") == self.cid for n in notes):
            return state
        if self.kind == "skill":
            name = str(self.payload["name"])
            key = name.casefold()
            skills = {k: v for k, v in skills.items() if k.casefold() != key}
            if not self.payload.get("tombstone"):
                skills[name] = render_skill(name, str(self.payload["description"]),
                                            str(self.payload["body"]))
            notes = [n for n in notes if not (n.get("kind") == "skill_ref" and str(
                n.get("name", "")).casefold() == key)]
            notes.append({"id": self.cid, "kind": "skill_ref", "name": name, "content": "",
                          "tombstone": bool(self.payload.get("tombstone")),
                          "artifact_sha256": self.artifact_sha256})
        else:
            target = str(self.payload["target"])
            entries = [str(e) for e in self.payload["entries"]]
            notes = [n for n in notes if not (n.get("kind") == "memory"
                                              and n.get("target") == target)]
            notes.append({"id": self.cid, "kind": "memory", "target": target,
                          "entries": entries, "content": MEMORY_DELIMITER.join(entries),
                          "artifact_sha256": self.artifact_sha256})
        return HarnessState(notes, skills, state.guards, rehearsal=state.rehearsal,
                            base=state.base)


def render_skill(name: str, description: str, body: str) -> str:
    """Canonical SKILL.md text of a ``skill`` post-state (UTF-8, LF; frontmatter keys other
    than name and description are not carried in V1)."""
    desc = " ".join(description.split())
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    return (f"---\nname: {name}\ndescription: {json.dumps(desc, ensure_ascii=False)}\n---\n"
            f"{body.rstrip(chr(10))}\n")


def target_conflicts(cands: Sequence[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    """Conflict exclusion: (kept, dropped). The first candidate per target key is kept; later
    ones with the same key are dropped (full post-states would overwrite each other)."""
    seen: set[str] = set()
    kept, dropped = [], []
    for c in cands:
        key = c.target_key
        if key is not None and key in seen:
            dropped.append(c)
            continue
        if key is not None:
            seen.add(key)
        kept.append(c)
    return kept, dropped


def apply_all(state: HarnessState, cands: Sequence[Candidate]) -> HarnessState:
    keys = [c.target_key for c in cands if c.target_key is not None]
    if len(set(keys)) != len(keys):
        raise ValueError("apply_all: two candidates share a target key (order-dependent)")
    for c in cands:
        state = c.apply(state)
    return state


@runtime_checkable
class Proposer(Protocol):
    """A proposer that costs money sets ``paid = True`` (or a non-zero estimate); either makes
    the harness require an owner-approved item cap."""

    last_cost_usd: float

    def estimate_cost_usd(self) -> float:
        """Worst-case cost of one ``propose`` call (reserved before it runs)."""

    def propose(self, batch: int, evidence: Sequence[Outcome], incumbent: HarnessState,
                retired: set[str]) -> list[Candidate]:
        """Candidates from evidence-role traces only; never re-proposes a retired id."""
