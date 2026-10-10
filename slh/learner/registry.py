"""Durable cross-round audit and conservative proposal-slot reservations.

Protocol amendments (learning system v2): ``amend`` appends one explicit
``protocol_amendment`` record (amendment id, the SHA-256 of the amendment
text, the new gate configuration hash, the new limits and the screening
rule). Nothing is reset: earlier rounds, their proposal-slot reservations
and the lifetime round / proposal caps registered by the first round keep
counting. After an amendment every round must declare it, use exactly its
gate configuration, screening rule and limits, and the amendment text must
still hash to the recorded value. The amended limits are enforced here,
under the registry's file lock:

- at most ``max_gate_candidates`` gate-selected candidates after the
  amendment (``reserve_gate_candidate``, one per round, never refunded);
- per round at most ``max_drafts_per_round`` drafted alternatives, each with
  at most ``max_critiques_per_draft`` critique and ``max_revisions_per_draft``
  revision (``record_authoring``; a refused call is logged and raises before
  its output can be used); where a config enables format repair, at most one
  ``format_repair`` call per drafted alternative, inside its draft slot and
  before its critique (``record_format_repair``; it reserves no extra slot);
- at most ``max_screen_alternatives`` alternatives screened per round;
- every draft-bearing round reserves exactly ``required_slots`` proposal
  slots (drafts + revisions: 4), when its first draft is dispatched; a round
  in which no draft is dispatched reserves none.

Integrity: every row appended by this code carries ``prev_sha256``, the
SHA-256 of all registry bytes before it (a hash chain), and a sidecar
``REGISTRY.head.json`` records the length and hash of the whole file after
each append. Editing or deleting any complete row before the last breaks the
chain; deleting rows at the end makes the file shorter than its head; deleting
the head of a chained registry refuses too. Every amendment also pins the
exact historical prefix it amends (mandatory at registration). Counters are
always derived from the verified rows.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from slh.learner.triage import public_path

AUTHORING_KINDS = ("draft", "critique", "revision", "recheck")
# Amendment 6b: one structured re-check of the revised artifact per drafted
# alternative, after its revision (it never feeds another revision and reserves
# no slot); refused unless the latest amendment registers ``grounding``.
MAX_RECHECKS_PER_DRAFT = 1


class AmendmentLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    max_gate_candidates: int = Field(default=3, ge=0)
    max_drafts_per_round: int = Field(default=2, ge=1)
    max_critiques_per_draft: int = Field(default=1, ge=0)
    max_revisions_per_draft: int = Field(default=1, ge=0)
    max_screen_alternatives: int = Field(default=2, ge=1)


def required_slots(limits: AmendmentLimits) -> int:
    """Slots a draft-bearing amended round reserves: every draft and revision."""
    return limits.max_drafts_per_round * (1 + limits.max_revisions_per_draft)


def reserved_total(rows: list[dict[str, Any]]) -> int:
    return (sum(r.get("reserved_proposals", 0) for r in rows if r.get("event") == "round_started")
            + sum(r.get("slots", 0) for r in rows if r.get("event") == "slots_reserved"))


def reserved_for(rows: list[dict[str, Any]], round_number: int) -> int:
    return (sum(r.get("reserved_proposals", 0) for r in rows
                if r.get("event") == "round_started" and r.get("round") == round_number)
            + sum(r.get("slots", 0) for r in rows
                  if r.get("event") == "slots_reserved" and r.get("round") == round_number))


def lifetime_caps(rows: list[dict[str, Any]]) -> dict[str, int] | None:
    """Round 1's registered lifetime caps, unless a registered amendment
    supersedes them (amendment 6: ``parameters.lifetime_caps``, 8 rounds and 34
    proposal slots); the latest superseding amendment wins. Consumption is
    never reset."""
    starts = [r for r in rows if r.get("event") == "round_started"]
    caps = None if not starts else {"max_rounds": starts[0]["max_rounds"],
                                    "max_total_proposals": starts[0]["max_total_proposals"]}
    for row in rows:
        if row.get("event") == "protocol_amendment" and isinstance(
                (row.get("parameters") or {}).get("lifetime_caps"), dict):
            caps = dict(row["parameters"]["lifetime_caps"])
    return caps


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


class Registry:
    def __init__(self, directory: Path):
        self.path = public_path(directory / "REGISTRY.jsonl")
        self.head_path = public_path(directory / "REGISTRY.head.json")
        # Second, sealed location: a copy of the head plus every amendment row's
        # offset and hash, so deleting the head (and rows) is still detected.
        self.seal_dir = public_path(directory / ".seal")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8") as out:
            out.write(json.dumps(value, sort_keys=True) + "\n")
            out.flush()
            os.fsync(out.fileno())
        tmp.replace(path)

    def _append(self, handle, event: dict[str, Any]) -> None:
        """Append one chained row; refused unless the current file verifies in full."""
        rows = self._rows(handle)
        body = self._body(handle)
        event["prev_sha256"] = hashlib.sha256(body).hexdigest()  # returned rows match
        line = json.dumps(event, sort_keys=True, allow_nan=False) + "\n"
        handle.seek(0, os.SEEK_END)
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
        whole = body + line.encode("utf-8")
        head = {"bytes": len(whole), "sha256": hashlib.sha256(whole).hexdigest(),
                "records": whole.count(b"\n")}
        if event.get("event") == "protocol_amendment":
            sealed = self._sealed_amendments() + [{
                "amendment_id": event.get("amendment_id"), "offset": len(body),
                "row_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest()}]
            self._write_json(self.seal_dir / "AMENDMENTS.json", sealed)
        self._write_json(self.head_path, head)
        self._write_json(self.seal_dir / "REGISTRY.head.json", head)
        del rows

    def _sealed_amendments(self) -> list[dict[str, Any]]:
        path = self.seal_dir / "AMENDMENTS.json"
        if not path.exists():
            return []
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            raise ValueError("registry seal is damaged; refuse") from None
        if not isinstance(value, list):
            raise ValueError("registry seal is damaged; refuse")
        return value

    @staticmethod
    def _body(handle) -> bytes:
        handle.seek(0)
        return handle.buffer.read() if hasattr(handle, "buffer") else handle.read().encode()

    def _rows(self, handle) -> list[dict[str, Any]]:
        # A torn audit is refused, never treated as free proposal budget.
        raw = self._body(handle)
        try:
            body = raw.decode("utf-8")
            if body and not body.endswith("\n"):
                raise ValueError("torn registry")
            rows = [json.loads(line) for line in body.splitlines()]
        except ValueError:
            raise ValueError("registry is damaged; refuse to reset caps") from None
        self._check_chain(raw, rows)
        self._check_prefixes(handle, rows)
        return rows

    def _check_chain(self, raw: bytes, rows: list[dict[str, Any]]) -> None:
        offsets, position = [], 0
        for line in raw.splitlines(keepends=True):
            offsets.append(position)
            position += len(line)
        chained = False
        for offset, row in zip(offsets, rows, strict=True):
            if "prev_sha256" in row:
                chained = True
                if row["prev_sha256"] != hashlib.sha256(raw[:offset]).hexdigest():
                    raise ValueError("registry hash chain is broken (rows edited or "
                                     "deleted); refuse to reset caps")
            elif chained:
                raise ValueError("registry row without its chain hash after chained rows; "
                                 "refuse")
        heads = [self.head_path, self.seal_dir / "REGISTRY.head.json"]
        sealed = self._sealed_amendments()
        amended = sealed or any(r.get("event") == "protocol_amendment" for r in rows)
        present = [h for h in heads if h.exists()]
        # Once anything was ever sealed or chained, both heads must exist; an
        # empty or unchained file cannot hide behind a missing head.
        if (chained or amended or present) and len(present) != len(heads):
            raise ValueError("registry head is missing or damaged; refuse")
        for path in present:
            try:
                head = json.loads(path.read_text(encoding="utf-8"))
                size, digest_ = int(head["bytes"]), head["sha256"]
            except (OSError, ValueError, KeyError, TypeError):
                raise ValueError("registry head is missing or damaged; refuse") from None
            if size > len(raw) or hashlib.sha256(raw[:size]).hexdigest() != digest_ or (
                    offsets and size < offsets[-1]):
                raise ValueError("registry is shorter than or differs from its recorded "
                                 "head (rows deleted); refuse to reset caps")
        for seal in sealed:
            offset = seal.get("offset")
            line = raw[offset:].split(b"\n", 1)[0] + b"\n" if isinstance(offset, int) \
                and offset < len(raw) else b""
            if hashlib.sha256(line).hexdigest() != seal.get("row_sha256"):
                raise ValueError("a sealed protocol amendment is missing from the registry; "
                                 "refuse")

    @classmethod
    def _check_prefixes(cls, handle, rows: list[dict[str, Any]]) -> None:
        """Every amendment pins the registry bytes before it; a changed prefix
        (edited, truncated or replaced history) refuses."""
        amendments = [r for r in rows if r.get("event") == "protocol_amendment"]
        if not amendments:
            return
        body = cls._body(handle)
        for amendment in amendments:
            size = amendment.get("prefix_bytes")
            if not isinstance(size, int) or hashlib.sha256(body[:size]).hexdigest() != \
                    amendment.get("prefix_sha256"):
                raise ValueError("registry prefix differs from the one the amendment "
                                 "registered; refuse")

    def append(self, event: dict[str, Any]) -> None:
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            self._append(handle, event)

    def rows(self) -> list[dict[str, Any]]:
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_SH)
            return self._rows(handle)

    @staticmethod
    def _amendment(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        found = [r for r in rows if r.get("event") == "protocol_amendment"]
        return found[-1] if found else None

    def amendment(self) -> dict[str, Any] | None:
        return self._amendment(self.rows())

    @staticmethod
    def _after(rows: list[dict[str, Any]], amendment: dict[str, Any]) -> list[dict[str, Any]]:
        return rows[rows.index(amendment) + 1:]

    @staticmethod
    def _chain_start(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        """The first amendment of the linked chain: amended limits (gate
        candidates) and the registered screen count from here, across links."""
        found = [r for r in rows if r.get("event") == "protocol_amendment"]
        return found[0] if found else None

    def amend(self, *, amendment_id: str, doc_path: str, doc_sha256: str,
              gate_config: dict, limits: AmendmentLimits, screen_rule: dict,
              expect_prefix_sha256: str, parameters: dict[str, Any] | None = None,
              links: str | None = None, links_doc_sha256: str | None = None,
              completes_round: int | None = None) -> dict:
        """Append the explicit amendment record; refuses a reused id.

        Records the amendment text hash, the previous gate hash, the hash and
        byte length of the registry prefix it amends, the carried-forward
        counters and the new gate settings, limits and screening rule.
        ``expect_prefix_sha256`` (mandatory): refuse unless the registry's
        current bytes hash to exactly that (the historical prefix the
        amendment text names), so an emptied or replaced registry cannot carry
        forward zero consumption. ``parameters``: every registered round
        parameter (gate, sentinel trials, alternatives, limits, screen, base
        arm); rounds must declare the same set.

        Linked amendments: once an amendment exists, a new one must name the
        latest registered amendment (``links``) and its exact text hash
        (``links_doc_sha256``); it supersedes it for later rounds while every
        counter (rounds, proposal slots, gate candidates since the first
        amendment, the registered screen) carries forward unchanged. An
        unlinked or mis-linked amendment is refused. A linked amendment whose
        parent was never appended to this registry is allowed: it records the
        parent's text hash and ``parent_registered: false`` and never
        fabricates the parent event.

        ``completes_round`` (amendment 5): the amendment completes the frozen
        evaluation of that already-started round, which must be the latest
        one; it authorizes no new round (``effective_from_round`` is None,
        ``authorizes_rounds`` 0) and later ``start`` calls are refused."""
        if not isinstance(expect_prefix_sha256, str) or len(expect_prefix_sha256) != 64:
            raise ValueError("the expected registry prefix hash is mandatory")
        if not amendment_id or not isinstance(amendment_id, str):
            raise ValueError("amendment id required")
        if not isinstance(doc_sha256, str) or len(doc_sha256) != 64:
            raise ValueError("amendment text hash required")
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            rows = self._rows(handle)
            if any(r.get("event") == "protocol_amendment" and r.get("amendment_id")
                   == amendment_id for r in rows):
                raise ValueError(f"amendment {amendment_id} is already registered")
            starts = [r for r in rows if r.get("event") == "round_started"]
            prefix = self._body(handle)
            prefix_sha = hashlib.sha256(prefix).hexdigest()
            if prefix_sha != expect_prefix_sha256:
                raise ValueError(f"registry prefix is {prefix_sha}, not the expected "
                                 f"{expect_prefix_sha256}")
            previous = self._amendment(rows)
            if previous is not None and (links != previous["amendment_id"]
                                         or links_doc_sha256 != previous["doc_sha256"]):
                raise ValueError(f"a new amendment must link the latest one "
                                 f"({previous['amendment_id']}) and its exact text hash")
            if links is not None and (not isinstance(links_doc_sha256, str)
                                      or len(links_doc_sha256) != 64):
                raise ValueError("a linked amendment needs its parent's exact text hash")
            if completes_round is not None and (
                    isinstance(completes_round, bool) or not isinstance(completes_round, int)
                    or not starts or starts[-1].get("round") != completes_round):
                raise ValueError("a completion amendment must name the latest started round")
            first = self._chain_start(rows)
            gate_candidates = 0 if first is None else len(
                [r for r in self._after(rows, first)
                 if r.get("event") == "gate_candidate_reserved"])
            event = {"event": "protocol_amendment", "amendment_id": amendment_id,
                     "links": links, "links_doc_sha256": links_doc_sha256,
                     "links_row_sha256": None if previous is None else digest(previous),
                     # The parent may never have been appended locally (amendment 2
                     # links amendment 1 by text hash); it is never fabricated.
                     "parent_registered": None if links is None else previous is not None,
                     "prefix_sha256": prefix_sha, "prefix_bytes": len(prefix),
                     "prefix_records": len(rows),
                     "previous_gate_sha256": previous["gate_sha256"] if previous else (
                         starts[0]["gate_sha256"] if starts else None),
                     "previous_amendment_id": previous["amendment_id"] if previous else None,
                     "doc_path": doc_path, "doc_sha256": doc_sha256,
                     "gate_sha256": digest(gate_config), "gate_config": gate_config,
                     "limits": limits.model_dump(), "screen_rule": screen_rule,
                     "screen_rule_sha256": digest(screen_rule),
                     "parameters": parameters, "parameters_sha256": digest(parameters),
                     "required_slots": required_slots(limits),
                     "effective_from_round": (len(starts) + 1 if completes_round is None
                                              else None),
                     **({"completes_round": completes_round, "authorizes_rounds": 0}
                        if completes_round is not None else {}),
                     "prior_consumption": {
                         "rounds": len(starts),
                         "reserved_proposals": reserved_total(rows),
                         "gate_candidates_since_first_amendment": gate_candidates,
                         "first_registration": None if not starts else {
                             "gate_sha256": starts[0]["gate_sha256"],
                             "max_rounds": starts[0]["max_rounds"],
                             "max_total_proposals": starts[0]["max_total_proposals"]}}}
            self._append(handle, event)
            return event

    def start(self, *, gate_config: dict, max_rounds: int, max_proposals: int,
              slots: int, out: Path, rehearsal: bool, compliance: list[dict],
              identity: dict[str, Any] | None = None,
              v2: dict[str, Any] | None = None) -> dict:
        """Register a round. ``identity``: base arm + incumbent state version
        (recorded on the row). ``v2``: the amended round's declaration
        (amendment_id, doc_sha256 as hashed now, limits, screen_rule,
        alternatives) -- required once an amendment exists."""
        gate_digest = digest(gate_config)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            rows = self._rows(handle)
            starts = [r for r in rows if r.get("event") == "round_started"]
            amendment = self._amendment(rows)
            reason = None
            caps = lifetime_caps(rows)
            if caps and (caps["max_rounds"] != max_rounds or
                         caps["max_total_proposals"] != max_proposals):
                reason = "gate configuration or lifetime caps changed after registration"
            elif amendment is None and v2 is not None:
                reason = "a v2 round needs a registered protocol amendment"
            elif amendment is None and starts and starts[0]["gate_sha256"] != gate_digest:
                reason = "gate configuration or lifetime caps changed after registration"
            elif amendment is not None and amendment.get("completes_round") is not None:
                reason = (f"amendment {amendment['amendment_id']} completes round "
                          f"{amendment['completes_round']} and authorizes no new round")
            elif amendment is not None:
                reason = self._amended_refusal(rows, amendment, gate_digest, v2)
            needed = slots
            if reason is None and amendment is not None:
                required = required_slots(AmendmentLimits.model_validate(amendment["limits"]))
                if slots not in (0, required):
                    reason = (f"a draft-bearing round reserves exactly {required} slots "
                              f"(refused {slots})")
                needed = required if (v2 or {}).get("may_draft", True) else 0
            if reason is None and len(starts) >= max_rounds:
                reason = "lifetime round cap reached"
            elif reason is None and reserved_total(rows) + needed > max_proposals:
                reason = "lifetime proposal cap reached"
            if reason:
                self._append(handle, {"event": "round_refused", "out": str(out),
                                      "gate_sha256": gate_digest, "reason": reason})
                raise ValueError(reason)
            event = {"event": "round_started", "round": len(starts) + 1,
                     "out": str(out), "rehearsal": rehearsal, "gate_sha256": gate_digest,
                     "max_rounds": max_rounds, "max_total_proposals": max_proposals,
                     "reserved_proposals": slots, "note_compliance": compliance}
            if amendment is not None:
                event["protocol_sha256"] = amendment["doc_sha256"]
            if identity is not None:
                event["identity"] = identity
            if v2 is not None:
                event["amendment_id"] = v2["amendment_id"]
                event["alternatives"] = v2.get("alternatives")
            self._append(handle, event)
            return event

    def _amended_refusal(self, rows: list[dict[str, Any]], amendment: dict[str, Any],
                         gate_digest: str, v2: dict[str, Any] | None) -> str | None:
        if v2 is None:
            return f"protocol amended ({amendment['amendment_id']}): rounds must declare it"
        if v2.get("amendment_id") != amendment["amendment_id"]:
            return "round declares another amendment than the registered one"
        if v2.get("doc_sha256") != amendment["doc_sha256"]:
            return "amendment text changed since it was registered"
        if gate_digest != amendment["gate_sha256"]:
            return "gate configuration differs from the amendment's"
        if v2.get("limits") != amendment["limits"]:
            return "round limits differ from the amendment's"
        if digest(v2.get("screen_rule")) != amendment["screen_rule_sha256"]:
            return "screening rule differs from the amendment's"
        if digest(v2.get("parameters")) != amendment.get("parameters_sha256"):
            return "registered round parameters (gate, sentinel, alternatives, base arm) " \
                   "differ from the amendment's"
        limits = AmendmentLimits.model_validate(amendment["limits"])
        if int(v2.get("alternatives", 0)) > min(limits.max_drafts_per_round,
                                                 limits.max_screen_alternatives):
            return "round asks for more alternatives than the amendment allows"
        used = [r for r in self._after(rows, self._chain_start(rows))
                if r.get("event") == "gate_candidate_reserved"]
        if len(used) >= limits.max_gate_candidates:
            return "amended gate-candidate limit reached"
        return None

    def record_authoring(self, round_number: int, alternative: int, kind: str,
                         raw: str) -> None:
        """Log one draft / critique / revision call; refuse beyond the amended limits."""
        if kind not in AUTHORING_KINDS:
            raise ValueError(f"unknown authoring call {kind!r}")
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            rows = self._rows(handle)
            amendment = self._amendment(rows)
            limits = AmendmentLimits.model_validate(amendment["limits"]) if amendment \
                else AmendmentLimits()
            mine = [r for r in rows if r.get("event") == "authoring_call"
                    and r.get("round") == round_number]
            counts = Counter((r["alternative"], r["kind"]) for r in mine)
            drafted = {r["alternative"] for r in mine if r["kind"] == "draft"}
            reason = None
            if kind == "draft" and alternative in drafted:
                reason = "an alternative is drafted once"
            elif kind == "draft" and len(drafted) >= limits.max_drafts_per_round:
                reason = "draft limit per round reached"
            elif kind != "draft" and alternative not in drafted:
                reason = f"{kind} without a draft"
            elif kind == "critique" and counts[(alternative, "critique")] >= \
                    limits.max_critiques_per_draft:
                reason = "critique limit per draft reached"
            elif kind == "revision" and counts[(alternative, "revision")] >= \
                    limits.max_revisions_per_draft:
                reason = "revision limit per draft reached"
            elif kind == "recheck" and not ((amendment or {}).get("parameters") or {}).get(
                    "grounding"):
                reason = "a re-check is registered only by amendment 6b (grounding)"
            elif kind == "recheck" and not counts[(alternative, "revision")]:
                reason = "recheck without a revision"
            elif kind == "recheck" and counts[(alternative, "recheck")] >= \
                    MAX_RECHECKS_PER_DRAFT:
                reason = "recheck limit per draft reached"
            event = {"event": "authoring_call", "round": round_number,
                     "alternative": alternative, "kind": kind, "raw": raw}
            reserve = 0
            if reason is None and kind == "draft" and amendment is not None:
                reserve = required_slots(limits) - reserved_for(rows, round_number)
                starts = [r for r in rows if r.get("event") == "round_started"]
                if reserve > 0 and starts and reserved_total(rows) + reserve > \
                        lifetime_caps(rows)["max_total_proposals"]:
                    reason = "lifetime proposal cap reached"
            if reason:
                self._append(handle, {**event, "event": "authoring_refused",
                                      "reason": reason})
                raise ValueError(reason)
            if reserve > 0:
                self._append(handle, {"event": "slots_reserved", "round": round_number,
                                      "slots": reserve})
            self._append(handle, event)

    def record_format_repair(self, round_number: int, alternative: int, raw: str) -> None:
        """Log one format-repair call (``event: format_repair``, phase input):
        refused unless the alternative was drafted, not yet repaired and not yet
        critiqued. It is part of the draft's slot and reserves nothing."""
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            rows = self._rows(handle)
            mine = [r for r in rows if r.get("round") == round_number
                    and r.get("alternative") == alternative]
            kinds = {r.get("kind") for r in mine if r.get("event") == "authoring_call"}
            repaired = any(r.get("event") == "format_repair" and r.get("phase") == "input"
                           for r in mine)
            reason = ("format repair without a draft" if "draft" not in kinds
                      else "an alternative is format-repaired at most once" if repaired
                      else "format repair after critique" if kinds - {"draft"} else None)
            event = {"event": "format_repair", "phase": "input", "round": round_number,
                     "alternative": alternative, "raw": raw}
            if reason:
                self._append(handle, {**event, "event": "format_repair_refused",
                                      "reason": reason})
                raise ValueError(reason)
            self._append(handle, event)

    def reserve_gate_candidate(self, round_number: int, identity: dict[str, Any]) -> dict:
        """Consume one amended gate-candidate slot before the DEV candidate replay."""
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            rows = self._rows(handle)
            amendment = self._amendment(rows)
            if amendment is None:
                raise ValueError("gate-candidate reservations need a protocol amendment")
            limits = AmendmentLimits.model_validate(amendment["limits"])
            used = [r for r in self._after(rows, self._chain_start(rows))
                    if r.get("event") == "gate_candidate_reserved"]
            reason = None
            if any(r.get("round") == round_number for r in used):
                reason = "this round already reserved its gate candidate"
            elif len(used) >= limits.max_gate_candidates:
                reason = "amended gate-candidate limit reached"
            if reason:
                self._append(handle, {"event": "gate_candidate_refused",
                                      "round": round_number, "reason": reason})
                raise ValueError(reason)
            event = {"event": "gate_candidate_reserved", "round": round_number,
                     "amendment_id": amendment["amendment_id"], "number": len(used) + 1,
                     "identity": identity}
            self._append(handle, event)
            return event

    def register_screen(self, *, tasks: list[str], selection: dict[str, Any],
                        source: dict[str, Any]) -> dict:
        """Register the screen task list once per amendment, before any screening."""
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            rows = self._rows(handle)
            amendment = self._amendment(rows)
            if amendment is None:
                raise ValueError("screen registration needs a protocol amendment")
            existing = [r for r in self._after(rows, self._chain_start(rows))
                        if r.get("event") == "screen_registered"
                        and digest((r.get("selection") or {}).get("rule"))
                        == amendment["screen_rule_sha256"]]
            if existing:
                raise ValueError("the screen task list is already registered for "
                                 f"{amendment['amendment_id']}")
            if digest(selection.get("rule")) != amendment["screen_rule_sha256"]:
                raise ValueError("screen selection rule differs from the amendment's")
            event = {"event": "screen_registered", "amendment_id": amendment["amendment_id"],
                     "tasks": list(tasks), "tasks_sha256": digest(list(tasks)),
                     "selection": selection, "source": source}
            self._append(handle, event)
            return event

    def registered_screen(self) -> dict[str, Any] | None:
        """The screen registered under the linked amendment chain, if the latest
        amendment keeps its rule (a changed rule needs a new registration)."""
        rows = self.rows()
        amendment = self._amendment(rows)
        if amendment is None:
            return None
        found = [r for r in self._after(rows, self._chain_start(rows))
                 if r.get("event") == "screen_registered"
                 and digest((r.get("selection") or {}).get("rule"))
                 == amendment["screen_rule_sha256"]]
        return found[0] if found else None

    def proposal(self, round_number: int, raw: str, parsed: str | None = None) -> None:
        """Log a writer output; ``parsed`` is the unfenced text its notes come from."""
        self.append({"event": "proposal_output", "round": round_number, "raw": raw})
        try:
            data = json.loads(raw if parsed is None else parsed)
        except ValueError:
            return  # malformed responses still consume the reserved batch slots
        if isinstance(data, dict) and isinstance(data.get("notes"), list):
            for index, note in enumerate(data["notes"]):
                self.append({"event": "candidate_note", "round": round_number,
                             "index": index, "note": note})
