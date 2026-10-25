"""One cycle of the V1 loop, the continuous driver, and the fail-closed rules around them.

Cycle ``b`` (docs/system/v1-design.md s. 2):

0. refuse while a system halt is latched; ``cycle_start``; invariants (audit chain, pinned
   adapter identity, guard modules and source hashes computed here, floor, canary conformance);
1. monitor (root always; root vs incumbent when they differ) -> rollback;
2. evidence (incumbent) -> trace store; 3. propose (metered; evidence only; policy, size and
   kind screens; pool cap with ``deferred`` records); 4. screen (F', carried state persisted
   in the screen record); 5. confirm: ``confirm_open`` with k BEFORE dispatch, then one
   ``confirm_close`` record that is also the adoption pointer (atomic adoption).

Every dispatch: per-cycle cap check and a ``SharedLedger`` reservation of the worst case
before, settlement after (a failed dispatch is charged in full); outcomes must biject with
requests and echo the applied state and guard chain; incidents are processed after every
dispatch of every role (root: latched system halt; incumbent: reset to root and quarantine
its adopted candidates; candidate: quarantine it); either way the cycle halts.
Restart: ``Harness(...)`` on the same directory reconciles open confirmations (abandoned: k is
never reused) and open cycles, charges unsettled reservations in full, and ``run()`` resumes at
the next batch.

Owner decisions 2026-10-23 (docs/system/v1-design.md s. 12):

* **Critical events** (``Outcome.critical_harm``, an unauthorised write or action) under a
  candidate configuration in screen or confirm are incidents (quarantine, cycle halts).
  ``critical_quarantine``: ``paired`` (default; an event counts only when the same unit under
  the current setup, i.e. the screen anchor or the confirmation's incumbent arm, had none) or
  ``literal`` (any event). The literal mode quarantines almost surely once the base rate is
  above zero (n runs at base rate p give an event with probability 1 - (1 - p)^n).
* **Production mode** (``production=True``): adoption needs a passing CONVERSATION-level
  confirmation (its own k, after the decision-level one passes); a cycle refuses to run while
  the current version has none (``certify_current`` runs it for a version adopted in
  development); rollbacks go to the nearest certified ancestor.
* **Paid adapters** (``AdapterInfo.paid``) are refused without an owner-approved item cap
  (``item_cap_usd``); the ledger cap is min(lifetime cap, item cap).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slh.budget import BudgetExceeded
from slh.ledger import SharedLedger
from slh.state import HarnessState
from slh.system import gate as G
from slh.system import guards as GD
from slh.system import replicate as RP
from slh.system.interfaces import (
    MEMORY_DELIMITER,
    STATUSES,
    AdapterBudgetStop,
    AgentAdapter,
    Candidate,
    Outcome,
    Proposer,
    RunRequest,
    Unit,
    apply_all,
    target_conflicts,
)
from slh.system.stores import (
    AuditLog,
    JsonlLog,
    SystemLock,
    TraceStore,
    UnitRegistry,
    VersionStore,
    now_utc,
)


class BudgetRefused(RuntimeError):
    pass


class InvariantBroken(RuntimeError):
    pass


class SystemHalted(RuntimeError):
    pass


class CycleHalted(RuntimeError):
    pass


class ConfirmDeferred(BudgetRefused):
    """The confirmation would not fit the remaining serving time (``Harness.confirm_guard``):
    nothing opened (no k spent), the advanced candidates are kept as the carry's
    ``pending_confirm``, the run stops cleanly, and the next cycle runs that confirmation
    first (no new screen look) once it fits."""


class ProductionRefused(RuntimeError):
    pass


CRITICAL_MODES = ("paired", "literal", "attributed")
LEVELS = ("decision", "conversation")

# Safety-bearing settings: pinned in ``start``, logged in every ``session``, changeable only
# through an owner ``config_amend`` record (``amend_safety_config``; CLI ``amend-config``).
# ``production`` may be switched on freely (stricter); switching it off needs an amendment.
SAFETY_KEYS = (
    "confirm.alpha", "confirm.margin_success", "confirm.margin_harm", "confirm.margin_error",
    "confirm.max_unscored_rate", "confirm.require_nonnegative_gain",
    "confirm.critical_block_on_increase", "confirm.critical_detect_level",
    "confirm.critical_margin", "monitor.alpha", "monitor.window_units", "monitor.min_units",
    "monitor.critical", "critical_quarantine", "guard_floor", "production",
    "item_cap_usd", "lifetime_cap_usd", "monitor_level", "evidence_units",
    "critical_attribution_alpha", "critical_attribution_min_base",
    "hold_replicates", "hold_replicate_rule",
    "critical_attribution_base_rule", "critical_attribution_min_base_rel",
    "critical_attribution_min_base_floor", "screen_hold_review",
)
# values for systems started before a key existed, where the default would change their
# behaviour: replicate-on-discordance (2026-10-24) was off; the hard trigger's baseline rule
# was absolute and a screen hard hold retired at look exhaustion (before 2026-10-25)
LEGACY_SAFETY = {"hold_replicates": 0, "critical_attribution_base_rule": "absolute",
                 "screen_hold_review": False}
# keys that act only in the attributed screen: a difference is not a resume refusal while
# neither the pinned nor the requested mode is ``attributed`` (switching to attributed is an
# amendment, and the pinned values then apply)
ATTRIBUTED_ONLY = ("critical_attribution_base_rule", "critical_attribution_min_base_rel",
                   "critical_attribution_min_base_floor", "screen_hold_review")


def _flat(config: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key in SAFETY_KEYS:
        v: Any = config
        for part in key.split("."):
            v = v[part]
        out[key] = v
    return out


def safety_config(cfg: SystemConfig) -> dict[str, Any]:
    return _json(_flat(asdict(cfg)))


def pinned_safety(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The safety settings in force: ``start`` (its config), updated by owner amendments;
    ``production`` stays on once a session ran with it (until an amendment turns it off)."""
    pinned: dict[str, Any] = {}
    config: dict[str, Any] = {}
    for r in rows:
        if r["kind"] == "start":
            config = r.get("config") or {}
            pinned = dict(r.get("safety") or {})
        elif r["kind"] == "session" and r.get("production"):
            pinned["production"] = True
        elif r["kind"] == "config_amend":
            pinned = dict(r["safety"])
    return _migrate(pinned, config) if rows else pinned


def _migrate(pinned: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Keys added to SAFETY_KEYS after a system started: taken from its start config when
    recorded there, else the default of the version that added them (SystemConfig), so a
    legacy resume with unchanged settings (e.g. ``paired``) is not rejected."""
    defaults = _json(asdict(SystemConfig()))
    out = dict(pinned)
    for key in SAFETY_KEYS:
        if key in out:
            continue
        if key in LEGACY_SAFETY and key not in config:
            out[key] = LEGACY_SAFETY[key]
            continue
        for source in (config, defaults):
            v: Any = source
            try:
                for part in key.split("."):
                    v = v[part]
            except (KeyError, TypeError):
                continue
            out[key] = _json(v)
            break
    return out


def safety_diff(pinned: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    diff = {k: {"pinned": pinned.get(k), "requested": current.get(k)} for k in SAFETY_KEYS
            if pinned.get(k) != current.get(k)}
    if "attributed" not in (pinned.get("critical_quarantine"),
                            current.get("critical_quarantine")):
        for k in ATTRIBUTED_ONLY:
            diff.pop(k, None)
    if diff.get("production") == {"pinned": False, "requested": True}:
        diff.pop("production")  # switching production on only tightens
    return diff


def amend_safety_config(root: Path, changes: dict[str, Any], owner_record: str,
                        by: str) -> dict[str, Any]:
    """Owner action: change pinned safety settings (under the writer lock). ``owner_record``
    references the owner's decision (message, document); it is required."""
    if not owner_record.strip() or not by.strip():
        raise ValueError("an owner record and the owner's name are required")
    bad = sorted(set(changes) - set(SAFETY_KEYS))
    if bad:
        raise ValueError(f"not safety settings: {bad}")
    root = Path(root)
    lock = SystemLock(root)
    try:
        audit = AuditLog(root)
        if not audit.rows("start"):
            raise ValueError("no system at this directory")
        before = pinned_safety(audit.rows())
        after = {**before, **_json(changes)}
        return audit.append("config_amend", by=by, owner_record=owner_record,
                            changes=_json(changes), before=before, safety=after)
    finally:
        lock.release()


# ---------------------------------------------------------------- human review of HOLDs
# Owner 2026-10-24: "Human review only". A confirmation HOLD (s. 15, s. 20) clears only by an
# audited human decision; replication is evidence in the packet and never releases.
# Final review of b7a4ae7 (s. 20 "Review integrity"): the audit chain is verified before any
# decision; an approval is bound to the packet's candidate version and pass status; a
# decision interrupted by a crash is completed (every precondition re-checked) or voided
# (re-decidable) by ``reconcile_reviews``; a clearance not made by a human (the legacy
# ``replicated_release``) is voided and any adoption resting on one is rolled back.

DECISIONS = {"approve": "approve", "release": "approve", "reject": "reject",
             "quarantine": "reject"}
RELEASING = ("approve", "release")  # "release": records written before 2026-10-24
NOT_A_REVIEWER = ("", "replication")  # ``by`` of the legacy automatic release
PACKET_BINDING = ("k", "candidates", "candidate_version", "incumbent",
                  "confirmation_passed_own_tests")


class ReviewRefused(ValueError):
    pass


def _is_human(r: dict[str, Any]) -> bool:
    return r.get("decision") in DECISIONS and \
        str(r.get("by") or "").strip() not in NOT_A_REVIEWER


def human_decisions(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """k -> the human decision in force on hold k: a ``hold_adjudication`` made by a human
    and not voided (``adjudication_void``). Anything else leaves the hold pending."""
    void = {r["adjudication_seq"] for r in rows if r["kind"] == "adjudication_void"}
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        if r["kind"] == "hold_adjudication" and r["seq"] not in void and _is_human(r):
            out.setdefault(r["k"], r)
    return out


def held_candidates(rows: list[dict[str, Any]]) -> set[str]:
    done = set(human_decisions(rows))
    return {c for r in rows if r["kind"] == "critical_hold" and r["k"] not in done
            for c in r["candidates"]}


def released_candidates(rows: list[dict[str, Any]]) -> set[str]:
    """Candidates a human APPROVED, minus any a human ever REJECTED (a rejection removes an
    earlier approval's exemption from retirement)."""
    dec = list(human_decisions(rows).values())
    rejected = {c for r in dec if DECISIONS[r["decision"]] == "reject" for c in r["candidates"]}
    return {c for r in dec if DECISIONS[r["decision"]] == "approve"
            for c in r["candidates"]} - rejected


def current_packet(rows: list[dict[str, Any]], k: int) -> dict[str, Any] | None:
    """The latest review packet of hold ``k`` (``critical_hold``, then ``hold_packet``)."""
    out = None
    for r in rows:
        if r["kind"] in ("critical_hold", "hold_packet") and r["k"] == k:
            out = {"packet": r["packet"], "packet_sha256": r["packet_sha256"]}
    return out


def _pointer(rows: list[dict[str, Any]]) -> str:
    for r in reversed(rows):
        if "pointer" in r and r["kind"] in ("start", "confirm_close", "rollback"):
            return r["pointer"]
    raise InvariantBroken("no pointer")


def _gone(rows: list[dict[str, Any]]) -> set[str]:
    return {r["cid"] for r in rows if r["kind"] in
            ("retire", "policy_drop", "size_drop", "kind_drop")}


def _halt(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    last = None
    for r in rows:
        if r["kind"] in ("system_halt", "halt_ack"):
            last = r
    return last if last is not None and last["kind"] == "system_halt" else None


def _opened(rows: list[dict[str, Any]], k: int) -> dict[str, Any] | None:
    return next((r for r in rows if r["kind"] == "confirm_open" and r["k"] == k), None)


def _adoptable(rows: list[dict[str, Any]], hold: dict[str, Any]) -> str | None:
    """None when an approved hold may adopt its confirmation's version now; else why not
    (the approval then only makes the candidates re-proposable)."""
    k = hold["k"]
    if hold.get("passed") is not True:
        return "the confirmation did not pass its own tests"
    opened = _opened(rows, k)
    voided = unreviewed_adoptions(rows)  # a legacy automatic adoption does not count
    closed = [r for r in rows if r["kind"] == "confirm_close" and r["k"] == k
              and r["seq"] not in voided]
    if opened is None or len(closed) != 1 or closed[0].get("void"):
        return "the confirmation record is incomplete or void"
    if opened.get("purpose") != "adopt":
        return "not an adoption confirmation"
    if _pointer(rows) != opened["incumbent"]:
        return "the pointer moved since the confirmation (its incumbent is not current)"
    level = opened.get("level", "decision")
    if pinned_safety(rows).get("production") and level != "conversation":
        return "production: adoption needs a conversation-level confirmation"
    if set(hold["candidates"]) & _gone(rows):
        return "a candidate was retired or dropped"
    return None


def _packet_mismatch(root: Path, cur: dict[str, Any], hold: dict[str, Any],
                     opened: dict[str, Any] | None) -> str | None:
    """None when the packet file matches its recorded hash AND carries exactly the audited
    confirmation's k, candidates, candidate version, incumbent and pass status."""
    path = Path(root) / "reviews" / cur["packet"]
    try:
        data = path.read_bytes()
        packet = json.loads(data)
    except (OSError, ValueError):
        return f"packet {cur['packet']} on disk is missing or unreadable"
    if hashlib.sha256(data).hexdigest() != cur["packet_sha256"]:
        return (f"packet {cur['packet']} on disk does not match its recorded hash (changed "
                "or missing)")
    if opened is None:
        return f"no confirm_open for k={hold['k']}"
    want = {"k": hold["k"], "candidates": hold["candidates"],
            "candidate_version": opened.get("candidate_version"),
            "incumbent": opened.get("incumbent"),
            "confirmation_passed_own_tests": hold.get("passed")}
    bad = [f for f in PACKET_BINDING if not isinstance(packet, dict) or f not in packet
           or packet[f] != want[f]]
    if bad:
        return f"the reviewed packet does not match the audited confirmation on {bad}"
    return None


def _verified(audit: AuditLog) -> None:
    try:
        audit.verify()
    except ValueError as e:
        raise ReviewRefused(f"audit chain fails verification ({e}): refusing any review "
                            "decision") from e


def _adopt_reviewed(audit: AuditLog, hold: dict[str, Any], rec: dict[str, Any],
                    **extra: Any) -> dict[str, Any]:
    rows = audit.rows()
    k, by = hold["k"], rec["by"]
    opened = _opened(rows, k)
    assert opened is not None
    closed = next(r for r in rows if r["kind"] == "confirm_close" and r["k"] == k)  # 1st
    result = {**(closed.get("result") or {}), "adopt": True, "hold": False,
              "approved_by": by}
    result["reasons"] = list(result.get("reasons") or []) + [
        f"HOLD approved by human review ({by}; adjudication seq {rec['seq']})"]
    return audit.append("confirm_close", batch=hold["batch"], k=k,
                        level=opened.get("level", "decision"), adopt=True, passed=True,
                        result=result, candidates=hold["candidates"],
                        version=opened["candidate_version"],
                        pointer=opened["candidate_version"], parent=opened["incumbent"],
                        via="human_review", reviewer=by, adjudication_seq=rec["seq"],
                        **extra)


def _recheck_approval(root: Path, rows: list[dict[str, Any]],
                      rec: dict[str, Any]) -> str | None:
    """Every precondition of an approve-and-adopt, re-checked when completing it later."""
    hold = next((r for r in rows if r["kind"] == "critical_hold" and r["k"] == rec["k"]),
                None)
    if hold is None:
        return "no critical hold"
    halt = _halt(rows)
    if halt is not None:
        return f"system halt latched ({halt['reason']})"
    cur = current_packet(rows, rec["k"]) or {}
    if (cur.get("packet"), cur.get("packet_sha256")) != (rec.get("packet"),
                                                         rec.get("packet_sha256")):
        return "the current packet is not the one approved"
    return _packet_mismatch(root, cur, hold, _opened(rows, rec["k"])) or \
        _adoptable(rows, hold)


def _certified(rows: list[dict[str, Any]], version: str) -> bool:
    return version == rows[0]["pointer"] or any(
        r["kind"] == "confirm_close" and r.get("level") == "conversation" and r.get("passed")
        and r.get("version") == version for r in rows)


def _adoption_of(rows: list[dict[str, Any]], version: str) -> dict[str, Any] | None:
    for r in reversed(rows):
        if r["kind"] == "confirm_close" and r.get("pointer") == version:
            return r
    return None


def unreviewed_adoptions(rows: list[dict[str, Any]]) -> set[int]:
    """Seqs of adoptions (``confirm_close`` with ``adopt``) of a HELD confirmation that do
    not rest on a human approval in force (e.g. the legacy automatic release)."""
    held_ks = {r["k"] for r in rows if r["kind"] == "critical_hold"}
    ok = {r["seq"] for r in human_decisions(rows).values()
          if DECISIONS[r["decision"]] == "approve"}
    return {r["seq"] for r in rows if r["kind"] == "confirm_close" and r.get("adopt")
            and r.get("k") in held_ks
            and not (r.get("via") == "human_review" and r.get("adjudication_seq") in ok)}


def reconcile_reviews(audit: AuditLog) -> list[dict[str, Any]]:
    """Writer only (lock held), on a verified chain. Idempotent. In order:

    1. a ``hold_adjudication`` not made by a human (legacy ``replicated_release``) is voided:
       its hold is pending human review again;
    2. an interrupted human decision is finished: a reject's candidates are retired; an
       approve whose adoption did not land adopts if EVERY precondition still holds
       (``_recheck_approval``), else the decision is voided (re-decidable);
    3. a pointer whose lineage contains an adoption without a human approval in force is
       rolled back below the earliest such adoption (production: to a certified ancestor).
    Returns the records written."""
    root = audit.path.parent
    out: list[dict[str, Any]] = []
    rows = audit.rows()
    void = {r["adjudication_seq"] for r in rows if r["kind"] == "adjudication_void"}
    for r in rows:
        if r["kind"] == "hold_adjudication" and r["seq"] not in void and not _is_human(r):
            out.append(audit.append(
                "adjudication_void", adjudication_seq=r["seq"], k=r["k"],
                reason=f"not a human decision ({r.get('decision')} by {r.get('by')!r}); the "
                       "hold awaits human review (owner 2026-10-24)"))
    rows = audit.rows()
    gone = _gone(rows)
    adopted_by = {r.get("adjudication_seq") for r in rows if r["kind"] == "confirm_close"
                  and r.get("via") == "human_review"}
    for k, r in sorted(human_decisions(rows).items()):
        if DECISIONS[r["decision"]] == "reject":
            for cid in r["candidates"]:
                if cid not in gone:
                    hb = next(h["batch"] for h in rows
                              if h["kind"] == "critical_hold" and h["k"] == k)
                    out.append(audit.append("retire", batch=hb, cid=cid, stage="quarantine",
                                            evidence={"hold_k": k, "adjudicated_by": r["by"],
                                                      "completed": "reconcile"}))
                    gone.add(cid)
        elif r.get("adopt") and r["seq"] not in adopted_by:
            why = _recheck_approval(root, audit.rows(), r)
            if why is None:
                hold = next(h for h in audit.rows()
                            if h["kind"] == "critical_hold" and h["k"] == k)
                out.append(_adopt_reviewed(audit, hold, r, completed="reconcile"))
            else:
                out.append(audit.append(
                    "adjudication_void", adjudication_seq=r["seq"], k=k,
                    reason=f"interrupted approval not completed: {why}; re-decide"))
    out += _reconcile_screen_reviews(audit)
    rows = audit.rows()
    bad = unreviewed_adoptions(rows)
    cur_v = _pointer(rows)
    v, lineage, seen = cur_v, [], set()
    while v != rows[0]["pointer"] and v not in seen:
        seen.add(v)
        rec = _adoption_of(rows, v)
        if rec is None:
            break
        lineage.append(rec)
        v = rec["parent"]
    tainted = [i for i, rec in enumerate(lineage) if rec["seq"] in bad]
    if tainted:
        target = lineage[tainted[-1]]["parent"]
        if pinned_safety(rows).get("production"):
            while not _certified(rows, target):
                rec = _adoption_of(rows, target)
                target = rec["parent"] if rec is not None else rows[0]["pointer"]
        batch = max([r["batch"] for r in rows if r["kind"] == "cycle_start"], default=-1)
        out.append(audit.append(
            "rollback", batch=batch, pointer=target, parent=cur_v,
            reason="adoption rested on a HOLD clearance not made by a human (legacy "
                   "replicated_release): reverted to the last human- or gate-adopted version",
            unreviewed=sorted(lineage[i]["seq"] for i in tainted)))
    return out


def _review_decide(root: Path, audit: AuditLog, k: int, decision: str, by: str, note: str,
                   packet_sha256: str | None) -> dict[str, Any]:
    """Caller holds the writer lock. The audit chain must verify (else refused); an
    interrupted earlier decision is reconciled first. ``approve`` needs ``packet_sha256``
    equal to the current packet's recorded hash AND to the file on disk, and the packet
    must carry the audited confirmation's candidate version, incumbent and pass status; an
    approved hold adopts the confirmation's version when ``_adoptable``, else releases the
    candidates (re-proposable, a NEW confirmation decides). ``reject`` retires them BEFORE
    its record (a crash leaves them retired). All in the audit hash chain."""
    if decision not in DECISIONS:
        raise ReviewRefused(f"decision is one of {sorted(DECISIONS)}")
    decision = DECISIONS[decision]
    if not str(by).strip() or not str(note).strip():
        raise ReviewRefused("a reviewer name and a note are required")
    if str(by).strip() in NOT_A_REVIEWER:
        raise ReviewRefused(f"{by!r} is not a human reviewer")
    _verified(audit)
    reconcile_reviews(audit)
    rows = audit.rows()
    hold = next((r for r in rows if r["kind"] == "critical_hold" and r["k"] == k), None)
    if hold is None:
        raise ReviewRefused(f"no critical hold for confirmation k={k}")
    if k in human_decisions(rows):
        raise ReviewRefused(f"hold k={k} already decided")
    cur = current_packet(rows, k)
    assert cur is not None
    path = Path(root) / "reviews" / cur["packet"]
    on_disk = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    if decision == "approve" or packet_sha256 is not None:
        if packet_sha256 != cur["packet_sha256"]:
            raise ReviewRefused(f"packet hash mismatch: the current packet of k={k} is "
                                f"{cur['packet']} sha256 {cur['packet_sha256']}; review it")
        if on_disk != cur["packet_sha256"]:
            raise ReviewRefused(f"packet {cur['packet']} on disk does not match its recorded "
                                "hash (changed or missing): refusing")
    opened = _opened(rows, k)
    if decision == "approve":
        halt = _halt(rows)
        if halt is not None:
            raise ReviewRefused(f"system halt latched ({halt['reason']}): ack it first")
        bad = _packet_mismatch(root, cur, hold, opened)
        if bad:
            raise ReviewRefused(f"{bad}: refusing")
    why = _adoptable(rows, hold) if decision == "approve" else None
    adopt = decision == "approve" and why is None
    if decision == "reject":
        gone = _gone(rows)
        for cid in hold["candidates"]:
            if cid not in gone:
                audit.append("retire", batch=hold["batch"], cid=cid, stage="quarantine",
                             evidence={"hold_k": k, "adjudicated_by": by})
    rec = audit.append("hold_adjudication", k=k, decision=decision, by=by, note=note,
                       candidates=hold["candidates"], packet=cur["packet"],
                       packet_sha256=cur["packet_sha256"], adopt=adopt,
                       candidate_version=(opened or {}).get("candidate_version"),
                       passed=hold.get("passed"),
                       not_adopted=why if decision == "approve" else None)
    if adopt:
        _adopt_reviewed(audit, hold, rec)
    return rec


# ----------------------------------------------------------------- screen-hold review
# (2026-10-25; owner: every hold ends in an audited human decision, never a silent
# auto-release). A candidate HELD at the attributed screen that would advance (or whose
# look budget is spent) is not advanced and not retired: ``Harness._screen_hold`` writes a
# review packet ``reviews/screen-hold-s<h>.json`` and a ``screen_hold`` record (h allocated
# from the records). A HUMAN decides with ``review_decide`` (target ``s<h>`` or a held
# candidate id): ``approve`` (packet hash bound) lets the candidates go to a confirmation
# in the next cycle, where the confirmation's own tests and its HOLD still apply;
# ``reject`` retires them (retire records BEFORE the decision record). No automatic
# release, ever; offline runs export the packets and the candidates stay where they are.

SCREEN_PACKET_BINDING = ("h", "batch", "candidates", "incumbent")


def screen_decisions(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """h -> the human decision in force on screen hold h (``screen_hold_adjudication`` by a
    human, not voided). Anything else leaves the hold pending."""
    void = {r["adjudication_seq"] for r in rows if r["kind"] == "adjudication_void"}
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        if r["kind"] == "screen_hold_adjudication" and r["seq"] not in void and _is_human(r):
            out.setdefault(r["h"], r)
    return out


def pending_screen_holds(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    done = set(screen_decisions(rows))
    return {r["h"]: r for r in rows if r["kind"] == "screen_hold" and r["h"] not in done}


def screen_held_candidates(rows: list[dict[str, Any]]) -> set[str]:
    """Candidates of screen holds awaiting a human decision."""
    return {c for r in pending_screen_holds(rows).values() for c in r["candidates"]}


def approved_screen_holds(rows: list[dict[str, Any]]
                          ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(hold, approval) of every APPROVED screen hold whose candidates have not yet entered
    a confirmation (no ``confirm_open`` after the approval names any of them)."""
    dec = screen_decisions(rows)
    out = []
    for r in rows:
        if r["kind"] != "screen_hold" or r["h"] not in dec:
            continue
        rec = dec[r["h"]]
        if DECISIONS[rec["decision"]] != "approve":
            continue
        used = any(o["kind"] == "confirm_open" and o["seq"] > rec["seq"]
                   and set(o["candidates"]) & set(r["candidates"]) for o in rows)
        if not used:
            out.append((r, rec))
    return out


def current_screen_packet(rows: list[dict[str, Any]], h: int) -> dict[str, Any] | None:
    out = None
    for r in rows:
        if r["kind"] == "screen_hold" and r["h"] == h:
            out = {"packet": r["packet"], "packet_sha256": r["packet_sha256"]}
    return out


def _screen_packet_mismatch(root: Path, cur: dict[str, Any], hold: dict[str, Any]
                            ) -> str | None:
    """None when the packet file matches its recorded hash AND carries exactly the audited
    hold's h, batch, candidates and incumbent."""
    path = Path(root) / "reviews" / cur["packet"]
    try:
        data = path.read_bytes()
        packet = json.loads(data)
    except (OSError, ValueError):
        return f"packet {cur['packet']} on disk is missing or unreadable"
    if hashlib.sha256(data).hexdigest() != cur["packet_sha256"]:
        return (f"packet {cur['packet']} on disk does not match its recorded hash (changed "
                "or missing)")
    bad = [f for f in SCREEN_PACKET_BINDING if not isinstance(packet, dict)
           or f not in packet or packet[f] != hold[f]]
    if bad:
        return f"the reviewed packet does not match the audited screen hold on {bad}"
    return None


def _reconcile_screen_reviews(audit: AuditLog) -> list[dict[str, Any]]:
    """Part of ``reconcile_reviews``: a ``screen_hold_adjudication`` not made by a human is
    voided (the hold is pending again); an interrupted reject's retirements are completed.
    An approval needs no completion (the next cycle confirms from the audit records)."""
    out: list[dict[str, Any]] = []
    rows = audit.rows()
    void = {r["adjudication_seq"] for r in rows if r["kind"] == "adjudication_void"}
    for r in rows:
        if r["kind"] == "screen_hold_adjudication" and r["seq"] not in void \
                and not _is_human(r):
            out.append(audit.append(
                "adjudication_void", adjudication_seq=r["seq"], h=r["h"],
                reason=f"not a human decision ({r.get('decision')} by {r.get('by')!r}); the "
                       "screen hold awaits human review"))
    rows = audit.rows()
    gone = _gone(rows)
    for h, r in sorted(screen_decisions(rows).items()):
        if DECISIONS[r["decision"]] != "reject":
            continue
        hold = next(x for x in rows if x["kind"] == "screen_hold" and x["h"] == h)
        for cid in hold["candidates"]:
            if cid not in gone:
                out.append(audit.append("retire", batch=hold["batch"], cid=cid,
                                        stage="quarantine",
                                        evidence={"screen_hold": h, "adjudicated_by": r["by"],
                                                  "completed": "reconcile"}))
                gone.add(cid)
    return out


def _screen_decide(root: Path, audit: AuditLog, h: int, decision: str, by: str, note: str,
                   packet_sha256: str | None) -> dict[str, Any]:
    """Caller holds the writer lock. As ``_review_decide`` for a SCREEN hold: the chain must
    verify; ``approve`` needs the current packet hash (recorded = on disk), the packet bound
    to the audited hold, no latched halt, the pointer still the hold's incumbent and no
    candidate retired or dropped; it ADVANCES the candidates to a confirmation in the next
    cycle (never an adoption by itself). ``reject`` retires them BEFORE its record."""
    if decision not in DECISIONS:
        raise ReviewRefused(f"decision is one of {sorted(DECISIONS)}")
    decision = DECISIONS[decision]
    if not str(by).strip() or not str(note).strip():
        raise ReviewRefused("a reviewer name and a note are required")
    if str(by).strip() in NOT_A_REVIEWER:
        raise ReviewRefused(f"{by!r} is not a human reviewer")
    _verified(audit)
    reconcile_reviews(audit)
    rows = audit.rows()
    hold = next((r for r in rows if r["kind"] == "screen_hold" and r["h"] == h), None)
    if hold is None:
        raise ReviewRefused(f"no screen hold s{h}")
    if h in screen_decisions(rows):
        raise ReviewRefused(f"screen hold s{h} already decided")
    cur = current_screen_packet(rows, h)
    assert cur is not None
    path = Path(root) / "reviews" / cur["packet"]
    on_disk = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    if decision == "approve" or packet_sha256 is not None:
        if packet_sha256 != cur["packet_sha256"]:
            raise ReviewRefused(f"packet hash mismatch: the current packet of s{h} is "
                                f"{cur['packet']} sha256 {cur['packet_sha256']}; review it")
        if on_disk != cur["packet_sha256"]:
            raise ReviewRefused(f"packet {cur['packet']} on disk does not match its recorded "
                                "hash (changed or missing): refusing")
    gone = _gone(rows)
    if decision == "approve":
        halt = _halt(rows)
        if halt is not None:
            raise ReviewRefused(f"system halt latched ({halt['reason']}): ack it first")
        bad = _screen_packet_mismatch(root, cur, hold)
        if bad:
            raise ReviewRefused(f"{bad}: refusing")
        if _pointer(rows) != hold["incumbent"]:
            raise ReviewRefused("the pointer moved since the screen hold (its incumbent is not "
                                "current): the screen evidence no longer applies; reject it")
        if set(hold["candidates"]) & gone:
            raise ReviewRefused("a candidate of the hold was retired or dropped")
    if decision == "reject":
        for cid in hold["candidates"]:
            if cid not in gone:
                audit.append("retire", batch=hold["batch"], cid=cid, stage="quarantine",
                             evidence={"screen_hold": h, "adjudicated_by": by})
    return audit.append("screen_hold_adjudication", h=h, decision=decision, by=by,
                        note=note, candidates=hold["candidates"], packet=cur["packet"],
                        packet_sha256=cur["packet_sha256"],
                        advance=decision == "approve", incumbent=hold["incumbent"])


def review_decide(root: Path, target: str | int, decision: str, reviewer: str, note: str,
                  packet_sha256: str | None = None) -> dict[str, Any]:
    """CLI ``review-decide``: the human decision on a pending HOLD, under the writer lock.
    ``target``: a held candidate id, or ``k<N>`` / an int for confirmation N."""
    root = Path(root)
    lock = SystemLock(root)
    try:
        audit = AuditLog(root)
        if not audit.rows("start"):
            raise ReviewRefused("no system at this directory")
        _verified(audit)
        reconcile_reviews(audit)  # a voided clearance makes its hold pending again
        rows = audit.rows()
        done = set(human_decisions(rows))
        pending = [r for r in rows if r["kind"] == "critical_hold" and r["k"] not in done]
        if str(target).startswith("s") and str(target)[1:].isdigit():
            return _screen_decide(root, audit, int(str(target)[1:]), decision, reviewer,
                                  note, packet_sha256)
        if isinstance(target, int) or (str(target).startswith("k")
                                       and str(target)[1:].isdigit()):
            k = int(str(target).lstrip("k"))
        else:
            hits = [("k", r["k"]) for r in pending if target in r["candidates"]] + \
                [("s", h) for h, r in pending_screen_holds(rows).items()
                 if target in r["candidates"]]
            if len(hits) != 1:
                raise ReviewRefused(f"candidate {target!r} has {len(hits)} pending holds")
            if hits[0][0] == "s":
                return _screen_decide(root, audit, hits[0][1], decision, reviewer, note,
                                      packet_sha256)
            k = hits[0][1]
        return _review_decide(root, audit, k, decision, reviewer, note, packet_sha256)
    finally:
        lock.release()


@dataclass
class SystemConfig:
    evidence_units: int = 40
    screen: G.ScreenConfig = field(default_factory=G.ScreenConfig)
    confirm: G.ConfirmConfig = field(default_factory=G.ConfirmConfig)
    monitor: G.MonitorConfig = field(default_factory=G.MonitorConfig)
    cycle_cap_usd: float = 10.0
    lifetime_cap_usd: float = 100.0
    guard_floor: tuple[str, ...] = ("slh.guards.identity",)  # tau2 retail: + consent (R)
    max_notes: int = 12
    max_note_chars: int = 500
    max_total_note_chars: int = 4000
    # owner decisions 2026-10-23
    critical_quarantine: str = "paired"  # paired | literal | attributed (s. 12; s. 18)
    critical_attribution_alpha: float = 0.05  # attributed mode: family-wise false quarantine
    critical_attribution_min_base: int = 100  # base runs per context for a "new failure"
    # (2026-10-25) the hard trigger's baseline rule (``slh.system.attribution``): "relative"
    # = a kind observed in the context's baseline is known at any baseline size; a NEW kind
    # needs min(min_base, max(floor, ceil(rel x the candidate's runs in the context)))
    # baseline runs; "absolute" (legacy) = min_base for every classification
    critical_attribution_base_rule: str = "relative"
    critical_attribution_min_base_rel: float = 1.5
    critical_attribution_min_base_floor: int = 30
    # (2026-10-25) a candidate HELD at the attributed screen that would advance (or whose
    # look budget is spent) gets a review packet and waits for a HUMAN decision
    # (``review_decide``: approve -> confirmation, which keeps its own HOLD; reject ->
    # retired). False (legacy): it is carried and retired at look exhaustion.
    screen_hold_review: bool = True
    # replication at the confirmation HOLD (s. 20): R repeated runs of the candidate and
    # every discordant reference per held unit, attached to the review packet as EVIDENCE.
    # Never releases (owner 2026-10-24: human review only). 0 = off (the default).
    hold_replicates: int = 0
    hold_replicate_rule: str = "ref_reproduces"  # legacy pinned key: no effect
    production: bool = False  # adoption needs a conversation-level confirmation
    conversation_units: int = 6000  # units of one conversation-level confirmation (~10x cost
    # per unit; the same n as decision level is needed for the 1-point harm margin)
    item_cap_usd: float | None = None  # owner-approved cap of the current spend item
    item_id: str | None = None  # the approved item (recorded in every session record)
    dispatch_chunk_runs: int = 2000  # runs per reserved dispatch (bounds an overspend)
    # monitor unit level: "decision", or "conversation" (the tau2 episode monitor; owner
    # 2026-10-23: decision-level confirmation in development WITH an episode monitor)
    monitor_level: str = "decision"
    # host learned-state kinds (skill, memory; host-integration-spike s. 4): caps and screens
    max_skills: int = 8
    max_skill_chars: int = 8000  # one rendered SKILL.md
    max_total_skill_chars: int = 32000
    max_skill_description_chars: int = 300
    max_memory_chars: int = 2200  # Hermes' own MEMORY.md limit (memory.memory_char_limit)


def _json(v: Any) -> Any:
    return json.loads(json.dumps(v))


def _ids(state: HarnessState) -> set[str]:
    return {n.get("id") for n in state.notes} | {g.get("id") for g in state.guards
                                                  if g.get("kind") == "checker"}


class Harness:
    def __init__(self, root: Path, adapter: AgentAdapter, proposer: Proposer,
                 cfg: SystemConfig | None = None, initial: HarnessState | None = None,
                 alert: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.cfg = cfg or SystemConfig()
        if self.cfg.critical_quarantine not in CRITICAL_MODES:
            raise ValueError(f"critical_quarantine must be one of {CRITICAL_MODES}")
        if self.cfg.monitor_level not in LEVELS:
            raise ValueError(f"monitor_level must be one of {LEVELS}")
        if self.cfg.critical_attribution_base_rule not in ("absolute", "relative"):
            raise ValueError("critical_attribution_base_rule must be 'absolute' or 'relative'")
        paid = bool(adapter.info().paid)
        metered = self.metered_components(adapter, proposer)
        if metered and self.cfg.item_cap_usd is None:
            raise BudgetRefused(f"metered components {metered}: refusing to start without an "
                                "owner-approved item cap (SystemConfig.item_cap_usd)")
        cap = (min(self.cfg.lifetime_cap_usd, self.cfg.item_cap_usd)
               if self.cfg.item_cap_usd is not None else self.cfg.lifetime_cap_usd)
        self.root = Path(root)
        self.lock = SystemLock(self.root)
        try:
            self._open(adapter, proposer, initial, alert, cap, paid, metered)
        except BaseException:
            self.lock.release()
            raise

    @staticmethod
    def metered_components(adapter: AgentAdapter, proposer: Proposer) -> list[str]:
        """Components that cost money: a paid adapter, or any non-zero cost estimate (the
        adapter's covers its runs and, from M6, checker and GPU-window shares)."""
        out = []
        if adapter.info().paid or max(float(adapter.estimate_cost_usd(1, level=lv))
                                      for lv in LEVELS) > 0:
            out.append("adapter")
        if getattr(proposer, "paid", False) or float(proposer.estimate_cost_usd()) > 0:
            out.append("proposer")
        return out

    def _open(self, adapter: AgentAdapter, proposer: Proposer, initial: HarnessState | None,
              alert: Callable[[dict[str, Any]], None] | None, cap: float, paid: bool,
              metered: list[str]) -> None:
        self.adapter, self.proposer = adapter, proposer
        self.metered = metered
        self.alert_fn = alert
        # Optional: (runs of the confirmation) -> reason to defer it, or None (M3: the
        # serving window's remaining request time at the measured rate).
        self.confirm_guard: Callable[[int], str | None] | None = None
        self._in_play: list[Candidate] = []  # candidates of the dispatch under way
        self.audit = AuditLog(self.root)
        self.traces = TraceStore(self.root)
        self.units = UnitRegistry(self.root)
        self.versions = VersionStore(self.root)
        self.alerts = JsonlLog(self.root / "alerts.jsonl")
        session = len(self.audit.rows("session")) + 1
        resuming = bool(self.audit.rows("start"))
        try:
            # Resuming requires the existing ledger: a missing one is never recreated.
            self.ledger = SharedLedger(self.root / "ledger.jsonl", cap,
                                       owner="slh-system", session=session,
                                       quarantine=self.root / "ledger.torn", wait_s=0.0,
                                       create=not resuming)
        except RuntimeError as e:
            raise InvariantBroken(f"resume: {e}") from e
        if resuming:
            recorded = max([float(r.get("ledger_charged_usd", 0.0)) for r in self.audit.rows()
                            if "ledger_charged_usd" in r], default=0.0)
            have = float(self.ledger.snapshot()["charged_usd"])
            if have + 1e-9 < recorded:
                raise InvariantBroken(f"resume: ledger holds ${have:.6f} charged but the audit "
                                      f"log recorded ${recorded:.6f}; refusing (spend history "
                                      "lost or replaced)")
            diff = safety_diff(pinned_safety(self.audit.rows()), safety_config(self.cfg))
            if diff:
                raise InvariantBroken(f"safety settings differ from the pinned ones {diff}; "
                                      "an owner amendment is required (python -m slh.system "
                                      "amend-config)")
        self.cycle_committed = 0.0
        self._chain_cache: dict[str, str] = {}
        info = adapter.info()
        if not self.audit.rows("start"):
            mods = tuple(info.guard_modules)
            if not mods or not set(self.cfg.guard_floor) <= set(mods):
                raise InvariantBroken("refusing to start: guard floor "
                                      f"{self.cfg.guard_floor} not within {mods}")
            root_state = initial or HarnessState.empty()
            self.versions.save(root_state)
            self.audit.append("start", pointer=root_state.version_id, batch=-1,
                              adapter=_json(asdict(info)),
                              guard_hashes=GD.module_hashes(mods),
                              floor=list(self.cfg.guard_floor), config=_json(asdict(self.cfg)),
                              safety=safety_config(self.cfg),
                              sources=G.registered_sources())
        else:
            self._reconcile()
        reconciled = self.ledger.reconcile()  # we hold the lock: earlier owners are dead
        self.ledger.open_session(0.0)
        self.audit.append("session", session=session, ledger_reconciled=len(reconciled),
                          ledger_cap_usd=cap, paid=paid, metered=metered,
                          ledger_charged_usd=self._charged(), safety=safety_config(self.cfg),
                          item_id=self.cfg.item_id,
                          item_cap_usd=self.cfg.item_cap_usd, production=self.cfg.production,
                          critical_quarantine=self.cfg.critical_quarantine,
                          repaired={p.path.name: (p.repaired or b"").decode("utf-8", "replace")
                                    for p in (self.audit.log, self.traces.log,
                                              self.units.log) if p.repaired is not None})
        self.carry = self._rebuild_carry()

    def close(self) -> None:
        self.lock.release()

    # ------------------------------------------------------------------ pointers

    def root_version(self) -> str:
        return self.audit.rows("start")[0]["pointer"]

    def _pointer_record(self) -> dict[str, Any]:
        for r in reversed(self.audit.rows()):
            if "pointer" in r and r["kind"] in ("start", "confirm_close", "rollback"):
                return r
        raise InvariantBroken("no pointer")

    def current_version(self) -> str:
        return self._pointer_record()["pointer"]

    def current(self) -> HarnessState:
        return self.versions.load(self.current_version())

    def parent_of(self, version: str) -> str:
        for r in reversed(self.audit.rows("confirm_close")):
            if r.get("pointer") == version:
                return r["parent"]
        return self.root_version()

    def certified(self, version: str) -> bool:
        """Root, or a version with a passing conversation-level confirmation."""
        if version == self.root_version():
            return True
        return any(r.get("level") == "conversation" and r.get("passed")
                   and r.get("version") == version for r in self.audit.rows("confirm_close"))

    def _rollback_target(self, version: str) -> str:
        """``version`` in development; in production the nearest certified ancestor."""
        if not self.cfg.production:
            return version
        while not self.certified(version):
            version = self.parent_of(version)
        return version

    def adoption_history(self) -> list[dict[str, Any]]:
        return [r for r in self.audit.rows() if "pointer" in r
                and r["kind"] in ("start", "confirm_close", "rollback")]

    # ------------------------------------------------------------------ restart

    def _reconcile(self) -> None:
        try:
            self.audit.verify()
        except ValueError as e:
            raise InvariantBroken(f"resume: audit chain fails verification ({e})") from e
        closed = {r["k"] for r in self.audit.rows("confirm_close")}
        for r in self.audit.rows("confirm_open"):
            if r["k"] not in closed:
                self.audit.append("confirm_close", k=r["k"], batch=r["batch"], adopt=False,
                                  void=True, reason="abandoned at restart (k never reused)",
                                  candidates=r["candidates"])
        ended = {r["batch"] for r in self.audit.rows("cycle_end")}
        for r in self.audit.rows("cycle_start"):
            if r["batch"] not in ended:
                self.audit.append("cycle_end", batch=r["batch"], status="abandoned")
        # interrupted review decisions, non-human (legacy) clearances, adoptions on them
        fixed = reconcile_reviews(self.audit)
        if fixed:
            self._alert("review_reconciled", records=[
                {"seq": r["seq"], "kind": r["kind"]} for r in fixed])

    def next_batch(self) -> int:
        starts = self.audit.rows("cycle_start")
        return (max(r["batch"] for r in starts) + 1) if starts else 0

    def halted(self) -> dict[str, Any] | None:
        last = self.audit.last("system_halt", "halt_ack")
        return last if last is not None and last["kind"] == "system_halt" else None

    def acknowledge_halt(self, by: str, note: str) -> None:
        """Owner action: clear a latched system halt after investigation."""
        self.audit.append("halt_ack", by=by, note=note)

    # ------------------------------------------------------------------ alerts, heartbeat

    def _charged(self) -> float:
        return float(self.ledger.snapshot()["charged_usd"])

    def _alert(self, kind: str, **detail: Any) -> None:
        """Notify. Always called AFTER the safety action is committed; a failing sink or
        callback is recorded and never propagates."""
        rec = {"ts": now_utc(), "kind": kind, **detail}
        try:
            self.alerts.append(_json(rec))
            if self.alert_fn is not None:
                self.alert_fn(rec)
        except Exception as e:  # noqa: BLE001 - notification must not undo safety actions
            self.audit.append("alert_failed", alert=kind, error=repr(e))

    def _heartbeat(self, batch: int, step: str) -> None:
        tmp = self.root / ".heartbeat.tmp"
        tmp.write_text(json.dumps({"ts": now_utc(), "batch": batch, "step": step,
                                   "pid": os.getpid()}))
        os.replace(tmp, self.root / "heartbeat.json")

    def _system_halt(self, reason: str, **detail: Any) -> None:
        self.audit.append("system_halt", reason=reason, **detail)
        self._alert("system_halt", reason=reason, **detail)

    # ------------------------------------------------------------------ invariants

    def check_invariants(self) -> None:
        self.audit.verify()
        start = self.audit.rows("start")[0]
        info = self.adapter.info()
        if _json(asdict(info)) != start["adapter"]:
            raise InvariantBroken("adapter identity (model, scorer, sampling, guards) changed")
        mods = tuple(info.guard_modules)
        if not set(start["floor"]) <= set(mods):
            raise InvariantBroken("guard floor not within the guard set")
        if GD.module_hashes(mods) != start["guard_hashes"]:
            raise InvariantBroken("guard module source changed")
        bad = GD.conformance(self.adapter, mods)
        if bad:
            raise InvariantBroken(f"guard conformance failed: {bad}")

    def _chain(self, state: HarnessState) -> str:
        v = state.version_id
        if v not in self._chain_cache:
            self._chain_cache[v] = GD.chain_sha256(self.adapter.info().guard_modules, state)
        return self._chain_cache[v]

    # ------------------------------------------------------------------ dispatch

    def _reserve(self, usd: float, what: str) -> str:
        if self.cycle_committed + usd > self.cfg.cycle_cap_usd:
            raise BudgetRefused(f"{what}: cycle cap ${self.cfg.cycle_cap_usd}")
        rid = self.ledger.reserve("agent", what, usd, 0)  # BudgetExceeded on lifetime cap
        self.cycle_committed += usd
        return rid

    def _settle(self, rid: str, reserved: float, actual: float, what: str,
                paid: bool = False, halt_on_overspend: bool = True) -> None:
        """Settle, then write a fsynced accounting checkpoint (``spend``) into the audit
        chain, so a restart can tell a lost or rolled-back ledger even after an interrupted
        cycle. ANY overspend (actual > reservation, whatever the component declares) latches
        a system halt: the estimate that bounds every reservation is wrong, so nothing more
        is dispatched until the owner acknowledges."""
        self.ledger.settle(rid, actual, "agent", what)
        self.cycle_committed += actual - reserved
        self.audit.append("spend", what=what, reserved=reserved, actual=actual,
                          ledger_charged_usd=self._charged())
        if halt_on_overspend and actual > reserved + 1e-12:
            self.audit.append("system_halt", reason="overspend", what=what,
                              reserved=reserved, actual=actual, declared_metered=paid)
            self._alert("overspend", what=what, reserved=reserved, actual=actual)
            self._alert("system_halt", reason="overspend", what=what)
            raise SystemHalted(f"overspend: {what} reserved ${reserved:.6f}, "
                               f"charged ${actual:.6f}")

    def _run(self, b: int, stage: str, requests: list[RunRequest], *, level: str = "decision",
             k: int | None = None, incumbent: HarnessState | None = None) -> list[Outcome]:
        keys = [r.key for r in requests]
        if len(set(keys)) != len(keys):
            raise InvariantBroken(f"{stage}: duplicate request keys (harness bug)")
        what = f"adapter:{stage}"
        paid = "adapter" in self.metered
        outs: list[Outcome] = []
        for chunk in self._chunks(requests):
            worst = float(self.adapter.estimate_cost_usd(len(chunk), level=level))
            if paid and not worst > 0:
                raise InvariantBroken(f"{stage}: a metered adapter estimated ${worst} for "
                                      f"{len(chunk)} runs")
            rid = self._reserve(worst, what)
            self._heartbeat(b, stage)
            try:
                got = list(self.adapter.run(chunk, budget_usd=worst))
            except AdapterBudgetStop as e:
                self._settle(rid, worst, worst, what)  # charged in full
                self.audit.append("system_halt", reason="adapter budget stop", what=what,
                                  reserved=worst, detail=str(e))
                self._alert("system_halt", reason="adapter budget stop", what=what)
                raise SystemHalted(f"adapter stopped at its reservation: {e}") from e
            except BaseException:
                self._settle(rid, worst, worst, what)  # a lost attempt is charged in full
                raise
            self._settle(rid, worst, sum(float(o.cost_usd) for o in got), what, paid=paid)
            outs += got
        self._validate(stage, requests, outs)
        self.traces.add(b, outs, stage=stage, k=k)
        inc = incumbent or self.current()
        self._incidents(b, stage, requests, outs, inc,
                        self._critical_events(stage, requests, outs, inc))
        return outs

    def _chunks(self, requests: list[RunRequest]) -> list[list[RunRequest]]:
        """Reserved dispatch chunks of about ``dispatch_chunk_runs`` runs, never splitting a
        unit's runs (paired checks need them together)."""
        size = max(1, self.cfg.dispatch_chunk_runs)
        chunks: list[list[RunRequest]] = []
        cur: list[RunRequest] = []
        for r in requests:
            if len(cur) >= size and r.unit.unit_id != cur[-1].unit.unit_id:
                chunks.append(cur)
                cur = []
            cur.append(r)
        if cur:
            chunks.append(cur)
        return chunks

    def _critical_events(self, stage: str, requests: list[RunRequest], outs: list[Outcome],
                         inc: HarnessState) -> list[Outcome]:
        """Critical events under a candidate configuration (screen, confirm) that count as
        incidents. ``paired``: only when the same unit's current-setup run (screen anchor,
        confirmation incumbent arm) was not critical; a missing current-setup run counts as
        clean (fail closed). ``literal``: every candidate-run event."""
        if stage != "screen" or self.cfg.critical_quarantine == "attributed":
            # CONFIRMATION: candidate-only critical events go to the HOLD with a review
            # packet (``_confirm_hold``), never an automatic retirement. SCREEN in the
            # attributed mode: the per-candidate test after the dispatch.
            return []
        by = {r.key: r for r in requests}
        base_ids = _ids(inc)

        def is_candidate(o: Outcome) -> bool:
            return bool(_ids(by[o.key].config) - base_ids) and o.tag not in ("incumbent",
                                                                              "root")
        current: dict[str, bool] = {}
        for o in outs:
            if (o.tag == "incumbent" or (o.tag.startswith("x=") and "1" not in o.tag)):
                current[o.unit_id] = o.critical_harm is True
        hits = [o for o in outs if o.critical_harm is True and is_candidate(o)]
        if self.cfg.critical_quarantine in ("paired", "attributed"):  # confirm: paired
            hits = [o for o in hits if not current.get(o.unit_id, False)]
        return hits

    def _validate(self, stage: str, requests: list[RunRequest], outs: list[Outcome]) -> None:
        by = {r.key: r for r in requests}
        seen: set[tuple[str, str, str, str]] = set()
        for o in outs:
            r = by.get(o.key)
            if r is None or o.key in seen:
                raise InvariantBroken(f"{stage}: outcome {o.key} matches no open request "
                                      "(unknown or duplicate)")
            seen.add(o.key)
            if o.applied_state_sha256 != r.config.content_sha256:
                raise InvariantBroken(f"{stage}: adapter applied another state for {o.key}")
            if o.context != r.unit.context:
                raise InvariantBroken(f"{stage}: context changed for {o.key}")
            if o.status not in STATUSES:
                raise InvariantBroken(f"{stage}: unknown status {o.status!r}")
            if o.status == "scored" and not (isinstance(o.success, bool)
                                             and isinstance(o.harm, bool)):
                raise InvariantBroken(f"{stage}: scored outcome without boolean scores")
            if o.critical_harm is not None and not isinstance(o.critical_harm, bool):
                raise InvariantBroken(f"{stage}: critical_harm must be True, False or None")
        if len(seen) != len(requests):
            raise InvariantBroken(f"{stage}: {len(requests) - len(seen)} requests unanswered")
        for o in outs:
            if not o.mediated or o.guard_chain_sha256 != self._chain(by[o.key].config):
                self._system_halt("guard mediation failure", stage=stage, unit=o.unit_id,
                                  mediated=o.mediated)
                raise SystemHalted("a run was not mediated by the pinned guard chain")

    def _incidents(self, b: int, stage: str, requests: list[RunRequest],
                   outs: list[Outcome], inc: HarnessState,
                   critical: list[Outcome] | None = None) -> None:
        critical = critical or []
        flagged = [o for o in outs if o.incident]
        seen = {o.key for o in flagged}
        hits = flagged + [o for o in critical if o.key not in seen]
        if not hits:
            return
        by = {r.key: r for r in requests}
        root_v, cur_v = self.root_version(), self.current_version()
        versions = sorted({o.config_version for o in hits})
        self.audit.append("incident", batch=b, stage=stage, count=len(hits),
                          flagged=len(flagged), critical=len(critical),
                          critical_mode=self.cfg.critical_quarantine,
                          versions=versions, units=sorted({o.unit_id for o in hits})[:20])
        note = {"batch": b, "stage": stage, "count": len(hits), "critical": len(critical),
                "versions": versions}
        if root_v in versions:
            self.audit.append("system_halt", reason="incident under the root configuration",
                              batch=b, stage=stage)  # latched before anyone is notified
            self._alert("incident", **note)
            self._alert("system_halt", reason="incident under the root configuration",
                        batch=b, stage=stage)
            raise SystemHalted("incident under root")
        # Candidates present in EVERY incident run are implicated; if no candidate is common
        # to all of them, every candidate present in any incident run is (fail closed).
        sets = [_ids(by[o.key].config) - _ids(inc) for o in hits]
        sets = [x for x in sets if x]
        implicated: set[str] = set.intersection(*sets) if sets else set()
        if sets and not implicated:
            implicated = set.union(*sets)
        rolled = cur_v in versions and cur_v != root_v
        if rolled:
            implicated |= _ids(self.current()) - _ids(self.versions.load(root_v))
            self.audit.append("rollback", batch=b, pointer=root_v, parent=cur_v,
                              reason="incident under the incumbent: reset to root",
                              critical=len(critical))
            self.carry = self._empty_carry()
        for cid in sorted(implicated):
            self.audit.append("retire", batch=b, cid=cid, stage="quarantine",
                              evidence={"incident_stage": stage})
        self.carry["pool"] = [d for d in self.carry["pool"] if d["cid"] not in implicated]
        if not rolled:
            # A candidate incident halts the BATCH only: every other candidate of the
            # screen pool is carried to the next cycle and re-screened on fresh screen
            # units (no look counted for the halted screen). Root / incumbent incidents
            # are handled above. (A halted CONFIRMATION spends its k and closes its
            # candidates void, as before: nothing is in play there.)
            pooled = {d["cid"] for d in self.carry["pool"]}
            kept = [c for c in self._in_play if c.cid not in implicated
                    and c.cid not in pooled and c.cid not in self.blocked()]
            self.carry["pool"] += [c.to_dict() for c in kept]
            self.audit.append("pool_carried", batch=b, stage=stage,
                              carried=[d["cid"] for d in self.carry["pool"]],
                              quarantined=sorted(implicated), carry=_json(self.carry))
        self._in_play = []
        self._alert("incident", **note, quarantined=sorted(implicated))
        if rolled:
            self._alert("rollback", batch=b, to=root_v, frm=cur_v, reason="incident")
        raise CycleHalted(f"incident in {stage}; quarantined {sorted(implicated)}")

    # ------------------------------------------------------------------ carry state

    def _empty_carry(self) -> dict[str, Any]:
        return {"incumbent": self.current_version(), "pool": [], "looks": {}, "batches": {},
                "meta": {}}

    def _rebuild_carry(self) -> dict[str, Any]:
        last = self.audit.last("screen", "confirm_deferred", "pool_carried", "confirm_close",
                               "rollback", "start")
        if last is None or last["kind"] not in ("screen", "confirm_deferred",
                                                "pool_carried") or \
                last["carry"]["incumbent"] != self.current_version():
            return self._empty_carry()
        carry = json.loads(json.dumps(last["carry"]))
        gone = self.retired()
        carry["pool"] = [d for d in carry["pool"] if d["cid"] not in gone]
        return carry

    def _carry_rows(self) -> list[dict[str, Any]]:
        rows = []
        batches = {int(b): ids for b, ids in self.carry["batches"].items()}
        for r in self.traces.rows("screen"):
            if r["batch"] in batches and r["stage"] == "screen":
                rows += G.screen_rows([self.traces.outcome(r)], batches[r["batch"]])
        return rows

    # ------------------------------------------------------------------ retired set

    def retired(self) -> set[str]:
        out = {r["cid"] for r in self.audit.rows() if r["kind"] in
               ("retire", "policy_drop", "size_drop", "kind_drop")}
        released = self.released_holds()
        for r in self.audit.rows("confirm_close"):
            out |= set(r.get("candidates", [])) - released
        return out

    def held(self) -> set[str]:
        """Candidates of confirmation HOLDs and of SCREEN holds without a human review
        decision."""
        rows = self.audit.rows()
        return held_candidates(rows) | screen_held_candidates(rows)

    def screen_reserved(self) -> set[str]:
        """Candidates under screen-hold review: pending, or approved and not yet in a
        confirmation. They are neither screened again nor re-proposed."""
        rows = self.audit.rows()
        return screen_held_candidates(rows) | {
            c for hold, _ in approved_screen_holds(rows) for c in hold["candidates"]}

    def adjudicate_screen_hold(self, h: int, decision: str, by: str, note: str,
                               packet_sha256: str | None = None) -> dict[str, Any]:
        """Human review of screen hold ``h`` (``review_decide`` target ``s<h>``): approve
        (needs the current packet hash; the next cycle confirms the candidates) or reject
        (retired)."""
        rec = _screen_decide(self.root, self.audit, h, decision, by, note, packet_sha256)
        gone = self.retired() | self.held()
        self.carry["pool"] = [d for d in self.carry["pool"] if d["cid"] not in gone]
        return rec

    def released_holds(self) -> set[str]:
        """Candidates whose hold a reviewer APPROVED (adopted, or re-proposable when the
        confirmation could not adopt any more); never by replication."""
        return released_candidates(self.audit.rows())

    def adjudicate_hold(self, k: int, decision: str, by: str, note: str,
                        packet_sha256: str | None = None) -> dict[str, Any]:
        """Human review of the hold of confirmation ``k`` (``review_decide``):
        ``approve`` (alias ``release``; needs the current packet hash) or ``reject``
        (alias ``quarantine``: retired)."""
        before = self.current_version()
        rec = _review_decide(self.root, self.audit, k, decision, by, note, packet_sha256)
        if rec.get("adopt") or self.current_version() != before:
            self.carry = self._empty_carry()
        else:
            gone = self.retired() | self.held()
            self.carry["pool"] = [d for d in self.carry["pool"] if d["cid"] not in gone]
        return rec

    def blocked(self) -> set[str]:
        """Ids that may never be confirmed or adopted: retired (screen, confirm, cap,
        quarantine), dropped (policy, size, kind), or HELD pending adjudication."""
        return {r["cid"] for r in self.audit.rows() if r["kind"] in
                ("retire", "policy_drop", "size_drop", "kind_drop")} | self.held()

    def adopted(self) -> list[str]:
        return [c for r in self.audit.rows("confirm_close") if r.get("adopt")
                for c in r["candidates"]]

    def _within_caps(self, state: HarnessState) -> str | None:
        # prompt notes only; skill_ref / memory records are capped below by their own limits
        notes = [n for n in state.notes if n.get("kind", "note") == "note"]
        if len(notes) > self.cfg.max_notes:
            return f"{len(notes)} notes > {self.cfg.max_notes}"
        total = sum(len(n["content"]) for n in notes)
        if total > self.cfg.max_total_note_chars:
            return f"{total} note characters > {self.cfg.max_total_note_chars}"
        return self._within_host_caps(state)

    def _within_host_caps(self, state: HarnessState) -> str | None:
        """Caps for the host learned-state kinds (skills and memory)."""
        skills = state.skills
        if len(skills) > self.cfg.max_skills:
            return f"{len(skills)} skills > {self.cfg.max_skills}"
        for name, text in skills.items():
            if len(text) > self.cfg.max_skill_chars:
                return f"skill {name}: {len(text)} characters > {self.cfg.max_skill_chars}"
        total = sum(len(t) for t in skills.values())
        if total > self.cfg.max_total_skill_chars:
            return f"{total} skill characters > {self.cfg.max_total_skill_chars}"
        for n in state.notes:
            if n.get("kind") == "memory" and len(n["content"]) > self.cfg.max_memory_chars:
                return (f"memory {n.get('target')}: {len(n['content'])} characters > "
                        f"{self.cfg.max_memory_chars}")
        return None

    def _host_kind_screen(self, c: Candidate) -> tuple[str, str] | None:
        """(record kind, reason) when a skill / memory candidate is inadmissible: support
        files (not admitted in V1), the adapter's policy screen, or size caps."""
        if c.kind == "skill" and c.payload.get("files"):
            return "kind_drop", "skill support files are not admitted in V1"
        reason = self.adapter.policy_conflict(c.text) if c.text else None
        if reason:
            return "policy_drop", reason
        if c.kind == "skill" and not c.payload.get("tombstone"):
            desc = str(c.payload["description"])
            if len(desc) > self.cfg.max_skill_description_chars:
                return "size_drop", f"description {len(desc)} characters"
            if len(c.text) > self.cfg.max_skill_chars:
                return "size_drop", f"skill {len(c.text)} characters"
        if c.kind == "memory":
            chars = len(MEMORY_DELIMITER.join(c.payload["entries"]))
            if chars > self.cfg.max_memory_chars:
                return "size_drop", f"memory {chars} characters"
        return None

    # ------------------------------------------------------------------ steps

    def _sample(self, b: int, role: str, n: int, k: int | None = None,
                level: str = "decision") -> list:
        units = list(self.adapter.sample_units(b, role, n, k=k, level=level))
        self.units.admit(units, role, n, b, k)
        return units

    def monitor(self, b: int) -> dict[str, Any]:
        cur_v, root_v = self.current_version(), self.root_version()
        root = self.versions.load(root_v)
        lv = self.cfg.monitor_level
        units = self._sample(b, "monitor", self.cfg.monitor.units, level=lv)
        if cur_v == root_v:
            outs = self._run(b, "monitor", [RunRequest(u, root, "root") for u in units],
                             level=lv)
            dec = {"action": "none", "root_only": True, "units": len(outs),
                   "root_success": sum(o.worst()[0] for o in outs) / len(outs),
                   "root_harm": sum(o.worst()[1] for o in outs) / len(outs),
                   "root_unscored": sum(not o.scored for o in outs) / len(outs)}
            self.audit.append("monitor_batch", batch=b, incumbent=cur_v, decision=dec)
            return dec
        cur = self.versions.load(cur_v)
        reqs = [RunRequest(u, s, t) for u in units for s, t in ((root, "root"),
                                                                 (cur, "incumbent"))]
        self._run(b, "monitor", reqs, level=lv)
        since = self._pointer_record()["batch"]
        rows = [r for r in self.traces.rows("monitor") if r["batch"] > since
                and r["config_version"] in (cur_v, root_v)]
        dec = G.monitor_decide([self.traces.outcome(r) for r in rows], self.cfg.monitor,
                               self.cfg.confirm)
        self.audit.append("monitor_batch", batch=b, incumbent=cur_v, decision=dec)
        if dec["action"] == "rollback":
            parent = self._rollback_target(self.parent_of(cur_v))
            why = ("critical-harm increase" if dec.get("reason") == "critical harm increase"
                   else "worse than root beyond margins")
            self.audit.append("rollback", batch=b, pointer=parent, parent=cur_v,
                              reason=f"monitor: incumbent {why}", evidence=dec)
            self._alert("rollback", batch=b, to=parent, frm=cur_v)
            self.carry = self._empty_carry()
        return dec

    def _attribute_critical(self, b: int, outs: list[Outcome], pool: list[Candidate],
                            inc: HarnessState) -> tuple[list[Candidate], set[str]]:
        """``attributed`` mode, SCREEN only, after a completed screen
        (``slh.system.attribution``): the hard trigger on semantic violation kinds and the
        per-candidate test of the average critical-risk effect (studentized sign-flip on
        unit means, pooled over looks, a fixed alpha per candidate spent over its frozen
        maximum looks). Returns (pool without the quarantined, the HELD ids: not advanced
        this look, carried and replayed). Guard flags still halt through the incident
        path; confirmation keeps the paired event quarantine and the HOLD.

        A hard-trigger HOLD is PERSISTENT (final review of b7a4ae7): its units and its
        new-kind events are kept in ``carry["crit_attr"][cid]["hard"]`` (and in the
        ``critical_attribution`` record's ``hard_state``, so a carry reset does not lose
        them); a candidate with one is never advanced (no automatic clearance; it is retired
        when its look budget is spent), and new-kind events of different looks accumulate
        toward ``min_units``."""
        from slh.system import attribution as AT

        ids = [c.cid for c in pool]
        prior = self.carry.setdefault("crit_attr", {})
        frozen = {cid: int((prior.get(cid) or {}).get("max_looks")
                           or self.cfg.screen.max_looks) for cid in ids}
        stat = AT.attribute(outs, ids, alpha=self.cfg.critical_attribution_alpha,
                            max_looks=frozen, prior=prior)
        base = self._attribution_base(inc.version_id)
        before = {cid: self._hard_state(cid) for cid in ids}
        detail: dict[str, dict[str, list[str]]] = {}
        report: dict[str, Any] = {}
        hard = AT.hard_trigger(outs, ids, base,
                               min_base=self.cfg.critical_attribution_min_base,
                               prior_kinds={c: h["kinds"] for c, h in before.items()},
                               detail=detail, rule=self.cfg.critical_attribution_base_rule,
                               min_base_rel=self.cfg.critical_attribution_min_base_rel,
                               min_base_floor=self.cfg.critical_attribution_min_base_floor,
                               report=report)
        state = {}
        for cid in ids:
            kinds = {**before[cid]["kinds"], **detail.get(cid, {})}
            units = sorted(set(before[cid]["units"])
                           | set((hard.get(cid) or {}).get("hold", [])))
            state[cid] = {"units": units, "kinds": kinds}
        quarantined = sorted({c for c, d in stat["candidates"].items()
                              if d["decision"] == "quarantine"}
                             | {c for c, h in hard.items() if h["quarantine"]})
        held = sorted(({c for c, d in stat["candidates"].items()
                        if d["decision"] in ("hold", "exhausted")}
                       | {c for c in ids if state[c]["units"]}) - set(quarantined))
        for cid, d in stat["candidates"].items():
            prior[cid] = {"d": d["d"], "d_worst": d["d_worst"], "looks": d["looks"],
                          "unresolved_pairs": d["unresolved_pairs"],
                          "max_looks": d["max_looks"], "hard": state[cid]}
        summary = {c: {k: v for k, v in d.items() if k not in ("d", "d_worst")}
                   for c, d in stat["candidates"].items()}
        self._last_attr = {"batch": b, "summary": summary, "hard": hard, "state": state,
                           "report": report}
        self.audit.append("critical_attribution", batch=b, candidates=_json(summary),
                          hard_trigger=_json(hard), quarantined=quarantined, held=held,
                          hard_state=_json({c: v for c, v in state.items()
                                            if v["units"] or v["kinds"]}),
                          baseline=_json({k: report.get(k) for k in (
                              "rule", "min_base", "min_base_rel", "min_base_floor",
                              "base")}),
                          events=_json((report.get("events") or [])[:200]))
        for cid in quarantined:
            self.audit.append("retire", batch=b, cid=cid, stage="quarantine",
                              evidence={"incident_stage": "screen", "rule": "attributed",
                                        "test": summary.get(cid), "hard": hard.get(cid)})
        if quarantined or held:
            self._alert("critical_attributed", batch=b, quarantined=quarantined, held=held)
        return [c for c in pool if c.cid not in quarantined], set(held)

    def _hard_state(self, cid: str) -> dict[str, Any]:
        """The persistent screen hard-trigger state of ``cid``: the UNION of the carry's and
        the latest ``critical_attribution`` record's. The state only ever grows (nothing
        clears a hard hold automatically), so the union is always the newest state: after a
        crash between the ``critical_attribution`` record and the next carry snapshot, the
        carry restored at restart is OLDER than the audit record and must not win (M-H1
        review; a carry reset keeps the record's state too)."""
        a = ((self.carry.get("crit_attr") or {}).get(cid) or {}).get("hard") or {}
        r_state: dict[str, Any] = {}
        for r in reversed(self.audit.rows("critical_attribution")):
            if cid in (r.get("hard_state") or {}):
                r_state = r["hard_state"][cid] or {}
                break
        units = sorted(set(a.get("units") or []) | set(r_state.get("units") or []))
        kinds: dict[str, list[Any]] = {}
        for src in (r_state, a):
            for k, v in (src.get("kinds") or {}).items():
                cur = kinds.setdefault(k, [])
                cur += [x for x in v if x not in cur]
        return {"units": units, "kinds": kinds}

    def _attribution_base(self, inc_v: str) -> list[Outcome]:
        """The hard trigger's baseline: EVERY labelled run of the current setup ``inc_v``
        recorded so far, of every earlier look and cycle: monitor and evidence runs, screen
        anchors (no candidate bit) and the incumbent arms of confirmations (each an unbiased
        draw of its role's units). Hold replicates are excluded (they are drawn on held
        units only)."""
        out = []
        for r in self.traces.rows():
            if r["config_version"] != inc_v:
                continue
            role, tag = r["role"], str(r["tag"])
            if role in ("monitor", "evidence") or (role == "screen" and "1" not in tag) \
                    or (role == "confirm" and tag == "incumbent"):
                out.append(self.traces.outcome(r))
        return out

    def _screen_hold(self, b: int, inc: HarnessState, cands: list[Candidate],
                     sd: dict[str, Any], outs: list[Outcome]) -> dict[str, Any]:
        """A SCREEN hold (``screen_hold_review``): the review packet
        ``reviews/screen-hold-s<h>.json`` (the screen decisions, the attribution test, the
        persistent hard state, every held event with the candidate's and its bit-flip
        partner's traces), then the ``screen_hold`` record. The candidates are neither
        advanced nor retired; a HUMAN decides (``review_decide`` s<h>)."""
        h = len(self.audit.rows("screen_hold")) + 1
        ids = [c.cid for c in cands]
        last = getattr(self, "_last_attr", None) or {}
        # every candidate-only event of these candidates over their looks (the hold
        # persists across looks), with both runs' traces from the trace store
        evs = [e for r in self.audit.rows("critical_attribution")
               for e in (r.get("events") or []) if e["candidate"] in ids]
        want = {(e["unit"], t) for e in evs for t in (e["tag"], e.get("partner_tag"))}
        by: dict[tuple[str, str], Outcome] = {(o.unit_id, o.tag): o for o in outs}
        for r in self.traces.rows("screen"):
            if (r["unit_id"], r["tag"]) in want and (r["unit_id"], r["tag"]) not in by:
                by[(r["unit_id"], r["tag"])] = self.traces.outcome(r)
        events = []
        for e in evs:
            a, pb = by.get((e["unit"], e["tag"])), by.get((e["unit"], e.get("partner_tag")))
            events.append({**e, "candidate_trace": a.trace if a else None,
                           "partner_trace": pb.trace if pb else None})
        hard_state = {cid: self._hard_state(cid) for cid in ids}
        packet = {"kind": "screen_hold", "h": h, "batch": b, "candidates": ids,
                  "incumbent": inc.version_id, "candidate_meta": {c.cid: c.to_dict()
                                                                  for c in cands},
                  "screen": {cid: sd["decisions"][cid] for cid in ids},
                  "attribution": {cid: (last.get("summary") or {}).get(cid) for cid in ids},
                  "hard_state": hard_state,
                  "baseline": {k: (last.get("report") or {}).get(k) for k in (
                      "rule", "min_base", "min_base_rel", "min_base_floor", "base")},
                  "events": events[:200], "events_total": len(events),
                  "decision_options": {
                      "approve": "the candidates go to a confirmation in the next cycle; its "
                                 "own tests and its per-unit HOLD still apply",
                      "reject": "the candidates are retired"},
                  "auto_release": RP.NEVER}
        name = f"screen-hold-s{h}.json"
        sha = self._write_packet(name, packet)
        self.audit.append("screen_hold", batch=b, h=h, candidates=ids,
                          incumbent=inc.version_id, packet=name, packet_sha256=sha,
                          units={cid: hard_state[cid]["units"] for cid in ids},
                          meta={c.cid: c.to_dict() for c in cands},
                          screen={cid: _json(sd["decisions"][cid]) for cid in ids})
        self._alert("screen_hold", batch=b, h=h, candidates=ids)
        return {"h": h, "candidates": ids, "packet": name}

    def _released_screen_holds(self, b: int, inc: HarnessState
                               ) -> tuple[list[Candidate], dict[str, Any], list[int]] | None:
        """Human-APPROVED screen holds not yet confirmed: (candidates, their screen
        decisions, hold ids) for a confirmation against ``inc``. An approval whose
        incumbent is no longer current is void for confirmation: its candidates are retired
        (the screen evidence was against another incumbent)."""
        rows = self.audit.rows()
        gone = _gone(rows)
        adv: list[Candidate] = []
        screen: dict[str, Any] = {}
        hs: list[int] = []
        for hold, rec in approved_screen_holds(rows):
            live = [c for c in hold["candidates"] if c not in gone]
            if not live:
                continue
            if hold["incumbent"] != inc.version_id:
                for cid in live:
                    self.audit.append("retire", batch=b, cid=cid, stage="screen_hold",
                                      evidence={"screen_hold": hold["h"],
                                                "approved_seq": rec["seq"],
                                                "reason": "the incumbent changed after the "
                                                          "approval"})
                continue
            hs.append(hold["h"])
            for cid in live:
                adv.append(Candidate.from_dict(hold["meta"][cid]))
                screen[cid] = {**(hold.get("screen") or {}).get(cid, {}),
                               "screen_hold": hold["h"], "approved_by": rec["by"],
                               "adjudication_seq": rec["seq"]}
        return (adv, screen, hs) if adv else None

    def _confirm_hold(self, outs: list[Outcome]) -> list[dict[str, Any]]:
        """CONFIRMATION policy, every quarantine mode: a candidate-only critical event (the
        candidate arm critical where the incumbent or root arm is not critical; in the
        ``literal`` mode any candidate critical event) and ANY candidate run with an
        UNRESOLVED critical label (worst case: it may be critical) HOLD the confirmation:
        no adoption, a review packet, a HUMAN review decides (``review_decide``). One row per
        (unit, discordant reference). Checked per unit, before any aggregate count could
        cancel it. Critical events at confirmation never auto-retire
        (guard ``incident`` flags still halt through the incident path)."""
        literal = self.cfg.critical_quarantine == "literal"
        by: dict[str, dict[str, Outcome]] = {}
        for o in outs:
            by.setdefault(o.unit_id, {})[o.tag] = o
        rows = []
        for uid, arms in by.items():
            c = arms.get("candidate")
            if c is None:
                continue
            if c.critical_harm is None:
                rows.append({"unit": uid, "reason": "unresolved candidate critical label",
                             "candidate_trace": c.trace, "candidate_status": c.status})
                continue
            if c.critical_harm is not True:
                continue
            for ref in ("incumbent", "root"):
                r = arms.get(ref)
                if r is not None and (literal or r.critical_harm is not True):
                    rows.append({"unit": uid, "reason": "candidate-only critical event",
                                 "reference": ref, "candidate_trace": c.trace,
                                 "reference_trace": r.trace,
                                 "reference_label": r.critical_harm})
                    # no break: EVERY discordant reference is a row (root as well)
        return rows

    def _propose_report(self, b: int, error: str | None = None) -> None:
        """The proposer's own report (raw writer replies, parse errors, op verdicts, its
        internal drops), persisted in the audit chain after every ``propose``."""
        rep = getattr(self.proposer, "last_report", None)
        if rep is None and error is None:
            return
        try:
            self.audit.append("propose_report", batch=b, report=_json(rep or {}),
                              error=error)
        except (TypeError, ValueError) as e:  # an unserialisable report is still noted
            self.audit.append("propose_report", batch=b, report=None,
                              error=error, report_error=f"{type(e).__name__}: {e}")

    def _propose(self, b: int, inc: HarnessState) -> list[Candidate]:
        est = float(self.proposer.estimate_cost_usd())
        rid = self._reserve(est, "proposer")
        if hasattr(self.proposer, "last_report"):
            self.proposer.last_report = {}
        try:
            cands = list(self.proposer.propose(b, self.traces.evidence(b), inc,
                                               self.retired()))
        except BaseException as e:
            self._settle(rid, est, est, "proposer")
            self._propose_report(b, error=f"{type(e).__name__}: {e}")
            raise
        self._settle(rid, est, float(self.proposer.last_cost_usd), "proposer",
                     paid="proposer" in self.metered)
        self._propose_report(b)
        retired, pooled = self.retired(), {c["cid"] for c in self.carry["pool"]}
        reserved = self.screen_reserved()
        new = []
        for c in cands:
            why = ("retired" if c.cid in retired else "already pooled" if c.cid in pooled
                   else "under screen-hold review" if c.cid in reserved
                   else "duplicate in this proposal" if c.cid in {x.cid for x in new}
                   else None)
            if why:
                self.audit.append("propose_drop", batch=b, cid=c.cid, reason=why)
                continue
            if c.kind == "weights":
                self.audit.append("kind_drop", batch=b, cid=c.cid, reason="weights from M6")
                continue
            if c.kind == "note":
                reason = self.adapter.policy_conflict(c.text)
                if reason:
                    self.audit.append("policy_drop", batch=b, cid=c.cid,
                                      candidate=c.to_dict(), reason=reason)
                    continue
                if len(c.text) > self.cfg.max_note_chars:
                    self.audit.append("size_drop", batch=b, cid=c.cid, chars=len(c.text))
                    continue
            if c.kind in ("skill", "memory"):
                drop = self._host_kind_screen(c)
                if drop:
                    self.audit.append(drop[0], batch=b, cid=c.cid, candidate=c.to_dict(),
                                      reason=drop[1])
                    continue
            self.audit.append("propose", batch=b, candidate=c.to_dict())
            new.append(c)
        return new

    def _refuse_uncertified(self) -> None:
        cur_v = self.current_version()
        if self.cfg.production and not self.certified(cur_v):
            raise ProductionRefused(
                f"production mode: current version {cur_v} has no conversation-level "
                "confirmation; run certify_current() (python -m slh.system certify)")

    def cycle(self, b: int) -> dict[str, Any]:
        if self.halted():
            raise SystemHalted(f"latched: {self.halted()['reason']}")
        self._refuse_uncertified()
        self.cycle_committed = 0.0
        self._in_play = []
        self.audit.append("cycle_start", batch=b)
        summary: dict[str, Any] = {"batch": b}
        try:
            self.check_invariants()
            summary["monitor"] = self.monitor(b)
            self._steps(b, summary)
        except CycleHalted as e:
            summary["halted"] = str(e)
            self.audit.append("cycle_end", batch=b, status="halted", summary=_json(summary),
                              ledger_charged_usd=self._charged())
            return summary
        self.audit.append("cycle_end", batch=b, status="ok", summary=_json(summary),
                          ledger_charged_usd=self._charged())
        return summary

    def _steps(self, b: int, summary: dict[str, Any]) -> None:
        inc = self.current()
        if self.carry["incumbent"] != inc.version_id:
            self.carry = self._empty_carry()
        ev = self._sample(b, "evidence", self.cfg.evidence_units)
        self._run(b, "evidence", [RunRequest(u, inc, "incumbent") for u in ev])

        pend = self.carry.get("pending_confirm")
        if pend:  # a confirmation deferred by the window guard: run it first
            adv = [Candidate.from_dict(d) for d in pend["candidates"]
                   if d["cid"] not in self.blocked()]
            if adv:
                summary["resumed_confirmation"] = [c.cid for c in adv]
                self._confirm_advanced(b, inc, adv, pend["screen"], summary)
                return
            self.carry.pop("pending_confirm", None)

        rel = self._released_screen_holds(b, inc)
        if rel:  # human-approved screen holds: their confirmation first (no new screen)
            adv, screen, hs = rel
            summary["screen_hold_confirmation"] = hs
            self._confirm_advanced(b, inc, adv, screen, summary)
            return

        new = self._propose(b, inc)
        summary["proposed"] = len(new)
        gone = self.retired() | self.screen_reserved()
        pool = [Candidate.from_dict(d) for d in self.carry["pool"] if d["cid"] not in gone] + new
        # conflict exclusion (skill / memory post-states of one target): keep the first
        pool, clash = target_conflicts(pool)
        for c in clash:
            self.audit.append("conflict_drop", batch=b, cid=c.cid, target=c.target_key)
        cap = self.cfg.screen.max_pool
        for c in pool[cap:]:
            self.audit.append("deferred", batch=b, cid=c.cid, reason="pool cap")
        pool = pool[:cap]
        summary["pool"] = [c.cid for c in pool]
        if not pool:
            return

        # screen
        self._in_play = list(pool)  # carried forward if a candidate incident halts the batch
        units = self._sample(b, "screen", self.cfg.screen.units)
        reqs = []
        for u in units:
            for v in G.screen_vectors(u.unit_id, len(pool), self.cfg.screen.draws):
                sel = [c for c, bit in zip(pool, v, strict=True) if bit]
                reqs.append(RunRequest(u, apply_all(inc, sel), "x=" + "".join(map(str, v))))
        souts = self._run(b, "screen", reqs, incumbent=inc)
        self._in_play = []
        screened_c = list(pool)
        screened = [c.cid for c in pool]  # the bit order of this batch's screen tags
        held: set[str] = set()
        if self.cfg.critical_quarantine == "attributed":
            pool, held = self._attribute_critical(b, souts, pool, inc)
            if not pool:
                summary["screen"] = {}
                return
        carry = self.carry
        for c in screened_c:  # quarantined ones stay in the rows as nuisance covariates
            carry["looks"][c.cid] = carry["looks"].get(c.cid, 0) + 1
            carry["meta"][c.cid] = c.to_dict()
        carry["batches"][str(b)] = screened
        while len(carry["batches"]) > self.cfg.screen.max_carry_batches:
            carry["batches"].pop(min(carry["batches"], key=int))
        live = {cid for ids in carry["batches"].values() for cid in ids}
        carry["meta"] = {k: v for k, v in carry["meta"].items() if k in live}
        sd = G.screen_decide(self._carry_rows(), pool, carry["meta"], carry["looks"],
                             self.cfg.screen)
        review: list[str] = []
        for cid in held:  # HELD by attribution (a hard-trigger hold persists): never advanced
            d = sd["decisions"].get(cid)
            if not d:
                continue
            spent = carry["crit_attr"].get(cid, {}).get("looks", 0) >= \
                carry["crit_attr"].get(cid, {}).get("max_looks", self.cfg.screen.max_looks)
            if self.cfg.screen_hold_review and (d["decision"] == "advance" or (
                    spent and d["decision"] == "carry")):
                # the hold, not the screen, stops it: a HUMAN decides (packet below)
                d["held"] = f"screen hold: human review (screen said {d['decision']})"
                d["decision"] = "review"
                sd["advanced"] = [x for x in sd["advanced"] if x != cid]
                review.append(cid)
            elif spent:  # legacy: frozen look budget used up, retired
                d["decision"], d["held"] = "retire", "look budget spent"
                sd["advanced"] = [x for x in sd["advanced"] if x != cid]
            elif d["decision"] == "advance":
                d["decision"], d["held"] = "carry", True
                sd["advanced"] = [x for x in sd["advanced"] if x != cid]
        carry["pool"] = [c.to_dict() for c in pool
                         if sd["decisions"][c.cid]["decision"] == "carry"]
        summary["screen"] = {k: v["decision"] for k, v in sd["decisions"].items()}
        if review:  # BEFORE the screen record: a crash in between never loses a candidate
            summary["screen_hold"] = self._screen_hold(b, inc, [c for c in pool
                                                                if c.cid in review],
                                                       sd, souts)
        for cid, d in sd["decisions"].items():
            if d["decision"] == "retire":
                self.audit.append("retire", batch=b, cid=cid, stage="screen", evidence=d)
        self.audit.append("screen", batch=b, incumbent=inc.version_id,
                          pool=[c.cid for c in pool], result=sd, carry=_json(carry))

        # confirm
        blocked = self.blocked()
        adv = [c for c in pool if c.cid in sd["advanced"] and c.cid not in blocked]
        summary["adopted"] = []
        if not adv:
            return
        self._in_play = []
        self._confirm_advanced(b, inc, adv, {c.cid: sd["decisions"][c.cid] for c in adv},
                               summary)

    def _confirm_advanced(self, b: int, inc: HarnessState, adv: list[Candidate],
                          screen: dict[str, Any], summary: dict[str, Any]) -> None:
        summary["adopted"] = []
        cand = apply_all(inc, adv)
        over = self._within_caps(cand)
        if over:
            for c in adv:
                self.audit.append("retire", batch=b, cid=c.cid, stage="cap", evidence=over)
            return
        self.versions.save(cand)
        ids = [c.cid for c in adv]
        if self.confirm_guard is not None:
            arms = 2 if self.root_version() == inc.version_id else 3
            why = self.confirm_guard(self.cfg.confirm.units * arms)
            if why:
                # the screen already advanced them: the NEXT cycle goes straight to this
                # confirmation (no new screen look), once it fits
                self.carry["pending_confirm"] = {"candidates": [c.to_dict() for c in adv],
                                                 "screen": screen}
                self.audit.append("confirm_deferred", batch=b, candidates=ids, reason=why,
                                  carry=_json(self.carry))
                raise ConfirmDeferred(why)
        self.carry.pop("pending_confirm", None)
        k, cd = self._confirmation(b, cand, inc, ids, "decision", "adopt")
        late = sorted(set(ids) & self.blocked())
        if cd["adopt"] and late:  # retired while the confirmation ran: never adopted
            cd = {**cd, "adopt": False, "reasons": [f"retired before adoption: {late}"]}
        if cd["adopt"] and self.cfg.production:
            self.audit.append("confirm_close", batch=b, k=k, level="decision", adopt=False,
                              passed=True, result=cd, candidates=ids,
                              version=cand.version_id,
                              reasons=["production: a conversation-level confirmation "
                                       "decides adoption"])
            k, cd = self._confirmation(b, cand, inc, ids, "conversation", "adopt")
            level = "conversation"
            late = sorted(set(ids) & self.blocked())
            if cd["adopt"] and late:
                cd = {**cd, "adopt": False, "reasons": [f"retired before adoption: {late}"]}
        else:
            level = "decision"
        if cd["adopt"]:
            self.audit.append("confirm_close", batch=b, k=k, level=level, adopt=True,
                              passed=True, result=cd, candidates=ids,
                              version=cand.version_id, pointer=cand.version_id,
                              parent=inc.version_id, screen=screen)
            summary["adopted"] = ids
            self.carry = self._empty_carry()
        else:
            self.audit.append("confirm_close", batch=b, k=k, level=level, adopt=False,
                              passed=False, result=cd, candidates=ids,
                              version=cand.version_id, reasons=cd["reasons"])

    def _confirmation(self, b: int, cand: HarnessState, inc: HarnessState, ids: list[str],
                      level: str, purpose: str) -> tuple[int, dict[str, Any]]:
        """One confirmation: k allocated by ``confirm_open`` before any dispatch; candidate
        vs ``inc`` (and root when it differs). The caller writes ``confirm_close``."""
        root_v = self.root_version()
        root_is_inc = root_v == inc.version_id
        k = len(self.audit.rows("confirm_open")) + 1
        n = self.cfg.confirm.units if level == "decision" else self.cfg.conversation_units
        cu = self._sample(b, "confirm", n, k, level=level)
        manifest = hashlib.sha256("\n".join(sorted(u.unit_id for u in cu)).encode()).hexdigest()
        self.audit.append("confirm_open", batch=b, k=k, level=level, purpose=purpose,
                          alpha_k=G.alpha_k(self.cfg.confirm.alpha, k),
                          candidates=ids, candidate_version=cand.version_id,
                          incumbent=inc.version_id, root=root_v, units=n,
                          units_sha256=manifest)
        arms = [(inc, "incumbent"), (cand, "candidate")]
        if not root_is_inc:
            arms.append((self.versions.load(root_v), "root"))
        try:
            outs = self._run(b, "confirm", [RunRequest(u, s, t) for u in cu for s, t in arms],
                             level=level,
                             k=k, incumbent=inc)
        except (CycleHalted, SystemHalted) as e:
            self.audit.append("confirm_close", batch=b, k=k, level=level, adopt=False,
                              void=True, passed=False, reason=f"halted: {e}",
                              reasons=[f"halted: {e}"], candidates=ids,
                              version=cand.version_id)
            raise
        cd = G.confirm_decide(outs, k, self.cfg.confirm, n_expected=n,
                              root_is_incumbent=root_is_inc)
        cd["level"] = level
        hold = self._confirm_hold(outs)
        if hold:  # never adopt with a candidate-only critical event: human review only
            passed = bool(cd["adopt"])
            cd["adopt"], cd["hold"] = False, True
            cd["hold_units"] = list(dict.fromkeys(h["unit"] for h in hold))
            cd["reasons"] = list(cd.get("reasons") or []) + [(
                f"HOLD: candidate-only critical event(s) on {len(cd['hold_units'])} unit(s); "
                "human review needed (review-decide)")]
            R = int(self.cfg.hold_replicates)
            packet = {"k": k, "level": level, "candidates": ids,
                      "candidate_version": cand.version_id, "incumbent": inc.version_id,
                      "root": root_v, "confirmation_passed_own_tests": passed,
                      "confirmation_reasons": cd["reasons"], "events": hold,
                      "auto_release": RP.NEVER,
                      "replication": {"R": R, "status": "pending" if R > 0 and passed else
                                      ("off (hold_replicates=0)" if R <= 0 else
                                       "not run: the confirmation failed its own tests")}}
            name = f"hold-k{k}.json"
            sha = self._write_packet(name, packet)
            self.audit.append("critical_hold", batch=b, k=k, candidates=ids,
                              units=cd["hold_units"], packet=name, packet_sha256=sha,
                              passed=passed, level=level)
            self._alert("critical_hold", batch=b, k=k, candidates=ids,
                        units=cd["hold_units"])
            # written BEFORE replication: a crash or halt mid-replication leaves the hold
            # and the original packet; replication only ever adds evidence, never releases
            if R > 0 and passed:
                self._replicate_evidence(b, k, level, cu, cand, inc, root_v, ids, hold,
                                         packet)
        return k, cd

    def _write_packet(self, name: str, packet: dict[str, Any]) -> str:
        pdir = self.root / "reviews"
        pdir.mkdir(exist_ok=True)
        tmp = pdir / f".{name}.tmp"
        tmp.write_text(json.dumps(_json(packet), indent=1))
        os.replace(tmp, pdir / name)
        return hashlib.sha256((pdir / name).read_bytes()).hexdigest()

    def _replicate_evidence(self, b: int, k: int, level: str, cu: list[Unit],
                            cand: HarnessState, inc: HarnessState, root_v: str,
                            ids: list[str], hold: list[dict[str, Any]],
                            packet: dict[str, Any]) -> None:
        """Replication as EVIDENCE (s. 20; owner 2026-10-24 "human review only"): R metered
        runs of the candidate (once per unit) and of EVERY discordant reference per held
        unit, in the same serving window when it fits (``confirm_guard``). The result is
        written to a new packet ``hold-k<k>-evidence.json`` (``hold_packet`` record); the
        hold NEVER clears here. Errored / unscored / unlabelled replicates are missing."""
        R = int(self.cfg.hold_replicates)
        units = {u.unit_id: u for u in cu}
        states = {"incumbent": inc, "candidate": cand}
        refs = ["incumbent"]
        if root_v != inc.version_id:
            states["root"] = self.versions.load(root_v)
            refs.append("root")
        arms = RP.plan(hold, refs)
        reqs = [RunRequest(units[uid], states[arm], RP.tag(arm, i))
                for uid, unit_arms in arms.items() for arm in unit_arms for i in range(R)]
        res: dict[str, Any]
        why = self.confirm_guard(len(reqs)) if self.confirm_guard is not None else None
        if why:
            res = {"skipped": f"does not fit the serving window: {why}"}
        else:
            try:
                outs = self._run(b, "replicate", reqs, level=level, k=k, incumbent=inc)
            except (BudgetRefused, BudgetExceeded, CycleHalted) as e:
                res = {"skipped": f"not run: {e}"}
            else:
                res = {"runs": len(reqs), **RP.evidence(hold, outs, R, arms)}
        res = {"release": False, "auto_release": RP.NEVER, "R": R,
               "seed_note": RP.SEED_NOTE, "arms": arms, **res}
        self.audit.append("hold_replication", batch=b, k=k, candidates=ids, **_json(res))
        name = f"hold-k{k}-evidence.json"
        sha = self._write_packet(name, {**packet, "replication": {"status": (
            "skipped" if "skipped" in res else "attached"), **res}})
        self.audit.append("hold_packet", batch=b, k=k, packet=name, packet_sha256=sha,
                          note="replication evidence attached (never releases)")

    def certify_current(self, b: int | None = None) -> dict[str, Any]:
        """Conversation-level confirmation of the CURRENT version (adopted in development)
        against its parent and the root, before production use. Pass: the version is
        certified. Fail: the pointer goes back to the nearest certified ancestor."""
        if self.halted():
            raise SystemHalted(f"latched: {self.halted()['reason']}")
        cur_v = self.current_version()
        if self.certified(cur_v):
            return {"version": cur_v, "certified": True, "ran": False}
        b = self.next_batch() if b is None else b
        self.cycle_committed = 0.0
        self.audit.append("cycle_start", batch=b, purpose="certify")
        cur = self.current()
        parent_v = self.parent_of(cur_v)
        parent = self.versions.load(parent_v)
        ids = sorted(_ids(cur) - _ids(parent))
        out: dict[str, Any] = {"version": cur_v, "ran": True}
        try:
            self.check_invariants()
            k, cd = self._confirmation(b, cur, parent, ids, "conversation", "certify")
        except CycleHalted as e:
            self.audit.append("cycle_end", batch=b, status="halted", purpose="certify")
            return {**out, "certified": False, "halted": str(e)}
        except BaseException:
            self.audit.append("cycle_end", batch=b, status="error", purpose="certify")
            raise
        ok = bool(cd["adopt"])
        self.audit.append("confirm_close", batch=b, k=k, level="conversation", adopt=False,
                          purpose="certify", passed=ok, result=cd, candidates=[],
                          version=cur_v, reasons=cd["reasons"])
        if not ok:
            target = parent_v
            while not self.certified(target):
                target = self.parent_of(target)
            self.audit.append("rollback", batch=b, pointer=target, parent=cur_v,
                              reason="certification failed (conversation level)",
                              evidence={"k": k, "reasons": cd["reasons"]})
            self._alert("rollback", batch=b, to=target, frm=cur_v, reason="certification")
            self.carry = self._empty_carry()
        self.audit.append("cycle_end", batch=b, status="ok", purpose="certify")
        return {**out, "certified": ok, "k": k, "reasons": cd["reasons"]}

    def run(self, batches: int, start: int | None = None) -> list[dict[str, Any]]:
        """Continuous driver. Resumes at the next batch; stops durably on any exception."""
        out = []
        b0 = self.next_batch() if start is None else start
        for b in range(b0, b0 + batches):
            try:
                out.append(self.cycle(b))
            except (BudgetRefused, BudgetExceeded) as e:
                self.audit.append("cycle_end", batch=b, status="stopped")
                self.audit.append("stop", batch=b, ledger_charged_usd=self._charged(), reason=f"budget: {e}")
                break
            except ProductionRefused as e:
                self.audit.append("stop", batch=b, ledger_charged_usd=self._charged(), reason=f"production: {e}")
                self._alert("production_refused", batch=b, reason=str(e))
                break
            except SystemHalted as e:
                if self.audit.last("cycle_start", "cycle_end")["kind"] == "cycle_start":
                    self.audit.append("cycle_end", batch=b, status="system_halt")
                self.audit.append("stop", batch=b, ledger_charged_usd=self._charged(), reason=f"system halt: {e}")
                break
            except InvariantBroken as e:
                self._system_halt(f"invariant: {e}", batch=b)
                self.audit.append("cycle_end", batch=b, status="system_halt")
                self.audit.append("stop", batch=b, ledger_charged_usd=self._charged(), reason=f"invariant: {e}")
                raise
            except Exception as e:
                self.audit.append("cycle_end", batch=b, status="error")
                self.audit.append("stop", batch=b, ledger_charged_usd=self._charged(), reason=f"error: {type(e).__name__}: {e}")
                self._alert("error", batch=b, error=repr(e))
                raise
        return out
