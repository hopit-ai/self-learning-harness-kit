"""Replication at the confirmation HOLD: EVIDENCE ONLY (owner 2026-10-24, "human review
only"; design s. 20).

A HOLD is a candidate-only critical event at confirmation (on a unit the candidate arm made
a critical write and a reference arm, the incumbent or the root, did not) or a candidate run
with an unresolved critical label. The critical margin is ZERO: any candidate showing more
critical events than a reference waits for human review. A zero margin cannot be certified
statistically (s. 20: even +0.01 pt harms pass a hold at 0.1-0.3 per confirmation), so no
replicate outcome ever releases a hold. The hold clears only through an audited human
decision (``slh.system.loop.review_decide``; CLI ``review-decide``).

When configured (``hold_replicates`` R > 0, default 0), both arms of every held unit are
re-run R times and the counts are attached to the review packet:

* the candidate arm once per unit (deduplicated across references);
* EVERY discordant reference arm of the unit (incumbent and root; for an unresolved
  candidate label, every reference arm, since discordance is unknown).

Replicates reuse the case seed (``per_unit_seed`` or the registered phase seed): they are
repeated serving calls on the same unit, not independent units. Any replicate that is not
``scored`` with a resolved critical label (absent, errored, unscored, unknown label) is
MISSING and listed as such. A violation kind the candidate shows on a unit (original or
replicate) that no reference arm shows there (original or replicate) is flagged."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from slh.system.interfaces import Outcome

RULES = ("ref_reproduces", "ref_ge_cand")  # pinned legacy key; no effect (never releases)

SEED_NOTE = ("replicates reuse the case seed (per_unit_seed / the registered phase seed): "
             "repeated serving calls on the same unit, NOT independent units")
NEVER = "never: replication is evidence only; a human decision clears a hold (owner 2026-10-24)"


def tag(arm: str, i: int) -> str:
    return f"rep{i}-{arm}"


def kind_of(trace: Mapping[str, Any] | None) -> str | None:
    return dict(trace or {}).get("violation")


def plan(hold: Sequence[Mapping[str, Any]], references: Sequence[str]) -> dict[str, list[str]]:
    """Unit -> arms to replicate: ``candidate`` plus every discordant reference of the unit
    (every reference in ``references`` when a row is an unresolved candidate label)."""
    out: dict[str, list[str]] = {}
    for row in hold:
        arms = out.setdefault(row["unit"], ["candidate"])
        refs = [row["reference"]] if row.get("reference") else list(references)
        for ref in refs:
            if ref not in arms:
                arms.append(ref)
    return out


def usable(o: Outcome | None) -> bool:
    return o is not None and o.status == "scored" and o.critical_harm is not None


def evidence(hold: Sequence[Mapping[str, Any]], outs: Sequence[Outcome], R: int,
             arms: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    """Descriptive per-unit counts for the review packet. Never a release decision."""
    by: dict[tuple[str, str], Outcome] = {(o.unit_id, o.tag): o for o in outs}
    rows_by: dict[str, list[Mapping[str, Any]]] = {}
    for row in hold:
        rows_by.setdefault(row["unit"], []).append(row)
    units, missing_total, flagged = [], 0, []
    for uid, unit_arms in arms.items():
        rec: dict[str, Any] = {"unit": uid, "arms": {}, "reasons": sorted(
            {r.get("reason", "") for r in rows_by.get(uid, [])})}
        kinds: dict[str, set[str]] = {}
        for row in rows_by.get(uid, []):  # the original confirmation events
            if row.get("candidate_trace") is not None and kind_of(row["candidate_trace"]):
                kinds.setdefault("candidate", set()).add(kind_of(row["candidate_trace"]))
            if row.get("reference") and row.get("reference_label") is True \
                    and kind_of(row.get("reference_trace")):
                kinds.setdefault(row["reference"], set()).add(kind_of(row["reference_trace"]))
        for arm in unit_arms:
            n_crit, missing, ks = 0, [], Counter()
            for i in range(R):
                o = by.get((uid, tag(arm, i)))
                if not usable(o):
                    missing.append({"i": i, "status": None if o is None else o.status,
                                    "critical_harm": None if o is None else o.critical_harm})
                    continue
                if o.critical_harm is True:
                    n_crit += 1
                    k = kind_of(o.trace) or "unknown-kind"
                    ks[k] += 1
                    kinds.setdefault(arm, set()).add(k)
            missing_total += len(missing)
            rec["arms"][arm] = {"R": R, "critical": n_crit, "usable": R - len(missing),
                                "missing": missing, "kinds": dict(ks)}
        ref_kinds = set().union(*[v for a, v in kinds.items() if a != "candidate"]) \
            if any(a != "candidate" for a in kinds) else set()
        new = sorted(kinds.get("candidate", set()) - ref_kinds)
        rec["candidate_kinds_no_reference_shows"] = new
        if new:
            flagged.append({"unit": uid, "kinds": new})
        units.append(rec)
    return {"release": False, "auto_release": NEVER, "R": R, "seed_note": SEED_NOTE,
            "missing_replicates": missing_total,
            "missing_note": ("replicates that are absent, errored, unscored or carry an "
                             "unresolved critical label are MISSING (not counted as clean)"),
            "kinds_flagged": flagged, "units": units}
