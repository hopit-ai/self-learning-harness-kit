"""Frozen sealed-test arms; all injected text uses demo1's single notes slot.

Removal is B1 by construction: reuse B1 rows and flag the reuse. It is not an
independent intervention or evidence of a causal fall after skill removal.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slh.demo1.schema import REPO

WRITERS = ("T", "S", "TS")
SNAPSHOTS = (12, 24, 48)
HUMAN = REPO / "demo1/human-procedure-address.md"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class Arm:
    name: str
    text: str = ""
    writer: str | None = None
    snapshot: int | None = None
    kind: str = "notes"
    reuse_b1: bool = False

    @property
    def notes(self) -> tuple[str, ...]:
        return (self.text,) if self.text else ()

    def record(self) -> dict[str, Any]:
        return {**asdict(self), "skill_sha256": sha(self.text) if self.text else None,
                "insertion": "demo1 HarnessState notes, before think instruction"}


def budget_schedule(case_ids: list[str], *, matched_usd: float,
                    attempt_usd: float, selector_usd: float, seed: int = 9109
                    ) -> dict[str, Any]:
    """Budget includes learning, writer/gate, endpoint inference and selection.

    Pairwise selection costs (N-1) selectors for N independent attempts. Freeze
    a shuffled allocation of floor(N) / ceil(N) attempts. Fractional remainder
    is rounded down over the whole test so the schedule never exceeds budget;
    randomization is over case IDs only and never observes test labels.
    """
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("schedule needs unique test case IDs")
    if any(not math.isfinite(v) for v in (matched_usd, attempt_usd, selector_usd)) \
            or attempt_usd <= 0 or selector_usd < 0:
        raise ValueError("budget units must be finite and attempt cost positive")
    per_case = matched_usd / len(case_ids)
    if per_case < attempt_usd:
        raise ValueError("matched budget cannot buy one attempt per test case")
    mean = (per_case + selector_usd) / (attempt_usd + selector_usd)
    low = math.floor(mean)
    extra = min(len(case_ids), math.floor((mean - low) * len(case_ids) + 1e-9))
    shuffled = sorted(case_ids)
    random.Random(seed).shuffle(shuffled)
    allocation = {cid: low + int(i < extra) for i, cid in enumerate(shuffled)}
    calls = sum(allocation.values())
    selectors = calls - len(case_ids)
    scheduled = calls * attempt_usd + selectors * selector_usd
    return {"schema": "poc9-budget-schedule-v1", "seed": seed,
            "attempts": allocation, "mean_attempts_target": mean,
            "mean_attempts_frozen": calls / len(case_ids), "matched_usd": matched_usd,
            "attempt_usd": attempt_usd, "selector_usd": selector_usd,
            "scheduled_usd": scheduled, "unspent_rounding_usd": matched_usd - scheduled,
            "candidate_calls": calls, "selector_calls_max": selectors,
            "selector": "same 9B, pairwise tournament, no test labels",
            "rounding": "frozen randomized allocation; floor aggregate fractional attempts"}


def definitions(snapshots: dict[str, dict[int, str]], generic: dict[str, str],
                history: str, *, human_path: Path = HUMAN,
                active: tuple[str, ...] | list[str] = WRITERS) -> list[Arm]:
    """B1 and the shared controls always; per-writer arms only for ``active``
    writers (a writer whose required generic control failed is dropped whole)."""
    active = tuple(w for w in WRITERS if w in active)
    if not active:
        raise ValueError("no writer has its required generic control")
    if set(snapshots) != set(active) or set(generic) != set(active):
        raise ValueError("every active writer needs its snapshots and generic control")
    arms = [Arm("b1", snapshot=0)]
    for writer in active:
        if set(snapshots[writer]) != {0, *SNAPSHOTS} or snapshots[writer][0]:
            raise ValueError("snapshots must be empty at 0 and present at 12/24/48")
        arms.extend(Arm(f"learned_{writer}@{n}", snapshots[writer][n], writer, n)
                    for n in SNAPSHOTS)
    from slh.demo1.learn import human_procedure

    arms.extend((Arm("full_context", history),
                 Arm("human", human_procedure("address", human_path.parent))))
    arms.extend(Arm(f"best_of_n_{w}", writer=w, snapshot=48, kind="best_of_n")
                for w in active)
    arms.extend(Arm(f"generic_{w}", generic[w], w, 48) for w in active)
    arms.extend(Arm(f"removal_{w}", writer=w, snapshot=0, reuse_b1=True) for w in active)
    return arms


def freeze(arms: list[Arm], schedule: dict[str, Any], *, seed: int = 9110
           ) -> dict[str, Any]:
    if len({a.name for a in arms}) != len(arms):
        raise ValueError("duplicate arm")
    order = [a.name for a in arms if not a.reuse_b1]
    random.Random(seed).shuffle(order)
    body = {"schema": "poc9-freeze-v1", "arms": [a.record() for a in arms],
            "arm_order": order, "order_seed": seed, "best_of_n": schedule,
            "removal_note": "identical to B1 by construction; reused rows, zero new calls"}
    body["sha256"] = sha(_canonical(body))
    return body


def _canonical(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def verify(frozen: dict[str, Any]) -> list[Arm]:
    body = {k: v for k, v in frozen.items() if k != "sha256"}
    if sha(_canonical(body)) != frozen["sha256"]:
        raise ValueError("frozen arms or schedule changed")
    result = []
    for row in frozen["arms"]:
        arm = Arm(**{k: row[k] for k in Arm.__dataclass_fields__})
        if row["skill_sha256"] != (sha(arm.text) if arm.text else None):
            raise ValueError("frozen skill hash differs")
        result.append(arm)
    return result
