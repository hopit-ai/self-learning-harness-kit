"""BFCL multi-turn data read as plain JSON (no ``bfcl_eval`` import; laptop-safe).

Terms used across the track:

- **root**: an originating scenario, the index ``i`` of ``multi_turn_base_<i>``
  (0..199);
- **siblings**: the augmented entries built from the same root, ``multi_turn_<cat>_<i>``
  for cat in long_context, miss_func, miss_param (index alignment audited in
  ``slh.xp.bfcl.splits``); a root and its siblings always share a partition;
- **instance**: one (root, category) pair that is run; the primary claim uses
  ``PRIMARY = (base, long_context)`` only (miss_func / miss_param reward asking for
  missing parameters or obtaining functions, which is out of scope for
  "learning beats asking");
- **domain** (stratum): the sorted tuple of the root's ``involved_classes``;
- **family**: one of four coarse domain families (``FAMILY_ORDER``), the context of the
  F' analysis: the first family in that order whose classes intersect the domain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slh.xp.bfcl import vendor

CATEGORIES = ("base", "long_context", "miss_func", "miss_param")
PRIMARY = ("base", "long_context")
N_ROOTS = 200
VERSION_PREFIX = "BFCL_v4"
FAMILY_ORDER = ("files", "vehicle", "finance", "travel")
FAMILIES = {"files": ("GorillaFileSystem", "MathAPI"), "vehicle": ("VehicleControlAPI",),
            "finance": ("TradingBot",), "travel": ("TravelAPI",)}
# Remaining classes (MessageAPI, TwitterAPI, TicketAPI) never occur alone in the pinned
# data; ``family_of`` refuses a domain that matches no family.


def package_dir(package: Path | None = None) -> Path:
    return package or vendor.PACKAGE


def entry_file(category: str, *, answers: bool = False) -> str:
    sub = "possible_answer/" if answers else ""
    return f"data/{sub}{VERSION_PREFIX}_multi_turn_{category}.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_category(category: str, package: Path | None = None, *, answers: bool = False
                  ) -> list[dict[str, Any]]:
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")
    return load_jsonl(package_dir(package) / entry_file(category, answers=answers))


def instance_id(root: int, category: str) -> str:
    if category not in CATEGORIES or not 0 <= int(root) < N_ROOTS:
        raise ValueError(f"bad instance ({root}, {category})")
    return f"multi_turn_{category}_{int(root)}"


def parse_instance(instance: str) -> tuple[int, str]:
    if not instance.startswith("multi_turn_"):
        raise ValueError(f"not a multi-turn id: {instance!r}")
    head, _, idx = instance[len("multi_turn_"):].rpartition("_")
    if head not in CATEGORIES or not idx.isdigit():
        raise ValueError(f"not a multi-turn id: {instance!r}")
    return int(idx), head


def domain_of(entry: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(entry["involved_classes"]))


def family_of(domain: tuple[str, ...] | list[str]) -> str:
    for fam in FAMILY_ORDER:
        if set(domain) & set(FAMILIES[fam]):
            return fam
    raise ValueError(f"domain {domain} matches no family")


def roots(package: Path | None = None) -> list[dict[str, Any]]:
    """One row per root: index, ids of every sibling, domain, family, turns."""
    base = load_category("base", package)
    if [e["id"] for e in base] != [instance_id(i, "base") for i in range(N_ROOTS)]:
        raise ValueError("base ids are not multi_turn_base_0..199 in order")
    out = []
    for i, e in enumerate(base):
        dom = domain_of(e)
        out.append({"root": i, "domain": list(dom), "family": family_of(dom),
                    "turns": len(e["question"]),
                    "instances": {c: instance_id(i, c) for c in CATEGORIES}})
    return out
