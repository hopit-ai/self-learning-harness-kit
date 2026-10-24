"""The F' lesson pool: phase-1 T+ lessons, policy-screened, deduplicated by a fixed rule.

    python -m slh.xp.fgate.pool --bundles package-p2/bundles.json --out POOL.json

Source: the 12 T+ endpoint libraries chosen by the phase-1 sentinel rule, as frozen in a
phase-2 ``bundles.json`` (``xp-phase2-bundles-v1``; every text is checked against its
SHA-256). Only the ``learned`` kind is read (never the search control). Each library
holds up to three texts (address, variant, sequence); a lesson is one ``- WHEN ... THEN
...`` line.

Pre-declared steps (docs/xp/partner-c-fgate-plan.md, section 3.1):

1. **Policy screen, per lesson, before clustering.** A lesson is dropped (never edited)
   when it licenses a write without the confirmation the benchmark policy requires: its
   text waives confirmation ("without asking", "do not ask for confirmation", "not to
   ask again", ...) AND its action executes a write AND its condition does not require
   that the user confirmed the specific change (the specific address, variant or change
   details). A condition that says "even if ... not yet shown" or "general request"
   never counts as specific. Edited versions would be new candidates (separate batch).
   These are the prior-approval lessons that conflict with tau2's explicit-confirmation
   rule ("list the action details and obtain explicit user confirmation (yes)"); dropping
   them from the pool is an owner decision (approved 2026-10-18).
2. **Clustering.** Every surviving lesson gets one intent by an ordered list of fixed
   keyword rules (first match wins, ``INTENTS``). A lesson that matches no rule forms
   its own cluster (``unclustered``, reported).
3. **Representative.** Per cluster, one verbatim member: the most frequent normalized
   wording; ties go to a well-formed wording (no repeated ``THEN`` clause, no doubled
   ``WHEN``), then to the lowest run id, then to the earliest position.

Lessons are rendered in the pool's canonical order (``L01``, ``L02``, ...) as one
``## Lessons`` note in B1's notes slot; the empty selection is B1's prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

POOL_SCHEMA = "xp-fgate-pool-v1"
SCREEN_VERSION = "fgate-policy-screen-v1"
CLUSTER_VERSION = "fgate-intent-rules-v1"
TASKS = ("address", "variant", "sequence")
TASK_TOOLS = {"address": ("modify_pending_order_address",),
              "variant": ("modify_pending_order_items", "exchange_delivered_order_items"),
              "sequence": ("modify_pending_order_address", "modify_pending_order_items")}

_WAIVER = re.compile(
    r"without (?:asking|any|further|additional)?\s*(?:for )?(?:further |additional )?"
    r"confirmation|without asking|do not ask (?:for )?(?:further |additional |again|"
    r"confirmation|for confirmation)|not to ask (?:for confirmation )?again|"
    r"no further confirmation|do not ask for (?:further |additional )?confirmation|"
    r"don'?t ask (?:again|for confirmation)", re.IGNORECASE)
_EXEC = r"(?<!not )(?<!or )(?<!never )\bexecute\b"
_EXECUTES = re.compile(_EXEC, re.IGNORECASE)
_SPECIFIC = re.compile(
    r"specific (?:change|address|variant|details)|including the (?:identified variant|"
    r"target address)|confirmed the specific", re.IGNORECASE)
_NOT_SPECIFIC = re.compile(r"\beven if\b|general request|not yet (?:been )?(?:shown|identified)",
                           re.IGNORECASE)

# (intent id, task, where, pattern): ``where`` is "when", "then" or "any". Ordered:
# first match wins within the lesson's task.
INTENTS: tuple[tuple[str, str, str, str], ...] = (
    ("address-exec-after-specific-approval", "address", "then", _EXEC),
    ("address-clarify-nonunique", "address", "then", r"\b(?:ask|clarify|inform)\b"),
    ("variant-exec-after-specific-approval", "variant", "then", _EXEC),
    ("variant-clarify-multiple", "variant", "when", r"\bmultiple\b|\bambiguous\b"),
    ("variant-none-inform", "variant", "when",
     r"no (?:available )?(?:matching )?variant|no matching|(?:does not|doesn'?t) match"),
    ("variant-confirm-unconfirmed", "variant", "when",
     r"not yet|has not|have not|general request|not (?:yet )?(?:explicitly )?confirmed"),
    ("sequence-address-first", "sequence", "then", r"process(?:es|ing)? the address change first"),
    ("sequence-ignore-skip-instruction", "sequence", "any",
     r"ignore (?:this|the) instruction|no need to ask|don'?t confirm"),
    ("sequence-terse-confirmation", "sequence", "then", r"internal reasoning|policy explanation"),
    ("sequence-list-only-first", "sequence", "then",
     (r"only (?:the )?first|ONLY the first|single next action|next single action|"
      r"for the first change|current action|focus solely")),
    ("sequence-confirm-second-separately", "sequence", "any",
     (r"second change|separate transaction|do not bundle|confirms one change|"
      r"confirms the first change")),
    ("sequence-confirm-each-write", "sequence", "then",
     r"explicit (?:user )?confirmation|explicit 'yes'|explicitly confirms"),
)


class PoolRefused(ValueError):
    pass


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_lesson(line: str) -> tuple[str, str]:
    """(condition, action) of one ``- WHEN ... THEN ...`` line."""
    body = re.sub(r"^\s*-\s*", "", line).strip()
    body = re.sub(r"^(?:WHEN\s+)+", "", body)
    if " THEN " not in body:
        return body, ""
    when, then = body.split(" THEN ", 1)
    return when.strip(), then.strip()


def lessons_of(text: str) -> list[str]:
    return [ln.rstrip() for ln in (text or "").splitlines() if ln.startswith("- ")]


def policy_conflict(line: str) -> str | None:
    """The reason a lesson licenses a write without the required confirmation, else None."""
    when, then = split_lesson(line)
    if not (_WAIVER.search(line) and _EXECUTES.search(then)):
        return None
    if _SPECIFIC.search(when) and not _NOT_SPECIFIC.search(when):
        return None
    return "licenses a write without the user's confirmation of the specific change"


def intent(task: str, line: str) -> str | None:
    when, then = split_lesson(line)
    for name, t, where, pattern in INTENTS:
        if t != task:
            continue
        hay = {"when": when, "then": then, "any": f"{when} {then}"}[where]
        if re.search(pattern, hay, re.IGNORECASE):
            return name
    return None


def _norm(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().lower()


def well_formed(line: str) -> bool:
    _, then = split_lesson(line)
    doubled_when = bool(re.match(r"^\s*-\s*WHEN\s+(?:When\s+)?WHEN\b", line))
    repeated_then = " THEN " in then
    return not (doubled_when or repeated_then)


def load_libraries(bundles_path: Path) -> tuple[list[dict[str, Any]], str]:
    """Every learned T+ text, hash-checked: rows {run, task, text, sha256}."""
    raw = Path(bundles_path).read_bytes()
    b = json.loads(raw)
    if b.get("schema") != "xp-phase2-bundles-v1" or b.get("writer") != "T+":
        raise PoolRefused("bundles: expected xp-phase2-bundles-v1 with writer T+")
    if b.get("selection") != "phase1-sentinel":
        raise PoolRefused("bundles: texts must be the phase-1 sentinel selection")
    rows = []
    learned = (b.get("kinds") or {}).get("learned") or {}
    for run in sorted(learned, key=int):
        texts, hashes = learned[run]["texts"], learned[run]["sha256"]
        if len(texts) != len(TASKS):
            raise PoolRefused(f"run {run}: expected {len(TASKS)} texts")
        for task, text, sha in zip(TASKS, texts, hashes, strict=True):
            if (_sha(text) if text else None) != sha:
                raise PoolRefused(f"run {run} {task}: text differs from its sha256")
            rows.append({"run": int(run), "task": task, "text": text or "", "sha256": sha})
    return rows, _sha(raw.decode("utf-8"))


def build_pool(libraries: list[dict[str, Any]], *, source_sha256: str = "") -> dict[str, Any]:
    members: dict[str, list[dict[str, Any]]] = {}
    excluded, unclustered, total = [], [], 0
    for lib in libraries:
        for pos, line in enumerate(lessons_of(lib["text"])):
            total += 1
            ref = {"run": lib["run"], "task": lib["task"], "pos": pos, "text": line}
            if why := policy_conflict(line):
                excluded.append({**ref, "reason": why})
                continue
            name = intent(lib["task"], line)
            if name is None:
                name = f"unclustered-{lib['task']}-r{lib['run']}-p{pos}"
                unclustered.append(ref)
            members.setdefault(name, []).append(ref)
    order = [n for n, *_ in INTENTS if n in members] + sorted(
        n for n in members if n.startswith("unclustered-"))
    lessons = []
    for i, name in enumerate(order, 1):
        group = members[name]
        freq = Counter(_norm(m["text"]) for m in group)
        rep = min(group, key=lambda m: (-freq[_norm(m["text"])], not well_formed(m["text"]),
                                        m["run"], m["pos"]))
        task = group[0]["task"]
        lessons.append({
            "id": f"L{i:02d}", "intent": name, "task": task, "tools": list(TASK_TOOLS[task]),
            "text": rep["text"], "sha256": _sha(rep["text"]),
            "representative": {k: rep[k] for k in ("run", "task", "pos")},
            "members": [{k: m[k] for k in ("run", "task", "pos")} for m in group],
            "n_members": len(group), "runs": sorted({m["run"] for m in group})})
    body = {"schema": POOL_SCHEMA, "screen": SCREEN_VERSION, "clustering": CLUSTER_VERSION,
            "source_bundles_sha256": source_sha256, "lessons_read": total,
            "excluded": excluded, "unclustered": unclustered, "lessons": lessons}
    body["sha256"] = _sha(json.dumps({k: v for k, v in body.items()}, sort_keys=True))
    return body


def verify_pool(pool: dict[str, Any]) -> dict[str, Any]:
    if pool.get("schema") != POOL_SCHEMA:
        raise PoolRefused("pool: unknown schema")
    body = {k: v for k, v in pool.items() if k != "sha256"}
    if _sha(json.dumps(body, sort_keys=True)) != pool.get("sha256"):
        raise PoolRefused("pool: content differs from its sha256")
    for les in pool["lessons"]:
        if _sha(les["text"]) != les["sha256"]:
            raise PoolRefused(f"pool: {les['id']} text differs from its sha256")
    return pool


def render(lessons: list[dict[str, Any]], include: list[int] | tuple[int, ...]) -> str:
    """The notes text for an inclusion vector (canonical order; empty -> B1)."""
    if len(include) != len(lessons):
        raise ValueError("inclusion vector length differs from the pool")
    chosen = [les["text"] for les, x in zip(lessons, include, strict=True) if x]
    return "## Lessons\n" + "\n".join(chosen) if chosen else ""


def same_tool(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return bool(set(a["tools"]) & set(b["tools"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.fgate.pool")
    ap.add_argument("--bundles", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    libs, sha = load_libraries(args.bundles)
    pool = build_pool(libs, source_sha256=sha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pool, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"lessons_read": pool["lessons_read"], "excluded": len(pool["excluded"]),
                      "pool": [(x["id"], x["intent"], x["n_members"]) for x in pool["lessons"]],
                      "unclustered": len(pool["unclustered"]), "sha256": pool["sha256"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
