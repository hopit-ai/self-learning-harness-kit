"""S1 headroom (oracle) probe for System One decision types 1-3 on Hermes + frozen Qwen3.5-9B,
tau2 retail, shipped in the M-H1 partner package (design: docs/xp/s1-headroom-probe-design.md).

It is a SEPARATE stage of the M-H1 run (``probe``, after ``trajectory``), with its own plan
file in the package (``probe-plan.json``, its own hash): the M-H1 plan, its A/B/C/H units,
seeds, sealed outcomes and blind machinery are untouched. Probe episodes run the empty state
(plain Hermes) on non-confirm units only (never a held-out eval or confirm user).

Mechanism (one decision = one boundary of one base episode):

1. **Base run (record).** A normal Hermes episode whose spend gate records every agent
   request (normalised digest) and response (raw bytes, kept on the node only), and whose
   runner reports the conversation at every Hermes post-tool compression boundary
   (``turn_tool_round.compress_after_tool_results``, where Hermes' proactive tool-result
   prune runs). Nothing is edited.
2. **Decisions.** At chosen boundaries: (1) context relevance: which bulky read-only tool
   results to keep (random groups, Hermes' own prune passes, a matched-budget heuristic, the
   prompted 9B's choice); (3) recovery after a read-only tool failure: native / retry /
   repair / stop (natural failures, plus an injected transient-failure stratum).
3. **Branches (paired continuations).** A fresh episode of the same unit whose gate replays
   the recorded responses for the first ``n_calls`` requests (each checked against the
   recorded digest), whose customer replays the recorded customer turns, and whose task tools
   run against a FRESH database (the prefix's calls re-executed: the environment snapshot is
   identical, verified by the boundary digest). At the boundary the option's edits are
   applied (tool-message CONTENT only), then the model continues live. Repeated ``keep`` /
   ``native`` branches measure serving noise.

Type 2 (review usefulness) is label schema only: rows built at export from M-H1's own
evidence episodes (B and C) and C's audit log, plus the prompted 9B's choice (c9b block).

Never deleted: user and assistant messages (consent, constraints), call messages and their
ids (call/result pairing), write-tool results, the pending (unread) round, identifiers
(a stub keeps every id found in the result).
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import random
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slh.system.hosts.hermes.meter import SpendGate

PROBE_SCHEMA = "slh-s1probe-plan-v1"
PROBE_ID = "s1probe-2026-10-25-v1"
PROBE_SALT = "s1probe-v1"
PLAN_FILE = "probe-plan.json"
DESIGN_DOC = "docs/xp/s1-headroom-probe-design.md"
BLOCKS = ("c9b", "dry")  # c9b only at the partner (cost; design s. 2); "dry" = CPU dry run
READ_ONLY = frozenset({"calculate", "find_user_id_by_email", "find_user_id_by_name_zip",
                       "get_item_details", "get_order_details", "get_product_details",
                       "get_user_details", "list_all_product_types"})
STUB_TAG = "slh-pruned"
PROMPT_TAG = "SLH-PROBE-DECISION"
INJECTED_ERROR = ('{"error": "temporarily unavailable (503): the request was not processed; '
                  'it may succeed if sent again"}')
NOTES = {
    "retry": "[harness recovery note: this failure may be transient; send the same call "
             "once more.]",
    "repair": "[harness recovery note: the call failed; check its arguments against what "
              "the customer said and earlier results, then make one corrected call.]",
    "stop": "[harness recovery note: do not call this tool again now; tell the customer "
            "what failed and ask for what you need.]",
}
RECOVERY = ("native", "retry", "repair", "stop")
LAST_K = 2  # recency arm default (reversible placeholders use it)
LAST_KS = (1, 2, 5, 10)  # keep-last-k grid; "all" = keep
# all-in cost units (probe_cost.TokenRates, relative H100 cost per token, pre-registered):
# a prefill token 1, a decoded token 3, a prefix-cache hit 0.1
COST_RATES = {"prefill": 1.0, "decode": 3.0, "cached": 0.1}
_ID_KEY = re.compile(r"(^|_)ids?$")
_ORDER = re.compile(r"#W\d{7}")


def _sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode() if isinstance(b, str) else b).hexdigest()


def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str)


def _u(*parts: Any) -> float:
    """A uniform number in [0, 1) fixed by the probe salt and ``parts``."""
    return int(_sha("|".join([PROBE_SALT, *map(str, parts)]))[:13], 16) / 16 ** 13


# ============================================================================ plan


@dataclass(frozen=True)
class ProbeSizes:
    base_episodes: int = 400  # plain-Hermes base runs (c9b), each a recorded prefix source
    context_boundaries: int = 2  # type 1: probed boundaries per base episode (at most)
    groups: int = 2  # type 1: random deletion groups per boundary
    keep_replicates: int = 2  # type 1 noise control: keep x2 (type 3: native x2)
    failures_per_episode: int = 2  # type 3: natural read-only failures probed (at most)
    injected_episodes: int = 150  # type 3: injected transient-failure stratum
    # every option runs twice on this share: replicate 0 selects (oracle), replicate 1
    # assesses (prereg additions s. 5: selection and assessment repeats, >= 2 each action)
    calibration_rate: float = 1.0
    bulky_chars: int = 400  # an eligible tool result has at least this many characters
    dialogue_sample: float = 0.02  # branches whose continuation text is exported
    chunk: int = 128


PROBE_SIZES = ProbeSizes()
PROBE_SMOKE = ProbeSizes(base_episodes=8, context_boundaries=1, groups=1, keep_replicates=2,
                         failures_per_episode=1, injected_episodes=4, calibration_rate=1.0,
                         dialogue_sample=0.25, chunk=64)
SPLIT_SHARES = (("fit", 0.6), ("calib", 0.2), ("test", 0.2))
FIDELITY_MAX_DIVERGED = 0.05  # smoke: prefix replay must reproduce >= 95 % of branches
BUDGET_BYTES = 2_000_000  # the probe's share of the < 20 MB return (s. 6)
# frozen per-parent byte bound (compressed: base deltas + its decisions + their branches
# + its share of sampled dialogues); admission stops at (budget - fixed) / bound parents
PER_PARENT_BOUND = 4_500
FIXED_BYTES = 100_000  # status, review rows, prompted review choices


def max_parents(budget: int = BUDGET_BYTES) -> int:
    return max(0, (budget - FIXED_BYTES) // PER_PARENT_BOUND)


def hypotheses() -> list[dict[str, Any]]:
    """Owner-facing, pre-registered (design s. 3)."""
    return [
        {"id": "HP1", "type": 1, "claim": "context trimming saves >= 10 % of all-in agent cost "
         "without success -2 points / harm +1 point",
         "test": ("per fixed arm (random group, Hermes native prune with lowered floors, "
                  "matched-budget heuristic, prompted 9B) vs keep replicate 1, paired by "
                  "decision: mean episode cost saving (cluster bootstrap 95 % CI by user), "
                  "success and harm differences (exact paired); keep replicate 2 vs 1 is the "
                  "serving-noise null"),
         "headroom_if": ("some arm OR the corrected oracle: saving >= 10 % (CI lower > 5 %), "
                         "success diff >= -2 pts (point) and one-sided 95 % loss bound <= 5 "
                         "pts, harm diff <= +1 pt (point), critical discordance not above the "
                         "keep-vs-keep count")},
        {"id": "HP2", "type": 2, "claim": "skipping the review fork halves review cost while "
         "losing few gate-surviving lessons",
         "test": ("share F of fork tokens spent on forks with no screen-surviving proposal "
                  "(arm C); best pre-declared observable rule (no task write in the episode; "
                  "episode success; short episode; prompted 9B says skip): saving and the "
                  "share of surviving forks it would skip"),
         "headroom_if": "F >= 50 % and a rule saves >= 50 % losing <= 20 % of surviving forks "
                        "(needs >= 5 surviving forks, else 'underpowered')"},
        {"id": "HP3", "type": 3, "claim": "per-failure recovery beats fixed rules",
         "test": ("success and cost of always-native / -retry / -repair / -stop vs the "
                  "per-failure oracle (best option per decision), oracle corrected for winner's "
                  "curse with the calibration replicates; natural and injected strata reported "
                  "separately"),
         "headroom_if": "corrected oracle - best fixed rule >= 3 success points, or equal "
                        "success with >= 10 % lower cost"},
    ]


def null_ladder() -> list[dict[str, str]]:
    return [
        {"rung": "R0 exposure", "fails_if": "type 1 < 200 decisions, type 3 < 100 failures "
         "(natural + injected), type 2 < 150 forks", "conclusion": "no exposure at this "
         "distribution (not 'no headroom')"},
        {"rung": "R1 replay fidelity", "fails_if": "> 5 % of branches diverge in the forced "
         "prefix", "conclusion": "the counterfactual is not identical; no headroom claim"},
        {"rung": "R2 serving noise", "fails_if": "an arm's effects are within the keep-vs-keep "
         "(native-vs-native) discordance", "conclusion": "indistinguishable from noise"},
        {"rung": "R3 oracle headroom", "fails_if": "HP1/HP2/HP3 bars not met", "conclusion":
         "no headroom for this decision type: do not label at scale"},
        {"rung": "R4+ (not tested here)", "fails_if": "labels inadequate -> not learnable at "
         "<= 2B -> sharing interference -> no episode benefit -> overhead erases savings -> "
         "gate underpowered", "conclusion": "the global plan's later rungs, each reported "
         "separately"},
    ]


def metrics() -> dict[str, str]:
    return {
        "success / harm / critical": "the M-H1 labels of the whole episode (prefix + "
                                     "continuation), from the customer's consent acts",
        "cost": "agent input + 3 x output tokens of the live continuation (the prefix is "
                "identical across the branches of a decision); cached input reported "
                "separately when the server returns it",
        "cost_share": "cost saving as a share of the keep branch's WHOLE episode cost "
                      "(prefix from the base run + continuation)",
        "latency": "branch wall seconds (p50 / p95 per arm)",
        "fidelity": "share of branches whose forced prefix reproduced the base exactly",
        "noise": "keep-vs-keep (type 3: native-vs-native) discordance per label",
        "review": "fork tokens, empty / duplicate / proposal, screen survival (C), audit flag",
    }


def split_of(case_id: str, user: str, order: int, n: int) -> dict[str, Any]:
    """Fixed BEFORE any label: user family bucket (prereg additions s. 7: F = fit, C =
    calibration with a family-level fit/audit part, P = untouched assessment; G, the
    future confirmation gate, is never drawn here), task family (the unit's context), and
    time (dispatch order; the last 20 % is the 'late' slice)."""
    u, acc = _u("split", user), 0.0
    bucket = SPLIT_SHARES[-1][0]
    for name, share in SPLIT_SHARES:
        acc += share
        if u < acc:
            bucket = name
            break
    pool = {"fit": "F", "calib": "C", "test": "P"}[bucket]
    return {"user_bucket": bucket, "pool": pool,
            "c_part": (("fit" if _u("c-part", user) < 0.5 else "audit")
                       if pool == "C" else None),
            "order": order, "late": order >= int(0.8 * n)}


def build_plan(cases: Sequence[Mapping[str, Any]], *, role_salt: str,
               sizes: ProbeSizes = PROBE_SIZES, smoke: ProbeSizes = PROBE_SMOKE,
               mh1_plan_sha256: str | None = None) -> dict[str, Any]:
    """``probe-plan.json``: sizes, units (full and smoke slices, disjoint), their splits,
    hypotheses, metrics, null ladder, arms, notes. Units: shipped cases whose user is NOT in
    the confirm partition (so never an eval or confirmation user), in probe-salt order."""
    from slh.system import tau2 as T

    cands = sorted((c for c in cases if T.role_of(str(c["entities"]["user_id"]), role_salt)
                    != "confirm"), key=lambda c: _sha(f"{PROBE_SALT}|unit|{c['case_id']}"))
    full = cands[:sizes.base_episodes]
    sm = cands[sizes.base_episodes:sizes.base_episodes + smoke.base_episodes]
    if len(sm) < smoke.base_episodes:  # a small (test) pool: the smoke slice reuses units
        sm = cands[:smoke.base_episodes]

    def units(lst: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [{"case": c["case_id"], "user": str(c["entities"]["user_id"]),
                 "family": T.context_of(c),
                 "split": split_of(c["case_id"], str(c["entities"]["user_id"]), i, len(lst))}
                for i, c in enumerate(lst)]

    body = {"schema": PROBE_SCHEMA, "probe_id": PROBE_ID, "design_doc": DESIGN_DOC,
            "blocks": list(BLOCKS), "state": "empty (plain Hermes)",
            "sizes": asdict(sizes), "smoke_sizes": asdict(smoke),
            "units": {"full": units(full), "smoke": units(sm)},
            "read_only_tools": sorted(READ_ONLY), "notes": dict(NOTES),
            "injected_error": INJECTED_ERROR, "recovery_options": list(RECOVERY),
            "context_arms": ["keep (C0)", "native_default (CH)", "group1..", "native_low",
                             "heuristic", *[f"last_k{k} (CK)" for k in LAST_KS],
                             "reversible (CP)", "prompted (P9)", "oracle (O)"],
            "recovery_arms": [*RECOVERY, "repair_fixed (deterministic map, when known)",
                              "prompted (P9)", "oracle (O)"],
            "deferred_arms": [("fitted feature scorer (probe_arms.FeatureScorer; needs "
                               "labels: the training package)"), "public small pruners",
                              "a head on the 9B's hidden states"],
            "last_k": LAST_K, "last_ks": list(LAST_KS), "cost_rates": dict(COST_RATES),
            "per_parent_bound": PER_PARENT_BOUND, "max_parents": max_parents(),
            "bootstrap": {"reps": BOOT_REPS, "seed": BOOT_SEED},
            "wording_audit_rate": WORDING_AUDIT,
            "fidelity_max_diverged": FIDELITY_MAX_DIVERGED,
            "hypotheses": hypotheses(), "null_ladder": null_ladder(), "metrics": metrics(),
            "budget_bytes": BUDGET_BYTES, "mh1_plan_sha256": mh1_plan_sha256,
            "gpu_estimate": gpu_estimate(sizes)}
    return {**body, "sha256": _sha(_canon(body))}


def load_plan(pkg_root: Path) -> dict[str, Any] | None:
    p = Path(pkg_root) / PLAN_FILE
    if not p.exists():
        return None
    plan = json.loads(p.read_text())
    if _sha(_canon({k: v for k, v in plan.items() if k != "sha256"})) != plan["sha256"]:
        raise ValueError("probe-plan.json does not hash to its sha256")
    return plan


def gpu_estimate(s: ProbeSizes = PROBE_SIZES, *, cont: float = 0.5,
                 rates: Sequence[float] = (5.0, 10.0, 20.0)) -> dict[str, Any]:
    """Episode-equivalents and H100-hours (c9b block). Assumptions (the smoke run measures):
    1.2 probed type-1 boundaries per episode, ~7 distinct type-1 edit sets,
    15 % of episodes with a natural read-only failure, a continuation = ``cont`` of an
    episode, one prompted-9B call per decision ~ 1/30 episode."""
    d1 = 1.2 * s.base_episodes
    d3 = min(s.failures_per_episode, 1) * 0.15 * s.base_episodes + s.injected_episodes
    # distinct non-reference edit sets per decision after sharing: ~9 of the type-1
    # options (2 placebos and the native+ids variant included), ~5.5 of type 3 (2
    # placebos included); the reference twice, the rest twice on the calibration share
    b1 = d1 * (2 + 9 * (1 + s.calibration_rate))
    b3 = d3 * (2 + 5.5 * (1 + s.calibration_rate))
    eq = s.base_episodes + (b1 + b3) * cont + (d1 + d3 + 180) / 30
    return {"decisions": {"type1": round(d1), "type3": round(d3)},
            "branches": {"type1": round(b1), "type3": round(b3)},
            "episode_equivalents": round(eq),
            "h100_hours": {f"{r:g}/gpu-min": round(eq / r / 60, 1) for r in rates}}


# ============================================================================ messages


def _content(c: Any) -> str:
    if isinstance(c, list):
        return " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return "" if c is None else str(c)


def _norm(messages: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    """Non-system messages as (role, content, call id, [(name, arguments)])."""
    out = []
    for m in messages:
        if not isinstance(m, Mapping) or m.get("role") == "system":
            continue
        calls = []
        for tc in m.get("tool_calls") or []:
            f = tc.get("function") or tc
            calls.append([str(f.get("name") or ""), str(f.get("arguments") or "")])
        out.append([m.get("role"), _content(m.get("content")),
                    str(m.get("tool_call_id") or ""), calls])
    return out


def digest(messages: Sequence[Mapping[str, Any]]) -> str:
    return _sha(_canon(_norm(messages)))


def request_digest(body: Mapping[str, Any]) -> str:
    return digest(body.get("messages") or [])


def call_fp(tc: Mapping[str, Any]) -> list[str]:
    """A tool call as (name, 16-hex digest of its canonical arguments), OpenAI or wire
    form: the reference ``probe_cost.EpisodeCost`` fingerprints (re-call detection)."""
    f = tc.get("function") or tc
    a: Any = f.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except (json.JSONDecodeError, ValueError):
            pass
    return [str(f.get("name") or ""), _sha(_canon(a))[:16]]


def removed_calls(wire: Sequence[Mapping[str, Any]], edits: Sequence[Mapping[str, Any]]
                  ) -> list[str]:
    """Fingerprints of the calls whose results ``edits`` stubbed or demoted."""
    by_id = {tc["id"]: tc for m in wire for tc in m.get("tool_calls") or [] if tc.get("id")}
    out = []
    for e in edits:
        if e.get("kind") in ("stub", "native", "native_ids", "reversible"):
            tc = by_id.get(str(wire[e["i"]].get("tool_call_id") or ""))
            if tc is not None:
                out.append(call_fp(tc))
    return out


def tool_names_by_id(wire: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {tc["id"]: tc["name"] for m in wire for tc in m.get("tool_calls") or []
            if tc.get("id")}


def pending_start(wire: Sequence[Mapping[str, Any]]) -> int:
    i = len(wire)
    while i > 0 and wire[i - 1].get("role") == "tool":
        i -= 1
    return i


def failed(content: str) -> str | None:
    """Failure kind of a tool result: ``tool_error`` (the environment's ``Error: ...``, a JSON
    error) or ``guard_block`` (the harness guard refused the call); None when it succeeded."""
    c = (content or "").strip()
    if "blocked by the harness guard" in c.lower():
        return "guard_block"
    if c.lower().startswith("error"):
        return "tool_error"
    try:
        d = json.loads(c)
    except (json.JSONDecodeError, ValueError):
        return None
    return "tool_error" if isinstance(d, dict) and d.get("error") else None


def items(wire: Sequence[Mapping[str, Any]], bulky: int) -> list[dict[str, Any]]:
    """Every tool result with its protection status. Eligible for deletion: a successful
    READ-ONLY result of at least ``bulky`` characters, outside the pending round, not stubbed."""
    names = tool_names_by_id(wire)
    ps = pending_start(wire)
    out = []
    for i, m in enumerate(wire):
        if m.get("role") != "tool":
            continue
        tool = names.get(str(m.get("tool_call_id") or ""), "")
        c = _content(m.get("content"))
        fk = failed(c)
        elig = (tool in READ_ONLY and i < ps and len(c) >= bulky and fk is None
                and STUB_TAG not in c[:40])
        out.append({"i": i, "tool": tool, "chars": len(c), "sha": _sha(c)[:16],
                    "pending": i >= ps, "failed": fk, "read_only": tool in READ_ONLY,
                    "eligible": elig})
    return out


def items_with_features(wire: Sequence[Mapping[str, Any]], bulky: int
                        ) -> list[dict[str, Any]]:
    feats = item_features(wire)
    return [{**it, "features": feats.get(it["i"])} for it in items(wire, bulky)]


def identifiers(content: str) -> list[str]:
    """Every identifier in a result (no cap: a user with many orders keeps them all):
    values of ``*_id`` / ``*_ids`` / ``id`` keys, dict keys that look like ids (payment
    methods), and order numbers anywhere."""
    found: list[str] = []

    def add(v: Any) -> None:
        s = str(v)
        if s and s not in found:
            found.append(s)

    def walk(x: Any, key: str = "") -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if _ID_KEY.search(str(k)):
                    for y in (v if isinstance(v, list) else [v]):
                        if isinstance(y, (str, int)):
                            add(y)
                if key.endswith("payment_methods"):
                    add(k)
                walk(v, str(k))
        elif isinstance(x, list):
            for y in x:
                walk(y, key)

    try:
        walk(json.loads(content))
    except (json.JSONDecodeError, ValueError):
        pass
    for m in _ORDER.findall(content or ""):
        add(m)
    if not found and re.fullmatch(r"[A-Za-z0-9_]{3,64}", (content or "").strip().strip('"')):
        add((content or "").strip().strip('"'))  # a bare id result (find_user_id_*)
    return found


def stub(tool: str, content: str) -> str:
    return (f"[{STUB_TAG}: the result of {tool} was removed to save context; identifiers "
            f"kept: {json.dumps(identifiers(content))}]")


def _edit_stub(wire: Sequence[Mapping[str, Any]], it: Mapping[str, Any]) -> dict[str, Any]:
    return {"i": it["i"], "kind": "stub",
            "content": stub(it["tool"], _content(wire[it["i"]].get("content")))}


def _edit_reversible(wire: Sequence[Mapping[str, Any]], it: Mapping[str, Any]
                     ) -> dict[str, Any]:
    """DTOC-style reversible placeholder: the stub says how to get the body back (the
    same read-only call again; a re-call is counted as a cost)."""
    c = _content(wire[it["i"]].get("content"))
    return {"i": it["i"], "kind": "reversible",
            "content": stub(it["tool"], c)[:-1] + "; to see it again, repeat the same "
                       f"{it['tool']} call]"}


def last_k_items(wire: Sequence[Mapping[str, Any]], elig: Sequence[Mapping[str, Any]],
                 k: int) -> list[Mapping[str, Any]]:
    """Recency (``probe_arms.keep_last_k`` semantics: the last ``k`` tool results of any
    kind are kept), restricted to the eligible results; the probe's identifier-keeping
    stub replaces ``probe_arms``' lexical gate, which protects nearly every retail result
    (ids everywhere; reported per item as ``lexical_protected``)."""
    tools = [i for i, m in enumerate(wire) if m.get("role") == "tool"]
    keep = set(tools[-k:]) if k else set()
    return [it for it in elig if it["i"] not in keep]


def _openai(wire: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in wire:
        if m.get("role") == "system":
            continue
        d = {k: v for k, v in m.items() if k != "tool_calls"}
        if m.get("tool_calls"):
            d["tool_calls"] = [{"id": tc["id"], "type": "function", "function": {
                "name": tc["name"], "arguments": tc["arguments"]}} for tc in m["tool_calls"]]
        out.append(d)
    return out


def item_features(wire: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """Classifier features per tool result (``probe_arms.extract_features``: age, size,
    tool, lexical overlap with later messages; snapshot-local, no label) and the
    conservative lexical protection flag (``probe_arms.is_protected_content``)."""
    from slh.system.hosts.hermes import probe_arms as PA

    sys_n = sum(1 for m in wire if m.get("role") == "system")
    msgs = _openai(wire)
    out: dict[int, dict[str, Any]] = {}
    try:
        feats = PA.extract_features(msgs)
    except ValueError:
        return out
    for j, f in feats.items():
        out[j + sys_n] = {"age": f.age, "size": f.size,
                          "overlap": round(f.referenced_later_overlap, 4),
                          "lexical_protected": PA.is_protected_content(msgs[j])}
    return out


def group_edits(wire, elig, *, key: str) -> list[dict[str, Any]]:
    """A random non-empty group: each eligible result in with probability 1/2."""
    pick = [it for it in elig if _u("group", key, it["i"]) < 0.5]
    if not pick and elig:
        pick = [elig[int(_u("group1", key) * len(elig))]]
    return [_edit_stub(wire, it) for it in pick]


def native_edits(wire, native: Sequence[Mapping[str, Any]] | None,
                 elig: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Hermes' own prune passes (lowered floors), restricted to the eligible results (the
    same protection rule as every arm)."""
    ok = {it["i"] for it in elig}
    return [{"i": int(e["i"]), "kind": "native", "content": str(e["content"])}
            for e in native or [] if int(e["i"]) in ok]


def native_ids_edits(wire, native: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``native+ids``: Hermes' own demotion text, plus every identifier of the ORIGINAL
    result that the demotion dropped (the probe's protection rule, as in every other
    arm). The plain native arms stay a faithful record of what Hermes does."""
    out = []
    for e in native:
        orig = _content(wire[int(e["i"])].get("content"))
        lost = [x for x in identifiers(orig) if x not in str(e["content"])]
        txt = str(e["content"]) + (f"\n[{STUB_TAG}: identifiers kept: {json.dumps(lost)}]"
                                   if lost else "")
        out.append({"i": int(e["i"]), "kind": "native_ids", "content": txt})
    return out


def drops_identifier(original: str, new: str) -> list[str]:
    """Identifiers of ``original`` missing from ``new`` (the runner's check; mirrored there
    in stdlib form)."""
    return [x for x in identifiers(original) if x not in new]


def _reserialise(c: str, variant: int) -> str:
    """A meaning-preserving rewrite (placebo): JSON re-serialised with another key order
    and whitespace; other text with added trailing whitespace. Ids stay verbatim."""
    try:
        d = json.loads(c)
    except (json.JSONDecodeError, ValueError):
        d = None
    if isinstance(d, (dict, list)):
        if variant == 1:
            out = json.dumps(d, indent=1, sort_keys=True)
        else:
            def rev(x: Any) -> Any:
                if isinstance(x, dict):
                    return {k: rev(x[k]) for k in reversed(list(x))}
                return [rev(y) for y in x] if isinstance(x, list) else x
            out = json.dumps(rev(d), separators=(", ", ": "))
    else:
        out = c + ("\n" if variant == 1 else " \n")
    return out if out != c else c + "\n" * variant


def placebo_edits(wire, i: int, variant: int, *, base: str | None = None
                  ) -> list[dict[str, Any]]:
    c = _content(wire[i].get("content")) if base is None else base
    return [{"i": i, "kind": "placebo" if base is None else "inject_placebo",
             "content": _reserialise(c, variant)}]


NATIVE_ARMS = ("native_low", "native_default")  # Hermes' text: may drop identifiers
PLACEBOS = ("placebo1", "placebo2")


def removed_chars(wire, edits: Sequence[Mapping[str, Any]]) -> int:
    return sum(max(0, len(_content(wire[e["i"]].get("content"))) - len(e["content"]))
               for e in edits)


def heuristic_edits(wire, elig, budget: int) -> list[dict[str, Any]]:
    """Matched budget: stub the OLDEST eligible results until ``budget`` characters are
    removed (the budget is native_low's, else the first random group's)."""
    out: list[dict[str, Any]] = []
    for it in sorted(elig, key=lambda x: x["i"]):
        if removed_chars(wire, out) >= budget > 0:
            break
        out.append(_edit_stub(wire, it))
    return out if budget > 0 else []


def recovery_edits(wire, i: int, option: str, *, injected: bool) -> list[dict[str, Any]]:
    base = INJECTED_ERROR if injected else _content(wire[i].get("content"))
    if option == "native":
        return [{"i": i, "kind": "inject", "content": base}] if injected else []
    # an injected failure replaces an UNREAD pending result (kind "inject*": exempt from
    # the runner's identifier check; the retry brings the real result back)
    return [{"i": i, "kind": f"inject_{option}" if injected else option,
             "content": base + "\n" + NOTES[option]}]


def deterministic_repair(wire: Sequence[Mapping[str, Any]], i: int) -> str | None:
    """The frozen repair map for known read-only error classes (prereg additions s. 3.1):
    a schema-valid argument normalisation, or None (unknown: abstain to native)."""
    by_id = {tc["id"]: tc for m in wire for tc in m.get("tool_calls") or [] if tc.get("id")}
    tc = by_id.get(str(wire[i].get("tool_call_id") or ""))
    if tc is None:
        return None
    try:
        a = json.loads(tc.get("arguments") or "{}")
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(a, dict):
        return None
    fixed = dict(a)
    for k, v in a.items():
        if not isinstance(v, str):
            continue
        w = v.strip()
        if k == "order_id" and w and not w.startswith("#"):
            w = "#" + w
        if k == "email":
            w = w.lower()
        if k in ("user_id", "item_id", "product_id", "payment_method_id"):
            w = w.replace(" ", "")
        fixed[k] = w
    if fixed == a:
        return None
    return f"{tc['name']}({json.dumps(fixed, sort_keys=True)})"


def edits_key(edits: Sequence[Mapping[str, Any]]) -> str:
    return _sha(_canon(sorted((int(e["i"]), e["content"]) for e in edits)))[:16]


# ============================================================================ gate


class _Tee:
    def __init__(self, inner: Any, buf: bytearray, cap: int = 8_000_000) -> None:
        self.inner, self.buf, self.cap = inner, buf, cap

    def write(self, b: bytes) -> int:
        n = self.inner.write(b)
        if len(self.buf) < self.cap:
            self.buf += b
        return n

    def flush(self) -> None:
        self.inner.flush()

    def __getattr__(self, k: str) -> Any:
        return getattr(self.inner, k)


class ProbeGate(SpendGate):
    """The episode's SpendGate (same ceilings), which also records the agent's calls
    (``record``) or replays the first ``len(forced)`` recorded responses after checking each
    request's normalised digest against the recorded one (``branch``). A request that differs
    goes live and the branch is marked diverged."""

    def __init__(self, *a: Any, mode: str, forced: Sequence[Mapping[str, Any]] = (),
                 **k: Any) -> None:
        super().__init__(*a, **k)
        self.mode, self.forced = mode, list(forced)
        self.calls: list[dict[str, Any]] = []
        self.diverged: list[dict[str, Any]] = []

    def _admit(self, body: dict[str, Any]) -> dict[str, Any] | None:
        rec = super()._admit(body)
        if rec is not None:
            rec["digest"] = request_digest(body)
            with self.lock:
                rec["idx"] = next(i for i, r in enumerate(self.requests) if r is rec)
        return rec

    def _forward(self, h: Any, method: str, path: str, data: bytes | None,
                 rec: dict[str, Any] | None) -> None:
        if rec is None:
            return super()._forward(h, method, path, data, rec)
        i = rec["idx"]
        if self.mode == "branch" and i < len(self.forced):
            f = self.forced[i]
            if f["digest"] == rec["digest"]:
                try:
                    h.wfile.write(base64.b64decode(f["raw"]))
                    h.wfile.flush()
                except OSError:
                    pass
                h.close_connection = True
                rec.update(usage=f.get("usage"), complete=bool(f.get("complete")),
                           replayed=True)
                return None
            self.diverged.append({"call": i, "why": "request differs from the recorded "
                                                    "prefix"})
        buf = bytearray()
        h.wfile = _Tee(h.wfile, buf)
        super()._forward(h, method, path, data, rec)
        if self.mode == "record":
            self.calls.append({"i": i, "digest": rec["digest"],
                               "raw": base64.b64encode(bytes(buf)).decode(),
                               "usage": rec.get("usage"), "complete": rec.get("complete")})
        return None

    def live_calls(self) -> list[list[int]]:
        """(input, output, cached) per live request, in order (the all-in cost's input)."""
        out = []
        for r in self.requests:
            if r.get("replayed"):
                continue
            u = r.get("usage") or {}
            det = u.get("prompt_tokens_details") or {}
            out.append([int(u.get("prompt_tokens") or r["input_bound"]),
                        int(u.get("completion_tokens") or 0),
                        None if det.get("cached_tokens") is None
                        else int(det["cached_tokens"])])
        return out

    def live(self) -> dict[str, int]:
        i = o = c = n = 0
        for r in self.requests:
            if r.get("replayed"):
                continue
            u = r.get("usage") or {}
            n += 1
            i += int(u.get("prompt_tokens") or r["input_bound"])
            o += int(u.get("completion_tokens") or 0)
            c += int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
        return {"calls": n, "input": i, "output": o, "cached": c,
                "replayed": sum(1 for r in self.requests if r.get("replayed"))}


# ============================================================================ customer


class ForcedCustomer:
    """The base run's customer turns (text and act, in order) for the forced prefix, then the
    live customer."""

    def __init__(self, inner: Any, turns: Sequence[Mapping[str, Any]]) -> None:
        self.inner, self.turns, self.k = inner, list(turns), 0

    def first_message(self, scenario: Mapping[str, Any]) -> tuple[str, Any]:
        if self.turns:
            self.k = 1
            return str(self.turns[0]["text"]), self.turns[0]["act"]
        return self.inner.first_message(scenario)

    def reply(self, scenario: Mapping[str, Any], transcript: Sequence[Mapping[str, Any]]
              ) -> tuple[bool, str, Any]:
        if self.k < len(self.turns):
            t = self.turns[self.k]
            self.k += 1
            return False, str(t["text"]), t["act"]
        return self.inner.reply(scenario, transcript)

    def __getattr__(self, k: str) -> Any:
        return getattr(self.inner, k)


# ============================================================================ hooks


def _write_gz_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0,
                                               filename="") as fh:
        fh.write(json.dumps(obj, default=str).encode())
    os.replace(tmp, p)


def _read_gz_json(p: Path) -> Any:
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def rec_path(root: Path, tag: str) -> Path:
    return Path(root) / "rec" / f"{_sha(tag)[:24]}.json.gz"


class ProbeHooks:
    """Shared by every worker of the probe stage. ``register(tag, spec)`` before dispatch;
    an episode whose tag is not registered runs unchanged (and the hooks are attached only
    while the probe stage runs)."""

    def __init__(self, root: Path, plan: Mapping[str, Any]) -> None:
        self.root = Path(root)
        self.plan = plan
        self.specs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._recs: dict[str, Any] = {}

    def register(self, tag: str, spec: Mapping[str, Any]) -> None:
        with self.lock:
            self.specs[tag] = dict(spec)

    def base_record(self, tag: str) -> dict[str, Any]:
        with self.lock:
            if tag not in self._recs:
                if len(self._recs) > 64:
                    self._recs.clear()
                self._recs[tag] = _read_gz_json(rec_path(self.root, tag))
            return self._recs[tag]

    def begin(self, r: Any, session: Any) -> _Handle | None:
        spec = self.specs.get(r.tag)
        return None if spec is None else _Handle(self, spec, r, session)


class _Handle:
    def __init__(self, hooks: ProbeHooks, spec: Mapping[str, Any], r: Any, session: Any
                 ) -> None:
        self.hooks, self.spec, self.r, self.session = hooks, spec, r, session
        self.mode = spec["mode"]
        self.bounds: list[dict[str, Any]] = []
        self.diverged: list[dict[str, Any]] = []
        self.applied = False
        self._gate: ProbeGate | None = None
        self.base = None
        if self.mode == "branch":
            self.base = hooks.base_record(spec["base_tag"])
            b0 = int(spec["b"])
            want = int(self.base["boundaries"][b0]["user_turns"])
            session.customer = ForcedCustomer(session.customer, self.base["users"][:want])

    def runner_spec(self) -> dict[str, Any]:
        return {"native_min_chars": int(self.hooks.plan["sizes"]["bulky_chars"])}

    def gate(self, model: Any) -> ProbeGate:
        forced = ()
        if self.mode == "branch":
            n = int(self.base["boundaries"][int(self.spec["b"])]["n_calls"])
            forced = self.base["calls"][:n]
        self._gate = ProbeGate(model.base_url, counter=model.counter,
                               max_input_tokens=model.worst_tokens_per_run[0],
                               max_output_tokens=model.worst_tokens_per_run[1],
                               max_output_per_request=model.max_output_tokens_per_request,
                               usd_per_mtok=model.usd_per_mtok, mode=self.mode, forced=forced)
        return self._gate

    def _users(self) -> int:
        return sum(1 for t in getattr(self.session, "transcript", []) if t["role"] == "user")

    def on_boundary(self, msg: Mapping[str, Any]) -> dict[str, Any]:
        b, wire = int(msg.get("b", -1)), list(msg.get("messages") or [])
        n_calls = len(self._gate.requests) if self._gate is not None else -1
        snap = {"b": b, "n_calls": n_calls, "user_turns": self._users(),
                "digest": digest(wire), "n_msgs": len(wire)}
        if self.mode == "record":
            self.bounds.append({**snap, "messages": wire, "native": msg.get("native"),
                                "native_default": msg.get("native_default"),
                                "native_default_n": len(msg.get("native_default") or [])})
            return {"edits": []}
        b0 = int(self.spec["b"])
        if b > b0:
            return {"edits": []}
        want = self.base["boundaries"][b]
        for k in ("digest", "n_calls", "user_turns"):
            if snap[k] != want[k]:
                self.diverged.append({"b": b, "why": f"{k} differs from the base"})
        if b < b0 or self.diverged:
            return {"edits": []}
        self.applied = True
        return {"edits": [{"i": int(e["i"]), "content": e["content"],
                           "kind": e.get("kind", "")} for e in self.spec["edits"]]}

    def finish(self, res: Mapping[str, Any] | None, gate: Any) -> dict[str, Any]:
        runner = (res or {}).get("probe") or {}
        if self.mode == "record":
            g = self._gate
            users = [{"text": t["content"], "act": t.get("act")}
                     for t in getattr(self.session, "transcript", []) if t["role"] == "user"]
            _write_gz_json(rec_path(self.hooks.root, self.r.tag),
                           {"tag": self.r.tag, "calls": g.calls if g else [],
                            "boundaries": self.bounds, "users": users,
                            "messages": (res or {}).get("messages")})
            return {"mode": "record", "boundaries": len(self.bounds),
                    "calls": len(g.calls) if g else 0,
                    "runner_errors": list(runner.get("errors") or [])[:5]}
        g = self._gate
        div = list(self.diverged) + (list(g.diverged) if g else [])
        n_edits = len(self.spec["edits"])
        if self.applied and len(runner.get("applied") or []) != n_edits:
            div.append({"why": f"runner applied {len(runner.get('applied') or [])} of "
                               f"{n_edits} edits"})
        if not self.applied:
            div.append({"why": "the boundary was never reached"})
        # the continuation's tool calls (fingerprints) and the calls whose results this
        # branch removed: a re-call of a removed result is a cost of the removal
        msgs = list((res or {}).get("messages") or [])
        n0 = int(self.base["boundaries"][int(self.spec["b"])]["n_msgs"])
        cont = [call_fp(tc) for m in msgs[n0:] if isinstance(m, Mapping)
                for tc in m.get("tool_calls") or []]
        return {"mode": "branch", "diverged": div[:5], "valid": not div,
                "live": g.live() if g else None,
                "live_calls": g.live_calls() if g else [], "cont_calls": cont,
                "runner_errors": list(runner.get("errors") or [])[:5]}


# ============================================================================ prompted 9B


def _ctx_lines(wire: Sequence[Mapping[str, Any]], labels: Mapping[int, str],
               cap: int = 12000) -> str:
    names = tool_names_by_id(wire)
    out = []
    for i, m in enumerate(wire):
        r, c = m.get("role"), _content(m.get("content"))
        if r == "system":
            continue
        if r == "user":
            out.append(f"CUSTOMER: {c[:800]}")
        elif r == "assistant":
            if c.strip():
                out.append(f"AGENT: {c[:800]}")
            for tc in m.get("tool_calls") or []:
                out.append(f"AGENT CALLS {tc['name']}({tc['arguments'][:200]})")
        elif r == "tool":
            lab = labels.get(i)
            tool = names.get(str(m.get("tool_call_id") or ""), "tool")
            head = f"RESULT [{lab}] " if lab else "RESULT "
            out.append(f"{head}{tool} ({len(c)} chars): {c[:400 if lab else 200]}")
    text = "\n".join(out)
    return text[-cap:]


def prompt_context(wire, elig) -> tuple[str, dict[str, int]]:
    labels = {it["i"]: f"R{k + 1}" for k, it in enumerate(elig)}
    p = (_ctx_lines(wire, labels) + "\n\nThe results marked [R...] are earlier tool results. "
         "Which of them can be removed from the agent's context without hurting the rest "
         "of this conversation (identifiers are kept when a result is removed)? Answer "
         'with JSON only: {"drop": ["R1", ...], "confidence": 0.0-1.0}.')
    return p, {v: k for k, v in labels.items()}


def prompt_recovery(wire, i: int, *, injected: bool, reverse: bool = False) -> str:
    names = tool_names_by_id(wire)
    tool = names.get(str(wire[i].get("tool_call_id") or ""), "tool")
    shown = [dict(m) for m in wire]
    if injected:
        shown[i]["content"] = INJECTED_ERROR
    desc = {"native": "'native' (no instruction)", "retry": "'retry' (send the same call "
            "again)", "repair": "'repair' (fix the arguments and call once more)",
            "stop": "'stop' (no more calls of this tool now; tell the customer and ask)"}
    order = list(reversed(RECOVERY)) if reverse else list(RECOVERY)
    opts_text = ", ".join(desc[o] for o in order[:-1]) + " or " + desc[order[-1]]
    return (_ctx_lines(shown, {i: "F"}) + f"\n\nThe read-only call {tool} [F] just failed. "
            f"What should the agent do next: {opts_text}? Answer with JSON only: "
            '{"choice": "' + "|".join(order) + '", "confidence": 0.0-1.0}.')


def prompt_review(messages: Sequence[Mapping[str, Any]] | None) -> str:
    return (_ctx_lines(_wire_of(messages or []), {}, cap=8000) + "\n\nThis customer "
            "conversation has ended. A background review can now look for skills or "
            "memories worth saving (it costs a model call). Should it run? Answer with JSON "
            'only: {"review": true|false, "confidence": 0.0-1.0}.')


def _wire_of(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        if not isinstance(m, Mapping):
            continue
        w = {"role": m.get("role"), "content": _content(m.get("content"))}
        if m.get("tool_call_id"):
            w["tool_call_id"] = str(m["tool_call_id"])
        if m.get("tool_calls"):
            w["tool_calls"] = [{"id": str(tc.get("id") or ""),
                                "name": str((tc.get("function") or {}).get("name") or ""),
                                "arguments": str((tc.get("function") or {}).get("arguments")
                                                 or "")} for tc in m["tool_calls"]]
        out.append(w)
    return out


SYSTEM = (f"{PROMPT_TAG}: you advise the harness of a customer-service agent on one "
          "decision. Answer with the JSON object requested and nothing else.")


def parse_choice(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    return d if isinstance(d, dict) else None


@dataclass
class PromptedChooser:
    """The frozen 9B, prompted (T = 0, the agent's seed), on the agent's own server; one
    request per decision, harness side (no Hermes, no tools)."""

    base_url: str
    model: str
    seed: int = 7401
    timeout_s: float = 120.0
    max_tokens: int = 200

    def ask(self, prompt: str) -> dict[str, Any]:
        import urllib.request

        body = {"model": self.model, "temperature": 0.0, "seed": self.seed,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt}]}
        req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                data = json.loads(r.read())
            text = str(data["choices"][0]["message"].get("content") or "")
            u = data.get("usage") or {}
        except Exception as e:
            # the server failed or stalled: NOT an abstention. Halt (exit 3, resumable;
            # answers given so far are cached), so one stalled server costs one timeout
            from slh.system.hosts.hermes.errors import HaltError

            raise HaltError(f"prompted-9B request failed ({type(e).__name__}: {e}); "
                            "rerun the same command") from e
        return {"ok": True, "parsed": parse_choice(text), "text_sha": _sha(text)[:16],
                "input": int(u.get("prompt_tokens") or 0),
                "output": int(u.get("completion_tokens") or 0),
                "seconds": round(time.time() - t0, 3)}


# ============================================================================ dry-run policy


class ProbeDryRunAgent:
    """CPU dry-run policy for the probe (no model; not a model of anything): within one
    customer turn it looks the user up, reads the user's details (a bulky read-only result),
    reads a non-existent order (a natural read-only failure), then lets the M-H1 dry-run
    policy answer; it answers the prompted-decision requests with fixed JSON."""

    def __init__(self, inner: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> None:
        self.inner = inner

    def __call__(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        from slh.system.hosts.hermes.stub_model import (
            calls_since_last_user,
            last_user_text,
            tool_names,
            tool_results_since_last_user,
        )

        msgs = body.get("messages") or []
        if msgs and PROMPT_TAG in _content(msgs[0].get("content")):
            u = last_user_text(body)
            if '"drop"' in u:
                return {"content": '{"drop": ["R1"], "confidence": 0.6}'}
            if '"choice"' in u:
                return {"content": '{"choice": "repair", "confidence": 0.5}'}
            return {"content": '{"review": false, "confidence": 0.5}'}
        if "management tools" in last_user_text(body):
            return self.inner(body)
        names, calls = tool_names(body), calls_since_last_user(body)
        if "find_user_id_by_email" in calls and "get_user_details" in names \
                and "get_user_details" not in calls:
            res = [_content(m.get("content")) for m in tool_results_since_last_user(body)]
            uid = (res[0].strip().strip('"') if res else "") or "unknown"
            return {"tool_calls": [{"name": "get_user_details",
                                    "arguments": {"user_id": uid}}]}
        if "get_user_details" in calls and "get_order_details" in names \
                and "get_order_details" not in calls:
            return {"tool_calls": [{"name": "get_order_details",
                                    "arguments": {"order_id": "#W0000000"}}]}
        if calls and calls[-1] == "get_order_details" and calls.count("get_order_details") == 1:
            last = _content((tool_results_since_last_user(body) or [{}])[-1].get("content"))
            if "send the same call" in last:  # retry note: one more identical call
                return {"tool_calls": [{"name": "get_order_details",
                                        "arguments": {"order_id": "#W0000000"}}]}
        return self.inner(body)


# ============================================================================ orchestration


def _empty_state() -> Any:
    from slh.state import HarnessState

    return HarnessState.empty()


def probe_unit(pool: Any, case_id: str) -> Any:
    from slh.system import tau2 as T
    from slh.system.interfaces import Unit

    # always a gate role: no review fork, learning writes blocked (whatever partition the
    # user belongs to; never confirm, see build_plan)
    return Unit(f"hep:{case_id}", "monitor", T.context_of(pool.cases[case_id]),
                pool.cluster(case_id), {"case_id": case_id, "level": "conversation"})


def outcome_compact(o: Any, d: Mapping[str, Any]) -> dict[str, Any]:
    u = (o.trace or {}).get("usage") or {}
    return {"status": o.status, "success": o.success, "harm": o.harm,
            "critical": o.critical_harm, "incident": o.incident, "mediated": o.mediated,
            "seconds": d.get("seconds"), "killed": d.get("killed"),
            "tokens": [int(u.get("session_prompt_tokens") or 0),
                       int(u.get("session_completion_tokens") or 0)],
            "api_calls": int(u.get("session_api_calls") or 0)}


DELTA_CHARS = 2_000  # per message (bounded bytes per parent); a cut is flagged ("...")


def _cap(c: str) -> str:
    return c if len(c) <= DELTA_CHARS else c[:DELTA_CHARS] + "\u2026[cut]"


def compact_messages(messages: Sequence[Mapping[str, Any]] | None) -> list[list[Any]]:
    """The base episode as minimal text deltas: agent text and calls, customer text; tool
    results as (length, sha256 prefix) only (rebuilt on our side from the database);
    system messages as a hash."""
    out = []
    for m in messages or []:
        if not isinstance(m, Mapping):
            continue
        r, c = m.get("role"), _content(m.get("content"))
        if r == "system":
            out.append(["s", _sha(c)[:16]])
        elif r == "tool":
            out.append(["t", str(m.get("tool_call_id") or ""), len(c), _sha(c)[:16]])
        elif r == "user":
            out.append(["u", _cap(c)])
        else:
            calls = [[str(tc.get("id") or ""), str((tc.get("function") or {}).get("name")),
                      str((tc.get("function") or {}).get("arguments") or "")]
                     for tc in m.get("tool_calls") or []]
            out.append(["a", _cap(c), calls])
    return out


@dataclass
class ProbeRun:
    """The probe stage of one block's M-H1 run folder (``<run>/probe/``); resumable."""

    out: Path
    plan: Mapping[str, Any]
    adapter: Any  # ParallelAdapter (the hooks are attached for this stage only)
    chooser: PromptedChooser | None
    smoke: bool
    log: Callable[[str], None] = print

    def __post_init__(self) -> None:
        from slh.system.hosts.hermes.mh1 import Jsonl

        self.dir = Path(self.out) / "probe"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sizes = ProbeSizes(**self.plan["smoke_sizes" if self.smoke else "sizes"])
        self.units = self.plan["units"]["smoke" if self.smoke else "full"][
            :self.sizes.base_episodes]
        self.hooks = ProbeHooks(self.dir, self.plan)
        self.base_log = Jsonl(self.dir / "base.jsonl")
        self.dec_log = Jsonl(self.dir / "decisions.jsonl")
        self.br_log = Jsonl(self.dir / "branches.jsonl")
        self.rev_log = Jsonl(self.dir / "review-prompted.jsonl")
        self.ask_log = Jsonl(self.dir / "prompted-cache.jsonl")  # node-local, not exported
        self._cache: dict[str, Any] | None = None

    # -------------------------------------------------------------- dispatch

    def _dispatch(self, reqs: list[tuple[Any, dict[str, Any]]]) -> list[tuple[Any, dict]]:
        from slh.system.interfaces import RunRequest

        for _, spec in reqs:
            self.hooks.register(spec["tag"], spec)
        ws = getattr(self.adapter, "workers", [self.adapter])
        for w in ws:
            w.probe = self.hooks
        try:
            return self.adapter.run_with_diag(
                [RunRequest(u, _empty_state(), spec["tag"]) for u, spec in reqs])
        finally:
            for w in ws:
                w.probe = None

    def _incidents(self, pairs: Sequence[tuple[Any, dict]]) -> None:
        from slh.system.hosts.hermes.errors import IncidentError
        from slh.system.hosts.hermes.mh1 import Jsonl

        bad = [(o.unit_id, (o.trace or {}).get("incident_reasons")) for o, _ in pairs
               if o.incident]
        if bad:
            Jsonl(self.out / "incidents.jsonl").append(
                [{"what": "probe", "unit": u, "reasons": r, "time": time.time()}
                 for u, r in bad])
            raise IncidentError(f"probe: {len(bad)} containment incident(s): {bad[:3]}")

    # -------------------------------------------------------------- phase 1: base

    def run_base(self) -> None:
        pool = self.adapter.surface.pool
        done = {r["case"] for r in self.base_log.rows()}
        # byte admission (prereg additions s. 8): never more parents than the frozen
        # per-parent bound allows inside the probe's share of the return
        cap = max_parents()
        if len(self.units) > cap:
            self.log(f"probe: capacity stop at {cap} parents (byte bound)")
        todo = [u for u in self.units[:cap] if u["case"] not in done]
        for k in range(0, len(todo), self.sizes.chunk):
            part = todo[k:k + self.sizes.chunk]
            reqs = [(probe_unit(pool, u["case"]), {"mode": "record", "tag": f"pb|{u['case']}"})
                    for u in part]
            pairs = self._dispatch(reqs)
            self._incidents(pairs)
            rows = []
            for u, (o, d) in zip(part, pairs, strict=True):
                pr = d.get("probe") or {}
                rec = _read_gz_json(rec_path(self.dir, f"pb|{u['case']}")) \
                    if rec_path(self.dir, f"pb|{u['case']}").exists() else {}
                bounds = [{"b": b["b"], "n_calls": b["n_calls"], "user_turns": b["user_turns"],
                           "digest": b["digest"][:16], "n_msgs": b["n_msgs"],
                           "native_default_n": b["native_default_n"],
                           "items": items(b["messages"], self.sizes.bulky_chars)}
                          for b in rec.get("boundaries") or []]
                rows.append({"case": u["case"], "tag": f"pb|{u['case']}",
                             **outcome_compact(o, d), "boundaries": bounds,
                             "calls": pr.get("calls"), "runner_errors": pr.get("runner_errors"),
                             "deltas": compact_messages(rec.get("messages"))})
            self.base_log.append(rows)
            self.log(f"probe base: {min(k + len(part), len(todo))}/{len(todo)}")

    # -------------------------------------------------------------- phase 2: decisions

    def _ask(self, prompt: str) -> dict[str, Any] | None:
        """The prompted 9B's answer, cached by prompt hash (a resume after a halt asks only
        what is still open). A model answer that does not parse is an abstention; a server
        failure halts (``PromptedChooser.ask``)."""
        if self.chooser is None:
            return None
        key = _sha(prompt)
        if self._cache is None:
            self._cache = {r["key"]: r["answer"] for r in self.ask_log.rows()}
        if key not in self._cache:
            ans = self.chooser.ask(prompt)
            self.ask_log.append([{"key": key, "answer": ans}])
            self._cache[key] = ans
        return self._cache[key]

    def plan_decisions(self) -> list[dict[str, Any]]:
        if self.dec_log.rows():
            return self.dec_log.rows()
        s = self.sizes
        units = {u["case"]: u for u in self.units}
        out: list[dict[str, Any]] = []
        inj_cases = sorted(units, key=lambda c: _u("inject", c))[:s.injected_episodes]
        for row in self.base_log.rows():
            case = row["case"]
            if row["status"] != "scored" or not row["boundaries"]:
                continue
            rec = _read_gz_json(rec_path(self.dir, row["tag"]))
            meta = {"case": case, "base_tag": row["tag"], "family": units[case]["family"],
                    "split": units[case]["split"]}
            # type 1: boundaries with >= 1 eligible result, in salt order
            bs = [b for b in rec["boundaries"]
                  if any(it["eligible"] for it in items(b["messages"], s.bulky_chars))]
            bs.sort(key=lambda b: _u("ctx-b", case, b["b"]))
            for b in bs[:s.context_boundaries]:
                out.append(self._ctx_decision(meta, b))
            # type 3: natural failures (pending failed read-only result)
            fs = [(b, it) for b in rec["boundaries"]
                  for it in items(b["messages"], s.bulky_chars)
                  if it["pending"] and it["read_only"] and it["failed"]]
            fs.sort(key=lambda x: _u("fail-b", case, x[0]["b"], x[1]["i"]))
            seen: set[int] = set()
            for b, it in fs:
                if b["b"] in seen or len(seen) >= s.failures_per_episode:
                    continue
                seen.add(b["b"])
                out.append(self._rec_decision(meta, b, it, injected=False))
            if case in inj_cases:
                ok = [(b, it) for b in rec["boundaries"]
                      for it in items(b["messages"], s.bulky_chars)
                      if it["pending"] and it["read_only"] and not it["failed"]]
                if ok:
                    b, it = min(ok, key=lambda x: (x[0]["b"], x[1]["i"]))
                    out.append(self._rec_decision(meta, b, it, injected=True))
        self.dec_log.write(out)
        return out

    def _calib(self, did: str) -> bool:
        return _u("calib", did) < self.sizes.calibration_rate

    def _ctx_decision(self, meta: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
        s, wire = self.sizes, b["messages"]
        its = items(wire, s.bulky_chars)
        elig = [it for it in its if it["eligible"]]
        did = f"ctx|{meta['case']}|b{b['b']}"
        opts: dict[str, list[dict[str, Any]]] = {"keep": []}
        for g in range(s.groups):
            opts[f"group{g + 1}"] = group_edits(wire, elig, key=f"{did}|g{g}")
        opts["native_low"] = native_edits(wire, b.get("native"), elig)
        budget = removed_chars(wire, opts["native_low"]) or removed_chars(wire, opts["group1"])
        opts["heuristic"] = heuristic_edits(wire, elig, budget)
        # CH: Hermes' native prune at its configured floors (it rarely fires here)
        opts["native_default"] = native_edits(wire, b.get("native_default"), elig)
        for k in LAST_KS:  # CK: the k grid (calibration picks k; every k is run)
            opts[f"last_k{k}"] = [_edit_stub(wire, it) for it in last_k_items(wire, elig, k)]
        # CP: the reversible variant at the default k
        opts["reversible"] = [_edit_reversible(wire, it)
                              for it in last_k_items(wire, elig, LAST_K)]
        opts["native_low_ids"] = native_ids_edits(wire, opts["native_low"])
        opts["native_default_ids"] = native_ids_edits(wire, opts["native_default"])
        target = elig[int(_u("placebo", did) * len(elig))]  # elig is never empty here
        for v, name in enumerate(PLACEBOS, 1):
            opts[name] = placebo_edits(wire, target["i"], v)
        prompt, lab = prompt_context(wire, elig)
        ans = self._ask(prompt)
        drop = ((ans or {}).get("parsed") or {}).get("drop")
        if isinstance(drop, list):
            pick = [it for it in elig if it["i"] in {lab.get(str(x)) for x in drop}]
            opts["prompted"] = [_edit_stub(wire, it) for it in pick]
        return {"id": did, "type": 1, **meta, "b": b["b"],
                "removed_calls": {k: removed_calls(wire, v) for k, v in opts.items()},
                "inputs": {"decision": "context_relevance", "n_msgs": b["n_msgs"],
                           "digest": b["digest"][:16],
                           "items": items_with_features(wire, s.bulky_chars),
                           "eligible": [it["i"] for it in elig],
                           "options": ["keep"] + [f"drop:{k}" for k in opts if k != "keep"]},
                "options": {k: [{"i": e["i"], "kind": e["kind"], "content": e["content"]}
                                for e in v] for k, v in opts.items()},
                "removed_chars": {k: removed_chars(wire, v) for k, v in opts.items()},
                "flags": {k: "native (may drop identifiers)" for k in NATIVE_ARMS
                          if any(drops_identifier(_content(wire[e["i"]].get("content")),
                                                  e["content"]) for e in opts[k])},
                "native_default_n": b.get("native_default_n"),
                "prompted": None if ans is None else {
                    "abstain": not isinstance(drop, list),
                    "choice": [str(x) for x in drop] if isinstance(drop, list) else None,
                    "confidence": ((ans.get("parsed") or {}).get("confidence")),
                    "tokens": [ans.get("input"), ans.get("output")], "ok": ans.get("ok")},
                "calibration": self._calib(did)}

    def _rec_decision(self, meta, b, it, *, injected: bool) -> dict[str, Any]:
        wire = b["messages"]
        did = f"rec|{meta['case']}|b{b['b']}|i{it['i']}" + ("|inj" if injected else "")
        opts = {o: recovery_edits(wire, it["i"], o, injected=injected) for o in RECOVERY}
        fix = None if injected else deterministic_repair(wire, it["i"])
        if fix:
            base = _content(wire[it["i"]].get("content"))
            opts["repair_fixed"] = [{"i": it["i"], "kind": "repair_fixed", "content": (
                base + f"\n[harness recovery note: the call failed; retry it exactly as "
                       f"{fix}.]")}]
        base_c = INJECTED_ERROR if injected else None
        for v, name in enumerate(PLACEBOS, 1):
            opts[name] = placebo_edits(wire, it["i"], v, base=base_c)
        ans = self._ask(prompt_recovery(wire, it["i"], injected=injected))
        rev = (self._ask(prompt_recovery(wire, it["i"], injected=injected, reverse=True))
               if _u("wording", did) < WORDING_AUDIT else None)
        rch = ((rev or {}).get("parsed") or {}).get("choice")
        ch = ((ans or {}).get("parsed") or {}).get("choice")
        return {"id": did, "type": 3, **meta, "b": b["b"], "injected": injected,
                "inputs": {"decision": "read_only_recovery", "n_msgs": b["n_msgs"],
                           "digest": b["digest"][:16], "tool": it["tool"],
                           "failure": "injected_transient" if injected else it["failed"],
                           "item": it, "options": list(RECOVERY)},
                "options": {k: [{"i": e["i"], "kind": e["kind"], "content": e["content"]}
                                for e in v] for k, v in opts.items()},
                "prompted": None if ans is None else {
                    "abstain": ch not in RECOVERY, "choice": ch if ch in RECOVERY else None,
                    "confidence": ((ans.get("parsed") or {}).get("confidence")),
                    "tokens": [ans.get("input"), ans.get("output")], "ok": ans.get("ok"),
                    # wording audit: the same question with the options in reverse order
                    "reversed_choice": (rch if rch in RECOVERY else "abstain")
                    if rev is not None else None},
                "calibration": self._calib(did)}

    # -------------------------------------------------------------- phase 3: branches

    def branch_specs(self, decisions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Distinct edit sets run once (twice on the calibration share); the reference
        option (keep / native) always twice. Options with the same edits share branches."""
        out = []
        for d in decisions:
            ref = "keep" if d["type"] == 1 else "native"
            by_key: dict[str, list[str]] = {}
            for name, ed in d["options"].items():
                by_key.setdefault(edits_key(ed), []).append(name)
            for key, names in sorted(by_key.items(), key=lambda kv: min(kv[1])):
                is_ref = ref in names
                reps = self.sizes.keep_replicates if is_ref or d["calibration"] else 1
                ed = d["options"][names[0]]
                for r in range(reps):
                    out.append({"mode": "branch", "decision": d["id"], "case": d["case"],
                                "base_tag": d["base_tag"], "b": d["b"], "edits": ed,
                                "removed_calls": d.get("removed_calls", {}).get(names[0], []),
                                "options": sorted(names), "edits_key": key, "rep": r,
                                "tag": f"pr|{d['id']}|{key}|r{r}"})
        return out

    def run_branches(self, decisions: Sequence[Mapping[str, Any]]) -> None:
        pool = self.adapter.surface.pool
        done = {r["tag"] for r in self.br_log.rows()}
        todo = [s for s in self.branch_specs(decisions) if s["tag"] not in done]
        for k in range(0, len(todo), self.sizes.chunk):
            part = todo[k:k + self.sizes.chunk]
            pairs = self._dispatch([(probe_unit(pool, s["case"]), s) for s in part])
            self._incidents(pairs)
            rows = []
            for s, (o, d) in zip(part, pairs, strict=True):
                pr = d.get("probe") or {}
                row = {"tag": s["tag"], "decision": s["decision"], "options": s["options"],
                       "edits_key": s["edits_key"], "rep": s["rep"], "n_edits": len(s["edits"]),
                       **outcome_compact(o, d), "valid": pr.get("valid"),
                       "diverged": pr.get("diverged"), "live": pr.get("live"),
                       "live_calls": pr.get("live_calls"),
                       "cont_calls": pr.get("cont_calls") or [],
                       "removed_calls": s.get("removed_calls") or [],
                       "runner_errors": pr.get("runner_errors")}
                if _u("dialogue", s["tag"]) < self.sizes.dialogue_sample or \
                        o.critical_harm is not False:
                    row["dialogue"] = [{"role": t.get("role"), "content": str(
                        t.get("content") or "")[:600]} for t in d.get("dialogue") or []]
                rows.append(row)
            self.br_log.append(rows)
            self.log(f"probe branches: {min(k + len(part), len(todo))}/{len(todo)}")

    # -------------------------------------------------------------- phase 4: review choice

    def run_review_prompted(self, run_dir: Path) -> None:
        """The prompted 9B's run/skip choice for every C evidence episode of this block (type
        2 baseline; the inputs are the episode's own messages, no labels)."""
        if self.chooser is None or self.rev_log.rows():
            return
        from slh.system.hosts.hermes.mh1 import Jsonl

        p = Path(run_dir) / "C" / "system" / "traces.jsonl"
        rows = []
        for r in Jsonl(p).rows() if p.exists() else []:
            if r.get("role") != "evidence":
                continue
            msgs = (r.get("trace") or {}).get("messages")
            ans = self._ask(prompt_review(msgs)) or {}
            v = (ans.get("parsed") or {}).get("review")
            rows.append({"arm": "C", "batch": r.get("batch"), "unit": r.get("unit_id"),
                         "episode": episode_digest(r.get("trace") or {}),
                         "choice": v if isinstance(v, bool) else None,
                         "confidence": (ans.get("parsed") or {}).get("confidence"),
                         "tokens": [ans.get("input"), ans.get("output")]})
        self.rev_log.write(rows)

    # -------------------------------------------------------------- all

    def fidelity(self) -> dict[str, Any]:
        rows = self.br_log.rows()
        n = len(rows)
        bad = sum(1 for r in rows if not r.get("valid"))
        return {"branches": n, "diverged": bad, "rate": round(bad / n, 4) if n else None}

    def run(self, run_dir: Path, *, require_fidelity: Mapping[str, Any] | None = None
            ) -> dict[str, Any]:
        """Phases 1-4. ``require_fidelity``: the smoke run's report; the full probe is
        skipped (recorded, M-H1 unaffected) when the smoke's prefix replay diverged."""
        st = self.dir / "status.json"
        if require_fidelity is not None:
            f = (require_fidelity or {}).get("fidelity") or {}
            if not f.get("branches") or (f.get("rate") or 0) > FIDELITY_MAX_DIVERGED:
                rep = {"skipped": "the smoke run's prefix replay fidelity did not pass",
                       "smoke_fidelity": f}
                st.write_text(json.dumps(rep, indent=1))
                self.log(f"probe skipped: {rep}")
                return rep
        t0 = time.time()
        self.run_base()
        dec = self.plan_decisions()
        self.run_branches(dec)
        self.run_review_prompted(run_dir)
        rep = {"probe_id": self.plan["probe_id"], "plan_sha256": self.plan["sha256"],
               "smoke": self.smoke, "base": len(self.base_log.rows()),
               "decisions": {t: sum(1 for d in dec if d["type"] == t) for t in (1, 3)},
               "fidelity": self.fidelity(), "minutes": round((time.time() - t0) / 60, 1),
               "review_prompted": len(self.rev_log.rows())}
        try:
            files = export_rows(run_dir)
        except ProbeBudgetError as e:
            rep["bytes"] = {"over_budget": str(e)}
            st.write_text(json.dumps(rep, indent=1))
            return rep
        nb = sum(len(gzip.compress(b, mtime=0)) for k, b in files.items()
                 if k not in ("probe/status.json", "probe/review-rows.jsonl",
                              "probe/review-prompted.jsonl"))
        rep["bytes"] = {"per_parent": round(nb / max(1, rep["base"])),
                        "bound": PER_PARENT_BOUND, "total": nb,
                        "within_bound": nb <= PER_PARENT_BOUND * max(1, rep["base"])}
        st.write_text(json.dumps(rep, indent=1))
        self.log(f"probe: {rep}")
        return rep


# ============================================================================ type 2 rows


def episode_digest(trace: Mapping[str, Any]) -> str:
    return _sha(_canon(_norm(trace.get("messages") or [])))[:16]


def _rec_sha(rec: Mapping[str, Any]) -> str:
    return _sha(_canon(rec.get("payload") or {}))[:16]


def review_rows(run_dir: Path, *, audit_rate: float = 0.2) -> list[dict[str, Any]]:
    """Type 2 labels (schema only, no new episode): every review-fork output of M-H1's
    evidence episodes (B and C) with an episode digest; ``kind`` empty / duplicate /
    proposals; for C the fate of the episode's records at our gate (dropped before the
    screen, screened: retired / carried / advanced, held, adopted); a random ``audit`` flag
    (forks a future skip policy must still run, to measure what skipping loses)."""
    from slh.system.hosts.hermes.mh1 import Jsonl

    run_dir = Path(run_dir)
    out: list[dict[str, Any]] = []
    sysd = run_dir / "C" / "system"
    audit = Jsonl(sysd / "audit.jsonl").rows() if (sysd / "audit.jsonl").exists() else []
    fate: dict[str, str] = {}  # record id -> furthest stage
    order = ["proposed", "dropped", "screen_retire", "screen_carry", "screen_hold",
             "screen_advance", "confirm_hold", "adopted"]

    def bump(rid: str, st: str) -> None:
        if order.index(st) > order.index(fate.get(rid, "proposed")):
            fate[rid] = st

    cid_recs: dict[str, list[str]] = {}
    for r in audit:
        k = r.get("kind")
        if k == "propose":
            c = r.get("candidate") or {}
            cid_recs[c.get("cid")] = [str(x) for x in
                                      (c.get("provenance") or {}).get("records") or []]
            for rid in cid_recs[c.get("cid")]:
                fate.setdefault(rid, "proposed")
        elif k in ("propose_drop", "policy_drop", "size_drop", "kind_drop", "conflict_drop"):
            for rid in cid_recs.get(r.get("cid"), []):
                bump(rid, "dropped")
        elif k == "screen":
            for cid, d in ((r.get("result") or {}).get("decisions") or {}).items():
                st = {"retire": "screen_retire", "carry": "screen_carry",
                      "advance": "screen_advance"}.get(str(d.get("decision")), "screen_carry")
                for rid in cid_recs.get(cid, []):
                    bump(rid, st)
        elif k == "screen_hold":
            for cid in r.get("candidates") or []:
                for rid in cid_recs.get(cid, []):
                    bump(rid, "screen_hold")
        elif k == "critical_hold":
            for cid in r.get("candidates") or []:
                for rid in cid_recs.get(cid, []):
                    bump(rid, "confirm_hold")
        elif k == "confirm_close" and r.get("adopt"):
            for cid in r.get("candidates") or r.get("adopted") or []:
                for rid in cid_recs.get(cid, []):
                    bump(rid, "adopted")

    def row(arm: str, batch: Any, unit: str, trace_or_row: Mapping[str, Any],
            staged: Sequence[Mapping[str, Any]], seen: set[str],
            episode: Mapping[str, Any]) -> dict[str, Any]:
        usage = trace_or_row.get("usage") or {}
        shas = [_rec_sha(x) for x in staged]
        kind = "empty" if not staged else ("duplicate" if all(s in seen for s in shas)
                                           else "proposals")
        seen.update(shas)
        fates = [fate.get(str(x.get("id")), "not_seen") for x in staged] if arm == "C" else []
        best = max(fates, key=lambda f: order.index(f) if f in order else -1) if fates else None
        rv = trace_or_row.get("review")
        return {"arm": arm, "batch": batch, "unit": unit,
                "episode": episode_digest(trace_or_row),
                "review": rv.get("status") if isinstance(rv, Mapping) else rv,
                "fork_tokens": [int(usage.get("fork_prompt_tokens") or 0),
                                int(usage.get("fork_completion_tokens") or 0)],
                "fork_calls": int(usage.get("fork_api_calls") or 0),
                "records": [{"subsystem": x.get("subsystem"), "sha": s}
                            for x, s in zip(staged, shas, strict=True)],
                "kind": kind, "c_fate": best,
                "survives": (best in ("screen_advance", "screen_hold", "confirm_hold",
                                      "adopted")) if arm == "C" else None,
                "audit": _u("review-audit", arm, unit, batch) < audit_rate,
                # the finished episode the fork would review (known BEFORE the fork runs:
                # inputs of the pre-registered skip rules 'no task write', 'episode
                # success', 'short episode'); numbers only
                **episode}

    seen_c: dict[Any, set[str]] = {}
    tp = sysd / "traces.jsonl"
    for r in Jsonl(tp).rows() if tp.exists() else []:
        if r.get("role") != "evidence":
            continue
        tr = r.get("trace") or {}
        out.append(row("C", r.get("batch"), r.get("unit_id"), tr, tr.get("staged") or [],
                       seen_c.setdefault(r.get("batch"), set()),
                       _episode_facts(r.get("status"), r.get("success"), tr.get("writes"),
                                      sum(1 for m in tr.get("dialogue") or []
                                          if m.get("role") == "user"),
                                      tr.get("usage"))))
    for p in sorted((run_dir / "B").glob("evidence-*.jsonl")):
        seen: set[str] = set()
        k = int(p.stem.split("-")[1])
        for r in Jsonl(p).rows():
            out.append(row("B", k, r.get("unit"), r, r.get("staged_records") or [], seen,
                           _episode_facts(r.get("status"), r.get("success"), r.get("writes"),
                                          r.get("acts"), r.get("usage"))))
    return out


def _episode_facts(status: Any, success: Any, writes: Any, turns: Any, usage: Any
                   ) -> dict[str, Any]:
    """Episode facts for the review skip rules: status, success (None unless scored),
    whether any task write was EXECUTED, customer turns (the dialogue's user turns, stop
    turn included) and the main agent's tokens."""
    u = usage or {}
    return {"ep_status": status, "ep_success": success if status == "scored" else None,
            "ep_task_write": bool(writes), "ep_turns": None if turns is None else int(turns),
            "ep_tokens": [int(u.get("session_prompt_tokens") or 0),
                          int(u.get("session_completion_tokens") or 0)]}


# ============================================================================ export


_HEX16 = re.compile(r"[0-9a-f]{16}")


def _short(x: Any) -> Any:
    """Every 16-hex reference shortened to 8 hex (unique within a parent; the export's
    size lever of the review)."""
    if isinstance(x, str):
        return x[:8] if _HEX16.fullmatch(x) else x
    if isinstance(x, list):
        return [_short(y) for y in x]
    if isinstance(x, dict):
        return {k: _short(v) for k, v in x.items()}
    return x


class ProbeBudgetError(RuntimeError):
    """The probe's rows do not fit its share of the return even fully trimmed."""


def _branch_export(r: Mapping[str, Any], *, lists: bool) -> dict[str, Any]:
    keep = ("decision", "options", "edits_key", "rep", "n_edits", "status", "success",
            "harm", "critical", "incident", "mediated", "seconds", "killed", "tokens",
            "api_calls", "valid", "live_calls")
    out = {k: r.get(k) for k in keep}
    removed = {tuple(c) for c in r.get("removed_calls") or []}
    cont = [tuple(c) for c in r.get("cont_calls") or []]
    out["recalls"] = sum(1 for c in cont if c in removed)
    out["n_cont_calls"], out["n_removed_calls"] = len(cont), len(removed)
    if lists:
        out["removed_calls"], out["cont_calls"] = r.get("removed_calls") or [], \
            r.get("cont_calls") or []
    for k in ("diverged", "runner_errors"):
        if r.get(k):
            out[k] = r[k]
    return out


def export_rows(run_dir: Path, budget: int = BUDGET_BYTES) -> dict[str, bytes]:
    """The probe's part of a block's export (file name -> uncompressed bytes), HARD-capped at
    ``budget`` compressed bytes. References only: tool results as (length, 8-hex hash),
    option edits as (index, kind, hash; Hermes' native demotion text kept), branches as
    outcomes and live token counts; minimal text deltas of the base episodes (agent and
    customer text, capped per message). Trimming, in this order, until it fits: the
    per-branch call lists (counts and re-calls stay); the sampled continuation dialogues
    (critical ones always stay); the base-delta text per message (1,000 / 500 / 250 / 120 /
    0 characters); the per-item features. Every step taken is recorded in ``probe/export-trims.json``. If even that does
    not fit, ``ProbeBudgetError`` (the export then says so; nothing is cut silently)."""
    from slh.system.hosts.hermes.mh1 import Jsonl

    d = Path(run_dir) / "probe"
    raw_rows: dict[str, list[dict[str, Any]]] = {}
    if d.exists():
        decs = []
        for r in Jsonl(d / "decisions.jsonl").rows():
            r = dict(r)
            r["options"] = {k: [{"i": e["i"], "kind": e["kind"],
                                 "sha": _sha(e["content"])[:8],
                                 # notes, stubs and placebos are rebuilt from the plan and
                                 # the base context; Hermes' native demotion is kept
                                 **({"content": e["content"]} if e["kind"] == "native"
                                    else {})}
                                for e in v] for k, v in r["options"].items()}
            decs.append(r)
        raw_rows = {"base": Jsonl(d / "base.jsonl").rows(), "decisions": decs,
                    "branches": Jsonl(d / "branches.jsonl").rows(),
                    "review_prompted": Jsonl(d / "review-prompted.jsonl").rows()}
    rev = review_rows(run_dir)
    fixed: dict[str, bytes] = {}
    for name in ("status.json", "error.json"):
        if (d / name).exists():
            fixed[f"probe/{name}"] = (d / name).read_bytes()

    def jl(rows: Sequence[Mapping[str, Any]]) -> bytes:
        return "".join(json.dumps(_short(r), sort_keys=True, default=str) + "\n"
                       for r in rows).encode()

    def build(dia_keep: float, chars: int | None, lists: bool, feats: bool
              ) -> dict[str, bytes]:
        fs = dict(fixed)
        base = []
        for r in raw_rows.get("base", []):
            r = dict(r)
            if chars is not None:
                r["deltas"] = [[m[0], str(m[1])[:chars], *m[2:]] if m[0] in ("a", "u")
                               else m for m in r.get("deltas") or []]
            base.append(r)
        decs = raw_rows.get("decisions", [])
        if not feats:
            decs = [{**x, "inputs": {**x["inputs"], "items": [
                {k: v for k, v in it.items() if k != "features"}
                for it in x["inputs"].get("items") or []]}} if x.get("inputs") else x
                for x in decs]
        brs = raw_rows.get("branches", [])
        dia = [{"decision": r["decision"], "edits_key": r.get("edits_key"), "rep": r["rep"],
                "dialogue": r["dialogue"]} for r in brs if r.get("dialogue")]
        crit = [x for x, r in zip(dia, [b for b in brs if b.get("dialogue")], strict=True)
                if r.get("critical") is not False]
        rest = [x for x in dia if x not in crit]
        dia = crit + rest[:int(len(rest) * dia_keep)]  # critical ones are always kept
        if raw_rows:
            fs["probe/base.jsonl"] = jl(base)
            fs["probe/decisions.jsonl"] = jl(decs)
            fs["probe/branches.jsonl"] = jl([_branch_export(r, lists=lists) for r in brs])
            fs["probe/dialogues.jsonl"] = jl(dia)
            fs["probe/review-prompted.jsonl"] = jl(raw_rows["review_prompted"])
        fs["probe/review-rows.jsonl"] = jl(rev)
        return fs

    def size(fs: Mapping[str, bytes]) -> int:
        return sum(len(gzip.compress(b, mtime=0)) for b in fs.values())

    # the per-branch call lists go first (their counts and the re-call count stay, which
    # is all the cost uses), then the sampled dialogues, then base-delta text, then the
    # per-item features
    steps = [(1.0, None, True, True), (1.0, None, False, True), (0.5, None, False, True),
             (0.25, None, False, True), (0.0, None, False, True)] + [
        (0.0, c, False, True) for c in (1000, 500, 250, 120, 0)] + [(0.0, 0, False, False)]
    trims = []
    for st in steps:
        files = build(*st)
        n = size(files)
        trims.append({"dialogue_share": st[0], "delta_chars": st[1], "call_lists": st[2],
                      "features": st[3], "bytes": n})
        if n > budget:
            continue
        files["probe/export-trims.json"] = json.dumps(
            {"budget": budget, "steps": trims}).encode()
        if size(files) <= budget:  # the FINAL part, trims record included: exact
            return files
    raise ProbeBudgetError(f"probe rows are {trims[-1]['bytes']} B compressed even fully "
                           f"trimmed; the probe's share is {budget} B")


# ============================================================================ our side


def rebuild_tool_results(deltas: Sequence[Sequence[Any]], retail_db: str
                         ) -> list[dict[str, Any]]:
    """Our side: re-execute the base episode's calls, in order, on a FRESH retail
    database and check every result against its exported (length, sha) reference. Returns
    one row per tool message: the rebuilt content and whether it matched."""
    from tau2.data_model.message import ToolCall

    from slh.xp import retaildb

    env = retaildb.env(retail_db)
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    out = []
    for m in deltas:
        if m[0] == "a":
            for cid, name, args in m[2]:
                try:
                    a = json.loads(args or "{}")
                except (json.JSONDecodeError, ValueError):
                    a = {}
                calls[cid] = (name, a if isinstance(a, dict) else {})
        elif m[0] == "t":
            cid, n, sha = m[1], m[2], m[3]
            if cid not in calls:
                out.append({"call_id": cid, "content": None, "ok": False})
                continue
            name, a = calls[cid]
            r = env.get_response(ToolCall(id=cid, name=name, arguments=a,
                                          requestor="assistant"))
            c = str(r.content)
            out.append({"call_id": cid, "content": c,
                        "ok": len(c) == n and _sha(c)[:len(sha)] == sha})
    return out


def _paired(a: Sequence[Any], b: Sequence[Any]) -> dict[str, Any]:
    """Paired binary outcomes (None skipped): n, mean difference a - b, discordant counts,
    exact two-sided McNemar p."""
    from slh.system.hosts.hermes.mh1 import _binom_two_sided

    pairs = [(bool(x), bool(y)) for x, y in zip(a, b, strict=True)
             if x is not None and y is not None]
    up = sum(1 for x, y in pairs if x and not y)
    dn = sum(1 for x, y in pairs if y and not x)
    n = len(pairs)
    return {"n": n, "diff": round((up - dn) / n, 4) if n else None, "n_plus": up,
            "n_minus": dn, "p": _binom_two_sided(min(up, dn), up + dn) if up + dn else 1.0}


BOOT_REPS, BOOT_SEED = 10_000, 20261025
WORDING_AUDIT = 0.2  # share of decisions also asked with the options reversed


def _boot_mean(vals: Sequence[float], clusters: Sequence[str], reps: int = BOOT_REPS
               ) -> tuple[float | None, float | None, float | None]:
    if not vals:
        return None, None, None
    rng = random.Random(BOOT_SEED)
    by: dict[str, list[float]] = {}
    for v, c in zip(vals, clusters, strict=True):
        by.setdefault(c, []).append(v)
    keys = sorted(by)
    means = []
    for _ in range(reps):
        pick = [by[keys[rng.randrange(len(keys))]] for _ in keys]
        flat = [x for p in pick for x in p]
        means.append(sum(flat) / len(flat))
    means.sort()
    return (round(sum(vals) / len(vals), 4), round(means[int(0.025 * reps)], 4),
            round(means[int(0.975 * reps)], 4))


def branch_cost(r: Mapping[str, Any], extra: Sequence[int] | None = None
                ) -> dict[str, Any] | None:
    """All-in cost of a branch's live continuation (``probe_cost.EpisodeCost``, units
    ``COST_RATES``): every live agent request (prefill net of reported prefix-cache hits;
    an unreported cache count is recorded as an estimate of 0), every continuation tool
    call (re-calls of a removed result counted), and ``extra`` = the decision's own
    (input, output) tokens for an arm that needed a model call (the prompted 9B)."""
    from slh.system.hosts.hermes import probe_cost as PC

    calls = r.get("live_calls")
    if calls is None:
        return None
    ec = PC.EpisodeCost(PC.TokenRates(**COST_RATES))
    for i, o, c in calls:
        u: dict[str, Any] = {"prompt_tokens": int(i), "completion_tokens": int(o)}
        if c is not None:
            u["prompt_tokens_details"] = {"cached_tokens": min(int(c), int(i))}
        ec.add_model_call(u)
    if extra and extra[0] is not None:
        ec.add_model_call({"prompt_tokens": int(extra[0]),
                           "completion_tokens": int(extra[1] or 0)})
    if "cont_calls" in r:
        for name, argsha in r.get("removed_calls") or []:
            ec.mark_pruned_call(name, argsha)
        for name, argsha in r.get("cont_calls") or []:
            ec.observe_tool_call(name or "tool", argsha)
    else:  # the export trimmed the call lists: counts (re-calls kept) stand in for them
        n_rc = int(r.get("recalls") or 0)
        for j in range(n_rc):
            ec.mark_pruned_call("removed", str(j))
            ec.observe_tool_call("removed", str(j))
        for j in range(max(0, int(r.get("n_cont_calls") or 0) - n_rc)):
            ec.observe_tool_call("other", str(j))
    secs = r.get("seconds")
    return ec.record(elapsed_ms=None if secs is None else float(secs) * 1000.0)


def _pct(vals: Sequence[float], q: float) -> float | None:
    from slh.system.hosts.hermes import probe_cost as PC

    return round(PC.percentile(vals, q), 2) if vals else None


def _cost(r: Mapping[str, Any], extra: Sequence[int] | None = None) -> float | None:
    rec = branch_cost(r, extra)
    return None if rec is None else float(rec["total_usd"]) * 1e6  # cost units


def analyze(exports: Sequence[Path]) -> dict[str, Any]:
    """Pre-registered headroom report (design s. 4) over the probe parts of the exports."""
    base, decs, brs, revs = [], [], [], []
    for e in exports:
        from slh.system.hosts.hermes.mh1 import _read_gz_jsonl

        e = Path(e)
        base += _read_gz_jsonl(e / "probe" / "base.jsonl.gz")
        decs += _read_gz_jsonl(e / "probe" / "decisions.jsonl.gz")
        brs += _read_gz_jsonl(e / "probe" / "branches.jsonl.gz")
        revs += _read_gz_jsonl(e / "probe" / "review-rows.jsonl.gz")
    by_dec: dict[str, list[dict[str, Any]]] = {}
    for r in brs:
        by_dec.setdefault(r["decision"], []).append(r)
    base_cost = {r["tag"]: COST_RATES["prefill"] * float(r["tokens"][0])
                 + COST_RATES["decode"] * float(r["tokens"][1]) for r in base}
    fid = {"branches": len(brs), "diverged": sum(1 for r in brs if not r.get("valid"))}
    fid["rate"] = round(fid["diverged"] / fid["branches"], 4) if fid["branches"] else None
    rep: dict[str, Any] = {"fidelity": fid, "base_episodes": len(base),
                           "fidelity_ok": fid["rate"] is not None
                           and fid["rate"] <= FIDELITY_MAX_DIVERGED}

    def arm_rows(d: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
        return [r for r in by_dec.get(d["id"], []) if name in r["options"] and r.get("valid")]

    def compare(dtype: int, ref: str, arms: Sequence[str]) -> dict[str, Any]:
        ds = [d for d in decs if d["type"] == dtype]
        res: dict[str, Any] = {"decisions": len(ds)}
        r1 = {d["id"]: next((r for r in arm_rows(d, ref) if r["rep"] == 0), None) for d in ds}
        r2 = {d["id"]: next((r for r in arm_rows(d, ref) if r["rep"] == 1), None) for d in ds}
        for name in [ref + "_rep2", *arms]:
            xs, ys, sv, cl = [], [], [], []
            hx, hy, cx, cy = [], [], [], []
            recalls, secs, nx, ny = 0, [], [], []
            for d in ds:
                a = r2[d["id"]] if name == ref + "_rep2" else next(
                    (r for r in arm_rows(d, name) if r["rep"] == 0), None)
                b = r1[d["id"]]
                if a is None or b is None:
                    continue
                xs.append(a["success"])
                ys.append(b["success"])
                hx.append(a["harm"])
                hy.append(b["harm"])
                cx.append(a["critical"])
                cy.append(b["critical"])
                nx.append(a["status"] != "scored")
                ny.append(b["status"] != "scored")
                ca = _cost(a, (d.get("prompted") or {}).get("tokens")
                           if name == "prompted" else None)
                cb = _cost(b)
                rec = branch_cost(a)
                recalls += int((rec or {}).get("pruned_recalls") or 0)
                if a.get("seconds") is not None:
                    secs.append(float(a["seconds"]))
                whole = base_cost.get(d["base_tag"])
                if ca is not None and cb is not None and whole:
                    sv.append((cb - ca) / whole)
                    cl.append(d["case"])
            m, lo, hi = _boot_mean(sv, cl)
            res[name] = {"success": _paired(xs, ys), "harm": _paired(hx, hy),
                         "critical": _paired([c is not False for c in cx],
                                             [c is not False for c in cy]),
                         "cost_saving_share": {"mean": m, "ci95": [lo, hi]},
                         "non_scored": _paired(nx, ny),
                         "recalls_of_removed": recalls,
                         "seconds_p50_p95": [_pct(secs, 50), _pct(secs, 95)]}
        real = [a for a in arms if a not in NATIVE_ARMS and a not in PLACEBOS]
        res.update(oracles(ds, real, ref))
        return res

    def arm_cost(d: Mapping[str, Any], name: str, r: Mapping[str, Any]) -> float | None:
        """A branch's all-in cost AS this option: the prompted arm also pays its chooser
        request (the same rule as the fixed-arm comparison)."""
        return _cost(r, (d.get("prompted") or {}).get("tokens") if name == "prompted"
                     else None)

    def row(d: Mapping[str, Any], name: str, rep: int) -> dict[str, Any] | None:
        return next((x for x in arm_rows(d, name) if x["rep"] == rep), None)

    def not_worse(r: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        return (bool(r["success"]) >= bool(b["success"]) and bool(r["harm"]) <= bool(b["harm"])
                and (r["critical"] is False or b["critical"] is not False)
                and (r["status"] == "scored" or b["status"] != "scored"))

    def pick(d: Mapping[str, Any], names: Sequence[str], rep: int, ref: str
             ) -> dict[str, Any] | None:
        """The pre-registered rule: on the selection repeat, the CHEAPEST option whose
        outcome is not worse than the reference's (success, harm, critical, non-scored);
        ties by the frozen action order; the reference is always eligible (fallback)."""
        b = row(d, ref, rep)
        if b is None:
            return None
        best = None
        for order, name in enumerate(names):
            r = row(d, name, rep)
            if r is None or not not_worse(r, b):
                continue
            c = arm_cost(d, name, r)
            k = (c if c is not None else float("inf"), name != ref, order)
            if best is None or k < best[0]:
                best = (k, {"name": name, "row": r})
        return best[1] if best else None

    def oracle(ds: Sequence[Mapping[str, Any]], names_of: Callable[[Mapping[str, Any]],
               Sequence[str]], ref: str) -> dict[str, Any]:
        """Select on replicate 0 (``pick``), assess on replicate 1: ``envelope`` = the
        chosen replicate-0 row vs ref replicate 1 (an optimistic envelope, never a gain);
        ``assessed`` = the chosen option's replicate 1 vs ref replicate 1. At T = 0 the
        repeats differ only by serving noise, so ``assessed`` does NOT remove the
        selection bias; headroom is judged against the placebo oracle (design s. 10)."""
        acc_ = {"envelope": {"s": [], "r": [], "sv": [], "cl": []},
                "assessed": {"s": [], "r": [], "sv": [], "cl": []}}
        chosen: dict[str, int] = {}
        for d in ds:
            ch = pick(d, names_of(d), 0, ref)
            r1 = row(d, ref, 1)
            if ch is None or r1 is None:
                continue
            chosen[ch["name"]] = chosen.get(ch["name"], 0) + 1
            whole = base_cost.get(d["base_tag"]) or 0.0
            for nm, rr in (("envelope", ch["row"]), ("assessed", row(d, ch["name"], 1))):
                if rr is None:
                    continue
                acc = acc_[nm]
                acc["s"].append(rr["success"])
                acc["r"].append(r1["success"])
                ca, cb = arm_cost(d, ch["name"], rr), _cost(r1)
                if ca is not None and cb is not None and whole:
                    acc["sv"].append((cb - ca) / whole)
                    acc["cl"].append(d["case"])
        out: dict[str, Any] = {"chosen": chosen}
        for nm, acc in acc_.items():
            m, lo, hi = _boot_mean(acc["sv"], acc["cl"])
            out[nm] = {"success": _paired(acc["s"], acc["r"]),
                       "cost_saving_share": {"mean": m, "ci95": [lo, hi]}}
        return out

    def oracles(ds: Sequence[Mapping[str, Any]], real: Sequence[str], ref: str
                ) -> dict[str, Any]:
        """Real-option oracle (native arms and placebos excluded), the placebo oracle
        (ref + the meaning-preserving placebos) and a size-matched real oracle (ref + as
        many real options as placebos, drawn per decision by hash). Headroom = matched real
        minus placebo, on the assessed repeat."""
        k = len(PLACEBOS)

        def matched(d: Mapping[str, Any]) -> list[str]:
            have = [n for n in real if row(d, n, 0) is not None]
            have.sort(key=lambda n: _u("matched", d["id"], n))
            return [ref, *have[:k]]

        o_real = oracle(ds, lambda d: [ref, *real], ref)
        o_plc = oracle(ds, lambda d: [ref, *PLACEBOS], ref)
        o_mat = oracle(ds, matched, ref)

        def mean(o: Mapping[str, Any], f: str) -> float | None:
            if f == "success":
                return o["assessed"]["success"]["diff"]
            return o["assessed"]["cost_saving_share"]["mean"]

        def diff(a: Any, b: Any) -> float | None:
            return None if a is None or b is None else round(a - b, 4)

        return {"oracle_real": o_real, "oracle_placebo": o_plc, "oracle_real_matched": o_mat,
                "headroom_vs_placebo": {
                    f: diff(mean(o_mat, f), mean(o_plc, f))
                    for f in ("success", "cost_saving_share")}}

    rep["type1"] = compare(1, "keep", ["native_default", "native_default_ids", "group1",
                                       "group2", "native_low", "native_low_ids", "heuristic",
                                       *[f"last_k{k}" for k in LAST_KS], "reversible",
                                       "prompted", *PLACEBOS])
    rep["native_arms_note"] = ("native_low / native_default are Hermes' own demotions (may "
                               "drop identifiers): reported as a record, excluded from the "
                               "oracle; compare the *_ids variants")
    for stratum, inj in (("natural", False), ("injected", True)):
        ds_ok = [d for d in decs if d["type"] == 3 and bool(d.get("injected")) == inj]
        saved = decs
        decs = [d for d in saved if d["type"] != 3] + ds_ok
        rep[f"type3_{stratum}"] = compare(3, "native", ["retry", "repair", "repair_fixed",
                                                        "stop", *PLACEBOS])
        decs = saved
    c_rows = [r for r in revs if r["arm"] == "C"]
    tok = [sum(r["fork_tokens"]) for r in c_rows]
    surv = [r for r in c_rows if r.get("survives")]
    rep["type2"] = {"forks": len(c_rows), "surviving_forks": len(surv),
                    "fork_tokens": sum(tok),
                    "share_not_surviving": round(sum(t for t, r in zip(tok, c_rows,
                                                                       strict=True)
                                                     if not r.get("survives")) / sum(tok), 4)
                    if sum(tok) else None,
                    "kinds": {k: sum(1 for r in c_rows if r["kind"] == k)
                              for k in ("empty", "duplicate", "proposals")},
                    "B_forks": sum(1 for r in revs if r["arm"] == "B"),
                    "underpowered": len(surv) < 5}
    rep["null_ladder"] = [x["rung"] for x in null_ladder()]
    return rep


__all__ = [
    "BLOCKS",
    "PROBE_SIZES",
    "PROBE_SMOKE",
    "ForcedCustomer",
    "ProbeDryRunAgent",
    "ProbeGate",
    "ProbeHooks",
    "ProbeRun",
    "ProbeSizes",
    "PromptedChooser",
    "analyze",
    "build_plan",
    "export_rows",
    "gpu_estimate",
    "items",
    "load_plan",
    "review_rows",
]
