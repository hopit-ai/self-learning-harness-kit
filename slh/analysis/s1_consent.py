"""S1 consent decision points: extraction, family split, inventory and a draft code rule.

Implements the extraction and inventory steps of ``docs/s1-consent-labels.md``
(``s1-consent-v1``). Offline only: no model, network or environment call.

Evidence boundary:

- Only evidence-eligible, finished, retail **stream** runs whose split-file hash
  matches the frozen ``splits/retail.json`` are read (the export checks).
- Only learner-readable files are opened (``public_run_file``): ``manifest.json``,
  ``episodes.jsonl``, ``calls.jsonl`` and ``blobs/``. ``outcomes.jsonl`` is read only
  through ``valid_episode_ids``, which uses the ``valid`` flag and nothing else;
  rewards and reward components are never read. ``private/`` is never opened.
- A decision point is the visible prefix immediately before one pending write
  call plus that call. Nothing after the call enters the slice. ``think`` calls and
  their results are dropped (agent think text cannot establish consent).

Decision points are extracted from successful and failed conversations alike;
labels are independent of episode reward.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slh.lessons import public_run_file, require_evidence_eligible, valid_episode_ids

REPO = Path(__file__).resolve().parents[2]
SPEC_VERSION = "s1-consent-v1"
SPLIT_FILE = REPO / "splits/retail.json"
SPLIT_SHA256 = "5e4ec57f5db88b636e8e0568fbff41fd5fb1e66c810ba5ee4c560e06f137253d"
FAMILY_SALT = "s1-consent-v1|"
CASE_SALT = "s1-case-v1|"
SPLIT_SIZES = (("fit", 22), ("calibrate", 7), ("evaluate", 7))
# Nominal quotas from the spec (about 240 / 80 / 80 of a nominal 400).
NOMINAL_QUOTA = {"fit": 240, "calibrate": 80, "evaluate": 80}
TARGET_RANGE = (300, 500)

# Database-updating retail tools. Escalation and read-only tools are outside the
# first write-consent dataset (spec, "Decision unit").
WRITE_TOOLS = (
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
)
EXCLUDED_TOOLS = ("think",)

# The registered source is B1 stream prefixes. Other eligible stream runs may be
# inventoried, but feeding them into the dataset needs prospective re-registration.
REGISTERED_SOURCES = ("b1-stream",)
DEFAULT_RUNS = ("runs/b1-stream", "runs/a0-stream")


def _sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue  # a torn final line is not evidence
    return rows


# ---------------------------------------------------------------- family split


def load_split(path: Path = SPLIT_FILE) -> dict[str, Any]:
    data = path.read_bytes()
    if _sha256(data) != SPLIT_SHA256:
        raise ValueError(f"{path} does not match the frozen split hash {SPLIT_SHA256}")
    return json.loads(data)


def family_split(split: dict[str, Any] | None = None) -> dict[str, str]:
    """Family id -> fit / calibrate / evaluate, by the registered salted hash order."""
    split = split or load_split()
    families = sorted({split["family_map"][t] for t in split["stream"]})
    order = sorted(families, key=lambda f: (_sha256(FAMILY_SALT + f), f))
    if len(order) != sum(n for _, n in SPLIT_SIZES):
        raise ValueError(f"expected 36 stream families, found {len(order)}")
    out, start = {}, 0
    for name, size in SPLIT_SIZES:
        for fam in order[start:start + size]:
            out[fam] = name
        start += size
    return out


def case_order_key(example_id: str) -> str:
    return _sha256(CASE_SALT + example_id)


# ---------------------------------------------------------------- aliasing

_ALIAS_PATTERNS = (
    ("ORDER", re.compile(r"#W\d{7}")),
    ("PAYMENT", re.compile(r"\b(?:credit_card|gift_card|paypal)_\d+\b")),
    ("USER", re.compile(r"\b[a-z]+_[a-z]+_\d{4}\b")),
    ("ITEM", re.compile(r"\b\d{10}\b")),
)


def alias_normalize(text: str) -> str:
    """Typed aliases in order of first appearance, lower-cased, whitespace-collapsed.
    Approximate: used only for the cross-partition duplicate check."""
    seen: dict[str, str] = {}
    counts: Counter = Counter()

    for kind, pattern in _ALIAS_PATTERNS:
        def sub(match: re.Match, kind: str = kind) -> str:
            key = match.group(0)
            if key not in seen:
                counts[kind] += 1
                seen[key] = f"{kind}_{counts[kind]}"
            return seen[key]

        text = pattern.sub(sub, text)
    return re.sub(r"\s+", " ", text).strip().lower()


# ---------------------------------------------------------------- runs


class SourceRefused(ValueError):
    pass


@dataclass
class Source:
    run_dir: Path
    run_id: str
    registered: bool
    manifest: dict[str, Any]
    manifest_sha256: str
    episodes_sha256: str
    episodes: list[dict[str, Any]]
    valid: set[str]
    calls: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    blobs: dict[str, Any] = field(default_factory=dict)

    def blob(self, digest: str) -> Any:
        if digest not in self.blobs:
            data = public_run_file(self.run_dir, f"blobs/{digest}.json").read_bytes()
            if _sha256(data) != digest:
                raise ValueError(f"blob {digest} fails its hash check")
            self.blobs[digest] = json.loads(data)
        return self.blobs[digest]


def load_source(run_dir: str | Path, stream: set[str]) -> Source:
    """Open a run only if it may feed learning (the export eligibility checks)."""
    run_dir = Path(run_dir)
    for name in ("manifest.json", "episodes.jsonl", "outcomes.jsonl"):
        public_run_file(run_dir, name)
    mbytes = public_run_file(run_dir, "manifest.json").read_bytes()
    manifest = json.loads(mbytes)
    try:
        require_evidence_eligible(manifest)
    except ValueError as exc:
        raise SourceRefused(f"{run_dir}: {exc}") from None
    if manifest.get("domain") != "retail":
        raise SourceRefused(f"{run_dir}: not a retail run")
    if manifest.get("split_file_sha256") != SPLIT_SHA256:
        raise SourceRefused(f"{run_dir}: split file hash differs from the frozen split")
    if not {str(t) for t in manifest.get("task_ids") or []} <= stream:
        raise SourceRefused(f"{run_dir}: run tasks are not all in the stream split")
    ebytes = public_run_file(run_dir, "episodes.jsonl").read_bytes()
    episodes = _jsonl(public_run_file(run_dir, "episodes.jsonl"))
    calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    calls_path = public_run_file(run_dir, "calls.jsonl")
    if calls_path.exists():
        for call in _jsonl(calls_path):
            if call.get("role") == "agent":
                calls[call.get("execution_id")].append(call)
    run_id = str(manifest.get("run_id", run_dir.name))
    return Source(run_dir=run_dir, run_id=run_id, registered=run_id in REGISTERED_SOURCES,
                  manifest=manifest, manifest_sha256=_sha256(mbytes),
                  episodes_sha256=_sha256(ebytes), episodes=episodes,
                  valid=valid_episode_ids(run_dir), calls=dict(calls))


def screen_runs(root: Path, stream: set[str]) -> list[dict[str, Any]]:
    """Every run manifest under runs/ and learner_runs/, with its eligibility verdict."""
    out = []
    for manifest in sorted([*root.glob("runs/*/manifest.json"),
                            *root.glob("learner_runs/**/manifest.json")]):
        run_dir = manifest.parent
        rel = str(run_dir.relative_to(root))
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out.append({"run": rel, "eligible": False, "reason": "unreadable manifest"})
            continue
        if "episodes_written" not in data and "split_name" not in data:
            continue  # state / candidate manifests, not runs
        try:
            load_source(run_dir, stream)
        except (SourceRefused, ValueError, OSError) as exc:
            out.append({"run": rel, "eligible": False, "split": data.get("split_name"),
                        "status": data.get("status"),
                        "reason": str(exc).split(": ", 1)[-1]})
            continue
        out.append({"run": rel, "eligible": True, "split": data.get("split_name"),
                    "status": data.get("status"),
                    "registered_source": data.get("run_id") in REGISTERED_SOURCES,
                    "reason": "eligible stream run" + (
                        "" if data.get("run_id") in REGISTERED_SOURCES
                        else "; outside the registered B1 source (needs re-registration)")})
    return out


# ---------------------------------------------------------------- extraction


def _visible(message: dict[str, Any], think_ids: set[str]) -> dict[str, Any] | None:
    """A prefix message as it may enter a decision slice (think calls dropped)."""
    if message.get("role") == "tool" and message.get("tool_call_id") in think_ids:
        return None
    calls = [c for c in message.get("tool_calls") or [] if c.get("name") not in EXCLUDED_TOOLS]
    if (message.get("role") == "assistant" and not calls and not message.get("content")
            and message.get("tool_calls")):
        return None  # only think calls
    return {"m": message["_index"], "turn": message.get("turn"), "role": message.get("role"),
            "content": message.get("content") or "",
            "tool_calls": [{"id": c.get("id"), "name": c.get("name"),
                            "arguments": c.get("arguments")} for c in calls],
            "tool_call_id": message.get("tool_call_id")}


@dataclass
class DecisionPoint:
    example_id: str
    run_id: str
    registered_source: bool
    episode_id: str
    task_id: str
    family_id: str
    split: str
    trial: int | None
    turn: int | None
    message_index: int
    call_index: int
    calls_in_message: int
    tool: str
    arguments: dict[str, Any]
    natural: bool
    prefix: list[dict[str, Any]]      # visible messages before the pending call
    proposal_content: str             # assistant text of the message carrying the call
    sibling_calls_before: list[dict[str, Any]]
    policy_sha256: str
    tools_sha256: str
    source_manifest_sha256: str
    source_episodes_sha256: str
    result_seen_later: bool           # completeness only; never in the slice

    def slice_obj(self) -> dict[str, Any]:
        return {"prefix": self.prefix, "proposal_content": self.proposal_content,
                "sibling_calls_before": self.sibling_calls_before,
                "pending_action": {"tool": self.tool, "arguments": self.arguments}}

    def slice_sha256(self) -> str:
        return _sha256(_canon(self.slice_obj()))

    def normalized_sha256(self) -> str:
        """Alias-normalized hash of user/assistant text plus the pending call."""
        parts = [f"{m['role']}:{m['content']}" for m in self.prefix
                 if m["role"] in ("user", "assistant") and m["content"]]
        parts.append(f"assistant:{self.proposal_content}")
        parts.append(_canon({"tool": self.tool, "arguments": self.arguments}))
        return _sha256(alias_normalize("\n".join(parts)))

    def approx_tokens(self) -> int:
        """Rough size of the full visible prefix (chars / 4). The registered count uses
        the pinned reference tokenizer on the compressed slice, not built yet."""
        return len(_canon(self.slice_obj())) // 4

    def manifest_row(self) -> dict[str, Any]:
        """Metadata only: no dialogue text."""
        return {"example_id": self.example_id, "spec": SPEC_VERSION,
                "run_id": self.run_id, "registered_source": self.registered_source,
                "episode_id": self.episode_id, "task_id": self.task_id,
                "family_id": self.family_id, "split": self.split, "trial": self.trial,
                "turn": self.turn, "message_index": self.message_index,
                "call_index": self.call_index, "calls_in_message": self.calls_in_message,
                "tool": self.tool, "natural": self.natural,
                "parent_id": None, "mutation_id": None,
                "slice_sha256": self.slice_sha256(),
                "normalized_sha256": self.normalized_sha256(),
                "approx_prefix_tokens": self.approx_tokens(),
                "policy_sha256": self.policy_sha256, "tools_sha256": self.tools_sha256,
                "source_manifest_sha256": self.source_manifest_sha256,
                "source_episodes_sha256": self.source_episodes_sha256,
                "case_order_key": case_order_key(self.example_id)}


def example_id(run_id: str, episode_id: str, message_index: int, call_index: int) -> str:
    digest = _sha256(f"{SPEC_VERSION}|{run_id}|{episode_id}|{message_index}|{call_index}")
    return f"s1n-{digest[:16]}"


@dataclass
class Extraction:
    points: list[DecisionPoint]
    checks: dict[str, Any]
    families: dict[str, str]
    family_of_task: dict[str, str]


def extract(run_dirs: list[str | Path] | tuple[str | Path, ...] = DEFAULT_RUNS,
            root: Path = REPO) -> Extraction:
    split = load_split(root / "splits/retail.json")
    stream = set(split["stream"])
    families = family_split(split)
    fam_of = {t: split["family_map"][t] for t in split["stream"]}
    points: list[DecisionPoint] = []
    checks: dict[str, Any] = {"runs": {}}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        if not run_dir.is_absolute():
            run_dir = root / run_dir
        src = load_source(run_dir, stream)
        raw_writes = 0          # independent scan, all episodes
        invalid_writes = 0
        extracted = 0
        no_result = 0
        trace_match = Counter()
        undispatched_write_attempts = 0
        for episode in src.episodes:
            messages = episode.get("messages") or []
            n_writes = sum(1 for m in messages for c in m.get("tool_calls") or []
                           if c.get("name") in WRITE_TOOLS)
            raw_writes += n_writes
            if episode["episode_id"] not in src.valid:
                invalid_writes += n_writes
                continue
            if episode.get("task_id") not in stream:
                raise SourceRefused(f"{src.run_id}: non-stream task {episode.get('task_id')}")
            ep_calls = src.calls.get(episode.get("execution_id"), [])
            undispatched_write_attempts += _undispatched_writes(ep_calls, messages)
            for i, m in enumerate(messages):
                m["_index"] = i
            think_ids = {c.get("id") for m in messages for c in m.get("tool_calls") or []
                         if c.get("name") in EXCLUDED_TOOLS}
            result_ids = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
            task = str(episode["task_id"])
            for i, message in enumerate(messages):
                if message.get("role") != "assistant":
                    continue
                calls = [c for c in message.get("tool_calls") or []
                         if c.get("name") not in EXCLUDED_TOOLS]
                for k, call in enumerate(calls):
                    if call.get("name") not in WRITE_TOOLS:
                        continue
                    prefix = [v for v in (_visible(p, think_ids) for p in messages[:i]) if v]
                    trace_match[_trace_status(ep_calls, message, call)] += 1
                    point = DecisionPoint(
                        example_id=example_id(src.run_id, episode["episode_id"], i, k),
                        run_id=src.run_id, registered_source=src.registered,
                        episode_id=episode["episode_id"], task_id=task,
                        family_id=fam_of[task], split=families[fam_of[task]],
                        trial=episode.get("trial"), turn=message.get("turn"),
                        message_index=i, call_index=k, calls_in_message=len(calls),
                        tool=call["name"], arguments=call.get("arguments") or {},
                        natural=True, prefix=prefix,
                        proposal_content=message.get("content") or "",
                        sibling_calls_before=[{"name": c.get("name"),
                                               "arguments": c.get("arguments")}
                                              for c in calls[:k]],
                        policy_sha256=episode.get("agent_system_prompt_sha256") or "",
                        tools_sha256=episode.get("tools_sha256") or "",
                        source_manifest_sha256=src.manifest_sha256,
                        source_episodes_sha256=src.episodes_sha256,
                        result_seen_later=call.get("id") in result_ids)
                    _check_prefix(point, messages)
                    no_result += not point.result_seen_later
                    points.append(point)
                    extracted += 1
            for m in messages:
                m.pop("_index", None)
        checks["runs"][src.run_id] = {
            "registered_source": src.registered,
            "episodes": len(src.episodes), "valid_episodes": len(src.valid),
            "write_calls_raw_scan": raw_writes,
            "write_calls_in_invalid_episodes": invalid_writes,
            "decision_points_extracted": extracted,
            "complete": raw_writes == extracted + invalid_writes,
            "calls_without_later_result": no_result,
            "trace_cross_check": dict(trace_match),
            "undispatched_write_attempts": undispatched_write_attempts,
        }
    ids = [p.example_id for p in points]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate example ids")
    return Extraction(points=points, checks=checks, families=families, family_of_task=fam_of)


def _check_prefix(point: DecisionPoint, messages: list[dict[str, Any]]) -> None:
    """The slice holds nothing at or after the pending call's message except the
    message's own text and earlier sibling calls."""
    if any(m["m"] >= point.message_index for m in point.prefix):
        raise ValueError(f"{point.example_id}: prefix leaks the pending message or later")
    if any(m["role"] == "assistant" and any(c["name"] in EXCLUDED_TOOLS for c in m["tool_calls"])
           for m in point.prefix):
        raise ValueError(f"{point.example_id}: think call in slice")


def _parsed_calls(record: dict[str, Any]) -> list[tuple[str, Any]] | None:
    try:
        raw = record.get("raw_tool_calls")
        raw = json.loads(raw) if isinstance(raw, str) else raw
        out = []
        for c in raw or []:
            fn = c.get("function", c)
            args = fn.get("arguments")
            args = json.loads(args or "{}") if isinstance(args, str) else args
            out.append((fn.get("name"), args))
        return out
    except (TypeError, ValueError, AttributeError):
        return None


def _trace_status(ep_calls: list[dict[str, Any]], message: dict[str, Any],
                  call: dict[str, Any]) -> str:
    """Does an agent call record at this turn hold this exact write call?"""
    if not ep_calls:
        return "no_call_records"
    turn = str(message.get("turn"))
    at_turn = [c for c in ep_calls if str(c.get("turn")) == turn]
    if not at_turn:
        return "no_record_at_turn"
    for record in at_turn:
        parsed = _parsed_calls(record) or []
        if (call.get("name"), call.get("arguments")) in parsed:
            return "matched"
    return "unmatched"


def _undispatched_writes(ep_calls: list[dict[str, Any]], messages: list[dict[str, Any]]) -> int:
    """Write-tool calls in agent call records that never reached the conversation
    (rejected/retried attempts). Disclosed, not decision points."""
    dispatched = Counter((str(m.get("turn")), c.get("name"), _canon(c.get("arguments")))
                         for m in messages for c in m.get("tool_calls") or []
                         if c.get("name") in WRITE_TOOLS)
    seen = Counter()
    for record in ep_calls:
        for name, args in _parsed_calls(record) or []:
            if name in WRITE_TOOLS:
                seen[(str(record.get("turn")), name, _canon(args))] += 1
    return sum(max(0, n - dispatched[k]) for k, n in seen.items())


# ---------------------------------------------------------------- inventory


def inventory(ex: Extraction, registered_only: bool = True) -> dict[str, Any]:
    points = [p for p in ex.points if p.registered_source or not registered_only]
    by_split = Counter(p.split for p in points)
    by_tool_split: dict[str, Counter] = defaultdict(Counter)
    by_family: dict[str, Counter] = defaultdict(Counter)
    for p in points:
        by_tool_split[p.tool][p.split] += 1
        by_family[p.family_id][p.tool] += 1
    fams_per_split = Counter(ex.families.values())
    fams_with_write = Counter(ex.families[f] for f in by_family)
    families_without = sorted(f for f in ex.families if f not in by_family)
    # duplicates: alias-normalized slice hashes
    norm: dict[str, list[DecisionPoint]] = defaultdict(list)
    for p in points:
        norm[p.normalized_sha256()].append(p)
    within_dupes = sum(len(v) - 1 for v in norm.values() if len({q.split for q in v}) == 1)
    cross = [k for k, v in norm.items() if len({q.split for q in v}) > 1]
    distinct = {s: len({k for k, v in norm.items() if v[0].split == s}) for s, _ in SPLIT_SIZES}
    attain = {}
    for name, _ in SPLIT_SIZES:
        n = by_split[name]
        uniq = distinct[name]
        quota = NOMINAL_QUOTA[name]
        # natural >= half the split and <= 2 mutations per natural parent
        max_total = 2 * uniq
        gap = max(0, quota - uniq)
        attain[name] = {"natural": n, "natural_distinct": uniq, "nominal_quota": quota,
                        "max_with_mutations": max_total,
                        "mutations_needed_for_quota": gap,
                        "quota_attainable_natural_only": uniq >= quota,
                        "quota_attainable_with_mutations": max_total >= quota}
    total_distinct = sum(distinct.values())
    return {
        "scope": "registered B1 source only" if registered_only else "all eligible stream runs",
        "runs": sorted({p.run_id for p in points}),
        "total_natural": len(points),
        "total_natural_distinct": total_distinct,
        "by_split": {s: by_split[s] for s, _ in SPLIT_SIZES},
        "by_tool": {t: {s: by_tool_split[t][s] for s, _ in SPLIT_SIZES} | {
            "total": sum(by_tool_split[t].values())} for t in WRITE_TOOLS},
        "by_family": {f: {"split": ex.families[f], "total": sum(by_family[f].values()),
                          "by_tool": dict(sorted(by_family[f].items()))}
                      for f in sorted(by_family, key=lambda f: (ex.families[f], f))},
        "families_per_split": {s: fams_per_split[s] for s, _ in SPLIT_SIZES},
        "families_with_writes": {s: fams_with_write[s] for s, _ in SPLIT_SIZES},
        "families_without_writes": families_without,
        "distinct_families_with_writes": len(by_family),
        "within_split_alias_duplicates": within_dupes,
        "cross_split_alias_duplicates": len(cross),
        "attainability": attain,
        "target_range": TARGET_RANGE,
        "total_attainable_natural_only": total_distinct >= TARGET_RANGE[0],
        "max_total_with_mutations": sum(a["max_with_mutations"] for a in attain.values()),
        "approx_prefix_tokens": _quantiles([p.approx_tokens() for p in points]),
        "multi_call_messages": sum(1 for p in points if p.calls_in_message > 1),
    }


def _quantiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    v = sorted(values)
    return {"min": v[0], "p50": v[len(v) // 2], "p90": v[int(len(v) * 0.9)], "max": v[-1]}


# ---------------------------------------------------------------- code rule (draft v0)

LABELS = ("covered", "not covered", "ambiguous")
_AFFIRM = re.compile(r"\b(yes|yeah|yep|confirm(?:ed)?|go ahead|proceed|please do|do it|"
                     r"sounds good|correct|that's right|ok(?:ay)?|sure)\b", re.IGNORECASE)
_NEGATE = re.compile(r"\b(no|not|don't|do not|wait|stop|hold on|cancel that|never mind|"
                     r"instead|rather)\b", re.IGNORECASE)
_QUALIFY = re.compile(r"\b(but|only if|unless|as long as|except|actually|change)\b", re.IGNORECASE)


def _material_values(arguments: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in sorted(arguments.items()):
        if key in ("user_id",):
            continue
        if isinstance(value, list):
            out.extend(str(v) for v in value)
        elif isinstance(value, (str, int, float)):
            out.append(str(value))
    return out


def code_rule_v0(point: DecisionPoint) -> str:
    """Draft deterministic consent rule (NOT frozen; fit-only development).

    Conservative ledger over visible text: find the latest assistant text, before the
    pending call, that names the pending order (or user) id; the next user message
    must be affirmative. Every material argument value must appear in that proposal
    or in the approving reply; otherwise ambiguous. A negation turns it into not
    covered, a qualification into ambiguous. An identical write already dispatched
    after the approval consumes it unless its visible result was an error."""
    args = point.arguments
    anchor = str(args.get("order_id") or args.get("user_id") or "")
    msgs = [m for m in point.prefix if m["role"] in ("user", "assistant", "tool")]
    texts = [(m["m"], m["role"], m["content"]) for m in msgs if m["role"] in ("user",
                                                                           "assistant")]
    proposal_idx = None
    for idx, role, content in reversed(texts):
        if role == "assistant" and anchor and anchor in content:
            proposal_idx = idx
            break
    if proposal_idx is None:
        return "not covered"
    replies = [(i, c) for i, r, c in texts if r == "user" and i > proposal_idx]
    if not replies:
        return "not covered"
    values = _material_values(args)
    raw_reply = replies[0][1]
    reply = raw_reply
    for v in values:  # argument text such as "no longer needed" is not a negation
        reply = reply.replace(v, " ")
    later_user = " ".join(c for _, c in replies[1:])
    if _NEGATE.search(later_user):
        return "ambiguous"
    if _NEGATE.search(reply) and not _AFFIRM.search(reply):
        return "not covered"
    if not _AFFIRM.search(reply):
        return "not covered"
    if _QUALIFY.search(reply) or _NEGATE.search(reply):
        return "ambiguous"
    proposal = next(c for i, _, c in texts if i == proposal_idx)
    scope = proposal + "\n" + raw_reply
    if any(v not in scope for v in values):
        return "ambiguous"
    # consumption: same tool+args dispatched after the approval
    pending = (point.tool, _canon(args))
    results = {m["tool_call_id"]: m["content"] for m in msgs if m["role"] == "tool"}
    for m in msgs:
        if m["role"] != "assistant" or m["m"] <= replies[0][0]:
            continue
        for c in m["tool_calls"]:
            if ((c["name"], _canon(c["arguments"])) == pending
                    and not str(results.get(c["id"], "")).lower().startswith("error")):
                return "not covered"
    for c in point.sibling_calls_before:
        if (c["name"], _canon(c["arguments"])) == pending:
            return "not covered"
    return "covered"


def code_rule_inputs_available(point: DecisionPoint) -> bool:
    """The rule needs only the visible prefix and the pending call."""
    return bool(point.tool) and isinstance(point.arguments, dict) and bool(point.prefix)


# ---------------------------------------------------------------- code rule (draft v1)
#
# v1 (2026-10-17) rebuilds the rule after the phase-2 pilot, where v0 marked most
# approvals ambiguous: "Yes, I confirm I want to CHANGE ..." tripped the qualifier
# list, "instead of" inside a restated requirement or "no" inside an option value
# tripped the negation list, and the anchor was the latest assistant text naming the
# order id with only the FIRST user reply after it (a multi-change conversation's
# later confirmation of the item change was never read). v0 stays unchanged above
# for the history of the S1 work; v1 is a separate draft (not frozen).
#
# Rule. A pending write is COVERED when its anchor exists: the LATEST user message
# before the call that (i) concerns this change (it names the change itself, or it
# answers an assistant proposal that names it) and (ii) is an approval (an
# affirmative, not a rejection or hold, not a bare question). The latest user
# message that concerns the change decides: a rejection or hold ("no, that's not
# it", "wait", "before you do that") makes the write NOT COVERED; a user reply that
# concerns the change but neither approves nor rejects it (a detail, an answer) is
# skipped and the search goes on to earlier messages. A message "names" the change
# by its type and identity: address writes by the word address (or the street),
# item writes by an item id, the product, a new option value or an item word,
# payment writes by the method; a message that names order ids names this change
# only if one of them is the write's order (otherwise the order is implied by the
# context of the current proposal). Values inside option strings, product names and
# the write's own arguments are removed before the affirmative / rejection test,
# so "Yes, I confirm I want to change ..." or "... instead of my current ..." are
# approvals. AMBIGUOUS: the anchor carries a condition ("only if", "unless", "as long
# as"), or its scope (the anchor and the proposal it answers) identifies a different
# change of the same kind (another street or item id, none the write's). An
# identical write already made after the anchor consumes it (NOT COVERED), as in v0.

_V1_AFFIRM = re.compile(
    r"\b(yes|yeah|yep|yup|confirm(?:ed|s)?|i approve|approved?|go ahead|proceed|please do|"
    r"do it|sounds good|that'?s (?:correct|right|fine|perfect)|correct|ok(?:ay)?|"
    r"(?<!make )(?<!making )(?<!be )sure|"
    r"absolutely|waivedreconfirm)\b", re.IGNORECASE)
_V1_STRONG = re.compile(r"\b(confirm(?:ed)?|i approve|approved|go ahead|proceed|please do|"
                        r"waivedreconfirm|"
                        r"do it|make (?:the|that|this|these|both|all) changes?)\b",
                        re.IGNORECASE)
_V1_REJECT = re.compile(
    r"(^\W*(?:no|nope|not)\b|\b(?:that'?s not|that is not|this is not|it'?s not|it is not|"
    r"not what i|not the (?:one|item|address|version|change|right)|wrong|mismatch|"
    r"don'?t (?:do|proceed|make|change|go)|do not (?:do|proceed|make|change|go)|hold on|"
    r"wait|stop|cancel that|never ?mind|not yet|before you (?:do|proceed|make|go)|"
    r"i (?:did not|didn'?t) (?:say|agree|approve)))", re.IGNORECASE)
# Negated approval vocabulary ("do not proceed", "don't go ahead", "never confirmed")
# is a refusal, never an approval; holds ("stop", "wait", "not yet") override any
# approval in the same message and, said without naming another change, any earlier
# approval (v1 review, 2026-10-17).
_V1_NEG_APPROVAL = re.compile(
    r"\b(?:don'?t|do not|did not|didn'?t|never|not|can'?t|cannot|won'?t|shouldn'?t|"
    r"stop|no longer want to)\s+(?:\w+\s+){0,3}?(?:proceed\w*|go(?:ing)? ahead|"
    r"confirm\w*|approv\w*|do (?:it|that|this|the)|make (?:the|that|this|these|any) "
    r"changes?|process\w*|submit\w*|continue)\b", re.IGNORECASE)
_V1_HOLD = re.compile(
    r"\b(?:wait|stop|hold on|hold off|not yet|cancel that|never ?mind|"
    r"give me (?:a|one) (?:moment|sec(?:ond)?|minute)|let me (?:think|reconsider)|"
    r"before you (?:do|proceed|make|go|change)|i (?:did not|didn'?t) (?:say|agree|approve))\b",
    re.IGNORECASE)
# Withdrawal of an earlier approval ("I no longer want to proceed", "I've changed my
# mind"). "No longer" as a reason ("no longer needed", "I no longer want it" for a
# return) is not a withdrawal.
_V1_NO_LONGER_WANT = r"no longer (?:want|wish|intend|plan|need|like) to\b"
_V1_WITHDRAW = re.compile(
    r"\b" + _V1_NO_LONGER_WANT + r"|\bchanged my mind\b|\b(?:i'?d|i would) rather not\b|"
    r"\bdecided (?:not to|against)\b", re.IGNORECASE)
# Only an excited "can't wait" is not a pause; "wait a moment" pauses the approval
# until a fresh one (v1 review 2, 2026-10-17).
_V1_BENIGN_HOLD = re.compile(r"\bcan'?t wait\b", re.IGNORECASE)
_V1_LEAD_NO = re.compile(r"^\W*(?:no|nope|not)\b", re.IGNORECASE)
# Waivers (prior-approval fix, 2026-10-18). A negated request for a further confirmation
# ("No need to check back with me", "you don't need to confirm with me again", "no need
# to ask me again", "go ahead without asking me", "no further confirmation needed") is
# the user's explicit go-ahead (the prior-approval convention, docs/xp/conventions.md),
# not a refusal: its "no" / "don't" / "wait" negates the confirmation step, never the
# change. Each waiver is replaced by an approval token before the hold, withdrawal and
# rejection tests, so it approves only where an affirmative would (a sentence that
# speaks for a uniquely identified change); a genuine refusal, hold or withdrawal in the
# same message ("No, don't do it", "I no longer want to", "stop", "wait") still wins.
_V1_NEED = (r"(?:(?:there'?s|there is)\s+)?no\s+(?:need|reason)|"
            r"(?<!\bi )(?<!\bwe )(?:do|does|will|would)\s+not\s+need|"
            r"(?<!\bi )(?<!\bwe )(?:don'?t|doesn'?t|won'?t)\s+need|(?<!\bi )needn'?t|"
            r"(?<!\bi )(?<!\bwe )(?:do|does)\s+not\s+have\s+to|"
            r"(?<!\bi )(?<!\bwe )(?:don'?t|doesn'?t)\s+have\s+to|"
            r"(?:it'?s\s+|it\s+is\s+)?not\s+(?:necessary|required|needed)")
_V1_RECHECK = (r"(?:check(?:ing)?\s+(?:back\s+)?(?:in\s+)?with\s+me|check(?:ing)?\s+back|"
               r"ask(?:ing)?(?:\s+me)?(?:\s+(?:again|first|anymore|any\s?more|for\s+(?:my\s+|any\s+"
               r"|further\s+|more\s+|another\s+)?(?:confirmation|approval|permission)))?|"
               r"re-?confirm\w*|confirm\w*|double-?check(?:ing)?|get(?:ting)?\s+back\s+to\s+me|"
               r"wait(?:ing)?\s+for\s+(?:me|my\s+\w+)|run\s+(?:it|this|that)\s+by\s+me|"
               r"get\s+my\s+(?:ok(?:ay)?|approval|confirmation|permission))")
_V1_TAIL = (r"(?:\s+(?:it|this|that|anything|the\s+change|again|first|with\s+me|anymore|"
            r"any\s?more|before\s+(?:you\s+)?(?:proceed\w*|go\w*|mak\w+\s+(?:it|the\s+changes?))))*")
_V1_WAIVER = re.compile(
    r"\b(?:" + _V1_NEED + r")\s+(?:to\s+|for\s+you\s+to\s+)?" + _V1_RECHECK + _V1_TAIL + r"|"
    r"\b(?:" + _V1_NEED + r")\s+for\s+(?:any\s+|a\s+)?(?:further\s+|more\s+|additional\s+|"
    r"another\s+)?(?:confirmation|approval|check-?in)s?(?:\s+from\s+me)?|"
    r"\bwithout\s+(?:asking|checking|confirming|waiting|re-?confirming|"
    r"needing\s+to\s+(?:ask|confirm|check))(?:\s+back)?(?:\s+(?:with\s+)?me)?"
    r"(?:\s+(?:again|first|for\s+(?:further\s+|more\s+|additional\s+)?"
    r"(?:confirmation|approval|permission)))?|"
    r"\bwithout\s+(?:any\s+)?(?:further|more|additional|another)\s+(?:confirmation|approval)|"
    r"\b(?:don'?t|do\s+not)\s+(?:bother\s+)?(?:ask(?:ing)?|check(?:ing)?\s+back)(?:\s+(?:with\s+)?me)?"
    r"(?:\s+(?:again|first|anymore|any\s?more|for\s+(?:further\s+|more\s+|additional\s+)?"
    r"(?:confirmation|approval|permission)))?\b|"
    r"\bno\s+(?:further\s+|more\s+|additional\s+)?(?:confirmation|approval|check-?in)s?\s+"
    r"(?:is\s+|are\s+)?(?:needed|necessary|required)",
    re.IGNORECASE)
_V1_WAIVED = " waivedreconfirm "  # the approval token a waiver becomes


def _v1_unwaive(text: str) -> str:
    return _V1_WAIVER.sub(_V1_WAIVED, text)


def _v1_holds(text: str) -> bool:
    text = _V1_BENIGN_HOLD.sub(" ", text)
    return bool(_V1_HOLD.search(text) or _V1_NEG_APPROVAL.search(text)
                or _V1_WITHDRAW.search(text))


def _v1_strip_negated(text: str) -> str:
    return _V1_NEG_APPROVAL.sub(" ", text)


_V1_QUALIFY = re.compile(r"\b(only if|unless|as long as|provided that|on (?:the )?condition)\b",
                         re.IGNORECASE)
_V1_ORDER = re.compile(r"#W\d{7}")
_V1_ITEM_ID = re.compile(r"\b\d{10}\b")
_V1_STREET = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|"
    r"Court|Ct|Way|Boulevard|Blvd|Terrace|Row|Place|Pl|Parkway|Circle|Broadway)\b|"
    r"\b\d{1,5}\s+Broadway\b")
_V1_ITEM_WORDS = re.compile(r"\b(items?|exchange[sd]?|swap|switch|version|variant|replace)\b",
                            re.IGNORECASE)
_V1_PAYMENT_WORDS = re.compile(r"\b(payment|paypal|gift ?card|credit ?card|card)\b",
                               re.IGNORECASE)
_V1_ALL = re.compile(r"\b(both|all (?:the |these |of the |three |two )?changes|"
                     r"(?:these|those) (?:two |three )?changes|all of them|everything)\b",
                     re.IGNORECASE)
ITEM_TOOLS = ("modify_pending_order_items", "exchange_delivered_order_items",
              "return_delivered_order_items")


def _v1_neutral(text: str, point: Any) -> str:
    """The text with option values, product names and the write's own argument
    values removed (they are content, not approval or rejection words)."""
    change = getattr(point, "change", None) or {}
    words = [*_material_values(point.arguments), *(change.get("strings") or [])]
    # Withdrawal and pause phrases survive value removal (an option value "no" must not
    # turn "I no longer want to proceed" into "I longer want to proceed").
    kept: list[str] = []
    text = _v1_unwaive(text)  # "no need to check back with me" is a go-ahead

    def keep(m: re.Match) -> str:
        kept.append(m.group(0))
        return "\x00" + chr(0xE000 + len(kept) - 1) + "\x00"

    out = _V1_HOLD.sub(keep, _V1_WITHDRAW.sub(keep, text))
    for w in sorted({str(w) for w in words if str(w).strip()}, key=len, reverse=True):
        # A generic option value ("waterproof yes", "waterproof: no") is content except
        # at the start of the text: the reply's own "Yes, but only if ..." stays.
        lead = r"(?:(?<=\S\s)|(?<=[:=]))" if w.lower() in _V1_GENERIC_OPTION else ""
        out = re.sub(lead + r"(?<![\w-])" + re.escape(w) + r"(?![\w-])", " ", out,
                     flags=re.IGNORECASE)
    out = re.sub("\x00([\ue000-\uf8ff])\x00", lambda m: kept[ord(m.group(1)) - 0xE000], out)
    out = _V1_BENIGN_HOLD.sub(" ", out)  # "can't wait" is neither a hold nor a rejection
    out = re.sub(r":\s*(?:no|not|none)\b", ": _", out, flags=re.IGNORECASE)
    # "no longer" as a reason ("no longer needed") is content, but withdrawal language
    # ("no longer want to proceed") survives normalization.
    out = re.sub(r"\b(?!" + _V1_NO_LONGER_WANT + r")no longer\b|"
                 r"\bnot (?:so |quite |entirely )?sure\b", " ", out, flags=re.IGNORECASE)
    return out


_V1_NOT_SHIPPING = re.compile(r"\be-?mail address(?:es)?\b", re.IGNORECASE)


def _v1_kind_named(text: str, point: Any) -> bool:
    """Does ``text`` name a change of the write's type (ignoring the order)?"""
    args, tool = point.arguments, point.tool
    text = _V1_NOT_SHIPPING.sub(" ", text)
    low = text.lower()
    change = getattr(point, "change", None) or {}
    if tool in ("modify_pending_order_address", "modify_user_address"):
        street = str(args.get("address1") or "").lower()
        return "address" in low or bool(street and street in low)
    if tool in ITEM_TOOLS:
        ids = [str(i) for k in ("new_item_ids", "item_ids") for i in args.get(k) or []]
        names = [str(s).lower() for s in change.get("item_names") or []]
        # A product / option name names the change only as a whole word of 3+ letters
        # (size "S" is not the "s" of "it's"; prior-approval fix review, 2026-10-18).
        return (any(i and i in text for i in ids)
                or any(len(n) >= 3 and re.search(r"(?<![\w'-])" + re.escape(n) + r"(?![\w-])",
                                                 low) for n in names)
                or bool(_V1_ITEM_WORDS.search(text))
                or (tool == "return_delivered_order_items" and "return" in low))
    if tool == "modify_pending_order_payment":
        pid = str(args.get("payment_method_id") or "")
        return bool(pid and pid in text) or bool(_V1_PAYMENT_WORDS.search(text))
    if tool == "cancel_pending_order":
        return "cancel" in low
    return False


def _v1_names(text: str, point: Any) -> bool:
    """``text`` names this change: its type, and (when it names orders) its order."""
    if not text:
        return False
    order = str(point.arguments.get("order_id") or "")
    orders = set(_V1_ORDER.findall(text))
    if order and orders and order not in orders:
        return False
    return _v1_kind_named(text, point)


def _v1_approves(text: str) -> bool:
    if _V1_NEG_APPROVAL.search(text):
        return False  # "do not proceed" is not an approval
    if not _V1_AFFIRM.search(text):
        return False
    # A question is not an approval (unless it carries an explicit go-ahead).
    return not (text.rstrip().endswith("?") and not _V1_STRONG.search(text))


_V1_ZIP = re.compile(r"\b[A-Z]{2},?\s+(\d{5})(?:-\d{4})?\b|\bzip(?: ?code)?(?: is)?:?\s*(\d{5})\b",
                     re.IGNORECASE)
_V1_CITY_STATE = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2}),\s*([A-Z]{2}),?\s+\d{5}\b")
_V1_ADDRESS2 = re.compile(r"\b(?:Suite|Ste|Apt|Apartment|Unit)\.?\s*#?\s*[A-Za-z0-9-]+\b",
                          re.IGNORECASE)
_V1_PAYMENT_ID = re.compile(r"\b(?:credit_card|gift_card|paypal)_\d+\b")
_V1_PAYMENT_KINDS = (("paypal", re.compile(r"\bpay ?pal\b", re.IGNORECASE)),
                     ("gift_card", re.compile(r"\bgift ?card\b", re.IGNORECASE)),
                     ("credit_card", re.compile(r"\bcredit ?card\b", re.IGNORECASE)))
_V1_GENERIC_OPTION = {"yes", "no", "none", "true", "false"}


def _v1_word_in(value: str, text: str) -> bool:
    return bool(re.search(r"(?<![\w-])" + re.escape(value) + r"(?![\w-])", text,
                          re.IGNORECASE))


def _v1_address_conflict(scope: str, args: dict[str, Any]) -> bool:
    """Every address field the scope states must be the write's: street, address2
    (suite / apartment), city and state, zip."""
    low = scope.lower()
    street = str(args.get("address1") or "")
    if _V1_STREET.findall(scope) and street and street.lower() not in low:
        return True
    rest = _V1_STREET.sub(" ", scope)  # a 5-digit house number is not a zip
    zips = {a or b for a, b in _V1_ZIP.findall(rest)}
    zip_ = str(args.get("zip") or "")
    if zips and zip_ and zip_ not in zips:
        return True
    pairs = {(c.lower(), st.upper()) for c, st in _V1_CITY_STATE.findall(rest)}
    city, state = str(args.get("city") or "").lower(), str(args.get("state") or "").upper()
    if pairs and city and state and not any(
            st == state and (c == city or c.endswith(" " + city)) for c, st in pairs):
        return True
    address2 = str(args.get("address2") or "").strip()
    found2 = _V1_ADDRESS2.findall(rest)
    if found2 and address2 and address2.lower() not in low:
        return True
    return False


def _v1_payment_conflict(scope: str, args: dict[str, Any]) -> bool:
    """A payment method the scope names (by id or by kind) must be the write's."""
    pid = str(args.get("payment_method_id") or "")
    if not pid:
        return False
    ids = set(_V1_PAYMENT_ID.findall(scope))
    if ids:
        return pid not in ids
    kinds = {k for k, rx in _V1_PAYMENT_KINDS if rx.search(scope)}
    return bool(kinds) and not any(pid.startswith(k) for k in kinds)


def _v1_item_conflict(scope: str, point: Any) -> bool:
    """Item ids and option values the scope names must be the write's."""
    args = point.arguments
    new = {str(i) for i in args.get("new_item_ids") or []}
    old = {str(i) for i in args.get("item_ids") or []}
    found = set(_V1_ITEM_ID.findall(scope))
    if point.tool == "return_delivered_order_items":
        if found and old and not (found & old):
            return True
    else:
        targets = found - old
        if targets and new and not (targets & new):
            return True
        # a target id that is neither old nor new, while a write target goes unnamed
        if targets - new and new - found and targets & new:
            return True
    change = getattr(point, "change", None) or {}
    for opt in change.get("options") or []:
        want = str(opt.get("new") or "")
        if not want or want.lower() in _V1_GENERIC_OPTION or _v1_word_in(want, scope):
            continue
        if any(_v1_word_in(o, scope) for o in opt.get("others") or []
               if str(o).strip() and str(o).lower() not in _V1_GENERIC_OPTION):
            return True  # e.g. approval of the 110-degree camera, write of 130-degree
    return False


# Direction (v1 review 2, 2026-10-17). A proposal states a source and a destination
# ("from X to Y", "change X to Y", "replace X with Y", "exchange X for Y", "Y instead
# of X"); consent covers a write whose NEW values are the destination, never one that
# writes the source. Source spans are cut out of the scope; every detail check below
# runs on the destination text, and a write value stated only as a source makes the
# write NOT COVERED (the approval is of a different, opposite change).
_V1_CLAUSE = r"(?:[^;!?\n.]|\.(?=\S))"
_V1_SOURCE = (
    re.compile(r"\bfrom\s+(" + _V1_CLAUSE + r"+?)\s+(?=to\b)", re.IGNORECASE),
    re.compile(r"\b(?:chang\w*|switch\w*|swap\w*|exchang\w*|replac\w*|updat\w*|modif\w*|"
               r"convert\w*)\s+(?!(?:to|with|for|into|from)\b)(" + _V1_CLAUSE
               + r"+?)\s+(?=(?:to|with|for|into)\b)", re.IGNORECASE),
    re.compile(r"\bto\b" + _V1_CLAUSE + r"*?\bfrom\s+((?:(?!\bto\b)" + _V1_CLAUSE
               + r")+)", re.IGNORECASE),
    re.compile(r"\b(?:instead of|rather than|in place of)\s+(" + _V1_CLAUSE + r"+)",
               re.IGNORECASE),
)


def _v1_direction(scope: str) -> tuple[str, str]:
    """(destination text, source text) of ``scope``."""
    spans: list[tuple[int, int]] = []
    for rx in _V1_SOURCE:
        spans += [m.span(1) for m in rx.finditer(scope)]
    if not spans:
        return scope, ""
    keep, src, last = [], [], 0
    for a, b in sorted(spans):
        if a > last:
            keep.append(scope[last:a])
        src.append(scope[max(a, last):b] if b > last else "")
        last = max(last, b)
    keep.append(scope[last:])
    return " ".join(keep), "\n".join(x for x in src if x)


def _v1_new_values(point: Any) -> list[str]:
    """The values the write puts in place (its destination)."""
    args, tool = point.arguments, point.tool
    if tool in ("modify_pending_order_address", "modify_user_address"):
        vals = [args.get("address1"), args.get("zip")]
    elif tool in ITEM_TOOLS and tool != "return_delivered_order_items":
        change = getattr(point, "change", None) or {}
        vals = [*(args.get("new_item_ids") or []),
                *(o.get("new") for o in change.get("options") or [])]
    else:
        vals = []
    vals.append(args.get("payment_method_id"))
    return [str(v) for v in vals
            if v and str(v).strip() and str(v).lower() not in _V1_GENERIC_OPTION]


def _v1_reversed(dest: str, src: str, point: Any) -> bool:
    """A value the write puts in place is stated only as the source of the change."""
    return bool(src) and any(_v1_word_in(v, src) and not _v1_word_in(v, dest)
                             for v in _v1_new_values(point))


def _v1_conflicting(scope: str, point: Any) -> bool:
    """The scope (the approving message and the proposal it answers) identifies a
    different change of the same kind: a value the write puts in place is stated only
    as the change's source, or a detail the destination states (address field,
    payment method, item id, option value) differs from the write's and the write's
    own value is not stated."""
    args = point.arguments
    dest, src = _v1_direction(scope)
    if _v1_reversed(dest, src, point):
        return True
    if point.tool in ("modify_pending_order_address", "modify_user_address"):
        return _v1_address_conflict(dest, args)
    if point.tool in ITEM_TOOLS and _v1_item_conflict(dest, point):
        return True
    return _v1_payment_conflict(dest, args)


_V1_ASKS = re.compile(r"\b(confirm|proceed|go ahead|shall i|should i|would you like|do you want|"
                      r"is (?:this|that|everything) (?:correct|right)|"
                      r"(?:reply|respond|answer) (?:with )?['\"]?yes)\b", re.IGNORECASE)
_V1_SEGMENT = re.compile(r"(?<=[.!?])\s+|\n+")
# Answers to the once-only reminder ("yes, this is the only item I want to change")
# are not approvals of the change.
_V1_ONLY_ITEM = re.compile(r"\b(?:and )?yes,?\s+(?:this|that|it)(?:'s| is) the only (?:item|one)\b",
                           re.IGNORECASE)


def _v1_segments(text: str) -> list[str]:
    return [x for x in (p.strip() for p in _V1_SEGMENT.split(text or "")) if x]


def _v1_reply(content: str, proposal: str, point: Any) -> str | None:
    """How a user message stands to this change: "approve", "reject" or None (does
    not concern it, or concerns it without approving or rejecting).

    Sentence by sentence: a sentence that names this change speaks for it; a sentence
    that names no change speaks for the change(s) of the proposal it answers, unless
    the message names a change of another kind or order (then a bare "yes" is not
    attributable). A rejection or hold in a sentence that speaks for the change wins
    over an approval in the same message."""
    order = str(point.arguments.get("order_id") or "")
    # The answer to the once-only reminder ("Yes, that's the only item") names no change.
    segs = [x for x in (_V1_ONLY_ITEM.sub(" ", s).strip(" ,.") for s in _v1_segments(content))
            if x]
    # A bare "yes" answers a proposal only when it asks for a confirmation (not an
    # identity question or a request for details).
    answered = bool(proposal) and _v1_names(proposal, point) and bool(
        _V1_ASKS.search(proposal))
    other = any((set(_V1_ORDER.findall(x)) - {order}) and not _v1_names(x, point)
                or _names_other_kind(x, point) for x in segs)
    # A bare sentence also speaks for a change the same message names ("Please change
    # X. Yes, I approve this exact change.").
    self_named = any(_v1_names(x, point) for x in segs)
    verdict = None
    for seg in segs:
        named = _v1_names(seg, point)
        names_any = named or _names_other_kind(seg, point) or bool(_V1_ORDER.findall(seg))
        neutral = _V1_ONLY_ITEM.sub(" ", _v1_neutral(seg, point))
        if not names_any and _v1_holds(neutral):
            return "reject"  # a bare hold stops every pending change
        if named or (not names_any and (answered or self_named) and not other) or (
                (answered or self_named) and _V1_ALL.search(seg)):
            if _v1_holds(neutral):
                return "reject"
            if _V1_REJECT.search(neutral):
                # Only a leading "no" can be followed by a real go-ahead ("No, please
                # proceed" answers "anything else before I proceed?").
                lead_only = (_V1_LEAD_NO.search(neutral)
                             and not _V1_REJECT.search(_V1_LEAD_NO.sub(" ", neutral)))
                if not (lead_only and _V1_STRONG.search(_v1_strip_negated(neutral))):
                    return "reject"
            if _v1_approves(neutral):
                verdict = "approve"
    return verdict


def code_rule_v1(point: Any) -> str:
    """Draft deterministic consent rule v1 (NOT frozen); see the section comment.

    ``point``: a decision point (``DecisionPoint`` or any object with ``tool``,
    ``arguments``, ``prefix``, ``proposal_content``, ``sibling_calls_before``) and
    optionally ``change`` = {"item_names": [...], "strings": [...]}: the product name
    and option values of the items the write touches (from the pinned database), used
    to recognise an item change by name and to keep option values out of the
    approval test."""
    msgs = [m for m in point.prefix if m["role"] in ("user", "assistant", "tool")]
    texts = [(m["m"], m["role"], m["content"] or "") for m in msgs
             if m["role"] in ("user", "assistant") and (m["content"] or "").strip()]
    anchor = None
    for k in range(len(texts) - 1, -1, -1):
        idx, role, content = texts[k]
        if role != "user":
            continue
        proposal = next((c for _, r, c in reversed(texts[:k]) if r == "assistant"), "")
        verdict = _v1_reply(content, proposal, point)
        if verdict == "reject":
            return "not covered"
        if verdict == "approve":
            anchor = (idx, content, proposal)
            break
    if anchor is None:
        return "not covered"
    idx, content, proposal = anchor
    if _V1_QUALIFY.search(_v1_neutral(content, point)):
        return "ambiguous"
    dest, src = _v1_direction(proposal + "\n" + content)
    if _v1_reversed(dest, src, point):
        return "not covered"  # the approval is of the opposite change (write = its source)
    if _v1_conflicting(proposal + "\n" + content, point):
        return "ambiguous"
    pending = (point.tool, _canon(point.arguments))
    results = {m["tool_call_id"]: m["content"] for m in msgs if m["role"] == "tool"}
    for m in msgs:
        if m["role"] != "assistant" or m["m"] <= idx:
            continue
        for c in m["tool_calls"]:
            if ((c["name"], _canon(c["arguments"])) == pending
                    and not str(results.get(c["id"], "")).lower().startswith("error")):
                return "not covered"
    for c in point.sibling_calls_before:
        if (c["name"], _canon(c["arguments"])) == pending:
            return "not covered"
    return "covered"


def _names_other_kind(text: str, point: Any) -> bool:
    """``text`` names a change of another type (address / item / payment) and not this
    write's type."""
    probes = {"modify_pending_order_address": {"address1": ""},
              "modify_pending_order_items": {"item_ids": [], "new_item_ids": []},
              "modify_pending_order_payment": {"payment_method_id": ""}}
    mine = _v1_kind_named(text, point)
    if mine:
        return False
    for tool, args in probes.items():
        if tool == point.tool:
            continue
        probe = type("P", (), {"tool": tool, "arguments": args, "change": None})()
        if _v1_kind_named(text, probe):
            return True
    return False
