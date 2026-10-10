"""Deterministic observations, never diagnoses based on hidden task data."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slh.lessons import require_evidence_eligible, valid_episode_ids
from slh.trace import check_no_grading_fields

COMPONENTS = frozenset({"DB", "ENV_ASSERTION", "NL_ASSERTION", "COMMUNICATE"})
WRITE_PREFIXES = ("cancel_", "return_", "exchange_", "modify_", "update_", "book_",
                  "create_", "delete_", "refund_", "transfer_")
CONFIRMATION = re.compile(
    r"\b(?:yes|confirm|confirmed|i agree|i approve|go ahead|please proceed)\b", re.IGNORECASE)
NEGATION = re.compile(r"\b(?:no|not|don't|do not|never|wait|unless|if)\b", re.IGNORECASE)
IDENTIFIERS = re.compile(
    r"\b[a-z]+_[a-z]+_\d+\b|\b(?:credit_card|gift_card)_\w+\b|"
    r"(?<!\w)(?:\+\d{1,3}[ .-]?)?\d{5}[ .-]\d{5}(?!\w)|"
    r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\w)|"
    r"\b[A-Za-z]+[-_]?\d+[A-Za-z\d_-]*\b|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"\b(?:order|user|customer|product|item|reservation)[_ -]?(?:id[_ :=-]*)?"
    r"(?:[A-Za-z]*\d[A-Za-z\d_-]*|[A-Za-z]+_[A-Za-z\d_]+)\b|"
    r"\b[A-Za-z]+_\d+(?:_[A-Za-z\d]+)*\b|(?<![#\w])#\w+(?![#\w])|\b\d{4,}\b", re.IGNORECASE)
IDENTITY_KEY = re.compile(r"(?:^|_)(?:id|ids|name|names|email|address|phone)$", re.IGNORECASE)


def public_path(path: str | Path) -> Path:
    """Reject private paths and symlink aliases before opening anything."""
    path = Path(path).absolute()
    # macOS temporary files live under /private/var. Only the filesystem-root
    # /private prefix is exempt; a run's nested private directory is forbidden.
    if "private" in path.parts[2:] or "private" in path.resolve().parts[2:]:
        raise ValueError("private paths are not learner-readable")
    from slh.sealing import is_sealed

    if is_sealed(path):  # sealed diagnostics are never learner input, even once released
        raise ValueError("sealed diagnostic paths are not learner-readable")
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("learner inputs must not use symlinks")
    return path.resolve()


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    lines = public_path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                break  # only a torn final record may be skipped
            raise ValueError("malformed interior evidence record") from None
        if not isinstance(row, dict):
            raise ValueError("evidence records must be objects")
        rows.append(row)
    return rows


def unique_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index = {}
    for row in rows:
        eid = row.get("episode_id")
        if not isinstance(eid, str) or not eid or eid in index:
            raise ValueError("episode ids must be nonempty and unique within a run")
        index[eid] = row
    return index


def score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not (
        math.isfinite(value) and 0 <= value <= 1
    ):
        raise ValueError("scores must be finite numbers in [0, 1]")
    return float(value)


def clean_messages(rows: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(rows, list):
        raise ValueError("messages must be a list")
    result = []
    think_ids = {call.get("id") for row in rows if isinstance(row, dict)
                 for call in row.get("tool_calls") or []
                 if isinstance(call, dict) and call.get("name") == "think" and call.get("id")}
    for row in rows:
        if not isinstance(row, dict) or row.get("role") not in {"user", "assistant", "tool"}:
            raise ValueError("unexpected evidence message")
        if row["role"] == "tool" and row.get("tool_call_id") in think_ids:
            continue
        content = row.get("content") or ""
        if not isinstance(content, str):
            raise ValueError("message content must be text")
        # Tool results may be JSON strings. Refuse embedded grading fields too.
        try:
            structured = json.loads(content)
        except ValueError:
            structured = None
        check_no_grading_fields(structured)
        calls = []
        for call in row.get("tool_calls") or []:
            name, args = call.get("name"), call.get("arguments", {})
            if name == "think":
                continue  # Only the dedicated sanitized projection is usable evidence.
            if not isinstance(name, str) or not isinstance(args, dict):
                raise ValueError("tool calls need a name and object arguments")
            check_no_grading_fields(args)
            cleaned = {"name": name, "arguments": args}
            if isinstance(call.get("id"), str):
                cleaned["id"] = call["id"]
            calls.append(cleaned)
        message = {"role": row["role"], "content": content, "tool_calls": calls}
        if isinstance(row.get("turn"), int):
            message["turn"] = row["turn"]
        if isinstance(row.get("tool_call_id"), str):
            message["tool_call_id"] = row["tool_call_id"]
        result.append(message)
    return tuple(result)


@dataclass(frozen=True)
class Evidence:
    run_manifest: str
    episode_id: str
    task_id: str  # bookkeeping only; never sent to the proposer
    messages: tuple[dict[str, Any], ...]
    termination_reason: str
    reward: float
    components: dict[str, bool]
    agent_system_prompt: str | None = None  # Local compliance only; never proposal input.
    family: str | None = None  # Split bookkeeping, separate from behavioural clusters.
    public_think: tuple[dict[str, Any], ...] = ()
    split_name: str = "stream"
    numeric_components: dict[str, float] = field(default_factory=dict)


def is_stub_run(manifest: dict[str, Any]) -> bool:
    """A run made entirely of stubs: stub agent, stub user, stub judge."""
    return (manifest.get("evidence_eligible") is False
            and (manifest.get("agent_model") or {}).get("backend") == "stub"
            and (manifest.get("user_model") or {}).get("backend") == "stub"
            and manifest.get("judge_mode") == "stub")


def require_rehearsal_run(manifest: dict[str, Any]) -> None:
    """Rehearsal accepts ONLY all-stub runs (never real evidence, never mixed), from
    a finished stream run; it marks every output as rehearsal."""
    if not is_stub_run(manifest):
        raise ValueError("rehearsal mode accepts only all-stub runs (stub agent, user and "
                         "judge); real evidence must go through a normal round")
    if manifest.get("split_name") != "stream":
        raise ValueError("rehearsal evidence must still come from a stream-split run")
    if manifest.get("status") not in ("complete", "complete_with_errors"):
        raise ValueError("rehearsal evidence must come from a finished run")


def load_evidence(run_dirs: list[str | Path], *, split_path: str | Path | None = None,
                  rehearsal: bool = False) -> list[Evidence]:
    split_path = public_path(split_path or Path(__file__).resolve().parents[2] /
                             "splits/retail.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    stream = set(split["stream"])
    result = []
    seen_runs = set()
    for directory in run_dirs:
        directory = public_path(directory)
        if directory in seen_runs:
            raise ValueError("duplicate evidence run")
        seen_runs.add(directory)
        manifest_path = public_path(directory / "manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if rehearsal:
            require_rehearsal_run(manifest)
        else:
            require_evidence_eligible(manifest)
        if manifest.get("domain") != "retail" or not set(manifest["task_ids"]) <= stream:
            raise ValueError("evidence tasks must belong to the frozen retail stream")
        # Check both paths before the shared helper opens them.
        for name in ("episodes.jsonl", "outcomes.jsonl"):
            public_path(directory / name)
        valid = valid_episode_ids(directory)
        episodes = unique_index(jsonl(directory / "episodes.jsonl"))
        outcomes = unique_index(jsonl(directory / "outcomes.jsonl"))
        thoughts = load_public_think(directory, episodes)
        for eid, episode in episodes.items():
            if eid not in valid:
                continue
            outcome = outcomes[eid]
            task = episode["task_id"]
            if task not in manifest["task_ids"] or outcome.get("task_id") != task:
                raise ValueError("episode task does not match verified stream manifest")
            numeric_components = {key: score(value) for key, value in
                                  (outcome.get("reward_breakdown") or {}).items() if key in COMPONENTS}
            components = {key: value == 1 for key, value in numeric_components.items()}
            result.append(Evidence(str(manifest_path), eid, task,
                                   clean_messages(episode.get("messages", [])),
                                   str(episode.get("termination_reason", "unknown")),
                                   score(outcome["reward"]), components,
                                   load_agent_prompt(directory, episode),
                                   split.get("family_map", {}).get(task),
                                   thoughts.get(eid, ()), numeric_components=numeric_components))
    return result


def load_public_think(directory: Path, episodes: dict[str, dict[str, Any]] | None = None
                      ) -> dict[str, tuple[dict[str, Any], ...]]:
    """Read only a dedicated public, producer-sanitized projection, if available.

    No fallback to raw think calls or private captures. Only thought text and
    its public trajectory position survive; metadata is not forwarded.

    A think-arm run, or any run carrying ``think_public.jsonl``, is refused
    unless ``slh.think_projection.verify`` confirms the file is the current
    strict projection (summary present, sources and output unchanged, and
    byte-identical to a fresh strict rebuild): a tampered ``sanitized: true``
    row or a missing summary never reaches the learner.
    """
    manifest_path = public_path(directory / "manifest.json")
    arm = json.loads(manifest_path.read_text(encoding="utf-8")).get("arm") \
        if manifest_path.exists() else None
    if arm == "think" or public_path(directory / "think_public.jsonl").exists():
        from slh.think_projection import verify

        verify(directory)
    found: dict[str, list[dict[str, Any]]] = {}
    for name in ("think_public.jsonl",):
        path = public_path(directory / name)
        if not path.exists():
            continue
        seen = set()
        for source_row, row in enumerate(jsonl(path), 1):
            if not row.get("execution_id"):
                continue  # Legacy rows cannot supply audited execution provenance.
            check_no_grading_fields(row)
            if row.get("sanitized") is not True:
                raise ValueError("think projection must be sanitized")
            eid = row.get("episode_id")
            if not isinstance(eid, str):
                raise ValueError("public think projection needs an episode id")
            if episodes is not None:
                if eid not in episodes:
                    raise ValueError("public think projection has an unknown episode")
                execution = episodes[eid].get("execution_id")
                if execution is not None and row.get("execution_id") != execution:
                    raise ValueError("public think projection execution does not match episode")
            execution = row.get("execution_id")
            if not isinstance(execution, str) or not execution:
                raise ValueError("public think projection needs an execution id")
            text = row.get("content")
            if not isinstance(text, str):
                raise ValueError("public think projection needs text")
            try:
                check_no_grading_fields(json.loads(text))
            except json.JSONDecodeError:
                pass
            turn = row.get("turn")
            if turn is not None and (isinstance(turn, bool) or not isinstance(turn, int) or turn < 0):
                raise ValueError("public think position must be an integer")
            index = row.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("public think index must be a nonnegative integer")
            key = (eid, execution, turn, index)
            if key in seen:
                raise ValueError("duplicate public think position")
            seen.add(key)
            found.setdefault(eid, []).append({"turn": turn, "index": index,
                                             "source_row": source_row, "content": text,
                                             "sanitized": True})
        break
    return {eid: tuple(rows) for eid, rows in found.items()}


def is_write(name: str) -> bool:
    return name != "transfer_to_human_agents" and name.startswith(WRITE_PREFIXES)


def signatures(episode: Evidence, *, failures_only: bool = True) -> tuple[str, ...]:
    if failures_only and episode.reward == 1:
        return ()
    found = set()
    calls: Counter[str] = Counter()
    last_user = ""
    writes = 0
    for message in episode.messages:
        if message["role"] == "user":
            last_user = message["content"]
        if message["role"] == "tool" and re.search(
                r'\b(?:error|failed|invalid|not found)\b', message["content"], re.IGNORECASE):
            found.add("tool_error")
        if message["role"] != "assistant":
            continue
        for call in message["tool_calls"]:
            if call["name"] == "think":
                continue
            calls[json.dumps({"name": call["name"], "arguments": call["arguments"]},
                             sort_keys=True)] += 1
            if is_write(call["name"]):
                writes += 1
                if not CONFIRMATION.search(last_user) or NEGATION.search(last_user):
                    found.add("write_without_explicit_confirmation")
    if any(count > 1 for count in calls.values()):
        found.add("repeated_identical_tool_call")
    reason = episode.termination_reason.casefold()
    if reason in {"max_steps", "max_turns", "too_many_steps"}:
        found.add("max_steps")
    elif reason in {"max_errors", "too_many_errors", "timeout", "error"}:
        found.add("termination_" + reason)
    if not writes and episode.components.get("DB") is False:
        found.add("agent_ended_without_write_db_failed")
    return tuple(sorted(found or ({"unclassified_failure"} if episode.reward != 1 else set())))


HEURISTIC_SIGNATURES = frozenset({"write_without_explicit_confirmation", "tool_error"})


def observed_signatures(episode: Evidence, *, failures_only: bool = True) -> tuple[str, ...]:
    """Direct events/component bits. Consent/error-text guesses are not labels."""
    if failures_only and episode.reward == 1:
        return ()
    found = set(signatures(episode, failures_only=False)) - HEURISTIC_SIGNATURES
    # A structured error field is observable; a word like 'invalid' is a heuristic.
    for message in episode.messages:
        if message["role"] != "tool":
            continue
        try:
            result = json.loads(message["content"])
        except ValueError:
            continue
        if isinstance(result, dict) and result.get("error"):
            found.add("tool_error")
    if episode.components.get("DB") is True and episode.components.get("NL_ASSERTION") is False:
        found.add("db_pass_nl_fail")
    if len(found) > 1:
        found.discard("unclassified_failure")
    if not found and episode.reward != 1:
        found.add("component_failure")
    return tuple(sorted(found))


def heuristic_signatures(episode: Evidence) -> tuple[str, ...]:
    return tuple(sorted(set(signatures(episode, failures_only=False)) -
                        set(observed_signatures(episode, failures_only=False))))


def redact(text: str, limit: int = 600) -> str:
    from slh.learner.aliases import AliasTable

    aliases = AliasTable()
    aliases.collect_text(text)
    # Legacy standalone helper; bundle projections also discover structured names.
    for match in IDENTIFIERS.finditer(text):
        aliases.register(match.group(), "ID")
    return aliases.text(text)[:limit]


def identifier_shaped(value: str) -> bool:
    """Only identifier-shaped values may veto durable procedural notes."""
    return bool(re.search(r"[\d_#@]", value) or IDENTIFIERS.fullmatch(value))


def identity_values(value: Any, *, context: str | None = None) -> set[str]:
    """Identifier-shaped JSON values; addresses/catalogue names are excluded."""
    from slh.learner.aliases import child_context, field_type, person_name_fields, public_name_field

    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            kind = field_type(key)
            if kind == "ADDRESS" or public_name_field(key, value, context):
                continue
            if key in person_name_fields(value, context):
                kind = "NAME"
            if kind:
                candidates = child if isinstance(child, list) else (
                    list(child) if isinstance(child, dict) and key in {"variants", "payment_methods"} else [child])
                found.update(str(v) for v in candidates if isinstance(v, (str, int, float))
                             and not isinstance(v, bool) and identifier_shaped(str(v)))
            found.update(identity_values(child, context=child_context(key, context)))
    elif isinstance(value, list):
        for child in value:
            found.update(identity_values(child, context=context))
    return found


def redact_object(value: Any) -> Any:
    from slh.learner.aliases import alias_bundle

    return alias_bundle(value)


def redact_content(text: str, limit: int = 120) -> str:
    try:
        value = json.loads(text)
    except ValueError:
        return redact(text, limit)
    return redact(json.dumps(redact_object(value), sort_keys=True), limit)


def excerpt(episode: Evidence) -> str:
    # First observable error or unconfirmed write is the decisive step. Include
    # preceding user/assistant context, rather than an arbitrary head/tail sample.
    decisive = max(0, len(episode.messages) - 1)
    last_user = ""
    for index, message in enumerate(episode.messages):
        if message["role"] == "user":
            last_user = message["content"]
        error = message["role"] == "tool" and re.search(
            r"\b(?:error|failed|invalid|not found)\b", message["content"], re.IGNORECASE)
        failing_write = any(is_write(c["name"])
                            for c in message["tool_calls"]) and (
            not CONFIRMATION.search(last_user) or NEGATION.search(last_user))
        if error or failing_write:
            decisive = index
            break
    lines = []
    for message in episode.messages[max(0, decisive - 2):decisive + 3]:
        content = redact_content(message["content"], max(260, len(message["content"])))
        marker = re.search(r"\b(?:error|failed|invalid|not found)\b", content, re.IGNORECASE)
        start = max(0, marker.start() - 60) if marker else 0
        calls = [c["name"] + " " + redact(json.dumps(redact_object(c["arguments"]),
                                                  sort_keys=True), 160)
                 for c in message["tool_calls"][:2]]
        # Tool names precede content so a long assistant turn cannot hide the write.
        text = message["role"] + ": " + " ".join(calls) + " " + content[start:start + 260]
        lines.append(text[:290])
    return redact("\n".join(lines), 1500)


def load_agent_prompt(directory: Path, episode: dict[str, Any]) -> str | None:
    """Only the exact public agent-prompt blob, never tools or simulator blobs."""
    digest = episode.get("agent_system_prompt_sha256")
    if digest is None:
        return None  # Legacy evidence cannot establish that a note was loaded.
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid agent prompt hash")
    body = public_path(directory / "blobs" / f"{digest}.json").read_bytes()
    if hashlib.sha256(body).hexdigest() != digest:
        raise ValueError("agent prompt blob hash mismatch")
    prompt = json.loads(body)
    if not isinstance(prompt, str):
        raise ValueError("agent prompt must be text")
    return prompt


def note_compliance(state, episodes: list[Evidence]) -> list[dict[str, Any]]:
    signals = []
    for note in state.notes:
        if not note.get("target_signature"):
            continue
        violated = followed = unknown = absent = 0
        for episode in episodes:
            if episode.agent_system_prompt is None:
                unknown += 1
            elif f"\n- {note['content']}\n" not in f"\n{episode.agent_system_prompt}\n":
                absent += 1
            elif note["target_signature"] in (set(signatures(episode, failures_only=False)) |
                                              set(observed_signatures(episode, failures_only=False))):
                violated += 1
            else:
                followed += 1
        signals.append({"lesson_id": note.get("lesson_id"),
                        "target_signature": note["target_signature"],
                        "loaded_but_violated": violated, "loaded_without_signature": followed,
                        "not_loaded": absent, "prompt_unknown": unknown})
    return signals


@dataclass
class SignatureGroup:
    signature: str
    count: int = 0
    examples: list[Evidence] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {"signature": self.signature, "count": self.count,
                "signal_kind": "observed" if any(self.signature in observed_signatures(e)
                                                 for e in self.examples) else "heuristic",
                "observed_signatures": sorted({s for e in self.examples
                                               for s in observed_signatures(e)}),
                "heuristic_signatures": sorted({s for e in self.examples
                                                for s in heuristic_signatures(e)}),
                "example_episode_ids": [e.episode_id for e in self.examples],
                "examples": [excerpt(e) for e in self.examples]}


def triage(episodes: list[Evidence], example_limit: int = 3) -> list[SignatureGroup]:
    if example_limit not in (2, 3):
        raise ValueError("keep two or three examples per signature when available")
    groups = {}
    for episode in episodes:
        for signature in sorted(set(signatures(episode)) | set(observed_signatures(episode))):
            group = groups.setdefault(signature, SignatureGroup(signature))
            group.count += 1
            if len(group.examples) < example_limit:
                group.examples.append(episode)
    return sorted(groups.values(), key=lambda g: (-g.count, g.signature))
