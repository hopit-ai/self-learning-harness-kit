"""Trace capture for the model project: a content-addressed blob store and
per-attempt call records.

- ``BlobStore``: ``<root>/<sha256>.json`` holds canonical JSON (sorted keys,
  no spaces, ASCII) and is named by the sha256 of exactly those bytes, so a
  reader verifies a blob by hashing the file. Writes are atomic (temp file +
  rename) and idempotent, so concurrent workers can store the same blob.
- ``CallRecorder``: the meter's ``on_attempt`` hook. It writes one record per
  model call attempt. Agent records go to the learner-readable ``calls.jsonl``
  after the same leak checks as episode metadata (the agent's own output is
  conversation and is not string-scanned, exactly as ``$.messages`` in
  episodes); user-simulator and judge records go to ``private/calls.jsonl``
  because their requests and outputs carry scenario and grading text.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from slh.tokens import TokenStore, level_rank, min_level
from slh.trace import append_jsonl, check_no_grading_fields, check_no_task_text

PUBLIC_ROLES = ("agent",)
# Agent-produced text: conversation, not metadata (same rule as episode messages).
AGENT_OUTPUT_PATHS = ("$.raw_content", "$.raw_tool_calls", "$.raw_reasoning_content")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def payload_sha256(payload: Any) -> str:
    """Hash of a request as the meter sized it (message dicts, then tool schemas)."""
    return sha256_json(payload)


class BlobStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def put(self, value: Any) -> str:
        data = canonical_bytes(value)
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / f"{digest}.json"
        if path.exists():
            return digest
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / f".{digest}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return digest

    def get(self, digest: str) -> Any:
        path = self.root / f"{digest}.json"
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"blob {digest} does not match its content hash")
        return json.loads(data)


def _turn(payload: Any) -> int | None:
    """Turn index of the message being produced: user and assistant messages in
    the request, counted as episode messages count them (tool results share the
    turn of their call; the system prompt is not a turn)."""
    if not isinstance(payload, list):
        return None
    return sum(1 for item in payload
               if isinstance(item, dict) and item.get("role") in ("user", "assistant"))


def _result_fields(result: Any) -> dict[str, Any]:
    """Output of a backend that made no wire call (stubs): what it returned."""
    calls = []
    for call in getattr(result, "tool_calls", None) or []:
        calls.append({"id": call.id, "type": "function",
                      "function": {"name": call.name,
                                   "arguments": json.dumps(call.arguments)}})
    return {"content": getattr(result, "content", None), "tool_calls": calls}


class CallRecorder:
    """Writes one record per model call attempt (see the module docstring)."""

    def __init__(self, public_path: Path, private_path: Path, *, session: int | None = None,
                 worker: str | None = None, token_store: TokenStore | None = None,
                 requested_levels: dict[str, str] | None = None) -> None:
        self.public_path = Path(public_path)
        self.private_path = Path(private_path)
        self.session = session
        self.worker = worker
        self.token_store = token_store
        self.requested_levels = dict(requested_levels or {})
        self.episode_id: str | None = None
        self.execution_id: str | None = None
        self.windows: frozenset[str] = frozenset()
        self._levels: list[str] = []
        self._count = 0

    def begin(self, episode_id: str, execution_id: str, windows: frozenset[str]) -> None:
        self.episode_id, self.execution_id, self.windows = episode_id, execution_id, windows
        self._levels, self._count = [], 0
        if self.token_store is not None:
            self.token_store.begin(episode_id, execution_id)

    def end(self) -> None:
        self.episode_id, self.execution_id, self.windows = None, None, frozenset()
        if self.token_store is not None:
            self.token_store.end()

    def episode_level(self) -> str:
        """Minimum capture level over this episode's agent calls that returned
        a response (``messages`` if there were none)."""
        return min_level(self._levels)

    def record(self, info: dict[str, Any]) -> dict[str, Any]:
        config = info["config"]
        responses = info["responses"]
        raw = responses[-1] if responses else None
        if raw is None and info["result"] is not None and not info["dispatched"]:
            out = _result_fields(info["result"])
            source = config.backend  # "stub": no wire call
            raw_fields = {"response_id": None, "model_reported": config.model,
                          "provider": config.backend, "finish_reason": None,
                          "usage": _usage_of(info["result"])}
        else:
            out = {"content": (raw or {}).get("content"),
                   "tool_calls": (raw or {}).get("tool_calls") or []}
            source = "wire" if raw is not None else None
            raw_fields = {k: (raw or {}).get(k) for k in
                          ("response_id", "model_reported", "provider", "finish_reason",
                           "usage")}
        if raw is not None or source == "stub":
            parse_ok: bool | None = info["error_type"] is None
        else:
            parse_ok = None  # no response came back: nothing to parse
        payload = info["request_payload"]
        record = {
            "episode_id": self.episode_id, "execution_id": self.execution_id,
            "session": self.session, "worker": self.worker,
            "role": info["role"],
            "turn": None if info["role"] == "judge" else _turn(payload),
            "attempt": info["attempt"],
            "model_requested": config.model,
            "model_reported": raw_fields["model_reported"],
            "provider": raw_fields["provider"],
            "pinned_provider": _pinned(config),
            "response_id": raw_fields["response_id"],
            "finish_reason": raw_fields["finish_reason"],
            "raw_content": out["content"],
            "raw_tool_calls": out["tool_calls"],
            "raw_reasoning_content": (raw or {}).get("reasoning_content"),
            "parse_ok": parse_ok,
            "error_type": info["error_type"],
            "rejected_before_dispatch": info["rejected_before_dispatch"],
            "responses": len(responses),
            "usage": raw_fields["usage"],
            "charged_usd": round(float(info["charged_usd"]), 12),
            "reservation_usd": round(float(info["reservation_usd"]), 12),
            "latency_s": (None if info["wire_latency_s"] is None
                          else round(info["wire_latency_s"], 4)),
            "attempt_s": round(info["attempt_s"], 4),
            "request_sha256": payload_sha256(payload),
            "source": source,
            "time_utc": dt.datetime.now(dt.UTC).isoformat(),
        }
        if raw is not None and "capture_error" in raw:
            record["capture_error"] = raw["capture_error"]
        capture = (raw or {}).get("token_capture") if source == "wire" else None
        requested = self.requested_levels.get(info["role"], "messages") \
            if info["role"] in PUBLIC_ROLES else "messages"
        graded = capture["level"] if capture else "messages"
        level = min(requested, graded, key=level_rank)
        record.update(capture_level_requested=requested, capture_level=level,
                      trainable=bool(capture and capture["trainable"] and level != "messages"),
                      capture_problems=list(capture["problems"]) if capture else [])
        if level == "messages" and capture and capture["problems"]:
            record["trainable"] = False
        self._count += 1
        record["call_id"] = f"{self.execution_id}-{self._count:03d}"
        record["_capture"] = capture if level != "messages" else None
        return record

    def __call__(self, info: dict[str, Any]) -> None:
        record = self.record(info)
        capture = record.pop("_capture")
        if record["role"] in PUBLIC_ROLES and record["source"] is not None:
            self._levels.append(record["capture_level"])
        if record["role"] in PUBLIC_ROLES and capture is not None and \
                self.token_store is not None:
            # Token ids and log-probs live in tokens/, referenced from the record.
            stored = self.token_store.add(record["call_id"], record["turn"],
                                          record["attempt"], {**capture,
                                                              "level": record["capture_level"]})
            if stored is not None:
                record["tokens_ref"] = {"file": stored[0], "line": stored[1]}
                record["parent_call_id"] = self.token_store.last_parent
        if record["role"] in PUBLIC_ROLES:
            check_no_grading_fields(record)
            check_no_task_text(record, self.windows, skip_prefixes=AGENT_OUTPUT_PATHS)
            append_jsonl(self.public_path, record)
        else:
            append_jsonl(self.private_path, record)


def _pinned(config: Any) -> str | None:
    provider = config.extra_params().get("provider") if hasattr(config, "extra_params") else None
    order = (provider or {}).get("order") if isinstance(provider, dict) else None
    return order[0] if order else None


def _usage_of(result: Any) -> dict[str, Any] | None:
    usage = getattr(result, "usage", None)
    return dict(usage) if isinstance(usage, dict) else None
