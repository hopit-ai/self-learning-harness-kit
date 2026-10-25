"""Text-free probe artifacts; reversible bodies live only in an explicit local store.

Nothing here writes files. Do not serialize ``ContentStore`` itself into a partner
return: serialize ``references()`` or ``compact_messages()`` instead. The return
budget is a strict upper bound on uncompressed UTF-8 JSON (20 MB, decimal).
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

RETURN_LIMIT_BYTES = 20_000_000


def canonical_bytes(value: Any) -> bytes:
    """Stable JSON encoding; rejects non-JSON values and NaN/infinity."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def content_reference(value: Any) -> dict[str, str | int]:
    raw = canonical_bytes(value)
    digest = hashlib.sha256(raw).hexdigest()
    return {"ref": f"sha256:{digest}", "sha256": digest, "bytes": len(raw)}


class ContentStore:
    """In-memory, content-addressed bodies. Callers own its lifetime and disposal."""

    def __init__(self) -> None:
        self._bodies: dict[str, Any] = {}

    def put(self, value: Any) -> str:
        ref = str(content_reference(value)["ref"])
        self._bodies.setdefault(ref, deepcopy(value))
        return ref

    def get(self, ref: str) -> Any:
        """Return an isolated copy; unknown references raise KeyError."""
        return deepcopy(self._bodies[ref])

    def references(self) -> list[dict[str, str | int]]:
        return [content_reference(self._bodies[ref]) for ref in sorted(self._bodies)]


def compact_messages(messages: list[dict[str, Any]], *,
                     store: ContentStore | None = None) -> list[dict[str, Any]]:
    """Replace *whole* messages (including arguments/IDs) with ordered references.

    An optional local store allows later reconstruction. No message text, tool
    arguments, or raw identifiers appear in the compact return.
    """
    records = []
    for index, message in enumerate(messages):
        ref = content_reference(message)
        if store is not None:
            store.put(message)
        records.append({"index": index, **ref})
    return records


def estimate_return_size(value: Any) -> int:
    """Exact uncompressed JSON byte count, usable on an accumulated return."""
    return len(canonical_bytes(value))


def ensure_return_size(value: Any, *, max_bytes: int = RETURN_LIMIT_BYTES) -> int:
    """Return size or fail before export; never silently truncate evidence."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    size = estimate_return_size(value)
    if size >= max_bytes:
        raise ValueError(f"probe return is {size} bytes; must be < {max_bytes}")
    return size
