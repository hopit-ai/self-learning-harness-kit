"""A library of short lessons: entries {id, when, then, source_case_ids, added_step}.

Rendering is deterministic: entries in id order (``L<n>``, numeric), one line
each, ``- WHEN <when> THEN <then>``, under one ``## Lessons`` heading. The
rendered text is what the agent sees (it enters B1 at the POC 9 skill slot,
``slh.poc9.loader``); ids and provenance are bookkeeping and never rendered
for the agent. An empty library renders to "" (B1's prompt unchanged).
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

from slh.demo1.learn import count_tokens, sha256_text

HEADING = "## Lessons"


def norm_text(text: str) -> str:
    """Whitespace collapsed, ends trimmed (the stored form of when/then)."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", norm_text(text).lower()).strip()


@dataclasses.dataclass(frozen=True)
class Entry:
    id: str
    when: str
    then: str
    source_case_ids: tuple[str, ...]
    added_step: int

    @property
    def number(self) -> int:
        return int(self.id[1:])

    def line(self) -> str:
        return f"- WHEN {self.when.rstrip(' ,;.')} THEN {self.then}"

    def key(self) -> tuple[str, str]:
        return _key(self.when), _key(self.then)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["source_case_ids"] = list(self.source_case_ids)
        return d


@dataclasses.dataclass
class Library:
    entries: list[Entry] = dataclasses.field(default_factory=list)
    next_number: int = 1  # ids are never reused after a removal

    def copy(self) -> Library:
        return Library(list(self.entries), self.next_number)

    def ordered(self) -> list[Entry]:
        return sorted(self.entries, key=lambda e: e.number)

    def get(self, entry_id: str) -> Entry | None:
        return next((e for e in self.entries if e.id == entry_id), None)

    # -- edits (callers validate first)
    def add(self, when: str, then: str, cases: list[str], step: int) -> Entry:
        entry = Entry(f"L{self.next_number}", norm_text(when), norm_text(then),
                      tuple(cases), step)
        self.next_number += 1
        self.entries.append(entry)
        return entry

    def replace(self, entry_id: str, when: str, then: str, cases: list[str], step: int
                ) -> Entry:
        entry = Entry(entry_id, norm_text(when), norm_text(then), tuple(cases), step)
        self.entries = [entry if e.id == entry_id else e for e in self.entries]
        return entry

    def remove(self, entry_id: str) -> None:
        self.entries = [e for e in self.entries if e.id != entry_id]

    # -- rendering
    def render(self) -> str:
        """What the agent sees (no ids)."""
        if not self.entries:
            return ""
        return "\n".join([HEADING, *(e.line() for e in self.ordered())])

    def render_for_writer(self) -> str:
        """What the writer sees: ids included, so ops can name entries."""
        if not self.entries:
            return "(empty)"
        return "\n".join(f"{e.id}: {e.line()[2:]}" for e in self.ordered())

    def tokens(self) -> int:
        text = self.render()
        return count_tokens(text) if text else 0

    def sha256(self) -> str | None:
        """Hash of the rendered text (None when empty), as POC 9's skill_sha256."""
        text = self.render()
        return sha256_text(text) if text else None

    def state_sha256(self) -> str:
        return sha256_text(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")))

    def to_dict(self) -> dict[str, Any]:
        return {"next_number": self.next_number,
                "entries": [e.to_dict() for e in self.ordered()]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Library:
        entries = [Entry(e["id"], e["when"], e["then"], tuple(e["source_case_ids"]),
                         int(e["added_step"])) for e in data.get("entries", [])]
        nxt = data.get("next_number") or (max((e.number for e in entries), default=0) + 1)
        return cls(entries, int(nxt))
