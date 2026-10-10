"""Allowlisted JSON-line projection with pre-decoding exclusion (amendment 5b).

``project_line(text, keys)`` reads one JSON object line and returns only its
allowlisted keys, at any depth. The value of every other key is validated
against the full JSON grammar (exactly where the standard decoder would fail,
the projection fails) and discarded without being decoded into a Python value. A reward, reward breakdown, grader
cost, judge verdict, transcript or model text therefore never exists as a
decoded value in the reading process: only allowlisted values are decoded.

Not part of the execution surface. Used by the identity reconciliation
(``slh.execution``) and the validity readers (``slh.learner.validity``).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_WS = " \t\n\r"


class Projected(dict):
    """A projected JSON object: its allowlisted members, plus ``members``, the
    KEYS (never the values) of every member the object had, so an identity
    object with an unexpected member is still told apart."""

    members: frozenset[str] = frozenset()


def members(value: Any) -> frozenset[str] | None:
    """Member keys of a (projected or fully decoded) object, else None."""
    if isinstance(value, Projected):
        return value.members
    if isinstance(value, dict):
        return frozenset(value)
    return None


def same_object(value: Any, expected: Any) -> bool:
    """Projected (or decoded) ``value`` equals ``expected`` exactly: the same
    member keys and equal allowlisted values (``expected`` fully decoded)."""
    if not isinstance(expected, dict) or not isinstance(value, dict):
        return value == expected
    return members(value) == frozenset(expected) and all(
        k in value and (same_object(value[k], v) if isinstance(v, dict) else value[k] == v)
        for k, v in expected.items())
_DECODER = json.JSONDecoder()


def _ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in _WS:
        i += 1
    return i


_HEX = frozenset("0123456789abcdefABCDEF")
_ESCAPES = frozenset('"\\/bfnrt')
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?")
# What the standard JSON decoder accepts as literals (it allows the three
# non-standard constants), so projection fails exactly where it fails.
_LITERALS = ("true", "false", "null", "NaN", "Infinity", "-Infinity")


def _skip_string(s: str, i: int) -> int:
    """Validate a JSON string starting at the quote ``s[i]`` (escapes, \\uXXXX,
    no raw control characters) and return its end, without building it."""
    i += 1
    while True:
        c = s[i]
        if c == '"':
            return i + 1
        if c == "\\":
            e = s[i + 1]
            if e == "u":
                if not all(h in _HEX for h in s[i + 2:i + 6]) or len(s[i + 2:i + 6]) != 4:
                    raise ValueError("invalid \\u escape")
                i += 6
            elif e in _ESCAPES:
                i += 2
            else:
                raise ValueError("invalid escape")
        elif c < " ":
            raise ValueError("control character in string")
        else:
            i += 1


def _skip_scalar(s: str, i: int) -> int:
    """Validate a JSON string, literal or number at ``i`` and return its end,
    without constructing it. Integers longer than the interpreter's integer
    string-conversion limit fail, exactly as the standard decoder fails."""
    c = s[i]
    if c == '"':
        return _skip_string(s, i)
    for literal in _LITERALS:
        if s.startswith(literal, i):
            return i + len(literal)
    match = _NUMBER.match(s, i)
    if match is None or match.end() == i:
        raise ValueError("invalid JSON value")
    text = match.group(0)
    limit = sys.get_int_max_str_digits()
    if limit and not any(ch in text for ch in ".eE") and len(text.lstrip("-")) > limit:
        raise ValueError("integer exceeds the interpreter's digit limit")
    return match.end()


def _read_key(s: str, i: int, build: bool) -> tuple[str | None, int]:
    if s[i] != '"':
        raise ValueError("object key expected")
    if build:
        key, i = _DECODER.raw_decode(s, i)  # the key name only
    else:
        key, i = None, _skip_string(s, i)
    i = _ws(s, i)
    if s[i] != ":":
        raise ValueError("':' expected")
    return key, _ws(s, i + 1)


def _finish(frame: list[Any]) -> Any:
    container = frame[2]
    if isinstance(container, Projected):
        container.members = frozenset(k for k in frame[3] if k is not None)
    return container


def _parse(s: str, i: int, keys: frozenset[str]) -> tuple[Any, int]:
    """Iterative (explicit-stack) parser of one JSON value: allowlisted members
    are built; every other value is validated against the full grammar and
    discarded without being constructed. No recursion: nesting depth is never
    the limit. A frame is [kind, building, container, member keys, pending key]."""
    stack: list[list[Any]] = []
    build = True
    value: Any = None
    while True:
        i = _ws(s, i)
        c = s[i]
        if c in "{[":
            frame = [c, build, (Projected() if c == "{" else []) if build else None, [], None]
            stack.append(frame)
            i = _ws(s, i + 1)
            if s[i] != ("}" if c == "{" else "]"):
                if c == "{":
                    key, i = _read_key(s, i, build)
                    frame[3].append(key)
                    frame[4] = key
                    build = frame[1] and key in keys
                continue  # read the first member / element
            i += 1
            value = _finish(stack.pop())
        elif build:
            value, i = _DECODER.raw_decode(s, i)
        else:
            value, i = None, _skip_scalar(s, i)
        # Attach the finished value to its parent, then read separators.
        while True:
            if not stack:
                return value, i
            frame = stack[-1]
            if frame[1]:
                if frame[0] == "{":
                    if frame[4] in keys:
                        frame[2][frame[4]] = value
                else:
                    frame[2].append(value)
            i = _ws(s, i)
            close = "}" if frame[0] == "{" else "]"
            if s[i] == ",":
                i = _ws(s, i + 1)
                if frame[0] == "{":
                    key, i = _read_key(s, i, frame[1])
                    frame[3].append(key)
                    frame[4] = key
                    build = frame[1] and key in keys
                else:
                    build = frame[1]
                break  # read the next value
            if s[i] != close:
                raise ValueError(f"',' or '{close}' expected")
            i += 1
            value = _finish(stack.pop())


def project_line(text: str, keys: frozenset[str]) -> dict[str, Any]:
    """The allowlisted projection of one JSON object line (ValueError if malformed)."""
    try:
        value, end = _parse(text, 0, keys)
    except IndexError:
        raise ValueError("truncated JSON") from None
    if not isinstance(value, dict) or _ws(text, end) != len(text):
        raise ValueError("one JSON object per line expected")
    return value


def read_projected_tolerant(path: Path, keys: frozenset[str]
                            ) -> tuple[list[dict[str, Any]], str | None]:
    """``slh.trace.read_jsonl_tolerant`` with the allowlist projection, with its
    exact tolerance and nothing more. Lines are split on LF only (U+2028 and
    U+2029 are valid inside JSON strings). Text after the last LF is a torn
    tail, returned as text. The last complete line may be returned as torn if
    it does not parse and no tail follows (a writer that died before its LF).
    Any other line that does not parse raises (fail closed)."""
    path = Path(path)
    if not path.exists():
        return [], None
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    tail = lines.pop()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(project_line(line, keys))
        except ValueError:
            if index == len(lines) - 1 and not tail:
                return records, line + "\n"
            raise ValueError(f"{path}: corrupt line {index + 1}") from None
    return records, (tail or None)
