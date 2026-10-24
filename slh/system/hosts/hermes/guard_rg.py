"""Surface guard for the Reasoning Gym-style smoke tasks (M-H0; a declared guard module).

The one write is ``rg_submit_answer``: at most one submission per episode is executed (a
second one is blocked). ``rg_calculate`` is a read. Observations are the (tool, arguments,
result) calls executed earlier in the episode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from slh.system import guards as GD

SUBMIT, CALC = "rg_submit_answer", "rg_calculate"
TOOLS = frozenset({SUBMIT, CALC})


def check(tool: str, arguments: Mapping[str, Any],
          observations: Sequence[tuple[str, Mapping[str, Any], str]]) -> tuple[bool, str]:
    if tool == SUBMIT:
        if any(t == SUBMIT for t, _, _ in observations):
            return False, "second submission"
        if not isinstance(arguments.get("answer"), (str, int, float)):
            return False, "malformed answer"
        return True, "first submission"
    return True, "not a write"


def probe(tool: str, arguments: Mapping[str, Any],
          observations: Sequence[tuple[str, Mapping[str, Any], str]]) -> bool:
    return check(tool, arguments, observations)[0]


CANARIES = [
    ("first-submit", SUBMIT, {"answer": "4"}, [], True),
    ("second-submit", SUBMIT, {"answer": "5"}, [(SUBMIT, {"answer": "4"}, "recorded")], False),
    ("malformed-submit", SUBMIT, {"answer": {"x": 1}}, [], False),
    ("calc-read", CALC, {"expression": "2+2"}, [(SUBMIT, {"answer": "4"}, "recorded")], True),
]
GD.CANARIES.setdefault(__name__, CANARIES)
