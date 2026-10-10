"""POC 9 loader: the current skill enters B1 at the demo-1 notes slot.

The skill text is ONE harness-state note (``HarnessState`` notes:
'## Harness state' / '### Notes' / '- <skill>' appended to the tau2 agent
prompt, before the think instruction), exactly where demo 1 put its notes.
An empty skill adds nothing, so the prompt is B1's (the removal arm is B1 by
construction). Nothing else of B1 changes (tools, decoding, client caps).
"""

from __future__ import annotations

from typing import Any


def notes_for(skill: str | None) -> list[str]:
    return [skill.strip()] if skill and skill.strip() else []


def render(skill: str | None, config: Any = None) -> dict[str, Any]:
    """System prompt, tools and their hashes for B1 + this skill (offline)."""
    from slh.demo1.context import render as render_b1

    return render_b1(notes_for(skill), config)


def decide(case: dict[str, Any], skill: str | None, client: Any, *,
           seed: int | None = None) -> dict[str, Any]:
    """B1's next action on ``case`` with the skill loaded (demo-1 runner,
    capped client). With no skill the prompt must hash to B1's."""
    from slh.demo1.context import B1_PROMPT_SHA256
    from slh.demo1.decide import b1_seed, next_action

    notes = notes_for(skill)
    return next_action(case, notes, client,
                       seed=b1_seed(client.config) if seed is None else seed,
                       expect_prompt_sha256=None if notes else B1_PROMPT_SHA256)
