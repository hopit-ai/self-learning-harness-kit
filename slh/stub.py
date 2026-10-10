"""Deterministic, offline backends for the agent and the user simulator.

They let a full episode run through tau2's real orchestrator, environment and
evaluator at $0. Neither imports or calls the provider library; their output is
a pure function of the messages they are shown.

Stub agent: authenticates from an email or a "First Last ... 12345" name and zip
found in the user's text, looks up each order id it sees (``#W`` + 7 digits),
then replies with what it found. It never calls a write tool, so the database is
left unchanged.

Stub user: opens with the known info and the reason for the call from its
scenario prompt (the simulator legitimately sees its own scenario), then ends
the conversation with tau2's stop token on its next turn.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.user.user_simulator_base import STOP

from slh.budget import ModelConfig, Usage

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
NAME_ZIP = re.compile(r"\b([A-Z][a-z]+) ([A-Z][a-z]+)\b\D{0,40}?\b(\d{5})\b")
ORDER_ID = re.compile(r"#W\d{7}\b")


def _usage(messages: list[Message], reply: AssistantMessage) -> Usage:
    """Token counts sized like a real request so accounting is exercised."""
    prompt = sum(len(m.content or "") for m in messages if hasattr(m, "content"))
    out = len(reply.content or "") + len(json.dumps(
        [c.arguments for c in reply.tool_calls or []]))
    return Usage(input_tokens=math.ceil(prompt / 4), output_tokens=math.ceil(out / 4))


def _with_usage(reply: AssistantMessage, usage: Usage) -> AssistantMessage:
    reply.cost = 0.0
    reply.usage = {"prompt_tokens": usage.input_tokens, "completion_tokens": usage.output_tokens}
    return reply


class StubAgentBackend:
    def __init__(self, config: ModelConfig) -> None:
        if config.backend != "stub":
            raise ValueError("StubAgentBackend needs a stub model config")
        self.config = config

    def complete(
        self, messages: list[Message], tools: list[Tool] | None, seed: int | None
    ) -> tuple[AssistantMessage, Usage]:
        reply = self._decide(messages, {t.name for t in tools or []})
        usage = _usage(messages, reply)
        return _with_usage(reply, usage), usage

    def _decide(self, messages: list[Message], tool_names: set[str]) -> AssistantMessage:
        user_text = " ".join(m.content or "" for m in messages if isinstance(m, UserMessage))
        made: list[tuple[str, str]] = []
        results: dict[str, ToolMessage] = {}
        calls_by_id: dict[str, ToolCall] = {}
        for m in messages:
            if isinstance(m, AssistantMessage) and m.tool_calls:
                for c in m.tool_calls:
                    made.append((c.name, json.dumps(c.arguments, sort_keys=True)))
                    calls_by_id[c.id] = c
            elif isinstance(m, ToolMessage):
                results[m.id] = m

        plan: list[tuple[str, dict[str, Any]]] = []
        email = EMAIL.search(user_text)
        name_zip = NAME_ZIP.search(user_text)
        if email:
            plan.append(("find_user_id_by_email", {"email": email.group(0)}))
        elif name_zip:
            first, last, zip_code = name_zip.groups()
            plan.append(("find_user_id_by_name_zip",
                         {"first_name": first, "last_name": last, "zip": zip_code}))
        for order_id in dict.fromkeys(ORDER_ID.findall(user_text)):
            plan.append(("get_order_details", {"order_id": order_id}))

        for name, arguments in plan:
            if name in tool_names and (name, json.dumps(arguments, sort_keys=True)) not in made:
                call_id = f"stub_call_{len(made)}"
                return AssistantMessage(role="assistant", content=None, tool_calls=[
                    ToolCall(id=call_id, name=name, arguments=arguments)])

        return AssistantMessage(role="assistant", content=self._summary(calls_by_id, results))

    @staticmethod
    def _summary(calls: dict[str, ToolCall], results: dict[str, ToolMessage]) -> str:
        if not calls:
            return ("Could you please share your email address, or your name and zip code, "
                    "so I can locate your account?")
        found = []
        for call_id, call in calls.items():
            result = results.get(call_id)
            if result is None:
                continue
            if result.error:
                found.append(f"{call.name} returned an error")
            elif call.name.startswith("find_user_id"):
                found.append(f"your user id is {result.content}")
            elif call.name == "get_order_details":
                try:
                    status = json.loads(result.content or "{}").get("status", "unknown")
                except ValueError:
                    status = "unknown"
                found.append(f"order {call.arguments.get('order_id')} is {status}")
        detail = "; ".join(found) if found else "I could not find matching records"
        return f"Thank you. Here is what I found: {detail}. I cannot make changes in this session."


SCENARIO_FIELD = re.compile(
    r"^\s*(Reason for call|Known info|Unknown info|Task instructions):\s*\n(.*?)"
    r"(?=^\s*(?:Reason for call|Known info|Unknown info|Task instructions|Domain):|\Z)",
    re.MULTILINE | re.DOTALL,
)


def scenario_fields(system_prompt: str) -> dict[str, str]:
    match = re.search(r"<scenario>(.*?)</scenario>", system_prompt, re.DOTALL)
    text = match.group(1) if match else system_prompt
    return {name: " ".join(body.split()) for name, body in SCENARIO_FIELD.findall(text)}


class StubUserBackend:
    def __init__(self, config: ModelConfig) -> None:
        if config.backend != "stub":
            raise ValueError("StubUserBackend needs a stub model config")
        self.config = config

    def complete(
        self, messages: list[Message], tools: list[Tool] | None, seed: int | None
    ) -> tuple[AssistantMessage, Usage]:
        # The simulator sees roles flipped: its own past turns are "assistant".
        own_turns = sum(isinstance(m, AssistantMessage) for m in messages)
        if own_turns == 0:
            system = next((m.content or "" for m in messages if isinstance(m, SystemMessage)), "")
            fields = scenario_fields(system)
            parts = [fields.get("Known info", ""), fields.get("Reason for call", "")]
            content = " ".join(p for p in ["Hi, I need some help.", *parts] if p)
        else:
            content = f"Thank you, that is all for now. {STOP}"
        reply = AssistantMessage(role="assistant", content=content)
        usage = _usage(messages, reply)
        return _with_usage(reply, usage), usage
