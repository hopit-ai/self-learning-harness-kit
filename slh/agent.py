"""The harness agent and user simulator wrappers on tau2's participant interfaces.

``HarnessAgent`` is a tau2 ``HalfDuplexAgent``. It uses tau2's own agent
instructions and policy prompt, and the harness state enters at exactly one
point: ``HarnessState.system_prompt_section()`` appended to that prompt, plus
the guard hook list built from ``HarnessState.guards``. The agent is never
given the task, so scenario and grading fields cannot reach its prompt.

``HarnessUser`` is tau2's ``UserSimulator`` (same guidelines, scenario prompt,
role flipping and stop tokens); only the model call is routed through the
harness backend so it is metered like every other call.

A backend turns (messages, tools) into an ``AssistantMessage``; the stub
backends live in ``slh.stub`` and the provider backend is ``LiteLLMBackend``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from tau2.agent.base_agent import HalfDuplexAgent, is_valid_agent_history_message
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT, LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.user.user_simulator import UserSimulator

from slh.budget import CAPTURE_FIELDS, Meter, ModelConfig, Usage
from slh.state import HarnessState

GUARD_RETRY_LIMIT = 2
SIBLING_BLOCKED = "not executed: another call in this message was blocked"
GUARD_GIVE_UP_TEXT = (
    "I'm sorry, I can't complete that action right now. Is there anything else I can help with?"
)


# --------------------------------------------------------------------------- backends


class Backend(Protocol):
    config: ModelConfig

    def complete(
        self, messages: list[Message], tools: list[Tool] | None, seed: int | None
    ) -> tuple[AssistantMessage, Usage | None]:
        """Return the next message and its token usage. Must not be called directly;
        callers go through ``metered_complete``."""
        ...


def request_payload(messages: list[Message], tools: list[Tool] | None) -> list[Any]:
    """The request as the meter sizes it: message dicts plus tool schemas."""
    from tau2.utils.llm_utils import to_litellm_messages

    payload: list[Any] = list(to_litellm_messages(messages))
    if tools:
        payload.extend(tool.openai_schema for tool in tools)
    return payload


def metered_complete(
    meter: Meter, role: str, backend: Backend, messages: list[Message],
    tools: list[Tool] | None, seed: int | None,
) -> AssistantMessage:
    return meter.call(role, backend.config, request_payload(messages, tools),
                      lambda: backend.complete(messages, tools, seed))


class LiteLLMBackend:
    """Provider call through tau2's ``generate`` (which uses litellm).

    Sends exactly what tau2's default participants send (``temperature`` and the
    per-trial ``seed``) plus only what the config sets: ``max_tokens`` when not
    None and whitelisted extras. Provider retries are off (``num_retries=0``);
    the meter retries and charges each attempt from the wire.
    """

    def __init__(self, config: ModelConfig, call_name: str, level: str = "messages") -> None:
        if config.backend != "litellm":
            raise ValueError("LiteLLMBackend needs a litellm model config")
        if level not in config.capture_levels():
            raise ValueError(f"capture level {level!r} not allowed by the config")
        self.config = config
        self.call_name = call_name
        # Hosted providers give at most log-probs (and only when the pinned
        # provider advertises them); the fields are exactly CAPTURE_FIELDS[level].
        self.level = level

    def request_kwargs(self, seed: int | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = self.config.extra_params()
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens
        if seed is not None:
            kwargs["seed"] = seed
        kwargs.update(CAPTURE_FIELDS[self.level])
        kwargs["num_retries"] = 0
        return kwargs

    def complete(
        self, messages: list[Message], tools: list[Tool] | None, seed: int | None
    ) -> tuple[AssistantMessage, Usage | None]:
        from tau2.utils.llm_utils import generate

        message = generate(model=self.config.model, messages=messages, tools=tools,
                           call_name=self.call_name, **self.request_kwargs(seed))
        # Usage was already charged from the raw response by the meter's guard.
        return message, None


# --------------------------------------------------------------------------- guards


@dataclass(frozen=True)
class GuardVerdict:
    allow: bool
    reason: str = ""


@dataclass(frozen=True)
class GuardContext:
    """What a guard may look at: the agent-visible history and the proposed message."""

    history: Sequence[Message]
    proposed: AssistantMessage


class Guard(Protocol):
    """A hook that can veto a tool call before the environment executes it."""

    name: str

    def check(self, call: ToolCall, context: GuardContext) -> GuardVerdict: ...


@dataclass
class GuardEvent:
    guard: str
    tool: str
    arguments: dict[str, Any]
    reason: str
    turn: int = field(default=-1)


def build_guards(specs: list[dict[str, Any]]) -> list[Guard]:
    """Instantiate guard hooks from harness state configuration. None exist yet."""
    if specs:
        names = [spec.get("name", "?") for spec in specs]
        raise NotImplementedError(f"no guard implementations are available yet: {names}")
    return []


# --------------------------------------------------------------------------- agent


class HarnessAgent(HalfDuplexAgent[LLMAgentState]):
    """tau2 half-duplex agent with the harness state injected at one point."""

    def __init__(
        self, tools: list[Tool], domain_policy: str, backend: Backend, meter: Meter,
        state: HarnessState, guards: list[Guard] | None = None,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.backend = backend
        self.meter = meter
        self.harness_state = state
        self.guards = list(guards or [])
        self.seed = backend.config.seed
        self.guard_events: list[GuardEvent] = []

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(domain_policy=self.domain_policy,
                                    agent_instruction=AGENT_INSTRUCTION)
        section = self.harness_state.system_prompt_section()
        return f"{base}\n\n{section}" if section else base

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    def get_init_state(self, message_history: list[Message] | None = None) -> LLMAgentState:
        history = list(message_history or [])
        if not all(is_valid_agent_history_message(m) for m in history):
            raise ValueError("invalid agent message history")
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=history,
        )

    def _complete(self, state: LLMAgentState) -> AssistantMessage:
        return metered_complete(self.meter, "agent", self.backend,
                                state.system_messages + state.messages, self.tools, self.seed)

    def generate_next_message(
        self, message: UserMessage | ToolMessage | MultiToolMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        for _ in range(GUARD_RETRY_LIMIT + 1):
            proposed = self._complete(state)
            vetoes = self._vetoes(proposed, state)
            if not vetoes:
                state.messages.append(proposed)
                return proposed, state
            # The vetoed message stays in the agent's private history with a
            # blocked result per call; nothing reaches the environment.
            state.messages.append(proposed)
            for call in proposed.tool_calls or []:
                reason = vetoes.get(call.id, SIBLING_BLOCKED)
                state.messages.append(ToolMessage(id=call.id, role="tool", requestor="assistant",
                                                  content=f"Blocked: {reason}", error=True))
        fallback = AssistantMessage(role="assistant", content=GUARD_GIVE_UP_TEXT, cost=0.0)
        state.messages.append(fallback)
        return fallback, state

    def _vetoes(self, proposed: AssistantMessage, state: LLMAgentState) -> dict[str, str]:
        if not self.guards or not proposed.is_tool_call():
            return {}
        context = GuardContext(history=tuple(state.messages), proposed=proposed)
        vetoes: dict[str, str] = {}
        for call in proposed.tool_calls or []:
            for guard in self.guards:
                verdict = guard.check(call, context)
                if not verdict.allow:
                    vetoes[call.id] = f"{guard.name}: {verdict.reason}"
                    self.guard_events.append(GuardEvent(guard=guard.name, tool=call.name,
                                                        arguments=call.arguments,
                                                        reason=verdict.reason))
                    break
        return vetoes


# --------------------------------------------------------------------------- user


class HarnessUser(UserSimulator):
    """tau2's user simulator with its model call routed through the meter."""

    def __init__(
        self, backend: Backend, meter: Meter, instructions: str | None,
        tools: list[Tool] | None = None,
    ) -> None:
        super().__init__(llm=backend.config.model, instructions=instructions, tools=tools)
        self.backend = backend
        self.meter = meter

    def _generate_next_message(self, message, state) -> UserMessage:
        # Same state handling as tau2's UserSimulator; only the call differs.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif (isinstance(message, ToolMessage) or message.has_content()
              or message.is_tool_call()):
            state.messages.append(message)
        messages = state.system_messages + state.flip_roles()
        reply = metered_complete(self.meter, "user", self.backend, messages, self.tools,
                                 self.llm_args.get("seed", self.backend.config.seed))
        user_message = UserMessage(role="user", content=reply.content, cost=reply.cost,
                                   usage=reply.usage, raw_data=reply.raw_data)
        if reply.tool_calls is not None:
            user_message.tool_calls = [
                ToolCall(id=c.id, name=c.name, arguments=c.arguments, requestor="user")
                for c in reply.tool_calls
            ]
        return user_message
