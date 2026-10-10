"""Arm 2: tau2's optional local think tool, hidden from the orchestrator."""

from dataclasses import asdict, dataclass
from typing import Any

from tau2.agent.llm_agent import LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

from slh.agent import (
    GUARD_GIVE_UP_TEXT,
    GUARD_RETRY_LIMIT,
    SIBLING_BLOCKED,
    Backend,
    Guard,
    HarnessAgent,
)
from slh.budget import Meter
from slh.state import HarnessState

MAX_THINK_ONLY_ROUNDS = 3
# Adapted from the published think-tool retail example; frozen in both configs.
THINK_INSTRUCTION = (
    "You may use the think tool to reason about policy before acting.\n"
    "Review relevant policy rules, required information, and user confirmation.\n"
    "Then choose a real tool call or a response to the user."
)
THINK_GIVE_UP_TEXT = "I'm unable to choose the next action right now."


# Verbatim quote: vendor/tau2-bench/src/tau2/domains/retail/tools.py:716-729
# (Pinned v1.0.1; the source contains no literal JSON schema. Tool derives it
# from this exact bound signature and docstring, as it does for retail tools.)
# @is_tool(ToolType.THINK)
# def think(self, thought: str) -> str:
#     """
#     Use the tool to think about something.
#     It will not obtain new information or change the database, but just append the thought to the log.
#     Use it when complex reasoning or some cache memory is needed.

#     Args:
#         thought: A thought to think about.

#     Returns:
#         Empty string
#     """
#     return ""
class _LocalTools:
    def think(self, thought: str) -> str:
        """
        Use the tool to think about something.
        It will not obtain new information or change the database, but just append the thought to the log.
        Use it when complex reasoning or some cache memory is needed.

        Args:
            thought: A thought to think about.

        Returns:
            Empty string
        """
        return ""


@dataclass(frozen=True)
class ThinkEvent:
    """Private episode telemetry; the thought itself stays in agent history."""

    call_id: str
    text_length: int


@dataclass(frozen=True)
class ThinkCapEvent:
    """A bounded local loop ended without an external action."""

    think_only_rounds: int
    event: str = "think_cap_hit"


class ThinkAgent(HarnessAgent):
    """Keep local thoughts in private history; return only external actions.

    All requests declare think and use the existing agent meter role.
    Guards still check every real call and retain
    the base agent's bounded retry behavior.
    """

    def __init__(
        self, tools: list[Tool], domain_policy: str, backend: Backend, meter: Meter,
        state: HarnessState, guards: list[Guard] | None = None, *,
        think_instruction: str = THINK_INSTRUCTION,
        think_cap_message: str = THINK_GIVE_UP_TEXT,
    ) -> None:
        if think_instruction != THINK_INSTRUCTION:
            raise ValueError("the think arm instruction must match the frozen config text")
        if think_cap_message != THINK_GIVE_UP_TEXT:
            raise ValueError("the think cap message must match the frozen config text")
        if any(tool.name == "think" for tool in tools):
            raise ValueError("the environment already defines a think tool")
        super().__init__(tools=[*tools, Tool(_LocalTools().think)], domain_policy=domain_policy,
                         backend=backend, meter=meter, state=state, guards=guards)
        self._think_events: list[ThinkEvent] = []
        self._think_cap_events: list[ThinkCapEvent] = []
        self._think_instruction = think_instruction
        self._think_cap_message = think_cap_message
        self._initial_model_calls = self._model_calls()

    def _model_calls(self) -> int:
        tally = self.meter.tally["agent"]
        return tally.calls + tally.failed_attempts

    @property
    def system_prompt(self) -> str:
        return f"{super().system_prompt}\n\n{self._think_instruction}"

    @property
    def private_episode_metrics(self) -> dict[str, Any]:
        """A copy for writing under run/private/, never into episode messages."""
        return {
            "think_count": len(self._think_events),
            "think_text_length": sum(event.text_length for event in self._think_events),
            "think_events": [asdict(event) for event in self._think_events],
            "think_cap_hit_count": len(self._think_cap_events),
            "think_cap_events": [asdict(event) for event in self._think_cap_events],
            "model_calls": self._model_calls() - self._initial_model_calls,
        }

    def get_init_state(self, message_history: list[Message] | None = None) -> LLMAgentState:
        state = super().get_init_state(message_history)
        self._think_events.clear()
        self._think_cap_events.clear()
        self._initial_model_calls = self._model_calls()
        return state

    def generate_next_message(
        self, message: UserMessage | ToolMessage | MultiToolMessage, state: LLMAgentState,
    ) -> tuple[AssistantMessage, LLMAgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        think_only_rounds = 0
        guard_retries = 0
        while guard_retries <= GUARD_RETRY_LIMIT:
            proposed = self._complete(state)
            thoughts = [call for call in proposed.tool_calls or [] if call.name == "think"]
            real_calls = [call for call in proposed.tool_calls or [] if call.name != "think"]
            # Keep the original (including raw output) privately. Strip raw_data
            # from filtered copies too: it can contain the removed think calls.
            external = proposed.model_copy(deep=True)
            # One external action may span several model calls. The meter is
            # the accounting source; never expose only the final call's values.
            external.cost = None
            external.usage = None
            external.tool_calls = [
                call for call in external.tool_calls or [] if call.name != "think"
            ] or None
            if thoughts:
                external.raw_data = None
            vetoes = self._vetoes(external, state) if real_calls else {}
            state.messages.append(proposed)
            for call in thoughts:
                thought = call.arguments.get("thought", "")
                self._think_events.append(ThinkEvent(
                    call_id=call.id, text_length=len(thought) if isinstance(thought, str) else 0))
                state.messages.append(ToolMessage(
                    id=call.id, role="tool", requestor="assistant", content=""))
            if thoughts and not real_calls and not external.has_content():
                think_only_rounds += 1
                if think_only_rounds > MAX_THINK_ONLY_ROUNDS:
                    self._think_cap_events.append(ThinkCapEvent(think_only_rounds))
                    break
                continue
            think_only_rounds = 0
            if not vetoes:
                return external, state
            for call in real_calls:
                reason = vetoes.get(call.id, SIBLING_BLOCKED)
                state.messages.append(ToolMessage(
                    id=call.id, role="tool", requestor="assistant",
                    content=f"Blocked: {reason}", error=True))
            guard_retries += 1
        text = GUARD_GIVE_UP_TEXT if guard_retries > GUARD_RETRY_LIMIT else self._think_cap_message
        fallback = AssistantMessage(role="assistant", content=text)
        state.messages.append(fallback)
        return fallback, state
