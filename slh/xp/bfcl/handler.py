"""The frozen 9B as a BFCL function-calling model, behind the official multi-turn loop.

Imports ``bfcl_eval`` lazily (partner / BFCL environment only; tests stub it).

What is official and unchanged (``bfcl-eval==2026.3.23``, pinned in ``vendor.py``):

- the episode loop ``BaseHandler.inference_multi_turn_FC`` (``@final`` upstream): turns,
  steps, the 20-step limit, decoding, execution against the official backends
  (``execute_multi_turn_func_call``), the long-context switch, and error handling;
- the per-entry driver ``_llm_response_generation.multi_threaded_inference``;
- the tool conversion, request shape, response parsing and chat-history handling of
  ``OpenAICompletionsHandler`` (the handler BFCL uses for OpenAI-compatible FC models);
- scoring by ``eval_runner._evaluate_single_multi_turn_entry`` (``score_entry``).

What this subclass adds, and nothing else:

1. ``_query_FC``: the official request (messages, model, temperature, store, tools) plus
   ``extra_body={"chat_template_kwargs": {"enable_thinking": False}}`` (the pinned
   non-thinking agent, as served by ``slh/serve_spec.py``) and, optionally, sampling
   fields (``top_p``) fixed by the run plan. Each raw response is captured for the trace
   export (``calls``).
2. ``_pre_query_processing_FC``: when the arm has notes (a lesson library, a reworded
   prompt, a memory note), the first message of the conversation is one system message
   with that text; with no notes the conversation is exactly the official one (B1).
   BFCL FC entries carry no system prompt of their own in the base and long-context
   categories (checked by ``check_no_system_prompt``).
3. ``inference``: the official dispatch, then the complete final message list is attached
   to the metadata (``slh_messages``) for the trace export.

**Unverified until the hand-off 1 feasibility step**: that vLLM 0.30.0 accepts the
``store`` field the official request sends (expected: ignored with a warning); that the
``qwen3_coder`` tool parser returns every BFCL call as ``tool_calls`` (the feasibility
step reports the share of steps with text that contains ``<tool_call>`` but no parsed
call).
"""

from __future__ import annotations

import copy
import threading
from typing import Any

NON_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def _dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_dump(v) for v in obj]
    return obj


def check_no_system_prompt(entry: dict[str, Any]) -> None:
    for turn in entry["question"]:
        for m in turn:
            if m.get("role") == "system":
                raise ValueError(f"{entry['id']}: the official entry has a system message")


def make_handler_class():
    """The handler class (built lazily so that importing this module needs no BFCL)."""
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )

    class SlhQwenFCHandler(OpenAICompletionsHandler):
        def __init__(self, model_name, temperature, registry_name, is_fc_model=True, *,
                     base_url: str, api_key: str = "EMPTY", notes: str = "",
                     sampling: dict[str, Any] | None = None, **kwargs) -> None:
            super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
            from openai import OpenAI

            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self.slh_notes = notes or ""
            self.slh_notes_by_id: dict[str, str] = {}  # per-instance notes (F' draws, memory)
            self.slh_sampling = dict(sampling or {})
            self._slh = threading.local()

        # -- capture helpers (one entry per thread at a time, as upstream runs them)
        def _slh_state(self) -> dict[str, Any]:
            st = getattr(self._slh, "state", None)
            if st is None:
                st = self._slh.state = {"calls": [], "inference_data": None}
            return st

        def _query_FC(self, inference_data: dict):
            message = inference_data["message"]
            tools = inference_data["tools"]
            inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
            kwargs = {"messages": message, "model": self.model_name,
                      "temperature": self.temperature, "store": False}
            if len(tools) > 0:
                kwargs["tools"] = tools
            kwargs.update(self.slh_sampling)
            kwargs["extra_body"] = copy.deepcopy(NON_THINKING)
            api_response, latency = self.generate_with_backoff(**kwargs)
            st = self._slh_state()
            st["calls"].append({"n_messages": len(message), "latency_s": latency,
                                "response": _dump(api_response)})
            return api_response, latency

        def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
            check_no_system_prompt(test_entry)
            notes = self.slh_notes_by_id.get(test_entry["id"], self.slh_notes)
            inference_data["message"] = [{"role": "system", "content": notes}] if notes else []
            st = self._slh_state()
            st["inference_data"] = inference_data
            return inference_data

        def inference(self, test_entry: dict, include_input_log: bool, exclude_state_log: bool):
            self._slh.state = {"calls": [], "inference_data": None}
            result, metadata = super().inference(test_entry, include_input_log,
                                                 exclude_state_log)
            st = self._slh_state()
            data = st.get("inference_data") or {}
            metadata = dict(metadata)
            metadata["slh_messages"] = _dump(data.get("message") or [])
            metadata["slh_tools"] = _dump(data.get("tools") or [])
            metadata["slh_calls"] = st["calls"]
            self._slh.state = None
            return result, metadata

    return SlhQwenFCHandler


def build_handler(*, served_model: str, registry_name: str, base_url: str,
                  api_key: str = "EMPTY", notes: str = "", temperature: float,
                  sampling: dict[str, Any] | None = None, client: Any = None):
    """``client`` replaces the OpenAI client (the dry run's fake model)."""
    import os

    # The official OpenAICompletionsHandler builds an OpenAI client from the environment
    # in its constructor; point it at the local endpoint (process-local variables).
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key or "EMPTY"
    cls = make_handler_class()
    h = cls(served_model, temperature, registry_name, True, base_url=base_url,
            api_key=api_key or "EMPTY", notes=notes, sampling=sampling)
    if client is not None:
        h.client = client
    return h


def run_entry(handler: Any, test_entry: dict[str, Any], *, exclude_state_log: bool = True
              ) -> dict[str, Any]:
    """The official per-entry driver (errors become a result string, as upstream)."""
    from bfcl_eval._llm_response_generation import multi_threaded_inference

    return multi_threaded_inference(handler, copy.deepcopy(test_entry), False,
                                    exclude_state_log)


def load_entries(category: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Official test entries (function docs populated) and possible answers, by id."""
    from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry

    cat = f"multi_turn_{category}"
    prompts = load_dataset_entry(cat, include_prereq=False, include_language_specific_hint=False)
    answers = load_ground_truth_entry(cat)
    return {e["id"]: e for e in prompts}, {a["id"]: a for a in answers}


def score_entry(handler: Any, result_row: dict[str, Any], prompt_entry: dict[str, Any],
                answer: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    """The official multi-turn scorer for one entry; returns its verdict dict."""
    from bfcl_eval.eval_checker.eval_runner import _evaluate_single_multi_turn_entry

    entry_id = result_row["id"]
    test_category = entry_id.rsplit("_", 1)[0]
    return _evaluate_single_multi_turn_entry(
        handler, entry_id, result_row["result"], answer["ground_truth"],
        copy.deepcopy(prompt_entry), model_name, test_category)
