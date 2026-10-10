"""The B1 decision context, rendered offline exactly as the runner renders it.

B1 is ``configs/arm-think-vllm.yaml`` with the empty harness state
(``state/588221e7aa723fdc``): tau2's agent prompt and retail policy, the frozen
think instruction and the retail tools plus the local think tool
(``slh.arms.think.ThinkAgent``). An arm that adds text puts it in the harness
state's notes, so it enters at the single point the runner uses for learned
notes (``HarnessState.system_prompt_section``, before the think instruction).
No model is called here.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from slh.demo1.schema import REPO, sha256_text

RUNNER_CONFIG = "configs/arm-think-vllm.yaml"
B1_STATE_VERSION = "588221e7aa723fdc"
# Recorded by every B1 episode (runs/b1-stream) and re-rendered by
# slh.execution.rendered_prompt_hashes for the empty state.
B1_PROMPT_SHA256 = "386fcf136aadb717d46189b82d80f0d120487a08d8d5214207aef0d763b2b54f"
B1_TOOLS_SHA256 = "ab158d0765ddd4411f4d51642099cae6c7e09827a88cedb275dd811995e56cae"
# The execution surface demo 1 runs on: after the serving GPU fallback
# (docs/surface-migration-2026-10-09-gpu.md; L40S, then one fallback to
# A100-40GB). Every demo-1 window-B arm runs fresh on it, in one window, so on
# one GPU type. PROPOSED: binding only once the owner appends the superseding
# registration row (docs/demo1/registration-plan-2026-10-09-gpu.md).
SURFACE_SHA256 = "e11234dc788821e9a03a7bf1bbcb260adf51ab83c6e9a4b5538a1c7e4bdf1d05"
# The surface this migration starts from (tunnel-wait fix,
# docs/surface-migration-2026-10-08.md; window A ran on it).
SURFACE_SHA256_BEFORE_MIGRATION = \
    "8d1ee551dedd3fabd066b50a3dd3b98dcdd05b66c80645dbf1c3eed113fa7615"
# History: the surface amendment 8 was first registered on.
SURFACE_SHA256_ORIGINAL = "960d25c127d2bf344557d34d25d9d5dd89efcb6e5bfa29fc217cc62a74d8ca19"
CONTEXT_FILE = REPO / "demo1/b1-context.json"
# B0: tau2's reference agent and the same served model (configs/a0-vllm.yaml);
# hashes recorded by every episode of runs/a0-stream and runs/a0-dev.
B0_CONFIG = "configs/a0-vllm.yaml"
B0_PROMPT_SHA256 = "3384b6e52d9395d9d8369777f4e06f423a8a801447ef3b58ad8897b56105f04b"
B0_TOOLS_SHA256 = "c1b3ff400ce83557f898d8603de1ebb9aee48ebd7ab0524698a8d5068b76ac73"
B0_CONTEXT_FILE = REPO / "demo1/b0-context.json"


class _Offline:
    def __init__(self, model: Any) -> None:
        self.config = model

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("offline rendering never calls a model")


def runner_config(path: str | Path = RUNNER_CONFIG) -> Any:
    from slh.runner import load_config

    path = Path(path)
    return load_config(path if path.is_absolute() else REPO / path)


def state_for(notes: list[str] | None = None) -> Any:
    from slh.state import HarnessState

    return HarnessState(notes=[{"content": n} for n in notes or []])


def build_agent(config: Any, state: Any, backend: Any, meter: Any,
                environment: Any = None) -> Any:
    """The agent object as ``slh.runner._run_pairs`` builds it: the think arm
    (B1) or the baseline arm (B0: tau2's reference agent, no think tool or
    instruction). ``environment``: a fresh retail environment unless passed."""
    from tau2.registry import registry

    from slh.agent import HarnessAgent, build_guards
    from slh.arms.think import ThinkAgent

    environment = environment or registry.get_env_constructor("retail")()
    common = {"tools": environment.get_tools(), "domain_policy": environment.get_policy(),
              "backend": backend, "meter": meter, "state": state,
              "guards": build_guards(state.guards)}
    if config.arm == "think":
        return ThinkAgent(**common, think_instruction=config.think_instruction,
                          think_cap_message=config.think_cap_message)
    if config.arm == "baseline":
        return HarnessAgent(**common)
    raise ValueError(f"unknown arm {config.arm!r}")


def render(notes: list[str] | None = None, config: Any = None) -> dict[str, Any]:
    """System prompt and tool schemas for B1 plus ``notes`` (offline)."""
    from slh.budget import Meter
    from slh.capture import sha256_json

    config = config or runner_config()
    state = state_for(notes)
    agent = build_agent(config, state, _Offline(config.agent), Meter(0.0, {}))
    tools = [tool.openai_schema for tool in agent.tools]
    return {"system_prompt": agent.system_prompt, "tools": tools,
            "system_prompt_sha256": sha256_json(agent.system_prompt),
            "tools_sha256": sha256_json(tools), "state_version": state.version_id}


def b1_context() -> dict[str, Any]:
    from slh.execution import base_arm_fingerprint

    config = runner_config()
    rendered = render([], config)
    if rendered["system_prompt_sha256"] != B1_PROMPT_SHA256 or \
            rendered["tools_sha256"] != B1_TOOLS_SHA256:
        raise ValueError("the rendered B1 prompt or tools differ from the recorded B1 hashes")
    if rendered["state_version"] != B1_STATE_VERSION:
        raise ValueError("the empty state is not the B1 state")
    return {"schema": "demo1-b1-context-v1", "runner_config": RUNNER_CONFIG,
            "base_arm": base_arm_fingerprint(config), "state_version": B1_STATE_VERSION,
            "notes_insertion": "HarnessState notes: '## Harness state' / '### Notes' / "
                               "'- <content>' appended to the tau2 agent prompt, before "
                               "the think instruction",
            **{k: rendered[k] for k in ("system_prompt", "tools", "system_prompt_sha256",
                                        "tools_sha256")}}


@functools.lru_cache(maxsize=1)
def b0_config() -> Any:
    return runner_config(B0_CONFIG)


def b0_context() -> dict[str, Any]:
    """The B0 prompt and tools, checked against the recorded B0 hashes; the
    agent model settings must equal B1's (same served model, decoding, seed)."""
    from slh.execution import base_arm_fingerprint

    config, b1 = b0_config(), runner_config()
    if config.arm != "baseline":
        raise ValueError("B0 is the baseline arm")
    if config.agent != b1.agent or config.seed != b1.seed:
        raise ValueError("B0 must use the same agent model settings and seed as B1")
    rendered = render([], config)
    if rendered["system_prompt_sha256"] != B0_PROMPT_SHA256 or \
            rendered["tools_sha256"] != B0_TOOLS_SHA256:
        raise ValueError("the rendered B0 prompt or tools differ from the recorded B0 hashes")
    return {"schema": "demo1-b0-context-v1", "runner_config": B0_CONFIG,
            "base_arm": base_arm_fingerprint(config),
            **{k: rendered[k] for k in ("system_prompt", "tools", "system_prompt_sha256",
                                        "tools_sha256")}}


def write_b0_context(path: Path = B0_CONTEXT_FILE) -> None:
    path.write_text(json.dumps(b0_context(), sort_keys=True, indent=1, ensure_ascii=True)
                    + "\n", encoding="utf-8")


def write_b1_context(path: Path = CONTEXT_FILE) -> dict[str, Any]:
    ctx = b1_context()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ctx, sort_keys=True, indent=1, ensure_ascii=True) + "\n",
                    encoding="utf-8")
    return {"file": str(path.relative_to(REPO)), "sha256": sha256_text(path.read_text())}
