"""slh-bridge: the harness's Hermes plugin (runs INSIDE the Hermes process; stdlib only).

* ``pre_tool_call`` sends every proposed call (main agent and review fork alike), with the
  arguments Hermes will execute (Hermes' own ``coerce_tool_args`` applied first, v2.2),
  to the harness over a Unix socket (``SLH_BRIDGE_SOCKET``) and returns the harness
  verdict (an allowed call whose arguments coercion changed is returned as ``modify``). Any
  bridge failure RAISES: Hermes' dispatcher turns a raising ``pre_tool_call`` callback into a
  block (``plugins_dispatch.invoke_hook``, fail closed). Hermes' own wrapper
  (``tool_executor._pre_tool_block``) swallows dispatch-level exceptions, so the harness
  also refuses to execute a task tool without a prior allow record, and checks after the run
  that every tool call in the transcript has a mediation record.
* Task tools (schemas in ``SLH_TASK_TOOLS``, toolset ``slh_task``) execute on the harness
  side: the handler forwards the call and returns the harness's result string.

The source of this file is part of the materialisation manifest; the runner attests that the
loaded module's file hashes to the manifest entry.
"""

from __future__ import annotations

import json
import os
import socket

TOOLSET = "slh_task"
COERCE_MAX_PASSES = 5
TIMEOUT_S = float(os.environ.get("SLH_BRIDGE_TIMEOUT_S", "30"))


class BridgeError(RuntimeError):
    pass


def request(msg: dict) -> dict:
    path = os.environ.get("SLH_BRIDGE_SOCKET", "")
    if not path:
        raise BridgeError("SLH_BRIDGE_SOCKET unset")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(TIMEOUT_S)
    try:
        s.connect(path)
        s.sendall((json.dumps(msg) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    finally:
        s.close()
    if not buf:
        raise BridgeError("empty bridge reply")
    reply = json.loads(buf)
    if not isinstance(reply, dict) or reply.get("error"):
        raise BridgeError(f"bridge refused: {reply!r}")
    return reply


def _origin() -> str:
    try:
        from tools.write_approval import current_origin

        return current_origin()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def _call_id() -> str:
    try:
        from tools.approval_context import _approval_tool_call_id

        return _approval_tool_call_id.get() or ""
    except Exception:  # noqa: BLE001
        return ""


def coerce(tool_name, args):
    """The arguments Hermes will EXECUTE: Hermes fires ``pre_tool_call`` with the model's raw
    arguments and only afterwards, in ``model_tools.handle_function_call``, applies
    ``tools.arg_coercion.coerce_tool_args`` (a bare string or a JSON-encoded string for an
    array parameter becomes a list, numeric strings become numbers, ...). The guard must
    decide on, and the harness record, exactly what will run: so the hook applies Hermes'
    own function (the pinned clone's; same registry schema) to a copy, first, repeated to a
    fixpoint (at most ``COERCE_MAX_PASSES``), so Hermes' own coercion at execution leaves
    the approved arguments unchanged and the broker can require exact equality. Raises (the
    hook then blocks, fail closed) when no fixpoint is reached or a key would be renamed or
    dropped (Hermes' shallow ``modify`` merge could not express it)."""
    import copy

    from tools.arg_coercion import coerce_tool_args

    raw = args if isinstance(args, dict) else {}
    out = raw
    # to a FIXPOINT: one pass can return early after parsing an array string, and Hermes
    # coerces again when it executes, so the approved arguments must be stable under it
    for _ in range(COERCE_MAX_PASSES):
        nxt = coerce_tool_args(str(tool_name), copy.deepcopy(out))
        if not isinstance(nxt, dict):
            raise BridgeError("argument coercion returned a non-dict")
        if nxt == out:
            break
        out = nxt
    else:
        raise BridgeError(f"argument coercion of {tool_name} did not settle in "
                          f"{COERCE_MAX_PASSES} passes")
    if set(out) != set(raw):
        raise BridgeError(f"argument coercion changed the argument names of {tool_name}")
    return out


def on_pre_tool_call(tool_name, args, tool_call_id="", **kw):
    """Harness verdict for one proposed call, on the arguments Hermes will execute (``coerce``);
    raises (= Hermes blocks) on any bridge or coercion fault."""
    raw = args if isinstance(args, dict) else {}
    final = coerce(tool_name, raw)
    reply = request({"op": "guard", "tool": str(tool_name), "args": final,
                     "call_id": str(tool_call_id or ""), "origin": _origin(),
                     "task_id": str(kw.get("task_id") or ""),
                     "session_id": str(kw.get("session_id") or ""),
                     "coerced": final != raw})
    action = reply.get("action")
    if action == "allow":
        # Hermes' modify merges these keys over the raw ones (the same key set): it then
        # executes exactly the approved arguments (a fixpoint of its own coercion)
        return {"action": "modify", "args": final} if final != raw else None
    if action == "modify" and isinstance(reply.get("args"), dict):
        return {"action": "modify", "args": reply["args"]}
    if action == "block":
        return {"action": "block", "message": str(reply.get("message") or "blocked by guard")}
    raise BridgeError(f"unknown guard action {action!r}")


def _make_handler(name):
    def handle(params, **kw):
        try:
            reply = request({"op": "tool", "tool": name, "args": params or {},
                             "call_id": _call_id()})
        except Exception as e:  # noqa: BLE001 - the model sees a tool error, never a result
            return json.dumps({"error": f"tool bridge failure: {e}"})
        return str(reply.get("result", ""))

    handle.__name__ = f"slh_task_{name}"
    return handle


def _task_tools() -> list:
    path = os.environ.get("SLH_TASK_TOOLS", "")
    if not path:
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def register(ctx):
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    for schema in _task_tools():
        ctx.register_tool(name=schema["name"], toolset=TOOLSET, schema=schema,
                          handler=_make_handler(schema["name"]))
