"""Harness side of the slh-bridge: a Unix-socket server per episode and the broker behind it.

The broker is where mediation happens and is recorded:

* ``guard``: the deployed chain (host floor ``guard_host`` by role, then the surface's
  guards on the episode's observations) decides allow / block; one mediation record per
  proposed call (main agent, review fork and the startup canary alike);
* ``tool``: a task tool executes only against an unconsumed allow record for the same call
  (by tool-call id, else by tool and canonical arguments); otherwise it is refused and logged
  as an unmediated execution attempt (Hermes swallows hook-dispatch exceptions, so this is
  the fail-closed backstop);
* ``hello``: the runner's attestation; the broker refuses a run whose materialisation hash
  or preflight does not match;
* ``first_user`` / ``user_turn``: the dialogue (task prompt, or a customer simulator);
* ``boundary`` (S1 headroom probe episodes only): the conversation at Hermes' post-tool
  compression boundary; the probe answers with content edits of tool messages.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from slh.system import guards as GD
from slh.system.hosts.hermes import guard_host


def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str)


# Hermes write origins of the main agent (tools.skill_provenance; turn_context sets
# "assistant_tool" for a turn). The review fork is "background_review"; anything else is denied.
FOREGROUND = frozenset({"foreground", "assistant_tool"})


class Broker:
    """Review fixes (2026-10-23): task tools only for foreground calls and only while the
    dialogue is open; execution bound to (call id, tool, canonical arguments), one use per id,
    no fallback; the customer call runs outside the broker lock."""

    def __init__(self, session: Any, *, role: str, task_tools: set[str],
                 surface_guard: Callable[..., tuple[bool, str, dict[str, Any]]],
                 expected_materialization: str,
                 on_hello: Callable[[Broker, Mapping[str, Any]], None] | None = None,
                 on_boundary: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None
                 ) -> None:
        self.session = session
        # S1 headroom probe only (``probe.py``): the runner's post-tool boundary report;
        # None in every M-H1 arm (the op is then unknown)
        self.on_boundary = on_boundary
        self.role = role
        self.task_tools = frozenset(task_tools)
        self.surface_guard = surface_guard
        self.expected_materialization = expected_materialization
        self.on_hello = on_hello
        self.lock = threading.Lock()
        self.turn_lock = threading.Lock()  # serialises dialogue ops (never held with ``lock``)
        self.mediation: list[dict[str, Any]] = []
        self.executed: list[dict[str, Any]] = []
        self.unmediated: list[dict[str, Any]] = []
        self._tokens: dict[str, dict[str, Any]] = {}  # call id -> outstanding allow
        self._used_ids: set[str] = set()  # call ids already decided (guard) or consumed
        self.hello: dict[str, Any] | None = None
        self.dialogue: list[tuple[str, dict[str, Any], str]] = []
        self.ended = False  # the customer / task said stop: no task tool runs after this

    # -- observations the surface guards read: dialogue turns and executed calls, in order
    def observations(self) -> list[tuple[str, dict[str, Any], str]]:
        return list(self.dialogue)

    def handle(self, msg: Mapping[str, Any]) -> dict[str, Any]:
        op = msg.get("op")
        if op in ("first_user", "user_turn"):
            return self._turn(op, msg)
        with self.lock:
            if op == "guard":
                return self._guard(msg)
            if op == "tool":
                return self._tool(msg)
            if op == "hello":
                return self._hello(msg)
            if op == "boundary" and self.on_boundary is not None:
                return self.on_boundary(msg)
        return {"error": f"unknown op {op!r}"}

    def _turn(self, op: str, msg: Mapping[str, Any]) -> dict[str, Any]:
        """A dialogue turn. The customer call (possibly a slow HTTP request) runs outside the
        broker lock, so concurrent guard / tool ops are never stalled behind it."""
        with self.turn_lock:
            if op == "first_user":
                text = str(self.session.first_user())
                with self.lock:
                    self.dialogue.append((GD.USER, {"content": text}, ""))
                return {"content": text}
            said = str(msg.get("assistant") or "")
            with self.lock:
                self.dialogue.append((GD.ASSISTANT, {"content": said}, ""))
            stop, text = self.session.user_turn(said)
            with self.lock:
                if stop:
                    self.ended = True
                    self._tokens.clear()  # nothing approved during the dialogue runs after it
                else:
                    self.dialogue.append((GD.USER, {"content": str(text)}, ""))
            return {"stop": bool(stop), "content": str(text or "")}

    def _hello(self, msg: Mapping[str, Any]) -> dict[str, Any]:
        self.hello = dict(msg)
        if self.on_hello is not None:
            self.on_hello(self, msg)  # test hook (fault injection), never in production
        att = msg.get("attestation") or {}
        pre = msg.get("preflight") or {}
        if att.get("materialization_sha256") != self.expected_materialization:
            return {"ok": False, "reason": "materialization hash mismatch"}
        if not att.get("files_ok"):
            return {"ok": False, "reason": f"immutable inputs differ: {att.get('mismatched')}"}
        if not pre.get("ok"):
            return {"ok": False, "reason": "preflight failed"}
        return {"ok": True}

    def _guard(self, msg: Mapping[str, Any]) -> dict[str, Any]:
        tool, args = str(msg.get("tool")), dict(msg.get("args") or {})
        call_id = str(msg.get("call_id") or "")
        origin = str(msg.get("origin") or "unknown")
        allow, reason = guard_host.check(tool, args, role=self.role, task_tools=self.task_tools)
        verdicts: dict[str, Any] = {"host": reason}
        if allow and tool in self.task_tools:
            if origin not in FOREGROUND:
                # Hermes' review whitelist is thread-local and does not reach its tool
                # workers: the fork (or any non-foreground origin) never runs a task tool
                allow, reason = False, f"task tool from origin {origin!r}"
            elif self.ended:
                allow, reason = False, "the dialogue has ended"
            elif not call_id or call_id in self._used_ids:
                allow, reason = False, "missing or reused tool-call id"
            else:
                try:
                    allow, why, more = self.surface_guard(tool, args, self.observations())
                    verdicts.update(more)
                except Exception as e:  # noqa: BLE001 - a failed guard made no decision: block
                    allow, why = False, f"guard error {type(e).__name__}: {e}"
                    reason = why
                verdicts["surface"] = why
            verdicts["host"] = reason if not allow else verdicts["host"]
        rec = {"seq": len(self.mediation), "call_id": call_id, "tool": tool, "args": args,
               "origin": origin, "allow": bool(allow), "verdicts": verdicts,
               "at": (self.session.position() if hasattr(self.session, "position")
                      else len(getattr(self.session, "transcript", []) or []))}
        self.mediation.append(rec)
        if call_id:
            self._used_ids.add(call_id)
        if allow and tool in self.task_tools:
            self._tokens[call_id] = {"tool": tool, "args": _canon(args), "seq": rec["seq"]}
        if allow:
            return {"action": "allow"}
        return {"action": "block", "message": f"Blocked by the harness guard: {reason}"
                if reason else "Blocked by the harness guard."}

    def _tool(self, msg: Mapping[str, Any]) -> dict[str, Any]:
        """Execute only against the allow record of the SAME call id, tool and canonical
        arguments; each allow is consumed once; there is no fallback match."""
        tool, args = str(msg.get("tool")), dict(msg.get("args") or {})
        call_id = str(msg.get("call_id") or "")
        tok = self._tokens.get(call_id) if call_id else None
        if (self.ended or tok is None or tok["tool"] != tool
                or tok["args"] != _canon(args)):
            self.unmediated.append({"tool": tool, "args": args, "call_id": call_id,
                                    "reason": "ended" if self.ended else
                                    "no matching allow for this id, tool and arguments"})
            return {"result": json.dumps({"error": "refused: no guard decision for this "
                                                   "call"})}
        del self._tokens[call_id]
        try:
            result = str(self.session.execute(tool, args))
        except Exception as e:  # noqa: BLE001 - a tool error is a tool result
            result = json.dumps({"error": f"{type(e).__name__}: {e}"})
        self.executed.append({"seq": tok["seq"], "call_id": call_id, "tool": tool,
                              "args": args, "result": result})
        self.dialogue.append((tool, args, result))
        return {"result": result}


class BridgeServer:
    """One Unix socket, one handler thread per connection (Hermes may call concurrently)."""

    def __init__(self, path: Path, broker: Broker) -> None:
        self.path = str(path)
        if len(self.path) > 100:
            raise ValueError(f"socket path too long for AF_UNIX: {self.path}")
        self.broker = broker
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if os.path.exists(self.path):
            os.unlink(self.path)
        self.sock.bind(self.path)
        self.sock.listen(16)
        self._stop = threading.Event()
        self.errors: list[str] = []
        self._thread = threading.Thread(target=self._serve, daemon=True, name="slh-bridge")

    def start(self) -> BridgeServer:
        self._thread.start()
        return self

    def _serve(self) -> None:
        self.sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=self._one, args=(conn,), daemon=True).start()

    def _one(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(60)
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
            try:
                reply = self.broker.handle(json.loads(buf))
            except Exception as e:  # noqa: BLE001 - the plugin turns an error into a block
                self.errors.append(repr(e))
                reply = {"error": repr(e)}
            conn.sendall((json.dumps(reply, default=str) + "\n").encode())
        finally:
            conn.close()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(2)
        self.sock.close()
        if os.path.exists(self.path):
            os.unlink(self.path)
