"""A fake OpenAI-compatible chat-completions server on 127.0.0.1 (tests and plumbing smoke).

No model, no network beyond the loopback interface. A *policy* maps one request (the parsed
JSON body) to one assistant turn: ``{"content": str | None, "tool_calls": [{"name", "arguments"}]}``.
Both the plain and the SSE streaming forms of ``/v1/chat/completions`` are served, plus
``/v1/models``. Every request is recorded (``requests``) so tests can check what the host
actually sent (temperature, seed, tools) and how many calls the review fork made.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

Policy = Callable[[Mapping[str, Any]], Mapping[str, Any]]

MODEL_ID = "stub-model"
CONTEXT_LENGTH = 131072


def _message(turn: Mapping[str, Any], n: int) -> dict[str, Any]:
    calls = [{"id": f"call_{n}_{i}", "type": "function",
              "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", {}))}}
             for i, c in enumerate(turn.get("tool_calls") or [])]
    msg: dict[str, Any] = {"role": "assistant", "content": turn.get("content")}
    if calls:
        msg["tool_calls"] = calls
    return msg


class StubModelServer:
    """``with StubModelServer(policy) as s: s.base_url`` (``http://127.0.0.1:<port>/v1``)."""

    def __init__(self, policy: Policy, *, host: str = "127.0.0.1", port: int = 0,
                 prompt_tokens: int = 10, completion_tokens: int = 20) -> None:
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("the stub serves the loopback interface only")
        self.policy = policy
        self.requests: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._lock = threading.Lock()
        self._n = 0
        self.usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens}
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:  # quiet
                return

            def _send(self, code: int, body: Mapping[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path.rstrip("/").endswith("/models"):
                    self._send(200, {"object": "list", "data": [
                        {"id": MODEL_ID, "object": "model", "owned_by": "stub",
                         "context_length": CONTEXT_LENGTH, "max_model_len": CONTEXT_LENGTH}]})
                else:
                    self._send(404, {"error": {"message": "not found"}})

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError:
                    self._send(400, {"error": {"message": "bad json"}})
                    return
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    self._send(404, {"error": {"message": f"unsupported path {self.path}"}})
                    return
                with server._lock:
                    server._n += 1
                    k = server._n
                    server.requests.append(body)
                try:
                    turn = dict(server.policy(body))
                except Exception as e:  # noqa: BLE001 - surfaced to the test, never hidden
                    server.errors.append(repr(e))
                    self._send(500, {"error": {"message": f"stub policy error {e!r}"}})
                    return
                msg = _message(turn, k)
                finish = "tool_calls" if msg.get("tool_calls") else "stop"
                if body.get("stream"):
                    self._stream(k, msg, finish, bool((body.get("stream_options") or {})
                                                     .get("include_usage")))
                else:
                    self._send(200, {"id": f"chatcmpl-{k}", "object": "chat.completion",
                                     "created": int(time.time()), "model": MODEL_ID,
                                     "choices": [{"index": 0, "message": msg,
                                                  "finish_reason": finish}],
                                     "usage": server.usage})

            def _stream(self, k: int, msg: dict[str, Any], finish: str, usage: bool) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                base = {"id": f"chatcmpl-{k}", "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": MODEL_ID}

                def emit(delta: dict[str, Any], fr: str | None = None,
                         extra: Mapping[str, Any] | None = None) -> None:
                    chunk = {**base, "choices": [{"index": 0, "delta": delta,
                                                  "finish_reason": fr}], **(extra or {})}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())

                emit({"role": "assistant", "content": ""})
                if msg.get("content"):
                    emit({"content": msg["content"]})
                for i, c in enumerate(msg.get("tool_calls") or []):
                    emit({"tool_calls": [{"index": i, "id": c["id"], "type": "function",
                                          "function": {"name": c["function"]["name"],
                                                       "arguments": c["function"]["arguments"]}}]})
                emit({}, finish)
                if usage:
                    self.wfile.write(("data: " + json.dumps({**base, "choices": [],
                                                             "usage": server.usage})
                                      + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._thread: threading.Thread | None = None

    def start(self) -> StubModelServer:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True,
                                        name="stub-model")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def __enter__(self) -> StubModelServer:  # noqa: PYI034
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


# ---------------------------------------------------------------------- policy helpers

def last_user_text(body: Mapping[str, Any]) -> str:
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
            return str(c or "")
    return ""


def tool_names(body: Mapping[str, Any]) -> set[str]:
    return {(t.get("function") or {}).get("name") for t in body.get("tools") or []}


def tool_results_since_last_user(body: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user":
            break
        if m.get("role") == "tool":
            out.append(m)
    return list(reversed(out))


def calls_since_last_user(body: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user":
            break
        for c in m.get("tool_calls") or []:
            names.append((c.get("function") or {}).get("name"))
    return list(reversed(names))
