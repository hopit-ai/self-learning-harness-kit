"""``SpendGate``: a loopback proxy between one Hermes episode and its model endpoint that
makes the agent's spend a hard bound (review round 4, finding 5).

Hermes reaches the model only through the gate (the sandbox allows only the gate's port).
For every chat-completions request the gate, BEFORE forwarding:

* sets ``max_tokens`` (and ``max_completion_tokens``) to at most the per-request output cap;
* counts the request's input with the model's counter (``ChatTemplateCounter`` of the served
  model for a priced model; the byte bound for free development) over messages AND tools;
* refuses the request (HTTP 400, never forwarded) when the run's input or output ceiling
  (``ModelEndpoint.worst_tokens_per_run``) would be passed.

So the episode's billed tokens never exceed the reservation ``worst_cost()``. Settlement per
request: the reported usage when the response completes with usage, else the request's
reserved bound (a lost or cut response is charged its bound).
"""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


class SpendGate:
    def __init__(self, upstream_base_url: str, *, counter: Any, max_input_tokens: int,
                 max_output_tokens: int, max_output_per_request: int,
                 usd_per_mtok: tuple[float, float] = (0.0, 0.0), timeout_s: float = 600.0
                 ) -> None:
        u = urlparse(upstream_base_url)
        if u.hostname not in ("127.0.0.1", "localhost"):
            raise ValueError("the gate forwards to a loopback endpoint only")
        self.up_host, self.up_port = u.hostname, int(u.port or 80)
        self.up_prefix = u.path.rstrip("/")  # e.g. /v1
        self.counter = counter
        self.max_in, self.max_out = int(max_input_tokens), int(max_output_tokens)
        self.max_out_req = int(max_output_per_request)
        if min(self.max_in, self.max_out, self.max_out_req) <= 0:
            raise ValueError("token ceilings must be positive")
        self.price = usd_per_mtok
        self.timeout_s = timeout_s
        self.lock = threading.Lock()
        self.in_reserved = self.out_reserved = 0
        self.requests: list[dict[str, Any]] = []
        self.refused: list[dict[str, Any]] = []
        self.breaches: list[dict[str, Any]] = []
        gate = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                return

            def _reply(self, code: int, body: Mapping[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
                self.close_connection = True

            def do_GET(self) -> None:
                gate._forward(self, "GET", self.path, None, None)

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n)
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    self._reply(404, {"error": {"message": "slh gate: unsupported path"}})
                    return
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    self._reply(400, {"error": {"message": "slh gate: bad json"}})
                    return
                rec = gate._admit(body)
                if rec is None:
                    self._reply(400, {"error": {"message": "slh gate: the run's token "
                                                           "ceiling is reached",
                                                "type": "slh_spend_ceiling"}})
                    return
                gate._forward(self, "POST", self.path, json.dumps(body).encode(), rec)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}{self.up_prefix}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True,
                                        name="slh-spend-gate")

    # ------------------------------------------------------------------ admission

    def _admit(self, body: dict[str, Any]) -> dict[str, Any] | None:
        msgs = body.get("messages") or []
        tools = [t.get("function", t) for t in body.get("tools") or [] if isinstance(t, dict)]
        try:
            n_in = int(self.counter.count(msgs, tools or None))
        except Exception:  # noqa: BLE001 - an uncountable request is refused (fail closed)
            with self.lock:
                self.refused.append({"input_bound": None, "output": None})
            return None
        asked = body.get("max_tokens") or body.get("max_completion_tokens") or self.max_out_req
        out = max(1, min(int(asked), self.max_out_req))
        with self.lock:
            if self.in_reserved + n_in > self.max_in or self.out_reserved + out > self.max_out:
                self.refused.append({"input_bound": n_in, "output": out})
                return None
            self.in_reserved += n_in
            self.out_reserved += out
            rec = {"input_bound": n_in, "output_cap": out, "usage": None, "complete": False}
            self.requests.append(rec)
        body["max_tokens"] = out
        if "max_completion_tokens" in body:
            body["max_completion_tokens"] = out
        return rec

    # ------------------------------------------------------------------ forwarding

    def _forward(self, h: BaseHTTPRequestHandler, method: str, path: str,
                 data: bytes | None, rec: dict[str, Any] | None) -> None:
        conn = http.client.HTTPConnection(self.up_host, self.up_port, timeout=self.timeout_s)
        try:
            headers = {"Content-Type": "application/json"}
            for k in ("Authorization", "Accept"):
                if h.headers.get(k):
                    headers[k] = h.headers[k]
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            h.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                    h.send_header(k, v)
            h.send_header("Connection", "close")
            h.end_headers()
            buf = b""
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                h.wfile.write(chunk)
                h.wfile.flush()
                if rec is not None and len(buf) < 4_000_000:
                    buf += chunk
            h.close_connection = True
            if rec is not None:
                rec["usage"] = _usage(buf)
                rec["complete"] = resp.status == 200
                u = rec["usage"] or {}
                if int(u.get("prompt_tokens") or 0) > rec["input_bound"] or \
                        int(u.get("completion_tokens") or 0) > rec["output_cap"]:
                    with self.lock:  # our bound is wrong: the adapter halts the system
                        self.breaches.append({"usage": u, "input_bound": rec["input_bound"],
                                              "output_cap": rec["output_cap"]})
        except Exception:  # noqa: BLE001 - the client sees a cut response; charged its bound
            h.close_connection = True
        finally:
            conn.close()

    # ------------------------------------------------------------------ accounting

    def charged_tokens(self) -> tuple[int, int]:
        """(input, output) to settle: reported usage of completed responses, the reserved
        bound otherwise (never more than the reservation when the counter is a bound)."""
        i = o = 0
        with self.lock:
            for r in self.requests:
                u = r["usage"]
                if r["complete"] and u and "prompt_tokens" in u and "completion_tokens" in u:
                    i += int(u["prompt_tokens"])
                    o += int(u["completion_tokens"])
                else:
                    i += r["input_bound"]
                    o += r["output_cap"]
        return i, o

    def cost_usd(self) -> float:
        i, o = self.charged_tokens()
        return (i * self.price[0] + o * self.price[1]) / 1e6

    def report(self) -> dict[str, Any]:
        i, o = self.charged_tokens()
        return {"requests": len(self.requests), "refused": len(self.refused),
                "breaches": list(self.breaches),
                "input_reserved": self.in_reserved, "output_reserved": self.out_reserved,
                "input_charged": i, "output_charged": o, "max_input": self.max_in,
                "max_output": self.max_out, "max_output_per_request": self.max_out_req}

    def start(self) -> SpendGate:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        # (v2.6) break gate -> server -> Handler class (closure over the gate) -> gate, so a
        # finished episode's gate is freed by reference counting, not left to the collector
        self._httpd.RequestHandlerClass = None


def _usage(raw: bytes) -> dict[str, Any] | None:
    """Usage from a JSON response or the last SSE chunk that carries it."""
    text = raw.decode("utf-8", "replace")
    try:
        d = json.loads(text)
        return d.get("usage") if isinstance(d, dict) else None
    except json.JSONDecodeError:
        pass
    found = None
    for line in text.splitlines():
        if line.startswith("data:") and "usage" in line:
            try:
                d = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict) and d.get("usage"):
                found = d["usage"]
    return found
