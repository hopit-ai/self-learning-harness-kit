#!/usr/bin/env python3
"""Fake OpenAI-compatible vLLM server for rehearsals (standard library only).

    fake_vllm.py serve MODEL --served-model-name NAME --port P [other vllm flags]

Waits SIM_VLLM_STARTUP_S (default 1) before listening (startup-time probes),
answers GET /health, /version, /v1/models and POST /v1/chat/completions with a
short clarifying message (no tool call). Requires the bearer token in
VLLM_API_KEY when set. SIM_VLLM_DIE_AFTER_S makes it exit (crash rehearsal);
SIM_VLLM_STATUS answers every chat request with that HTTP status.
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]
if argv[:1] == ["serve"]:
    argv = argv[1:]


def flag(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


NAME = flag("--served-model-name", argv[0] if argv else "model")
PORT = int(flag("--port", "8000"))
KEY = os.environ.get("VLLM_API_KEY")
REPLY = os.environ.get("SIM_VLLM_REPLY",
                       "Could you please confirm which order and which item you mean?")


COUNTERS = {"prompt": 0, "generation": 0, "requests": 0, "running": 0}
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[fake-vllm {NAME}] {fmt % args}\n")

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _auth(self):
        if KEY and self.headers.get("Authorization") != "Bearer " + KEY:
            self._send(401, {"error": {"message": "Unauthorized"}})
            return False
        return True

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {})
        elif self.path == "/metrics":
            with LOCK:
                c = dict(COUNTERS)
            text = ("# vLLM-style metrics (fake)\n"
                    f'vllm:prompt_tokens_total{{model_name="{NAME}"}} {c["prompt"]}\n'
                    f'vllm:generation_tokens_total{{model_name="{NAME}"}} {c["generation"]}\n'
                    f'vllm:request_success_total{{finished_reason="stop"}} {c["requests"]}\n'
                    f'vllm:num_requests_running{{model_name="{NAME}"}} {c["running"]}\n'
                    f'vllm:prefix_cache_queries_total{{model_name="{NAME}"}} {c["prompt"]}\n'
                    f'vllm:prefix_cache_hits_total{{model_name="{NAME}"}} {c["prompt"] // 2}\n')
            data = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/version":
            self._send(200, {"version": "0.30.0"})
        elif self.path == "/v1/models":
            if self._auth():
                self._send(200, {"object": "list", "data": [{"id": NAME, "object": "model"}]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": {"message": "not found"}})
            return
        if not self._auth():
            return
        forced = os.environ.get("SIM_VLLM_STATUS")  # e.g. 401: a halt-class error
        if forced:
            self._send(int(forced), {"error": {"message": "forced by SIM_VLLM_STATUS"}})
            return
        if body.get("model") != NAME:
            self._send(404, {"error": {"message": "model {!r} not served".format(body.get("model"))}})
            return
        prompt = sum(len(str(m.get("content") or "")) for m in body.get("messages", [])) // 4
        with LOCK:
            COUNTERS["prompt"] += max(1, prompt)
            COUNTERS["generation"] += 14
            COUNTERS["requests"] += 1
        self._send(200, {
            "id": "chatcmpl-sim", "object": "chat.completion", "created": int(time.time()),
            "model": NAME,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": REPLY,
                                     "tool_calls": []}}],
            "usage": {"prompt_tokens": max(1, prompt), "completion_tokens": 14,
                      "total_tokens": max(1, prompt) + 14}})


def main():
    time.sleep(float(os.environ.get("SIM_VLLM_STARTUP_S", "1")))
    die = os.environ.get("SIM_VLLM_DIE_AFTER_S")
    if die:
        threading.Timer(float(die), lambda: os._exit(1)).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
