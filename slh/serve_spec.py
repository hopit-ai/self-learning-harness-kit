"""Pinned definition of the self-served agent (stdlib only: copied into the
Modal Sandbox image next to serve_entry.py, and imported by slh/serving.py)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

# ---------------------------------------------------------------- what is served

# Pinned 2026-10-06 (Hugging Face model API and PyPI / Docker Hub, read only).
MODEL_REPO = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"  # main, 2026-03-02
SERVED_MODEL_NAME = "qwen3.5-9b-vllm"  # the price-table key, $0 per token
VLLM_VERSION = "0.30.0"  # released 2026-09-22; 0.31.0 (2026-10-05) is a day old
VLLM_IMAGE = ("vllm/vllm-openai:v0.30.0@sha256:"
              "8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90")
# Ordered GPU preference (surface migration 2026-10-09,
# docs/surface-migration-2026-10-09-gpu.md). Modal 1.5.3 refuses a list for
# Sandbox.create(gpu=...) ("Sandboxes do not support configuring a list of
# GPUs"), so slh/serving.py requests one GPU type per sandbox: the first; and,
# only if that sandbox was never scheduled (no container start and empty logs
# GPU_SCHEDULE_WAIT_S after creation), one fallback sandbox on the next, in the
# same window. A window's reservation is priced at the most expensive entry.
GPU_PREFERENCE = ("L40S", "A100-40GB")
GPU = GPU_PREFERENCE[0]  # the primary GPU type
# Every successful start so far began its container 3-82 s after creation
# (12 serve windows, 2026-10-06..08); the two never-scheduled starts (D4, D6)
# never began one. 300 s is the wait before a start counts as unscheduled.
GPU_SCHEDULE_WAIT_S = 300
# (request, limit) tuples: Modal throttles CPU at the limit and enforces the
# memory limit, so the limits are the billed maxima the reservation prices.
CPU_CORES = (4.0, 6.0)
MEMORY_MIB = (32768, 40960)
PREFETCH_CPU_CORES = (2.0, 2.0)
PREFETCH_MEMORY_MIB = (8192, 8192)
PROBE_CPU_CORES = (0.25, 0.25)
PROBE_MEMORY_MIB = (512, 512)
MAX_MODEL_LEN = 131072
MAX_NUM_SEQS = 8
GPU_MEMORY_UTILIZATION = 0.92

# Networking inside the sandbox: vLLM listens on loopback only; the proxy is
# the only tunnelled port and requires the bearer token on every request.
VLLM_PORT = 8001
PROXY_PORT = 8000
ALLOWED_ROUTES = frozenset({("POST", "/v1/chat/completions"), ("GET", "/v1/models")})
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024
MAX_CONNECTIONS = 32

MAX_WINDOW_S = 10800  # 3 hours: the owner's hard stop
PREFETCH_TIMEOUT_S = 1800
PROBE_TIMEOUT_S = 120
START_REFUSE_MARGIN_S = 300  # the entrypoint does not start vLLM this close to the deadline
IN_SANDBOX_STOP_BEFORE_S = 60  # the entrypoint exits this long before the deadline
SANDBOX_APP_NAME = "slh-serve-sandboxes"  # holds sandboxes only: no functions, no endpoints
# Dedicated Modal Environment with server-side limits set by the owner
# (Maximum Concurrent Containers 3, Maximum Concurrent GPUs 1).
MODAL_ENVIRONMENT = "self-learning-harness"


# ---------------------------------------------------------------- vLLM command line


def vllm_args(model_path: str, served_name: str = SERVED_MODEL_NAME) -> list[str]:
    """The exact ``vllm serve`` argv (flags checked against vLLM 0.30.0's
    ``vllm/engine/arg_utils.py`` and ``vllm/entrypoints/launchers/cli_args.py``,
    and again inside the image by ``serve prefetch``).

    - ``--host 127.0.0.1``: vLLM is reachable only through the in-sandbox proxy;
    - ``--tool-call-parser qwen3_coder``: Qwen3.5's chat template emits the
      Qwen3-Coder XML tool format (``<tool_call><function=..><parameter=..>``);
    - ``--reasoning-parser qwen3``: any thinking text goes to
      ``reasoning_content``, as on OpenRouter;
    - ``--default-chat-template-kwargs``: non-thinking by default; requests
      also send ``chat_template_kwargs: {enable_thinking: false}``;
    - ``--generation-config vllm``: no sampling defaults from the model repo;
    - ``--language-model-only``: text only; the vision tower is not loaded;
    - capture: ``--return-tokens-as-token-ids --logprobs-mode processed_logprobs``.
    vLLM also gets the token as ``VLLM_API_KEY`` (environment, never argv).
    """
    return [
        "vllm", "serve", model_path,
        "--served-model-name", served_name,
        "--dtype", "bfloat16",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--language-model-only",
        "--generation-config", "vllm",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
        "--default-chat-template-kwargs", json.dumps({"enable_thinking": False}),
        "--return-tokens-as-token-ids",
        "--logprobs-mode", "processed_logprobs",
        "--host", "127.0.0.1", "--port", str(VLLM_PORT),
        "--uvicorn-log-level", "warning",
    ]


def serving_spec() -> dict[str, Any]:
    """Everything that defines what is served (recorded per window)."""
    return {"model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
            "served_model_name": SERVED_MODEL_NAME, "dtype": "bfloat16",
            "vllm_version": VLLM_VERSION, "image": VLLM_IMAGE,
            "gpu": list(GPU_PREFERENCE), "gpu_schedule_wait_s": GPU_SCHEDULE_WAIT_S,
            "cpu_cores": list(CPU_CORES), "memory_mib": list(MEMORY_MIB),
            "max_model_len": MAX_MODEL_LEN, "max_num_seqs": MAX_NUM_SEQS,
            "vllm_argv": vllm_args("<snapshot>"), "proxy_port": PROXY_PORT,
            "allowed_routes": sorted(" ".join(r) for r in ALLOWED_ROUTES),
            "mechanism": "modal.Sandbox", "region": None, "nonpreemptible": False}


def spec_sha256() -> str:
    return hashlib.sha256(json.dumps(serving_spec(), sort_keys=True).encode()).hexdigest()
