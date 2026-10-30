# Plain runbook: environment and servers (plan mh1-2026-10-25-v2.6)

Code snapshot `32e53223b6cdc86dba4c3c0298811db60cf19e6f`. This kit runs no phase 1 and ships no phase-1
package: the release's runbook (see README.md) names the packages and commands to run.
Sections 1 and 2 below are the environment and the server settings it refers to.

## 1. Setup (Python 3.12, no network on the node)

Copy the kit at its tag to the node (for example the tag's source archive, fetched on any
machine with network). Nothing below opens a connection: the pinned benchmark and every
wheel come from the release assets (`tau2-bench-<commit>.tar.gz`,
`wheelhouse-linux-x86_64-py312.tar.gz`, `offline-assets.json`, in one directory):

```bash
python3.12 -m venv .venv
PYTHON=.venv/bin/python scripts/xp/setup_offline.sh <assets dir>        # ends "ok": true
```

To keep an existing environment (the one you ran phase 1 with) instead, use its Python
and add `--reuse-env`: only the pinned benchmark is added and the check then decides
(`PYTHON=/path/to/phase1-env/bin/python scripts/xp/setup_offline.sh <assets dir> --reuse-env`).
`python -m slh.xp.partner.offline check` repeats the check at any time (every network
connection is refused while it runs).

Run everything from this directory with `.venv/bin/python` (`python` below).

## 2. Start the two servers yourself (vLLM 0.30.0, one 8-GPU node)

| Setting | 9B agent | teacher |
|---|---|---|
| `model` | `Qwen/Qwen3.5-9B` @ `c20223623576` | `Qwen/Qwen3.8-Flash-Next-FP8` @ `236dfdf28582` |
| `served name` | `qwen3.5-9b-vllm` | `qwen3.8-flash-next-fp8` |
| `port` | 8001 | 8002 |
| `tensor x data parallel` | 4 x 2 | 8 x 1, expert parallel |
| `memory share per GPU` | 0.16 | 0.68 |
| `max model length` | 131072 | 131072 |
| `mode` | pinned arguments (`slh/serve_spec.py`) | CUDA graphs (eager only if graphs fail) |
| `extra` | - | `--language-model-only --engram-config={"cpu_offload":false}` |

Both on the same 8 GPUs; start the teacher first:

```bash
vllm serve /path/to/models/teacher --served-model-name qwen3.8-flash-next-fp8 --tensor-parallel-size 8 --max-model-len 131072 --gpu-memory-utilization 0.68 --generation-config vllm --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' --host 127.0.0.1 --port 8002 --uvicorn-log-level warning --enable-prefix-caching --enable-chunked-prefill --enable-expert-parallel --language-model-only '--engram-config={"cpu_offload":false}'
vllm serve /path/to/models/Qwen3.5-9B --served-model-name qwen3.5-9b-vllm --dtype bfloat16 --max-model-len 131072 --max-num-seqs 32 --gpu-memory-utilization 0.16 --language-model-only --generation-config vllm --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' --return-tokens-as-token-ids --logprobs-mode processed_logprobs --host 127.0.0.1 --port 8001 --uvicorn-log-level warning --tensor-parallel-size 4 --data-parallel-size 2 --enable-prefix-caching --enable-chunked-prefill
```

Replace the model paths with your downloads. If CUDA graphs fail on your node, add
`--enforce-eager` to the teacher only and tell us (it is about 25 times slower). A token
(`VLLM_API_KEY` for the servers) is optional: name its variable in `api_key_env`.
Then write `endpoint.yaml` in the kit directory (JSON is valid YAML; leave the template
`configs/xp-partner-endpoint.yaml` unchanged: it is registered code). `serving_args`
and `execution.teacher_argv` are the exact argv you started (recorded in the run
identity); keep them and every server setting fixed for the whole run:

```json
{
 "base_url": "http://127.0.0.1:8001/v1",
 "model": "qwen3.5-9b-vllm",
 "api_key_env": null,
 "model_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
 "vllm_version": "0.30.0",
 "serving_args": ["vllm", "serve", "/path/to/models/Qwen3.5-9B", "--served-model-name", "qwen3.5-9b-vllm", "--dtype", "bfloat16", "--max-model-len", "131072", "--max-num-seqs", "32", "--gpu-memory-utilization", "0.16", "--language-model-only", "--generation-config", "vllm", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3", "--default-chat-template-kwargs", "{\"enable_thinking\": false}", "--return-tokens-as-token-ids", "--logprobs-mode", "processed_logprobs", "--host", "127.0.0.1", "--port", "8001", "--uvicorn-log-level", "warning", "--tensor-parallel-size", "4", "--data-parallel-size", "2", "--enable-prefix-caching", "--enable-chunked-prefill"],
 "replicas": 2,
 "concurrency": 32,
 "learning_units": 12,
 "teacher": {
  "mode": "self_hosted",
  "base_url": "http://127.0.0.1:8002/v1",
  "model": "qwen3.8-flash-next-fp8",
  "api_key_env": null,
  "extra": {
   "chat_template_kwargs": {
    "enable_thinking": false
   }
  },
  "weights": "Qwen/Qwen3.8-Flash-Next-FP8@236dfdf285828023ca3bcd3f37366c58a3469b13"
 },
 "execution": {
  "teacher_argv": ["vllm", "serve", "/path/to/models/teacher", "--served-model-name", "qwen3.8-flash-next-fp8", "--tensor-parallel-size", "8", "--max-model-len", "131072", "--gpu-memory-utilization", "0.68", "--generation-config", "vllm", "--reasoning-parser", "qwen3", "--default-chat-template-kwargs", "{\"enable_thinking\": false}", "--host", "127.0.0.1", "--port", "8002", "--uvicorn-log-level", "warning", "--enable-prefix-caching", "--enable-chunked-prefill", "--enable-expert-parallel", "--language-model-only", "--engram-config={\"cpu_offload\":false}"],
  "teacher_extra_args": ["--language-model-only", "--engram-config={\"cpu_offload\":false}"]
 }
}
```

