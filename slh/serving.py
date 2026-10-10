"""Self-served agent inference on Modal Sandboxes: bounded window, metered spend.

``python -m slh serve probe-timeout|prefetch|plan|start|stop|status|owner-clear``
(``watchdog`` is internal). docs/self-served-agent.md explains every layer and
the residual risks.

One *session directory* (``serving/<name>/``, git-ignored) holds
``session.json`` (learner-safe: every window, no token), ``private/`` (the
token, mode 0600, and redacted launch tracebacks), ``ledger.jsonl`` (the
session's spend ledger, cumulative under ``--cap``) and the watchdog logs.
One project-wide lock (``PROJECT_LOCK``) serialises capacity check,
admission, image preparation and creation across every session directory on
this machine.

Every window is one ``modal.Sandbox`` in the app ``slh-serve-sandboxes``
(sandboxes only: no functions, web endpoints or schedules):

- ``probe``: a tiny CPU sandbox that sleeps with ``timeout=120``. It passes
  only if Modal ended it *by timeout* (result status TIMEOUT) and Modal's
  created->finished lifetime is within [timeout-10, timeout+30] s, i.e. the
  timeout counts from creation. Required before any GPU window.
- ``prefetch``: a CPU sandbox that caches the pinned weights and checks the
  vLLM version, flags, parsers and the non-thinking template. Required (same
  spec) before any GPU window.
- ``serve``: the GPU sandbox. Its timeout is ``deadline - now`` computed
  immediately before the create request (after the image is built and every
  dependency hydrated), so it is anchored to the approved deadline. The
  entrypoint gets the deadline too (``slh/serve_entry.py``).

Window state machine (never backwards):
``launching -> serving -> stopping -> stopped -> verified``. Creation happens
under the session lock and only from ``launching``. ``verified`` is written in
the same locked update as the window's settlement, and only after an
authoritative per-id check (``poll`` says finished, the id is absent from the
running list). Admission requires every window verified and settled, no open
discrepancy and no uncleared excess.

Money: per-second rates; a window reserves rate x window + its image-build
allowance before anything is built or created. Settlement charges rate x
Modal's created->finished lifetime (+ the allowance); without both
timestamps it charges the full reservation. Never clamped: an excess is
recorded and blocks starts until ``serve owner-clear``.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import math
import os
import re
import secrets
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slh.budget import BudgetExceeded
from slh.ledger import SharedLedger, read_records
from slh.serve_spec import (
    ALLOWED_ROUTES,
    CPU_CORES,
    GPU_PREFERENCE,
    GPU_SCHEDULE_WAIT_S,
    IN_SANDBOX_STOP_BEFORE_S,
    MAX_WINDOW_S,
    MEMORY_MIB,
    MODAL_ENVIRONMENT,
    PREFETCH_CPU_CORES,
    PREFETCH_MEMORY_MIB,
    PREFETCH_TIMEOUT_S,
    PROBE_CPU_CORES,
    PROBE_MEMORY_MIB,
    PROBE_TIMEOUT_S,
    PROXY_PORT,
    SANDBOX_APP_NAME,
    SERVED_MODEL_NAME,
    START_REFUSE_MARGIN_S,
    VLLM_IMAGE,
    serving_spec,
    spec_sha256,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_LOCK = REPO_ROOT / "serving" / ".project.lock"  # replaced in tests
FORMAT = "slh-serving.v3"

DEFAULT_STOP_MARGIN_S = 300
DEFAULT_REQUEST_MARGIN_S = 120
DEFAULT_EPISODE_MARGIN_S = 480
DEFAULT_STARTUP_TIMEOUT_S = 1200
DEFAULT_IDLE_S = 900
MAX_ENV_GPUS = 1  # the environment's server-side limits we require
MAX_ENV_CONTAINERS = 3
VERIFY_STOP_S = 120.0
WATCHDOG_POLL_S = 30.0
WATCHDOG_GIVE_UP_S = 3600.0
CONTAINER_CHECK_S = 300.0
LAUNCH_GRACE_S = 120.0
TUNNEL_POLL_S = 15  # one tunnel lookup; repeated until the startup deadline
TUNNEL_RETRY_PAUSE_S = 2.0
MAX_GPU_FALLBACKS = 1  # at most one fallback sandbox per window
MEASURE_GRACE_S = 300.0  # measurement mode: watchdog backstop after the deadline
MEASURE_ENTRY_BACKSTOP_S = 120  # measurement mode: in-sandbox exit after the deadline
MEASURE_MAX_WINDOW_S = 1800
MEASURE_MAX_CAP_USD = 2.0
MEASURE_EARLY_TOLERANCE_S = 10.0
MEASURE_LATE_TOLERANCE_S = 60.0
PROBE_EARLY_TOLERANCE_S = 10.0
PROBE_LATE_TOLERANCE_S = 30.0
PROBE_WAIT_EXTRA_S = 240.0
TIMEOUT_STATUS = "GENERIC_STATUS_TIMEOUT"
TIMEOUT_RETURNCODE = 124  # what the SDK reports for a sandbox ended by its timeout

TRANSITIONS = {"launching": {"serving", "stopping"}, "serving": {"stopping"},
               "stopping": {"stopped"}, "stopped": {"verified"}, "verified": set()}
LIVE = ("launching", "serving")

PROBE_ROUTES = (
    ("GET", "/v1/models"), ("POST", "/v1/chat/completions"), ("POST", "/v1/completions"),
    ("POST", "/invocations"), ("GET", "/metrics"), ("GET", "/docs"),
    ("GET", "/openapi.json"), ("GET", "/health"), ("GET", "/version"),
    ("POST", "/tokenize"), ("POST", "/detokenize"), ("POST", "/v1/responses"),
    ("POST", "/v1/embeddings"), ("GET", "/server_info"), ("POST", "/reset_prefix_cache"),
    ("POST", "/sleep"), ("POST", "/wake_up"), ("POST", "/collective_rpc"),
    ("POST", "/scale_elastic_ep"), ("POST", "/v1/load_lora_adapter"),
    ("DELETE", "/v1/models"), ("GET", "/v1/chat/completions"), ("PATCH", "/v1/models"),
)

_now = time.time  # replaced in tests
_sleep = time.sleep


def _iso(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch, dt.UTC).isoformat()


def _ceil6(value: float) -> float:
    """Round money *up* to the micro-dollar, so a reservation is never below
    the exact per-second product."""
    return math.ceil(value * 1_000_000 - 1e-6) / 1_000_000


class ServingError(RuntimeError):
    """A serve command refused or failed (message says why)."""


class TunnelNotReady(ServingError):
    """The tunnel lookup reached its deadline (the message is unchanged)."""


class NotOurSandbox(ServingError):
    """A sandbox id that our app in our environment does not list."""


class InvalidTransition(ServingError):
    """A window state change that the state machine forbids."""


class ServingWindowClosed(BudgetExceeded):
    """No agent request may be sent: the window's request deadline passed, or
    it is not serving. A ``BudgetExceeded`` so every meter path refuses; the
    runner leaves the episode unwritten so the run resumes in a later window."""


# ---------------------------------------------------------------- redaction

_SECRET_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password|authorization)(\s*[=:]\s*)\S+"),
    re.compile(r"\b(sk|ak|as|tk)-[A-Za-z0-9_-]{16,}\b"),
)


def redact(text: str, *known: str | None) -> str:
    """Remove known secrets and anything secret-shaped."""
    out = str(text)
    for value in known:
        if value and len(value) >= 8:
            out = out.replace(value, "[REDACTED]")
    out = _SECRET_PATTERNS[0].sub("Bearer [REDACTED]", out)
    out = _SECRET_PATTERNS[1].sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", out)
    return _SECRET_PATTERNS[2].sub("[REDACTED]", out)


# ---------------------------------------------------------------- rates


@dataclass(frozen=True)
class Rate:
    """USD per second of one sandbox at its billed maximum (resource limits)."""

    gpu: str
    gpu_per_s: float
    cpu_core_per_s: float
    memory_gib_per_s: float
    cpu_cores: float
    memory_gib: float
    source: str
    priced_gpu: str | None = None  # the most expensive allowed GPU type (serve)

    @property
    def per_second(self) -> float:
        return (self.gpu_per_s + self.cpu_core_per_s * self.cpu_cores
                + self.memory_gib_per_s * self.memory_gib)

    @property
    def per_hour(self) -> float:
        return self.per_second * 3600.0

    def usd(self, seconds: float) -> float:
        return _ceil6(self.per_second * max(0.0, seconds))

    def as_dict(self) -> dict[str, Any]:
        return {"gpu": self.gpu, "gpu_per_s": self.gpu_per_s,
                "cpu_core_per_s": self.cpu_core_per_s, "cpu_cores": self.cpu_cores,
                "memory_gib_per_s": self.memory_gib_per_s, "memory_gib": self.memory_gib,
                "per_second": self.per_second, "per_hour": round(self.per_hour, 6),
                "source": self.source, "priced_gpu": self.priced_gpu}


def _modal_prices(prices_path: str | Path) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(Path(prices_path).read_text(encoding="utf-8")) or {}
    modal = raw.get("modal")
    if not isinstance(modal, dict):
        raise ServingError(f"{prices_path}: no 'modal' price section")
    if modal.get("region_multiplier_applied") is not False or \
            modal.get("nonpreemptible_multiplier_applied") is not False:
        raise ServingError(f"{prices_path}: the modal rates must state that no region "
                           "selection and no non-preemptible multiplier apply")
    return modal


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not (
            math.isfinite(value) and value >= 0):
        raise ServingError(f"modal {name} must be a finite nonnegative number")
    return float(value)


def _per_second(per_s: Any, per_h: Any, name: str) -> float:
    """The published per-second figure, never below the hourly figure / 3600."""
    return max(_number(per_s, f"{name} per second"), _number(per_h, f"{name} per hour") / 3600)


def load_rate(prices_path: str | Path, kind: str = "serve") -> Rate:
    """Billed maximum per second of one ``kind`` sandbox (resource limits)."""
    modal = _modal_prices(prices_path)
    per_s = modal.get("per_second") or {}
    cpu = _per_second(per_s.get("cpu_core"), modal.get("cpu_core_per_hour"), "cpu_core")
    mem = _per_second(per_s.get("memory_gib"), modal.get("memory_gib_per_hour"), "memory_gib")
    source = str(modal.get("source", ""))
    if kind == "serve":
        # Priced at the most expensive GPU type the window may run on: any
        # allowed type (primary or fallback) fits the reservation.
        gpus_h, gpus_s = modal.get("gpu_per_hour") or {}, per_s.get("gpu") or {}
        # The primary GPU needs both published figures, as before. A fallback
        # GPU with only an hourly figure (A100-40GB: $2.10/h, no per-second
        # entry in prices.yaml) is billed at hourly / 3600, which is the floor
        # max(per-second, hourly / 3600) applies anyway. prices.yaml is not
        # edited: its hash is part of every config's resolved identity.
        rates = {}
        for gpu in GPU_PREFERENCE:
            primary = gpu == GPU_PREFERENCE[0]
            if gpu not in gpus_h or (primary and gpu not in gpus_s):
                raise ServingError(f"{prices_path}: no price for GPU {gpu!r}; refusing")
            rates[gpu] = _per_second(gpus_s.get(gpu, 0.0), gpus_h[gpu], f"gpu {gpu}")
        priced = max(GPU_PREFERENCE, key=lambda g: rates[g])
        return Rate("|".join(GPU_PREFERENCE), rates[priced], cpu, mem,
                    CPU_CORES[1], MEMORY_MIB[1] / 1024, source, priced)
    cores, memory = {"prefetch": (PREFETCH_CPU_CORES, PREFETCH_MEMORY_MIB),
                     "probe": (PROBE_CPU_CORES, PROBE_MEMORY_MIB)}[kind]
    return Rate("none", 0.0, cpu, mem, cores[1], memory[1] / 1024, source)


def build_allowance_usd(prices_path: str | Path) -> float:
    """Reserved and charged with every window that prepares an image."""
    return _number(_modal_prices(prices_path).get("image_build_allowance_usd"),
                   "image_build_allowance_usd")


# ---------------------------------------------------------------- window plan


def check_window_s(window_s: int, allow_over_3h: bool = False) -> int:
    if isinstance(window_s, bool) or not isinstance(window_s, int) or window_s < 600:
        raise ServingError("window must be a whole number of seconds, at least 10 minutes")
    if window_s > MAX_WINDOW_S and not allow_over_3h:
        raise ServingError(f"window {window_s}s exceeds the {MAX_WINDOW_S}s (3 h) hard stop; "
                           "pass --allow-over-3h only with the owner's explicit approval")
    if window_s > 86400:
        raise ServingError("window exceeds Modal's 24 h sandbox lifetime")
    return window_s


def plan_window(window_s: int, rate: Rate, now: float, *, kind: str = "serve",
                allow_over_3h: bool = False, headroom_usd: float = 0.0,
                fixed_charge_usd: float = 0.0, measure_timeout: bool = False,
                stop_margin_s: int = DEFAULT_STOP_MARGIN_S,
                request_margin_s: int = DEFAULT_REQUEST_MARGIN_S,
                episode_margin_s: int = DEFAULT_EPISODE_MARGIN_S,
                startup_timeout_s: int | None = None,
                idle_s: int = DEFAULT_IDLE_S) -> dict[str, Any]:
    """Deadlines and reservation. ``now`` is the admission time; the sandbox
    timeout is computed later, immediately before the create request, as
    ``deadline - now`` (never more than the window)."""
    if kind == "serve":
        check_window_s(window_s, allow_over_3h)
    if startup_timeout_s is None:
        startup_timeout_s = min(DEFAULT_STARTUP_TIMEOUT_S, window_s // 3)
    margins = {"stop_margin_s": stop_margin_s, "request_margin_s": request_margin_s,
               "episode_margin_s": episode_margin_s, "startup_timeout_s": startup_timeout_s,
               "idle_s": idle_s}
    for name, value in margins.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ServingError(f"{name} must be a nonnegative whole number of seconds")
    min_remaining = (startup_timeout_s + START_REFUSE_MARGIN_S + 60 if kind == "serve"
                     else window_s - 30)
    if kind == "serve" and (stop_margin_s + request_margin_s + episode_margin_s
                            + startup_timeout_s >= window_s or min_remaining >= window_s):
        raise ServingError("startup timeout plus the stop, request and episode margins leave "
                           "no time to serve in this window")
    deadline = now + window_s
    request_deadline = deadline - stop_margin_s - request_margin_s
    # In timeout-measurement mode nothing of ours ends the sandbox: the
    # watchdog fires only MEASURE_GRACE_S *after* the deadline, as a backstop.
    watchdog = deadline + MEASURE_GRACE_S if measure_timeout else deadline - stop_margin_s
    return {
        "kind": kind, "window_s": window_s, "admitted_epoch": now,
        "deadline_epoch": deadline, "deadline_utc": _iso(deadline),
        "watchdog_fire_epoch": watchdog, "watchdog_fire_utc": _iso(watchdog),
        "request_deadline_epoch": request_deadline,
        "request_deadline_utc": _iso(request_deadline),
        "episode_start_deadline_epoch": request_deadline - episode_margin_s,
        "episode_start_deadline_utc": _iso(request_deadline - episode_margin_s),
        "startup_deadline_epoch": now + startup_timeout_s,
        "min_create_remaining_s": min_remaining,
        "measure_timeout": measure_timeout,
        "entry_exit_offset_s": (MEASURE_ENTRY_BACKSTOP_S if measure_timeout
                                else -IN_SANDBOX_STOP_BEFORE_S),
        **margins, "rate": rate.as_dict(),
        "reservation_usd": _ceil6(rate.usd(window_s) + headroom_usd + fixed_charge_usd),
        "headroom_usd": headroom_usd, "fixed_charge_usd": fixed_charge_usd,
    }


def _lifetime(timing: dict[str, Any] | None) -> float | None:
    """Modal's created->finished lifetime, or None if either is missing."""
    timing = timing or {}
    created, finished = timing.get("created_at"), timing.get("finished_at")
    if (isinstance(created, int | float) and isinstance(finished, int | float)
            and finished >= created > 0):
        return finished - created
    return None


def settlement(window: dict[str, Any], timing: dict[str, Any] | None) -> dict[str, Any]:
    """Charge for a window whose sandbox is verified finished.

    - never created (known: creation happens only under the lock): the fixed
      charge only;
    - Modal's ``created_at`` and ``finished_at`` both present: rate x
      (finished - created) + the fixed charge (``started_at`` is never used);
    - otherwise: the full reservation (never a local clock difference).
    A window with a GPU fallback adds each earlier (unscheduled) sandbox's
    created->finished lifetime, recorded when it was verified stopped, at the
    same rate (the most expensive allowed GPU); if any is missing, the full
    reservation. Never clamped; ``excess_usd`` is anything above the reservation."""
    fixed = window.get("fixed_charge_usd", 0.0)
    timing = timing or {}
    per_s = window["rate"]["per_second"]
    earlier = [_lifetime(a.get("timing")) for a in window.get("gpu_attempts") or []]
    current = _lifetime(timing) if window.get("create_called") else 0.0
    if not window.get("create_called") and not earlier:
        lifetime, basis, charge = 0.0, "never_created", _ceil6(fixed)
    elif current is not None and all(x is not None for x in earlier):
        lifetime = current + sum(earlier)
        basis = "modal_created_to_finished" + ("_with_fallback" if earlier else "")
        charge = _ceil6(per_s * lifetime + fixed)
    else:
        lifetime, basis, charge = None, "no_timestamps_full_reservation", window[
            "reservation_usd"]
    timeout = window.get("create_timeout_s")
    return {"lifetime_s": None if lifetime is None else round(lifetime, 3), "basis": basis,
            "charge_usd": charge,
            "excess_usd": _ceil6(max(0.0, charge - window["reservation_usd"])),
            "lifetime_over_timeout_s": (round(max(0.0, current - timeout), 3)
                                        if lifetime is not None and current and timeout
                                        else None),
            "earlier_attempt_lifetimes_s": [None if x is None else round(x, 3)
                                            for x in earlier] or None,
            "modal_timing": timing or None}


# ---------------------------------------------------------------- the sandbox request


def sandbox_request(window: dict[str, Any], timeout_s: int, issued_at: float) -> dict[str, Any]:
    """The exact ``modal.Sandbox.create`` arguments, minus the prepared image,
    volume and secret objects. No ``region``, ``cloud`` or non-preemptible
    option, so no price multiplier applies."""
    kind = window["kind"]
    gpu = window.get("gpu_requested") or GPU_PREFERENCE[0]
    if kind == "serve" and gpu not in GPU_PREFERENCE:
        raise ServingError(f"GPU {gpu!r} is not in the pinned preference {GPU_PREFERENCE}")
    tags = {"slh_session": window["session"], "slh_window": window["window_id"],
            "slh_kind": kind}
    entry = ["python3", "/opt/slh_serve/serve_entry.py"]
    if kind == "serve":
        command = [*entry, "serve", "--deadline", repr(window["deadline_epoch"]),
                   "--issued-at", repr(issued_at), "--budget-s", str(timeout_s),
                   "--served-name", SERVED_MODEL_NAME, "--idle-s", str(window["idle_s"])]
        measuring = bool(window.get("measure_timeout"))
        if measuring:
            # No idle exit; the in-sandbox exit becomes a backstop *after* the
            # deadline, so Modal's timeout should end the sandbox first.
            command[command.index("--idle-s") + 1] = str(10 * MAX_WINDOW_S)
            command += ["--exit-offset-s", str(MEASURE_ENTRY_BACKSTOP_S)]
        return {"command": command, "timeout": timeout_s,
                "idle_timeout": None if measuring else window["idle_s"],
                "gpu": gpu, "cpu": CPU_CORES, "memory": MEMORY_MIB,
                "encrypted_ports": [PROXY_PORT], "tags": tags}
    if kind == "prefetch":
        return {"command": [*entry, "prefetch"], "timeout": timeout_s, "idle_timeout": None,
                "gpu": None, "cpu": PREFETCH_CPU_CORES, "memory": PREFETCH_MEMORY_MIB,
                "encrypted_ports": [], "tags": tags}
    return {"command": [*entry, "sleep"], "timeout": timeout_s, "idle_timeout": None,
            "gpu": None, "cpu": PROBE_CPU_CORES, "memory": PROBE_MEMORY_MIB,
            "encrypted_ports": [], "tags": tags}


async def scoped_sandbox_async(client: Any, app_id: str, sandbox_id: str, environment: str,
                               pages: int = 50) -> dict[str, Any] | None:
    """Find one sandbox in *our* app in *our* environment (``SandboxList`` with
    ``app_id`` and ``environment_name``, finished sandboxes included). Returns
    its timestamps, or None if it is not ours. This is the membership check
    that every by-ID operation needs: the SDK's by-ID requests
    (``SandboxWaitRequest``, ``SandboxTerminateRequest``, tunnels) carry no
    environment, so an arbitrary id could otherwise reach another environment."""
    from modal_proto import api_pb2

    before = None
    for _ in range(pages):
        response = await client.stub.SandboxList(api_pb2.SandboxListRequest(
            app_id=app_id, before_timestamp=before, include_finished=True,
            environment_name=environment))
        if not response.sandboxes:
            return None
        for info in response.sandboxes:
            if info.id == sandbox_id:
                task = info.task_info
                return {"app_id": app_id, "environment": environment,
                        "created_at": info.created_at or None,
                        "started_at": getattr(task, "started_at", None) or None,
                        "finished_at": getattr(task, "finished_at", None) or None}
        before = response.sandboxes[-1].created_at
    return None


timing_async = scoped_sandbox_async  # the same scoped lookup gives the timestamps


async def environment_limits_async(client: Any, environment: str) -> dict[str, Any] | None:
    """The environment's server-side limits and current usage (read only)."""
    from google.protobuf import empty_pb2

    response = await client.stub.EnvironmentList(empty_pb2.Empty())
    for item in response.items:
        if item.name == environment:
            return {"name": item.name,
                    "max_containers": (item.max_concurrent_tasks
                                       if item.HasField("max_concurrent_tasks") else None),
                    "max_gpus": (item.max_concurrent_gpus
                                 if item.HasField("max_concurrent_gpus") else None),
                    "current_containers": item.current_concurrent_tasks,
                    "current_gpus": item.current_concurrent_gpus}
    return None


def _blocking(async_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Run an async SDK call through the SDK's own synchronizer (replaced in tests)."""
    from modal._utils.async_utils import synchronizer

    return synchronizer.create_blocking(async_fn)


async def _client_from_env() -> Any:
    from modal.client import _Client

    return await _Client.from_env()


def _modal() -> Any:
    import modal

    return modal


RESTORE_ORIGINAL_PACKAGES = (
    "python3 -c \"import subprocess, sys; "
    "parse = lambda text: dict(l.split('==', 1) for l in text.split() if '==' in l); "
    "orig = parse(open('/opt/slh_orig_freeze.txt').read()); "
    "now = parse(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True)); "
    "diff = [k + '==' + v for k, v in orig.items() if now.get(k) != v]; "
    "print('restoring', diff); "
    "diff and subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', *diff])\""
)
# Import-only: `vllm serve --help` needs a GPU to infer its device, and image
# builds run on CPU. These are the modules whose imports broke.
VLLM_IMPORT_CHECK = (
    "python3 -c 'import typing_extensions, fastapi, vllm.entrypoints.cli.main, "
    "vllm.entrypoints.openai.api_server; print(\"vllm imports ok\")'"
)


class ModalControl:
    """Every Modal operation, pinned to one Modal Environment.

    The SDK's ``environment_name`` is passed wherever it exists (App.lookup,
    Volume.from_name, SandboxList); everything else (Secret.from_dict,
    Sandbox.list, Image.build via the app) resolves the environment from
    ``MODAL_ENVIRONMENT``, which ``_pin`` sets and asserts before every call.
    CLI calls get ``--env`` and the variable. Tests substitute a fake."""

    def __init__(self, environment: str = MODAL_ENVIRONMENT,
                 allow_other_environment: bool = False) -> None:
        if not environment:
            raise ServingError("a Modal environment is required")
        if environment != MODAL_ENVIRONMENT and not allow_other_environment:
            raise ServingError(f"Modal environment must be {MODAL_ENVIRONMENT!r} (got "
                               f"{environment!r}); --allow-other-modal-env exists for CPU "
                               "work only")
        self.environment = environment
        self.gpu_allowed = environment == MODAL_ENVIRONMENT
        self._app: Any = None
        self._members: set[str] = set()

    def _scoped(self, sandbox_id: str) -> dict[str, Any] | None:
        app_id = self._app_handle().app_id
        self._pin()

        async def run() -> dict[str, Any] | None:
            return await scoped_sandbox_async(await _client_from_env(), app_id, sandbox_id,
                                              self.environment)

        return _blocking(run)()

    def _member(self, sandbox_id: str) -> Any:
        """Refuse any by-ID operation on a sandbox that an app- and
        environment-scoped listing does not show as ours."""
        if not isinstance(sandbox_id, str) or not sandbox_id.startswith("sb-"):
            raise NotOurSandbox(f"not a sandbox id: {sandbox_id!r}")
        if sandbox_id not in self._members:
            if self._scoped(sandbox_id) is None:
                raise NotOurSandbox(f"sandbox {sandbox_id} is not in app {SANDBOX_APP_NAME} "
                                    f"in environment {self.environment!r}; refusing")
            self._members.add(sandbox_id)
        return self._pin().Sandbox.from_id(sandbox_id)

    def _pin(self) -> Any:
        os.environ["MODAL_ENVIRONMENT"] = self.environment
        modal = _modal()
        with contextlib.suppress(ImportError):
            from modal.config import config

            if config.get("environment") != self.environment:
                raise ServingError("MODAL_ENVIRONMENT is not pinned; refusing the Modal call")
        return modal

    def _app_handle(self) -> Any:
        modal = self._pin()
        if self._app is None:
            self._app = modal.App.lookup(SANDBOX_APP_NAME, create_if_missing=True,
                                         environment_name=self.environment)
        return self._app

    def environment_limits(self) -> dict[str, Any] | None:
        self._pin()

        async def run() -> dict[str, Any] | None:
            return await environment_limits_async(await _client_from_env(), self.environment)

        return _blocking(run)()

    def prepare(self, window: dict[str, Any], token: str | None) -> dict[str, Any]:
        """Build the image and hydrate every dependency now, so the create
        request that follows is quick and its timeout is anchored to the deadline."""
        modal = self._pin()
        here = Path(__file__).resolve().parent
        if window["kind"] == "probe":
            image = modal.Image.debian_slim(python_version="3.12")
        else:
            # The vLLM image ships python3 but no `python`, which Modal's image setup
            # runs; link it before Modal's own steps.
            image = (modal.Image.from_registry(
                        VLLM_IMAGE,
                        setup_dockerfile_commands=[
                            "RUN ln -sf \"$(command -v python3)\" /usr/local/bin/python",
                            "RUN python3 -m pip freeze --all > /opt/slh_orig_freeze.txt"])
                     .entrypoint([])
                     # Modal's legacy image setup downgrades some of vLLM's packages
                     # (typing_extensions broke vLLM's imports); restore the image's
                     # original versions, then prove vLLM's server imports on CPU so a
                     # broken image fails at build time, never on a GPU.
                     .run_commands(RESTORE_ORIGINAL_PACKAGES, VLLM_IMPORT_CHECK)
                     .env({"HF_HOME": "/root/.cache/huggingface", "VLLM_NO_USAGE_STATS": "1",
                           "DO_NOT_TRACK": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}))
        image = (image.add_local_file(here / "serve_entry.py", "/opt/slh_serve/serve_entry.py",
                                      copy=True)
                 .add_local_file(here / "serve_spec.py", "/opt/slh_serve/serve_spec.py",
                                 copy=True))
        app = self._app_handle()
        self._pin()
        image = image.build(app)
        volumes = {}
        if window["kind"] != "probe":
            self._pin()
            volume = modal.Volume.from_name("slh-hf-cache", create_if_missing=True,
                                            environment_name=self.environment)
            if window["kind"] == "serve":
                volume = volume.with_mount_options(read_only=True)
            volume.hydrate()
            volumes["/root/.cache/huggingface"] = volume
        secrets_ = []
        if token:
            self._pin()
            secret = modal.Secret.from_dict({"SLH_SERVE_TOKEN": token})
            secret.hydrate()
            secrets_.append(secret)
        return {"app": app, "image": image, "volumes": volumes, "secrets": secrets_,
                "environment": self.environment}

    def create(self, prepared: dict[str, Any], window: dict[str, Any], timeout_s: int,
               issued_at: float) -> str:
        modal = self._pin()
        if prepared.get("environment") != self.environment:
            raise ServingError("prepared objects belong to another Modal environment")
        if window["kind"] == "serve" and not self.gpu_allowed:
            raise ServingError("GPU sandboxes run only in the dedicated environment "
                               f"{MODAL_ENVIRONMENT!r}")
        request = sandbox_request(window, timeout_s, issued_at)
        sandbox = modal.Sandbox.create(
            *request["command"], app=prepared["app"], image=prepared["image"],
            volumes=prepared["volumes"], secrets=prepared["secrets"],
            timeout=request["timeout"], idle_timeout=request["idle_timeout"],
            gpu=request["gpu"], cpu=request["cpu"], memory=request["memory"],
            encrypted_ports=request["encrypted_ports"], tags=request["tags"])
        self._members.add(sandbox.object_id)  # created just now, in our app
        return sandbox.object_id

    def tunnel_url(self, sandbox_id: str, timeout: int) -> str:
        return self._member(sandbox_id).tunnels(timeout=timeout)[PROXY_PORT].url

    def state(self, sandbox_id: str) -> dict[str, Any]:
        sandbox = self._member(sandbox_id)
        returncode = sandbox.poll()
        result = getattr(sandbox, "_result", None)
        status = None
        if result is not None and getattr(result, "status", 0):
            from modal_proto import api_pb2

            status = api_pb2.GenericResult.GenericStatus.Name(result.status)
        return {"sandbox_id": sandbox_id, "finished": returncode is not None,
                "returncode": returncode, "status": status}

    def terminate(self, sandbox_id: str) -> None:
        self._member(sandbox_id).terminate()

    def list_running(self) -> list[dict[str, Any]]:
        app = self._app_handle()
        modal = self._pin()
        return [{"sandbox_id": sb.object_id, "tags": sb.get_tags()}
                for sb in modal.Sandbox.list(app_id=app.app_id)]

    def timing(self, sandbox_id: str) -> dict[str, Any] | None:
        info = self._scoped(sandbox_id)
        if info is None:
            raise NotOurSandbox(f"sandbox {sandbox_id} is not ours; no timestamps")
        return info

    def output(self, sandbox_id: str) -> str:
        return self._member(sandbox_id).stdout.read()

    def logs(self, sandbox_id: str) -> str:
        """stdout and stderr of a finished sandbox (read only after it ended)."""
        sandbox = self._member(sandbox_id)
        return sandbox.stdout.read() + sandbox.stderr.read()

    def cli_env(self) -> dict[str, str]:
        return {**os.environ, "MODAL_ENVIRONMENT": self.environment}

    def running_containers(self) -> int:
        out = subprocess.run(["modal", "container", "list", "--env", self.environment,
                              "--json"], check=True, capture_output=True, text=True,
                             timeout=120, env=self.cli_env()).stdout
        containers = json.loads(out)
        if not isinstance(containers, list):
            raise ServingError("unexpected `modal container list --json` output")
        return len(containers)


# ---------------------------------------------------------------- locks and the session directory


@contextlib.contextmanager
def project_lock() -> Iterator[None]:
    """One lock for the whole project on this machine (every session dir)."""
    path = Path(PROJECT_LOCK)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class _Held:
    def __init__(self, session: Session, data: dict[str, Any]) -> None:
        self.session, self.data = session, data

    def write(self) -> None:
        self.session._write(self.data)


class Session:
    """``serving/<name>/``: session.json, private/, ledger.jsonl, lock."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.path = self.root / "session.json"
        self.token_path = self.root / "private" / "token"
        self.ledger_path = self.root / "ledger.jsonl"
        self.lock_path = self.root / "session.lock"

    @contextlib.contextmanager
    def hold(self) -> Iterator[_Held]:
        """The session lock, with explicit (durable) writes inside it."""
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield _Held(self, self.read() or {"format": FORMAT, "session": self.root.name,
                                                  "windows": [], "status_checks": []})
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @contextlib.contextmanager
    def locked(self) -> Iterator[dict[str, Any]]:
        """Read-modify-write; written on a clean exit, discarded if the block raises."""
        with self.hold() as held:
            yield held.data
            held.write()

    def read(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)

    def private(self) -> Path:
        path = self.root / "private"
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)
        return path

    def new_token(self) -> str:
        self.private()
        token = secrets.token_urlsafe(32)
        fd = os.open(self.token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(token)
        return token

    def token(self) -> str:
        return self.token_path.read_text(encoding="utf-8").strip()

    def token_or_none(self) -> str | None:
        try:
            return self.token()
        except OSError:
            return None

    def ledger(self, cap_usd: float, owner: str, session: int,
               create: bool = False) -> SharedLedger:
        return SharedLedger(self.ledger_path, cap_usd, owner=owner, session=session,
                            quarantine=self.root / "private" / "quarantine.jsonl",
                            wait_s=0.0, create=create)


def resolve_session_dir(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def window_by_id(data: dict[str, Any], window_id: str) -> dict[str, Any]:
    for window in data["windows"]:
        if window["window_id"] == window_id:
            return window
    raise ServingError(f"no window {window_id!r} in this session")


def current_window(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if not data or not data.get("windows"):
        return None
    return data["windows"][-1]


def transition(window: dict[str, Any], new: str, reason: str) -> None:
    """Move a window along the state machine, or raise InvalidTransition."""
    old = window["status"]
    if old == new:
        return
    if new not in TRANSITIONS.get(old, set()):
        raise InvalidTransition(f"window {window['window_id']}: {old} -> {new} is not allowed")
    if new == "verified" and window.get("settlement") is None:
        raise InvalidTransition(f"window {window['window_id']}: not settled; cannot verify")
    window["status"] = new
    window.setdefault("history", []).append({"from": old, "to": new, "reason": reason,
                                             "time_utc": _iso(_now())})


def _update(session: Session, window_id: str, fields: dict[str, Any] | None = None,
            status: str | None = None, reason: str = "") -> dict[str, Any]:
    with session.locked() as data:
        window = window_by_id(data, window_id)
        if status is not None:
            transition(window, status, reason)
        window.update(fields or {})
        return dict(window)


# ---------------------------------------------------------------- runner side


class ServingGate:
    """What the agent backend and the runner ask before each request / episode."""

    def __init__(self, session_dir: str | Path, model: str) -> None:
        self.session = Session(resolve_session_dir(session_dir))
        self.model = model

    def _window(self) -> dict[str, Any]:
        try:
            data = self.session.read()
        except (OSError, ValueError) as exc:
            raise ServingWindowClosed(f"serving session unreadable: {type(exc).__name__}") \
                from None
        windows = [w for w in (data or {}).get("windows", []) if w.get("kind") == "serve"]
        window = windows[-1] if windows else None
        if window is None:
            raise ServingWindowClosed(f"no serving window in {self.session.root}; run "
                                      "`python -m slh serve start` first")
        if window.get("status") != "serving":
            raise ServingWindowClosed(f"serving window {window['window_id']} is "
                                      f"{window.get('status')}, not serving")
        if window.get("served_model") != self.model:
            raise ServingWindowClosed(f"window serves {window.get('served_model')!r}, "
                                      f"the config asks for {self.model!r}")
        return window

    def check_request(self) -> tuple[str, str]:
        window = self._window()
        if _now() >= window["request_deadline_epoch"]:
            raise ServingWindowClosed(
                f"serving window {window['window_id']} request deadline "
                f"{window['request_deadline_utc']} has passed; resume the run in a new window")
        token = self.session.token_or_none()
        if not token:
            raise ServingWindowClosed("serving token missing")
        return window["endpoint_url"].rstrip("/") + "/v1", token

    def check_episode_start(self) -> dict[str, Any]:
        window = self._window()
        if _now() >= window["episode_start_deadline_epoch"]:
            raise ServingWindowClosed(
                f"serving window {window['window_id']} episode-start deadline "
                f"{window['episode_start_deadline_utc']} has passed; no new episode started")
        return window

    def identity(self) -> dict[str, Any] | None:
        with contextlib.suppress(OSError, ValueError):
            windows = [w for w in (self.session.read() or {}).get("windows", [])
                       if w.get("kind") == "serve"]
            if windows:
                window = windows[-1]
                return {k: window.get(k) for k in (
                    "window_id", "sandbox_id", "endpoint_url", "status", "deadline_utc",
                    "request_deadline_utc", "served_model", "spec_sha256", "gpu_used",
                    "gpu_preference")} | {
                    "session_dir": str(self.session.root)}
        return None


def gate_for(config: Any) -> ServingGate | None:
    session = getattr(config, "serving_session", None)
    return None if session is None else ServingGate(session, config.model)


# ---------------------------------------------------------------- verification and stop


def _sandbox_id(window: dict[str, Any], running: list[dict[str, Any]]) -> str | None:
    if window.get("sandbox_id"):
        return window["sandbox_id"]
    tagged = [r for r in running
              if (r.get("tags") or {}).get("slh_window") == window["window_id"]]
    return tagged[0]["sandbox_id"] if len(tagged) == 1 else None


def verify_finished(control: Any, window: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Verified only if (a) no create was ever requested and nothing runs with
    the window's tag, or (b) ``poll`` on the id says finished and the id is not
    in the running list. Anything missing, malformed or failing: unverified."""
    evidence: dict[str, Any] = {"time_utc": _iso(_now())}
    try:
        running = control.list_running()
        if not isinstance(running, list) or not all(
                isinstance(r, dict) and isinstance(r.get("sandbox_id"), str) for r in running):
            raise ServingError("malformed running-sandbox list")
    except Exception as exc:  # noqa: BLE001 - unverified
        evidence["error"] = f"list_running:{type(exc).__name__}"
        return False, evidence
    running_ids = {r["sandbox_id"] for r in running}
    if not window.get("create_called"):
        tagged = [r for r in running
                  if (r.get("tags") or {}).get("slh_window") == window["window_id"]]
        evidence["basis"] = "never_created"
        return not tagged, evidence
    sandbox_id = _sandbox_id(window, running)
    if sandbox_id is None:
        evidence["error"] = "sandbox_id_unknown"
        return False, evidence
    evidence["sandbox_id"] = sandbox_id
    try:
        state = control.state(sandbox_id)
    except Exception as exc:  # noqa: BLE001
        evidence["error"] = f"state:{type(exc).__name__}"
        return False, evidence
    if not isinstance(state, dict) or not isinstance(state.get("finished"), bool):
        evidence["error"] = "malformed_state"
        return False, evidence
    evidence.update(finished=state["finished"], returncode=state.get("returncode"),
                    status=state.get("status"), listed_running=sandbox_id in running_ids,
                    basis="poll_and_list")
    return state["finished"] and sandbox_id not in running_ids, evidence


def _timing(control: Any, window: dict[str, Any]) -> dict[str, Any] | None:
    if not (window.get("create_called") and window.get("sandbox_id")):
        return None
    try:
        timing = control.timing(window["sandbox_id"])
    except Exception as exc:  # noqa: BLE001 - no timestamps: full reservation
        return {"error": type(exc).__name__}
    return timing if isinstance(timing, dict) else None


def settle_once(session: Session, cap_usd: float, window: dict[str, Any], charge: float,
                owner: str) -> bool:
    """Settle a window's reservation in the ledger exactly once (keyed by its
    reservation id): a retry after a crash between the ledger write and the
    session write is a no-op. Returns whether it wrote."""
    rid = window.get("reservation_id")
    if not rid:
        return False
    if session.ledger_path.is_file() and any(
            r.get("op") in ("settle", "charge") and r.get("id") == rid
            for r in read_records(session.ledger_path)):
        return False
    session.ledger(cap_usd, owner=owner, session=window["ledger_session"]).settle(
        rid, charge, "agent", f"{window['kind']}:{window['rate']['gpu']}")
    return True


def _close(session: Session, window_id: str, verified: bool, evidence: dict[str, Any],
           record: dict[str, Any], timing: dict[str, Any] | None, owner: str) -> dict[str, Any]:
    """One locked update: stop record, stopped, and (if verified) settlement,
    excess and verified together."""
    with session.locked() as data:
        window = window_by_id(data, window_id)
        window["stop"] = {**record, "verified": verified, "evidence": evidence}
        if window["status"] == "stopping":
            transition(window, "stopped", record["reason"])
        if verified and window["status"] == "stopped" and window.get("settlement") is None:
            result = settlement(window, timing)
            result["ledger_written"] = settle_once(session, data.get("cap_usd", 0.0), window,
                                                   result["charge_usd"], owner)
            window["settlement"] = result
            if result["excess_usd"] > 0:
                window["excess"] = {"usd": result["excess_usd"], "cleared": False}
            if window.get("measure_timeout"):
                window["timeout_measurement"] = probe_outcome(
                    {"finished": evidence.get("finished"), "status": evidence.get("status"),
                     "returncode": evidence.get("returncode")}, timing,
                    window.get("create_timeout_s") or window["window_s"],
                    MEASURE_EARLY_TOLERANCE_S, MEASURE_LATE_TOLERANCE_S)
            transition(window, "verified", record["reason"])
        return dict(window)


def stop_window(session: Session, window_id: str, control: Any, reason: str,
                owner: str = "serve-stop") -> dict[str, Any]:
    """Terminate, verify, settle; ``verified`` only together with the settlement."""
    with session.locked() as data:
        window = window_by_id(data, window_id)
        if window["status"] == "verified":
            return dict(window)
        if window["status"] in LIVE:
            transition(window, "stopping", reason)
        window.setdefault("stop_requests", []).append({"reason": reason, "owner": owner,
                                                       "time_utc": _iso(_now())})
        snapshot = dict(window)
    errors = []
    sandbox_id = snapshot.get("sandbox_id")
    had_id = bool(sandbox_id)
    if not sandbox_id and snapshot.get("create_called"):
        with contextlib.suppress(Exception):
            sandbox_id = _sandbox_id(snapshot, control.list_running())
    if sandbox_id:
        try:
            control.terminate(sandbox_id)
        except Exception as exc:  # noqa: BLE001 - verification decides
            errors.append(f"terminate:{type(exc).__name__}")
    snapshot["sandbox_id"] = sandbox_id
    deadline = _now() + VERIFY_STOP_S
    while True:
        verified, evidence = verify_finished(control, snapshot)
        if verified or _now() >= deadline:
            break
        _sleep(5.0)
    if sandbox_id and not had_id:
        _update(session, window_id, {"sandbox_id": sandbox_id})
    timing = _timing(control, snapshot) if verified else None
    return _close(session, window_id, verified, evidence,
                  {"reason": reason, "owner": owner, "errors": errors,
                   "time_utc": _iso(_now())}, timing, owner)


def launch_stale(window: dict[str, Any], now: float) -> bool:
    return (window.get("status") == "launching"
            and now >= window["startup_deadline_epoch"] + LAUNCH_GRACE_S)


def sandbox_ended(control: Any, window: dict[str, Any]) -> bool:
    try:
        return verify_finished(control, window)[0]
    except Exception:  # noqa: BLE001
        return False


def recheck_verified(session: Session, control: Any) -> list[str]:
    """A verified window whose sandbox is listed running: open discrepancy
    (blocks starts) and terminate."""
    data = session.read()
    if not data:
        return []
    running_ids = {r["sandbox_id"] for r in control.list_running()}
    found = []
    for window in data["windows"]:
        if window["status"] == "verified" and window.get("sandbox_id") in running_ids:
            found.append(window["window_id"])
            with contextlib.suppress(Exception):
                control.terminate(window["sandbox_id"])
            _update(session, window["window_id"], {"discrepancy": {
                "found_utc": _iso(_now()), "detail": "verified window's sandbox listed running",
                "open": True}})
    return found


def resolve_discrepancies(session: Session, control: Any) -> list[str]:
    resolved = []
    for window in (session.read() or {"windows": []})["windows"]:
        if (window.get("discrepancy") or {}).get("open"):
            with contextlib.suppress(Exception):
                control.terminate(window["sandbox_id"])
            if verify_finished(control, window)[0]:
                _update(session, window["window_id"], {"discrepancy": {
                    **window["discrepancy"], "open": False, "resolved_utc": _iso(_now())}})
                resolved.append(window["window_id"])
    return resolved


def check_environment(session: Session, control: Any) -> None:
    """Every window of a session lives in one Modal environment: the control
    must target that environment."""
    for window in (session.read() or {"windows": []})["windows"]:
        if window.get("modal_environment") != control.environment:
            raise ServingError(f"window {window['window_id']} is in Modal environment "
                               f"{window.get('modal_environment')!r}, not "
                               f"{control.environment!r}; refusing")


def reconcile(session: Session, control: Any, owner: str = "reconcile") -> list[dict[str, Any]]:
    check_environment(session, control)
    data = session.read()
    if not data:
        return []
    now = _now()
    actions = []
    for window in data["windows"]:
        status_now = window["status"]
        if status_now == "verified":
            continue
        if now >= window["watchdog_fire_epoch"]:
            reason = "reconcile_overdue"
        elif launch_stale(window, now):
            reason = "reconcile_launch_stale"
        elif status_now == "serving" and sandbox_ended(control, window):
            reason = "reconcile_sandbox_ended"
        elif status_now in LIVE:
            continue
        else:
            reason = f"reconcile_{status_now}"
        stopped = stop_window(session, window["window_id"], control, reason=reason,
                              owner=owner)
        actions.append({"window_id": window["window_id"], "reason": reason,
                        "status": stopped["status"]})
    for window_id in recheck_verified(session, control):
        actions.append({"window_id": window_id, "reason": "discrepancy_running",
                        "status": "verified"})
    return actions


def blockers(data: dict[str, Any] | None) -> list[str]:
    out = []
    for window in (data or {}).get("windows", []):
        wid = window["window_id"]
        if window["status"] != "verified":
            out.append(f"window {wid} is {window['status']} (not verified stopped)")
        elif window.get("settlement") is None:
            out.append(f"window {wid} is not settled")
        if (window.get("discrepancy") or {}).get("open"):
            out.append(f"window {wid} has an open discrepancy (run `serve stop`)")
        if window.get("excess") and not window["excess"].get("cleared"):
            out.append(f"window {wid} settled ${window['excess']['usd']:.6f} above its "
                       "reservation (owner must review and run `serve owner-clear`)")
    return out


def status(session: Session, control: Any) -> dict[str, Any]:
    actions = reconcile(session, control, owner="status")
    running_ids = {r["sandbox_id"] for r in control.list_running()}
    data = session.read() or {"windows": []}
    ours = {w.get("sandbox_id") for w in data["windows"]}
    rows = [{"window_id": w["window_id"], "kind": w["kind"], "status": w["status"],
             "sandbox_id": w.get("sandbox_id"),
             "running": w.get("sandbox_id") in running_ids,
             "settled_usd": (w.get("settlement") or {}).get("charge_usd"),
             "excess_usd": (w.get("excess") or {}).get("usd")} for w in data["windows"]]
    stray = [r for r in rows if r["status"] not in LIVE and r["running"]]
    check = {"time_utc": _iso(_now()), "reconciled": actions, "windows": rows,
             "running_in_app": len(running_ids),
             "running_not_in_this_session": sorted(running_ids - ours),
             "closed_windows_all_stopped": not stray, "blockers": blockers(data)}
    if data["windows"]:
        with session.locked() as locked:
            locked.setdefault("status_checks", []).append(check)
    return check


# ---------------------------------------------------------------- watchdog


def watchdog_loop(session: Session, window_id: str, control: Any) -> str:
    next_check = _now() + CONTAINER_CHECK_S
    while True:
        window = window_by_id(session.read() or {"windows": []}, window_id)
        status_now = window["status"]
        if status_now == "verified":
            return "verified"
        now = _now()
        if (now >= window["watchdog_fire_epoch"] or status_now not in LIVE
                or launch_stale(window, now)):
            reason = ("watchdog_deadline" if now >= window["watchdog_fire_epoch"]
                      else "watchdog_launch_stale" if launch_stale(window, now)
                      else f"watchdog_{status_now}")
            try:
                stopped = stop_window(session, window_id, control, reason=reason,
                                      owner="watchdog")
            except Exception as exc:  # noqa: BLE001 - retried below
                print(f"{_iso(now)} watchdog stop failed: {type(exc).__name__}", flush=True)
                stopped = {"status": "stopped"}
            if stopped["status"] == "verified":
                return "verified"
            if _now() >= window["deadline_epoch"] + WATCHDOG_GIVE_UP_S:
                return "unverified"
            _sleep(60.0)
            continue
        if status_now == "serving" and now >= next_check:
            next_check = now + CONTAINER_CHECK_S
            if sandbox_ended(control, window):
                stop_window(session, window_id, control, reason="watchdog_sandbox_ended",
                            owner="watchdog")
                continue
        _sleep(min(WATCHDOG_POLL_S, max(1.0, window["watchdog_fire_epoch"] - now)))


def spawn_watchdog(session: Session, window_id: str, environment: str = MODAL_ENVIRONMENT
                   ) -> int:
    log = (session.root / f"watchdog-{window_id}.log").open("a")
    process = subprocess.Popen(
        [sys.executable, "-m", "slh", "serve", "watchdog", "--out", str(session.root),
         "--window", window_id, "--modal-env", environment],
        cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
        env={**os.environ, "MODAL_ENVIRONMENT": environment}, start_new_session=True)
    return process.pid


# ---------------------------------------------------------------- ingress check


def http_request(method: str, url: str, token: str | None,
                 body: bytes | None = None, timeout: float = 20.0) -> tuple[int, Any]:
    """One HTTP request returning (status, parsed JSON or None). Replaced in tests."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            try:
                return response.status, json.loads(raw)
            except ValueError:
                return response.status, None
    except urllib.error.HTTPError as exc:
        return exc.code, None


def _is_timeout(exc: BaseException) -> bool:
    """A lookup that ran out of time (the provider's timeout errors subclass
    a class named ``TimeoutError``; the builtin one counts too)."""
    return any(cls.__name__ == "TimeoutError" for cls in type(exc).__mro__)


def wait_tunnel(control: Any, sandbox_id: str, deadline: float) -> str:
    """The sandbox's tunnel URL: short lookups repeated until ``deadline`` (the
    window's startup deadline). Refuses once the sandbox has finished; any
    error other than a lookup timeout is raised as is. No lookup starts at or
    after the deadline, and each lookup's timeout (whole seconds, at least 1)
    ends at or before it; with less than 1 s left the wait fails at the
    deadline without a lookup."""
    while True:
        if _now() >= deadline:
            raise TunnelNotReady("tunnel not ready before startup deadline")
        if control.state(sandbox_id).get("finished"):
            raise ServingError("the sandbox finished while waiting for its tunnel")
        remaining = deadline - _now()  # after the state check, which takes time
        if remaining < 1:  # the lookup timeout is whole seconds (>= 1): no lookup fits
            _sleep(max(0.0, remaining))
            raise TunnelNotReady("tunnel not ready before startup deadline")
        try:
            return control.tunnel_url(
                sandbox_id, timeout=min(TUNNEL_POLL_S, math.floor(remaining)))
        except Exception as exc:  # only timeouts are retried
            if not _is_timeout(exc):
                raise
        _sleep(min(TUNNEL_RETRY_PAUSE_S, max(0.0, deadline - _now())))


def _started(timing: Any) -> bool | None:
    """True if the record shows a container start; False only if it carries
    ``started_at`` explicitly as None (the scoped lookup's "not started");
    None (unknown) if the record is missing, an error, or has no
    ``started_at`` key. Only False permits a fallback."""
    if not isinstance(timing, dict) or timing.get("error") or "started_at" not in timing:
        return None
    return timing["started_at"] is not None


def container_started(control: Any, sandbox_id: str) -> bool | None:
    """Whether Modal reports the sandbox's container as started (``started_at``
    in its scoped record); None if that cannot be read (see ``_started``)."""
    try:
        timing = control.timing(sandbox_id)
    except Exception:  # noqa: BLE001 - unknown: never a reason to fall back
        return None
    return _started(timing)


def fall_back(session: Session, window_id: str, control: Any, prepared: Any,
              sandbox_id: str, capacity: dict[str, Any], next_gpu: str) -> str:
    """Replace a never-scheduled sandbox by one on ``next_gpu`` in the same
    window. The old sandbox is terminated and must be verified finished, never
    started and silent (empty stdout and stderr); its Modal lifetime is
    recorded (it is charged at the window's rate). The new one is created under
    the project lock with the same capacity checks, ``timeout = deadline -
    now`` and a fresh startup deadline (``create_locked(fallback=True)``)."""
    with contextlib.suppress(Exception):
        control.terminate(sandbox_id)
    snapshot = window_by_id(session.read(), window_id)
    deadline = _now() + VERIFY_STOP_S
    while True:
        verified, evidence = verify_finished(control, snapshot)
        if verified or _now() >= deadline:
            break
        _sleep(5.0)
    if not verified:
        raise ServingError("no GPU fallback: the unscheduled sandbox is not verified stopped")
    timing = _timing(control, snapshot)
    try:
        logs = control.logs(sandbox_id)
    except Exception as exc:  # noqa: BLE001
        raise ServingError(f"no GPU fallback: cannot read the sandbox's logs "
                           f"({type(exc).__name__})") from None
    if _started(timing) is not False:
        raise ServingError("no GPU fallback: the sandbox's container started (or its start "
                           "cannot be read)")
    # Exactly empty: any byte, whitespace included, counts as output.
    if not isinstance(logs, str) or logs != "":
        raise ServingError("no GPU fallback: the sandbox wrote output")
    with session.locked() as data:
        window = window_by_id(data, window_id)
        if window["status"] != "launching":
            raise ServingError(f"window {window_id} is {window['status']}: no GPU fallback")
        window["gpu_attempts"].append({
            "gpu": window["gpu_requested"], "sandbox_id": sandbox_id,
            "create_requested_utc": window.get("create_requested_utc"),
            "issued_at_epoch": window.get("issued_at_epoch"),
            "create_timeout_s": window.get("create_timeout_s"),
            "outcome": "not_scheduled", "schedule_wait_s": GPU_SCHEDULE_WAIT_S,
            "stop_evidence": evidence, "timing": timing, "logs_empty": True,
            "replaced_utc": _iso(_now())})
        window.update(gpu_requested=next_gpu, sandbox_id=None, create_called=False)
    with project_lock():
        new_id, _ = create_locked(
            session, window_id, control, prepared, fallback=True,
            precreate=lambda: _recheck_capacity(control, "serve", capacity))
    return new_id


def wait_tunnel_or_fall_back(session: Session, window_id: str, control: Any, prepared: Any,
                             sandbox_id: str, capacity: dict[str, Any]) -> tuple[str, str]:
    """The tunnel URL and the sandbox that serves it.

    While a later GPU type remains in the preference (and no fallback was used
    yet), the first lookup waits only until ``GPU_SCHEDULE_WAIT_S`` after the
    create request (never past the startup deadline). If the tunnel is not
    ready then: a container that started (or whose start cannot be read) keeps
    waiting until the startup deadline, as before; one that never started is
    replaced once by a sandbox on the next GPU type (``fall_back``). The last
    GPU type always waits until its startup deadline."""
    while True:
        window = window_by_id(session.read(), window_id)
        preference = window.get("gpu_preference") or [window.get("gpu_requested")]
        index = preference.index(window["gpu_requested"])
        startup = window["startup_deadline_epoch"]
        can_fall_back = (index + 1 < len(preference)
                         and len(window.get("gpu_attempts") or []) < MAX_GPU_FALLBACKS)
        if not can_fall_back:
            return wait_tunnel(control, sandbox_id, startup), sandbox_id
        schedule = min(startup, window["issued_at_epoch"] + GPU_SCHEDULE_WAIT_S)
        try:
            return wait_tunnel(control, sandbox_id, schedule), sandbox_id
        except TunnelNotReady:
            pass
        if container_started(control, sandbox_id) is not False or _now() >= startup:
            return wait_tunnel(control, sandbox_id, startup), sandbox_id
        sandbox_id = fall_back(session, window_id, control, prepared, sandbox_id, capacity,
                               preference[index + 1])


def check_ingress(endpoint: str, token: str, served_name: str, deadline: float,
                  alive: Callable[[], bool],
                  request: Callable[..., tuple[int, Any]] = http_request) -> dict[str, Any]:
    """Wait for /v1/models with the token; then every route in PROBE_ROUTES
    must be 401 without it, and with it only the allow-list may be forwarded
    (others 404)."""
    base = endpoint.rstrip("/")
    last = None
    while True:
        if _now() >= deadline:
            raise ServingError(f"server not healthy before the startup deadline (last: {last})")
        if not alive():
            raise ServingError("the sandbox finished during startup")
        try:
            code, body = request("GET", base + "/v1/models", token)
            ids = [m.get("id") for m in (body or {}).get("data", [])] if code == 200 else []
            if served_name in ids:
                break
            last = f"HTTP {code}"
        except Exception as exc:  # noqa: BLE001 - not up yet
            last = type(exc).__name__
        _sleep(10.0)
    matrix, problems = [], []
    for method, path in PROBE_ROUTES:
        body = b"{}" if method in ("POST", "PATCH") else None
        anonymous, _ = request(method, base + path, None, body)
        authed, _ = request(method, base + path, token, body)
        allowed = (method, path) in ALLOWED_ROUTES
        matrix.append({"route": f"{method} {path}", "allowed": allowed,
                       "without_token": anonymous, "with_token": authed})
        if anonymous != 401:
            problems.append(f"{method} {path} answered {anonymous} without the token")
        if allowed and authed in (401, 403, 404, 405):
            problems.append(f"{method} {path} not forwarded ({authed})")
        if not allowed and authed != 404:
            problems.append(f"{method} {path} answered {authed} with the token (must be 404)")
    if problems:
        raise ServingError("ingress check failed: " + "; ".join(problems))
    return {"healthy_utc": _iso(_now()), "routes": matrix}


# ---------------------------------------------------------------- admission and creation


def _capacity(control: Any, kind: str) -> dict[str, Any]:
    """Checked under the project lock, immediately before admission: the
    environment exists and has its server-side limits (at most one GPU, at
    most MAX_ENV_CONTAINERS containers); no GPU and no sandbox of our app is
    already running in it; a container is free."""
    limits = control.environment_limits()
    if not limits:
        raise ServingError(f"Modal environment {control.environment!r} not found; refusing")
    max_gpus, max_containers = limits.get("max_gpus"), limits.get("max_containers")
    if max_gpus is None or max_gpus < 1 or max_gpus > MAX_ENV_GPUS:
        raise ServingError(f"environment {control.environment!r} must have a server-side "
                           f"GPU limit of {MAX_ENV_GPUS} (has {max_gpus}); refusing")
    if max_containers is None or max_containers > MAX_ENV_CONTAINERS:
        raise ServingError(f"environment {control.environment!r} must have a server-side "
                           f"container limit <= {MAX_ENV_CONTAINERS} (has {max_containers})")
    used = control.running_containers()
    if isinstance(used, bool) or not isinstance(used, int) or used < 0:
        raise ServingError("cannot read the environment's running containers")
    if kind == "serve" and not getattr(control, "gpu_allowed", True):
        raise ServingError(f"GPU windows run only in {MODAL_ENVIRONMENT!r}")
    used = max(used, int(limits.get("current_containers") or 0))
    if limits.get("current_gpus"):
        raise ServingError(f"a GPU container is already running in {control.environment!r}; "
                           "not launching")
    if used + 1 > max_containers:
        raise ServingError(f"environment {control.environment!r} has {used} of "
                           f"{max_containers} containers running; not launching")
    running = control.list_running()
    if running:
        raise ServingError(f"{len(running)} sandbox(es) already running in "
                           f"{SANDBOX_APP_NAME}; one window at a time (run `serve status`)")
    return {"environment": control.environment, "limits": limits,
            "running_containers": used}


def _probe_acceptable(result: dict[str, Any], accept_undetermined: bool) -> bool:
    if result.get("ok"):
        return True
    return (accept_undetermined and result.get("ended_by_timeout") is True
            and result.get("lifetime_within_range") is True
            and result.get("timeout_origin_supported") == "undetermined")


def _recheck_capacity(control: Any, kind: str, before: dict[str, Any]) -> dict[str, Any]:
    """Immediately before the create request: the limits must be unchanged and
    still safe, and nothing of ours or on the GPU may have started meanwhile."""
    now = _capacity(control, kind)
    keys = ("max_gpus", "max_containers")
    if any(now["limits"].get(k) != before["limits"].get(k) for k in keys):
        raise ServingError("the environment's limits changed during preparation "
                           f"({ {k: before['limits'].get(k) for k in keys} } -> "
                           f"{ {k: now['limits'].get(k) for k in keys} }); refusing")
    return now


def _require_passed(data: dict[str, Any], kind: str,
                    accept_undetermined: bool = False) -> dict[str, Any]:
    passed = [w for w in data["windows"] if w["kind"] == kind and w["status"] == "verified"
              and (_probe_acceptable(w.get("result") or {}, accept_undetermined)
                   if kind == "probe" else (w.get("result") or {}).get("ok"))
              and (kind != "prefetch" or w.get("spec_sha256") == spec_sha256())]
    if not passed:
        name = "probe-timeout" if kind == "probe" else kind
        raise ServingError(f"no passed `serve {name}` in this session (required before a "
                           "GPU window)")
    return passed[-1]


def admit(session: Session, cap_usd: float, plan_fn: Callable[[float], dict[str, Any]],
          kind: str, environment: str = MODAL_ENVIRONMENT,
          accept_undetermined_probe: bool = False) -> dict[str, Any]:
    """One locked transaction: blockers (every window verified *and* settled),
    the cap, the reservation (window + build allowance) and the window record."""
    if not math.isfinite(cap_usd) or cap_usd < 0:
        raise ServingError("cap must be a finite nonnegative number")
    with session.locked() as data:
        problems = blockers(data)
        if problems:
            raise ServingError("; ".join(problems))
        if kind == "serve":
            probe = _require_passed(data, "probe", accept_undetermined_probe)
            _require_passed(data, "prefetch")
            if not probe["result"].get("ok"):
                data.setdefault("owner_accepted_undetermined_probe", []).append(
                    {"probe_window": probe["window_id"], "time_utc": _iso(_now())})
        if data["windows"] and not session.ledger_path.is_file():
            raise ServingError("the session's ledger is missing; refusing with unknown spend")
        plan = plan_fn(_now())
        records = read_records(session.ledger_path) if session.ledger_path.is_file() else []
        ledger_session = sum(1 for r in records if r.get("op") == "session")
        ledger = session.ledger(cap_usd, owner=f"serve-{kind}", session=ledger_session,
                                create=True)
        committed = ledger.snapshot()["committed_usd"]
        if committed + plan["reservation_usd"] > cap_usd + 1e-12:
            raise ServingError(
                f"cap ${cap_usd:.2f} cannot cover this window: ${plan['reservation_usd']:.6f} "
                f"(${plan['rate']['per_second']:.8f}/s x {plan['window_s']}s + allowances) "
                f"plus ${committed:.6f} already committed in this session; nothing is built "
                "or created")
        ledger.open_session(sum((w.get("settlement") or {}).get("charge_usd", 0.0)
                                for w in data["windows"]))
        rid = ledger.reserve("agent", f"{kind}:{plan['rate']['gpu']}", plan["reservation_usd"],
                             1)
        window_id = dt.datetime.fromtimestamp(plan["admitted_epoch"], dt.UTC).strftime(
            "w%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:6]}"
        window = {"window_id": window_id, "session": session.root.name, "status": "launching",
                  "sandbox_id": None, "create_called": False,
                  "served_model": SERVED_MODEL_NAME if kind == "serve" else None,
                  "spec": serving_spec(), "spec_sha256": spec_sha256(),
                  "ledger_session": ledger_session, "reservation_id": rid,
                  "modal_environment": environment, "settlement": None, **plan}
        if kind == "serve":
            window.update(gpu_preference=list(GPU_PREFERENCE),
                          gpu_requested=GPU_PREFERENCE[0], gpu_attempts=[], gpu_used=None)
        data["cap_usd"] = cap_usd
        data["windows"].append(window)
        return dict(window)


def create_locked(session: Session, window_id: str, control: Any, prepared: Any,
                  reanchor: bool = False,
                  precreate: Callable[[], dict[str, Any]] | None = None,
                  fallback: bool = False) -> tuple[str, int]:
    """Create the sandbox under the session lock, only from ``launching``, with
    ``timeout = deadline - now`` computed immediately before the request.
    ``create_called`` is written durably before the request is sent. With
    ``reanchor`` (probe, prefetch) the deadline is moved to now + window.
    ``fallback`` (a GPU fallback sandbox in the same window): the deadline,
    request and episode deadlines stay; the startup deadline becomes now +
    the window's own startup timeout, and must fall before the episode-start
    deadline."""
    with session.hold() as held:
        window = window_by_id(held.data, window_id)
        if window["status"] != "launching":
            raise ServingError(f"window {window_id} is {window['status']}: a stop arrived "
                               "before creation; nothing created")
        if precreate is not None:  # final capacity read, *before* the clock is read
            window["capacity_at_create"] = precreate()
        # Everything that can take time is done: the clock is read now, and the
        # timeout and the minimum are computed from it, right before the request.
        now = _now()
        if reanchor:
            window["deadline_epoch"] = now + window["window_s"]
            window["deadline_utc"] = _iso(window["deadline_epoch"])
            window["watchdog_fire_epoch"] = window["deadline_epoch"]
        timeout = min(window["window_s"], math.floor(window["deadline_epoch"] - now))
        if timeout < window["min_create_remaining_s"]:
            raise ServingError(f"only {timeout}s left before the window's deadline "
                               f"(minimum {window['min_create_remaining_s']}s); not creating")
        if fallback:
            startup = now + window["startup_timeout_s"]
            if startup >= window["episode_start_deadline_epoch"]:
                raise ServingError("no fallback: its startup deadline would not fall before "
                                   "the window's episode-start deadline; not creating")
            window["startup_deadline_epoch"] = startup
        window.update(create_called=True, create_timeout_s=timeout, issued_at_epoch=now,
                      create_requested_utc=_iso(now))
        held.write()
        try:
            sandbox_id = control.create(prepared, window, timeout, now)
        except BaseException as exc:
            window["create_error"] = type(exc).__name__
            held.write()
            raise
        window["sandbox_id"] = sandbox_id
        held.write()
        return sandbox_id, timeout


def _fail(session: Session, window_id: str, control: Any, exc: BaseException,
          token: str | None) -> ServingError:
    """Never raises. Redact, log the redacted traceback privately, stop, and
    terminate the window's sandbox by id unconditionally. If any cleanup step
    fails, the failure is recorded (redacted) and the window simply stays
    unverified with its reservation charged."""
    message = redact(f"{type(exc).__name__}: {exc}", token)[:300]
    cleanup: list[str] = []
    try:
        text = redact("".join(traceback.format_exception(exc)), token)
        with (session.private() / "launch_errors.log").open("a") as handle:
            handle.write(f"--- {_iso(_now())} {window_id}\n{text}\n")
    except BaseException as error:  # noqa: BLE001 - must not escape
        cleanup.append(f"log:{type(error).__name__}")
    try:
        _update(session, window_id, {"launch_error": message})
        stopped = stop_window(session, window_id, control, reason="launch_failed",
                              owner="start")
        sandbox_id = stopped.get("sandbox_id")
        if not sandbox_id and stopped.get("create_called"):
            with contextlib.suppress(Exception):
                sandbox_id = _sandbox_id(stopped, control.list_running())
        if sandbox_id:
            with contextlib.suppress(Exception):
                control.terminate(sandbox_id)  # unconditionally
            if not verify_finished(control, {**stopped, "sandbox_id": sandbox_id})[0] \
                    and stopped["status"] == "verified":
                _update(session, window_id, {"discrepancy": {
                    "found_utc": _iso(_now()), "open": True,
                    "detail": "sandbox still running after a failed launch"}})
    except BaseException as error:  # noqa: BLE001 - must not escape
        cleanup.append(f"cleanup:{type(error).__name__}:"
                       f"{redact(str(error), token)[:120]}")
    verdict = "NOT verified stopped"
    try:
        final = window_by_id(session.read(), window_id)
        if final["status"] == "verified" and not (final.get("discrepancy") or {}).get("open"):
            verdict = "verified stopped"
        if cleanup:
            _update(session, window_id, {"cleanup_errors": cleanup})
    except BaseException as error:  # noqa: BLE001 - must not escape
        cleanup.append(f"record:{type(error).__name__}")
    note = f"; cleanup failed ({', '.join(cleanup)})" if cleanup else ""
    return ServingError(f"launch failed ({message}); sandbox {verdict}{note}")


def _launch(session: Session, cap_usd: float, control: Any, plan_fn: Callable, kind: str,
            token_fn: Callable[[], str | None], spawn: Callable | None, reanchor: bool,
            accept_undetermined_probe: bool = False
            ) -> tuple[dict[str, Any], str, str | None, dict[str, Any], Any]:
    """Under the project lock: reconcile, capacity, admit, prepare, create.
    Returns the window, sandbox id, token, capacity and prepared objects."""
    with project_lock():
        reconcile(session, control, owner=f"{kind}-start")
        capacity = _capacity(control, kind)
        window = admit(session, cap_usd, plan_fn, kind, control.environment,
                       accept_undetermined_probe)
        wid, token = window["window_id"], None
        try:
            token = token_fn()
            if spawn is not None:
                _update(session, wid, {"watchdog_pid": spawn(session, wid,
                                                             control.environment)})
            prepared = control.prepare(window, token)
            sandbox_id, _ = create_locked(
                session, wid, control, prepared, reanchor=reanchor,
                precreate=lambda: _recheck_capacity(control, kind, capacity))
        except BaseException as exc:  # noqa: BLE001 - re-raised, redacted
            raise _fail(session, wid, control, exc, token) from None
    return window_by_id(session.read(), wid), sandbox_id, token, capacity, prepared


# ---------------------------------------------------------------- commands


ORIGIN_RESOLUTION_S = 5.0  # creation vs start lifetimes closer than this cannot be told apart


def probe_outcome(state: dict[str, Any], timing: dict[str, Any] | None, timeout_s: int,
                  early_s: float = PROBE_EARLY_TOLERANCE_S,
                  late_s: float = PROBE_LATE_TOLERANCE_S) -> dict[str, Any]:
    """A timeout measurement. ``ok`` only if Modal ended the sandbox by its
    timeout, the created->finished lifetime is within [timeout-early,
    timeout+late], and the evidence distinguishes creation from container
    start as the origin. ``undetermined`` (not ok) if ``started_at`` is
    missing, or the two lifetimes differ by less than ORIGIN_RESOLUTION_S, or
    neither fits; the uncertainty is recorded."""
    timing = timing or {}
    created, started, finished = (timing.get("created_at"), timing.get("started_at"),
                                  timing.get("finished_at"))
    by_timeout = state.get("status") == TIMEOUT_STATUS or (
        state.get("status") is None and state.get("returncode") == TIMEOUT_RETURNCODE)
    low, high = timeout_s - early_s, timeout_s + late_s
    from_created = finished - created if created and finished else None
    from_started = finished - started if started and finished else None

    def fits(value: float | None) -> bool:
        return value is not None and low <= value <= high

    separation = (None if from_created is None or from_started is None
                  else abs(from_created - from_started))
    if from_created is None:
        origin, why = "undetermined", "created_at or finished_at missing"
    elif from_started is None:
        origin, why = "undetermined", "started_at missing"
    elif separation < ORIGIN_RESOLUTION_S:
        origin, why = "undetermined", (f"start delay {separation:.1f}s is below the "
                                       f"{ORIGIN_RESOLUTION_S:.0f}s resolution")
    elif fits(from_created) and abs(from_created - timeout_s) < abs(from_started - timeout_s):
        origin, why = "creation", "created->finished matches the timeout"
    elif fits(from_started) and abs(from_started - timeout_s) < abs(from_created - timeout_s):
        origin, why = "container_start", "started->finished matches the timeout"
    else:
        origin, why = "undetermined", "neither lifetime matches the timeout"
    lifetime_ok = fits(from_created)
    ok = bool(state.get("finished")) and by_timeout and lifetime_ok and origin == "creation"
    return {"ok": ok, "timeout_s": timeout_s, "ended_by_timeout": by_timeout,
            "final_state": state, "created_at": created, "started_at": started,
            "finished_at": finished,
            "lifetime_from_created_s": None if from_created is None else round(from_created, 3),
            "lifetime_from_started_s": None if from_started is None else round(from_started, 3),
            "lifetime_within_range": lifetime_ok, "origin_separation_s": (
                None if separation is None else round(separation, 3)),
            "timeout_origin_supported": origin, "origin_reason": why,
            "accept_range_s": [low, high]}


def probe_timeout(out: str | Path, cap_usd: float, *, control: Any = None,
                  prices_path: str | Path = REPO_ROOT / "configs" / "prices.yaml"
                  ) -> dict[str, Any]:
    """Measure that Modal ends a sandbox by its timeout counted from creation."""
    control = control or ModalControl()
    session = Session(resolve_session_dir(out))
    rate = load_rate(prices_path, "probe")
    allowance = build_allowance_usd(prices_path)
    window, sandbox_id, _, _, _ = _launch(
        session, cap_usd, control, lambda now: plan_window(
            PROBE_TIMEOUT_S, rate, now, kind="probe", stop_margin_s=0, request_margin_s=0,
            episode_margin_s=0, startup_timeout_s=0, idle_s=0, fixed_charge_usd=allowance,
            headroom_usd=rate.usd(PROBE_LATE_TOLERANCE_S)),
        "probe", lambda: None, None, reanchor=True)
    wid = window["window_id"]
    try:
        _update(session, wid, status="serving", reason="probe_created")
        give_up = window["issued_at_epoch"] + window["create_timeout_s"] + PROBE_WAIT_EXTRA_S
        state: dict[str, Any] = {"finished": False}
        while _now() < give_up:
            state = control.state(sandbox_id)
            if state.get("finished"):
                break
            _sleep(5.0)
    except BaseException as exc:  # noqa: BLE001 - re-raised, redacted
        raise _fail(session, wid, control, exc, None) from None
    result = probe_outcome(state, _timing(control, window), window["create_timeout_s"])
    _update(session, wid, {"result": result})
    stopped = stop_window(session, wid, control, reason="probe_done", owner="probe")
    if not result["ok"]:
        raise ServingError(f"timeout probe FAILED: {json.dumps(result, default=str)}")
    return stopped


def prefetch(out: str | Path, cap_usd: float, *, control: Any = None,
             prices_path: str | Path = REPO_ROOT / "configs" / "prices.yaml",
             dry_run: bool = False) -> dict[str, Any]:
    """CPU sandbox: cache the weights in the Volume, check vLLM and the template."""
    control = control or ModalControl()
    session = Session(resolve_session_dir(out))
    rate = load_rate(prices_path, "prefetch")
    allowance = build_allowance_usd(prices_path)

    def plan_fn(now: float) -> dict[str, Any]:
        return plan_window(PREFETCH_TIMEOUT_S, rate, now, kind="prefetch", stop_margin_s=0,
                           request_margin_s=0, episode_margin_s=0, startup_timeout_s=0,
                           idle_s=0, fixed_charge_usd=allowance)

    if dry_run:
        return {"dry_run": True, "plan": plan_fn(_now())}
    window, sandbox_id, _, _, _ = _launch(session, cap_usd, control, plan_fn, "prefetch",
                                       lambda: None, None, reanchor=True)
    wid = window["window_id"]
    try:
        _update(session, wid, status="serving", reason="prefetch_created")
        state: dict[str, Any] = {"finished": False}
        while _now() < window["deadline_epoch"] + 60:
            state = control.state(sandbox_id)
            if state.get("finished"):
                break
            _sleep(10.0)
        lines = [line for line in str(control.output(sandbox_id)).splitlines()
                 if line.startswith("SLH_RESULT ")]
        result = json.loads(lines[-1][len("SLH_RESULT "):]) if lines else {
            "ok": False, "problems": ["no_result_line"]}
    except BaseException as exc:  # noqa: BLE001 - re-raised, redacted
        raise _fail(session, wid, control, exc, None) from None
    result["final_state"] = state
    _update(session, wid, {"result": result})
    stopped = stop_window(session, wid, control, reason="prefetch_done", owner="prefetch")
    if not result.get("ok"):
        raise ServingError(f"prefetch checks failed: {result.get('problems')}")
    return stopped


def start(out: str | Path, window_s: int, cap_usd: float, *, control: Any = None,
          prices_path: str | Path = REPO_ROOT / "configs" / "prices.yaml",
          allow_over_3h: bool = False, dry_run: bool = False,
          measure_timeout: bool = False, accept_undetermined_probe: bool = False,
          spawn: Callable[..., int] = spawn_watchdog,
          request: Callable[..., tuple[int, Any]] = http_request,
          **margins: int) -> dict[str, Any]:
    """Under the project lock: reconcile, capacity, admit, watchdog, prepare,
    create; then verify ingress and serve.

    ``measure_timeout`` (the mandatory GPU smoke measurement; window <= 30 min,
    cap <= $2): no idle exit, no in-sandbox early exit, watchdog only after the
    deadline, so Modal's timeout ends the sandbox; the outcome is recorded as
    ``timeout_measurement`` when the window is settled."""
    control = control or ModalControl()
    session = Session(resolve_session_dir(out))
    rate = load_rate(prices_path, "serve")
    allowance = build_allowance_usd(prices_path)

    def plan_fn(now: float) -> dict[str, Any]:
        # In measurement mode our backstops sit after the deadline (in-sandbox
        # exit +120 s, watchdog +300 s): the reservation covers the latter.
        return plan_window(window_s, rate, now, allow_over_3h=allow_over_3h,
                           fixed_charge_usd=allowance, measure_timeout=measure_timeout,
                           headroom_usd=rate.usd(MEASURE_GRACE_S) if measure_timeout else 0.0,
                           **margins)

    if measure_timeout and (window_s > MEASURE_MAX_WINDOW_S or cap_usd > MEASURE_MAX_CAP_USD):
        raise ServingError(f"--measure-timeout needs a window <= {MEASURE_MAX_WINDOW_S // 60} "
                           f"min and a cap <= ${MEASURE_MAX_CAP_USD:.2f}")
    plan = plan_fn(_now())  # validate before any Modal call
    if dry_run:  # read only: the environment, its limits and current usage
        check_environment(session, control)
        return {"dry_run": True, "plan": plan, "spec": serving_spec(),
                "capacity": _capacity(control, "serve"),
                "blockers": blockers(session.read()),
                "sandbox_request": sandbox_request(
                    {**plan, "window_id": "<id>", "session": session.root.name},
                    window_s, plan["admitted_epoch"])}
    window, sandbox_id, token, capacity, prepared = _launch(
        session, cap_usd, control, plan_fn, "serve", session.new_token, spawn, reanchor=False,
        accept_undetermined_probe=accept_undetermined_probe)
    wid = window["window_id"]
    try:
        _update(session, wid, {"capacity_before": capacity})
        began = _now()
        try:
            endpoint, sandbox_id = wait_tunnel_or_fall_back(
                session, wid, control, prepared, sandbox_id, capacity)
        finally:
            _update(session, wid, {"tunnel_wait_s": round(_now() - began, 3)})
        window = window_by_id(session.read(), wid)  # a fallback moves the startup deadline
        _update(session, wid, {"endpoint_url": endpoint})
        ingress = check_ingress(
            endpoint, token, SERVED_MODEL_NAME, window["startup_deadline_epoch"],
            alive=lambda: not control.state(sandbox_id).get("finished"), request=request)
        _update(session, wid, {"ingress": ingress, "gpu_used": window["gpu_requested"]},
                status="serving", reason="healthy")
    except BaseException as exc:  # noqa: BLE001 - re-raised, redacted
        raise _fail(session, wid, control, exc, token) from None
    return window_by_id(session.read(), wid)


def owner_clear(out: str | Path, window_id: str, note: str,
                control: Any = None) -> dict[str, Any]:
    """The owner's acknowledgement after checking the Modal dashboard: clears
    an excess, or closes an unverifiable window at its full reservation. A
    sandbox listed running is never cleared."""
    if len(note.strip()) < 10:
        raise ServingError("a note of what was checked (at least 10 characters) is required")
    control = control or ModalControl()
    session = Session(resolve_session_dir(out))
    check_environment(session, control)
    running = {r["sandbox_id"] for r in control.list_running()}
    with session.locked() as data:
        window = window_by_id(data, window_id)
        if window.get("sandbox_id") in running:
            raise ServingError("that window's sandbox is listed running; run `serve stop`")
        record = {"note": note.strip(), "time_utc": _iso(_now())}
        window.setdefault("owner_notes", []).append(record)  # always, append-only
        if window.get("excess") and not window["excess"].get("cleared"):
            window["excess"].update(cleared=True, **record)
        if window["status"] != "verified":
            for step in ("stopping", "stopped"):
                if step in TRANSITIONS[window["status"]]:
                    transition(window, step, "owner_clear")
            result = settlement({**window, "create_called": True}, None)  # full reservation
            result["ledger_written"] = settle_once(session, data.get("cap_usd", 0.0), window,
                                                   result["charge_usd"], "owner-clear")
            window["settlement"] = result
            transition(window, "verified", "owner_clear")
            window["owner_verified"] = record
        return dict(window)


# ---------------------------------------------------------------- command line


SERVING_CONFIG = REPO_ROOT / "configs" / "serving.yaml"


def modal_environment(config_path: str | Path | None = None, override: str | None = None,
                      allow_other: bool = False) -> str:
    """The Modal environment: configs/serving.yaml (or ``--modal-env``). It
    must be the dedicated ``self-learning-harness`` environment unless
    ``--allow-other-modal-env`` is given, and then GPU windows are refused."""
    value = override
    path = Path(config_path or SERVING_CONFIG)
    if value is None and path.is_file():
        import yaml

        value = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get(
            "modal_environment")
        if not isinstance(value, str) or not value:
            raise ServingError(f"{path}: modal_environment must be a nonempty string")
    value = value or MODAL_ENVIRONMENT
    if value != MODAL_ENVIRONMENT and not allow_other:
        raise ServingError(f"Modal environment must be {MODAL_ENVIRONMENT!r} (got {value!r}); "
                           "--allow-other-modal-env permits CPU work only")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m slh serve", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serving-config", type=Path, default=None,
                        help="default: configs/serving.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe-timeout", help="CPU sandbox, timeout 120 s: measure "
                                "that Modal ends it by timeout counted from creation")
    probe.add_argument("--out", required=True, type=Path)
    probe.add_argument("--cap", required=True, type=float, metavar="USD")
    pre = commands.add_parser("prefetch", help="CPU sandbox: cache weights, check vLLM")
    pre.add_argument("--out", required=True, type=Path)
    pre.add_argument("--cap", required=True, type=float, metavar="USD")
    pre.add_argument("--dry-run", action="store_true")
    for name in ("start", "plan"):
        cmd = commands.add_parser(name, help="launch a GPU window" if name == "start" else
                                  "free dry run of start: validation, costs, exact request")
        cmd.add_argument("--out", required=True, type=Path)
        cmd.add_argument("--window-min", required=True, type=int)
        cmd.add_argument("--cap", required=True, type=float, metavar="USD",
                         help="cap for this session (all its windows, cumulative)")
        cmd.add_argument("--allow-over-3h", action="store_true")
        cmd.add_argument("--measure-timeout", action="store_true",
                         help="GPU timeout measurement: nothing of ours ends the sandbox "
                              "(window <= 30 min, cap <= $2)")
        cmd.add_argument("--accept-undetermined-probe", action="store_true",
                         help="owner: accept a CPU probe whose timeout origin is undetermined")
        cmd.add_argument("--startup-timeout-min", type=int, default=None)
        cmd.add_argument("--idle-min", type=int, default=DEFAULT_IDLE_S // 60)
        cmd.add_argument("--stop-margin-min", type=int, default=DEFAULT_STOP_MARGIN_S // 60)
        cmd.add_argument("--episode-margin-min", type=int,
                         default=DEFAULT_EPISODE_MARGIN_S // 60)
    stop = commands.add_parser("stop", help="stop unverified windows, verify, settle, fix "
                               "discrepancies")
    stop.add_argument("--out", required=True, type=Path)
    stat = commands.add_parser("status", help="reconcile and verify what is running")
    stat.add_argument("--out", required=True, type=Path)
    clear = commands.add_parser("owner-clear", help="owner acknowledges an excess or an "
                                "unverifiable window after checking the Modal dashboard")
    clear.add_argument("--out", required=True, type=Path)
    clear.add_argument("--window", required=True)
    clear.add_argument("--note", required=True)
    watch = commands.add_parser("watchdog", help=argparse.SUPPRESS)
    watch.add_argument("--out", required=True, type=Path)
    watch.add_argument("--window", required=True)
    for sub in commands.choices.values():
        sub.add_argument("--modal-env", default=None,
                         help="Modal environment (default: configs/serving.yaml)")
        sub.add_argument("--allow-other-modal-env", action="store_true",
                         help="permit a non-dedicated environment (CPU work only)")
    return parser


def main(argv: list[str] | None = None, control: Any = None) -> int:
    args = _parser().parse_args(argv)
    session = Session(resolve_session_dir(args.out))
    try:
        if control is None:
            environment = modal_environment(args.serving_config, args.modal_env,
                                            args.allow_other_modal_env)
            control = ModalControl(environment, args.allow_other_modal_env)
        check_environment(session, control)
        if args.command == "probe-timeout":
            window = probe_timeout(args.out, args.cap, control=control)
            print(json.dumps({"window_id": window["window_id"], "status": window["status"],
                              "result": window.get("result")}, indent=2, default=str))
            return 0
        if args.command == "prefetch":
            result = prefetch(args.out, args.cap, control=control, dry_run=args.dry_run)
            print(json.dumps({k: result.get(k) for k in (
                "dry_run", "plan", "window_id", "status", "result", "settlement")
                if k in result}, indent=2, default=str))
            return 0
        if args.command in ("start", "plan"):
            margins = {"idle_s": args.idle_min * 60, "stop_margin_s": args.stop_margin_min * 60,
                       "episode_margin_s": args.episode_margin_min * 60}
            if args.startup_timeout_min is not None:
                margins["startup_timeout_s"] = args.startup_timeout_min * 60
            result = start(args.out, args.window_min * 60, args.cap, control=control,
                           allow_over_3h=args.allow_over_3h, dry_run=args.command == "plan",
                           measure_timeout=args.measure_timeout,
                           accept_undetermined_probe=args.accept_undetermined_probe,
                           **margins)
            if args.command == "plan":
                print(json.dumps(result, indent=2, default=str))
                return 0
            print(f"serving {result['served_model']} at {result['endpoint_url']}/v1 "
                  f"(sandbox {result['sandbox_id']}, GPU {result.get('gpu_used')}, "
                  f"timeout {result['create_timeout_s']}s); "
                  f"requests stop {result['request_deadline_utc']}, watchdog "
                  f"{result['watchdog_fire_utc']}, deadline {result['deadline_utc']}; "
                  f"reserved ${result['reservation_usd']:.6f}")
            return 0
        if args.command == "watchdog":
            print(f"{_iso(_now())} watchdog: {watchdog_loop(session, args.window, control)}",
                  flush=True)
            return 0
        if args.command == "owner-clear":
            window = owner_clear(args.out, args.window, args.note, control=control)
            print(json.dumps({k: window.get(k) for k in (
                "window_id", "status", "excess", "owner_verified", "settlement")},
                indent=2, default=str))
            return 0
        if args.command == "stop":
            ok = True
            for window in (session.read() or {"windows": []})["windows"]:
                if window["status"] != "verified":
                    done = stop_window(session, window["window_id"], control, reason="manual")
                    ok &= done["status"] == "verified"
            resolve_discrepancies(session, control)
            check = status(session, control)
            print(json.dumps(check, indent=2, default=str))
            return 0 if ok and check["closed_windows_all_stopped"] and not any(
                "discrepancy" in b for b in check["blockers"]) else 1
        check = status(session, control)
        print(json.dumps(check, indent=2, default=str))
        return 0 if check["closed_windows_all_stopped"] else 1
    except (ServingError, BudgetExceeded) as exc:
        print(redact(f"serve {args.command} refused: {exc}", session.token_or_none()),
              file=sys.stderr)
        return 2
