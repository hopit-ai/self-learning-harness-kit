"""The paid provider behind the user simulator and judge: preflight, error
classes and the credit-failure stop (docs/protocol-amendment-5.md).

``python -m slh provider preflight | credit-stop | classify``

Nothing here is part of the execution surface (``slh.execution``
``RUNTIME_MODULES``): it never shapes an episode. The preflight runs BEFORE
any GPU starts; the credit stop runs beside a window stage and acts only
through the run's existing shared spend ledger (a ``halt`` record, the same
mechanism a worker's own refusal uses), so no runtime file changes.

**Preflight** (``preflight``): reads the provider account's available balance
(``GET <api>/credits``: total credits minus total usage) and the key's
remaining limit (``GET <api>/key``: ``limit_remaining``, null when the key has
no limit); the allowance (the smaller of the two) must cover the window's
token reservation plus the synthetic request's bound. Then exactly one metered
synthetic request goes through the exact user-simulator route: the same model
string, the same request arguments as ``slh.agent.LiteLLMBackend`` builds them
(provider pin and fallback setting, reasoning switch, temperature and the full
configured ``max_tokens``, never a reduced output allowance), with its cost
bounded before sending and charged from the returned usage afterwards. The
record holds counts, prices and classes only: never the key, the account id or
the provider's error text.

**Error classes** (``classify_message``): ``provider_credit`` (HTTP 402 with
credit wording: the account cannot afford the request), ``provider_rate_limit``
(429), ``provider_server`` (5xx, connection, timeout), ``provider_auth`` (401,
403) or ``other``. ``pair_category`` maps one recorded outcome and its private
error record to a registered pair category; ``INFRASTRUCTURE_CATEGORIES`` are
the categories that are verified infrastructure, never task outcomes.

**Credit stop** (``credit_stop``): watches the error records of the run
directories under its roots. On a confirmed account-wide credit error (a
``provider_credit`` record) it appends one ``halt`` record (owner
``credit-stop``) to the shared ledger of every running run under the roots:
every worker is refused at its next call and dispatches no further episode;
the coordinator drains, merges and writes the manifest (``not_run``). It then
writes a flag file, so the window script skips later stages and stops the GPU.
Pairs never dispatched are recorded as missing by the learner and diagnostic
code that reads the run (``pair_category``: ``not_dispatched`` or
``credit_stop_halt``), never as failures. It prints counts only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Self

REPO = Path(__file__).resolve().parent.parent

ERROR_CLASSES = ("provider_credit", "provider_rate_limit", "provider_server", "provider_auth",
                 "other")
INFRASTRUCTURE_CATEGORIES = ("provider_credit", "provider_rate_limit", "provider_server",
                             "serving_window_closed", "credit_stop_halt",
                             "credit_stop_interrupted", "not_dispatched")
PAIR_CATEGORIES = ("valid", *INFRASTRUCTURE_CATEGORIES, "provider_auth", "budget_refused",
                   "unverified", "other_error")
CREDIT_STOP_OWNER = "credit-stop"
CREDIT_STOP_REASON = "provider_credit_exhausted"
PREFLIGHT_SCHEMA = "provider-preflight-v1"
PREFLIGHT_MAX_COST_USD = 0.01
SYNTHETIC_MESSAGES = (
    {"role": "system", "content": "You are a customer contacting an online retail store's "
                                  "support agent. Reply in one short sentence."},
    {"role": "user", "content": "Hello, how can I help you today?"})

_STATUS = re.compile(r"\"code\"\s*:\s*(\d{3})\b|\bstatus(?:_code)?\s*[=:]\s*(\d{3})\b|"
                     r"\b(?:Error code|HTTP|status)\s*:?\s*(\d{3})\b")
_CREDIT = re.compile(r"requires more credits|insufficient (?:credits|balance|funds)|"
                     r"add more credits|out of credits|credit balance|_credits\"|"
                     r"can only afford", re.IGNORECASE)
_RATE = ("RateLimitError",)
_SERVER = ("ServiceUnavailableError", "InternalServerError", "APIConnectionError",
           "Timeout", "TimeoutError", "BadGatewayError", "ReadTimeout", "ConnectTimeout")
_AUTH = ("AuthenticationError", "PermissionDeniedError")


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------------------------- error classes


def _statuses(message: str) -> set[int]:
    return {int(next(g for g in match if g)) for match in _STATUS.findall(message or "")}


def classify_message(message: str | None, type_name: str | None = None) -> str:
    """One of ERROR_CLASSES for a recorded provider error message."""
    message = message or ""
    statuses = _statuses(message)
    if 402 in statuses and _CREDIT.search(message):
        return "provider_credit"
    if 429 in statuses or type_name in _RATE:
        return "provider_rate_limit"
    if any(500 <= s <= 599 for s in statuses) or type_name in _SERVER:
        return "provider_server"
    if statuses & {401, 403} or type_name in _AUTH:
        return "provider_auth"
    return "other"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, NotADirectoryError):
        return rows
    for line in text.split("\n"):  # LF only: U+2028/U+2029 are valid inside strings
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def error_files(run_dir: Path) -> list[Path]:
    """The run's private error records: the merged file once the coordinator
    has merged, else every worker's own file."""
    merged = run_dir / "private" / "errors.jsonl"
    if merged.is_file():
        return [merged]
    try:
        return sorted((run_dir / "private" / "workers").glob("*/errors.jsonl"))
    except (PermissionError, OSError):
        return []


def error_records(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = {}
    for path in error_files(run_dir):
        for row in _jsonl(path):
            found.setdefault(str(row.get("episode_id")), []).append(row)
    return found


def episode_records(run_dir: Path) -> dict[str, dict[str, Any]]:
    """The run's merged episode records by episode id (first record wins)."""
    found: dict[str, dict[str, Any]] = {}
    for row in _jsonl(run_dir / "episodes.jsonl"):
        found.setdefault(str(row.get("episode_id")), row)
    return found


def credit_halted(run_dir: Path) -> bool:
    """True if the credit stop halted this run's ledger."""
    return any(r.get("op") == "halt" and r.get("owner") == CREDIT_STOP_OWNER
               for r in _jsonl(run_dir / "ledger.jsonl"))


def record_matches(outcome: dict[str, Any], record: dict[str, Any]) -> bool:
    """The private record is the one behind this outcome's public error: same
    code and type, and its message hashes to the outcome's recorded hash."""
    error = outcome.get("error") if isinstance(outcome.get("error"), dict) else {}
    message = record.get("message")
    return (isinstance(message, str) and error.get("code") == record.get("code")
            and error.get("type") == record.get("type")
            and error.get("sha256") == record.get("sha256")
            == hashlib.sha256(message.encode("utf-8", "replace")).hexdigest())


def _halt_refusal(record: dict[str, Any]) -> bool:
    """The private record of a refusal caused by the credit stop's ledger halt
    (``SharedLedger`` names the halting owner and reason in its message)."""
    message = record.get("message")
    return isinstance(message, str) and (
        f"run halted by {CREDIT_STOP_OWNER}: {CREDIT_STOP_REASON}" in message)


def _no_model_call(episode: dict[str, Any] | None) -> bool:
    """True if the episode's record shows no token and no spend for any role:
    it was refused at its first model call, so nothing was dispatched."""
    if not isinstance(episode, dict):
        return False
    usage = episode.get("usage")
    return (isinstance(usage, dict) and bool(usage)
            and all(v == 0 for v in usage.values() if isinstance(v, int | float))
            and episode.get("usd") in (0, 0.0))


def pair_category(verdict: str, outcome: dict[str, Any] | None,
                  records: list[dict[str, Any]], *, halted: bool,
                  episode: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """(category, detail) of one registered pair from its verified outcome
    (``diagnostics.verified_outcomes`` verdict), its private error records and
    its episode record. ``detail`` carries the stage, code, type and error hash
    only.

    A budget refusal is ``credit_stop_halt`` (never dispatched, missing) only
    if the run's ledger carries the credit stop's halt, the refusal is bound
    to its private record naming that halt, it happened at stage
    ``simulation``, and the episode made no model call at all. A bound halt
    refusal after a model call or at grading is ``credit_stop_interrupted``
    (verified infrastructure, not missing). Any other refusal is the run's own
    cap (``budget_refused``), never infrastructure."""
    if verdict == "missing":
        return "not_dispatched", {}
    if verdict == "valid":
        return "valid", {}
    if verdict != "invalid" or outcome is None:
        return "unverified", {}
    error = outcome.get("error") if isinstance(outcome.get("error"), dict) else {}
    code = error.get("code")
    matched = [r for r in records if record_matches(outcome, r)]
    detail = {"code": code, "type": error.get("type"), "error_sha256": error.get("sha256"),
              "stage": matched[0].get("stage") if matched else None}
    if code == "serving_window_closed" and matched:
        return "serving_window_closed", detail
    if code in ("budget_refused", "grading_budget_refused"):
        if not (halted and matched and _halt_refusal(matched[0])):
            return "budget_refused", detail
        if code == "budget_refused" and detail["stage"] == "simulation" and \
                _no_model_call(episode):
            return "credit_stop_halt", detail
        return "credit_stop_interrupted", detail
    if not matched:
        return "unverified", detail
    if code in ("simulation_crash", "grading_failed"):
        cls = classify_message(matched[0].get("message"), matched[0].get("type"))
        detail["error_class"] = cls
        return (cls if cls != "other" else "other_error"), detail
    return "other_error", detail


# --------------------------------------------------------------------------- route and preflight


def user_route(runner_config: str | Path) -> dict[str, Any]:
    """The user simulator's exact request route: model string and the request
    arguments ``LiteLLMBackend`` sends (per-trial seed aside)."""
    from slh.agent import LiteLLMBackend
    from slh.runner import load_config

    config = load_config(runner_config)
    if config.user.backend != "litellm":
        # Offline stand-ins (stub) have a route too, so their hash can be
        # required; the preflight itself refuses them.
        return {"model": config.user.model, "kwargs": {}, "backend": config.user.backend,
                "api_base": None, "api_key_env": None}
    kwargs = LiteLLMBackend(config.user, call_name="user_response").request_kwargs(None)
    return {"model": config.user.model, "kwargs": kwargs, "backend": "litellm",
            "api_base": config.user.api_base, "api_key_env": config.user.api_key_env}


def route_sha256(route: dict[str, Any]) -> str:
    if route.get("backend", "litellm") != "litellm":
        return digest({"model": route["model"], "backend": route["backend"]})
    return digest({"model": route["model"], "kwargs": route["kwargs"]})


def resolve_endpoint(route: dict[str, Any]) -> tuple[str, str]:
    """(API base, key environment variable) exactly as the simulator's client
    library resolves them for the route's provider prefix: the base from
    ``<PREFIX>_API_BASE`` if set, else the provider's default; the key from
    ``<PREFIX>_API_KEY``. No override: the allowance checks and the synthetic
    request use the same resolved key and base, and the runs use them too."""
    prefix = route["model"].split("/", 1)[0]
    base = os.environ.get(f"{prefix.upper()}_API_BASE") or f"https://{prefix}.ai/api/v1"
    return base.rstrip("/"), f"{prefix.upper()}_API_KEY"


def _http_get_json(url: str, key: str, timeout: float = 20.0) -> tuple[int, Any]:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, None


def _litellm_completion(**kwargs: Any) -> Any:
    import litellm

    return litellm.completion(**kwargs)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    get = (usage.get if isinstance(usage, dict) else lambda k, d=None: getattr(usage, k, d))
    return int(get("prompt_tokens", 0) or 0), int(get("completion_tokens", 0) or 0)


def _finish(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return None
    first = choices[0]
    reason = first.get("finish_reason") if isinstance(first, dict) else getattr(
        first, "finish_reason", None)
    return reason if isinstance(reason, str) and reason.isidentifier() else None


def preflight(*, runner_configs: list[str | Path], reservation_usd: float,
              max_cost_usd: float = PREFLIGHT_MAX_COST_USD,
              prices_path: str | Path | None = None,
              http_get: Callable[[str, str], tuple[int, Any]] | None = None,
              completion: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Balance and key allowance against the window's token reservation, then
    one metered synthetic request through the exact user-simulator route.
    Every runner config must use the same route. Returns the record
    (``ok`` true only if every check passed)."""
    from slh.budget import input_token_bound, load_prices
    from slh.runner import load_config

    if not runner_configs:
        raise ValueError("at least one runner config is required")
    if not (isinstance(reservation_usd, int | float) and 0 <= reservation_usd < float("inf")):
        raise ValueError("the window token reservation must be a finite nonnegative amount")
    routes = [user_route(c) for c in runner_configs]
    if any(r["backend"] != "litellm" for r in routes):
        raise ValueError("the provider preflight checks a hosted (litellm) user simulator")
    if any(r.get("api_base") or r.get("api_key_env") for r in routes):
        raise ValueError("the runs ignore a configured api_base / api_key_env for the "
                         "simulator; the preflight would check another endpoint")
    shas = {route_sha256(r) for r in routes}
    if len(shas) != 1:
        raise ValueError("the runner configs use different user-simulator routes")
    route = routes[0]
    record: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA, "checked_utc": _now(), "ok": False,
        "route_sha256": shas.pop(), "route": {"model": route["model"],
                                              "kwargs": route["kwargs"]},
        "runner_configs": [str(Path(c)) for c in runner_configs],
        "reservation_usd": round(float(reservation_usd), 6), "failures": []}
    config = load_config(runner_configs[0])
    prices = load_prices(prices_path or config.prices_path)
    price = prices.get(route["model"])
    if price is None:
        record["failures"].append("no price entry for the user-simulator model")
        return record
    max_tokens = route["kwargs"].get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens < 1:
        record["failures"].append("the user-simulator route has no max_tokens bound")
        return record
    request_bound = round((input_token_bound(list(SYNTHETIC_MESSAGES)) * price.input
                           + max_tokens * price.output) / 1e6, 9)
    record["synthetic_bound_usd"] = request_bound
    if request_bound > max_cost_usd:
        record["failures"].append("the synthetic request's bound exceeds the preflight cap")
        return record
    base, env = resolve_endpoint(route)
    record["endpoint_sha256"] = hashlib.sha256(base.encode()).hexdigest()
    key = os.environ.get(env)
    if not key:
        record["failures"].append("the provider API key is not set")
        return record
    get = http_get or _http_get_json
    allowance: dict[str, Any] = {}
    status, body = get(f"{base}/credits", key)
    data = (body or {}).get("data") if isinstance(body, dict) else None
    total, used = (_number((data or {}).get("total_credits")),
                   _number((data or {}).get("total_usage")))
    if status != 200 or total is None or used is None:
        record["failures"].append(f"account balance unavailable (HTTP {status})")
    else:
        allowance["account_available_usd"] = round(total - used, 6)
    status, body = get(f"{base}/key", key)
    data = (body or {}).get("data") if isinstance(body, dict) else None
    if status != 200 or not isinstance(data, dict):
        record["failures"].append(f"key allowance unavailable (HTTP {status})")
    else:
        remaining = data.get("limit_remaining")
        allowance["key_limit_remaining_usd"] = (None if remaining is None
                                                else _number(remaining))
        if remaining is not None and _number(remaining) is None:
            record["failures"].append("key allowance is not a number")
    record["allowance"] = allowance
    if record["failures"]:
        return record
    values = [v for v in (allowance["account_available_usd"],
                          allowance["key_limit_remaining_usd"]) if v is not None]
    available = min(values)
    record["allowance_usd"] = round(available, 6)
    needed = round(float(reservation_usd) + request_bound, 6)
    record["required_usd"] = needed
    if available < needed:
        record["failures"].append("the available allowance is below the window token "
                                  "reservation")
        return record
    kwargs = dict(route["kwargs"])
    kwargs["seed"] = config.seed
    # The same resolved key and base as the allowance checks (and as the runs).
    kwargs["api_key"], kwargs["api_base"] = key, base
    synthetic: dict[str, Any] = {"max_tokens": max_tokens}
    try:
        response = (completion or _litellm_completion)(
            model=route["model"], messages=[dict(m) for m in SYNTHETIC_MESSAGES], **kwargs)
    except Exception as exc:  # noqa: BLE001 - classified, never echoed
        cls = classify_message(str(exc), type(exc).__name__)
        synthetic.update(ok=False, error_class=cls, error_type=type(exc).__name__
                         if type(exc).__name__.isidentifier() else "other")
        # A failed attempt may still be billed: charged at its full bound.
        synthetic["charged_usd"] = request_bound
        record["synthetic"] = synthetic
        record["failures"].append(f"synthetic request failed ({cls})")
        return record
    prompt, output = _usage(response)
    cost = round((prompt * price.input + output * price.output) / 1e6, 9)
    synthetic.update(ok=True, prompt_tokens=prompt, completion_tokens=output,
                     charged_usd=cost, finish_reason=_finish(response))
    record["synthetic"] = synthetic
    if cost > request_bound + 1e-12:
        record["failures"].append("the synthetic request cost more than its bound")
        return record
    record["ok"] = True
    return record


def registry_event(record: dict[str, Any]) -> dict[str, Any]:
    """The ``provider_preflight`` registry row (classes and numbers only)."""
    synthetic = record.get("synthetic") or {}
    return {"event": "provider_preflight", "schema": PREFLIGHT_SCHEMA, "ok": record["ok"],
            "checked_utc": record["checked_utc"], "route_sha256": record["route_sha256"],
            "reservation_usd": record["reservation_usd"],
            "allowance_usd": record.get("allowance_usd"),
            "synthetic_bound_usd": record.get("synthetic_bound_usd"),
            "synthetic_ok": synthetic.get("ok"),
            "synthetic_charged_usd": synthetic.get("charged_usd"),
            "synthetic_error_class": synthetic.get("error_class"),
            "failures": list(record.get("failures") or [])}


def require_preflight(rows: list[dict[str, Any]], route_sha: str, *, after: dict[str, Any],
                      max_age_s: int, now: float | None = None) -> dict[str, Any]:
    """The latest ``provider_preflight`` registry row after ``after`` (the
    authorizing amendment): it must have passed, for this exact route, within
    ``max_age_s``. Raises ValueError otherwise."""
    if after not in rows:
        raise ValueError("the authorizing amendment is not in the registry")
    found = [r for r in rows[rows.index(after) + 1:] if r.get("event") == "provider_preflight"]
    if not found:
        raise ValueError("no provider preflight is recorded after the authorizing amendment")
    latest = found[-1]
    if latest.get("ok") is not True:
        raise ValueError("the latest provider preflight did not pass")
    if latest.get("route_sha256") != route_sha:
        raise ValueError("the provider preflight checked another user-simulator route")
    try:
        checked = dt.datetime.fromisoformat(latest["checked_utc"]).timestamp()
    except (KeyError, TypeError, ValueError):
        raise ValueError("the provider preflight has no valid time") from None
    age = (time.time() if now is None else now) - checked
    if not 0 <= age <= max_age_s:
        raise ValueError(f"the provider preflight is {int(age)} s old (limit {max_age_s} s)")
    return latest


# --------------------------------------------------------------------------- credit stop


def _timestamp(value: Any) -> float | None:
    try:
        return dt.datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return None


def _run_dirs(roots: Iterable[Path]) -> list[Path]:
    found = []
    for root in roots:
        root = Path(root)
        try:
            # A run directory: a shared ledger (parallel) or a manifest (any run).
            for name in ("ledger.jsonl", "manifest.json"):
                if (root / name).is_file():
                    found.append(root)
                for marker in root.rglob(name):
                    if marker.parent != root and marker.parent not in found:
                        found.append(marker.parent)
        except (PermissionError, OSError):
            continue
    return sorted(set(found))


def _created(run_dir: Path) -> float | None:
    """When the run was created: its manifest's ``created_utc``, else its
    ledger's first record (a run starting up)."""
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        created = _timestamp(manifest.get("created_utc"))
        if created is not None:
            return created
    except (OSError, ValueError):
        pass
    first = next(iter(_jsonl(run_dir / "ledger.jsonl")), None)
    return None if first is None else _timestamp(first.get("time_utc"))


def window_runs(roots: Iterable[Path], since: float) -> list[Path]:
    """Run directories under ``roots`` created at or after ``since`` (this
    window's runs only; finished historical runs are never read)."""
    # Records carry whole seconds: ``since`` counts from the start of its second.
    return [run for run in _run_dirs(roots) if (_created(run) or -1.0) >= int(since)]


def _current_session(run_dir: Path) -> int | None:
    sessions = [r.get("session") for r in _jsonl(run_dir / "ledger.jsonl")
                if r.get("op") == "session" and isinstance(r.get("session"), int)]
    return max(sessions) if sessions else None


def _running(run_dir: Path) -> bool:
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True  # starting: no manifest yet, but a ledger exists
    return manifest.get("status") == "running"


def halt_run(run_dir: Path) -> bool:
    """Append the credit-stop halt to a running run's shared ledger (the
    session's existing halt mechanism). Returns True if a halt was written."""
    from slh.ledger import SharedLedger

    session = _current_session(run_dir)
    if session is None or credit_halted(run_dir) or not _running(run_dir):
        return False
    ledger = SharedLedger(run_dir / "ledger.jsonl", 0.0, owner=CREDIT_STOP_OWNER,
                          session=session, create=False)
    ledger.halt(CREDIT_STOP_REASON)
    return True


def credit_seen(run_dirs: Iterable[Path], since: float) -> bool:
    """A provider-credit error record written at or after ``since``."""
    return any(classify_message(r.get("message"), r.get("type")) == "provider_credit"
               and (_timestamp(r.get("time_utc")) or -1.0) >= int(since)
               for run in run_dirs for path in error_files(run) for r in _jsonl(path))


def _write_flag(flag: Path, body: dict[str, Any]) -> None:
    tmp = flag.with_name(flag.name + ".tmp")
    tmp.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(flag)


def credit_scan(roots: list[Path], *, flag: Path, since: float, state: dict[str, Any],
                out: Callable[[str], None] = print) -> dict[str, Any]:
    """One scan: confirm a credit error of this window's runs (or an existing
    flag), then halt every running run of this window under ``roots``."""
    runs = window_runs(roots, since)
    if not state.get("credit_stop") and (flag.is_file() or credit_seen(runs, since)):
        state["credit_stop"] = True
        state["confirmed_utc"] = _now()
        out("credit stop: account-wide provider credit error confirmed; halting runs")
    if state.get("credit_stop"):
        halted = sum(1 for run in runs if halt_run(run))
        if halted or not flag.is_file():
            state["runs_halted"] = state.get("runs_halted", 0) + halted
            _write_flag(flag, {"credit_stop": True, "confirmed_utc": state.get("confirmed_utc"),
                               "runs_halted": state["runs_halted"],
                               "reason": CREDIT_STOP_REASON})
            if halted:
                out(f"credit stop: halted {halted} running run(s)")
    return state


def credit_stop(roots: list[str | Path], *, flag: str | Path, since: float | None = None,
                poll_s: float = 2.0, once: bool = False, max_polls: int | None = None,
                ready: str | Path | None = None,
                out: Callable[[str], None] = print) -> dict[str, Any]:
    """Watch ``roots`` (never a sealed tree); on a confirmed credit error in a
    run created at or after ``since`` (default: now) halt every running run of
    this window under them, write ``flag`` and keep halting runs that start
    later. ``ready``: written once watching has begun (pid and start time).
    ``once``: a single scan (tests and manual checks)."""
    from slh import sealing

    roots = [Path(r) for r in roots]
    for root in roots:
        if sealing.is_sealed(root):
            raise ValueError("the credit-stop supervisor never reads a sealed tree (the "
                             "sealed recovery watches its own runs)")
    flag = Path(flag)
    since = time.time() if since is None else since
    state: dict[str, Any] = {"credit_stop": False, "runs_halted": 0, "polls": 0}
    if ready is not None:
        _write_flag(Path(ready), {"pid": os.getpid(), "since": since, "roots": len(roots)})
    while True:
        state["polls"] += 1
        credit_scan(roots, flag=flag, since=since, state=state, out=out)
        if once or (max_polls is not None and state["polls"] >= max_polls):
            return state
        time.sleep(poll_s)


class CreditWatcher:
    """The credit stop inside one process (the sealed B1i recovery): a
    background thread scanning this generation's own new runs, plus
    ``check()`` for a synchronous scan between trials. Writes the same flag."""

    def __init__(self, roots: list[Path], flag: Path, *, poll_s: float = 1.0) -> None:
        import threading

        self.roots, self.flag, self.poll_s = roots, Path(flag), poll_s
        self.since = time.time()
        self.state: dict[str, Any] = {"credit_stop": False, "runs_halted": 0}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def check(self) -> bool:
        with self._lock:
            credit_scan(self.roots, flag=self.flag, since=self.since, state=self.state,
                        out=lambda _line: None)
            return bool(self.state["credit_stop"])

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                self.check()
            except Exception:  # noqa: BLE001,S110 - the next scan retries
                pass

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m slh provider", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight", help=(
        "account / key allowance against the window token reservation, then one metered "
        "synthetic request through the exact user-simulator route (exit 0 ok, 3 failed)"))
    pre.add_argument("--runner-config", type=Path, action="append", required=True,
                     help="each runner config the window runs (all must share one route)")
    pre.add_argument("--reservation-usd", type=float, required=True,
                     help="the window's token reservation (every token subcap)")
    pre.add_argument("--max-cost-usd", type=float, default=PREFLIGHT_MAX_COST_USD)
    pre.add_argument("--out", type=Path, required=True, help="the record (JSON), new file")
    pre.add_argument("--registry-dir", type=Path, default=None,
                     help="append the provider_preflight event to this registry")
    stop = commands.add_parser("credit-stop", help=(
        "watch run directories; on a confirmed provider credit error halt every running "
        "run's ledger and write the flag file"))
    stop.add_argument("--watch", type=Path, action="append", required=True)
    stop.add_argument("--flag", type=Path, required=True)
    stop.add_argument("--poll-s", type=float, default=2.0)
    stop.add_argument("--ready", type=Path, default=None,
                      help="written once watching has begun (the window waits for it)")
    stop.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            if args.out.exists():
                raise FileExistsError(f"refusing to overwrite {args.out}")
            record = preflight(runner_configs=args.runner_config,
                               reservation_usd=args.reservation_usd,
                               max_cost_usd=args.max_cost_usd)
            args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
            if args.registry_dir is not None:
                from slh.learner.registry import Registry

                Registry(args.registry_dir).append(registry_event(record))
            print(json.dumps({k: record.get(k) for k in (
                "ok", "route_sha256", "reservation_usd", "allowance_usd", "failures")},
                sort_keys=True))
            return 0 if record["ok"] else 3
        state = credit_stop(args.watch, flag=args.flag, poll_s=args.poll_s, once=args.once,
                            ready=args.ready)
        print(json.dumps(state, sort_keys=True))
        return 0
    except (ValueError, OSError) as exc:
        print(f"provider {args.command} refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
