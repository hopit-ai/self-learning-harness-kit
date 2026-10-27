"""Hermes episode runner: executed by the PINNED Hermes venv's interpreter (``python -I -B``),
one process per episode, inside the sandbox profile (``sandbox.py``). Stdlib + Hermes only:
nothing from ``slh`` is importable here.

    python -I -B runner.py <episode.json>

Modes (``episode["mode"]``):

* ``episode``: egress guard -> preflight attestation (re-hash the immutable inputs against
  the manifest; effective config re-read from inside the process; ``write_approval``
  importable and on; safe-worker off; our plugin loaded with the manifest's source hash;
  delegation absent) -> ``hello`` to the harness (which may refuse) -> startup canaries (a
  planted forbidden call must come back blocked; planted memory and skill writes must be
  staged, not committed) -> the conversation (user turns from the harness: a task prompt or
  a customer simulator) -> evidence only: an explicit review fork, joined with a bounded
  wait -> ``out.json`` (messages, fork transcript, ShareGPT trajectory, usage, attestation,
  canaries, review status).
* ``replay``: replay staged records with Hermes' own approval appliers
  (``apply_skill_pending`` / ``apply_memory_pending``) on a scratch home materialised from
  the incumbent; the harness reads the resulting files back as full post-states.

Exit codes: 0 done; 3 refused by the harness at preflight; 4 preflight failed; 5 canary
failed; 6 conversation crashed (out.json still written when possible).
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import threading
import time
import traceback

T0 = time.time()


# ----------------------------------------------------------------------------- egress


def install_egress_guard(allowed_port: int | None, unix_path: str | None = None,
                         extra_unix: tuple = ()) -> None:
    """In-process egress guard (the sandbox profile is the OS-level one): only the loopback
    model endpoint and this episode's bridge socket (and, on Linux, the model relay socket)
    may be connected."""
    unix_ok = {p for p in (unix_path, *extra_unix) if p}
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_gai = socket.getaddrinfo

    def allowed(sock, address) -> bool:
        if sock.family == getattr(socket, "AF_UNIX", None):
            return address in unix_ok
        if isinstance(address, tuple) and len(address) >= 2:
            host, port = address[0], address[1]
            return host in ("127.0.0.1", "::1", "localhost") and port == allowed_port
        return False

    def connect(sock, address):
        if not allowed(sock, address):
            raise PermissionError(f"slh egress guard: connect to {address!r} refused")
        return real_connect(sock, address)

    def connect_ex(sock, address):
        if not allowed(sock, address):
            raise PermissionError(f"slh egress guard: connect to {address!r} refused")
        return real_connect_ex(sock, address)

    def getaddrinfo(host, *a, **k):
        if host not in ("127.0.0.1", "::1", "localhost", None):
            raise socket.gaierror(f"slh egress guard: resolving {host!r} refused")
        return real_gai(host, *a, **k)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo


# ----------------------------------------------------------------------------- bridge


def bridge(msg: dict) -> dict:
    path = os.environ["SLH_BRIDGE_SOCKET"]
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(float(os.environ.get("SLH_BRIDGE_TIMEOUT_S", "30")))
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
    return json.loads(buf)


# ----------------------------------------------------------------------------- hashing


def sha_file(p: str) -> str:
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def canon(v) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def attest_files(home: str, manifest: dict) -> dict:
    bad = []
    for rel, h in manifest["files"].items():
        p = os.path.join(home, rel)
        if not os.path.isfile(p) or os.path.islink(p) or sha_file(p) != h:
            bad.append(rel)
    with open(os.path.join(home, ".slh", "manifest.json"), encoding="utf-8") as fh:
        on_disk = json.load(fh)
    return {"files_ok": not bad and on_disk == manifest, "mismatched": sorted(bad),
            "materialization_sha256": hashlib.sha256(canon(on_disk).encode()).hexdigest(),
            "content_sha256": on_disk.get("content_sha256")}


# ----------------------------------------------------------------------------- preflight


def preflight(ep: dict, agent) -> dict:
    from hermes_cli.config import load_config

    cfg = load_config() or {}
    mem, sk = cfg.get("memory", {}) or {}, cfg.get("skills", {}) or {}
    aux = (cfg.get("auxiliary", {}) or {}).get("background_review", {}) or {}
    checks: dict = {}
    try:
        from tools import write_approval as wa

        checks["write_approval_importable"] = True
        checks["memory_write_approval"] = wa.write_approval_enabled(wa.MEMORY)
        checks["skills_write_approval"] = wa.write_approval_enabled(wa.SKILLS)
    except Exception as e:  # noqa: BLE001
        checks["write_approval_importable"] = False
        checks["error"] = repr(e)
    checks["memory_nudge_interval"] = mem.get("nudge_interval")
    checks["skills_creation_nudge_interval"] = sk.get("creation_nudge_interval")
    checks["review_defer"] = aux.get("defer")
    checks["review_enabled"] = aux.get("enabled")
    checks["user_profile_enabled"] = mem.get("user_profile_enabled")
    try:
        from agent.safe_worker_policy import safe_worker_enabled

        checks["safe_worker"] = bool(safe_worker_enabled())
    except Exception as e:  # noqa: BLE001
        checks["safe_worker"] = f"unknown: {e!r}"
    # our plugin: loaded, hook registered, source hash = manifest entry
    mod = sys.modules.get("hermes_plugins.slh_bridge")
    checks["plugin_loaded"] = mod is not None
    want = ep["manifest"]["files"].get("plugins/slh-bridge/__init__.py")
    checks["plugin_hash_ok"] = bool(mod is not None and getattr(mod, "__file__", None)
                                    and sha_file(mod.__file__) == want)
    try:
        from hermes_cli.plugins import get_plugin_manager

        hooks = get_plugin_manager()._hooks.get("pre_tool_call", [])
        checks["guard_hook_registered"] = any(
            getattr(h, "__module__", "") == "hermes_plugins.slh_bridge"
            and getattr(h, "__name__", "") == "on_pre_tool_call" for h in hooks)
        checks["other_pre_tool_hooks"] = sorted(
            f"{getattr(h, '__module__', '?')}.{getattr(h, '__name__', '?')}" for h in hooks
            if getattr(h, "__module__", "") != "hermes_plugins.slh_bridge")
    except Exception as e:  # noqa: BLE001
        checks["guard_hook_registered"] = False
        checks["error_hooks"] = repr(e)
    # the Hermes actually imported is the pinned clone at the pinned commit (review fix 7)
    import run_agent

    clone = os.path.realpath(ep.get("hermes_clone") or "")
    imported = os.path.realpath(os.path.dirname(run_agent.__file__))
    checks["imported_clone"] = imported
    checks["imported_clone_ok"] = bool(clone) and imported == clone
    checks["imported_head"] = _git_head(imported)
    checks["imported_head_ok"] = checks["imported_head"] == ep.get("hermes_sha")
    tools = sorted(agent.valid_tool_names)
    checks["tools"] = tools
    checks["delegation_absent"] = not any(t.startswith("delegate") for t in tools)
    checks["skip_background_review"] = bool(getattr(agent, "skip_background_review", None))
    checks["approval_callback"] = _approval_callback_present()
    checks["ok"] = bool(
        checks.get("write_approval_importable") and checks.get("memory_write_approval")
        and checks.get("skills_write_approval") and checks["memory_nudge_interval"] == 1
        and checks["skills_creation_nudge_interval"] == 1 and checks["review_defer"] == "never"
        and checks["user_profile_enabled"] is False and checks["safe_worker"] is False
        and checks["plugin_loaded"] and checks["plugin_hash_ok"]
        and checks["guard_hook_registered"] and not checks["other_pre_tool_hooks"]
        and checks["delegation_absent"] and not checks["approval_callback"]
        and set(tools) <= set(ep["allowed_tools"])
        and checks["skip_background_review"] is True
        and checks["imported_clone_ok"] and checks["imported_head_ok"])
    return checks


def _git_head(clone: str) -> str:
    try:
        with open(os.path.join(clone, ".git", "HEAD"), encoding="utf-8") as fh:
            ref = fh.read().strip()
        if not ref.startswith("ref: "):
            return ref
        p = os.path.join(clone, ".git", ref[5:])
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                return fh.read().strip()
        with open(os.path.join(clone, ".git", "packed-refs"), encoding="utf-8") as fh:
            for line in fh:
                if line.strip().endswith(" " + ref[5:]):
                    return line.split()[0]
    except OSError:
        pass
    return "unknown"


def _approval_callback_present() -> bool:
    try:
        from tools.terminal_tool import _get_approval_callback

        return _get_approval_callback() is not None
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------------------- canaries


def _pending_files(home: str, sub: str) -> list[str]:
    d = os.path.join(home, "pending", sub)
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def canaries(ep: dict, agent) -> dict:
    home = os.environ["HERMES_HOME"]
    nonce = ep["canary_nonce"]
    res: dict = {}
    # 1. a planted forbidden call through the real dispatch path must come back blocked
    from hermes_cli.plugins import _dispatch_pre_tool_call_hooks

    block, _ = _dispatch_pre_tool_call_hooks(
        "slh_canary_forbidden", {"nonce": nonce}, tool_call_id=f"slh-canary-{nonce}",
        task_id="slh-canary", session_id="slh-canary", turn_id="slh-canary")
    res["forbidden_call_blocked"] = block is not None
    # 2. planted memory and skill writes must be STAGED, not committed
    mem_before = _read(os.path.join(home, "memories", "MEMORY.md"))
    before = set(_pending_files(home, "memory"))
    from tools.memory_tool import memory_tool

    out = memory_tool(action="add", target="memory", content=f"slh canary {nonce}",
                      store=agent._memory_store)
    new = sorted(set(_pending_files(home, "memory")) - before)
    staged = [f for f in new if nonce in _read(os.path.join(home, "pending", "memory", f))]
    res["memory_write_staged"] = (len(staged) == 1
                                  and _read(os.path.join(home, "memories", "MEMORY.md"))
                                  == mem_before and nonce not in _store_text(agent))
    res["memory_tool_reply"] = str(out)[:300]
    for f in staged:
        os.unlink(os.path.join(home, "pending", "memory", f))
    sk_before = set(_pending_files(home, "skills"))
    skills_before = sorted(_walk(os.path.join(home, "skills")))
    from tools.skill_manager_tool import skill_manage

    name = f"slh-canary-{nonce}"
    out = skill_manage(action="create", name=name,
                       content=f"---\nname: {name}\ndescription: canary\n---\ncanary {nonce}\n")
    new = sorted(set(_pending_files(home, "skills")) - sk_before)
    staged = [f for f in new if nonce in _read(os.path.join(home, "pending", "skills", f))]
    res["skill_write_staged"] = (len(staged) == 1
                                 and sorted(_walk(os.path.join(home, "skills")))
                                 == skills_before)
    res["skill_tool_reply"] = str(out)[:300]
    for f in staged:
        os.unlink(os.path.join(home, "pending", "skills", f))
    # 3. (v2.2) Hermes' argument coercion: a task-tool call with MISTYPED array arguments
    # through the real dispatch; the plugin coerces before the guard (the harness checks
    # the recorded arguments are lists); the guard must block it
    cc = ep.get("coerce_canary")
    if cc:
        block, _ = _dispatch_pre_tool_call_hooks(
            cc["tool"], dict(cc["args"]), tool_call_id=f"slh-canary-coerce-{nonce}",
            task_id="slh-canary", session_id="slh-canary", turn_id="slh-canary")
        res["coerce_call_blocked"] = block is not None
    res["ok"] = bool(res["forbidden_call_blocked"] and res["memory_write_staged"]
                     and res["skill_write_staged"] and res.get("coerce_call_blocked", True))
    return res


def _read(p: str) -> str:
    try:
        with open(p, encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""


def _walk(root: str) -> list[str]:
    out = []
    for r, _, fs in os.walk(root):
        out += [os.path.relpath(os.path.join(r, f), root) for f in fs]
    return out


def _store_text(agent) -> str:
    st = getattr(agent, "_memory_store", None)
    try:
        return json.dumps(getattr(st, "memory_entries", None) or [], default=str)
    except Exception:  # noqa: BLE001
        return ""


# ----------------------------------------------------------------------------- episode


FORK_TRANSCRIPTS: list = []
FORK_USAGE: list = []


def _wrap_review_fork() -> None:
    import agent.background_review as BR

    orig = BR._run_review_fork

    def wrapped(agent, messages_snapshot, prompt, task_cfg, review_run, st, *a, **k):
        try:
            return orig(agent, messages_snapshot, prompt, task_cfg, review_run, st, *a, **k)
        finally:
            FORK_TRANSCRIPTS.append(list(getattr(st, "review_messages", None) or []))
            FORK_USAGE.append(dict(getattr(st, "review_usage", None) or {}))

    BR._run_review_fork = wrapped


PROBE: dict = {"boundaries": 0, "applied": [], "errors": []}


def _wire(messages: list) -> list:
    """The conversation as the harness's probe sees it (no persistence fields)."""
    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
        w = {"role": m.get("role"), "content": None if c is None else str(c)}
        if m.get("tool_call_id"):
            w["tool_call_id"] = str(m["tool_call_id"])
        if m.get("tool_calls"):
            w["tool_calls"] = [{"id": str(tc.get("id") or ""),
                                "name": str((tc.get("function") or {}).get("name") or ""),
                                "arguments": str((tc.get("function") or {}).get("arguments")
                                                 or "")}
                               for tc in m["tool_calls"] if isinstance(tc, dict)]
        out.append(w)
    return out


def _native_prune(agent, messages: list, min_chars: int, tail: int | None) -> list | None:
    """Hermes' own deterministic tool-result prune passes on a COPY of ``messages``
    (``ContextCompressor._prune_old_tool_results``): the content it would demote, as
    ``[{"i", "content"}]``. ``tail`` None = Hermes' configured ``protect_last_n``."""
    try:
        cc = agent.context_compressor
        n_tail = cc.protect_last_n if tail is None else int(tail)
        pruned, _ = cc._prune_old_tool_results([dict(m) for m in messages],
                                               protect_tail_count=n_tail,
                                               protect_tail_tokens=None,
                                               min_prune_chars=int(min_chars))
        return [{"i": i, "content": str(p.get("content"))}
                for i, (m, p) in enumerate(zip(messages, pruned, strict=False))
                if isinstance(m, dict) and m.get("role") == "tool"
                and p.get("content") != m.get("content")]
    except Exception as e:  # noqa: BLE001 - recorded; the probe treats it as "no native"
        PROBE["errors"].append(f"native: {e!r}"[:300])
        return None


def _ids(content: str) -> list:
    """Identifiers in a tool result (stdlib mirror of ``probe.identifiers``, tested equal):
    values of ``*_id``/``*_ids``/``id`` keys, payment-method keys, order numbers, a bare id
    result; no cap."""
    import re

    found: list = []

    def add(v):
        v = str(v)
        if v and v not in found:
            found.append(v)

    def walk(x, key=""):
        if isinstance(x, dict):
            for k, v in x.items():
                if re.search(r"(^|_)ids?$", str(k)):
                    for y in (v if isinstance(v, list) else [v]):
                        if isinstance(y, (str, int)):
                            add(y)
                if key.endswith("payment_methods"):
                    add(k)
                walk(v, str(k))
        elif isinstance(x, list):
            for y in x:
                walk(y, key)

    try:
        walk(json.loads(content))
    except (ValueError, TypeError):
        pass
    for m in re.findall(r"#W\d{7}", content or ""):
        add(m)
    bare = (content or "").strip().strip('"')
    if not found and re.fullmatch(r"[A-Za-z0-9_]{3,64}", bare):
        add(bare)
    return found


def _apply_edits(msgs: list, edits: list, b: int) -> None:
    """Apply the harness's edits IN PLACE (the same list object Hermes holds). Refused (and
    recorded, so the harness marks the branch invalid): an edit of anything but an existing
    ``tool`` message, and any edit that drops an identifier of the original result, except
    kind ``native`` (Hermes' own demotion text, a faithful record of the native policy) and
    kinds ``inject*`` (an injected failure replaces an UNREAD pending result)."""
    for e in edits:
        i = int(e["i"])
        if not (0 <= i < len(msgs) and isinstance(msgs[i], dict)
                and msgs[i].get("role") == "tool"):
            PROBE["errors"].append(f"edit refused at b{b}: index {i}")
            continue
        old = msgs[i].get("content")
        old = old if isinstance(old, str) else json.dumps(old)
        new = str(e["content"])
        kind = str(e.get("kind", ""))
        lost = ([] if kind == "native" or kind.startswith("inject")
                else [x for x in _ids(old) if x not in new])
        if lost:
            PROBE["errors"].append(f"edit refused at b{b}: index {i} drops {len(lost)} "
                                   "identifier(s)")
            continue
        msgs[i] = {**msgs[i], "content": new}
        PROBE["applied"].append({"b": b, "i": i})


def _install_boundary_hook(spec: dict) -> None:
    """S1 headroom probe (docs/xp/s1-headroom-probe-design.md): at Hermes' own post-tool
    compression boundary (``turn_tool_round.compress_after_tool_results``, where the
    proactive tool-result prune runs) report the conversation to the harness and apply the
    edits it returns (``_apply_edits``: only the CONTENT of existing ``tool`` messages; the
    call message, its id and the call/result pairing are never touched; no non-native
    edit may drop an identifier). Installed only for probe episodes (``ep["probe"]``);
    never for the M-H1 arms."""
    import agent.turn_tool_round as TTR

    orig = TTR.compress_after_tool_results
    lo = int(spec.get("native_min_chars", 400))

    def wrapped(ag, *a, **k):
        v = orig(ag, *a, **k)
        if threading.current_thread() is not threading.main_thread():
            return v  # a review fork's or helper's tool round: not the conversation
        try:
            msgs = v.messages
            b = PROBE["boundaries"]
            PROBE["boundaries"] += 1
            pend = 0  # the pending round: tool results the model has not read yet
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") == "tool":
                    pend += 1
                else:
                    break
            dflt = getattr(ag.context_compressor, "proactive_prune_min_result_chars", 8000)
            rep = bridge({"op": "boundary", "b": b, "messages": _wire(msgs),
                          "native": _native_prune(ag, msgs, lo, pend),
                          "native_default": _native_prune(ag, msgs, dflt, None)})
            _apply_edits(msgs, rep.get("edits") or [], b)
        except Exception as e:  # noqa: BLE001 - the harness marks the branch invalid
            PROBE["errors"].append(f"boundary: {e!r}"[:300])
        return v

    TTR.compress_after_tool_results = wrapped


def _usage(agent) -> dict:
    """Main-agent counters plus the review forks' (charged to the evidence run)."""
    keys = ("session_prompt_tokens", "session_completion_tokens", "session_total_tokens",
            "session_api_calls", "session_input_tokens", "session_output_tokens",
            "session_cache_read_tokens", "session_cache_write_tokens",
            "session_reasoning_tokens")
    u = {k: int(getattr(agent, k, 0) or 0) for k in keys}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "api_calls",
              "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
              "reasoning_tokens"):
        u[f"fork_{k}"] = sum(int(f.get(k) or 0) for f in FORK_USAGE)
    return u


def make_agent(ep: dict):
    from run_agent import AIAgent

    with open(os.path.join(os.environ["HERMES_HOME"], ".slh", "notes.md"),
              encoding="utf-8") as fh:
        notes = fh.read()
    return AIAgent(
        base_url=ep["model"]["base_url"], api_key=ep["model"].get("api_key", "local"),
        provider="custom", model=ep["model"]["name"],
        max_iterations=int(ep["max_iterations"]),
        run_budget_seconds=float(ep["run_budget_seconds"]),
        enabled_toolsets=list(ep["toolsets"]),
        quiet_mode=True, save_trajectories=False,
        skip_context_files=True, skip_memory=False,
        # automatic (per-turn) forks off in every role; evidence runs get ONE explicit fork
        # after the conversation (bypasses deferral), joined with a bounded wait
        skip_background_review=True,
        request_overrides=dict(ep["model"].get("request_overrides") or {}),
        ephemeral_system_prompt=(ep.get("system_preamble", "") + (
            "\n\n## Notes\n" + notes if notes.strip() else "")) or None,
        cwd=ep["workspace"], session_id=ep["session_id"], platform="batch",
    )


def run_episode(ep: dict) -> int:
    out: dict = {"schema": "slh-hermes-episode-out-v1", "unit_id": ep["unit_id"],
                 "role": ep["role"], "tag": ep["tag"]}
    home = os.environ["HERMES_HOME"]
    out["attestation"] = attest_files(home, ep["manifest"])
    code = 0
    agent = None
    try:
        _wrap_review_fork()
        if ep.get("probe"):
            _install_boundary_hook(ep["probe"])
        agent = make_agent(ep)
        out["preflight"] = preflight(ep, agent)
        hello = bridge({"op": "hello", "attestation": out["attestation"],
                        "preflight": out["preflight"]})
        if not hello.get("ok"):
            out["refused"] = hello.get("reason", "refused")
            return _finish(ep, out, agent, 3)
        if not (out["preflight"]["ok"] and out["attestation"]["files_ok"]):
            return _finish(ep, out, agent, 4)
        out["canaries"] = canaries(ep, agent)
        if not out["canaries"]["ok"]:
            return _finish(ep, out, agent, 5)
        first = bridge({"op": "first_user"})
        msg, history, turns = first["content"], None, []
        while True:
            r = agent.run_conversation(msg, conversation_history=history,
                                       task_id=ep["session_id"])
            history = r.get("messages") or []
            turns.append({"final_response": r.get("final_response"),
                          "completed": r.get("completed"),
                          "interrupted": r.get("interrupted")})
            nxt = bridge({"op": "user_turn", "assistant": r.get("final_response") or ""})
            if nxt.get("stop"):
                break
            msg = nxt["content"]
        out["messages"] = history
        out["turns"] = turns
        out["first_user"] = first["content"]
        if ep["role"] == "evidence":
            out["review"] = run_review(agent, history, float(ep["review_join_s"]))
        out["usage"] = _usage(agent)
        try:
            out["sharegpt"] = agent._convert_to_trajectory_format(
                history, first["content"], bool(turns and turns[-1]["completed"]))
        except Exception as e:  # noqa: BLE001
            out["sharegpt_error"] = repr(e)
    except Exception as e:  # noqa: BLE001
        out["crash"] = "".join(traceback.format_exception(e))[-4000:]
        code = 6
    return _finish(ep, out, agent, code)


def run_review(agent, history: list, join_s: float) -> dict:
    """Explicit review (bypasses deferral), on a thread we own, joined with a bounded wait."""
    from agent.background_review import (
        prepare_background_review_run,
        spawn_background_review_thread,
    )

    t = time.time()
    review_run = prepare_background_review_run(agent)
    if review_run is None:
        return {"status": "not_started", "seconds": 0.0}
    target, _ = spawn_background_review_thread(
        agent, history, review_memory=True, review_skills=True, review_run=review_run,
        explicit=True)
    th = threading.Thread(target=target, daemon=True, name="slh-review")
    th.start()
    th.join(join_s)
    others = [x for x in threading.enumerate() if x.name == "bg-review"]
    for x in others:  # an automatic fork should not exist (skip flag); join it if it does
        x.join(max(0.0, join_s - (time.time() - t)))
    alive = th.is_alive() or any(x.is_alive() for x in others)
    return {"status": "timeout" if alive else "joined", "automatic_forks": len(others),
            "seconds": round(time.time() - t, 3), "forks": len(FORK_TRANSCRIPTS)}


def _finish(ep: dict, out: dict, agent, code: int) -> int:
    out["fork_transcripts"] = FORK_TRANSCRIPTS
    if ep.get("probe"):
        out["probe"] = PROBE
    out["exit_code"] = code
    out["seconds"] = round(time.time() - T0, 3)
    if agent is not None and "usage" not in out:
        out["usage"] = _usage(agent)
    p = os.path.join(ep["out_dir"], "out.json")
    with open(p + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(out, fh, default=str)
    os.replace(p + ".tmp", p)
    return code


# ----------------------------------------------------------------------------- replay


def run_replay(ep: dict) -> int:
    """Apply staged records in order with Hermes' own appliers; report per-record results."""
    from tools.memory_tool import apply_memory_pending, load_on_disk_store
    from tools.skill_manager_tool import apply_skill_pending

    results = []
    store = load_on_disk_store()
    for rec in ep["records"]:
        try:
            if rec["subsystem"] == "memory":
                r = apply_memory_pending(dict(rec["payload"]), store)
            else:
                r = json.loads(apply_skill_pending(dict(rec["payload"])))
            results.append({"id": rec.get("id"), "ok": bool(r.get("success")),
                            "error": r.get("error", "")})
        except Exception as e:  # noqa: BLE001
            results.append({"id": rec.get("id"), "ok": False, "error": repr(e)})
    p = os.path.join(ep["out_dir"], "out.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"schema": "slh-hermes-replay-out-v1", "results": results}, fh)
    return 0


def start_model_relay(port: int, unix_path: str) -> None:
    """Linux form (network namespace with only its own loopback): listen on
    ``127.0.0.1:<port>`` INSIDE the namespace and forward every connection to the model relay
    socket in the episode root (the harness forwards it to the episode's spend gate)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", int(port)))
    srv.listen(64)

    def pump(src, dst):
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for x in (src, dst):
                try:
                    x.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def pair(c):
        u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            u.connect(unix_path)
        except OSError:
            c.close()
            u.close()
            return
        threading.Thread(target=pump, args=(u, c), daemon=True).start()
        pump(c, u)

    def loop():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=pair, args=(c,), daemon=True).start()

    threading.Thread(target=loop, daemon=True, name="slh-model-relay").start()


def main(argv: list[str]) -> int:
    with open(argv[1], encoding="utf-8") as fh:
        ep = json.load(fh)
    relay = os.environ.get("SLH_MODEL_RELAY_SOCKET") if ep["mode"] == "episode" else None
    if relay:
        start_model_relay(int(ep["model"]["port"]), relay)
    install_egress_guard(ep.get("model", {}).get("port"),
                         os.environ.get("SLH_BRIDGE_SOCKET") if ep["mode"] == "episode" else None,
                         (relay,) if relay else ())
    if ep["mode"] == "replay":
        return run_replay(ep)
    return run_episode(ep)


if __name__ == "__main__":
    rc = main(sys.argv)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)  # daemon threads (an unjoined review fork) must not keep the process alive
