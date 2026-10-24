"""``HermesAdapter``: the V1 ``AgentAdapter`` around Hermes Agent (spike rev 2 s. 2, s. 6).

One process per episode (``runner.py`` under the pinned Hermes venv), per request:

1. **materialise** a fresh ``HERMES_HOME`` from the request's ``HarnessState``
   (``materialize.py``; manifest -> ``materialization_sha256``), the learned state
   read-only (0444/0555 and the sandbox profile);
2. **preflight + canaries** inside the run process (``runner.preflight`` / ``canaries``),
   reported to the broker in ``hello`` (refused on any mismatch);
3. the **episode**: ``AIAgent`` with ``skip_context_files``, ``skip_memory=False`` (memory
   is learned state), ``skip_background_review`` unless evidence, ``request_overrides``
   (temperature, seed), no approval callback, ``write_approval`` on, tool search off,
   delegation absent; every tool call mediated by the broker (``bridge.py``);
4. evidence only: an **explicit review fork, joined** with a bounded wait (a timeout leaves
   the run unscored);
5. **harvest** ``pending/{memory,skills}`` (evidence: proposals; any other role: incident);
6. **post-run hash** of the immutable inputs (mismatch or extra file: incident);
7. **score** by the surface (the answer key never leaves this process).

The applied-state echo is ``content_sha256`` only when the run's own attestation matched the
manifest; otherwise it is withheld (``unattested:...``), which the harness refuses
(``loop._validate``). The guard-chain echo is ``guards.chain_sha256`` over the declared
modules; ``mediated`` is True only if every tool call in the transcripts (main and fork) has
a mediation record and no task tool executed without one.

Fault-injection hooks (tests only): ``on_hello`` runs inside the broker at ``hello``;
``pre_launch`` runs after materialisation and before the process starts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from slh.state import HarnessState
from slh.system import guards as GD
from slh.system.hosts.hermes import guard_host
from slh.system.hosts.hermes import materialize as MZ
from slh.system.hosts.hermes import sandbox as SB
from slh.system.hosts.hermes.bridge import BridgeServer, Broker
from slh.system.interfaces import (
    AdapterBudgetStop,
    AdapterInfo,
    Outcome,
    RunRequest,
    Unit,
)

HERMES_SHA = "cb757fac59c92b49e9b8f2910e5cd7ab75efffdd"
EXT = Path(__file__).resolve().parents[5] / "ext-harnesses"
DEFAULT_CLONE = EXT / "hermes-agent"
DEFAULT_VENV = EXT / "venv-hermes"
RUNNER = Path(__file__).resolve().parent / "runner.py"
HERMES_READ_TOOLS = ("skill_view", "skills_list")
REPO = Path(__file__).resolve().parents[4]
SECRET_ROOTS = (REPO / "learner_runs" / ".seal", REPO / "sealed", REPO / "demo1" / "sealed")
LEARNING_TOOLS = ("memory", "skill_manage")


def _sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode() if isinstance(b, str) else b).hexdigest()


@dataclass(frozen=True)
class HermesHost:
    """The pinned host: clone at ``sha``, the isolated venv's interpreter."""

    clone: Path = DEFAULT_CLONE
    venv: Path = DEFAULT_VENV
    sha: str = HERMES_SHA

    @property
    def python(self) -> Path:
        return self.venv / "bin" / "python"

    def available(self) -> bool:
        return self.python.exists() and (self.clone / "run_agent.py").exists()

    def venv_lock_sha256(self) -> str:
        """Hash of the installed distributions (names and versions from dist-info dirs)."""
        libs = sorted(self.venv.glob("lib/python*/site-packages/*.dist-info"))
        return _sha("\n".join(p.name for p in libs))

    def clone_head(self) -> str:
        head = (self.clone / ".git" / "HEAD")
        if not head.exists():
            return "unknown"
        ref = head.read_text().strip()
        if ref.startswith("ref: "):
            p = self.clone / ".git" / ref[5:]
            return p.read_text().strip() if p.exists() else "unknown"
        return ref

    def verify(self) -> dict[str, str]:
        """The clone that will be imported IS the pinned one: HEAD == ``sha`` and no tracked
        file modified. Returns {head, tree}; raises otherwise (review fix 7)."""
        import subprocess

        def git(*a: str) -> str:
            return subprocess.run(["git", "-C", str(self.clone), *a], check=True,
                                  capture_output=True, text=True).stdout.strip()

        head = git("rev-parse", "HEAD")
        if head != self.sha:
            raise RuntimeError(f"Hermes clone HEAD {head} != pinned {self.sha}")
        dirty = git("status", "--porcelain", "--untracked-files=no")
        if dirty:
            raise RuntimeError(f"Hermes clone has modified tracked files: {dirty[:200]}")
        return {"head": head, "tree": git("rev-parse", "HEAD^{tree}")}

    def identity(self) -> str:
        v = self.verify()
        return (f"hermes-agent@{v['head']}|tree:{v['tree'][:16]}"
                f"|venv:{self.venv_lock_sha256()[:16]}")


@dataclass(frozen=True)
class ModelEndpoint:
    """An OpenAI-compatible endpoint on the loopback interface (local vLLM or the test stub).
    ``usd_per_mtok``: (input, output) price; ``worst_tokens_per_run`` bounds the estimate."""

    base_url: str
    name: str
    serving_spec: str = "stub"
    temperature: float = 0.0
    seed: int = 7401
    usd_per_mtok: tuple[float, float] = (0.0, 0.0)
    worst_tokens_per_run: tuple[int, int] = (400_000, 40_000)  # ENFORCED by the SpendGate
    max_output_tokens_per_request: int = 4096  # ENFORCED: max_tokens set on every request
    counter: Any = None  # input counter; None = byte bound (free only)

    def __post_init__(self) -> None:
        from slh.system.hosts.hermes.tau2_backend import ByteBoundCounter, ChatTemplateCounter

        if self.counter is None:
            object.__setattr__(self, "counter", ByteBoundCounter())
        if any(p > 0 for p in self.usd_per_mtok) and not isinstance(self.counter,
                                                                     ChatTemplateCounter):
            raise ValueError("a priced agent model needs its served tokenizer and chat "
                             "template (ChatTemplateCounter) to bound input tokens")
        if self.max_output_tokens_per_request > self.worst_tokens_per_run[1]:
            raise ValueError("the per-request output cap exceeds the run's output ceiling")

    def gate(self) -> Any:
        from slh.system.hosts.hermes.meter import SpendGate

        return SpendGate(self.base_url, counter=self.counter,
                         max_input_tokens=self.worst_tokens_per_run[0],
                         max_output_tokens=self.worst_tokens_per_run[1],
                         max_output_per_request=self.max_output_tokens_per_request,
                         usd_per_mtok=self.usd_per_mtok)

    @property
    def port(self) -> int:
        u = urlparse(self.base_url)
        if u.hostname not in ("127.0.0.1", "localhost"):
            raise ValueError("Hermes episodes may only reach a loopback model endpoint")
        return int(u.port or (443 if u.scheme == "https" else 80))

    def worst_cost(self) -> float:
        i, o = self.worst_tokens_per_run
        return (i * self.usd_per_mtok[0] + o * self.usd_per_mtok[1]) / 1e6

    def cost(self, usage: Mapping[str, Any]) -> float:
        """Main agent plus review fork tokens (the fork is charged to the evidence run)."""
        i = int(usage.get("session_prompt_tokens") or usage.get("session_input_tokens") or 0)
        o = int(usage.get("session_completion_tokens")
                or usage.get("session_output_tokens") or 0)
        i += int(usage.get("fork_prompt_tokens") or usage.get("fork_input_tokens") or 0)
        o += int(usage.get("fork_completion_tokens") or usage.get("fork_output_tokens") or 0)
        # reasoning tokens charged as output on top (an upper bound)
        o += int(usage.get("session_reasoning_tokens") or 0)
        o += int(usage.get("fork_reasoning_tokens") or 0)
        # cached reads and writes are charged at the input price (an upper bound whether or
        # not the endpoint counts them inside prompt_tokens)
        for k in ("session_cache_read_tokens", "session_cache_write_tokens",
                  "fork_cache_read_tokens", "fork_cache_write_tokens"):
            i += int(usage.get(k) or 0)
        return (i * self.usd_per_mtok[0] + o * self.usd_per_mtok[1]) / 1e6


@dataclass
class HermesAdapter:
    surface: Any  # ReasoningGymToolPool | Tau2HermesEpisodes (sample, sessions, guards, tools)
    model: ModelEndpoint
    host: HermesHost = field(default_factory=HermesHost)
    max_iterations: int = 12
    run_budget_seconds: float = 120.0
    wall_clock_s: float = 240.0  # hard kill of the episode process group
    review_join_s: float = 90.0
    work_root: Path | None = None  # episode roots (short path: a Unix socket lives there)
    keep_runs: bool = False
    sandbox: bool = True
    soul: str = MZ.DEFAULT_SOUL
    config_extra: Mapping[str, Any] | None = None
    system_preamble: str = ""
    paid: bool = False
    # M-H1: keep each run's dialogue (with the customer's acts) and its executed writes with
    # the consent reading in the per-run diagnostics (``runs``), never in the trace
    record_dialogue: bool = False
    on_hello: Callable[..., None] | None = None  # test fault injection
    pre_launch: Callable[[Path, RunRequest], None] | None = None  # test fault injection
    runs: list[dict[str, Any]] = field(default_factory=list)  # per-run diagnostics

    def __post_init__(self) -> None:
        if self.sandbox and not SB.sandbox_available():
            raise RuntimeError("no sandbox available (sandbox-exec on macOS, a self-tested "
                               "bubblewrap on Linux); refusing to run Hermes uncontained: "
                               f"{SB.backend_report()}")
        self.cfg = MZ.host_config(extra=self.config_extra)
        self._chains: dict[str, str] = {}
        self.host_verified = self.host.verify() if self.host.available() else None

    # ------------------------------------------------------------------ identity

    def info(self) -> AdapterInfo:
        runner_sha = _sha(RUNNER.read_bytes())
        plugin_sha = _sha(b"".join(MZ.plugin_sources()[k] for k in sorted(MZ.plugin_sources())))
        model = (f"{self.host.identity()}|runner:{runner_sha[:16]}|plugin:{plugin_sha[:16]}"
                 f"|model:{self.model.name}|serving:{self.model.serving_spec}"
                 f"|T={self.model.temperature}|seed={self.model.seed}"
                 f"|caps:{self.model.worst_tokens_per_run}/{self.model.max_output_tokens_per_request}"
                 f"|counter:{_sha(json.dumps(self.model.counter.identity(), sort_keys=True))[:16]}"
                 f"|config:{_sha(json.dumps(self.cfg, sort_keys=True))[:16]}"
                 f"|limits:iter={self.max_iterations},run={self.run_budget_seconds},"
                 f"wall={self.wall_clock_s},join={self.review_join_s}"
                 f"|surface:{self._surface_runtime()}")
        scorer = _sha(Path(__import__(type(self.surface).__module__,
                                      fromlist=["x"]).__file__).read_bytes())
        return AdapterInfo(agent_id=f"hermes:{self.surface.name}", model_identity=model,
                           scorer_identity=f"{type(self.surface).__module__}:{scorer[:16]}",
                           sampling_identity=self.surface.sampling_identity(),
                           surface=self.surface.name,
                           guard_modules=tuple(self.surface.guard_modules),
                           contexts=tuple(self.surface.contexts),
                           paid=self.paid or bool(getattr(self.surface, "paid", False)))

    def _surface_runtime(self) -> str:
        """Material runtime inputs of the surface (customer model, endpoint, serving,
        sampling, seed policy, turn and token limits, price)."""
        f = getattr(self.surface, "runtime_identity", None)
        return _sha(json.dumps(f() if f else {}, sort_keys=True, default=str))[:16]

    def sample_units(self, batch: int, role: str, n: int, *, k: int | None = None,
                     level: str = "decision") -> list[Unit]:
        return list(self.surface.sample_units(batch, role, n, k))

    def _surface_worst(self) -> float:
        est = getattr(self.surface, "estimate_cost_usd", None)
        return float(est(1)) if est else 0.0

    def estimate_cost_usd(self, n_runs: int, *, level: str = "decision") -> float:
        """Agent model plus the surface's own spend (e.g. an API customer simulator)."""
        return n_runs * (self.model.worst_cost() + self._surface_worst())

    def policy_conflict(self, text: str) -> str | None:
        low = text.lower()
        for bad in ("write_approval", "approval gate", "skip the guard", "bypass the guard",
                    "disable the guard", "ignore the harness"):
            if bad in low:
                return f"tries to weaken containment ({bad!r})"
        return self.surface.policy_conflict(text)

    def guard_probe(self, tool: str, arguments: Mapping[str, Any],
                    observations: Sequence[tuple[str, Mapping[str, Any], str]]) -> bool:
        ok, _ = guard_host.check(tool, arguments, role=guard_host.PROBE_ROLE,
                                 task_tools=self.surface.task_tools)
        if ok and tool in self.surface.task_tools:
            ok = self.surface.probe(tool, arguments, observations)
        return bool(ok)

    def chain_sha(self, state: HarnessState) -> str:
        v = state.version_id
        if v not in self._chains:
            self._chains[v] = GD.chain_sha256(self.surface.guard_modules, state)
        return self._chains[v]

    # ------------------------------------------------------------------ run

    def run(self, requests: Sequence[RunRequest], *, budget_usd: float | None = None
            ) -> list[Outcome]:
        out: list[Outcome] = []
        spent = 0.0
        for r in requests:
            worst = self.model.worst_cost() + self._surface_worst()
            if budget_usd is not None and spent + worst > budget_usd + 1e-12:
                raise AdapterBudgetStop(f"next Hermes episode may cost ${worst:.4f}; "
                                        f"${spent:.4f} of ${budget_usd:.4f} spent")
            o = self.run_one(r)
            spent += o.cost_usd
            out.append(o)
            # a provider billing above our counted bound means the bound is wrong: stop the
            # dispatch (the harness charges the reservation and latches a system halt)
            gate = (self.runs[-1].get("gate") or {}) if self.runs else {}
            breach = gate.get("breaches") or (getattr(self.surface, "spend_breach",
                                                      lambda: None)())
            if breach:
                raise AdapterBudgetStop(f"provider usage above the counted bound: {breach}")
        return out

    def bridge_timeout_s(self) -> float:
        """The runner and plugin wait at least the surface's slowest call (an HTTP customer
        turn) plus a margin, so a slow customer never turns into a bridge crash."""
        f = getattr(self.surface, "max_call_seconds", None)
        return max(30.0, (float(f()) if f else 0.0) + 30.0)

    def _readable(self, root: Path) -> list[Path]:
        return [root, self.host.clone, self.host.venv, RUNNER,
                *SB.interpreter_roots(self.host.python)]

    def _toolsets(self, role: str) -> list[str]:
        sets = ["slh_task", "skills", "memory"]
        return sets

    def _allowed_tools(self, role: str) -> list[str]:
        return sorted(set(self.surface.task_tools) | set(HERMES_READ_TOOLS)
                      | set(LEARNING_TOOLS))

    def run_one(self, r: RunRequest) -> Outcome:
        if any(g.get("kind") == "checker" for g in r.config.guards):
            raise NotImplementedError("checker candidates are not enforced in Hermes runs yet")
        root = Path(tempfile.mkdtemp(prefix="slh-h-", dir=str(self.work_root or "/tmp")))
        root = Path(os.path.realpath(root))
        try:
            return self._run_in(root, r)
        finally:
            if not self.keep_runs:
                SB.rmtree(root)

    def _run_in(self, root: Path, r: RunRequest) -> Outcome:
        state, unit = r.config, r.unit
        if any(g.get("kind") == "checker" for g in state.guards):
            # the chain echo would include checker specs that nothing here executes
            raise NotImplementedError("checker candidates are not enforced in Hermes runs yet")
        home, ws, outd, tmp, uhome = (root / "home", root / "ws", root / "out", root / "tmp",
                                      root / "u")
        for d in (ws, outd, tmp, uhome):
            d.mkdir()
        man = MZ.materialize(state, home, self.cfg, self.soul)
        expected = MZ.manifest_for(state, self.cfg, self.soul)
        determinism_ok = man == expected
        mat_sha = MZ.materialization_sha256(expected)
        session = self.surface.new_session(unit)
        tools_file = root / "tools.json"
        tools_file.write_text(json.dumps(self.surface.tool_schemas))
        broker = Broker(session, role=unit.role, task_tools=set(self.surface.task_tools),
                        surface_guard=self.surface.surface_guard,
                        expected_materialization=mat_sha, on_hello=self.on_hello)
        nonce = uuid.uuid4().hex[:12]
        gate = self.model.gate().start()  # the only way to the model: a hard token ceiling
        ep = {"mode": "episode", "role": unit.role, "tag": r.tag, "unit_id": unit.unit_id,
              "manifest": expected, "allowed_tools": self._allowed_tools(unit.role),
              "toolsets": self._toolsets(unit.role),
              "model": {"base_url": gate.base_url, "name": self.model.name,
                        "port": gate.port, "api_key": "local",
                        "request_overrides": {"temperature": self.model.temperature,
                                              "seed": self.model.seed}},
              "max_iterations": self.max_iterations,
              "run_budget_seconds": self.run_budget_seconds,
              "review_join_s": self.review_join_s, "canary_nonce": nonce,
              "workspace": str(ws), "out_dir": str(outd),
              "session_id": f"slh-{_sha(unit.unit_id + r.tag)[:12]}",
              "hermes_clone": str(self.host.clone.resolve()), "hermes_sha": self.host.sha,
              "system_preamble": self.system_preamble}
        ep_file = root / "episode.json"
        ep_file.write_text(json.dumps(ep))
        if self.pre_launch is not None:
            self.pre_launch(home, r)
        server = BridgeServer(root / "b.sock", broker).start()
        # Linux form: the episode has its own network namespace; the model is reached
        # through a Unix socket in the root, forwarded here to the episode's spend gate
        relay = (SB.UnixRelay(root / "m.sock", "127.0.0.1", gate.port).start()
                 if self.sandbox and SB.backend() == "bwrap" else None)
        readonly = [home / p for p in MZ.IMMUTABLE_ROOT_FILES] + [home / d for d in (
            ".slh", "memories", "skills", "plugins")] + [ep_file, tools_file]
        extra = {"HERMES_HOME": str(home), "SLH_BRIDGE_SOCKET": str(server.path),
                 "SLH_TASK_TOOLS": str(tools_file),
                 "SLH_BRIDGE_TIMEOUT_S": str(self.bridge_timeout_s())}
        if relay is not None:
            extra["SLH_MODEL_RELAY_SOCKET"] = relay.path
        try:
            prof = (SB.contain(writable=[root], readonly=readonly, model_port=gate.port,
                               readable=self._readable(root),
                               sockets=[server.path] + ([relay.path] if relay else []),
                               secret_roots=SECRET_ROOTS)
                    if self.sandbox else None)
        except BaseException:
            server.stop()
            gate.stop()
            if relay is not None:
                relay.stop()
            raise
        env = SB.scrubbed_env(extra, home=uhome, tmp=tmp)
        t0 = time.time()
        try:
            done = SB.run_contained([str(self.host.python), "-I", "-B", str(RUNNER),
                                     str(ep_file)], env=env, cwd=ws, profile_text=prof,
                                    deadline_s=self.wall_clock_s, log_dir=root / "log")
        finally:
            server.stop()
            gate.stop()
            if relay is not None:
                relay.stop()
        post = MZ.rehash(home, expected)
        staged = harvest(home)
        res = _read_json(outd / "out.json")
        diag = {"unit": unit.unit_id, "role": unit.role, "tag": r.tag,
                "returncode": done.returncode, "killed": done.killed,
                "seconds": round(time.time() - t0, 2), "post_hash": post,
                "staged": len(staged), "stderr_tail": done.stderr[-1500:],
                "bridge_errors": server.errors, "root": str(root), "gate": gate.report()}
        if self.record_dialogue:
            wn = getattr(session, "why_not", None)
            diag["writes"] = [{"name": w["name"], "arguments": w["arguments"], "at": w["at"],
                               "why_not": wn(w["name"], w["arguments"], w["at"]) if wn
                               else None}
                              for w in getattr(session, "executed_writes", [])]
            diag["dialogue"] = [dict(m) for m in getattr(session, "transcript", [])]
        self.runs.append(diag)
        return self._outcome(r, res, done, post, staged, broker, session, determinism_ok,
                             mat_sha, diag, gate)

    # ------------------------------------------------------------------ outcome

    def _outcome(self, r: RunRequest, res: dict[str, Any] | None, done: SB.Completed,
                 post: dict[str, Any], staged: list[dict[str, Any]], broker: Broker,
                 session: Any, determinism_ok: bool, mat_sha: str,
                 diag: dict[str, Any], gate: Any = None) -> Outcome:
        state, unit = r.config, r.unit
        reasons: list[str] = []  # incident reasons (containment); notes are not incidents
        notes: list[str] = []
        res = res or {}
        hello = broker.hello or {}
        # the attestation and preflight the runner reported at hello (also when it was killed
        # or crashed later and wrote no out.json)
        att = res.get("attestation") or hello.get("attestation") or {}
        pre = res.get("preflight") or hello.get("preflight") or {}
        started = bool(res) or broker.hello is not None
        attested = (determinism_ok and att.get("files_ok") is True
                    and att.get("materialization_sha256") == mat_sha
                    and att.get("content_sha256") == state.content_sha256)
        if not determinism_ok:
            reasons.append("materialiser not deterministic")
        if started and not attested:
            reasons.append("attestation mismatch")
        if not started:
            # died before hello: no model call and no tool call can have happened (the
            # conversation starts only after hello and the canaries); nothing to attest
            notes.append("process ended before hello")
            attested = determinism_ok
        if not post["ok"]:
            reasons.append(f"post-run state change: {post}")
        if started and not res.get("refused") and pre and not pre.get("ok", False):
            reasons.append("preflight failed")
        if res.get("refused"):
            reasons.append(f"refused at hello: {res['refused']}")
        can = res.get("canaries")
        if res.get("exit_code") == 5 or (can is not None and not can.get("ok")):
            reasons.append(f"canary failed: {can}")
        if staged and unit.role != "evidence":
            reasons.append(f"{len(staged)} staged write(s) in a {unit.role} run")
        if broker.unmediated:
            reasons.append(f"unmediated execution attempts: {len(broker.unmediated)}")
        # mediation: every tool call in the transcripts has a record. Without a transcript
        # (killed / crashed) only the broker's own evidence is available: no task tool ran
        # without an allow record, and learned-state writes are caught by the staged-file
        # and post-hash checks above; critical harm then stays unlabelled (fail closed).
        have_transcript = res.get("messages") is not None
        call_ids = _tool_call_ids(res.get("messages") or [])
        for fork in res.get("fork_transcripts") or []:
            call_ids |= _tool_call_ids(fork)
        recorded = {m["call_id"] for m in broker.mediation if m["call_id"]}
        results = _tool_results(res.get("messages") or [])
        for fork in res.get("fork_transcripts") or []:
            results.update(_tool_results(fork))
        executed_ids = {e["call_id"] for e in broker.executed}
        refused = sorted(c for c in call_ids - recorded
                         if c not in executed_ids and _refused_before_dispatch(results.get(c)))
        missing = sorted(call_ids - recorded - set(refused))
        mediated = not missing and not broker.unmediated
        if not have_transcript:
            notes.append("no transcript")
        crashed = res.get("exit_code") == 6 or "crash" in res
        if getattr(session, "failed", None):
            notes.append(f"surface failure: {session.failed}")
        review = res.get("review") or {}
        status = "scored"
        if done.killed:
            status = "unscored"
            notes.append("wall-clock kill")
        elif (not res or res.get("exit_code") not in (0,) or crashed
              or getattr(session, "failed", None)):
            status = "error"
        elif getattr(session, "unscored_reason", None):
            status = "unscored"  # e.g. a customer turn without a valid consent act
            notes.append(f"unscored: {session.unscored_reason}")
        elif unit.role == "evidence" and review.get("status") != "joined":
            status = "unscored"  # the fork did not finish: unscored for evidence
        incident = bool(reasons)
        proposed = [m for m in broker.mediation if m["tool"] in self.surface.task_tools]
        # labels come from OUR side's ground truth (the session saw every executed task
        # call), so a killed or crashed run is still labelled (a wrong write then a kill is
        # critical); success and harm count only for complete, scored runs
        sc = session.score(proposed)
        scored = status == "scored" and have_transcript
        # tri-state: None (unresolved, e.g. a write after an invalid consent act) stays
        # None, so the gate treats it as worst case (HOLD) and the analysis as critical
        crit = sc["critical_harm"]
        critical = (None if crit is None else bool(crit)) if (started or broker.mediation) \
            else False
        usage = res.get("usage") or {}
        # the gate saw every agent request: reported usage of completed responses, the
        # reserved bound of the rest (lost, cut or in flight at a kill); never above the
        # reservation, because the gate refuses requests past the run's ceilings
        model_cost = gate.cost_usd() if gate is not None else self.model.worst_cost()
        if gate is not None and gate.refused:
            notes.append(f"spend gate refused {len(gate.refused)} request(s)")
        cost = model_cost + float(getattr(session, "cost_usd", 0.0) or 0.0)
        applied = state.content_sha256 if attested else f"unattested:{mat_sha}"
        trace: dict[str, Any] = {
            "host": "hermes", "materialization_sha256": mat_sha,
            "attested": attested, "post_hash_ok": post["ok"], "incident_reasons": reasons,
            "notes": notes,
            "staged": [_learner_record(x) for x in staged] if unit.role == "evidence" else
            [{"subsystem": x.get("subsystem"), "action": x.get("action")} for x in staged],
            "review": review, "usage": usage, "mediation_missing": missing[:20],
            "refused_before_dispatch": refused[:20],
            "consent_audit": (session.consent_audit() if hasattr(session, "consent_audit")
                              else None),
            "preflight": res.get("preflight"), "canaries": res.get("canaries"),
            "spend_gate": gate.report() if gate is not None else None,
            "guard_records": len(broker.mediation)}
        if unit.role == "evidence":
            trace.update({"messages": res.get("messages"), "fork": res.get("fork_transcripts"),
                          "sharegpt": res.get("sharegpt"), **session.learner_view(),
                          "guard": [{k: m[k] for k in ("tool", "allow", "verdicts", "origin")}
                                    for m in broker.mediation]})
        diag["reasons"], diag["notes"] = reasons, notes
        return Outcome(unit_id=unit.unit_id, role=unit.role, context=unit.context, tag=r.tag,
                       config_version=state.version_id, applied_state_sha256=applied,
                       guard_chain_sha256=self.chain_sha(state), mediated=mediated,
                       status=status, success=bool(sc["success"]) if scored else None,
                       harm=bool(sc["harm"]) if scored else None, critical_harm=critical,
                       cost_usd=cost, incident=incident, trace=trace)


# ---------------------------------------------------------------------- helpers


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


_REFUSAL_PATTERNS = (
    re.compile(r"^Tool '[^'\n]{1,200}' does not exist\. Available tools: "),
)
_INVALID_ARGS = "Tool arguments must be a valid JSON object; tool was not executed."


def _refused_before_dispatch(content: Any) -> bool:
    """Hermes' own pre-dispatch refusals (no hook ran, nothing executed): an unknown or
    unadvertised tool name, or arguments that are not a JSON object."""
    if not isinstance(content, str):
        return False
    if any(p.match(content) for p in _REFUSAL_PATTERNS):
        return True
    try:
        d = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return False
    return (isinstance(d, dict) and d.get("error") == "Invalid tool arguments"
            and d.get("message") == _INVALID_ARGS)


def _tool_results(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {str(m.get("tool_call_id")): m.get("content") for m in messages or []
            if m.get("role") == "tool" and m.get("tool_call_id")}


def _tool_call_ids(messages: Sequence[Mapping[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for m in messages or []:
        for c in m.get("tool_calls") or []:
            if isinstance(c, Mapping) and c.get("id"):
                ids.add(str(c["id"]))
    return ids


def harvest(home: Path) -> list[dict[str, Any]]:
    """Staged records under ``pending/{memory,skills}``, oldest first."""
    recs = []
    for sub in ("memory", "skills"):
        d = Path(home) / "pending" / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                rec = {"id": p.stem, "subsystem": sub, "unreadable": True}
            if isinstance(rec, dict):
                rec.setdefault("subsystem", sub)
                recs.append(rec)
    recs.sort(key=lambda x: (float(x.get("created_at") or 0), str(x.get("id"))))
    return recs


def _learner_record(rec: Mapping[str, Any]) -> dict[str, Any]:
    return {k: rec.get(k) for k in ("id", "subsystem", "action", "summary", "origin",
                                    "created_at", "payload")}


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, symlinks=True)
