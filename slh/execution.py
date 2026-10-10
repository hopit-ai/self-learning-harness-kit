"""Execution identity: what must be equal for two runs' episodes to be comparable.

A finished run may stand in for a fresh one (a reused gate incumbent, a trial
extension, a merged run) only when everything that shapes an episode is the
same. ``verify_run_identity`` ENFORCES this; any mismatch refuses.

What counts (``execution_identity``):

- **Code surface** (``surface_fingerprint``): the normalized AST (docstrings,
  comments and line numbers removed) of every runtime module in
  ``RUNTIME_MODULES``, WHOLE FILES: the runner (episode loop, pairing and
  seeds, workers, coordination), the worker entry point, the agent and guards,
  the think arm, harness-state rendering, the meter and ledger, the
  self-served client and serving gate, the sandbox entry point and pinned
  serving spec, trace and capture, tokens, stubs and split verification. Any
  edit to them, including a module-level rebinding, changes the surface.
  Learner code (``slh/learner/``), the CLI, demo, report, export, preflight,
  lessons and this module (identity checks only; it never shapes an episode)
  stay out. A legacy run (no recorded fingerprint) is fingerprinted from
  ``git show <its commit>:<path>``; it must have run on a clean code tree.
- **Dependencies** (``dependency_lock``): every installed distribution and
  version in the interpreter's environment, and the full Python version. Runs
  that did not record them cannot be verified (refused).
- **Configuration**: the run's resolved config without its cap
  (``base_arm_fingerprint``: arm, think text, agent / user / judge models,
  orchestrator limits, seed and the price table hash).
- **Prompts and tool schemas**: re-rendered offline from the current code,
  the config and the harness state; must equal every episode's recorded hash.
- **Serving**: every session pinned one serving window at start
  (``serving_pin``: window id and spec hash, refused unless the spec equals the
  code's pinned ``serve_spec``) and the runner refused any request or episode
  once the active window differed; every episode records its window. All pins
  must carry the current spec.
- **tau2**: the pinned commit and the data file hashes.
- **Records** (``reconcile_records``): episode and outcome lines themselves,
  not only the manifest: exact task x trial coverage, no duplicates, the
  episode id, seed for its trial index, harness version, base arm, unique
  execution id and serving window of every episode; one valid, task-matched
  outcome each.
"""

from __future__ import annotations

import ast
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
FINGERPRINT_VERSION = 2

# Every module on the execution path, hashed whole.
RUNTIME_MODULES = (
    "slh/__init__.py",
    "slh/agent.py",
    "slh/arms/__init__.py",
    "slh/arms/think.py",
    "slh/budget.py",
    "slh/capture.py",
    "slh/ledger.py",
    "slh/openai_compat.py",
    "slh/runner.py",
    "slh/serve_entry.py",
    "slh/serve_spec.py",
    "slh/serving.py",
    "slh/splits.py",
    "slh/state.py",
    "slh/stub.py",
    "slh/tokens.py",
    "slh/trace.py",
    "slh/worker.py",
)


class IdentityMismatch(ValueError):
    """A run does not have the execution identity it is required to have."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_sha(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


# --------------------------------------------------------------------------- code surface


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _dump(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _parse(source: str) -> ast.Module:
    return _strip_docstrings(ast.parse(source))  # type: ignore[return-value]


def surface_fingerprint(read: Callable[[str], str]) -> dict[str, Any]:
    """The code surface of one source tree; ``read(relpath)`` returns the text."""
    parts = {path: _sha(_dump(_parse(read(path))).encode()) for path in RUNTIME_MODULES}
    return {"version": FINGERPRINT_VERSION, "parts": parts, "sha256": _json_sha(parts)}


def current_surface(root: Path = REPO_ROOT) -> dict[str, Any]:
    return surface_fingerprint(lambda rel: (root / rel).read_text(encoding="utf-8"))


def surface_at(commit: str, root: Path = REPO_ROOT) -> dict[str, Any]:
    """The surface of a committed tree, read with ``git show`` (nothing is checked out)."""
    def read(rel: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(root), "show", f"{commit}:{rel}"],
                                  check=True, capture_output=True, text=True).stdout
        except subprocess.CalledProcessError:
            raise IdentityMismatch(f"cannot read {rel} at commit {commit}") from None
    return surface_fingerprint(read)


def dependency_lock() -> dict[str, Any]:
    """Installed distributions (``name==version``, sorted) and the Python version.
    The harness's own editable install is excluded: its code is the surface."""
    from importlib import metadata

    pins = sorted({f"{(d.metadata['Name'] or '').lower()}=={d.version}"
                   for d in metadata.distributions()
                   if (d.metadata["Name"] or "").lower() not in ("", "self-learning-harness",
                                                                 "slh")})
    return {"python": sys.version.split()[0], "implementation": sys.implementation.name,
            "distributions": pins, "sha256": _json_sha(pins)}


# --------------------------------------------------------------------------- base arm


def resolved_without_cap(resolved: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in resolved.items() if k != "cap_usd"}


def base_arm_fingerprint(config: Any) -> dict[str, str]:
    """The fixed arm a harness state runs on: B0 (baseline) and B1 (think) share
    the empty-state hash, so every identity check pairs the state version with
    this. The cap is not part of it."""
    return {"arm": config.arm,
            "config_sha256": _json_sha(resolved_without_cap(config.resolved()))}


def base_arm_of_run(run_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    """Recorded, or rebuilt from the run's snapshot/resolved.json (legacy runs)."""
    if manifest.get("base_arm"):
        return dict(manifest["base_arm"])
    resolved = json.loads((Path(run_dir) / "snapshot" / "resolved.json").read_text())
    return {"arm": manifest.get("arm", "baseline"),
            "config_sha256": _json_sha(resolved_without_cap(resolved))}


# --------------------------------------------------------------------------- prompts


def rendered_prompt_hashes(config: Any, state: Any, domain: str = "retail") -> dict[str, str]:
    """Agent system prompt and tool schema hashes, rendered offline exactly as
    ``_run_pairs`` stores them (no backend call; the backend is never used)."""
    from tau2.registry import registry

    from slh.agent import HarnessAgent, build_guards
    from slh.arms.think import ThinkAgent
    from slh.budget import Meter
    from slh.capture import sha256_json

    class _Offline:
        def __init__(self, model: Any) -> None:
            self.config = model

        def complete(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("offline prompt rendering never calls a model")

    environment = registry.get_env_constructor(domain)()
    kwargs = ({"think_instruction": config.think_instruction,
               "think_cap_message": config.think_cap_message} if config.arm == "think" else {})
    agent_type = ThinkAgent if config.arm == "think" else HarnessAgent
    agent = agent_type(tools=environment.get_tools(), domain_policy=environment.get_policy(),
                       backend=_Offline(config.agent), meter=Meter(0.0, {}), state=state,
                       guards=build_guards(state.guards), **kwargs)
    return {"agent_system_prompt_sha256": sha256_json(agent.system_prompt),
            "tools_sha256": sha256_json([tool.openai_schema for tool in agent.tools])}


# Identity checks read run records through an allowlist projection (amendment
# 5b): scores, verdicts and transcripts are never decoded (slh.projection).
PROMPT_KEYS = frozenset({"agent_system_prompt_sha256", "tools_sha256"})
RECONCILE_EPISODE_KEYS = frozenset({
    "episode_id", "task_id", "trial", "seed", "harness_version", "execution_id",
    "base_arm", "arm", "config_sha256", "serving_window_id"})
RECONCILE_OUTCOME_KEYS = frozenset({"episode_id", "task_id", "valid"})


def recorded_prompt_hashes(run_dir: Path) -> dict[str, list[str]]:
    from slh.projection import read_projected_tolerant

    prompts, tools = set(), set()
    for episode in read_projected_tolerant(Path(run_dir) / "episodes.jsonl", PROMPT_KEYS)[0]:
        prompts.add(episode.get("agent_system_prompt_sha256"))
        tools.add(episode.get("tools_sha256"))
    return {"agent_system_prompt_sha256": sorted(map(str, prompts)),
            "tools_sha256": sorted(map(str, tools))}


# --------------------------------------------------------------------------- identity


SEED_DERIVATION = "tau2: random.Random(base_seed); one randint(0, 1_000_000) per trial index"


def compose_base_fingerprint(*, base_arm: dict[str, str], surface_sha256: str,
                             serving_spec_sha256: str | None, tau2_commit: str | None,
                             data_sha256: dict[str, str] | None,
                             base_prompts: dict[str, str], base_seed: int | None,
                             dependencies_sha256: str | None, python: str | None) -> str:
    """The single base fingerprint of docs/protocol-amendment-1.md: everything
    that defines the base arm except learned artifact content (which the state
    version hashes): arm and resolved config (models, decoding, retries,
    orchestrator limits, think text, prices), the execution code surface, the
    serving spec, tau2 commit and data, the artifact-free agent prompt and tool
    schemas (policy included), the dependency lock and Python version, the base
    seed and the seed derivation."""
    return _json_sha({"version": FINGERPRINT_VERSION, "base_arm": base_arm,
                      "surface_sha256": surface_sha256,
                      "dependencies_sha256": dependencies_sha256, "python": python,
                      "serving_spec_sha256": serving_spec_sha256,
                      "tau2_commit": tau2_commit, "data_sha256": data_sha256,
                      "base_prompts": base_prompts, "base_seed": base_seed,
                      "seed_derivation": SEED_DERIVATION})


def current_execution_fingerprint(config: Any, domain: str = "retail") -> dict[str, Any]:
    """Recorded in every new run manifest at start."""
    from slh import serve_spec
    from slh.runner import data_hashes, tau2_commit
    from slh.state import HarnessState

    surface = current_surface()
    deps = dependency_lock()
    served = getattr(config.agent, "serving_session", None) is not None
    spec = serve_spec.spec_sha256() if served else None
    base_prompts = rendered_prompt_hashes(config, HarnessState.empty(), domain)
    return {"version": FINGERPRINT_VERSION, "surface_sha256": surface["sha256"],
            "surface_parts": surface["parts"], "serving_spec_sha256": spec,
            "base_prompts": base_prompts, "python": deps["python"],
            "dependencies_sha256": deps["sha256"], "dependencies": deps["distributions"],
            "base_fingerprint_sha256": compose_base_fingerprint(
                base_arm=base_arm_fingerprint(config), surface_sha256=surface["sha256"],
                serving_spec_sha256=spec, tau2_commit=tau2_commit(),
                data_sha256=data_hashes(domain), base_prompts=base_prompts,
                base_seed=config.seed, dependencies_sha256=deps["sha256"],
                python=deps["python"])}


def run_surface(manifest: dict[str, Any]) -> tuple[str, str]:
    """(surface sha256, how it was obtained) for a run manifest."""
    recorded = manifest.get("execution_fingerprint")
    if recorded:
        return recorded["surface_sha256"], "recorded"
    identity = manifest.get("code_identity") or {}
    sha = identity.get("repo_git_sha")
    if not sha:
        raise IdentityMismatch("run records neither an execution fingerprint nor a commit")
    if identity.get("code_diff_sha256") != EMPTY_SHA256 or identity.get("untracked_code_files"):
        raise IdentityMismatch("legacy run executed uncommitted code; its surface cannot be "
                               "rebuilt from git")
    shas = {s.get("repo_git_sha") for s in manifest.get("sessions") or []} | {sha}
    if len(shas) != 1:
        raise IdentityMismatch("legacy run sessions ran at different commits")
    return surface_at(sha)["sha256"], f"git:{sha}"


def run_serving_specs(manifest: dict[str, Any]) -> list[str | None]:
    """Spec hashes of the windows the sessions pinned (legacy: sampled identity)."""
    return sorted({(s.get("serving_pin") or s.get("serving") or {}).get("spec_sha256")
                   for s in manifest.get("sessions") or []}, key=str)


def run_serving_windows(manifest: dict[str, Any]) -> set[str]:
    return {s["serving_pin"]["window_id"] for s in manifest.get("sessions") or []
            if s.get("serving_pin")}


def reconcile_records(run_dir: Path, manifest: dict[str, Any], *,
                      require_valid: bool = True) -> list[str]:
    """Problems found in the run's own episode and outcome lines (empty = none).

    Exactly one episode per (task, trial index) of the manifest, with the
    canonical episode id, the tau2 seed of its trial index, the run's harness
    version, a unique execution id, and (for runs that record them) the run's
    base arm and a serving window the session pinned; exactly one outcome per
    episode with the same task id, and (``require_valid``) ``valid: true``."""
    from slh.projection import read_projected_tolerant, same_object

    problems: list[str] = []
    episodes, torn_e = read_projected_tolerant(Path(run_dir) / "episodes.jsonl",
                                               RECONCILE_EPISODE_KEYS)
    outcomes, torn_o = read_projected_tolerant(Path(run_dir) / "outcomes.jsonl",
                                               RECONCILE_OUTCOME_KEYS)
    if torn_e is not None or torn_o is not None:
        problems.append("torn episode or outcome line")
    tasks = list(manifest.get("task_ids") or [])
    indices = trial_indices(manifest)
    seed_of = dict(zip(indices, manifest.get("seeds") or [], strict=False))
    expected = {(t, i) for t in tasks for i in indices}
    new_run = bool(manifest.get("execution_fingerprint"))
    served = bool((manifest.get("agent_model") or {}).get("serving_session"))
    windows = run_serving_windows(manifest)
    seen: dict[tuple[Any, Any], str] = {}
    executions: set[str] = set()
    task_of: dict[str, Any] = {}
    for episode in episodes:
        key = (episode.get("task_id"), episode.get("trial"))
        eid = episode.get("episode_id")
        if key in seen:
            problems.append(f"duplicate episode for task {key[0]} trial {key[1]}")
        seen[key] = eid
        task_of[eid] = key[0]
        if key not in expected:
            problems.append(f"episode {eid} is outside the task x trial set")
        if eid != f"ep-{key[0]}-{key[1]}":
            problems.append(f"episode id {eid} does not match its task and trial")
        if episode.get("seed") != seed_of.get(key[1]):
            problems.append(f"episode {eid} seed is not its trial's tau2 seed")
        if episode.get("harness_version") != manifest.get("harness_version_id"):
            problems.append(f"episode {eid} ran another harness version")
        execution = episode.get("execution_id")
        if not execution or execution in executions:
            problems.append(f"episode {eid} execution id missing or repeated")
        executions.add(execution)
        if new_run and not same_object(episode.get("base_arm"), manifest.get("base_arm")):
            problems.append(f"episode {eid} ran another base arm")
        if new_run and served and episode.get("serving_window_id") not in windows:
            problems.append(f"episode {eid} ran on a window no session pinned")
    if set(seen) != expected:
        problems.append("episodes do not cover the exact task x trial set")
    by_episode: dict[Any, dict[str, Any]] = {}
    for outcome in outcomes:
        eid = outcome.get("episode_id")
        if eid in by_episode:
            problems.append(f"duplicate outcome for {eid}")
        by_episode[eid] = outcome
        if eid not in task_of or outcome.get("task_id") != task_of[eid]:
            problems.append(f"outcome {eid} has no matching episode or task")
        if require_valid and outcome.get("valid") is not True:
            problems.append(f"outcome {eid} is not valid")
    if set(by_episode) != set(task_of):
        problems.append("outcomes do not pair one-to-one with episodes")
    return sorted(set(problems))


def execution_identity(run_dir: str | Path) -> dict[str, Any]:
    """The comparable execution identity of a finished run (read-only)."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    surface, source = run_surface(manifest)
    base = base_arm_of_run(run_dir, manifest)
    specs = run_serving_specs(manifest) \
        if (manifest.get("agent_model") or {}).get("serving_session") else [None]
    recorded = (manifest.get("execution_fingerprint") or {}).get("base_fingerprint_sha256")
    if recorded is None:
        # Legacy run: rebuilt from its own surface, config, serving and data,
        # with the artifact-free prompts rendered from the (verified) config.
        from slh.runner import load_config
        from slh.state import HarnessState

        recorded = compose_base_fingerprint(
            base_arm=base, surface_sha256=surface,
            serving_spec_sha256=specs[0] if len(specs) == 1 else "mixed",
            tau2_commit=manifest.get("tau2_commit"), data_sha256=manifest.get("data_sha256"),
            base_prompts=rendered_prompt_hashes(load_config(manifest["config_path"]),
                                                HarnessState.empty(),
                                                manifest.get("domain", "retail")),
            base_seed=manifest.get("base_seed"), dependencies_sha256=None, python=None)
    fingerprint = manifest.get("execution_fingerprint") or {}
    return {"surface_sha256": surface, "surface_source": source,
            "base_fingerprint_sha256": recorded, "base_arm": base,
            "dependencies_sha256": fingerprint.get("dependencies_sha256"),
            "python": fingerprint.get("python"),
            "harness_version_id": manifest.get("harness_version_id"),
            "state_base_arm": manifest.get("state_base_arm"),
            "serving_spec_sha256": specs,
            "prompts": recorded_prompt_hashes(run_dir),
            "tau2_commit": manifest.get("tau2_commit"),
            "data_sha256": manifest.get("data_sha256"),
            "domain": manifest.get("domain"), "split_file_sha256":
                manifest.get("split_file_sha256"),
            "task_ids": list(manifest.get("task_ids") or []),
            "base_seed": manifest.get("base_seed"), "seeds": list(manifest.get("seeds") or []),
            "trial_indices": trial_indices(manifest)}


def trial_indices(manifest: dict[str, Any]) -> list[int]:
    if manifest.get("trial_indices") is not None:
        return list(manifest["trial_indices"])
    offset = int(manifest.get("trial_offset", 0))
    return list(range(offset, offset + len(manifest.get("seeds") or [])))


def verify_run_identity(run_dir: str | Path, config: Any, state: Any, *,
                        require_complete: bool = True,
                        expect: dict[str, Any] | None = None,
                        check_prompts: bool = True,
                        allow_missing_pairs: bool = False) -> dict[str, Any]:
    """ENFORCE that a finished run executed exactly what ``config`` + ``state``
    would execute now. ``config`` is a runner RunConfig (its cap is ignored).
    ``expect`` adds run-coverage requirements (task_ids, seeds, trial_indices,
    base_seed, split_file_sha256, split_name, trials). Raises IdentityMismatch
    listing every difference; returns the verified identity otherwise.

    ``allow_missing_pairs`` (with ``require_complete=False``; the B1i recovery
    audit): absent (task, trial) pairs and episodes without an outcome are not
    identity problems; every record that IS present must still verify."""
    if allow_missing_pairs and require_complete:
        raise ValueError("allow_missing_pairs needs require_complete=False")
    from slh import serve_spec
    from slh.runner import check_tau2_pinned, data_hashes, trial_seeds

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    problems: list[str] = []

    def need(ok: bool, what: str) -> None:
        if not ok:
            problems.append(what)

    if require_complete:
        need(manifest.get("status") == "complete", "status is not complete")
        need(manifest.get("invalid_episodes") == 0, "invalid episodes")
        need(not manifest.get("not_run"), "episodes not run")
        need(manifest.get("episodes_written")
             == len(manifest.get("task_ids") or []) * len(manifest.get("seeds") or []),
             "episode count")
    coverage_gaps = {"episodes do not cover the exact task x trial set",
                     "outcomes do not pair one-to-one with episodes"}
    problems += [f"records: {p}" for p in reconcile_records(run_dir, manifest,
                                                           require_valid=require_complete)
                 if not (allow_missing_pairs and p in coverage_gaps)]
    fingerprint = manifest.get("execution_fingerprint") or {}
    deps = dependency_lock()
    need(fingerprint.get("dependencies_sha256") == deps["sha256"],
         "dependency lock differs or was not recorded")
    need(fingerprint.get("python") == deps["python"], "Python version differs or was not "
         "recorded")
    try:
        surface, source = run_surface(manifest)
    except IdentityMismatch as exc:
        surface, source = None, str(exc)
        problems.append(f"code surface unavailable ({exc})")
    current = current_surface()
    if surface is not None:
        need(surface == current["sha256"], "execution code surface differs from the current "
             f"code ({source})")
    base = base_arm_of_run(run_dir, manifest)
    need(base == base_arm_fingerprint(config), "base arm / resolved config (cap aside) differs")
    need(manifest.get("arm", "baseline") == config.arm, "arm")
    need(manifest.get("harness_version_id") == state.version_id, "harness state version")
    need(manifest.get("state_rehearsal") is False and not state.rehearsal, "rehearsal state")
    if getattr(state, "base", None) is not None:
        need(state.base == base_arm_fingerprint(config), "state is bound to another base arm")
    seeds = list(manifest.get("seeds") or [])
    indices = trial_indices(manifest)
    need(manifest.get("trials") == len(seeds) == len(indices), "trials")
    base_seed = manifest.get("base_seed")
    need(base_seed == config.seed, "base seed")
    if isinstance(base_seed, int) and indices:
        need(seeds == trial_seeds(base_seed, max(indices) + 1)[min(indices):max(indices) + 1]
             and indices == list(range(min(indices), max(indices) + 1)),
             "trial seeds are not tau2's per-trial derivation for their indices")
    if (manifest.get("agent_model") or {}).get("serving_session"):
        need(run_serving_specs(manifest) == [serve_spec.spec_sha256()],
             "serving spec (model revision, vLLM image, argv) differs")
        need(all(s.get("serving_pin") for s in manifest.get("sessions") or []),
             "a session did not pin its serving window")
    need(manifest.get("tau2_commit") == check_tau2_pinned(), "tau2 commit")
    need(manifest.get("data_sha256") == data_hashes(manifest.get("domain", "retail")),
         "tau2 data hashes")
    if check_prompts:
        recorded = recorded_prompt_hashes(run_dir)
        rendered = rendered_prompt_hashes(config, state, manifest.get("domain", "retail"))
        for key in ("agent_system_prompt_sha256", "tools_sha256"):
            need(recorded[key] == [rendered[key]], f"{key} (rendered now) differs")
    current_fp = current_execution_fingerprint(config, manifest.get("domain", "retail"))
    recorded_fp = (manifest.get("execution_fingerprint") or {}).get("base_fingerprint_sha256")
    if recorded_fp is not None:
        need(recorded_fp == current_fp["base_fingerprint_sha256"], "base fingerprint differs")
    for key, value in (expect or {}).items():
        if key == "trial_indices":
            need(indices == list(value), "trial indices")
        elif key == "task_ids":
            need(list(manifest.get("task_ids") or []) == list(value), "task ids")
        else:
            need(manifest.get(key) == value, key.replace("_", " "))
    if problems:
        raise IdentityMismatch(f"{run_dir.name}: " + "; ".join(problems))
    return {"run_dir": str(run_dir), "run_id": manifest.get("run_id"),
            "surface_sha256": surface, "surface_source": source, "base_arm": base,
            "base_fingerprint_sha256": current_fp["base_fingerprint_sha256"],
            "protocol": manifest.get("protocol"),
            "harness_version_id": manifest.get("harness_version_id"),
            "trial_indices": indices, "seeds": seeds, "task_ids": manifest.get("task_ids"),
            "split_file_sha256": manifest.get("split_file_sha256"),
            "config_resolved_sha256": manifest.get("config_resolved_sha256"),
            "budget_cap_usd": manifest.get("budget_cap_usd"),
            "spend": manifest.get("spend"), "verified": True}


# --------------------------------------------------------------------------- migration


def check_protocol_state(state: Any, protocol: Any) -> None:
    """Pre-flight (outside the runtime surface) for protocol runs: a nonempty
    legacy learned state without a base-arm binding is refused on every
    protocol run, B0 included (B1 refuses it in the runner already). Called by
    the CLI before ``run``/``run --extend`` with ``--protocol``; v2 learner
    rounds only replay the frozen empty incumbent and bound candidates."""
    if protocol is None or state.base is not None:
        return
    if state.notes or state.skills or state.guards:
        raise IdentityMismatch(f"harness state {state.version_id} is a legacy learned state "
                               "without a base-arm binding; protocol runs accept only bound "
                               "states. Migrate it explicitly: python -m slh bind-state "
                               "--round")


def verify_binding_provenance(state: Any, base_arm: dict[str, str],
                              round_dir: str | Path) -> dict[str, Any]:
    """Refuse to bind a legacy learned state unless the learner round that
    produced it shows it was learned against ``base_arm``: the round's
    candidate or published state is this state; every learned note carries a
    lesson id recorded in that round's ``lessons.jsonl``; and the round's base
    arm (recorded, or for legacy rounds the fingerprint of its runner config)
    equals ``base_arm``."""
    from slh.runner import load_config

    round_dir = Path(round_dir)
    report = json.loads((round_dir / "round.json").read_text(encoding="utf-8"))
    produced = {report.get("candidate_version"),
                Path(report["new_state"]).name if report.get("new_state") else None}
    problems = []
    if state.version_id not in produced:
        problems.append("the round did not produce this state")
    if state.skills or state.guards:
        problems.append("skills or guards have no lesson provenance")
    lessons_path = round_dir / "lessons.jsonl"
    lesson_ids = set()
    if lessons_path.exists():
        lesson_ids = {json.loads(line).get("id") for line in
                      lessons_path.read_text(encoding="utf-8").split("\n") if line.strip()}
    learned = state.notes
    if not learned or any(not note.get("lesson_id") or note["lesson_id"] not in lesson_ids
                          for note in learned):
        problems.append("a note has no lesson record in the round")
    recorded = report.get("base_arm")
    if recorded is None:
        runner_config = (report.get("config") or {}).get("runner_config")
        if not runner_config:
            problems.append("the round records neither its base arm nor its runner config")
        else:
            recorded = base_arm_fingerprint(load_config(REPO_ROOT / "configs" /
                                                        Path(runner_config).name))
    if recorded is not None and recorded != base_arm:
        problems.append(f"the round learned against base arm {recorded.get('arm')}, not "
                        f"{base_arm['arm']}")
    if problems:
        raise IdentityMismatch("bind-state refused: " + "; ".join(problems))
    return {"round": str(round_dir), "round_number": report.get("round"),
            "base_arm": base_arm, "lesson_ids": sorted(lesson_ids & {
                n.get("lesson_id") for n in learned})}


# --------------------------------------------------------------------------- merging


def _file_sha(path: Path) -> str | None:
    return _sha(path.read_bytes()) if path.exists() else None


def run_files_digest(run_dir: Path) -> dict[str, str | None]:
    return {name: _file_sha(Path(run_dir) / name)
            for name in ("episodes.jsonl", "outcomes.jsonl", "calls.jsonl")}


def merge_trials(parent: str | Path, child: str | Path, out: str | Path) -> dict[str, Any]:
    """Merge a run and its trial extension into a NEW run directory.

    Verified before anything is written: both complete with no invalid or
    missing episodes; the child names this parent and the parent's files are
    unchanged since the extension started; identical execution identity (code
    surface, base arm, harness state version and binding, serving spec, tau2
    commit and data), task ids and order, split hash and base seed; disjoint
    trial indices whose union is 0..n-1 with tau2's per-trial seeds.
    Sources are only read; ``out`` must not exist.
    """
    from slh.runner import trial_seeds
    from slh.trace import read_jsonl_tolerant

    parent, child, out = Path(parent), Path(child), Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to write into existing {out}")
    manifests = [json.loads((p / "manifest.json").read_text(encoding="utf-8"))
                 for p in (parent, child)]
    pm, cm = manifests
    problems: list[str] = []

    def need(ok: bool, what: str) -> None:
        if not ok:
            problems.append(what)

    for name, m in (("parent", pm), ("child", cm)):
        need(m.get("status") == "complete", f"{name} status is not complete")
        need(m.get("invalid_episodes") == 0, f"{name} has invalid episodes")
        need(not m.get("not_run"), f"{name} has episodes not run")
    for name, path, m in (("parent", parent, pm), ("child", child, cm)):
        problems.extend(f"{name} records: {p}" for p in reconcile_records(path, m))
    link = cm.get("parent_run") or {}
    need(link.get("run_id") == pm.get("run_id"), "child does not name this parent")
    need(link.get("files_sha256") == run_files_digest(parent),
         "parent files changed since the extension started")
    need(link.get("manifest_sha256") == _file_sha(parent / "manifest.json"),
         "parent manifest changed since the extension started")
    identities = [execution_identity(p) for p in (parent, child)]
    for key in ("surface_sha256", "base_fingerprint_sha256", "base_arm", "harness_version_id",
                "state_base_arm", "dependencies_sha256", "python",
                "serving_spec_sha256", "prompts", "tau2_commit", "data_sha256", "domain",
                "split_file_sha256", "task_ids", "base_seed"):
        need(identities[0][key] == identities[1][key], f"{key.replace('_', ' ')} differs")
    for key in ("split_name", "judge_mode", "arm", "state_rehearsal", "evidence_eligible"):
        need(pm.get(key) == cm.get(key), f"{key} differs")
    pi, ci = identities[0]["trial_indices"], identities[1]["trial_indices"]
    need(not set(pi) & set(ci), "trial indices overlap")
    union = sorted(pi + ci)
    need(union == list(range(len(union))), "merged trial indices are not 0..n-1")
    need(isinstance(pm.get("base_seed"), int), "base seed is not an integer")
    if isinstance(pm.get("base_seed"), int):
        expected = trial_seeds(pm["base_seed"], len(union))
        by_index = dict(zip(pi, pm.get("seeds") or [], strict=False)) | dict(
            zip(ci, cm.get("seeds") or [], strict=False))
        need([by_index.get(i) for i in range(len(union))] == expected,
             "seeds are not tau2's per-trial derivation")
    if problems:
        raise IdentityMismatch("merge-trials refused: " + "; ".join(problems))

    order = {i: k for k, i in enumerate(union)}
    tasks = list(pm["task_ids"])
    rank = {t: k for k, t in enumerate(tasks)}
    records: dict[str, list[dict[str, Any]]] = {}
    for name in ("episodes.jsonl", "outcomes.jsonl"):
        rows: dict[str, dict[str, Any]] = {}
        for source in (parent, child):
            found, torn = read_jsonl_tolerant(source / name)
            if torn is not None:
                raise IdentityMismatch(f"{source.name}/{name} has a torn line")
            for row in found:
                if row["episode_id"] in rows:
                    raise IdentityMismatch(f"episode {row['episode_id']} in both runs")
                rows[row["episode_id"]] = row
        records[name] = list(rows.values())
    trial_of = {e["episode_id"]: e["trial"] for e in records["episodes.jsonl"]}
    task_of = {e["episode_id"]: e["task_id"] for e in records["episodes.jsonl"]}
    seed_by_index = dict(zip(union, expected, strict=True))
    tuples = [(e["task_id"], e["trial"], e["seed"]) for e in records["episodes.jsonl"]]
    if sorted(tuples, key=str) != sorted(((t, i, seed_by_index[i]) for t in tasks
                                          for i in union), key=str):
        raise IdentityMismatch("merged episodes do not cover every task x trial x seed "
                               "exactly once")
    outcome_tasks = {o["episode_id"]: o.get("task_id") for o in records["outcomes.jsonl"]}
    if outcome_tasks != task_of or any(o.get("valid") is not True
                                       for o in records["outcomes.jsonl"]):
        raise IdentityMismatch("merged outcomes do not pair with episodes as valid, "
                               "task-matched results")

    def key(row: dict[str, Any]) -> tuple[int, int]:
        return order[trial_of[row["episode_id"]]], rank[task_of[row["episode_id"]]]

    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        from slh.runner import _write_jsonl_atomic

        for name, rows in records.items():
            _write_jsonl_atomic(staging / name, sorted(rows, key=key))
        calls: list[dict[str, Any]] = []
        think: list[dict[str, Any]] = []
        for source in (parent, child):
            calls += read_jsonl_tolerant(source / "calls.jsonl")[0]
            if (source / "think_public.jsonl").exists():
                think += read_jsonl_tolerant(source / "think_public.jsonl")[0]
        _write_jsonl_atomic(staging / "calls.jsonl", calls)
        for sub in ("blobs", "tokens"):
            for source in (parent, child):
                if (source / sub).is_dir():
                    shutil.copytree(source / sub, staging / sub, dirs_exist_ok=True)
        if (parent / "snapshot").is_dir():
            shutil.copytree(parent / "snapshot", staging / "snapshot")
        private = staging / "private"
        private.mkdir()
        (private / "README.txt").write_text(
            "NOT FOR THE LEARNER.\n\nMerged run: private records stay in the source runs "
            f"({parent.name}, {child.name}).\n", encoding="utf-8")
        spend = {"spent_usd": round(sum((m.get("spend") or {}).get("spent_usd", 0.0)
                                        for m in manifests), 9),
                 "sources": {m["run_id"]: m.get("spend") for m in manifests}}
        merged = dict(pm)
        merged.update(
            run_id=out.name, created_utc=_now(), status="complete", merged=True,
            merged_from=[{"run_id": m["run_id"], "path": str(p.resolve()),
                          "trial_indices": trial_indices(m),
                          "manifest_sha256": _file_sha(p / "manifest.json"),
                          "files_sha256": run_files_digest(p),
                          "budget_cap_usd": m.get("budget_cap_usd"),
                          "config_resolved_sha256": m.get("config_resolved_sha256")}
                         for p, m in ((parent, pm), (child, cm))],
            trials=len(union), seeds=[by_index[i] for i in range(len(union))],
            trial_offset=0, trial_indices=list(range(len(union))),
            episodes_written=len(trial_of), invalid_episodes=0, not_run=[],
            spend=spend, sessions=list(pm.get("sessions") or []) + list(
                cm.get("sessions") or []),
            base_arm=identities[0]["base_arm"],
            execution_fingerprint=cm.get("execution_fingerprint") or {
                "version": FINGERPRINT_VERSION,
                "surface_sha256": identities[0]["surface_sha256"],
                "serving_spec_sha256": identities[0]["serving_spec_sha256"][0],
                "base_fingerprint_sha256": identities[0]["base_fingerprint_sha256"]},
            # A verified backfill for a legacy parent: sources and hash, the
            # historical run itself is never edited.
            identity_backfill={"parent_surface_source": identities[0]["surface_source"],
                               "base_fingerprint_sha256":
                                   identities[0]["base_fingerprint_sha256"]},
            protocol=cm.get("protocol") or pm.get("protocol"),
            parent_run=None, workers=None, assignment=None)
        if think:
            _write_jsonl_atomic(staging / "think_public.jsonl", think)
            merged["think_projection"] = {
                "file": "think_public.jsonl", "thoughts": len(think),
                "sources": [(m.get("think_projection") or {}) for m in manifests]}
        (staging / "manifest.json").write_text(json.dumps(merged, indent=2) + "\n",
                                               encoding="utf-8")
        staging.rename(out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return merged


def _now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- worker slots

# Concurrent agent requests one serving window takes without queueing
# (serve_spec.MAX_NUM_SEQS). Every run against one serving session holds one
# slot per worker for its whole duration, across processes.
SESSION_WORKER_SLOTS = 8


def _slot_dir(session: str) -> Path:
    # serving_session paths are relative to the repo root (as the serving gate reads them).
    path = Path(session) if Path(session).is_absolute() else REPO_ROOT / session
    key = _sha(str(path.resolve()).encode())[:16]
    path = Path(os.environ.get("SLH_WORKER_SLOT_DIR") or
                Path(tempfile.gettempdir()) / "slh-worker-slots") / key
    path.mkdir(parents=True, exist_ok=True)
    return path


# A slot whose owner record survives was not released normally: its holder
# (a coordinator) died, and its workers may still be finishing. They exit
# within the worker watchdog's 0.25 s poll of the coordinator's death; the slot
# is re-admitted only after the recorded owner pid is gone AND this grace.
ORPHAN_GRACE_S = 2.0
OWNER_WAIT_S = 60.0


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _admit_slot(directory: Path, index: int) -> None:
    """Crash-recovery guard for a slot just locked: wait out a crashed owner's
    orphaned workers, then record this process as the owner."""
    import time

    owner = directory / f"slot-{index}.owner.json"
    if owner.exists():
        try:
            pid = json.loads(owner.read_text(encoding="utf-8")).get("pid")
        except ValueError:
            pid = None
        deadline = time.monotonic() + OWNER_WAIT_S
        while pid != os.getpid() and _pid_alive(pid):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"serving slot {index}: previous owner {pid} still alive")
            time.sleep(0.1)
        if pid != os.getpid():
            time.sleep(ORPHAN_GRACE_S)
    owner.write_text(json.dumps({"pid": os.getpid()}) + "\n", encoding="utf-8")


def _release_slot(directory: Path, index: int) -> None:
    with contextlib.suppress(OSError):
        (directory / f"slot-{index}.owner.json").unlink()


def wait_for_old_workers(run_dir: str | Path, timeout: float = 60.0) -> list[int]:
    """Crash recovery: before a parallel run is resumed, wait until every worker
    pid recorded by earlier sessions (``workers/*/status.json``) has exited;
    refuse after ``timeout``. Returns the pids that were waited for."""
    import time

    pids = []
    for status in sorted(Path(run_dir).glob("workers/*/status.json")):
        try:
            pid = json.loads(status.read_text(encoding="utf-8")).get("pid")
        except ValueError:
            continue
        if _pid_alive(pid) and pid != os.getpid():
            pids.append(pid)
    deadline = time.monotonic() + timeout
    while any(_pid_alive(pid) for pid in pids):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"old workers {[p for p in pids if _pid_alive(p)]} of "
                               f"{Path(run_dir).name} are still running; refusing to resume")
        time.sleep(0.1)
    return pids


@contextlib.contextmanager
def worker_slots(session: str | None, count: int,
                 total: int = SESSION_WORKER_SLOTS) -> Iterator[list[int]]:
    """Hold ``count`` of the serving session's ``total`` worker slots (fcntl
    locks, released when the process exits). Refuses rather than waits, so two
    concurrent runs (e.g. a screen and an incumbent top-up, 4 workers each) can
    never together exceed what the server serves at once. No-op without a
    serving session."""
    if session is None:
        yield []
        return
    directory = _slot_dir(session)
    held: list[tuple[int, Any]] = []
    admitted: list[int] = []
    try:
        for index in range(total):
            if len(held) == count:
                break
            handle = (directory / f"slot-{index}.lock").open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            held.append((index, handle))
        if len(held) < count:
            raise RuntimeError(f"serving session {session}: only {len(held)} of {count} "
                               f"worker slots free ({total} in all); another run is using "
                               "the server")
        for index, _ in held:
            _admit_slot(directory, index)
        admitted = [index for index, _ in held]
        yield list(admitted)
    finally:
        # Orderly exit (the runner has joined or terminated its workers by now):
        # drop the owner records. Process death skips this, leaving the record
        # that makes the next admission wait for orphans.
        for index in admitted:
            _release_slot(directory, index)
        for _, handle in held:
            handle.close()
