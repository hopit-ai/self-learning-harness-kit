"""Deterministic materialiser: ``HarnessState`` -> a fresh ``HERMES_HOME`` (spike rev 2 s. 4.3).

Layout (UTF-8, LF, sorted, fixed modes; every file below is an IMMUTABLE input):

    config.yaml                  host config (canonical JSON, which is valid YAML), role-free
    SOUL.md                      fixed persona text (from the host config)
    .slh/notes.md                prompt notes (kind "note"), passed as the ephemeral prompt
    .slh/guards.json             adopted checker specs (the chain runs on our side)
    memories/MEMORY.md           the "memory" note's entries, Hermes' delimiter
    skills/<name>/SKILL.md       one per skill
    plugins/slh-bridge/*         our guard/tool bridge plugin (copied from this package)

``.slh/manifest.json`` binds them: ``{schema, content_sha256, version_id, files: {path:
sha256}, dirs: [...]}``; ``materialization_sha256`` is the SHA-256 of its canonical JSON. The
manifest is a pure function of (state, host config, plugin source), so it can be recomputed
and stored with the version (``save_manifest``) and verified on load. Mutable paths
(``pending/``, the session DB, logs, caches, the workspace) are outside the binding and are
harvested separately. The learned-state directories (``IMMUTABLE_DIRS``) must hold exactly the
manifest's files: any extra file there after a run is a state change (incident).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from slh.state import HarnessState
from slh.system.interfaces import MEMORY_DELIMITER

SCHEMA = "slh-hermes-materialization-v1"
PLUGIN_NAME = "slh-bridge"
PLUGIN_DIR = Path(__file__).resolve().parent / "plugin"
PLUGIN_FILES = ("plugin.yaml", "__init__.py")
IMMUTABLE_DIRS = (".slh", "memories", "skills", f"plugins/{PLUGIN_NAME}")
IMMUTABLE_ROOT_FILES = ("config.yaml", "SOUL.md")
MANIFEST = ".slh/manifest.json"
# (v2.4) Hermes' memory store takes an flock on ``<memory file>.lock`` next to each memory
# file (``MemoryStore._file_lock``: O_RDWR|O_CREAT|O_NOFOLLOW, then fchmod 0600) before it
# stages a ``patch``/``remove`` (it pins the matched entry under the lock). In the read-only
# learned state that open failed, the exception escaped the memory tool, and Hermes' outer
# loop answered every call of that assistant message (siblings never reached the hook):
# the partner's c9b smoke halt (design s. 20). The lock files are created EMPTY and
# writable here, outside the manifest; the sandbox re-allows exactly these files; the
# post-run check requires them to be empty regular files.
LOCK_FILES = ("memories/MEMORY.md.lock", "memories/USER.md.lock")
HOOK_CALLBACK_TIMEOUT_S = 300
LOCK_MODE = 0o600
FILE_MODE = 0o444
DIR_MODE = 0o555
DEFAULT_SOUL = ("You are a careful assistant. Use the provided tools to complete the task, "
                "and give a short final answer.\n")


def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def host_config(*, memory_char_limit: int = 2200,
                extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The Hermes ``config.yaml`` every run uses (role-free, so one materialisation serves
    every role): write approval on for memory and skills, nudge 1 (review fires every turn),
    deferral off, USER.md off, curator off, tool search off, our plugin enabled."""
    cfg: dict[str, Any] = {
        "memory": {"memory_enabled": True, "user_profile_enabled": False,
                   "write_approval": True, "nudge_interval": 1,
                   "memory_char_limit": memory_char_limit, "provider": ""},
        "skills": {"write_approval": True, "creation_nudge_interval": 1, "ledger": False,
                   "inline_shell": False, "guard_agent_created": False, "auto_load": []},
        "auxiliary": {"background_review": {"enabled": True, "defer": "never"}},
        "curator": {"enabled": False},
        "tools": {"tool_search": {"enabled": "off"}, "connectors": {"enabled": False}},
        # (v2.6) Hermes' wall-clock cap on one pre_tool_call callback (our guard round
        # trip): default 30 s, max 600 s; a timeout blocks the call and suppresses the hook
        # for 60 s (fail closed). 300 s: far above any healthy guard decision (milliseconds),
        # below the episode's wall clock.
        "plugins": {"enabled": [PLUGIN_NAME], "hook_callback_timeout": HOOK_CALLBACK_TIMEOUT_S},
    }
    for k, v in (extra or {}).items():
        cfg[k] = v
    return cfg


def plugin_sources() -> dict[str, bytes]:
    return {f"plugins/{PLUGIN_NAME}/{n}": (PLUGIN_DIR / n).read_bytes() for n in PLUGIN_FILES}


def _notes_text(state: HarnessState) -> str:
    notes = [n for n in state.notes if n.get("kind", "note") == "note"]
    return "".join(f"- {n['content']}\n" for n in notes)


def _memory_text(state: HarnessState) -> str | None:
    mem = [n for n in state.notes if n.get("kind") == "memory" and n.get("target") == "MEMORY"]
    if len(mem) > 1:
        raise ValueError("more than one MEMORY post-state in one harness state")
    return MEMORY_DELIMITER.join(mem[0]["entries"]) if mem else None


def files_for(state: HarnessState, cfg: Mapping[str, Any], soul: str = DEFAULT_SOUL
              ) -> dict[str, bytes]:
    """Every immutable file (path -> bytes), in canonical form. Pure."""
    out: dict[str, bytes] = {
        "config.yaml": (_canon(dict(cfg)) + "\n").encode(),
        "SOUL.md": soul.replace("\r\n", "\n").encode(),
        ".slh/notes.md": _notes_text(state).encode(),
        ".slh/guards.json": (_canon([g for g in state.guards if g.get("kind") == "checker"])
                             + "\n").encode(),
    }
    mem = _memory_text(state)
    if mem is not None:
        out["memories/MEMORY.md"] = mem.encode()
    for name, text in state.skills.items():
        out[f"skills/{name}/SKILL.md"] = text.replace("\r\n", "\n").encode()
    out.update(plugin_sources())
    return out


def _dirs_of(paths: list[str]) -> list[str]:
    ds = set(IMMUTABLE_DIRS)
    for p in paths:
        parts = p.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            ds.add("/".join(parts[:i]))
    return sorted(d for d in ds if d)


def manifest_for(state: HarnessState, cfg: Mapping[str, Any], soul: str = DEFAULT_SOUL
                 ) -> dict[str, Any]:
    files = files_for(state, cfg, soul)
    return {"schema": SCHEMA, "content_sha256": state.content_sha256,
            "version_id": state.version_id,
            "files": {p: sha256_bytes(b) for p, b in sorted(files.items())},
            "dirs": _dirs_of(sorted(files))}


def materialization_sha256(manifest: Mapping[str, Any]) -> str:
    return sha256_bytes(_canon(dict(manifest)).encode())


def materialize(state: HarnessState, home: Path, cfg: Mapping[str, Any],
                soul: str = DEFAULT_SOUL) -> dict[str, Any]:
    """Write a fresh HERMES_HOME (must not exist or be empty); return the manifest. Files are
    0444 and learned-state dirs 0555 after writing (the sandbox profile also denies writes)."""
    home = Path(home)
    if home.exists() and any(home.iterdir()):
        raise FileExistsError(f"{home} is not a fresh HERMES_HOME")
    home.mkdir(parents=True, exist_ok=True)
    files = files_for(state, cfg, soul)
    man = manifest_for(state, cfg, soul)
    for d in man["dirs"]:
        (home / d).mkdir(parents=True, exist_ok=True)
    for p in sorted(files):
        fp = home / p
        with open(fp, "wb") as fh:
            fh.write(files[p])
        os.chmod(fp, FILE_MODE)
    mp = home / MANIFEST
    mp.write_bytes((_canon(man) + "\n").encode())
    os.chmod(mp, FILE_MODE)
    for p in LOCK_FILES:
        fp = home / p
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(b"")
        os.chmod(fp, LOCK_MODE)
    for d in sorted(man["dirs"], key=lambda x: -x.count("/")):
        os.chmod(home / d, DIR_MODE)
    # mutable areas Hermes needs (created writable, outside the binding)
    for d in ("pending/memory", "pending/skills", "sessions", "logs", "cache"):
        (home / d).mkdir(parents=True, exist_ok=True)
    return man


def rehash(home: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Post-run check: every manifest file's hash, and no extra file in the learned-state
    directories. Returns {ok, changed, missing, extra}."""
    home = Path(home)
    changed, missing, extra = [], [], []
    for p, h in manifest["files"].items():
        fp = home / p
        if not fp.is_file() or fp.is_symlink():
            missing.append(p)
        elif sha256_bytes(fp.read_bytes()) != h:
            changed.append(p)
    for p in LOCK_FILES:  # (v2.4) present, a regular file, and still empty
        fp = home / p
        if fp.is_symlink() or not fp.is_file():
            missing.append(p)
        elif fp.stat().st_size != 0:
            changed.append(p)
    known = set(manifest["files"]) | {MANIFEST} | set(LOCK_FILES)
    for d in IMMUTABLE_DIRS:
        root = home / d
        if not root.exists():
            missing.append(d + "/")
            continue
        for fp in root.rglob("*"):
            rel = fp.relative_to(home).as_posix()
            if "__pycache__" in rel.split("/"):
                extra.append(rel)  # a byte-code write into learned state is a change too
            elif (fp.is_file() or fp.is_symlink()) and rel not in known:
                extra.append(rel)
    mp = home / MANIFEST
    if not mp.is_file() or json.loads(mp.read_text()) != dict(manifest):
        changed.append(MANIFEST)
    return {"ok": not (changed or missing or extra), "changed": sorted(changed),
            "missing": sorted(missing), "extra": sorted(extra)}


def make_writable(home: Path) -> None:
    """Restore write permission (for cleanup only)."""
    for root, dirs, files in os.walk(home):
        os.chmod(root, 0o755)
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                os.chmod(fp, 0o644)


def save_manifest(root: Path, state: HarnessState, cfg: Mapping[str, Any],
                  soul: str = DEFAULT_SOUL) -> Path:
    """Store the manifest with the version (``<root>/<version_id>.hermes-manifest.json``);
    an existing one must be identical (never overwritten)."""
    man = manifest_for(state, cfg, soul)
    p = Path(root) / f"{state.version_id}.hermes-manifest.json"
    text = _canon(man) + "\n"
    if p.exists():
        if p.read_text() != text:
            raise ValueError(f"{p}: stored manifest differs (materialiser or host config "
                             "changed)")
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def verify_manifest(root: Path, state: HarnessState, cfg: Mapping[str, Any],
                    soul: str = DEFAULT_SOUL) -> dict[str, Any]:
    p = Path(root) / f"{state.version_id}.hermes-manifest.json"
    man = json.loads(p.read_text())
    if man != manifest_for(state, cfg, soul) or man.get("content_sha256") != \
            state.content_sha256:
        raise ValueError(f"{p}: manifest does not bind this state")
    return man
