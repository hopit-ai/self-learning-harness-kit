"""Disconnected setup: pinned vendor checkout and wheelhouse as release assets.

Our side (with network, once per kit tag; ``scripts/xp/offline_assets.sh``)::

    python -m slh.xp.partner.offline lock > requirements.lock   # from our environment
    pip download --only-binary=:all: --platform manylinux_2_28_x86_64 \
        --python-version 3.12 -r requirements.lock -d wheelhouse/
    python -m slh.xp.partner.offline build-requirements > build-requirements.txt
    pip download --only-binary=:all: --platform manylinux_2_28_x86_64 \
        --python-version 3.12 -r build-requirements.txt -d wheelhouse/
    python -m slh.xp.partner.offline verify-wheelhouse --wheelhouse wheelhouse/ \
        --lock requirements.lock --write-build-lock   # fails unless the closure is complete
    python -m slh.xp.partner.offline manifest --assets DIR       # sha256 of each asset

Partner side (no network; ``scripts/xp/setup_offline.sh``)::

    tar xzf tau2-bench-<commit>.tar.gz -C vendor/
    python -m slh.xp.partner.offline verify-wheelhouse --wheelhouse wheelhouse/ \
        --lock wheelhouse/requirements.lock
    pip install --no-index --find-links wheelhouse/ -r wheelhouse/requirements.lock
    pip install --no-index --find-links wheelhouse/ --no-deps -e vendor/tau2-bench
    python -m slh.xp.partner.offline check [--assets DIR] [--wheelhouse wheelhouse/]

The vendored benchmark builds with hatchling (``[build-system]`` of its
pyproject.toml) and an editable install also needs ``editables``; pip builds it
in an isolated environment filled from the wheelhouse, so the wheelhouse must
hold the build backend AND its whole dependency closure for Linux x86_64 /
CPython 3.12 (packaging, pathspec, pluggy, trove-classifiers, ...). They are
downloaded explicitly (never taken from our environment, where they need not be
installed) and ``verify-wheelhouse`` resolves the closure from the wheels' own
metadata, failing on any requirement without a matching wheel.

``check`` opens no connection (every socket connect is refused while it
runs): the vendor checkout equals the pin (commit, tree, data hashes), the
installed benchmark is the kit's own vendor copy, the required packages import
at a version the lock allows (an existing phase-1 environment with vLLM is
reused as is when it passes), the build backend is importable here or
installable offline from ``--wheelhouse`` (complete closure), and
(``--assets``) every asset matches the manifest. Exit 0 = ready, 1 = not ready
(the problems are listed).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import socket
import sys
import tomllib
import zipfile
from collections.abc import Iterator
from email.parser import Parser
from importlib import metadata
from pathlib import Path
from typing import Any

from slh.xp.partner import common as pc
from slh.xp.partner import vendor

ROOTS = ("tau2", "pyyaml", "jinja2", "pydantic")  # pyproject.toml dependencies
# Imported by the phase-1/2 runtime (name to import, distribution).
IMPORTS = (("tau2", "tau2"), ("yaml", "PyYAML"), ("jinja2", "Jinja2"),
           ("pydantic", "pydantic"), ("litellm", "litellm"), ("loguru", "loguru"),
           ("numpy", "numpy"))
BUILD_BACKEND = "hatchling"  # the vendored benchmark's [build-system] backend
EDITABLE_BUILD = ("editables",)  # hatchling needs it for an editable install
BUILD_LOCK = "build.lock"
LINUX = {"sys_platform": "linux", "platform_system": "Linux", "os_name": "posix",
         "platform_machine": "x86_64", "python_version": "3.12",
         "python_full_version": "3.12.0", "implementation_name": "cpython",
         "platform_python_implementation": "CPython"}
VENDOR_ASSET = f"tau2-bench-{vendor.COMMIT}.tar.gz"
WHEELHOUSE_ASSET = "wheelhouse-linux-x86_64-py312.tar.gz"
MANIFEST = "offline-assets.json"
MANIFEST_SCHEMA = "xp-offline-assets-v1"


def _norm(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def lock(roots: tuple[str, ...] = ROOTS, env: dict[str, str] | None = None) -> list[str]:
    """``name==version`` for every distribution the roots need on the partner's
    platform (Linux x86_64, Python 3.12), at the versions installed HERE; the
    benchmark itself is installed from the vendor checkout, not from the lock."""
    from packaging.requirements import Requirement

    env = {**LINUX, **(env or {})}
    seen: dict[str, str] = {}
    todo: list[tuple[str, frozenset[str]]] = [(r, frozenset()) for r in roots]
    while todo:
        name, extras = todo.pop()
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            raise SystemExit(f"{name} is not installed here; cannot lock it") from None
        key = _norm(dist.metadata["Name"])
        first = key not in seen
        seen[key] = dist.version
        for raw in dist.requires or []:
            req = Requirement(raw)
            ok = req.marker is None or any(
                req.marker.evaluate({**env, "extra": e}) for e in (extras or {""}))
            if ok and (first or extras):
                todo.append((req.name, frozenset(req.extras)))
    seen.pop("tau2", None)
    # The build backend is NOT taken from this environment (it need not be
    # installed here): ``build_requirements`` + ``pip download`` + verify_wheelhouse.
    return [f"{k}=={v}" for k, v in sorted(seen.items())]


def build_requirements(root: Path = pc.REPO) -> list[str]:
    """The vendored benchmark's build requirements ([build-system].requires of its
    pyproject.toml) plus what an editable install adds; refused unless the
    backend is the one this module knows how to check."""
    path = Path(root) / "vendor/tau2-bench/pyproject.toml"
    system = tomllib.loads(path.read_text(encoding="utf-8")).get("build-system") or {}
    if not str(system.get("build-backend", "")).startswith(BUILD_BACKEND):
        raise SystemExit(f"{path}: build backend {system.get('build-backend')!r} is not "
                         f"{BUILD_BACKEND}; update slh.xp.partner.offline")
    reqs = list(system.get("requires") or [])
    from packaging.requirements import Requirement

    have = {_norm(Requirement(r).name) for r in reqs}
    return reqs + [e for e in EDITABLE_BUILD if _norm(e) not in have]


_WHEEL = re.compile(r"^(?P<name>[^-]+)-(?P<version>[^-]+)(?:-\d[^-]*)?-[^-]+-[^-]+-[^-]+\.whl$")


def _wheels(wheelhouse: Path) -> dict[str, list[tuple[str, Path]]]:
    out: dict[str, list[tuple[str, Path]]] = {}
    for p in sorted(Path(wheelhouse).glob("*.whl")):
        m = _WHEEL.match(p.name)
        if m:
            out.setdefault(_norm(m["name"]), []).append((m["version"], p))
    return out


def _wheel_requires(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as z:
        meta = [n for n in z.namelist()
                if n.endswith(".dist-info/METADATA") and n.count("/") == 1]
        if not meta:
            raise ValueError(f"{path.name}: no METADATA")
        msg = Parser().parsestr(z.read(meta[0]).decode("utf-8", "replace"))
    return msg.get_all("Requires-Dist") or []


def wheelhouse_closure(wheelhouse: Path, roots: list[str],
                       env: dict[str, str] | None = None) -> dict[str, Any]:
    """Resolve ``roots`` (requirement strings) against the wheels in ``wheelhouse``
    using each wheel's own Requires-Dist (markers for Linux x86_64, CPython 3.12),
    no network. Returns {"pins": {name: version}, "problems": [...]}: a
    requirement with no wheel satisfying it is a problem."""
    from packaging.requirements import Requirement
    from packaging.version import InvalidVersion, Version

    env = {**LINUX, **(env or {})}
    wheels = _wheels(Path(wheelhouse))
    pins: dict[str, str] = {}
    problems: list[str] = []
    done: set[tuple[str, frozenset[str]]] = set()
    todo = [(r, "root") for r in roots]
    while todo:
        raw, why = todo.pop()
        req = Requirement(raw)
        key = _norm(req.name)
        # Every incoming constraint is checked, also against a package already
        # pinned; only the expansion of its own requirements is deduplicated.
        if key in pins:
            try:
                ok = req.specifier.contains(Version(pins[key]), prereleases=True)
            except InvalidVersion:
                ok = False
            if not ok:
                problems.append(f"{key} {pins[key]} does not satisfy {raw} (needed by {why})")
                continue
        if (key, frozenset(req.extras)) in done:
            continue
        done.add((key, frozenset(req.extras)))
        found = None
        for version, path in sorted(wheels.get(key, []), key=lambda vp: _vkey(vp[0]),
                                    reverse=True):
            try:
                ok = req.specifier.contains(Version(version), prereleases=True)
            except InvalidVersion:
                ok = False
            if ok and (key not in pins or pins[key] == version):
                found = (version, path)
                break
        if found is None:
            problems.append(f"no wheel for {raw} (needed by {why})")
            continue
        pins[key] = found[0]
        try:
            requires = _wheel_requires(found[1])
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            problems.append(f"{found[1].name}: unreadable ({exc})")
            continue
        for dep in requires:
            d = Requirement(dep)
            if d.marker is None or any(d.marker.evaluate({**env, "extra": e})
                                       for e in (req.extras or {""})):
                todo.append((dep, f"{key} {found[0]}"))
    return {"pins": dict(sorted(pins.items())), "problems": sorted(set(problems))}


def _vkey(version: str) -> Any:
    from packaging.version import InvalidVersion, Version

    try:
        return (1, Version(version))
    except InvalidVersion:
        return (0, version)


def wheelhouse_problems(wheelhouse: Path, *, lockfile: Path | None = None,
                        root: Path = pc.REPO, write_build_lock: bool = False
                        ) -> dict[str, Any]:
    """The wheelhouse holds the build backend's full closure (and, with
    ``lockfile``, every pinned runtime requirement's closure). With
    ``write_build_lock`` (our side) the resolved build pins are written to
    ``build.lock`` in the wheelhouse; otherwise an existing build.lock must
    agree with the resolution."""
    wheelhouse = Path(wheelhouse)
    build = wheelhouse_closure(wheelhouse, build_requirements(root))
    problems = [f"build backend: {p}" for p in build["problems"]]
    if lockfile is not None:
        pinned = [f"{k}=={v}" for k, v in _locked(lockfile).items()]
        if not pinned:
            problems.append(f"{lockfile}: no pins")
        runtime = wheelhouse_closure(wheelhouse, pinned)
        problems += [f"runtime: {p}" for p in runtime["problems"]]
    lock_path = wheelhouse / BUILD_LOCK
    text = "".join(f"{k}=={v}\n" for k, v in build["pins"].items())
    if write_build_lock and not build["problems"]:
        lock_path.write_text(text, encoding="utf-8")
    elif lock_path.is_file() and _locked(lock_path) != build["pins"]:
        problems.append(f"{BUILD_LOCK} differs from the build closure resolved from the "
                        "wheelhouse")
    return {"ok": not problems, "problems": problems, "build_pins": build["pins"]}


def build_backend_problems(wheelhouse: Path | None, root: Path = pc.REPO) -> list[str]:
    """The benchmark's build backend is importable here, or installable offline
    (its whole closure is in ``wheelhouse``)."""
    missing = []
    for module in ("hatchling.build", *EDITABLE_BUILD):
        try:
            __import__(module)
        except Exception:  # noqa: BLE001 - reported
            missing.append(module)
    if not missing:
        return []
    if wheelhouse is None:
        return [(f"build backend: {missing} not importable here and no --wheelhouse to "
                 "install it from offline")]
    if not Path(wheelhouse).is_dir():
        return [f"build backend: wheelhouse {wheelhouse} missing"]
    return [f"build backend: {p}" for p in
            wheelhouse_closure(Path(wheelhouse), build_requirements(root))["problems"]]


@contextlib.contextmanager
def no_network() -> Iterator[None]:
    """Refuse every outgoing connection while the block runs."""
    def refuse(*_a: Any, **_k: Any) -> None:
        raise OSError("network disabled during the offline check")

    saved = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)
    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.socket.connect_ex = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = saved


def asset_problems(assets: Path) -> list[str]:
    path = Path(assets) / MANIFEST
    if not path.is_file():
        return [f"{MANIFEST} missing from {assets}"]
    man = pc.read_json(path)
    out = []
    if man.get("schema") != MANIFEST_SCHEMA:
        out.append(f"{MANIFEST}: unknown schema")
    if man.get("vendor_commit") != vendor.COMMIT:
        out.append(f"{MANIFEST}: assets for another benchmark commit")
    for name, meta in sorted((man.get("files") or {}).items()):
        f = Path(assets) / name
        if not f.is_file():
            out.append(f"{name}: missing")
        elif pc.sha_file(f) != meta["sha256"]:
            out.append(f"{name}: sha256 differs from {MANIFEST}")
    return out


def manifest(assets: Path) -> dict[str, Any]:
    """Our side: hash every asset next to the manifest."""
    assets = Path(assets)
    files = {p.name: {"sha256": pc.sha_file(p), "bytes": p.stat().st_size}
             for p in sorted(assets.iterdir()) if p.is_file() and p.name != MANIFEST}
    body = {"schema": MANIFEST_SCHEMA, "vendor_commit": vendor.COMMIT,
            "vendor_tree": vendor.TREE, "files": files}
    pc.write_json(assets / MANIFEST, body)
    return body


def _locked(lockfile: Path | None) -> dict[str, str]:
    if lockfile is None or not Path(lockfile).is_file():
        return {}
    out = {}
    for line in Path(lockfile).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            out[_norm(name)] = version.strip()
    return out


def installed_origin(module: str) -> Path | None:
    """Where ``module`` resolves in this environment from outside the kit directory
    (its installed binding, ignoring the current directory); None if not installed."""
    import subprocess
    import tempfile

    code = ("import importlib.util as u, sys\n"
            "s = u.find_spec(sys.argv[1])\n"
            "print(s.origin if s and s.origin else '')")
    with tempfile.TemporaryDirectory() as cwd:
        res = subprocess.run([sys.executable, "-I", "-c", code, module], cwd=cwd,
                             capture_output=True, text=True, check=False)
    out = res.stdout.strip()
    return Path(out).resolve() if res.returncode == 0 and out else None


def slh_binding_problems(root: Path = pc.REPO) -> list[str]:
    """An ``slh`` installed in the environment must be this kit's (or absent: the kit
    runs from its own directory)."""
    at = installed_origin("slh")
    kit = (Path(root) / "slh").resolve()
    if at is None or kit in at.parents:
        return []
    return [(f"slh is installed from {at.parent}, not from this kit ({kit}); fix: rerun "
             "scripts/xp/setup_offline.sh ASSETS_DIR --reuse-env, or "
             f"{sys.executable} -m pip uninstall -y slh (the kit runs from its directory)")]


def check(*, assets: Path | None = None, lockfile: Path | None = None,
          wheelhouse: Path | None = None, root: Path = pc.REPO) -> dict[str, Any]:
    """Everything the run needs, with the network disabled."""
    with no_network():
        problems = [f"vendor: {p}" for p in vendor.problems(Path(root) / "vendor/tau2-bench")]
        problems += build_backend_problems(wheelhouse, root)
        packages: dict[str, Any] = {}
        locked = _locked(lockfile)
        for module, dist in IMPORTS:
            try:
                mod = __import__(module)
                version = metadata.version(dist)
            except Exception as exc:  # noqa: BLE001 - reported
                problems.append(f"{module}: not importable ({type(exc).__name__})")
                continue
            packages[dist] = version
            want = locked.get(_norm(dist))
            if want and want != version and module != "tau2":
                problems.append(f"{dist} {version} differs from the lock ({want})")
            if module == "tau2":
                src = Path(getattr(mod, "__file__", "") or "").resolve()
                vend = (Path(root) / "vendor/tau2-bench").resolve()
                if vend not in src.parents:
                    problems.append(f"tau2 is imported from {src.parent}, not from this kit's "
                                    f"{vend} (an environment reused from another kit "
                                    "checkout keeps that checkout's editable tau2); fix: "
                                    "rerun scripts/xp/setup_offline.sh ASSETS_DIR "
                                    "--reuse-env, which rebinds it offline, or run "
                                    f"{sys.executable} -m pip install --no-index "
                                    "--find-links .offline/wheelhouse --no-deps -e "
                                    "vendor/tau2-bench from the kit directory")
        problems += slh_binding_problems(root)
        if assets is not None:
            problems += asset_problems(Path(assets))
        data = None
        if not any(p.startswith("tau2") for p in problems):
            from slh.runner import data_hashes

            data = data_hashes("retail")
    return {"ok": not problems, "problems": problems, "packages": packages,
            "python": sys.version.split()[0], "data_sha256": data,
            "vendor": vendor.state(Path(root) / "vendor/tau2-bench")}


# --------------------------------------------------------------------------- hand-off 1

# Exported in the shell that starts the vLLM server and runs every part (runbook section 0).
H1_RUN_ENV = {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "HF_HUB_OFFLINE": "1",
              "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}
LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")


def _initial_environ() -> dict[str, str] | None:
    """The environment this process started with (Linux ``/proc/self/environ``), before
    ``slh`` set its in-process defaults; None where it cannot be read."""
    try:
        raw = Path("/proc/self/environ").read_bytes()
    except OSError:
        return None
    return dict(item.split("=", 1) for item in raw.decode("utf-8", "replace").split("\0")
                if "=" in item)


def _shell_env(name: str) -> str | None:
    import os

    initial = _initial_environ()
    return initial.get(name) if initial is not None else os.environ.get(name)


def endpoint_problems(path: Path) -> list[str]:
    """An endpoint file of hand-off 1: local server only, the pinned model and vLLM, and
    no server-side sampling defaults."""
    from urllib.parse import urlparse

    import yaml

    from slh import serve_spec

    try:
        ep = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return [f"{path}: unreadable ({type(exc).__name__})"]
    out = []
    host = urlparse(str(ep.get("base_url") or "")).hostname or ""
    if host not in LOOPBACK:
        out.append(f"{path}: base_url host {host!r} is not this node (127.0.0.1): the run "
                   "would need the network")
    if ep.get("model") != serve_spec.SERVED_MODEL_NAME:
        out.append(f"{path}: model must be {serve_spec.SERVED_MODEL_NAME!r}")
    if ep.get("vllm_version") != serve_spec.VLLM_VERSION:
        out.append(f"{path}: vllm_version {ep.get('vllm_version')!r} is not the pinned "
                   f"{serve_spec.VLLM_VERSION}")
    if ep.get("model_revision") != serve_spec.MODEL_REVISION:
        out.append(f"{path}: model_revision is not the pinned {serve_spec.MODEL_REVISION}")
    out += [f"{path}: {p}" for p in pc.serving_sampling_problems(ep.get("serving_args"))]
    return out


def vllm_problems(vllm_cmd: str | None) -> tuple[list[str], str | None]:
    """``VLLM --version`` (the phase-1 vLLM environment, reused) with every proxy pointed
    at a closed local port and the hub offline: must print the pinned version."""
    import os
    import shutil
    import subprocess

    from slh import serve_spec

    cmd = vllm_cmd or shutil.which("vllm")
    if not cmd:
        return [(f"vLLM not found: reuse your phase-1 vLLM {serve_spec.VLLM_VERSION} "
                 "environment and pass --vllm /path/to/bin/vllm (or put it on PATH)")], None
    closed = "http://127.0.0.1:9"
    env = {**os.environ, **H1_RUN_ENV, "HTTP_PROXY": closed, "HTTPS_PROXY": closed,
           "http_proxy": closed, "https_proxy": closed, "NO_PROXY": "", "no_proxy": ""}
    try:
        res = subprocess.run([cmd, "--version"], capture_output=True, text=True, env=env,
                             timeout=900, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"vLLM: {cmd} --version failed ({type(exc).__name__})"], None
    lines = [ln.strip() for ln in (res.stdout + res.stderr).splitlines() if ln.strip()]
    version = next((ln for ln in reversed(lines) if re.fullmatch(r"\d+\.\d+\.\d+\S*", ln)),
                   None)
    if res.returncode != 0 or version is None:
        return [f"vLLM: {cmd} --version exited {res.returncode} without a version"], None
    if version != serve_spec.VLLM_VERSION:
        return [f"vLLM {version} at {cmd} is not the pinned {serve_spec.VLLM_VERSION}"], version
    return [], version


# The 9B weights at the pinned revision (``serve_spec.MODEL_REPO`` @ ``MODEL_REVISION``),
# from the hub's file listing of that revision: name -> (bytes, kind, digest). "blob" is
# the git blob SHA-1 of a small file, "sha256" the LFS SHA-256 of a large one. Config,
# index and tokenizer identify the revision; the index must name exactly these shards.
WEIGHTS_PIN: dict[str, tuple[int, str, str]] = {
    "config.json": (3126, "blob", "273ce437e01baf96a07cd9eb3d5f48bac8d7c657"),
    "model.safetensors.index.json": (79657, "blob", "e4c1cb7dba5096b43b9d92bc781aba5e3aa8acd8"),
    "tokenizer_config.json": (16710, "blob", "eda48d3e75a8e59a8479ee4ec8b37f76e711d9c1"),
    "tokenizer.json": (12807982, "sha256",
                       "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
    "model.safetensors-00001-of-00004.safetensors": (
        5276436216, "sha256", "db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e"),
    "model.safetensors-00002-of-00004.safetensors": (
        5335161512, "sha256", "31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330"),
    "model.safetensors-00003-of-00004.safetensors": (
        5368717440, "sha256", "7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec"),
    "model.safetensors-00004-of-00004.safetensors": (
        3325995712, "sha256", "b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6"),
}
WEIGHTS_INDEX = "model.safetensors.index.json"


def _git_blob_sha1(path: Path) -> str:
    import hashlib

    body = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest()


def weights_problems(model_dir: Path, *, hash_shards: bool = False,
                     pin: dict[str, tuple[int, str, str]] | None = None) -> list[str]:
    """The weights directory holds the pinned revision: every pinned file with its size;
    config, index and tokenizers byte-identical (digests); the index names exactly the
    pinned shards. Shard SHA-256s (about 19 GB to read) only with ``hash_shards``."""
    pin = WEIGHTS_PIN if pin is None else pin
    d = Path(model_dir)
    if not d.is_dir():
        return [f"model weights: {d} is not a directory"]
    problems: list[str] = []
    for name, (size, kind, digest) in sorted(pin.items()):
        f = d / name
        if not f.is_file():
            problems.append(f"model weights: {name} missing in {d}")
            continue
        if f.stat().st_size != size:
            problems.append(f"model weights: {name} has {f.stat().st_size} bytes, "
                            f"the pinned revision {size}")
            continue
        shard = name.endswith(".safetensors")
        if shard and not hash_shards:
            continue
        got = _git_blob_sha1(f) if kind == "blob" else pc.sha_file(f)
        if got != digest:
            problems.append(f"model weights: {name} differs from the pinned revision "
                            f"({kind} {got[:12]} != {digest[:12]})")
    index = d / WEIGHTS_INDEX
    if index.is_file():
        try:
            named = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        except (ValueError, KeyError, AttributeError) as exc:
            problems.append(f"model weights: {WEIGHTS_INDEX} unreadable ({exc})")
        else:
            pinned = {n for n in pin if n.endswith(".safetensors")}
            if named != pinned:
                problems.append(f"model weights: {WEIGHTS_INDEX} names shards "
                                f"{sorted(named ^ pinned)} outside the pinned set")
    return problems


def preflight_h1(*, package_h1: Path | None, package_bfcl: Path | None,
                 endpoints: list[Path], vllm_cmd: str | None, model_dir: Path | None,
                 assets: Path | None = None, lockfile: Path | None = None,
                 wheelhouse: Path | None = None, skip_vllm: bool = False,
                 hash_weights: bool = False, root: Path = pc.REPO) -> dict[str, Any]:
    """Hand-off 1, disconnected: everything parts A and B need is on this node, and
    nothing would reach the network. Fails (``ok`` false) on any problem."""
    problems: list[str] = []
    env = {k: _shell_env(k) for k in H1_RUN_ENV}
    for k, want in H1_RUN_ENV.items():
        if env[k] != want:
            problems.append(f"run shell: export {k}={want} (found {env[k]!r})")
    base = check(assets=assets, lockfile=lockfile, wheelhouse=wheelhouse, root=root)
    problems += base["problems"]
    tokenizer = None
    with no_network():
        try:
            import litellm

            if not getattr(litellm, "model_cost", None):
                problems.append("litellm: no local price map")
        except Exception as exc:  # noqa: BLE001 - reported
            problems.append(f"litellm: not importable offline ({type(exc).__name__})")
        try:
            from slh.learner import note_tokens

            tokenizer = str(note_tokens.tokenizer_path())
            note_tokens.count_tokens("check")
        except Exception as exc:  # noqa: BLE001 - reported
            problems.append(f"agent tokenizer: {exc} (export SLH_AGENT_TOKENIZER="
                            "/path/to/Qwen3.5-9B/tokenizer.json)")
        if package_h1 is not None:
            from slh.xp.fgate import units

            try:
                pkg = units.load_package(Path(package_h1))
                problems += [f"package-h1 code: {m}" for m in
                             units.code_mismatches(pkg["plan"]["code"], root)]
            except Exception as exc:  # noqa: BLE001 - reported
                problems.append(f"package-h1: {exc}"[:300])
        if package_bfcl is not None:
            from slh.xp.bfcl import runner as bfcl_runner

            try:
                bfcl_runner.load_package(Path(package_bfcl))
            except Exception as exc:  # noqa: BLE001 - reported
                problems.append(f"package-bfcl-h1: {exc}"[:300])
    for ep in endpoints:
        problems += endpoint_problems(ep)
    if model_dir is not None:
        problems += weights_problems(Path(model_dir), hash_shards=hash_weights)
    version = None
    if not skip_vllm:
        vp, version = vllm_problems(vllm_cmd)
        problems += vp
    return {"ok": not problems, "problems": problems, "run_env": env,
            "tokenizer": tokenizer, "vllm": version, "packages": base["packages"],
            "python": base["python"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.partner.offline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="disconnected setup check (no network)")
    p.add_argument("--assets", type=Path, help="directory with the release assets")
    p.add_argument("--lock", type=Path, help="requirements.lock from the wheelhouse")
    p.add_argument("--wheelhouse", type=Path, help="the unpacked wheelhouse (the build "
                                                   "backend must be installable from it)")
    sub.add_parser("lock", help="our side: print the pinned requirements")
    sub.add_parser("build-requirements", help="our side: print the benchmark's build "
                                              "requirements (for pip download)")
    p = sub.add_parser("verify-wheelhouse", help="the wheelhouse holds the full build "
                                                 "(and runtime) closure; no network")
    p.add_argument("--wheelhouse", type=Path, required=True)
    p.add_argument("--lock", type=Path)
    p.add_argument("--write-build-lock", action="store_true",
                   help="our side: record the resolved build pins as build.lock")
    p = sub.add_parser("manifest", help="our side: hash the release assets")
    p.add_argument("--assets", type=Path, required=True)
    p = sub.add_parser("preflight-h1", help="hand-off 1: disconnected preflight (fails if "
                                            "anything would need the network)")
    p.add_argument("--assets", type=Path, help="directory with the release assets")
    p.add_argument("--lock", type=Path)
    p.add_argument("--wheelhouse", type=Path)
    p.add_argument("--package-h1", type=Path)
    p.add_argument("--package-bfcl", type=Path)
    p.add_argument("--endpoint", type=Path, action="append", default=[])
    p.add_argument("--vllm", help="the phase-1 vLLM executable (default: vllm on PATH)")
    p.add_argument("--model-dir", type=Path, help="the 9B weights directory")
    p.add_argument("--skip-vllm", action="store_true", help="rehearsals without vLLM only")
    p.add_argument("--hash-weights", action="store_true",
                   help="also hash every weight shard (reads about 19 GB)")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if args.cmd == "lock":
        print("\n".join(lock()))
        return 0
    if args.cmd == "build-requirements":
        print("\n".join(build_requirements()))
        return 0
    if args.cmd == "verify-wheelhouse":
        with no_network():
            res = wheelhouse_problems(args.wheelhouse, lockfile=args.lock,
                                      write_build_lock=args.write_build_lock)
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 1
    if args.cmd == "manifest":
        print(json.dumps(manifest(args.assets), indent=1))
        return 0
    if args.cmd == "preflight-h1":
        res = preflight_h1(package_h1=args.package_h1, package_bfcl=args.package_bfcl,
                           endpoints=args.endpoint, vllm_cmd=args.vllm,
                           model_dir=args.model_dir, assets=args.assets, lockfile=args.lock,
                           wheelhouse=args.wheelhouse, skip_vllm=args.skip_vllm,
                           hash_weights=args.hash_weights)
        print(json.dumps(res, indent=1, default=str))
        return 0 if res["ok"] else 1
    res = check(assets=args.assets, lockfile=args.lock, wheelhouse=args.wheelhouse)
    print(json.dumps(res, indent=1, default=str))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
