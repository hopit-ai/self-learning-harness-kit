"""Trusted bootstrap for the XP runner (standard library only; imports nothing from slh
until the checks pass). scripts/xp/run_phase1.sh runs it:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=<fresh empty dir> \\
        python scripts/xp/xp_boot.py --package PKG [runner arguments...]

Before ``slh`` is imported:

1. Python 3.12 or newer; bytecode writing off; ``sys.pycache_prefix`` set to an
   existing EMPTY directory, so cached bytecode is read only from there (never from a
   ``__pycache__`` in the kit): planted ``.pyc`` files are not executed. Stale
   ``__pycache__`` directories in the kit are removed.
2. No executable import artifact in the kit or the benchmark copy: no ``.pyc``/``.pyo``
   outside ``__pycache__`` (Python imports a sourceless ``.pyc``), no ``.so``/``.pyd``,
   no ``.pth``, no ``sitecustomize.py`` / ``usercustomize.py``.
3. ``slh`` is not yet imported and resolves to this kit's ``slh/``.
4. Every registered code file (``registration.json`` ``source.paths``) has its
   registered git blob id, and the code hash equals ``code_sha256``.

Then it runs ``slh.xp.partner.runner`` (or, with ``--module gate``, the phase gate)
as ``__main__`` (or, with ``--serve-spec``,
prints the pinned serving values from ``slh/serve_spec.py`` as JSON for the Slurm
helpers, which never import ``slh`` themselves). Any failure prints a JSON
refusal and exits 4 (as the runner does for a registration mismatch).
"""

import hashlib
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_GLOBS = ("slh/**/*.py", "scripts/**/*.sh", "scripts/**/*.py", "configs/**/*.yaml",
              "pyproject.toml")  # = slh.xp.partner.common.CODE_GLOBS (a test checks)
PARTNER_EDITABLE = frozenset({"configs/xp-partner-endpoint.yaml"})
FORBIDDEN_SUFFIX = (".pyc", ".pyo", ".so", ".pyd", ".pth")
FORBIDDEN_NAMES = frozenset({"sitecustomize.py", "usercustomize.py"})
SKIP_TOP = frozenset({".git", ".venv", "runs"})
MODEL_PLACEHOLDER = "<MODEL>"
MODULES = frozenset({"runner", "gate"})


def refuse(reason):
    print(json.dumps({"status": "refused", "reason": "bootstrap: " + reason}, indent=1))
    sys.exit(4)


def blob_id(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_sha256(root):
    files = {}
    for pattern in CODE_GLOBS:
        for path in root.glob(pattern):
            rel = path.relative_to(root).as_posix()
            if path.is_file() and "__pycache__" not in path.parts and \
                    rel not in PARTNER_EDITABLE:
                files[rel] = sha_file(path)
    canon = json.dumps(dict(sorted(files.items())), sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def import_artifacts(root):
    """Executable import artifacts under root (paths), removing __pycache__ dirs."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        if rel.parts and rel.parts[0] in SKIP_TOP:
            dirnames[:] = []
            continue
        if "__pycache__" in dirnames:
            shutil.rmtree(Path(dirpath) / "__pycache__")  # ignored anyway (prefix)
            dirnames.remove("__pycache__")
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames:
            if name.endswith(FORBIDDEN_SUFFIX) or name in FORBIDDEN_NAMES:
                found.append((rel / name).as_posix())
    return found


def main(argv):
    if sys.version_info < (3, 12):  # noqa: UP036 - a runtime guard at the partner site
        refuse(f"Python 3.12 or newer is required, not {sys.version.split()[0]}")
    if not sys.flags.dont_write_bytecode:
        refuse("run with PYTHONDONTWRITEBYTECODE=1 (scripts/xp/run_phase1.sh sets it)")
    prefix = sys.pycache_prefix
    if not prefix or not Path(prefix).is_dir() or any(Path(prefix).iterdir()):
        refuse("PYTHONPYCACHEPREFIX must name an existing empty directory")
    if any(m == "slh" or m.startswith("slh.") for m in sys.modules):
        refuse("slh was imported before the bootstrap")
    bad = import_artifacts(ROOT)
    if bad:
        refuse(f"executable import artifacts in the kit: {bad[:5]}")
    if "--package" not in argv:
        refuse("--package is required")
    package = Path(argv[argv.index("--package") + 1])
    try:
        reg = json.loads((package / "registration.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        refuse(f"registration.json unreadable: {exc}")
    paths = (reg.get("source") or {}).get("paths") or {}
    if paths:
        differ = sorted(p for p, b in paths.items()
                        if not (ROOT / p).is_file() or blob_id(ROOT / p) != b)
        if differ:
            refuse(f"code files differ from the registration: {differ[:5]}")
    if code_sha256(ROOT) != reg.get("code_sha256"):
        refuse("code hash differs from the registration")
    sys.path.insert(0, str(ROOT))
    import importlib.util
    spec = importlib.util.find_spec("slh")
    if spec is None or Path(spec.origin).resolve().parent != ROOT / "slh":
        refuse("slh does not resolve to this kit")
    if "--serve-spec" in argv:
        # The pinned serving values for the Slurm helpers, read under the same
        # isolation (they never import slh themselves).
        import slh.serve_spec as ss

        print(json.dumps({"model_repo": ss.MODEL_REPO, "model_revision": ss.MODEL_REVISION,
                          "served_model_name": ss.SERVED_MODEL_NAME,
                          "vllm_version": ss.VLLM_VERSION,
                          "vllm_args": ss.vllm_args(MODEL_PLACEHOLDER)}, indent=1))
        return
    module = "runner"
    if "--module" in argv:  # registered entry points only
        i = argv.index("--module")
        module = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
        if module not in MODULES:
            refuse(f"module {module!r} is not an entry point")
    sys.argv = [f"python -m slh.xp.partner.{module}", *argv]
    runpy.run_module(f"slh.xp.partner.{module}", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main(sys.argv[1:])
