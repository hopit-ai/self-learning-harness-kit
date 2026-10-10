"""Sealed storage for diagnostic outputs (amendment 2, B1i).

A sealed tree is a directory holding a ``SEAL.json`` marker. Everything below
it, and everything under the repository's ``sealed/`` root, is SEALED:

- learner inputs refuse it always (``slh.learner.triage.public_path``): dev
  diagnostics are never learner evidence, even after release;
- every other ``python -m slh`` command refuses a path inside a sealed tree
  until ``python -m slh diagnostic unseal`` has verified, against the registry,
  that the round's candidate hashes are frozen and written ``UNSEALED.json``
  (``refuse_unreleased``);
- the raw files are written under umask 077 (files 600, directories 700) and,
  once sealed, every file and directory below the marker is mode 000, so a
  reader that ignores the check still fails with a permission error.

A path that cannot be inspected (a mode-000 ancestor) counts as sealed.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEALED_ROOT = REPO / "sealed"
MARKER = "SEAL.json"
RELEASE = "UNSEALED.json"

_writer: contextvars.ContextVar[bool] = contextvars.ContextVar("sealed_writer", default=False)


class SealedPath(ValueError):
    """A sealed diagnostic path was requested by a consumer that may not read it."""


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def seal_root(path: str | Path) -> Path | None:
    """The sealed tree containing ``path`` (its marker directory, or the
    ``sealed/`` root itself), or None. Uninspectable ancestors count as sealed."""
    path = Path(os.path.abspath(path))
    for candidate in (path, *path.parents):
        try:
            os.stat(candidate / MARKER)
        except PermissionError:
            return candidate
        except OSError:
            continue
        return candidate
    with contextlib.suppress(OSError):
        resolved = path.resolve()
        for root in (SEALED_ROOT, SEALED_ROOT.resolve()):
            if _under(path, root) or _under(resolved, root):
                return root
    return None


def is_sealed(path: str | Path) -> bool:
    return seal_root(path) is not None


def released(root: Path) -> bool:
    try:
        os.stat(root / RELEASE)
    except OSError:
        return False
    return True


def refuse_unreleased(paths: list[str]) -> None:
    """Refuse any path inside a sealed tree that has not been unsealed (and the
    whole ``sealed/`` root, whose children release individually)."""
    if _writer.get():
        return
    for raw in paths:
        token = raw.split("=", 1)[1] if raw.startswith("--") and "=" in raw else raw
        if not token or token.startswith("-"):
            continue
        root = seal_root(token)
        if root is None:
            continue
        if root in (SEALED_ROOT, SEALED_ROOT.resolve()) or not released(root):
            raise SealedPath(f"{token} is sealed diagnostic output; it is released only by "
                             "`python -m slh diagnostic unseal` once candidate hashes are "
                             "frozen in the registry")


@contextlib.contextmanager
def writer() -> Iterator[None]:
    """The isolated diagnostic process: may write into (and audit) a sealed tree,
    with umask 077 so every file it or its workers create is owner-only."""
    token = _writer.set(True)
    old = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(old)
        _writer.reset(token)


def tree_hashes(root: Path) -> dict[str, str]:
    """SHA-256 of every regular file below ``root`` (relative POSIX paths)."""
    found = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        found[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def lock_tree(root: Path) -> None:
    """Mode 000 for every file and directory below and including ``root``
    (children first, so the walk can finish)."""
    entries = sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    for path in entries:
        if not path.is_symlink():
            path.chmod(0o000)
    root.chmod(0o000)


def open_tree(root: Path) -> None:
    """Owner-only access again (directories 700, files 600), parents first."""
    root.chmod(0o700)
    for child in root.iterdir():
        if child.is_symlink():
            continue
        if child.is_dir():
            open_tree(child)
        else:
            child.chmod(0o600)
