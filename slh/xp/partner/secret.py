"""Secret generation seeds for partner packages (docs/xp/partner-package.md, threat model).

The case generators are deterministic and ship in the bundle, so a public seed
would let anyone regenerate the test split with its gold. Every case of a
partner package is therefore generated from a 128-bit seed derived from a
secret that never leaves us::

    seed(task) = int(HMAC-SHA256(secret, "xp-partner-seed-v1|<plan_id>|<task>")[:32 hex], 16)

The secret lives in ``.xp-secret/test-seed.key`` (git-ignored, mode 0600; 32
random bytes, hex). Shipped rows carry no seed (``strip_seed``); case ids and
group ids are hashes over the seed, so recovering it means a 2^128 search.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from pathlib import Path
from typing import Any

from slh.xp.partner.common import REPO

SECRET_PATH = REPO / ".xp-secret" / "test-seed.key"
LABEL = "xp-partner-seed-v1"


def create_secret(path: Path = SECRET_PATH) -> bytes:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; a secret is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(key.hex() + "\n")
    return key


def load_secret(path: Path = SECRET_PATH) -> bytes:
    key = bytes.fromhex(Path(path).read_text(encoding="utf-8").strip())
    if len(key) < 32:
        raise ValueError("the secret must be at least 32 bytes")
    return key


def derive_seed(secret: bytes, task: str, plan_id: str) -> int:
    if len(secret) < 32:
        raise ValueError("the secret must be at least 32 bytes")
    mac = hmac.new(secret, f"{LABEL}|{plan_id}|{task}".encode(), hashlib.sha256).hexdigest()
    return int(mac[:32], 16)


def fingerprint(secret: bytes) -> str:
    """Identifies which secret built a package (kept with our keys, never shipped)."""
    return hmac.new(secret, b"fingerprint", hashlib.sha256).hexdigest()[:16]


def generate(task: str, *, secret: bytes, plan_id: str, **kwargs: Any) -> list[dict[str, Any]]:
    """All splits of ``task`` from the secret seed (``slh.xp.tasks.cli.build``)."""
    from slh.xp.tasks.cli import build

    return build(task, seed=derive_seed(secret, task, plan_id), **kwargs)


def strip_seed(case: dict[str, Any]) -> dict[str, Any]:
    gen = {k: v for k, v in (case.get("generator") or {}).items() if k != "seed"}
    return {**case, "generator": gen}
