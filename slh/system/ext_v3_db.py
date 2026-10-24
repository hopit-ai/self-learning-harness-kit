"""Runtime-only registration of the ext-v3 retail database (no generators).

The public kit's part H (M-H1) ships this module, NOT ``slh.system.ext_v3`` (database and
pool builders), ``slh.xp.retaildb_ext`` (the database generator) or ``slh.xp.tasks`` (the
case generators): the ext-v3 database travels only in the private M-H1 data package, so the
public kit cannot regenerate the pool or its answers. Constants equal ``slh.system.ext_v3``'s
(checked by a test).
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
from pathlib import Path

NAME = "ext-v3"
DOMAIN = "retail-ext-v3"
SHA256 = "2e2a4f45df73e3304456d5de8ac936a3d9527a55a0869b77a5a0e9f70be7d0cc"


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unpack(gz: Path, cache: Path) -> Path:
    """``gz`` unpacked to ``cache`` (atomic; per-process temporary name) and checked."""
    cache = Path(cache)
    if cache.exists() and _sha_file(cache) == SHA256:
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    import os

    tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
    with gzip.open(gz, "rb") as src, tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    if _sha_file(tmp) != SHA256:
        tmp.unlink()
        raise RuntimeError(f"{gz} does not unpack to the pinned ext-v3 database")
    os.replace(tmp, cache)
    return cache


def register_file(path: Path) -> str:
    """Add the (unpacked, pinned) ext-v3 database to the runtime registry."""
    from slh.xp import retaildb as RDB

    RDB.DBS[NAME] = {"path": Path(path), "sha256": SHA256, "domain": DOMAIN}
    RDB._CACHE.pop(("verified", NAME), None)
    RDB._CACHE.pop(("text", NAME), None)
    return RDB.verify(NAME)
