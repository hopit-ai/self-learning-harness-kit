"""Pinned tau2 retail databases: the ORIGINAL ``db.json`` and our synthetic extension.

Rule (docs/xp/retail-db-extension.md): official tau2 tasks always run on the ORIGINAL
database. Our generated decision prefixes and scenarios may use the extension
(``ext-v2``, current: the original 500 users and 1,000 orders byte-identical, plus 5,000
synthetic users and their orders drawn from the existing product catalogue by
``slh.xp.retaildb_ext``; ``ext-v1``, 2,000 users, kept for provenance).

* ``use(name)``: the database the XP generators draw from and build their tool results on
  (``slh.xp.tasks.common.load_db`` / ``new_env``); default ``original``, unchanged.
* ``env(name)``: a fresh tau2 retail environment on that database (own copy; writes do
  not leak). For ``original`` it is tau2's registered constructor, exactly as before.
* ``domain(name)``: the tau2 domain name a conversation run uses (``retail``,
  ``retail-ext-v1`` or ``retail-ext-v2``); ``register_domains()`` registers each
  extension with tau2's registry (same policy, same tools, only the database differs), so
  the runner, tau2's evaluator and replay scoring all build their environments from the
  same pinned file.
* every file is checked against its pinned SHA-256 before use (a different file refuses).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
ORIGINAL = "original"
EXT_V1 = "ext-v1"
EXT_V2 = "ext-v2"
# The database our GENERATED decision prefixes / test sets default to (CLI defaults of
# fgate.design, fgate.build, partner.release h1). Official tau2 tasks: always ORIGINAL.
GENERATED_DEFAULT = EXT_V2
ORIGINAL_PATH = REPO / "vendor/tau2-bench/data/tau2/domains/retail/db.json"
EXT_V1_PATH = REPO / "data/tau2-retail-ext/db-ext-v1.json"  # kept for provenance
EXT_V2_PATH = REPO / "data/tau2-retail-ext/db-ext-v2.json"
# SHA-256 of the file bytes (the original is the vendor pin, slh.xp.partner.vendor).
ORIGINAL_SHA256 = "413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765"
EXT_V1_SHA256 = "c19d1242f666289c473f78f30af915e8233a458ba8297e74a0a251a370a2ef57"
EXT_V2_SHA256 = "3d03098276f5b2dd2ec08988f11e4f3d239e7ac0d052991f63807f35760130e5"
DBS: dict[str, dict[str, Any]] = {
    ORIGINAL: {"path": ORIGINAL_PATH, "sha256": ORIGINAL_SHA256, "domain": "retail"},
    EXT_V1: {"path": EXT_V1_PATH, "sha256": EXT_V1_SHA256, "domain": "retail-ext-v1"},
    EXT_V2: {"path": EXT_V2_PATH, "sha256": EXT_V2_SHA256, "domain": "retail-ext-v2"},
}
_ACTIVE = {"name": ORIGINAL}
_CACHE: dict[str, Any] = {}


class DbRefused(ValueError):
    """Unknown database name, or a file that differs from its pin."""


def names() -> tuple[str, ...]:
    return tuple(DBS)


def spec(name: str) -> dict[str, Any]:
    if name not in DBS:
        raise DbRefused(f"unknown retail database {name!r} (known: {', '.join(DBS)})")
    return DBS[name]


def path(name: str) -> Path:
    return Path(spec(name)["path"])


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(name: str) -> str:
    """The pinned SHA-256 of ``name`` after checking the file on disk against it."""
    s = spec(name)
    key = ("verified", name)
    if key not in _CACHE:
        p = Path(s["path"])
        if not p.is_file():
            raise DbRefused(f"{name}: {p} is missing (python -m slh.xp.retaildb_ext build)")
        have = sha256_file(p)
        if have != s["sha256"]:
            raise DbRefused(f"{name}: {p} has SHA-256 {have}, pinned {s['sha256']}")
        _CACHE[key] = have
    return _CACHE[key]


def name_for_sha(sha: str) -> str:
    for n, s in DBS.items():
        if s["sha256"] == sha:
            return n
    raise DbRefused(f"no pinned retail database has SHA-256 {sha}")


def identity(name: str) -> dict[str, str]:
    """What a package records about the database its cases were built on."""
    return {"name": name, "sha256": verify(name), "file": path(name).name,
            "domain": spec(name)["domain"]}


def domain(name: str) -> str:
    return spec(name)["domain"]


def name_for_domain(dom: str) -> str:
    for n, s in DBS.items():
        if s["domain"] == dom:
            return n
    raise DbRefused(f"no pinned retail database for tau2 domain {dom!r}")


def is_retail_domain(dom: str) -> bool:
    return any(s["domain"] == dom for s in DBS.values())


# --------------------------------------------------------------------------- active db


def active() -> str:
    return _ACTIVE["name"]


@contextlib.contextmanager
def use(name: str) -> Iterator[str]:
    """Make ``name`` the database the XP generators read (restored on exit)."""
    spec(name)
    verify(name)
    previous = _ACTIVE["name"]
    _ACTIVE["name"] = name
    try:
        yield name
    finally:
        _ACTIVE["name"] = previous


def select(name: str) -> str:
    """Make ``name`` the active database for the rest of the process (generator CLIs)."""
    spec(name)
    verify(name)
    _ACTIVE["name"] = name
    return name


def load(name: str | None = None) -> dict[str, Any]:
    """The database as plain JSON (a fresh object each call)."""
    name = name or active()
    if name != ORIGINAL:
        verify(name)
    return json.loads(path(name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- tau2


def _retail_db(name: str) -> Any:
    from tau2.domains.retail.data_model import RetailDB

    key = ("text", name)
    if key not in _CACHE:
        verify(name)
        _CACHE[key] = path(name).read_text(encoding="utf-8")
    return RetailDB.model_validate_json(_CACHE[key])  # a fresh copy (faster than a deep copy)


def env(name: str | None = None, **kwargs: Any) -> Any:
    """A fresh tau2 retail environment on ``name`` (default: the active database)."""
    from tau2.registry import registry

    name = name or active()
    if name == ORIGINAL:
        return registry.get_env_constructor("retail")(**kwargs)
    register_domains()
    return registry.get_env_constructor(domain(name))(**kwargs)


def _constructor(name: str) -> Any:
    def get_environment(db: Any = None, solo_mode: bool = False) -> Any:
        from tau2.domains.retail.environment import get_environment as retail_env

        return retail_env(db=db if db is not None else _retail_db(name), solo_mode=solo_mode)

    get_environment.__name__ = f"get_environment_{name.replace('-', '_')}"
    return get_environment


def register_domains() -> list[str]:
    """Register every extension database as its own tau2 domain (idempotent). The
    environment is tau2's retail environment (same policy text, same tools, same
    ``domain_name``) on the pinned extension database. Tasks are not registered: a
    generated scenario brings its own task."""
    from tau2.registry import registry

    out = []
    for n, s in DBS.items():
        if n == ORIGINAL:
            continue
        if s["domain"] not in registry.get_domains():
            registry.register_domain(_constructor(n), s["domain"])
        out.append(s["domain"])
    return out


def data_hashes(dom: str) -> dict[str, str] | None:
    """For an extension domain: the original retail data hashes with ``db.json``
    replaced by the extension file's hash (None for any other domain)."""
    if dom == "retail" or not is_retail_domain(dom):
        return None
    from slh import runner

    base = runner.data_hashes("retail")
    name = name_for_domain(dom)
    out = {k: v for k, v in base.items() if not k.endswith("/db.json")}
    out[f"domains/{dom}/{path(name).name}"] = verify(name)
    return out
