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
import threading
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


# ---- (v2.6) the shared, read-only database and each conversation's copy-on-write view

_CHANGE_ATTEMPTS = [0]  # attempts to change the shared database, this process (never > 0)


def shared_change_attempts() -> int:
    """How many times anything tried to change the shared database in this process; every
    episode compares it at its end with its start (a change is an incident)."""
    return _CHANGE_ATTEMPTS[0]


def _deny(*_: Any, **__: Any) -> Any:
    _CHANGE_ATTEMPTS[0] += 1
    raise TypeError("the shared retail database is read-only: change a conversation's own "
                    "copy (db.<table>[key]) instead")


class _ReadOnlyList(list):
    """A list of the shared database: every mutator raises (and is counted)."""

    __slots__ = ()
    append = extend = insert = remove = pop = clear = sort = reverse = _deny
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _deny


class _ReadOnlyDict(dict):
    """A dict of the shared database: every mutator raises (and is counted)."""

    __slots__ = ()
    __setitem__ = __delitem__ = setdefault = pop = popitem = clear = update = _deny
    __ior__ = _deny


_FROZEN: dict[type, type] = {}
_FROZEN_LOCK = threading.Lock()


def _frozen_class(cls: type) -> type:
    """A subclass of a pydantic model whose attribute writes raise (and are counted)."""
    with _FROZEN_LOCK:
        if cls not in _FROZEN:
            _FROZEN[cls] = type(f"Shared{cls.__name__}", (cls,),
                                {"__setattr__": _deny, "__delattr__": _deny,
                                 "__module__": __name__})
        return _FROZEN[cls]


def _freeze(v: Any) -> Any:
    """Deeply read-only in place: models get a raising class, lists and dicts read-only
    subclasses (still ``list`` / ``dict``: serialisation and hashing are unchanged)."""
    from pydantic import BaseModel

    if isinstance(v, BaseModel):
        d = v.__dict__
        for k in list(d):
            d[k] = _freeze(d[k])
        v.__class__ = _frozen_class(type(v))
        return v
    if isinstance(v, list) and not isinstance(v, _ReadOnlyList):
        return _ReadOnlyList(_freeze(x) for x in v)
    if isinstance(v, dict) and not isinstance(v, _ReadOnlyDict):
        return _ReadOnlyDict((k, _freeze(x)) for k, x in v.items())
    return v


class _CowTable(dict):
    """One conversation's view of a table of the SHARED database (copy on write). The
    stored values start as the shared records (deeply read-only: any write through them
    raises and is counted); a record becomes the conversation's own, fresh and writable
    (validated from the record's JSON), the first time it is looked up by key
    (``table[key]`` / ``get`` / ``setdefault``), the path every retail tool that changes a
    record uses. ``items`` / ``values`` return the conversation's own record where it has one,
    else the read-only shared one; ``pop`` / ``popitem`` / ``__delitem__`` / ``update`` /
    ``__setitem__`` / ``clear`` change only this view; a shallow ``copy`` raises; a deep copy
    is the conversation's state as fresh writable records. Iteration goes through these
    methods (``__iter__`` is defined, so CPython's dict fast paths do not bypass them). It is
    a real ``dict``: tau2's hashing sees exactly this conversation's state."""

    __slots__ = ("_model", "_own", "_records")

    def __init__(self, base: dict[str, Any], records: dict[str, str], model: type) -> None:
        super().__init__(base)
        self._own: set[str] = set()
        self._records, self._model = records, model

    def _fresh(self, key: str) -> Any:
        return self._model.model_validate_json(self._records[key])

    def __getitem__(self, key: str) -> Any:
        if key not in self._own:
            dict.__getitem__(self, key)  # KeyError for a missing key, as a dict
            v = self._fresh(key)
            dict.__setitem__(self, key, v)
            self._own.add(key)
            return v
        return dict.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self:
            return default
        return self.__getitem__(key)

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self.__getitem__(key)
        self[key] = default
        return default

    def __setitem__(self, key: str, value: Any) -> None:
        dict.__setitem__(self, key, value)
        self._own.add(key)

    def __delitem__(self, key: str) -> None:
        dict.__delitem__(self, key)
        self._own.discard(key)

    def pop(self, key: str, *default: Any) -> Any:
        if key not in self:
            if default:
                return default[0]
            raise KeyError(key)
        v = self.__getitem__(key)
        self.__delitem__(key)
        return v

    def popitem(self) -> tuple[str, Any]:
        key = next(reversed(dict.keys(self)))
        return key, self.pop(key)

    def update(self, *a: Any, **kw: Any) -> None:
        for k, v in dict(*a, **kw).items():
            self[k] = v

    def clear(self) -> None:
        dict.clear(self)
        self._own.clear()

    def __iter__(self) -> Any:
        return iter(list(dict.keys(self)))

    def items(self) -> Any:  # type: ignore[override]
        return [(k, dict.__getitem__(self, k)) for k in dict.keys(self)]

    def values(self) -> Any:  # type: ignore[override]
        return [dict.__getitem__(self, k) for k in dict.keys(self)]

    def copy(self) -> Any:
        raise TypeError("a conversation's database table cannot be shallow-copied "
                        "(it would share the read-only records); deep-copy it")

    __copy__ = copy

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        import copy as _copy

        return {k: (_copy.deepcopy(dict.__getitem__(self, k), memo) if k in self._own
                    else self._fresh(k)) for k in dict.keys(self)}

    def __reduce__(self) -> Any:
        return (dict, (self.__deepcopy__({}),))


_BASE_LOCK = threading.Lock()


def _base(name: str) -> Any:
    """The database validated ONCE per process, then made deeply read-only (the shared
    copy), with every record's JSON text for the conversations' own copies."""
    from tau2.domains.retail.data_model import RetailDB

    key = ("base", name)
    with _BASE_LOCK:
        if key not in _CACHE:
            verify(name)
            text = path(name).read_text(encoding="utf-8")
            raw = json.loads(text)
            records = {t: {k: json.dumps(v, ensure_ascii=False) for k, v in rows.items()}
                       for t, rows in raw.items()}
            del raw
            _CACHE[key] = (_freeze(RetailDB.model_validate_json(text)), records)
        return _CACHE[key]


def _retail_db(name: str) -> Any:
    """A fresh copy of the database for one conversation. (2026-10-29, M-H1 v2.6) v2.5 built a
    full validated copy per conversation (~1.4 M garbage-collector-tracked objects, one
    0.5 s GIL-holding call); with 64 conversations in one harness process, full garbage
    collections over ~90 M objects stopped the process for tens of seconds and our guard
    round trip passed Hermes' 30 s hook timeout (the partner's c27b learning halt). Now every
    conversation shares one deeply read-only copy per process and owns only the records it
    looks up (``_CowTable``): building a copy is three shallow dict copies, the tracked heap
    no longer grows with the number of conversations, the state each conversation sees and
    changes is the same as with its own full copy (tested), and nothing can change the shared
    copy (any attempt raises and is counted: an incident at the episode's end)."""
    from tau2.domains.retail.data_model import Order, Product, RetailDB, User

    base, rec = _base(name)
    return RetailDB.model_construct(
        products=_CowTable(base.products, rec["products"], Product),
        users=_CowTable(base.users, rec["users"], User),
        orders=_CowTable(base.orders, rec["orders"], Order))


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
