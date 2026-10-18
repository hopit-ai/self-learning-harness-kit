"""Deterministic identity guard (retail): account access only for the authenticated
customer, live and offline.

Owner-approved 2026-10-17; v2 (2026-10-18, guard review): the conversation is
bound to ONE customer and account reads are guarded too. tau2 retail policy: the
agent must authenticate the user by ``find_user_id_by_email`` or
``find_user_id_by_name_zip`` before acting on an account, and may help only that
user. A user id typed by the user, or a ``get_user_details`` result, is NOT
authentication.

Rule (no learning, no task or grading data). The conversation is bound to the
FIRST user id an authentication call returned without error (the agent saw the
result). A later authentication that returns a different id is not credited: it
never re-binds the conversation and never authenticates that other user. Then

1. an account-scoped call (``ACCOUNT_SCOPED_TOOLS``: the state-changing calls of
   ``trace.WRITE_TOOLS``, the retail ``ToolType.WRITE`` set, plus the account reads
   ``get_user_details`` and ``get_order_details``) is allowed only once the
   conversation is bound, and
2. only when it targets the bound user, on visible evidence: the ``user_id``
   argument of ``get_user_details`` / ``modify_user_address``; for order tools the
   owner of ``order_id`` as shown by an earlier tool result (a user-details result
   listing the order, or an order result's ``user_id``). Exception: once the
   conversation is bound, ``get_order_details`` of an order whose owner was never
   shown is allowed (tau2 gold looks orders up straight after authentication); its
   own result then establishes the owner. A WRITE on an order whose owner was never
   shown is blocked (``OWNER_UNKNOWN``: look the order up with ``get_order_details``
   first). Once any result shows an order belongs to another user, every further
   read or write of that order is blocked (``NOT_OWNER``); the one lookup that
   revealed it cannot be prevented, because nothing before it shows the owner.

Authentication calls, product and catalogue reads (``get_product_details``,
``get_item_details``, ``list_all_product_types``), ``calculate`` and
``transfer_to_human_agents`` always pass. A blocked call is answered with a
tool-result-style message (``BLOCK_UNAUTHENTICATED`` / ``BLOCK_NOT_OWNER`` /
``BLOCK_OWNER_UNKNOWN``) that names the procedure only: no user id, owner, order
content or task answer is revealed, and the not-owner message denies
cross-account work in this conversation (it never suggests authenticating
another account holder). The agent may retry.

Live use composes with the harness's existing veto contract
(``slh.agent.Guard``: ``check(call, GuardContext) -> GuardVerdict``; the agent
answers a veto with ``Blocked: <guard>: <reason>`` in its private history and
re-asks the model, at most ``slh.agent.GUARD_RETRY_LIMIT`` extra times per
turn; a vetoed call blocks its whole message, so an authentication batched with a
write must be re-issued alone). ``install()`` rebinds ``slh.runner.build_guards``
at runtime so that, while ``active(True)``, the episode loop's guard list gains
``IdentityGuard``. Nothing in the pinned execution surface
(``slh.execution.RUNTIME_MODULES``) is edited and the harness state
(``state.guards``, version id) is unchanged; a run that turns the guard on records
``descriptor()`` in its own identity.

Offline use: ``counterfactual(messages)`` replays a recorded conversation and
reports, for every write (and, with ``include_reads``, every account read) the
agent actually made, whether this guard would have blocked it (trajectory as
recorded; the conversation after a block would differ).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slh.guards.trace import AUTH_TOOLS, WRITE_TOOLS, order_key

NAME = "identity_guard"
VERSION = 2
ALLOW, UNAUTHENTICATED, NOT_OWNER, OWNER_UNKNOWN = (
    "allow", "unauthenticated", "not_owner", "owner_unknown")
# User- or order-scoped reads (retail ``ToolType.READ`` tools that reveal one account).
ACCOUNT_READ_TOOLS = frozenset({"get_user_details", "get_order_details"})
ACCOUNT_SCOPED_TOOLS = ACCOUNT_READ_TOOLS | WRITE_TOOLS
USER_ARG_TOOLS = frozenset({"get_user_details", "modify_user_address"})

BLOCK_UNAUTHENTICATED = (
    "the customer's identity is not verified in this conversation. Before looking up or "
    "changing any order or account, authenticate the customer with find_user_id_by_email "
    "(their email address) or find_user_id_by_name_zip (first name, last name and zip "
    "code). A user id given by the customer does not authenticate them. Ask the customer "
    "for these details, look the user up, then retry the action.")
BLOCK_NOT_OWNER = (
    "the order or account requested does not belong to the customer authenticated in this "
    "conversation. This conversation is bound to that customer: no other account, and no "
    "order of another account, may be looked up or changed here. Continue only with the "
    "authenticated customer's own account and orders.")
BLOCK_OWNER_UNKNOWN = (
    "the order has not been shown to belong to the customer authenticated in this "
    "conversation. Before changing it, look the order up with get_order_details (or find "
    "it in the authenticated customer's get_user_details), confirm it belongs to them, "
    "then retry the action.")
MESSAGES = {UNAUTHENTICATED: BLOCK_UNAUTHENTICATED, NOT_OWNER: BLOCK_NOT_OWNER,
            OWNER_UNKNOWN: BLOCK_OWNER_UNKNOWN}


def _json(text: Any) -> Any:
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _auth_id(result: str) -> str | None:
    text = result.strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = text  # the usual form: the bare user id
    if not isinstance(parsed, str):
        return None  # null, numbers, objects, lists: not a user id
    text = parsed.strip()
    text = text.strip('"').strip()
    if not text or text.lower().startswith("error") or any(c.isspace() for c in text):
        return None
    return text


@dataclass
class IdentityState:
    """What the agent has seen so far in one conversation."""

    bound: str | None = None  # the first successfully authenticated user id
    other_auth_ids: list[str] = field(default_factory=list)  # later, different: not credited
    order_owner: dict[str, str] = field(default_factory=dict)

    @property
    def auth_ids(self) -> list[str]:
        """The authenticated user(s) the guard credits: at most the bound one."""
        return [self.bound] if self.bound is not None else []

    def observe(self, name: str, args: dict[str, Any], result: str | None,
                error: bool = False) -> None:
        """One executed call and its result (errors teach nothing)."""
        if result is None or error or str(result).startswith("Error"):
            return
        if name in AUTH_TOOLS:
            uid = _auth_id(str(result))
            if uid is None:
                return
            if self.bound is None:
                self.bound = uid
            elif uid != self.bound and uid not in self.other_auth_ids:
                self.other_auth_ids.append(uid)
            return
        data = _json(result)
        if not isinstance(data, dict):
            return
        if "order_id" in data and data.get("user_id") is not None:
            self.order_owner[order_key(data["order_id"])] = str(data["user_id"])
        if name in USER_ARG_TOOLS and data.get("user_id"):
            for oid in data.get("orders") or []:
                self.order_owner.setdefault(order_key(oid), str(data["user_id"]))

    def decide(self, name: str, args: dict[str, Any]) -> str:
        """``ALLOW``, ``UNAUTHENTICATED``, ``NOT_OWNER`` or ``OWNER_UNKNOWN`` for a
        proposed call. Account-scoped calls need the bound user and positive
        ownership evidence."""
        if name not in ACCOUNT_SCOPED_TOOLS:
            return ALLOW
        if self.bound is None:
            return UNAUTHENTICATED
        target = self.target_user(name, args)
        if target is None:
            # After authentication an order may be looked up directly (tau2 gold does
            # this): the read's own result then establishes the owner, and every write
            # still needs that owner to be the bound user (review 2026-10-18).
            if name == "get_order_details":
                return ALLOW
            return OWNER_UNKNOWN
        return ALLOW if target == self.bound else NOT_OWNER

    def target_user(self, name: str, args: dict[str, Any]) -> str | None:
        if name in USER_ARG_TOOLS:
            return None if args.get("user_id") is None else str(args["user_id"])
        if "order_id" in args:
            return self.order_owner.get(order_key(args["order_id"]))
        return None


# --------------------------------------------------------------------------- live


def _args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    parsed = _json(value)
    return parsed if isinstance(parsed, dict) else {}


def state_from_history(history: Iterable[Any]) -> IdentityState:
    """Rebuild the state from the agent's message history (tau2 messages): each
    tool result is matched to the call that requested it by id."""
    state = IdentityState()
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    for message in history:
        for call in getattr(message, "tool_calls", None) or []:
            calls[str(call.id)] = (str(call.name), _args(call.arguments))
        inner = getattr(message, "tool_messages", None)
        for tool_message in inner if inner is not None else [message]:
            if getattr(tool_message, "role", None) != "tool":
                continue
            found = calls.get(str(getattr(tool_message, "id", "")))
            if found is None:
                continue
            state.observe(found[0], found[1], getattr(tool_message, "content", None),
                          bool(getattr(tool_message, "error", False)))
    return state


class IdentityGuard:
    """``slh.agent.Guard``: vetoes a write until the target user is authenticated.
    Stateless: every check rebuilds the conversation state from the history, so
    one instance serves every episode."""

    name = NAME

    def check(self, call: Any, context: Any) -> Any:
        from slh.agent import GuardVerdict

        verdict = state_from_history(context.history).decide(str(call.name),
                                                             _args(call.arguments))
        if verdict == ALLOW:
            return GuardVerdict(allow=True)
        return GuardVerdict(allow=False, reason=MESSAGES[verdict])


_INSTALLED: dict[str, Any] = {}
_ACTIVE = {"on": False}


def _wrap_build_guards(original: Any) -> Any:
    def build_guards(specs: list[dict[str, Any]]) -> list[Any]:
        guards = list(original(specs))
        if _ACTIVE["on"]:
            guards.append(IdentityGuard())
        return guards

    build_guards.__wrapped__ = original  # type: ignore[attr-defined]
    return build_guards


def install() -> None:
    """Rebind ``slh.runner.build_guards`` once per process (idempotent). The
    wrapper adds the guard only while ``active``; installed but inactive it
    returns exactly what the original returns."""
    if _INSTALLED:
        return
    from slh import runner

    _INSTALLED["build_guards"] = runner.build_guards
    runner.build_guards = _wrap_build_guards(runner.build_guards)


def uninstall() -> None:
    """Tests: restore the original binding and switch off."""
    if not _INSTALLED:
        return
    from slh import runner

    runner.build_guards = _INSTALLED.pop("build_guards")
    _ACTIVE["on"] = False


def set_active(on: bool) -> None:
    if on:
        install()
    _ACTIVE["on"] = bool(on)


@contextlib.contextmanager
def active(on: bool) -> Iterator[None]:
    previous = _ACTIVE["on"]
    set_active(on)
    try:
        yield
    finally:
        _ACTIVE["on"] = previous


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def descriptor(arms: Sequence[str]) -> dict[str, Any]:
    """What a guarded run records in its identity (absent when the guard is off)."""
    return {"name": NAME, "version": VERSION, "arms": sorted(set(arms)),
            "source_sha256": source_sha256(), "write_tools": sorted(WRITE_TOOLS),
            "account_read_tools": sorted(ACCOUNT_READ_TOOLS),
            "auth_tools": sorted(AUTH_TOOLS), "binding": "first-authenticated-user",
            "selected_by": "arm name"}


def parse_arms(value: str | None, allowed: Iterable[str]) -> tuple[str, ...] | None:
    """CLI value ``b1,learned`` (or ``all``) -> arm names; None/empty -> off."""
    if not value:
        return None
    allowed = sorted(set(allowed))
    arms = allowed if value.strip() == "all" else [a.strip() for a in value.split(",")
                                                    if a.strip()]
    bad = sorted(set(arms) - set(allowed))
    if bad:
        raise ValueError(f"identity guard: unknown arm(s) {bad}; choose from {allowed}")
    return tuple(sorted(set(arms)))


# --------------------------------------------------------------------------- offline


def counterfactual(messages: list[dict[str, Any]], include_reads: bool = False
                   ) -> list[dict[str, Any]]:
    """Every write call (with ``include_reads`` also every account read,
    ``ACCOUNT_READ_TOOLS``; rows carry ``kind`` "write" / "read") of a recorded
    conversation with the guard's verdict at
    that point. Strictly chronological, as the live guard sees it: the calls of
    one assistant message are judged against the results that arrived BEFORE
    that message (a batch holding an authentication and a write does not credit
    the authentication to the write); a tool result updates the state only when
    its own message is reached. ``executed`` says whether the recorded write
    actually ran (non-error result): an executed blocked write is a mutation the
    guard would have prevented, a non-executed one only an attempt."""
    state = IdentityState()
    pending: dict[str, tuple[str, dict[str, Any]]] = {}
    out: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for position, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                name, args = str(call.get("name")), _args(call.get("arguments"))
                pending[str(call.get("id"))] = (name, args)
                if name in WRITE_TOOLS or (include_reads and name in ACCOUNT_READ_TOOLS):
                    row = {"message": position, "tool": name,
                           "kind": "write" if name in WRITE_TOOLS else "read",
                           "verdict": state.decide(name, args), "executed": False,
                           "args": args}
                    out.append(row)
                    by_id[str(call.get("id"))] = row
        elif role == "tool":
            cid = str(message.get("tool_call_id"))
            found = pending.pop(cid, None)
            if found is None:
                continue
            content = message.get("content")
            error = content is None or str(content).startswith("Error")
            if cid in by_id:
                by_id[cid]["executed"] = not error
            state.observe(found[0], found[1], None if content is None else str(content),
                          error)
    return out
