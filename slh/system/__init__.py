"""Working system V1: a safe self-improvement loop around a frozen agent (skeleton).

Design: docs/system/v1-design.md. Modules:

- ``interfaces``: the agent adapter, proposer and record types every component shares.
- ``stores``: trace store (role-separated), adoption store (versioned ``slh.state.HarnessState``)
  and the hash-chained audit log.
- ``gate``: the screen (the registered F' analysis, ``slh.xp.fgate.analysis``, imported, never
  copied), the confirmation (paired non-inferiority on success and harm, vs the incumbent and
  the root, with lifetime alpha spending) and the monitor (rollback).
- ``loop``: one cycle (evidence -> propose -> screen -> confirm -> adopt/reject -> monitor) and
  the continuous driver with budgets and invariant checks.
- ``sim``: an offline synthetic adapter and a planted proposer (no model, no network).
- ``dashboard``: read-only status and the static HTML dashboard rendered from the audit log.
- ``cli`` (``python -m slh.system``): run / status / render / ack / certify / sweep.
- ``sweep``: the offline seed sweep over planted lessons.
"""
