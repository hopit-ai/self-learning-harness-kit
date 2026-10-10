"""Error classes for the partner runner (docs/xp/partner-package.md, section 5).

``halt``: the run stops at once and writes no row for the case (spend cap,
credit, authentication, configuration, connection, serving-window errors, and
anything unrecognised, so nothing is consumed by a systematic failure); rerun
after the fix and the case runs again. ``case``: a per-case model outcome that
is final (a timeout after the client's retries, the case wall-clock limit,
malformed model output, a context-length refusal).

Credit failures also latch POC 9's process-wide credit gate
(``slh.poc9.pipeline.latch_credit``), so every other request stops before it is
sent.
"""

from __future__ import annotations

import json
import re
from typing import Any

from slh.poc9.pipeline import CreditHalt


class RunHalted(CreditHalt):
    """A halt-class error: the run stops; resumable after the cause is fixed.
    A ``CreditHalt`` so POC 9 / XP learning loops stop at their credit handler."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


_CASE_TYPES = ("CaseTimeLimit", "Timeout", "TimeoutError", "ReadTimeout", "JSONDecodeError",
               "ContextLimitExceeded", "OpsParseError")
_CASE_WORDS = re.compile(r"timed out|timeout|context length|context window|max_model_len|"
                         r"malformed|could not parse|unparseable|invalid tool call", re.IGNORECASE)
_HALT_WORDS = re.compile(r"spend|budget|\bcap\b|credit|api[ _-]?key|unauthori[sz]ed|forbidden|"
                         r"bearer|access token|auth token|not set|invalid url|ssl|"
                         r"certificate|refused|"
                         r"authentication|permission|connection (refused|reset|error|aborted)|"
                         r"failed to establish|name or service not known|nodename nor servname|"
                         r"no route to host|serving window|not configured|config", re.IGNORECASE)
_HALT_TYPES = ("BudgetExceeded", "Overspend", "UnmeteredCall", "ServingWindowClosed",
               "CreditHalt", "RunHalted", "APIConnectionError", "ConnectionError",
               "ConnectionRefusedError", "URLError", "ImportError", "ModuleNotFoundError",
               "FileNotFoundError", "PermissionError", "KeyError", "ConnectTimeout",
               "ConnectTimeoutError", "NewConnectionError", "MaxRetryError")
_CONTEXT = re.compile(r"context length|context window|max_model_len", re.IGNORECASE)
_STATUS = re.compile(r"\b(?:HTTP|status)[ :]*([45]\d\d)\b", re.IGNORECASE)


def classify(error: BaseException | str) -> tuple[str, str]:
    """(``halt`` | ``case``, label)."""
    if isinstance(error, BaseException):
        name, message = type(error).__name__, f"{type(error).__name__}: {error}"
        names = {c.__name__ for c in type(error).__mro__}
    else:
        message = error
        name = message.split(":", 1)[0].strip()
        names = {name}
    from slh.poc9.pipeline import credit_failure

    if credit_failure(message):
        return "halt", "credit"
    if names & set(_HALT_TYPES):
        return "halt", name
    statuses = {int(s) for s in _STATUS.findall(message)}
    if statuses & {401, 402, 403, 404, 407}:
        return "halt", f"http_{min(statuses & {401, 402, 403, 404, 407})}"
    # Halting signatures (configuration, auth, connection, credit) always win over
    # any case-level class or wording, whatever order the message puts them in.
    if _HALT_WORDS.search(message):
        return "halt", "systematic"
    if _CONTEXT.search(message):
        return "case", "context_length"
    if names & set(_CASE_TYPES) or isinstance(error, json.JSONDecodeError) or \
            _CASE_WORDS.search(message):
        return "case", name
    return "halt", f"unrecognised:{name}"


def halt_record(error: BaseException | str, **where: Any) -> dict[str, Any]:
    kind, label = classify(error)
    text = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else error
    return {"event": "halt", "class": kind, "label": label, "error": text[:500], **where}
