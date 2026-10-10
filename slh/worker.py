"""Entry point of a parallel-run worker process.

Kept free of heavy imports: a spawned worker imports this module first, so
console logging is quieted before tau2 (imported by ``slh.runner``) logs
anything. All tau2 and provider logs of a run go to the worker's private/
directory once the run starts.

Workers die with their coordinator (``start_parent_watchdog``): the
coordinator holds the run's serving worker slots (``slh.execution.
worker_slots``), and those are released when it exits, so an orphaned worker
must never keep sending requests against slots another run may now hold. The
watchdog polls the parent pid and hard-exits the worker (including any
in-flight request) within ``PARENT_POLL_S`` of the coordinator's death.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

PARENT_POLL_S = 0.25
ORPHAN_EXIT_CODE = 70


def _watch(parent: int, interval: float) -> None:
    while True:
        time.sleep(interval)
        if os.getppid() != parent:
            os._exit(ORPHAN_EXIT_CODE)  # no cleanup: the coordinator is gone


def start_parent_watchdog(interval: float = PARENT_POLL_S) -> threading.Thread:
    """Exit this process as soon as its parent (the coordinator) is gone."""
    parent = os.getppid()
    if parent <= 1:
        os._exit(ORPHAN_EXIT_CODE)
    thread = threading.Thread(target=_watch, args=(parent, interval), daemon=True,
                              name="parent-watchdog")
    thread.start()
    return thread


def worker_entry(*args: Any) -> None:
    start_parent_watchdog()
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING",
               filter=lambda record: not record["name"].startswith(("tau2", "litellm")))
    from slh.runner import _worker_main

    _worker_main(*args)
