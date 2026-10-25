"""The M-H1 kit's exit-code exceptions, in ONE module: ``mh1`` (also when it runs as
``python -m slh.system.hosts.hermes.mh1``, i.e. as ``__main__``) and ``probe`` import the
same classes, so a halt raised in the probe is the halt the CLI catches."""

from __future__ import annotations


class RefusedError(RuntimeError):
    """Exit 4: something differs from what is registered."""


class HaltError(RuntimeError):
    """Exit 3: resumable (server down, interruption)."""


class IncidentError(RuntimeError):
    """Exit 5: a containment incident."""
