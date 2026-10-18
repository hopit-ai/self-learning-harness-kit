"""Offline policy predicates and provenance-backed structural guards (retail).

Pure functions over public traces: the dialogue, the agent's tool calls and the
tool results. Nothing here is imported by the runtime (``slh.execution``
``RUNTIME_MODULES``) and nothing here reads rewards, grading fields or task
scenarios. ``trace`` parses an episode, ``structural`` holds the guards (shadow
mode only), ``detectors`` holds the frozen policy-to-predicate specification and
``audit`` applies both to a run directory.
"""
