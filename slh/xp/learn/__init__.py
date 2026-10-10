"""S+ / T+: a lesson library learned by small edit operations gated on a dev split.

- ``library``: entries {id, when, then, source_case_ids, added_step}, deterministic rendering.
- ``ops``: add/replace/remove schema, strict parser, per-op gates.
- ``loop``: failure batches, proposals, critic (teacher/self/none), dev gate,
  rejected-edit memory, stopping rule, ceilings, compact JSONL trace.
"""
