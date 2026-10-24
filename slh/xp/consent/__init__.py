"""Track C, conversation stage (step 4; docs/xp/step4-plan.md).

* ``scenarios``: consent-sensitive scenario roots (``genpilot/c-roots.yaml``) on the
  retail database ext-v2: revoked approval, changed target, partial approval and
  valid-prior-approval controls; records, validation, structure and dialogue audits,
  the draw under the ban list, audit/freeze files and the capacity report.
* ``metrics``: per-conversation consent metrics (key + consent v1), false blocks among
  gold-act writes, success, turns, checker latency; per-arm summaries.
* ``analysis``: the pre-specified C-S1 versus R analysis and its power simulation.

The live guard chain (identity guard v2.1, then the consent checker) is
``slh.guards.consent``. Nothing here calls a model or the network.
"""
