"""Track C development: a small consent/target checker (docs/xp/partner-c-fgate-plan.md, s. 2).

Our side (never in the partner kit): ``examples`` (decision examples, deterministic labeller, second reader,
counterfactual minimal pairs), ``build`` (hand-off 2 package, adjudication sheet),
``adjudicate`` (two independent LLM reviewers on the blinded sheet; disagreements go
to the owner), ``gate`` (counterfactual gate with independent denominators). Partner side: ``train``
(LoRA checker, encoder baseline, optional prompted 9B checker; dry/smoke/full/export).
Hand-off 1 (the 9B proposals) is shared with track F' (``slh.xp.fgate``).
"""
