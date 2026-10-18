"""Larger experiment (XP): decision-level case generators beyond POC 9 (frozen).

Importing any ``slh.xp`` module forces ``LITELLM_LOCAL_MODEL_COST_MAP=True`` (``slh``
already sets it as a default, before tau2 or LiteLLM can be imported): LiteLLM then reads
its bundled price map and never fetches one over the network, even when a shell exported
another value. The partner runbook also exports it in the run shell, which the
disconnected preflight (``python -m slh.xp.partner.offline preflight-h1``) checks.
"""

import os

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
