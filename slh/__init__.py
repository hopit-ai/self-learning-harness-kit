"""Self-learning harness for tau2-bench tool-use agents."""

import os

# The model-pricing library fetches a remote price table at import time unless
# told to use its bundled copy. Harness accounting uses configs/prices.yaml, so
# the remote table is never needed; this keeps imports offline.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

__version__ = "0.0.1"
