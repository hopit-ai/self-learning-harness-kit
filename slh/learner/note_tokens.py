"""Token counting for learned note content under the pinned agent tokenizer.

Counting rule (registered with the content limit):

- tokenizer: the agent model's ``tokenizer.json`` at the pinned revision
  (``slh.serve_spec.MODEL_REPO`` @ ``MODEL_REVISION``), whose exact bytes must
  hash to ``PINNED_TOKENIZER_SHA256``; any other file is refused;
- text: the note's ``content`` string exactly as stored (it is the only field
  rendered into the agent prompt), with no normalisation, stripping or bullet
  prefix;
- count: ``len(Tokenizer.encode(content, add_special_tokens=False).ids)``.

``CONTENT_TOKEN_LIMIT`` (200) bounds one note's content; the active-addition
ceiling (``ACTIVE_ADDITION_TOKEN_LIMIT``, 600, protocol amendment 1) bounds the
sum over every active learned note's content. The tokenizer is read offline
from the local Hugging Face cache (``HF_HOME`` or ``~/.cache/huggingface``) or
from ``SLH_AGENT_TOKENIZER`` (a path to the same file); nothing is downloaded.
"""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path

CONTENT_TOKEN_LIMIT = 200
ACTIVE_ADDITION_TOKEN_LIMIT = 600
PINNED_TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"


class TokenizerUnavailable(ValueError):
    """The pinned agent tokenizer is not available offline (or does not match)."""


def tokenizer_path() -> Path:
    from slh.serve_spec import MODEL_REPO, MODEL_REVISION

    override = os.environ.get("SLH_AGENT_TOKENIZER")
    if override:
        return Path(override)
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    return (home / "hub" / f"models--{MODEL_REPO.replace('/', '--')}" / "snapshots"
            / MODEL_REVISION / "tokenizer.json")


@functools.lru_cache(maxsize=1)
def _tokenizer(path: str):
    from tokenizers import Tokenizer

    try:
        body = Path(path).read_bytes()
    except OSError:
        raise TokenizerUnavailable(f"pinned agent tokenizer not found at {path}") from None
    if hashlib.sha256(body).hexdigest() != PINNED_TOKENIZER_SHA256:
        raise TokenizerUnavailable("tokenizer file is not the pinned agent tokenizer")
    return Tokenizer.from_str(body.decode("utf-8"))


def count_tokens(text: str) -> int:
    """Tokens of ``text`` under the pinned agent tokenizer (the registered rule)."""
    return len(_tokenizer(str(tokenizer_path())).encode(text, add_special_tokens=False).ids)


ACTIVE_ADDITION_TEXT = "HarnessState.system_prompt_section()"
COUNTING_RULE = "len(Tokenizer.encode(text, add_special_tokens=False).ids)"


def rendered_active_tokens(state) -> tuple[int, str]:
    """Amendment 3: tokens of the EXACT rendered learned section (headings,
    bullet prefixes, separators and whitespace included; one combined string,
    not a sum over notes) and that string's SHA-256. Diagnostic (non-learned)
    notes are excluded; audit metadata is never rendered."""
    from slh.state import HarnessState

    learned = HarnessState(notes=[n for n in state.notes if not n.get("diagnostic")],
                           skills=state.skills, guards=state.guards, base=state.base)
    text = learned.system_prompt_section()
    return count_tokens(text), hashlib.sha256(text.encode("utf-8")).hexdigest()


def counting_receipt() -> dict:
    """What the counts were made with: the pinned tokenizer asset (verified by
    hash), its implementation and version, the serving spec it belongs to and
    the counting rule. Raises TokenizerUnavailable if the pin cannot be verified."""
    import tokenizers

    from slh import serve_spec

    path = tokenizer_path()
    _tokenizer(str(path))  # verifies the pinned bytes
    return {"tokenizer_repo": serve_spec.MODEL_REPO, "revision": serve_spec.MODEL_REVISION,
            "tokenizer_json_sha256": PINNED_TOKENIZER_SHA256,
            "implementation": "tokenizers", "version": tokenizers.__version__,
            "serving_spec_sha256": serve_spec.spec_sha256(), "rule": COUNTING_RULE,
            "content_limit": CONTENT_TOKEN_LIMIT,
            "active_addition_limit": ACTIVE_ADDITION_TOKEN_LIMIT,
            "active_addition_text": ACTIVE_ADDITION_TEXT}
