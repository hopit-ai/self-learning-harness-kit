"""POC 9 skill writers: T (teacher), S (the 9B agent), TS (teacher drafts, 9B rewrites).

One interface, ``Writer.propose_patch(current_skill, evidence_batch) -> PatchResult``,
with identical evidence (``render_evidence``), identical prompts, the same
per-call output cap and the same mechanical checks with ONE repair call by the
writer's own final model. The learner's gate (``slh.poc9.learn.gate``) adds the
binding critique on top.

- T: the teacher (``configs/poc9-teacher.yaml``: a hosted OpenAI-compatible
  endpoint with one pinned provider, metered by its own ``slh.budget.Meter``)
  writes the patch and does the repair.
- S: the 9B agent through the demo-1 capped client (``slh.demo1.learn._ask``).
- TS: the teacher drafts; the 9B rewrites in its own words, tagging each line
  with the draft claims it expresses; ``fidelity`` then checks mechanically
  that every normative claim of the draft is preserved (dropped, invented and
  altered claims, including changed conditions and changed order of
  connectives/negation/actions), and, if that passes, one binding teacher call
  checks semantic equivalence (any difference = altered). The 9B does the one
  repair; the repaired text is re-checked (mechanically, then semantically).

A patch is a full replacement SKILL.md (300-500 tokens; sections Trigger,
Steps, Pitfalls, Verification, Exceptions) plus provenance lines kept
separately (never injected). The writer may answer NO CHANGE.

``generic_note(writer, target_tokens)``: the same writer and rules with NO
evidence, in the same length band (the same-writer generic-note control).
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from slh.demo1.learn import (
    _ID_PATTERNS,
    _facts,
    clean,
    count_tokens,
    evidence_values,
    sha256_text,
    value_hits,
)
from slh.demo1.schema import REPO

TEACHER_CONFIG = REPO / "configs/poc9-teacher.yaml"
USER_CONFIG = REPO / "configs/user.yaml"
PLACEHOLDER_MARK = "TBD"
SKILL_TOKENS = (300, 500)
SECTIONS = ("Trigger", "Steps", "Pitfalls", "Verification", "Exceptions")
NO_CHANGE = "NO CHANGE"
EXTRA_GENERAL = frozenset({"full", "new", "same", "other", "standard", "default", "home",
                           "work", "office", "main", "primary"})
TASK = ("choosing the new shipping address when a user asks to ship a pending order to an "
        "address they describe by pointing to one of their other orders: write the change "
        "when the description matches exactly one address and the user has approved it, ask "
        "for confirmation when it is not yet approved, and ask the user when the description "
        "matches no address or more than one")

# A chat call: (system, user, purpose=...) -> (text or None, accounting, error or None).
Chat = Callable[..., tuple[str | None, dict[str, int], str | None]]

SYSTEM = ("You maintain one SKILL.md for a retail customer-service agent. The agent must "
          "follow this policy (excerpt):\n\n<policy>\n{policy}\n</policy>")
FORMAT_RULES = (
    "Rules for the skill:\n"
    "- Markdown with exactly these sections, in order: " +
    ", ".join(f"'## {s}'" for s in SECTIONS) + ".\n"
    "- Length: at most {words_high} words and at most {high} tokens (at least {low} "
    "tokens); aim for {words_low}-{words_high} words. Five sections only: no other "
    "headings, no preamble, no closing remarks.\n"
    "- Use only what the policy and the evidence support. Do not add new obligations.\n"
    "- Do not mention specific order ids, item ids, user ids, emails, names, addresses, "
    "cities, product names, prices or episode numbers inside the skill. Describe the "
    "general situation.\n"
    "- Cover when to act, when to ask for confirmation and when to ask which address the "
    "user means.")
PATCH_PROMPT = (
    "Task of the skill: {task}.\n\nCurrent SKILL.md:\n<skill>\n{skill}\n</skill>\n\n"
    "New evidence: {n} episodes the agent just handled with the current skill. For each "
    "you see the conversation, the agent's action, the deterministic outcome and the "
    "user's correction, if any.\n\n{evidence}\n\n"
    "Update the skill so the agent handles new cases of this task correctly. Reply with "
    "the full updated skill inside <skill>...</skill>, then inside "
    "<provenance>...</provenance> one line per change of the form "
    "'- E<episode number>: <what that episode showed>'. If the skill needs no change, "
    "reply exactly NO CHANGE.\n\n" + FORMAT_RULES)
GENERIC_PROMPT = (
    "Task of the skill: {task}.\n\nWrite a SKILL.md that would help the agent handle this "
    "task correctly. Reply with the skill inside <skill>...</skill>.\n\n" + FORMAT_RULES)
REPAIR_PROMPT = (
    "Your reply breaks these rules:\n{problems}\n\nRewrite it to fix only these problems "
    "and keep its content otherwise.{length}Reply in the same form as before "
    "(<skill>...</skill>{prov}).\n\nYour previous reply:\n{text}")
LENGTH_NOTE = (
    " LENGTH: the skill measures {tokens} tokens; the band is {low}-{high} tokens "
    "(at most {words_high} words; aim for {words_low}-{words_high} words). Section sizes "
    "now: {sizes}. Shorten mostly {longest}: merge overlapping lines, drop examples and "
    "restatements. ")
SHORTEN_PROMPT = (
    "The skill below measures {tokens} tokens; it must be at most {high} tokens (about "
    "{words_target} words). Shorten it without changing its meaning: keep every rule, "
    "condition, exception and their order, keep the five sections and any [C<n>] tags, "
    "only cut wording. Section sizes now: {sizes}. Reply with the skill inside "
    "<skill>...</skill>.\n\n<skill>\n{text}\n</skill>")
SHORTEN_SLACK = 0.15  # a second, mechanical-only repair is allowed up to 15% over the band

# Equal per-writer ceilings (registered; the pipeline imports these): every
# writer call (draft, write, rewrite, repair, shorten, equivalence, critique),
# generic note included. A call is made only if one more call fits and a full
# CALL_OUTPUT_RESERVE of output still fits; otherwise the proposal is rejected
# as budget_exhausted (previous skill kept). Nothing raises mid-run.
WRITER_CALL_CEILING = 96
CALL_OUTPUT_RESERVE = 4096
WRITER_OUTPUT_TOKEN_CEILING = WRITER_CALL_CEILING * CALL_OUTPUT_RESERVE
BUDGET_EXHAUSTED = "budget_exhausted"


class BudgetExhausted(Exception):
    """Internal: the next writer call does not fit the ceilings."""


@dataclasses.dataclass
class WriterBudget:
    calls: int = 0
    output_tokens: int = 0
    call_ceiling: int = WRITER_CALL_CEILING
    output_token_ceiling: int = WRITER_OUTPUT_TOKEN_CEILING

    def can_call(self) -> bool:
        return self.calls + 1 <= self.call_ceiling and \
            self.output_tokens + CALL_OUTPUT_RESERVE <= self.output_token_ceiling

    def reserve(self) -> None:
        if not self.can_call():
            raise BudgetExhausted

    def charge(self, usage: dict[str, int]) -> None:
        self.calls += 1
        self.output_tokens += int(usage.get("output_tokens", 0) or 0)

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)
REWRITE_PROMPT = (
    "Task of the skill: {task}.\n\nBelow is a draft SKILL.md written by someone else, and "
    "its normative claims numbered C1..C{k}.\n\n<draft>\n{draft}\n</draft>\n\nClaims:\n"
    "{claims}\n\nRewrite the skill in your own words, with the same sections: one "
    "sentence per draft claim, keeping every condition, exception, \"only\" and stated "
    "order, adding no new rule. End each sentence with its single claim id, for example "
    "[C3]. Never merge two claims into one sentence (no [C1, C2]); a claim may take two "
    "sentences, each tagged with its id. You may reorder sentences, except where the "
    "draft says First/Then/Next/Afterwards/Finally. Reply with the rewritten "
    "skill inside <skill>...</skill>.\n\n" + FORMAT_RULES)


# --------------------------------------------------------------------------- results


@dataclasses.dataclass
class PatchResult:
    writer: str
    ok: bool
    changed: bool
    text: str | None            # the proposed full SKILL.md (None if not ok)
    provenance: list[str]
    problems: list[str]
    calls: list[dict[str, Any]]  # one record per model call (role, purpose, accounting)
    repairs: int = 0             # repair calls used (format repair + optional shorten)
    budget_exhausted: bool = False
    draft: str | None = None     # TS only: the teacher draft
    fidelity: dict[str, Any] | None = None  # TS only

    @property
    def sha256(self) -> str | None:
        return sha256_text(self.text) if self.text else None

    @property
    def tokens(self) -> int:
        return count_tokens(self.text) if self.text else 0

    def call_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.calls:
            out[c["role"]] = out.get(c["role"], 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.update(sha256=self.sha256, tokens=self.tokens, call_counts=self.call_counts())
        return d


@dataclasses.dataclass
class EvidenceBatch:
    """Episodes of one patch step: dicts with ``index`` (1-based stream
    position), ``case``, ``action``, ``score`` and ``correction`` (or None)."""

    episodes: list[dict[str, Any]]
    policy: str

    def text(self) -> str:
        return render_evidence(self.episodes)

    def forbidden(self) -> set[str]:
        """Case values the skill may not contain (demo-1 ``evidence_values``),
        minus words of the policy excerpt, the writer prompts and
        ``EXTRA_GENERAL`` (option values such as 'full' are ordinary words)."""
        cases = [e["case"] for e in self.episodes]
        results = [{"case_id": e["case"]["case_id"], "action": e["action"]}
                   for e in self.episodes]
        vocab = set(re.findall(r"[a-z]+", (self.policy + PATCH_PROMPT + TASK).lower()))
        return {v for v in evidence_values(cases, results)
                if v.lower() not in vocab | EXTRA_GENERAL}

    def indices(self) -> set[int]:
        return {e["index"] for e in self.episodes}


def render_action(action: dict[str, Any]) -> str:
    from slh.demo1.decide import render_action as render

    return render(action)


def render_evidence(episodes: list[dict[str, Any]]) -> str:
    """What every writer sees: visible conversation, lookups, the agent's action,
    the deterministic outcome, lookup errors and the scripted correction. No gold."""
    blocks = []
    for e in episodes:
        case = e["case"]
        turns = []
        for m in case["messages"]:
            if m["role"] in ("user", "assistant") and m.get("content"):
                turns.append(f"  {m['role']}: {m['content']}")
        errors = [m["content"] for m in case["messages"]
                  if m["role"] == "tool" and m.get("error")]
        errors += [f"{r['name']} failed" for r in e.get("reads") or [] if r.get("error")]
        corr = e.get("correction")
        score = e["score"]
        blocks.append("\n".join([
            f"Episode E{e['index']}",
            "Conversation:", *turns,
            "Lookup facts:", *[f"  {x}" for x in _facts({**case, "task": "address"})],
            f"Agent's action: {render_action(e['action'])}",
            (f"Outcome: {score.get('outcome')} "
             f"({'correct' if score.get('correct') else 'not correct'})"),
            f"Lookup errors: {'; '.join(errors) if errors else 'none'}",
            f"User's correction: {corr['text'] if corr else 'none'}"]))
    return "\n\n".join(blocks)


def policy_excerpt() -> str:
    from slh.learner.evidence import policy_excerpt as excerpt

    return excerpt("Modify pending order")["text"]


# --------------------------------------------------------------------------- parsing and checks


def band_args(band: tuple[int, int]) -> dict[str, int]:
    """Token band plus its word guidance (500 tokens ~ 330 words)."""
    return {"low": band[0], "high": band[1], "words_low": round(band[0] * 5 / 6),
            "words_high": round(band[1] * 0.66)}


def section_sizes(text: str) -> dict[str, int]:
    parts = re.split(r"^(?=#+\s)", text, flags=re.MULTILINE)
    out = {}
    for part in parts:
        m = re.match(r"#+\s*(.+?)\s*$", part, re.MULTILINE)
        if m:
            out[m.group(1)] = count_tokens(part)
    return out


def length_only_over(problems: list[str], text: str | None, band: tuple[int, int]) -> bool:
    """The only problem is length, over the band by at most SHORTEN_SLACK."""
    return bool(text) and len(problems) == 1 and "the length must be between" in problems[0] \
        and band[1] < count_tokens(text) <= band[1] * (1 + SHORTEN_SLACK)


def length_note(text: str | None, band: tuple[int, int]) -> str:
    if not text:
        return " "
    tokens = count_tokens(text)
    if band[0] <= tokens <= band[1]:
        return " "
    sizes = section_sizes(text)
    longest = sorted(sizes, key=sizes.get, reverse=True)[:2]
    return LENGTH_NOTE.format(tokens=tokens, sizes=", ".join(f"{k} {v}" for k, v in
                                                             sizes.items()) or "unknown",
                              longest=" and ".join(longest) or "the longest sections",
                              **band_args(band))


def shorten_prompt(text: str, band: tuple[int, int]) -> str:
    sizes = section_sizes(text)
    return SHORTEN_PROMPT.format(tokens=count_tokens(text), high=band[1],
                                 words_target=round(band[1] * 0.6),
                                 sizes=", ".join(f"{k} {v}" for k, v in sizes.items()),
                                 text=text)


def parse_reply(raw: str | None) -> tuple[str | None, list[str], bool]:
    """(skill text, provenance lines, no_change). A reply without <skill> tags
    that starts with a markdown heading is taken as the skill (minus any
    provenance block)."""
    text = clean(raw)
    if text.strip().rstrip(".").upper() == NO_CHANGE:
        return None, [], True
    m = re.search(r"<skill>(.*?)</skill>", text, re.DOTALL | re.IGNORECASE)
    skill = m.group(1).strip() if m else None
    if skill is None and re.match(r"#+\s", text):
        skill = re.sub(r"<provenance>.*?(</provenance>|$)", "", text,
                       flags=re.DOTALL | re.IGNORECASE).strip() or None
    p = re.search(r"<provenance>(.*?)</provenance>", text, re.DOTALL | re.IGNORECASE)
    prov = [ln.strip() for ln in (p.group(1) if p else "").splitlines() if ln.strip()]
    return skill, prov, False


def check_skill(text: str | None, *, forbidden: set[str], band: tuple[int, int] = SKILL_TOKENS,
                provenance: list[str] | None = None, indices: set[int] | None = None
                ) -> list[str]:
    """Mechanical checks: format (sections, length band), policy hygiene (no
    case-specific values or ids; demo-1 checks) and evidence support
    (provenance cites episodes of this batch only). ``indices`` None: no
    provenance required (generic note)."""
    if not text:
        return ["no <skill>...</skill> block in the reply"]
    problems = []
    heads = re.findall(r"^#+\s*(.+?)\s*$", text, re.MULTILINE)
    if [h for h in heads if h in SECTIONS] != list(SECTIONS):
        problems.append("the sections must be exactly " + ", ".join(SECTIONS) + ", in order")
    tokens = count_tokens(text)
    if not band[0] <= tokens <= band[1]:
        problems.append(f"{tokens} tokens; the length must be between {band[0]} and {band[1]}")
    for label, pattern in _ID_PATTERNS:
        if pattern.search(text):
            problems.append(f"it contains a specific {label}")
    if re.search(r"\bE\d+\b", text):
        problems.append("the skill cites episode numbers (they belong in the provenance)")
    hits = value_hits(text, forbidden)
    if hits:
        problems.append(f"it contains {len(hits)} case-specific value(s) "
                        f"(for example {hits[0]!r})")
    if indices is not None:
        cited = {int(n) for line in provenance or [] for n in re.findall(r"\bE(\d+)\b", line)}
        if not cited:
            problems.append("the provenance cites no episode")
        elif cited - indices:
            problems.append(f"the provenance cites episodes not in this evidence: "
                            f"{sorted(cited - indices)}")
    return problems


# --------------------------------------------------------------------------- fidelity (TS)

_NORMATIVE = re.compile(
    r"\b(must|never|always|only|do not|don't|should|ask|confirm|check|verify|call|use|"
    r"write|update|modify|proceed|tell|if|when|unless|before|after|avoid)\b", re.IGNORECASE)
_NEG = re.compile(r"\b(never|not|don't|no|none|nothing|neither|without|avoid)\b", re.IGNORECASE)
ACTS = {
    "ask": ("ask", "question", "clarif", "which"),
    "confirm": ("confirm", "approv", "consent", "yes"),
    "write": ("write", "call", "modify", "update", "change", "proceed", "submit", "apply"),
    "transfer": ("transfer", "human"),
}
_TAGS = re.compile(r"\[\s*(C\d+(?:\s*,\s*C\d+)*)\s*\]")


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(]?[A-Z])")


def sentences(line: str) -> list[str]:
    return [x.strip() for x in _SENTENCE_END.split(line) if x.strip()]


def claims(text: str) -> list[str]:
    """Normative claims: sentences of non-heading lines (bullets or paragraphs)
    with a normative marker, in text order. One claim per sentence, so a
    rewrite may reorder or merge sentences without changing a claim."""
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if line and not line.startswith("#"):
            out.extend(x for x in sentences(line) if _NORMATIVE.search(x))
    return out


# A claim that starts with one of these is ordered after the claim before it
# (and a claim starting with "First" before the claim after it).
_THEN_START = re.compile(r"^(then|next|after that|afterwards|finally|subsequently|"
                         r"once (?:done|that)|after (?:this|that))\b", re.IGNORECASE)
_FIRST_START = re.compile(r"^first\b", re.IGNORECASE)


def ordered_pairs(draft_claims: list[str]) -> list[tuple[int, int]]:
    """(i, j), 1-based: explicit sequence relations between consecutive claims."""
    return [(j - 1, j) for j in range(2, len(draft_claims) + 1)
            if _THEN_START.search(draft_claims[j - 1]) or
            _FIRST_START.search(draft_claims[j - 2])]


def _acts(text: str) -> set[str]:
    low = text.lower()
    return {a for a, stems in ACTS.items() if any(s in low for s in stems)}


# Condition features (canonical label -> phrases). Multi-word phrases are
# matched first and removed, so "exactly one" is ONE, not also a bare "one".
CONDITIONS = (
    ("count:one", r"exactly one|only one|just one|a single|single|unique(?:ly)?|one(?! of)"),
    ("count:many", r"more than one|two or more|several|multiple|many|both|two|ambiguous"),
    ("count:none", r"no|none|zero|nothing|neither"),
    ("count:any", r"any"),
    ("count:all", r"all|every|each"),
    ("only", r"only"),
    ("status:pending", r"pending"),
    ("status:delivered", r"delivered"),
    ("status:processed", r"processed|shipped"),
    ("status:cancelled", r"cancell?ed"),
    ("approval", r"approv\w*|confirm\w*|consent\w*|agreed|said yes"),
    ("clause:if", r"if|when|whenever|once|where"),
    ("clause:except", r"unless|except|other than"),
    ("clause:before", r"before|until|first"),
    ("clause:after", r"after|afterwards"),
)
_COND_RES = [(label, re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE))
             for label, pattern in CONDITIONS]


def conditions(text: str) -> set[str]:
    """Condition features of a claim: quantifiers, statuses, approval words and
    condition/exception/order clauses (a phrase counts once, longest first)."""
    found, rest = set(), text.lower()
    for label, pattern in _COND_RES:
        if pattern.search(rest):
            found.add(label)
            rest = pattern.sub(" ", rest)
    return found


# Ordered logic tokens of a claim: connectives, negation, clause markers,
# conditions and actions, in text order (longest phrase first; consecutive
# repeats collapsed). "then" right after an if-clause with no action yet is
# just the if-clause's "then" and is skipped.
SEQUENCE = (
    ("none", r"no|none|nothing|neither|zero"),
    ("not", r"never|not|don't|do not|without|avoid"),
    ("and", r"and|as well as|also"),
    ("or", r"or|either"),
    ("except", r"unless|except|other than"),
    ("if", r"if|when|whenever|once|where"),
    ("before", r"before|until|first|prior to"),
    ("after", r"after|afterwards|then|once done"),
    ("one", r"exactly one|only one|just one|a single|single|unique(?:ly)?|one(?! of)"),
    ("many", r"more than one|two or more|several|multiple|many|both|two|ambiguous"),
    ("any", r"any"), ("all", r"all|every|each"), ("only", r"only"),
    ("pending", r"pending"), ("delivered", r"delivered"),
    ("processed", r"processed|shipped"), ("cancelled", r"cancell?ed"),
    ("ask", r"ask\w*|question\w*|clarif\w*|which"),
    ("confirm", r"confirm\w*|approv\w*|consent\w*|agreed|said yes|yes"),
    ("write", (r"write\w*|writing|call\w*|modif\w*|updat\w*|chang\w*|proceed\w*|"
               r"submit\w*|apply\w*|make the change")),
    ("transfer", r"transfer\w*|human"),
)
_PHRASES = sorted(((p, label) for label, pats in SEQUENCE for p in pats.split("|")),
                  key=lambda x: -len(x[0].replace("\\w*", "")))
_SEQ_RE = re.compile(r"\b(?:" + "|".join(f"({p})" for p, _ in _PHRASES) + r")\b",
                     re.IGNORECASE)
ACTIONS = frozenset({"ask", "confirm", "write", "transfer"})


def logic_sequence(text: str) -> list[str]:
    out: list[str] = []
    for m in _SEQ_RE.finditer(text):
        label = _PHRASES[m.lastindex - 1][1]
        if label == "after" and m.group(0).lower() == "then" and "if" in out and \
                not (ACTIONS - {"confirm"}) & set(out[len(out) - out[::-1].index("if"):]):
            continue
        if not out or out[-1] != label:
            out.append(label)
    return out


EQUIVALENCE_PROMPT = (
    "Draft rules:\n<draft>\n{draft}\n</draft>\n\nRewrite:\n<rewrite>\n{rewrite}\n</rewrite>\n\n"
    "Does the rewrite say exactly the same rules, conditions, exceptions and order as the "
    "draft? Wording may differ. Answer EQUIVALENT, or list each difference on its own line "
    "starting with '- '.")


def parse_equivalence(raw: str | None) -> list[str] | None:
    """[] when EQUIVALENT, the listed differences otherwise, None if unreadable."""
    if raw is None:
        return None
    text = clean(raw)
    if re.fullmatch(r"(?is)\W*equivalent\W*", text):
        return []
    diffs = [ln.strip()[2:].strip() for ln in text.splitlines() if ln.strip().startswith("- ")]
    if diffs:
        return diffs
    return [text.strip()] if text.strip() else None


def tagged_sentences(rewrite: str) -> list[tuple[str, list[int]]]:
    """Every sentence of every non-heading line with the claim ids it carries
    (repeats deduplicated): text up to a tag group is split into sentences; the
    last one carries the group, any earlier ones carry none. Text after the
    last tag group of a line is an untagged sentence."""
    out: list[tuple[str, list[int]]] = []

    def add(text: str, tags: list[int]) -> None:
        parts = sentences(text.strip(" .;:,")) or ([""] if tags else [])
        for part in parts[:-1]:
            out.append((part.strip(" .;:,"), []))
        if parts:
            out.append((parts[-1].strip(" .;:,"), tags))

    for line in rewrite.splitlines():
        body = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if not body or body.startswith("#"):
            continue
        pos = 0
        for m in _TAGS.finditer(body):
            tags = list(dict.fromkeys(int(t.strip()[1:]) for t in m.group(1).split(",")))
            text = body[pos:m.start()]
            if not text.strip(" .;:,") and out and pos > 0:
                # "Sentence. [C1] [C2]": a second group on the same sentence.
                out[-1] = (out[-1][0], list(dict.fromkeys(out[-1][1] + tags)))
            else:
                add(text, tags)
            pos = m.end()
        rest = body[pos:]
        if re.search(r"[A-Za-z]{2,}", rest):
            add(rest, [])
    return out


def fidelity(draft_claims: list[str], rewrite: str) -> dict[str, Any]:
    """Mechanical claim preservation. Claims are draft sentences; every
    rewritten sentence carries exactly one claim tag (``tagged_sentences``).

    - merged: a sentence tagged with more than one claim (restate each claim as
      its own sentence);
    - invented: a normative sentence without a tag, or a tag naming no claim;
    - dropped: a draft claim no sentence carries;
    - altered: per claim, all sentences tagged with it, joined in order, against
      that claim alone: negation flipped, an act kind (ask/confirm/write/
      transfer) lost, a condition feature (``conditions``: counts, statuses,
      approval, if/unless/before/after, 'only') lost or added, or its ordered
      logic tokens (``logic_sequence``) changed; and an explicit draft order
      (First/Then/Next/Afterwards/Finally between consecutive claims) reversed
      by the position of the first sentence tagged with each claim.
    Sentences may otherwise be reordered freely."""
    k = len(draft_claims)
    by_claim: dict[int, list[str]] = {}
    first_pos: dict[int, int] = {}
    merged: list[str] = []
    merged_ids: set[int] = set()
    invented, altered = [], []
    for pos, (sentence, tags) in enumerate(tagged_sentences(rewrite)):
        if not tags:
            if _NORMATIVE.search(sentence):
                invented.append(sentence)
            continue
        if [t for t in tags if not 1 <= t <= k]:
            invented.append(sentence)
        valid = [t for t in tags if 1 <= t <= k]
        if len(valid) > 1:
            merged.append(f"{', '.join(f'C{t}' for t in valid)}: {sentence}")
            merged_ids.update(valid)
            continue
        for t in valid:
            by_claim.setdefault(t, []).append(sentence)
            first_pos.setdefault(t, pos)
    dropped = [f"C{i}: {draft_claims[i - 1]}" for i in range(1, k + 1)
               if i not in by_claim and i not in merged_ids]
    for i, lines in sorted(by_claim.items()):
        claim, joined = draft_claims[i - 1], " ".join(lines)
        want, got = logic_sequence(claim), logic_sequence(joined)
        lost = conditions(claim) - conditions(joined)
        added = conditions(joined) - conditions(claim)
        if bool(_NEG.search(claim)) != bool(_NEG.search(joined)):
            altered.append(f"C{i}: negation differs")
        elif _acts(claim) - _acts(joined):
            altered.append(f"C{i}: loses {sorted(_acts(claim) - _acts(joined))}")
        elif lost:
            altered.append(f"C{i}: loses condition {sorted(lost)}")
        elif added:
            altered.append(f"C{i}: adds condition {sorted(added)}")
        elif want != got:
            altered.append(f"C{i}: logic order differs ({got} vs {want})")
    for a, b in ordered_pairs(draft_claims):
        if a in first_pos and b in first_pos and first_pos[a] > first_pos[b]:
            altered.append(f"C{b}: now comes before C{a}, which the draft orders first")
    return {"claims": k, "dropped": dropped, "invented": invented, "altered": altered,
            "merged": merged, "ok": not (dropped or invented or altered or merged)}


def restore_list(draft_claims: list[str], fid: dict[str, Any]) -> list[str]:
    """Repair instructions naming each merged/dropped/altered draft claim, quoted
    (every claim an error names)."""
    def quote(ids: list[int]) -> str:
        return "; ".join(f'C{n}: "{draft_claims[n - 1]}"' for n in dict.fromkeys(ids)
                         if 1 <= n <= len(draft_claims))

    out = []
    for m in fid.get("merged", []):
        ids = [int(x) for x in re.findall(r"C(\d+)", m.split(":", 1)[0])]
        out.append(f"claims were merged into one sentence ({m}); split them: restate each "
                   f"claim as its own sentence tagged with its single id: {quote(ids)}")
    for d in fid["dropped"]:
        n = int(re.match(r"C(\d+)", d).group(1))
        out.append(f"draft claim C{n} is missing; restore it as its own sentence tagged "
                   f"[C{n}]: {quote([n])}")
    for a in fid["altered"]:
        head = "" if a.startswith("semantic:") else a.split(":", 1)[0]
        ids = [int(x) for x in re.findall(r"C(\d+)", head)]
        ids += [int(x) for x in re.findall(r"before C(\d+)", a)]
        if ids:
            out.append(f"draft claim(s) changed ({a}); restore every condition, exception, "
                       f'"only" and order they state: {quote(ids)}')
        else:
            out.append(f"the rewrite differs from the draft: {a}")
    out += [f"rule without a single claim tag (tag it with its one claim or remove it): {x}"
            for x in fid["invented"]]
    return out


def strip_tags(text: str) -> str:
    return "\n".join(_TAGS.sub("", ln).rstrip() for ln in text.splitlines()).strip()


# --------------------------------------------------------------------------- chats


def agent_chat(client: Any) -> Chat:
    """The 9B agent through the demo-1 capped client (metered on its meter)."""
    from slh.demo1.learn import _ask

    def chat(system: str, user: str, purpose: str = "write"):
        return _ask(client, system, user)

    return chat


def _norm_model(model: str) -> str:
    name = model.strip().lower().split("/")[-1]
    return re.sub(r"-(\d{4}-\d{2}-\d{2}|\d{4})$", "", name)


def forbidden_models(user_path: Path = USER_CONFIG) -> dict[str, str]:
    from slh.runner import judge_config

    user = yaml.safe_load(user_path.read_text(encoding="utf-8"))["model"]
    return {"user simulator": user, "judge": judge_config().model}


TEACHER_KEYS = ("cap_usd", "price")


def load_teacher(path: Path = TEACHER_CONFIG, *, forbidden: dict[str, str] | None = None
                 ) -> dict[str, Any]:
    """The teacher config, refused while a placeholder remains, without one
    pinned provider and no fallbacks, without thinking off, or when the model
    is the user simulator or the judge."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    todo = [k for k in ("model", "provider") if PLACEHOLDER_MARK in str(raw.get(k))]
    price = raw.get("price") or {}
    todo += [f"price.{k}" for k in ("input_per_mtok", "output_per_mtok")
             if not isinstance(price.get(k), int | float) or isinstance(price.get(k), bool)
             or not math.isfinite(price[k])]
    if todo:
        raise ValueError(f"{Path(path).name}: the owner has not filled {todo}")
    provider = raw.get("provider") or {}
    if len(provider.get("order") or []) != 1 or provider.get("allow_fallbacks") is not False:
        raise ValueError("the teacher pins exactly one provider with allow_fallbacks: false")
    if (raw.get("reasoning") or {}).get("enabled") is not False or raw.get("temperature") != 0:
        raise ValueError("the teacher runs non-thinking at temperature 0")
    for role, model in (forbidden if forbidden is not None else forbidden_models()).items():
        if _norm_model(raw["model"]) == _norm_model(model):
            raise ValueError(f"the teacher model {raw['model']!r} is the {role} model "
                             f"({model!r}); choose another")
    return raw


def teacher_model_config(raw: dict[str, Any]) -> Any:
    from slh.budget import ModelConfig

    return ModelConfig.from_dict({k: v for k, v in raw.items() if k not in TEACHER_KEYS})


class TeacherChat:
    """The teacher behind its own budget meter (role 'agent' of that meter is
    the teacher). Calls run inside ``completion_guard`` (checked and charged at
    the wire). Log-probs are requested only when the config sets
    ``capture: logprobs`` (allowed only with ``provider_advertises_logprobs``,
    checked at preflight against the pinned endpoint); nothing depends on them.

    ``log_to(path)``: every attempt is appended to ``path`` (teacher-calls.jsonl)
    in full: purpose, system and user text, the pinned provider, the provider
    and model id each response reports, response text, usage, charged cost,
    error and ``logprobs_available``. ``backend`` may be injected (tests)."""

    def __init__(self, raw: dict[str, Any], *, backend: Any = None) -> None:
        from slh.agent import LiteLLMBackend
        from slh.budget import Meter, Price

        self.raw = raw
        self.model = teacher_model_config(raw)
        price = raw["price"]
        self.meter = Meter(float(raw["cap_usd"]), {self.model.model: Price(
            input=float(price["input_per_mtok"]), output=float(price["output_per_mtok"]),
            cached_input=float(price.get("cached_input_per_mtok", price["input_per_mtok"])),
            max_output_tokens=price.get("max_output_tokens"),
            context_window=price.get("context_window"))})
        self.level = "logprobs" if "logprobs" in self.model.capture_levels() else "messages"
        self.backend = backend or LiteLLMBackend(self.model, "teacher", self.level)
        self.log_path: Path | None = None
        self._current: dict[str, Any] = {}
        self.meter.on_attempt = self._on_attempt

    def log_to(self, path: Path) -> None:
        self.log_path = Path(path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_attempt(self, event: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        responses = [{k: v for k, v in r.items() if k != "token_capture"}
                     | {"logprobs_available": bool(
                         (r.get("token_capture") or {}).get("logprobs"))}
                     for r in event["responses"]]
        row = {**self._current, "attempt": event["attempt"],
               "model_requested": self.model.model,
               "provider_pinned": thaw_order(self.model),
               "request_level": self.level,
               "responses": responses,
               "provider_reported": [r.get("provider") for r in responses],
               "model_reported": [r.get("model_reported") for r in responses],
               "logprobs_available": any(r["logprobs_available"] for r in responses),
               "charged_usd": event["charged_usd"], "reservation_usd": event["reservation_usd"],
               "error_type": event["error_type"], "dispatched": event["dispatched"],
               "attempt_s": event["attempt_s"]}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def __call__(self, system: str, user: str, purpose: str = "write"):
        from tau2.data_model.message import SystemMessage, UserMessage

        from slh.agent import metered_complete
        from slh.budget import BudgetExceeded, completion_guard

        self._current = {"purpose": purpose, "system": system, "user": user,
                         "user_sha256": sha256_text(user)}
        t = self.meter.tally["agent"]
        before = (t.calls, t.failed_attempts, t.input_tokens, t.output_tokens)
        error = None
        try:
            with completion_guard():
                reply = metered_complete(self.meter, "agent", self.backend,
                                         [SystemMessage(role="system", content=system),
                                          UserMessage(role="user", content=user)], None, None)
            text = reply.content
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded; the patch is not valid
            text, error = None, f"{type(exc).__name__}: {exc}"
        t = self.meter.tally["agent"]
        after = (t.calls, t.failed_attempts, t.input_tokens, t.output_tokens)
        keys = ("model_calls", "failed_attempts", "input_tokens", "output_tokens")
        return text, {k: a - b for k, a, b in zip(keys, after, before, strict=True)}, error

    @property
    def spent_usd(self) -> float:
        return self.meter.spent_usd


def thaw_order(model: Any) -> list[str]:
    return list((model.extra_params().get("provider") or {}).get("order") or [])


# --------------------------------------------------------------------------- writers


class Writer:
    """Base: ``draft`` (one call by the drafting model), then mechanical checks
    with one repair by ``final`` (the model whose words the skill is in)."""

    name = "?"

    def __init__(self, final: Chat, final_role: str) -> None:
        self.final, self.final_role = final, final_role
        self.budget = WriterBudget()

    def chats(self) -> list[Chat]:
        return [self.final]

    def _call(self, chat: Chat, role: str, purpose: str, system: str, user: str,
              calls: list[dict[str, Any]]) -> str | None:
        self.budget.reserve()
        raw, usage, error = chat(system, user, purpose=purpose)
        self.budget.charge(usage)
        calls.append({"role": role, "purpose": purpose, "prompt_sha256": sha256_text(user),
                      "raw": raw, "error": error, **usage})
        return None if error else raw

    def _draft(self, system: str, prompt: str, calls: list[dict[str, Any]]) -> str | None:
        return self._call(self.final, self.final_role, "write", system, prompt, calls)

    def _finish(self, raw: str | None, system: str, calls: list[dict[str, Any]], *,
                forbidden: set[str], band: tuple[int, int], indices: set[int] | None
                ) -> PatchResult:
        if raw is None:
            return PatchResult(self.name, False, False, None, [],
                               [f"writer call failed: {calls[-1]['error']}"], calls)
        skill, prov, no_change = parse_reply(raw)
        if no_change and indices is not None:
            return PatchResult(self.name, True, False, None, [], [], calls)
        problems = check_skill(skill, forbidden=forbidden, band=band, provenance=prov,
                               indices=indices)
        repairs = 0
        if problems:
            repair = REPAIR_PROMPT.format(
                problems="\n".join(f"- {p}" for p in problems), text=clean(raw),
                length=length_note(skill, band),
                prov=", <provenance>...</provenance>" if indices is not None else "")
            raw = self._call(self.final, self.final_role, "repair", system, repair, calls)
            repairs += 1
            if raw is None:
                return PatchResult(self.name, False, False, None, prov,
                                   problems + [f"repair call failed: {calls[-1]['error']}"],
                                   calls, repairs)
            skill2, prov2, _ = parse_reply(raw)
            skill, prov = skill2, prov2 or prov
            problems = check_skill(skill, forbidden=forbidden, band=band, provenance=prov,
                                   indices=indices)
        if length_only_over(problems, skill, band):
            # One extra mechanical-only repair: shorten without changing meaning.
            raw = self._call(self.final, self.final_role, "shorten", system,
                             shorten_prompt(skill, band), calls)
            repairs += 1
            if raw is not None:
                skill = parse_reply(raw)[0]
                problems = check_skill(skill, forbidden=forbidden, band=band,
                                       provenance=prov, indices=indices)
            else:
                problems = problems + [f"shorten call failed: {calls[-1]['error']}"]
        ok = not problems
        return PatchResult(self.name, ok, ok, skill if ok else None, prov, problems, calls,
                           repairs)

    def propose_patch(self, current_skill: str, evidence_batch: EvidenceBatch) -> PatchResult:
        system = SYSTEM.format(policy=evidence_batch.policy)
        prompt = PATCH_PROMPT.format(task=TASK, skill=current_skill or "(empty)",
                                     n=len(evidence_batch.episodes),
                                     evidence=evidence_batch.text(),
                                     **band_args(SKILL_TOKENS))
        calls: list[dict[str, Any]] = []
        return self._guarded(system, prompt, calls, forbidden=evidence_batch.forbidden(),
                             band=SKILL_TOKENS, indices=evidence_batch.indices())

    def _guarded(self, system, prompt, calls, **kw) -> PatchResult:
        """``_produce`` with the ceilings: a call that does not fit rejects the
        proposal as budget_exhausted instead of raising."""
        try:
            return self._produce(system, prompt, calls, **kw)
        except BudgetExhausted:
            return PatchResult(self.name, False, False, None, [], [BUDGET_EXHAUSTED], calls,
                               sum(c["purpose"] in ("repair", "shorten") for c in calls),
                               budget_exhausted=True)

    def _produce(self, system, prompt, calls, *, forbidden, band, indices) -> PatchResult:
        raw = self._draft(system, prompt, calls)
        return self._finish(raw, system, calls, forbidden=forbidden, band=band,
                            indices=indices)

    def generic_note(self, policy: str, target_tokens: int | None = None,
                     forbidden: set[str] | None = None) -> PatchResult:
        band = generic_band(target_tokens)
        system = SYSTEM.format(policy=policy)
        prompt = GENERIC_PROMPT.format(task=TASK, **band_args(band))
        return self._guarded(system, prompt, [], forbidden=forbidden or set(), band=band,
                             indices=None)


class TeacherWriter(Writer):
    name = "T"

    def __init__(self, teacher: Chat) -> None:
        super().__init__(teacher, "teacher")


class AgentWriter(Writer):
    name = "S"

    def __init__(self, agent: Chat) -> None:
        super().__init__(agent, "agent")


class TeacherThenAgentWriter(Writer):
    """Teacher drafts (same prompt as T); the 9B rewrites with claim tags; the
    fidelity check rejects dropped/invented/altered claims; the 9B repairs."""

    name = "TS"

    def __init__(self, teacher: Chat, agent: Chat) -> None:
        super().__init__(agent, "agent")
        self.teacher = teacher

    def chats(self) -> list[Chat]:
        return [self.teacher, self.final]

    def _produce(self, system, prompt, calls, *, forbidden, band, indices) -> PatchResult:
        raw = self._call(self.teacher, "teacher", "draft", system, prompt, calls)
        if raw is None:
            return PatchResult(self.name, False, False, None, [],
                               [f"teacher draft failed: {calls[-1]['error']}"], calls)
        draft, prov, no_change = parse_reply(raw)
        if no_change and indices is not None:
            return PatchResult(self.name, True, False, None, [], [], calls)
        if not draft:
            return PatchResult(self.name, False, False, None, prov,
                               ["the teacher draft has no <skill> block"], calls)
        draft_claims = claims(draft)
        rewrite = REWRITE_PROMPT.format(
            task=TASK, draft=draft, k=len(draft_claims),
            claims="\n".join(f"C{i}: {c}" for i, c in enumerate(draft_claims, 1)),
            **band_args(band))
        raw = self._call(self.final, "agent", "rewrite", system, rewrite, calls)
        tagged, _, _ = parse_reply(raw) if raw else (None, [], False)
        fid, problems = self._verify(draft, draft_claims, tagged, system, calls,
                                     forbidden=forbidden, band=band, prov=prov,
                                     indices=indices)
        repairs = 0
        if not fid["ok"] or problems:
            # One repair by the 9B on the tagged text; everything is re-checked after it.
            issues = problems + restore_list(draft_claims, fid)
            repair = REPAIR_PROMPT.format(problems="\n".join(f"- {p}" for p in issues),
                                          text=f"<skill>\n{tagged or ''}\n</skill>", prov="",
                                          length=length_note(strip_tags(tagged or ""), band))
            raw = self._call(self.final, "agent", "repair", system, repair, calls)
            repairs += 1
            tagged, _, _ = parse_reply(raw) if raw else (None, [], False)
            fid, problems = self._verify(draft, draft_claims, tagged, system, calls,
                                         forbidden=forbidden, band=band, prov=prov,
                                         indices=indices)
        if fid["ok"] and length_only_over(problems, strip_tags(tagged or ""), band):
            raw = self._call(self.final, "agent", "shorten", system,
                             shorten_prompt(tagged, band), calls)
            repairs += 1
            tagged, _, _ = parse_reply(raw) if raw else (None, [], False)
            fid, problems = self._verify(draft, draft_claims, tagged, system, calls,
                                         forbidden=forbidden, band=band, prov=prov,
                                         indices=indices)
        if not fid["ok"]:
            problems = problems + ["fidelity: draft claims not preserved"]
        ok = not problems
        text = strip_tags(tagged) if ok and tagged else None
        return PatchResult(self.name, ok, ok, text, prov, problems, calls, repairs,
                           draft=draft, fidelity=fid)

    def _verify(self, draft, draft_claims, tagged, system, calls, *, forbidden, band, prov,
                indices) -> tuple[dict[str, Any], list[str]]:
        """Mechanical fidelity and checks; then, only if those pass, ONE binding
        semantic-equivalence call by the teacher (any difference = altered)."""
        fid = fidelity(draft_claims, tagged or "")
        problems = check_skill(strip_tags(tagged) if tagged else None, forbidden=forbidden,
                               band=band, provenance=prov, indices=indices)
        fid["semantic"] = None
        if fid["ok"] and not problems:
            raw = self._call(self.teacher, "teacher", "equivalence", system,
                             EQUIVALENCE_PROMPT.format(draft=draft, rewrite=strip_tags(tagged)),
                             calls)
            diffs = parse_equivalence(raw)
            fid["semantic"] = {"differences": diffs, "raw": raw}
            if diffs is None:
                diffs = ["equivalence check unreadable or failed (fail closed)"]
            fid["altered"] = fid["altered"] + [f"semantic: {d}" for d in diffs]
            fid["ok"] = not diffs
        return fid, problems


def generic_band(target_tokens: int | None) -> tuple[int, int]:
    """The learned skill's band, narrowed to target +/- 20% when a target is given."""
    if target_tokens is None:
        return SKILL_TOKENS
    return (max(SKILL_TOKENS[0], int(target_tokens * 0.8)),
            min(SKILL_TOKENS[1], round(target_tokens * 1.2)))


def generic_note(writer: Writer, target_tokens: int | None = None, *,
                 policy: str | None = None, forbidden: set[str] | None = None) -> PatchResult:
    """The same writer with no evidence, in the same length band."""
    return writer.generic_note(policy if policy is not None else policy_excerpt(),
                               target_tokens, forbidden)


def make_writer(name: str, *, agent_client: Any = None, teacher: Chat | None = None,
                agent: Chat | None = None, budget: WriterBudget | None = None) -> Writer:
    """Build writer T, S or TS. ``teacher`` defaults to the configured teacher
    (refused while placeholders remain); ``agent`` to the 9B capped client."""
    if name not in ("T", "S", "TS"):
        raise ValueError("writer is T, S or TS")
    if name in ("T", "TS") and teacher is None:
        teacher = TeacherChat(load_teacher())
    if name in ("S", "TS") and agent is None:
        agent = agent_chat(agent_client)
    writer = TeacherWriter(teacher) if name == "T" else AgentWriter(agent) if name == "S" \
        else TeacherThenAgentWriter(teacher, agent)
    if budget is not None:
        writer.budget = budget
    return writer
