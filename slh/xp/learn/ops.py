"""Edit operations on a lesson library: schema, strict parser, per-op gates.

Reply format (``OPS_SCHEMA``)::

    {"ops": [{"op": "add", "when": "...", "then": "...", "cases": ["E1"]},
             {"op": "replace", "id": "L2", "when": "...", "then": "...", "cases": ["E3"]},
             {"op": "remove", "id": "L4", "cases": ["E2"]}]}

``{"ops": []}`` or ``NO CHANGE`` means no edit. The parser is strict: the reply
must hold one JSON object whose only key is ``ops`` (a list); anything else is
a parse error (the loop allows one repair call). Each op is then checked on
its own (``apply_ops``) and rejected with reasons, never silently fixed:

- shape: known op, exactly its fields, string when/then, ``cases`` a non-empty
  list of this batch's labels (add/replace: the failure it comes from);
- at most ``max_ops`` ops; one op per entry id; replace/remove name an
  existing entry;
- size: when+then at most ``entry_tokens`` tokens (a short rule);
- hygiene (POC 9's checks per entry): no ids, emails, prices or case numbers
  (demo-1 ``_ID_PATTERNS``), no case labels (E3) and no case-specific value of
  this batch (``case_value_hits``, which ignores list numbering such as "1.");
- not an exact duplicate of another entry, not a re-proposal of a rejected edit;
- no contradiction with another entry of the candidate (``contradicts``).

The library-level gates (soft token budget, optional critic, dev split) are in
``slh.xp.learn.loop``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from slh.demo1.learn import _ID_PATTERNS, clean, count_tokens, value_hits
from slh.poc9.writers import _NEG, _acts, conditions
from slh.xp.learn.library import Entry, Library, _key, norm_text

OPS_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["ops"],
    "properties": {"ops": {"type": "array", "items": {"oneOf": [
        {"type": "object", "additionalProperties": False,
         "required": ["op", "when", "then", "cases"],
         "properties": {"op": {"const": "add"}, "when": {"type": "string"},
                        "then": {"type": "string"},
                        "cases": {"type": "array", "minItems": 1,
                                  "items": {"type": "string"}}}},
        {"type": "object", "additionalProperties": False,
         "required": ["op", "id", "when", "then", "cases"],
         "properties": {"op": {"const": "replace"}, "id": {"type": "string"},
                        "when": {"type": "string"}, "then": {"type": "string"},
                        "cases": {"type": "array", "minItems": 1,
                                  "items": {"type": "string"}}}},
        {"type": "object", "additionalProperties": False, "required": ["op", "id"],
         "properties": {"op": {"const": "remove"}, "id": {"type": "string"},
                        "cases": {"type": "array", "items": {"type": "string"}}}},
    ]}}}}
FIELDS = {"add": ({"op", "when", "then", "cases"}, set()),
          "replace": ({"op", "id", "when", "then", "cases"}, set()),
          "remove": ({"op", "id"}, {"cases"})}
MIN_WORDS = 2
SIMILAR_WHEN = 0.6  # Jaccard of content words for two conditions to count as the same
_STOP = frozenset(["the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with", "is", "are", "be", "it", "its", "that", "this", "they", "their", "them", "user", "users", "agent", "when", "then", "any"])

# A list marker at the start of a line ("1.", "2)", "10.") is numbering, not a
# case value; POC 9's check counted it when a case carried the value "1".
_LIST_MARKER = re.compile(r"(?m)^(\s*)\d{1,3}[.)](?=\s|$)")


class OpsParseError(ValueError):
    pass


def parse_ops(raw: str | None) -> list[Any]:
    """The raw op list of a reply, or ``OpsParseError``."""
    text = clean(raw)
    if not text:
        raise OpsParseError("empty reply")
    if text.strip().rstrip(".").upper() == "NO CHANGE":
        return []
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise OpsParseError("no JSON object in the reply")
    try:
        data = json.loads(text[start:end + 1])
    except ValueError as exc:
        raise OpsParseError(f"invalid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise OpsParseError("the reply must be a JSON object")
    extra = sorted(set(data) - {"ops"})
    if extra or "ops" not in data:
        raise OpsParseError(f"the object must have exactly one key 'ops' (got {sorted(data)})")
    if not isinstance(data["ops"], list):
        raise OpsParseError("'ops' must be a list")
    return data["ops"]


def case_value_hits(text: str, forbidden: set[str]) -> list[str]:
    """``value_hits`` with list numbering removed first (the POC 9 false positive)."""
    return value_hits(_LIST_MARKER.sub(r"\1", text), forbidden)


def hygiene(text: str, forbidden: set[str]) -> list[str]:
    problems = [f"it contains a specific {label}" for label, pattern in _ID_PATTERNS
                if pattern.search(text)]
    if re.search(r"\b[EL]\d+\b", text):
        problems.append("it cites a case label or entry id (they belong in 'cases'/'id')")
    hits = case_value_hits(text, forbidden)
    if hits:
        problems.append(f"it contains {len(hits)} case-specific value(s) (for example "
                        f"{hits[0]!r})")
    return problems


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in _STOP}


def contradicts(a: Entry, b: Entry) -> str | None:
    """Mechanical contradiction: the same situation (content words of ``when``
    overlap >= SIMILAR_WHEN and the same condition features) with conflicting
    actions (a shared act with opposite negation, or disjoint act kinds)."""
    wa, wb = _words(a.when), _words(b.when)
    if not wa or not wb or len(wa & wb) / len(wa | wb) < SIMILAR_WHEN:
        return None
    if conditions(a.when) != conditions(b.when):
        return None
    acts_a, acts_b = _acts(a.then), _acts(b.then)
    neg_a, neg_b = bool(_NEG.search(a.then)), bool(_NEG.search(b.then))
    if acts_a & acts_b and neg_a != neg_b:
        return f"contradicts {b.id}: same situation, opposite instruction"
    if acts_a and acts_b and not acts_a & acts_b:
        return f"contradicts {b.id}: same situation, different action " \
               f"({sorted(acts_a)} vs {sorted(acts_b)})"
    return None


def op_key(op: dict[str, Any]) -> tuple[str, str, str, str]:
    """Identity of an edit for the rejected-edit memory."""
    return (str(op.get("op")), str(op.get("id", "")), _key(str(op.get("when", ""))),
            _key(str(op.get("then", ""))))


def compact_op(op: Any) -> Any:
    if not isinstance(op, dict):
        return op
    return {k: op[k] for k in ("op", "id", "when", "then", "cases") if k in op}


def _shape(op: Any, labels: dict[str, str]) -> list[str]:
    if not isinstance(op, dict):
        return ["an op must be a JSON object"]
    kind = op.get("op")
    if kind not in FIELDS:
        return [f"unknown op {kind!r} (add, replace or remove)"]
    required, optional = FIELDS[kind]
    problems = []
    if missing := sorted(required - set(op)):
        problems.append(f"missing field(s) {missing}")
    if extra := sorted(set(op) - required - optional):
        problems.append(f"unknown field(s) {extra}")
    for f in ("when", "then", "id"):
        if f in op and not isinstance(op[f], str):
            problems.append(f"'{f}' must be a string")
    for f in ("when", "then"):
        if isinstance(op.get(f), str) and len(norm_text(op[f]).split()) < MIN_WORDS:
            problems.append(f"'{f}' is empty or too short")
    if "cases" in op:
        cases = op["cases"]
        if not isinstance(cases, list) or not all(isinstance(c, str | int) for c in cases):
            problems.append("'cases' must be a list of case labels such as \"E1\"")
        else:
            norm = [c if isinstance(c, str) else f"E{c}" for c in cases]
            if kind != "remove" and not norm:
                problems.append("'cases' must name the failure case(s) the lesson comes from")
            if unknown := [c for c in norm if c.strip().upper() not in labels]:
                problems.append(f"'cases' names labels not in this batch: {unknown}")
    return problems


def apply_ops(library: Library, ops: list[Any], *, labels: dict[str, str], step: int,
              forbidden: set[str], max_ops: int, entry_tokens: int,
              rejected_keys: set[tuple[str, str, str, str]] | None = None
              ) -> tuple[Library, list[dict[str, Any]]]:
    """Validate ``ops`` one by one against the growing candidate; return the
    candidate (valid ops applied, in order) and one verdict per op.
    ``labels`` maps batch labels ("E1") to case ids."""
    cand = library.copy()
    verdicts: list[dict[str, Any]] = []
    touched: set[str] = set()
    for i, op in enumerate(ops):
        problems = [] if i < max_ops else [f"over the limit of {max_ops} ops per step"]
        problems = problems or _shape(op, labels)
        if not problems:
            kind = op["op"]
            if kind in ("replace", "remove"):
                if cand.get(op["id"]) is None:
                    problems.append(f"no entry {op['id']!r} in the library")
                elif op["id"] in touched:
                    problems.append(f"entry {op['id']} is already edited in this step")
            if kind != "remove":
                text = f"{op['when']} {op['then']}"
                tokens = count_tokens(text)
                if tokens > entry_tokens:
                    problems.append(f"{tokens} tokens; a lesson is at most {entry_tokens}")
                problems += hygiene(text, forbidden)
            if rejected_keys and op_key(op) in rejected_keys:
                problems.append("re-proposes an edit that was already rejected")
        if not problems and op["op"] != "remove":
            cases = [labels[(c if isinstance(c, str) else f"E{c}").strip().upper()]
                     for c in op["cases"]]
            new = Entry(op.get("id", "L?"), norm_text(op["when"]), norm_text(op["then"]),
                        tuple(dict.fromkeys(cases)), step)
            others = [e for e in cand.entries if e.id != op.get("id")]
            if any(e.key() == new.key() for e in others):
                problems.append("duplicates an existing entry")
            problems += [r for e in others if (r := contradicts(new, e))]
        if not problems:
            if op["op"] == "add":
                entry = cand.add(new.when, new.then, list(new.source_case_ids), step)
                touched.add(entry.id)
            elif op["op"] == "replace":
                cand.replace(op["id"], new.when, new.then, list(new.source_case_ids), step)
                touched.add(op["id"])
            else:
                cand.remove(op["id"])
                touched.add(op["id"])
        verdicts.append({"index": i, "op": compact_op(op), "ok": not problems,
                         "problems": problems})
    return cand, verdicts
