"""The learning step, its controls and the owner review (draft amendment 8).

- ``evidence_text``: the train evidence of one task: each train case's request,
  the decisive lookup facts, the agent's own next action and the correct one.
  Built from train cases only; calibration and sealed cases are never read.
- ``write_procedure``: the same 9B agent model (the served B1 client, no
  tools) writes ONE short WHEN/THEN procedure from the policy and that
  evidence. One mechanical format repair is allowed; nothing else.
- ``write_generic``: the same writer, the same rules and length target, but
  WITHOUT the evidence (the same-length generic-note control).
- ``full_context_note``: the evidence text itself, used as a note (the
  full-context control: same evidence, no procedure, same insertion point).
- Owner review: accept or reject a procedure by its hash; no editing path.
- ``human_procedure``: the person-written procedure, loaded from
  ``demo1/human-procedure-{task}.md`` (a placeholder until the owner supplies it).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from slh.demo1.decide import AgentClient, render_action
from slh.demo1.schema import REPO

TOKEN_LIMIT = 200
GENERIC_LENGTH_BAND = (0.8, 1.2)
HUMAN_DIR = REPO / "demo1"
PLACEHOLDER = "PLACEHOLDER: the owner supplies this text before the window."
TASK_DESCRIPTION = {
    "address": "choosing the new shipping address when a user asks to ship a pending order "
               "to an address they describe by pointing to one of their other orders (or to "
               "ask the user when that description matches no address or more than one)",
    "variant": "choosing the replacement variant when a user asks to exchange or modify an "
               "item for a version they describe by options and price (or to ask the user "
               "when no available variant or more than one matches)",
}
WRITER_SYSTEM = ("You write short procedures for a retail customer-service agent. The agent "
                 "must follow this policy:\n\n<policy>\n{policy}\n</policy>")
RULES = (
    "Rules for the procedure:\n"
    "- Write one or more lines. Each line has the form: WHEN <situation the agent can "
    "observe> THEN <what the agent does>.\n"
    "- At most {limit} tokens in total.\n"
    "- Use only what the policy and the tool results support. Do not add new obligations.\n"
    "- Do not mention specific order ids, item ids, user ids, emails, names, addresses, "
    "cities, product names, option values, prices or case numbers. Describe the general "
    "situation.\n"
    "- Cover when to act and when to ask the user.\n"
    "Reply with the procedure only.")
LEARN_PROMPT = (
    "Below are {n} decisions the agent made on one task: {task}. For each case you see the "
    "user's request, the facts from the agent's lookups, the agent's next action and the "
    "correct next action.\n\n{evidence}\n\nWrite ONE short procedure that would make the "
    "agent choose the correct next action on new cases of this task.\n\n" + RULES)
GENERIC_PROMPT = (
    "The agent works on one task: {task}. Write ONE short procedure that would help the "
    "agent choose the correct next action on this task. Aim for about {target} tokens "
    "(between {low} and {high}).\n\n" + RULES)
REPAIR_PROMPT = (
    "Your procedure breaks these format rules:\n{problems}\n\nRewrite it to fix only these "
    "problems and keep its content otherwise. Reply with the procedure only.\n\n"
    "Your procedure:\n{text}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_tokens(text: str) -> int:
    from slh.learner.note_tokens import count_tokens as count

    return count(text)


# --------------------------------------------------------------------------- evidence


def _facts(case: dict[str, Any]) -> list[str]:
    """Decisive lookup facts, compressed from the case's own tool results."""
    lines = []
    for m in case["messages"]:
        if m["role"] != "tool":
            continue
        try:
            body = json.loads(m["content"])
        except (ValueError, TypeError):
            if m.get("error"):
                lines.append(f"lookup error: {m['content']}")
            continue
        if not isinstance(body, dict):
            continue
        if case["task"] == "address" and "order_id" in body:
            a = body["address"]
            pays = [p["payment_method_id"] for p in body.get("payment_history", [])
                    if p.get("transaction_type") == "payment"]
            lines.append(
                f"order {body['order_id']}: status {body['status']}; items "
                f"{', '.join(i['name'] for i in body['items'])}; paid with "
                f"{pays[0] if pays else 'unknown'}; address {a['address1']}, "
                f"{a['address2']}, {a['city']}, {a['state']} {a['zip']}, {a['country']}")
        elif case["task"] == "variant" and "variants" in body:
            for item_id, v in sorted(body["variants"].items()):
                opts = ", ".join(f"{k} {val}" for k, val in v["options"].items())
                lines.append(f"variant {item_id}: {opts}; price {v['price']}; "
                             f"{'available' if v['available'] else 'not available'}")
        elif case["task"] == "variant" and "order_id" in body:
            for item in body["items"]:
                if item["item_id"] == case["request"]["item_id"]:
                    opts = ", ".join(f"{k} {val}" for k, val in item["options"].items())
                    lines.append(f"current item {item['item_id']} in order "
                                 f"{body['order_id']} ({body['status']}): {opts}; price "
                                 f"{item['price']}")
    return lines


def evidence_text(cases: list[dict[str, Any]], results: list[dict[str, Any]]) -> str:
    """Train evidence for one task (train split only; refuses anything else)."""
    if any(c["split"] != "train" for c in cases):
        raise ValueError("evidence comes from train cases only")
    by_id = {r["case_id"]: r for r in results}
    blocks = []
    for n, case in enumerate(cases, 1):
        result = by_id[case["case_id"]]
        user_turns = [m["content"] for m in case["messages"] if m["role"] == "user"]
        proposal = [m["content"] for m in case["messages"]
                    if m["role"] == "assistant" and "To confirm:" in (m.get("content") or "")]
        gold = case["gold"]
        correct = render_action(gold) if gold["type"] == "tool_call" else \
            "message to the user asking what they mean (the description matches " + (
                "no address/variant)" if gold["reason"] == "no_match" else
                "more than one address/variant)")
        blocks.append("\n".join([
            f"Case {n}",
            f"User: {user_turns[0]}",
            f"Agent proposal: {proposal[0] if proposal else ''}",
            f"User: {user_turns[-1]}",
            "Lookup facts:", *[f"  {line}" for line in _facts(case)],
            f"Agent's next action: {render_action(result['action'])}",
            f"Correct next action: {correct}",
            f"Verdict: {'correct' if result['correct'] else 'incorrect'}"]))
    return "\n\n".join(blocks)


def full_context_note(task: str, evidence: str) -> str:
    return (f"Past decisions on the task of {TASK_DESCRIPTION[task]}, with the correct next "
            f"action for each:\n\n{evidence}")


# --------------------------------------------------------------------------- validation

_ID_PATTERNS = (
    ("order id", re.compile(r"#?W\d{7}")),
    ("item or product id", re.compile(r"\b\d{10}\b")),
    ("payment id", re.compile(r"\b(?:credit_card|gift_card|paypal)_\d+\b")),
    ("user id", re.compile(r"\b[a-z]+_[a-z]+_\d{4}\b")),
    ("email", re.compile(r"\S+@\S+")),
    ("price", re.compile(r"\$\s?\d|(?<![\w.])\d+\.\d{2}(?![\w.])")),
    ("case number", re.compile(r"\bcase\s*#?\d+", re.IGNORECASE)),
)


# Field values that are general vocabulary in the policy and tool schemas, not
# case-specific values (statuses, transaction and payment-source words, yes/no).
GENERAL_WORDS = frozenset({
    "yes", "no", "none", "true", "false", "pending", "delivered", "processed", "cancelled",
    "payment", "refund", "usa", "credit_card", "paypal", "gift_card", "assistant", "user"})


def _leaves(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [x for v in value.values() for x in _leaves(v)]
    if isinstance(value, list):
        return [x for v in value for x in _leaves(v)]
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, float):
        return [repr(value), f"{value:.2f}"]
    return [str(value)]


def case_values(cases: list[dict[str, Any]]) -> set[str]:
    """Every value the writer's evidence carries for these (train) cases: all
    leaves of every lookup result, every tool-call argument, the request and the
    gold, and the user's email. General vocabulary (``GENERAL_WORDS``) and empty
    strings are left out; short strings are kept (matched case-sensitively)."""
    values: set[str] = set()
    for case in cases:
        for m in case["messages"]:
            if m["role"] == "tool":
                try:
                    values.update(_leaves(json.loads(m["content"])))
                except (ValueError, TypeError):
                    values.add(m["content"])
            for call in m.get("tool_calls") or []:
                values.update(_leaves(call["arguments"]))
            if m["role"] == "user" and "@" in m["content"]:
                values.add(m["content"].split("it's ")[-1].rstrip("."))
        values.update(_leaves({k: v for k, v in case["request"].items()
                               if k not in ("keep_others", "select")}))
        if case["gold"]["type"] == "tool_call":
            values.update(_leaves(case["gold"]["arguments"]))
    return {v.strip() for v in values
            if v.strip() and v.strip().lower() not in GENERAL_WORDS
            and not v.startswith("Error:")}


def value_hits(text: str, forbidden: set[str]) -> list[str]:
    hits = []
    for v in forbidden:
        flags = 0 if len(v) <= 3 else re.IGNORECASE
        if re.search(rf"(?<![\w']){re.escape(v)}(?![\w'])", text, flags):
            hits.append(v)
    return sorted(hits)


def clean(text: str | None) -> str:
    """Whitespace and code-fence stripping only (no content change)."""
    text = (text or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    return text


def check(text: str, *, forbidden: set[str], length: tuple[int, int] | None = None
          ) -> list[str]:
    problems = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ["the procedure is empty"]
    bad = [ln for ln in lines if not re.match(r"^(?:[-*]\s*|\d+[.)]\s*)?WHEN\b.*\bTHEN\b", ln)]
    if bad:
        problems.append(f"{len(bad)} line(s) do not have the form 'WHEN ... THEN ...'")
    tokens = count_tokens(text)
    if tokens > TOKEN_LIMIT:
        problems.append(f"{tokens} tokens; the limit is {TOKEN_LIMIT}")
    if length is not None and not length[0] <= tokens <= length[1]:
        problems.append(f"{tokens} tokens; the length must be between {length[0]} and "
                        f"{length[1]}")
    for label, pattern in _ID_PATTERNS:
        if pattern.search(text):
            problems.append(f"it contains a specific {label}")
    hits = value_hits(text, forbidden)
    if hits:
        problems.append(f"it contains {len(hits)} case-specific value(s) "
                        f"(for example {hits[0]!r})")
    return problems


# --------------------------------------------------------------------------- writer


def _ask(client: AgentClient, system: str, user: str
         ) -> tuple[str | None, dict[str, int], str | None]:
    """One writer call: (text, accounting, error). A failed call keeps the
    accounting of every attempt it consumed (failed attempts included)."""
    from tau2.data_model.message import SystemMessage, UserMessage

    from slh.agent import metered_complete
    from slh.demo1.decide import _delta, _tally, b1_seed
    from slh.serving import ServingWindowClosed

    backend, meter, _ = client.parts()
    from slh.demo1.decide import _new_decision

    _new_decision(backend)  # one writer call = one decision for a growth-tracking guard
    before = _tally(meter)
    try:
        reply = metered_complete(meter, "agent", backend,
                                 [SystemMessage(role="system", content=system),
                                  UserMessage(role="user", content=user)],
                                 None, b1_seed(client.config))
    except ServingWindowClosed:
        raise
    except Exception as exc:  # noqa: BLE001 - recorded; the artifact is not valid
        return None, _delta(_tally(meter), before), f"{type(exc).__name__}: {exc}"
    return reply.content, _delta(_tally(meter), before), None


def _policy() -> str:
    from tau2.registry import registry

    return registry.get_env_constructor("retail")().get_policy()


def _write(client: AgentClient, prompt: str, *, forbidden: set[str],
           length: tuple[int, int] | None) -> dict[str, Any]:
    system = WRITER_SYSTEM.format(policy=_policy())
    raw, usage, error = _ask(client, system, prompt)
    attempts = [{"prompt_sha256": sha256_text(prompt), "raw": raw, "error": error, **usage}]
    text = clean(raw)
    problems = [f"writer call failed: {error}"] if error else \
        check(text, forbidden=forbidden, length=length)
    attempts[-1]["problems"] = problems
    if problems and not error:
        repair = REPAIR_PROMPT.format(problems="\n".join(f"- {p}" for p in problems), text=text)
        raw, usage, error = _ask(client, system, repair)
        attempts.append({"prompt_sha256": sha256_text(repair), "raw": raw, "error": error,
                         **usage, "repair": True})
        text = clean(raw)
        problems = [f"writer call failed: {error}"] if error else \
            check(text, forbidden=forbidden, length=length)
        attempts[-1]["problems"] = problems
    ok = not problems
    return {"text": text if ok else None, "sha256": sha256_text(text) if ok else None,
            "tokens": count_tokens(text) if ok else 0, "ok": ok, "problems": problems,
            "attempts": attempts, "repaired": len(attempts) > 1,
            **{k: sum(a[k] for a in attempts) for k in (
                "model_calls", "failed_attempts", "input_tokens", "output_tokens")}}


def evidence_values(cases: list[dict[str, Any]], results: list[dict[str, Any]]) -> set[str]:
    """Every value in the evidence the writer receives: the cases' values
    (``case_values``) plus the values of the agent's own train actions shown as
    evidence (tool-call arguments; ids and numbers in its messages)."""
    values = set(case_values(cases))
    ids = {c["case_id"] for c in cases}
    for r in results:
        if r["case_id"] not in ids:
            continue
        action = r["action"]
        if action.get("type") == "tool_call":
            values.update(v.strip() for v in _leaves(action.get("arguments") or {})
                          if v.strip() and v.strip().lower() not in GENERAL_WORDS)
        elif action.get("type") == "message":
            values.update(re.findall(r"#?W\d{7}|\b\d[\w.-]*", action.get("content") or ""))
    return values


def write_procedure(task: str, cases: list[dict[str, Any]], results: list[dict[str, Any]],
                    client: AgentClient) -> dict[str, Any]:
    evidence = evidence_text(cases, results)
    prompt = LEARN_PROMPT.format(n=len(cases), task=TASK_DESCRIPTION[task], evidence=evidence,
                                 limit=TOKEN_LIMIT)
    out = _write(client, prompt, forbidden=evidence_values(cases, results), length=None)
    return {"task": task, "artifact": "learned_procedure", **out,
            "evidence_sha256": sha256_text(evidence)}


def generic_band(target: int) -> tuple[int, int]:
    low = max(1, int(target * GENERIC_LENGTH_BAND[0]))
    return low, min(TOKEN_LIMIT, round(target * GENERIC_LENGTH_BAND[1]))


def write_generic(task: str, target_tokens: int, cases: list[dict[str, Any]],
                  client: AgentClient) -> dict[str, Any]:
    low, high = generic_band(target_tokens)
    prompt = GENERIC_PROMPT.format(task=TASK_DESCRIPTION[task], target=target_tokens, low=low,
                                   high=high, limit=TOKEN_LIMIT)
    out = _write(client, prompt, forbidden=case_values(cases), length=(low, high))
    return {"task": task, "artifact": "generic_note", "target_tokens": target_tokens, **out}


# --------------------------------------------------------------------------- human and owner


def human_path(task: str, root: Path = HUMAN_DIR) -> Path:
    return root / f"human-procedure-{task}.md"


def human_procedure(task: str, root: Path = HUMAN_DIR) -> str:
    """The person-written procedure (refuses the placeholder). The file's text
    after the first line '---' is the procedure, used verbatim."""
    text = human_path(task, root).read_text(encoding="utf-8")
    if PLACEHOLDER in text:
        raise ValueError(f"{human_path(task, root).name} is still the placeholder")
    _, sep, body = text.partition("\n---\n")
    if not sep or not body.strip():
        raise ValueError("the human procedure follows a line '---'")
    procedure = body.strip()
    if count_tokens(procedure) > TOKEN_LIMIT:
        raise ValueError("the human procedure exceeds the token limit")
    return procedure


def owner_record(path: Path, *, task: str, procedure_sha256: str, verdict: str,
                 reviewer: str = "owner", note: str = "") -> dict[str, Any]:
    if verdict not in ("accept", "reject"):
        raise ValueError("verdict is accept or reject (the owner does not edit)")
    row = {"event": "demo1_owner_verdict", "task": task, "procedure_sha256": procedure_sha256,
           "verdict": verdict, "reviewer": reviewer, "note": note,
           "time_utc": dt.datetime.now(dt.UTC).isoformat()}
    existing = owner_verdicts(path)
    if any(r["task"] == task for r in existing):
        raise ValueError(f"a verdict for the {task} procedure is already recorded")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def owner_verdicts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
