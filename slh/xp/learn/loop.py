"""Writer S+ / T+: a lesson library learned by small, validated edit steps.

Loop (``run_ops_learning``):

1. Run the current library on the train cases; the failures (score not
   correct) form the pool, taking the FIRST error of each trajectory
   (``first_error``: a POC 9 case is one decision, so that is its action).
2. Each step takes the next ``batch_size`` failures (seeded order, cyclic) and
   asks the writer for at most ``max_ops`` edit ops (``slh.xp.learn.ops``),
   showing the library with ids, the failures (POC 9 evidence rendering) and
   the most recent rejected edits with their reasons. An unparseable reply
   gets one repair call.
3. Gates: per-op checks (``apply_ops``); the candidate's rendered size against
   the soft ``token_budget`` (over it: every op rejected, explicitly); the
   optional binding critic (POC 9 ``critique``: policy and evidence
   contradictions); then the dev split. The candidate is kept only if its
   dev mean score exceeds the current one by more than ``min_delta``.
4. Rejected ops (gate or dev) go to the rejected-edit memory, which later
   prompts show and the op gate enforces (identical re-proposals rejected).
5. Stop after ``patience`` consecutive steps without an accepted edit, at
   ``max_steps``, when no failure is left, when a writer/critic call or an
   agent decision no longer fits its ceiling, or on a credit error.

Critic (recorded on every trace row and the summary): ``teacher`` (runs are
"teacher-checked"), ``self`` (the 9B agent) or ``none`` (mechanical checks +
dev validation only). Critic calls count against the writer's ceilings
(POC 9 ``WriterBudget``: calls and output tokens, one ``CALL_OUTPUT_RESERVE``
per call). Agent decisions (train mining + dev) have their own ceiling.

Credit errors reuse POC 9's latch (``slh.poc9.pipeline``): the gate is checked
before every call; a credit failure (HTTP 402 / credit wording) latches it and
the run halts with ``stop_reason = credit_halt``.

Outputs under ``out``: ``trace.jsonl`` (compact rows: start, one per step,
stop), ``library.json`` and ``library.md`` (final), ``summary.json``.
"""

from __future__ import annotations

import dataclasses
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from slh.demo1.learn import sha256_text
from slh.poc9.pipeline import CreditHalt, credit_failure, credit_gate_check, latch_credit
from slh.poc9.writers import (
    BUDGET_EXHAUSTED,
    TASK,
    BudgetExhausted,
    Chat,
    EvidenceBatch,
    Writer,
    WriterBudget,
)
from slh.xp.learn.library import Library
from slh.xp.learn.ops import OpsParseError, apply_ops, compact_op, op_key, parse_ops

CRITICS = ("teacher", "self", "none")
SYSTEM = ("You maintain a library of short lessons for a retail customer-service agent. "
          "The agent must follow this policy (excerpt):\n\n<policy>\n{policy}\n</policy>")
OPS_EXAMPLE = ('{"ops": [{"op": "add", "when": "<general situation>", "then": "<what to do>", '
               '"cases": ["E1"]}, {"op": "replace", "id": "L2", "when": "...", "then": "...", '
               '"cases": ["E3"]}, {"op": "remove", "id": "L4", "cases": ["E2"]}]}')
REPAIR = ("Your reply could not be read: <error>.\n\nReply again with JSON only, in exactly "
          "this form: " + OPS_EXAMPLE + ' (or {"ops": []}).\n\nYour previous reply:\n<raw>')


@dataclasses.dataclass
class LoopConfig:
    batch_size: int = 8          # failure cases per step
    max_ops: int = 3             # ops per proposal
    max_steps: int = 20
    patience: int = 5            # consecutive steps without an accepted edit
    token_budget: int = 2000     # soft cap on the rendered library (explicit rejection)
    entry_tokens: int = 80       # one lesson (when + then)
    min_delta: float = 0.0       # accept iff dev mean > current + min_delta
    rejected_shown: int = 20     # most recent rejected edits shown to the writer
    seed: int = 0
    max_decisions: int = 1500    # agent decisions (train mining + dev evaluations)
    remine_on_accept: bool = True

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Proposal:
    ops: list[Any]
    calls: list[dict[str, Any]]
    parse_error: str | None = None
    call_error: str | None = None
    budget_exhausted: bool = False


class OpsWriter(Writer):
    """The POC 9 writer abstraction (one Chat, its ``WriterBudget``, ``_call``
    reserving before and charging after every call) proposing edit ops."""

    def __init__(self, name: str, chat: Chat, role: str) -> None:
        super().__init__(chat, role)
        self.name = name

    def propose_ops(self, system: str, prompt: str) -> Proposal:
        calls: list[dict[str, Any]] = []
        try:
            raw = self._call(self.final, self.final_role, "ops", system, prompt, calls)
            if raw is None:
                return Proposal([], calls, call_error=calls[-1]["error"])
            try:
                return Proposal(parse_ops(raw), calls)
            except OpsParseError as exc:
                error = str(exc)
            raw = self._call(self.final, self.final_role, "repair", system,
                             REPAIR.replace("<error>", error).replace("<raw>", raw), calls)
            if raw is None:
                return Proposal([], calls, parse_error=error, call_error=calls[-1]["error"])
            try:
                return Proposal(parse_ops(raw), calls)
            except OpsParseError as exc:
                return Proposal([], calls, parse_error=str(exc))
        except BudgetExhausted:
            return Proposal([], calls, budget_exhausted=True)


def make_ops_writer(name: str, *, agent_client: Any = None, teacher: Chat | None = None,
                    agent: Chat | None = None, budget: WriterBudget | None = None
                    ) -> OpsWriter:
    """S+ (the 9B agent writes) or T+ (the teacher writes), built like POC 9's
    ``make_writer``."""
    from slh.poc9.writers import TeacherChat, agent_chat, load_teacher

    if name == "T+":
        writer = OpsWriter(name, teacher or TeacherChat(load_teacher()), "teacher")
    elif name == "S+":
        writer = OpsWriter(name, agent or agent_chat(agent_client), "agent")
    else:
        raise ValueError("writer is S+ or T+")
    if budget is not None:
        writer.budget = budget
    return writer


def resolve_critic(critic: str, *, teacher: Chat | None = None, agent: Chat | None = None
                   ) -> Chat | None:
    """The critic chat for ``critic`` in CRITICS (``self`` is the 9B agent)."""
    if critic not in CRITICS:
        raise ValueError(f"critic is one of {CRITICS}")
    if critic == "none":
        return None
    chat = teacher if critic == "teacher" else agent
    if chat is None:
        raise ValueError(f"critic {critic!r} needs its chat client")
    return chat


CRITIC_MODES = ("binding", "advisory")


def check_label(critic: str, mode: str = "binding") -> str:
    """``advisory`` (owner decision 2026-10-12): the critic is called and logged but
    never blocks an edit; mechanical checks, dev validation and sentinel selection
    decide."""
    if critic == "none":
        return "unchecked"
    if mode == "advisory":
        return f"advisory-{critic}"
    return {"teacher": "teacher-checked", "self": "self-checked"}[critic]


def first_error(episode: dict[str, Any]) -> dict[str, Any]:
    """The first wrong step of a trajectory. With ``steps`` (dicts with
    ``action``/``score``/``correction``), the first whose score is not correct;
    otherwise the episode itself (one decision per POC 9 case)."""
    for s in episode.get("steps") or []:
        if not (s.get("score") or {}).get("correct"):
            return {**episode, **{k: s[k] for k in ("action", "score", "correction", "reads")
                                  if k in s}}
    return episode


class _Batch:
    """Duck type for POC 9 ``critique``: ``text()`` and ``policy``."""

    def __init__(self, text: str, policy: str) -> None:
        self._text, self.policy = text, policy

    def text(self) -> str:
        return self._text


class _Halt(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason, self.detail = reason, detail


def _credit(message: str | None) -> bool:
    return bool(message) and (message.startswith("CreditHalt") or credit_failure(message))


def build_prompt(library: Library, evidence: str, n: int, rejected: list[dict[str, Any]],
                 config: LoopConfig, task_text: str = TASK) -> str:
    shown = rejected[-config.rejected_shown:] if config.rejected_shown else []
    rej = "\n".join(f"- step {r['step']}: {json.dumps(r['op'], sort_keys=True)} -> rejected: "
                    f"{'; '.join(r['reasons'])}" for r in shown) or "(none)"
    return (
        f"Task of the lessons: {task_text}.\n\n"
        f"Current library ({len(library.entries)} lessons, {library.tokens()} of about "
        f"{config.token_budget} tokens):\n{library.render_for_writer()}\n\n"
        f"Failure cases: {n} decisions the agent got wrong with the current library. For "
        f"each you see the conversation, the agent's first wrong action, the deterministic "
        f"outcome and the user's correction, if any.\n\n{evidence}\n\n"
        f"Edits already tried and rejected (do not propose them again):\n{rej}\n\n"
        f"Propose at most {config.max_ops} edits to the library that would fix these "
        f"failures on new cases of this task. A lesson is one short rule: WHEN <general "
        f"situation> THEN <what to do>, at most {config.entry_tokens} tokens, tied to the "
        f"failure case(s) it comes from (their labels in 'cases'). Replace or remove a "
        f"lesson that caused a failure. Use only what the policy and the failures support; "
        f"do not mention specific order ids, item ids, user ids, emails, names, addresses, "
        f"cities, product names, prices, case labels or lesson ids inside 'when' or 'then'.\n\n"
        f"Reply with JSON only: {OPS_EXAMPLE}\nReply {{\"ops\": []}} if no edit would help.")


def _default_scoring() -> tuple[Callable, Callable]:
    from slh.poc9 import corrections, score

    return score.score, corrections.explain


def _default_forbidden(episodes: list[dict[str, Any]], policy: str) -> set[str]:
    return EvidenceBatch(episodes, policy).forbidden()


def _default_render(episodes: list[dict[str, Any]]) -> str:
    from slh.poc9.writers import render_evidence

    return render_evidence(episodes)


def _usage(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: c.get(k) for k in ("role", "purpose", "raw", "error", "model_calls",
                                   "input_tokens", "output_tokens") if k in c} for c in calls]


def run_ops_learning(writer: OpsWriter, *, train: list[dict[str, Any]],
                     dev: list[dict[str, Any]],
                     decide: Callable[[dict[str, Any], str], dict[str, Any]],
                     out: Path, critic: str = "none", critic_chat: Chat | None = None,
                     policy: str | None = None, config: LoopConfig | None = None,
                     score_fn: Callable | None = None, correct_fn: Callable | None = None,
                     render_fn: Callable[[list[dict[str, Any]]], str] | None = None,
                     forbidden_fn: Callable[[list[dict[str, Any]], str], set[str]] | None = None,
                     initial: Library | None = None,
                     task_text: str | None = None,
                     critic_mode: str = "binding") -> dict[str, Any]:
    """One S+/T+ run. ``decide(case, library_text)`` returns a decision dict
    (``action``, ``reads``, ``model_calls``), normally ``slh.poc9.loader.decide``."""
    config = config or LoopConfig()
    if critic_mode not in CRITIC_MODES:
        raise ValueError(f"critic mode is one of {CRITIC_MODES}")
    if critic not in CRITICS:
        raise ValueError(f"critic is one of {CRITICS}")
    if (critic == "none") != (critic_chat is None):
        raise ValueError("critic 'none' takes no chat; 'teacher'/'self' need one")
    if not train or not dev:
        raise ValueError("train and dev cases are required")
    if {c["case_id"] for c in train} & {c["case_id"] for c in dev}:
        raise ValueError("train and dev overlap")
    if score_fn is None or correct_fn is None:
        ds, dc = _default_scoring()
        score_fn, correct_fn = score_fn or ds, correct_fn or dc
    if policy is None:
        from slh.poc9.writers import policy_excerpt
        policy = policy_excerpt()
    render_fn = render_fn or _default_render
    forbidden_fn = forbidden_fn or _default_forbidden
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    base = {"writer": writer.name, "critic": critic, "check": check_label(critic, critic_mode),
            "critic_mode": critic_mode}
    would_block = 0

    def emit(row: dict[str, Any]) -> None:
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**base, **row}, sort_keys=True, separators=(",", ":"),
                                    default=str) + "\n")

    state = {"decisions": 0, "agent_model_calls": 0}

    def evaluate(cases: list[dict[str, Any]], text: str) -> tuple[list[dict[str, Any]], float]:
        if state["decisions"] + len(cases) > config.max_decisions:
            raise _Halt("decision_budget", f"{len(cases)} more decisions do not fit "
                        f"{config.max_decisions} (used {state['decisions']})")
        eps = []
        for case in cases:
            credit_gate_check()
            error = None
            try:
                decision = decide(case, text)
            except CreditHalt:
                raise
            except Exception as exc:  # noqa: BLE001 - recorded; the case scores as an error
                error = f"{type(exc).__name__}: {exc}"
                if _credit(error):
                    latch_credit(error)
                    raise _Halt("credit_halt", error) from None
                decision = {"action": {"type": "none", "reason": "error"}, "reads": []}
            state["decisions"] += 1
            state["agent_model_calls"] += decision.get("model_calls", 0)
            action = decision["action"]
            eps.append({"case": case, "action": action, "reads": decision.get("reads") or [],
                        "score": score_fn(case, action), "correction": correct_fn(case, action),
                        "error": error, **({"steps": decision["steps"]}
                                           if decision.get("steps") else {})})
        mean = sum(bool(e["score"].get("correct")) for e in eps) / len(eps)
        return eps, mean

    library = initial.copy() if initial else Library()
    rejected: list[dict[str, Any]] = []
    rejected_keys: set[tuple[str, str, str, str]] = set()
    rng = random.Random(config.seed)
    steps_run = accepted = since_accept = 0
    stop_reason, stop_detail = "max_steps", None
    dev_curve: list[dict[str, Any]] = []
    pool: list[dict[str, Any]] = []
    cursor = 0

    def mine() -> None:
        nonlocal pool, cursor
        eps, _ = evaluate(train, library.render())
        pool = [first_error(e) for e in eps if not e["score"].get("correct")]
        rng.shuffle(pool)
        cursor = 0

    cur_dev = None
    try:
        mine()
        _, cur_dev = evaluate(dev, library.render())
        dev_curve.append({"step": 0, "dev": cur_dev, "library_sha256": library.sha256()})
        emit({"type": "start", "config": config.to_dict(), "library_sha256": library.sha256(),
              "library_tokens": library.tokens(), "entries": len(library.entries),
              "train": len(train), "dev": len(dev), "train_failures": len(pool),
              "dev_score": cur_dev, "writer_budget": writer.budget.to_dict()})
        for step in range(1, config.max_steps + 1):
            if not pool:
                stop_reason = "no_failures"
                break
            take = min(config.batch_size, len(pool))
            batch = [pool[(cursor + i) % len(pool)] for i in range(take)]
            cursor = (cursor + take) % len(pool)
            episodes = [{**e, "index": i} for i, e in enumerate(batch, 1)]
            labels = {f"E{e['index']}": e["case"]["case_id"] for e in episodes}
            evidence = render_fn(episodes)
            prompt = build_prompt(library, evidence, len(episodes), rejected, config,
                                  task_text or TASK)
            credit_gate_check()
            proposal = writer.propose_ops(SYSTEM.format(policy=policy), prompt)
            steps_run += 1
            row: dict[str, Any] = {
                "type": "step", "step": step, "batch": [labels[k] for k in sorted(
                    labels, key=lambda x: int(x[1:]))],
                "prompt_sha256": sha256_text(prompt), "calls": _usage(proposal.calls),
                "parse_error": proposal.parse_error, "call_error": proposal.call_error,
                "ops": [compact_op(o) for o in proposal.ops], "verdicts": [],
                "budget_status": None, "critique": None, "dev_before": cur_dev,
                "dev_after": None, "accepted": False}
            if proposal.call_error and _credit(proposal.call_error):
                latch_credit(proposal.call_error)
                emit({**row, "library_sha256": library.sha256()})
                raise _Halt("credit_halt", proposal.call_error)
            if proposal.budget_exhausted:
                emit({**row, "problems": [BUDGET_EXHAUSTED], "library_sha256": library.sha256()})
                raise _Halt("writer_budget", BUDGET_EXHAUSTED)
            forbidden = forbidden_fn(episodes, policy)
            cand, verdicts = apply_ops(library, proposal.ops, labels=labels, step=step,
                                       forbidden=forbidden, max_ops=config.max_ops,
                                       entry_tokens=config.entry_tokens,
                                       rejected_keys=rejected_keys)
            row["verdicts"] = verdicts
            valid = [v for v in verdicts if v["ok"]]
            group_reason = None
            if valid and cand.tokens() > config.token_budget:
                group_reason = (f"over budget: the library would measure {cand.tokens()} "
                                f"tokens; the budget is {config.token_budget}")
                row["budget_status"] = group_reason
            if valid and group_reason is None and critic_chat is not None:
                if not writer.budget.can_call():
                    emit({**row, "problems": [BUDGET_EXHAUSTED],
                          "library_sha256": library.sha256()})
                    raise _Halt("writer_budget", f"{BUDGET_EXHAUSTED} (critic)")
                credit_gate_check()
                from slh.poc9.learn import critique
                crit = critique(critic_chat, cand.render(), _Batch(evidence, policy))
                writer.budget.charge(crit["call"])
                crit["call"]["role"] = critic
                row["critique"] = {"ok": crit["ok"], "binding": crit["binding"],
                                   "dismissed": crit["dismissed"], "problem": crit["problem"],
                                   "call": _usage([crit["call"]])[0]}
                if crit["call"].get("error") and _credit(crit["call"]["error"]):
                    latch_credit(crit["call"]["error"])
                    emit({**row, "library_sha256": library.sha256()})
                    raise _Halt("credit_halt", crit["call"]["error"])
                if not crit["ok"]:
                    if critic_mode == "advisory":  # logged, never blocking
                        row["critique"]["would_have_blocked"] = True
                        would_block += 1
                    else:
                        group_reason = f"critic ({critic}): {crit['problem']}"
                else:
                    row["critique"]["would_have_blocked"] = False
            if valid and group_reason is None:
                _, cand_dev = evaluate(dev, cand.render())
                row["dev_after"] = cand_dev
                if cand_dev > cur_dev + config.min_delta:
                    row["accepted"] = True
                else:
                    group_reason = (f"dev score did not rise ({cur_dev:.4f} -> "
                                    f"{cand_dev:.4f})")
            for v in verdicts:
                reasons = v["problems"] if not v["ok"] else (
                    [group_reason] if group_reason else [])
                if reasons and isinstance(v["op"], dict):
                    rejected.append({"step": step, "op": v["op"], "reasons": reasons})
                    if v["ok"] and row["budget_status"] is None:
                        # A valid edit that the critic or the dev split rejected:
                        # an identical re-proposal is rejected by the op gate.
                        rejected_keys.add(op_key(v["op"]))
            if proposal.parse_error and not proposal.ops:
                rejected.append({"step": step, "op": "(unreadable reply)",
                                 "reasons": [proposal.parse_error]})
            if row["accepted"]:
                library, cur_dev = cand, row["dev_after"]
                accepted += 1
                since_accept = 0
                dev_curve.append({"step": step, "dev": cur_dev,
                                  "library_sha256": library.sha256()})
            else:
                since_accept += 1
            row.update(group_reason=group_reason, library_sha256=library.sha256(),
                       library_tokens=library.tokens(), entries=len(library.entries),
                       writer_budget=writer.budget.to_dict(), decisions=state["decisions"])
            emit(row)
            if row["accepted"] and config.remine_on_accept:
                mine()
            if since_accept >= config.patience:
                stop_reason = "patience"
                break
    except _Halt as halt:
        stop_reason, stop_detail = halt.reason, halt.detail
    except CreditHalt as exc:
        stop_reason, stop_detail = "credit_halt", str(exc)

    summary = {**base, "stop_reason": stop_reason, "stop_detail": stop_detail,
               "steps": steps_run, "accepted": accepted, "dev_score": cur_dev,
               "dev_curve": dev_curve, "library_sha256": library.sha256(),
               "library_state_sha256": library.state_sha256(),
               "library_tokens": library.tokens(), "entries": len(library.entries),
               "rejected_edits": len(rejected), "writer_budget": writer.budget.to_dict(),
               "critic_would_have_blocked": would_block,
               "decisions": state["decisions"], "agent_model_calls": state["agent_model_calls"],
               "config": config.to_dict()}
    emit({"type": "stop", **{k: summary[k] for k in (
        "stop_reason", "stop_detail", "steps", "accepted", "dev_score", "library_sha256",
        "library_tokens", "entries", "decisions")}, "library": library.to_dict()})
    (out / "library.json").write_text(json.dumps(library.to_dict(), indent=1, sort_keys=True)
                                      + "\n", encoding="utf-8")
    (out / "library.md").write_text(library.render() + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                      encoding="utf-8")
    return summary


def run(writer_name: str, out: Path, *, critic: str, client: Any = None,
        config: LoopConfig | None = None) -> dict[str, Any]:
    """Wire the real pieces (POC 9 train stream, validation split as dev, the
    capped 9B client, the configured teacher only when T+ or critic=teacher)."""
    from slh.demo1.decide import AgentClient
    from slh.poc9 import cases as poc9_cases
    from slh.poc9 import loader
    from slh.poc9.pipeline import _new_credit_generation
    from slh.poc9.writers import TeacherChat, agent_chat, load_teacher

    _new_credit_generation()  # this run's own credit latch (POC 9: one per stage)
    client = client or AgentClient.for_window()
    agent = agent_chat(client)
    teacher = TeacherChat(load_teacher()) if critic == "teacher" or writer_name == "T+" \
        else None
    writer = make_ops_writer(writer_name, teacher=teacher, agent=agent)
    return run_ops_learning(writer, train=poc9_cases.load("train"),
                            dev=poc9_cases.load("validation"),
                            decide=lambda case, text: loader.decide(case, text, client),
                            out=out, critic=critic,
                            critic_chat=resolve_critic(critic, teacher=teacher, agent=agent),
                            config=config)
