"""The partner's learning stage (docs/xp/partner-package.md; plan v1.1 writers and controls).

Per task x run x writer, ``PartnerLearner.learn`` writes into ``out``:

- ``skill.md``: the learned text. S+/T+: the lesson library of
  ``slh.xp.learn.loop.run_ops_learning`` (critic per plan: teacher, self or
  none). S/T: POC 9's whole-note loop (``run_note_learning``: one patch per 4
  episodes, mechanical checks with one repair, binding critique when a critic
  is set), generalised to every task (task text, task-aware evidence, XP
  scoring and scripted corrections, ``slh.xp.partner.corrections``).
- ``search-control.md``: the same writer and loop with the same proposal
  opportunities and selection, but no evidence (``NO_EVIDENCE`` in place of
  the episodes).
- ``generic.md``: the same writer with no evidence, length-matched to the
  learned text (POC 9 ``generic_note``).
- ``evidence.md``: full context: every training case with B1's action, the
  deterministic outcome and the correction (the writers' evidence, uncompressed).
- ``trace.jsonl`` / ``patches.jsonl`` / ``episodes.jsonl`` / ``search_trace.jsonl``
  and ``summary.json``.

Chats are ``(system, user, purpose=...) -> (text, usage, error)`` (POC 9
``Chat``). A halt-class error (``slh.xp.partner.errors``) in any chat or agent
decision raises ``RunHalted``; the half-written directory is removed so the
learning run starts again on resume. ``FakeChat`` makes the whole stage run
offline (dry run, tests).
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from slh.demo1.learn import count_tokens, sha256_text
from slh.poc9 import writers as w9
from slh.poc9.pipeline import CreditHalt
from slh.poc9.writers import EvidenceBatch, Writer, WriterBudget
from slh.xp.partner import common as pc
from slh.xp.partner.errors import RunHalted, classify

TASK_TEXT = {
    "address": w9.TASK,
    "variant": ("choosing the exact replacement variant when a user asks to exchange or "
                "modify an item and describes the new one by its attributes: write the change "
                "when exactly one available variant matches every attribute and the user has "
                "approved it, ask for confirmation naming that variant when it is not yet "
                "approved, and ask the user when no variant or more than one matches"),
    "sequence": ("handling a user who wants both the address and an item of one pending order "
                 "changed: the address change must come first (an item change locks the "
                 "order), each change needs the user's explicit yes to its listed details, "
                 "write the next approved change, and ask for confirmation of the next change "
                 "when it is not yet approved"),
}
NO_EVIDENCE = ("(Search control: no episodes are shown at this step. Improve the text from the "
               "task description and the policy only.)")
WRITERS = ("S", "T", "S+", "T+")


# --------------------------------------------------------------------------- evidence


def render_task_evidence(episodes: list[dict[str, Any]]) -> str:
    """POC 9's evidence rendering with each case's own task for the lookup facts."""
    from slh.demo1.learn import _facts

    blocks = []
    for e in episodes:
        case = e["case"]
        turns = [f"  {m['role']}: {m['content']}" for m in case["messages"]
                 if m["role"] in ("user", "assistant") and m.get("content")]
        errors = [m["content"] for m in case["messages"] if m["role"] == "tool" and m.get("error")]
        errors += [f"{r['name']} failed" for r in e.get("reads") or [] if r.get("error")]
        try:
            facts = _facts({**case, "task": pc.task_of(case)})
        except (KeyError, TypeError):
            facts = []
        corr, score = e.get("correction"), e["score"]
        blocks.append("\n".join([
            f"Episode E{e['index']}", "Conversation:", *turns,
            "Lookup facts:", *[f"  {x}" for x in facts],
            f"Agent's action: {w9.render_action(e['action'])}",
            (f"Outcome: {score.get('outcome')} "
             f"({'correct' if score.get('correct') else 'not correct'})"),
            f"Lookup errors: {'; '.join(errors) if errors else 'none'}",
            f"User's correction: {corr['text'] if corr else 'none'}"]))
    return "\n\n".join(blocks)


class TaskEvidence(EvidenceBatch):
    def text(self) -> str:
        return render_task_evidence(self.episodes)


class NoEvidence(EvidenceBatch):
    def text(self) -> str:
        return NO_EVIDENCE


def _task_prompt(prompt: str, task: str) -> str:
    if task == "address":
        return prompt
    return prompt.replace("when to ask which address the user means",
                          "when to ask the user what they mean")


class TaskNoteWriter(Writer):
    """POC 9's whole-note writer (T or S) with the task's own description."""

    def __init__(self, name: str, chat: Any, role: str, task: str) -> None:
        super().__init__(chat, role)
        self.name, self.task = name, task

    def propose_patch(self, current_skill: str, evidence_batch: EvidenceBatch):
        system = w9.SYSTEM.format(policy=evidence_batch.policy)
        prompt = _task_prompt(w9.PATCH_PROMPT.format(
            task=TASK_TEXT[self.task], skill=current_skill or "(empty)",
            n=len(evidence_batch.episodes), evidence=evidence_batch.text(),
            **w9.band_args(w9.SKILL_TOKENS)), self.task)
        return self._guarded(system, prompt, [], forbidden=evidence_batch.forbidden(),
                             band=w9.SKILL_TOKENS, indices=evidence_batch.indices())

    def generic_note(self, policy: str, target_tokens: int | None = None,
                     forbidden: set[str] | None = None):
        band = w9.generic_band(target_tokens)
        prompt = _task_prompt(w9.GENERIC_PROMPT.format(task=TASK_TEXT[self.task],
                                                       **w9.band_args(band)), self.task)
        return self._guarded(w9.SYSTEM.format(policy=policy), prompt, [],
                             forbidden=forbidden or set(), band=band, indices=None)


# --------------------------------------------------------------------------- note loop


def run_note_learning(writer: TaskNoteWriter, *, train: list[dict[str, Any]],
                      decide: Callable[[dict[str, Any], str], dict[str, Any]],
                      critic: Any, out: Path, policy: str, score_fn: Callable,
                      correct_fn: Callable, can_decide: Callable[[], bool],
                      batch: int = 4, evidence_cls: type = TaskEvidence, prefix: str = ""
                      ) -> dict[str, Any]:
    """POC 9's loop over a run's training stream (the registered sample, in its
    order). Stops early when the decision budget is spent (``can_decide``)."""
    from slh.poc9.learn import gate

    out = Path(out)
    for name in ("episodes", "patches"):
        (out / f"{prefix}{name}.jsonl").write_text("", encoding="utf-8")
    skill, pending, snapshots = "", [], [""]
    patches = accepted = episodes = 0
    stop = "end"
    for n, case in enumerate(train, 1):
        if not can_decide():
            stop = "decision_budget"
            break
        error = None
        try:
            decision = decide(case, skill)
        except CreditHalt:  # RunHalted included: a halt never becomes an episode
            raise
        except Exception as exc:  # noqa: BLE001 - a case-level error; the episode counts
            error = f"{type(exc).__name__}: {exc}"
            decision = {"action": {"type": "none", "reason": "error"}, "reads": []}
        episodes += 1
        score = score_fn(case, decision["action"])
        correction = correct_fn(case, decision["action"])
        pc.append_jsonl(out / f"{prefix}episodes.jsonl", [{
            "case_id": case["case_id"], "stream_position": n, "action": decision["action"],
            "score": score, "correction": correction, "error": error,
            "skill_sha256": sha256_text(skill) if skill else None}])
        pending.append({"index": n, "case": case, "action": decision["action"],
                        "reads": decision.get("reads") or [], "score": score,
                        "correction": correction})
        if n % batch and n != len(train):
            continue
        evidence = evidence_cls(pending, policy)
        patch = writer.propose_patch(skill, evidence)
        if critic is None:
            ok = patch.ok and patch.changed and not w9.check_skill(
                patch.text, forbidden=evidence.forbidden(), band=w9.SKILL_TOKENS,
                provenance=patch.provenance, indices=evidence.indices())
            verdict = {"accepted": bool(ok), "stage": "checks", "problems": patch.problems,
                       "critique": None}
        else:
            verdict = gate(patch, evidence, critic, writer.budget)
        patches += 1
        if verdict["accepted"]:
            accepted += 1
            skill = patch.text
            snapshots.append(skill)
        pc.append_jsonl(out / f"{prefix}patches.jsonl", [{
            "after_episode": n, "writer": writer.name, "episodes": sorted(evidence.indices()),
            "evidence_sha256": sha256_text(evidence.text()), "accepted": verdict["accepted"],
            "stage": verdict["stage"], "problems": verdict["problems"],
            "critique": {k: v for k, v in (verdict["critique"] or {}).items() if k != "call"}
            or None, "patch": patch.to_dict(), "writer_budget": writer.budget.to_dict(),
            "skill_sha256_after": sha256_text(skill) if skill else None}])
        pending = []
    return {"skill": skill, "snapshots": snapshots, "proposals": patches,
            "accepted": accepted, "episodes": episodes, "stop_reason": stop}


# --------------------------------------------------------------------------- chats


class FakeChat:
    """Offline stand-in for the 9B agent or the teacher as a writer/critic:
    valid ops replies, a valid five-section skill with provenance, empty critiques."""

    LESSONS = ("the user describes the change but has not said yes to the exact details",
               "more than one option fits what the user described",
               "nothing fits what the user described",
               "the user already said yes to the exact change after it was listed",
               "two changes are requested on one pending order")
    THEN = ("list the exact change and ask for an explicit yes before writing",
            "ask the user which option they mean and change nothing yet",
            "tell the user nothing fits and ask what to do",
            "make the approved change without asking again",
            "do the address change first and the item change after it")
    SKILL = "\n".join([
        "## Trigger",
        ("A user asks to change a pending or delivered order and describes the wanted "
         "result in words instead of giving every detail directly."),
        "## Steps", *[f"- {t[0].upper() + t[1:]}." for t in THEN],
        ("- Read the lookup results carefully and compare every requested attribute with "
         "every candidate before deciding what to do next."),
        ("- Restate the full details of the change in one message so the user can approve "
         "it with a single explicit yes."),
        "## Pitfalls",
        ("- Do not guess between candidates that fit equally well, and do not treat a vague "
         "agreement as approval of details that were never listed."),
        ("- Do not ask again when the user has already approved the listed details; asking "
         "twice wastes the user's time and is a mistake."),
        "- Do not change anything when the description fits nothing; explain and ask.",
        "## Verification",
        ("- Before any write, check that exactly one candidate fits every requested "
         "attribute and that the user said yes after the details were listed."),
        "- After a write, confirm what changed and what remains to be done.",
        "## Exceptions",
        ("- When the user asks for a human agent only because of a policy limit, follow the "
         "transfer rule of the policy instead of this procedure."),
        ("- When the order status does not allow the change, explain the limit and ask the "
         "user how to proceed.")])

    def __init__(self, *, malformed: bool = False, max_dispatches: int = 1) -> None:
        self.calls = 0
        self.malformed = malformed
        self.max_dispatches = max_dispatches

    def __call__(self, system: str, user: str, purpose: str = "write"):
        self.calls += 1
        usage = {"model_calls": 1, "failed_attempts": 0, "input_tokens": len(user) // 4,
                 "output_tokens": 4096 if self.malformed else 60}
        if self.malformed:
            return "not a valid reply", usage, None
        if purpose == "critique":
            return '{"contradictions": []}', usage, None
        if system.startswith("You maintain a library"):
            labels = re.findall(r"\bE(\d+)\b", user)
            i = self.calls % len(self.LESSONS)
            op = {"op": "add", "when": self.LESSONS[i], "then": self.THEN[i],
                  "cases": [f"E{labels[0] if labels else 1}"]}
            return json.dumps({"ops": [op]}), usage, None
        cited = re.findall(r"Episode E(\d+)", user)
        prov = f"\n<provenance>\n- E{cited[0]}: the agent needed this procedure\n</provenance>" \
            if cited else ""
        return f"<skill>\n{self.SKILL}\n</skill>{prov}", usage, None


class HaltingChat:
    """Raises ``RunHalted`` on a halt-class error (spend cap, credit, auth,
    connection, configuration) and refuses every call once the run is halted;
    case-level errors pass through as POC 9 errors."""

    def __init__(self, chat: Any, learner: PartnerLearner) -> None:
        self.chat, self.learner = chat, learner
        # Transport dispatches one call can make (1 + retries + meter pauses); the
        # writer budget reserves this many before every call.
        self.max_dispatches = max(1, int(getattr(chat, "max_dispatches", 1)))

    def __call__(self, system: str, user: str, purpose: str = "write"):
        self.learner.check_halt()
        try:
            text, usage, error = self.chat(system, user, purpose=purpose)
        except CreditHalt as exc:  # RunHalted included: latch, then stop
            self.learner.halt(f"{type(exc).__name__}: {exc}")
            raise RunHalted("chat", f"{type(exc).__name__}: {exc}") from exc
        except Exception as exc:
            if classify(exc)[0] == "halt":
                self.learner.halt(f"{type(exc).__name__}: {exc}")
                raise RunHalted("chat", f"{type(exc).__name__}: {exc}") from exc
            text, usage, error = None, {"model_calls": 0, "failed_attempts": 1,
                                        "input_tokens": 0, "output_tokens": 0}, \
                f"{type(exc).__name__}: {exc}"
        if error and classify(error)[0] == "halt":
            self.learner.halt(error)
            raise RunHalted("chat", error)
        return text, usage, error


def guard_backend(backend: Any) -> Any:
    """Check the process-wide halt latch (POC 9's credit gate) before EVERY
    request attempt of this backend (each retry included) and latch it on a
    halt-class failure, so no request leaves after any halt."""
    from slh.poc9.pipeline import credit_gate_check, latch_credit

    if getattr(backend, "_xp_guarded", False):
        return backend
    complete = backend.complete

    def guarded(*args: Any, **kwargs: Any) -> Any:
        credit_gate_check()
        try:
            return complete(*args, **kwargs)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if classify(message)[0] == "halt":
                latch_credit(message)
            raise

    backend.complete = guarded
    backend._xp_guarded = True
    return backend


# Request extras a partner may set for the self-hosted teacher: only the switch to
# non-thinking mode. Decoding settings (max_tokens, temperature, retries, seed,
# top_p, ...) are fixed here and cannot be overridden.
TEACHER_CFG_KEYS = frozenset({"mode", "base_url", "model", "api_key_env", "extra", "weights"})
TEACHER_EXTRA_KEYS = frozenset({"chat_template_kwargs"})


def teacher_extras(extra: Any) -> dict[str, Any]:
    extra = dict(extra or {})
    if set(extra) - TEACHER_EXTRA_KEYS:
        raise ValueError(f"teacher extras may only set {sorted(TEACHER_EXTRA_KEYS)}, not "
                         f"{sorted(set(extra) - TEACHER_EXTRA_KEYS)}")
    ctk = extra.get("chat_template_kwargs")
    if ctk is not None and (not isinstance(ctk, dict) or set(ctk) - {"enable_thinking"}
                            or not isinstance(ctk.get("enable_thinking", False), bool)):
        raise ValueError("chat_template_kwargs may only hold enable_thinking (bool)")
    return extra


class SelfHostedChat:
    """The teacher on the partner's own OpenAI-compatible endpoint ($0 per token;
    temperature 0, the plan's output cap, 2 retries, non-thinking extras only).
    Built like POC 9's ``TeacherChat``; not exercised against a live endpoint here."""

    def __init__(self, cfg: dict[str, Any], *, max_tokens: int = 4096) -> None:
        from slh.budget import Meter, ModelConfig, Price
        from slh.openai_compat import OpenAICompatibleBackend

        if set(cfg) - TEACHER_CFG_KEYS:
            raise ValueError(f"unknown teacher settings {sorted(set(cfg) - TEACHER_CFG_KEYS)}")
        fixed = {"backend": "openai_compatible", "model": cfg["model"], "temperature": 0.0,
                 "max_tokens": int(max_tokens), "reasoning_reserve_tokens": 0, "retries": 2,
                 "capture": "messages", "api_base": cfg["base_url"],
                 "api_key_env": cfg.get("api_key_env")}
        self.model = ModelConfig.from_dict({**teacher_extras(cfg.get("extra")), **fixed})
        body = self.model.extra_params()
        if set(body) - TEACHER_EXTRA_KEYS or self.model.max_tokens != int(max_tokens) or \
                self.model.temperature != 0.0 or self.model.retries != 2:
            raise ValueError("teacher decoding settings must stay fixed")
        # No pause-and-retry beyond the configured retries: at most retries + 1
        # transport dispatches per call (charged against the writer's ceiling).
        self.meter = Meter(1.0, {self.model.model: Price(input=0.0, output=0.0,
                                                         cached_input=0.0)}, max_pauses=0)
        self.backend = guard_backend(OpenAICompatibleBackend(self.model, "teacher"))
        self.max_dispatches = self.model.retries + 1

    def __call__(self, system: str, user: str, purpose: str = "write"):
        from tau2.data_model.message import SystemMessage, UserMessage

        from slh.agent import metered_complete

        t = self.meter.tally["agent"]
        before = (t.calls, t.failed_attempts, t.input_tokens, t.output_tokens)
        error = None
        try:
            reply = metered_complete(self.meter, "agent", self.backend,
                                     [SystemMessage(role="system", content=system),
                                      UserMessage(role="user", content=user)], None, None)
            text = reply.content
        except Exception as exc:  # noqa: BLE001 - classified by HaltingChat
            text, error = None, f"{type(exc).__name__}: {exc}"
        t = self.meter.tally["agent"]
        after = (t.calls, t.failed_attempts, t.input_tokens, t.output_tokens)
        keys = ("model_calls", "failed_attempts", "input_tokens", "output_tokens")
        return text, {k: a - b for k, a, b in zip(keys, after, before, strict=True)}, error


# --------------------------------------------------------------------------- budgets


@dataclasses.dataclass
class DispatchBudget(WriterBudget):
    """POC 9's writer budget counted in transport dispatches: every call reserves
    its worst case (``per_call`` dispatches) and is charged what it actually
    dispatched (successful + failed attempts), so retries never pass the ceiling."""

    per_call: int = 1

    def can_call(self) -> bool:
        return self.calls + self.per_call <= self.call_ceiling and \
            self.output_tokens + w9.CALL_OUTPUT_RESERVE <= self.output_token_ceiling

    def charge(self, usage: dict[str, int]) -> None:
        sent = int(usage.get("model_calls", 0) or 0) + int(usage.get("failed_attempts", 0) or 0)
        self.calls += max(1, sent)
        self.output_tokens += int(usage.get("output_tokens", 0) or 0)


@dataclasses.dataclass
class Component:
    """One registered budget of a learning run (``plan.learning.budgets``)."""

    name: str
    calls: int
    output_tokens: int
    decisions: int
    used_decisions: int = 0
    writer: Any = None

    def writer_budget(self, per_call: int = 1) -> Any:
        self.writer = DispatchBudget(call_ceiling=self.calls,
                                     output_token_ceiling=self.output_tokens,
                                     per_call=max(1, per_call))
        return self.writer

    def can_decide(self) -> bool:
        return self.used_decisions < self.decisions

    def take(self) -> None:
        if not self.can_decide():
            raise RuntimeError(f"{self.name}: decision budget of {self.decisions} exceeded")
        self.used_decisions += 1

    def report(self) -> dict[str, int]:
        w = self.writer
        return {"calls": w.calls if w else 0, "output_tokens": w.output_tokens if w else 0,
                "decisions": self.used_decisions, "limit_calls": self.calls,
                "limit_output_tokens": self.output_tokens, "limit_decisions": self.decisions}


@dataclasses.dataclass
class LearnSettings:
    ops: dict[str, Any] = dataclasses.field(default_factory=dict)  # LoopConfig fields
    note_batch: int = 4
    budgets: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_plan(cls, plan: dict[str, Any]) -> LearnSettings:
        from slh.xp.partner.plan import check_budgets, default_budgets

        learning = plan.get("learning") or {}
        budgets = learning.get("budgets") or default_budgets()
        check_budgets(budgets)
        return cls(ops=dict(learning.get("ops") or {}),
                   note_batch=int(learning.get("note_batch", 4)), budgets=budgets)

    def components(self) -> dict[str, Component]:
        from slh.xp.partner.plan import BUDGET_COMPONENTS

        return {name: Component(name, int(self.budgets[name]["calls"]),
                                int(self.budgets[name]["output_tokens"]),
                                int(self.budgets[name]["decisions"]))
                for name in BUDGET_COMPONENTS}


# --------------------------------------------------------------------------- selection


def select_on_sentinel(snapshots: list[str], sentinel: list[dict[str, Any]], *,
                       evaluate: Callable[[str], float | None]) -> tuple[str, dict[str, Any]]:
    """Final selection (plan v1.1: a separate sentinel set). Snapshots run from the
    empty start to the endpoint. Keep the endpoint only if it beats its
    predecessor on the sentinel split; otherwise revert to the best-on-sentinel
    snapshot (ties: the earlier one). ``evaluate`` returns None when the
    selection budget cannot cover a snapshot."""
    seq: list[str] = []
    for s in snapshots:
        if not seq or seq[-1] != s:
            seq.append(s)
    end = seq[-1]
    record: dict[str, Any] = {"applied": False, "reason": None,
                              "snapshots": [sha256_text(x) if x else None for x in seq],
                              "endpoint_sha256": sha256_text(end) if end else None,
                              "selected_sha256": sha256_text(end) if end else None,
                              "reverted": False, "scores": []}
    if not sentinel:
        record["reason"] = "no sentinel split"
        return end, record
    if len(seq) < 2:
        record["reason"] = "no accepted change"
        return end, record
    order = [len(seq) - 1, len(seq) - 2, *range(len(seq) - 3, -1, -1)]
    scores: dict[int, float] = {}
    for i in order:
        value = evaluate(seq[i])
        if value is None:
            break
        scores[i] = value
    record["scores"] = [{"index": i, "sha256": sha256_text(seq[i]) if seq[i] else None,
                         "score": scores[i]} for i in sorted(scores)]
    last, prev = len(seq) - 1, len(seq) - 2
    if last not in scores or prev not in scores:
        record["reason"] = "selection budget too small"
        return end, record
    record["applied"] = True
    if scores[last] > scores[prev]:
        return end, record
    best = max(sorted(scores), key=lambda i: (scores[i], -i))
    chosen = seq[best]
    record.update(reverted=chosen != end, selected_sha256=sha256_text(chosen) if chosen else None)
    return chosen, record


# --------------------------------------------------------------------------- learner


class PartnerLearner:
    def __init__(self, *, agent_chat: Any, teacher_chat: Any,
                 decide: Callable[[dict[str, Any], str], dict[str, Any]],
                 settings: LearnSettings | None = None, policy: str | None = None) -> None:
        self.agent_chat, self.teacher_chat = agent_chat, teacher_chat
        self.decide_fn, self.settings = decide, settings or LearnSettings()
        self.policy = policy
        self.halted: str | None = None

    def halt(self, error: str) -> None:
        from slh.poc9.pipeline import latch_credit

        self.halted = self.halted or error
        latch_credit(f"learning halt: {error}")  # no request leaves anywhere after this

    def check_halt(self) -> None:
        if self.halted:
            raise RunHalted("learning", self.halted)

    def _decider(self, comp: Component, texts: dict[str, str] | None = None
                 ) -> Callable[[dict[str, Any], str], dict[str, Any]]:
        def decide(case: dict[str, Any], text: str) -> dict[str, Any]:
            from slh.poc9.pipeline import credit_gate_check

            self.check_halt()
            try:
                credit_gate_check()
            except CreditHalt as exc:
                self.halt(f"{type(exc).__name__}: {exc}")
                raise RunHalted("credit", str(exc)) from exc
            comp.take()
            if texts is not None and text:
                texts[sha256_text(text)] = text
            try:
                return self.decide_fn(case, text)
            except CreditHalt as exc:  # RunHalted included: latch, then stop
                self.halt(f"{type(exc).__name__}: {exc}")
                raise RunHalted("decision", f"{type(exc).__name__}: {exc}") from exc
            except Exception as exc:
                if classify(exc)[0] == "halt":
                    self.halt(f"{type(exc).__name__}: {exc}")
                    raise RunHalted("decision", f"{type(exc).__name__}: {exc}") from exc
                raise
        return decide

    def _scoring(self, case: dict[str, Any]) -> tuple[Callable, Callable]:
        if str(case.get("schema", "")).startswith("poc9-"):
            from slh.poc9 import corrections, score

            return score.score, corrections.explain
        from slh.xp.partner import corrections
        from slh.xp.score import score

        return score, corrections.explain

    def full_context(self, train: list[dict[str, Any]], comp: Component) -> tuple[str, int]:
        """B1 on every training case of the run's sample with outcome and
        correction, within the acquisition budget."""
        from slh.demo1.decide import render_transcript

        decide = self._decider(comp)
        score_fn, correct_fn = self._scoring(train[0])
        blocks = []
        rows = []
        for i, case in enumerate(train, 1):
            if not comp.can_decide():
                break
            d = decide(case, "")
            fb = {"action": d["action"], "score": score_fn(case, d["action"]),
                  "correction": correct_fn(case, d["action"])}
            rows.append({"case_id": case["case_id"], "stream_position": i, **fb})
            blocks.append(f"Episode {i}:\n{render_transcript(case['messages'])}\n"
                          + json.dumps(fb, sort_keys=True))
        self._acquisition = rows
        return ("Training histories and deterministic feedback:\n\n" + "\n\n".join(blocks),
                len(blocks))

    def learn(self, *, task: str, writer: str, run: int, critic: str,
              train: list[dict[str, Any]], dev: list[dict[str, Any]], out: Path,
              seed: int = 0, sentinel: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if writer not in WRITERS:
            raise ValueError(f"writer is one of {WRITERS}")
        out = Path(out)
        if out.exists():
            shutil.rmtree(out)  # an unfinished earlier attempt (no summary.json)
        out.mkdir(parents=True)
        try:
            summary = self._learn(task, writer, run, critic, train, dev, sentinel or [],
                                  out, seed)
            self.check_halt()
        except CreditHalt as exc:  # RunHalted included
            self.halt(f"{type(exc).__name__}: {exc}")
            shutil.rmtree(out, ignore_errors=True)
            raise exc if isinstance(exc, RunHalted) else RunHalted("learning", str(exc)) \
                from exc
        return summary

    def _ops(self, writer: str, chat: Any, role: str, comp: Component, *, out: Path,
             render: Callable, common: dict[str, Any], seed: int, per_call: int = 1
             ) -> dict[str, Any]:
        from slh.xp.learn import loop as xl

        texts: dict[str, str] = {}
        cfg = xl.LoopConfig(**{**self.settings.ops, "seed": seed,
                               "max_decisions": comp.decisions})
        ops_writer = xl.OpsWriter(writer, chat, role)
        ops_writer.budget = comp.writer_budget(per_call)
        res = xl.run_ops_learning(ops_writer, out=out, config=cfg, render_fn=render,
                                  decide=self._decider(comp, texts), **common)
        if res["stop_reason"] == "credit_halt":  # the loop caught a CreditHalt: latch it
            self.halt(res.get("stop_detail") or "credit_halt")
        self.check_halt()  # a halt inside the loop stops the whole run here
        steps = [r for r in pc.read_jsonl(out / "trace.jsonl") if r.get("type") == "step"]
        missing = [r["library_sha256"] for r in steps if r["accepted"]
                   and r["library_sha256"] and r["library_sha256"] not in texts]
        if missing:
            raise RuntimeError(f"accepted library {missing[0][:12]} was never evaluated")
        snapshots = [""] + [texts.get(r["library_sha256"], "") if r["library_sha256"] else ""
                            for r in steps if r["accepted"]]
        text = (out / "library.md").read_text(encoding="utf-8").strip() \
            if res["entries"] else ""
        if text and snapshots[-1] != text:
            snapshots.append(text)
        for name in ("library.json", "library.md"):
            (out / name).unlink(missing_ok=True)
        return {"skill": text, "snapshots": snapshots, "proposals": res["steps"],
                "accepted": res["accepted"], "stop_reason": res["stop_reason"],
                "episodes": res["decisions"]}

    def _learn(self, task: str, writer: str, run: int, critic: str,
               train: list[dict[str, Any]], dev: list[dict[str, Any]],
               sentinel: list[dict[str, Any]], out: Path, seed: int) -> dict[str, Any]:
        from slh.xp.learn import loop as xl

        comps = self.settings.components()
        policy = self.policy if self.policy is not None else w9.policy_excerpt()
        teacher_side = writer.startswith("T")
        chat = HaltingChat(self.teacher_chat if teacher_side else self.agent_chat, self)
        role = "teacher" if teacher_side else "agent"
        critic_chat = None
        if critic != "none":
            critic_chat = HaltingChat(self.teacher_chat if critic == "teacher"
                                      else self.agent_chat, self)
        per_call = max(chat.max_dispatches,
                       critic_chat.max_dispatches if critic_chat else 1)
        score_fn, correct_fn = self._scoring(train[0])

        evidence, acquired = self.full_context(train, comps["acquisition"])
        pc.append_jsonl(out / "acquisition.jsonl", self._acquisition)
        self.check_halt()
        if writer.endswith("+"):
            common = {"train": train, "dev": dev, "critic": critic, "critic_chat": critic_chat,
                      "policy": policy, "score_fn": score_fn, "correct_fn": correct_fn,
                      "task_text": TASK_TEXT[task]}
            main = self._ops(writer, chat, role, comps["learn"], out=out,
                             render=render_task_evidence, common=common, seed=seed,
                             per_call=per_call)
            search = self._ops(writer, chat, role, comps["search"], out=out / "search",
                               render=lambda eps: NO_EVIDENCE, common=common, seed=seed,
                               per_call=per_call)
            shutil.copy(out / "search/trace.jsonl", out / "search_trace.jsonl")
            loop = "ops"
        else:
            kw = {"train": train, "critic": critic_chat, "policy": policy,
                  "score_fn": score_fn, "correct_fn": correct_fn,
                  "batch": self.settings.note_batch}
            note = TaskNoteWriter(writer, chat, role, task)
            note.budget = comps["learn"].writer_budget(per_call)
            main = run_note_learning(note, out=out, decide=self._decider(comps["learn"]),
                                     can_decide=comps["learn"].can_decide, **kw)
            self.check_halt()
            ctrl = TaskNoteWriter(writer, chat, role, task)
            ctrl.budget = comps["search"].writer_budget(per_call)
            search = run_note_learning(ctrl, out=out, decide=self._decider(comps["search"]),
                                       can_decide=comps["search"].can_decide,
                                       evidence_cls=NoEvidence, prefix="search_", **kw)
            self.check_halt()
            (out / "search_patches.jsonl").replace(out / "search_trace.jsonl")
            (out / "search_episodes.jsonl").unlink(missing_ok=True)
            loop = "note"

        def evaluator(comp: Component) -> Callable[[str], float | None]:
            decide = self._decider(comp)

            def evaluate(text: str) -> float | None:
                if comp.decisions - comp.used_decisions < len(sentinel):
                    return None
                hits = [bool(score_fn(c, decide(c, text)["action"]).get("correct"))
                        for c in sentinel]
                return round(sum(hits) / len(hits), 4)
            return evaluate

        # Separate, equal, registered budgets (sized at registration for every
        # snapshot an arm can produce): neither arm can starve the other.
        skill, sel_learned = select_on_sentinel(main["snapshots"], sentinel,
                                                evaluate=evaluator(comps["selection_learned"]))
        self.check_halt()
        control, sel_search = select_on_sentinel(search["snapshots"], sentinel,
                                                 evaluate=evaluator(comps["selection_search"]))
        self.check_halt()

        generic_writer = TaskNoteWriter(writer, chat, role, task)
        generic_writer.budget = comps["generic"].writer_budget(per_call)
        gen = generic_writer.generic_note(policy, count_tokens(skill) if skill else None)
        self.check_halt()
        generic = gen.text if gen.ok and gen.text else ""
        files = {"skill.md": skill, "search-control.md": control, "generic.md": generic,
                 "evidence.md": evidence}
        for name, text in files.items():
            (out / name).write_text(text, encoding="utf-8")
        shutil.rmtree(out / "search", ignore_errors=True)
        budgets = {name: c.report() for name, c in comps.items()}
        budgets["total"] = {k: sum(b[k] for b in budgets.values())
                            for k in ("calls", "output_tokens", "decisions")}
        summary = {"task": task, "writer": writer, "run": run, "seed": seed, "critic": critic,
                   "check": xl.check_label(critic), "loop": loop,
                   "train_case_ids_sha256": pc.sha_text(pc.canonical(
                       [c["case_id"] for c in train])),
                   "train_cases": len(train), "dev": len(dev), "sentinel": len(sentinel),
                   "proposals": main["proposals"], "accepted": main["accepted"],
                   "episodes": main["episodes"], "stop_reason": main["stop_reason"],
                   "search_control": {"proposals": search["proposals"],
                                      "accepted": search["accepted"],
                                      "stop_reason": search["stop_reason"]},
                   "selection": {"learned": sel_learned, "search_control": sel_search},
                   "budgets": budgets, "evidence_cases": acquired,
                   "skill_sha256": sha256_text(skill) if skill else None,
                   "generic_sha256": sha256_text(generic) if generic else None,
                   "evidence_sha256": sha256_text(evidence),
                   "search_control_sha256": sha256_text(control) if control else None,
                   "skill_tokens": count_tokens(skill) if skill else 0,
                   "generic_ok": bool(generic), "generic_problems": list(gen.problems)}
        pc.write_json(out / "summary.json", summary)
        return summary
