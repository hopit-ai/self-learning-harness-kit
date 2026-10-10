"""POC 9 learning loop: one writer, the 48-episode training stream, snapshots.

- Start with an empty skill (snapshot 0 = B1).
- Episode = B1 with the current skill (``slh.poc9.loader.decide``) on the next
  train case in stream order, scored (``slh.poc9.score.score``) and given the
  scripted correction (``slh.poc9.corrections.explain``). Episodes are training
  only: their rows are written for the record and never reported as results.
- After every 4 episodes the writer proposes ONE patch from that batch's
  evidence (``slh.poc9.writers.EvidenceBatch``: visible conversation, action,
  deterministic outcome, lookup errors, correction, policy excerpt; no gold).
- Gate (identical for every writer): the writer's mechanical checks with one
  repair (format, 300-500 tokens, no case values, provenance cites this batch:
  evidence support and policy hygiene), then the mechanical checks again, then
  ONE binding critique call by the fixed critic: a contradiction with the
  policy or the evidence whose quotes are both found verbatim blocks the patch
  (no override, no second critique); an unreadable critique blocks it too.
- Snapshots (skill text + SHA-256) after episodes 0/12/24/48; every rejected
  patch is kept with its reasons.

Outputs under ``out``: ``episodes.jsonl`` (poc9-result-v1 rows, arm
``learning``), ``patches.jsonl``, ``snapshots/skill-<n>.md``, ``summary.json``
(call counts by role and purpose, teacher spend).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from slh.demo1.learn import count_tokens, sha256_text
from slh.poc9.writers import (
    BUDGET_EXHAUSTED,
    SKILL_TOKENS,
    EvidenceBatch,
    PatchResult,
    Writer,
    WriterBudget,
    check_skill,
    policy_excerpt,
)

BATCH = 4
EPISODES = 48
SNAPSHOTS = (0, 12, 24, 48)

CRITIQUE_PROMPT = (
    "Check the proposed SKILL.md below against the policy (in your instructions) and the "
    "evidence. List every statement of the skill that contradicts the policy or the "
    "evidence. Quote exactly. Reply with JSON only: {{\"contradictions\": [{{\"skill_quote\": "
    "\"<exact words from the skill>\", \"source\": \"policy\" or \"evidence\", "
    "\"source_quote\": \"<exact words from the policy or evidence>\", \"reason\": \"...\"}}]}}"
    " and an empty list if there is none.\n\nEvidence:\n{evidence}\n\nProposed skill:\n"
    "<skill>\n{skill}\n</skill>")
CRITIC_SYSTEM = ("You review procedures for a retail customer-service agent that must follow "
                 "this policy (excerpt):\n\n<policy>\n{policy}\n</policy>")
MIN_QUOTE = 8


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_critique(raw: str | None) -> list[dict[str, Any]] | None:
    """The contradictions list, or None if the reply is unreadable."""
    if raw is None:
        return None
    text = raw.strip()
    if re.fullmatch(r"(?i)\W*(none|no contradictions?\.?)\W*", text):
        return []
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    items = data.get("contradictions") if isinstance(data, dict) else None
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        return None
    return items


def critique(critic: Callable, skill: str, batch: EvidenceBatch) -> dict[str, Any]:
    """One binding critique call. A contradiction binds when its skill quote is
    in the skill and its source quote is in the policy or the evidence."""
    evidence = batch.text()
    raw, usage, error = critic(CRITIC_SYSTEM.format(policy=batch.policy),
                               CRITIQUE_PROMPT.format(evidence=evidence, skill=skill),
                               purpose="critique")
    call = {"role": "critic", "purpose": "critique", "raw": raw, "error": error, **usage}
    items = None if error else parse_critique(raw)
    if items is None:
        return {"ok": False, "binding": [], "dismissed": [], "call": call,
                "problem": "critique unreadable or failed (fail closed)"}
    sources = {"policy": _norm(batch.policy), "evidence": _norm(evidence)}
    binding, dismissed = [], []
    for item in items:
        sq, oq = _norm(str(item.get("skill_quote", ""))), _norm(str(item.get("source_quote", "")))
        found = len(sq) >= MIN_QUOTE and len(oq) >= MIN_QUOTE and sq in _norm(skill) and \
            any(oq in s for s in sources.values())
        (binding if found else dismissed).append(item)
    return {"ok": not binding, "binding": binding, "dismissed": dismissed, "call": call,
            "problem": f"{len(binding)} unresolved contradiction(s)" if binding else None}


def gate(patch: PatchResult, batch: EvidenceBatch, critic: Callable,
         budget: WriterBudget | None = None) -> dict[str, Any]:
    """Evidence support + policy check (mechanical, re-run) + binding critique.
    The critique counts against the writer's ceilings (``budget``); if it does
    not fit, the proposal is rejected as budget_exhausted."""
    if not patch.ok or not patch.changed:
        return {"accepted": False, "stage": "writer", "problems": patch.problems,
                "critique": None}
    problems = check_skill(patch.text, forbidden=batch.forbidden(), band=SKILL_TOKENS,
                           provenance=patch.provenance, indices=batch.indices())
    if problems:
        return {"accepted": False, "stage": "checks", "problems": problems, "critique": None}
    if budget is not None and not budget.can_call():
        return {"accepted": False, "stage": "budget", "problems": [BUDGET_EXHAUSTED],
                "critique": None}
    crit = critique(critic, patch.text, batch)
    if budget is not None:
        budget.charge(crit["call"])
    return {"accepted": crit["ok"], "stage": "critique",
            "problems": [crit["problem"]] if crit["problem"] else [], "critique": crit}


def _default_scoring() -> tuple[Callable, Callable]:
    from slh.poc9 import corrections, score

    return score.score, corrections.explain


def _result_row(writer: str, case: dict[str, Any], decision: dict[str, Any],
                score: dict[str, Any], correction: dict[str, Any] | None, skill: str,
                error: str | None) -> dict[str, Any]:
    from slh.demo1.context import SURFACE_SHA256

    return {"schema": "poc9-result-v1", "arm": "learning", "writer": writer, "snapshot": None,
            "skill_sha256": sha256_text(skill) if skill else None,
            "case_id": case["case_id"], "group": case["group"], "variant": case["variant"],
            "kind": case["kind"], "split": case["split"],
            "stream_index": case.get("stream_index"), "training_only": True,
            "action": decision["action"], "score": score, "correction": correction,
            "surface_sha256": SURFACE_SHA256,
            "prompt_sha256": decision.get("system_prompt_sha256"),
            "tokens": {"prompt": decision.get("input_tokens", 0),
                       "completion": decision.get("output_tokens", 0)},
            "model_calls": decision.get("model_calls", 0), "error": error}


def run_learning(writer: Writer, *, cases: list[dict[str, Any]],
                 decide: Callable[[dict[str, Any], str], dict[str, Any]],
                 critic: Callable, out: Path, policy: str | None = None,
                 score_fn: Callable | None = None, correct_fn: Callable | None = None,
                 teacher_meter: Any = None) -> dict[str, Any]:
    """One learning run of ``writer`` over the train stream. ``decide(case,
    skill)`` returns a ``slh.demo1.decide.next_action`` dict (normally
    ``loader.decide`` bound to the capped 9B client)."""
    if score_fn is None or correct_fn is None:
        default_score, default_correct = _default_scoring()
        score_fn, correct_fn = score_fn or default_score, correct_fn or default_correct
    stream = sorted(cases, key=lambda c: c["stream_index"])
    if len(stream) != EPISODES or any(c["split"] != "train" for c in stream) or \
            [c["stream_index"] for c in stream] != list(range(EPISODES)):
        raise ValueError("learning consumes exactly the 48 train cases, stream order 0..47")
    policy = policy if policy is not None else policy_excerpt()
    out = Path(out)
    for chat in {id(c): c for c in (critic, *writer.chats())}.values():
        if hasattr(chat, "log_to"):
            (out / "teacher-calls.jsonl").unlink(missing_ok=True)
            chat.log_to(out / "teacher-calls.jsonl")
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl", "patches.jsonl"):
        (out / name).write_text("", encoding="utf-8")

    def append(name: str, row: dict[str, Any]) -> None:
        with (out / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    skill = ""
    snapshots: dict[int, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    episode_calls = 0

    def snapshot(n: int) -> None:
        sha = sha256_text(skill) if skill else None
        (out / "snapshots" / f"skill-{n}.md").write_text(skill, encoding="utf-8")
        snapshots[n] = {"episodes": n, "skill_sha256": sha,
                        "tokens": count_tokens(skill) if skill else 0}

    snapshot(0)
    batch: list[dict[str, Any]] = []
    patches = accepted = unchanged = exhausted = 0
    for n, case in enumerate(stream, 1):
        error = None
        try:
            decision = decide(case, skill)
        except Exception as exc:  # noqa: BLE001 - recorded; the episode still counts
            error = f"{type(exc).__name__}: {exc}"
            decision = {"action": {"type": "none", "reason": "error"}, "reads": []}
        episode_calls += decision.get("model_calls", 0)
        score = score_fn(case, decision["action"])
        correction = correct_fn(case, decision["action"])
        append("episodes.jsonl", _result_row(writer.name, case, decision, score, correction,
                                             skill, error))
        batch.append({"index": n, "case": case, "action": decision["action"],
                      "reads": decision.get("reads") or [], "score": score,
                      "correction": correction})
        if n % BATCH == 0:
            evidence = EvidenceBatch(batch, policy)
            patch = writer.propose_patch(skill, evidence)
            verdict = gate(patch, evidence, critic, writer.budget)
            exhausted += BUDGET_EXHAUSTED in verdict["problems"]
            calls = patch.calls + ([verdict["critique"]["call"]] if verdict["critique"] else [])
            for c in calls:
                key = f"{c['role']}:{c['purpose']}"
                counts[key] = counts.get(key, 0) + 1
            patches += 1
            unchanged += patch.ok and not patch.changed
            if verdict["accepted"]:
                accepted += 1
                skill = patch.text
            append("patches.jsonl", {
                "after_episode": n, "writer": writer.name,
                "episodes": sorted(evidence.indices()),
                "evidence_sha256": sha256_text(evidence.text()),
                "accepted": verdict["accepted"], "no_change": patch.ok and not patch.changed,
                "stage": verdict["stage"], "problems": verdict["problems"],
                "budget_exhausted": BUDGET_EXHAUSTED in verdict["problems"],
                "critique": {k: v for k, v in (verdict["critique"] or {}).items()
                             if k != "call"} or None,
                "patch": patch.to_dict(), "critique_call": (verdict["critique"] or {}).get(
                    "call"), "skill_sha256_after": sha256_text(skill) if skill else None})
            batch = []
        if n in SNAPSHOTS:
            snapshot(n)
    summary = {"writer": writer.name, "episodes": EPISODES, "patches": patches,
               "accepted": accepted, "no_change": unchanged,
               "rejected": patches - accepted - unchanged,
               "budget_exhausted": exhausted, "writer_budget": writer.budget.to_dict(),
               "snapshots": snapshots, "calls": dict(sorted(counts.items())),
               "episode_model_calls": episode_calls,
               "teacher_usd": round(teacher_meter.spent_usd, 6) if teacher_meter else 0.0}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                      encoding="utf-8")
    return summary


def run(writer_name: str, out: Path, client: Any = None) -> dict[str, Any]:
    """Wire the real pieces: train cases, the capped 9B client (serving window),
    the configured teacher (writer T/TS and the critic, the same for every writer)."""
    from slh.demo1.decide import AgentClient
    from slh.poc9 import cases as poc9_cases
    from slh.poc9 import loader
    from slh.poc9.writers import TeacherChat, load_teacher, make_writer

    client = client or AgentClient.for_window()
    teacher = TeacherChat(load_teacher())
    writer = make_writer(writer_name, agent_client=client, teacher=teacher)
    return run_learning(writer, cases=poc9_cases.load("train"),
                        decide=lambda case, skill: loader.decide(case, skill, client),
                        critic=teacher, out=out, teacher_meter=teacher.meter)
