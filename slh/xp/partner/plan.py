"""Frozen phase plan, arm instances, prompt-hash dedupe, and the package we send.

Plan (``xp-partner-plan-v1``, self-hashed like ``slh.poc9.arms.freeze``)::

    {"schema", "phase", "plan_id", "seed", "tasks", "writers", "runs",
     "critic": {writer: "teacher"|"self"|"none"},
     "cases": {task: {split: {"file", "sha256", "rows"}}},
     "arms": [{"name", "kind", "scope", "text"?, "n"?}],
     "limits": {"case_limit_s", "max_output_tokens", "concurrency",
                "export_cap_bytes", "max_shards"},
     "learning": {...loop settings...},
     "protocol": {"file", "sha256"}, "sha256"}

Arm kinds: ``b0`` (model only), ``notes`` (inline text, e.g. B1 = "" or the
person-written procedure), ``learned`` / ``generic`` / ``full_context`` /
``search_control`` (texts the learning stage writes per task x run x writer),
``best_of_n`` (``n`` candidates + pairwise selector on the B1 prompt).
``scope`` is ``shared`` (once per task) or ``writer`` (per task x run x writer).

``prompt_hash`` is the hash of what the agent is actually sent: the rendered
system prompt and tools after the loader's normalisation (``slh.poc9.loader``
strips the note), B0's prompt, or B1's prompt plus n for best-of-n. Instances
with equal hashes run once; an empty or whitespace-only skill is B1.

Package (``write_package``): test cases leave as question rows; every split is
generated from the secret seed (``slh.xp.partner.secret``) and shipped rows
carry no seed; splits must be disjoint by case id, question payload, group,
family and user before anything is written (fail closed). Our side keeps the
test answer keys and the frozen test membership (case ids and counts).
"""

from __future__ import annotations

import functools
import random
from pathlib import Path
from typing import Any

from slh.xp.partner import common as pc

PLAN_SCHEMA = "xp-partner-plan-v1"
REGISTRATION_SCHEMA = "xp-partner-registration-v1"
ARM_KINDS = ("b0", "notes", "learned", "generic", "full_context", "search_control",
             "best_of_n")
WRITER_TEXT_KINDS = ("learned", "generic", "full_context", "search_control")
TEXT_FILES = {"learned": "skill.md", "generic": "generic.md", "full_context": "evidence.md",
              "search_control": "search-control.md"}
SPLITS = ("train", "validation", "sentinel", "test")
WITHHELD = ("test",)  # splits that leave as questions only
EXPORT_CAP_BYTES = 100_000_000  # 100 MB, decimal; registered per plan


def _limits(tasks: int) -> dict[str, Any]:
    from slh.demo1.decide import CASE_LIMIT_S, MAX_OUTPUT_TOKENS

    return {"case_limit_s": CASE_LIMIT_S, "max_output_tokens": MAX_OUTPUT_TOKENS,
            "concurrency": 8, "export_cap_bytes": EXPORT_CAP_BYTES,
            "max_shards": max(1, tasks)}


def default_arms(human_text: str = "", *, search_control: bool = True
                 ) -> list[dict[str, Any]]:
    """Plan v1.1 phase-1 arms (code rule and raw-excerpt controls added later)."""
    arms = [{"name": "b0", "kind": "b0", "scope": "shared"},
            {"name": "b1", "kind": "notes", "scope": "shared", "text": ""}]
    if human_text:
        arms.append({"name": "human", "kind": "notes", "scope": "shared", "text": human_text})
    arms += [{"name": "learned", "kind": "learned", "scope": "writer"},
             {"name": "generic", "kind": "generic", "scope": "writer"},
             {"name": "full_context", "kind": "full_context", "scope": "writer"}]
    if search_control:
        arms.append({"name": "search_control", "kind": "search_control", "scope": "writer"})
    arms.append({"name": "best_of_n", "kind": "best_of_n", "scope": "shared", "n": 2})
    return arms


def seal(body: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in body.items() if k != "sha256"}
    return {**body, "sha256": pc.sha_text(pc.canonical(body))}


def verify(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"not a {PLAN_SCHEMA} plan")
    body = {k: v for k, v in plan.items() if k != "sha256"}
    if pc.sha_text(pc.canonical(body)) != plan.get("sha256"):
        raise ValueError("plan hash does not verify (the plan changed after freezing)")
    names = [a["name"] for a in plan["arms"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate arm name in the plan")
    for arm in plan["arms"]:
        if arm["kind"] not in ARM_KINDS or arm["scope"] not in ("shared", "writer"):
            raise ValueError(f"bad arm {arm}")
        if (arm["kind"] in WRITER_TEXT_KINDS) != (arm["scope"] == "writer"):
            raise ValueError(f"{arm['name']}: writer-text arms (and only they) are per writer")
    budgets = (plan.get("learning") or {}).get("budgets")
    if budgets is not None:
        check_budgets(budgets)
    check_selection_capacity(plan)
    check_minimum_work(plan)
    return plan


# Rehearsal / dry-run smoke: fewer runs and test cases and a shorter lesson loop.
# A runtime setting recorded in the run identity; the registered plan is unchanged.
SMOKE = {"runs": 1, "cases": 6, "ops": {"max_steps": 2, "patience": 1}}


def effective(plan: dict[str, Any], smoke: dict[str, Any] | None) -> dict[str, Any]:
    """The plan as a smoke run executes it (a local copy; never re-sealed)."""
    if not smoke:
        return plan
    return {**plan, "runs": min(int(plan["runs"]), int(smoke["runs"]))}


def smoke_ids(ids: list[str], smoke: dict[str, Any] | None) -> list[str]:
    return sorted(ids)[:int(smoke["cases"])] if smoke else sorted(ids)


def load(path: Path) -> dict[str, Any]:
    return verify(pc.read_json(path))


# --------------------------------------------------------------------------- prompts


@functools.lru_cache(maxsize=4096)
def _rendered(text: str) -> tuple[str, str]:
    from slh.poc9 import loader

    r = loader.render(text or None)
    return r["system_prompt_sha256"], r["tools_sha256"]


def prompt_key(kind: str, text: str = "", n: int | None = None) -> dict[str, Any]:
    """What the agent is sent (rendered hashes), never the raw note text."""
    from slh.demo1.context import B0_PROMPT_SHA256, B1_PROMPT_SHA256, B1_TOOLS_SHA256

    if kind == "b0":
        return {"class": "b0", "system_prompt_sha256": B0_PROMPT_SHA256}
    if kind == "best_of_n":
        return {"class": "best_of_n", "n": int(n or 2), "system_prompt_sha256":
                B1_PROMPT_SHA256, "tools_sha256": B1_TOOLS_SHA256}
    system, tools = _rendered((text or "").strip())
    return {"class": "notes", "system_prompt_sha256": system, "tools_sha256": tools}


def prompt_hash(kind: str, text: str = "", n: int | None = None) -> str:
    return pc.sha_text(pc.canonical(prompt_key(kind, text, n)))


# --------------------------------------------------------------------------- instances


def instance_slots(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Every (task, run, writer, arm, kind) the plan requires; shared arms once per task."""
    out = []
    for task in plan["tasks"]:
        for arm in plan["arms"]:
            base = {"arm": arm["name"], "kind": arm["kind"], "n": arm.get("n")}
            if arm["scope"] == "shared":
                out.append({"task": task, "run": None, "writer": None, **base})
            else:
                out += [{"task": task, "run": r, "writer": w, **base}
                        for r in range(plan["runs"]) for w in plan["writers"]]
    return out


def slot_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["task"], row["run"], row["writer"], row["arm"])


def learning_slots(plan: dict[str, Any]) -> list[tuple[str, int, str]]:
    return [(t, r, w) for t in plan["tasks"] for r in range(plan["runs"])
            for w in plan["writers"]]


def instances(plan: dict[str, Any], text_for: Any) -> list[dict[str, Any]]:
    """Arm instances with prompt hashes. ``text_for(task, run, writer, kind)``
    returns the learning stage's text for writer-scoped text arms."""
    arms = {a["name"]: a for a in plan["arms"]}
    out = []
    for slot in instance_slots(plan):
        arm = arms[slot["arm"]]
        kind = arm["kind"]
        if kind in WRITER_TEXT_KINDS:
            text = text_for(slot["task"], slot["run"], slot["writer"], kind) or ""
        else:
            text = arm.get("text", "") if kind == "notes" else ""
        out.append({**slot, "prompt_hash": prompt_hash(kind, text, arm.get("n")),
                    "skill_sha256": pc.sha_text(text) if text else None, "_text": text})
    return out


# --------------------------------------------------------------------------- disjointness


def check_disjoint(cases_by_task: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    """Fail closed unless splits are disjoint.

    Within and across tasks: case id and question payload globally unique; a
    group in one split only. Within a task: families of test and train shared
    with no other split (validation and sentinel draw from one family pool by
    design). Across ALL tasks: no user of a test split appears in a non-test
    split, and no test gold shares its exact call (name + arguments), its order
    id or its (address1, zip) with a non-test gold. Catalogue item ids alone
    are shared by design (one catalogue); an item id is tied to an answer only
    with its order, which the call and order-id checks cover."""
    seen_id: dict[str, str] = {}
    seen_q: dict[str, str] = {}
    by: dict[str, dict[str, set[Any]]] = {}
    side: dict[bool, dict[str, set[Any]]] = {
        flag: {"user": set(), "call": set(), "order": set(), "address": set()}
        for flag in (True, False)}
    for task, splits in cases_by_task.items():
        for split, rows in splits.items():
            if split not in SPLITS:
                raise ValueError(f"unknown split {split!r}")
            test = split in WITHHELD
            for c in rows:
                where = f"{task}/{split}"
                cid = c["case_id"]
                if cid in seen_id:
                    raise ValueError(f"case {cid} appears in {seen_id[cid]} and {where}")
                seen_id[cid] = where
                q = pc.sha_text(pc.canonical({"task": pc.task_of(c),
                                              "messages": c["messages"]}))
                if q in seen_q:
                    raise ValueError(f"case {cid} ({where}) repeats the question of "
                                     f"{seen_q[q]}")
                seen_q[q] = f"{where}:{cid}"
                bucket = by.setdefault(f"{task}|{split}", {"group": set(), "family": set()})
                bucket["group"].add(c.get("group"))
                if c.get("family"):
                    bucket["family"].add(c["family"])
                user = (c.get("entities") or {}).get("user_id")
                if user:
                    side[test]["user"].add(user)
                atoms = pc.gold_atoms(c)
                if atoms:
                    side[test]["call"].add(atoms["call"])
                    side[test]["order"].add(atoms["order_id"])
                    if atoms["address"]:
                        side[test]["address"].add(atoms["address"])
    groups: dict[Any, str] = {}
    for key, bucket in by.items():
        for g in bucket["group"] - {None}:
            if g in groups:
                raise ValueError(f"groups shared by {groups[g]} and {key}")
            groups[g] = key
    for task in cases_by_task:
        present = [s for s in SPLITS if f"{task}|{s}" in by]
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                if ("test" in (a, b) or "train" in (a, b)) and \
                        by[f"{task}|{a}"]["family"] & by[f"{task}|{b}"]["family"]:
                    raise ValueError(f"{task}: families shared by {a} and {b}")
    for name in ("user", "call", "order", "address"):
        if side[True][name] & side[False][name]:
            raise ValueError(f"a test {name} also appears in a non-test split "
                             f"(across tasks): {sorted(map(str, side[True][name] & side[False][name]))[:1]}")


def distinctive_addresses(cases_by_task: dict[str, dict[str, list[dict[str, Any]]]]
                          ) -> list[list[str]]:
    """Test answer addresses (address1, zip) that appear nowhere in the shipped
    non-test cases: only these mark a leak on their own (the public database
    may show the same address elsewhere). Kept on our side (membership.json)."""
    blob = "\n".join(pc.canonical(c) for splits in cases_by_task.values()
                     for s, rows in splits.items() if s not in WITHHELD for c in rows)
    out = set()
    for splits in cases_by_task.values():
        for c in splits.get("test", []):
            atoms = pc.gold_atoms(c)
            if atoms and atoms["address"] and not (atoms["address"][0] in blob
                                                   and atoms["address"][1] in blob):
                out.add(atoms["address"])
    return [list(a) for a in sorted(out)]


def membership(cases_by_task: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    out = {}
    for task, splits in cases_by_task.items():
        ids = sorted(c["case_id"] for c in splits.get("test", []))
        out[task] = {"count": len(ids), "case_ids": ids,
                     "sha256": pc.sha_text(pc.canonical(ids))}
    return out


# --------------------------------------------------------------------------- generation


TASK_ORDER = ("address", "variant", "sequence")  # most constrained pool first
# Training pool per task (cases). The address pool cannot exceed about 150 cases
# next to a 1,000-case test split (measured 2026-10-10: 240 exhausts the pool).
POOL_CASES = {"address": 150, "variant": 400, "sequence": 400}
SAMPLE_CASES = 48  # each learning run's own training sample


def generation(tasks: list[str], *, test_cases: int = 1000,
               pool_cases: dict[str, int] | None = None, validation_cases: int = 60,
               sentinel_cases: int = 30) -> dict[str, Any]:
    pools = {**POOL_CASES, **(pool_cases or {})}
    order = [t for t in TASK_ORDER if t in tasks] + [t for t in tasks if t not in TASK_ORDER]
    return {"order": order, "sizes": {t: {"train": pools[t], "validation": validation_cases,
                                          "sentinel": sentinel_cases, "test": test_cases}
                                      for t in order}}


def build_cases(secret: bytes, plan_id: str, gen: dict[str, Any]
                ) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Every split of every task from the secret-derived seed, users carried
    across tasks (a user stays in one split class everywhere). The only source
    of packaged cases: provenance by construction."""
    from slh.xp.partner import secret as sec
    from slh.xp.tasks import common as cm
    from slh.xp.tasks.cli import task_module

    user_split: dict[str, str] = {}
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    db = cm.load_db()
    owners = {cm.canonical(cm.addr_args(u["address"])): uid for uid, u in db["users"].items()}
    for task in gen["order"]:
        s = gen["sizes"][task]
        sizes = cm.group_sizes(s["test"], train_cases=s["train"],
                               validation_cases=s["validation"], sentinel_cases=s["sentinel"])
        cases = task_module(task).generate(seed=sec.derive_seed(secret, task, plan_id),
                                           sizes=sizes, user_split=user_split)
        for c in cases:
            user_split.setdefault(c["entities"]["user_id"], c["split"])
            new = (c.get("request") or {}).get("new_address")
            if isinstance(new, dict):  # the owner of an answer address stays in this split
                owner = owners.get(cm.canonical(cm.addr_args(new)))
                if owner:
                    user_split.setdefault(owner, c["split"])
        splits: dict[str, list[dict[str, Any]]] = {}
        for c in cases:
            splits.setdefault(c["split"], []).append(c)
        out[task] = splits
    return out


def train_samples(cases_by_task: dict[str, dict[str, list[dict[str, Any]]]], *, runs: int,
                  seed: int, size: int = SAMPLE_CASES) -> dict[str, Any]:
    """Each run's own seeded sample of the training pool (case ids in stream
    order), plus the measured overlap between runs."""
    out: dict[str, Any] = {}
    for task, splits in cases_by_task.items():
        pool = sorted(c["case_id"] for c in splits.get("train", []))
        if len(pool) < size:
            raise ValueError(f"{task}: training pool of {len(pool)} is smaller than {size}")
        per_run = {}
        for r in range(runs):
            rng = random.Random(pc.sha_text(f"xp-train-sample|{seed}|{task}|{r}"))
            per_run[str(r)] = rng.sample(pool, size)
        pairs = [(a, b) for a in range(runs) for b in range(a + 1, runs)]
        shared = [len(set(per_run[str(a)]) & set(per_run[str(b)])) / size for a, b in pairs]
        out[task] = {"pool": len(pool), "size": size, "runs": per_run,
                     "overlap": {"pairs": len(pairs),
                                 "mean": round(sum(shared) / len(shared), 4) if shared else 0.0,
                                 "max": round(max(shared), 4) if shared else 0.0,
                                 "expected": round(size / len(pool), 4)}}
    return out


# --------------------------------------------------------------------------- budgets


BUDGET_COMPONENTS = ("acquisition", "learn", "search", "selection_learned",
                     "selection_search", "generic")
DEFAULT_MAX_STEPS = 20  # slh.xp.learn.loop.LoopConfig.max_steps
DEFAULT_NOTE_BATCH = 4


def max_snapshots(*, sample: int, max_steps: int, note_batch: int,
                  writers: list[str] | tuple[str, ...] = ("S", "T", "S+", "T+")) -> int:
    """Most snapshots one arm can produce (the empty start + one per accepted
    proposal): ops loop max_steps + 1; note loop ceil(sample / batch) + 1."""
    out = 1
    if any(w.endswith("+") for w in writers):
        out = max(out, max_steps + 1)
    if any(not w.endswith("+") for w in writers):
        out = max(out, -(-sample // note_batch) + 1)
    return out


SELF_HOSTED_TEACHER_RETRIES = 2  # slh.xp.partner.learners.SelfHostedChat
GENERIC_CALLS = 3  # write + repair + shorten
LOOP_MIN_CALLS = 4  # one proposal: write/ops + repair + shorten + critique


def production_dispatch() -> dict[str, int]:
    """Worst-case transport dispatches per writer call of the production chats:
    the 9B agent (configs/agent-vllm.yaml retries + 1 + meter pauses) and the
    teacher (the larger of the API teacher, configs/poc9-teacher.yaml, and the
    self-hosted teacher, which has no pauses)."""
    import yaml

    from slh.budget import MAX_PAUSES

    def retries(name: str) -> int:
        raw = yaml.safe_load((pc.REPO / "configs" / name).read_text(encoding="utf-8"))
        return int(raw.get("retries", 3))

    return {"agent": retries("agent-vllm.yaml") + 1 + MAX_PAUSES,
            "teacher": max(retries("poc9-teacher.yaml") + 1 + MAX_PAUSES,
                           SELF_HOSTED_TEACHER_RETRIES + 1)}


def per_call_dispatch(writers: list[str] | tuple[str, ...], critic: dict[str, str] | None,
                      dispatch: dict[str, int]) -> int:
    """The largest reservation any writer call of this plan makes (writer chat
    or its critic, as the learner reserves ``max`` of both)."""
    roles = {"teacher" if w.startswith("T") else "agent" for w in writers}
    roles |= {"teacher" if c == "teacher" else "agent"
              for c in (critic or {}).values() if c != "none"}
    return max([1, *(int(dispatch[r]) for r in roles)])


def default_budgets(*, sample: int = SAMPLE_CASES, sentinel: int = 30,
                    max_steps: int = DEFAULT_MAX_STEPS, note_batch: int = DEFAULT_NOTE_BATCH,
                    writers: list[str] | tuple[str, ...] = ("S", "T", "S+", "T+"),
                    critic: dict[str, str] | None = None,
                    dispatch: dict[str, int] | None = None) -> dict[str, Any]:
    """One registered budget per learning-run component; ``total`` is their sum.
    Learn and search are equal (search-control parity); so are the two sentinel
    selections, each sized for every snapshot its arm can produce. Call ceilings
    are in transport dispatches: logical calls x the plan's per-call reservation
    (``production_dispatch``), so retries and reservations never starve a component."""
    from slh.poc9.writers import CALL_OUTPUT_RESERVE, WRITER_CALL_CEILING

    per_call = per_call_dispatch(writers, critic, dispatch or production_dispatch())
    loop = {"calls": WRITER_CALL_CEILING * per_call, "output_tokens": WRITER_CALL_CEILING *
            CALL_OUTPUT_RESERVE, "decisions": 1500}
    select = {"calls": 0, "output_tokens": 0, "decisions": sentinel * max_snapshots(
        sample=sample, max_steps=max_steps, note_batch=note_batch, writers=writers)}
    comp = {"acquisition": {"calls": 0, "output_tokens": 0, "decisions": sample},
            "learn": dict(loop), "search": dict(loop),
            "selection_learned": dict(select), "selection_search": dict(select),
            "generic": {"calls": GENERIC_CALLS * per_call,
                        "output_tokens": GENERIC_CALLS * CALL_OUTPUT_RESERVE, "decisions": 0}}
    return {**comp, "total": {k: sum(c[k] for c in comp.values())
                              for k in ("calls", "output_tokens", "decisions")}}


def check_budgets(budgets: dict[str, Any]) -> None:
    if set(budgets) != {*BUDGET_COMPONENTS, "total"}:
        raise ValueError("learning budgets need exactly the registered components and total")
    for k in ("calls", "output_tokens", "decisions"):
        if sum(int(budgets[c][k]) for c in BUDGET_COMPONENTS) != int(budgets["total"][k]):
            raise ValueError(f"learning budget {k}: components do not sum to the total")
    for a, b in (("learn", "search"), ("selection_learned", "selection_search")):
        if budgets[a] != budgets[b]:
            raise ValueError(f"learning budgets {a} and {b} must be equal (parity)")


def check_minimum_work(plan: dict[str, Any]) -> None:
    """At registration: every component can finish its minimum work under the
    worst-case dispatch reservations (``learning.dispatch``): the generic note
    (write + repair + shorten), one full learning / search-control proposal,
    the full-context pass over the sample, and (lesson loops) mining the sample
    plus two dev evaluations."""
    from slh.poc9.writers import CALL_OUTPUT_RESERVE

    learning = plan.get("learning") or {}
    budgets, samples = learning.get("budgets"), learning.get("train_samples")
    dispatch = learning.get("dispatch")
    if budgets is None or samples is None:
        return
    if not dispatch or set(dispatch) != {"agent", "teacher"}:
        raise ValueError("the plan registers no per-call dispatch reservations")
    per_call = per_call_dispatch(plan["writers"], plan.get("critic"), dispatch)
    need = {"generic": GENERIC_CALLS, "learn": LOOP_MIN_CALLS, "search": LOOP_MIN_CALLS}
    for comp, calls in need.items():
        b = budgets[comp]
        if int(b["calls"]) < calls * per_call or \
                int(b["output_tokens"]) < calls * CALL_OUTPUT_RESERVE:
            raise ValueError(f"{comp} budget ({b['calls']} dispatches, {b['output_tokens']} "
                             f"tokens) cannot finish {calls} calls at {per_call} dispatches "
                             "each")
    for task, splits in plan["cases"].items():
        sample = int(samples[task]["size"])
        dev = int((splits.get("validation") or {}).get("rows", 0))
        if int(budgets["acquisition"]["decisions"]) < sample:
            raise ValueError(f"{task}: acquisition budget cannot cover the {sample}-case sample")
        loop_need = sample + (2 * dev if any(w.endswith("+") for w in plan["writers"]) else 0)
        for comp in ("learn", "search"):
            if int(budgets[comp]["decisions"]) < loop_need:
                raise ValueError(f"{task}: {comp} decision budget below {loop_need}")


def check_selection_capacity(plan: dict[str, Any]) -> None:
    """At registration: each sentinel selection budget covers every snapshot its
    arm can produce, for every task (never discovered at run time)."""
    learning = plan.get("learning") or {}
    budgets, samples = learning.get("budgets"), learning.get("train_samples")
    if budgets is None or samples is None:
        return
    ops = learning.get("ops") or {}
    for task, splits in plan["cases"].items():
        sentinel = int((splits.get("sentinel") or {}).get("rows", 0))
        need = sentinel * max_snapshots(
            sample=int(samples[task]["size"]),
            max_steps=int(ops.get("max_steps", DEFAULT_MAX_STEPS)),
            note_batch=int(learning.get("note_batch", DEFAULT_NOTE_BATCH)),
            writers=plan["writers"])
        for comp in ("selection_learned", "selection_search"):
            if int(budgets[comp]["decisions"]) < need:
                raise ValueError(f"{task}: {comp} budget {budgets[comp]['decisions']} is "
                                 f"below the {need} decisions its snapshots can need")


# --------------------------------------------------------------------------- package


def build_plan(*, plan_id: str, tasks: list[str], writers: list[str], runs: int,
               arms: list[dict[str, Any]], cases: dict[str, dict[str, dict[str, Any]]],
               protocol: dict[str, str], critic: dict[str, str] | None = None,
               seed: int = 7401, limits: dict[str, Any] | None = None,
               learning: dict[str, Any] | None = None, generation_spec: dict | None = None,
               phase: int = 1) -> dict[str, Any]:
    lim = {**_limits(len(tasks)), **(limits or {})}
    if lim["export_cap_bytes"] > EXPORT_CAP_BYTES:
        raise ValueError(f"the export cap may not exceed {EXPORT_CAP_BYTES} bytes")
    return verify(seal({"schema": PLAN_SCHEMA, "phase": phase, "plan_id": plan_id,
                        "seed": seed, "tasks": list(tasks), "writers": list(writers),
                        "runs": int(runs), "critic": critic or {w: "none" for w in writers},
                        "cases": cases, "arms": arms, "limits": lim,
                        "learning": learning or {}, "generation": generation_spec or {},
                        "protocol": protocol}))


def write_package(out: Path, keys_dir: Path, *, secret: bytes, plan_id: str,
                  tasks: list[str], writers: list[str], runs: int,
                  arms: list[dict[str, Any]], protocol_text: str,
                  gen: dict[str, Any] | None = None, critic: dict[str, str] | None = None,
                  seed: int = 7401, limits: dict[str, Any] | None = None,
                  learning: dict[str, Any] | None = None, sample_cases: int = SAMPLE_CASES,
                  root: Path = pc.REPO, registration_extra: dict[str, Any] | None = None
                  ) -> dict[str, Any]:
    """Generate every case from the secret seed (``build_cases``), check
    disjointness, and write ``out/{cases/, plan.json, protocol.md,
    registration.json}``; test answer keys and ``membership.json`` go to
    ``keys_dir`` (outside ``out``). There is no way to pass cases in: a package
    holds only cases this function generated from the secret."""
    from slh.xp.partner import secret as sec
    from slh.xp.partner import vendor

    out, keys_dir = Path(out), Path(keys_dir)
    try:
        keys_dir.resolve().relative_to(out.resolve())
        raise PermissionError("answer keys may not be written inside the package")
    except ValueError:
        pass
    gen = gen or generation(tasks)
    if sorted(gen["order"]) != sorted(tasks):
        raise ValueError("generation spec does not cover exactly the plan's tasks")
    cases_by_task = build_cases(secret, plan_id, gen)
    check_disjoint(cases_by_task)
    samples = train_samples(cases_by_task, runs=runs, seed=seed, size=sample_cases)
    learning = dict(learning or {})
    ops = dict(learning.get("ops") or {})
    dispatch = dict(learning.get("dispatch") or production_dispatch())
    budgets = learning.get("budgets") or default_budgets(
        sample=sample_cases,
        sentinel=max(len(s.get("sentinel", [])) for s in cases_by_task.values()),
        max_steps=int(ops.get("max_steps", DEFAULT_MAX_STEPS)),
        note_batch=int(learning.get("note_batch", DEFAULT_NOTE_BATCH)), writers=writers,
        critic=critic, dispatch=dispatch)
    check_budgets(budgets)
    learning.update(budgets=budgets, train_samples=samples, dispatch=dispatch)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    keys_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, dict[str, Any]]] = {}
    keys: dict[str, Path] = {}
    for task in tasks:
        splits = cases_by_task[task]
        files[task] = {}
        for split in SPLITS:
            rows = splits.get(split)
            if not rows:
                continue
            if split in WITHHELD:
                body, n = pc.jsonl_bytes(pc.question_row(c) for c in rows)
                rel = f"cases/{task}-{split}.questions.jsonl"
                key_body, _ = pc.jsonl_bytes(pc.answer_key_rows(rows))
                keys[task] = keys_dir / f"{task}-{split}.answer-key.jsonl"
                keys[task].write_bytes(key_body)
            else:
                body, n = pc.jsonl_bytes(sec.strip_seed({**c, "task": pc.task_of(c)})
                                         for c in rows)
                rel = f"cases/{task}-{split}.jsonl"
            (out / rel).write_bytes(body)
            files[task][split] = {"file": rel, "sha256": pc.sha_bytes(body), "rows": n}
    (out / "protocol.md").write_text(protocol_text, encoding="utf-8")
    protocol = {"file": "protocol.md", "sha256": pc.sha_text(protocol_text)}
    plan = build_plan(plan_id=plan_id, tasks=list(tasks), writers=writers, runs=runs,
                      arms=arms, cases=files, protocol=protocol, critic=critic, seed=seed,
                      limits=limits, learning=learning, generation_spec=gen)
    pc.write_json(out / "plan.json", plan)
    member = membership({t: cases_by_task[t] for t in tasks})
    pc.write_json(keys_dir / "membership.json", {
        "plan_id": plan_id, "tasks": member, "secret_fingerprint": sec.fingerprint(secret),
        "distinctive_addresses": distinctive_addresses(cases_by_task)})
    registration = {"schema": REGISTRATION_SCHEMA, "plan_id": plan_id,
                    "git_commit": pc.git_state(root)["git_commit"],
                    "code_sha256": pc.code_hashes(root)["code_sha256"],
                    "plan_sha256": plan["sha256"], "protocol_sha256": protocol["sha256"],
                    "case_files": case_file_hashes(plan),
                    "test_membership": {t: {"count": m["count"], "sha256": m["sha256"]}
                                        for t, m in member.items()},
                    "export_cap_bytes": plan["limits"]["export_cap_bytes"],
                    "max_shards": plan["limits"]["max_shards"],
                    "vendor": vendor.expected(), **(registration_extra or {})}
    pc.write_json(out / "registration.json", registration)
    return {"plan": plan, "registration": registration, "keys": keys,
            "membership": keys_dir / "membership.json", "cases": cases_by_task}


def case_file_hashes(plan: dict[str, Any]) -> dict[str, str]:
    return {f["file"]: f["sha256"] for splits in plan["cases"].values()
            for f in splits.values()}


def run_train(plan: dict[str, Any], task: str, run: int,
              pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The run's registered training sample, in its stream order."""
    ids = plan["learning"]["train_samples"][task]["runs"][str(run)]
    by = {c["case_id"]: c for c in pool}
    missing = [i for i in ids if i not in by]
    if missing:
        raise ValueError(f"{task} run {run}: sample cases missing from the pool: {missing[:3]}")
    return [by[i] for i in ids]
