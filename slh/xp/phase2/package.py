"""Phase-2 package: what a run needs, without any answer (docs/xp/phase2-design.md).

Built on our side; the same package drives our pilot and the full run at the
partner. Layout (``xp-phase2-package-v1``)::

    plan.json          the Phase2Plan (arms, runs, sizes, budget) + its sha256
    scenarios.jsonl    the TEST split: generated scenarios and the official shard
    smoke.jsonl        SMOKE-ONLY scenarios (rehearsal; never test, never scored
                       as evidence)
    bundles.json       the arm texts per learning run (``slh.xp.phase2.bundles``)
    registration.json  sha256 of every file above, the code hash, the git commit

Each scenario row is a ``PublicScenario`` (the simulator's instructions) plus
its split; the matching ``ScenarioKey`` rows (expected writes, expected
database hash, fidelity checklist, predicate binding, grading criteria) go to a
separate keys file that never enters the package. ``build`` refuses before
writing anything when a public row carries a key field or a value only the key
should know (a new item id, an address the user never states), when ids repeat
or a smoke scenario is also a test scenario (by id, by content fingerprint of
the simulator's instructions, whatever its id, or by configuration), or when
the plan has problems.

Every conversation runs as a tau2 task WITHOUT evaluation criteria
(``to_tau2_task``): nothing is graded where the run happens; scoring replays the
exported transcripts on our side (``slh.xp.phase2.check``).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from slh.xp import phase2 as P
from slh.xp.partner import common as pc

PACKAGE_SCHEMA = "xp-phase2-package-v1"
SCENARIO_SCHEMA = "xp-phase2-scenario-v1"
KEY_SCHEMA = "xp-phase2-key-v1"
SPLITS = ("test", "smoke")
PACKAGE_FILES = ("plan.json", "scenarios.jsonl", "smoke.jsonl", "bundles.json")
INSTRUCTION_FIELDS = P.INSTRUCTION_FIELDS
GENERATOR_VERSION = "genpilot-generator-v1"
OFFICIAL_PREFIX = "off-"
WRITE_TOOLS = frozenset({
    "cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
    "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
    "return_delivered_order_items"})


class PackageRefused(ValueError):
    pass


# --------------------------------------------------------------------------- plan json


def plan_to_json(plan: P.Phase2Plan) -> dict[str, Any]:
    body = asdict(plan)
    body["schema"] = "xp-phase2-plan-v1"
    body.pop("sha256", None)
    return {**body, "sha256": pc.sha_text(pc.canonical(body))}


def plan_from_json(data: dict[str, Any]) -> P.Phase2Plan:
    body = {k: v for k, v in data.items() if k not in ("sha256", "schema")}
    if pc.sha_text(pc.canonical({**body, "schema": data.get("schema")})) != data.get("sha256"):
        raise PackageRefused("plan.json differs from its sha256")
    arms = tuple(P.Arm(**{**a, "runs": tuple(a["runs"])}) for a in body.pop("arms"))
    budget = P.Budget(**body.pop("budget"))
    for k in ("primary", "secondary"):
        if body.get(k) is not None:
            body[k] = tuple(body[k])
    return P.Phase2Plan(arms=arms, budget=budget, **body)


# --------------------------------------------------------------------------- scenarios


def _write_values(writes: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Values of the expected writes that only the key may hold, by kind."""
    items, addresses = set(), set()
    for w in writes:
        args = w.get("arguments") or {}
        for v in args.get("new_item_ids") or []:
            items.add(str(v))
        if args.get("address1"):
            addresses.add(str(args["address1"]))
    return {"new_item_ids": items, "address1": addresses}


def from_genpilot(s: dict[str, Any], *, scenario_id: str | None = None,
                  generator_version: str | None = None
                  ) -> tuple[P.PublicScenario, dict[str, Any]]:
    """A generated scenario (``slh.genpilot.generate`` or phase-2 generator record)
    as (public, key row).

    The key carries the expected writes, the predicate binding and the fidelity
    checklist; ``stated_values``: expected-write values the user's own
    instructions state (an explicit new address), which the leak scan allows.
    ``generator_version``: default the record's own, else the pilot's."""
    generator_version = generator_version or s.get("generator_version") or GENERATOR_VERSION
    text = s["text"]
    sid = scenario_id or s["task_id"]
    public = P.PublicScenario(
        scenario_id=sid, source="generated", decision_type=s["cell"],
        subvariation=s["subvariation"],
        user_scenario={k: text[k] for k in INSTRUCTION_FIELDS},
        generator_version=generator_version)
    public = P.PublicScenario(**{**asdict(public), "scenario_hash": public.computed_hash()})
    blob = json.dumps(public.body())
    explicit = (s.get("root") or {}).get("address") == "explicit"
    values = _write_values(s["expected_writes"])
    stated = sorted(v for v in values["address1"] if explicit and v in blob)
    binding = s.get("predicate")
    key = P.ScenarioKey(
        scenario_id=sid, scenario_hash=public.scenario_hash,
        expected_writes=tuple(s["expected_writes"]), expected_db_sha256="",
        must_voice_in_opening=tuple((s.get("fidelity_checklist") or {}).get(
            "must_voice_in_opening") or ()),
        forbidden=tuple((s.get("fidelity_checklist") or {}).get("forbidden") or ()),
        predicate=(binding or {}).get("predicate"),
        user_id=(s.get("sample") or {}).get("user_id") or (s.get("constraints") or {}).get(
            "user_id", ""),
        configuration=pc.sha_text(pc.canonical({
            "user": (s.get("sample") or {}).get("user_id"),
            "target": (s.get("sample") or {}).get("target"), "source": "generated"})))
    criteria = {"actions": [{"action_id": f"{sid}_{i}", "requestor": "assistant",
                             "name": w["name"], "arguments": w["arguments"], "info": None}
                            for i, w in enumerate(s["expected_writes"])],
                "communicate_info": [], "nl_assertions": None, "reward_basis": ["DB"]}
    row = key_row(key, binding=binding, criteria=criteria, stated_values=stated)
    if s.get("p2") is not None:
        # Generator v2, key only: family, prior approval, phase-2 success predicates
        # (slh.xp.phase2.check.P2_PREDICATES) and the specification values the
        # secondary outcomes use.
        row["p2"] = s["p2"]
    return public, row


def from_official(task: dict[str, Any]) -> tuple[P.PublicScenario, dict[str, Any]]:
    """An official tau2 retail task (the frozen test split) as (public, key row)."""
    us = task["user_scenario"]
    instr = us["instructions"]
    if not isinstance(instr, dict) or instr.get("domain") != "retail":
        raise PackageRefused(f"official task {task['id']}: unstructured instructions")
    if task.get("initial_state") is not None:
        raise PackageRefused(f"official task {task['id']}: initial_state is not supported")
    sid = OFFICIAL_PREFIX + str(task["id"])
    scenario = {k: instr[k] for k in INSTRUCTION_FIELDS}
    if us.get("persona") is not None:
        scenario["persona"] = us["persona"]
    public = P.PublicScenario(scenario_id=sid, source="official_test", decision_type="mixed",
                              subvariation="official", user_scenario=scenario,
                              generator_version="tau2-retail-official")
    public = P.PublicScenario(**{**asdict(public), "scenario_hash": public.computed_hash()})
    criteria = task.get("evaluation_criteria") or {}
    writes = [{"name": a["name"], "arguments": a["arguments"]}
              for a in criteria.get("actions") or [] if a["name"] in WRITE_TOOLS]
    key = P.ScenarioKey(scenario_id=sid, scenario_hash=public.scenario_hash,
                        expected_writes=tuple(writes), expected_db_sha256="",
                        must_voice_in_opening=(), user_id="",
                        configuration=pc.sha_text(f"official|{task['id']}"))
    return public, key_row(key, binding=None, criteria=criteria, stated_values=[],
                           official_task_id=str(task["id"]))


def key_row(key: P.ScenarioKey, *, binding: dict[str, Any] | None,
            criteria: dict[str, Any], stated_values: list[str],
            official_task_id: str | None = None) -> dict[str, Any]:
    row = {"schema": KEY_SCHEMA, **asdict(key), "predicate_binding": binding,
           "evaluation_criteria": criteria, "stated_values": list(stated_values),
           "official_task_id": official_task_id}
    row["expected_writes"] = list(row["expected_writes"])
    return row


def scenario_row(public: P.PublicScenario, split: str) -> dict[str, Any]:
    if split not in SPLITS:
        raise PackageRefused(f"unknown split {split!r}")
    row = {"schema": SCENARIO_SCHEMA, "split": split, **asdict(public)}
    pc.assert_no_gold(row, f"scenario {public.scenario_id}")
    return row


def public_of(row: dict[str, Any]) -> P.PublicScenario:
    return P.PublicScenario(**{k: row[k] for k in (
        "scenario_id", "source", "decision_type", "subvariation", "user_scenario",
        "generator_version", "scenario_hash")})


def to_tau2_task(row: dict[str, Any]) -> dict[str, Any]:
    """The tau2 task a conversation runs: the simulator's instructions only, NO
    evaluation criteria (nothing is graded where the run happens). For official
    tasks the user scenario equals the benchmark's, so the simulator's prompt is
    the same as in our earlier official-split runs."""
    problems = P.user_scenario_problems(row["user_scenario"])
    if problems:
        raise PackageRefused(f"{row['scenario_id']}: {'; '.join(problems)}")
    view = P.simulator_view(row["user_scenario"])
    return {"id": row["scenario_id"], "description": None,
            "user_scenario": {"persona": view["persona"],
                              "instructions": {"domain": "retail",
                                               **{k: view[k] for k in INSTRUCTION_FIELDS}}},
            "initial_state": None, "evaluation_criteria": None}


def leak_problems(rows: list[dict[str, Any]], keys: dict[str, dict[str, Any]]) -> list[str]:
    """Public rows that carry a key field, or an expected-write value the user's
    own instructions do not state (new item ids never; an address only when the
    scenario has the user state it)."""
    out = []
    for row in rows:
        out += P.public_problems(public_of(row))
        key = keys.get(row["scenario_id"])
        if key is None:
            out.append(f"{row['scenario_id']}: no key")
            continue
        if key["scenario_hash"] != row["scenario_hash"]:
            out.append(f"{row['scenario_id']}: key and scenario hashes differ")
        if row["source"] != "generated":
            continue  # official instructions are the public benchmark's own text
        blob = json.dumps(row)
        values = _write_values(key["expected_writes"])
        allowed = set(key.get("stated_values") or [])
        for kind, found in values.items():
            for v in sorted(found - allowed):
                if v in blob:
                    out.append(f"{row['scenario_id']}: expected {kind} value in public text")
    return out


def overlap_problems(rows: dict[str, list[dict[str, Any]]],
                     keys: dict[str, dict[str, Any]] | None = None) -> list[str]:
    """Smoke/test isolation independent of scenario ids: no two scenarios (within
    or across splits) share a content fingerprint (``phase2.content_fingerprint``
    of the simulator's instructions), and (with the keys, our side) no smoke
    scenario shares a (user, target order) configuration with a test scenario."""
    out = []
    seen: dict[str, str] = {}
    for split in SPLITS:
        for r in rows.get(split, []):
            fp = P.content_fingerprint(r["user_scenario"])
            if fp in seen:
                other = seen[fp]
                what = ("a smoke scenario is also a test scenario"
                        if {other.split(":")[0], split} == {"smoke", "test"}
                        else "a scenario repeats")
                out.append(f"{what} (same instructions: {other.split(':', 1)[1]} and "
                           f"{r['scenario_id']})")
            else:
                seen[fp] = f"{split}:{r['scenario_id']}"
    if keys is not None:
        def configs(split: str) -> dict[str, str]:
            return {keys[r["scenario_id"]]["configuration"]: r["scenario_id"]
                    for r in rows.get(split, []) if r["scenario_id"] in keys
                    and keys[r["scenario_id"]].get("configuration")}

        shared = sorted(set(configs("smoke")) & set(configs("test")))
        if shared:
            out.append(f"a smoke scenario is also a test scenario ({len(shared)} shared "
                       "(user, target order) configurations)")
    return out


# --------------------------------------------------------------------------- build


def expected_db_hash(key: dict[str, Any]) -> str:
    """tau2's database hash after the expected writes on a fresh retail database.

    Generated scenarios: every expected write must succeed (an erroring write is a
    generator defect). Official tasks: the gold DB the way tau2's own evaluator
    builds it (``EnvironmentEvaluator.calculate_reward``): every gold action of the
    benchmark's criteria, in order, with a tool error logged and skipped. Some
    official tasks list a gold action that errors (e.g. task 64 exchanges a pending
    order, then modifies it); such writes leave the database unchanged, so they
    are dropped from the key's ``expected_writes`` (recorded in
    ``gold_writes_erroring``) and never count against a run's later writes."""
    if key.get("official_task_id") is not None:
        return _official_gold(key)
    from tau2.data_model.message import ToolCall

    from slh.xp import retaildb

    env = retaildb.env()  # the package's pinned database (build: retaildb.use)
    for i, w in enumerate(key["expected_writes"]):
        response = env.get_response(ToolCall(id=f"k{i}", name=w["name"],
                                             arguments=w["arguments"]))
        if response.error:
            raise PackageRefused(f"{key['scenario_id']}: expected write {i} errors on the "
                                 "pinned database")
    return env.get_db_hash()


def _official_gold(key: dict[str, Any]) -> str:
    """Replay an official task's gold actions exactly as tau2's evaluator does and
    set the key's effective expected writes (those that do not error)."""
    from tau2.data_model.tasks import Action
    from tau2.registry import registry

    criteria = key.get("evaluation_criteria") or {}
    actions = [Action.model_validate(a) for a in criteria.get("actions") or []]
    env = registry.get_env_constructor("retail")()
    ok: list[dict[str, Any]] = []
    erroring: list[dict[str, Any]] = []
    for a in actions:
        try:
            env.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
            failed = False
        except Exception:  # tau2: logger.warning(...) and continue
            failed = True
        if a.name in WRITE_TOOLS:
            (erroring if failed else ok).append({"name": a.name, "arguments": a.arguments})
    gold = [{"name": a.name, "arguments": a.arguments} for a in actions
            if a.name in WRITE_TOOLS]
    listed = [{"name": w["name"], "arguments": w["arguments"]}
              for w in key["expected_writes"]]
    if listed != gold:
        raise PackageRefused(f"{key['scenario_id']}: expected writes differ from the "
                             "official task's gold actions")
    key["expected_writes"] = ok
    key["gold_writes_erroring"] = erroring
    return env.get_db_hash()


def retail_db_of(registration: dict[str, Any]) -> str:
    """The pinned retail database a package's generated scenarios use (``slh.xp.retaildb``;
    a registration without ``retail_db`` is the original database)."""
    return (registration.get("retail_db") or {}).get("name", "original")


def tau2_domain(registration: dict[str, Any]) -> str:
    """The tau2 domain conversations of this package run in: ``retail`` (original) or
    the registered extension domain (same policy and tools, the pinned extension DB)."""
    from slh.xp import retaildb

    name = retail_db_of(registration)
    if name != retaildb.ORIGINAL:
        retaildb.register_domains()
    return retaildb.domain(name)


def benchmark_data_hashes(name: str = "original") -> dict[str, str]:
    """The data files a run on database ``name`` reads (the runner's ``data_hashes``;
    for the extension, ``db.json`` replaced by the pinned extension file)."""
    from slh import runner
    from slh.xp import retaildb

    if name == retaildb.ORIGINAL:
        return runner.data_hashes("retail")
    return retaildb.data_hashes(retaildb.domain(name))


def build(out: Path, keys_out: Path, *, plan: P.Phase2Plan,
          test: list[tuple[P.PublicScenario, dict[str, Any]]],
          smoke: list[tuple[P.PublicScenario, dict[str, Any]]],
          bundles: dict[str, Any], root: Path = pc.REPO,
          check_plan: bool = True, calibration: bool = False,
          retail_db: str = "original") -> dict[str, Any]:
    """Write the package to ``out`` and the keys to ``keys_out`` (never inside it).

    ``retail_db``: the pinned database the generated scenarios run and are scored on
    (``slh.xp.retaildb``); recorded in the registration when not the original. Official
    tau2 tasks always use the ORIGINAL database, so a package on the extension refuses
    any official scenario.

    ``calibration``: a calibration package (generator v2's calib split in the smoke
    slot, run with ``run smoke`` / the pilot): it carries NO test scenario, so the
    test-size and official-shard checks do not apply; its registration says so."""
    from slh.xp.phase2 import bundles as B

    out, keys_out = Path(out), Path(keys_out)
    if out.exists() and any(out.iterdir()):
        raise PackageRefused(f"{out} is not empty")
    if out.resolve() in keys_out.resolve().parents:
        raise PackageRefused("the keys file must not be inside the package")
    problems = plan.problems() if check_plan else []
    rows = {"test": [scenario_row(p, "test") for p, _ in test],
            "smoke": [scenario_row(p, "smoke") for p, _ in smoke]}
    keys = {k["scenario_id"]: k for _, k in [*test, *smoke]}
    ids = [r["scenario_id"] for split in SPLITS for r in rows[split]]
    if len(set(ids)) != len(ids):
        problems.append("scenario ids repeat (within or across splits)")
    hashes = {split: {r["scenario_hash"] for r in rows[split]} for split in SPLITS}
    if hashes["test"] & hashes["smoke"]:
        problems.append("a smoke scenario is also a test scenario")
    problems += overlap_problems(rows, keys)
    gen = [r for r in rows["test"] if r["source"] == "generated"]
    per_type = {t: sum(r["decision_type"] == t for r in gen) for t in P.DECISION_TYPES}
    official = sum(r["source"] == "official_test" for r in rows["test"])
    if calibration:
        if rows["test"]:
            problems.append("a calibration package carries no test scenario")
    else:
        if per_type != plan.scenarios_per_type:
            problems.append(f"generated test scenarios per type {per_type} differ from the "
                            f"plan {plan.scenarios_per_type}")
        if official != plan.official_test_tasks:
            problems.append(f"{official} official test tasks, the plan has "
                            f"{plan.official_test_tasks}")
    if not rows["smoke"]:
        problems.append("no smoke scenarios")
    if any(r["source"] != "generated" for r in rows["smoke"]):
        problems.append("smoke scenarios must be generated (never official test tasks)")
    problems += leak_problems([*rows["test"], *rows["smoke"]], keys)
    problems += B.problems(bundles, plan)
    if retail_db != "original" and any(r["source"] != "generated"
                                       for split in SPLITS for r in rows[split]):
        problems.append("official tau2 tasks always use the original retail database "
                        f"(this package is on {retail_db})")
    if problems:
        raise PackageRefused("; ".join(problems[:20]))
    from slh.xp import retaildb

    with retaildb.use(retail_db):
        for key in keys.values():
            key["expected_db_sha256"] = expected_db_hash(key)
    out.mkdir(parents=True, exist_ok=True)
    pc.write_json(out / "plan.json", plan_to_json(plan))
    for split, name in (("test", "scenarios.jsonl"), ("smoke", "smoke.jsonl")):
        raw, _ = pc.jsonl_bytes(sorted(rows[split], key=lambda r: r["scenario_id"]))
        (out / name).write_bytes(raw)
    pc.write_json(out / "bundles.json", bundles)
    git = pc.git_state(root)
    registration = {"schema": PACKAGE_SCHEMA, "plan_id": plan.plan_id,
                    "data_sha256": benchmark_data_hashes(retail_db),
                    "files": {n: pc.sha_file(out / n) for n in PACKAGE_FILES},
                    "counts": {s: len(rows[s]) for s in SPLITS},
                    "code_sha256": pc.code_hashes(root)["code_sha256"],
                    "git_commit": git["git_commit"], "git_dirty": git["git_dirty"]}
    if calibration:
        registration["purpose"] = "calibration (no test scenarios; never evidence)"
    if retail_db != "original":
        registration["retail_db"] = retaildb.identity(retail_db)
    pc.write_json(out / "registration.json", registration)
    registration_sha = pc.sha_file(out / "registration.json")
    keys_out.parent.mkdir(parents=True, exist_ok=True)
    raw, _ = pc.jsonl_bytes([{**k, "package_registration_sha256": registration_sha}
                             for _, k in sorted(keys.items())])
    keys_out.write_bytes(raw)
    return {"package": str(out), "keys": str(keys_out), "registration_sha256": registration_sha,
            **registration["counts"]}


# --------------------------------------------------------------------------- load


class Package:
    def __init__(self, path: Path, *, root: Path = pc.REPO, check_code: bool = True) -> None:
        self.path = Path(path)
        self.registration = verify(self.path, root=root, check_code=check_code)
        self.registration_sha256 = pc.sha_file(self.path / "registration.json")
        self.plan_json = pc.read_json(self.path / "plan.json")
        self.plan = plan_from_json(self.plan_json)
        self.bundles = pc.read_json(self.path / "bundles.json")
        self.rows = {"test": pc.read_jsonl(self.path / "scenarios.jsonl"),
                     "smoke": pc.read_jsonl(self.path / "smoke.jsonl")}
        for split, rows in self.rows.items():
            for row in rows:
                pc.assert_no_gold(row, f"scenario {row.get('scenario_id')}")
                if row.get("split") != split:
                    raise PackageRefused(f"{row.get('scenario_id')}: split {row.get('split')} "
                                         f"in the {split} file")
                problems = P.public_problems(public_of(row))
                if problems or not row.get("scenario_hash"):
                    raise PackageRefused(f"{row['scenario_id']}: {problems or 'no hash'}")
        overlap = overlap_problems(self.rows)
        if overlap:
            raise PackageRefused("; ".join(overlap[:5]))

    def scenarios(self, split: str) -> list[dict[str, Any]]:
        return list(self.rows[split])


def verify(path: Path, *, root: Path = pc.REPO, check_code: bool = True) -> dict[str, Any]:
    """Refuse unless every package file matches registration.json and (by default)
    this checkout's code hash equals the registered one."""
    path = Path(path)
    reg_path = path / "registration.json"
    if not reg_path.is_file():
        raise PackageRefused("registration.json missing from the package")
    reg = pc.read_json(reg_path)
    if reg.get("schema") != PACKAGE_SCHEMA:
        raise PackageRefused(f"unknown package schema {reg.get('schema')!r}")
    for name in PACKAGE_FILES:
        if not (path / name).is_file() or pc.sha_file(path / name) != reg["files"].get(name):
            raise PackageRefused(f"{name}: missing or differs from the registration")
    if check_code:
        have = pc.code_hashes(root)["code_sha256"]
        if have != reg.get("code_sha256"):
            raise PackageRefused("the code in this checkout differs from the registered code "
                                 "(code_sha256)")
    return reg


def load_keys(path: Path, *, registration_sha256: str | None = None
              ) -> dict[str, dict[str, Any]]:
    """Our side only: scenario id -> key row, bound to one package registration."""
    out: dict[str, dict[str, Any]] = {}
    for row in pc.read_jsonl(Path(path)):
        if row.get("schema") != KEY_SCHEMA:
            raise PackageRefused(f"{row.get('scenario_id')}: not a phase-2 key row")
        if registration_sha256 is not None and \
                row.get("package_registration_sha256") != registration_sha256:
            raise PackageRefused(f"{row['scenario_id']}: key bound to another package")
        if row["scenario_id"] in out:
            raise PackageRefused(f"{row['scenario_id']}: duplicate key")
        out[row["scenario_id"]] = row
    return out


# --------------------------------------------------------------------------- cli


def _genpilot_file(path: Path) -> list[dict[str, Any]]:
    data = pc.read_json(path)
    return data["scenarios"] if isinstance(data, dict) else data


def main(argv: list[str] | None = None) -> int:
    """python -m slh.xp.phase2.package [--out package-p2] --keys KEYS --test gen.json
    --smoke smoke.json --phase1-export DIR [...] [--plan minimum|full] [--runs 12]
    [--sizes 75,75,75] [--official-split test] [--root SNAPSHOT]"""
    import argparse
    import sys

    from slh.xp.phase2 import bundles as B

    ap = argparse.ArgumentParser(prog="python -m slh.xp.phase2.package")
    ap.add_argument("--out", type=Path, default=Path(P.PACKAGE_DIR),
                    help=f"package directory (default {P.PACKAGE_DIR}/, apart from phase 1's "
                         "package/)")
    ap.add_argument("--keys", type=Path, required=True, help="our side only; never shipped")
    ap.add_argument("--test", type=Path,
                    help="generated test scenarios (generator records); not with "
                         "--calibration")
    ap.add_argument("--calibration", action="store_true",
                    help="calibration package: --smoke names the calib split's key file; "
                         "no test scenario and no official shard")
    ap.add_argument("--smoke", type=Path, required=True, help="smoke-only scenarios")
    ap.add_argument("--phase1-export", type=Path, action="append", required=True)
    ap.add_argument("--plan", choices=("minimum", "full"), default="minimum")
    ap.add_argument("--plan-id", default="xp-p2")
    ap.add_argument("--phase1-plan-sha256", default="0" * 64)
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--sizes", help="address,variant,sequence (default: the plan's)")
    ap.add_argument("--official-split", default="test",
                    help="frozen official split shipped as the official shard ('' for none)")
    ap.add_argument("--root", type=Path, default=pc.REPO,
                    help="code tree whose hash is registered (the kit snapshot)")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    kw: dict[str, Any] = {"runs": args.runs, "phase1_plan_sha256": args.phase1_plan_sha256}
    if args.sizes:
        kw["scenarios"] = tuple(int(x) for x in args.sizes.split(","))
    if bool(args.calibration) == bool(args.test):
        print(json.dumps({"status": "refused", "reason": "pass --test, or --calibration "
                                                         "(and no --test)"}, indent=1))
        return 4
    official: list[dict[str, Any]] = []
    if args.official_split and not args.calibration:
        ids = pc.read_json(pc.REPO / "splits/retail.json")[args.official_split]
        tasks = {t["id"]: t for t in pc.read_json(
            pc.REPO / "vendor/tau2-bench/data/tau2/domains/retail/tasks.json")}
        official = [tasks[i] for i in ids]
    kw["official_test_tasks"] = len(official)
    make = P.minimum_plan if args.plan == "minimum" else P.default_plan
    plan = make(plan_id=args.plan_id, **kw)
    test = ([] if args.calibration else [from_genpilot(s) for s in _genpilot_file(args.test)]
            ) + [from_official(t) for t in official]
    smoke = [from_genpilot(s) for s in _genpilot_file(args.smoke)]
    bundles = B.from_phase1_exports(args.phase1_export, runs=args.runs,
                                    human=B.person_procedures())
    try:
        result = build(args.out, args.keys, plan=plan, test=test, smoke=smoke,
                       bundles=bundles, root=args.root, calibration=args.calibration)
    except PackageRefused as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, indent=1))
        return 4
    print(json.dumps({"status": "ok", **result}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
