"""Verify partner export shard(s) and score them against our answer keys (fail closed).

    python -m slh.xp.partner.check --export SHARD [--export SHARD2 ...] \
        --registration registration.json --plan plan.json --membership KEYS/membership.json \
        --key variant=KEYS/variant-test.answer-key.jsonl --key ... \
        [--out check-report.json] [--reveal-incomplete] [--allow-dry-run]

The registered plan and OUR frozen test membership (case ids and counts per task,
``membership.json`` written with the package) are required; coverage is measured
against them, never against what an answer-key file happens to contain.

Per shard: manifest (required file set, no unlisted file, SHA-256/bytes/rows),
size within the registered cap, registration hashes, run identity (all shards one
run; a dry run is refused unless allowed), typed allowlist of every row, and rows
belonging to the shard's declared tasks. Shards declare disjoint task sets; at
most ``max_shards``; coverage is computed over their union.

Then: every arm instance must match the plan exactly (task, run, writer, arm,
kind, n) and its prompt hash must be the one the plan text or the exported skill
text renders to; duplicate identities fail; decisions join our key by case id
with the same case_hash, must belong to a prompt some arm uses, and their arm
label must be an instance with that prompt; answer signatures (exact gold
argument sets of our test key) anywhere outside decision actions fail, as does a
test case id in learning, skill or conversation rows or beside a ``kind``.

Status: ``failed`` (no scores), ``incomplete`` (coverage gaps: missing tasks,
arm instances, learning provenance or cases; counts only unless revealed) or
``ok``. Exit code 0 only for ok.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from slh.xp.partner import common as pc
from slh.xp.partner import plan as pp
from slh.xp.partner import schemas as sc

REG_KEYS = ("git_commit", "code_sha256", "plan_sha256", "protocol_sha256")
EXPORT_SCHEMA = "xp-partner-export-v2"
_ORDER = re.compile(r"#W\d+")
_ID_TOKEN = re.compile(r"[A-Za-z0-9]+-[0-9a-f]{6,}")
LOG_ID_FIELDS = frozenset({"name", "task", "case_id", "prompt_hash", "error", "error_class",
                           "event", "stage"})


class CheckFailed(ValueError):
    pass


# --------------------------------------------------------------------------- inputs


def load_membership(membership: dict[str, Any], registration: dict[str, Any],
                    plan: dict[str, Any]) -> dict[str, list[str]]:
    if membership.get("plan_id") != plan["plan_id"]:
        raise CheckFailed("membership file belongs to another plan")
    out = {}
    for task in plan["tasks"]:
        m = (membership.get("tasks") or {}).get(task)
        reg = (registration.get("test_membership") or {}).get(task)
        if m is None or reg is None:
            raise CheckFailed(f"{task}: no registered test membership")
        ids = m["case_ids"]
        if ids != sorted(set(ids)) or len(ids) != m["count"] or not ids:
            raise CheckFailed(f"{task}: membership list malformed")
        digest = pc.sha_text(pc.canonical(ids))
        if digest != m["sha256"] or digest != reg["sha256"] or m["count"] != reg["count"]:
            raise CheckFailed(f"{task}: membership differs from the registration")
        out[task] = ids
    return out


def load_keys(keys: dict[str, Path], members: dict[str, list[str]]
              ) -> dict[str, dict[str, dict[str, Any]]]:
    out = {}
    for task, ids in members.items():
        path = keys.get(task)
        if path is None or not Path(path).is_file():
            raise CheckFailed(f"{task}: answer key missing")
        key = pc.load_answer_key(Path(path))
        if sorted(key) != ids:
            raise CheckFailed(f"{task}: answer key does not hold exactly the registered test "
                              f"cases ({len(key)} rows vs {len(ids)})")
        out[task] = key
    return out


def signatures(keys: dict[str, dict[str, dict[str, Any]]]) -> dict[str, list[str]]:
    """Order id -> serialisations of the gold argument sets that name it."""
    out: dict[str, list[str]] = defaultdict(list)
    for key in keys.values():
        for case in key.values():
            args = (case.get("gold") or {}).get("arguments")
            if not isinstance(args, dict) or not args.get("order_id"):
                continue
            for value in (args, case["gold"]):
                out[args["order_id"]] += [pc.canonical(value), json.dumps(value, sort_keys=True),
                                          json.dumps(value)]
    return out


# --------------------------------------------------------------------------- shards


def read_shard(export: Path, registration: dict[str, Any], plan: dict[str, Any]
               ) -> dict[str, Any]:
    export = Path(export)
    man_path = export / "manifest.json"
    if not man_path.is_file():
        raise CheckFailed(f"{export}: manifest.json missing")
    man = sc.check_object("manifest.json", pc.read_json(man_path), sc.MANIFEST_FIELDS)
    if man.get("schema") != EXPORT_SCHEMA or man.get("plan_id") != plan["plan_id"]:
        raise CheckFailed(f"{export}: not an export of plan {plan['plan_id']}")
    listed = set(man.get("files") or {})
    if any(p.is_dir() for p in export.iterdir()):
        raise CheckFailed(f"{export}: unexpected directory in the delivery")
    present = {p.name for p in export.iterdir() if p.is_file()} - {"manifest.json"}
    if present - listed:
        raise CheckFailed(f"{export}: files not in the manifest: {sorted(present - listed)}")
    missing = set(sc.REQUIRED_FILES) - listed
    if missing:
        raise CheckFailed(f"{export}: required files missing: {sorted(missing)}")
    if listed - set(sc.REQUIRED_FILES) - set(sc.OPTIONAL_FILES):
        raise CheckFailed(f"{export}: unknown files listed: {sorted(listed)}")
    total = man_path.stat().st_size
    for name, meta in sorted(man["files"].items()):
        path = export / name
        if not path.is_file():
            raise CheckFailed(f"{name}: listed in the manifest but missing")
        if path.stat().st_size != meta["bytes"] or pc.sha_file(path) != meta["sha256"]:
            raise CheckFailed(f"{name}: SHA-256 or size differs from the manifest")
        total += meta["bytes"]
    cap = int(registration["export_cap_bytes"])
    if man.get("cap_bytes") != cap:
        raise CheckFailed(f"{export}: manifest cap {man.get('cap_bytes')} is not the "
                          f"registered {cap}")
    if total > cap:
        raise CheckFailed(f"{export}: delivery is {total} bytes, over the registered cap {cap}")
    hashes = sc.check_object("manifest hashes", man.get("hashes"), sc.HASH_FIELDS)
    for key in REG_KEYS:
        if hashes.get(key) != registration.get(key):
            raise CheckFailed(f"{key}: export has {hashes.get(key)!r}, registered "
                              f"{registration.get(key)!r}")
    if hashes["case_files"] != registration.get("case_files"):
        raise CheckFailed("case-file hashes differ from the registration")
    tasks = (man.get("shard") or {}).get("tasks")
    if not isinstance(tasks, list) or not tasks or set(tasks) - set(plan["tasks"]):
        raise CheckFailed(f"{export}: bad shard task list {tasks!r}")
    rows: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(listed - {"env.json"}):
        data = pc.read_jsonl(export / name)
        if len(data) != man["files"][name]["rows"]:
            raise CheckFailed(f"{name}: row count differs from the manifest")
        for r in data:
            sc.strict(name, r)
            if "task" in r and r["task"] not in tasks:
                raise CheckFailed(f"{name}: row of task {r['task']} in a shard of {tasks}")
        rows[name] = data
    env = sc.check_object("env.json", pc.read_json(export / "env.json"), sc.ENV_FIELDS)
    if env["mode"] != hashes["run_identity"]["mode"] or \
            env["endpoint_sha256"] != hashes["run_identity"]["endpoint_sha256"]:
        raise CheckFailed("env.json does not match the run identity")
    return {"path": str(export), "tasks": tasks, "hashes": hashes, "env": env, "rows": rows,
            "manifest": man,
            "bytes": total}


def merge(shards: list[dict[str, Any]], registration: dict[str, Any], *, allow_dry: bool
          ) -> dict[str, list[dict[str, Any]]]:
    if len(shards) > int(registration["max_shards"]):
        raise CheckFailed(f"{len(shards)} shards; at most {registration['max_shards']} "
                          "registered")
    seen: set[str] = set()
    for s in shards:
        if seen & set(s["tasks"]):
            raise CheckFailed(f"task delivered in two shards: {sorted(seen & set(s['tasks']))}")
        seen |= set(s["tasks"])
    identities = {pc.canonical(s["hashes"]["run_identity"]) for s in shards}
    if len(identities) != 1:
        raise CheckFailed("shards come from different runs (mode or endpoint differ)")
    if shards[0]["hashes"]["run_identity"]["mode"] != "real" and not allow_dry:
        raise CheckFailed("the delivery is a dry run (fake client)")
    merged: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name in (*sc.REQUIRED_FILES, *sc.OPTIONAL_FILES):
        if name != "env.json":
            merged[name] = []
    for s in shards:
        for name, rows in s["rows"].items():
            merged[name] += rows
    return merged


# --------------------------------------------------------------------------- scans


def atoms_index(keys: dict[str, dict[str, dict[str, Any]]],
                distinctive: list[list[str]] | None = None
                ) -> tuple[dict[str, list[set[str]]], dict[str, set[str]]]:
    """Test-gold atoms: order id -> the answer values that pin it (address
    fields, item ids); zip -> address1 of a test answer address that appears in
    no shipped non-test case (``membership.distinctive_addresses``)."""
    by_order: dict[str, list[set[str]]] = defaultdict(list)
    by_zip: dict[str, set[str]] = defaultdict(set)
    for key in keys.values():
        for case in key.values():
            atoms = pc.gold_atoms(case)
            if atoms is None:
                continue
            by_order[atoms["order_id"]].append(atoms["values"])
    for address1, zip_code in distinctive or []:
        by_zip[zip_code].add(address1)
    return by_order, by_zip


_ZIP = re.compile(r"(?<!\d)\d{5}(?!\d)")


def scan(merged: dict[str, list[dict[str, Any]]], extras: list[tuple[str, Any]],
         sigs: dict[str, list[str]], test_ids: set[str], kinds: set[str],
         atoms: tuple[dict[str, list[set[str]]], dict[str, set[str]]]) -> None:
    """Every string the partner sends (rows, env, manifest), outside decision
    actions: no exact gold serialisation, no test order id together with one
    of its answer values, no test answer address (address1 with its zip)."""
    by_order, by_zip = atoms

    def gold_in(text: str, where: str) -> None:
        for order in set(_ORDER.findall(text)):
            for sig in sigs.get(order, ()):
                if sig in text:
                    raise CheckFailed(f"{where}: carries the answer of a test case")
            for values in by_order.get(order, ()):
                if any(v in text for v in values):
                    raise CheckFailed(f"{where}: carries the answer of a test case "
                                      "(order and answer value together)")
        for z in set(_ZIP.findall(text)):
            if any(a in text for a in by_zip.get(z, ())):
                raise CheckFailed(f"{where}: carries a test answer address")

    def walk(value: Any, where: str) -> None:
        if isinstance(value, dict):
            if "kind" in value and any(isinstance(v, str) and v in test_ids
                                       for v in value.values()):
                raise CheckFailed(f"{where}: a test case id beside a kind")
            for k, v in value.items():
                gold_in(k, where)
                walk(v, where)
        elif isinstance(value, list):
            for v in value:
                walk(v, where)
        elif isinstance(value, str):
            gold_in(value, where)  # serialised answers inside a string
            if value.lstrip().startswith(("{", "[")):
                try:
                    parsed = json.loads(value)
                except ValueError:
                    return
                gold_in(pc.canonical(parsed), where)
                walk(parsed, where)

    for name, rows in merged.items():
        for i, row in enumerate(rows):
            where = f"{name}[{i}]"
            body = {k: v for k, v in row.items() if k != "action"} \
                if name == "decisions.jsonl.gz" else row
            text = pc.canonical(body) + "\n" + json.dumps(body)
            gold_in(text, where)
            walk(body, where)
            if name in ("learning.jsonl.gz", "skills.jsonl.gz", "conversations.jsonl.gz"):
                hit = set(_ID_TOKEN.findall(text)) & test_ids
                if hit:
                    raise CheckFailed(f"{where}: names test case {min(hit)}")
            if name == "logs.jsonl.gz":
                r = row["row"]
                if any(isinstance(v, str) and v in test_ids for v in r.values()):
                    if set(r) - LOG_ID_FIELDS:
                        raise CheckFailed(f"{where}: a log row about a test case has extra "
                                          "fields")
                    if any(isinstance(v, str) and v in kinds for k, v in r.items()
                           if k != "error"):
                        raise CheckFailed(f"{where}: a log row pairs a test case with a kind")
    for where, value in extras:
        text = pc.canonical(value) + "\n" + json.dumps(value)
        gold_in(text, where)
        walk(value, where)
        if set(_ID_TOKEN.findall(text)) & test_ids:
            raise CheckFailed(f"{where}: names a test case")


# --------------------------------------------------------------------------- learning


def _trace_ok(rows: list[dict[str, Any]], loop: str, proposals: int, where: str) -> None:
    if loop == "ops":
        types = Counter(r.get("type") for r in rows)
        if types["start"] != 1 or types["stop"] != 1 or types["step"] != proposals:
            raise CheckFailed(f"{where}: trace has {dict(types)}; {proposals} proposals "
                              "summarised")
        for r in rows:
            if r.get("type") == "step" and not (isinstance(r.get("accepted"), bool) and
                                                isinstance(r.get("verdicts"), list) and
                                                isinstance(r.get("calls"), list)):
                raise CheckFailed(f"{where}: a step lacks its proposal, verdicts or calls")
    else:
        if len(rows) != proposals:
            raise CheckFailed(f"{where}: {len(rows)} patch rows; {proposals} proposals "
                              "summarised")
        for r in rows:
            if not (isinstance(r.get("accepted"), bool) and isinstance(r.get("stage"), str)
                    and isinstance(r.get("patch"), dict)
                    and isinstance(r.get("writer_budget"), dict)):
                raise CheckFailed(f"{where}: a patch row lacks its proposal, verdict or "
                                  "budget")


SELECTION_REASONS = ("no sentinel split", "no accepted change", "selection budget too small")


def _dedupe(seq: list[Any]) -> list[Any]:
    out: list[Any] = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def verify_selection(sel: dict[str, Any], trace_snapshots: list[str | None], sentinel: int,
                     decisions: int, where: str, limit: int) -> None:
    """Recompute the sentinel selection from its own record: snapshots equal the
    accepted states in the trace, scores belong to those snapshots and are
    possible sentinel accuracies, the chosen snapshot follows the registered rule
    (keep the endpoint only if it beats its predecessor, else the best, ties to
    the earlier), and the selection decisions match the scores evaluated."""
    snaps = sel["snapshots"]
    if snaps != _dedupe(trace_snapshots):
        raise CheckFailed(f"{where}: selection snapshots differ from the accepted trace states")
    if sel["endpoint_sha256"] != snaps[-1]:
        raise CheckFailed(f"{where}: selection endpoint is not the last snapshot")
    scores = sel["scores"]
    idx = [s["index"] for s in scores]
    if len(set(idx)) != len(idx) or any(not 0 <= i < len(snaps) for i in idx):
        raise CheckFailed(f"{where}: selection score indices are invalid")
    for s in scores:
        if s["sha256"] != snaps[s["index"]]:
            raise CheckFailed(f"{where}: a selection score names another snapshot")
        if not 0.0 <= s["score"] <= 1.0 or (sentinel and abs(
                s["score"] * sentinel - round(s["score"] * sentinel)) > 1e-3 * sentinel + 1e-9):
            raise CheckFailed(f"{where}: a selection score is not a sentinel accuracy")
    if decisions != len(scores) * sentinel:
        raise CheckFailed(f"{where}: selection decisions do not match the scores evaluated")
    last, prev = len(snaps) - 1, len(snaps) - 2
    # What the registered rule and budget make mandatory: evaluate endpoint,
    # predecessor, then older snapshots, as far as the registered capacity allows.
    if not sentinel:
        expected, want = "no sentinel split", []
    elif len(snaps) < 2:
        expected, want = "no accepted change", []
    else:
        order = [last, prev, *range(last - 2, -1, -1)]
        want = order[:min(len(snaps), limit // sentinel)]
        expected = None if len(want) >= 2 else "selection budget too small"
    if sorted(idx) != sorted(want):
        raise CheckFailed(f"{where}: selection evaluated {sorted(idx)}, but the registered "
                          f"capacity ({limit} decisions) requires {sorted(want)}")
    if expected is not None:
        if sel["applied"] or sel["reason"] != expected or sel["reverted"] or \
                sel["selected_sha256"] != snaps[-1]:
            raise CheckFailed(f"{where}: an unapplied selection is inconsistent")
        return
    if not sel["applied"] or sel["reason"] is not None:
        raise CheckFailed(f"{where}: selection was required by the registered capacity")
    by = {s["index"]: s["score"] for s in scores}
    chosen = last if by[last] > by[prev] else max(sorted(by), key=lambda i: (by[i], -i))
    if sel["selected_sha256"] != snaps[chosen] or \
            sel["reverted"] != (snaps[chosen] != snaps[last]):
        raise CheckFailed(f"{where}: the selected snapshot does not follow the registered rule")


def _states(rows: list[dict[str, Any]], loop: str) -> list[str | None]:
    if loop == "ops":
        return [None] + [r.get("library_sha256") for r in rows
                         if r.get("type") == "step" and r.get("accepted")]
    return [None] + [r.get("skill_sha256_after") for r in rows if r.get("accepted")]


def _sample_use(by: dict[str, list[dict[str, Any]]], s: dict[str, Any], ids: list[str],
                where: str) -> None:
    """Episode and acquisition case ids are the run's registered sample, in order."""
    acq = [r.get("case_id") for r in by.get("acquisition", [])]
    if acq != ids[:s["evidence_cases"]]:
        raise CheckFailed(f"{where}: full-context cases are not the registered sample")
    if s["loop"] == "note":
        eps = [r.get("case_id") for r in by.get("episodes", [])]
        if eps != ids[:s["episodes"]]:
            raise CheckFailed(f"{where}: episodes are not the registered sample in order")
        for r in by.get("search_trace", []):
            if any(not isinstance(i, int) or not 1 <= i <= len(ids)
                   for i in r.get("episodes") or [None]):
                raise CheckFailed(f"{where}: search-control evidence outside the sample")
    else:
        sample = set(ids)
        for source in ("trace", "search_trace"):
            for r in by.get(source, []):
                if r.get("type") == "step" and not set(r.get("batch") or [None]) <= sample:
                    raise CheckFailed(f"{where}: a {source} step used cases outside the "
                                      "registered sample")


def verify_learning(plan: dict[str, Any], learning: list[dict[str, Any]],
                    texts: dict[tuple[Any, ...], str]) -> None:
    """Every delivered learning summary: typed schema, its own registered
    training sample, artifact hashes equal to the exported texts, sentinel
    selection consistent with them, budgets within the registered ones, and the
    per-run trace rows (proposals, verdicts, budget accounting) it summarises."""
    rows: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list))
    for r in learning:
        rows[(r["task"], r["run"], r["writer"])][r["source"]].append(r["row"])
    registered = (plan.get("learning") or {}).get("budgets")
    samples = (plan.get("learning") or {}).get("train_samples") or {}
    if registered is None:
        raise CheckFailed("the plan registers no learning budgets")
    for (task, run, writer), by in sorted(rows.items(), key=str):
        where = f"learning {task}/{writer}/run-{run}"
        if len(by.get("summary", [])) != 1:
            raise CheckFailed(f"{where}: no summary (or more than one)")
        s = sc.check_object(where, by["summary"][0], sc.SUMMARY_FIELDS)
        if (s["task"], s["run"], s["writer"]) != (task, run, writer):
            raise CheckFailed(f"{where}: summary names another run")
        ids = samples.get(task, {}).get("runs", {}).get(str(run))
        if ids is None or s["train_case_ids_sha256"] != pc.sha_text(pc.canonical(ids)) or \
                s["train_cases"] != len(ids):
            raise CheckFailed(f"{where}: not trained on its registered sample")
        for kind, field in (("learned", "skill_sha256"), ("generic", "generic_sha256"),
                            ("full_context", "evidence_sha256"),
                            ("search_control", "search_control_sha256")):
            text = texts.get((task, run, writer, kind))
            if text is None:
                raise CheckFailed(f"{where}: exported {kind} text missing")
            if (pc.sha_text(text) if text else None) != s[field]:
                raise CheckFailed(f"{where}: {kind} text differs from its summary hash")
        if s["selection"]["learned"]["selected_sha256"] != s["skill_sha256"] or \
                s["selection"]["search_control"]["selected_sha256"] != \
                s["search_control_sha256"]:
            raise CheckFailed(f"{where}: selected snapshot is not the exported text")
        used = s["budgets"]
        for comp, limits in registered.items():
            if comp == "total":
                continue
            for k in sc.BUDGET_KEYS:
                if used[comp][f"limit_{k}"] != int(limits[k]) or used[comp][k] > int(limits[k]):
                    raise CheckFailed(f"{where}: {comp} {k} outside the registered budget")
        for k in sc.BUDGET_KEYS:
            if sum(used[c][k] for c in used if c != "total") > int(registered["total"][k]):
                raise CheckFailed(f"{where}: total {k} over the registered budget")
        if s["evidence_cases"] > used["acquisition"]["decisions"]:
            raise CheckFailed(f"{where}: full context counts more cases than it decided")
        main = "trace" if s["loop"] == "ops" else "patches"
        _trace_ok(by.get(main, []), s["loop"], s["proposals"], where)
        _trace_ok(by.get("search_trace", []), s["loop"], s["search_control"]["proposals"],
                  f"{where} (search control)")
        _sample_use(by, s, ids, where)
        sentinel = int((plan["cases"][task].get("sentinel") or {}).get("rows", 0))
        if s["sentinel"] != sentinel:
            raise CheckFailed(f"{where}: sentinel size differs from the plan")
        verify_selection(s["selection"]["learned"], _states(by.get(main, []), s["loop"]),
                         sentinel, used["selection_learned"]["decisions"], where,
                         int(registered["selection_learned"]["decisions"]))
        verify_selection(s["selection"]["search_control"],
                         _states(by.get("search_trace", []), s["loop"]), sentinel,
                         used["selection_search"]["decisions"], f"{where} (search control)",
                         int(registered["selection_search"]["decisions"]))
        if s["loop"] == "note" and len(by.get("episodes", [])) != s["episodes"]:
            raise CheckFailed(f"{where}: episode rows do not match the summary")


# --------------------------------------------------------------------------- check


def _accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"accuracy": None, "group_accuracy": None, "wrong_write": None}
    groups: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(int(bool(r["score"]["correct"])))
    return {"accuracy": round(statistics.mean(int(bool(r["score"]["correct"]))
                                              for r in rows), 4),
            "group_accuracy": round(statistics.mean(statistics.mean(v)
                                                    for v in groups.values()), 4),
            "wrong_write": round(statistics.mean(int(bool(r["score"].get("wrong_write")))
                                                 for r in rows), 4)}


def check(exports: Path | list[Path], registration: dict[str, Any], keys: dict[str, Path], *,
          plan: dict[str, Any], membership: dict[str, Any], reveal_incomplete: bool = False,
          allow_dry: bool = False, allow_smoke: bool = False) -> dict[str, Any]:
    exports = [exports] if isinstance(exports, str | Path) else list(exports)
    report: dict[str, Any] = {"schema": "xp-partner-check-v2",
                              "exports": [str(e) for e in exports]}
    try:
        body = _check(exports, registration, keys, plan, membership, reveal_incomplete,
                      allow_dry, allow_smoke)
    except (CheckFailed, sc.SchemaError) as exc:
        return {**report, "status": "failed", "reason": str(exc)}
    except (ValueError, KeyError, TypeError, OSError) as exc:  # malformed input: fail closed
        return {**report, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    return {**report, **body}


def _check(exports: list[Path], registration: dict[str, Any], key_paths: dict[str, Path],
           plan: dict[str, Any], membership: dict[str, Any], reveal_incomplete: bool,
           allow_dry: bool, allow_smoke: bool = False) -> dict[str, Any]:
    if plan is None or membership is None:
        raise CheckFailed("the registered plan and our test membership are required")
    pp.verify(plan)
    if plan["sha256"] != registration.get("plan_sha256") or \
            plan["plan_id"] != registration.get("plan_id"):
        raise CheckFailed("the plan passed to the checker is not the registered plan")
    if not exports:
        raise CheckFailed("no export given")
    members = load_membership(membership, registration, plan)
    keys = load_keys(key_paths, members)
    shards = [read_shard(e, registration, plan) for e in exports]
    merged = merge(shards, registration, allow_dry=allow_dry)
    smoke = shards[0]["hashes"]["run_identity"]["smoke"]
    if smoke is not None:
        if not allow_smoke:
            raise CheckFailed("the delivery is a smoke run (rehearsal setting)")
        if smoke != pp.SMOKE:
            raise CheckFailed("unknown smoke setting")
        members = {t: pp.smoke_ids(ids, smoke) for t, ids in members.items()}
    plan = pp.effective(plan, smoke)  # registered plan already verified above
    delivered = {t for s in shards for t in s["tasks"]}
    test_ids = {cid for ids in members.values() for cid in ids}
    kinds = {case["kind"] for key in keys.values() for case in key.values()}
    extras = [(f"{s['path']}/env.json", s["env"]) for s in shards] + \
        [(f"{s['path']}/manifest.json", s["manifest"]) for s in shards]
    scan(merged, extras, signatures(keys), test_ids, kinds,
         atoms_index(keys, membership.get("distinctive_addresses")))

    arms = merged["arms.jsonl.gz"]
    skills = merged["skills.jsonl.gz"]
    learning = merged["learning.jsonl.gz"]
    decisions = merged["decisions.jsonl.gz"]
    gaps: list[dict[str, Any]] = [{"task": t, "problem": "task not delivered"}
                                  for t in plan["tasks"] if t not in delivered]

    # skills: unique identities, hashes, planned learning runs only
    learn_slots = set(pp.learning_slots(plan))
    texts: dict[tuple[Any, ...], str] = {}
    for s in skills:
        ident = (s["task"], s["run"], s["writer"], s["kind"])
        if ident in texts:
            raise CheckFailed(f"duplicate skill text {ident}")
        if (s["task"], s["run"], s["writer"]) not in learn_slots:
            raise CheckFailed(f"skill text for an unplanned learning run {ident}")
        if (pc.sha_text(s["text"]) if s["text"] else None) != s["sha256"]:
            raise CheckFailed(f"skill text hash differs {ident}")
        texts[ident] = s["text"]
    # learning provenance: one summary per planned learning run of a delivered task
    summaries = Counter((r["task"], r["run"], r["writer"]) for r in learning
                        if r["source"] == "summary")
    for ident, n in summaries.items():
        if n > 1:
            raise CheckFailed(f"duplicate learning summary {ident}")
    for r in learning:
        if (r["task"], r["run"], r["writer"]) not in learn_slots:
            raise CheckFailed(f"learning rows for an unplanned run {(r['task'], r['run'])}")
    for t, r, w in sorted(learn_slots, key=str):
        if t in delivered and (t, r, w) not in summaries:
            gaps.append({"slot": [t, r, w], "problem": "learning provenance missing"})
    verify_learning(plan, learning, texts)

    # arms: exact match with the plan
    want = {pp.slot_key(s): s for s in pp.instance_slots(plan)}
    plan_arms = {a["name"]: a for a in plan["arms"]}
    slots: dict[tuple[Any, ...], dict[str, Any]] = {}
    for a in arms:
        k = pp.slot_key(a)
        if k in slots:
            raise CheckFailed(f"duplicate arm instance {k}")
        spec = want.get(k)
        if spec is None:
            raise CheckFailed(f"arm instance not in the plan: {k}")
        if a["kind"] != spec["kind"] or a["n"] != spec["n"]:
            raise CheckFailed(f"arm instance {k}: kind/n differ from the plan")
        if spec["kind"] in pp.WRITER_TEXT_KINDS:
            text = texts.get((a["task"], a["run"], a["writer"], a["kind"]))
            if text is None:
                raise CheckFailed(f"no exported text for writer arm {k}")
        else:
            text = plan_arms[a["arm"]].get("text", "") if spec["kind"] == "notes" else ""
        if pp.prompt_hash(spec["kind"], text, spec["n"]) != a["prompt_hash"] or \
                a["skill_sha256"] != (pc.sha_text(text) if text else None):
            raise CheckFailed(f"arm instance {k}: prompt hash does not match its text")
        slots[k] = a
    for k in sorted(set(want) - set(slots), key=str):
        if k[0] in delivered:
            gaps.append({"slot": list(k), "problem": "arm instance missing"})

    # decisions
    by_prompt: dict[tuple[str, str], set[tuple[Any, ...]]] = defaultdict(set)
    for k, a in slots.items():
        by_prompt[(a["task"], a["prompt_hash"])].add(k)
    scored: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    mismatches = []
    for row in decisions:
        task, cid = row["task"], row["case_id"]
        case = keys[task].get(cid)
        if case is None:
            raise CheckFailed(f"{task}/{cid}: not a registered test case")
        if row["case_hash"] != case["case_hash"]:
            raise CheckFailed(f"{task}/{cid}: case_hash differs from our answer key")
        pkey = (task, row["prompt_hash"])
        if pkey not in by_prompt:
            raise CheckFailed(f"{task}/{cid}: decision for a prompt no arm uses")
        if (task, row["run"], row["writer"], row["arm"]) not in by_prompt[pkey]:
            raise CheckFailed(f"{task}/{cid}: arm label does not match its prompt hash")
        bucket = scored[pkey]
        if cid in bucket:
            raise CheckFailed(f"{task}/{cid}: duplicate decision for prompt "
                              f"{row['prompt_hash'][:12]}")
        score_fn, _ = pc.scorer(task, case)
        s = score_fn(case, row["action"])
        if "reported_correct" in row and bool(row["reported_correct"]) != bool(s["correct"]):
            mismatches.append({"task": task, "case_id": cid})
        bucket[cid] = {"case_id": cid, "group": case["group"], "kind": case["kind"],
                       "score": s, "error": row["error"]}
    if mismatches:
        raise CheckFailed(f"{len(mismatches)} reported scores differ from ours "
                          f"(first: {mismatches[0]})")

    table = []
    for k, a in sorted(slots.items(), key=lambda kv: str(kv[0])):
        rows = scored.get((a["task"], a["prompt_hash"]), {})
        missing = sorted(set(members[a["task"]]) - set(rows))
        if missing:
            gaps.append({"slot": list(k), "problem": "cases missing", "missing": len(missing),
                         "first": missing[:3]})
        table.append({"arm": a, "rows": list(rows.values()), "missing": len(missing)})
    complete = not gaps
    reveal = complete or reveal_incomplete
    instances = []
    for inst in table:
        a, rows = inst["arm"], inst["rows"]
        entry = {"task": a["task"], "run": a["run"], "writer": a["writer"], "arm": a["arm"],
                 "kind": a["kind"], "prompt_hash": a["prompt_hash"], "n": len(rows),
                 "missing": inst["missing"], "errors": sum(1 for r in rows if r["error"])}
        if reveal:
            entry.update(_accuracy(rows))
            first = keys[a["task"]][members[a["task"]][0]]
            _, summarize = pc.scorer(a["task"], first)
            entry["summary"] = summarize([r["score"] for r in rows], [r["kind"] for r in rows])
        instances.append(entry)
    pooled = []
    if reveal:
        by: dict[tuple[Any, Any], list[float]] = defaultdict(list)
        for e in instances:
            if e["group_accuracy"] is not None:
                by[(e["writer"], e["arm"])].append(e["group_accuracy"])
        pooled = [{"writer": w, "arm": arm, "instances": len(v),
                   "mean_group_accuracy": round(statistics.mean(v), 4)}
                  for (w, arm), v in sorted(by.items(), key=str)]
    return {"status": "ok" if complete else "incomplete", "coverage_gaps": gaps,
            "scores_concealed": not reveal,
            "shards": [{"path": s["path"], "tasks": s["tasks"], "bytes": s["bytes"]}
                       for s in shards],
            "mode": shards[0]["hashes"]["run_identity"]["mode"],
            "decision_rows": dict(Counter(r["task"] for r in decisions)),
            "instances": instances, "pooled": pooled}


def main(argv: list[str] | None = None) -> int:
    import sys

    ap = argparse.ArgumentParser(prog="python -m slh.xp.partner.check")
    ap.add_argument("--export", type=Path, action="append", required=True)
    ap.add_argument("--registration", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--membership", type=Path, required=True)
    ap.add_argument("--key", action="append", default=[], help="TASK=PATH (repeatable)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--reveal-incomplete", action="store_true")
    ap.add_argument("--allow-dry-run", action="store_true")
    ap.add_argument("--allow-smoke", action="store_true",
                    help="accept a rehearsal smoke run (plan.SMOKE coverage)")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    keys = {}
    for item in args.key:
        task, _, path = item.partition("=")
        keys[task] = Path(path)
    report = check(args.export, pc.read_json(args.registration), keys,
                   plan=pc.read_json(args.plan), membership=pc.read_json(args.membership),
                   reveal_incomplete=args.reveal_incomplete, allow_dry=args.allow_dry_run,
                   allow_smoke=args.allow_smoke)
    out = args.out or args.export[0].parent / "check-report.json"
    pc.write_json(out, report)
    print(json.dumps({k: report.get(k) for k in ("status", "reason", "decision_rows",
                                                 "scores_concealed")}, indent=1))
    print(f"report: {out}")
    return 0 if report["status"] == "ok" else (1 if report["status"] == "incomplete" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
