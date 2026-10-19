"""ALFWorld (text-only) feasibility at the partner: the frozen 9B on 48 development games.

    python -m slh.xp.alfworld.feasibility select --data $ALFWORLD_DATA            # prints the games
    python -m slh.xp.alfworld.feasibility run --data $ALFWORLD_DATA --out runs/alfworld-dry --dry-run
    python -m slh.xp.alfworld.feasibility run --data $ALFWORLD_DATA --endpoint endpoint-bfcl.yaml \\
        --out runs/alfworld-smoke --smoke
    python -m slh.xp.alfworld.feasibility run --data $ALFWORLD_DATA --endpoint endpoint-bfcl.yaml \\
        --out runs/alfworld-feas
    python -m slh.xp.alfworld.feasibility export --run runs/alfworld-feas --out export-alfworld-feas

Step 7 of the owner decisions of 2026-10-18 (docs/xp/bfcl-plan.md s. 9): measure the
headroom of the standard harness before ALFWorld is adopted as a second domain.

Games: ``json_2.1.1/valid_train`` (200 games of the training distribution, disjoint from
``train``, ``valid_seen`` and ``valid_unseen`` by directory; checked 2026-10-18 on the
release archive), **8 per task type x 6 types = 48**, chosen by a salted SHA-256 rank of
the game's relative path (``SALT``), never by any outcome. ``valid_seen`` and
``valid_unseen`` (the later transfer endpoints) are never touched here; these 48 games
are excluded from every later discovery or gate pool.

Standard harness (arm B1, the same for every later arm): one system message with the
game's own command list (the ``help`` text of ALFWorld 0.4.2's ``alfred.twl2``), the
initial observation (the TextWorld banner removed) as the first user message, then one
command per assistant message and the game's feedback as the next user message. No
admissible-command list, no examples, no notes. At most ``MAX_STEPS`` = 50 commands; the
episode ends when the game reports ``won`` or at the limit. Decoding as the BFCL track
(owner decision 1): temperature 0.7, top-p 0.8, non-thinking, per-request seed.

Environment: ``textworld.gym.register_game(game.tw-pddl, EnvInfos(won=True),
max_episode_steps=50, wrappers=[AlfredDemangler(shuffle=False)])`` exactly as ALFWorld's
``alfworld-play-tw`` script does (**unverified** until the partner's dry run; the dry run
plays three fixed commands in every game with no model and no GPU).

Outputs per game (``games.jsonl``): success, steps, invalid commands ("Nothing
happens."), the longest run of one repeated command, failure class, tokens, wall time and
the full transcript (commands, feedback, raw replies). A request the server rejects by its
own request validation (HTTP 400/413/422 with vLLM's OpenAI-format error body, e.g. the
context window exceeded on a long game) ends that game at once as a recorded failure
(class ``request_rejected``, reason ``request_rejected:<code>``; never retried: a rerun
gets the same answer); ``summary.json`` counts it as a failed game and reports it
separately. ``export`` writes
``summary.json`` (success overall and per type, failure classes, the pre-declared go /
no-go) next to the rows. Exit 3 = halted (the endpoint is unreachable; ``halt.json``);
a rerun resumes (finished games are kept).

Go / no-go (pre-declared, s. 9 of the plan): proceed only if success is between 25% and
80% and at least a quarter of the failures are procedural (not format failures: an
episode where under 30% of commands were invalid and the reply was parseable).

Licences: ALFWorld (MIT) and TextWorld (MIT) are imported here; TextWorld loads
Jericho (GPL-2.0-or-later) and Fast Downward (GPL-3.0, ``libdownward.so``) as separately
installed, unmodified third-party packages. This module does not import, copy or
modify either GPL component (``check`` only verifies that they are installed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

SCHEMA = "xp-alfworld-feasibility-v1"
SPLIT = "valid_train"
SALT = "slh-alfworld-feasibility-v1"
TASK_TYPES = ("pick_and_place_simple", "look_at_obj_in_light",
              "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
              "pick_cool_then_place_in_recep", "pick_two_obj_and_place")
PER_TYPE = 8
MAX_STEPS = 50
SMOKE_GAMES = 2
SERVED_MODEL = "qwen3.5-9b-vllm"
DECODING = {"temperature": 0.7, "top_p": 0.8, "max_tokens": 128}
EXIT_HALT, EXIT_REFUSED = 3, 4
DRY_COMMANDS = ("look", "inventory", "examine room")
INVALID = "Nothing happens."
GO = {"success_min": 0.25, "success_max": 0.80, "procedural_share_min": 0.25,
      "format_invalid_share": 0.30}

# The command list printed by the game's own ``help`` command (ALFWorld 0.4.2,
# alfworld/data/alfred.twl2), verbatim.
COMMANDS = (
    "Available commands:\n"
    "  look:                             look around your current location\n"
    "  inventory:                        check your current inventory\n"
    "  go to (receptacle):               move to a receptacle\n"
    "  open (receptacle):                open a receptacle\n"
    "  close (receptacle):               close a receptacle\n"
    "  take (object) from (receptacle):  take an object from a receptacle\n"
    "  move (object) to (receptacle):  place an object in or on a receptacle\n"
    "  examine (something):              examine a receptacle or an object\n"
    "  use (object):                     use an object\n"
    "  heat (object) with (receptacle):  heat an object using a receptacle\n"
    "  clean (object) with (receptacle): clean an object using a receptacle\n"
    "  cool (object) with (receptacle):  cool an object using a receptacle\n"
    "  slice (object) with (object):     slice an object using a sharp object\n")
SYSTEM = ("You are in a household and must complete the task stated in the first "
          "message by typing text commands. After each command you will see what "
          "happened.\n\n" + COMMANDS + "\nReply with exactly one command and nothing else.")


class Refused(RuntimeError):
    pass


class Halt(RuntimeError):
    pass


class Rejected(RuntimeError):
    """The server rejected the request by its own validation (deterministic)."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"HTTP {code}: {message}")
        self.code = code
        self.message = message


# --------------------------------------------------------------------------- games


def task_type(rel: str) -> str:
    """The task type from the game directory (``<type>-<objects>-<scene>/trial_.../``)."""
    return rel.split("/")[0].split("-")[0]


def all_games(data: Path, split: str = SPLIT) -> list[str]:
    root = Path(data) / "json_2.1.1" / split
    return sorted(p.parent.relative_to(root).as_posix()
                  for p in root.rglob("game.tw-pddl"))


def select(games: list[str]) -> list[dict[str, str]]:
    """8 per task type by a salted hash rank; play order interleaves the types."""
    def rank(rel: str) -> str:
        return hashlib.sha256(f"{SALT}:{rel}".encode()).hexdigest()

    by_type: dict[str, list[str]] = {t: [] for t in TASK_TYPES}
    for g in games:
        if task_type(g) in by_type and "movable" not in g and "Sliced" not in g:
            by_type[task_type(g)].append(g)
    short = {t: len(v) for t, v in by_type.items() if len(v) < PER_TYPE}
    if short:
        raise Refused(f"too few games per type: {short}")
    chosen = {t: sorted(v, key=rank)[:PER_TYPE] for t, v in by_type.items()}
    return [{"game": chosen[t][i], "type": t} for i in range(PER_TYPE) for t in TASK_TYPES]


def selection_sha(sel: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(sel, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- env + model


def make_env(gamefile: Path) -> Any:  # pragma: no cover - needs the ALFWorld install
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

    infos = textworld.EnvInfos(won=True)
    env_id = textworld.gym.register_game(str(gamefile), infos, max_episode_steps=MAX_STEPS,
                                         wrappers=[AlfredDemangler(shuffle=False)])
    return textworld.gym.make(env_id)


def strip_banner(obs: str) -> str:
    parts = obs.split("\n\n")
    return "\n\n".join(parts[1:]).strip() if "Welcome to TextWorld" in parts[0] else obs.strip()


def parse_command(reply: str) -> str | None:
    for line in (reply or "").splitlines():
        line = line.strip().strip("`").strip()
        line = re.sub(r"^(>|action:|command:)\s*", "", line, flags=re.I).strip()
        if line:
            return line.rstrip(".").lower()
    return None


REJECT_STATUS = (400, 413, 422)  # request validation; recorded, never retried


def validation_error(status: int, data: bytes) -> str | None:
    """The message when ``data`` is vLLM's OpenAI-format error JSON for ``status``
    (``{"object": "error", "message", "type", "code"}`` or ``{"error": {...}}``), else
    None (HTML, a proxy's own body, a code that disagrees with the status)."""
    try:
        doc = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    if doc.get("object") == "error":
        err = doc
    elif set(doc) == {"error"} and isinstance(doc["error"], dict):
        err = doc["error"]
    else:
        return None
    ok = (isinstance(err.get("message"), str) and bool(err["message"].strip())
          and isinstance(err.get("type"), str) and bool(err["type"])
          and type(err.get("code")) is int and err["code"] == status)
    return err["message"] if ok else None


def chat(endpoint: dict[str, Any], messages: list[dict[str, str]], seed: int,
         retries: int = 3) -> dict[str, Any]:
    body = {"model": endpoint["model"], "messages": messages, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False}, **DECODING}
    key = os.environ.get(endpoint.get("api_key_env") or "", "") or "EMPTY"
    req = urllib.request.Request(endpoint["base_url"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:  # before URLError, its superclass
            try:
                data = exc.read()
            except Exception:  # noqa: BLE001 - an unreadable body is not vLLM's answer
                data = b""
            msg = validation_error(exc.code, data) if exc.code in REJECT_STATUS else None
            if msg is not None:
                raise Rejected(exc.code, msg) from None
            last = exc  # 5xx, 408/429, a proxy's own 4xx: retried, then halt
            time.sleep(2 * (attempt + 1))
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise Halt(f"endpoint unreachable: {type(last).__name__}: {last}"[:300])


def dry_model(step: int) -> dict[str, Any]:
    cmd = DRY_COMMANDS[step % len(DRY_COMMANDS)]
    return {"choices": [{"message": {"content": cmd}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}}


def play(gamefile: Path, rel: str, *, endpoint: dict[str, Any] | None,
         env_factory: Callable[[Path], Any] = make_env, max_steps: int = MAX_STEPS
         ) -> dict[str, Any]:
    """One episode. ``endpoint`` None = dry run (three fixed commands, no model)."""
    t0 = time.time()
    env = env_factory(gamefile)
    obs, info = env.reset()
    first = strip_banner(obs)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": first}]
    steps, invalid, won, unparsed = [], 0, bool(info.get("won")), 0
    prompt_tok = completion_tok = 0
    seed0 = int(hashlib.sha256(rel.encode()).hexdigest()[:8], 16)
    limit = len(DRY_COMMANDS) if endpoint is None else max_steps
    rejected: Rejected | None = None
    for i in range(limit):
        try:
            resp = dry_model(i) if endpoint is None else chat(endpoint, messages, seed0 + i)
        except Rejected as exc:  # terminal for this game; recorded, not retried
            rejected = exc
            break
        reply = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        usage = resp.get("usage") or {}
        prompt_tok += usage.get("prompt_tokens") or 0
        completion_tok += usage.get("completion_tokens") or 0
        cmd = parse_command(reply)
        if cmd is None:
            unparsed += 1
            cmd = "look"  # an empty reply still costs a step; recorded as unparsed
        obs, _score, done, info = env.step(cmd)
        obs = str(obs).strip()
        invalid += obs.startswith(INVALID)
        steps.append({"reply": reply, "command": cmd, "feedback": obs})
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": obs}]
        won = bool(info.get("won"))
        if won or done:
            break
    run = best = 0
    for a, b in zip([None] + [s["command"] for s in steps], [s["command"] for s in steps]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    n = len(steps)
    extra: dict[str, Any] = {}
    if rejected is not None:
        cls = "request_rejected"  # the server refused a request (e.g. context overflow)
        extra = {"reason": f"request_rejected:{rejected.code}",
                 "rejection": rejected.message[:500]}
    elif won:
        cls = "success"
    elif n and (invalid + unparsed) / n >= GO["format_invalid_share"]:
        cls = "format"           # commands the game does not accept, or no command
    elif best >= 4:
        cls = "loop"             # the same command four or more times in a row
    else:
        cls = "procedure"        # valid commands that did not complete the task
    return {"game": rel, "type": task_type(rel), "won": won, "steps": n,
            "invalid": invalid, "unparsed": unparsed, "max_repeat": best,
            "class": cls, "prompt_tokens": prompt_tok, "completion_tokens": completion_tok,
            "wall_s": round(time.time() - t0, 2), "first_observation": first,
            "transcript": steps, **extra}


def _play_job(args: tuple[str, str, dict[str, Any] | None]) -> dict[str, Any]:
    gamefile, rel, endpoint = args
    try:
        return play(Path(gamefile), rel, endpoint=endpoint)
    except Halt as exc:
        return {"game": rel, "halt": str(exc)}
    except Exception as exc:  # noqa: BLE001 - an environment error fails the game, recorded
        return {"game": rel, "type": task_type(rel), "won": False, "steps": 0,
                "class": "env_error", "error": f"{type(exc).__name__}: {exc}"[:500]}


# --------------------------------------------------------------------------- run / export


def load_endpoint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    import yaml

    ep = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if ep.get("model") != SERVED_MODEL or not ep.get("base_url"):
        raise Refused(f"endpoint needs base_url and model {SERVED_MODEL!r}")
    from slh.xp.partner.common import serving_sampling_problems

    problems = serving_sampling_problems(ep.get("serving_args"))
    if problems:  # decoding is sent per request (DECODING); the server adds nothing
        raise Refused("; ".join(problems))
    return {k: ep.get(k) for k in ("base_url", "model", "api_key_env", "serving_args",
                                   "replicas", "model_revision", "vllm_version")}


def _done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {json.loads(line)["game"] for line in path.read_text().splitlines() if line.strip()}


def run(data: Path, out: Path, *, endpoint: dict[str, Any] | None, smoke: bool,
        workers: int, job: Callable[[tuple], dict[str, Any]] = _play_job) -> int:
    sel = select(all_games(data))
    if smoke:
        sel = sel[:SMOKE_GAMES]
    mode = "dry" if endpoint is None else ("smoke" if smoke else "full")
    ident = {"schema": SCHEMA, "mode": mode, "selection_sha256": selection_sha(sel),
             "endpoint": endpoint, "decoding": DECODING, "max_steps": MAX_STEPS,
             "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest()}
    out.mkdir(parents=True, exist_ok=True)
    idp = out / "identity.json"
    if idp.is_file() and json.loads(idp.read_text()) != ident:
        raise Refused(f"{out} belongs to another run; use a new --out")
    idp.write_text(json.dumps(ident, indent=1, sort_keys=True))
    (out / "selection.json").write_text(json.dumps(sel, indent=1))
    rows = out / "games.jsonl"
    done = _done(rows)
    todo = [s for s in sel if s["game"] not in done]
    root = Path(data) / "json_2.1.1" / SPLIT
    t0 = time.time()
    halted = None
    jobs = [(str(root / s["game"] / "game.tw-pddl"), s["game"], endpoint) for s in todo]

    def keep(row: dict[str, Any]) -> None:
        nonlocal halted
        if "halt" in row:
            halted = halted or row["halt"]
            return
        with rows.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    if workers <= 0:  # in-process (tests)
        for j in jobs:
            keep(job(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for f in as_completed([pool.submit(job, j) for j in jobs]):
                keep(f.result())
    status = {"games": len(sel), "finished": len(_done(rows)), "wall_s": round(time.time() - t0, 1)}
    (out / "status.json").write_text(json.dumps(status, indent=1))
    if halted:
        (out / "halt.json").write_text(json.dumps({"reason": halted}, indent=1))
        print(json.dumps({"halted": halted}))
        return EXIT_HALT
    print(json.dumps(status))
    return 0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    wins = sum(r["won"] for r in rows)
    fails = [r for r in rows if not r["won"]]
    classes = Counter(r["class"] for r in fails)
    per_type = {t: {"games": sum(r["type"] == t for r in rows),
                    "won": sum(r["won"] for r in rows if r["type"] == t)} for t in TASK_TYPES}
    success = wins / n if n else None
    proc_share = (classes.get("procedure", 0) + classes.get("loop", 0)) / len(fails) \
        if fails else None
    rejected = Counter(r.get("reason", "request_rejected") for r in fails
                       if r["class"] == "request_rejected")
    go = (success is not None and GO["success_min"] <= success <= GO["success_max"]
          and proc_share is not None and proc_share >= GO["procedural_share_min"]
          and classes.get("env_error", 0) == 0)
    return {"games": n, "won": wins, "success": success, "per_type": per_type,
            "failure_classes": dict(classes), "procedural_share_of_failures": proc_share,
            # counted above as failed games; reported separately (request validation)
            "request_rejected": {"games": sum(rejected.values()),
                                 "by_reason": dict(sorted(rejected.items()))},
            "mean_steps": sum(r["steps"] for r in rows) / n if n else None,
            "invalid_command_share": (sum(r.get("invalid", 0) for r in rows)
                                      / max(1, sum(r["steps"] for r in rows))),
            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rows),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in rows),
            "go_rule": GO, "go": go}


def export(run_dir: Path, out: Path) -> dict[str, Any]:
    import platform
    import shutil
    from importlib import metadata

    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (run_dir / "games.jsonl").read_text().splitlines()
            if line.strip()]
    for name in ("games.jsonl", "identity.json", "selection.json", "status.json",
                 "halt.json"):
        if (run_dir / name).is_file():
            shutil.copy(run_dir / name, out / name)
    summ = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summ, indent=1, sort_keys=True))
    versions = {}
    for dist in ("alfworld", "textworld", "jericho", "fast-downward-textworld", "spacy",
                 "numpy"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    (out / "env.json").write_text(json.dumps({"python": sys.version,
                                              "platform": platform.platform(),
                                              "packages": versions}, indent=1))
    return summ


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.alfworld.feasibility")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--data", type=Path, default=os.environ.get("ALFWORLD_DATA"))
    r = sub.add_parser("run")
    r.add_argument("--data", type=Path, default=os.environ.get("ALFWORLD_DATA"))
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--endpoint", type=Path)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--workers", type=int, default=16)
    e = sub.add_parser("export")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "select":
            sel = select(all_games(Path(args.data)))
            print(json.dumps({"sha256": selection_sha(sel), "games": sel}, indent=1))
            return 0
        if args.cmd == "run":
            if args.data is None:
                raise Refused("set ALFWORLD_DATA or --data")
            if not args.dry_run and args.endpoint is None:
                raise Refused("--endpoint is required (or --dry-run)")
            ep = None if args.dry_run else load_endpoint(args.endpoint)
            return run(Path(args.data), args.out, endpoint=ep, smoke=args.smoke,
                       workers=args.workers)
        print(json.dumps(export(args.run, args.out), indent=1))
        return 0
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
