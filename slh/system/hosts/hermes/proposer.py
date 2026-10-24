"""``HermesStagedProposer``: Hermes' own learning proposals -> typed ``skill`` / ``memory``
candidates (spike rev 2 s. 2 "Learning writes routed through the gate", s. 4.1-4.2).

Input: the evidence outcomes of one batch (only these; the harness never shows a proposer
gate data). Their traces carry the records Hermes STAGED under ``write_approval``
(``pending/{memory,skills}/<id>.json``: ``origin`` and the exact replay ``payload``).

* Only **scored** evidence runs count (an unjoined review fork leaves the run unscored and
  its records unused).
* Records are grouped by **target**: a skill by name (case-folded), memory by target.
  ``user``-target memory writes are dropped (USER.md is disabled on benchmarks).
* Each group is **replayed with Hermes' own approval appliers** (``apply_skill_pending``,
  ``apply_memory_pending``) in record order, on a scratch ``HERMES_HOME`` materialised from
  the incumbent, in a sandboxed Hermes process with no network. ``patch`` / ``replace`` /
  ``remove`` therefore become **full post-states** with Hermes' exact semantics; a record
  that fails to replay is skipped and logged.
* The post-state is read back: one ``skill`` candidate per changed skill (``tombstone`` when
  the skill was deleted; support files listed in ``files``, which the harness drops in V1),
  one ``memory`` candidate when MEMORY changed. Every candidate of one call has a distinct
  target (conflict exclusion at the source; the pool enforces it again).

Free: no model call (the fork's tokens were charged to the evidence runs).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from slh.state import HarnessState
from slh.system.hosts.hermes import materialize as MZ
from slh.system.hosts.hermes import sandbox as SB
from slh.system.hosts.hermes.adapter import RUNNER, HermesHost
from slh.system.interfaces import MEMORY_DELIMITER, Candidate, Outcome


def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(s: str | bytes) -> str:
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def parse_skill(text: str) -> tuple[dict[str, Any], str]:
    """(frontmatter, body) of a SKILL.md text; no frontmatter -> ({}, text)."""
    t = text.replace("\r\n", "\n")
    if t.startswith("---\n"):
        end = t.find("\n---", 4)
        if end != -1:
            try:
                fm = yaml.safe_load(t[4:end]) or {}
            except yaml.YAMLError:
                fm = {}
            body = t[end + 4:]
            body = body.removeprefix("\n")
            return (fm if isinstance(fm, dict) else {}), body
    return {}, t


def record_target(rec: dict[str, Any]) -> str | None:
    """Conflict key of one staged record, or None when it is not admissible."""
    p = rec.get("payload") or {}
    if rec.get("subsystem") == "skills":
        name = p.get("name")
        return f"skill:{str(name).casefold()}" if isinstance(name, str) and name else None
    if rec.get("subsystem") == "memory":
        targets = {op.get("target", "memory") for op in (p.get("operations") or [p])
                   if isinstance(op, dict)}
        return "memory:MEMORY" if targets == {"memory"} else None
    return None


@dataclass
class HermesStagedProposer:
    host: HermesHost = field(default_factory=HermesHost)
    config_extra: dict[str, Any] | None = None
    soul: str = MZ.DEFAULT_SOUL
    work_root: Path | None = None
    sandbox: bool = True
    replay_wall_s: float = 120.0
    keep_runs: bool = False
    paid: bool = False
    last_cost_usd: float = 0.0
    log: list[dict[str, Any]] = field(default_factory=list)

    def estimate_cost_usd(self) -> float:
        return 0.0

    # ------------------------------------------------------------------ propose

    def propose(self, batch: int, evidence: Sequence[Outcome], incumbent: HarnessState,
                retired: set[str]) -> list[Candidate]:
        self.last_cost_usd = 0.0
        groups: dict[str, list[dict[str, Any]]] = {}
        units: dict[str, list[str]] = {}
        for o in evidence:
            if o.role != "evidence" or not o.scored or not o.trace:
                continue
            for rec in o.trace.get("staged") or []:
                if not isinstance(rec, dict) or not rec.get("payload"):
                    continue
                key = record_target(rec)
                if key is None:
                    self.log.append({"batch": batch, "drop": "inadmissible target",
                                     "record": rec.get("id"), "unit": o.unit_id})
                    continue
                groups.setdefault(key, []).append(dict(rec))
                units.setdefault(key, []).append(o.unit_id)
        out: list[Candidate] = []
        for key in sorted(groups):
            cand = self._candidate(batch, key, groups[key], units[key], incumbent)
            if cand is not None and cand.cid not in retired:
                out.append(cand)
        return out

    def _candidate(self, batch: int, key: str, recs: list[dict[str, Any]], units: list[str],
                   incumbent: HarnessState) -> Candidate | None:
        home, results = self.replay(incumbent, recs)
        try:
            kind, name = key.split(":", 1)
            prov = {"source": "hermes-staged", "batch": batch, "target": key,
                    "records": [r.get("id") for r in recs], "units": sorted(set(units)),
                    "origins": sorted({str(r.get("origin")) for r in recs}),
                    "replay": results}
            if not any(x["ok"] for x in results):
                self.log.append({"batch": batch, "drop": "nothing replayed", "target": key,
                                 "results": results})
                return None
            if kind == "skill":
                payload = self._skill_post_state(home, name, incumbent)
            else:
                payload = self._memory_post_state(home, incumbent)
            if payload is None:
                self.log.append({"batch": batch, "drop": "no change", "target": key})
                return None
            cid = f"{kind}-{_slug(payload.get('name', 'memory'))}-{_sha(_canon(payload))[:10]}"
            return Candidate(cid, "skill" if kind == "skill" else "memory", payload,
                             ("any",), prov)
        finally:
            if not self.keep_runs:
                SB.rmtree(home.parent)

    @staticmethod
    def _skill_post_state(home: Path, folded: str, inc: HarnessState) -> dict[str, Any] | None:
        hits = [p for p in sorted((home / "skills").rglob("SKILL.md"))
                if p.parent.name.casefold() == folded]
        before = {k.casefold(): (k, v) for k, v in inc.skills.items()}
        if not hits:
            if folded in before:
                return {"name": before[folded][0], "tombstone": True}
            return None
        skill_dir = hits[0].parent
        fm, body = parse_skill(hits[0].read_text(encoding="utf-8"))
        name = skill_dir.name
        payload: dict[str, Any] = {"name": name,
                                   "description": " ".join(str(fm.get("description") or
                                                               "").split()),
                                   "body": body}
        files = {p.relative_to(skill_dir).as_posix(): _sha(p.read_bytes())
                 for p in sorted(skill_dir.rglob("*")) if p.is_file() and p.name != "SKILL.md"}
        if files:
            payload["files"] = files
        from slh.system.interfaces import render_skill

        if (not files and folded in before
                and before[folded][1] == render_skill(name, payload["description"], body)):
            return None
        return payload

    @staticmethod
    def _memory_post_state(home: Path, inc: HarnessState) -> dict[str, Any] | None:
        p = home / "memories" / "MEMORY.md"
        text = p.read_text(encoding="utf-8") if p.exists() else ""
        entries = [e.strip() for e in text.split(MEMORY_DELIMITER) if e.strip()]
        cur = [n for n in inc.notes if n.get("kind") == "memory" and n.get("target") == "MEMORY"]
        if cur and [str(e) for e in cur[0]["entries"]] == entries:
            return None
        if not cur and not entries:
            return None
        return {"target": "MEMORY", "entries": entries}

    # ------------------------------------------------------------------ replay

    def replay(self, incumbent: HarnessState, recs: list[dict[str, Any]]
               ) -> tuple[Path, list[dict[str, Any]]]:
        """Replay ``recs`` with Hermes' appliers on a scratch home; (home, per-record
        results). The caller removes ``home.parent``."""
        root = Path(tempfile.mkdtemp(prefix="slh-r-", dir=str(self.work_root or "/tmp")))
        root = root.resolve()
        home, outd, tmp, uhome = root / "home", root / "out", root / "tmp", root / "u"
        for d in (outd, tmp, uhome):
            d.mkdir()
        MZ.materialize(incumbent, home, MZ.host_config(extra=self.config_extra), self.soul)
        MZ.make_writable(home)  # a scratch copy: the replay writes into it
        ep = {"mode": "replay", "records": recs, "out_dir": str(outd), "model": {}}
        ep_file = root / "episode.json"
        ep_file.write_text(json.dumps(ep))
        from slh.system.hosts.hermes.adapter import SECRET_ROOTS

        # no sockets at all during replay (review fix 6)
        prof = SB.contain(writable=[root], readonly=[ep_file], model_port=None,
                          readable=[root, self.host.clone, self.host.venv, RUNNER,
                                    *SB.interpreter_roots(self.host.python)],
                          sockets=(), secret_roots=SECRET_ROOTS) if self.sandbox else None
        env = SB.scrubbed_env({"HERMES_HOME": str(home)}, home=uhome, tmp=tmp)
        done = SB.run_contained([str(self.host.python), "-I", "-B", str(RUNNER), str(ep_file)],
                                env=env, cwd=root, profile_text=prof,
                                deadline_s=self.replay_wall_s, log_dir=root / "log")
        try:
            res = json.loads((outd / "out.json").read_text())["results"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            res = [{"id": r.get("id"), "ok": False,
                    "error": f"replay process failed rc={done.returncode} "
                             f"killed={done.killed}: {done.stderr[-500:]}"} for r in recs]
        return home, res


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in str(name))[:40]
