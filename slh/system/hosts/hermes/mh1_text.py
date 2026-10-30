"""M-H1 text reports: the partner's return as pasted TEXT (owner 2026-10-30: no files can come
back, only about two to three pages of text). ANALYSIS ONLY, run after the run on its
exports: it imports the kit's own functions, writes nothing into a run or export folder, and
is not part of the run's code identity (it is not in the package's recorded ``code_files``;
it ships as a follow-up to the published v2.6 kit). Design s. 23.

    PY=python
    $PY -m slh.system.hosts.hermes.mh1_text part1 --package package-mh1 \\
        --block c9b=export-mh1-c9b --block c27b=export-mh1-c27b \\
        --smoke c9b=export-mh1-c9b-smoke --smoke c27b=export-mh1-c27b-smoke
    #   paste part 1; WAIT for our hold decisions; then:
    $PY -m slh.system.hosts.hermes.mh1_text part2 --package package-mh1 \\
        --block c9b=export-mh1-c9b --block c27b=export-mh1-c27b \\
        --decisions "c9b:k1=approve c27b:k2=reject" --part1-check <its CHECK value>
    #   ours: python -m slh.system.hosts.hermes.mh1_text check --text pasted.txt

A multi-node block takes every export folder of the block, comma separated (learning node
and every eval shard): they are merged first (``mh1.merge_exports``, bytes only).

Part 1 (<= 1 page): identity, per block completion / exit code / timing, the verify verdict
(coverage, identity, sealed sizes, incidents, mediation), the smoke summary, and every held
change with its proposed text, why it is held and the critical-event counts FROM ITS PACKET
ONLY: no eval outcome (the blind hold decision is made from it).

Part 2 (<= 2 pages; after the decisions): the pre-registered analysis WITHOUT the sealed
C_hold file (it cannot come back and our key never leaves us): B-A, C_nohold-A, C_nohold-B
(C_nohold = C's pointer). When no approved passing hold applies, C IS C's pointer and these
are the registered primary contrasts C-A, B-A, C-B; when an approved hold applies, C is that
hold's version, which is NOT computable without the sealed file, and these are reported as
secondaries. Harm non-inferiority labels, per-arm success / harm / critical / non-scored,
degraded flags, tokens, and the S1 probe headroom summary (``s1_probe_ourside``: calibration
on pool C, then the report: oracle envelope, matched-real minus placebo, both calibration
rules, the null-ladder rung, the review-skip rule statistics).

Each part ends with ``CHECK partN <16 hex>``: the SHA-256 of every line above it (trailing
spaces stripped), so a pasted copy can be checked for transcription errors (``check``).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TEXT_SCHEMA = "slh-mh1-text-v1"
EXIT_OF_HALT = {"refused": 4, "incident": 5, "halted": 3, "error": 4}
CHANGE_CHARS = 220  # per proposed change text
MAX_HOLDS_FULL = 6  # holds listed in full per block; the rest one line each


# ----------------------------------------------------------------------------- helpers


def tool_hashes() -> str:
    """This tool's and the frozen probe analysis's own hashes (they are outside the run's
    code identity, so every report names them; we compare with ours)."""
    from slh.xp.ourside import s1_probe_ourside as SO

    me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    return f"tool {me} probe-analysis {SO.module_sha256()[:12]}"


def _norm(text: str) -> list[str]:
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def checksum(body: str) -> str:
    return hashlib.sha256("\n".join(_norm(body)).encode()).hexdigest()[:16]


def _seal(body: list[str], part: str) -> str:
    text = "\n".join(body)
    return text + f"\nCHECK {part} {checksum(text)}\n"


def check_text(text: str) -> dict[str, Any]:
    """Ours: recompute the CHECK line of a pasted part."""
    lines = _norm(text)
    if not lines or not lines[-1].startswith("CHECK "):
        return {"ok": False, "error": "no CHECK line at the end"}
    _, part, want = (lines[-1].split() + ["", ""])[:3]
    got = checksum("\n".join(lines[:-1]))
    return {"ok": got == want, "part": part, "check": want, "recomputed": got}


def _gz(p: Path) -> Any:
    if not p.exists():
        return None
    raw = gzip.decompress(p.read_bytes())
    return json.loads(raw) if raw.strip() else None


def _clip(s: Any, n: int) -> str:
    t = " ".join(str(s).split())
    return t if len(t) <= n else t[:n - 1] + "…"


def _pct(k: int, n: int) -> str:
    return f"{k}/{n} ({k / n:.1%})" if n else f"{k}/0"


def _f(x: Any, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:+.{nd}f}" if nd else str(x)
    return str(x)


def _ci(ci: Any) -> str:
    if not ci or ci[0] is None:
        return "[n/a]"
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def resolve_block(pkg: Any, dirs: Sequence[Path], work: Path) -> Path:
    """One export folder, or a multi-node block merged into one (bytes only)."""
    from slh.system.hosts.hermes import mh1 as M

    dirs = [Path(d) for d in dirs]
    if len(dirs) == 1:
        return dirs[0]
    roles = {d: json.loads((d / "MANIFEST.json").read_text())["identity"].get("role")
             for d in dirs}
    learn = [d for d, r in roles.items() if r == "learn"]
    if len(learn) != 1:
        raise SystemExit("a multi-node block needs exactly one learning-node export")
    out = work / f"merged-{len(list(work.iterdir()))}"
    M.merge_exports(learn[0], [d for d in dirs if d != learn[0]], out, pkg)
    return out


def _blocks(pairs: Sequence[str]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for p in pairs:
        b, _, ds = p.partition("=")
        out[b] = [Path(x) for x in ds.split(",") if x]
    return out


# ----------------------------------------------------------------------------- part 1


def _completion(e: Path) -> tuple[str, str]:
    man = json.loads((e / "MANIFEST.json").read_text())
    halt = _gz(e / "halt.json.gz")
    if halt:
        code = EXIT_OF_HALT.get(str(halt.get("kind")), "?")
        why = _clip(str(halt.get("error") or "").splitlines()[0] if halt.get("error")
                    else "", 90)
        return f"exit {code} ({halt.get('kind')}: {why})", "no"
    return ("exit 0 (done)" if man.get("complete") else "no done/halt record"), \
        ("yes" if man.get("complete") else "no")


def _timing(e: Path) -> str:
    raw = (e / "throughput.jsonl.gz")
    rows = [json.loads(x) for x in gzip.decompress(raw.read_bytes()).decode().splitlines()
            if x.strip()] if raw.exists() else []
    if not rows:
        return "timing n/a"
    eps = sum(int(r.get("episodes") or 0) for r in rows)
    mins = sum(float(r.get("wall_minutes") or 0) for r in rows)
    return f"{mins:.0f} min, {eps} episodes" + (f", {eps / mins:.1f}/min" if mins else "")


def _coverage(e: Path, pkg: Any) -> str:
    from slh.system.hosts.hermes import mh1 as M

    rows = M._read_gz_jsonl(e / "eval" / "primary.jsonl.gz")
    ident = json.loads((e / "MANIFEST.json").read_text())["identity"]
    n = pkg.sizes(ident.get("mode") != "full").eval_units
    per = {a: len({r["unit"] for r in rows if r["arm"] == a}) for a in M.ARMS}
    cov = json.loads(gzip.decompress((e / "eval/hold-coverage.json.gz").read_bytes())) \
        if (e / "eval/hold-coverage.json.gz").exists() else {}
    holds = " ".join(f"{a}={v.get('units')}" for a, v in sorted(cov.items()))
    return ("coverage " + "/".join(f"{a}{per[a]}" for a in M.ARMS) + f" of {n}"
            + (f", sealed holds {holds}" if holds else ""))


def _seals(e: Path, pkg: Any) -> str:
    from slh.system.hosts.hermes import mh1 as M

    man = json.loads((e / "MANIFEST.json").read_text())
    fs = M.sealed_files(man)
    if not fs:
        return "seal none"
    sizes = []
    for f in fs:
        head, _, ct = (e / f).read_bytes().partition(b"\n")
        try:
            pad = json.loads(head).get("padded")
        except ValueError:
            pad = None
        sizes.append(f"{len(ct)}{'' if len(ct) == pad else '!'}")
    return f"seal {len(fs)} file(s) {','.join(sizes)} B"


def _smoke_line(s: Path | None) -> str:
    if s is None:
        return "smoke: (no smoke export given)"
    sm = _gz(Path(s) / "smoke.json.gz") or {}
    st = sm.get("status") or {}
    gl = sm.get("guard_load") or {}
    ans = sm.get("hermes_answered_without_execution") or {}
    inc = _gz_rows(Path(s) / "incidents.jsonl.gz")
    return (f"smoke: {'passed' if sm.get('passed') else 'NOT passed ' + _clip(sm.get('reasons'), 80)}"
            f" | {sm.get('episodes')} episodes, unscored {st.get('unscored')}, error "
            f"{st.get('error')} | consent acts {(sm.get('consent_acts') or {}).get('checked')}"
            f" | guard load {gl.get('workers')}w max {gl.get('max_s')} s"
            f" | guard_max_s {sm.get('guard_max_s_in_episodes')}"
            f" | hermes_answered {','.join(f'{k}={v}' for k, v in sorted(ans.items())) or 0}"
            f" | incidents {len(inc)}")


def _gz_rows(p: Path) -> list[dict[str, Any]]:
    from slh.system.hosts.hermes import mh1 as M

    return M._read_gz_jsonl(p)


def _packet(e: Path, name: str, sha: str) -> dict[str, Any] | None:
    p = e / "C" / "reviews" / f"{name}.gz"
    if not p.exists():
        return None
    raw = gzip.decompress(p.read_bytes())
    if hashlib.sha256(raw).hexdigest() != sha:
        return {"_bad": True}
    return json.loads(raw)


def _change(c: Mapping[str, Any]) -> str:
    p = dict(c.get("payload") or {})
    bits = []
    if p.get("tombstone"):
        bits.append(f"removes skill {p.get('name')}")
    for k in ("name", "target"):
        if p.get(k):
            bits.append(f"{k}={_clip(p[k], 40)}")
    for k in ("description", "body", "content", "text"):
        if p.get(k):
            bits.append(f"{k}: {_clip(p[k], CHANGE_CHARS)}")
            break
    if p.get("entries") is not None:
        bits.append("entries: " + " | ".join(_clip(x, 90) for x in p["entries"][:3])
                    + (f" (+{len(p['entries']) - 3})" if len(p["entries"]) > 3 else ""))
    return f"{c.get('cid')} ({c.get('kind')}): " + "; ".join(bits)


def _tally(evs: Sequence[Mapping[str, Any]]) -> str:
    kinds: dict[str, int] = {}
    for e in evs:
        k = str(e.get("violation") or e.get("kind") or "?")
        kinds[k] = kinds.get(k, 0) + 1
    return ",".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "none"


def holds(e: Path) -> list[dict[str, Any]]:
    """Every hold of a block, from C's exported audit chain and its packets ONLY."""
    from slh.system import loop as L
    from slh.system.hosts.hermes import mh1 as M

    rows = M._read_gz_jsonl(e / "C" / "audit.jsonl.gz")
    props = {r["candidate"]["cid"]: r["candidate"] for r in rows if r.get("kind") == "propose"}
    passing = M.held_passing(rows)
    out = []
    for r in (x for x in rows if x.get("kind") == "critical_hold"):
        k = r["k"]
        cur = L.current_packet(rows, k)
        pk = _packet(e, cur["packet"], cur["packet_sha256"]) or {}
        evs = pk.get("events") or []
        out.append({"id": f"k{k}", "kind": "confirm", "required": k in passing,
                    "batch": r.get("batch"), "packet_ok": not pk.get("_bad"),
                    "packet_sha256": cur["packet_sha256"],
                    "changes": [_change(props.get(c) or {"cid": c})
                                for c in pk.get("candidates") or r.get("candidates") or []],
                    "why": "; ".join(str(x) for x in pk.get("confirmation_reasons") or []),
                    "events": len(evs), "units": len({x.get("unit") for x in evs}),
                    "kinds": _tally(evs), "auto_release": pk.get("auto_release")})
    for r in (x for x in rows if x.get("kind") == "screen_hold"):
        h = r["h"]
        cur = L.current_screen_packet(rows, h)
        pk = _packet(e, cur["packet"], cur["packet_sha256"]) or {}
        evs = pk.get("events") or []
        meta = pk.get("candidate_meta") or {}
        out.append({"id": f"s{h}", "kind": "screen", "required": False,
                    "batch": r.get("batch"), "packet_ok": not pk.get("_bad"),
                    "packet_sha256": cur["packet_sha256"],
                    "changes": [_change(meta.get(c) or props.get(c) or {"cid": c})
                                for c in pk.get("candidates") or []],
                    "why": "candidate-only critical event the attribution rule could not clear",
                    "events": int(pk.get("events_total", len(evs))),
                    "units": len({x.get("unit") for x in evs}), "kinds": _tally(evs),
                    "auto_release": None})
    return out


def part1(package: Path, blocks: Mapping[str, Sequence[Path]],
          smokes: Mapping[str, Path] | None = None) -> str:
    from slh.system.hosts.hermes import mh1 as M

    pkg = M.Package.load(Path(package), check_code=False)
    work = Path(tempfile.mkdtemp(prefix="mh1-text-"))
    try:
        body = [(f"M-H1 TEXT PART 1 ({TEXT_SCHEMA}) plan {pkg.plan['plan_id']} package "
                 f"{pkg.manifest['sha256'][:16]} code {pkg.manifest['code_sha256'][:16]} | "
                 f"{tool_hashes()}")]
        for b in sorted(blocks):
            e = resolve_block(pkg, blocks[b], work)
            man = json.loads((e / "MANIFEST.json").read_text())
            ident = man["identity"]
            done, complete = _completion(e)
            v = M.verify(e, pkg)
            n = (man.get("shards") or {}).get("n") or 1
            body.append(f"BLOCK {b}: export {hashlib.sha256((e / 'MANIFEST.json').read_bytes()).hexdigest()[:12]}"
                        f" | mode {ident.get('mode')} | nodes {n} | complete {complete} | "
                        f"{done} | {_timing(e)}")
            body.append(f" verify: {'ok' if v['ok'] else 'NOT ok'} | {_coverage(e, pkg)} | "
                        f"{_seals(e, pkg)} | incidents {v['incidents']} | unmediated "
                        f"{v['unmediated']} | probe episodes {v['probe']['episodes']} bad "
                        f"{v['probe']['bad']}")
            for p in v["problems"][:3]:
                body.append(f"  problem: {_clip(p, 110)}")
            body.append(" " + _smoke_line((smokes or {}).get(b)))
            hs = holds(e)
            req = [h for h in hs if h["required"]]
            body.append(f" holds: {len(hs)} ({len(req)} need a decision: "
                        f"{' '.join(h['id'] for h in req) or 'none'})")
            for i, h in enumerate(hs):
                head = (f"  {h['id']} {h['kind']}{' REQUIRED' if h['required'] else ''} batch "
                        f"{h['batch']} packet {h['packet_sha256'][:12]}"
                        f"{'' if h['packet_ok'] else ' (PACKET HASH MISMATCH)'}: critical "
                        f"events {h['events']} on {h['units']} unit(s) [{h['kinds']}]")
                body.append(head)
                if i < MAX_HOLDS_FULL:
                    for c in h["changes"][:2]:
                        body.append(f"   change {c}")
                    body.append(f"   why held: {_clip(h['why'], 160)}")
        return _seal(body, "part1")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ----------------------------------------------------------------------------- part 2


def _decisions(spec: str) -> dict[tuple[str, int], str]:
    out = {}
    for tok in spec.split():
        b, _, rest = tok.partition(":")
        k, _, d = rest.partition("=")
        if not (k.startswith("k") and d in ("approve", "reject")):
            raise SystemExit(f"bad decision {tok!r}: use BLOCK:kN=approve|reject")
        out[(b, int(k[1:]))] = d
    return out


def _arm_line(rows: Sequence[Mapping[str, Any]], arm: str, label: str) -> str:
    from slh.system.hosts.hermes import mh1 as M

    s, h, c = (M._rate(rows, arm, f) for f in ("success", "harm", "critical"))
    ns = sum(1 for r in rows if r["arm"] == arm and r["status"] != "scored")
    toks = [r.get("usage") or {} for r in rows if r["arm"] == arm]
    tin = sum(int(t.get("session_prompt_tokens") or 0) for t in toks)
    tout = sum(int(t.get("session_completion_tokens") or 0) for t in toks)
    n = s["n"]
    return (f"  {label}: n {n} | success {_pct(s['k'], n)} | harm {_pct(h['k'], n)} | critical "
            f"{_pct(c['k'], n)} | non-scored {_pct(ns, n)}{' DEGRADED' if M.degraded(rows, arm) else ''}"
            f" | tokens/ep {tin // max(1, n)}/{tout // max(1, n)}")


def _contrast_line(c: Mapping[str, Any], label: str, reject: bool | None) -> list[str]:
    s, h, k = c["success"], c["harm"], c["critical"]
    holm = "" if reject is None else (" Holm-reject" if reject else " Holm-keep")
    return [(f"  {label}: success +{s['plus']}/-{s['minus']} of {s['n']} diff {_f(s['diff'])} "
            f"CI95 {_ci(s['ci95'])} p {s['p_two_sided']:.4g}{holm}"),
            (f"   harm +{h['plus']}/-{h['minus']} diff {_f(h['diff'])} upper(bonf4) "
            f"{_f(h['upper_bonf4'])} margin {h['margin']} -> {h['conclusion']}"
            f"{'' if h.get('noninferior') is None else (' NI yes' if h['noninferior'] else ' NI no')}"
            f" | critical +{k['plus']}/-{k['minus']} increase p {k['increase_p']:.4g}")]


def _mci(v: Any) -> str:
    """A bootstrap summary ``[mean, lo, hi]`` as ``mean [lo, hi]``."""
    if not v or v[0] is None:
        return "n/a"
    lo, hi = (v[1], v[2]) if len(v) > 2 else (None, None)
    return f"{v[0]:+.4f} [{_f(lo)}, {_f(hi)}]"


def _point(ev: Mapping[str, Any] | None) -> str:
    if not isinstance(ev, Mapping):
        return "n/a"
    if "status" in ev and "decisions" not in ev:
        return f"{ev.get('point')} ({ev['status']})"
    s, h = ev.get("success") or {}, ev.get("harm") or {}
    return (f"{ev.get('point')} miss {ev.get('missed_positives')}/{ev.get('positives')} "
            f"fam {_f(ev.get('family_miss_rate'))} succ {_f(s.get('diff'))} (lo "
            f"{_f(s.get('lower'))}) harm {_f(h.get('diff'))} (up {_f(h.get('upper'))}) cost "
            f"{ev.get('mean_cost')}")


def _cal(c: Mapping[str, Any]) -> str:
    return " ".join(
        f"{r} {(c.get(r) or {}).get('chosen_on_fit')}->{(c.get(r) or {}).get('frozen')}"
        f"{'' if (c.get(r) or {}).get('audit_pass') else ' (audit fail)'}"
        for r in ("control", "selector"))


def _probe_lines(exports: Sequence[Path], package: Path) -> list[str]:
    from slh.xp.ourside import s1_probe_ourside as SO

    exports = [e for e in exports if (Path(e) / "probe" / "decisions.jsonl.gz").exists()]
    if not exports:
        return ["S1 PROBE: no probe part in these exports"]
    with tempfile.TemporaryDirectory() as td:
        cal = SO.calibrate(exports, package)  # pool C only (prereg s. 7)
        p = Path(td) / "calibration.json"
        p.write_text(json.dumps(cal, indent=1, sort_keys=True, default=str))
        rep = SO.report(exports, package, p, hashlib.sha256(p.read_bytes()).hexdigest())
    d = rep.get("design_report") or {}
    fid = d.get("fidelity") or {}
    out = [(f"S1 PROBE: base {d.get('base_episodes')} | replay diverged {fid.get('diverged')}/"
           f"{fid.get('branches')} | calibration {cal['sha256'][:12]} (pool C; primary bars "
           f"{cal.get('primary_quality_mode')}, secondary {SO._secondary_mode()})")]
    for key in ("type1", "type3_natural", "type3_injected"):
        t = rep.get(key) or {}
        om = t.get("oracle_means") or {}
        hr = t.get("headroom_matched_minus_placebo") or {}
        q = t.get("matched_oracle_quality") or {}
        ck = cal.get(key) or {}
        onp = (rep.get("calibrated_on_P") or {}).get(key) or {}
        lad = (rep.get("ladder") or {}).get("type1" if key == "type1" else "type3") or {}
        env = " | ".join(f"{a} succ {_f((om.get(a) or {}).get('success'))} env "
                         f"{_f((om.get(a) or {}).get('envelope_success'))} save "
                         f"{_f((om.get(a) or {}).get('saving'))}"
                         for a in ("real", "matched", "placebo"))
        out += [f" {key}: decisions {t.get('decisions')} | {env}",
                (f"   matched-placebo succ {_mci((hr.get('success') or {}).get('mean_ci95'))} "
                f"cost {_mci((hr.get('cost_share') or {}).get('mean_ci95'))} | quality succ "
                f"{_f(q.get('success_diff'))} harm {_f(q.get('harm_diff'))} crit-new "
                f"{q.get('critical_new')} | class {t.get('hp_class')} | ladder stops "
                f"{lad.get('stops_at')}"),
                (f"   calib bounds: {_cal(ck.get('primary') or {})} | points: "
                f"{_cal((ck.get('secondary_point_bars') or {}).get('primary') or {})}"),
                f"   on P control {_point(onp.get('control'))}",
                f"   on P selector {_point(onp.get('selector'))}"]
    t2 = rep.get("type2") or {}
    out.append(f" type2 review: C forks {t2.get('forks_C')} surviving {t2.get('surviving')} "
               f"| F not surviving {_f(t2.get('F_share_not_surviving'))} | HP2 "
               f"{(t2.get('HP2') or {}).get('class')} | ladder stops "
               f"{((rep.get('ladder') or {}).get('type2') or {}).get('stops_at')}"
               + (f" | not testable {t2.get('rules_not_testable')}"
                  if t2.get("rules_not_testable") else ""))
    for k in sorted(x for x in t2 if x.startswith("rule_") and isinstance(t2[x], dict)):
        r = t2[k]
        out.append(f"   {k}: skipped {r.get('skipped')}/{r.get('forks')} net saving "
                   f"{_f(r.get('net_saving'))} loss {_f(r.get('loss'))} surviving skipped "
                   f"{r.get('surviving_skipped')}/{r.get('surviving')}")
    return out


def part2(package: Path, blocks: Mapping[str, Sequence[Path]], decisions: str,
          part1_check: str | None = None) -> str:
    from slh.system.hosts.hermes import mh1 as M

    pkg = M.Package.load(Path(package), check_code=False)
    dec = _decisions(decisions or "")
    work = Path(tempfile.mkdtemp(prefix="mh1-text-"))
    try:
        body = [(f"M-H1 TEXT PART 2 ({TEXT_SCHEMA}) plan {pkg.plan['plan_id']} package "
                 f"{pkg.manifest['sha256'][:16]} | {tool_hashes()} | part1 "
                 f"{part1_check or 'not given'} | decisions {decisions or 'none'}"),
                "(no sealed C_hold file: C including an approved hold is NOT computable here)"]
        pvals: dict[str, float] = {}
        per_block: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        exports: dict[str, Path] = {}
        for b in sorted(blocks):
            e = resolve_block(pkg, blocks[b], work)
            exports[b] = e
            prim = M._read_gz_jsonl(e / "eval" / "primary.jsonl.gz")
            audit = M._read_gz_jsonl(e / "C" / "audit.jsonl.gz")
            pointer = {r["arm"]: r["version"]
                       for r in M._read_gz_jsonl(e / "eval" / "arms.jsonl.gz")}.get("C")
            held = M.held_passing(audit)
            missing = [k for k in held if (b, k) not in dec]
            approved = {k for k in held if dec.get((b, k)) == "approve"}
            k_c = None if missing else M._c_choice(audit, pointer, approved)
            if missing:
                cdef = (f"C undetermined: no decision for {' '.join(f'k{k}' for k in missing)}"
                        " -> reported C_nohold (C's pointer)")
            elif k_c is None:
                cdef = "C = C's pointer (no approved passing hold applies): primary contrasts"
            else:
                cdef = (f"C = H{k_c} (approved hold): NOT computable without the sealed file;"
                        " C_nohold reported as secondary")
            primary_c = not missing and k_c is None
            rows = [r for r in prim if r["arm"] in ("A", "B", "C")]
            body.append(f"BLOCK {b}: {cdef}")
            body += [_arm_line(rows, "A", "A"), _arm_line(rows, "B", "B"),
                     _arm_line(rows, "C", "C" if primary_c else "C_nohold")]
            cs = []
            for x, y in (("B", "A"), ("C", "A"), ("C", "B")):
                c, p = M._contrast(rows, x, y)
                lab = f"{x}-{y}" if (primary_c or x == "B") else f"{x}_nohold-{y}"
                key = f"{b}:{lab}"
                if primary_c or x == "B":
                    pvals[key] = p
                cs.append((key, c))
            per_block[b] = cs
        rej = M.holm(pvals) if pvals else {}
        for b in sorted(per_block):
            body.append(f"CONTRASTS {b} (paired by unit; Holm over the registered "
                        f"contrasts computable here: {len(pvals)})")
            for key, c in per_block[b]:
                body += _contrast_line(c, key.split(":", 1)[1], rej.get(key))
            e = exports[b]
            traj = M._read_gz_jsonl(e / "eval" / "trajectory.jsonl.gz")
            arms = sorted({r["arm"] for r in traj})
            if arms:
                body.append("  trajectory success: " + " ".join(
                    f"{a} {M._rate(traj, a, 'success')['k']}/{M._rate(traj, a, 'success')['n']}"
                    for a in arms))
            g = M.c_gate_summary(M._read_gz_jsonl(e / "C" / "audit.jsonl.gz"))
            body.append(f"  C gate: adopted {g.get('adopted')} confirmations "
                        f"{g.get('confirmations')} holds pending "
                        f"{g.get('confirmation_holds_pending')} screen holds "
                        f"{g.get('screen_holds_pending')} retired {g.get('screen_retired')}")
        probe_exports = [exports[b] for b in sorted(exports)
                         if (exports[b] / "probe" / "decisions.jsonl.gz").exists()]
        body += _probe_lines(probe_exports or list(exports.values()), Path(package))
        return _seal(body, "part2")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.system.hosts.hermes.mh1_text")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("part1", "part2"):
        p = sub.add_parser(n)
        p.add_argument("--package", type=Path, default=Path("package-mh1"))
        p.add_argument("--block", action="append", default=[], required=True,
                       help="BLOCK=EXPORT[,EXPORT...] (every node's export of a block)")
        if n == "part1":
            p.add_argument("--smoke", action="append", default=[], help="BLOCK=SMOKE_EXPORT")
        else:
            p.add_argument("--decisions", default="",
                           help='"c9b:k1=approve c27b:k2=reject" (from us, after part 1)')
            p.add_argument("--part1-check", default=None)
    c = sub.add_parser("check", help="ours: recompute the CHECK line of a pasted part")
    c.add_argument("--text", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "part1":
        sys.stdout.write(part1(a.package, _blocks(a.block),
                               {b: ds[0] for b, ds in _blocks(a.smoke).items()}))
    elif a.cmd == "part2":
        sys.stdout.write(part2(a.package, _blocks(a.block), a.decisions, a.part1_check))
    else:
        r = check_text(a.text.read_text())
        print(json.dumps(r))
        return 0 if r["ok"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
