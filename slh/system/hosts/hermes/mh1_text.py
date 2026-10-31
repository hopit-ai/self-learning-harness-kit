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

Part 3 (v2.6.2; ``probe``, about one page, no decision needed): the S1 probe from the RUN
folders, because an export drops the probe rows when they exceed its byte share (it writes
``probe/over_budget.json``; the v2.6.1 c9b return lost them this way). Per node: what the
probe did (ran / skipped and why / error / rows over the share), its rows on disk, then the
same analysis as part 2 on every row (a text return has no byte share). Part 1 now also
names each node's probe outcome.

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


def _status_word(st: Mapping[str, Any] | None, err: Any, over: Any) -> str:
    """One node's probe outcome in a few words (status.json, error.json, the export's
    over_budget.json): ran / skipped (why) / error / rows over the byte share."""
    bits = []
    st = st or {}
    if st.get("skipped"):
        f = st.get("smoke_fidelity") or {}
        bits.append(f"SKIPPED ({_clip(st['skipped'], 60)}; smoke branches {f.get('branches')}"
                    f" diverged {f.get('diverged')})")
    elif st:
        f = st.get("fidelity") or {}
        d = st.get("decisions") or {}
        bits.append(f"ran: base {st.get('base')} decisions t1 {d.get('1')} t3 {d.get('3')} "
                    f"branches {f.get('branches')} diverged {f.get('diverged')}"
                    + (f" parts {','.join(st['parts'])}" if st.get("parts") else ""))
        b = st.get("bytes") or {}
        if b.get("over_budget"):
            bits.append(f"ROWS OVER BYTE SHARE ({_clip(b['over_budget'], 70)})")
    else:
        bits.append("no status")
    if err:
        e = err.get("error") if isinstance(err, Mapping) else err
        bits.append(f"ERROR {_clip(str(e).strip().splitlines()[-1] if e else '?', 90)}")
    if over:
        bits.append("export: rows DROPPED (over_budget.json)")
    return " | ".join(bits)


def _export_probe_line(e: Path) -> str | None:
    """(v2.6.2) What the probe did on every node of a block, from the export: the learning
    node's (or the one node's) probe/status.json and each shard's under shards/s<i>/."""
    nodes = [("", e)] + sorted((d.name, d) for d in (e / "shards").glob("s*")
                               if d.is_dir()) if (e / "shards").exists() else [("", e)]
    out = []
    for name, d in nodes:
        st, err, over = (_gz(d / "probe" / f"{n}.json.gz")
                         for n in ("status", "error", "over_budget"))
        if st is None and err is None and over is None:
            continue
        out.append(f"{name or 'node'} {_status_word(st, err, over)}")
    return ("probe: " + " ;; ".join(out)) if out else None


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
            pl = _export_probe_line(e)
            if pl:
                body.append(" " + pl)
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
    """``BLOCK:kN=approve|reject`` tokens; empty or ``none`` (the runbook's word when no
    hold needs a decision) is no decision (v2.6.2)."""
    out: dict[tuple[str, int], str] = {}
    if (spec or "").strip().lower() in ("", "none"):
        return out
    for tok in spec.split():
        b, _, rest = tok.partition(":")
        k, _, d = rest.partition("=")
        if not (k.startswith("k") and d in ("approve", "reject")):
            raise SystemExit(f"bad decision {tok!r}: use BLOCK:kN=approve|reject")
        out[(b, int(k[1:]))] = d
    return out


def _dec_string(dec: Mapping[tuple[str, int], str]) -> str:
    """The decisions as the header prints them (and our record stores them)."""
    return " ".join(f"{b}:k{k}={d}" for (b, k), d in sorted(dec.items())) or "none"


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


def _numeric(digest: Any) -> bool:
    """A call digest that JSON reads as a number (all digits, or ``<digits>e<digits>``,
    which overflows to inf): the frozen cost code (``probe_cost.call_fingerprint``)
    json-parses string arguments, so such a digest became a float and ``canonical_bytes``
    refused inf (the partner's v2.6.2 part 3 crash)."""
    if not isinstance(digest, str):
        return False
    try:
        json.loads(digest)
    except ValueError:
        return False
    return True


def safe_branch_row(r: Mapping[str, Any]) -> dict[str, Any]:
    """(v2.6.3, analysis path only) A branch row whose call lists the frozen cost code can
    fingerprint: every digest prefixed ``h:`` (never valid JSON, so it is hashed as the
    string it is; equal digests stay equal, so re-calls of removed calls still match) and
    an empty tool name read as ``tool`` (what the continuation side already does). On rows
    without numeric-looking digests the cost is unchanged (tested)."""
    out = dict(r)
    for k in ("cont_calls", "removed_calls"):
        if isinstance(r.get(k), list):
            out[k] = [[str(c[0] or "tool"), f"h:{c[1]}"] if isinstance(c, (list, tuple))
                      and len(c) == 2 else c for c in r[k]]
    return out


def _safe_copy(exports: Sequence[Path], work: Path) -> list[Path]:
    """The probe parts of the exports, copied with ``safe_branch_row`` applied (the input
    folders are never written)."""
    out = []
    for i, e in enumerate(exports):
        e, d = Path(e), work / f"safe{i}"
        (d / "probe").mkdir(parents=True)
        if (e / "MANIFEST.json").exists():
            shutil.copy(e / "MANIFEST.json", d / "MANIFEST.json")
        for f in (e / "probe").iterdir():
            if f.name == "branches.jsonl.gz":
                rows = _gz_rows(f)
                data = "".join(json.dumps(safe_branch_row(r), sort_keys=True) + "\n"
                               for r in rows).encode()
                (d / "probe" / f.name).write_bytes(gzip.compress(data, mtime=0))
            elif f.is_file():
                shutil.copy(f, d / "probe" / f.name)
        out.append(d)
    return out


def _probe_lines(exports: Sequence[Path], package: Path) -> list[str]:
    from slh.xp.ourside import s1_probe_ourside as SO

    exports = [e for e in exports if (Path(e) / "probe" / "decisions.jsonl.gz").exists()]
    if not exports:
        return ["S1 PROBE: no probe part in these exports"]
    with tempfile.TemporaryDirectory() as td:
        exports = _safe_copy(exports, Path(td))
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
                 f"{part1_check or 'not given'} | decisions {_dec_string(dec)}"),
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


# ----------------------------------------------------------------------------- part 3


UNCAPPED = 10 ** 12  # text return: the export's byte share does not apply


def _run_ident(d: Path) -> dict[str, Any]:
    p = Path(d) / "run.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _rows_on_disk(d: Path) -> tuple[dict[str, int], str]:
    """Row counts of a node's probe logs and one hash over them (they stay in the run
    folder whatever the export did)."""
    h = hashlib.sha256()
    n = {}
    for name in ("base", "decisions", "branches", "review-prompted"):
        f = Path(d) / "probe" / f"{name}.jsonl"
        raw = f.read_bytes() if f.exists() else b""
        h.update(name.encode() + b"\0" + raw + b"\0")
        n[name] = sum(1 for x in raw.splitlines() if x.strip())
    return n, h.hexdigest()[:12]


def part3(package: Path, runs: Mapping[str, Sequence[Path]],
          smoke_runs: Mapping[str, Path] | None = None) -> str:
    """(v2.6.2) The S1 probe from the RUN folders (not the exports): what the probe did on
    every node (ran / skipped and why / error / rows over the export's byte share), the
    rows on disk, and, when there are rows, the same frozen analysis as part 2 on ALL of
    them (the export's byte share is a file-return limit; a text return has none). No
    M-H1 outcome is read: probe rows and the learning node's review rows only."""
    from slh.system.hosts.hermes import mh1 as M
    from slh.system.hosts.hermes import probe as PR

    pkg = M.Package.load(Path(package), check_code=False)
    work = Path(tempfile.mkdtemp(prefix="mh1-text-"))
    try:
        body = [(f"M-H1 TEXT PART 3 PROBE ({TEXT_SCHEMA}) plan {pkg.plan['plan_id']} package "
                 f"{pkg.manifest['sha256'][:16]} | {tool_hashes()}")]
        dirs: list[Path] = []
        has_rows = False
        for b in sorted(runs):
            body.append(f"BLOCK {b}: {len(runs[b])} run folder(s)")
            sm = (smoke_runs or {}).get(b)
            if sm is not None:
                st = _run_probe_status(Path(sm))
                body.append(f" smoke {Path(sm).name}: {_status_word(*st)}")
            for d in runs[b]:
                d = Path(d)
                ident = _run_ident(d)
                bad = [k for k, want in (("plan_sha256", pkg.manifest["plan_sha256"]),
                                         ("package_sha256", pkg.manifest["sha256"]),
                                         ("code_sha256", pkg.manifest["code_sha256"]))
                       if ident.get(k) != want]
                if ident.get("probe_plan_sha256") not in (
                        None, pkg.manifest.get("probe_plan_sha256")):
                    bad.append("probe_plan_sha256")
                role = ident.get("role") or "one node"
                sh = ident.get("shard")
                name = role if sh is None else f"{role} {sh}/{ident.get('shards')}"
                n, hx = _rows_on_disk(d)
                body.append(f" {d.name} ({name}{'; IDENTITY DIFFERS ' + ','.join(bad) if bad else ''})"
                            f" done {'yes' if (d / 'done.json').exists() else 'no'} | "
                            f"{_status_word(*_run_probe_status(d))}")
                body.append(f"   rows on disk: base {n['base']} decisions {n['decisions']} "
                            f"branches {n['branches']} review-prompted {n['review-prompted']}"
                            f" | sha {hx}")
                if bad:
                    continue
                out = work / f"n{len(dirs)}"
                for f, data in PR.export_rows(d, UNCAPPED).items():
                    q = out / f"{f}.gz"
                    q.parent.mkdir(parents=True, exist_ok=True)
                    q.write_bytes(gzip.compress(data, mtime=0))
                    if f.endswith(".jsonl") and data.strip():
                        has_rows = True
                dirs.append(out)
        if has_rows:
            body.append("ANALYSIS on every row above (uncapped; the frozen analysis of part 2):")
            body += _probe_lines(dirs, Path(package))
        else:
            body.append("ANALYSIS: no probe rows in these run folders")
        return _seal(body, "part3")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _run_probe_status(d: Path) -> tuple[Any, Any, Any]:
    """A run folder's probe status.json and error.json (and no export flag)."""
    def rd(n: str) -> Any:
        f = Path(d) / "probe" / n
        try:
            return json.loads(f.read_text()) if f.exists() else None
        except ValueError:
            return {"error": f"unreadable {n}"}
    return rd("status.json"), rd("error.json"), None


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
    p3 = sub.add_parser("probe", help="(v2.6.2) part 3: the S1 probe from the RUN folders")
    p3.add_argument("--package", type=Path, default=Path("package-mh1"))
    p3.add_argument("--run", action="append", default=[], required=True,
                    help="BLOCK=RUN[,RUN...] (every node's run folder of a block)")
    p3.add_argument("--smoke-run", action="append", default=[], help="BLOCK=SMOKE_RUN")
    c = sub.add_parser("check", help="ours: recompute the CHECK line of a pasted part")
    c.add_argument("--text", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "part1":
        sys.stdout.write(part1(a.package, _blocks(a.block),
                               {b: ds[0] for b, ds in _blocks(a.smoke).items()}))
    elif a.cmd == "part2":
        sys.stdout.write(part2(a.package, _blocks(a.block), a.decisions, a.part1_check))
    elif a.cmd == "probe":
        sys.stdout.write(part3(a.package, _blocks(a.run),
                               {b: ds[0] for b, ds in _blocks(a.smoke_run).items()}))
    else:
        r = check_text(a.text.read_text())
        print(json.dumps(r))
        return 0 if r["ok"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
