"""Plain-English results.md and a self-contained results.html from results.json.

The page follows the POC 9 demo page (site/index.html): a ledger-paper column,
tokens on :root with a dark theme, Google Fonts only, inline SVG charts, no
scripts, readable at phone width. No local paths and no tool or vendor names.
"""

from __future__ import annotations

import html as H
from typing import Any

TASK_NAMES = {"address": "Address", "variant": "Variant", "sequence": "Sequence"}
WRITER_NAMES = {"T+": "T+ (teacher, lesson library)", "S+": "S+ (self-written by the 9B, advisory self-critic)",
                "T": "T (teacher, POC 9 note loop)", None: "shared"}
ARM_SHORT = {"b0": "B0", "b1": "B1", "human": "Person-written", "learned": "Learned",
             "generic": "Generic note", "full_context": "Full context",
             "search_control": "Search control", "best_of_n": "Best-of-N"}


def pts(x: Any, signed: bool = True) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.1f}" if signed else f"{x:.1f}"


def iv(i: Any) -> str:
    if not i:
        return "[n/a]"
    return f"[{i[0]:+.1f}, {i[1]:+.1f}]"


def pct(x: Any) -> str:
    return "n/a" if x is None else f"{x:.1f}%"


def _yes(b: Any) -> str:
    return "yes" if b else "no"


ROUNDING_NOTE = ("* shown to more decimals: one-decimal rounding would meet or cross the "
                 "decision threshold; results.json keeps the unrounded value the gate used.")


def bound(x: Any, thr: float) -> str:
    """A decision bound at one decimal, or with enough decimals (marked *) when
    one-decimal rounding would meet or cross the threshold ``thr``."""
    if x is None:
        return "n/a"
    for d in range(1, 9):
        txt = f"{x:+.{d}f}"
        shown = float(txt)
        if shown != thr and (shown > thr) == (x > thr):
            return txt if d == 1 else txt + "*"
    return f"{x:+.3g}*"


def giv(i: Any, lo_thr: float | None = None, hi_thr: float | None = None) -> str:
    """An interval whose lower (superiority) or upper (non-inferiority) bound decides."""
    if not i:
        return "[n/a]"
    lo = bound(i[0], lo_thr) if lo_thr is not None else f"{i[0]:+.1f}"
    hi = bound(i[1], hi_thr) if hi_thr is not None else f"{i[1]:+.1f}"
    return f"[{lo}, {hi}]"


def confirmatory(r: dict[str, Any]) -> bool:
    return bool(r.get("confirmatory", r.get("status") == "analysed")) and \
        bool((r.get("gate") or {}).get("evaluable"))


def _tested(r: dict[str, Any]) -> bool:
    return bool(((r.get("gate") or {}).get("secondary") or {}).get("tested"))


# --------------------------------------------------------------------------- text


def headline(r: dict[str, Any]) -> list[str]:
    g = r["gate"]
    p = r.get("primary") or {}
    a = p.get("accuracy") or {}
    w = p.get("wrong_write") or {}
    lines = []
    conf = confirmatory(r)
    m = r["registered"]["wrong_write_margin"]
    if not conf:
        why = (r.get("label") or "DESCRIPTIVE").split(":", 1)[-1].strip()
        lines.append(f"This is a DESCRIPTIVE report ({why}): no confirmatory verdict and no "
                     "phase-2 decision can be read from it.")
    if r.get("rehearsal"):
        lines.append("REHEARSAL: the delivery is a dry run with synthetic outcomes; the "
                     "numbers below test the analysis, they are not results.")
    if a and not conf:
        above = a.get("interval") and a["interval"][0] > 0
        lines.append(
            f"Descriptively, the T+ learned skill scored {pts(a.get('estimate'))} points "
            f"against its no-evidence search control over "
            f"{len(a.get('tasks_used') or [])} tasks (95% interval {iv(a.get('interval'))}, "
            f"which {'lies above' if above else 'includes or lies below'} 0); no test is read.")
    if w and not conf:
        lines.append(f"Wrong writes changed by {pts(w.get('estimate'))} points "
                     f"{iv(w.get('interval'))} (descriptive; no non-inferiority test).")
    if a and conf:
        sup = a.get("interval") and a["interval"][0] > 0
        lines.append(
            f"The T+ learned skill scored {pts(a.get('estimate'))} points against its "
            f"no-evidence search control, pooled over {len(a.get('tasks_used') or [])} tasks "
            f"(95% interval {giv(a.get('interval'), 0.0)}): superiority "
            f"{'shown' if sup else 'not shown'}; the estimate "
            f"{'reaches' if (a.get('estimate') or 0) >= r['registered']['minimum_worthwhile_gain'] else 'does not reach'} "
            f"the 5-point minimum worthwhile gain.")
    if w and conf:
        lines.append(f"Wrong writes changed by {pts(w.get('estimate'))} points "
                     f"{giv(w.get('interval'), None, m)}; non-inferiority (upper bound below "
                     f"{m:g} points) "
                     f"{'holds' if w.get('interval') and w['interval'][1] < m else 'does not hold'}.")
    if conf and r.get("secondary") and not _tested(r):
        lines.append("Secondary (S+): not tested (primary failed); its numbers are "
                     "descriptive.")
    if g.get("evaluable"):
        lines.append(f"Phase 2 gate: {g['phase2']}" + (f" ({'; '.join(g['reasons'])})"
                                                        if g.get("reasons") else "") + ".")
    else:
        lines.append(f"Phase 2 gate: {g.get('phase2')} ({g.get('reason')}).")
    return lines


def _contrast_row(c: dict[str, Any], margin: float | None = None) -> str:
    a, w = c.get("accuracy") or {}, c.get("wrong_write") or {}
    lo = 0.0 if margin is not None else None
    return (f"| {c['name']} | {c['role']} | {pts(a.get('estimate'))} | "
            f"{giv(a.get('interval'), lo)} | {pts(w.get('estimate'))} | "
            f"{giv(w.get('interval'), None, margin)} |")


def markdown(r: dict[str, Any]) -> str:
    d = r["design"]
    g = r["gate"]
    out = [f"# Phase 1 results ({d['plan_id']})", ""]
    if r.get("label"):
        out += [f"**{r['label']}**", ""]
    if g.get("label"):
        out += [f"**{g['label']}**", ""]
    out += ["## Headline", ""] + [f"- {x}" for x in headline(r)] + [""]
    out += ["## Design", "",
            f"- Tasks: {', '.join(d['tasks'])}; test cases per task: "
            + ", ".join(f"{t} {n} ({d['groups'][t]} groups)" for t, n in d["test_cases"].items()),
            f"- Writers: {', '.join(d['writers'])}; learning runs per writer and task: {d['runs']}",
            f"- Arms: {', '.join(d['arms'])}",
            (f"- Checker: status {r['check']['status']}, mode {r['check']['mode']}, "
            f"decision rows {sum((r['check'].get('decision_rows') or {}).values())}"), ""]
    out += ["## Registered analysis", "",
            (f"- Primary: T+ endpoint vs its no-evidence search control, accuracy, pooled over "
            f"tasks with equal weights. Superiority: {r['registered']['superiority']}."),
            ("- Secondary (tested only if the primary succeeds): S+ (self-written, advisory self-critic) endpoint vs "
            "its search control."),
            f"- Unit: {r['registered']['unit']}. Interval: {r['registered']['interval']}.",
            (f"- Wrong-write non-inferiority: upper 95% bound below "
            f"{r['registered']['wrong_write_margin']:g} points."), ""]
    conf = confirmatory(r)
    margin = r["registered"]["wrong_write_margin"]
    out += ["## Verdicts" if conf else "## Estimates (descriptive; no verdict)", "",
            ("| Contrast | Role | Accuracy (points) | 95% interval | Wrong writes (points) "
            "| 95% interval |"), "|---|---|---|---|---|---|"]
    for c in (r.get("primary"), r.get("secondary")):
        if c:
            out.append(_contrast_row(c, margin if conf else None))
    out.append("")
    if g.get("evaluable"):
        s = g.get("secondary") or {}
        out += ["## Phase 2 gate", "",
                f"- Rule: {g['rule']}.",
                (f"- Primary superiority: {_yes(g['primary']['superiority'])} "
                f"(estimate {pts(g['primary']['estimate'])}, interval "
                f"{giv(g['primary']['interval'], 0.0)}); reaches 5 points: "
                f"{_yes(g['primary']['reaches_minimum_worthwhile_gain'])}."),
                (f"- Primary wrong-write non-inferiority: {_yes(g['primary_wrong_write']['holds'])}"
                f" (upper bound {bound((g['primary_wrong_write']['interval'] or [None, None])[1], margin)})."),
                f"- Secondary: {s.get('gate_reason')}"
                + (f"; superiority {_yes(s.get('superiority'))} {giv(s.get('interval'), 0.0)}; "
                   f"wrong-write non-inferiority {_yes((s.get('wrong_write') or {}).get('holds'))} "
                   f"{giv((s.get('wrong_write') or {}).get('interval'), None, margin)}"
                   if s.get("tested") else "") + ".",
                (f"- **Phase 2: {g['phase2']}** (narrow reading, primary writer's wrong writes "
                f"only: {g['phase2_narrow_reading']})."), ""]
    else:
        out += ["## Phase 2 gate", "", f"- {g.get('phase2')}: {g.get('reason')}.", ""]
    out += ["## Per task (secondary, descriptive)", "",
            "| Contrast | Task | Points | 95% interval |", "|---|---|---|---|"]
    for c in (r.get("primary"), r.get("secondary")):
        if c:
            for t, v in (c["accuracy"].get("per_task") or {}).items():
                out.append(f"| {c['name']} | {t} | {pts(v.get('estimate'))} | "
                           f"{iv(v.get('interval'))} |")
    out.append("")
    p = r.get("primary") or {}
    sens = (p.get("accuracy") or {}).get("sensitivity")
    if sens:
        a = p["accuracy"]
        out += ["## Sensitivity of the primary interval (descriptive)", "",
                (f"- Percentile parametric bootstrap: {iv(a.get('percentile_interval'))}; "
                f"closed-form Satterthwaite: {iv(a.get('satterthwaite_interval'))}."),
                *(f"- Non-parametric run x {k} bootstrap: {iv(v['interval'])}"
                  for k, v in sens.items()),
                (f"- Between-run SD of the pooled run means: "
                f"{pts(a.get('between_run_sd'), False)} points."), ""]
    out += ["## Other contrasts (descriptive)", "",
            ("| Contrast | Role | Accuracy (points) | 95% interval | Wrong writes (points) "
            "| 95% interval |"), "|---|---|---|---|---|---|"]
    out += [_contrast_row(c) for c in r.get("descriptive_contrasts") or []]
    out.append("")
    out += ["## Arms, pooled over tasks (mean over runs)", "",
            ("| Writer | Arm | Accuracy | Wrong write | Unnecessary question | "
            "Missed clarification |"), "|---|---|---|---|---|---|"]
    for a in r.get("arms") or []:
        p_ = a.get("pooled") or {}
        out.append(f"| {a['writer'] or 'shared'} | {a['arm']} | {pct(p_.get('correct'))} | "
                   f"{pct(p_.get('wrong_write'))} | {pct(p_.get('unnecessary_question'))} | "
                   f"{pct(p_.get('missed_clarification'))} |")
    out.append("")
    for t in d["tasks"]:
        out += [f"### {t}", "", ("| Writer | Arm | Accuracy | Run SD | Wrong write | "
                "Unnecessary question | Missed clarification | Outcomes |"),
                "|---|---|---|---|---|---|---|---|"]
        for a in r.get("arms") or []:
            v = (a.get("per_task") or {}).get(t)
            if not v:
                continue
            oc = ", ".join(f"{k} {n}" for k, n in sorted(v["outcomes"].items()))
            out.append(f"| {a['writer'] or 'shared'} | {a['arm']} | {pct(v['correct'])} | "
                       f"{pts(v.get('run_sd'), False)} | {pct(v['wrong_write'])} | "
                       f"{pct(v['unnecessary_question'])} | {pct(v['missed_clarification'])} "
                       f"| {oc} |")
        out.append("")
    out += ["## Endpoint scores per learning run", "",
            ("| Writer | Task | Mean difference | Run SD | Runs above 0 | Reverted on sentinel "
             "(learned) | Reverted on sentinel (search control) |"),
            "|---|---|---|---|---|---|---|"]
    for w, t, vals, rev, crev in _endpoint_summary(r):
        if vals:
            m = sum(vals) / len(vals)
            sd = (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5 \
                if len(vals) > 1 else None
            out.append(f"| {w} | {t} | {pts(m)} | {pts(sd, False)} | "
                       f"{sum(x > 0 for x in vals)}/{len(vals)} | {rev}/{len(vals)} | "
                       f"{crev}/{len(vals)} |")
    out.append("")
    out += ["## Learning curves", "",
            ("Mean over runs; dev accuracy of the current skill after each proposal (lesson "
            "loops) or training-stream accuracy per batch (note loop)."), "",
            "| Writer / task | Arm | Start | End | Sentinel reverts |", "|---|---|---|---|---|"]
    for key, c in (r.get("learning_curves") or {}).items():
        for label in ("learned", "search_control"):
            if label in c:
                m = c[label]["mean"]
                out.append(f"| {key.replace('|', ' / ')} | {label} ({c[label]['kind']}) | "
                           f"{pct(m[0])} | {pct(m[-1])} | "
                           f"{_reverts(c, label)}/{c['runs']} |")
    out.append("")
    cost = r.get("costs") or {}
    dec = cost.get("decisions") or {}
    out += ["## Costs", "",
            f"- GPU-hours: {pts(cost.get('gpu_hours'), False) if cost.get('gpu_hours') is not None else 'not in the delivery'}"
            + (f" (from the {cost['gpu_hours_source']})" if cost.get("gpu_hours_source") else ""),
            (f"- Test decisions: {dec.get('decisions', 0):,}; model calls {dec.get('model_calls', 0):,}; "
            f"prompt tokens {dec.get('prompt_tokens', 0):,}; completion tokens "
            f"{dec.get('completion_tokens', 0):,}; errors {dec.get('errors', 0)}; time limits "
            f"{dec.get('time_limits', 0)}"),
            (f"- Agent time on test decisions: {pts(cost.get('decision_agent_hours'), False)} "
            f"hours (sum of per-case wall time); learning wall time "
            f"{pts(cost.get('learning_wall_hours'), False)} hours"),
            "- Learning budgets used (all runs): " + ", ".join(
                f"{k} {v:,}" for k, v in sorted((cost.get("learning_budget_used") or {}).items())),
            ""]
    out += ["## Training-sample overlap between runs", ""]
    for t, o in (r.get("training_overlap") or {}).items():
        if o:
            out.append(f"- {t}: mean {100 * o['mean']:.0f}%, max {100 * o['max']:.0f}% "
                       f"(expected {100 * o['expected']:.0f}%)")
    out.append("")
    for title, key in (("Missing arms", "missing_arms"), ("Deviations", "deviations"),
                       ("Ambiguities in the registered text and how they were read",
                        "ambiguities"), ("Caveats", "caveats")):
        items = r.get(key) or []
        out += [f"## {title}", ""] + ([f"- {x}" for x in items] or ["- none"]) + [""]
    text = "\n".join(out)
    if "*" in "".join(x for x in out if x.startswith(("- ", "| "))).replace("**", ""):
        text += "\n" + ROUNDING_NOTE + "\n"
    return text


def _reverts(c: dict[str, Any], arm: str) -> int:
    v = c.get("sentinel_reverted") or {}
    return int(v.get(arm, 0)) if isinstance(v, dict) else 0


def _endpoint_summary(r: dict[str, Any]) -> list[tuple[str, str, list[float], int, int]]:
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for e in r.get("endpoints") or []:
        by.setdefault((e["writer"], e["task"]), []).append(e)
    out = []
    for (w, t), rows in by.items():
        vals = [e["difference"] for e in rows if e["difference"] is not None]
        out.append((w, t, vals, sum(1 for e in rows if e.get("learned_reverted")),
                    sum(1 for e in rows if e.get("control_reverted"))))
    return out


# --------------------------------------------------------------------------- html


CSS = """
:root{--bg:#EEF2F0;--surface:#FFFFFF;--surface-2:#F6F8F7;--ink:#15201C;--muted:#56645F;
--faint:#7E8B86;--line:#D3DCD8;--accent:#1B5C49;--accent-soft:#DCEAE4;--label:#E9B949;
--label-soft:#FAF0D2;--good:#1F7A4E;--good-soft:#DAEFE2;--bad:#B02A20;--bad-soft:#F8DEDA;
--warn:#8C5A00;--warn-soft:#F5E8C8;--neutral-soft:#E4E9E7;--s0:#A3AFAA;--s1:#73918A;
--s2:#3E7D69;--s3:#1B5C49;
--f-display:"Bricolage Grotesque","Avenir Next","Segoe UI",system-ui,sans-serif;
--f-body:"IBM Plex Sans","Helvetica Neue",Arial,system-ui,sans-serif;
--f-mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
--bg:#101614;--surface:#18201D;--surface-2:#1D2622;--ink:#E4ECE8;--muted:#A3B1AB;--faint:#7F8D87;
--line:#2E3A35;--accent:#6CC3A3;--accent-soft:#1E3A30;--label:#E9B949;--label-soft:#3A3120;
--good:#6FCF9A;--good-soft:#1C3A2A;--bad:#F08A7E;--bad-soft:#43201C;--warn:#E5B35A;
--warn-soft:#3C2F16;--neutral-soft:#26302C;--s0:#5E6B66;--s1:#6E918A;--s2:#55A88C;--s3:#7FD6B3;
color-scheme:dark}}
:root[data-theme="dark"]{--bg:#101614;--surface:#18201D;--surface-2:#1D2622;--ink:#E4ECE8;
--muted:#A3B1AB;--faint:#7F8D87;--line:#2E3A35;--accent:#6CC3A3;--accent-soft:#1E3A30;
--label:#E9B949;--label-soft:#3A3120;--good:#6FCF9A;--good-soft:#1C3A2A;--bad:#F08A7E;
--bad-soft:#43201C;--warn:#E5B35A;--warn-soft:#3C2F16;--neutral-soft:#26302C;--s0:#5E6B66;
--s1:#6E918A;--s2:#55A88C;--s3:#7FD6B3;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--f-body);font-size:16px;
line-height:1.55;margin:0;-webkit-text-size-adjust:100%}
.wrap{max-width:1080px;margin:0 auto;padding-inline:clamp(16px,4vw,40px);padding-block:28px 64px}
h1,h2,h3{font-family:var(--f-display);text-wrap:balance;margin:0;line-height:1.15;letter-spacing:-.01em}
h1{font-size:clamp(1.7rem,4.2vw,2.6rem);font-weight:700;max-width:26ch}
h2{font-size:clamp(1.3rem,2.6vw,1.7rem);font-weight:700}
h3{font-size:1.05rem;font-weight:600}
p{margin:0}
.mono{font-family:var(--f-mono)}
.eyebrow{font-family:var(--f-mono);font-size:.74rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.lede{color:var(--muted);max-width:70ch}
.small{font-size:.85rem;color:var(--muted)}
section{display:grid;gap:16px;padding-block:40px;border-top:1px solid var(--line)}
section:first-of-type{border-top:0;padding-top:8px}
.mast{display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px 16px;align-items:baseline;
padding-bottom:20px;border-bottom:1px solid var(--line);margin-bottom:24px}
.mast .mono{font-size:.78rem;color:var(--muted)}
.stamp{display:inline-block;font-family:var(--f-mono);font-size:.72rem;letter-spacing:.06em;
text-transform:uppercase;padding:2px 8px;border:1.5px solid var(--accent);color:var(--accent);border-radius:3px}
.stamp.warn{border-color:var(--warn);color:var(--warn)}
.banner{background:var(--label-soft);border:1px solid var(--label);border-radius:8px;padding:10px 14px;font-weight:500}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:14px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 18px;display:grid;gap:8px;align-content:start}
.card .big{font-family:var(--f-display);font-weight:700;font-size:1.9rem;line-height:1}
.verdict{display:inline-block;font-family:var(--f-mono);font-size:.78rem;font-weight:500;padding:2px 8px;border-radius:3px}
.verdict.go{background:var(--good-soft);color:var(--good)}
.verdict.no{background:var(--bad-soft);color:var(--bad)}
.verdict.na{background:var(--neutral-soft);color:var(--muted)}
.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}
th,td{padding:7px 11px;text-align:left;border-top:1px solid var(--line);vertical-align:top}
thead th{border-top:0;font-family:var(--f-mono);font-size:.68rem;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);font-weight:500;white-space:nowrap}
td.r,th.r{text-align:right;white-space:nowrap}
tr.hl td{background:var(--accent-soft)}
.chart{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px clamp(10px,2vw,20px)}
.chart{overflow-x:auto}
.chart svg{display:block;width:100%;height:auto}
.chart svg.wide{min-width:540px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:12px}
ul{margin:0;padding-left:1.2em;display:grid;gap:6px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.82rem;color:var(--muted)}
.legend i{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:6px}
svg text{font-family:var(--f-mono);fill:var(--muted)}
"""


def e(x: Any) -> str:
    return H.escape(str(x), quote=True)


def _verdict(v: Any) -> str:
    cls = {"GO": "go", "NO-GO": "no"}.get(v, "na")
    return f'<span class="verdict {cls}">{e(v)}</span>'


def forest(rows: list[tuple[str, float | None, list[float] | None, str]], *,
           ref: float | None = 5.0, ref_label: str = "minimum worthwhile", width: int = 640
           ) -> str:
    """Horizontal interval chart: (label, estimate, interval, css colour var)."""
    rows = [x for x in rows if x[1] is not None]
    if not rows:
        return ""
    vals = [v for _, est, i, _ in rows for v in ([est] + list(i or []))]
    lo, hi = min(vals + [0.0, ref or 0.0]), max(vals + [0.0, ref or 0.0])
    pad = (hi - lo) * 0.08 + 0.5
    lo, hi = lo - pad, hi + pad
    left, right, rh = 220, 70, 26
    h = 30 + rh * len(rows) + 24
    span = width - left - right

    def x(v: float) -> float:
        return left + (v - lo) / (hi - lo) * span

    parts = [(f'<svg class="wide" viewBox="0 0 {width} {h}" role="img" '
             f'aria-label="Intervals in points">')]
    step = 1 if hi - lo <= 12 else 2 if hi - lo <= 25 else 5
    t = int(lo // step) * step
    while t <= hi:
        if t >= lo:
            parts.append(f'<line x1="{x(t):.1f}" y1="20" x2="{x(t):.1f}" y2="{h - 22}" '
                         f'stroke="var(--line)" stroke-width="1"/>')
            parts.append(f'<text x="{x(t):.1f}" y="{h - 6}" font-size="11" '
                         f'text-anchor="middle">{t:+d}</text>')
        t += step
    parts.append(f'<line x1="{x(0):.1f}" y1="16" x2="{x(0):.1f}" y2="{h - 22}" '
                 f'stroke="var(--ink)" stroke-width="1.5"/>')
    if ref is not None:
        parts.append(f'<line x1="{x(ref):.1f}" y1="16" x2="{x(ref):.1f}" y2="{h - 22}" '
                     f'stroke="var(--label)" stroke-width="1.5" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{x(ref):.1f}" y="12" font-size="10" text-anchor="middle">'
                     f'{e(ref_label)} {ref:+g}</text>')
    for i, (label, est, interval, colour) in enumerate(rows):
        y = 30 + rh * i + rh / 2
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" font-size="12" text-anchor="end" '
                     f'style="fill:var(--ink)">{e(label)}</text>')
        if interval:
            parts.append(f'<line x1="{x(interval[0]):.1f}" y1="{y:.1f}" '
                         f'x2="{x(interval[1]):.1f}" y2="{y:.1f}" stroke="var({colour})" '
                         f'stroke-width="3" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{x(est):.1f}" cy="{y:.1f}" r="5" fill="var({colour})" '
                     f'stroke="var(--surface)" stroke-width="1.5"/>')
    parts.append("</svg>")
    return "".join(parts)


def curve_panel(title: str, series: list[tuple[str, list[float], list[list[float]], str]],
                *, width: int = 320, height: int = 190) -> str:
    """Small multiple: mean curve (thick) with each run (thin)."""
    allv = [v for _, m, runs, _ in series for v in m + [x for r in runs for x in r]
            if v is not None]
    if not allv:
        return ""
    lo, hi = max(0.0, min(allv) - 5), min(100.0, max(allv) + 5)
    if hi - lo < 10:
        hi = min(100.0, lo + 10)
    n = max(len(m) for _, m, _, _ in series)
    left, right, top, bottom = 36, 10, 24, 26
    pw, ph = width - left - right, height - top - bottom

    def xy(i: int, v: float) -> tuple[float, float]:
        xx = left + (i / max(1, n - 1)) * pw
        yy = top + (1 - (v - lo) / (hi - lo)) * ph
        return xx, yy

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{e(title)}">',
             f'<text x="{left}" y="14" font-size="11" style="fill:var(--ink)">{e(title)}</text>']
    for v in (lo, (lo + hi) / 2, hi):
        _, yy = xy(0, v)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" '
                     f'stroke="var(--line)"/>')
        parts.append(f'<text x="{left - 4}" y="{yy + 3:.1f}" font-size="9" '
                     f'text-anchor="end">{v:.0f}</text>')
    for i in range(n):
        xx, _ = xy(i, lo)
        parts.append(f'<text x="{xx:.1f}" y="{height - 8}" font-size="9" '
                     f'text-anchor="middle">{i}</text>')
    for _, mean, runs, colour in series:
        for r in runs:
            pts_ = " ".join(f"{a:.1f},{b:.1f}" for a, b in (xy(i, v) for i, v in enumerate(r)
                                                            if v is not None))
            parts.append(f'<polyline points="{pts_}" fill="none" stroke="var({colour})" '
                         f'stroke-opacity=".22" stroke-width="1"/>')
        pts_ = " ".join(f"{a:.1f},{b:.1f}" for a, b in (xy(i, v) for i, v in enumerate(mean)
                                                        if v is not None))
        parts.append(f'<polyline points="{pts_}" fill="none" stroke="var({colour})" '
                     f'stroke-width="2.5"/>')
    parts.append("</svg>")
    return "".join(parts)


def strip(rows: list[tuple[str, list[float]]], *, width: int = 640) -> str:
    """Per-run endpoint differences as dots, one row per writer/task."""
    rows = [r for r in rows if r[1]]
    if not rows:
        return ""
    vals = [v for _, vs in rows for v in vs] + [0.0]
    lo, hi = min(vals) - 2, max(vals) + 2
    left, right, rh = 130, 20, 24
    h = 20 + rh * len(rows) + 22
    span = width - left - right

    def x(v: float) -> float:
        return left + (v - lo) / (hi - lo) * span

    parts = [(f'<svg class="wide" viewBox="0 0 {width} {h}" role="img" '
             f'aria-label="Per-run differences">'),
             f'<line x1="{x(0):.1f}" y1="10" x2="{x(0):.1f}" y2="{h - 22}" stroke="var(--ink)"/>']
    step = 2 if hi - lo <= 20 else 5 if hi - lo <= 50 else 10
    t = int(lo // step) * step
    while t <= hi:
        if t >= lo:
            parts.append(f'<text x="{x(t):.1f}" y="{h - 6}" font-size="10" '
                         f'text-anchor="middle">{t:+d}</text>')
        t += step
    for i, (label, vs) in enumerate(rows):
        y = 20 + rh * i + rh / 2
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" font-size="11" text-anchor="end" '
                     f'style="fill:var(--ink)">{e(label)}</text>')
        for v in vs:
            parts.append(f'<circle cx="{x(v):.1f}" cy="{y:.1f}" r="4" fill="var(--s2)" '
                         f'fill-opacity=".7"/>')
        m = sum(vs) / len(vs)
        parts.append(f'<line x1="{x(m):.1f}" y1="{y - 9:.1f}" x2="{x(m):.1f}" y2="{y + 9:.1f}" '
                     f'stroke="var(--accent)" stroke-width="3"/>')
    parts.append("</svg>")
    return "".join(parts)


def _table(head: list[str], rows: list[list[str]], right: set[int] | None = None,
           hl: set[int] | None = None) -> str:
    right = right or set()
    th = "".join(f'<th{" class=r" if i in right else ""}>{e(h)}</th>'
                 for i, h in enumerate(head))
    body = []
    for j, r in enumerate(rows):
        tds = "".join(f'<td{" class=r" if i in right else ""}>{c}</td>' for i, c in enumerate(r))
        body.append(f'<tr{" class=hl" if hl and j in hl else ""}>{tds}</tr>')
    return (f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def html(r: dict[str, Any]) -> str:
    d, g = r["design"], r["gate"]
    p, s = r.get("primary") or {}, r.get("secondary") or {}
    pa, sa = p.get("accuracy") or {}, s.get("accuracy") or {}
    stamps = []
    if r.get("rehearsal"):
        stamps.append('<span class="stamp warn">Rehearsal: synthetic delivery</span>')
    conf = confirmatory(r)
    margin = r["registered"]["wrong_write_margin"]
    if not conf:
        stamps.append('<span class="stamp warn">Descriptive only</span>')
    else:
        stamps.append('<span class="stamp">Registered analysis</span>')
    out = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>Phase 1 Results</title>',
           '<link rel="preconnect" href="https://fonts.googleapis.com">',
           '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
           ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+'
           'Grotesque:opsz,wght@12..96,600;12..96,700&amp;family=IBM+Plex+Sans:wght@400;500;600'
           '&amp;family=IBM+Plex+Mono:wght@400;500&amp;display=swap">'),
           f"<style>{CSS}</style></head><body><main class=\"wrap\">",
           (f'<header class="mast"><span class="eyebrow">Self-learning harness · phase 1 · '
           f'{e(d["plan_id"])}</span><span>{" ".join(stamps)}</span></header>')]
    # headline
    out.append('<section><div style="display:grid;gap:10px">'
               '<h1>Does the learned skill beat its no-evidence search control?</h1>')
    for line in headline(r):
        cls = "banner" if line.startswith(("REHEARSAL", "This is a DESCRIPTIVE")) else "lede"
        out.append(f'<p class="{cls}">{e(line)}</p>')
    out.append("</div>")
    cards = []
    for c, title in ((p, "Primary: T+ vs its search control"),
                     (s, "Secondary: S+ (self-written, advisory self-critic) vs its search control")):
        a, w = c.get("accuracy") or {}, c.get("wrong_write") or {}
        if not a:
            continue
        cards.append(f'<div class="card"><span class="eyebrow">{e(title)}</span>'
                     f'<span class="big">{e(pts(a.get("estimate")))} pts</span>'
                     f'<span class="mono small">95% '
                     f'{e(giv(a.get("interval"), 0.0 if conf else None))}</span>'
                     f'<span class="small">{e(c.get("role"))}</span>'
                     f'<span class="small">Wrong writes {e(pts(w.get("estimate")))} '
                     f'{e(giv(w.get("interval"), None, margin if conf else None))}</span></div>')
    reason = "; ".join(g.get("reasons") or []) or g.get("reason") or "all registered conditions met"
    cards.append(f'<div class="card"><span class="eyebrow">Phase 2 gate</span>'
                 f'<span>{_verdict(g.get("phase2"))}</span>'
                 f'<span class="small">{e(reason)}</span>'
                 + (f'<span class="small">Narrow reading (primary wrong writes only): '
                    f'{_verdict(g.get("phase2_narrow_reading"))}</span>'
                    if g.get("evaluable") else "")
                 + "</div>")
    out.append(f'<div class="cards">{"".join(cards)}</div></section>')
    # verdict chart
    rows = []
    if pa:
        rows.append(("Primary: T+ pooled", pa.get("estimate"), pa.get("interval"), "--s3"))
        for t, v in (pa.get("per_task") or {}).items():
            rows.append((f"  T+ {t}", v.get("estimate"), v.get("interval"), "--s1"))
    if sa:
        lab = "Secondary: S+ pooled" if (conf and _tested(r)) else \
            "S+ pooled (not tested)" if conf else "S+ pooled"
        rows.append((lab, sa.get("estimate"), sa.get("interval"), "--s3"))
        for t, v in (sa.get("per_task") or {}).items():
            rows.append((f"  S+ {t}", v.get("estimate"), v.get("interval"), "--s1"))
    out.append('<section><div class="sec-head" style="display:grid;gap:8px">'
               f'<span class="eyebrow">{"Registered contrasts" if conf else "Descriptive estimates"}'
               '</span><h2>Endpoint minus search control, accuracy points</h2>'
               f'<p class="lede">{e(_forest_note(r, conf))}</p></div>'
               f'<div class="chart">{forest(rows)}</div>')
    ww_rows = []
    for c, lab in ((p, "T+ wrong writes"), (s, "S+ wrong writes")):
        w = c.get("wrong_write") or {}
        if w:
            ww_rows.append((lab, w.get("estimate"), w.get("interval"), "--bad"))
    if ww_rows:
        out.append('<h3>Wrong writes, endpoint minus search control (non-inferiority margin '
                   f'{r["registered"]["wrong_write_margin"]:g} points, dashed)</h3>'
                   f'<div class="chart">{forest(ww_rows, ref=r["registered"]["wrong_write_margin"], ref_label="margin")}</div>')
    if pa.get("sensitivity"):
        srows = [["Bootstrap-t (registered method)", e(iv(pa.get("interval")))],
                 ["Percentile parametric bootstrap", e(iv(pa.get("percentile_interval")))],
                 ["Closed-form Satterthwaite", e(iv(pa.get("satterthwaite_interval")))]]
        srows += [[f"Non-parametric run x {e(k)} bootstrap", e(iv(v["interval"]))]
                  for k, v in pa["sensitivity"].items()]
        out.append('<h3>Sensitivity of the primary interval (descriptive)</h3>'
                   + _table(["Method", "95% interval"], srows, right={1}))
    out.append("</section>")
    # gate details
    if g.get("evaluable"):
        sec = g.get("secondary") or {}
        grows = [["Primary superiority (lower bound &gt; 0)",
                  _verdict("GO" if g["primary"]["superiority"] else "NO-GO"),
                  e(giv(g["primary"]["interval"], 0.0))],
                 ["Primary reaches 5 points (reported, not gated)",
                  e(_yes(g["primary"]["reaches_minimum_worthwhile_gain"])),
                  e(pts(g["primary"]["estimate"]))],
                 ["Primary wrong-write non-inferiority",
                  _verdict("GO" if g["primary_wrong_write"]["holds"] else "NO-GO"),
                  e(giv(g["primary_wrong_write"]["interval"], None, margin))],
                 ["Secondary tested", e(_yes(sec.get("tested"))), e(sec.get("gate_reason"))]]
        if sec.get("tested"):
            grows += [["Secondary superiority",
                       _verdict("GO" if sec.get("superiority") else "NO-GO"),
                       e(giv(sec.get("interval"), 0.0))],
                      ["Secondary wrong-write non-inferiority",
                       _verdict("GO" if (sec.get("wrong_write") or {}).get("holds") else "NO-GO"),
                       e(giv((sec.get("wrong_write") or {}).get("interval"), None, margin))]]
        out.append('<section><span class="eyebrow">Gate</span><h2>Phase 1 to phase 2</h2>'
                   f'<p class="lede">{e(g["rule"])}.</p>'
                   + _table(["Condition", "Result", "Evidence"], grows) + "</section>")
    # arms
    arms = r.get("arms") or []
    bars = []
    for a in arms:
        pooled = a.get("pooled") or {}
        if pooled.get("correct") is not None:
            lab = f"{a['writer'] + ' ' if a['writer'] else ''}{ARM_SHORT.get(a['arm'], a['arm'])}"
            bars.append((lab, pooled["correct"]))
    if bars:
        out.append('<section><span class="eyebrow">Arms</span><h2>Accuracy by arm, pooled '
                   'over tasks</h2><p class="lede">Mean over learning runs for per-writer arms; '
                   'shared arms are evaluated once per task. All rates in % of test cases.</p>'
                   f'<div class="chart">{_bars(bars)}</div>')
        trows, hl = [], set()
        for a in arms:
            pooled = a.get("pooled") or {}
            if a["writer"] == "T+" and a["arm"] in ("learned", "search_control"):
                hl.add(len(trows))
            trows.append([e(a["writer"] or "shared"), e(ARM_SHORT.get(a["arm"], a["arm"])),
                          e(pct(pooled.get("correct"))), e(pct(pooled.get("wrong_write"))),
                          e(pct(pooled.get("unnecessary_question"))),
                          e(pct(pooled.get("missed_clarification")))])
        out.append(_table(["Writer", "Arm", "Accuracy", "Wrong write", "Unnecessary question",
                           "Missed clarification"], trows, right={2, 3, 4, 5}, hl=hl))
        for t in d["tasks"]:
            trows = []
            for a in arms:
                v = (a.get("per_task") or {}).get(t)
                if v:
                    oc = ", ".join(f"{k} {n}" for k, n in sorted(v["outcomes"].items()))
                    trows.append([e(a["writer"] or "shared"),
                                  e(ARM_SHORT.get(a["arm"], a["arm"])), e(pct(v["correct"])),
                                  e(pts(v.get("run_sd"), False)), e(pct(v["wrong_write"])),
                                  e(pct(v["unnecessary_question"])),
                                  e(pct(v["missed_clarification"])),
                                  f'<span class="small">{e(oc)}</span>'])
            out.append(f"<h3>{e(TASK_NAMES.get(t, t))}</h3>" + _table(
                ["Writer", "Arm", "Accuracy", "Run SD", "Wrong write", "Unnecessary question",
                 "Missed clarification", "Outcomes"], trows, right={2, 3, 4, 5, 6}))
        out.append("</section>")
    # other contrasts
    desc = r.get("descriptive_contrasts") or []
    if desc:
        frows = [(_short(c), (c.get("accuracy") or {}).get("estimate"),
                  (c.get("accuracy") or {}).get("interval"), "--s1") for c in desc]
        out.append('<section><span class="eyebrow">Controls</span><h2>Every other contrast '
                   '(descriptive)</h2><p class="lede">Learned endpoint minus each control, '
                   'same analysis, no claim attached. B1 equals the removal arm.</p>'
                   f'<div class="chart">{forest(frows, ref=None)}</div>'
                   + _table(["Contrast", "Accuracy", "95% interval", "Wrong writes",
                             "95% interval"],
                            [[e(c["name"]), e(pts(c["accuracy"].get("estimate"))),
                              e(iv(c["accuracy"].get("interval"))),
                              e(pts(c["wrong_write"].get("estimate"))),
                              e(iv(c["wrong_write"].get("interval")))] for c in desc],
                            right={1, 2, 3, 4}) + "</section>")
    # per-run endpoints
    srows = [(f"{w} {t}", vals) for w, t, vals, _, _ in _endpoint_summary(r)]
    if srows:
        out.append('<section><span class="eyebrow">Learning runs</span><h2>Endpoint minus '
                   'search control, per run</h2><p class="lede">Each dot is one independent '
                   'learning run on the sealed test; the bar is the mean.</p>'
                   f'<div class="chart">{strip(srows)}</div>')
        erows = []
        for w, t, vals, rev, crev in _endpoint_summary(r):
            if vals:
                m = sum(vals) / len(vals)
                erows.append([e(w), e(t), e(pts(m)), e(f"{sum(v > 0 for v in vals)}/{len(vals)}"),
                              e(f"{rev}/{len(vals)}"), e(f"{crev}/{len(vals)}")])
        out.append(_table(["Writer", "Task", "Mean difference", "Runs above 0",
                           "Sentinel reverts (learned)", "Sentinel reverts (search control)"],
                          erows, right={2, 3, 4, 5}) + "</section>")
    # curves
    curves = r.get("learning_curves") or {}
    if curves:
        panels = []
        order = {w: i for i, w in enumerate(d["writers"])}
        for key, c in sorted(curves.items(), key=lambda kv: (
                order.get(kv[0].split("|")[0], 99), d["tasks"].index(kv[0].split("|")[1])
                if kv[0].split("|")[1] in d["tasks"] else 99)):
            series = []
            for label, colour in (("learned", "--s3"), ("search_control", "--warn")):
                if label in c:
                    series.append((label, c[label]["mean"], c[label]["runs"], colour))
            kind = next((c[x]["kind"] for x in ("learned", "search_control") if x in c), "")
            panels.append(f'<div class="chart">{curve_panel(f"{key.replace("|", " · ")} ({kind})", series)}</div>')
        out.append('<section><span class="eyebrow">Learning curves</span><h2>Development '
                   'scores during learning</h2><p class="lede">Lesson loops: dev accuracy of '
                   'the current skill after each proposal (x = proposal). Note loop: '
                   'training-stream accuracy per batch. Thin lines: runs; thick: mean.</p>'
                   '<div class="legend"><span><i style="background:var(--s3)"></i>learned'
                   '</span><span><i style="background:var(--warn)"></i>no-evidence search '
                   f'control</span></div><div class="grid3">{"".join(panels)}</div></section>')
    # costs
    cost = r.get("costs") or {}
    dec = cost.get("decisions") or {}
    crow = [["GPU-hours", e(pts(cost.get("gpu_hours"), False)
                            if cost.get("gpu_hours") is not None else "not in the delivery")],
            ["Test decisions", e(f"{dec.get('decisions', 0):,}")],
            ["Model calls (test)", e(f"{dec.get('model_calls', 0):,}")],
            ["Prompt / completion tokens (test)",
             e(f"{dec.get('prompt_tokens', 0):,} / {dec.get('completion_tokens', 0):,}")],
            ["Errors / time limits", e(f"{dec.get('errors', 0)} / {dec.get('time_limits', 0)}")],
            ["Agent time on test decisions (h)", e(pts(cost.get("decision_agent_hours"), False))],
            ["Learning wall time (h)", e(pts(cost.get("learning_wall_hours"), False))]]
    crow += [[e(f"Learning budget used: {k}"), e(f"{v:,}")]
             for k, v in sorted((cost.get("learning_budget_used") or {}).items())]
    orow = [[e(t), e(f"{100 * o['mean']:.0f}%"), e(f"{100 * o['max']:.0f}%"),
             e(f"{100 * o['expected']:.0f}%")]
            for t, o in (r.get("training_overlap") or {}).items() if o]
    out.append('<section><span class="eyebrow">Costs</span><h2>What the run used</h2>'
               + _table(["Item", "Amount"], crow, right={1})
               + ("<h3>Training-sample overlap between runs</h3>"
                  + _table(["Task", "Mean", "Max", "Expected"], orow, right={1, 2, 3})
                  if orow else "") + "</section>")
    # caveats
    for title, key in (("Missing arms", "missing_arms"), ("Deviations", "deviations"),
                       ("How ambiguities in the registered text were read", "ambiguities"),
                       ("Caveats", "caveats")):
        items = r.get(key) or []
        lis = "".join(f"<li>{e(x)}</li>" for x in items) or "<li>none</li>"
        out.append(f'<section><span class="eyebrow">Read with care</span><h2>{e(title)}</h2>'
                   f"<ul>{lis}</ul></section>")
    body = "\n".join(out)
    if "*</" in body or "*]" in body:
        out.append(f'<p class="small">{e(ROUNDING_NOTE)}</p>')
    out.append("</main></body></html>")
    return "\n".join(out)


def refusal_markdown(body: dict[str, Any]) -> str:
    return ("# Phase 1 results: refused\n\nNo analysis was run and no verdict exists for "
            f"this delivery. {body.get('reason', '')}\n")


def refusal_html(body: dict[str, Any]) -> str:
    chk = body.get("check") or {}
    return "\n".join([
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Phase 1 Refused</title>", f"<style>{CSS}</style></head><body>",
        ('<main class="wrap"><header class="mast"><span class="eyebrow">Self-learning '
        'harness · phase 1</span><span class="stamp warn">Refused</span></header>'),
        "<section><h1>No analysis: the delivery was refused</h1>",
        f'<p class="banner">{e(body.get("reason", ""))}</p>',
        (f'<p class="lede">Checker status: {e(chk.get("status"))}. No scores, estimates or '
        "verdict exist for this delivery; any earlier page in this folder has been "
        "replaced.</p></section></main></body></html>")])


def _forest_note(r: dict[str, Any], conf: bool) -> str:
    if not conf:
        return ("Descriptive only: no row is a test. Bootstrap-t 95% intervals, crossed runs x "
                "scenario groups. Dashed: the 5-point minimum worthwhile gain.")
    note = ("The T+ pooled row is the registered test (bootstrap-t 95% interval, crossed runs "
            "x scenario groups); ")
    note += ("the S+ pooled row is the gated secondary test; " if _tested(r) else
             "S+ is not tested (primary failed) and is descriptive; ")
    return note + "per-task rows are descriptive. Dashed: the 5-point minimum worthwhile gain."


def _short(c: dict[str, Any]) -> str:
    ctrl = ARM_SHORT.get(c["control"], c["control"])
    if c.get("writer"):
        return f"{c['writer']} vs {ctrl}"
    return f"{ARM_SHORT.get(c['endpoint'], c['endpoint'])} vs {ctrl}"


def _bars(rows: list[tuple[str, float]], *, width: int = 640) -> str:
    left, right, rh = 190, 60, 22
    h = 10 + rh * len(rows) + 10
    span = width - left - right
    parts = [(f'<svg class="wide" viewBox="0 0 {width} {h}" role="img" '
             f'aria-label="Accuracy by arm">')]
    for i, (label, v) in enumerate(rows):
        y = 10 + rh * i
        w = max(0.0, min(100.0, v)) / 100 * span
        colour = "--s3" if "Learned" in label else "--s1"
        parts.append(f'<text x="{left - 10}" y="{y + 15}" font-size="11" text-anchor="end" '
                     f'style="fill:var(--ink)">{e(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y + 3}" width="{span}" height="{rh - 6}" '
                     f'fill="var(--surface-2)"/>')
        parts.append(f'<rect x="{left}" y="{y + 3}" width="{w:.1f}" height="{rh - 6}" '
                     f'fill="var({colour})"/>')
        parts.append(f'<text x="{left + w + 6:.1f}" y="{y + 15}" font-size="11">{v:.1f}%</text>')
    parts.append("</svg>")
    return "".join(parts)
