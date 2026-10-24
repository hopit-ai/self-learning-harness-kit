"""Our side: the F' screen analysis (pre-specified; docs/xp/partner-c-fgate-plan.md, s. 3.5).

    python -m slh.xp.fgate.analysis --table TABLE.jsonl --pool POOL.json --out REPORT.json \
        [--plan PKG/plan.json]

The report also carries ``difficulty_check`` (the synthetic-difficulty check of hand-off 1b
part A, ``difficulty_check``; ``incomplete`` unless every registered pair, arm and phrasing
is observed exactly once), None for a package without it.

**Model** (per outcome: ``y`` success, ``harm`` wrong write, ``reask`` unnecessary
question; logistic for each, all binary at the decision level)::

    logit P(y_r) = m_c + u_i + sum_l x_rl (b_l + d_lc) + sum_(l<m) x_rl x_rm g_lm

r = decision, i = decision prefix (partial pooling: u_i ~ N(0, s_a^2) around its
context mean m_c ~ N(0, 2.5^2)), c = context (task x gold type), x = inclusion bits.
b_l ~ N(mu_b, s_b^2) with mu_b and s_b calibrated **predictively on the probability
scale**: s_b = rd_prior_sd / (p0 (1 - p0)), mu_b = rd_prior_mean / (p0 (1 - p0)) where p0 is
the empty-library anchors' mean (so the prior of a lesson's risk difference near the
baseline has mean ``rd_prior_mean`` and SD ``rd_prior_sd``). **Three named priors**
(``PRIORS``, owner 2026-10-18), each fitted: ``sceptical`` (primary) 0 +- 3 points,
``weak`` 0 +- 6 points, ``literature`` -1 +- 3 points on success (harm centred 0 under
all three); the interaction and tau scales stay at the base 3 points under every prior. Context deviations d_lc ~ N(0, tau^2) with one shared tau (shared
regularization). Every pairwise interaction is included and strongly shrunk:
g ~ N(0, (0.25 s_b)^2) for lessons that touch the same tool, N(0, (0.1 s_b)^2) across
tools (cross-tool interference allowed but strongly shrunk: with 11 partners per lesson,
looser interaction priors add their summed noise to every standardized RD). s_a and tau are chosen by Laplace marginal
likelihood over a fixed grid (empirical Bayes, per prior); every grid point is reported as
a sensitivity (primary prior). The prior scale x2 sensitivity is superseded by the gated
weakly informative prior.

**Fit.** Posterior mode by Newton steps with the prefix intercepts eliminated through
the Schur complement; posterior draws from the Laplace (Gaussian) approximation, the
intercepts drawn from their conditional Gaussian. (A full sampler can replace it at
registration; the estimands and the gate do not change.)

**Estimands.** Posterior-standardized risk differences on the probability scale over a
declared target population: the randomized screen decisions (the distribution of the
other lessons is the design's: each included with p = 0.5), every context weighted
equally (``weights="equal"``; ``"sampled"`` as a sensitivity). RD_l: inclusion of l vs
not, other lessons as randomized. RD_lc: the same within context c. RD_S for a subset
S: every lesson of S on vs every one off, the others as randomized (the joint contrast
the design identifies); "exactly S vs the empty library" is reported descriptively (it
is an extrapolation corner of a p = 0.5 design, so it is left to the conversation gate). These are not intrinsic lesson
values and not contributions within a final library.

**Gate** (``gate``): per lesson, all of
  E1  P(RD_l > 1 point) >= 0.90 (success);
  E2  every context has exposure >= ``min_exposure`` decisions with the lesson included
      (positive evidence is required; unexposed contexts fail);
  E3  every context has P(RD_lc < -1 point) < 0.10, i.e. P(RD_lc >= -1 point) > 0.90
      (tightened 2026-10-19 from "no context with P(RD_lc < -1) >= 0.80" by simulation,
      docs/xp/partner-c-fgate-plan.md s. 3.5; ``context_harm_bound`` /
      ``p_context_harm_bound`` add an optional per-context bound, off by default);
  E4  harm: P(RD^harm_l > +1 point) <= 0.20 AND P(RD^harm_l < +2.5 points) >= 0.95
      (positive posterior support for bounded harm).
**Three priors** (``gate_priors``): E1-E4 are evaluated under each named prior; a lesson
passes only if it passes under all three (intersection), and each prior's stand-alone
result is reported (``by_prior``, ``table``). The shortlist is at most 4
intersection-passing lessons by primary posterior mean RD; every non-empty subset
of the shortlist is evaluated jointly (RD_S and RD^harm_S) and the advanced set is the
subset with the highest posterior mean RD_S among those with P(RD_S > 1 point) >= 0.90
and P(RD^harm_S > 1 point) <= 0.20 under every prior, ranked by the primary prior (ties:
the smaller subset). A sequencing lesson (intent ``sequence-*``) that does not pass but holds E2-E4 and
P(RD_l > 0) >= 0.5 under every prior (so it fails only E1, under some prior) may be listed as
the single "sequencing candidate" for the conversation gate (a one-step metric can miss
a sequencing benefit). The advanced lessons go to the small conversation-level gate;
only that gate freezes a library.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPORT_SCHEMA = "xp-fgate-report-v1"
DELTA = 0.01
RULES = {"delta": DELTA, "p_efficacy": 0.90, "context_harm_delta": -0.01,
         "p_context_harm": 0.10, "harm_delta": 0.01, "p_harm": 0.20,
         "harm_bound": 0.025, "p_harm_bound": 0.95, "min_exposure": 100,
         "shortlist": 4, "p_sequencing": 0.5}
S_A_GRID = (1.0, 2.0, 3.0)
TAU_GRID = (0.25, 0.5, 1.0)  # x s_b
M_SD = 2.5
PAIR_SAME, PAIR_CROSS = 0.25, 0.1  # interaction prior SDs as multiples of s_b
BASE_RD_SD = 0.03  # the base (sceptical) scale: interactions and tau are multiples of it
# Named priors for the lesson main effect on the risk-difference scale (owner 2026-10-18).
# A lesson passes only if the gate holds under all three; ``sceptical`` is primary (ranking,
# reported estimates). ``literature``: learned prompt-level lessons are mostly neutral or
# mildly harmful out of distribution (docs/lit/generalization-2026-10-17.md: SEAGym
# cross-backend OOD "mostly neutral or negative", TF-GRPO OOD -2.5, PRISM GEPA prompt-only
# -4.7 held-out, Recuris +2.0 n.s. at 9B), so success is centred at -1 point. Harm is
# centred at 0 under every prior (the evidence is about task success, not wrong writes).
PRIORS: dict[str, dict[str, Any]] = {
    "sceptical": {"mean": 0.0, "sd": 0.03, "primary": True},
    "weak": {"mean": 0.0, "sd": 0.06},
    "literature": {"mean": -0.01, "sd": 0.03}}


# --------------------------------------------------------------------------- data


class Design:
    """Rows of one outcome: prefix index, context index, inclusion matrix."""

    def __init__(self, rows: list[dict[str, Any]], outcome: str, n_lessons: int,
                 tools: list[list[str]] | None = None) -> None:
        self.rows = rows
        self.L = n_lessons
        self.contexts = sorted({r["context"] for r in rows})
        cidx = {c: i for i, c in enumerate(self.contexts)}
        prefixes = sorted({r["case_id"] for r in rows})
        pidx = {p: i for i, p in enumerate(prefixes)}
        self.prefixes = prefixes
        self.c = np.array([cidx[r["context"]] for r in rows], dtype=np.int64)
        self.i = np.array([pidx[r["case_id"]] for r in rows], dtype=np.int64)
        self.x = np.array([[int(b) for b in r["x"]] for r in rows], dtype=np.float64)
        if self.x.shape[1] != n_lessons:
            raise ValueError("inclusion vectors differ from the pool size")
        self.y = np.array([int(r[outcome]) for r in rows], dtype=np.float64)
        self.anchor = ~self.x.any(axis=1)
        self.pairs = list(itertools.combinations(range(n_lessons), 2))
        tools = tools or [["any"]] * n_lessons
        self.same_tool = np.array([bool(set(tools[a]) & set(tools[b])) for a, b in self.pairs])
        self.C = len(self.contexts)
        self.n_prefix = len(prefixes)

    @property
    def k(self) -> int:
        return self.C + self.L + self.L * self.C + len(self.pairs)

    def matrix(self, x: np.ndarray | None = None) -> np.ndarray:
        """Columns: context means | lesson mains | lesson x context | pairs."""
        x = self.x if x is None else x
        n = len(self.c)
        ctx = np.zeros((n, self.C))
        ctx[np.arange(n), self.c] = 1.0
        lc = np.zeros((n, self.L * self.C))
        for c in range(self.C):
            m = self.c == c
            lc[m, c::self.C] = x[m]  # column l * C + c
        pairs = np.stack([x[:, a] * x[:, b] for a, b in self.pairs], axis=1) \
            if self.pairs else np.zeros((n, 0))
        return np.hstack([ctx, x, lc, pairs])


def prior_sds(d: Design, s_b: float, tau: float, s_main: float | None = None) -> np.ndarray:
    """Prior SDs. ``s_b`` scales the interactions; ``s_main`` (default ``s_b``) is the SD of
    the lesson main effects (the named priors change only the main-effect prior)."""
    pair_sd = np.where(d.same_tool, PAIR_SAME * s_b, PAIR_CROSS * s_b)
    return np.concatenate([np.full(d.C, M_SD), np.full(d.L, s_b if s_main is None else s_main),
                           np.full(d.L * d.C, tau), pair_sd])


def prior_means(d: Design, mu_main: float = 0.0) -> np.ndarray:
    """Prior means: ``mu_main`` for every lesson main effect b_l, 0 elsewhere."""
    mu = np.zeros(d.k)
    mu[d.C:d.C + d.L] = mu_main
    return mu


# --------------------------------------------------------------------------- fit


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _loglik(y: np.ndarray, eta: np.ndarray) -> float:
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def fit(d: Design, *, s_a: float, s_b: float, tau: float, s_main: float | None = None,
        mu_main: float = 0.0, max_iter: int = 50, tol: float = 1e-7) -> dict[str, Any]:
    """Posterior mode and Laplace quantities (u eliminated by Schur complement). Lesson main
    effects b_l ~ N(mu_main, s_main^2) (``s_main`` default ``s_b``)."""
    X = d.matrix()
    sd = prior_sds(d, s_b, tau, s_main)
    prec = 1.0 / sd ** 2
    mu = prior_means(d, mu_main)
    n_i = d.n_prefix
    u = np.zeros(n_i)
    th = np.zeros(X.shape[1])

    def nlp(u: np.ndarray, th: np.ndarray) -> float:
        eta = X @ th + u[d.i]
        return (-_loglik(d.y, eta) + 0.5 * float(u @ u) / s_a ** 2
                + 0.5 * float((th - mu) ** 2 @ prec))

    cur = nlp(u, th)
    for _ in range(max_iter):
        eta = X @ th + u[d.i]
        p = _sigmoid(eta)
        w = p * (1 - p)
        r = d.y - p
        g_u = -np.bincount(d.i, weights=r, minlength=n_i) + u / s_a ** 2
        g_t = -(X.T @ r) + (th - mu) * prec
        D = np.bincount(d.i, weights=w, minlength=n_i) + 1.0 / s_a ** 2
        B = np.zeros((n_i, X.shape[1]))
        np.add.at(B, d.i, w[:, None] * X)
        C = (X.T * w) @ X + np.diag(prec)
        S = C - (B.T / D) @ B
        rhs = -g_t + B.T @ (g_u / D)
        dt = np.linalg.solve(S, rhs)
        du = (-g_u - B @ dt) / D
        step = 1.0
        while True:
            new = nlp(u + step * du, th + step * dt)
            if new <= cur + 1e-12 or step < 1e-4:
                break
            step *= 0.5
        u, th = u + step * du, th + step * dt
        done = abs(cur - new) < tol * max(1.0, abs(cur)) and step * max(
            np.abs(dt).max(initial=0), np.abs(du).max(initial=0)) < 1e-5
        cur = new
        if done:
            break
    eta = X @ th + u[d.i]
    p = _sigmoid(eta)
    w = p * (1 - p)
    D = np.bincount(d.i, weights=w, minlength=n_i) + 1.0 / s_a ** 2
    B = np.zeros((n_i, X.shape[1]))
    np.add.at(B, d.i, w[:, None] * X)
    C = (X.T * w) @ X + np.diag(prec)
    S = C - (B.T / D) @ B
    with np.errstate(all="ignore"):  # spurious BLAS flags on some platforms; checked below
        sign, logdet_s = np.linalg.slogdet(S)
    ok = bool(sign > 0 and np.isfinite(logdet_s) and np.isfinite(th).all())
    loglik = _loglik(d.y, eta)
    penalty = 0.5 * float(u @ u) / s_a ** 2 + 0.5 * float((th - mu) ** 2 @ prec)
    log_ml = (loglik - penalty - n_i * math.log(s_a) - float(np.log(sd).sum())
              - 0.5 * (float(np.log(D).sum()) + logdet_s))
    return {"u": u, "theta": th, "D": D, "B": B, "S": S, "log_ml": log_ml,
            "s_a": s_a, "s_b": s_b, "tau": tau, "s_main": float(sd[d.C]) if d.L else s_b,
            "mu_main": mu_main, "converged": ok}


def draws(d: Design, f: dict[str, Any], n: int, rng: np.random.Generator
          ) -> tuple[np.ndarray, np.ndarray]:
    """(theta draws n x k, u draws n x prefixes) from the Laplace approximation."""
    cov = np.linalg.inv(f["S"])
    cov = 0.5 * (cov + cov.T)
    th = rng.multivariate_normal(f["theta"], cov, size=n, method="cholesky")
    shift = (f["B"] @ (th - f["theta"]).T).T / f["D"]  # n x prefixes
    u = f["u"] - shift + rng.standard_normal((n, d.n_prefix)) / np.sqrt(f["D"])
    return th, u


def _split(d: Design, th: np.ndarray) -> tuple[np.ndarray, ...]:
    C, L = d.C, d.L
    m = th[:, :C]
    b = th[:, C:C + L]
    dlc = th[:, C + L:C + L + L * C].reshape(-1, L, C)
    g = th[:, C + L + L * C:]
    G = np.zeros((th.shape[0], L, L))
    for j, (a, bb) in enumerate(d.pairs):
        G[:, a, bb] = g[:, j]
        G[:, bb, a] = g[:, j]
    return m, b, dlc, G


def risk_differences(d: Design, th: np.ndarray, u: np.ndarray, *, weights: str = "equal"
                     ) -> dict[str, np.ndarray]:
    """Per draw: RD_l (draws x L) and RD_lc (draws x L x C) over the randomized rows."""
    rows = ~d.anchor
    x, c, i = d.x[rows], d.c[rows], d.i[rows]
    m, b, dlc, G = _split(d, th)
    n_draw = th.shape[0]
    rd = np.zeros((n_draw, d.L))
    rdc = np.zeros((n_draw, d.L, d.C))
    counts = np.bincount(c, minlength=d.C).astype(float)
    w = (np.where(counts > 0, 1.0, 0.0) / max(1, (counts > 0).sum())) if weights == "equal" \
        else counts / counts.sum()
    for s in range(n_draw):
        main = b[s][None, :] + dlc[s][:, c].T  # rows x L
        eta = m[s][c] + u[s][i] + (x * main).sum(1) + 0.5 * ((x @ G[s]) * x).sum(1)
        delta = main + x @ G[s]  # change of eta when l switches on (others as observed)
        e1 = eta[:, None] + (1 - x) * delta
        e0 = eta[:, None] - x * delta
        diff = _sigmoid(e1) - _sigmoid(e0)
        for cc in range(d.C):
            mask = c == cc
            if mask.any():
                rdc[s, :, cc] = diff[mask].mean(0)
        rd[s] = rdc[s] @ w
    return {"rd": rd, "rdc": rdc, "weights": w}


def subset_rd(d: Design, th: np.ndarray, u: np.ndarray, subset: tuple[int, ...],
              mode: str = "randomized") -> np.ndarray:
    """Per draw, contexts weighted equally. ``randomized`` (the gate's joint contrast):
    every lesson of ``subset`` on vs every one off, the other lessons as randomized, over
    the randomized rows. ``exact``: exactly ``subset`` vs the empty library on every
    prefix (descriptive: an extrapolation corner of a p = 0.5 design)."""
    m, b, dlc, G = _split(d, th)
    s_idx = list(subset)
    if mode == "exact":
        first = np.unique(d.i, return_index=True)[1]
        c, i = d.c[first], d.i[first]
        x_on = np.zeros((len(first), d.L))
        x_on[:, s_idx] = 1.0
        x_off = np.zeros_like(x_on)
    else:
        rows = ~d.anchor
        c, i = d.c[rows], d.i[rows]
        x_on = d.x[rows].copy()
        x_on[:, s_idx] = 1.0
        x_off = d.x[rows].copy()
        x_off[:, s_idx] = 0.0
    counts = np.bincount(c, minlength=d.C).astype(float)
    w = np.where(counts > 0, 1.0, 0.0) / max(1, (counts > 0).sum())
    out = np.zeros(th.shape[0])
    for s in range(th.shape[0]):
        main = b[s][None, :] + dlc[s][:, c].T
        base = m[s][c] + u[s][i]

        def eta(x: np.ndarray, base: np.ndarray = base, main: np.ndarray = main,
                g: np.ndarray = G[s]) -> np.ndarray:
            return base + (x * main).sum(1) + 0.5 * ((x @ g) * x).sum(1)

        diff = _sigmoid(eta(x_on)) - _sigmoid(eta(x_off))
        per = np.array([diff[c == cc].mean() if (c == cc).any() else 0.0 for cc in range(d.C)])
        out[s] = per @ w
    return out


# --------------------------------------------------------------------------- analysis


def prior_scale(d: Design, rd_prior_sd: float) -> float:
    p0 = float(d.y[d.anchor].mean()) if d.anchor.any() else float(d.y.mean())
    p0 = min(max(p0, 0.1), 0.9)
    return rd_prior_sd / (p0 * (1 - p0))


def fit_outcome(rows: list[dict[str, Any]], outcome: str, lessons: list[dict[str, Any]], *,
                rd_prior_sd: float = 0.03, n_draws: int = 1000, seed: int = 0,
                eb: bool = True, s_a: float = 2.0, tau_mult: float = 0.5,
                prior_mult: float = 1.0, weights: str = "equal", rd_prior_mean: float = 0.0,
                interaction_rd_sd: float | None = None) -> dict[str, Any]:
    """Fit one outcome. Lesson main effects b_l ~ N(mu, s_b^2) with mu and s_b the
    risk-difference prior (``rd_prior_mean``, ``rd_prior_sd``) mapped to the logit scale at
    the anchors' baseline. Interactions and the context-deviation scale tau are multiples of
    ``interaction_rd_sd`` mapped the same way (default ``rd_prior_sd``: the original single
    scale); the named priors pass the base 3 points so only the main-effect prior varies."""
    d = Design(rows, outcome, len(lessons), [les.get("tools") or ["any"] for les in lessons])
    s_b = prior_scale(d, rd_prior_sd) * prior_mult
    s_int = prior_scale(d, rd_prior_sd if interaction_rd_sd is None else interaction_rd_sd) \
        * prior_mult
    mu_b = prior_scale(d, rd_prior_mean)  # same linear map (RD ~ p0 (1 - p0) b near p0)
    grid = []
    if eb:
        for sa in S_A_GRID:
            for tm in TAU_GRID:
                f = fit(d, s_a=sa, s_b=s_int, tau=tm * s_int, s_main=s_b, mu_main=mu_b)
                grid.append({"s_a": sa, "tau_mult": tm, "log_ml": f["log_ml"], "fit": f})
        best = max(grid, key=lambda g: g["log_ml"])
        f = best["fit"]
    else:
        f = fit(d, s_a=s_a, s_b=s_int, tau=tau_mult * s_int, s_main=s_b, mu_main=mu_b)
    rng = np.random.default_rng(seed)
    th, u = draws(d, f, n_draws, rng)
    rds = risk_differences(d, th, u, weights=weights)
    exposure = np.zeros((d.L, d.C), dtype=int)
    rows_mask = ~d.anchor
    for cc in range(d.C):
        m = rows_mask & (d.c == cc)
        exposure[:, cc] = d.x[m].sum(0).astype(int)
    return {"design": d, "fit": f, "theta": th, "u": u, "rd": rds["rd"], "rdc": rds["rdc"],
            "exposure": exposure, "s_b": s_b, "s_int": s_int, "mu_b": mu_b,
            "rd_prior": {"mean": rd_prior_mean, "sd": rd_prior_sd},
            "grid": [{k: v for k, v in g.items() if k != "fit"} for g in grid]}


def lesson_checks(succ: dict[str, Any], harm: dict[str, Any], lessons: list[dict[str, Any]],
                  rules: dict[str, Any] = RULES) -> list[dict[str, Any]]:
    """Per-lesson gate E1-E4 under one prior's posterior."""
    d = succ["design"]
    per = []
    for li, les in enumerate(lessons):
        rd, rdc = succ["rd"][:, li], succ["rdc"][:, li, :]
        hr = harm["rd"][:, li]
        p_eff = float((rd > rules["delta"]).mean())
        exposed = succ["exposure"][li] >= rules["min_exposure"]
        ctx_harm = [(d.contexts[c], float((rdc[:, c] < rules["context_harm_delta"]).mean()))
                    for c in range(d.C)]
        e3 = all(p < rules["p_context_harm"] for _, p in ctx_harm)
        ctx_bound = None
        if rules.get("context_harm_bound") is not None:  # E3 bound part (mirrors E4)
            ctx_bound = {d.contexts[c]: float((rdc[:, c] > rules["context_harm_bound"]).mean())
                         for c in range(d.C)}
            e3 = e3 and all(v >= rules["p_context_harm_bound"] for v in ctx_bound.values())
        p_harm = float((hr > rules["harm_delta"]).mean())
        p_bound = float((hr < rules["harm_bound"]).mean())
        checks = {"E1": p_eff >= rules["p_efficacy"], "E2": bool(exposed.all()),
                  "E3": e3,
                  "E4": p_harm <= rules["p_harm"] and p_bound >= rules["p_harm_bound"]}
        per.append({"id": les["id"], "intent": les.get("intent"),
                    "rd_mean": float(rd.mean()), "rd_ci90": _ci(rd), "p_rd_gt_delta": p_eff,
                    "p_rd_gt_0": float((rd > 0).mean()),
                    "rd_by_context": {d.contexts[c]: {"mean": float(rdc[:, c].mean()),
                                                      "ci90": _ci(rdc[:, c]),
                                                      "exposure": int(succ["exposure"][li, c])}
                                      for c in range(d.C)},
                    "p_context_harm": dict(ctx_harm),
                    "max_p_context_harm": max(p for _, p in ctx_harm),
                    "min_p_context_above_bound": (min(ctx_bound.values()) if ctx_bound
                                                  else None),
                    "harm_rd_mean": float(hr.mean()), "harm_rd_ci90": _ci(hr),
                    "p_harm_gt_delta": p_harm, "p_harm_lt_bound": p_bound,
                    "checks": checks, "passes": all(checks.values())})
    return per


def _sequencing_ok(p: dict[str, Any], rules: dict[str, Any]) -> bool:
    """Sequencing-candidate conditions under one prior other than failing: intent
    ``sequence-*``, E2-E4 hold and P(RD > 0) >= p_sequencing. The caller adds "does not
    pass" (so with E2-E4 holding, E1 is what fails)."""
    c = p["checks"]
    return ((p["intent"] or "").startswith("sequence-") and c["E2"] and c["E3"] and c["E4"]
            and p["p_rd_gt_0"] >= rules["p_sequencing"])


class _Joint:
    """Joint (subset) contrasts per prior, cached across the per-prior and final selections."""

    def __init__(self, fits: dict[str, tuple[dict[str, Any], dict[str, Any]]],
                 lessons: list[dict[str, Any]], rules: dict[str, Any],
                 cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] | None = None) -> None:
        self.fits, self.rules = fits, rules
        self.ids = [les["id"] for les in lessons]
        self.cache = {} if cache is None else cache

    def stats(self, prior: str, sub: tuple[str, ...]) -> dict[str, Any]:
        key = (prior, sub)
        if key not in self.cache:
            succ, harm = self.fits[prior]
            d, idx = succ["design"], tuple(self.ids.index(s) for s in sub)
            rd_s = subset_rd(d, succ["theta"], succ["u"], idx)
            hr_s = subset_rd(harm["design"], harm["theta"], harm["u"], idx)
            exact = subset_rd(d, succ["theta"], succ["u"], idx, mode="exact")
            st = {"rd_mean": float(rd_s.mean()), "rd_ci90": _ci(rd_s),
                  "exact_vs_empty_rd_mean": float(exact.mean()),
                  "exact_vs_empty_ci90": _ci(exact),
                  "p_rd_gt_delta": float((rd_s > self.rules["delta"]).mean()),
                  "harm_rd_mean": float(hr_s.mean()),
                  "p_harm_gt_delta": float((hr_s > self.rules["harm_delta"]).mean())}
            st["ok"] = (st["p_rd_gt_delta"] >= self.rules["p_efficacy"]
                        and st["p_harm_gt_delta"] <= self.rules["p_harm"])
            self.cache[key] = st
        return self.cache[key]

    def select(self, shortlist: list[str], priors: list[str], primary: str
               ) -> tuple[list[dict[str, Any]], list[str]]:
        """Every non-empty subset of ``shortlist``; a subset is eligible only if its joint
        rule holds under every prior in ``priors``; the advanced set is the eligible subset
        with the highest ``primary`` posterior mean RD_S (ties: the smaller subset)."""
        subsets = []
        for r in range(1, len(shortlist) + 1):
            for sub in itertools.combinations(shortlist, r):
                by = {pr: self.stats(pr, sub) for pr in priors}
                row = {"lessons": list(sub), **{k: v for k, v in by[primary].items()
                                                if k != "ok"},
                       "ok": all(v["ok"] for v in by.values())}
                if len(priors) > 1:
                    row["by_prior"] = by
                subsets.append(row)
        ok = [s for s in subsets if s["ok"]]
        advanced = max(ok, key=lambda s: (s["rd_mean"], -len(s["lessons"])))["lessons"] \
            if ok else []
        return subsets, advanced


def gate_priors(fits: dict[str, tuple[dict[str, Any], dict[str, Any]]],
                lessons: list[dict[str, Any]], rules: dict[str, Any] = RULES, *,
                primary: str | None = None,
                joint_cache: dict[Any, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The gate under several named priors (``fits``: prior -> (success fit, harm fit)).

    E1-E4 are evaluated per prior; a lesson passes only if it passes under **every** prior
    (intersection). The shortlist is the <= 4 intersection-passing lessons with the highest
    ``primary``-prior posterior mean RD; a subset is eligible only if its joint rule holds
    under every prior, and the advanced set is the eligible subset with the highest primary
    posterior mean RD_S. The sequencing candidate must qualify under every prior. Each
    prior's own stand-alone result (passing, shortlist, advanced) is reported in
    ``by_prior``; only the intersection is the decision. ``joint_cache`` (optional) shares the
    joint-subset statistics between calls on the same fits whose rules differ only in E1-E4
    thresholds (the simulation compares E3 variants on one set of fits)."""
    names = list(fits)
    primary = primary or names[0]
    per_prior = {pr: lesson_checks(*fits[pr], lessons, rules) for pr in names}
    joint = _Joint(fits, lessons, rules, joint_cache)
    by_prior: dict[str, Any] = {}
    for pr in names:
        passing = sorted((p for p in per_prior[pr] if p["passes"]), key=lambda p: -p["rd_mean"])
        sl = [p["id"] for p in passing[:rules["shortlist"]]]
        _, adv = joint.select(sl, [pr], pr)
        by_prior[pr] = {"passing": [p["id"] for p in passing], "shortlist": sl, "advanced": adv}
    combined = []
    for li in range(len(lessons)):
        rows = {pr: per_prior[pr][li] for pr in names}
        base = dict(rows[primary])
        base["checks"] = {k: all(rows[pr]["checks"][k] for pr in names)
                          for k in base["checks"]}
        base["passes_by_prior"] = {pr: rows[pr]["passes"] for pr in names}
        base["passes"] = all(base["passes_by_prior"].values())
        base["sequencing_ok"] = all(_sequencing_ok(rows[pr], rules) for pr in names)
        if len(names) > 1:
            base["by_prior"] = {pr: {k: rows[pr][k] for k in (
                "rd_mean", "rd_ci90", "p_rd_gt_delta", "p_rd_gt_0", "p_context_harm",
                "harm_rd_mean", "p_harm_gt_delta", "p_harm_lt_bound", "checks", "passes")}
                for pr in names}
        combined.append(base)
    passing = sorted((p for p in combined if p["passes"]), key=lambda p: -p["rd_mean"])
    shortlist = [p["id"] for p in passing[:rules["shortlist"]]]
    subsets, advanced = joint.select(shortlist, names, primary)
    seq = [p for p in combined if p["sequencing_ok"] and not p["passes"]]
    seq_candidate = max(seq, key=lambda p: p["rd_mean"])["id"] if seq else None
    table = [{"id": p["id"], **{pr: p["passes_by_prior"][pr] for pr in names},
              "final": p["passes"]} for p in combined]
    return {"lessons": combined, "shortlist": shortlist, "subsets": subsets,
            "advanced": advanced, "sequencing_candidate": seq_candidate, "rules": rules,
            "priors": names, "primary": primary, "by_prior": by_prior, "table": table}


def gate(succ: dict[str, Any], harm: dict[str, Any], lessons: list[dict[str, Any]],
         rules: dict[str, Any] = RULES) -> dict[str, Any]:
    """The gate under one prior (the fits given)."""
    g = gate_priors({"single": (succ, harm)}, lessons, rules)
    for k in ("priors", "primary", "by_prior", "table"):
        g.pop(k)
    return g


def fit_priors(rows: list[dict[str, Any]], lessons: list[dict[str, Any]], *,
               priors: dict[str, dict[str, Any]] | None = None, n_draws: int = 1000,
               seed: int = 0, **kw: Any) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """(success, harm) fits under each named prior (``PRIORS``). The named prior sets the
    lesson main-effect prior of success (mean, SD) and of harm (mean 0, same SD); the
    interactions and tau keep the base 3-point scale. Same seeds under every prior."""
    out = {}
    for name, pr in (priors or PRIORS).items():
        common = {"rd_prior_sd": pr["sd"], "interaction_rd_sd": BASE_RD_SD,
                  "n_draws": n_draws, **kw}
        out[name] = (fit_outcome(rows, "y", lessons, seed=seed, rd_prior_mean=pr["mean"],
                                 **common),
                     fit_outcome(rows, "harm", lessons, seed=seed + 1,
                                 rd_prior_mean=pr.get("harm_mean", 0.0), **common))
    return out


def _ci(v: np.ndarray) -> list[float]:
    return [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))]


def analyse(table: list[dict[str, Any]], pool: dict[str, Any], *, n_draws: int = 1000,
            seed: int = 0, sensitivities: bool = True) -> dict[str, Any]:
    lessons = pool["lessons"]
    rows = [r for r in table if r["track"] == "fgate"]
    errors = sum(1 for r in rows if r.get("error_class"))
    fits = fit_priors(rows, lessons, n_draws=n_draws, seed=seed)
    primary = next(k for k, v in PRIORS.items() if v.get("primary"))
    succ = fits[primary][0]
    reask = fit_outcome(rows, "reask", lessons, n_draws=n_draws, seed=seed + 2)
    g = gate_priors(fits, lessons, primary=primary)
    d = succ["design"]
    anchors = {c: {"n": int(((d.c == i) & d.anchor).sum()),
                   "success": float(d.y[(d.c == i) & d.anchor].mean())
                   if ((d.c == i) & d.anchor).any() else None}
               for i, c in enumerate(d.contexts)}
    report = {"schema": REPORT_SCHEMA, "pool_sha256": pool.get("sha256"), "rows": len(rows),
              "prefixes": d.n_prefix, "error_rows": errors, "contexts": d.contexts,
              "anchors": anchors, "s_b": succ["s_b"], "eb_grid": succ["grid"],
              "chosen": {"s_a": succ["fit"]["s_a"], "tau": succ["fit"]["tau"]},
              "priors": {name: {"rd_mean": pr["mean"], "rd_sd": pr["sd"],
                                "primary": name == primary,
                                "s_b": fits[name][0]["s_b"], "mu_b": fits[name][0]["mu_b"],
                                "chosen": {"s_a": fits[name][0]["fit"]["s_a"],
                                           "tau": fits[name][0]["fit"]["tau"]}}
                         for name, pr in PRIORS.items()},
              "gate": g,
              "reask": {les["id"]: {"rd_mean": float(reask["rd"][:, k].mean()),
                                    "rd_ci90": _ci(reask["rd"][:, k])}
                        for k, les in enumerate(lessons)}}
    if sensitivities:
        sens = {}
        # (prior x2 is superseded by the named weakly informative prior, which is gated)
        f2 = fit_priors(rows, lessons, n_draws=n_draws, seed=seed, weights="sampled")
        sens["weights_sampled"] = {"advanced": gate_priors(f2, lessons,
                                                           primary=primary)["advanced"]}
        clean = [r for r in rows if not r.get("error_class")]
        if len(clean) != len(rows):
            f3 = fit_priors(clean, lessons, n_draws=n_draws, seed=seed)
            sens["without_error_rows"] = {"advanced": gate_priors(
                f3, lessons, primary=primary)["advanced"]}
        for g_pt in succ["grid"]:  # empirical-Bayes grid under the primary prior
            s4 = fit_outcome(rows, "y", lessons, n_draws=n_draws, seed=seed, eb=False,
                             s_a=g_pt["s_a"], tau_mult=g_pt["tau_mult"],
                             rd_prior_sd=PRIORS[primary]["sd"],
                             rd_prior_mean=PRIORS[primary]["mean"],
                             interaction_rd_sd=BASE_RD_SD)
            sens[f"s_a={g_pt['s_a']},tau={g_pt['tau_mult']}xs_b"] = {
                "rd_mean": [round(float(s4["rd"][:, k].mean()), 4) for k in range(len(lessons))]}
        report["sensitivity"] = sens
    return report


DIFFICULTY_ARMS = ("original", "synthetic")


def pair_interval(d: np.ndarray, rng: np.random.Generator, *, n_boot: int = 4000
                  ) -> dict[str, Any]:
    """Mean of the per-pair differences ``d`` with percentile-bootstrap 90% and 95%
    intervals over pairs (pairs are independent configuration slots)."""
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    q = np.quantile(boots, [0.025, 0.05, 0.95, 0.975])
    return {"pairs": len(d), "diff": float(d.mean()), "ci90": [float(q[1]), float(q[2])],
            "ci95": [float(q[0]), float(q[3])]}


def difficulty_verdict(s: dict[str, Any], *, large: float, equivalence: float
                       ) -> dict[str, Any]:
    """Pre-specified verdict (plan section 3.6). Primary question: is there a difficulty
    shift, powered for a large one (|diff| >= ``large``)?
      ``shift``           the two-sided 95% CI excludes 0 (alpha 0.05);
      ``no large shift``  not ``shift`` and the 90% CI lies inside (-large, +large)
                          (two one-sided tests at alpha 0.05 against +-large);
      ``inconclusive``    otherwise.
    ``equivalent_reported`` (secondary, reported only, not a planned claim): the 90% CI
    happens to lie inside (-equivalence, +equivalence)."""
    lo95, hi95 = s["ci95"]
    lo90, hi90 = s["ci90"]
    if lo95 > 0 or hi95 < 0:
        verdict = "shift"
    elif -large < lo90 and hi90 < large:
        verdict = "no large shift"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "large_shift_threshold": large,
            "estimate_at_least_large": abs(s["diff"]) >= large,
            "equivalent_reported": -equivalence < lo90 and hi90 < equivalence}


def difficulty_completeness(rows: list[dict[str, Any]], sizes: dict[str, int], *,
                            phrasings: int = 3) -> list[str]:
    """Problems that make the difficulty data incomplete (empty = complete): every
    registered pair ``task:0 .. task:n-1`` of ``sizes``, both arms, exactly ``phrasings``
    distinct phrasings (``variant`` 0 .. phrasings-1) per pair and arm, each observed
    exactly once (unique unit and unique (pair, arm, variant)), nothing unregistered."""
    problems: list[str] = [] if rows else ["no difficulty rows (sizes are registered)"]
    expected = {f"{t}:{i}" for t, n in sizes.items() for i in range(int(n))}
    units = [r.get("unit_id") for r in rows]
    if None in units or len(set(units)) != len(units):
        problems.append("duplicate or missing unit ids")
    cells: dict[tuple[str, str], list[Any]] = {}
    for r in rows:
        pair, arm = r.get("pair"), r.get("arm")
        if arm not in DIFFICULTY_ARMS or not isinstance(pair, str) or ":" not in pair:
            problems.append(f"row {r.get('unit_id')}: no arm/pair")
            continue
        if pair.split(":", 1)[0] != r.get("task"):
            problems.append(f"row {r.get('unit_id')}: pair {pair} of another task")
        cells.setdefault((pair, arm), []).append(r.get("variant"))
    seen = {p for p, _ in cells}
    if extra := sorted(seen - expected):
        problems.append(f"{len(extra)} unregistered pairs (e.g. {extra[0]})")
    want = list(range(phrasings))
    bad = 0
    for pair in sorted(expected):
        for arm in DIFFICULTY_ARMS:
            vs = cells.get((pair, arm), [])
            if sorted(v for v in vs if isinstance(v, int)) != want or len(vs) != phrasings:
                bad += 1
    if bad:
        problems.append(f"{bad} of {2 * len(expected)} registered (pair, arm) cells lack "
                        f"exactly the {phrasings} phrasings once each")
    return problems


def difficulty_check(table: list[dict[str, Any]], *, sizes: dict[str, int] | None = None,
                     phrasings: int = 3, large: float | None = None,
                     equivalence: float | None = None, n_boot: int = 4000, seed: int = 0
                     ) -> dict[str, Any] | None:
    """Synthetic-difficulty check (pre-specified; plan section 3.6). Rows of track
    ``difficulty``: B1 (empty library) on matched prefixes, ``arm`` original | synthetic,
    ``pair`` = task:slot (same kind, family, pattern, style in both arms: enforced at
    build), ``variant`` = phrasing. **Completeness first**: the registered pairs (``sizes``,
    default ``design.DIFFICULTY_SIZES``; pass the plan's), both arms, every phrasing once
    (``difficulty_completeness``); otherwise the verdict is ``incomplete`` and no estimate
    or interval is reported. Then per pair the arm's mean over its phrasings; estimate =
    mean of (synthetic - original) over pairs; percentile bootstrap over pairs; verdict by
    ``difficulty_verdict`` on success (primary: shift, powered for |diff| >= ``large``,
    default ``design.DIFFICULTY_LARGE_SHIFT``; +-``equivalence`` reported only). Harm,
    re-ask and each task are reported with the same estimator (not gated). None only when
    there are no difficulty rows and no sizes were registered (``sizes`` None); registered
    sizes with no difficulty rows at all are ``incomplete``, never silently absent."""
    from slh.xp.fgate import design as D

    rows = [r for r in table if r.get("track") == "difficulty"]
    if not rows and not sizes:
        return None
    sizes = dict(D.DIFFICULTY_SIZES if sizes is None else sizes)
    large = D.DIFFICULTY_LARGE_SHIFT if large is None else large
    equivalence = D.DIFFICULTY_EQUIVALENCE if equivalence is None else equivalence
    head = {"large_shift": large, "equivalence_reported": equivalence, "sizes": sizes,
            "phrasings": phrasings, "rows": len(rows),
            "expected_rows": 2 * phrasings * sum(sizes.values()),
            "errors": sum(1 for r in rows if r.get("error_class"))}
    if problems := difficulty_completeness(rows, sizes, phrasings=phrasings):
        return {**head, "verdict": "incomplete", "complete": False, "problems": problems}
    rng = np.random.default_rng(seed)

    def summary(rs: list[dict[str, Any]], outcome: str) -> dict[str, Any]:
        cell: dict[str, dict[str, list[int]]] = {}
        for r in rs:
            cell.setdefault(r["pair"], {}).setdefault(r["arm"], []).append(int(r[outcome]))
        d = np.array([np.mean(v["synthetic"]) - np.mean(v["original"])
                      for _, v in sorted(cell.items())], dtype=float)
        s = pair_interval(d, rng, n_boot=n_boot)
        arms = {a: float(np.mean([int(r[outcome]) for r in rs if r["arm"] == a]))
                for a in DIFFICULTY_ARMS}
        return {**s, "rate": arms, "excludes_zero_95": s["ci95"][0] > 0 or s["ci95"][1] < 0}

    succ = summary(rows, "y")
    return {**head, "complete": True, "success": succ,
            **difficulty_verdict(succ, large=large, equivalence=equivalence),
            "harm": summary(rows, "harm"), "reask": summary(rows, "reask"),
            "by_task": {t: summary([r for r in rows if r["task"] == t], "y")
                        for t in sorted(sizes)}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.fgate.analysis")
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--plan", type=Path,
                    help="the package's plan.json: registered difficulty-check sizes and "
                         "thresholds (default: design.DIFFICULTY_SIZES, the full design)")
    args = ap.parse_args(argv)
    from slh.xp.fgate import pool as fp
    from slh.xp.partner import common as pc

    pool = fp.verify_pool(pc.read_json(args.pool))
    table = pc.read_jsonl(args.table)
    report = analyse(table, pool, n_draws=args.draws,
                     sensitivities=not args.no_sensitivity)
    reg = (pc.read_json(args.plan).get("difficulty") or {}) if args.plan else {}
    report["difficulty_check"] = difficulty_check(
        table, sizes=reg.get("sizes"), phrasings=reg.get("phrasings", 3),
        large=reg.get("large_shift"), equivalence=reg.get("equivalence_reported"))
    pc.write_json(args.out, report)
    dc = report["difficulty_check"]
    print(json.dumps({"difficulty_verdict": dc and dc["verdict"],
                      "advanced": report["gate"]["advanced"],
                      "advanced_by_prior": {k: v["advanced"]
                                            for k, v in report["gate"]["by_prior"].items()},
                      "shortlist": report["gate"]["shortlist"],
                      "sequencing_candidate": report["gate"]["sequencing_candidate"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
