"""The registered interval: crossed run x group random-effects model, task fixed.

Plan v1.2 (docs/xp/plan-v1.md, "Analysis"): per case paired difference (endpoint
minus control, same test case and run), averaged within scenario group, crossed
with learning run, pooled over tasks with equal task weights; interval from a
parametric bootstrap of the crossed random-effects model (10,000 draws, seed
registered here).

For task k the input is a matrix ``D_k`` (runs x groups) of group-mean
differences in points. Moment (ANOVA) estimators as in ``slh.xp.power``:
``MS_R``, ``MS_G``, ``MS_E``; ``sa2 = max(0, (MS_R - MS_E)/G)``,
``sb2 = max(0, (MS_G - MS_E)/R)``, ``se2 = MS_E``;
``Var(mean_k) = sa2/R + sb2/G + se2/(R G)`` (truncated components dropped) and,
pooled, ``Var = s2(D_r)/R + sum_k sb2_k/(K^2 G_k)`` where ``D_r`` is run r's
task-averaged difference (its spread carries the run effects, including the part
common to the tasks of one run, and the residual).

Parametric bootstrap: the fitted model (run effects multivariate normal across
tasks with the measured between-task covariance, group and residual terms per
task) is simulated at mean 0 through its sufficient statistics (run means
normal; MS_G and MS_E scaled chi-square), the same estimator is applied to every
draw, and the studentised draws ``t* = est*/se*`` give the interval
``est - q(0.975) se, est - q(0.025) se`` (bootstrap-t). The studentised form is
the conservative reading of "parametric bootstrap": a plain percentile interval
ignores that the variance components are themselves estimated from 12 runs (it
is reported alongside, with the closed-form Satterthwaite interval of
``slh.xp.power``).

A non-parametric pigeonhole bootstrap (runs and clusters resampled
independently) is a descriptive sensitivity check, with clusters = groups,
request families, or users.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from slh.xp.power import t_quantile

DRAWS = 10_000  # registered (plan v1.2)
SEED = 20261010  # registered: the analysis script's bootstrap seed
SENSITIVITY_DRAWS = 2_000
ALPHA = 0.05


# --------------------------------------------------------------------------- moments


def anova(d: np.ndarray) -> dict[str, Any]:
    """Two-way ANOVA without replication on one task's runs x groups matrix."""
    r, g = d.shape
    rows, cols, grand = d.mean(axis=1), d.mean(axis=0), d.mean()
    ms_r = g * rows.var(ddof=1) if r > 1 else float("nan")
    ms_g = r * cols.var(ddof=1) if g > 1 else float("nan")
    resid = d - rows[:, None] - cols[None, :] + grand
    df_e = (r - 1) * (g - 1)
    ms_e = float((resid ** 2).sum() / df_e) if df_e > 0 else float("nan")
    return {"runs": r, "groups": g, "row_means": rows, "ms_r": float(ms_r),
            "ms_g": float(ms_g), "ms_e": ms_e}


def estimate(d_run: np.ndarray, ms_g: np.ndarray, ms_e: np.ndarray, groups: np.ndarray
             ) -> dict[str, np.ndarray]:
    """Vectorised estimator over leading axes.

    ``d_run``: (..., K, R) run means per task; ``ms_g``, ``ms_e``: (..., K);
    ``groups``: (K,) groups per task. Returns per-task and pooled estimates,
    variances and Satterthwaite degrees of freedom."""
    k, r = d_run.shape[-2], d_run.shape[-1]
    g = np.asarray(groups, dtype=float)
    rg = r * g
    est_k = d_run.mean(axis=-1)
    ms_r = g * d_run.var(axis=-1, ddof=1)
    df_r, df_g, df_e = r - 1, g - 1, (r - 1) * (g - 1)
    a_on = ms_r > ms_e
    b_on = ms_g > ms_e
    c_r, c_g = a_on / rg, b_on / rg
    c_e = (1.0 - a_on - b_on) / rg
    v_k = np.maximum(c_r * ms_r + c_g * ms_g + c_e * ms_e, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        den = (c_r * ms_r) ** 2 / df_r + (c_g * ms_g) ** 2 / df_g + (c_e * ms_e) ** 2 / df_e
        df_k = np.where(den > 0, v_k ** 2 / den, np.inf)
    d_pool = d_run.mean(axis=-2)
    est = d_pool.mean(axis=-1)
    s_run = d_pool.var(axis=-1, ddof=1) / r
    cg = b_on / (k * k * rg)
    v = np.maximum(s_run + (cg * (ms_g - ms_e)).sum(axis=-1), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        den = s_run ** 2 / df_r + ((cg * ms_g) ** 2 / df_g + (cg * ms_e) ** 2 / df_e).sum(
            axis=-1)
        df = np.where(den > 0, v ** 2 / den, np.inf)
    return {"est_task": est_k, "var_task": v_k, "df_task": df_k, "est": est, "var": v,
            "df": df}


def _psd(m: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((m + m.T) / 2)
    return (v * np.clip(w, 0, None)) @ v.T


def _quantiles(t: np.ndarray) -> tuple[float, float] | None:
    t = t[np.isfinite(t)]
    if t.size < 100:
        return None
    lo, hi = np.quantile(t, [ALPHA / 2, 1 - ALPHA / 2])
    return float(lo), float(hi)


def _interval(est: float, se: float, q: tuple[float, float] | None) -> list[float] | None:
    if not math.isfinite(est):
        return None
    if se == 0:
        return [est, est]
    if q is None:
        return None
    return [est - q[1] * se, est - q[0] * se]


def crossed(mats: dict[str, np.ndarray], *, draws: int = DRAWS, seed: int = SEED
            ) -> dict[str, Any]:
    """Registered analysis of one contrast.

    ``mats``: task -> (R, G_k) group-mean differences in points; every task has
    the same runs in the same order. Returns pooled (equal task weights) and
    per-task estimates with bootstrap-t, percentile and Satterthwaite intervals."""
    tasks = list(mats)
    if not tasks:
        raise ValueError("no task to analyse")
    runs = {m.shape[0] for m in mats.values()}
    if len(runs) != 1:
        raise ValueError("every task needs the same learning runs")
    r = runs.pop()
    fits = {t: anova(np.asarray(mats[t], dtype=float)) for t in tasks}
    groups = np.array([fits[t]["groups"] for t in tasks], dtype=float)
    d_run = np.stack([fits[t]["row_means"] for t in tasks])  # (K, R)
    ms_g = np.array([fits[t]["ms_g"] for t in tasks])
    ms_e = np.array([fits[t]["ms_e"] for t in tasks])
    k = len(tasks)
    out: dict[str, Any] = {"tasks": tasks, "runs": r,
                           "groups": {t: fits[t]["groups"] for t in tasks}}
    if r < 2 or groups.min() < 2:
        est_k = d_run.mean(axis=1)
        out.update(estimate=float(est_k.mean()), interval=None,
                   per_task={t: {"estimate": float(est_k[i]), "interval": None}
                             for i, t in enumerate(tasks)},
                   note="too few runs or groups for an interval")
        return out
    obs = estimate(d_run, ms_g, ms_e, groups)
    # Fitted components.
    sa2 = np.maximum((np.array([fits[t]["ms_r"] for t in tasks]) - ms_e) / groups, 0.0)
    sb2 = np.maximum((ms_g - ms_e) / r, 0.0)
    se2 = np.maximum(ms_e, 0.0)
    cov = np.cov(d_run) if k > 1 else np.array([[0.0]])
    cov = np.atleast_2d(cov)
    sigma = cov.copy()
    np.fill_diagonal(sigma, sa2)
    # A correlation above 1 is impossible: clip off-diagonals to the components.
    lim = np.sqrt(np.outer(sa2, sa2))
    sigma = np.clip(sigma, -lim, lim)
    np.fill_diagonal(sigma, sa2)
    sigma = _psd(sigma)
    rng = np.random.default_rng(seed)
    a = rng.multivariate_normal(np.zeros(k), sigma, size=(draws, r), method="eigh")
    a = np.swapaxes(a, 1, 2)  # (B, K, R)
    b_bar = rng.normal(0, 1, (draws, k, 1)) * np.sqrt(sb2 / groups)[None, :, None]
    e_bar = rng.normal(0, 1, (draws, k, r)) * np.sqrt(se2 / groups)[None, :, None]
    d_star = a + b_bar + e_bar
    df_g, df_e = groups - 1, (r - 1) * (groups - 1)
    ms_g_star = (se2 + r * sb2) * rng.chisquare(df_g, (draws, k)) / df_g
    ms_e_star = se2 * rng.chisquare(df_e, (draws, k)) / df_e
    boot = estimate(d_star, ms_g_star, ms_e_star, groups)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_pool = boot["est"] / np.sqrt(boot["var"])
        t_task = boot["est_task"] / np.sqrt(boot["var_task"])

    def summary(est: float, var: float, df: float, t: np.ndarray, e_star: np.ndarray
                ) -> dict[str, Any]:
        se = math.sqrt(max(var, 0.0))
        tq = float(t_quantile(0.975, df)) if math.isfinite(df) else 1.959963984540054
        pct = np.quantile(e_star, [ALPHA / 2, 1 - ALPHA / 2])
        return {"estimate": est, "se": se,
                "interval": _interval(est, se, _quantiles(t)),
                "percentile_interval": [est - float(pct[1]), est - float(pct[0])],
                "satterthwaite_interval": [est - tq * se, est + tq * se],
                "df": df if math.isfinite(df) else None}

    out.update(summary(float(obs["est"]), float(obs["var"]), float(obs["df"]), t_pool,
                       boot["est"]))
    out["per_task"] = {t: summary(float(obs["est_task"][i]), float(obs["var_task"][i]),
                                  float(obs["df_task"][i]), t_task[:, i],
                                  boot["est_task"][:, i]) for i, t in enumerate(tasks)}
    out["components"] = {t: {"run_sd": math.sqrt(sa2[i]), "group_sd": math.sqrt(sb2[i]),
                             "residual_sd": math.sqrt(se2[i])} for i, t in enumerate(tasks)}
    pooled_runs = d_run.mean(axis=0)
    out["run_means"] = [float(x) for x in pooled_runs]
    out["between_run_sd"] = float(pooled_runs.std(ddof=1))
    out["draws"], out["seed"] = draws, seed
    return out


# --------------------------------------------------------------------------- sensitivity


def pigeonhole(sums: dict[str, np.ndarray], counts: dict[str, np.ndarray], *,
               draws: int = SENSITIVITY_DRAWS, seed: int = SEED + 1) -> dict[str, Any]:
    """Non-parametric crossed bootstrap: runs (jointly across tasks) and clusters
    (per task) resampled with replacement. ``sums[t]`` and ``counts[t]`` are
    (R, C) matrices of summed group-mean differences and group counts per run
    and cluster; the estimate is the group-weighted mean per task, then the
    equal-weight mean over tasks. Percentile interval."""
    tasks = list(sums)
    r = sums[tasks[0]].shape[0]
    rng = np.random.default_rng(seed)
    wr = rng.multinomial(r, np.full(r, 1 / r), size=draws).astype(float)
    per = {}
    for t in tasks:
        s, n = np.asarray(sums[t], float), np.asarray(counts[t], float)
        c = s.shape[1]
        wc = rng.multinomial(c, np.full(c, 1 / c), size=draws).astype(float)
        num = np.einsum("br,rc,bc->b", wr, s, wc)
        den = np.einsum("br,rc,bc->b", wr, n, wc)
        with np.errstate(divide="ignore", invalid="ignore"):
            per[t] = num / den
    pooled = np.mean([per[t] for t in tasks], axis=0)
    est = float(np.mean([sums[t].sum() / counts[t].sum() for t in tasks]))
    ok = pooled[np.isfinite(pooled)]
    lo, hi = np.quantile(ok, [ALPHA / 2, 1 - ALPHA / 2])
    return {"estimate": est, "interval": [float(lo), float(hi)], "draws": draws,
            "clusters": {t: int(sums[t].shape[1]) for t in tasks}}
