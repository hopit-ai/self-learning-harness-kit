"""Finite-sample valid one-sided bounds for paired binary differences (confirmation and monitor).

Paired data: for each unit, d = a - b with a, b in {0, 1}; N+ = #(a=1, b=0), N- = #(a=0, b=1),
n units. The estimand is delta = p+ - p-. N+ ~ Binomial(n, p+) and N- ~ Binomial(n, p-)
marginally (whatever their dependence), so with exact Clopper-Pearson bounds at level a/2 each:

    P(UCB_{a/2}(p+) < p+) <= a/2   and   P(LCB_{a/2}(p-) > p-) <= a/2

and by the union bound ``delta_upper = UCB_{a/2}(N+) - LCB_{a/2}(N-)`` satisfies
P(delta_upper < delta) <= a for every (p+, p-), every n, every discordance pattern (Bonferroni
split of two exact marginal bounds; no normal approximation). Symmetrically for the lower bound.
Zero discordance still gives a positive width (UCB(0) = 1 - (a/2)^(1/n)). Units drawn without
replacement from a finite pool give hypergeometric counts, for which the binomial
Clopper-Pearson bounds remain valid (conservative) for the pool's own rates.

``exact_size`` computes the exact rejection probability of the NI test over the trinomial
(N+, N-) distribution (no simulation), used by the tests to check size <= nominal.

numpy only; the bisection is vectorized over all counts at once.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def _logfact(n: int) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1)))])


def _log_binom_pmf(n: int, p: np.ndarray) -> np.ndarray:
    """(len(p), n+1) matrix of log P(X = k; n, p_i)."""
    k = np.arange(n + 1)
    lf = _logfact(n)
    logc = lf[n] - lf[k] - lf[n - k]
    p = np.clip(np.asarray(p, dtype=float)[:, None], 1e-300, 1 - 1e-16)
    return logc[None, :] + k[None, :] * np.log(p) + (n - k)[None, :] * np.log1p(-p)


def _tail_le(n: int, x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """P(X <= x_i; n, p_i), in log space."""
    lp = _log_binom_pmf(n, p)
    mask = np.arange(n + 1)[None, :] <= x[:, None]
    lp = np.where(mask, lp, -np.inf)
    m = lp.max(1, keepdims=True)
    return np.exp(m[:, 0]) * np.exp(lp - m).sum(1)


def _tail_ge(n: int, x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """P(X >= x_i; n, p_i), in log space (accurate for tiny tails)."""
    lp = _log_binom_pmf(n, p)
    mask = np.arange(n + 1)[None, :] >= x[:, None]
    lp = np.where(mask, lp, -np.inf)
    m = lp.max(1, keepdims=True)
    return np.exp(m[:, 0]) * np.exp(lp - m).sum(1)


def cp_upper_many(x: np.ndarray, n: int, a: float, iters: int = 64) -> np.ndarray:
    """Exact one-sided upper (1 - a) Clopper-Pearson bounds for counts x (vectorized)."""
    x = np.asarray(x, dtype=int)
    lo, hi = x / n, np.ones(len(x))
    for _ in range(iters):
        mid = (lo + hi) / 2
        too_low = _tail_le(n, x, mid) > a  # P(X <= x; mid) still > a: bound is higher
        lo, hi = np.where(too_low, mid, lo), np.where(too_low, hi, mid)
    return np.where(x >= n, 1.0, hi)


def cp_lower_many(x: np.ndarray, n: int, a: float, iters: int = 64) -> np.ndarray:
    """Exact one-sided lower (1 - a) Clopper-Pearson bounds for counts x (vectorized)."""
    x = np.asarray(x, dtype=int)
    lo, hi = np.zeros(len(x)), x / n
    for _ in range(iters):
        mid = (lo + hi) / 2
        too_high = _tail_ge(n, x, mid) > a  # P(X >= x; mid) still > a: bound is lower
        lo, hi = np.where(too_high, lo, mid), np.where(too_high, mid, hi)
    return np.where(x <= 0, 0.0, lo)


def cp_upper(x: int, n: int, a: float) -> float:
    return float(cp_upper_many(np.array([x]), n, a)[0])


def cp_lower(x: int, n: int, a: float) -> float:
    return float(cp_lower_many(np.array([x]), n, a)[0])


def diff_upper(n_plus: int, n_minus: int, n: int, a: float) -> float:
    """Upper (1 - a) bound on p+ - p- (finite-sample valid)."""
    return cp_upper(n_plus, n, a / 2) - cp_lower(n_minus, n, a / 2)


def diff_lower(n_plus: int, n_minus: int, n: int, a: float) -> float:
    """Lower (1 - a) bound on p+ - p- (finite-sample valid)."""
    return cp_lower(n_plus, n, a / 2) - cp_upper(n_minus, n, a / 2)


def increase_pvalue(n_plus: int, n_minus: int) -> float:
    """Exact one-sided p-value against "no increase" (p+ <= p-): conditional on the
    discordant total d = N+ + N-, N+ ~ Binomial(d, q) with q <= 1/2 under the null, so
    P(Binomial(d, 1/2) >= N+) is a valid (exact, conservative) p-value. d = 0 gives 1."""
    d = n_plus + n_minus
    if d == 0:
        return 1.0
    return float(min(1.0, _tail_ge(d, np.array([n_plus]), np.array([0.5]))[0]))


def wald_upper(n_plus: int, n_minus: int, n: int, a: float) -> float:
    """The V0 skeleton's normal-approximation bound, kept only to reproduce its failure."""
    from statistics import NormalDist

    mean = (n_plus - n_minus) / n
    var = (n_plus + n_minus) / n - mean ** 2
    var = var * n / max(1, n - 1)
    return mean + NormalDist().inv_cdf(1 - a) * float(np.sqrt(max(var, 0.0) / n))


@lru_cache(maxsize=64)
def _bounds_table(n: int, a: float, side: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(n + 1)
    if side == "upper":  # delta_upper = U(N+) - L(N-)
        return cp_upper_many(x, n, a / 2), cp_lower_many(x, n, a / 2)
    return cp_lower_many(x, n, a / 2), cp_upper_many(x, n, a / 2)


def _trinomial(n: int, p_plus: float, p_minus: float) -> np.ndarray:
    lf = _logfact(n)
    a = np.arange(n + 1)[:, None]
    b = np.arange(n + 1)[None, :]
    c = n - a - b
    p0 = 1.0 - p_plus - p_minus
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = (lf[n] - lf[a] - lf[b] - lf[np.clip(c, 0, n)]
              + np.where(a > 0, a * np.log(max(p_plus, 1e-300)), 0.0)
              + np.where(b > 0, b * np.log(max(p_minus, 1e-300)), 0.0)
              + np.where(c > 0, c * np.log(max(p0, 1e-300)), 0.0))
    lp = np.where(c >= 0, lp, -np.inf)
    if p_plus == 0:
        lp = np.where(a > 0, -np.inf, lp)
    if p_minus == 0:
        lp = np.where(b > 0, -np.inf, lp)
    return np.exp(lp)


def exact_size(n: int, p_plus: float, p_minus: float, a: float, margin: float,
               method: str = "cp") -> float:
    """Exact P(the harm-type NI test passes) when the truth is p+ - p- = ``margin`` (a null
    on its boundary): pass iff upper bound on p+ - p- < margin. ``method``: ``cp`` (this
    module) or ``wald`` (the V0 bound, for the reproduction)."""
    prob = _trinomial(n, p_plus, p_minus)
    if method == "cp":
        up, lo = _bounds_table(n, a, "upper")
        ub = up[:, None] - lo[None, :]
    else:
        x = np.arange(n + 1)
        ub = np.array([[wald_upper(i, j, n, a) if i + j <= n else np.inf for j in x]
                       for i in x])
    return float(prob[ub < margin].sum())
