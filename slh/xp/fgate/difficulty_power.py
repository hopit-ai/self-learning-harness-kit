"""Power and coverage of the synthetic-difficulty check (plan section 3.6).

    python -m slh.xp.fgate.difficulty_power [--reps 300] [--boot 600] [--pairs 180] \
        [--large 0.10] [--out SIM.json]

Simulates the pre-specified estimator of ``fgate.analysis.difficulty_check`` (per pair the
synthetic minus original mean over 3 phrasings; percentile bootstrap over pairs) and its
verdicts, on synthetic outcomes with:

* ``p``: original-arm success (0.7, 0.8, 0.9); the synthetic arm is ``p + delta``;
* ``rho``: correlation of the binary outcomes of one group's 3 phrasings (0, 0.5, 1): with
  probability ``rho`` a group's phrasings share one draw, else they are independent;
* ``het``: SD of a per-slot success shift shared by both arms (0 = independent arms, the
  conservative case; matched slots can only add positive cross-arm correlation).

Reported per cell: P(shift) at delta 0 (false positive), power at the pre-registered large
shift separately per direction (``power_minus_large`` at -large, ``power_plus_large`` at
+large; they differ because the binomial variance changes with the synthetic arm's rate,
so power is higher where the synthetic rate moves towards 0 or 1; ``power_plus_large`` is
None when p + large > 1), P(no large shift) at delta 0, P(90% CI inside +-5) at delta 0
(the secondary, reported-only equivalence), and the coverage of the 90% / 95% intervals.
Plus the analytic (normal) number of pairs needed for 80% power of the +-5 equivalence.
Light: numpy, single thread, seconds on a laptop at the defaults (2,000 replicates with
1,000 bootstrap resamples: under a minute).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from slh.xp.fgate import analysis as A
from slh.xp.fgate import design as D

PHRASINGS = 3
RATES = (0.7, 0.8, 0.9)
RHOS = (0.0, 0.5, 1.0)


def _arm(p: np.ndarray, rho: float, reps: int, rng: np.random.Generator) -> np.ndarray:
    """reps x pairs mean success over the 3 phrasings of one arm."""
    pairs = p.shape[-1]
    shared = rng.random((reps, pairs)) < p
    indep = rng.random((reps, pairs, PHRASINGS)) < p[..., None]
    use_shared = rng.random((reps, pairs)) < rho
    y = np.where(use_shared[..., None], shared[..., None], indep)
    return y.mean(axis=2)


def simulate_cell(*, p: float, delta: float, rho: float, het: float = 0.0, pairs: int = 180,
                  reps: int = 300, n_boot: int = 600, seed: int = 0,
                  large: float | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    slot = rng.normal(0.0, het, size=(reps, pairs)) if het else np.zeros((reps, pairs))
    po = np.clip(p + slot, 0.0, 1.0)
    ps = np.clip(p + delta + slot, 0.0, 1.0)
    true = float((ps - po).mean())
    d = _arm(ps, rho, reps, rng) - _arm(po, rho, reps, rng)
    out = {"shift": 0, "no_large_shift": 0, "equivalent_5": 0, "cover90": 0, "cover95": 0}
    for r in range(reps):
        s = A.pair_interval(d[r], rng, n_boot=n_boot)
        v = A.difficulty_verdict(s, large=D.DIFFICULTY_LARGE_SHIFT if large is None else large,
                                 equivalence=D.DIFFICULTY_EQUIVALENCE)
        out["shift"] += v["verdict"] == "shift"
        out["no_large_shift"] += v["verdict"] == "no large shift"
        out["equivalent_5"] += bool(v["equivalent_reported"])
        out["cover90"] += s["ci90"][0] <= true <= s["ci90"][1]
        out["cover95"] += s["ci95"][0] <= true <= s["ci95"][1]
    return {k: round(v / reps, 3) for k, v in out.items()}


def pairs_for_equivalence(p: float, rho: float, *, margin: float = 0.05, power: float = 0.8,
                          het: float = 0.0) -> int:
    """Normal approximation: pairs so that P(90% CI inside +-margin | delta 0) = power,
    i.e. 2 Phi(margin / se - 1.645) - 1 = power."""
    z = 1.6449 + _norm_ppf((1 + power) / 2)
    var_arm = p * (1 - p) * (1 + (PHRASINGS - 1) * rho) / PHRASINGS
    var_pair = 2 * max(var_arm - het ** 2, 1e-12)  # a shared slot shift cancels in the pair
    return math.ceil(var_pair * (z / margin) ** 2)


def _norm_ppf(q: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run(*, reps: int, n_boot: int, pairs: int, seed: int = 0,
        large: float | None = None) -> dict[str, Any]:
    large = D.DIFFICULTY_LARGE_SHIFT if large is None else large
    cells = []
    grid = [(p, rho, 0.0) for p in RATES for rho in RHOS] + \
        [(0.8, rho, 0.1) for rho in (0.5, 1.0)]
    for i, (p, rho, het) in enumerate(grid):
        kw = {"p": p, "rho": rho, "het": het, "pairs": pairs, "reps": reps, "n_boot": n_boot,
              "large": large}
        null = simulate_cell(delta=0.0, seed=seed + 10 * i, **kw)
        down = simulate_cell(delta=-large, seed=seed + 10 * i + 1, **kw)
        up = simulate_cell(delta=+large, seed=seed + 10 * i + 2, **kw) \
            if p + large <= 1 + 1e-9 else None
        cells.append({"p": p, "rho": rho, "het": het,
                      "false_shift": null["shift"],
                      "power_minus_large": down["shift"],
                      "power_plus_large": up["shift"] if up else None,
                      "power_min_direction": min(down["shift"], up["shift"]) if up
                      else down["shift"],
                      "no_large_shift_at_0": null["no_large_shift"],
                      "equivalent_5_at_0": null["equivalent_5"],
                      "cover90_at_0": null["cover90"], "cover95_at_0": null["cover95"],
                      "cover95_at_large": down["cover95"],
                      "pairs_for_80pct_equivalence_5": pairs_for_equivalence(p, rho, het=het)})
    return {"pairs": pairs, "phrasings": PHRASINGS, "reps": reps, "boot": n_boot,
            "large_shift": large, "equivalence_reported": D.DIFFICULTY_EQUIVALENCE,
            "cells": cells}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m slh.xp.fgate.difficulty_power")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--boot", type=int, default=600)
    ap.add_argument("--pairs", type=int, default=sum(D.DIFFICULTY_SIZES.values()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--large", type=float, help="large-shift threshold (default: registered)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    res = run(reps=args.reps, n_boot=args.boot, pairs=args.pairs, seed=args.seed, large=args.large)
    text = json.dumps(res, indent=1)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
