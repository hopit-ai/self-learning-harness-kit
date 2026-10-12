# Self-learning harness: partner kit

Public packages for running our self-learning harness experiments on a partner's GPUs.
Each package is one tag; check out the tag, never `main`, so we know exactly what ran.

| Tag | What |
|---|---|
| `xp-p1-v1` | plan `xp-p1-2026-10-10`: one command for phase 0 (v2), the gate and phase 1 |
| `xp-p1-phase0-v1` | plan `xp-p1-2026-10-10`: phase 0 v1, the first site check (done) |

Start with [README-phase1.md](README-phase1.md).

What is here: the code that runs (`slh/`, `configs/`, `scripts/xp/`, `pyproject.toml`),
the frozen package (`package/`: plan, registration, protocol, cases), the Slurm job scripts
(`slurm/`) and `MANIFEST.json` (SHA-256 of every file). The test cases ship as questions
only: their answers stay with us and the results are scored on our side. The runner
refuses to start if the code, the package or the benchmark copy differ from
`package/registration.json`.

Licence: MIT (LICENSE). The benchmark (tau2-bench, MIT) is fetched at a pinned commit,
not included; see NOTICE.
