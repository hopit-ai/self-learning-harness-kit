# Self-learning harness: partner kit (plain)

**Hand-off 2 (function-calling benchmark) (current): start with [docs/xp/handoff2-bfcl-runbook.md](docs/xp/handoff2-bfcl-runbook.md)**;
it refers to README-plain.md for the environment and the agent server.

Plan `xp-h2b-2026-10-19`: the code that runs (`slh/`, `configs/`, `scripts/xp/`,
`pyproject.toml`), the hand-off packages listed below and `MANIFEST.json` (SHA-256 of every
file). This release runs no phase 1 and ships no phase-1 package. No job scripts: you
start the servers yourself ([README-plain.md](README-plain.md): environment and servers)
and follow the runbook above.

## Release `xp-h2b-v1`

This commit is tag `xp-h2b-v1` of `hopit-ai/self-learning-harness-kit` (a new tag; earlier tags are never
moved). The hand-off packages are part of this tree (hashed in `MANIFEST.json`, their code
registration checked against this tree when the release was built); this release runs only part D (function-calling benchmark: randomized lesson inclusion on the gate split):

| Directory | What | plan.json sha256 |
|---|---|---|
| `package-bfcl-h2/` | part D: function-calling benchmark, randomized lesson inclusion on the gate split (empty-library anchors) | `c26b6fd8e589d735...` |

Offline assets attached to the release (put them next to this directory; checksums also in
`MANIFEST.json` under `release.assets`):

| Asset | sha256 | bytes |
|---|---|---:|
| `offline-assets.json` | `7cea783c97db18bc98f4693d7c125decbf1d1da84f0233c16bf0e6901e168514` | 500 |
| `tau2-bench-672227c6b6676edc20d57ea53b7000262aae77b9.tar.gz` | `a65736d75b1f5cc712c3e068704c64ca7fd4592d0e7cc94248ef45f1c05afd76` | 165984479 |
| `wheelhouse-bfcl-linux-x86_64-py312.tar.gz` | `5cd8aed9d3c2a9e7d543bbc18f8b6051c7e6edffb8b2a92f3ba5950be8a8f34c` | 106569207 |
| `wheelhouse-bfcl.sha256` | `c78e63ccaef9e63566376dfbe8cc3cf1357c77e5e5f4ebd891bcc0e308aaa32d` | 108 |
| `wheelhouse-linux-x86_64-py312.tar.gz` | `d67df2fa5518996ff5a5f74db92f27bfa1b800b73e744421882ced5b5405021e` | 67497161 |

Licence: MIT (LICENSE). The benchmark (tau2-bench, MIT) is used at a pinned commit, not
included; see NOTICE.
