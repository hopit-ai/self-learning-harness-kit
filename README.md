# Self-learning harness: partner kit (plain)

**Hand-off 1b (current): start with [docs/xp/handoff1b-runbook.md](docs/xp/handoff1b-runbook.md)**;
it refers to README-plain.md for the environment and the agent server.

Plan `xp-h1b-2026-10-19`: the code that runs (`slh/`, `configs/`, `scripts/xp/`,
`pyproject.toml`), the hand-off packages listed below and `MANIFEST.json` (SHA-256 of every
file). This release runs no phase 1 and ships no phase-1 package. No job scripts: you
start the servers yourself ([README-plain.md](README-plain.md): environment and servers)
and follow the runbook above.

## Release `xp-h1b-v1`

This commit is tag `xp-h1b-v1` of `hopit-ai/self-learning-harness-kit` (a new tag; earlier tags are never
moved). The hand-off packages are part of this tree (hashed in `MANIFEST.json`, their code
registration checked against this tree when the release was built); this release runs only part A (decision screen and proposal collection):

| Directory | What | plan.json sha256 |
|---|---|---|
| `package-h1/` | part A: decision screen and proposal collection (F' / C) | `d5251d7234f556e9...` |

The decision prefixes were built on the retail database `ext-v2` (`db-ext-v2.json`, sha256 `3d03098276f5b2dd2ec08988f11e4f3d239e7ac0d052991f63807f35760130e5`); the database is attached to the release for reference only (nothing in this kit reads it at run time).

Offline assets attached to the release (put them next to this directory; checksums also in
`MANIFEST.json` under `release.assets`):

| Asset | sha256 | bytes |
|---|---|---:|
| `db-ext-v2.json` | `3d03098276f5b2dd2ec08988f11e4f3d239e7ac0d052991f63807f35760130e5` | 28698817 |
| `db-ext-v2.sha256` | `b2992e6ab8518bb55782b6811d0dec7b7be9f739511e5766faecc890f7adf23d` | 81 |
| `offline-assets.json` | `7cea783c97db18bc98f4693d7c125decbf1d1da84f0233c16bf0e6901e168514` | 500 |
| `tau2-bench-672227c6b6676edc20d57ea53b7000262aae77b9.tar.gz` | `a65736d75b1f5cc712c3e068704c64ca7fd4592d0e7cc94248ef45f1c05afd76` | 165984479 |
| `wheelhouse-linux-x86_64-py312.tar.gz` | `d67df2fa5518996ff5a5f74db92f27bfa1b800b73e744421882ced5b5405021e` | 67497161 |

Licence: MIT (LICENSE). The benchmark (tau2-bench, MIT) is used at a pinned commit, not
included; see NOTICE.
