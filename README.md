# Self-learning harness: partner kit (plain)

**Hand-off: Hermes Agent on the retail benchmark (current): start with [docs/xp/handoff-mh1-runbook.md](docs/xp/handoff-mh1-runbook.md)**;
it refers to README-plain.md for the environment and the agent server.

Plan `mh1-2026-10-25-v2.6`: the code that runs (`slh/`, `configs/`, `scripts/xp/`,
`pyproject.toml`), the hand-off packages listed below and `MANIFEST.json` (SHA-256 of every
file). This release runs no phase 1 and ships no phase-1 package. No job scripts: you
start the servers yourself ([README-plain.md](README-plain.md): environment and servers)
and follow the runbook above.

## Release `mh1-2026-10-25-v2.6`

This commit is tag `mh1-2026-10-25-v2.6` of `hopit-ai/self-learning-harness-kit` (a new tag; earlier tags are never
moved). The hand-off packages are part of this tree (hashed in `MANIFEST.json`, their code
registration checked against this tree when the release was built); this release runs only part H (Hermes Agent on the retail benchmark: three learning setups, two customer models (code only; task data handed over privately)):

| Directory | What | plan.json sha256 |
|---|---|---|


Offline assets attached to the release (put them next to this directory; checksums also in
`MANIFEST.json` under `release.assets`):

| Asset | sha256 | bytes |
|---|---|---:|
| `cpython-3.14.6+20260804-x86_64-unknown-linux-gnu-install_only.tar.gz` | `8fa4614f8c2d47430d5573098fad787c44bb314830fd2e826952335446706e45` | 124756578 |
| `hermes-assets.sha256` | `6f79b344895c2708b4fb69d0dbaf328a0aa60e7fc1dd4d369149889216e72af5` | 245 |
| `offline-assets.json` | `7cea783c97db18bc98f4693d7c125decbf1d1da84f0233c16bf0e6901e168514` | 500 |
| `tau2-bench-672227c6b6676edc20d57ea53b7000262aae77b9.tar.gz` | `a65736d75b1f5cc712c3e068704c64ca7fd4592d0e7cc94248ef45f1c05afd76` | 165984479 |
| `wheelhouse-hermes-linux-x86_64-py314.tar.gz` | `ce5e424e74d77aa1fd112cce2ae2bc958cc774ff2a9fcb472d0e3c2b13b0fc8b` | 45831063 |
| `wheelhouse-linux-x86_64-py312.tar.gz` | `d67df2fa5518996ff5a5f74db92f27bfa1b800b73e744421882ced5b5405021e` | 67497161 |

Licence: MIT (LICENSE). The benchmark (tau2-bench, MIT) is used at a pinned commit, not
included; see NOTICE.
