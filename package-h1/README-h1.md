# Hand-off 1: decision units (xp-h1b-2026-10-19-h1)

23010 decision units (20400 lesson-screen units, 1530
collection units, 1080 difficulty-check units) on 3810
decision prefixes. One model only: the frozen 9B
agent served with vLLM exactly as in the plain phase-1 runbook (README-plain.md,
section 2: the agent server; no teacher server is needed). No network beyond localhost.

```bash
P="--package package-h1 --endpoint endpoint.yaml"
python -m slh.xp.fgate.units run --package package-h1 --out runs/h1-dry --dry-run   # no GPU; "export ok"
python -m slh.xp.fgate.units run $P --out runs/h1-smoke --smoke                      # small slice
python -m slh.xp.fgate.units run $P --out runs/h1                                    # full
python -m slh.xp.fgate.units export --run runs/h1 --out export-h1
```

Rerun the same command after an interruption (finished units are kept). Exit 3 = halted
(cause in `halt.json`), exit 4 = refused (package, code or settings differ). Send back
`export-h1/` (and the smoke export) unchanged. Estimated GPU time: 2.1-6.4
H100-hours (1-3 decisions per second per H100 (phase-1 assumption 3/s with prefix caching; lesson-varied system prompts cache less)).
