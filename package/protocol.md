# Protocol xp-p1-2026-10-10

Frozen copy of docs/xp/plan-v1.md.

# Larger experiment: plan v1.2 (2026-10-10; owner decisions folded in)

Status: v1.2 is the version registered with the partner package (mode B,
docs/xp/partner-package.md v3.2; owner decisions docs/xp/owner-decisions-2026-10-10.md).
Sections after "Registered design (v1.2)" are the v1.1 background; where they differ,
v1.2 wins.

## Registered design (v1.2)

- **Execution:** mode B. The partner runs all GPU work; the teacher is self-hosted at the
  partner (FP8 on 8xH100, recorded as a deviation from POC 9's API endpoint). Training,
  validation and sentinel answers ship (the learning signal); test answers stay with us.
- **Writers and critics:** T+ (teacher, lesson library; critic teacher), S+ (the 9B,
  lesson library; critic self), T (teacher, POC 9 whole-note loop; critic teacher, for
  continuity with POC 9). S and TS are dropped.
- **Tasks and size:** address, variant, sequence; about 1,000 test cases per task (333
  groups of three phrasings); validation 60, sentinel 30; training pool 400 (address 150)
  with each run's own seeded 48-case sample; 12 learning runs per writer and task.
- **Budgets:** separate, equal learn and search-control budgets per run (about 196
  logical writer calls per run in all), registered in transport dispatches; final
  selection on the sentinel split. Catalogue item ids are shared across splits by design.
- **Primary contrast:** T+ endpoint skill (after sentinel selection) vs the T+
  no-evidence search control (same writer, loop, proposals, dev and sentinel selection;
  no evidence), accuracy on the sealed test, pooled over the three tasks.
- **Secondary contrast** (tested only if the primary succeeds): S+ (critic = self) endpoint
  vs its search control, same analysis.
- **Minimum worthwhile gain:** 5 points pooled (the planning minimum detectable gain at
  80% power, 12 runs, between-run SD 6; docs/xp/power.md). The confirmatory test is
  superiority: the lower bound of the two-sided 95% interval above 0; the report states
  whether the estimate reaches 5 points.
- **Wrong-write non-inferiority:** the upper 95% bound of (endpoint minus search control)
  wrong-write rate below 2 points, for the primary and, if tested, the secondary writer.
- **Analysis:** per case paired difference (endpoint minus control, same test case and run);
  averaged within scenario group; crossed with learning run; pooled over tasks with equal
  task weights. Interval: crossed random-effects model (run x group, task fixed) with a
  parametric bootstrap (10,000 draws, seed registered in the analysis script). Coverage
  must be complete (checker status ok) before any score is read. All other arms (B0, B1,
  generic note, full context, best-of-N, T vs its control, per-task contrasts) are
  descriptive. The measured overlap of training samples between runs is reported.
- **Person-written procedure arm:** left out pending an owner decision; adding it later
  appends one shared arm and changes no other arm or prompt hash (a new registration).
- **Gates:** phase 1 to phase 2 (conversations) only if the primary contrast succeeds and
  wrong-write non-inferiority holds; phase 2 then uses the T+ endpoint skills chosen by the
  registered rule, never by test scores. Any deviation is logged before unblinding.

## Question

Can the harness of a frozen 9B agent learn, repeatably, from its own traces, so that
it beats every simple alternative on held-out cases? Does the gain hold across
decision types and in full conversations? Who should write the lessons?

## Evidence so far (POC 9, docs/poc9/results-2026-10-09.md)

- Teacher-written skill: +10.0 points over B1 [+2.5, +18.3]; beats its same-length
  generic note by +6.7 [+0.8, +12.5]; wrong writes 8.3% to 0%.
- Self-written skill: +5.8 over B1 [-2.5, +14.2]; equal to its generic note (0.0).
- Teacher draft rewritten by the 9B: no rewrite kept the teacher's rules; no skill.
- Limits: one learning run per writer, one decision type, 120 sealed cases.

## Roles

- Partner: all GPU work on H100s. Returns reports only, no model weights; up to
  100 MB of gz data.
- Us: all API token work where possible (teacher via API, user simulator, judge,
  AI annotators); answer keys; scoring; analysis.
- Open: execution mode (A or B below) depends on the partner.

### Execution mode A (preferred): partner serves, we orchestrate

The partner hosts the 9B (and optionally the teacher) as a private, authenticated
endpoint. We run every experiment from our side. API keys, answer keys and all raw
data stay with us. The laptop only makes network calls.

### Execution mode B (fallback): partner runs everything

We send a frozen package (code commit, protocol, cases without answers). The
partner runs it and returns a compact export (below). Simulator and judge calls use
a separate key with a hard spend limit. We score every answer against the answer
key that we keep.

## Writers

| Writer | Who writes the lesson | Loop |
|---|---|---|
| T | teacher | POC 9 loop (whole-skill rewrite) |
| S | the 9B | POC 9 loop |
| S+ | the 9B | library of short entries; add/replace/remove edits; keep only if dev improves; rejected-edit memory; no tight length cap |
| T+ | teacher | the S+ loop |
| TS (optional) | teacher draft, 9B rewrite | POC 9 loop; failed fidelity in POC 9 |

## Tasks (decision level)

1. Address resolution (POC 9 task).
2. Variant selection from a long variant list.
3. Order of changes on one pending order (address before items; confirmation).

Each task: a training pool (400 cases for variant and sequence; 150 for address, the
largest the address pool supports next to a 1,000-case test split) from which every
learning run draws its own seeded sample of 48 (case ids registered per run; overlap
between runs measured and reported), validation 60 (20 groups; development scores during learning),
sentinel 30 (10 groups; a separate set used only to choose among candidate skills),
sealed test about 1,000 cases (groups of three phrasings, split by group and request
family; validation and sentinel share the selection families but no group or user;
every kind appears in both). Generators: `slh/xp/tasks/` (`python -m slh.xp.tasks.cli`).

### Address task (POC 9's decision at XP scale)

`slh/xp/tasks/address.py`: POC 9's kinds and gold rules, XP conventions (group = one
configuration in three phrasings; family = referent kind x wording style). Seed 7301
(POC 9: 9101). Every (user, target order) pair of POC 9's train and validation cases
is excluded from every split. POC 9's sealed test cases are not read; their (user,
target) pairs are known only as group-id hashes in the public `poc9/manifest.json`
and are avoided unless the pool runs out (default sizes: 3 of 333 test groups touch a
POC 9 test pair, all of them `ambiguous`; 36 of 500 at 1,500 test cases).
TODO: overlap with POC 9's sealed test set is possible but unlikely beyond that count;
check it on our side at configuration level (user, target, referent) in a logged
sealed-access window before the XP test set is registered. The address pool is
small for `ambiguous` (about 135 configurations, 41 users), so test groups share
users (about 96 users for 333 groups); cluster by user in a sensitivity analysis.

## Controls added after review

- Critic is a recorded parameter: teacher, self or none. In POC 9 the teacher
  critic gated every writer, so S was "self-written, teacher-checked". Self-learning
  claims use critic = self or none.
- Writer parity: equal call and token ceilings, evidence, proposal opportunities,
  repairs and selection feedback; actual tokens and cost reported.
- Search control: a no-evidence writer gets the same number of proposals and the
  same dev selection as the learned writers (not one generic note).
- Prompt length: a raw-excerpt/retrieval control at the skill's token budget.
- Best-of-N economics: acquisition, writer, screening and selector costs included;
  amortization volume registered.
- Realism: generators and templates frozen; family leakage audited; official results
  kept separate; model versions and serving settings pinned.

## Cuts (keep runs, drop checkpoints and variants)

- Large test: only the frozen endpoint skill; 12/24-episode curves on development
  data only.
- Drop TS, removal re-evaluations and repeated B0 evaluations.
- One editing loop for both T and S, chosen on development evidence (the 2x2
  writer x loop design only if the loop effect is a claim).
- More distinct scenarios, fewer paraphrases (a smaller paraphrase-robustness sample).
- Validation: 60 dev cases (was 12), a cap on proposals, and a separate 30-case
  sentinel set for selection.

## Arms (every task, every run)

Model only (B0); B1; learned skill at 0/12/24/48 episodes per writer; full
context (the writers' evidence text); budget-matched best-of-N; same-length generic
note; person-written procedure; code rule where one exists; removal (= B1).
Arms with byte-identical prompts are run once (dedupe by prompt hash).

## Repeats, unit of analysis and power

- An independent learning run means a new training sample/stream and a new
  learning seed, not a repeated evaluation of the same skill. Each run draws its own
  48-case sample from the task's training pool (`slh.xp.partner.plan.train_samples`);
  samples may overlap (expected overlap 48/pool: 12% at 400, 32% at 150), and the
  measured overlap is reported with the results, since overlap makes runs less
  independent than the power simulation assumes.
- Final selection on the sentinel split: a run keeps its endpoint skill only if it
  beats its predecessor there; otherwise it reverts to its best-on-sentinel snapshot
  (recorded per run).
- Unit of analysis: paired accuracy differences averaged within each scenario group,
  crossed with learning run; cluster by request family where groups share a
  template. Inference over both future cases and learning runs uses a crossed
  random-effects model with parametric-bootstrap intervals.
- About 1,000 cases (about 333 groups) per task makes a fixed artifact's gain of
  about 5-7 points detectable. Repeatability across runs is the weak point: five
  runs give few degrees of freedom ("4 of 5 positive" is descriptive only, sign test
  p = 0.19). Power is simulated before registration over plausible run and group
  variance; the number of runs is set from that simulation (likely 8-10, paid for
  by the cuts below).
- Power simulation (docs/xp/power.md, `slh/xp/power.py`): runs, not test size,
  limit power once the gain varies between learning runs. Recommendation: 12
  learning runs per writer and task, 1,000 test cases per task; minimum detectable
  pooled gain at 80% power 4.8 points if the between-run SD of the gain is 6
  points (3.0 at SD 3, 6.8 at SD 9); per-task contrasts are secondary (6.3 points
  at SD 6). 10 runs: 5.2 points; 8 runs: 5.9.

## Phases and gates

| Phase | Content | Gate to the next phase |
|---|---|---|
| 1 Decisions | all writers x 3 tasks x N runs, all arms | registered primary contrast (one writer, one loop, endpoint skill vs its generic note and best-of-N, pooled over tasks) has a lower 95% bound above a registered minimum gain, and wrong writes are not worse than B1 by more than a registered margin |
| 2 Conversations | the registered writer's endpoint skills (chosen by the phase-1 rule, not by test scores) in full retail conversations on fresh generated scenarios and, separately, the official test; blinded fidelity audit with a human-calibrated sample; all-scenarios analysis alongside the fidelity-filtered one | registered sample size, metric and margin |
| 3 Second cycle (deferred unless continual improvement is a headline claim) | new traces; update one skill selected on development data; evaluate once on untouched sealed cases | update helps |
| 4 System One | not in this hand-off (weights cannot come back) | later, separately |

## Compact export (mode B; under 100 MB gz)

| Content | Format | Estimate (gz) |
|---|---|---|
| Decision answers (all arms, runs, tasks) | case_id, arm, run, action, tokens, time | about 20 MB |
| Conversations | per conversation: actions, reward, final DB hash, fidelity flags; full transcripts for a sample (about 500 plus all failures on key arms) | about 15-30 MB |
| Learning traces | every proposal, ops, gate verdicts, dev scores, accepted hashes | about 5-10 MB |
| Learned skills | text plus hash | under 1 MB |
| Logs | costs, timings, errors; hashes of code, cases, protocol | about 1-2 MB |

An exporter writes this format with a manifest (file hashes, row counts) and refuses
to exceed 100 MB. A checker on our side verifies hashes, scores answers against our
answer key and rebuilds every table.

## Cost (estimates)

| Item | Paid by | Estimate |
|---|---|---|
| GPU (9B agent; teacher if self-hosted) | partner | about 200-400 H100-hours (phase 1 about 165-345 at 12 runs x 4 writers x 3 tasks; replaced by the pilot measurement) |
| User simulator and judge (conversations) | us | about $45-180 |
| Teacher via API | us | about $10 |
| AI annotators (fidelity audits) | us | about $10-20 |

## Integrity

- Code commit, cases, protocol and arm list hashed and registered before any run.
- Answer keys never leave us. In mode B, cases go out without answers.
- Scores are concealed until coverage is complete.
- Every deviation is logged, as in POC 9.

## Open decisions (owner)

1. Execution mode A or B (partner capability).
2. Include S+, T+ and TS.
3. Whether a paper names the models.
4. The primary claim to register (writer, loop, critic, minimum gain, wrong-write margin).
