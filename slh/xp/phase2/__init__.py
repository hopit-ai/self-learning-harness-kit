"""Phase 2 (full retail conversations): plan schema, seeds and sizing (design skeleton).

Design: docs/xp/phase2-design.md. This module holds only what de-risks the design:

* the scenario schema, split into the PUBLIC part that ships to the partner
  (user instructions for the simulator, no answer) and the KEY that stays with us
  (expected writes, fidelity checklist);
* the arm and plan schema, with ``Phase2Plan.problems()`` checking the rules the
  design depends on (endpoint skills chosen by the phase-1 sentinel rule, never by
  test scores; the confirmatory arms paired run by run; budget admission at 0.8 x
  the caps);
* secret-seed derivation (same construction as phase 1, a new label) and the
  public per-(scenario, run) simulator seed that pairs arms;
* conversation counts, the $ estimate and the power design used in the doc.

The runnable path lives in the submodules: ``package`` (what ships, keys kept
apart), ``bundles`` (arm texts from the phase-1 export, sentinel rule),
``run`` (the conversation driver, the same code at the partner and on our
side), ``export``, ``check`` (coverage by identity, replay scoring, safety
metric), ``pilot`` (our-side smoke pilot under an explicit cap) and
``generator`` (the scenario generator: ``python -m slh.xp.phase2 generate``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

DECISION_TYPES = ("address", "variant", "sequence")
SOURCES = ("generated", "official_test")
SEED_LABEL = "xp-phase2-seed-v1"
SIM_SEED_LABEL = "xp-phase2-sim-seed-v1"
SELECTION_RULE = "phase1-sentinel"  # how endpoint skills were chosen (plan v1.2)
ARM_KINDS = ("b0", "b1", "learned", "search_control", "generic", "full_context", "human")
# confirmatory: in the primary contrast (H1) and the safety test; secondary: the
# comparator of the secondary contrast only (H2, always reported, not a gate:
# docs/xp/registration-phase2-draft.md); descriptive: diagnostic arms.
ROLES = ("confirmatory", "secondary", "descriptive")
# Phase 2's own package directory (phase 1's kit already ships ``package/``).
PACKAGE_DIR = "package-p2"
ADMISSION = 0.8  # estimate must stay within 0.8 x every cap (phase-1 gate rule)
# Measured on the B1 stream run (126 conversations): simulator $0.4530 and judge
# $0.6573 in all, i.e. per conversation; the judge only runs on NL-assertion tasks.
SIM_USD_PER_CONV = 0.0036
JUDGE_USD_PER_CONV = 0.0052
# Two same-seed B1 dev runs agreed on 57 of 72 conversations.
SAME_SEED_AGREEMENT = (57, 72)


# --------------------------------------------------------------------------- seeds


def derive_seed(secret: bytes, plan_id: str, decision_type: str) -> int:
    """128-bit generator seed per decision type; the secret never leaves us."""
    if len(secret) < 32:
        raise ValueError("the secret must be at least 32 bytes")
    if decision_type not in DECISION_TYPES:
        raise ValueError(f"unknown decision type {decision_type!r}")
    mac = hmac.new(secret, f"{SEED_LABEL}|{plan_id}|{decision_type}".encode(),
                   hashlib.sha256).hexdigest()
    return int(mac[:32], 16)


def simulator_seed(plan_id: str, scenario_id: str, run: int) -> int:
    """Public seed of the user simulator for one (scenario, run).

    Every arm evaluated on this scenario in this run uses the same seed, so the
    contrast endpoint(run r) - B1(seed r) is paired; different runs use different
    seeds, so simulator noise is run-specific (the crossed model's residual)."""
    digest = hashlib.sha256(f"{SIM_SEED_LABEL}|{plan_id}|{scenario_id}|{run}".encode())
    return int(digest.hexdigest()[:8], 16)


# --------------------------------------------------------------------------- scenarios


@dataclass(frozen=True)
class PublicScenario:
    """What ships: the simulator's instructions, nothing that names the answer."""

    scenario_id: str
    source: str  # "generated" | "official_test"
    decision_type: str  # one of DECISION_TYPES; official tasks: the dominant one or "mixed"
    subvariation: str
    user_scenario: dict[str, str]  # reason_for_call, known_info, unknown_info, task_instructions
    generator_version: str
    scenario_hash: str = ""

    def body(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("scenario_hash")
        return d

    def computed_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.body(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ScenarioKey:
    """Stays with us: what a correct agent writes, and the fidelity checklist."""

    scenario_id: str
    scenario_hash: str
    expected_writes: tuple[dict[str, Any], ...]
    expected_db_sha256: str
    must_voice_in_opening: tuple[str, ...]
    forbidden: tuple[str, ...] = ()
    predicate: str | None = None  # e.g. "same-order-write-precedence-v1"
    user_id: str = ""  # for the user-clustered sensitivity analysis
    configuration: str = ""  # (user, target order, source) hash; at most 2 scenarios each


# Answer-key field names. A public scenario must not carry one as a field name,
# nor name one in its text (``expected_writes: ...``, ``gold_address``); the
# plain word "gold" (a colour option) is ordinary scenario text. The value check
# (every expected item id and address) is ``package.leak_problems``.
KEY_FIELDS = ("expected_writes", "expected_db_sha256", "expected_outcome",
              "evaluation_criteria", "predicate_binding", "answer_key")
_KEY_TEXT = re.compile(r"\b(?:" + "|".join(KEY_FIELDS) + r")\b|\bgold_[A-Za-z0-9_]+"
                       r"|[\"']gold\\?[\"']\s*:", re.IGNORECASE)


def _field_names(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(k) for k in value] + [n for v in value.values() for n in _field_names(v)]
    if isinstance(value, list):
        return [n for v in value for n in _field_names(v)]
    return []


def key_leaks(value: Any) -> list[str]:
    """Answer-key field names or key labels in a public value (names, not values)."""
    found = [n for n in _field_names(value)
             if n.lower() in KEY_FIELDS or n.lower() == "gold" or n.lower().startswith("gold_")]
    found += [m.group(0) for m in _KEY_TEXT.finditer(json.dumps(value))]
    return sorted(set(found))


# What the simulator is told: tau2's structured instructions plus the persona
# (``package.to_tau2_task`` forwards exactly these; nothing else is sent).
INSTRUCTION_FIELDS = ("reason_for_call", "known_info", "unknown_info", "task_instructions")
SIMULATOR_FIELDS = (*INSTRUCTION_FIELDS, "persona")


def simulator_view(user_scenario: dict[str, Any]) -> dict[str, Any]:
    """The canonical simulator-facing projection of a user scenario: the
    instruction fields and the persona (None when absent), nothing else."""
    return {"persona": user_scenario.get("persona"),
            **{k: user_scenario.get(k) for k in INSTRUCTION_FIELDS}}


def user_scenario_problems(user_scenario: Any) -> list[str]:
    """Only the simulator-facing fields are allowed (any other field would be
    dropped before the simulator sees it, so it could make two identical
    conversations look different); every instruction field must be present."""
    if not isinstance(user_scenario, dict):
        return ["user_scenario is not a mapping"]
    out = []
    extra = sorted(set(user_scenario) - set(SIMULATOR_FIELDS))
    if extra:
        out.append(f"unsupported user_scenario fields {extra} (the simulator is sent only "
                   f"{list(SIMULATOR_FIELDS)})")
    missing = [k for k in INSTRUCTION_FIELDS if k not in user_scenario]
    if missing:
        out.append(f"user_scenario lacks {missing}")
    return out


def content_fingerprint(user_scenario: dict[str, Any]) -> str:
    """ID-independent fingerprint of exactly what the simulator is told
    (``simulator_view``: instructions and persona, whitespace and case
    normalised): a renamed copy of a scenario has the same fingerprint, and a
    field the simulator never sees cannot change it."""
    norm = {k: None if v is None else " ".join(str(v).split()).lower()
            for k, v in sorted(simulator_view(user_scenario).items())}
    return hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest()


def public_problems(s: PublicScenario) -> list[str]:
    out = []
    if s.source not in SOURCES:
        out.append(f"{s.scenario_id}: unknown source {s.source!r}")
    if s.source == "generated" and s.decision_type not in DECISION_TYPES:
        out.append(f"{s.scenario_id}: unknown decision type {s.decision_type!r}")
    out += [f"{s.scenario_id}: {p}" for p in user_scenario_problems(s.user_scenario)]
    for k in key_leaks(s.body()):
        out.append(f"{s.scenario_id}: key field {k!r} in a public scenario")
    if s.scenario_hash and s.scenario_hash != s.computed_hash():
        out.append(f"{s.scenario_id}: scenario_hash does not match its body")
    return out


def generate_scenarios(secret: bytes, plan_id: str, *, sizes: dict[str, int],
                       exclusions: dict[str, Any] | None = None,
                       db: dict[str, Any] | None = None, split: str = "test"
                       ) -> tuple[list[PublicScenario], list[dict[str, Any]]]:
    """Fresh generated scenarios per decision type (docs/xp/phase2-design.md, section 2).

    Implemented in ``slh.xp.phase2.generator`` on top of ``slh.genpilot.generate``:
    roots x seeded sampler over the pinned public db, seed ``derive_seed(secret,
    plan_id, type)``; no model call; users of official tasks and every (user, target
    order) configuration of phase-1 training, validation or sentinel cases
    (``exclusions``: ``generator.phase1_configurations``) excluded; at most 2
    scenarios per (user, target order); each scenario independently validated.
    ``sizes``: test scenarios per type. Returns the ``split``'s public scenarios
    and our key rows, in the same order. ``db``, when given, must be the pinned
    database."""
    from slh.genpilot import data as gdata
    from slh.xp.phase2 import generator as GEN
    from slh.xp.phase2 import package as PK

    if db is not None and db != gdata.load_db():
        raise ValueError("the generator runs on the pinned retail db.json only")
    result = GEN.generate(secret, plan_id, test_sizes=sizes, phase1=exclusions,
                          through="test" if split == "test" else "smoke")
    pairs = [PK.from_genpilot(r) for r in sorted(result["splits"][split],
                                                 key=lambda r: r["task_id"])]
    return [p for p, _ in pairs], [k for _, k in pairs]


# --------------------------------------------------------------------------- arms and plan


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str  # ARM_KINDS
    role: str  # ROLES
    writer: str | None = None  # "T+", "S+" for writer arms
    runs: tuple[int, ...] = ()  # phase-1 learning runs whose skill bundles are used
    trials_per_run: int = 1  # conversations per (scenario, run)
    selection: str | None = None  # writer arms: how the phase-1 snapshot was chosen
    subset_per_type: int | None = None  # diagnostic arms: first n scenarios per type (seeded)
    official: bool = True  # also runs on the official-test shard

    def conversations_per_scenario(self) -> int:
        return len(self.runs) * self.trials_per_run

    def conversations(self, scenarios_per_type: dict[str, int], official_tasks: int) -> int:
        n = sum(v if self.subset_per_type is None else min(v, self.subset_per_type)
                for v in scenarios_per_type.values())
        off = official_tasks if self.official else 0
        return self.conversations_per_scenario() * (n + off)


@dataclass(frozen=True)
class Budget:
    key_cap_usd: float  # hard limit set on the partner's simulator key
    meter_cap_usd: float  # our meter halts first (below the key cap)
    gpu_hours_cap: float
    export_cap_bytes: int = 100_000_000  # per shard


@dataclass(frozen=True)
class Phase2Plan:
    plan_id: str
    phase1_plan_sha256: str
    phase1_gate: str  # "GO" required (phase-1 registered gate result)
    arms: tuple[Arm, ...]
    scenarios_per_type: dict[str, int]
    official_test_tasks: int
    budget: Budget
    primary: tuple[str, str] = ("tplus_endpoint", "b1")
    secondary: tuple[str, str] | None = ("tplus_endpoint", "tplus_search")
    min_worthwhile_gain: float = 5.0
    wrong_write_margin: float = 2.5  # phase-2 safety margin (registration draft; power.Setting)
    max_steps: int = 200  # tau2 orchestrator default
    conversation_time_limit_s: int = 2400
    notes: dict[str, Any] = field(default_factory=dict)

    def arm(self, name: str) -> Arm:
        for a in self.arms:
            if a.name == name:
                return a
        raise KeyError(name)

    def generated_scenarios(self) -> int:
        return sum(self.scenarios_per_type.values())

    def conversations(self) -> dict[str, int]:
        full = [a for a in self.arms if a.subset_per_type is None]
        per = sum(a.conversations_per_scenario() for a in full)
        gen = sum(a.conversations(self.scenarios_per_type, 0) for a in self.arms)
        runs_cap = self.notes.get("official_runs")
        off = sum((len(a.runs) if runs_cap is None else min(len(a.runs), runs_cap))
                  * a.trials_per_run * self.official_test_tasks
                  for a in self.arms if a.official)
        return {"per_scenario": per, "generated": gen, "official_test": off,
                "total": gen + off}

    def problems(self) -> list[str]:
        out: list[str] = []
        if self.phase1_gate != "GO":
            out.append("phase 2 runs only after a phase-1 GO")
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            out.append("duplicate arm names")
        for a in self.arms:
            if a.kind not in ARM_KINDS:
                out.append(f"{a.name}: unknown kind {a.kind!r}")
            if a.role not in ROLES:
                out.append(f"{a.name}: unknown role {a.role!r}")
            if not a.runs or a.trials_per_run < 1:
                out.append(f"{a.name}: no conversations")
            if a.kind in ("learned", "search_control", "generic", "full_context"):
                if a.writer is None:
                    out.append(f"{a.name}: writer arm without a writer")
                if a.selection != SELECTION_RULE:
                    out.append(f"{a.name}: skills must be chosen by {SELECTION_RULE!r}, "
                               f"never by test scores")
        contrasts = [("primary", self.primary)]
        if self.secondary is not None:
            contrasts.append(("secondary", self.secondary))
        for label, (x, y) in contrasts:
            try:
                ax, ay = self.arm(x), self.arm(y)
            except KeyError as exc:
                out.append(f"{label}: missing arm {exc.args[0]}")
                continue
            # Primary: both arms confirmatory. Secondary: the first arm is a primary
            # (confirmatory) arm; its comparator may be confirmatory or 'secondary'.
            allowed = {ax.name: ("confirmatory",),
                       ay.name: ("confirmatory",) if label == "primary"
                       else ("confirmatory", "secondary")}
            for a in (ax, ay):
                if a.role not in allowed[a.name]:
                    out.append(f"{label}: {a.name} must be "
                               f"{' or '.join(allowed[a.name])}")
            if (ax.runs, ax.trials_per_run) != (ay.runs, ay.trials_per_run):
                out.append(f"{label}: {x} and {y} are not paired run by run")
        secondary_arms = set(self.secondary or ())
        for a in self.arms:
            if a.role == "confirmatory" and a.name not in {*self.primary, *secondary_arms}:
                out.append(f"{a.name}: confirmatory arm outside the registered contrasts")
            if a.role == "secondary" and a.name not in secondary_arms - set(self.primary):
                out.append(f"{a.name}: a secondary arm must be the secondary contrast's "
                           f"comparator")
            if a.role in ("confirmatory", "secondary") and a.subset_per_type is not None:
                out.append(f"{a.name}: a {a.role} arm runs on every scenario")
        if not 0 < self.wrong_write_margin:
            out.append("the wrong-write margin must be positive")
        for t in DECISION_TYPES:
            if self.scenarios_per_type.get(t, 0) < 1:
                out.append(f"no scenarios for {t}")
        b = self.budget
        if not 0 < b.meter_cap_usd <= b.key_cap_usd:
            out.append("the meter cap must be positive and at most the key cap")
        est = cost_estimate(self)["simulator_usd_with_margin"]
        if est > ADMISSION * b.meter_cap_usd:
            out.append(f"planning estimate ${est:.2f} (with margin) exceeds {ADMISSION} x "
                       f"meter cap ${b.meter_cap_usd:.2f}: the plan could never be admitted")
        return out

    def admit(self, est_usd: float, est_gpu_hours: float) -> list[str]:
        """Gate admission: estimates within ADMISSION x each cap."""
        out = []
        if est_usd > ADMISSION * self.budget.meter_cap_usd:
            out.append(f"simulator estimate ${est_usd:.2f} > {ADMISSION} x meter cap")
        if est_gpu_hours > ADMISSION * self.budget.gpu_hours_cap:
            out.append(f"GPU estimate {est_gpu_hours:.0f} h > {ADMISSION} x cap")
        return out


def default_plan(plan_id: str = "xp-p2-draft", *, phase1_plan_sha256: str = "0" * 64,
                 runs: int = 12, scenarios: tuple[int, int, int] = (100, 150, 150),
                 official_test_tasks: int = 40, official_runs: int | None = None,
                 human_subset: int = 20, human_runs: int = 3,
                 meter_cap_usd: float = 115.0, key_cap_usd: float = 130.0,
                 gpu_hours_cap: float = 160.0) -> Phase2Plan:
    """The v2 design (doc section 1; registration draft): B1, T+ endpoint and T+ search
    control on every scenario and all ``runs`` (paired run by run). Primary (H1) = T+
    endpoint vs B1 (both confirmatory); secondary (H2, not a gate) = T+ endpoint vs
    the search control (role 'secondary'); safety margin 2.5 points. The
    person-written procedure on a
    diagnostic subset of ``human_subset`` scenarios per type with ``human_runs`` seeds.
    ``scenarios`` = (address, variant, sequence); address may be smaller.
    ``official_runs``: runs of the three arms on the official shard (default: all)."""
    all_runs = tuple(range(runs))
    sel = SELECTION_RULE
    arms = [
        Arm("tplus_endpoint", "learned", "confirmatory", "T+", all_runs, 1, sel),
        Arm("tplus_search", "search_control", "secondary", "T+", all_runs, 1, sel),
        Arm("b1", "b1", "confirmatory", None, all_runs, 1),
        Arm("human", "human", "descriptive", None, tuple(range(human_runs)), 1,
            subset_per_type=human_subset, official=False),
    ]
    plan = Phase2Plan(plan_id=plan_id, phase1_plan_sha256=phase1_plan_sha256, phase1_gate="GO",
                      arms=tuple(arms),
                      scenarios_per_type=dict(zip(DECISION_TYPES, scenarios, strict=True)),
                      official_test_tasks=official_test_tasks,
                      budget=Budget(key_cap_usd=key_cap_usd, meter_cap_usd=meter_cap_usd,
                                    gpu_hours_cap=gpu_hours_cap))
    if official_runs is not None:
        plan.notes["official_runs"] = official_runs
    return plan


def minimum_plan(**kw: Any) -> Phase2Plan:
    """The 'minimum credible' variant: 12 runs x 75 scenarios per type, official
    shard on 3 runs only."""
    kw = {"scenarios": (75, 75, 75), "official_runs": 3, "meter_cap_usd": 65.0,
          "key_cap_usd": 75.0, "gpu_hours_cap": 90.0, **kw}
    return default_plan(**kw)


# --------------------------------------------------------------------------- cost and power


# Our-side serving (docs/poc9/pricing-correction.md): full sandbox rates per hour.
SANDBOX_USD_PER_HOUR = {"L40S": 3.763952, "A100-40GB": 3.911952}
# H100 is not in slh/serve_spec.GPU_PREFERENCE nor in configs/prices.yaml; this is
# an UNVERIFIED planning figure (GPU about $3.95/h plus the corrected CPU and memory
# reservation $1.81/h); read the pricing page before any use.
H100_USD_PER_HOUR_UNVERIFIED = 5.76
WINDOW_S = 10800  # the owner's 3-hour hard stop (slh/serve_spec.MAX_WINDOW_S)
WINDOW_OVERHEAD_MIN = 25  # startup 10 + preflight 5 + drain 10 (amendment 6 allowances)
# Measured: 4 conversations in flight on one L40S, mean 147 s each (runs/b1-stream).
MEASURED_CONV_PER_MIN = 4 * 60 / 147.0


def _official_conversations(plan: Phase2Plan) -> int:
    return plan.conversations()["official_test"]


def cost_estimate(plan: Phase2Plan, *, sim_usd: float = SIM_USD_PER_CONV,
                  judge_usd: float = JUDGE_USD_PER_CONV, length_factor: float = 1.5,
                  smoke_conversations: int = 60, prescreen_seeds: int = 2,
                  prescreen_replacement: float = 0.15) -> dict[str, float]:
    """$ for the simulator (one key: agent conversations, smoke and the oracle
    pre-screen) and for the official judge (NL tasks of the official shard).
    ``length_factor``: generated multi-change conversations may be longer than the
    measured stream conversations (planning margin)."""
    n = plan.conversations()
    official = _official_conversations(plan)
    agent_convs = n["generated"] + official + smoke_conversations
    prescreen = round(prescreen_seeds * plan.generated_scenarios() * (1 + prescreen_replacement))
    sim_convs = agent_convs + prescreen
    sim = sim_convs * sim_usd
    judge = official * judge_usd
    return {"agent_conversations": agent_convs, "official_conversations": official,
            "prescreen_conversations": prescreen, "simulator_conversations": sim_convs,
            "simulator_usd": round(sim, 2),
            "simulator_usd_with_margin": round(sim * length_factor, 2),
            "judge_usd": round(judge, 2), "judge_usd_with_margin": round(judge * length_factor, 2)}


def gpu_estimate(plan: Phase2Plan, *, conv_per_min: float, gpu: str = "L40S",
                 usd_per_hour: float | None = None, window_s: int = WINDOW_S,
                 overhead_min: float = WINDOW_OVERHEAD_MIN, parallel_gpus: int = 1
                 ) -> dict[str, float]:
    """Our-side serving: windows of ``window_s`` on one GPU each, billed in full
    (conservative), ``conv_per_min`` conversations per GPU-minute of useful time."""
    convs = cost_estimate(plan)["agent_conversations"]
    useful = window_s / 60 - overhead_min
    minutes = convs / conv_per_min
    windows = math.ceil(minutes / useful)
    hours = windows * window_s / 3600
    rate = SANDBOX_USD_PER_HOUR[gpu] if usd_per_hour is None else usd_per_hour
    return {"conversations": convs, "conv_per_min": round(conv_per_min, 2),
            "serving_hours": round(minutes / 60, 1), "windows": windows,
            "billed_gpu_hours": hours, "gpu_usd": round(hours * rate, 2),
            "wall_days_continuous": round(windows * window_s / 86400 / parallel_gpus, 1)}


def noise_sd_points(agreement: tuple[int, int] = SAME_SEED_AGREEMENT) -> float:
    """SD (points) of one paired conversation difference from pure rerun noise:
    discordance q = 1 - agree/n, Var(D) = q under no gain, SD = 100 sqrt(q)."""
    agree, n = agreement
    return 100 * math.sqrt(1 - agree / n)


def conversation_design(scenarios_per_type: int, *, runs: int = 12, tau: float = 3.0,
                        sd_scenario: float | None = None, shared: float = 0.4,
                        rho: float = 0.5) -> Any:
    """Map the conversation design onto ``slh.xp.power.Design``: a 'group' is one
    scenario (one conversation per arm, scenario and run), tasks = decision types."""
    from slh.xp.power import Design

    sd = noise_sd_points() if sd_scenario is None else sd_scenario
    return Design(runs=runs, cases=3 * scenarios_per_type, tau=float(tau), rho=rho,
                  sd_group=float(sd), shared=shared, tasks=len(DECISION_TYPES))


def power_grid(scenarios: tuple[int, ...] = (100, 150, 200), taus: tuple[float, ...] = (3, 6),
               sds: tuple[tuple[float, float], ...] = ((46, 0.2), (55, 0.4), (63, 0.4)),
               runs: int = 12, reps: int = 2000, seed: int = 0, gain: float = 5.0
               ) -> list[dict[str, Any]]:
    """Small grid for the doc: pooled and per-type minimum detectable gain (80%) and
    pooled power at ``gain``. Light: a few seconds on a laptop."""
    from slh.xp.power import mdg, power, simulate

    rows = []
    for s in scenarios:
        for tau in taus:
            for sd, shared in sds:
                d = conversation_design(s, runs=runs, tau=tau, sd_scenario=sd, shared=shared)
                sim = simulate(d, reps=reps, seed=seed)
                m = mdg(sim)
                rows.append({"scenarios_per_type": s, "tau": tau, "sd": sd, "shared": shared,
                             "mdg_pooled": round(m["pooled"], 1),
                             "mdg_per_type": round(m["per_task"], 1),
                             f"power_at_{gain:g}": round(power(sim, gain)["pooled"], 2)})
    return rows


def main(argv: list[str] | None = None) -> int:  # python -m slh.xp.phase2 [generate ...]
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "generate":
        from slh.xp.phase2 import generator

        return generator.main(argv[1:])
    for name, plan in (("full", default_plan()), ("minimum", minimum_plan())):
        gpu = {label: gpu_estimate(plan, conv_per_min=rate, gpu=g, usd_per_hour=usd)
               for label, rate, g, usd in (
                   ("L40S, 4 in flight (measured)", MEASURED_CONV_PER_MIN, "L40S", None),
                   ("L40S, 8 in flight (planning)", 8 / (2.45 * 1.25), "L40S", None),
                   ("H100, 32 in flight (unverified)", 32 / (2.45 * 1.25 * 1.2), "L40S",
                    H100_USD_PER_HOUR_UNVERIFIED))}
        print(json.dumps({"plan": name, "conversations": plan.conversations(),
                          "cost": cost_estimate(plan), "gpu": gpu,
                          "problems": plan.problems()}, indent=1))
    print(json.dumps({"noise_sd": round(noise_sd_points(), 1)}))
    return 0
