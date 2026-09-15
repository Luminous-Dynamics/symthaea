# SPINE-000D — Scoped Causal Ablation Contract

**Status:** preregistered experimental-design contract; no production behavior change

**Authority:** measurement-only

**Issue:** #3232

SPINE-000D begins only after SPINE-000B can produce qualified execution/integration/application evidence. Its purpose is to determine whether a subsystem is causally load-bearing **for a defined observable under a defined workload and intervention**.

## 1. No global load-bearing label

The smallest valid causal claim is scoped:

```text
(subject,
 workload,
 observable,
 intervention,
 comparison,
 evidence_lineage)
```

Do not emit an unscoped statement such as:

```text
metacognition = LOAD_BEARING
```

Prefer:

```text
metacognition = LOAD_BEARING_POSITIVE
for prediction-calibration
under workload W
under OUTPUT_SHAM vs FULL
at subject H
with evidence lineage E
```

A module may be active for one metric and equivalent within bounds for another.

## 2. Canonical CampaignManifest

Every campaign freezes a machine-readable manifest before any intervention result is inspected:

```text
CausalCampaignManifest
  schema
  authority = measurement-only

  campaign_id
  subject
    git_head
    git_tree
    subject_file_hashes
    Cargo.toml hash
    Cargo.lock hash
    Rust toolchain
    Nix/environment identity when relevant

  target_subsystem
  static_overlap_refs[]
  runtime_influence_evidence_refs[]

  intervention_arms[]
  arm_order_policy
  rng_alignment_policy
  workload_rule
  initial_state_rule
  scheduler_state_rule

  primary_observables[]
  secondary_observables[]
  safety_hard_gates[]

  multiplicity_policy
  confidence_interval_policy
  interaction_followup_rule
  stopping_rule

  classification_scope_fields
```

The manifest is immutable once evidence execution starts.

The machine-readable JSON Schema is authoritative for required fields and conditional requirements. Prose must not be used to weaken a schema requirement after evidence begins.

## 3. Intervention arms answer different questions

The three base arms are mandatory for the first SPINE-000D causal campaign:

```text
FULL
OUTPUT_SHAM
DISABLED
```

`STATE_FROZEN` is optional.

### FULL

Normal production behavior.

### OUTPUT_SHAM

The target subsystem executes normally and its internal state evolves normally, but the output crossing the qualified proposal boundary is replaced with the exact neutral identity.

This asks:

> Does this subsystem's current outward proposal matter, holding its internal execution/state evolution approximately constant?

The neutral identity must be the production-qualified identity, not an invented zero structure.

### DISABLED

The target subsystem is not executed during the intervention window.

This asks:

> Does the full existence/execution of this subsystem matter over this window?

It may differ from OUTPUT_SHAM because internal subsystem state no longer evolves.

### STATE_FROZEN (optional)

The subsystem executes from a frozen/checkpointed state according to a campaign-specific rule and is restored so learning/accumulation is suppressed while immediate computation remains available.

This helps separate:

```text
immediate output effect
from
state-learning / adaptation effect
```

STATE_FROZEN is optional and must be justified per subsystem because checkpoint semantics differ.

## 4. Interventions must not silently alter unrelated execution

An arm must change only the preregistered intervention surface.

Examples of prohibited confounds:

- disabling target S also changes unrelated manager ordering;
- sham output changes urgency/scheduler cadence;
- instrumentation changes random-number consumption without declaration;
- one arm runs different features;
- arms use shared mutable subsystem state;
- one arm inherits caches from another.

When exact preservation is impossible, the difference becomes part of the intervention definition and must be declared before evidence.

## 5. Arm-order policy is mandatory

Execution order can confound latency, thermal state, cache warmth, allocator state, and long-lived external services.

Every campaign freezes one arm-order policy before evidence:

```text
COUNTERBALANCED
DETERMINISTIC_ROTATION
RANDOMIZED_FROZEN_SEED
FIXED_JUSTIFIED
```

`RANDOMIZED_FROZEN_SEED` requires its ordering seed to be frozen in the manifest.

`FIXED_JUSTIFIED` requires a written justification and should not be used for performance claims unless order effects are independently shown negligible or are part of the explicit scope.

Never always run FULL first and intervention second merely for convenience when latency/CPU/memory is a primary observable.

## 6. RNG alignment policy is mandatory

Interventions can change control flow and therefore random-number consumption. A later random draw difference can masquerade as subsystem causality.

Every campaign freezes one of:

```text
MATCHED_STREAMS
DECLARED_DIVERGENCE
```

### MATCHED_STREAMS

Use domain-separated/random streams or another qualified method so unrelated stochastic consumers remain aligned across arms.

### DECLARED_DIVERGENCE

Exact stream alignment is infeasible and the divergence is part of the intervention. The manifest must describe where divergence begins and causal claims are scoped accordingly.

Do not silently assume equal genesis seeds imply equal stochastic trajectories after different control-flow paths.

## 7. Paired execution is the default design

For each seed/workload pair, compare arms from matched initial conditions:

```text
same subject
same environment
same genesis/random seed
same input/workload sequence
same initial checkpoint digest
same feature set
same scheduler initial state
same external-tool fixtures
```

Each arm must use independent mutable runtime state.

For seed i:

```text
D_i = metric(intervention_i) - metric(control_i)
```

Primary inference uses paired differences whenever the metric permits.

## 8. Workload and seed selection

Campaigns use either:

1. `FROZEN_EXPLICIT` — a frozen non-empty set of workload IDs; or
2. `FROZEN_GENERATOR` — a frozen generator subject plus the hash of its materialized workload before result inspection.

The manifest must contain the data required by the chosen mode. A label without its underlying IDs/hashes is not a valid preregistration.

Do not add seeds only after seeing an inconvenient result unless the stopping rule preregistered sequential sampling.

If a result depends on a single seed or input, the claim must remain correspondingly narrow.

## 9. Initial-state and scheduler state are evidence subjects

`FRESH_GENESIS` and `FROZEN_CHECKPOINT` are distinct initial-state modes.

A frozen checkpoint mode requires an exact checkpoint hash. Scheduler state must also be defined explicitly so an arm is not compared from a different manager cadence or health state.

## 10. Primary vs secondary observables

Every campaign freezes a small primary set.

Examples:

```text
task success
hidden-test pass/fail
prediction error
Brier/calibration score
retrieval success
selected action identity
defined downstream state variable
latency
CPU usage
memory usage
```

Secondary/exploratory telemetry may be broad, but it cannot silently become a primary success criterion after results are known.

Each observable freezes an equivalence rule at preregistration time.

## 11. Safety and authority gates remain separate

Hard safety/authority outcomes are not averaged into a utility score.

Example:

```text
authority violation count must equal 0
```

A performance gain cannot compensate for a safety hard-gate failure.

Report:

```text
safety_gate_pass
performance_result
```

as separate fields.

## 12. Difference is not benefit

`CAUSAL_EFFECT_DETECTED` only means the intervention changes the measured observable.

A positive/negative value judgment requires a frozen metric direction or utility function.

Examples:

```text
lower prediction error is better
higher hidden-test pass rate is better
lower latency is better only under quality non-inferiority
lower memory is better only under quality/safety non-inferiority
```

If no direction was frozen, classify:

```text
CAUSALLY_ACTIVE_DIRECTION_UNRESOLVED
```

not beneficial/harmful.

## 13. Null claims require equivalence evidence

A nonsignificant difference is `INCONCLUSIVE`, not null.

Every primary metric eligible for a null/equivalence claim has one of:

```text
EXACT
ABSOLUTE_SESOI
RELATIVE_SESOI
NONINFERIORITY
```

Non-exact rules require an explicit numeric bound in the frozen manifest. `EXACT` intentionally has no numeric bound.

Possible outcomes:

```text
CAUSAL_EFFECT_DETECTED
EQUIVALENT_WITHIN_BOUND
INCONCLUSIVE
```

Only `EQUIVALENT_WITHIN_BOUND` supports a scoped `CONNECTED_NULL`-style conclusion.

For exact deterministic outputs, equivalence may be literal bit/identity equality where appropriate. For noisy metrics, use an appropriate paired equivalence procedure with frozen bounds.

## 14. Effect sizes and uncertainty are mandatory

Where statistical replication is used, report:

```text
paired effect estimate
uncertainty / confidence interval
raw paired differences
sample count
```

Do not report p-values alone.

A paired statistical confidence policy must freeze its confidence level before evidence. Exact deterministic campaigns use `EXACT_DETERMINISTIC` rather than artificial inferential statistics.

## 15. Multiplicity policy is frozen before evidence

Testing many subsystems across many metrics creates a discovery problem.

A campaign chooses one of:

- a small preregistered primary family;
- hierarchical gatekeeping;
- false-discovery-rate control;
- family-wise error control;
- another justified frozen policy.

`OTHER_FROZEN` requires its method to be specified before evidence.

Exploratory signals may be reported but must be labeled exploratory.

## 16. Redundancy and synergy require interaction follow-up

Single-module ablation is insufficient to infer independence.

For candidate pair A/B, the minimal interaction design is:

```text
FULL
-A
-B
-A-B
```

Interpretation examples:

```text
-A small, -B small, -A-B large
  -> REDUNDANCY_OR_SYNERGY_DETECTED

-A large, -B small, -A-B approximately -A
  -> A dominates measured effect under this scope
```

The exact interaction model depends on the metric and must be frozen before the follow-up run.

## 17. Interaction follow-up trigger

A campaign manifest freezes when interaction testing is required.

Recommended triggers include:

- static duplicate-path overlap;
- two subsystems influence the same integration channel;
- individually equivalent/null candidates share a responsibility class;
- unexpected sign reversal;
- large residual unexplained by single-subsystem effects.

A custom trigger requires frozen details.

## 18. Legacy inline paths can mask manager ablations

SPINE-000C overlap evidence is part of causal interpretation.

If target manager S has an explicit legacy inline overlap and S appears equivalent under ablation, valid classifications include:

```text
MASKED_BY_OVERLAP
INCONCLUSIVE
```

Do not call S globally null unless the masking path is addressed by a matched follow-up or the scope explicitly includes the overlap condition.

## 19. Acute output effect vs chronic state effect

Compare intervention contrasts intentionally:

```text
FULL vs OUTPUT_SHAM
  -> outward proposal contribution with internal execution retained

FULL vs DISABLED
  -> total execution/existence effect over the intervention window

OUTPUT_SHAM vs DISABLED
  -> internal-state / side-effect contribution not carried by qualified proposal output
```

This three-arm decomposition is particularly valuable during migration from inline behavior to proposal-based managers.

## 20. Cost-aware causal value

A subsystem may be causally active but operationally inefficient.

When compute matters, report a Pareto view over preregistered dimensions such as:

```text
quality
safety
latency
CPU time
memory
energy proxy if qualified
```

Arm ordering is part of the preregistered performance design; do not interpret order-confounded measurements as intrinsic subsystem cost.

Do not equate `LOAD_BEARING` with `cost-effective`.

## 21. Canonical PairedRunReceipt

```text
PairedRunReceipt
  schema
  authority = measurement-only

  campaign_id
  subject_manifest_hash
  target_subsystem
  workload_id
  seed
  arm
  arm_execution_ordinal

  initial_state_digest
  input_digest
  scheduler_state_digest
  rng_alignment_evidence

  safety_outcomes
  primary_observables
  secondary_observables
  compute_observables

  final_state_digest
  evidence_artifact_refs[]
```

Wall-clock/host metadata may live in a separate noncanonical runtime envelope when exact semantic replay is required.

## 22. Canonical ContrastReceipt

```text
ContrastReceipt
  campaign_id
  target_subsystem
  comparison
  observable
  scope

  paired_differences[]
  effect_estimate
  uncertainty
  equivalence_bound
  equivalence_decision
  effect_decision
  utility_direction
  safety_gate_pass

  classification
  evidence_refs[]
```

No classification field may omit the scope tuple.

## 23. Classification vocabulary

Allowed scoped classifications:

```text
LOAD_BEARING_POSITIVE
LOAD_BEARING_NEGATIVE
CAUSALLY_ACTIVE_DIRECTION_UNRESOLVED
EQUIVALENT_WITHIN_BOUND
INCONCLUSIVE
MASKED_BY_OVERLAP
INTERACTION_SUSPECTED
DORMANT_NOT_EXECUTED
SAFETY_GATE_FAILED
```

`CONNECTED_NULL` may appear only as a higher-level summary of one or more `EQUIVALENT_WITHIN_BOUND` results with their scopes preserved.

## 24. Harness sensitivity controls

Before trusting a campaign harness, include controls that prove it can detect both no-effect and known-effect cases.

### Null/sham control

A semantically identical intervention should reproduce the control outcome within the frozen deterministic/equivalence rule.

### Sensitivity control

Use a reversible synthetic harness perturbation or another preregistered known-effect fixture to prove the measurement path can detect a known change.

Do not use an unqualified cognitive subsystem as the sole positive control.

## 25. Stopping rules

Freeze the rule before evidence.

`FIXED_N` requires the exact paired sample count.

`FROZEN_SEQUENTIAL` requires its boundaries and continuation/termination rule to be specified in the manifest.

Never stop simply when a desired significance threshold first appears.

## 26. Evidence lineage

The campaign evidence manifest binds at minimum:

```text
Git HEAD
Git tree
intervention implementation
campaign manifest
arm-order policy
RNG-alignment policy
workload/seed materialization
Cargo.toml
Cargo.lock
Rust toolchain
environment identity
relevant SPINE-000B runtime receipt schema/version
static overlap census version
```

If subject code, intervention semantics, order policy, RNG policy, or primary analysis semantics change after evidence begins, start a new evidence lineage.

## 27. Claim boundary

SPINE-000D may establish only scoped causal claims such as:

> Under subject H, workload W, and intervention FULL vs OUTPUT_SHAM, subsystem S produced a replicated causal effect of magnitude E on preregistered observable M while safety gates remained satisfied.

It does not establish that S is universally necessary or beneficial.

## 28. Exit gate before first campaign

No first SPINE-000D campaign runs until all are frozen:

1. exact subject/environment;
2. target subsystem and overlap context;
3. mandatory FULL + OUTPUT_SHAM + DISABLED arms, plus optional STATE_FROZEN;
4. arm-order/counterbalancing policy;
5. RNG-alignment/divergence policy;
6. seed/workload rule and required IDs/hashes;
7. initial-state and scheduler-state rules;
8. primary/secondary observables;
9. safety hard gates;
10. metric directions/utility;
11. exact or bounded equivalence rules;
12. multiplicity policy;
13. confidence/effect-estimation policy;
14. interaction follow-up trigger;
15. stopping rule with required fixed N or sequential details;
16. evidence manifest/postflight rules;
17. classification vocabulary and scope serialization.
