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
  causal_claim_scope_version

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
  seed_or_workload_generation_rule
  initial_state_rule
  scheduler_state_rule

  primary_observables[]
  secondary_observables[]
  safety_hard_gates[]

  metric_direction_or_utility[]
  equivalence_bounds[]
  multiplicity_policy
  confidence_interval_policy
  interaction_followup_rule
  stopping_rule

  result_fields_allowed
  preregistration_hash
```

The manifest is immutable once evidence execution starts.

## 3. Intervention arms answer different questions

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

## 4. Interventions must not silently alter unrelated scheduling

An arm must change only the preregistered intervention surface.

Examples of prohibited confounds:

- disabling target S also changes manager ordering;
- sham output changes urgency/scheduler cadence;
- instrumentation changes random-number consumption;
- one arm runs different features;
- arms use shared mutable subsystem state;
- one arm inherits caches from another.

When exact preservation is impossible, the difference becomes part of the intervention definition and must be declared before evidence.

## 5. Paired execution is the default design

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

## 6. Workload and seed selection

Campaigns may use either:

1. a frozen explicit workload/seed set; or
2. a frozen untouched generation rule whose materialized set is hashed before result inspection.

Do not add seeds only after seeing an inconvenient result unless the stopping rule preregistered sequential sampling.

If a result depends on a single seed or input, the claim must remain correspondingly narrow.

## 7. Primary vs secondary observables

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

## 8. Safety and authority gates remain separate

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

## 9. Difference is not benefit

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

## 10. Null claims require equivalence evidence

A nonsignificant difference is `INCONCLUSIVE`, not null.

Every primary metric eligible for a null/equivalence claim must have a preregistered smallest effect size of interest (SESOI), exact deterministic tolerance, or non-inferiority bound.

Possible outcomes:

```text
CAUSAL_EFFECT_DETECTED
EQUIVALENT_WITHIN_BOUND
INCONCLUSIVE
```

Only `EQUIVALENT_WITHIN_BOUND` supports a scoped `CONNECTED_NULL`-style conclusion.

For exact deterministic outputs, equivalence may be literal bit/identity equality where appropriate. For noisy metrics, use an appropriate paired equivalence procedure with frozen bounds.

## 11. Effect sizes and uncertainty are mandatory

Where statistical replication is used, report:

```text
paired effect estimate
uncertainty / confidence interval
raw paired differences
sample count
```

Do not report p-values alone.

Where exact deterministic replay applies, report exact differences and replay identity instead of artificial inferential statistics.

## 12. Multiplicity policy is frozen before evidence

Testing many subsystems across many metrics creates a discovery problem.

A campaign must choose one of:

- a small preregistered primary family;
- hierarchical gatekeeping;
- false-discovery-rate control;
- family-wise error control;
- another justified frozen policy.

Exploratory signals may be reported but must be labeled exploratory.

## 13. Redundancy and synergy require interaction follow-up

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

## 14. Interaction follow-up trigger

A campaign manifest must freeze when interaction testing is required.

Recommended triggers include:

- static duplicate-path overlap;
- two subsystems influence the same integration channel;
- individually equivalent/null candidates share a responsibility class;
- unexpected sign reversal;
- large residual unexplained by single-subsystem effects.

## 15. Legacy inline paths can mask manager ablations

SPINE-000C overlap evidence is part of causal interpretation.

If target manager S has an explicit legacy inline overlap and S appears equivalent under ablation, valid classifications include:

```text
MASKED_BY_OVERLAP
INCONCLUSIVE
```

Do not call S globally null unless the masking path is addressed by a matched follow-up or the scope explicitly includes the overlap condition.

## 16. Acute output effect vs chronic state effect

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

## 17. Cost-aware causal value

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

Do not equate `LOAD_BEARING` with `cost-effective`.

## 18. Canonical PairedRunReceipt

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

  initial_state_digest
  input_digest
  scheduler_state_digest

  safety_outcomes
  primary_observables
  secondary_observables
  compute_observables

  final_state_digest
  evidence_artifact_refs[]
```

Wall-clock/host metadata may live in a separate noncanonical runtime envelope when exact semantic replay is required.

## 19. Canonical ContrastReceipt

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

## 20. Classification vocabulary

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

## 21. Harness sensitivity controls

Before trusting a campaign harness, include controls that prove it can detect both no-effect and known-effect cases.

### Null/sham control

A semantically identical intervention should reproduce the control outcome within the frozen deterministic/equivalence rule.

### Sensitivity control

Use a reversible synthetic harness perturbation or another preregistered known-effect fixture to prove the measurement path can detect a known change.

Do not use an unqualified cognitive subsystem as the sole positive control.

## 22. Stopping rules

Freeze the rule before evidence, for example:

```text
fixed N paired seeds/workloads
```

or a justified sequential rule with frozen boundaries.

Never stop simply when a desired significance threshold first appears.

## 23. Evidence lineage

The campaign evidence manifest binds at minimum:

```text
Git HEAD
Git tree
intervention implementation
campaign manifest
workload/seed materialization
Cargo.toml
Cargo.lock
Rust toolchain
environment identity
relevant SPINE-000B runtime receipt schema/version
static overlap census version
```

If subject code or intervention semantics change after evidence begins, start a new evidence lineage.

## 24. Claim boundary

SPINE-000D may establish only scoped causal claims such as:

> Under subject H, workload W, and intervention FULL vs OUTPUT_SHAM, subsystem S produced a replicated causal effect of magnitude E on preregistered observable M while safety gates remained satisfied.

It does not establish that S is universally necessary or beneficial.

## 25. Exit gate before first campaign

No first SPINE-000D campaign runs until all are frozen:

1. exact subject/environment;
2. target subsystem and overlap context;
3. intervention arms;
4. seed/workload rule;
5. initial-state rule;
6. primary/secondary observables;
7. safety hard gates;
8. metric directions/utility;
9. equivalence bounds;
10. multiplicity policy;
11. confidence/effect-estimation policy;
12. interaction follow-up trigger;
13. stopping rule;
14. evidence manifest/postflight rules;
15. classification vocabulary and scope serialization.
