# REGEN-042S — Symthaea Compound-Shock Kernel Profile v1

Status: implementation-profile preregistration only

Mycelix normative campaign contract: REGEN-042 / Luminous-Dynamics/mycelix#1431

Mycelix transition-semantics hardening: REGEN-042A / Luminous-Dynamics/mycelix#1519

Cross-repo ownership boundary: Luminous-Dynamics/mycelix#1523

Symthaea predecessors: REGEN-040 / #3800 and REGEN-041 / #3803 are assessor/model hardening profiles, not authoritative replacements for Mycelix service/dependency contracts.

## Purpose

Define the first executable Symthaea implementation profile for deterministic compound-shock analysis while preserving the cross-repo ownership boundary:

```text
Mycelix adopted/evidence state
        |
        | immutable versioned assessment envelope
        v
Symthaea REGEN-042S kernel
        |
        | deterministic model receipt / witness
        v
Mycelix evidence/review path
```

REGEN-042S does not own essential-service definitions, rights, ecological obligations, quality/safety authority, procurement, emergency policy, or physical control.

## 1. Kernel theorem

```text
validated immutable assessment envelope
+ canonical model state
+ typed shock effects
+ deterministic interval reducer
+ finite reserve/recovery accounting
+ explicit unresolved/error states
= replayable compound-shock model receipt
```

not:

```text
model receipt
= real service observation
= disaster forecast
= real hazard probability
= emergency authority
= actuator command
```

## 2. Proposed crate boundary

The initial executable target should be a small domain crate conceptually named:

```text
crates/domains/symthaea-regenerative-resilience
```

The exact repository path may be adjusted before implementation, but the dependency boundary is normative.

The core should not depend directly on:

- Holochain;
- Mycelix runtime crates;
- networking;
- databases;
- LLMs;
- adaptive learning;
- robotics/actuator HALs;
- helicopter control code;
- governance runtime;
- marketplace/finance runtime.

A bridge/adapter layer may translate frozen Mycelix assessment payloads into core types.

## 3. Reuse concepts, not aviation authority

Existing `symthaea-helicopter::service_resilience` and `common_cause` contain useful ideas:

- service criticality;
- explicit dependency graphs;
- outage/recovery objectives;
- common-cause failure domains;
- independence not inferred from distinct component IDs.

REGEN-042S should generalize these concepts into neutral model types rather than importing aviation-specific roles, lanes, or flight authority.

## 4. Reuse protected-reserve concept, not mission policy

Existing subterranean logistics demonstrates the useful invariant that discretionary work cannot consume a protected return reserve.

REGEN-042S should generalize reserve accounting while leaving reserve policy in the supplied Mycelix/adopted profile.

```text
model can enforce supplied reserve class
!= model can decide reserve policy
```

## 5. Canonical identity layer

The kernel should use opaque, ordered newtypes for external identities.

Conceptually:

```rust
pub struct ServiceId(CanonicalId);
pub struct DependencyId(CanonicalId);
pub struct StockId(CanonicalId);
pub struct FailureDomainId(CanonicalId);
pub struct EventId(CanonicalId);
pub struct RecoveryResourceId(CanonicalId);
pub struct CampaignRevisionId(CanonicalId);
pub struct ProfileRevisionId(CanonicalId);
pub struct EvidenceSnapshotId(CanonicalId);
pub struct ModelRevisionId(CanonicalId);
```

The IDs should preserve upstream identity exactly or bind an exact adapter mapping.

Symthaea must not silently mint replacement canonical IDs for authoritative Mycelix entities.

## 6. Canonical quantity representation

The transition kernel should avoid unvalidated bare floating-point quantities at authority/evidence boundaries.

A useful initial representation is an exact fixed-decimal quantity:

```rust
pub struct CanonicalQuantity {
    pub mantissa: i128,
    pub scale: u8,
    pub unit: UnitId,
    pub basis: BasisId,
}
```

with canonical normalization rules.

Alternative exact representations may be selected, but the following are mandatory:

- finite by construction;
- explicit unit/basis identity;
- exact comparison/addition when compatible;
- checked overflow;
- no implicit unit conversion;
- no stock/rate conflation.

## 7. Non-negative physical stocks

For conserved/non-negative resources, use a separate validated type rather than relying on convention.

Conceptually:

```rust
pub struct NonNegativeQuantity(CanonicalQuantity);
```

A physical-stock transition may not produce a negative amount.

## 8. Time model

The first kernel should use integer campaign ticks with an explicit tick-duration identity supplied by the campaign.

Conceptually:

```rust
pub struct Tick(pub u64);

pub struct TimeInterval {
    pub start: Tick,
    pub end_exclusive: Tick,
}
```

Half-open intervals remove boundary ambiguity.

Wall-clock timestamps may be carried as evidence metadata but should not define deterministic transition ordering unless the campaign explicitly maps them to canonical ticks.

## 9. Immutable input envelope

The core receives an already validated snapshot conceptually shaped as:

```rust
pub struct AssessmentEnvelope {
    pub campaign_revision: CampaignRevisionId,
    pub service_profile_revision: ProfileRevisionId,
    pub dependency_graph_revision: ProfileRevisionId,
    pub evidence_snapshot: EvidenceSnapshotId,
    pub authority_profile_revisions: Vec<ProfileRevisionId>,
    pub ecology_profile_revisions: Vec<ProfileRevisionId>,
    pub quality_safety_profile_revisions: Vec<ProfileRevisionId>,
    pub timebase: Timebase,
    pub horizon: TimeInterval,
    pub input_commitment: Commitment,
}
```

The kernel does not perform mutable live lookups.

## 10. Service state

A service should remain plural and profile-bound.

Conceptually:

```rust
pub enum ServiceAvailability {
    Satisfied,
    Degraded,
    Failed,
    Unresolved,
}

pub struct ServiceState {
    pub service_id: ServiceId,
    pub availability: ServiceAvailability,
    pub delivered: Option<CanonicalQuantity>,
    pub required_floor: Option<CanonicalQuantity>,
    pub affected_scope_ref: ScopeId,
    pub active_dependency_refs: Vec<DependencyId>,
}
```

The kernel may not invent a service floor when the authoritative profile omits or leaves it unresolved.

## 11. Dependency state

Conceptually:

```rust
pub enum DependencyAvailability {
    Available,
    Degraded,
    Unavailable,
    Unresolved,
}

pub struct DependencyState {
    pub dependency_id: DependencyId,
    pub availability: DependencyAvailability,
    pub nominal_capacity: Option<CanonicalQuantity>,
    pub usable_capacity: Option<CanonicalQuantity>,
    pub committed_capacity: Option<CanonicalQuantity>,
    pub failure_domains: Vec<FailureDomainId>,
}
```

Availability and capacity remain separate.

## 12. Stock state

Conceptually:

```rust
pub struct StockState {
    pub stock_id: StockId,
    pub commodity: CommodityId,
    pub quantity: NonNegativeQuantity,
    pub protected_quantity: NonNegativeQuantity,
    pub committed_quantity: NonNegativeQuantity,
}
```

Required invariant:

```text
protected + committed <= total
```

unless the supplied profile explicitly distinguishes overlapping reservation classes with another validated accounting scheme.

## 13. Shock effect algebra

The first kernel should implement an intentionally small effect enum matching REGEN-042A semantics.

Conceptually:

```rust
pub enum ShockEffect {
    DependencyUnavailable {
        dependency: DependencyId,
    },
    CapacityScaled {
        dependency: DependencyId,
        factor: ExactRatio,
    },
    CapacityCapped {
        dependency: DependencyId,
        maximum: NonNegativeQuantity,
    },
    DemandScaled {
        service: ServiceId,
        factor: ExactRatio,
    },
    LeadTimeExtended {
        path: ProvisionPathId,
        delta_ticks: u64,
    },
    InventoryLoss {
        stock: StockId,
        quantity: NonNegativeQuantity,
    },
    FailureDomainUnavailable {
        failure_domain: FailureDomainId,
    },
    RecoveryResourceUnavailable {
        resource: RecoveryResourceId,
    },
}
```

New effect classes require a contract revision rather than being smuggled through a generic key/value payload.

## 14. Exact ratios

Capacity/demand scaling should avoid unconstrained floats.

Conceptually:

```rust
pub struct ExactRatio {
    numerator: u64,
    denominator: NonZeroU64,
}
```

with checked multiplication and an explicit rounding policy when applied to integer/fixed-decimal quantities.

The rounding rule is part of the model revision.

## 15. Event identity

Conceptually:

```rust
pub struct ShockEvent {
    pub event_id: EventId,
    pub interval: TimeInterval,
    pub source_class: ShockSourceClass,
    pub effects: Vec<ShockEffect>,
    pub profile_ref: ProfileRevisionId,
}
```

`source_class` distinguishes at least synthetic fixtures from evidence-backed scenario inputs.

It does not assign real-world probability.

## 16. Exogenous vs derived state

A shock event records only exogenous/model-input effects.

Service deficits, dependency propagation, recovery delays, and cascading consequences are emitted as derived transition consequences.

```text
injected event
!= derived failure
```

This prevents double-counting one causal chain as multiple independent shocks.

## 17. Canonical event ordering

The event set active at a tick is ordered by canonical `EventId` only for stable processing/receipt representation.

Ordering alone must not decide conflicting same-target semantics.

Before reduction, the kernel performs conflict analysis.

## 18. Same-target conflict rule

The initial v1 should be conservative.

Effects targeting the same state dimension at the same tick are accepted only if the pair is in an explicitly reviewed reducer table.

Otherwise:

```rust
TransitionError::AmbiguousSimultaneousEffects { ... }
```

is returned.

This is preferable to inventing an arbitrary winner.

## 19. Initial safe reducer examples

Potential explicitly reviewed same-target reducers may include:

- duplicate `DependencyUnavailable` for the same dependency is idempotent;
- duplicate `FailureDomainUnavailable` for the same domain is idempotent;
- identical `CapacityCapped` values are idempotent.

Mixed `CapacityScaled` + `CapacityCapped`, loss + replenishment, recovery + outage, or multiple non-identical caps should remain unsupported in v1 unless an explicit reducer is frozen.

## 20. Disjoint commutativity

Effects over disjoint state targets should be proven through metamorphic tests to commute when their downstream derived dependency/service updates are also independent.

The kernel must not assume graph-level independence merely because the immediate IDs differ; shared dependencies can make effects interact.

## 21. No-op event

The core should include an explicit no-op fixture representation or allow an empty effect vector only when the campaign contract permits it.

Substantive state identity law:

```text
Apply(S, NoOp) = S
```

The transition receipt may still append an evaluation record.

## 22. Interval execution order

A simple first interval reducer may freeze this order:

```text
1. validate prior state
2. resolve active exogenous effects
3. apply accepted exogenous state changes
4. propagate failure-domain/dependency effects
5. account stocks/reserves/commitments
6. evaluate already-qualified fallback/substitution availability
7. allocate finite recovery resources under supplied scheduler profile
8. advance recovery timers
9. compute delivered service against adopted floors
10. emit successor state + transition receipt
```

The order is normative for a model revision.

Changing it creates a new model revision and requires requalification.

## 23. No dynamic substitute invention

The core consumes only substitute/fallback paths frozen in the assessment envelope.

A search module may later propose new candidates, but those candidates do not enter the current campaign lineage until admitted through the Mycelix contract/profile path.

## 24. Reserve accounting

A reserve is a finite stock with reservation semantics.

Conceptually:

```rust
pub struct ReserveClaim {
    pub reserve_id: StockId,
    pub service_id: ServiceId,
    pub quantity: NonNegativeQuantity,
    pub interval: TimeInterval,
}
```

Concurrent claims cannot exceed service-usable reserve capacity.

## 25. Protected reserve

Protected reserve cannot be consumed by ordinary fallback logic unless the supplied authoritative profile explicitly permits release for that service/scenario.

The kernel does not infer emergency release authority.

## 26. Fallback state

A fallback should carry at least:

- availability;
- usable capacity;
- dependencies;
- failure domains;
- activation delay;
- finite backing stock where relevant;
- cooldown/dwell constraints where supplied;
- profile/evidence refs.

`fallback_active` alone is insufficient.

## 27. Recovery resource state

Conceptually:

```rust
pub struct RecoveryResourceState {
    pub resource_id: RecoveryResourceId,
    pub capacity: u32,
    pub commitments: Vec<RecoveryCommitment>,
    pub availability: DependencyAvailability,
}
```

A worker/tool/spare with capacity one cannot repair two simultaneous failures at the same interval.

## 28. Recovery candidate state machine

A useful v1 model distinction is:

```rust
pub enum RecoveryState {
    Proposed,
    Feasible,
    Scheduled,
    InProgress,
    AwaitingVerification,
    Restored,
    Unresolved,
}
```

The kernel may omit stages not represented by the authoritative profile, but it must not collapse proposal directly into restored service.

## 29. Recovery scheduling

The kernel consumes an explicit scheduler profile ID.

The scheduler may be deterministic, but its priority rule is a modeling/policy input.

Symthaea does not invent a morally or politically binding emergency priority ordering.

## 30. Horizon censoring

Conceptually:

```rust
pub enum RecoveryOutcome {
    RestoredAt(Tick),
    NotRestoredWithinHorizon,
    UnrecoverableUnderModel,
    Unresolved,
}
```

`NotRestoredWithinHorizon` must never be relabeled `UnrecoverableUnderModel`.

## 31. Invalid fixture vs resilience failure

Execution status remains separate from modeled service outcome.

Conceptually:

```rust
pub enum ExecutionStatus {
    Completed,
    CompletedWithUnresolvedState,
    InvalidFixture,
    UnsupportedModelState,
    ExecutionFailure,
}
```

A malformed fixture is not evidence that the modeled service failed.

## 32. Transition receipt

Every interval should emit a canonical receipt conceptually containing:

```rust
pub struct TransitionReceipt {
    pub prior_state_commitment: Commitment,
    pub interval: TimeInterval,
    pub applied_event_refs: Vec<EventId>,
    pub derived_consequence_refs: Vec<DerivedConsequenceId>,
    pub recovery_commitment_refs: Vec<RecoveryCommitmentId>,
    pub service_outcome_refs: Vec<ServiceOutcomeId>,
    pub unresolved_refs: Vec<UnresolvedId>,
    pub successor_state_commitment: Commitment,
    pub model_revision: ModelRevisionId,
}
```

Receipts are append-only evidence artifacts.

## 33. Canonical commitment

A state commitment must use a versioned canonical serialization or another explicitly frozen commitment algorithm.

Do not hash:

- Rust `Debug` output;
- unordered map iteration;
- host-dependent floating formatting;
- pointer/allocator-derived values.

## 34. Deterministic collections

Use ordered data structures or canonical sorting before scientific commitments.

`BTreeMap`/`BTreeSet` or explicit stable sorted vectors are preferable in the first implementation.

## 35. Bridge remains outside core

A Mycelix adapter conceptually performs:

```text
Mycelix immutable REGEN assessment payload
-> schema validation
-> exact identity/unit/time mapping
-> REGEN-042S core state
```

The core receives only the validated result.

The adapter itself must not alter authoritative meaning.

## 36. Unknown preservation

The adapter and core must preserve:

```text
Missing
Unknown
Unresolved
Unavailable
Failed
```

as distinct states where supplied by the contract.

No `unwrap_or_default()` may convert scientific/authority uncertainty into success.

## 37. Schema skew

Unknown newer schema/effect/profile versions should produce explicit unsupported-schema/model state.

The kernel must not ignore unknown fields when those fields may alter semantics unless the wire contract explicitly classifies them as non-semantic extensions.

## 38. No network inside deterministic run

The first qualification target should execute entirely from local frozen fixtures.

No HTTP, Holochain, database, DNS, clock service, or remote model call belongs in the deterministic transition path.

This makes replay and causal attribution inspectable.

## 39. No adaptive learning in transition kernel

Model parameters/rules remain frozen throughout a campaign lineage.

A learned model may later produce an input/proposal or run in shadow comparison, but the v1 transition kernel does not self-update while evaluating a campaign.

## 40. No physical-control output type

The core should export analysis/receipt/result types only.

It should not define actuator commands, equipment setpoints, or generic `execute()` APIs.

## 41. Comparison key

Counterfactual comparison requires an exact key conceptually containing:

```rust
pub struct ComparisonKey {
    pub baseline_state: Commitment,
    pub service_profile_revision: ProfileRevisionId,
    pub dependency_graph_revision: ProfileRevisionId,
    pub event_schedule_commitment: Commitment,
    pub effect_profile_revisions: Vec<ProfileRevisionId>,
    pub model_revision: ModelRevisionId,
    pub endpoint_revision: ProfileRevisionId,
    pub horizon: TimeInterval,
    pub seed_or_sample_ref: Option<CanonicalId>,
}
```

Only declared experimental dimensions may differ between paired runs.

## 42. Comparison rejection

The analysis layer must reject ordinary paired-comparison claims when material comparison-key fields differ unexpectedly.

It may still display the runs side-by-side as non-paired scenarios if clearly labeled.

## 43. Randomness policy

The first kernel should preferably be deterministic.

If randomness is added later:

- RNG algorithm/version is frozen;
- seed set is campaign input;
- draws are reproducible;
- paired counterfactuals use common random numbers where appropriate;
- synthetic draw frequencies are not real hazard probabilities.

## 44. First module decomposition

A compact initial crate could be organized as:

```text
src/
  ids.rs
  quantity.rs
  time.rs
  state.rs
  event.rs
  reducer.rs
  dependency.rs
  reserve.rs
  recovery.rs
  service.rs
  receipt.rs
  comparison.rs
  error.rs
```

Do not create plugin systems or a generic simulation framework before the REGEN fixture set proves the need.

## 45. Trusted computing base

The scientific TCB for v1 should be deliberately small:

- canonical type validators;
- event conflict checker/reducer;
- stock/reserve arithmetic;
- dependency propagation;
- finite recovery-resource scheduler;
- service-floor evaluator;
- canonical receipt/commitment encoder.

Visualization, dashboards, explanations, optimization and HDC reasoning remain outside this core.

## 46. First test fixture family

The first executable campaign should implement the REGEN-042A fixture family, including:

1. no-op identity;
2. zero-duration identity;
3. disjoint commuting events;
4. intentionally non-commuting sequences;
5. ambiguous same-target simultaneous effects rejected;
6. binary dependency outage;
7. partial capacity loss;
8. demand shock without supply mutation;
9. inventory loss bounded by stock;
10. availability loss without material destruction;
11. fallback switching delay;
12. fallback reserve exhaustion;
13. shared failure-domain fallback rejection;
14. fallback under-capacity case;
15. two recoveries contending for one finite resource;
16. derived secondary service deficit from recovery contention;
17. unrecovered-at-horizon censoring;
18. exact paired-comparison success;
19. comparison-key mismatch rejection;
20. deterministic replay and witness export.

## 47. Additional property tests

Property-based tests should exercise:

- stock never negative;
- protected + committed never exceeds total under valid state;
- checked arithmetic never wraps;
- no-op preserves substantive commitment;
- rejected ambiguous events leave state unchanged;
- failed transition emits no successor state;
- canonical event-set ordering is stable;
- same frozen campaign replay produces identical canonical receipts;
- unknown input cannot become Satisfied without an explicit transition rule;
- adding an unavailable recovery resource cannot increase usable recovery capacity.

## 48. Mutation targets

A future mutation campaign should kill mutations that:

- drop a dependency edge;
- ignore a failure domain;
- reset inventory between shocks;
- permit reserve overcommitment;
- treat unknown as available;
- ignore activation delay;
- release protected reserve implicitly;
- mark recovery complete before verification;
- treat horizon censoring as successful recovery;
- skip comparison-key fields;
- reorder ambiguous same-target events instead of rejecting them.

## 49. Independent oracle strategy

Not every graph result needs a second full simulator.

Use narrow independent oracles where possible:

- hand-computed stock/runway vectors;
- exact small dependency cuts;
- analytic capacity arithmetic;
- exhaustive enumeration for tiny event sets;
- independently authored expected transition vectors;
- metamorphic identities.

The implementation agreeing with itself is not validation.

## 50. Qualification stages

A sensible execution program is:

```text
042S-0  types + canonical quantity/time validation
042S-1  immutable state + commitments
042S-2  typed effects + conflict checker
042S-3  deterministic reducer + no-op/ordering fixtures
042S-4  dependency/common-cause propagation
042S-5  reserves/fallback capacity
042S-6  recovery contention + censoring
042S-7  service outcomes
042S-8  comparison keys
042S-9  canonical receipts/witnesses
042S-10 adversarial/property/mutation qualification
042S-11 cross-repo bridge fixtures
```

Each stage should freeze exact-head evidence before widening the theorem.

## 51. Cross-repo bridge qualification

After the pure core is qualified, a separate adapter campaign should prove that a frozen Mycelix fixture and its Symthaea representation preserve:

- exact service/dependency identity;
- profile revisions;
- unknown/unresolved states;
- units/bases;
- time horizon;
- failure-domain identity;
- campaign/effect identity;
- canonical commitment.

The adapter qualification does not transfer automatically from core qualification.

## 52. Symthaea reasoning layer

Only after deterministic execution exists should higher Symthaea cognition consume receipts to:

- explain bottlenecks;
- search candidate substitutions;
- identify informative experiments;
- compare qualified strategies;
- surface counterexamples.

Its outputs remain proposals/recommendations.

## 53. Experiment proposal boundary

An experiment candidate emitted from model uncertainty must bind the exact uncertainty/witness that motivated it.

```text
model uncertainty
-> experiment proposal
!= experiment authorization
```

## 54. Common-mode witness

When the model finds fake redundancy, emit a machine-readable witness containing exact:

- campaign revision;
- state commitment;
- event refs;
- failure-domain refs;
- affected dependencies;
- affected services;
- transition receipts.

That witness becomes the natural input to Mycelix REGEN-047 adversarial review.

## 55. Counterexample-first review

A failed invariant or surprising path is retained and minimized before changing the model or fixture.

Do not repair the campaign after observing a failure merely to obtain a more favorable resilience result.

## 56. Performance is secondary to traceability

The first kernel should prioritize deterministic inspectability over large-scale speed.

Only optimize after a reference implementation and frozen fixture corpus exist.

Any optimization must demonstrate semantic equivalence against the reference corpus.

## 57. No universal resilience score

The kernel returns plural service trajectories, dependency states, deficits, recovery outcomes, and unresolved items.

It does not expose a normative `resilience_score`, tier, grade, or winner.

## 58. No real-world probability

Even a fully enumerated synthetic campaign is sensitivity/falsification evidence unless a separately qualified probabilistic hazard model supplies real-world sampling meaning.

## 59. Safety and authority boundary

REGEN-042S contains no:

- emergency commands;
- infrastructure control;
- resource seizure;
- water/energy dispatch;
- equipment repair procedures;
- procurement commitments;
- ecological exemptions;
- rights overrides;
- physical actuation.

## 60. Promotion gate

An initial executable REGEN-042S kernel should not be described as qualified until its exact ProductHead demonstrates at least:

- pinned toolchain/dependency state;
- deterministic replay;
- canonical identity/unit/time validation;
- no-op identity;
- explicit same-time conflict semantics;
- checked stock/reserve arithmetic;
- common-cause propagation;
- fallback availability/capacity verification;
- finite recovery-resource contention;
- horizon censoring;
- service-floor evaluation;
- comparison-key enforcement;
- canonical transition receipts;
- property/adversarial oracle coverage;
- postflight immutability;
- no authority-bearing output.

## 61. Deliberate non-claims

REGEN-042S does not establish real hazard probability, real disaster forecasts, guaranteed continuity, community self-sufficiency, policy optimality, emergency authority, resource ownership, ecological/right exemptions, infrastructure safety, procurement decisions, repair instructions, or physical control.

It establishes only the implementation profile for a small deterministic Symthaea kernel capable of executing the exact bounded compound-shock semantics frozen on the Mycelix side.
