# REGEN-042S2 — Typed Exogenous Effect Validation v1

Status: preregistration only.

State predecessor: REGEN-042S1 / #3823 plus hardening #3832 and #3833.
Primitive predecessor: REGEN-042S0A / #3831.
Authoritative transition semantics: Luminous-Dynamics/mycelix#1519 / REGEN-042A.
Cross-repo ownership: Luminous-Dynamics/mycelix#1523.

## 1. Purpose

S1/S1A/S1B freeze immutable canonical model state. S2 freezes the effect grammar and conflict-analysis boundary that must exist before any effect is allowed to transform that state.

Core theorem:

```text
validated committed prior state
+ immutable exogenous event set
+ typed effects
+ exact target/timing validation
+ deterministic conflict analysis
= validated effect plan
```

not:

```text
validated effect plan
= successor state
= resilience outcome
= real-world event
= hazard probability
= authority
```

S2 performs validation/canonicalization only. S3 will own state transition/reduction.

## 2. Exogenous effects are not derived consequences

S2 accepts only exogenous campaign inputs.

The following remain distinct:

```text
ExogenousShockEffect
DerivedDependencyFailure
DerivedServiceDeficit
DerivedRecoveryDelay
```

A service deficit produced by a grid-loss fixture is not reintroduced as a second exogenous shock.

## 3. Shock labels are not executable semantics

Labels such as `drought`, `grid-loss`, `severe`, or `extreme` do not enter the S2 core as overloaded severity values.

The bridge resolves labels/profiles into exact typed effects before validation.

```text
scenario label
-> exact profile revision
-> typed effect set
```

An unresolved profile is not executable input.

## 4. Initial event envelope

Conceptually:

```rust
pub struct ShockEventInput {
    pub event_id: EventId,
    pub timing: EventTiming,
    pub profile_revision: ProfileRevisionId,
    pub effects: Vec<ShockEffectInput>,
}
```

A validated form should have private fields and canonical ordering.

Changing the profile revision changes event identity/context even when the resolved effect payload happens to be numerically equal.

## 5. Initial typed effect family

The first Symthaea executable subset should cover only state dimensions already frozen by S1/S1B:

```rust
pub enum ShockEffectInput {
    NoOpFixture,
    DependencyUnavailable { dependency: DependencyId },
    DependencyCapacityScaled { dependency: DependencyId, factor: ExactRatio },
    DependencyCapacityCapped { dependency: DependencyId, maximum: NonNegativeQuantity },
    ScenarioDemandScaled { service: ServiceId, factor: ExactRatio },
    ProvisionLeadTimeExtended { path: ProvisionPathId, delta_ticks: u64 },
    InventoryLoss { stock: StockId, quantity: NonNegativeQuantity, destination: LossDestination },
    FailureDomainUnavailable { failure_domain: FailureDomainId },
}
```

Exact Rust spelling may differ. Materially different semantics must not be collapsed into one `severity` scalar.

## 6. Deferred effect classes remain explicit

REGEN-042A also names quality-state and recovery-resource effects. S2 v1 deliberately defers them until their target state is separately represented and qualified.

```text
unsupported effect class
!= ignored effect
```

The bridge/kernel returns a typed unsupported result rather than dropping the effect.

## 7. Demand effect targets scenario demand only

`ScenarioDemandScaled` targets the S1B `scenario_demand` field.

It may not mutate:

- adopted service floor;
- service-profile identity;
- authoritative policy state;
- delivered-service history.

This is a hard target-compatibility rule.

## 8. Physical inventory loss requires disposition

`InventoryLoss` represents physical modeled quantity leaving the stock boundary.

It must preserve an explicit destination/disposition such as:

```rust
pub enum LossDestination {
    ExternalSink(CanonicalId),
    Unresolved(CanonicalId),
}
```

Exact naming may differ.

Access loss, custody loss, or rights unavailability must not be represented by physically subtracting stock merely for convenience.

```text
unavailable != destroyed
```

## 9. Inventory loss cannot exceed modeled stock in v1

S2 v1 chooses a strict fixture rule:

```text
requested physical loss <= current modeled total stock
```

A larger loss request is an invalid fixture under this profile, not silent negative inventory and not implicit clamping.

A future profile may model capped loss plus unmet-loss metadata, but that requires an explicit revision.

## 10. Capacity effects target capacity-bearing dependencies

A capacity scale/cap effect requires the target dependency to exist and expose compatible known capacity semantics required by the selected effect.

The validator must reject unit/basis mismatch for a cap.

No implicit conversion is allowed.

A factor need not be interpreted as morally or physically “worse”; S2 validates representation, not the scenario author's narrative label.

## 11. Lead-time extension targets provision-path time

A lead-time effect targets an identified provision path with represented lead-time semantics.

S2 verifies the target exists and the delta can be represented.

S3 later owns checked application/overflow into successor state.

## 12. Failure-domain effect does not directly rewrite dependencies in S2

`FailureDomainUnavailable` targets the failure-domain state only.

Derived dependency consequences belong to the reducer/propagation stage.

This prevents one exogenous effect from being logged once at the domain and again as multiple independent dependency shocks.

## 13. Explicit no-op fixture

A no-op campaign control is represented explicitly rather than inferred from an empty/malformed event.

A `NoOpFixture` event must contain no mutating effects.

Its later S3 identity law is:

```text
substantive successor state == prior state
```

apart from transition/evaluation receipt metadata.

## 14. Empty event handling

An event with no effects and no explicit no-op semantics is invalid input.

This catches profile-resolution bugs that otherwise appear as successful harmless shocks.

## 15. Event identities are unique

Duplicate `event_id` values in one validated campaign slice are rejected before canonical collection materialization.

No last-write-wins and no implicit deduplication are allowed.

## 16. Effect target key

S2 derives a canonical direct-write target for conflict analysis, conceptually:

```rust
pub enum EffectTargetKey {
    DependencyAvailability(DependencyId),
    DependencyCapacity(DependencyId),
    ServiceScenarioDemand(ServiceId),
    ProvisionLeadTime(ProvisionPathId),
    StockQuantity(StockId),
    FailureDomainAvailability(FailureDomainId),
}
```

`NoOpFixture` has no mutating target.

## 17. Direct target conflict is not causal dependence

Two effects with different direct target keys may still be causally connected through the dependency graph.

S2's direct-write conflict checker does not claim causal independence.

It only proves that the effects do not attempt ambiguous concurrent writes to the same modeled state dimension.

## 18. Temporal overlap is explicit

Target conflict exists only when same-target effects overlap in campaign time.

S2 must correctly handle:

- instant vs instant;
- instant inside interval;
- overlapping intervals;
- adjacent half-open intervals.

For half-open intervals:

```text
[a,b) and [b,c) are adjacent, not overlapping
```

S0A owns the checked interval algebra consumed here.

## 19. V1 same-target overlap rule is fail-closed

For S2 v1:

```text
same direct target
+ overlapping timing
+ more than one mutating effect
=> conflict / not executable
```

S2 does not try to be clever by multiplying scales, selecting the strongest cap, or ordering event IDs.

A later reducer profile may explicitly qualify selected same-target compositions.

## 20. Duplicate equal effects are still duplicates

Two identical overlapping effects do not become safe merely because applying both would appear idempotent in one implementation.

Fixture duplication is preserved as an input error/conflict unless an explicit later profile defines idempotent duplicate semantics.

## 21. Same-target non-overlapping effects may coexist

Two effects on the same target are not a simultaneous-write conflict when their event timings do not overlap.

Their order remains part of campaign history and S3 transition sequencing.

## 22. Canonical event ordering

Validated events are canonically ordered by exact `EventId` bytes only for representation/replay.

Canonical ordering does not resolve a semantic conflict.

```text
stable sort
!= conflict reducer
```

## 23. Effect ordering inside an event

Mutating effects in one event are canonically ordered by target key and frozen variant tag after validation.

If two effects would occupy the same overlapping direct target, validation fails rather than relying on that order.

## 24. Effect validation is state-bound

S2 validation consumes the exact prior `StateCommitment`/validated state identity against which targets and quantity compatibility were checked.

A validated effect plan cannot be replayed against an unrelated state merely because IDs happen to exist there.

Conceptually the validated plan binds:

```text
prior_state_commitment
event/profile identities
validated target set
```

## 25. State drift requires revalidation

If the intended prior state changes before S3 transition execution, the effect plan must be revalidated or rejected according to the later transition contract.

This mirrors state-bound authorization discipline without creating real-world authority.

## 26. No hidden randomization

S2 contains no RNG and no adaptive effect generation.

If a campaign samples stochastic shocks, the resulting sampled event set and seed/sample identity are frozen upstream before S2 validation.

## 27. No network/live profile lookup

S2 performs no network, database, Holochain, or live Mycelix lookup.

It consumes already-resolved immutable event/profile/input state.

A changed authoritative profile creates a new bridge/campaign input lineage.

## 28. Error taxonomy

S2 should expose stable typed errors such as:

```rust
DuplicateEventId
EmptyUnmarkedEvent
NoOpMixedWithMutatingEffects
MissingDependencyTarget
MissingServiceTarget
MissingProvisionPathTarget
MissingStockTarget
MissingFailureDomainTarget
QuantityUnitMismatch
QuantityBasisMismatch
InventoryLossExceedsStock
UnsupportedEffectClass
SameTargetTemporalConflict
InvalidTiming
StateCommitmentMismatch
```

Exact enum names may differ. Control logic must not parse display strings.

## 29. Validated plan is opaque

Conceptually:

```rust
pub struct ValidatedEffectPlan { /* private */ }

pub fn validate_effect_plan(
    prior: &ModelState,
    events: Vec<ShockEventInput>,
) -> Result<ValidatedEffectPlan, EffectValidationError>;
```

Callers cannot instantiate a trusted/validated plan by setting public fields.

## 30. Validated plan does not contain successor state

S2 output contains only validated/canonicalized exogenous inputs and their binding to the prior state.

It does not precompute or cache a successor state as a hidden side effect.

That keeps validation and transition independently testable.

## 31. First regression campaign

A later executable S2 should include at least:

1. explicit no-op fixture accepted;
2. empty unmarked event rejected;
3. no-op mixed with mutating effect rejected;
4. duplicate event ID rejected;
5. missing dependency target rejected;
6. missing service target rejected;
7. missing provision path rejected;
8. missing stock rejected;
9. missing failure domain rejected;
10. inventory-loss unit mismatch rejected;
11. inventory-loss basis mismatch rejected;
12. inventory loss above stock rejected;
13. stock physical loss preserves explicit destination;
14. scenario-demand effect binds scenario demand, not adopted floor;
15. same-target same-instant effects conflict;
16. instant-inside-interval same-target effects conflict;
17. overlapping interval same-target effects conflict;
18. adjacent half-open same-target intervals do not conflict;
19. disjoint-target simultaneous effects validate;
20. same-target non-overlapping events validate as ordered history;
21. insertion-order permutation yields identical validated canonical plan;
22. canonical sort does not rescue an otherwise conflicting fixture;
23. unsupported quality/recovery effect class fails visibly;
24. changed prior-state commitment invalidates/requires revalidation of the plan.

## 32. Metamorphic properties

The initial qualification should include properties such as:

```text
permuting raw order of valid disjoint events
-> same validated canonical plan

adding an unrelated disjoint no-op control event
-> same mutating target set

shifting one same-target interval from overlapping to exactly adjacent
-> conflict becomes non-conflict
```

without claiming final-state equivalence before S3 exists.

## 33. Independent conflict oracle

At least the temporal-overlap/direct-target conflict matrix should have an independently specified oracle or fixed expected table separate from the production conflict checker.

Self-agreement is not enough for the core ambiguity firewall.

## 34. Handoff to S3

Only a `ValidatedEffectPlan` may enter the deterministic reducer.

S3 should bind:

```text
prior_state_commitment
validated_effect_plan identity
ruleset/model revision
successor_state_commitment
```

and preserve the prior state unchanged.

S3 must not re-resolve scenario labels or silently widen the effect set.

## 35. Authority boundary

Effect validation does not create:

- emergency authority;
- procurement authority;
- operating authority;
- ecological/right exemption;
- infrastructure-control authority;
- physical actuation.

Even a perfectly validated effect plan remains synthetic/model input.

## 36. Deliberate non-claims

REGEN-042S2 establishes no S0/S0A/S1 executable PASS, no successor-state theorem, no real disaster event, no hazard probability, no model adequacy, no service continuity, no emergency policy, no recommendation correctness, no governance authority, and no physical-action permission.

Its proposition is narrow:

> before a compound-shock model mutates state, exogenous campaign inputs should be reduced to a small typed effect grammar, checked against the exact prior state, and rejected whenever target/timing semantics are ambiguous rather than relying on container order or implicit severity logic.
