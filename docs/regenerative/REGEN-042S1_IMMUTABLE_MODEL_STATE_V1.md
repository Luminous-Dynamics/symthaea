# REGEN-042S1 — Immutable Model State and Commitment Contract v1

Status: preregistration only

Parent implementation profile: REGEN-042S / #3817

Executable predecessor: REGEN-042S0 / #3818

Mycelix authoritative semantics: Luminous-Dynamics/mycelix#1431, #1519, #1523

## 1. Purpose

Freeze the exact state boundary that must exist before Symthaea is allowed to apply any compound-shock effect.

REGEN-042S0 establishes canonical identity, exact quantity, exact ratio, and campaign-time primitives.

REGEN-042S1 adds the first immutable validated model state and a deterministic commitment over that state.

Core theorem:

```text
canonical primitives
+ immutable state snapshot
+ referential closure
+ conservation/coherence invariants
+ canonical ordered encoding
+ versioned cryptographic commitment
= replayable pre-transition model state
```

not:

```text
model state
= observed real-world truth
= adopted service policy
= authority
= emergency decision
= physical control
```

No shock reducer, dependency propagation, recovery scheduler, recommendation, or actuator surface is introduced by this contract.

## 2. Ownership boundary remains unchanged

Mycelix and authoritative domain systems remain owners of adopted service/profile/evidence/rights/ecology/quality/authority state.

Symthaea receives an immutable assessment snapshot and represents it for deterministic analysis.

```text
Mycelix immutable revision refs
        |
        v
validated REGEN-042S model state
        |
        v
future deterministic transitions
```

Symthaea must not silently repair, normalize, broaden, reinterpret, or replace authoritative identities during model-state construction.

## 3. State is validated, not merely constructed

The executable API should prevent arbitrary callers from manufacturing a semantically valid-looking state with dangling references or inconsistent accounting.

Conceptually:

```rust
pub struct ModelState { /* private fields */ }

impl ModelState {
    pub fn validate(input: ModelStateInput) -> Result<Self, StateValidationError>;
}
```

`ModelStateInput` may be public transport/construction material.

`ModelState` is the validated immutable state accepted by later transition stages.

No public field mutation belongs on `ModelState`.

## 4. Canonical ordered collections

Scientific commitments must not depend on hash-map insertion order or host behavior.

The first implementation should use ordered collections such as:

```rust
BTreeMap<ServiceId, ServiceState>
BTreeMap<DependencyId, DependencyState>
BTreeMap<StockId, StockState>
BTreeMap<FailureDomainId, FailureDomainState>
BTreeMap<ProvisionPathId, ProvisionPathState>
BTreeSet<DependencyId>
BTreeSet<FailureDomainId>
```

or an equivalently explicit canonical sort before commitment encoding.

A builder must reject duplicate input identities rather than allowing last-write-wins replacement before the `BTreeMap` exists.

## 5. Model identity

Every state binds at minimum:

```text
campaign_revision
model_revision
evidence_snapshot
service_profile_revision
dependency_graph_revision
current_tick
```

Changing any of these creates a different committed state even if all numeric values are identical.

The kernel does not treat equal payload values under different authoritative revisions as the same scientific subject.

## 6. Service state

A first executable representation may be:

```rust
pub enum ServiceAvailability {
    Satisfied,
    Degraded,
    Failed,
    Unresolved,
}

pub struct ServiceState {
    pub availability: ServiceAvailability,
    pub delivered: Resolution<CanonicalQuantity>,
    pub required_floor: Resolution<CanonicalQuantity>,
    pub scope: ScopeId,
    pub active_dependencies: BTreeSet<DependencyId>,
}
```

The `ServiceId` is the map key and is not redundantly stored inside the value.

Where both delivered quantity and required floor are known, unit and basis must be compatible.

The kernel must not invent a missing service floor.

## 7. Explicit resolution state

`Option<T>` is insufficient for scientific/authority-bearing unknowns when absence could mean materially different things.

The initial state contract should distinguish at least:

```rust
pub enum Resolution<T> {
    Known(T),
    Unresolved(UnresolvedId),
    NotApplicable,
}
```

A missing mandatory field at the bridge boundary is a validation error, not `Unresolved`.

`Unresolved` is a represented state with provenance/reason identity.

`NotApplicable` is not zero and not unresolved.

Later wire schemas may distinguish additional upstream states, but must never map unknown/unresolved into success by default.

## 8. Dependency state

Conceptually:

```rust
pub enum DependencyAvailability {
    Available,
    Degraded,
    Unavailable,
    Unresolved,
}

pub struct DependencyState {
    pub availability: DependencyAvailability,
    pub nominal_capacity: Resolution<NonNegativeQuantity>,
    pub usable_capacity: Resolution<NonNegativeQuantity>,
    pub committed_capacity: Resolution<NonNegativeQuantity>,
    pub failure_domains: BTreeSet<FailureDomainId>,
}
```

When all three capacities are `Known`, v1 requires compatible unit/basis and:

```text
committed_capacity <= usable_capacity <= nominal_capacity
```

A future profile that intentionally uses different capacity semantics must use another explicit model revision rather than weakening this invariant silently.

`Unavailable` does not require the stored nominal capacity to become zero; availability and nominal design capacity remain different facts.

## 9. Stock state

Conceptually:

```rust
pub struct StockState {
    pub commodity: CommodityId,
    pub total: NonNegativeQuantity,
    pub protected: NonNegativeQuantity,
    pub committed: NonNegativeQuantity,
}
```

All three quantities must share exact unit and basis.

The v1 accounting theorem is:

```text
protected + committed <= total
```

Protected and committed are disjoint reservation classes in this model revision.

If an authoritative profile requires overlapping reservation classes, it needs a separately specified accounting model instead of double counting inside v1.

Derived freely usable quantity may be computed as:

```text
total - protected - committed
```

with checked exact arithmetic only.

## 10. Failure-domain state

Failure-domain membership belongs on dependency state, while domain state records current exogenous/model availability.

Conceptually:

```rust
pub enum FailureDomainAvailability {
    Available,
    Unavailable,
    Unresolved,
}

pub struct FailureDomainState {
    pub availability: FailureDomainAvailability,
}
```

A failure domain appearing in any dependency membership set must exist in the state-level failure-domain table.

No dependency consequence is propagated in S1; only referential closure is validated.

## 11. Provision-path state

The state must reserve a canonical place for later lead-time and substitution effects without claiming a path is qualified merely because it exists.

Conceptually:

```rust
pub enum ProvisionAvailability {
    Available,
    Unavailable,
    Unresolved,
}

pub struct ProvisionPathState {
    pub availability: ProvisionAvailability,
    pub lead_time_ticks: Resolution<u64>,
    pub dependencies: BTreeSet<DependencyId>,
}
```

Every referenced dependency must exist.

Qualification/authority of the path remains upstream and is represented by the adopted profile revisions bound to the assessment envelope, not inferred by Symthaea.

## 12. Referential closure

Validation rejects at least:

- a service referencing a missing dependency;
- a dependency referencing a missing failure domain;
- a provision path referencing a missing dependency;
- duplicate service/dependency/stock/failure-domain/provision-path identities in the input builder;
- an unresolved reference with no stable unresolved identity;
- empty mandatory service portfolio when the campaign requires at least one service.

The kernel must not auto-create placeholder nodes to make a graph close.

## 13. No hidden normalization

Validation may establish structural invariants but may not silently change authoritative meaning.

Examples:

```text
unknown capacity -> not zero
unavailable dependency -> nominal capacity not erased
unresolved service floor -> not zero
missing reference -> not auto-created
negative stock -> rejected, not clamped
protected+committed overflow -> rejected, not truncated
```

## 14. Immutable successor discipline

Future transition stages must produce a new `ModelState`; they do not mutate the prior state in place.

Conceptually:

```text
prior_state
+ accepted transition
-> successor_state
```

with both prior and successor commitments retained in the transition receipt.

This preserves replay and makes accidental partial mutation much harder to hide.

## 15. Canonical state encoding v1

A commitment must not hash Rust `Debug`, host memory layout, unordered iteration, or ad-hoc JSON formatting.

S1 should freeze a small canonical binary encoding with these rules:

1. domain-separation prefix exactly `symthaea.regen042s.model-state.v1` followed by one zero byte;
2. integers encoded big-endian at their declared fixed width;
3. signed `i128` quantity mantissas encoded as two's-complement big-endian 16 bytes;
4. identifier strings encoded as `u32` byte length followed by their exact ASCII bytes;
5. enum variants encoded by explicit frozen `u8` tags;
6. ordered maps/sets emitted in canonical key order;
7. collection lengths encoded as big-endian `u32` and overflow rejected;
8. `Resolution<T>` variants use explicit tags and never omit payload discriminators;
9. quantities encode canonical mantissa, canonical scale, exact unit ID, and exact basis ID;
10. no implementation-language metadata is included.

Changing any encoding rule requires a new commitment/serialization version.

## 16. Commitment algorithm v1

The first commitment algorithm should be:

```text
SHA-256(canonical_state_bytes_v1)
```

represented as exactly 32 bytes and rendered externally as lowercase hexadecimal only at display/wire boundaries.

The commitment type should remain opaque:

```rust
pub struct StateCommitment([u8; 32]);
```

No caller may construct a trusted commitment from an arbitrary display string without validation.

The implementation should use a maintained cryptographic library rather than custom SHA-256 code.

This commitment proves byte identity of the canonical model state; it does not prove truth, authority, freshness, authenticity, or real-world correspondence.

## 17. Commitment binds revisions and tick

The state commitment includes campaign/model/profile/evidence revision identities and the current campaign tick.

Therefore:

```text
same numeric state at tick 5
!= same committed state at tick 6
```

and:

```text
same values under evidence snapshot A
!= same committed state under snapshot B
```

This prevents lineage metadata from being carried only beside, rather than inside, the scientific state identity.

## 18. Insertion-order metamorphism

Given two semantically identical validated inputs whose raw collections arrived in different orders:

```text
commitment(A) == commitment(B)
```

must hold.

The test is mandatory.

## 19. Semantic-change sensitivity

At least the following one-field changes must alter the commitment:

- campaign revision;
- model revision;
- evidence snapshot;
- current tick;
- service availability;
- delivered quantity;
- dependency availability;
- capacity;
- stock quantity/protected/committed state;
- failure-domain membership/state;
- provision lead time/availability;
- unresolved identity.

This is a commitment-correctness property, not a cryptographic collision proof.

## 20. State validation errors remain typed

S1 should expose stable typed failures such as:

```rust
pub enum StateValidationError {
    DuplicateIdentity { ... },
    MissingDependencyReference { ... },
    MissingFailureDomainReference { ... },
    MissingProvisionDependencyReference { ... },
    QuantityUnitMismatch { ... },
    QuantityBasisMismatch { ... },
    CapacityOrderingViolation { ... },
    StockReservationExceedsTotal { ... },
    ArithmeticOverflow,
    EmptyRequiredServiceSet,
    CollectionTooLarge,
    UnsupportedCommitmentVersion,
}
```

Display text is diagnostic only; program behavior keys on variants.

## 21. S1 must not smuggle S2 semantics

No `ShockEffect`, `ShockEvent`, same-target conflict reducer, outage propagation, demand scaling, inventory-loss application, or lead-time extension is implemented in S1.

The transition surface remains closed until state validation and commitment identity earn their own execution evidence.

## 22. First executable regression campaign

The S1 implementation should include at least these independent cases:

1. one minimal valid state;
2. duplicate service ID rejected before map materialization;
3. dangling service dependency rejected;
4. dangling failure-domain reference rejected;
5. dangling provision dependency rejected;
6. known service delivered/floor unit mismatch rejected;
7. known service delivered/floor basis mismatch rejected;
8. committed dependency capacity above usable rejected;
9. usable dependency capacity above nominal rejected;
10. capacity unit/basis mismatch rejected;
11. protected stock above total rejected;
12. committed + protected stock above total rejected;
13. stock unit mismatch rejected;
14. stock basis mismatch rejected;
15. arithmetic overflow rejected rather than wrapped;
16. unresolved service floor preserved as unresolved;
17. unavailable dependency retains nominal-capacity identity;
18. insertion-order permutation yields identical commitment;
19. one semantic field mutation changes commitment;
20. revision-only mutation changes commitment;
21. tick-only mutation changes commitment;
22. canonical encode -> commit repeated twice yields identical bytes/digest;
23. invalid state cannot obtain a `StateCommitment` through the public API;
24. prior committed state remains byte/semantic-identical after constructing an independent successor candidate.

## 23. Qualification boundary

S1 implementation is blocked on an acceptable S0 exact-head execution result.

A later S1 executable PR should bind:

- exact S0 predecessor identity;
- exact model-state source/test scope;
- exact dependency additions required for SHA-256, if any;
- formatting;
- targeted crate tests;
- strict Clippy;
- canonical commitment vectors;
- insertion-order metamorphic tests;
- clean postflight checkout;
- no unsafe code;
- no network/database/Holochain/LLM/actuator dependency.

No S0 PASS is inferred by this document.

## 24. Handoff to S2

Only after S1 earns its own exact-subject evidence should S2 add the typed exogenous `ShockEffect` algebra and simultaneous-effect conflict checker.

The intended sequencing is:

```text
S0 canonical primitives
-> S1 immutable validated state + commitment
-> S2 typed effects + conflict analysis
-> S3 deterministic reducer
```

This keeps representation, state validity, effect grammar, and transition semantics independently reviewable.

## 25. Deliberate non-claims

REGEN-042S1 establishes no real service sufficiency, disaster probability, hazard severity, resilience superiority, ecological/right/quality conclusion, emergency policy, procurement priority, recommendation correctness, governance authority, or physical-action permission.

Its proposition is narrow:

> before a resilience simulator can transform state, it should prove that the prior state is immutable, referentially closed, arithmetically coherent, canonically encoded, and bound to an exact versioned cryptographic commitment.
