# Temporal State API V1

Status: **contract draft / no runtime behavior change**

Parent program: #3604  
Root abstraction: #3586  
Current correctness findings: #3580 #3583 #3584 #3585 #3587 #3588

## Purpose

This document freezes the first version of Symthaea's temporal-state lifecycle contract before implementation changes land.

The central rule is:

```text
ProjectedStateObservation
  != ExactEvolutionSnapshot
  != DynamicReset
  != Perturbation
  != CheckpointState
  != TrainingContinuationState
```

A backend may support some of these capabilities and not others. A caller may not infer one from another.

This contract is intentionally backend-neutral. It does not select CfC, HdcLtc, HierarchicalCfC, or any future temporal substrate as preferred.

## Why V1 is required

The current `TemporalNetwork` wrapper presents heterogeneous backend semantics through similar-looking operations.

### Classic CfC

`CfCNetwork::read_state()` returns only the final cell's hidden state while `inject(state)` writes that vector into every cell. Since the default network is multi-layer, this pair is not an exact restore of the default cognitive-loop subject.

### HdcLtc

The bridge's `read_state()` returns a small cached output projection. `inject(state)` resets the HDC/LTC network and seeds only that cache. The engine does have a strong `NetworkStateSnapshot`, and `predict_forward()` is internally state-pure, but destructive-operation wrapper backup does not currently include every bridge-visible field such as `current_output`.

### HierarchicalCfC

The wrapper exposes a slow-layer projected state while `inject()` resets the hierarchy. Exact hierarchy snapshot/restore is not available yet.

Therefore V1 separates observation, reset, exact restore, perturbation, and training state at the type/API level.

---

## 1. Capability identity

Capabilities are profile-bound, not backend-name booleans.

Conceptually:

```rust
pub enum TemporalCapabilityKind {
    ProjectedObservation,
    DynamicReset,
    ExactEvolutionSnapshot,
    PurePrediction,
    RecurrentStateMask,
    HistoricalStartTraining,
    OnlineAdaptation,
}

pub struct TemporalCapabilityId {
    pub backend: TemporalBackendId,
    pub kind: TemporalCapabilityKind,
    pub profile: CapabilityProfileId,
    pub version: u32,
}
```

Exact Rust names may change, but qualification receipts MUST bind equivalent identity.

A capability exists for evidence purposes only after its named profile passes conformance tests. Merely implementing a method does not qualify the capability.

---

## 2. Projected state observation

A projected observation is a read-only observable view suitable for diagnostics or downstream metrics whose scientific meaning explicitly matches the projection.

Conceptual API:

```rust
pub struct ProjectedTemporalState {
    pub profile: ProjectionProfileId,
    pub values: Array1<f32>,
}

pub fn projected_state(&self) -> Result<ProjectedTemporalState, TemporalStateError>;
```

### Required properties

- observational / non-mutating;
- projection identity is explicit;
- no implication of injectability or exact restoration;
- dimensions and semantics are documented per backend/profile;
- a projection change requires a new profile/version;
- a metric must not relabel a projection as the full recurrent state.

### Initial projection registry

```text
Classic CfC
  current legacy projection: final cell hidden state

HdcLtc
  current legacy projection: bridge current_output cache

HierarchicalCfC
  current legacy projection: selected slowest hierarchy/cell projection
```

These are observations, not exact snapshots.

---

## 3. Dynamic reset

Reset is an explicit state transition, not an injection side effect.

Conceptual API:

```rust
pub enum TemporalResetProfile {
    DynamicStateV1,
    InitialCheckpointStateV1,
}

pub fn reset_dynamic_state(
    &mut self,
    profile: TemporalResetProfile,
) -> Result<ResetReceipt, TemporalStateError>;
```

### Required properties

- does not accept an arbitrary projected vector;
- backend-specific reset semantics are documented;
- hidden-state reset is distinguishable from exact checkpoint-state restore;
- intentional reset callers use this API directly;
- zero-vector `inject()` MUST NOT remain the generic mechanism for requesting reset in evidence-sensitive code.

Legacy `inject()` may remain temporarily for compatibility, but V1 callers MUST NOT treat it as an exact lifecycle primitive unless a backend/profile explicitly qualifies projected-state injection.

---

## 4. Exact evolution snapshots

An exact evolution snapshot captures every mutable value required to reproduce the future inference trajectory for one named subject/checkpoint/execution profile.

Conceptual wrapper:

```rust
pub enum TemporalEvolutionSnapshot {
    CfC(CfCInferenceSnapshotV1),
    HdcLtc(HdcLtcBridgeEvolutionSnapshotV1),
    HierarchicalCfC(HierarchicalCfCInferenceSnapshotV1),
}
```

Capture/restore:

```rust
pub fn snapshot_evolution_state(
    &self,
) -> Result<TemporalEvolutionSnapshot, TemporalStateError>;

pub fn restore_evolution_state(
    &mut self,
    snapshot: &TemporalEvolutionSnapshot,
) -> Result<(), TemporalStateError>;
```

### Identity guard

Every exact snapshot profile MUST bind or validate enough identity to reject restoration into an incompatible subject.

At minimum audit:

- snapshot schema/profile version;
- backend/profile;
- exact shape census;
- realized configuration identity;
- initialization/subject identity where relevant;
- checkpoint/parameter identity where relevant;
- numerical execution profile when exact replay depends on it.

A mismatch MUST return a typed error. Silent no-op restore is forbidden in evidence-sensitive V1 paths.

### Atomic restore

Malformed or mismatched snapshots MUST NOT partially mutate live state.

Validation precedes mutation where possible. If implementation requires staged mutation, rollback or an equivalent atomicity theorem is required before qualification.

---

## 5. Snapshot contents are semantic, not reflexively comprehensive

Do not serialize every counter or cache merely because it exists.

Classify mutable values:

```text
InferenceSemantic
MeasurementOnly
TrainingContinuationOnly
CheckpointParameter
```

An inference snapshot contains all and only the state required for its claimed replay theorem, plus identity metadata.

Measurement-only counters may have a separate telemetry snapshot. Training optimizer state belongs in a training-continuation profile unless it changes inference behavior directly.

---

## 6. Classic CfC exact snapshot V1 requirements

The current last-cell vector is insufficient for a default two-layer network.

`CfCInferenceSnapshotV1` MUST source its state from the engine, not reconstruct it from a projected observation.

Minimum audit/candidate contents:

```text
ordered hidden state for every cell/layer
actual mutable tau arrays if inference can change them
semantic network/cell counters if future outputs depend on them
online-adaptation state when that profile is enabled
subject/config/checkpoint binding metadata
```

Optimizer/Adam state is NOT automatically part of inference V1. It belongs only if needed for the named capability.

### Required negative control

Construct a multi-layer CfC with deliberately distinct layer states and prove:

```text
legacy read_state -> inject(read_state)
```

cannot reproduce the original multilayer state.

The conformance suite MUST retain this regression fixture.

---

## 7. HdcLtc exact snapshot V1 requirements

The existing `NetworkStateSnapshot` remains the engine-level reference and SHOULD be reused rather than duplicated.

Wrapper-level exact lifecycle state requires an owning wrapper snapshot such as conceptually:

```rust
pub struct HdcLtcBridgeEvolutionSnapshotV1 {
    pub network: NetworkStateSnapshot,
    pub current_output: Vec<f32>,
    // additional fields only if audit proves semantic necessity
}
```

Audit before freezing inclusion of:

- `total_steps`;
- state diversity cache;
- adaptive-dimension counters/state;
- other wrapper fields.

Do not serialize unrelated telemetry by reflex.

`predict_forward()` is a separate internally pure capability and does not require external wrapper save/restore merely because destructive replay does.

---

## 8. HierarchicalCfC exact snapshot V1 requirements

This profile is blocked on deterministic realized construction and exact lifecycle work tracked by #3577/#3578/#3579.

Minimum semantic audit includes:

```text
ordered hidden state for every hierarchy layer/cell
actual current tau arrays
hierarchy total_steps when it gates top-down behavior
recent_slow_outputs / contextual state
future mutable hierarchy inference state
subject/config/checkpoint binding metadata
```

Nominal configured hierarchy time constants are NOT a substitute for actual realized tau state.

The historical cumulative tau algorithm remains unchanged by lifecycle work unless #3581 independently qualifies a different execution profile.

---

## 9. Typed perturbation

Causal/state perturbations MUST NOT be implemented by mutating an arbitrary projected observation and calling a generic inject method.

Conceptual API:

```rust
pub enum TemporalPerturbationSpec {
    RecurrentMask(RecurrentMaskSpec),
    Silence(SilenceSpec),
    Clamp(ClampSpec),
    Impulse(ImpulseSpec),
    Noise(NoiseSpec),
}

pub enum PerturbationOutcome {
    Applied(PerturbationReceipt),
    Unsupported,
    Rejected(TemporalStateError),
}
```

### Truthfulness theorem

If the receipt says `Applied`, the named perturbation occurred on exactly the declared state surface.

Unsupported perturbations MUST NOT degrade into reset, projection replication, or another intervention.

This is required for NEUROARCH effectome work.

---

## 10. Prediction purity

Purity is profile-specific.

Conceptual capability:

```rust
pub fn prediction_capability(&self) -> PredictionCapability;
```

A pure prediction profile MUST satisfy paired replay:

```text
capture S
A = future(I, D) from S

restore S
run prediction probes P
restore S if the prediction profile uses external snapshots
B = future(I, D)

A == B
```

Equality includes every semantic field belonging to the profile, not only returned output.

### Current descriptive status

```text
HdcLtc internally pure predict_forward: strong existing candidate
Classic CfC predict_forward: destructive today
HierarchicalCfC predict_forward: destructive today
```

Do not promote the latter two until exact lifecycle conformance passes.

---

## 11. Historical-start training

Training start state is distinct from the live online inference trajectory.

A historical-start profile MUST define:

```text
online inference evolution state
training-start evolution state
learned/checkpoint parameter state
training-continuation optimizer state
```

These are not interchangeable.

For sequence pair `(enc_{t-1} -> enc_t)`, a claimed historical-start profile MUST actually begin from the committed historical evolution state required by the experiment.

If a backend lacks that capability, the operation is `Unsupported` for that profile; silently training from live state while emitting a historical-start claim is forbidden.

Parameter updates may persist while online inference evolution is restored, but that interaction with the new checkpoint MUST be explicitly defined and tested.

---

## 12. Capability reporting

The wrapper SHOULD expose a stable capability description rather than scattering backend-name conditionals through callers.

Conceptual structure:

```rust
pub struct TemporalCapabilities {
    pub projected_observation: Option<ProjectionProfileId>,
    pub reset: Option<ResetProfileId>,
    pub exact_snapshot: Option<SnapshotProfileId>,
    pub pure_prediction: Option<PredictionProfileId>,
    pub recurrent_mask: Option<PerturbationProfileId>,
    pub historical_start_training: Option<TrainingProfileId>,
}
```

This describes supported code paths. Qualification status belongs in evidence receipts, not hard-coded marketing-style flags.

---

## 13. V1 conformance laws

### 13.1 Observation purity

```text
S0 = exact semantic state
observe projected state
S1 = exact semantic state
assert S0 == S1
```

### 13.2 Exact restore

```text
S = capture()
mutate/evolve
restore(S)
assert exact_semantic_state == S
assert future trajectory matches untouched twin
```

### 13.3 Repeated restore

Restoring the same immutable snapshot repeatedly produces the same future trajectory.

### 13.4 Snapshot ownership

Snapshot data cannot alias mutable live state such that later evolution changes the stored snapshot.

### 13.5 Mismatch rejection

Restoring the wrong backend/profile/shape/subject/checkpoint fails explicitly and leaves live state unchanged.

### 13.6 Reset distinction

Reset is tested independently from restore. Passing reset tests does not qualify exact restore.

### 13.7 Perturbation truthfulness

Applied perturbation telemetry is verified against exact state before/after. Unsupported operations leave semantic state unchanged.

### 13.8 Prediction purity

Prediction probes cannot alter subsequent semantic trajectory for profiles advertised as pure.

### 13.9 Measurement-only state

Resetting or excluding a field classified as measurement-only MUST NOT alter the semantic future trajectory under the qualified inference profile.

### 13.10 Training separation

A detached/historical training profile may change intended parameters while preserving/restoring online inference evolution according to its declared semantics.

---

## 14. Evidence receipt V1

A lifecycle qualification receipt SHOULD bind at least:

```text
schema/profile version
backend identity
capability profile
exact code/head
subject/config identity
checkpoint identity where relevant
snapshot schema/profile
numerical execution profile
input sequence commitment
dt/timestamp sequence commitment
probe/mutation/perturbation commitment
pre-state commitment
post-restore state commitment
future-trace commitment
resource accounting
result status
```

Allowed result vocabulary:

```text
Qualified
NotQualified
Inconclusive
InfrastructureIndeterminate
```

Queued/no-step/runnerless execution is not qualification evidence.

---

## 15. Resource accounting

Snapshot-based correctness has cost and V1 makes it visible.

Record separately where applicable:

- snapshot retained bytes;
- capture copied bytes/elements;
- restore copied bytes/elements;
- capture operations/time;
- restore operations/time;
- perturbation operations/time;
- prediction forward work;
- instrumentation/trace overhead.

Do not hide lifecycle instrumentation inside candidate inference cost in architecture comparisons.

---

## 16. Migration order

### Phase A — containment
Refuse/skip known-invalid restore, training, replay, or perturbation paths with explicit telemetry rather than silently approximating them.

### Phase B — introduce explicit APIs
Add projected observation, reset, exact snapshot/restore, capability reporting and typed perturbation surfaces.

### Phase C — migrate intentional reset callers
Replace zero-vector inject reset patterns.

### Phase D — qualify exact snapshots
Classic CfC, HdcLtc wrapper, then HierarchicalCfC profile(s).

### Phase E — migrate prediction/replay/training
Re-enable operations only under capabilities they actually require.

### Phase F — deprecate ambiguous inject
Narrow/remove generic projected-state injection once callers no longer depend on mixed semantics.

---

## 17. NEUROARCH boundary

A future `LiquidIsland` may advertise only capabilities qualified for its exact backend/profile.

Examples:

```text
DeterministicReplay
ExactSnapshotRestore
PurePrediction(profile)
RealizedTauObservability
TypedPerturbation(profile)
HistoricalStartTraining(profile)
```

No capability is inherited merely because another backend implements it.

Effectome experiments MUST NOT claim paired same-start causality unless exact restore for the subject/profile has qualified.

---

## 18. Evidence-impact rule

Discovering a lifecycle defect does not erase all historical output.

Classify each affected claim:

```text
Unaffected
RawRetainedInterpretationLimited
RerunRequired
InvalidInterventionLabel
```

The classification depends on whether the claim materially relied on the broken lifecycle assumption.

---

## 19. V1 non-goals

This contract does not:

- rank temporal architectures;
- select a production backend;
- redesign HierarchicalCfC tau modulation;
- establish consciousness;
- establish biological realism;
- require every backend to support every capability;
- claim exact training continuation from an inference-only snapshot;
- retroactively alter historical raw results.

## Qualification boundary

Landing this document alone establishes only an explicit lifecycle contract. Runtime correctness is established only by later implementation plus executable conformance evidence on exact code/config/profile lineages.
