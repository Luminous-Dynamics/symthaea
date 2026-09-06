# Symthaea RCA Observer Bridge

This crate provides the opt-in construction and one-way observation lane for Recursive Cognitive Architecture v1.

Ordinary `CognitiveLoopService::new(...)` remains unchanged and carries no RCA execution-lineage claim.

## Boundary

```text
ordinary CognitiveLoopService::new(config)
        ↓
ordinary cognitive execution
        ↓
NO RCA execution-lineage claim


source-generation identity
        +
validated config
        +
canonical config projection
        +
live OS-entropy lineage issuance
        ↓
RcaObservableCognitiveLoopV1
        ↓
RcaCompletedCycleV1
        ↓ immutable one-way projection
ValidatedFrozenCycleObservationV1
        ↓
shadow analysis / receipts
        ✕
        └── no return path into cognition
```

The wrapper owns both the cognitive service and `IssuedCognitiveExecutionLineageV1`. Neither can be extracted through this API.

## Config identity

`CognitiveLoopConfig` contains unordered structures, including `HashMap` state, so raw `serde_json::to_vec(&config)` is not treated as canonical identity.

`rca-cognitive-loop-config-tree-v1` instead:

1. serializes the complete config to a value tree;
2. recursively encodes explicit type tags and little-endian lengths;
3. sorts object keys by raw UTF-8 bytes;
4. encodes strings as raw UTF-8 rather than JSON text escaping;
5. binds the projection algorithm itself with a domain-separated BLAKE3 contract digest.

Changing those projection semantics requires a new profile identity.

## Execution and cycle identity

A cognitive-loop cycle counter is meaningful only within one execution. The wrapper therefore owns a separate monotonic `u64` counter that is not tied to internal statistics resets.

Every returned `RcaCompletedCycleV1` privately binds:

```text
source_generation_digest
        +
execution_lineage_digest
        +
wrapper_cycle_index
        +
CycleResult
```

The completed-cycle type is non-serializable. Its provenance cannot be manufactured with a public struct literal.

## RCA-002.1 shadow projection

`adapt_completed_cycle_v1(&RcaCompletedCycleV1)` is an immutable, one-way function. It returns only a validated detached observation. It does not invoke the shadow observer itself and it receives no mutable cognition handle.

The first projection deliberately admits a narrow surface:

```text
source generation
execution lineage
cycle index
cycle duration
prediction error
peak attention weight
learning flag
primitive count
output commitment
thought commitment
selected metadata commitment
optional language commitment + source
```

### Metric semantics

Prediction error is a genuine unit-interval quantity in the predictive encoder, so it is stored as ppm after fail-closed range validation.

Peak attention is **not** a probability. Attention weights may exceed `1.0`, so the adapter preserves the exact `f32::to_bits()` representation. Non-finite or negative attention fails validation; it is never clamped or normalized.

### Explicit metadata v1

The adapter does not hash the whole evolving `CycleMetadata` object. That would allow unrelated telemetry additions to silently widen RCA-v1 evidence semantics.

The v1 metadata commitment contains only:

- surprise/prefrontal veto state;
- reasoning confidence and gate evaluated/blocked state;
- predictive-self safety and behavioral error;
- narrative-GWT veto;
- cycle urgency;
- epistemic quality, conflict count, gate confidence and gate approval;
- metacognitive anomaly and safety-block state;
- feedback conflict ratio and cross-module agreement;
- prediction coherence;
- startup-suppression and self-model accuracy;
- predictive-budget gating.

Each field is committed by explicit label, length, and exact primitive bits/bytes. Adding another metadata field requires changing the declared adapter contract/profile.

### Explicitly outside v1

The first projection does **not** admit:

- `training_loss`;
- predictive-compression diagnostics (`bits_saved_*`, `bits_kappa`);
- episodic-recall diagnostics;
- `wisdom_hv`;
- feature-gated cycle payloads;
- detected primitive identities (only the count is admitted);
- every `CycleMetadata` field not named by the v1 metadata contract.

Tests deliberately prove both positive sensitivity and negative invariance: admitted fields change observation identity; excluded fields do not.

## No escape hatch

This crate deliberately provides no:

- `Deref` / `DerefMut` to `CognitiveLoopService`;
- `service()` / `service_mut()` accessor;
- `into_inner()` / `into_parts()` escape;
- accessor for `IssuedCognitiveExecutionLineageV1`;
- shadow-result feedback path.

The adapter likewise contains no GWT/workspace admission, MetaRouter control, memory mutation, motor/action path, recursive-improvement promotion, clock, filesystem, network, or callback dependency.

## Authority limits

An externally supplied source-generation digest is **identity**, not proof that the source is qualified.

Likewise:

```text
observation provenance
        !=
epistemic admission
        !=
cognitive influence
        !=
action authority
        !=
self-improvement promotion
```

This crate creates the provenance-preserving observation boundary only. Later RCA stages must establish separate authority for every subsequent transition.
