# Symthaea RCA Observer Bridge

This crate provides the opt-in construction lane for cognitive executions that may later be observed by the Recursive Cognitive Architecture shadow plane.

It deliberately leaves ordinary `CognitiveLoopService::new(...)` unchanged.

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
```

The wrapper owns both the cognitive service and `IssuedCognitiveExecutionLineageV1`. Neither can be extracted through the RCA-002.0b API.

## Config identity

`CognitiveLoopConfig` contains unordered structures, including `HashMap` state, so raw `serde_json::to_vec(&config)` is not treated as canonical identity.

RCA-002.0b instead defines `rca-cognitive-loop-config-tree-v1`:

1. serialize the complete config to a `serde_json::Value` tree;
2. recursively encode explicit type tags and little-endian lengths;
3. sort object keys by their raw UTF-8 bytes;
4. encode strings as raw UTF-8 rather than JSON text escaping;
5. bind the exact projection algorithm itself with a domain-separated BLAKE3 contract digest.

Changing those projection semantics requires a new profile identity.

## Cycle identity

A cognitive-loop cycle counter is meaningful only within an execution. The wrapper therefore owns a separate monotonic `u64` counter that is never reset with the service's internal statistics.

Every returned `RcaCompletedCycleV1` binds:

```text
execution_lineage_digest
        +
wrapper_cycle_index
        +
CycleResult
```

Its fields are private and the type is not serializable in this tranche. The later shadow adapter will turn it into the already-isolated `FrozenCycleObservationV1`.

## No escape hatch

RCA-002.0b deliberately provides no:

- `Deref` / `DerefMut` to `CognitiveLoopService`;
- `service()` / `service_mut()` accessor;
- `into_inner()` / `into_parts()` escape;
- accessor for `IssuedCognitiveExecutionLineageV1`;
- shadow feedback path.

The only behavioral delegation added here is `cycle(&mut self, input)`.

## Authority limits

An externally supplied source-generation digest is **identity**, not proof that the source is qualified. This bridge does not perform build/source qualification, epistemic admission, GWT admission, action authorization, or self-improvement promotion.

RCA-002.1 should add only the one-way projection:

```text
RcaCompletedCycleV1
        ↓
FrozenCycleObservationV1
```

No shadow result should be consumed by the cognitive loop.
