# Symthaea RCA Runtime Evidence Candidate Bridge

This crate converts a **validated detached RCA observation** into narrowly typed **candidate evidence**.

It exists because an observation of Symthaea's own instrumented runtime is not honestly described by the existing canonical evidence authorities:

- it is not an external `EmpiricalObservation`;
- it is not an `InternalInference`;
- it is not an `InternalSimulation`;
- and observation alone does not grant evidence-use admission.

The bridge therefore stops before `CognitiveEvidenceRefV1`.

## Boundary

```text
ValidatedFrozenCycleObservationV1
        ↓
shared ObservationEventRoot
        ↓
closed lossless field selector
        ↓
InstrumentedRuntimeClaimV1
        ↓
InstrumentedRuntimeEvidenceCandidateV1
        ↓
provenance lineage bookkeeping only

        ✕ no canonical evidence authority
        ✕ no currentness self-declaration
        ✕ no belief/workspace admission
        ✕ no action authority
        ✕ no self-improvement promotion
```

## Observation identity versus field-candidate identity

A frozen cycle is one observation event. It receives a shared domain-separated `observation_root_id` derived from:

```text
runtime-observation-root profile contract
        +
shadow-observer contract digest
        +
exact observation commitment
```

Every field candidate extracted from that exact observation shares this same provenance root.

The selected field claim receives its own `claim_digest`, and the candidate identity binds:

```text
candidate_id = H(
    candidate-profile contract,
    shared observation-root id,
    selected claim digest
)
```

Therefore:

```text
same observation + different selected fields
        = different candidate claims
        = same provenance root
        = NOT independent observations
```

while:

```text
different frozen observation events
        = different observation roots
```

This prevents multiple fields from one cycle from being miscounted as independent corroboration.

## Lineage fragment

`lineage_fragment()` yields two generic governance nodes:

```text
ObservationEventRoot
    derivation = RootObservation
    parents = []
        ↓
FieldCandidate
    derivation = Transformation
    parents = [ObservationEventRoot]
```

The candidate itself is deliberately **not** a `RootObservation`.

Two candidates from the same frozen observation therefore resolve to `SameRoot` under the RCA lineage/independence policy. Candidates from distinct observation events can be independent when their root sets are disjoint.

## Closed claim surface

V1 can expose only:

- cycle duration;
- prediction-error ppm;
- exact peak-attention f32 bits;
- learning-occurrence flag;
- detected primitive count;
- output commitment;
- thought commitment;
- metadata commitment;
- optional language-output commitment plus source identity.

A caller cannot supply an arbitrary claim payload or arbitrary claim digest.

Changing the allowed claim set or projection semantics requires a new profile/contract.

## Persistence

Persisted candidates contain the validated detached observation. Deserialization recomputes:

1. observer profile and contract;
2. observation commitment;
3. shared observation-root identity;
4. selected claim;
5. claim digest;
6. candidate identity.

Any mismatch fails closed.

## Currentness

This crate deliberately does not construct or assess `EvidenceCurrentnessV1`.

The historical fact that a particular cycle had a particular value and the proposition that the same state is relevant **now** are different claims. Currentness/relevance belongs to a later proposition/use-context boundary.

## Dependency direction

The crate depends on:

```text
symthaea-rca-shadow
symthaea-epistemic-governance
```

It does not depend on the live RCA observer wrapper or root cognitive loop. It also does not directly depend on canonical cognitive evidence types.
