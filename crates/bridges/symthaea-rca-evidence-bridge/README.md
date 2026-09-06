# Symthaea RCA Runtime Evidence Candidate Bridge

This crate converts a **validated detached RCA observation** into a narrowly typed **candidate evidence** object.

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

## Claim identity versus candidate identity

The selected field claim receives its own domain-separated `claim_digest`.

The candidate identity additionally binds the exact observation context:

```text
candidate_id = H(
    candidate profile contract,
    shadow observer contract,
    exact observation commitment,
    selected claim digest
)
```

Therefore two cycles that report the same prediction-error value still produce distinct candidate evidence when their full detached observations differ.

This is intentional:

```text
same observed value != same evidence event
```

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
3. selected claim;
4. claim digest;
5. candidate identity.

Any mismatch fails closed.

## Currentness

This crate deliberately does not construct or assess `EvidenceCurrentnessV1`.

The historical fact that a particular cycle had a particular value and the proposition that the same state is relevant **now** are different claims. Currentness belongs to a later proposition/use-context boundary.

## Dependency direction

The crate depends on:

```text
symthaea-rca-shadow
symthaea-epistemic-governance
```

It does not depend on the live RCA observer wrapper or root cognitive loop. It also does not directly depend on canonical cognitive evidence types.
