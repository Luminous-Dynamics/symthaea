# Symthaea Physical Agency — Simulation Context Binding RFC v0.1

**Status:** Draft architecture RFC  
**Stack:** PA-09, above PA-08 (#558)  
**Execution posture:** simulation-only; no new physical authority

## Purpose

PA-08 binds physical-agency deliberation and simulation qualification to the same
immutable `WorldSnapshotRef`. That closes substitution between those two layers, but
it does **not** yet establish that the numerical simulation request was actually
constructed from, or evaluated against, that snapshot.

The current `symthaea-sim-bridge::SimulationRequest` carries an id, engineering domain,
solver kind, objective, model parameters, and requested metrics. It has no typed channel
for immutable world, geometry, material, calibration, boundary-condition, or other
context lineage.

PA-09 defines the contract required to close that gap without encoding context identity
inside free-form strings or pretending that a backend's self-report is authenticated
truth.

The target relationship is:

```text
immutable world / digital-twin context
        |
        v
SimulationContextRef[]
        |
        v
SimulationRequest + canonical request lineage
        |
        v
adapter renders solver input
        |
        v
external solver
        |
        v
SimulationEvidence + consumed context refs
        |
        v
SimulationRegistry exact-context checks
        |
        v
RegistryValidatedSimulationEvidence
        |
        v
Physical Agency qualification
```

This is still an epistemic/evidence path. It creates no actuator or execution authority.

## Non-goals

PA-09 does not:

- authenticate an arbitrary backend as honest;
- prove that an external solver is mathematically correct;
- prove that a context producer's digest accurately represents reality;
- create a HAL path or execution permit;
- add acoustic, photonic, plasma, or other physical actuation;
- replace specialist solver/model validation.

A malicious backend can still lie unless separately qualified or sandboxed. PA-09 makes
such claims explicit and mechanically cross-checkable; it does not turn self-attestation
into trust.

## Context vocabulary

The simulation bridge should gain a small solver-neutral context reference type.
Conceptually:

```text
SimulationContextRef {
    schema_version,
    kind,
    context_id,
    digest_algorithm,
    digest,
    frame_id?,
    provenance_ref?
}
```

The exact Rust representation is deferred to the code PR, but the semantics are fixed
here.

### `schema_version`

Version of the context-reference canonicalization contract. Unknown versions fail
closed where exact lineage is required.

### `kind`

A typed context family, initially including at least:

```text
WorldSnapshot
GeometrySnapshot
MaterialDataset
BoundaryConditions
CalibrationSnapshot
EnvironmentSnapshot
InitialState
Custom
```

`Custom` requires a non-empty namespaced `context_id`; it must never mean "accept any
unknown context".

### `context_id`

Stable logical identity within a context kind, distinct from the content digest. Examples
could identify a particular digital twin, geometry asset, calibration set, or initial
state. It is metadata, not authority.

### `digest_algorithm` and `digest`

Content-addressed identity of the immutable context. The implementation must use an
explicit cryptographic digest algorithm already approved for the repository rather than
`DefaultHasher`, debug-string hashes, floating-point encodings, or process-local hashes.

The digest does not certify truth. It certifies equality of the referenced bytes/state
under the producer's canonicalization contract.

### `frame_id`

Optional spatial/reference-frame identity. A `WorldSnapshot` used by Physical Agency must
carry the frame needed to match the transition target. Other context families may omit it
when spatial framing is irrelevant.

### `provenance_ref`

Optional audit reference to the producer record. This is descriptive evidence metadata,
not an authorization token.

## SimulationRequest extension

`SimulationRequest` should gain:

```text
contexts: Vec<SimulationContextRef>
```

with a serde default of the empty vector for backward-compatible deserialization.

Existing callers that do not require context binding therefore remain valid. Physical
Agency, however, will require a `WorldSnapshot` context for snapshot-bound qualification.

Request validation must reject:

- malformed/empty digest metadata;
- duplicate context identities;
- duplicate `(kind, context_id)` entries with different digests;
- invalid or unsupported schema versions when strict context binding is requested;
- non-empty `WorldSnapshot` context whose frame conflicts with the intended physical
  target when the caller supplies that requirement.

The bridge should preserve deterministic ordering for any canonical lineage calculation;
input vector ordering must not create accidental identity differences.

## Canonical request lineage

The bridge needs a deterministic cryptographic lineage digest for the normalized request.
It must bind at least:

```text
schema version
request id
engineering domain
solver kind
model parameters + units + provenance + uncertainty
requested metrics
sorted SimulationContextRef set
```

Whether the human-readable `objective` is included should be decided once and frozen.
If included, semantically irrelevant wording changes alter lineage; if excluded, it must
never carry safety-relevant machine state. PA-09 recommends **excluding `objective` from
safety-relevant lineage** and enforcing that all machine-relevant inputs live in typed
fields.

The lineage digest is not an execution permit. It is a stable identity for the normalized
request presented to an adapter.

## Adapter contract

An adapter receives the complete normalized `SimulationRequest`, including all contexts.
For a context-bound run, its result must declare the exact context refs it claims to have
consumed.

Conceptually `SimulationEvidence` gains:

```text
request_lineage_digest
consumed_contexts: Vec<SimulationContextRef>
```

Existing `input_digest` continues to identify the fully rendered solver input and
`output_digest` the raw solver output parsed by the adapter.

These are deliberately different identities:

```text
request_lineage_digest
    = normalized Symthaea request + typed contexts

input_digest
    = concrete adapter-rendered external-solver input

output_digest
    = concrete external-solver output
```

Conflating them would hide adapter translation errors.

## Registry checks

For `ExecutionMode::ExternalSolver`, `SimulationRegistry::run` should additionally fail
closed when context lineage is in use and any of the following is true:

1. returned request-lineage digest does not equal the bridge-computed request lineage;
2. consumed context set differs from the request context set;
3. a context is missing;
4. an unexpected context is added;
5. a `(kind, context_id)` identity is returned with a different digest;
6. duplicate or malformed context evidence is returned.

Comparison is set/canonical-order based, not raw input-vector order.

`DryRun` may echo contexts for orchestration tests, but such output remains non-engineering
evidence and cannot satisfy Physical Agency simulation qualification.

`Unknown` evidence remains non-evidence for the strict path.

## Honest evidence-tier naming

PA-09 preserves the naming discipline introduced by PA-07.

A successful registry check may establish:

```text
RegistryValidatedSimulationEvidence
```

meaning:

- the result came through the registered backend path;
- normalized request/result identity matched;
- external-evidence structural provenance was complete;
- request lineage matched;
- claimed consumed contexts exactly matched the requested contexts.

It must **not** be renamed to `Attested`, `Trusted`, `Authentic`, or similar merely because
these checks pass.

A stronger future tier would require independent adapter/runtime evidence such as a
qualified wrapper, sandbox measurement, signed execution receipt, reproducible replay,
or provider verification. That is a separate design problem.

## Physical Agency composition

PA-08's `WorldSnapshotRef` remains the physical-agency-facing abstraction. PA-09 should
add a conversion/binding step that creates the corresponding sim-bridge
`SimulationContextRef::WorldSnapshot` without dropping:

```text
frame_id
snapshot digest algorithm
digest
producer/canonicalization identity where available
```

The strict run path should conceptually become:

```text
SelectedCandidate
    |
    | exact WorldSnapshotRef
    v
build_context_bound_request(...)
    |
    v
SimulationRequest {
    contexts: [WorldSnapshot(...), ...]
}
    |
    v
SimulationRegistry::run
    |
    v
RegistryValidatedSimulationEvidence
    |
    | request lineage + exact context set
    v
qualify_selected_simulation_candidate
```

Physical Agency should no longer accept a separate caller-written snapshot digest in the
simulation binding once the context-bound request exists. The snapshot identity should be
reconstructed from the non-serializable selected receipt and the typed request itself,
reducing duplicated mutable claims.

## Required adversarial tests

The implementation PR should include at least the following regressions.

### Missing context

A request contains world snapshot `A`; backend evidence claims no consumed context.
Registry rejects it.

### Substituted context

Request contains snapshot `A`; evidence claims snapshot `B`. Registry rejects it.

### Extra context

Request contains `A`; evidence claims `A + X`. Registry rejects it rather than silently
accepting an undeclared dependency.

### Reordered context

Request/evidence contain identical sets in different vector order. Registry accepts after
canonicalization.

### Duplicate identity

Two contexts use the same `(kind, context_id)` with different digests. Request validation
rejects before adapter dispatch.

### Request-lineage mismatch

Evidence returns a stale or altered request-lineage digest. Registry rejects it.

### Dry-run spoof

A dry-run result contains perfectly matching contexts and digests. Physical Agency still
rejects it as execution/safety evidence.

### Physical Agency cross-snapshot attempt

A `SelectedCandidate` bound to snapshot `A` cannot construct/qualify a strict simulation
request whose world context is snapshot `B`.

### Context omitted during translation

A test adapter receives a context-bearing request but returns external evidence omitting
that context. Registry rejects the adapter result. This is the minimum mechanical proof
that context propagation is not optional.

## Migration strategy

The code change should be additive:

1. introduce context types and canonical validation;
2. add `contexts` to `SimulationRequest` with serde default;
3. add context/request-lineage fields to `SimulationEvidence` with conservative defaults;
4. preserve legacy behavior for requests with no contexts;
5. activate strict exact-context checks when contexts are present;
6. add Physical Agency snapshot-to-context conversion;
7. update PHYSIS to use only the context-bound strict path;
8. do not update real physical-domain adapters until the context path is hosted-green.

This prevents a repository-wide flag day while ensuring Physical Agency cannot silently
fall back to an unbound solver path.

## Phase exit gate

PA-09 is complete only when hosted tests establish this exact chain:

```text
SelectedCandidate(snapshot A)
        |
        v
SimulationRequest(context A)
        |
        v
SimulationRegistry
        |
        v
external-result structural evidence(context A)
        |
        v
RegistryValidatedSimulationEvidence(context A)
        |
        v
DeliberationBoundSimulationCandidate(snapshot A)
```

and demonstrate that substitution, omission, addition, stale lineage, malformed context,
and dry-run paths all fail closed.

Only after that should the first real acoustic diagnostic adapter be allowed to join the
Physical Agency stack.
