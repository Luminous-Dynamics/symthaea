# Verification-Method Resolution Snapshot Contract

## Purpose

A verification report that records a resolver snapshot must bind the resolved verification method and the snapshot identifier to the same resolution event.

The snapshot is evidence about the resolver state consulted by verification. It is not a claim that the underlying attestation, observation, or external-world fact is true.

## Required semantics

### 1. Resolution and snapshot are one evidence unit

Implementations backed by mutable or remote state should implement `VerificationMethodResolver::resolve_with_snapshot` directly.

The implementation MUST obtain:

- the resolved verification method;
- the lifecycle state used for verification;
- the proof-purpose authorization state used for verification; and
- the durable snapshot identifier/fingerprint

from one atomic or otherwise consistency-preserving view.

A caller MUST NOT assume that calling `resolve()` and then `snapshot_fingerprint_for()` is atomic.

The default trait implementation intentionally exists for source compatibility with existing resolvers. It is a compatibility fallback, not an atomicity guarantee.

### 2. Snapshot identifiers are opaque

A snapshot fingerprint is an opaque resolver-provided identifier.

Consumers MUST NOT assume:

- a particular hash algorithm;
- a fixed hexadecimal length;
- that the identifier is locally recomputable;
- or that it represents the entire resolver registry.

A resolver may instead return a durable database revision, transparency-log position, content-addressed identifier, or another stable identifier whose semantics are documented by that resolver.

### 3. Scope should match the evidence being asserted

Where a resolver can provide it, `snapshot_fingerprint_for(method)` SHOULD identify the state relevant to the requested verification method rather than unrelated registry state.

The in-memory resolver therefore exposes:

- the historical whole-registry v1 fingerprint for compatibility; and
- a method-scoped v2 fingerprint for new resolution evidence.

Changing unrelated verification methods should not invalidate a scoped snapshot for a method whose resolution facts did not change.

### 4. Missing snapshots are explicit

`None` means that the resolver did not provide a durable snapshot identifier.

Consumers that require replayable/auditable resolution evidence SHOULD require an explicit snapshot identifier rather than silently treating `None` as proof that resolution was durable.

### 5. Snapshot evidence does not establish truth

A valid snapshot proves only that the verifier recorded a particular resolver state or durable state reference.

It does not establish:

- truth of the attested observation;
- correctness of an external-world claim;
- signer intent;
- evaluator independence;
- or correctness of the resolver's own data.

Those remain separate epistemic boundaries in the Evidence Fabric.

## Replayability guidance

For durable deployments, retaining only a resolver identifier may be insufficient if the referenced state can later disappear.

A deployment SHOULD retain enough resolver evidence to reconstruct the verification decision at the time it was made. Depending on the resolver, that can mean retaining the relevant signed registry statement, authenticated key material, authorization policy, lifecycle evidence, and the durable state identifier.

This mirrors the broader auditability principle in RFC 9943: systems intended for later audit should retain enough information to reproduce the checks that were applicable when a statement was accepted. RFC 9943 also distinguishes authentication/registration evidence from the truth or accuracy of the statement itself.

## Compatibility rule

Historical report fingerprints must remain reconstructable.

New snapshot semantics may therefore require a new versioned canonical representation while preserving the old representation for historical evidence. The current v1 whole-registry fingerprint is retained for that reason; the v2 method-scoped fingerprint is a new representation and must not silently rewrite historical v1 identities.

## Threat model

The primary failure this contract prevents is state skew:

1. resolve a key from state A;
2. the backing registry changes to state B;
3. independently compute a snapshot for state B;
4. emit evidence that appears to bind the key to B.

That evidence is internally coherent only at the serialization layer; it does not prove that B produced the key actually used for verification.

The atomic `resolve_with_snapshot` contract removes this ambiguity for resolvers that can provide a consistency-preserving view.
