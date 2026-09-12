# ADR-017: RSK verified semantic schema registry boundary

**Status**: Accepted for reference design

**Date**: 2026-09-12

**Change Class**: A

## Context

Drafts #1936 and #1966 establish schema-bound and structurally schema-validated RSK semantic values. They deliberately do not prove that the resolved schema definition came from the current authorized semantic registry.

The remaining distinction is:

```text
resolved schema != verified schema provenance
```

A structurally valid schema table supplied by an attacker could otherwise mark arbitrary capability bits assignable or redefine a resource dimension while still satisfying the local validation algorithm.

Issue #1969 tracks this provenance boundary. Static review also found that the current v0.1 resource golden corpus does not commit the Rust runtime numeric dimension mapping inside the resource schema digest; #1973 tracks that correction.

## Decision

Freeze a dedicated verified schema-registry contract before implementing cryptographic registry code.

The future verifier will consume:

- exact signed canonical registry evidence;
- a separately verified fresh trust snapshot;
- registry verification policy;
- trusted time/continuity evidence;
- durable anti-rollback registry state.

It may produce an opaque in-process `VerifiedSchemaRegistrySnapshot` only after all predicates pass.

## Internal precedent

The Fabrication Kernel already demonstrates a useful pattern:

- canonical digest binding;
- raw signed evidence distinct from opaque verified capability;
- lifecycle/usage-aware trust snapshots;
- snapshot freshness;
- sequence rollback/collision detection;
- verified capability fields retaining trust/evaluation evidence.

RSK will reuse this architectural shape without importing Fabrication-specific authority semantics into the schema registry.

## Canonical registry properties

The registry snapshot is sequence-numbered, freshness-bounded, canonically encoded, domain-separated at the registry-snapshot digest layer, and binds exact schema entries.

Each schema entry binds exact canonical schema bytes and a claimed schema ID that the verifier independently recomputes.

Duplicate/ambiguous schema keys fail closed.

## Signer independence

Signer identity and key identity remain distinct.

Multiple keys belonging to one identity cannot inflate signer quorum. Failure-domain independence is derived from trusted metadata rather than signer assertion.

Missing required domain metadata fails closed.

## Rollback/freeze behavior

The verifier/tracker rejects sequence rollback, same-sequence different-content collision, issuance regression, configured previous-digest mismatch, and stale/frozen snapshots.

A collision/fork becomes non-operational rather than selecting a favorable branch.

Exact replay may reconstruct evidence but does not refresh expiry.

## Resource numeric-ID rule

Production resource schema identity must commit the runtime numeric `ResourceDimensionId` mapping used by the semantic arithmetic TCB.

The current v0.1 test golden digest is not retroactively treated as proving that mapping. A revised canonical resource schema encoding must produce a new scheme identity.

## Authority separation

Verified schema provenance cannot by itself:

- mint replication grants;
- satisfy replication grant quorum;
- clear negative state;
- extend expiry;
- widen ceilings;
- approve cross-schema migration.

A verified registry capability is semantic provenance only.

## Restart rule

`VerifiedSchemaRegistrySnapshot` is not a durable deserializable authority object.

Persistent storage retains raw/signed evidence plus durable anti-rollback state. Restart requires reverification under current trust, lifecycle, policy, and time evidence.

## Validation plan

`RSK_SCHEMA_REGISTRY_VERIFICATION_PLAN_V0_1.md` defines stable negative-test families for canonicalization, signature/lifecycle failure, failure-domain collapse, rollback/fork/freeze, schema lifecycle, restart, authority non-amplification, and composition with structural schema validation.

Authored test plans are not executed evidence.

## Non-goals

This ADR does not:

- implement registry cryptography;
- modify the current RSK authority evaluator/ledger;
- approve a production capability vocabulary;
- define physical resource measurements;
- prove a schema scientifically correct;
- approve cross-schema translation;
- establish production admission.

## Consequence

The semantic trust pipeline becomes explicit:

```text
canonical schema definition
 -> exact schema identity
 -> verified registry provenance
 -> structural value validation
 -> verified policy/grant evaluation
 -> bounded authorization
 -> atomic ledger commit
```

No single layer is allowed to silently stand in for another.

Production admission remains **DENIED / NOT YET ELIGIBLE**.
