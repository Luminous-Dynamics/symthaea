# ADR-028 — RSK/Xenia state-witness golden vectors

**Status**: Proposed  
**Date**: 2026-09-13  
**Change Class**: A

## Context

RSK's monotonic-anchor contract and the Xenia generic state-witness draft are independently fail-closed, but cross-system safety also depends on exact agreement about protocol bytes and deterministic identity derivation.

The adapter contract in ADR-027 freezes the semantic mapping. Without executable vectors, an implementation could still drift in field ordering, length-prefix encoding, integer endianness, target/epoch derivation, state-digest handling, or bootstrap predecessor semantics while remaining locally well-typed.

## Decision

Add a committed Class A golden fixture and execute it through the existing `test_rsk_monotonic_anchor.py` self-test.

The fixture pins:

- Xenia reference PR/head identity;
- the exact RSK adapter namespace;
- target-ID derivation from admitted deployment identity plus registry identity;
- epoch-ID derivation from target ID plus the positive RSK recovery epoch;
- exact Xenia state-commitment bytes for counter 0;
- the published Xenia BLAKE3 fingerprint for that counter-0 commitment;
- exact Xenia state-commitment bytes for counter 1;
- counter 1 binding the exact counter-0 Xenia fingerprint;
- the published Xenia BLAKE3 fingerprint for counter 1;
- the rule that the RSK tracker SHA-256 digest is decoded to 32 raw bytes without rehashing.

The Python side deliberately does **not** reimplement BLAKE3. Xenia owns the fingerprint algorithm and its Rust golden-vector test; RSK consumes the published fingerprint as cross-system evidence and independently verifies every byte preceding that fingerprint.

## Rationale

This creates two independent checks without creating a second cryptographic implementation:

1. Xenia pins canonical commitment bytes plus BLAKE3 fingerprints in Rust.
2. RSK independently reproduces the same canonical commitment bytes and deterministic SHA-256 adapter identities using its existing canonical-JSON profile.

A mismatch becomes an explicit qualification failure rather than an implicit protocol fork.

## Safety boundary

These vectors prove protocol compatibility only. They do not prove:

- production witness-key custody;
- independent administrative/failure domains;
- trusted time;
- persistent compare-and-swap storage;
- external retention;
- hardware monotonicity;
- recovery authorization;
- replication authority.

Production admission remains **DENIED / NOT YET ELIGIBLE**.

## Consequences

Any future change to the Xenia commitment schema, domain separator, encoding order, integer representation, target/epoch derivation, or predecessor mapping must deliberately update both sides' vectors and receive Class A review.

A new vector version is preferred over silently changing the meaning of v0.1.
