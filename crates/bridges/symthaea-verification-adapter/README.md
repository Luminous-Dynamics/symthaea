# symthaea-verification-adapter

Durable controller-document resolution for the Symthaea verification contract.

## Purpose

This crate turns an application-owned controller-document snapshot into a `VerificationMethodResolution` that can be consumed by the substrate-neutral verification evidence layer.

The adapter is deliberately narrower than a complete cryptographic verifier:

- it reads one versioned snapshot envelope from durable storage;
- it verifies the snapshot against an expected content-addressed reference;
- it parses the controller document and checks its identifier;
- it resolves the requested verification method;
- it checks method/controller binding;
- it checks membership in the requested verification relationship;
- it carries method lifecycle timestamps into the resolution;
- it binds the declared method type and a SHA-256 identity digest of the exact public verification material;
- it can return that exact public material to a downstream cryptographic verifier without re-resolving the snapshot;
- it emits an `ApplicationSnapshot` dereference attestation with the exact SHA-256 digest and standards-oriented `digestMultibase` artifact.

It does **not** perform network access, DNS resolution, controller authorization, digital-signature verification, or truth/reliability assessment.

## Snapshot identity

The filesystem path is only a retrieval hint. The durable identity is:

`sha256:<64 lowercase hexadecimal characters>`

The digest covers:

1. snapshot schema version;
2. controller-document URL;
3. historical `state_at`;
4. dereference `resolved_at`;
5. response media type; and
6. the exact controller-document text consumed by the parser.

Consequently, copying a snapshot to another path does not change its identity, while changing any of those fields makes the reference mismatch.

## Security posture

The adapter is fail-closed at the durable-resolution boundary.

A current mutable controller document is not substituted for a historical snapshot. The snapshot's `state_at` must match the proof lifecycle reference time, and its `resolved_at` must be the same instant recorded by the dereference attestation.

The requested verification method must be present exactly once in the controller document and its controller must equal the controller-document URL. The method must also occur exactly in the requested verification relationship.

Relationship entries are normalized to absolute URLs before duplicate checks, so two syntactically different references cannot silently become two distinct authorization members.

The raw persisted envelope is bounded to 32 MiB and the decoded controller document is independently bounded by the request's response-size policy.

## Standards relationship

The implementation follows the security-critical structure of the W3C Controlled Identifiers v1.0 retrieval algorithm without claiming generic W3C conformance for the Symthaea contract.

The resolution also binds the exact public verification-material identity so a downstream cryptographic verifier can refuse key-material substitution between resolution and signature checking.

The optional `digestMultibase` receipt artifact is aligned with the W3C Verifiable Credential Data Integrity 1.1 Working Draft resource-integrity property. The current implementation intentionally emits one SHA-256 Multibase/Multihash value and binds it to the independently recorded SHA-256 document digest.

## CI

The repository sub-crate matrix includes both `symthaea-epistemic-types` and `symthaea-verification-adapter`, so these contracts are exercised by ordinary CI rather than existing only as locally targeted tests.
