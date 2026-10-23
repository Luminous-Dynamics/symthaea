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

The structural resolver does **not** perform network access, DNS resolution, controller authorization, or digital-signature verification.
The crate now also contains a deliberately narrow `eddsa-jcs-2022` verifier: it consumes the already-resolved `Multikey` material, applies RFC 8785 JCS, computes the W3C SHA-256 pair-hash input, and verifies the detached Ed25519 signature. It is not a generic Data Integrity 1.1 processor and does not implement the RDF canonicalization suite, proof sets/chains, network retrieval, or controller-document resolution itself.

Truth/reliability assessment remains outside this cryptographic boundary.

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

### Cryptographic profile

The `eddsa-jcs-2022` path is intentionally a single-suite profile. It requires `DataIntegrityProof`, `eddsa-jcs-2022`, `verificationMethod`, `proofPurpose`, base58-btc `proofValue`, and an Ed25519 `Multikey` carrying the `0xed01` multicodec header. The resolver also recognizes the normative CID public Multikey headers and rejects private-key encodings as public material. Its cryptographic input is exactly `SHA-256(proofConfig) || SHA-256(transformedDocument)`, followed by pure Ed25519 verification. The typed receipt records the canonical-document digest, proof-configuration digest, the digest of that 64-byte cryptographic input, the full proof identity, and the detached proof value.

The wire and programmatic entry points enforce the same strict JSON/JCS input boundary before canonicalization: duplicate object properties, Unicode surrogate/noncharacter code points, and trailing JSON data are rejected. Symthaea also applies a stronger numeric interoperability profile that rejects integers not exactly representable as IEEE-754 binary64; this is intentionally stricter than RFC 7493's SHOULD NOT guidance. For `eddsa-jcs-2022`, the caller must also bind the exact expected transformed-document SHA-256 digest into the `VerificationRequest`; the adapter refuses to verify when that binding is absent or mismatched, and the typed cryptographic receipt preserves it for replay/audit.

The implementation deliberately does not claim conformance to the complete W3C Verifiable Credential Data Integrity processing model. It implements the specified `eddsa-jcs-2022` cryptographic core and keeps the controller-document/admission/replay boundaries explicit.

The temporal profile uses the shared XML Schema 1.1 `dateTimeStamp` lexical boundary used by the core verification contract, including explicit timezone offsets, year zero/expanded years within Chrono's representable range, XSD end-of-day `24:00:00(.0+)`, and XML whitespace collapse. Fractional seconds beyond nanosecond precision are accepted only when the additional digits are zero; values outside the finite internal temporal range or with non-representable sub-nanosecond precision fail closed. The original lexical timestamp spelling remains preserved in snapshot/evidence identity, so semantic timestamp equivalence does not erase provenance.

The optional `digestMultibase` receipt artifact is aligned with the W3C Verifiable Credential Data Integrity 1.1 Working Draft resource-integrity property. The current implementation intentionally emits one SHA-256 Multibase/Multihash value and binds it to the independently recorded SHA-256 document digest.

## CI

The repository sub-crate matrix includes both `symthaea-epistemic-types` and `symthaea-verification-adapter`, so these contracts are exercised by ordinary CI rather than existing only as locally targeted tests.


## Standards status

The concrete `eddsa-jcs-2022` cryptographic core targets the W3C Data Integrity EdDSA Cryptosuites v1.0 Recommendation published 15 May 2025. The broader Data Integrity 1.1 document is still a Working Draft as of 30 September 2026, so this crate does not claim v1.1 processor conformance.
