# WCARE-42 — Builder attestation verifier protocol v1

Status: `SOURCE_REVIEW_CANDIDATE`
Authority: `MeasurementOnly`
Verifier protocol: `wcare42-builder-attestation-verifier-v1`
Subject protocol: `wcare41-authenticated-preregistration-v1`

## Purpose

WCARE-42 implements the builder-signature channel frozen by WCARE-41. It verifies one exact `BuilderProvenance` or `BuilderRelation` attestation at a time.

The governing distinction is:

`accepted single attestation != aggregate builder authentication`

A successful WCARE-42 receipt can later contribute to WCARE-41 coverage. It does not by itself establish that all WCARE-40 builder evidence is authenticated or that the WCARE-40 independence result is correct.

## Inputs

The standalone verifier accepts exactly:

1. WCARE-41 builder attestation envelope;
2. WCARE-41 builder issuer trust policy;
3. exact WCARE-40 replication plan;
4. exact WCARE-40 result;
5. exact WCARE-40 builder provenance or relation receipt;
6. canonical evaluation UTC.

The verifier is outside the Symthaea workspace and uses pinned `ed25519-dalek = 2.1.0`. It must not implement custom signature mathematics.

## Canonical signed bytes

WCARE-42 implements exactly the canonical builder-attestation bytes frozen by WCARE-41, including final LF and domain `builder-evidence-attestation`.

The envelope public key is part of the signed body. Signature validity can therefore be determined even for an issuer that is absent from or untrusted by the supplied trust policy.

## Exact subject binding

Acceptance requires the envelope to bind:

- exact WCARE-40 plan byte SHA-256;
- exact WCARE-40 builder subject receipt byte SHA-256;
- exact WCARE-41 issuer trust-policy byte SHA-256;
- exact subject kind;
- exact provenance/relation strength claim copied from the subject receipt.

When `wcare40_result_sha256` is a 64-hex digest, that exact WCARE-40 result is also signed and must match the supplied bytes.

When `wcare40_result_sha256 = -`, the attestation is explicitly **pre-result**: the plan and subject receipt are signature-bound, but the supplied later WCARE-40 result is context only and `result_bound_by_signature = false`.

Pre-result signing must never be represented as if the later result itself was signed.

## Trust policy

Signature validity and issuer authorization are separate.

The exact trust-policy bytes are SHA-bound by the signature. Trust additionally requires:

- exact key ID + public key match;
- policy creation no later than attestation issue time;
- issue time inside the key validity interval;
- issue time before revocation takes effect;
- evaluation at or after issue and before attestation expiry when expiry exists;
- subject kind authorized;
- exact claimed strength authorized.

An otherwise valid signature whose key is not authorized yields `SIGNATURE_VALID_ISSUER_UNTRUSTED`, not `ATTESTATION_ACCEPTED`.

## Dispositions

- `ATTESTATION_ACCEPTED`
- `SIGNATURE_VALID_ISSUER_UNTRUSTED`
- `ATTESTATION_REJECTED`
- `INFRASTRUCTURE_INDETERMINATE`

`ATTESTATION_ACCEPTED` means only that this exact single attestation passed signature, subject-binding, currentness, and bound trust-policy checks.

## Cross-implementation vector

The review unit contains a frozen Ed25519 vector produced with Python `cryptography`, not by the Rust verifier under test. The Rust tests must reconstruct the WCARE-41 canonical message, match the frozen message SHA-256, and verify the frozen signature.

This detects canonical-byte or Ed25519 verification drift that could be hidden if signing and verifying used the same implementation.

## Executable qualification blocker

The standalone verifier must have its own committed `Cargo.lock` before executable qualification can produce PASS. The WCARE-42 qualifier fails closed while that lock is absent.

A source review, a valid golden vector, or workspace CI that does not execute the locked standalone verifier is not executable cryptographic qualification.

## Claim boundary

WCARE-42 does not establish panel-wide builder authentication, builder independence, temporal preregistration, subject correctness, reviewer independence, network/sandbox isolation, consciousness, phenomenal experience, suffering, moral patienthood, consent, veto/self-preservation authority, or solved alignment.

`builder_authentication_established`, `preregistration_temporal_precedence_established`, `subject_correctness_established`, and `runtime_authority_granted` remain false in every single-attestation WCARE-42 receipt.
