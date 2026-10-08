# RFC 9942 PQ/T Hybrid Assurance Design

Status: design proposal; deliberately separate from the ES256 qualification gate in PR #6737.

## Decision

Use a **PQC/classical hybrid assurance lane**, but do **not** replace ES256 in the current RFC 9942 qualification tranche and do **not** encode the current IETF composite COSE algorithm identifiers as if they were standardized.

Near-term profile:

- Classical component: existing ES256 (COSE alg -7) for the RFC 9942 receipt path.
- PQ component: ML-DSA-65 (COSE alg -49, standardized by RFC 9964).
- Combination: an application-level **PQ-bound attestation** over the exact Receipt identity and the already-verified classical capability. Both verifications are required for the stronger "PQ-hybrid qualified" state; the two signatures are not currently a standardized composite and do not sign an identical transcript.
- Existing RFC 9942 ES256 receipt remains independently valid and independently qualified.
- The hybrid layer is a second assurance result; it does not change the meaning of the RFC 9942 VDS proof or make a selection decision into cryptographic proof.

This is intentionally an application-level dual-signature profile rather than the still-evolving JOSE/COSE composite-signature profile.

## Why now

The current system already has a strict semantic boundary:

exact wire
→ cryptographic verification
→ verified capability
→ selection witness
→ durable Holochain projection

A quantum-resistant component belongs at the **cryptographic verification / capability** seam, before selection and projection.

The system should therefore distinguish at least:

- ClassicalVerified
- PqVerified
- HybridVerified

and never infer the third state from either component alone.

## Standards position (2026-10-08)

NIST has finalized FIPS 204 for ML-DSA. RFC 9964 standardizes ML-DSA for JOSE/COSE and registers:

- ML-DSA-44 as COSE -48
- ML-DSA-65 as COSE -49
- ML-DSA-87 as COSE -50

The active IETF JOSE/COSE composite-signature draft proposes:

- ML-DSA-65-ES256
- a requested COSE assignment of -55

but that composite algorithm identifier is **TBD**, not a final IANA registration.

Therefore Symthaea must not place -55 on the wire and call it an interoperable standardized algorithm today.

## Recommended profile

### Hybrid policy identifier

The implementation currently uses this private assurance profile identifier:

symthaea-swarm/rfc9942-pq-bound-mldsa65-v1

This is a **policy/profile identifier**, not a COSE alg value.

### Authenticated message — current implementation seam

The current branch deliberately implements an **out-of-band PQ attestation**, not a COSE composite and not a second ES256 signature over a new transcript. RFC 9942's existing ES256 signature is verified according to RFC 9942/COSE; the ML-DSA-65 provider verifies a separate fixed-width transcript that binds the exact Receipt bytes and the already-verified classical capability identity.

This distinction matters: the ES256 signature does **not** sign the 186-byte hybrid transcript. The transcript links the PQ attestation to a particular classical verification result, rather than pretending that the two component signatures use a standardized composite construction.

The current authenticated byte layout is exactly 186 bytes:

| Offset | Width | Value |
|---|---:|---|
| 0 | 2 | Hybrid profile version, unsigned big-endian |
| 2 | 8 | ML-DSA-65 COSE algorithm identifier -49, signed big-endian |
| 10 | 32 | Key-policy/snapshot SHA-256 |
| 42 | 16 | Application key-policy ID |
| 58 | 32 | SHA-256 of exact ML-DSA public-key bytes |
| 90 | 32 | SHA-256 of exact serialized RFC 9942 Receipt bytes |
| 122 | 32 | SHA-256 of the verified classical capability |
| 154 | 32 | Domain-separated transcript digest |

The transcript digest uses the private domain `symthaea-swarm/rfc9942-pq-bound-mldsa65-transcript-v1`, the profile version, algorithm ID, key-policy digest, key ID, public-key digest, and length-framed Receipt/capability digests. The full receipt bytes are bounded and hashed before this metadata is formed; parsed Rust structs are not re-serialized to define the signed meaning.

The verifier must retain the exact Receipt digest, classical capability digest, key-policy snapshot digest, key ID, public-key digest, PQ signature digest, transcript digest, evaluation time, and private hybrid capability identity.

**Not yet implemented:** a serialized PQ-attestation envelope, the RFC 9964 AKP COSE_Key/thumbprint adapter, an actual ML-DSA provider, hybrid-required selection admission, and a Holochain projection field that durably carries the hybrid assurance. The 16-byte key ID is an application registry identifier, not a claim to be an RFC 9964 key thumbprint. Those are explicit follow-on seams, not implied capabilities.

Do not claim that the existing ES256 signature authenticates PQ-specific metadata or that this interim profile is interoperable as one composite COSE object.

## Key separation

Do **not** reuse an existing standalone ML-DSA key as a component of a future standardized Composite-ML-DSA key.

The current IETF composite work explicitly requires both component keys to be freshly generated for the composite and prohibits reuse of component key material in standalone or other composite contexts.

For the interim application-level dual-signature profile, key identities must remain independently addressable:

- classical_key_id
- pq_key_id

and the verified capability must retain both key fingerprints.

The future migration target is a freshly generated composite key pair once the JOSE/COSE composite specification is standardized.

## ML-DSA parameter choice

Prefer **ML-DSA-65**.

Reasons:

- security category 3;
- 1952-byte public key;
- 3309-byte signature;
- substantially smaller than ML-DSA-87;
- stronger margin than ML-DSA-44;
- standardized COSE algorithm identifier -49.

The wire-size increase is material but acceptable for a receipt/provenance channel where correctness and long-term verifiability are more important than minimum packet size.

## Implementation / oracle strategy

Do not make an unaudited implementation the only cryptographic acceptance oracle.

The current RustCrypto `ml-dsa` crate tracks FIPS 204 and its latest published release is the right kind of API candidate for a Rust implementation, but its documentation explicitly states that the implementation has **never been independently audited**. It has also had multiple 2026 security advisories, including a signature-verification malleability regression that was fixed in later releases.

For qualification, use an independent implementation as an oracle. OpenSSL 3.5+ documents ML-DSA-44/65/87 support in both its default and FIPS providers and exposes one-shot sign/verify operations. This makes OpenSSL a useful independent vector-generation and cross-verification oracle even if the runtime Rust implementation is kept separate.

The acceptance rule should therefore be:

runtime verifier passes
+ independent implementation verifies the same bytes
+ independent implementation rejects the negative vectors
+ exact wire/transcript identity matches

No single library implementation should be treated as its own proof of correctness.

## Capability model

Add a private capability layer, conceptually:

Rfc9942VerifiedHybridSignature

with private fields for:

- exact classical capability identity;
- exact PQ capability identity;
- exact authenticated transcript identity;
- classical signature identity;
- PQ signature identity;
- classical verification-key fingerprint;
- PQ verification-key fingerprint;
- policy/profile identifier and version;
- receipt / outer-object identity.

Expose only accessors that preserve the distinction between:

- algorithm verification;
- hybrid verification;
- capability identity;
- authorization.

A hybrid capability must never expose an API that implies "true" merely because one component verified.

## Selection boundary

Receipt selection remains downstream:

wire → verify-classical → verify-PQ attestation over exact Receipt + classical capability → form-hybrid-capability → (not yet implemented: hybrid-required selection admission) → durable projection

The existing ReceiptSelectionDecision remains audit/evaluator evidence only.

The durable witness should eventually bind:

source collection + selected receipt wire + classical capability + PQ capability + hybrid capability

and require the expected hybrid policy.

A receipt with only ES256 is still a valid **classical-qualified** receipt.

A receipt with both valid components is **hybrid-qualified**.

A malformed or partially verified PQ component must not silently downgrade to the classical state when the caller requested hybrid policy.

## Failure taxonomy

The hybrid verifier needs typed failures, not generic "invalid signature" strings:

- PqAlgorithmMismatch
- PqPublicKeyWrongLength
- PqSignatureWrongLength
- PqPublicKeyInvalid
- PqSignatureInvalid
- HybridTranscriptMismatch
- HybridPolicyMismatch
- HybridKeyIdentityMismatch
- HybridComponentMissing
- HybridResourceLimitExceeded

These should map into the existing selection rejection vocabulary without collapsing resource failures into semantic-invalid failures.

## Decoder/resource limits

ML-DSA is larger than ES256, so the hybrid parser needs explicit limits.

At minimum:

- ML-DSA-65 public key: exactly 1952 bytes;
- ML-DSA-65 signature: exactly 3309 bytes;
- explicit aggregate envelope cap;
- explicit nested-object depth/entry cap;
- no attacker-controlled allocation before exact length admission.

The existing generic bounds are generous enough for these values, but hybrid-specific fixed-width admission should still be implemented. A generic 64 KiB signature limit is not a substitute for an exact ML-DSA-65 signature-length check.

## Authentication versus provenance

The same rule already enforced for RFC 9052 must continue:

- protected headers used by a COSE signature are authenticated;
- unprotected headers are not authenticated by that signature.

No PQ COSE envelope is implemented yet. If a later wire format places an attestation in an unprotected container, it must carry its own authenticated binding and must not be treated as authenticated merely because the surrounding object parses.

The durable capability must record whether a field was:

- authenticated by the classical signature;
- authenticated by the PQ signature;
- provenance only.

## Qualification strategy

Do not add this to PR #6737.

PR #6737 currently provides an important exact-head ES256 qualification boundary. Adding ML-DSA would introduce a new dependency, new vectors, new wire cases, new parser limits, and new interoperability obligations while the current gate is still blocked in Cargo's locked-resolution stage.

Instead, create a stacked follow-up qualification lane after #6737 obtains an uncontested PASS.

The hybrid qualification should include:

1. RFC 9964 ML-DSA-65 fixed known-answer vectors from an independent implementation.
2. Positive and negative COSE interoperability vectors.
3. Exact transcript-binding tests.
4. Classical-component tamper rejection.
5. PQ-component tamper rejection.
6. Wrong-PQ-key rejection.
7. Missing-component rejection under hybrid policy.
8. Unprotected-header substitution tests.
9. Signature/public-key length boundary fuzz cases.
10. Deterministic capability identity tests.
11. Selection-witness binding tests.
12. Holochain projection requiring the hybrid witness.
13. Cross-implementation verification against at least one independent ML-DSA implementation.
14. Fuzzing for nested hybrid/COSE parser exhaustion.

The qualification result must remain fail-closed:

**no hybrid PASS until exact-head checkout, locked resolution, parser, cryptographic, semantic, selection, projection, and fuzz stages all pass.**

## Composite migration target

Once the IETF JOSE/COSE composite-signature specification has a final RFC and the COSE identifier is registered, evaluate migration to the standardized composite form.

Do not simply replace a private policy identifier with the final composite alg value.

Instead:

1. generate fresh composite component keys;
2. validate the final key serialization;
3. verify both components;
4. run a new interop corpus;
5. compare capability semantics between the application-level dual-signature profile and the standardized composite profile;
6. keep explicit versioned policy IDs so old receipts remain interpretable.

## Security posture

The architectural rule should be:

> Classical verification protects today's interoperability.
> PQ verification protects the long-term transition.
> Hybrid qualification requires both.

This also keeps a clean migration path if a future PQ implementation is found to have a flaw. A compromised PQ component must not silently manufacture a stronger capability, and a compromised classical component must not erase the requirement for the PQ component once a hybrid policy is requested.

## References

- NIST FIPS 204, Module-Lattice-Based Digital Signature Standard.
- RFC 9964, ML-DSA for JOSE and COSE.
- RFC 9052, COSE Structures and Process.
- RFC 9942, Signature Validation for VDP Receipts.
- IETF Internet-Draft, PQ/T Hybrid Composite Signatures for JOSE and COSE, draft-ietf-jose-pq-composite-sigs-05 (current as of 2026-10-08).
