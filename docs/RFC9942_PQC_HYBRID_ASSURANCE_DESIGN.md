# RFC 9942 PQ/T Hybrid Assurance Design

Status: design proposal; deliberately separate from the ES256 qualification gate in PR #6737.

## Decision

Use a **PQC/classical hybrid assurance lane**, but do **not** replace ES256 in the current RFC 9942 qualification tranche and do **not** encode the current IETF composite COSE algorithm identifiers as if they were standardized.

Near-term profile:

- Classical component: existing ES256 (COSE alg -7) for the RFC 9942 receipt path.
- PQ component: ML-DSA-65 (COSE alg -49, standardized by RFC 9964).
- Combination: an application-level **dual signature / hybrid attestation** in which both signatures are required for the stronger "PQ-hybrid qualified" state.
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

Use a private application policy identifier such as:

symthaea-swarm/rfc9942-dual-signature-mldsa65-es256-v1

This is a **policy/profile identifier**, not a COSE alg value.

### Authenticated message

Define one canonical **hybrid transcript** and have both components sign that exact byte string independently.

Recommended transcript:

hybrid_message = Domain || version || len(receipt_wire) || receipt_wire

where:

- Domain is a fixed, protocol-owned domain separator;
- version is the profile version;
- receipt_wire is the exact byte sequence being bound by the hybrid attestation.

The classical ES256 hybrid signature and the ML-DSA-65 signature must both authenticate hybrid_message. This avoids making the PQ signature depend on the internal representation of the ES256 COSE Sig_structure and keeps the two component signatures semantically parallel.

The existing RFC 9942 ES256 signature remains independently verified; the hybrid attestation is a second assurance layer over the exact receipt wire.

The hybrid verifier should retain:

- SHA-256 of the exact hybrid transcript bytes;
- SHA-256 of the exact receipt wire bytes;
- SHA-256 of the exact PQ COSE protected header bytes;
- SHA-256 of the exact PQ signature bytes;
- SHA-256 of the exact PQ public-key bytes;
- SHA-256 of the hybrid ES256 signature bytes;
- both algorithm identifiers;
- hybrid policy/version.

The canonical transcript must be length-delimited and domain-separated. Do not reconstruct it from parsed semantic fields after verification.

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

wire → verify-classical → verify-pq → form-hybrid-capability → selection witness → durable projection

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

Therefore a PQ hybrid attestation placed in an unprotected container must carry its own cryptographic binding and must not be treated as authenticated merely because the surrounding object parses.

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
