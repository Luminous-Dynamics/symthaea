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

This is intentionally an application-level PQ-bound attestation profile rather than the still-evolving JOSE/COSE composite-signature profile.

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

symthaea-swarm/rfc9942-es256-pq-bound-mldsa65-v1

This is a **policy/profile identifier**, not a COSE alg value.

### Authenticated message — current implementation seam

The current branch deliberately implements an **out-of-band PQ attestation**, not a COSE composite and not a second ES256 signature over a new transcript. RFC 9942's existing ES256 signature is verified according to RFC 9942/COSE; the ML-DSA-65 provider verifies a separate fixed-width transcript that binds the exact Receipt bytes and the already-verified classical capability identity.

This distinction matters: the ES256 signature does **not** sign the 186-byte hybrid transcript. The transcript links the PQ attestation to a particular classical verification result, rather than pretending that the two component signatures use a standardized composite construction.

The current authenticated byte layout is exactly 194 bytes. This profile is restricted to ES256 classical receipts (COSE algorithm -7), and that algorithm ID is also explicitly signed:

| Offset | Width | Value |
|---|---:|---|
| 0 | 2 | Hybrid profile version, unsigned big-endian |
| 2 | 8 | ML-DSA-65 COSE algorithm identifier -49, signed big-endian |
| 10 | 8 | Classical ES256 COSE algorithm identifier `-7`, signed big-endian |
| 18 | 32 | Key-policy/snapshot SHA-256 |
| 50 | 16 | Application key-policy ID |
| 66 | 32 | SHA-256 of exact ML-DSA public-key bytes |
| 98 | 32 | SHA-256 of exact serialized RFC 9942 Receipt bytes |
| 130 | 32 | SHA-256 of the verified classical capability |
| 162 | 32 | Domain-separated transcript digest |

The transcript digest uses the private domain `symthaea-swarm/rfc9942-es256-pq-bound-mldsa65-transcript-v1`, the profile version, both algorithm IDs, key-policy digest, key ID, public-key digest, and length-framed Receipt/capability digests. Both construction and verification reject a classically valid non-ES256 receipt before consulting the PQ key policy or verifier. The full receipt bytes are bounded and hashed before this metadata is formed; parsed Rust structs are not re-serialized to define the signed meaning.

The verifier must retain the exact Receipt digest, classical capability digest, key-policy snapshot digest, key ID, public-key digest, PQ signature digest, transcript digest, evaluation time, and private hybrid capability identity.

**Implemented as semantic seams on the current branch:** the snapshot-bound key-authorization result, fail-closed ClassicalAllowed / HybridRequired admission, binding of the PQ capability to the exact selected Receipt and classical capability, and schema-v2 Holochain projection metadata. The projection records the required/not-required bit plus hybrid capability identity, key-policy digest, evaluation time, key ID, both explicit algorithm IDs (ES256 `-7` and ML-DSA-65 `-49`), verification-key fingerprint, PQ signature fingerprint, Receipt/capability identities, and transcript digest. Its hybrid fields are read-only externally and are populated from the admission constructor rather than caller-supplied metadata. The only public constructor for a durable receipt-selection context requires an explicit assurance-admission object; there is no alternate public classical-only projection constructor that silently skips that decision.

**Not yet implemented or qualified:** a serialized PQ-attestation envelope, RFC 9964 AKP COSE_Key/thumbprint adapter, concrete ML-DSA-65 provider, independent known-answer/interoperability corpus, cross-implementation oracle, and a concrete trusted key-lifecycle snapshot implementation. The policy is an explicit trait contract; no concrete trust registry is claimed. The 16-byte key ID is an application registry identifier, not an RFC 9964 key thumbprint. The targeted locked test workflow and all current code still require an actual exact-head successful run; skipped/queued jobs do not qualify this seam.

Do not claim that the existing ES256 signature authenticates PQ-specific metadata or that this interim profile is interoperable as one composite COSE object. The Holochain projection is provenance/identity metadata, not a substitute for independently re-verifying retained cryptographic material at the integrity boundary.

## Key separation

Do **not** reuse an existing standalone ML-DSA key as a component of a future standardized Composite-ML-DSA key.

The current IETF composite work explicitly requires both component keys to be freshly generated for the composite and prohibits reuse of component key material in standalone or other composite contexts.

For the interim application-level PQ-bound attestation profile, key identities must remain independently addressable:

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

### ML-DSA verifier corpus must include mutation and boundary cases

Published ML-DSA signature-verification vectors are necessary but not sufficient by themselves. A public analysis in `usnistgov/ACVP-Server#470` (2026-08-11) reports five verifier mutations that survived the published `ML-DSA-sigVer-FIPS204` sets across the pinned repository states measured, while pinned Project Wycheproof community vectors detected them. The reported mutation classes include disabling or mis-stating the infinity-norm check, relaxing strict hint-index ordering, and disabling the hint-weight bound. Follow-up discussion also identifies the Algorithm 21 rule that unused tail entries in the hint-index array must be zero; vector regeneration had made that condition inconsistently exercised in the measured history. Treat this as community research, not as an official NIST finding or validation result.

The concrete ML-DSA-65 provider qualification corpus must therefore pin source revisions and SHA-256 digests and include, at minimum:

- invalid signatures with repeated hint indices (the hint indices must be strictly increasing);
- invalid signatures whose unused hint-index tail contains nonzero entries (unused entries must be zero);
- invalid signatures whose `z` vector has `||z||_∞ >= gamma_1 - beta`;
- invalid signatures whose hint weight exceeds `omega`;
- valid signatures immediately below the `gamma_1 - beta` boundary, so over-strict verifiers are detected as well as under-strict ones;
- unmodified positive/negative known-answer vectors and cross-implementation checks against an independent implementation.

Run these through the actual provider adapter with the RFC 9964 empty context, not only through isolated parsing helpers. Preserve per-vector expected result, corpus revision, file digest, provider/version, and exact test-command receipt. A passing corpus does not prove cryptographic correctness, but it closes concrete gaps that ordinary published verification sets may not expose.

Research reference: [usnistgov/ACVP-Server issue #470](https://github.com/usnistgov/ACVP-Server/issues/470).

## Cross-project crypto reuse decision: Mycelix

**Decision: consolidate the cryptographic implementation in Mycelix over time, but do not treat the current `mycelix-crypto` high-level hybrid API as a drop-in implementation or as qualified evidence for this RFC 9942 profile.** Reuse the primitive implementation through a narrow provider adapter once the packaging and verification gates below are met.

Inspection was performed against the pinned Mycelix `main` commit [`445a84abdaa64f3d05e2d5c51115e4d060b78ec7`](https://github.com/Luminous-Dynamics/mycelix/tree/445a84abdaa64f3d05e2d5c51115e4d060b78ec7):

- Mycelix already has `mycelix-identity/crates/mycelix-crypto`, its `AlgorithmId` registry, tagged key/signature types, and a RustCrypto `hybrid-rc` backend. The manifest pins `ml-dsa = 0.1.1`; the same feature pins `ml-kem = 0.3.0-rc.2`. The manifest explicitly labels the hybrid backend experimental and pending crypto audit. The crate docs for the ML-DSA implementation also warn that it has not been independently audited.
- The separate `native` backend still uses `pqcrypto-dilithium` and other pqcrypto crates. Mycelix has two open, directly relevant source-audit issues: [#252](https://github.com/Luminous-Dynamics/mycelix/issues/252) documents `AlgorithmId::MlDsa65` being used for both the legacy `pqcrypto_dilithium::dilithium3` backend and FIPS 204 ML-DSA-65 despite incompatible wire sizes/encodings; [#261](https://github.com/Luminous-Dynamics/mycelix/issues/261) records the analogous unqualified Dilithium5/ML-DSA-87 and Kyber/ML-KEM identity collision. Those issue descriptions are the repository's open findings, not independently reproduced test results here. **Do not use the current `native` backend to satisfy standardized FIPS ML-DSA policy until these contracts are repaired and exact conformance tests pass.** Do not silently treat backends as interchangeable: each needs its own exact version, corpus, target, and conformance receipts.
- Mycelix's current `hybrid_sig.rs` API signs the **same message** with Ed25519 and ML-DSA-65 and requires both signatures to verify. Symthaea's current profile instead verifies an RFC 9942 **ES256** Receipt and then checks an ML-DSA-65 attestation over the separate 194-byte transcript binding the exact Receipt and verified ES256 capability. These are different protocol constructions; the Mycelix high-level `HybridSigner` is not the correct API to call from this verifier.

These are not cosmetic naming issues: a crypto-agility identifier must bind the exact standard revision, key/signature or ciphertext encoding, and operation semantics. Signature/key lengths can detect some mismatches but do not establish full interoperability. The intended shared dependency should either remove/quarantine the legacy backends or give them distinct legacy identifiers that cannot satisfy standardized FIPS policy.

There is also a packaging blocker: the Mycelix repository has no root `Cargo.toml`; `mycelix-identity/Cargo.toml` is the workspace manifest that owns the `mycelix-crypto` member. The current crate therefore is not yet a clean, root-addressable shared dependency for an unrelated repository. The Mycelix PQC roadmap itself lists promotion of this crate to a shared workspace/root package as deferred structural work. Do not add an unpinned path assumption, duplicate the entire Mycelix identity workspace, or add another crypto implementation simply to bypass that packaging seam.

### Intended reuse boundary

1. Keep the RFC 9942 Receipt parsing/proof checks, ES256 capability, 194-byte transcript, key-policy authorization, `HybridRequired` admission, and Holochain projection in Symthaea. These are protocol-specific policy and provenance semantics.
2. Make `mycelix-crypto` the long-term shared home for algorithm-tagged key types and vetted primitive adapters. First publish/extract a versioned package from a real root workspace or standalone repository, with its dependency graph and target support pinned.
3. Add a small adapter that implements Symthaea's `MlDsa65Verifier::verify_with_empty_context` by invoking only the ML-DSA-65 primitive over the exact provided transcript bytes. The adapter must not reinterpret the Receipt, make key authorization decisions, or construct a second transcript.
4. Qualify that adapter with exact RFC 9964/empty-context vectors, pinned ACVP and Project Wycheproof negative/positive boundary corpora (including the Algorithm 21 hint-padding case described above), and cross-implementation checks against OpenSSL's ML-DSA provider. Store corpus revisions and file digests in the qualification receipt. Passing a simple sign/verify round trip is not enough.
5. Until the package extraction, provider tests, and independent corpus gate pass, retain the current trait-only boundary and make **no production-provider or hybrid-qualified claim**.

This gives Mycelix one crypto-agility control point and prevents Symthaea from forking duplicate cryptographic primitives, while keeping the application-specific transcript and admission rules independently reviewable.

Pinned implementation references:
- [Mycelix crypto manifest](https://github.com/Luminous-Dynamics/mycelix/blob/445a84abdaa64f3d05e2d5c51115e4d060b78ec7/mycelix-identity/crates/mycelix-crypto/Cargo.toml)
- [Mycelix RustCrypto hybrid signature implementation](https://github.com/Luminous-Dynamics/mycelix/blob/445a84abdaa64f3d05e2d5c51115e4d060b78ec7/mycelix-identity/crates/mycelix-crypto/src/hybrid_sig.rs)
- [Mycelix PQC roadmap](https://github.com/Luminous-Dynamics/mycelix/blob/445a84abdaa64f3d05e2d5c51115e4d060b78ec7/mycelix-workspace/PQC_ROADMAP_2026-07-07.md)
- [OpenSSL ML-DSA provider documentation](https://docs.openssl.org/3.5/man7/EVP_PKEY-ML-DSA/)

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

wire → verify-classical → verify PQ attestation over exact Receipt + classical capability → form-hybrid-capability → hybrid-required admission gate → schema-v2 durable projection

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

1. RFC 9964 ML-DSA-65 known-answer vectors from an independent implementation, with pinned corpus revision and SHA-256 digests.
2. Community negative vectors for repeated hint indices, nonzero unused hint-index tails, hint-weight overflow, and `||z||_∞ >= gamma_1 - beta` norm violations.
3. Positive boundary vectors immediately below `gamma_1 - beta` to catch over-rejection.
4. Positive and negative COSE interoperability vectors using the empty RFC 9964 context.
5. Exact transcript-binding tests and classical-component tamper rejection.
6. PQ-component tamper rejection, wrong-PQ-key rejection, and missing-component rejection under hybrid policy.
7. Unprotected-header substitution tests.
8. Signature/public-key length boundary fuzz cases.
9. Deterministic capability identity and selection-witness binding tests.
10. Holochain projection requiring the hybrid witness.
11. Cross-implementation verification against at least one independent ML-DSA implementation.
12. Fuzzing for nested hybrid/COSE parser exhaustion.

Published ACVP signature-verification vectors alone are not sufficient; retain both the reference vectors and the mutation/boundary corpus described above.

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
5. compare capability semantics between the application-level PQ-bound attestation profile and the standardized composite profile;
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
