# RFC 9942 post-quantum and hybrid signature boundary

Status: proposed implementation boundary
Date: 2026-10-08

## Decision

Symthaea should migrate long-lived evidence from classical-only public-key signatures toward a **PQ/T hybrid** model, but it should **not replace Ed25519 or invent a composite COSE algorithm today**.

The target durable policy is:

- **Ed25519** remains available for classical interoperability and existing protocol compatibility.
- **ML-DSA-65** is the preferred post-quantum signature component for new durable evidence.
- A **hybrid durable witness requires both classical and post-quantum verification** over the same exact transcript/artifact.
- A failure of either component is a failure of the hybrid witness. There is no partial-pass durable state.
- The hybrid witness binds exact wire identity, algorithm identifiers, verification-key identities, signature identities, and the domain-separated transcript.
- Holochain durable projection should ultimately accept only this stronger hybrid witness for evidence classes designated quantum-resistant.

This is intentionally an application/proof-boundary decision rather than a change to the RFC 9942 wire format.

## Why not simply replace Ed25519?

Ed25519 is still useful because it is compact, fast, mature, and broadly interoperable. Replacing it immediately would cause unnecessary compatibility breakage and would not solve the migration problem cleanly.

It is, however, a classical public-key signature. A future sufficiently capable quantum computer would invalidate its security assumptions. NIST's current migration guidance says organizations should begin moving to quantum-resistant cryptography now, with quantum-vulnerable algorithms ultimately being deprecated and removed from NIST standards by 2035, and higher-risk systems moving earlier.

For Symthaea, the relevant evidence is deliberately long-lived: receipts, transparency proofs, capability hashes, and Holochain projections are intended to remain meaningful after the event that created them. Those are exactly the artifacts where post-quantum migration matters most.

## Why hybrid rather than PQ-only?

A PQ-only migration would give future quantum resistance but would discard the mature classical verification path too early.

A hybrid construction gives two independent cryptographic anchors:

1. a classical algorithm for present-day ecosystem interoperability;
2. a post-quantum algorithm for future resistance.

For the durable evidence boundary, Symthaea should use an **AND rule**: both signatures must verify before a hybrid witness can be constructed.

This is stronger than accepting either signature independently and, critically, prevents a verifier from silently falling back to the classical component when the PQ component is unavailable or invalid.

The generic IETF terminology permits several hybrid constructions (parallel, composite, or nested) and emphasizes that security depends on the combiner and attacker model. This implementation therefore treats the combiner as part of the verified capability rather than as an informal policy outside the cryptographic transcript.

## RFC 9942 compatibility constraint

RFC 9942 defines receipts as COSE Single Signer Data Objects and leaves the choice of signature algorithm to the profile/application.

The current COSE standardization path already gives ML-DSA-44/65/87 registered algorithm identifiers. In particular:

- ML-DSA-44: COSE alg -48
- **ML-DSA-65: COSE alg -49**
- ML-DSA-87: COSE alg -50

ML-DSA-65 is therefore safe to adopt as a **pure ML-DSA COSE signature algorithm** without inventing a new algorithm identifier.

By contrast, the current IETF JOSE/COSE composite-signature draft is still a working document and its proposed composite COSE algorithm values remain **TBD**. Symthaea MUST NOT allocate or hard-code one of those temporary values as though it were an IANA assignment.

Therefore the migration should proceed in two layers:

### Layer A — standards-aligned ML-DSA

Add a feature-gated ML-DSA-65 verifier using the registered COSE algorithm identifier -49. Preserve the current exact COSE Sig_structure and provenance accounting.

### Layer B — hybrid durable witness

Until the composite COSE algorithms are finalized and implemented interoperably, compose **two independently verifiable artifacts at the Symthaea application boundary**:

- the existing classical signature artifact;
- a separately encoded ML-DSA-65 signature over the same canonical/domain-separated transcript.

The durable witness proves that both were checked against the same exact artifact identity. This avoids pretending that a non-standard two-signature COSE encoding is an RFC 9942 feature.

Once a final interoperable composite COSE representation exists, it can be considered as a wire-level optimization. It should not be a prerequisite for establishing the stronger internal trust boundary.

## Exact transcript requirements

The hybrid witness must bind, at minimum:

- protocol/domain identifier;
- exact signed artifact bytes or exact artifact hash;
- classical algorithm identifier;
- post-quantum algorithm identifier;
- classical verification-key hash;
- post-quantum verification-key hash;
- classical signature hash;
- post-quantum signature hash;
- any external AAD/context hash;
- payload transport mode where relevant;
- RFC 9942 Receipt/Signature_With_Receipt exact-wire fingerprints;
- any proof capability identifiers that were used to establish the semantic state.

A hybrid capability MUST NOT be constructed by independently verifying two signatures and then merely assuming they refer to the same statement. The binder must fail closed on any transcript, wire, key, algorithm, or context mismatch.

## Interaction with the current Symthaea implementation

The current RFC 9942 implementation already has unusually strong provenance discipline:

- parsed COSE objects retain exact source bytes;
- protected and unprotected header bytes are fingerprinted separately;
- payload, external AAD, signature, proof, Receipt, collection, and outer wrapper identities are retained;
- the verified-selection witness is private and cannot be manufactured by callers;
- durable Holochain projection crosses that witness boundary.

Those mechanisms are the correct foundation for PQ hybridization.

One important boundary remains: **the current verified-selection durable path is classical-cryptographic, not post-quantum**. The current atomic selection constructor uses ES256 for the outer and selected Receipt path. Ed25519 verification APIs exist as additional classical verification paths, but neither Ed25519 nor ES256 by itself is a quantum-resistant durable evidence policy.

The migration should therefore strengthen the witness rather than merely add another standalone verifier.

## Backend decision

ML-DSA-65 is the target parameter set because it is the middle FIPS 204 security category and is explicitly registered for COSE.

The Rust implementation choice must remain conservative:

- use a final FIPS 204 implementation;
- pin the dependency exactly;
- keep it feature-gated;
- add cross-implementation known-answer/interop tests before treating the verifier as evidence-grade;
- do not claim independent cryptographic audit or FIPS 140 validation merely because the algorithm itself is standardized.

The currently available Rust implementations examined for this work are not independently audited. This means **standardized algorithm != audited implementation**. The implementation layer therefore needs its own qualification evidence.

## Migration policy

Recommended policy states:

```
CLASSICAL_ONLY
    Existing compatibility / transitional artifacts.

PQ_ONLY
    New artifacts where classical interoperability is unnecessary.

HYBRID_REQUIRED
    Durable evidence, identity roots, release/attestation roots,
    and any artifact whose validity is expected to survive a
    future quantum break of classical public-key cryptography.
```

The durable default should move to **HYBRID_REQUIRED**, with explicit policy exceptions rather than silent downgrade.

A verifier encountering a hybrid-required artifact with only Ed25519/ES256 available should return an explicit policy failure, not a classical success.

## Non-goals

This decision does not:

- invent a new COSE algorithm number;
- redefine RFC 9942 receipt semantics;
- claim that ML-DSA is the only long-term PQ signature that will ever be needed;
- imply that a quantum-resistant signature makes a statement truthful;
- replace the existing exact-wire/provenance model;
- treat cryptographic verification as authorization.

The existing distinction remains essential:

**signature validity != truth != authorization.**

The purpose of the hybrid layer is narrower: preserve cryptographic authenticity and artifact integrity against the failure of either one of its component security assumptions, while keeping present-day interoperability.

## Qualification gates

Before a hybrid durable path is promoted:

1. RFC 9964 ML-DSA-65 COSE algorithm -49 positive and negative vectors pass.
2. At least one independent/reference implementation verifies the emitted signature.
3. Protected-header, payload, AAD, key, signature, and exact-wire mutations fail or produce a distinct capability.
4. Classical-valid/PQ-invalid and PQ-valid/classical-invalid combinations never produce a hybrid witness.
5. Reordering or substitution of the two component artifacts never preserves hybrid identity.
6. The hybrid witness cannot be constructed through a public raw constructor.
7. Holochain projection accepts only the hybrid witness for HYBRID_REQUIRED policy.
8. CI records exact-head evidence for the complete vector/fuzz/interop suite.
9. The PQ backend's audit status is recorded explicitly; no “secure” or “FIPS validated” claim is inferred from algorithm standardization alone.

## Current recommendation

**Yes: adopt PQ/T hybrid for the durable evidence boundary.**

Do not remove Ed25519 now. Do not make Ed25519 the durable root of new quantum-resistant evidence. Add ML-DSA-65 as the standards-aligned PQ component, and make the future durable witness require **both** classical and PQ verification over one exact, domain-separated transcript.

The existing RFC 9942 parser/fuzz qualification remains a separate change stream. This PQC migration is intentionally isolated so the current qualification run can finish without being invalidated by an unrelated cryptographic dependency migration.
