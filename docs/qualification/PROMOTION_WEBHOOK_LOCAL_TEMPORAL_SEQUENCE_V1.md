# Promotion Webhook Temporal Evidence v2

This tranche hardens the cross-domain clock boundary that was intentionally left abstract in the first bounded-time model.

## Clock relation is evidence, not a boolean

A usable relation is now a frozen composition of three distinct records:

    ClockRelationEvidenceV1
        ↓
    ClockRelationVerificationV1
        ↓
    ClockRelationTrustSnapshotV1
        ↓
    ClockRelationV1

The evidence records where the relation came from, its provider/local clock domains, its worst-case skew bound, when that bound was measured, its local validity interval, an evidence identity, and the source-authentication class.

The verification record binds to the exact evidence digest and records the verifier identity, verifier-policy digest, verification time, validity interval, freshness ceiling, and revocation epoch.

The trust snapshot binds the verifier policy and revocation epoch used for the admission decision. This is a local trust snapshot, not a claim that the external time source is inherently truthful.

A generic verified=true flag is no longer sufficient.

## Freshness and validity

The skew bound is interpreted as a conservative bound over the entire declared validity interval, not as an indefinitely reusable instantaneous measurement.

The relation is admitted only when:

    relation evidence covers dispatch and observation
    verification covers dispatch and observation
    verification time <= dispatch
    observation - verification_time <= freshness_max_age
    the trust snapshot existed by dispatch

Therefore a relation that was valid yesterday cannot be silently reused for today's promotion merely because its max_skew_ms still looks plausible.

A relation that becomes stale before local observation fails closed as:

    clock-relation-stale

A validity-window miss fails closed as:

    clock-relation-outside-validity

Verification that occurs after dispatch fails closed as:

    clock-relation-established-after-dispatch

The trust snapshot must also predate the verifier decision. Otherwise the verifier could appear to have made an accepted decision using a trust state that did not yet exist.

That fails closed as:

    clock-relation-trust-snapshot-after-verification

Evidence measurement must not occur after verification. Such a record would amount to future evidence being used to justify an earlier decision and is structurally invalid.

## Policy drift and revocation

The operation compares the verification's policy digest and revocation epoch against an explicit local trust snapshot.

Policy mismatch is:

    clock-relation-policy-drift

Revocation-epoch mismatch is:

    clock-relation-revoked

This does not retroactively rewrite historical evidence. The intended model is a decision-time snapshot: a relation is qualified under the exact trust state that existed for that operation. A later trust-state change can prevent new admission or force reconciliation to treat a fresh decision as unavailable without changing the old bytes.

## Why this shape

Secure time protocols treat freshness as a first-class property. Current Roughtime protocol work binds a server response to a fresh client nonce, includes an explicit uncertainty radius, and uses time-limited online keys; NTS similarly has explicit concerns around certificate validity and replay. RFC 8633 also recommends multiple independent time sources and monitoring because authenticated packets alone do not make the reported time infallible. These mechanisms motivate the data-model separation here, but they do not make this deterministic promotion oracle a real time-synchronization implementation.

## Remaining boundary

Even a provenance-complete and fresh clock relation still does not establish:

    provider truthfulness
    immunity to network delay/asymmetry attacks
    cryptographic synchronization of the GitHub clock to the local clock
    provider-side topology CAS
    operation causality
    production atomicity
    promotion success

The bounded temporal gate therefore remains a qualification predicate, not an oracle of universal time.

The local monotonic sequence remains a separate ordering primitive. Its fixture semantics were also corrected so an explicit missing sequence stays missing instead of being silently replaced by a default sequence.

Related: #7176, #7180.

## Source-message lineage

The clock relation is now bound to a fresh source-message lineage:

    promotion operation identity
        +
    fresh challenge
        ↓ exact challenge digest
    provider time response
        ↓ exact response digest
    verifier attestation
        ↓
    ClockRelationEvidenceV1

The challenge carries a 32-byte non-zero nonce, the exact promotion operation identity digest, a challenge validity interval, and a pinned trust-anchor identifier.

The response binds the exact challenge digest, provider clock domain, provider time, uncertainty radius, response identity/digest, and the same trust anchor.

The verifier attestation binds the exact challenge and response digests, trust anchor, verifier identity and policy, verification time, validity interval, cryptographic verification scheme, and accepted decision.

The timing artifact carries the same promotion operation identity digest and rejects a time-source challenge belonging to another operation as clock-relation-operation-identity-mismatch.

This is a stronger provenance shape than a free-form source-authentication string. It still does not implement the external signature verification primitive inside this provider-free oracle: signature_verified represents an external verifier decision and is not itself claimed to be cryptographic proof.

Roughtime provides the relevant protocol precedent: responses are bound to a fresh client nonce, include a timestamp and uncertainty radius, and are verified against configured long-term trust roots. RFC 10049 also explicitly distinguishes a valid response from proof that the timestamp itself is globally correct. citeturn577066search2turn577066search1


## Local sequence provenance

The local monotonic sequence is no longer sufficient merely because its three integers are ordered.

Temporal admission now requires sequence provenance:

    LocalTemporalSequenceEvidenceV1
        ↓
    LocalTemporalSequenceV1
        ↓
    ProviderWebhookEffectTimingV1

The evidence binds the exact promotion operation identity, local sequence source identity, source generation, capture identity, sequence triple, sequence-triple digest, and source kind.

A timing artifact carrying a correctly ordered sequence from another operation is rejected as local-sequence-operation-identity-mismatch.

Clock-relation evidence also commits the source challenge, response, and verifier-attestation digests into its canonical evidence digest. This prevents mutation of one underlying source-message layer from leaving the outer clock-evidence identity unchanged.

A correctly ordered sequence with missing provenance is rejected as local-sequence-provenance-missing. A malformed or internally inconsistent provenance record is rejected as local-sequence-source-invalid.

The reference model deliberately does not claim a kernel or hardware attestation of the sequence source. It establishes provenance and binding semantics so a bare caller-authored sequence cannot satisfy the full cross-domain timing predicate.

Rust's current Instant documentation describes Instant as monotonically nondecreasing but not necessarily steady, and notes that platform or virtualization bugs can still violate practical monotonicity guarantees. That supports keeping monotonic ordering separate from wall-clock accuracy and source trust. citeturn670450search1turn670450search0


## Cryptographic verification receipt

The source attestation no longer uses a caller-authored signature_verified boolean.

It now records a typed verification receipt containing:

    signed payload digest
    signature digest
    public-key digest
    signature algorithm
    signature context
    key role
    trust-anchor identifier
    verifier identity
    verifier implementation digest
    verifier policy digest
    verification time and validity
    verification result

The exact receipt digest becomes part of the surrounding clock evidence identity.

This still does not implement Ed25519 or another signature primitive inside the provider-free oracle. The receipt represents the output of an external cryptographic verifier, but the input/output transcript is now explicit enough to be independently reconstructed and checked against captured artifacts.

Roughtime's response validation requires the client to verify the certificate's long-term signature, the request-derived Merkle path, and the response signature over the SREP value, while explicitly stating that a valid response does not prove the timestamp itself is correct. This distinction is preserved here. citeturn946860search1

The schema remains algorithm-agile rather than declaring post-quantum readiness. NIST's ML-DSA standard is available as a separate signature family, but introducing a string such as ML-DSA-65 into this receipt must not be interpreted as implementation support; the verifier policy must establish the allowed algorithm set. citeturn946860search2


## Independent cryptographic replay

The receipt schema is now exercised against an actual Ed25519 signature rather than only synthetic digests.

The captured vector is:

    docs/qualification/fixtures/CLOCK_RELATION_CRYPTO_RFC8032_V1.json

and the independent replay harness is:

    scripts/verify_clock_relation_crypto_v1.py

The harness is intentionally separate from the provider-free promotion oracle. It independently reconstructs the SHA-256 digests of the exact signed payload, signature, and public key; verifies the Ed25519 signature equation; confirms the typed receipt says `signature-valid`; and rejects mutated payload, signature, and public-key material.

This establishes an important distinction:

    receipt schema integrity
        !=
    independent verification of captured cryptographic material
        !=
    proof that a real external provider response was truthful

The implementation is a deterministic qualification/test harness, not production cryptographic code. It is not constant-time, never handles private signing keys, and does not claim to replace a maintained cryptographic library.

The vector follows the Ed25519 verification structure specified by RFC 8032, including decoding R and A, requiring S < L, computing the challenge from R || A || M, and checking the cofactor-cleared group equation. RFC 8032 also supplies standardized Ed25519 test vectors. citeturn348751search3turn348751search5

Roughtime remains a separate protocol-level source of truth about how a real time response should be validated: RFC 10049 requires validation of the long-term certificate signature, request-derived Merkle proof, and response signature, and explicitly says a valid response does not by itself prove the timestamp is correct. citeturn348751search0turn348751search2

Hosted qualification remains fail-closed: the PR-triggered workflows for this draft branch are still not a qualification claim. The new replay step only becomes executable as part of the existing trusted-default-branch manual lab.


## Independent-source quorum

A single authenticated clock source is no longer treated as sufficient for the stronger temporal disposition.

`ClockRelationSourceSetV1` requires a policy minimum of three sources and requires each source to carry distinct:

    source_id
    operator_id
    trust_anchor_id

The set also requires a common provider clock domain and a non-empty intersection of the sources' uncertainty intervals.

The resulting predicate is:

    independent-source quorum
        +
    common time interval
        +
    per-source cryptographic validity
        ->
    clock-source-quorum-admissible

The model deliberately returns failure states rather than selecting a preferred source:

    clock-source-policy-too-weak
    clock-source-quorum-insufficient
    clock-source-independence-invalid
    clock-source-domain-mismatch
    clock-source-time-disagreement

Distinct source identifiers alone are not enough: the operator and trust-anchor identities are also required to be distinct. This is a conservative model of source independence, not proof that two supposedly separate operators are actually independent.

RFC 10049 specifies that Roughtime clients use a list with at least three operational servers not run by the same parties, and its multi-server mode checks reported times for causal consistency. The present model borrows the anti-single-source principle and interval reasoning while remaining a local qualification model rather than a Roughtime implementation. citeturn348751search0turn348751search2

The source-set agreement predicate still does not establish global clock correctness, network-path symmetry, provider honesty, or causal attribution. It only makes single-source temporal evidence insufficient for the stronger disposition.


## Causally chained multi-source measurement

The source quorum now has a second evidence layer: a sequential measurement chain.

`ClockSourceMeasurementSequenceV1` requires two rounds over the same ordered source set. Each round contains at least three sources, and each step binds:

    operation identity
    round and sequence index
    source identity
    exact relation digest
    request-nonce digest
    previous-response digest
    fresh chain-random digest
    deterministic chain-link digest
    local receive time

The first step in each round starts from the fresh request nonce. Each subsequent step commits to the immediately preceding response digest plus fresh chain entropy. Replaying an earlier response across rounds is rejected.

The model also checks causal ordering between sequentially received responses using the uncertainty intervals. A response that is already too late to be causally compatible with the next response produces:

    clock-source-causal-order-contradiction

Other chain failures are explicit:

    clock-source-measurement-quorum-insufficient
    clock-source-measurement-round-mismatch
    clock-source-measurement-invalid
    clock-source-measurement-replay
    clock-source-measurement-chain-mismatch

This is deliberately a qualification-model abstraction rather than an implementation of the Roughtime wire format. RFC 10049 requires at least three operational servers not run by the same parties, sequentially queries them, repeats the sequence twice in the same order, chains later query nonces to the prior response plus fresh randomness, and checks pairwise causal consistency using the reported midpoint and radius. citeturn312740search0turn312740search1

The important boundary is preserved:

    authenticated + independently verified source
        +
    independent multi-source agreement
        +
    causally linked repeated measurement
        ->
    stronger temporal evidence

but not:

    universal time correctness
    network-path symmetry proof
    provider honesty
    proof of operation causality
    production promotion success

The chain digest is order-sensitive because measurement order is semantically meaningful, unlike the source-set digest which is intentionally order-invariant.
