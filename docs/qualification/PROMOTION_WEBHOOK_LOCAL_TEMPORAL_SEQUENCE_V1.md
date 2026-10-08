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
