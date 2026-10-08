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
