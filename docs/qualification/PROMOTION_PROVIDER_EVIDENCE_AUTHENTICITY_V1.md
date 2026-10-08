# Promotion Provider Evidence Authenticity v1

This tranche separates preservation of provider response bytes from source authentication and provider attestation.

## Three evidence layers

A captured provider result may have three independent properties:

    CaptureIntegrity
    SourceAuthentication
    ProviderAttestation

These are intentionally not represented as one generic authenticated flag.

### CaptureIntegrity

Capture integrity binds:

    exact raw bytes
    byte-level digest
    durable storage identity
    positive capture sequence

A local digest proves preservation of the captured bytes. It does not prove their origin.

### SourceAuthentication

The reference model recognizes two provider-source authentication mechanisms:

    authenticated-api-channel
    webhook-hmac-verified

A local-untrusted object is never accepted as preserved provider evidence.

For webhook evidence, GitHub's documented `X-Hub-Signature-256` mechanism provides HMAC-SHA256 verification of the request body, and `X-GitHub-Delivery` supplies globally unique delivery identity. The webhook signature is evidence about the delivery, not a proof that an unrelated asynchronous merge operation caused an effect.

### ProviderAttestation

Provider attestation remains a separate layer.

HTTPS/API authentication and webhook HMAC validation are not relabeled as a provider-signed attestation of the merge result.

The reference model does not claim provider attestation for GitHub merge results unless a future independently verifiable provider attestation mechanism is added.

## Delivery replay and conflict

Webhook delivery identity is tracked independently from payload bytes.

For an already-seen delivery ID:

    same payload digest -> duplicate-identical
    different payload digest -> delivery-id-conflict

A conflicting delivery ID fails closed. A redelivery of the same authenticated payload may be recognized as an identical delivery and deduplicated.

Signature verification must occur before any delivery becomes trusted evidence.

## Causal boundary

The causal resolver consumes preserved provider evidence for a direct provider async result.

Therefore:

    local capture integrity alone
        -> no requested-effect causality

    webhook HMAC authenticity alone
        -> no merge-operation causality

    authenticated API result
        + exact provider operation identity
        + exact requested PR/head/merge parameters
        + provider result = merged
        -> requested-effect-causal in the synthetic reference model

This is a semantic model, not a claim that an HTTPS channel or webhook proves more than GitHub's documented authentication mechanism establishes.

## Relationship to retention

A provider result may remain causally useful after the provider stops retaining the UUID if the exact provider result was durably captured locally.

Conversely, provider expiry without a durable captured result permanently removes that direct result evidence. Later PR merge observation remains effect evidence only.

## Required fail-closed conditions

- forged local provider-result JSON;
- digest over forged local bytes;
- non-durable capture;
- missing provider identity for transport-authenticated evidence;
- invalid webhook signature;
- tampered payload under a reused signature;
- delivery ID reused for different payload bytes;
- webhook evidence presented as direct merge-operation evidence;
- local reconciliation receipt presented as provider evidence;
- provider attestation asserted without an independently verified attestation layer.

## Claim ceiling

This establishes only deterministic evidence-layer semantics in the provider-free reference model.

It does not establish provider cryptographic attestation where none exists, provider truthfulness, provider-side topology CAS, production atomicity, governance legitimacy, or successful external promotion.

Related: #7101, #7136, #7138, #7139, #7140, #7149.
