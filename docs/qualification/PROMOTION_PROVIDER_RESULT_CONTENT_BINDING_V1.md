# Promotion Provider Result Content Binding v1

This tranche binds the normalized provider async merge result to the exact captured response bytes used as its evidence source.

## Content binding

ProviderMergeResultV1 now carries result_payload_digest.

Direct requested-effect causal resolution requires:

    result_payload_digest == captured raw-byte digest
    AND source authentication method == authenticated-api-channel

Therefore a normalized merge result cannot be attached to an arbitrary evidence envelope merely because the fields look compatible.

## Why both conditions matter

Capture integrity prevents the normalized result from being rebound to different captured bytes.

Source-class binding prevents a webhook delivery from being treated as the direct async merge-operation result.

These are separate from provider attestation.

## Missing and conflicting evidence

The resolver rejects:

- missing result payload digest;
- result digest differing from captured bytes;
- locally authored normalized result without preserved provider evidence;
- webhook evidence used as direct async-operation causality even when its bytes are identical to the normalized result's digest;
- untrusted source authentication even when the content digest matches.

## Retention interaction

A captured provider response may remain useful as direct evidence after the provider no longer exposes its async UUID.

If the response was never durably captured, expiry removes that direct result evidence and later effect observation remains effect-only.

## Non-equivalence

A content digest proves byte association, not semantic truth.

An authenticated API channel proves the evidence arrived through the configured authenticated transport, not that the provider's application-level claim is truthful.

A verified provider result therefore still does not establish governance legitimacy, provider truthfulness, or an exact-stack provider-side CAS.

## Claim ceiling

This establishes deterministic content binding between a normalized provider-result object and its preserved response evidence in the provider-free reference model.

It does not establish provider attestation, provider truthfulness, provider-side topology CAS, causal attribution beyond the existing resolver, governance legitimacy, production atomicity, or external promotion success.

Related: #7138, #7139, #7140, #7149, #7150, #7159, #7160.