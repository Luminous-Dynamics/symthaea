# Promotion Webhook Context Authentication v1

This tranche removes circular context verification from the webhook evidence boundary.

## Separate inputs

Webhook evidence has three distinct inputs:

    signed payload bytes
    received request context
    trusted expected configuration

GitHub documents HMAC-SHA256 verification for the request body and separately exposes hook ID, event type, and delivery ID as request headers.

Therefore the captured receipt context must not be used as the expected context for verifying that same receipt.

## Required verification

The reference parser requires:

- ProviderWebhookRequestContextV1 representing the received request metadata;
- trusted expected hook ID;
- trusted expected event type;
- trusted expected repository;
- exact payload bytes and HMAC secret.

Verification requires:

    received context == captured receipt context
    AND received context == trusted expected context
    AND HMAC(payload, secret) == captured signature
    AND captured payload digest == SHA-256(payload)

This makes context comparison non-circular.

## Delivery identity

The delivery ID is compared between received context and the captured receipt and remains available for replay and deduplication handling.

## Important non-equivalence

Valid HMAC proves integrity/authenticity of the signed request body under the configured secret.

It does not by itself prove that independently supplied hook, event, or repository context is correct.

Conversely, matching context does not prove payload integrity without successful HMAC verification.

## Fail-closed cases

Reject:

- valid HMAC with wrong expected hook;
- valid HMAC with wrong expected event;
- valid HMAC with wrong expected repository;
- received context differing from captured receipt context;
- empty expected context;
- payload tampering;
- wrong HMAC secret;
- context-valid webhook presented as an async-operation causal receipt.

## Causality boundary

Non-circular webhook authentication enables trusted source evidence to enter semantic merge-effect parsing.

It still does not establish that a specific async promotion operation caused the later pull-request effect.

## Claim ceiling

This establishes only deterministic, non-circular webhook context authentication semantics in the provider-free reference model.

It does not establish provider attestation, provider truthfulness, provider-side topology CAS, causal attribution, governance legitimacy, production atomicity, or external promotion success.

Related: #7149, #7150, #7152, #7154, #7155, #7157.