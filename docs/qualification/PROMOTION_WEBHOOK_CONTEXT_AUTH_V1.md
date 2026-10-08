# Promotion Webhook Context Authentication v1

This tranche removes circular context verification from the webhook evidence boundary.

## Separate inputs

Webhook evidence has three different inputs:

    signed payload bytes
    received request context
    trusted expected configuration

GitHub documents HMAC-SHA256 verification for the request body and separately exposes hook ID, event type, and delivery ID as request headers.

Therefore a receipt's stored context must not be used as the expected context for its own verification.

## Required verification

The reference parser requires:

- a ProviderWebhookRequestContextV1 representing the received request metadata;
- trusted expected hook ID;
- trusted expected event type;
- trusted expected repository;
- the exact payload bytes and HMAC secret.

The receipt is admitted only when:

    received context == captured receipt context
    AND received context == trusted expected context
    AND HMAC(payload, secret) == captured signature
    AND captured payload digest == SHA-256(payload)

This prevents a caller from changing the receipt's hook/event/repository fields and then asking the receipt to authenticate those same substituted values.

## Delivery identity

The delivery ID is compared between the received context and the captured receipt and remains available for replay/deduplication handling.

Changing the delivery ID, hook ID, event type, or repository in the received context invalidates the captured receipt.

## Important non-equivalence

Valid HMAC proves integrity/authenticity of the signed request body under the configured secret.

It does not, by itself, prove that independently supplied hook/event/repository context is correct.

Conversely, matching context does not prove payload integrity without successful HMAC verification.

## Fail-closed cases

Reject:

- valid HMAC with wrong expected hook;
- valid HMAC with wrong expected event;
- valid HMAC with wrong expected repository;
- received context differing from captured receipt context;
- missing/empty required context;
- payload tampering;
- wrong HMAC secret;
- context-valid webhook presented as an async-operation causal receipt.

## Causality boundary

Non-circular webhook authentication enables trusted source evidence to enter the semantic merge-effect parser.

It still does not establish that a specific async promotion operation caused the later pull-request effect.

That remains the independent causal boundary.

## Claim ceiling

This establishes only deterministic, non-circular webhook context authentication semantics in the provider-free reference model.

It does not establish provider attestation, provider truthfulness, provider-side topology CAS, causal attribution, governance legitimacy, production atomicity, or external promotion success.

Related: #7149, #7150, #7152, #7154, #7155, #7157.