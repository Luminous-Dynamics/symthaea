# Promotion Provider Result Retention v1

GitHub asynchronous merge results are not indefinitely recoverable from their provider UUID.

This tranche treats result retention as evidence liveness rather than as merge-state semantics.

## Evidence states

The reference model distinguishes:

    provider-result-captured-locally
    provider-result-provider-recoverable
    provider-result-expired
    provider-result-unavailable
    invalid-retention-state

A local capture remains available after provider expiration.

An uncaptured provider result is only recoverable while the provider still exposes the result.

## Critical non-equivalence

Provider result expiry does not mean the external effect did not happen.

Likewise:

    later EffectObserved
        !=
    recreated RequestedEffectCausal

When the provider result has expired and no durable local provider-result evidence exists, later durable PR state can establish the effect but cannot recreate the missing operation-to-effect evidence.

## Boundary

The retention model intentionally sits beside the causal resolver.

A `ProviderMergeResultV1` object represents preserved provider evidence in the reference model. `ProviderResultRetentionV1` describes whether that evidence would remain recoverable from the provider or has already been durably captured locally.

A locally authored `PromotionEffectReceipt` remains a separate reconciliation record and cannot become provider evidence merely because the provider result later expires.

## Adversarial cases

The reference tests cover:

- durable local capture surviving well beyond provider retention;
- uncaptured result recoverable before expiry;
- expiry at the retention boundary;
- expiry after the retention window;
- provider unavailability before expiry;
- invalid negative age;
- invalid non-positive retention window;
- exact effect observation after result expiry remaining effect-only.

## Claim ceiling

This establishes only deterministic provider-result evidence retention semantics in the reference model.

It does not establish provider truthfulness, provider-side topology CAS, causal attribution after missing evidence, governance legitimacy, production atomicity, or successful external promotion.

Related: #7101, #7136, #7138, #7139.
