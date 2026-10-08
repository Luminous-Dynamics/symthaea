# Promotion Webhook-Derived Stack Effect Set v1

This tranche closes the construction path between authenticated per-PR webhook observations and the exact reserved stacked effect set.

## Derivation

The reference path is:

    authenticated webhook delivery
        ->
    ProviderPullRequestMergeObservationV1
        ->
    PromotionStackEffectSetV1

The stack effect set is constructed from the observations rather than from caller-authored merge commit records.

## Construction requirements

Given a reserved PromotionOperationIdentityV1, the constructor:

- requires every reserved PR exactly once;
- rejects duplicate observations for a PR;
- rejects observations from another repository;
- requires the observed PR/head/merged/action semantics to match each reserved stack entry;
- rejects missing or extra PRs;
- emits effects in the reserved stack order;
- binds the resulting set to the reserved operation digest.

Input observation order does not affect the resulting order. The semantic operation order remains the reserved stack order.

## Provenance preservation

The compact PromotionStackEffectSetV1 remains an effect projection.

The originating webhook observations retain:

    delivery ID
    repository
    event type/action
    observed PR/head
    merge commit
    payload digest

The effect-set projection does not replace those source evidence records.

A future durable evidence capsule should retain both the normalized effect projection and its source observations so the effect can always be traced back to the authenticated delivery evidence.

## Fail-closed cases

The constructor rejects:

- missing lower-stack or requested observation;
- extra unrelated observation;
- duplicate observation for the same PR;
- mixed repository observations;
- wrong observed head;
- semantically invalid observation;
- incomplete derived effect set.

## Causality boundary

An authenticated webhook-derived effect set establishes an exact observed effect set.

It does not establish:

    local async operation caused those effects

and it does not establish:

    exact reserved stack operation was provider-side atomically fenced

Those remain the separate causal and provider-topology boundaries.

## Claim ceiling

This establishes only deterministic derivation of an exact stacked effect projection from authenticated per-PR webhook observations in the provider-free reference model.

It does not establish provider truthfulness, provider-side topology CAS, causal attribution, governance legitimacy, production atomicity, or successful external promotion.

Related: #7118, #7119, #7133, #7136, #7138, #7140, #7149, #7150, #7151, #7152, #7153.