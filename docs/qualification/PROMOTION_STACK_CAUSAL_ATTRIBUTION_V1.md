# Promotion Stack Causal Attribution v1

This tranche separates causal attribution of the requested pull request from causal attribution of the exact reserved stack operation.

## Requested-effect causality

A provider merge result can establish requested-effect causality only when the preserved provider-operation evidence contains:

    provider async-result source
    status = merged
    non-empty provider UUID
    requested PR number
    exact expected requested PR head SHA
    exact merge method
    exact merge action
    non-empty observed merge commit

This is stronger than merely observing that a PR is merged.

## Exact-stack causality

Exact-stack causal attribution requires all of the requested-effect conditions plus:

    exact reserved stack effect set
    matching provider topology evidence
    provider-side topology CAS

A matching provider observation without provider CAS remains:

    requested-effect-causal
    stack-effect-causal = false

This is intentional. A read followed by a matching read does not become a provider-side conditional mutation fence merely because the values are equal.

## Conservative outcomes

The reference resolver emits exactly one of:

    requested-effect-causal
    stack-effect-causal
    effect-observed-only
    causality-unestablished

The intended ordering is:

    stack-effect-causal
        > requested-effect-causal
        > effect-observed-only
        > causality-unestablished

where the ordering expresses evidence strength, not a numerical score.

## Adversarial distinctions

The corpus keeps these cases separate:

- direct async merge result with exact requested identity;
- direct result plus exact effect set but observation-only topology;
- direct result plus exact effect set and explicit provider topology CAS;
- enqueued result followed by exact durable effect observation;
- already-merged retry with no provider UUID;
- expired/missing async result followed by exact effect observation;
- wrong requested head;
- conflicting requested merge commit;
- incomplete effect set;
- local receipt or non-provider result source.

A local receipt can preserve reconciliation evidence but cannot mint provider causality.

## Claim ceiling

This establishes only deterministic causal-evidence resolution in the provider-free reference model.

It does not establish:

- that GitHub currently supplies topology CAS;
- provider truthfulness;
- production atomicity;
- governance legitimacy;
- successful external promotion.

Related: #7101, #7118, #7119, #7128, #7130, #7132, #7133, #7136.
