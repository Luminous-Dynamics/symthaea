# Provider Stack Topology CAS Boundary v1

This tranche makes the remaining time-of-check/time-of-use boundary explicit.

## Observation is not a provider CAS

A matching provider stack observation establishes correspondence at an evidence point.

It does not establish that the provider will execute the eventual merge against that same topology.

The local operation digest is not a conditional mutation predicate accepted by the provider.

## Revalidation

The reference model permits two provider observations:

    initial observation
        -> pre-submit revalidation
        -> provider operation

Both observations must match the same reserved operation identity.

A changed or malformed pre-submit observation fails closed.

Observation sequence values must increase so that an older observation cannot be relabeled as a newer revalidation.

## Claim classes

The model distinguishes:

    unobserved
    initial-mismatch
    unrevalidated
    stale-before-submit
    invalid-observation-order
    invalid-observation-sequence
    invalid-provider-operation-options
    observed-not-cas
    provider-topology-cas

The final class requires explicit CAS evidence; it is not a Boolean switch.

observed-not-cas is the normal conservative class when the provider exposes stack state but does not expose an independent conditional topology fence.

provider-topology-cas is now evidence-gated in the synthetic model. A matching observation alone remains observed-not-cas; the stronger class requires a typed ProviderTopologyCasEvidenceV1 whose predicate digest identifies an explicit ProviderTopologyCasPredicateV1 over the exact reserved operation identity, exact pre-submit observation, and pre-submit sequence. The evidence carries a canonical ProviderTopologyCasSubmissionV1 binding the provider-assigned operation handle to one exact request, plus a canonical ProviderTopologyCasProviderResultV1 that binds the later result to that exact submission and request. Crucially, provider result admission and predicate enforcement are separate facts: an accepted submission is not enough to establish CAS semantics. The provider result therefore carries a typed ProviderTopologyCasExecutionV1 witness whose exact request digest, submission digest, predicate digest, provider operation identity, execution source, and enforced result must all validate; the execution witness itself is content-addressed and its digest is bound into the provider result. ProviderTopologyCasEvidenceV1 carries the exact submission and provider-result digests, so neither acceptance nor enforcement can be reconstructed from a weaker local receipt or silently substituted. The provider request contains the requested PR, expected head, merge method, merge action, bypass mode, and predicate digest. The executable oracle includes negative controls for missing enforcement, unenforced results, execution-digest splicing, execution-field splicing, wrong observation, sequence drift, non-provider source, empty operation identity, non-accepted admission, provider-result digest splicing, and result-field splicing. This witness chain remains synthetic reference-model evidence; it must not be represented as GitHub capability unless an independent provider surface actually supplies the corresponding conditional predicate and its enforcement result.
## GitHub interpretation

GitHub's asynchronous stacked merge operation takes the requested pull request and expected head SHA, together with merge parameters. Its documented stacked semantics operate on the open downstack at execution time.

Therefore:

    observation match
        !=
    GitHub topology CAS

The provider-specific adapter must keep this boundary visible unless a future provider surface supplies an independently evidenced conditional topology predicate. In particular, the async merge UUID and expected requested head are not, by themselves, a topology-CAS predicate over the selected downstack. Because GitHub also accepts `bypass_rules`, the promotion operation identity binds that option rather than treating normal-rule and bypassed execution as interchangeable requests. For a multi-PR stacked operation, the reference model rejects `bypass_rules=true`: GitHub's stacked-PR documentation permits bypass-rule merging only for the bottom pull request, not for merging the whole stack.

## Claim ceiling

This tranche establishes only a deterministic classification of provider-topology observation, revalidation, and explicitly represented conditional-predicate evidence.

It does not establish:

- provider-side topology CAS for GitHub;
- provider truthfulness;
- atomicity between observation and merge execution;
- causal attribution of the exact reserved stack;
- governance legitimacy;
- successful external promotion.

Related: #7096, #7101, #7117, #7118, #7119, #7128, #7130, #7132.
