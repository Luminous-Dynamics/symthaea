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
    observed-not-cas
    provider-topology-cas

The final class requires explicit CAS evidence; it is not a Boolean switch.

observed-not-cas is the normal conservative class when the provider exposes stack state but does not expose an independent conditional topology fence.

provider-topology-cas is now evidence-gated in the synthetic model. A matching observation alone remains observed-not-cas; the stronger class requires a typed ProviderTopologyCasEvidenceV1 witness bound to the exact reserved operation digest, the exact pre-submit observation digest, and the pre-submit observation sequence. The complete provider observation is content-addressed, so the witness binds the whole observed payload rather than a hand-picked subset. The executable oracle includes negative controls for an unbound witness and sequence-mismatched witness. This witness is still only synthetic evidence in the reference model; it must not be represented as GitHub capability unless an independent provider surface actually supplies the corresponding conditional predicate.

## GitHub interpretation

GitHub's asynchronous stacked merge operation takes the requested pull request and expected head SHA, together with merge parameters. Its documented stacked semantics operate on the open downstack at execution time.

Therefore:

    observation match
        !=
    GitHub topology CAS

The provider-specific adapter must keep this boundary visible unless a future provider surface supplies an independently evidenced conditional topology predicate.

## Claim ceiling

This tranche establishes only a deterministic classification of provider-topology observation and revalidation evidence.

It does not establish:

- provider-side topology CAS for GitHub;
- provider truthfulness;
- atomicity between observation and merge execution;
- causal attribution of the exact reserved stack;
- governance legitimacy;
- successful external promotion.

Related: #7096, #7101, #7117, #7118, #7119, #7128, #7130, #7132.
