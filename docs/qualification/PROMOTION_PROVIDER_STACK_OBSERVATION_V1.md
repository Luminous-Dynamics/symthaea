# Promotion Provider Stack Observation v1

This tranche separates the locally reserved stacked-operation identity from the provider-side stack topology observed at dispatch.

## Provider observation

The normalized provider observation records:

    observation source
    provider stack number
    full stack size
    requested pull-request position
    stack base ref
    stack base tip SHA
    ordered provider stack entries
    optional provider observation identity

The observation is evidence about the provider-visible target topology. It is not a provider-enforced mutation fence.

## Exact selected operation

A stacked merge requested for a pull request operates on that pull request and every open pull request below it. Pull requests above the requested pull request remain outside the selected operation.

Therefore the reserved operation identity represents the exact selected downstack:

    [bottom ... requested]

The provider observation may contain additional entries above the requested PR. Those upper entries may grow or change without invalidating the selected-operation match.

The match requires:

    provider stack number == reserved provider stack number
    AND provider stack base ref == reserved base ref
    AND provider stack base tip SHA == reserved base tip SHA
    AND provider requested position == reserved operation depth
    AND provider selected prefix == reserved ordered stack exactly
    AND provider requested PR head == reserved requested PR head

The provider's full stack size must equal the number of observed entries, and the requested position must identify the requested PR.

## Fail-closed boundaries

No provider observation is not a topology match.

The following invalidate the match:

- stack-number drift;
- stack-base ref drift;
- stack-base tip drift;
- requested-position drift;
- lower-stack member insertion/removal;
- lower-stack head drift;
- requested head drift;
- malformed stack size or position;
- provider observation that is internally inconsistent.

An exact final effect set must not retroactively manufacture a missing dispatch-time provider topology observation.

## Observation source

The normalized object can be populated from provider surfaces that expose stack membership, such as REST pull-request/stack reads, GraphQL read-only stack fields, or pull-request webhooks. The observation source and optional observation identifier remain metadata; matching is determined by the exact provider-visible topology fields above.

## Evidence vs causality

A matching provider topology observation does not by itself prove that a later merge was caused by the local operation.

The intended future composition is:

    reserved local identity
        + matching provider dispatch topology
        + provider operation result/provenance
        + exact observed effect set
        -> conditional causal attribution

That causal step remains separately bounded by #7101 and must not be minted by a local receipt.

## Claim ceiling

This establishes only deterministic matching of a normalized provider-side stack observation to a reserved operation identity.

It does not establish:

- provider truthfulness;
- provider-side atomicity beyond the documented API contract;
- causal attribution;
- governance legitimacy;
- production promotion success.

Related: #7096, #7101, #7117, #7118, #7119, #7128.
