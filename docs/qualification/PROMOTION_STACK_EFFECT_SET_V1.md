# Promotion Stack Effect Set v1

This tranche makes stacked terminal effect completeness explicit in the provider-free reference model.

## Effect-set identity

A stacked promotion is not adequately represented by one observed merge commit.

The reference model represents the terminal observation as:

    operation_identity_digest
    ordered effects:
      PR number
      expected PR-head SHA
      observed merge commit

The observed effect order must match the reserved stack order exactly.

## Completeness rule

An observed effect set is complete only when:

    operation identity matches
    AND effect cardinality matches stack cardinality
    AND each PR appears exactly once
    AND each expected head SHA matches
    AND every observed merge commit is non-empty
    AND observed ordering equals reserved ordering

Missing, extra, duplicated, reordered, or head-mismatched effects fail closed.

## Why observation stays separate from causality

This object proves completeness of the effects that were actually observed.

It does not prove that every observed effect was caused by the local provider operation. That remains the separate #7101 causal-attribution boundary.

Likewise, provider group atomicity does not authorize the local system to synthesize per-PR merge commits that were never observed.

## Claim ceiling

This establishes only deterministic completeness checking for a synthetic stacked-effect set.

It does not establish:

- provider truthfulness;
- causal attribution;
- production atomicity;
- governance legitimacy;
- successful external promotion.

Related: #7096, #7117, #7101, #7087.
