# ASSURE-000 — Generic Claim–Evidence Qualification Kernel

## Governing theorem

```text
evidence exists
    != claim established
    != stronger claim established
    != deployment authority
```

ASSURE-000 defines a domain-neutral vocabulary for exact subject identity, claims, evidence provenance, support strength, negative findings, claim ceilings, and invalidation conditions.

## Positive support ladder

- `Structural`
- `Observed`
- `CausallySupported`
- `FunctionallySupported`
- `IndependentlyReproduced`
- `DeploymentQualified`

The ladder is not a generic score. Each successor tier requires explicit predicates. `DeploymentQualified` additionally requires an explicit deployment envelope. `IndependentlyReproduced` and stronger support require evidence with a verifier distinct from producer and executor.

## Negative and orthogonal findings

- `NotDemonstrated`
- `Contradicted`
- `Inconclusive`
- `Expired`
- `Invalidated`

These are intentionally not placed below `Structural` on one ordinal scale. A contradiction is not a weak positive result, and inconclusive evidence is not a scientific refutation.

## Exact identity

`SubjectManifest` canonicalizes its component order and content-addresses the result with SHA-256. `Claim` separately binds the exact subject identity, claim ID, textual proposition, and declared scope.

Evidence artifacts bind both subject and claim identities. Rebinding evidence to another subject or claim fails closed.

## Claim ceiling

`QualificationPlan::maximum_support` is a hard upper bound. A proposed result above that ceiling is rejected rather than silently promoted.

## Invalidation

Qualification results retain explicit invalidation conditions. Matching a declared condition converts a positive result to `Negative(Invalidated { ... })` rather than preserving stale qualification.

## Deliberate nonclaims

ASSURE-000 does not establish certification, compliance, agent safety, deployment authorization, independent audit, runtime isolation, red-team completeness, or regulatory conformity. It provides the semantic substrate later assurance campaigns can use to make narrower evidence-backed claims.
