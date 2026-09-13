# ASSURE-000 — Generic Claim–Evidence Qualification Kernel

## Governing theorem

```text
evidence exists
    != claim established
    != stronger claim established
    != deployment authority
```

ASSURE-000 defines a domain-neutral vocabulary for exact subject identity, claims, evidence provenance, support strength, negative findings, claim ceilings, and invalidation conditions.

## Deterministic identities

The kernel content-addresses:

- `SubjectManifest` — canonical component order + optional deployment envelope;
- `Claim` — exact subject, claim ID, statement, and scope;
- `QualificationPlan` — exact claim, maximum support ceiling, and invalidation set;
- `QualificationResult` — exact subject/claim/plan identity, evidence IDs, outcome, ceiling, and invalidation set.

Evidence cannot be rebound to a different subject or claim. Duplicate evidence IDs fail closed rather than being silently deduplicated.

## Positive support ladder

- `Structural` requires architecture-inspection evidence;
- `Observed` requires observation evidence;
- `CausallySupported` requires controlled-intervention evidence;
- `FunctionallySupported` requires controlled-intervention plus functional-benchmark evidence;
- `IndependentlyReproduced` additionally requires independent-reproduction evidence with a verifier distinct from producer and executor;
- `DeploymentQualified` additionally requires runtime-receipt evidence and an explicit deployment envelope.

The ladder is not a generic score. Each successor tier has explicit predicates, and `QualificationPlan::maximum_support` is a hard claim ceiling.

## Negative and orthogonal findings

- `NotDemonstrated`
- `Contradicted`
- `Inconclusive`
- `Expired`
- `Invalidated`

These are intentionally not placed below `Structural` on one ordinal scale. A contradiction is not a weak positive result, and inconclusive evidence is not a scientific refutation.

## Invalidation

Qualification results retain an explicit invalidation set. Matching a declared condition changes the result to `Negative(Invalidated { ... })` and therefore changes the result identity. ASSURE-004 will later add predecessor/requalification lineage; ASSURE-000 deliberately does not imply that a mutated result preserves historical qualification authority.

## Deliberate nonclaims

ASSURE-000 does not establish certification, compliance, agent safety, deployment authorization, independent audit, runtime isolation, red-team completeness, or regulatory conformity. It provides the semantic substrate later assurance campaigns can use to make narrower evidence-backed claims.
