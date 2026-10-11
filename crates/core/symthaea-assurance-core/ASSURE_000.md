# ASSURE-000 — Generic Claim–Evidence Qualification Kernel

## Governing theorem

```text
evidence exists
    != claim established
    != stronger claim established
    != deployment authority
```

ASSURE-000 defines a domain-neutral vocabulary for exact subject identity, claims, evidence provenance, support strength, negative findings, reproduction-evidence status, claim ceilings, and invalidation conditions.

## Deterministic identities

The kernel content-addresses:

- `SubjectManifest` — canonical committed subject components;
- `Claim` — exact subject, claim ID, statement, and declared scope label;
- `QualificationPlan` — exact claim, maximum support ceiling, and invalidation set;
- `EvidenceArtifact` — exact subject/claim binding, evidence kind, artifact digest, and producer/executor/verifier/signer provenance;
- `QualificationResult` — exact subject/claim/plan identity, evidence commitments, support outcome, reproduction-evidence status, ceiling, and invalidation set.

Evidence cannot be rebound to a different subject or claim. Duplicate evidence IDs fail closed rather than being silently deduplicated. Changing evidence bytes or recorded provenance changes the evidence commitment and therefore changes the qualification-result identity.

## Support strength

The positive support ladder is intentionally narrow:

- `Structural` requires architecture-inspection evidence;
- `Observed` requires observation evidence;
- `CausallySupported` requires controlled-intervention evidence;
- `FunctionallySupported` requires controlled-intervention plus functional-benchmark evidence.

The ladder is not a generic score. Each successor tier has explicit predicates, and `QualificationPlan::maximum_support` is a hard claim ceiling.

## Reproduction evidence is orthogonal to support

Reproduction evidence is not treated as a fifth support tier. A structural, observational, causal, or functional result can in principle be reproduced; the presence of reproduction evidence therefore describes another axis of the result rather than a stronger point on the support ladder.

ASSURE-000 exposes:

```text
ReproductionStatus::NotClaimed
ReproductionStatus::EvidenceFromDistinctVerifier
```

`EvidenceFromDistinctVerifier` requires an admitted `Reproduction` evidence artifact whose verifier identity differs from both producer and executor. That status is independently bound into the result digest.

Crucially, ASSURE-000 does **not** infer that the reproduction succeeded or semantically matched the original result. Successful replication, failed replication, and contradiction-aware interpretation require a resolver that can compare heterogeneous evidence; those semantics belong to ASSURE-003.

## Reproduction evidence is not common-cause independence

Identity-distinct reproduction evidence is deliberately not called independent verification.

```text
different verifier ID
    != different organization
    != different review process
    != different verification toolchain
    != different evidence source
    != common-cause independence
```

Symthaea already has a stronger generic verifier-diversity model in PR #1912, where reviewed verifier profiles carry organization, review-process, toolchain, and evidence-source fault domains. ASSURE-000 does not duplicate that ontology. A later assurance layer must bind and evaluate those stronger independence facts before using independent-verification language.

## Support is not deployment eligibility

ASSURE-000 also deliberately keeps deployment off the support ladder.

```text
stronger evidentiary support
    != deployment eligibility
    != deployment authorization
```

A runtime receipt may be admitted as evidence, but its presence does not create another support tier. ASSURE-001 may bind an exact deployment envelope as part of a richer external-AI subject manifest, and later campaign/policy layers may assess deployment-specific claims. The core kernel itself never turns evidence strength into authority.

## Outcome binding is not evidence resolution

`QualificationResult::validate_and_bind` accepts caller-supplied result dimensions and validates them against exact identities, claim ceilings, evidence classes, and reproduction-evidence predicates. It does not infer a heterogeneous evidence verdict or establish that a plan existed before evidence production.

Those stronger theorems are intentionally deferred:

- ASSURE-002 — preregistered qualification plans;
- ASSURE-003 — contradiction-aware/non-scalar evidence resolution.

## Negative and orthogonal findings

- `NotDemonstrated`
- `Contradicted`
- `Inconclusive`
- `Expired`
- `Invalidated`

These are intentionally not placed below `Structural` on one ordinal scale. A contradiction is not a weak positive result, and inconclusive evidence is not a scientific refutation.

## Invalidation preserves historical values

Qualification results retain an explicit invalidation set. `QualificationResult::invalidated_by` returns a **new** result with `Negative(Invalidated { ... })` when the supplied condition was declared by the plan. The original content-addressed result remains unchanged, preserving its prior digest and outcome.

ASSURE-004 will later add explicit predecessor/currentness/requalification lineage. ASSURE-000 deliberately does not claim that returning a new invalidated value by itself proves when the invalidating event occurred or which result is currently authoritative.

## Qualification environment

ASSURE-000 declares Rust 1.96 as its minimum supported toolchain because that is the exact toolchain exercised by its focused qualification lane. The workflow proves exact candidate HEAD, formatting, compilation/check, tests, strict Clippy, and committed root-lock parity. The qualification workflow is read-only; lock drift fails closed rather than granting PR execution standing repository-write authority.

## Deliberate nonclaims

ASSURE-000 does not establish certification, compliance, agent safety, preregistration, heterogeneous evidence resolution, successful replication, deployment eligibility, deployment authorization, common-cause verifier independence, independent audit, runtime isolation, red-team completeness, or regulatory conformity. It provides the semantic substrate later assurance campaigns can use to make narrower evidence-backed claims.
