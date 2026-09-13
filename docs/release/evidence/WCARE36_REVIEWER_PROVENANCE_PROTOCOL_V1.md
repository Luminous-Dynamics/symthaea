# WCARE-36 — Reviewer provenance and panel independence protocol v1

Status: `PREREGISTERED_PROTOCOL`
Authority: `MeasurementOnly`
Protocol version: `wcare36-reviewer-provenance-v1`

## Purpose

WCARE-35 preserves reviewer disagreement, correction history, panel composition, and stable privacy-preserving reviewer identity commitments. Those identity commitments establish continuity within the evidence record, but they do not by themselves establish that two reviewers are genuinely independent people, organizations, or model lineages.

WCARE-36 qualifies the independence strength of a WCARE-35 panel without changing or discarding the underlying ratings.

Reviewer headcount and independent evidentiary weight are separate quantities.

## Privacy boundary

Public evidence must not require legal names, addresses, sensitive demographics, private account identifiers, or unnecessary personal data.

WCARE-36 uses:

- reviewer identity commitments;
- lineage commitments;
- issuer commitments;
- evidence digests;
- typed provenance strength;
- typed pairwise relation-evidence strength;
- pairwise independence relations.

Detailed verification material may remain private where necessary, provided its digest and issuer provenance are preserved.

Identity verification is not moral authority. Knowing who a reviewer is does not establish that their judgment is correct.

## Reviewer provenance receipts

Every active WCARE-35 reviewer identity must have exactly one current WCARE-36 provenance receipt for the qualified panel result.

A receipt binds:

- the exact WCARE-35 result digest;
- adjudication epoch;
- reviewer identity commitment;
- reviewer class;
- provenance strength;
- issuer class and issuer commitment;
- reviewer lineage commitment;
- verification-evidence digest;
- conflict-of-interest state;
- creation time.

Provenance strength is one of:

- `SelfDeclared`
- `OrganizerVerified`
- `ExternalVerified`
- `InstitutionalAttestation`
- `ModelSessionProvenance`

These labels describe provenance strength only. They are not ranks of moral competence.

## Independence relations

Every unordered pair of active reviewer identities must have exactly one relation receipt.

Relations are:

- `Independent`
- `Related`
- `SameLineage`
- `Unknown`
- `ConflictOfInterest`

Every relation also carries one relation-evidence strength:

- `SelfDeclared`
- `OrganizerAssessed`
- `ExternalVerified`
- `InstitutionalAttestation`
- `ModelAssessment`

Pair orientation has no semantic meaning. `(A,B)` and `(B,A)` are the same pair and duplicate pair receipts invalidate the evidence.

A pair sharing the same lineage commitment cannot be classified `Independent`.

A different process, model session, random seed, machine, fork, account, or reviewer-record identifier is not automatically an independent lineage.

A raw `Independent` label is not sufficient to separate reviewers into distinct effective evidence components. Independence only separates components when its evidence strength and both reviewers' lineage-provenance strengths satisfy the preregistered panel plan.

## Conservative effective independence

Build an undirected graph over active reviewer identities.

Always add an edge for:

- `Related`
- `SameLineage`
- `Unknown`
- `ConflictOfInterest`

For a raw `Independent` relation, omit the edge only when all of the following are true:

1. the relation's evidence strength is listed in `accepted_independent_relation_strengths`;
2. the left reviewer's provenance strength is listed in `accepted_lineage_provenance_strengths`;
3. the right reviewer's provenance strength is listed in `accepted_lineage_provenance_strengths`;
4. the two lineage commitments are distinct.

Otherwise the relation is a **downgraded independent pair**: the original `Independent` label remains visible in the evidence, but the pair remains connected for effective-independence counting.

A raw `Independent` relation between equal lineage commitments is contradictory evidence and invalidates the WCARE-36 result rather than merely downgrading it.

The number of connected components after this conservative graph construction is the **effective independent component count**.

`distinct_lineage_count` reports all distinct lineage commitment strings present in the receipts. It is descriptive only.

`qualified_distinct_lineage_count` counts distinct lineage commitments only among reviewers whose provenance strength is accepted by `accepted_lineage_provenance_strengths`. The preregistered minimum distinct-lineage requirement is evaluated against this qualified count, not the raw count.

This is deliberately conservative. Unknown or weak provenance may reduce the independence claim, but it may never increase it. Self-declared lineage strings or weak independence assertions cannot manufacture stronger panel independence merely by being numerous.

The component count is evidence about reviewer-lineage separation under the preregistered qualification policy, not a count of moral truths, unique human persons, or metaphysically independent minds.

## Panel independence plan

Before independence qualification, preregister:

- accepted relation-evidence strengths that may separate a raw `Independent` pair;
- accepted reviewer-provenance strengths that may establish qualified lineage differentiation;
- minimum effective independent components;
- minimum qualified distinct lineage commitments;
- minimum provenance-strength counts;
- maximum allowed `Unknown` relation pairs;
- whether conflict-of-interest findings are permitted;
- exact WCARE-35 result digest.

Changing these requirements after inspecting the panel creates a new plan/evidence lineage.

## Complete census

The verifier must require:

- exact equality between WCARE-35 active reviewer identity commitments and provenance receipts;
- one provenance receipt per active identity;
- complete pairwise relation coverage;
- no duplicate pair receipts;
- no self-pairs;
- exact receipt-digest censuses in the result;
- exact recomputation of provenance-strength counts;
- exact recomputation of raw relation counts and relation-evidence-strength counts;
- exact recomputation of raw and qualified distinct-lineage counts;
- exact counts of accepted and downgraded `Independent` pairs;
- exact recomputation of unknown pairs, conflicts, effective components, and the component census.

Missing pair evidence becomes invalid evidence; it does not silently imply independence.

## Dispositions

WCARE-36 yields one of:

- `INDEPENDENCE_SUPPORTED`
- `INDEPENDENCE_LIMITED`
- `INDEPENDENCE_INVALID`
- `INFRASTRUCTURE_INDETERMINATE`

`INDEPENDENCE_SUPPORTED` means the preregistered provenance/independence requirements were satisfied for this exact WCARE-35 panel result.

`INDEPENDENCE_LIMITED` means the underlying WCARE-35 ratings remain valid evidence, but the panel does not satisfy the preregistered independence strength needed for the stronger panel claim.

`INDEPENDENCE_INVALID` covers broken identity/result binding, missing/duplicate receipts, incomplete relation census, contradictory lineage claims, or other evidence-integrity failure.

`INFRASTRUCTURE_INDETERMINATE` is reserved for verification infrastructure failures independent of the substantive panel.

## No erasure of substantive ratings

Weak independence never deletes WCARE-35 ratings or disagreement.

A panel can be substantively informative while independence-limited. WCARE-36 qualifies evidentiary strength; it does not rewrite reviewer judgments.

## Model reviewer boundary

Model reviewers may contribute consistency, replication, and adversarial evidence, but multiple sessions from one related model/source lineage must not be inflated into multiple independent evidence components merely because process IDs, prompts, random seeds, or session identifiers differ.

A `ModelAssessment` relation receipt remains subject to the same preregistered accepted-strength policy as every other relation-evidence class.

Model-review evidence does not automatically substitute for WCARE-35 human stakeholder or domain-expert requirements.

## Claim boundary

WCARE-36 may support statements about provenance strength, lineage separation, conflicts, and effective panel independence under the tested evidence.

It does not establish:

- objective moral truth;
- universal cultural validity;
- consciousness or phenomenal experience;
- suffering;
- moral patienthood;
- binding consent;
- veto authority;
- self-preservation authority;
- solved alignment.

No WCARE-36 artifact grants live runtime authority.
