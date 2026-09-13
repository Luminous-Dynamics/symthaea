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

Pair orientation has no semantic meaning. `(A,B)` and `(B,A)` are the same pair and duplicate pair receipts invalidate the evidence.

A pair sharing the same lineage commitment cannot be classified `Independent`.

A different process, model session, random seed, machine, fork, or account is not automatically an independent lineage.

## Conservative effective independence

Build an undirected graph over active reviewer identities.

Add an edge for every relation except `Independent`:

- `Related`
- `SameLineage`
- `Unknown`
- `ConflictOfInterest`

Also add an edge whenever two provenance receipts carry the same lineage commitment, regardless of the pairwise relation label.

The number of connected components is the **effective independent component count**.

This is deliberately conservative. Unknown provenance may reduce the independence claim, but it may never increase it.

The component count is evidence about reviewer-lineage separation, not a count of moral truths or human persons.

## Panel independence plan

Before independence qualification, preregister:

- minimum effective independent components;
- minimum distinct lineage commitments;
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
- exact recomputation of provenance-strength counts, relation counts, distinct lineages, unknown pairs, conflicts, and effective components.

Missing pair evidence becomes invalid/limited evidence; it does not silently imply independence.

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

Model reviewers may contribute consistency, replication, and adversarial evidence, but multiple sessions from one related model/source lineage must not be inflated into multiple independent evidence components merely because the process IDs differ.

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
