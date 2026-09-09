# Qualification Lineage Binding v1 (SCI-Q5)

Status: architecture contract only; non-authorizing.

Date: 2026-09-09

## Purpose

SCI-Q1 separates execution observations from qualification disposition and exposes only verifier-owned reporting authority.

SCI-Q3 separates qualification artifact identity from provenance, execution, scientific truth, and materialization authority.

SCI-Q4 separates declared execution profile, realized environment, execution occurrence, verified execution identity, reproducibility, and scientific qualification.

SCI-Q5 defines the missing join theorem between those independently valid objects.

The central invariant is:

`valid receipt + valid artifact identity + valid execution identity != valid qualification lineage`

unless the three objects are proven to describe the same qualification subject and the same execution occurrence.

This document does not implement the join and does not grant materialization authority.

## Why this layer is necessary

Independent validators create a composition risk.

Suppose:

- receipt `R_A` is a legitimate `Qualified` receipt for candidate A;
- artifact identity `A_B` is a legitimate identity for candidate B;
- execution identity `E_C` is a legitimate identity for run C.

Each object can be valid in isolation while the tuple `(R_A, A_B, E_C)` is scientifically meaningless.

A consumer that checks only `R_A.is_qualified()`, `A_B.is_valid()`, and `E_C.is_valid()` can accidentally launder authority across subjects.

The join itself therefore requires qualification.

## Non-equivalences

SCI-Q5 preserves all of the following distinctions:

`receipt validity != artifact validity`

`artifact validity != execution validity`

`execution validity != receipt validity`

`same repository != same candidate`

`same base != same candidate patch`

`same patch bytes != same result tree`

`same workflow bytes != same qualification profile`

`same profile != same execution occurrence`

`same run provider ID != same execution identity`

`same candidate != same qualification attempt`

`same qualification attempt != same materialization attempt`

`same result tree != same provenance`

`same result tree != same execution environment`

`same scientific theorem != same candidate artifact`

`same disposition != same lineage`

`valid lineage != materialization authority`

## Threat model

SCI-Q5 is designed to reject at least the following composition failures.

### 1. Qualified-receipt replay across candidates

A valid qualified receipt for candidate A is paired with candidate B's artifact identity.

Expected outcome: reject.

### 2. Artifact replay across workflow versions

The exact same source candidate is qualified under workflow W1, but a consumer pairs the receipt with workflow identity W2.

Expected outcome: reject unless an explicit qualified relation proves W1 and W2 interchangeable for this authority purpose.

### 3. Profile replay

A receipt produced under qualification profile P1 is paired with profile semantics P2 while retaining the same human-readable profile name/version.

Expected outcome: reject.

### 4. Execution replay across candidates

A valid execution identity from candidate A is attached to candidate B because both ran the same commands/toolchain.

Expected outcome: reject.

### 5. Run-ID laundering

Two executions share or spoof a provider-level run number/string.

Expected outcome: provider locator alone is non-authoritative; reject unless the full execution occurrence binding matches.

### 6. Result-tree substitution

The candidate patch digest and base match, but the claimed result tree is from another application or contains an extra mutation.

Expected outcome: reject.

### 7. Changed-scope substitution

The patch digest matches, but a consumer substitutes a changed-path scope from another candidate or omits an authoritative path.

Expected outcome: reject.

### 8. Cross-repository replay

A candidate with matching-looking Git/hash values is moved between repository contexts.

Expected outcome: reject unless repository context is exactly bound.

### 9. Execution-environment substitution

Receipt and artifact identity are for one candidate, but the execution identity comes from a different toolchain/dependency/environment closure.

Expected outcome: reject.

### 10. Partial-success laundering

A receipt whose scientific theorem passed but admissibility failed is paired with a fully valid artifact/execution identity and treated as qualified.

Expected outcome: reject. Lineage coherence cannot upgrade disposition.

### 11. Cancelled/incomplete replay

A cancelled qualification attempt is combined with valid identities from another run to manufacture a complete-looking record.

Expected outcome: reject.

### 12. Superseded-head replay

A previously queued/failed candidate head is superseded by new candidate bytes, but its historical receipt is attached to the new head.

Expected outcome: reject.

## Qualification subject

SCI-Q5 introduces the conceptual object `QualificationSubjectV1`.

It is not a human label such as a PR number or branch name.

The subject is the exact thing being qualified.

Conceptually:

`QualificationSubjectV1 = artifact identity + qualification-profile semantic identity`

The artifact identity should eventually include the SCI-Q3 coordinates:

- repository context;
- exact base Git commit identity;
- exact candidate patch byte digest;
- exact independently produced result-tree Git identity;
- exact workflow byte digest;
- canonical changed-path scope;
- qualification-profile semantic commitment.

The profile commitment may live inside the artifact envelope or be referenced separately, but there must be exactly one authoritative value in a validated lineage. Duplicate independently writable copies are dangerous.

## Execution occurrence

SCI-Q5 also introduces the conceptual `QualificationExecutionOccurrenceV1`.

An execution occurrence is one attempt to evaluate one qualification subject under one realized execution environment.

Conceptually it binds:

- qualification subject identity;
- declared execution profile identity;
- realized execution environment identity;
- exact command/instrument semantics;
- input artifact identities;
- stochasticity/nondeterminism state;
- controlled/uncontrolled external-state declaration;
- provider locator metadata where available;
- start/end/terminal execution state;
- execution observations emitted by that occurrence.

Provider run IDs are useful locators but are not sufficient identity.

## Receipt binding

The qualification receipt must be interpreted as evidence from exactly one execution occurrence over exactly one qualification subject.

A future receipt integration therefore needs two non-optional references:

1. `subject_identity`
2. `execution_occurrence_identity`

These references must not be caller-chosen aliases that can be rebound after validation.

The strongest implementation direction is verifier-owned construction:

- artifact validator produces an opaque verified artifact witness;
- execution validator consumes that witness and produces an opaque execution witness bound to it;
- receipt validator consumes the same execution witness and produces an opaque validated qualification-lineage witness.

The caller should never construct the final capability by copying strings/digests into a public struct and asserting equality manually.

## Proposed future stages

The eventual implementation may look conceptually like:

1. `verify_artifact_identity(observed, expected) -> VerifiedArtifactIdentity`
2. `verify_execution_identity(artifact, profile, occurrence) -> VerifiedExecutionIdentity`
3. `validate_qualification_receipt(profile, observations) -> ValidatedQualificationReceipt`
4. `bind_qualification_lineage(artifact, execution, receipt) -> VerifiedQualificationLineage`

But stage 4 MUST additionally prove that:

- the receipt names/references the exact execution occurrence;
- that occurrence references the exact artifact identity;
- the profile semantic commitment is identical across all relevant layers;
- no contradictory duplicate identities are present;
- the receipt disposition remains unchanged when recomputed;
- reporting authority remains verifier-derived;
- no action/materialization authority is implied.

## Why simple digest equality is not enough

A naive implementation might hash each whole object and require three hashes.

That is useful as tamper evidence but insufficient as a semantic join.

Two individually valid objects can have valid hashes while referring to different subjects.

SCI-Q5 requires both:

- object integrity/identity; and
- relation validity between objects.

`valid node != valid edge`

This is deliberately analogous to the Scientific Evidence Dependency Graph principle that independently valid evidence nodes do not establish independence or replication without qualified relationship semantics.

## Relationship identity

The join itself should eventually be content-addressed once SCI-Q2/SCI-Q3 canonicalization rules are qualified.

A future relationship commitment may include:

- schema/domain separator;
- subject identity commitment;
- execution identity commitment;
- receipt commitment;
- qualification profile commitment;
- validator/profile version identity.

However, SCI-Q5 does not yet select a universal canonical serialization or hash protocol.

A relationship digest is not authority by itself.

## Profile consistency rule

Qualification profile semantics are security/scientific authority semantics.

Therefore:

- human-readable profile ID is locator metadata;
- integer/string version is useful but insufficient;
- exact profile semantic commitment is authoritative;
- receipt, artifact identity, execution identity, and lineage validator must agree on the same commitment.

If profile semantics drift without a new commitment, validation must fail closed.

## Workflow consistency rule

Workflow bytes and qualification profile semantics are related but not identical.

One workflow can potentially implement multiple profiles; one profile could theoretically be implemented by multiple qualified workflows.

Therefore:

`workflow digest != profile semantic commitment`

Q5 must preserve both when both are authority-relevant.

Equivalence between two workflow implementations is a future qualification claim, not assumed equality.

## Execution consistency rule

Execution identity must bind the artifact identity it evaluated.

It is not enough that the execution used the same repository and base.

It must bind the exact candidate/result/workflow/profile subject expected by the qualification lineage.

This prevents a run against candidate A from being reused to qualify candidate B.

## Receipt consistency rule

A validated receipt may contain a derived disposition such as `Qualified`, but the lineage validator must never trust that cached field in isolation.

It must recompute or consume an opaque receipt witness whose disposition was recomputed under the trusted profile.

Lineage validation cannot upgrade:

- `PremiseNotReproduced`;
- `CandidateRejected`;
- `CandidateHygieneFailure`;
- `CandidateArtifactMismatch`;
- `HarnessFailure`;
- `InfrastructureInterrupted`;
- `Incomplete`.

A coherent lineage with a non-qualified receipt is still non-qualified.

## Failed executions are still bindable evidence

A failed/cancelled/timed-out execution can have exact artifact and execution identity.

Q5 should permit a coherent lineage witness for historical/audit purposes even when qualification failed.

Therefore two concepts must remain distinct:

- `VerifiedQualificationLineage`: the objects belong together;
- `QualifiedLineage`: the coherent lineage also carries a `Qualified` receipt disposition.

Even `QualifiedLineage` still does not imply materialization authority in Q5.

## Anti-replay test matrix for a future Q5 implementation

A future executable qualification should create at least two subjects (A/B), two execution occurrences (A1/B1), and receipts spanning success/failure states.

It should require:

1. A artifact + A execution + A receipt -> coherent.
2. B artifact + B execution + B receipt -> coherent.
3. A artifact + B execution + A receipt -> reject.
4. A artifact + A execution + B receipt -> reject.
5. B artifact + A execution + A receipt -> reject.
6. A receipt copied with candidate-patch identity altered -> reject.
7. A receipt copied with result-tree identity altered -> reject.
8. A receipt copied with workflow identity altered -> reject.
9. A receipt copied with profile commitment altered -> reject.
10. A execution copied with toolchain/environment identity altered -> reject.
11. Same provider run locator with different full execution identity -> reject.
12. Qualified receipt + coherent identities -> qualified-lineage witness only.
13. Hygiene-failed receipt + coherent identities -> coherent historical lineage but NOT qualified lineage.
14. Cancelled receipt + coherent identities -> coherent historical lineage but NOT qualified lineage.
15. Forged cached `Qualified` field whose recomputation is non-qualified -> reject.
16. Duplicate contradictory subject references -> reject.
17. Unknown future schema/profile semantics under older validator -> fail closed.

## No materialization authority

SCI-Q5 intentionally stops before action authority.

The following must remain impossible:

`VerifiedQualificationLineage -> apply patch`

`QualifiedLineage -> merge PR`

`QualifiedLineage -> write main`

A later SCI-Q6+ materialization layer must additionally bind:

- exact target base state at materialization time;
- exact qualified patch/artifact identity;
- exact expected resulting tree;
- materialization policy;
- target repository/ref constraints;
- authorization actor/quorum/delegation policy if required;
- post-application tree/diff equivalence;
- single-use/replay constraints where appropriate.

Materialization is an action authority problem, not merely a scientific qualification problem.

## Interaction with provenance/authentication

Q5 proves identity relationships, not who produced them.

A fully coherent qualification lineage can still lack authenticated provenance.

Future provenance/signature layers may bind:

- validator identity;
- runner/operator identity;
- external verifier identity;
- signatures/attestations;
- trust snapshots;
- delegation/quorum state.

Those are independent authority dimensions and must not be inferred from hashes or GitHub metadata.

## Interaction with reproducibility

One coherent qualification execution is not replication.

Repeating the same subject under the same environment may demonstrate replay consistency but not independent replication.

Repeating under meaningfully different environments may support robustness/triangulation, but independence remains SCI-006 evidence-dependency semantics.

## Interaction with Q1

Q1 should remain responsible for:

- observations;
- profile-relative disposition;
- fail-closed missing/duplicate evidence handling;
- verifier-owned reporting capability.

Q5 should not duplicate Q1's classifier.

Instead it binds Q1's validated output to Q3/Q4 identities.

## Interaction with Q3

Q3 owns artifact identity semantics.

Q5 must consume Q3's verified artifact witness rather than reconstruct candidate identity using its own hash conventions.

## Interaction with Q4

Q4 owns execution identity semantics.

Q5 must consume the future Q4 verified execution witness rather than treating provider run metadata or environment labels as sufficient.

## Migration rule

Do not retrofit Q5 lineage authority into existing qualification workflows until:

1. Q1 exact hosted candidate qualifies;
2. Q3 exact hosted candidate qualifies;
3. a Q4 executable identity candidate qualifies;
4. Q5 anti-replay implementation itself passes hostile composition tests.

Historical workflows can be represented as legacy evidence, but must not be retroactively upgraded to `QualifiedLineage` without sufficient exact identity/execution data.

## Boundary statement

SCI-Q5 establishes only the design requirement:

`valid objects do not imply valid composition`

and the future relationship that must be proven:

`receipt ↔ qualification subject ↔ execution occurrence`

It does not establish:

- that Q1 is qualified;
- that Q3 is qualified;
- a Q4 implementation;
- canonical receipt hashing;
- provenance/authentication;
- independent replication;
- scientific truth;
- merge/materialization authority.
