# Exclusion Evidence Registry — Design Contract

Status: **design-only / not implementation-authorized**

Base lineage: Native Interoception v0.1 exact qualification candidate
`1007949d5c60fd2d7dd650e8bb4521e2b2803c48`.

This document addresses one specific evidence-lineage weakness without mutating the
current v0.1 qualification target: an `ExclusionCriterionDecision` currently stores a
well-formed SHA-256 string, but the receipt does not prove that the digest corresponds
to a known, preserved artifact in the realized study evidence package.

The goal is to make exclusion decisions reproducibly attributable to concrete evidence
while preserving the current scientific rule that exclusions are preregistered,
mechanical, and never inferred from inconvenient hypothesis outcomes.

## 1. Scope and non-goals

This design is about **evidence identity and accounting**.

It is not:

- a new native interoceptive state variable;
- a change to homeostatic or allostatic dynamics;
- a new exclusion criterion;
- an authorization to exclude surprising outcomes;
- an authentication oracle for external evidence;
- evidence for affect, emotion, feeling, sentience, or consciousness.

A future implementation belongs to a new evidence-schema lineage unless review decides
that it is required before v0.1 can be qualified.

## 2. Current failure mode

Today the exclusion path can establish all of the following:

1. the criterion identifier was preregistered;
2. there is exactly one decision per criterion;
3. the decision status is `NotTriggered`, `Triggered`, or `Indeterminate`;
4. `evidence_sha256` is syntactically a lowercase 64-hex digest;
5. the exclusion receipt is bound to the exact study and execution digest.

But it cannot establish:

> the digest named by the decision actually identifies an artifact that exists in the
> realized evidence package and is preserved under the same study lineage.

A random 64-hex value is therefore structurally indistinguishable from the digest of a
real exclusion-evidence artifact.

## 3. Core invariant

For every exclusion decision `D` there must exist exactly one evidence artifact `A`
such that:

- `D.evidence_ref.artifact_id == A.artifact_id`;
- `D.evidence_ref.sha256 == A.sha256`;
- `A.scope` permits use for `D.criterion_id`;
- `A` is included in the immutable exclusion-evidence registry bound into the realized
  study evidence root;
- the registry itself belongs to the same study-preregistration and study-execution
  lineage as the decision receipt.

Missing, duplicate, mismatched, or cross-lineage evidence fails closed.

## 4. Proposed typed objects

### 4.1 `ExclusionEvidenceArtifact`

Conceptual shape:

- `artifact_id: String`
  - opaque stable identifier;
  - unique inside one registry;
  - must not encode hypothesis direction or semantic result labels.
- `sha256: String`
  - digest of the exact artifact bytes under the declared hashing procedure.
- `artifact_kind: ExclusionEvidenceArtifactKind`
  - typed producer/representation identity.
- `media_type: String`
  - e.g. `application/json`, `text/plain`;
  - descriptive, not trusted as content verification by itself.
- `schema_id: Option<String>`
  - stable schema/version identifier when the artifact is machine-readable.
- `scope: ExclusionEvidenceScope`
  - declares which criterion or criterion set may cite this artifact.
- `producer: ExclusionEvidenceProducer`
  - identifies the subsystem/harness that produced the evidence.

Suggested artifact kinds:

- `MechanicalIntegrityReport`
- `TraceValidationReport`
- `ProtocolConformanceReport`
- `EnvironmentVerificationReport`
- `ExternalObservation`
- `CompositeExclusionEvidence`

The enum should stay narrow. Unknown future classes should require a schema bump rather
than silently becoming an untyped string.

### 4.2 `ExclusionEvidenceScope`

Prefer explicit scope over implied reuse:

- `Criterion { criterion_id }`
- `Criteria { criterion_ids }`
- `StudyWideMechanicalIntegrity`

Rules:

- `Criterion` is the strongest and preferred default;
- `Criteria` is allowed only when the artifact genuinely supports every listed
  criterion;
- `StudyWideMechanicalIntegrity` is reserved for evidence whose meaning is truly global
  to the study execution;
- a decision may not cite an artifact outside its declared scope.

This prevents criterion A from opportunistically reusing criterion B's evidence merely
because both are valid digests.

### 4.3 `ExclusionEvidenceRef`

Replace the unscoped digest in the future receipt shape with:

- `artifact_id: String`
- `sha256: String`

Retaining the digest in the reference is intentional. Registry lookup by ID alone would
allow an artifact's bytes/digest to change while a stale decision continued to point at
the same logical name. Requiring both creates an explicit ID+content binding.

### 4.4 `ExclusionEvidenceRegistry`

Conceptual fields:

- `schema_version`
- `study_preregistration_sha256`
- `study_execution_sha256`
- `artifacts: Vec<ExclusionEvidenceArtifact>`

Required properties:

- deterministic ordering or canonical sort by `artifact_id`;
- no duplicate artifact IDs;
- no duplicate canonical artifact identities unless explicitly permitted;
- every digest is valid lowercase SHA-256;
- all declared criterion scopes refer to preregistered exclusion criteria;
- empty criterion scopes are invalid;
- canonical JSON under the pinned dependency set;
- stable registry SHA-256.

## 5. Receipt binding

A future `ExclusionDecisionReceipt` schema should additionally bind:

- `exclusion_evidence_registry_sha256`;
- each decision's `ExclusionEvidenceRef`.

Validation order should be deterministic:

1. validate study preregistration;
2. validate execution against study;
3. validate registry against study and execution;
4. validate receipt study/execution/registry digests;
5. validate one decision per preregistered criterion;
6. resolve each evidence reference by `artifact_id`;
7. require exact digest equality;
8. require criterion-scope compatibility;
9. derive run disposition only after all evidence references validate.

`disposition_against` must never operate on an evidence receipt that has not first
passed the registry-bound validation path.

## 6. Realized evidence root

The registry should not be an orphan object. A realized study evidence package should
bind its digest alongside at least:

- study preregistration digest;
- exact execution digest;
- exclusion decision receipt digest;
- blinded metric digest;
- confirmatory evaluation digest when one exists;
- raw execution/analysis artifact identities;
- source/toolchain/environment provenance required by the qualification capsule.

This creates the chain:

`StudyPreregistration`
→ `StudyExecutionTrace`
→ `ExclusionEvidenceRegistry`
→ `ExclusionDecisionReceipt`
→ `StudyBlindedMetricReport`
→ `ConfirmatoryHypothesisEvaluation`
→ realized evidence root.

Removing or replacing exclusion evidence after blinded analysis therefore changes the
root identity or fails validation.

## 7. Relationship to `EvidenceCapsuleManifest.artifacts`

The existing evidence capsule already stores named raw artifact digests. Reuse its
hashing/provenance philosophy, but do not silently overload `ArtifactDigest` as the
entire exclusion registry because it currently lacks:

- stable opaque artifact IDs distinct from filenames;
- typed artifact kind;
- criterion scope;
- producer identity;
- schema identity;
- direct study/execution lineage binding.

Two integration strategies are acceptable:

### A. Registry references capsule artifacts

The exclusion registry stores rich metadata and every registry artifact must also
appear in `EvidenceCapsuleManifest.artifacts` with the same digest.

Advantages:

- one raw-artifact inventory remains the outer qualification root;
- exclusion evidence becomes a semantically richer view over preserved artifacts.

### B. Capsule binds registry as one canonical artifact

The complete registry canonical JSON is included as an artifact in the evidence
capsule, while the registry itself accounts for its members.

Advantages:

- smaller capsule surface;
- cleaner separation between generic capsule provenance and study-specific evidence.

Preferred initial design: **A plus registry digest**. It gives both raw-artifact
presence and criterion-scoped semantic linkage. If this becomes cumbersome, move to B
only with an explicit schema/design review.

## 8. External-verification boundary

Registry membership proves provenance consistency, not the truth of external evidence.

For internally produced mechanical evidence, the study harness can usually recompute
or verify the artifact directly.

For external observations, the harness may additionally require:

- source identity;
- retrieval timestamp or acquisition epoch;
- signed attestation or transparency-log proof when available;
- independent verifier receipt;
- declared trust class.

Those features should not be falsely implied by a SHA-256 alone.

## 9. Exclusion-policy firewall

The registry must not become a backdoor for outcome-dependent exclusion.

Hard rules:

- an exclusion artifact cannot be generated from semantic hypothesis satisfaction unless
  the exclusion criterion prospectively names that exact dependency;
- primary hypothesis outcome values are not exclusion evidence by default;
- semantic arm labels should remain unavailable to exclusion producers unless the
  preregistered criterion explicitly requires them;
- unexpected effect direction, small effect size, or null result are never mechanical
  exclusion evidence;
- `NotTriggered` requires preserved evidence just as `Triggered` does;
- `Indeterminate` preserves the evidence explaining why determination failed.

## 10. Deterministic canonicalization

Recommended canonicalization rules:

- sort registry artifacts by `artifact_id` before hashing;
- sort multi-criterion scopes lexicographically and reject duplicates;
- require normalized stable IDs rather than filesystem paths as primary keys;
- avoid unordered maps in canonical serialized structures;
- preserve explicit schema versions;
- do not normalize or rewrite raw artifact bytes during verification;
- hash exact serialized bytes under the pinned implementation/dependency set.

Round-trip serialization must preserve exact equality and digest identity.

## 11. Adversarial test matrix

A future implementation should include at least these tests.

### Presence and identity

1. random well-formed digest not present in registry is rejected;
2. known artifact ID with wrong digest is rejected;
3. unknown artifact ID with a digest equal to some known artifact is rejected;
4. duplicate artifact IDs invalidate the registry;
5. duplicate criterion IDs inside a scope invalidate the registry;
6. empty artifact IDs / producer IDs / required schema IDs are rejected as applicable.

### Scope

7. criterion A cannot cite criterion B-only evidence;
8. a study-wide artifact can be reused only by criteria that accept the declared
   study-wide evidence class;
9. scope referring to an unknown preregistered criterion is rejected;
10. a multi-criterion artifact can be cited only by criteria explicitly listed.

### Lineage

11. registry study digest from another preregistration is rejected;
12. registry execution digest from another run is rejected;
13. valid receipt + valid registry from different executions cannot be cross-paired;
14. removing an exclusion artifact after receipt creation breaks realized evidence-root
    validation;
15. changing artifact bytes while preserving the artifact ID changes the digest and
    invalidates the decision reference.

### Decision semantics

16. `NotTriggered` without resolvable evidence is rejected;
17. `Triggered` without resolvable evidence is rejected;
18. `Indeterminate` without resolvable evidence is rejected;
19. all decisions resolve before `RunDisposition` is derived;
20. excluded and indeterminate runs retain their evidence registry in final accounting.

### Outcome-manipulation resistance

21. a semantic hypothesis report cannot be substituted for mechanical exclusion
    evidence under an incompatible artifact kind/scope;
22. changing hypothesis outcome satisfaction cannot change exclusion disposition when
    all preregistered exclusion evidence is unchanged;
23. reordering registered artifacts does not change canonical registry identity after
    canonical sort;
24. reordering decisions does not alter derived disposition, while canonical receipt
    hashing follows its declared ordering policy.

## 12. Schema/version consequences

Likely future changes:

- `EXCLUSION_DECISION_RECEIPT_SCHEMA_VERSION`: increment;
- introduce `EXCLUSION_EVIDENCE_REGISTRY_SCHEMA_VERSION`;
- possibly increment a realized-study-evidence schema if one is introduced;
- update blinded metric / confirmatory validation only where their bound exclusion
  receipt identity changes structurally.

This does **not** require changing `INTEROCEPTIVE_MODEL_SEMANTICS_VERSION` because the
native viability state, recovery, intervention, and allostatic forecast behavior are
unchanged.

## 13. Implementation sequence

If authorized after the current v0.1 qualification attempt:

1. add typed artifact/scope/ref/registry structures;
2. canonical registry validation + hashing;
3. add registry-bound exclusion receipt schema v2;
4. update study-level validation to require registry resolution before disposition;
5. update blinded study metric extraction to bind the new receipt;
6. update confirmatory recomputation path;
7. integrate registry artifacts with `EvidenceCapsuleManifest`;
8. add adversarial tests above;
9. add migration fixtures proving v1 receipts cannot be silently interpreted as v2;
10. issue a new evidence-schema qualification lineage.

## 14. Stop rule

Do not merge or reinterpret this design as part of the current v0.1 qualified candidate
while `1007949d5c60fd2d7dd650e8bb4521e2b2803c48` is awaiting its gates.

After those gates execute, classify this work explicitly as one of:

- **qualification-blocking**: implement as a new v0.1 evidence-schema head and rerun all
  qualification gates;
- **post-v0.1 hardening**: retain the qualified v0.1 baseline and implement this as the
  next evidence lineage before confirmatory studies depend on exclusion evidence.

No result from the current v0.1 head should be retroactively rewritten to claim that it
had registry-bound exclusion evidence when it did not.
