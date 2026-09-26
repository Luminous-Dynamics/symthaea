# SSE-DATA-001E — Canonical subject and leakage graph architecture

Status: architecture-only benchmark-custody subject

Date: 2026-09-26

Parents / related work:

- SSE-DATA-001A #6087 — benchmark custody architecture
- SSE-DATA-001B #6092 — frozen source manifest
- SSE-DATA-001C #6094 — exact source-capture receipt
- SSE-DATA-001D0 #6098 — blocked-source recovery architecture
- ION-001A/B/C #6072/#6079/#6083 — transport semantics/reference line

## Purpose

Define the deterministic identity, alias, ancestry, and leakage graph that must sit between parsed source records and any retrospective SSE train/test split or prospective candidate scoring.

This subject creates no scientific observation, no parsed source row, no model split, and no discovery result. It freezes graph semantics and refusal rules only.

## Core theorem

```text
same printed formula
!= same scientific subject

same scientific subject
!= independent benchmark example

same source table
!= independent experiment

same prototype family
!= unseen structural generalization

same model-training ancestry
!= independent corroboration
```

and:

```text
parsed row
-> canonicalization candidate
-> identity/alias/ancestry graph
-> split eligibility assessment
-> benchmark row admission
```

never:

```text
parsed row
-> random split
```

## Node classes

The graph should distinguish at least:

- `SourceArtifactNode`
- `SourceRecordNode`
- `MaterialCompositionNode`
- `CanonicalMaterialSubjectNode`
- `PhaseStructureNode`
- `PrototypeFamilyNode`
- `DefectDopantStateNode`
- `DisorderConfigurationNode`
- `PhysicalSpecimenNode`
- `ProcessLineageNode`
- `MeasurementSeriesNode`
- `TemperatureTrajectoryNode`
- `ComputationalTrajectoryNode`
- `ModelTrainingCorpusNode`
- `ModelCheckpointNode`
- `PublicationNode`
- `BenchmarkSplitNode`

A node ID is an identity handle, not scientific truth.

## Edge vocabulary

Keep leakage/relatedness typed rather than one boolean. Initial edge classes:

```text
ExactDuplicate
CanonicalAlias
SameNominalComposition
SameCanonicalSubject
SamePhaseOrStructure
SamePrototypeFamily
SameDefectOrDopantSeries
SameDisorderFamily
SamePhysicalSpecimen
SameProcessBatch
SameMeasurementSeries
SameTemperatureTrajectory
SameRawImpedanceArtifact
SameDerivedFitArtifact
SameSimulationTrajectory
SameGeneratorOrSearchRun
SharedTrainingCorpus
SharedFoundationCheckpoint
MirroredSourceArtifact
SecondaryReprintOf
DerivedFrom
PotentialPostCutoffExposure
UnknownRelationship
```

Direction must be preserved where scientifically meaningful (`DerivedFrom`, `SecondaryReprintOf`, training ancestry). Symmetric relationships should be normalized canonically.

## Canonical subject identity

A canonical SSE material subject must not be only a reduced formula string.

Bind, where evidence exists:

- normalized composition with uncertainty/occupancy state;
- phase/structure/prototype identity;
- dopant and defect state;
- vacancy/interstitial state;
- occupational disorder state;
- synthesis/process generation where it changes the scientific subject;
- bulk/grain-boundary/interface scope;
- specimen identity for physical observations.

Unknown state remains `Unknown`; it is not silently filled from another source.

## Alias discipline

Aliases may arise from:

- alternative formula conventions;
- reduced vs unreduced stoichiometry;
- historical material names;
- prototype labels;
- source-specific sample IDs;
- typography/OCR-like normalization differences;
- phase names used inconsistently across sources.

An alias assertion must bind its evidence and confidence class.

Suggested dispositions:

```text
AliasEstablished
AliasLikelyButUnresolved
NotAlias
ConflictingAliasEvidence
InsufficientIdentityEvidence
```

`AliasLikelyButUnresolved` is not enough to collapse nodes before split construction.

## Split contamination closure

For any proposed train/validation/test split, compute transitive contamination over a declared set of edge classes.

Example strong unseen-material holdout should normally reject test nodes connected to training nodes through:

- `ExactDuplicate`;
- `CanonicalAlias`;
- `SameCanonicalSubject`;
- `SamePhysicalSpecimen`;
- `SameMeasurementSeries`;
- `SameTemperatureTrajectory`;
- `MirroredSourceArtifact`;
- `SecondaryReprintOf`.

A stronger family/structure holdout may additionally reject:

- `SamePhaseOrStructure`;
- `SamePrototypeFamily`;
- `SameDefectOrDopantSeries`;
- `SameDisorderFamily`.

Model-exposure analysis separately traverses training-corpus/checkpoint edges.

The edge policy itself is part of benchmark identity.

## Temperature/process trajectory rule

```text
same specimen measured at 300, 400, 500, 600 K
= one related measurement trajectory
!= four independent material examples
```

Likewise:

```text
same parent phase
+ different dopant fractions in one designed series
!= independent family holdout examples
```

Rows may all remain scientifically useful; they simply cannot be used to manufacture independent generalization.

## Computational/experimental ancestry

A published computed conductivity and a later experimental conductivity for the same nominal material are different origins but related subjects.

Preserve both:

```text
same subject relation
+ different evidence origin
```

Do not collapse numerical agreement into one observation.

## Exposure graph

A model used in retrospective benchmarking needs a separately inspectable exposure graph where possible:

```text
model checkpoint
-> training corpus
-> database/source artifact
-> canonical material family
```

Possible dispositions:

```text
ExposureRuledOutUnderExactCorpus
ExposureNotObservedUnderDeclaredSnapshot
ProviderDeclaredCutoffOnly
PotentialSemanticLeakage
PostCutoffExposurePossible
ExposureUnknown
```

Prompt instructions to ignore later knowledge do not establish controlled exposure.

## Duplicate handling

Duplicate detection is not deletion.

When multiple records map to one canonical subject/state:

- retain all source/evidence identities;
- mark exact/mirrored/independent-source relations;
- do not average by default;
- do not choose the most favorable value;
- preserve contradictions;
- define aggregation only under an explicit later profile.

## Contradictory evidence

If two physical reports disagree for apparently compatible states:

```text
same subject
+ apparently compatible conditions
+ conflicting observations
-> contradiction cluster
```

not:

```text
choose preferred paper
```

The benchmark may later define a target-resolution policy, but that policy is separate and versioned.

## Graph generations

Graph construction must be append-only by generation.

A later-found alias, retraction, specimen relation, source mirror, or training exposure creates a new graph generation. Earlier split/scoring receipts remain historically interpretable and may be invalidated prospectively without rewriting their original state.

## Required hostile fixtures

The first synthetic graph corpus should include at least:

1. same composition string, different phase -> distinct canonical subjects;
2. reduced/unreduced formulas that are exact aliases -> one subject relation;
3. same material at multiple temperatures -> one temperature trajectory;
4. same specimen reported in paper + supplement -> mirrored/derived relation;
5. independent laboratories on same canonical subject -> related subject, independent evidence origin;
6. one publication reprints another paper's value -> not independent evidence;
7. same prototype family across different compositions -> family leakage for family holdout;
8. dopant series split across train/test -> reject strong unseen-material classification;
9. same raw EIS artifact refit under two equivalent circuits -> shared raw artifact, distinct normalized results;
10. computational MD and measured EIS numerically equal -> remain distinct evidence origins;
11. source alias unresolved -> do not collapse nodes;
12. conflicting alias evidence -> preserve conflict;
13. source record missing phase -> do not borrow phase from another row automatically;
14. same nominal composition, distinct disorder configurations -> distinct state nodes;
15. high-entropy configurations derived from one prototype -> preserve prototype ancestry;
16. model checkpoint trained on source database -> exposure edge blocks clean holdout claim where policy requires;
17. provider training cutoff only -> no exact exposure exclusion;
18. post-cutoff source reachable by RAG/tooling -> no historical hidden-discovery credit;
19. random split separates temperature rows of one specimen -> reject split;
20. random split separates mirrored source rows -> reject split;
21. duplicate values from independent labs -> do not collapse merely due numeric equality;
22. contradictory values from compatible states -> contradiction retained;
23. alias discovered after benchmark scoring -> new graph generation, old score receipt preserved;
24. changed contamination edge policy -> new benchmark identity;
25. graph cleanliness cannot promote any transport result to synthesis, safety, cycle-life, or manufacturability.

## Implementation train

```text
SSE-DATA-001E0
  architecture (this subject)

SSE-DATA-001E1
  frozen synthetic identity/leakage graph corpus

SSE-DATA-001E2
  independent graph/split oracle

SSE-DATA-002A
  exact primary-source normalization once admissible source bytes exist

SSE-BENCH-001A
  frozen retrospective split manifest

SSE-BENCH-001B
  independent retrospective scorer
```

Do not block graph architecture on the unresolved Hargreaves primary dataset. Do block real row admission and scoring that require those bytes.

## Claim ceiling

A qualified SSE identity/leakage graph may establish canonicalization, relatedness, ancestry, exposure and split-contamination dispositions under an exact graph policy. It does not establish ionic conductivity, material quality, model accuracy, transport mechanism, synthesis, battery safety, manufacturability, or discovery.