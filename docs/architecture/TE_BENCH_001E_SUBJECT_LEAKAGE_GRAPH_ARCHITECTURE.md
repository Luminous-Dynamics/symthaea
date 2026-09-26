# TE-BENCH-001E — Canonical subject and leakage graph architecture

Status: architecture-only benchmark-custody subject

Date: 2026-09-26

Parents / related work:

- TE-BENCH-001A #6088 — benchmark custody architecture
- TE-BENCH-001B #6093 — frozen source manifest
- TE-BENCH-001C #6095 — exact source-capture receipt
- TE-BENCH-001D0 #6097 — workbook parser normalization architecture
- TE-BENCH-001D1 #6102 — hostile XLSX fixture
- TE-CORE-001A/B/C #6074/#6085/#6086 — thermoelectric evidence semantics/reference line

## Purpose

Define the deterministic identity, parent-family, specimen, source, trajectory, and model-exposure graph that must sit between parsed thermoelectric source records and any retrospective train/test split, property composition, or prospective scoring.

This subject creates no parsed ESTM row, no canonical thermoelectric observation, no model split, and no scientific result. It freezes graph semantics and split-refusal rules only.

## Core theorem

```text
same formula
!= same thermoelectric subject state

same parent compound
!= independent unseen-material example

same dopant/alloy series
!= independent family evidence

same specimen across temperatures
!= independent material examples

same source experiment
!= independent corroboration
```

and:

```text
parsed source record
-> canonical subject/state candidate
-> alias/parent/specimen/exposure graph
-> split eligibility
-> compatible-property admission
```

never:

```text
parsed spreadsheet row
-> random benchmark split
```

## Node classes

At minimum distinguish:

- `SourceArtifactNode`
- `WorkbookRecordNode`
- `CompositionNode`
- `CanonicalMaterialSubjectNode`
- `PhaseStructureNode`
- `PrototypeFamilyNode`
- `ParentCompoundNode`
- `DopantAlloyStateNode`
- `CarrierStateNode`
- `PhysicalSpecimenNode`
- `ProcessBatchNode`
- `MeasurementSeriesNode`
- `TemperatureTrajectoryNode`
- `CarrierTrajectoryNode`
- `DirectionalTransportStateNode`
- `ElectronicStructureArtifactNode`
- `PhononArtifactNode`
- `ModelTrainingCorpusNode`
- `ModelCheckpointNode`
- `PublicationNode`
- `BenchmarkSplitNode`

## Edge vocabulary

Use typed relationships rather than one `leaked=true` flag:

```text
ExactDuplicate
CanonicalAlias
SameCanonicalSubject
SamePhaseOrStructure
SamePrototypeFamily
SameParentCompound
SameDopantSeries
SameAlloySeries
SameCarrierTrajectory
SameTemperatureTrajectory
SamePhysicalSpecimen
SameProcessBatch
SameMeasurementSeries
SameElectronicStructureArtifact
SamePhononArtifact
SameSourceExperiment
MirroredSourceArtifact
SecondaryReprintOf
DerivedFrom
SharedTrainingCorpus
SharedFoundationCheckpoint
PotentialPostCutoffExposure
UnknownRelationship
```

Direction is preserved for derived/reprint/training ancestry edges; symmetric relations are canonically normalized.

## Canonical TE subject state

A thermoelectric subject identity must bind, when available:

- canonical composition;
- exact phase/structure/prototype;
- dopant species/site/fraction;
- alloy/disorder state;
- defect state;
- carrier type and concentration/chemical potential source;
- temperature;
- transport direction/tensor scope;
- physical specimen/process state for measurements;
- computational structure/evaluator state for calculations.

A formula and temperature alone are insufficient to establish a fully compatible TE state.

Unknown state remains explicit. It is not backfilled from another row merely because formulas match.

## Parent-compound ancestry

Half-Heusler work often explores related doped/alloyed derivatives from one parent.

Represent:

```text
ScNiSb parent
  -> Y substitution series
  -> Ti substitution series
  -> Sn substitution series
  -> combined derivatives
```

as ancestry, not independent families.

A split may legitimately test interpolation within a parent series, but it may not label that result `unseen_parent_material`.

## Temperature and carrier trajectories

```text
same specimen at multiple temperatures
= one related temperature trajectory
```

and:

```text
same computed structure swept over chemical potential/carrier concentration
= one related carrier trajectory
```

Rows remain useful for state-conditioned modeling while being ineligible as independent unseen-material examples under strong holdout profiles.

## Component-property ancestry

TE properties may share upstream artifacts.

Examples:

```text
same electronic structure
-> S(T,n)
-> sigma/tau(T,n)
-> electronic transport derivatives
```

and:

```text
same harmonic/anharmonic force constants
-> phonon properties
-> kappa_l(T)
```

A benchmark must not count several descendants of one underlying artifact as independent scientific corroboration.

## Specimen compatibility

Experimental component composition must retain specimen/process ancestry.

```text
S from specimen A
+ sigma from specimen A
+ kappa from specimen B
!= automatically compatible zT evidence
```

A later explicit compatibility profile may permit cross-specimen composition only under declared evidence; default behavior is fail closed.

## Alias discipline

Possible aliases include:

- reduced/unreduced formulas;
- site-order notation variants;
- historical material labels;
- source sample codes;
- Unicode/subscript formatting differences;
- alloy notation ranges;
- nominal vs analyzed composition.

Use dispositions such as:

```text
AliasEstablished
AliasLikelyButUnresolved
NotAlias
ConflictingAliasEvidence
InsufficientIdentityEvidence
```

Do not collapse unresolved aliases before split construction.

## Split contamination closure

Each benchmark split profile declares which edge classes constitute contamination.

A strong unseen-material holdout should normally reject train/test connectivity through:

- exact duplicate / canonical alias;
- same canonical subject;
- same parent compound;
- same physical specimen/process batch;
- same measurement series;
- same temperature/carrier trajectory;
- same source experiment;
- mirrored/reprinted source artifact.

A structure/family holdout may additionally reject:

- same phase/structure;
- same prototype family;
- same dopant/alloy series.

The contamination edge policy is part of benchmark identity.

## Exposure graph

Where possible track:

```text
model checkpoint
-> training corpus
-> source/database
-> parent/prototype/material family
```

Exposure dispositions:

```text
ExposureRuledOutUnderExactCorpus
ExposureNotObservedUnderDeclaredSnapshot
ProviderDeclaredCutoffOnly
PotentialSemanticLeakage
PostCutoffExposurePossible
ExposureUnknown
```

Modern open-world models with unknown training data cannot receive controlled hidden-discovery credit merely from a prompt cutoff.

## Duplicates and contradictions

When multiple records map to apparently the same subject/state:

- retain every source identity;
- classify exact/mirrored/independent relationships;
- preserve lexical precision;
- do not average by default;
- do not select the best zT/value;
- preserve contradictory measurements;
- require a separately versioned reconciliation policy before aggregation.

Numerical equality does not prove duplication; numerical disagreement does not prove distinct subjects.

## Derived quantities

Graph identity must preserve that reported and recomputed quantities can share component evidence but remain distinct artifacts.

```text
reported zT
!= recomputed zT
```

Likewise:

```text
sigma/tau
!= sigma

kappa_total
!= kappa_l
```

Graph edges can record derivation ancestry but cannot erase property-class distinctions.

## Graph generations

Later alias discoveries, corrected specimen identities, source retractions, newly known training exposure, or parent-series relationships create a new graph generation.

Do not rewrite historical split/scoring receipts. Instead mark whether later graph evidence invalidates the earlier interpretation.

## Required hostile fixtures

The first synthetic graph corpus should include at least:

1. same formula, different crystal phase -> distinct subjects;
2. same formula/phase, different dopant site -> distinct state;
3. reduced/unreduced formula exact alias -> related canonical subject;
4. same parent half-Heusler at different temperatures -> one trajectory;
5. same parent across dopant fractions -> parent/dopant-series leakage;
6. same computed bands across carrier sweeps -> shared electronic-structure ancestry;
7. same phonon artifact feeding several kappa rows -> shared phonon ancestry;
8. same specimen used for S and sigma at several temperatures -> shared specimen/measurement series;
9. S from specimen A and kappa from specimen B -> no default zT compatibility;
10. same experiment represented in paper and supplement -> mirrored/derived relation;
11. secondary paper reprints zT -> not independent evidence;
12. independent laboratory reproduces same subject -> related subject but independent evidence origin;
13. alias unresolved -> no collapse;
14. conflicting alias evidence -> preserve both possibilities;
15. nominal composition differs from analyzed composition -> distinct state/uncertainty relation;
16. row lacks carrier state -> do not borrow from adjacent row;
17. row lacks direction -> no anisotropic compatibility assumption;
18. train/test split separates temperature rows of one specimen -> reject strong unseen-material claim;
19. train/test split separates carrier-grid rows from same band structure -> reject strong unseen-material claim;
20. train/test split separates doped derivatives from same parent and calls them new family -> reject;
21. model training corpus contains source workbook -> exposure edge blocks clean holdout where required;
22. provider cutoff only -> exact exposure exclusion unavailable;
23. post-cutoff RAG/web access -> no historical hidden-discovery credit;
24. alias relation discovered after scoring -> new graph generation; old receipt preserved;
25. changed contamination policy -> new benchmark identity;
26. graph cleanliness cannot promote zT to synthesis, device efficiency, manufacturability, or economic value.

## Implementation train

```text
TE-BENCH-001E0
  architecture (this subject)

TE-BENCH-001E1
  frozen synthetic subject/leakage graph corpus

TE-BENCH-001E2
  independent graph/split oracle

TE-BENCH-002A
  parser-bound normalized source records once parser qualifies

TE-BENCH-002B
  canonical subject/state graph over admitted records

TE-BENCH-003A
  frozen retrospective split manifest

TE-BENCH-003B
  independent retrospective scorer
```

The graph architecture may proceed while the parser fixture/qualifier is pending. Real normalized-row admission must not.

## Claim ceiling

A qualified TE identity/leakage graph may establish canonicalization, ancestry, exposure, and split-contamination dispositions under an exact policy. It does not establish Seebeck coefficient, conductivity, thermal conductivity, zT, material quality, model accuracy, synthesis, device performance, manufacturability, or discovery.