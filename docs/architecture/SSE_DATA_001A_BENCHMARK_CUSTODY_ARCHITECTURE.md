# SSE-DATA-001A — Solid-electrolyte benchmark custody architecture

Status: architecture/data-custody design only

Date: 2026-09-26

Related:

- SSE-001 #4993
- ION-001 #4996
- MAT-015 external-provider provenance
- MAT-MLIP-001 #4992
- MAT-CALC-001 #5114
- MAT-010 negative/search memory
- MAT-RD-001 #4997

## Purpose

Define the first auditable retrospective benchmark-custody contract for solid-state electrolytes before any Symthaea model is allowed to rank or score benchmark answers.

This subject does not create an electrolyte dataset, does not normalize external measurements, does not execute DFT/MD, and does not establish any material property. It freezes what must be captured and what must remain distinct when the benchmark is later materialized.

## Core theorem

```text
paper mentions material
!= exact material subject
!= exact conductivity observation
!= compatible measurement state
!= transport benchmark row
!= model-training eligibility
!= model-test eligibility
!= prospective discovery
```

The benchmark must preserve source and condition ancestry strongly enough that an apparently excellent model cannot succeed by learning duplicate compositions, temperature trajectories, aliases, or post-cutoff answers.

## Benchmark generations

Use explicit immutable generations.

Initial plan:

```text
SSE-BENCH-G0
  design/custody freeze date: 2026-09-26
  retrospective literature/provider benchmark
  no prospective claim

SSE-PROSPECT-G1
  candidate/ranking commitment frozen later
  high-fidelity answers unavailable to scorer at commitment time
```

A later source correction or newly published result creates a new generation rather than rewriting G0.

## Source-cutoff semantics

For every benchmark generation bind:

- exact cutoff timestamp/date policy;
- publication-online date and version-of-record date where different;
- provider snapshot/export date;
- exact artifact URL/DOI/provider identifier;
- exact captured source bytes or source receipt where permitted;
- access/retrieval timestamp as provenance only;
- source license/redistribution constraints;
- parser/extraction profile identity;
- whether post-cutoff material was accessible to any candidate generator/model.

Preserve:

```text
published before cutoff
!= known to model

retrieved after cutoff
!= automatically contaminated

provider live endpoint
!= frozen benchmark snapshot
```

## Source classes

Keep evidence source classes explicit, for example:

```text
PeerReviewedArticle
SupplementaryArtifact
CuratedExperimentalDatabase
MaterialsProviderRecord
ComputedDatabaseRecord
AuthorReleasedTrajectory
AuthorReleasedStructure
IndependentReplication
ReviewOrPerspective
DerivedSecondaryCompilation
```

Reviews/perspectives may motivate schema design but should not silently become primary property observations.

## Initial source families

The first manifest may include, subject to exact source custody and licensing:

1. curated experimental lithium solid-electrolyte conductivity data with explicit temperature conditions;
2. halide-family experimental studies with exact composition/process/measurement conditions;
3. 2025/2026 ML+DFT/AIMD halide-screening studies as retrospective computational targets;
4. transport-focused MLFF benchmark studies for LLZO/LYC/LGPS as capability/calibration context;
5. explicit negative or low-conductivity material examples;
6. failed/nonconverged computational targets where recoverable.

External examples are benchmark candidates, not assumed ground truth.

## Candidate subject identity

A benchmark row must not be keyed only by formula.

Bind where known:

- canonical composition;
- composition normalization convention;
- phase/space group/prototype;
- exact structure artifact when available;
- dopant identities, fractions, and sites where known;
- vacancy/interstitial state;
- occupational disorder state;
- amorphous/crystalline state;
- polymorph identity;
- sample/process lineage for physical observations;
- source-specific aliases;
- deterministic canonical subject identity.

```text
Li3YCl6
!= every Li3YCl6 specimen/phase/process state
```

## Observation identity

Every conductivity observation must bind conditions rather than becoming `material -> scalar`.

Capture where available:

- mobile species;
- conductivity value and units;
- temperature;
- pressure;
- AC/DC/EIS method;
- frequency range;
- equivalent-circuit/fitting method;
- specimen geometry/density;
- pellet processing;
- electrodes/contacts;
- bulk/grain-boundary/composite assignment;
- activation energy and its fit window where reported;
- uncertainty/error bars;
- source table/figure/artifact locator;
- parser/extraction receipt;
- physical vs computational origin.

Unknown fields remain `Unknown`, never zero/default.

## Computational transport records

Keep separate computational propositions such as:

```text
NEB barrier
AIMD diffusion coefficient
MLIP-MD diffusion coefficient
Nernst-Einstein conductivity
collective conductivity
activation-energy fit
```

Computational rows additionally bind:

- exact structure/state;
- evaluator/solver family;
- exchange-correlation/model identity;
- supercell size;
- trajectory length;
- temperature points;
- diffusion-analysis profile;
- finite-size/time caveats;
- whether conductivity is direct/collective or Nernst-Einstein derived;
- execution/convergence evidence when local Symthaea calculations are eventually added.

ION-001 owns transport semantics. SSE-DATA owns benchmark custody only.

## Do not normalize away physics

The benchmark should not force every record into one room-temperature conductivity scalar.

Preserve raw temperature-conditioned observations first.

Optional normalized views may later be created only through an explicit transformation profile that binds:

- source observations;
- fit model;
- temperature window;
- extrapolation/interpolation policy;
- uncertainty propagation;
- applicability/refusal state.

```text
measured 300 K
!= Arrhenius-extrapolated 300 K
```

## Leakage graph

Represent contamination as a graph, not one boolean.

Possible relations:

```text
ExactDuplicate
FormulaAlias
SameCrystalStructure
SameParentCompound
SameDopingTrajectory
SameTemperatureTrajectory
SameSourceExperiment
SameAuthorDerivedRecord
SharedSimulationTrajectory
SharedTrainingDataset
PostCutoffExposurePossible
SemanticNearDuplicate
```

A benchmark split is identified by the exact leakage graph snapshot used to build it.

## Required split families

At minimum support:

### Composition/parent holdout

All temperature/process rows of the same parent subject remain on one side unless the benchmark explicitly tests condition interpolation.

### Chemistry-family holdout

Examples may include halide/oxide/sulfide family or narrower chemistry-group exclusions.

### Structure/prototype holdout

Prevent nearly identical crystal prototypes from trivially crossing train/test.

### Dopant-family holdout

Keep closely related substitution series together where appropriate.

### Temporal/source holdout

Freeze exact cutoff and hold later publications/sources separately.

### Transport-state holdout

A separate benchmark may test interpolation/extrapolation across temperature or defect state, but must not call that unseen-material generalization.

## First halide benchmark cohort

The first bounded halide cohort should deliberately contain multiple evidence kinds and outcomes.

Candidate families/examples for source-custody investigation include:

- Li3YCl6-family halides;
- Li2ZrCl6 and substituted variants;
- other Li-M-Cl/Br/F families with explicit measurements;
- Rb2LiAlF6 as a recent ML+DFT/AIMD retrospective candidate;
- at least one low-conductivity/failed candidate from the same literature universe;
- at least one disorder/amorphous halide case;
- at least one out-of-family oxide/sulfide control for transport-method calibration only.

Named materials are not assumed benchmark winners.

## Property planes

Do not make one `electrolyte_score`.

Track independently where source evidence exists:

- ionic conductivity;
- diffusion coefficient;
- activation energy;
- thermodynamic/phase stability;
- dynamic stability;
- electrochemical stability evidence;
- elastic/mechanical properties;
- electronic conductivity;
- air/moisture sensitivity;
- interface/decomposition evidence;
- synthesis/process state;
- criticality/resource tags where separately sourced.

A benchmark may have missing planes for many rows. Missing is not failure.

## Negative-result custody

The benchmark must retain:

- low conductivity;
- unstable phase;
- solver nonconvergence;
- insufficient trajectory duration;
- no observed hopping under bounded simulation;
- contradictory measurements;
- failed synthesis where source evidence exists;
- unresolved phase identity;
- ambiguous EIS decomposition.

Do not build only from successful superionic conductors.

## Contradictory observations

Conflicting literature values remain separate evidence records.

Possible disposition:

```text
CompatibleAndConsistent
CompatibleButDisagree
ConditionMismatch
SubjectIdentityAmbiguous
MethodAssignmentAmbiguous
InsufficientMetadata
```

No hidden averaging across incompatible specimen/process states.

## ML training/test exposure

For every model used in retrospective evaluation record where knowable:

- provider-declared training cutoff;
- exact training corpus if open;
- fine-tuning corpus;
- retrieval/web/tool access during trial;
- whether benchmark records may be in pretraining;
- whether structure/provider records overlap training datasets.

Classifications may include:

```text
ExposureControlled
ExactCorpusOverlapKnown
ProviderCutoffOnly
PostCutoffExposurePossible
ExposureUnknown
```

A model with unknown historical exposure cannot earn strong hidden-discovery credit.

## Benchmark manifests

Future materialization should split into at least:

```text
SSE-DATA-001B
  exact source manifest + source receipts

SSE-DATA-001C
  canonical subject/alias/leakage graph

SSE-DATA-001D
  frozen observation corpus

SSE-DATA-001E
  independent corpus/custody validator
```

Do not combine source capture, normalization, splitting, and scoring into one opaque import script.

## Independent validation requirements

An independent validator should verify at minimum:

- exact source manifest identity;
- unique canonical row/observation IDs;
- aliases resolve deterministically;
- no duplicate primary benchmark slots;
- required condition fields are present or explicitly Unknown;
- physical/computational origins remain distinct;
- train/test leakage rules are reproduced independently;
- temperature/doping trajectories do not straddle material-holdout splits;
- source cutoff rules are respected;
- post-cutoff sources cannot enter frozen G0;
- transformed/normalized values bind their exact source observations;
- no one scalar silently overwrites contradictory evidence.

## Relationship to ION-001

SSE-DATA may proceed through custody architecture while ION implementation remains unqualified.

However:

```text
benchmark row represented
!= ION transport evidence established
```

Production benchmark consumers should use qualified ION contracts once available rather than inventing SSE-local diffusion/conductivity types.

## Relationship to MAT-MLIP

The benchmark should later support transport-specific model calibration such as:

- force/energy errors on transport-relevant configurations;
- barrier/path agreement;
- diffusion/activation agreement under exact state;
- OOD slices;
- finite-size/time sensitivity;
- high-temperature-to-lower-temperature extrapolation behavior.

Preserve:

```text
low force RMSE
!= transport capability
```

## Relationship to prospective SSE discovery

Retrospective benchmark success is not the final theorem.

Required later progression:

```text
frozen retrospective benchmark
-> calibrated evaluator profiles
-> frozen candidate universe
-> frozen ranking/controls
-> fresh high-fidelity evaluation
-> independent scorecard
-> later physical validation
```

Do not tune the prospective campaign on its withheld answers.

## External design motivation

Architecture motivation only; these sources do not become Symthaea evidence without exact custody/materialization:

- Yan, Tang & Zhu, 2026, transport-focused MLFF perspective:
  https://www.nature.com/articles/s44456-026-00014-4
- Choong et al., 2025/2026, ML+DFT halide screening including AIMD candidate verification:
  https://doi.org/10.1021/acsaem.5c03277
- curated experimental lithium-ion conductivity database:
  https://www.nature.com/articles/s41524-022-00951-z
- 2026 amorphous halide electrolyte work:
  https://www.nature.com/articles/s41467-026-69737-x

## Adversarial corpus requirements

The future custody corpus should include at least:

1. same formula, different phase -> distinct subjects;
2. same material, different vacancy state -> distinct subjects;
3. same specimen measured at multiple temperatures -> grouped trajectory, not duplicate independent materials;
4. two papers reproduce one source table -> source-ancestry relation retained;
5. room-temperature measured value vs extrapolated value -> distinct observation types;
6. EIS bulk assignment vs total conductivity -> distinct propositions;
7. unknown density/geometry -> remains Unknown;
8. changed equivalent-circuit fit -> new normalized-observation identity;
9. same numeric conductivity from MD and experiment -> distinct origins;
10. computational barrier cannot substitute for conductivity;
11. post-cutoff paper accidentally inserted -> reject G0;
12. alias causes same subject in train and test -> reject split;
13. temperature rows of one material split across unseen-material train/test -> reject classification;
14. dopant series leakage -> expose graph relation;
15. contradictory compatible experiments -> both retained;
16. failed/nonconverged candidate omitted -> reject complete benchmark census;
17. source corrected/retracted -> new benchmark generation, historical receipt retained;
18. model training overlap unknown -> hidden-discovery claim remains bounded;
19. review article value copied without primary-source custody -> reject strong observation authority;
20. benchmark result promoted to battery safety/cycle life/manufacturability -> reject authority promotion.

## Claim ceiling

SSE-DATA-001A establishes only benchmark design/custody requirements. A later qualified benchmark may establish a frozen, provenance-complete retrospective evaluation corpus under declared split/exposure rules. It does not establish any electrolyte as fast, stable, safe, synthesizable, manufacturable, commercially useful, or prospectively discovered.