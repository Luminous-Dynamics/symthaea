# TE-BENCH-001A — Half-Heusler thermoelectric benchmark custody architecture

Status: architecture/data-custody design only

Date: 2026-09-26

Related:

- TE-001 #4994
- TE-CORE-001A #6074
- MAT-015 external-provider provenance
- MAT-MLIP-001 #4992
- MAT-CALC-001 #5114
- MAT-010 negative/search memory
- MAT-RD-001 #4997

## Purpose

Define an auditable benchmark-custody contract for the first half-Heusler thermoelectric program before any model is allowed to tune on or score hidden benchmark answers.

The benchmark must test compatible-state thermoelectric reasoning, not merely formula-to-zT regression.

This subject creates no dataset, no solver execution, no candidate ranking, and no thermoelectric scientific result.

## Core theorem

```text
formula in database
!= exact phase/material state
!= exact carrier/doping state
!= exact temperature state
!= compatible component transport evidence
!= zT evidence
!= unseen-material benchmark row
!= prospective discovery
```

And:

```text
same parent compound at many T/n rows
!= many independent materials
```

## Benchmark generations

Use immutable generations.

```text
TE-BENCH-G0
  design/custody freeze: 2026-09-26
  retrospective half-Heusler benchmark
  no prospective claim

TE-PROSPECT-G1
  later frozen candidate/ranking/control commitment
  high-fidelity target answers withheld until after commitment
```

New sources, corrected data, changed split policy, or changed target definitions create a new generation.

## Why half-Heuslers first

Half-Heuslers provide a bounded structural family with substantial computational and experimental history, known doping/alloy trajectories, and clear multi-physics tension among electronic transport, lattice thermal conductivity, stability, and carrier optimization.

That makes them useful for testing whether Symthaea can distinguish:

- parent material from doped derivative;
- carrier sweep from explicit dopant state;
- power factor from zT;
- harmonic stability from anharmonic transport;
- computational transport from measured specimen behavior;
- material property from device performance.

## Source-cutoff semantics

Every benchmark generation binds:

- exact cutoff date/time policy;
- publication-online/version-of-record dates;
- provider snapshot/export identity;
- DOI/provider/source identifiers;
- exact source artifact/receipt where permitted;
- retrieval timestamp as provenance only;
- source license/redistribution constraints;
- parser/extraction profile;
- model exposure/training overlap state.

Preserve:

```text
published before cutoff
!= absent from model pretraining

live provider record
!= frozen source snapshot
```

## Source classes

Represent explicitly:

```text
PeerReviewedArticle
SupplementaryArtifact
ExperimentalTransportTable
ComputedTransportDataset
MaterialsProviderRecord
StructureArtifact
PhononOrIFCArtifact
IndependentReplication
ReviewOrHighlight
DerivedSecondaryCompilation
```

Editorial/review material may motivate benchmark design but cannot substitute for primary transport evidence.

## Initial source families

The first source manifest may investigate, subject to exact custody/licensing:

1. established experimental half-Heusler transport datasets and literature;
2. experimentally validated ML-guided ScNiSb-family work;
3. half-Heusler lattice-thermal-conductivity computational datasets;
4. current 2026 hierarchical thermoelectric inverse-design work as workflow motivation/control context;
5. explicit low-performance, unstable, or computationally failed half-Heuslers;
6. provider/database structure and stability records where exact snapshots are available.

Published candidates are retrospective targets, not assumed truths or winners.

## Material subject identity

Formula alone is insufficient.

Bind where known:

- canonical composition;
- half-Heusler prototype/space group;
- exact structure artifact;
- ordered/disordered state;
- dopant species;
- dopant sites/fractions;
- alloy/substitution state;
- defect/compensation state;
- phase purity/multiphase state for physical specimens;
- sample/process lineage;
- source aliases;
- deterministic subject identity.

```text
ScNiSb
!= Sc0.7Y0.3NiSb0.97Sn0.03
!= every specimen of either formula
```

## Thermoelectric state identity

Every transport record binds a state, including where applicable:

- temperature;
- carrier type;
- carrier concentration or chemical potential;
- measured vs imposed carrier state;
- dopant/alloy configuration;
- pressure/strain;
- crystallographic/measurement direction;
- polycrystal/single-crystal scope;
- microstructure/density/porosity;
- specimen identity;
- measurement/calculation method.

Changing any claim-critical state creates distinct evidence.

## Component-property custody

Keep independent observations for:

```text
S
sigma
sigma/tau
kappa_e
kappa_l
kappa_total
carrier concentration
Hall mobility
power factor
zT
```

A source-reported zT does not erase the need to capture component conditions when available.

Derived local Symthaea zT must later reference exact compatible components under TE-CORE.

## Do not normalize away carrier physics

Carrier optimization often produces curves over chemical potential/doping and temperature.

Preserve raw trajectories/grids before creating summaries.

```text
maximum zT over any carrier concentration
!= material property at one realized carrier state
```

Any optimum summary must bind:

- search grid/domain;
- carrier model;
- scattering model;
- state at optimum;
- whether the carrier state is computational, chemically explicit, or measured;
- optimization procedure.

## Electronic-transport method custody

Capture where applicable:

- electronic-structure source;
- DFT/evaluator profile;
- SOC treatment;
- k mesh/interpolation;
- Boltzmann transport solver/profile;
- scattering model;
- relaxation time or `sigma/tau` convention;
- carrier concentration/chemical-potential grid;
- convergence evidence;
- units/tensor convention.

Preserve:

```text
sigma/tau
!= absolute sigma
```

## Lattice-transport method custody

Capture where applicable:

- harmonic force-constant/phonon evidence;
- second/third/higher-order IFC identities;
- supercell/q mesh;
- isotope/scattering profile;
- boundary/grain-size assumptions;
- RTA vs iterative transport method;
- temperature grid;
- q-mesh/supercell convergence state;
- exact solver/evaluator identity.

Preserve:

```text
harmonic stability
!= kappa_l
```

## Experimental custody

Physical observations should bind where available:

- specimen/sample ID;
- synthesis/process route;
- phase/composition characterization;
- density/porosity;
- grain size/microstructure;
- dopant state;
- geometry/orientation;
- instrument/calibration;
- raw/processed artifact refs;
- measurement temperature;
- uncertainty;
- whether component measurements came from the same specimen/state.

Do not combine best S, sigma and kappa from unrelated specimens without explicit compatibility evidence.

## Leakage graph

Represent related rows/materials explicitly.

Relations may include:

```text
ExactDuplicate
SameParentCompound
SamePrototype
SameDopingSeries
SameAlloySeries
SameTemperatureTrajectory
SameCarrierTrajectory
SameSpecimen
SameSourceExperiment
SameComputedBandStructure
SamePhononArtifact
SameProviderRecord
SharedTrainingDataset
PostCutoffExposurePossible
SemanticNearDuplicate
```

## Required split families

### Parent-compound holdout

All T/carrier rows of one parent material remain together for unseen-material evaluation.

### Doping/alloy-family holdout

Closely related substitution trajectories remain grouped where the question is cross-family generalization.

### Prototype/structure holdout

Where enough structural variation exists, preserve a stronger structural generalization split.

### Chemistry-family holdout

Hold chemically related parent groups together to avoid elemental-neighbor leakage.

### Temporal/source holdout

Freeze exact source cutoff and evaluate later sources separately.

### State-interpolation benchmark

Temperature/carrier interpolation can be a separate valid task, but must not be labeled unseen-material discovery.

## First benchmark cohort

The first bounded cohort should contain:

- known experimentally studied half-Heusler thermoelectrics;
- ScNiSb-family parent and doped derivatives as one explicit family lineage;
- additional p-type and n-type half-Heusler families;
- high-kappa_l/low-performance controls;
- dynamically or thermodynamically unstable candidates where evidence exists;
- calculation failures/nonconvergence where recoverable;
- candidates with strong power factor but unfavorable kappa_l;
- candidates with low kappa_l but poor electronic transport.

The benchmark should not contain only successful thermoelectrics.

## Target planes

No universal `TE_score`.

Track independent target/evidence planes such as:

- phase/hull stability;
- dynamic stability;
- Seebeck coefficient;
- electrical conductivity or sigma/tau with explicit distinction;
- carrier concentration;
- power factor;
- electronic thermal conductivity;
- lattice thermal conductivity;
- total thermal conductivity;
- compatible derived zT;
- experimental vs computational origin;
- temperature range;
- chemistry/structure family;
- failure/nonconvergence state.

## Retrospective benchmark questions

Separate questions rather than one leaderboard.

Examples:

```text
Q1: can the method identify stable half-Heusler subjects?
Q2: can it reproduce component electronic-transport trends under exact carrier/T state?
Q3: can it reproduce kappa_l under qualified methods?
Q4: can it derive compatible zT without state mixing?
Q5: can it rank high-fidelity evaluation candidates better than simple controls?
```

Success on Q1 cannot mint success on Q4/Q5.

## Exposure and contamination

For every model/profile record where knowable:

- declared training cutoff;
- exact open training corpus;
- structure/property database overlap;
- fine-tuning/RAG sources;
- web/tool access during trial;
- known overlap with benchmark subjects;
- unknown exposure.

Possible classifications:

```text
ExposureControlled
ExactCorpusOverlapKnown
ProviderCutoffOnly
PostCutoffExposurePossible
ExposureUnknown
```

Do not call retrospective rediscovery hidden discovery when exposure cannot be bounded.

## Baselines

The first benchmark should include simple transparent baselines where applicable:

- elemental/descriptor regression;
- nearest-neighbor composition/prototype baseline;
- simple known-family heuristic;
- random feasible candidate ordering;
- diversity-only ordering;
- calibrated ML surrogate later.

Do not assume the sophisticated method wins.

## Source/materialization train

Recommended sequence:

```text
TE-BENCH-001A
  benchmark custody/split architecture

TE-BENCH-001B
  exact source manifest + cutoff receipts

TE-BENCH-001C
  canonical subject/alias/leakage graph

TE-BENCH-001D
  frozen condition-bearing observation corpus

TE-BENCH-001E
  independent custody/split validator

TE-BENCH-002
  retrospective scorer over qualified TE-CORE/evaluator owners
```

Do not put scraping, normalization, splitting, transport derivation and scoring into one script.

## Independent validator requirements

The reference validator should independently prove:

- exact source-manifest identity;
- exact corpus bytes;
- unique subject/observation identities;
- exact 1:N parent-to-state trajectories;
- aliases do not create train/test duplicates;
- same parent compound cannot straddle unseen-material splits;
- doping/alloy family grouping follows the frozen policy;
- `sigma/tau` never appears as absolute sigma;
- incompatible T/carrier/direction states cannot form a derived zT row;
- experimental components from incompatible specimens are not silently combined;
- post-cutoff sources cannot enter frozen G0;
- missing values remain missing;
- failures and nonconvergence remain in the census.

## Relationship to TE-CORE

TE-BENCH custody work may proceed while TE-CORE implementation is unqualified.

But:

```text
benchmark record represented
!= qualified TE transport evidence
```

Production scoring should consume qualified TE-CORE contracts rather than inventing duplicate benchmark-local transport semantics.

## Relationship to 2026 inverse design

Current inverse-design work is useful as architecture motivation because it explicitly uses hierarchical screening from generated candidates through ML, DFT, electronic transport, phonons, and third-order force constants.

Symthaea should preserve that fidelity separation while adding stricter benchmark custody, exposure tracking, negative-result memory, and preregistered prospective scoring.

The published candidate list is not local Symthaea evidence.

## Relationship to prospective discovery

A strong future theorem is:

```text
retrospective benchmark frozen
+ evaluator calibration frozen
+ candidate universe frozen
+ ranking/controls frozen before expensive answers
+ exact high-fidelity execution
+ complete failure census
+ independent scorecard
-> bounded prospective computational discovery evidence
```

Physical validation remains separate.

## External design motivation

Architecture motivation only:

- Wang et al., 2026, hierarchical thermoelectric inverse design:
  https://www.nature.com/articles/s41524-026-02307-3
- Jia et al., 2022, ML-guided half-Heusler discovery with experimental ScNiSb-family validation:
  https://www.nature.com/articles/s41524-022-00723-9
- Miyazaki et al., 2021, half-Heusler lattice-thermal-conductivity ML dataset/workflow:
  https://www.nature.com/articles/s41598-021-92030-4
- Shafique et al., 2026, high-throughput TE screening demonstrating explicit sigma/tau and later higher-fidelity scattering/phonon treatment:
  https://www.nature.com/articles/s41699-026-00715-z

## Adversarial corpus requirements

The future custody corpus should include at least:

1. same parent material at 300 K/train and 700 K/test -> reject unseen-material split;
2. same parent with two carrier concentrations -> grouped parent identity retained;
3. rigid-band row and explicit doped specimen -> distinct states;
4. sigma/tau copied into sigma field -> reject;
5. changed relaxation-time model -> distinct evidence;
6. S and sigma from incompatible carrier states -> no PF/zT composition;
7. kappa_l from another phase -> no zT composition;
8. in-plane and cross-plane components mixed -> reject without exact tensor policy;
9. harmonic stability used as kappa_l -> reject;
10. experimental values from different specimens silently combined -> reject;
11. one source table mirrored by a second database -> ancestry preserved;
12. alias puts same subject in train and test -> reject split;
13. dopant-series neighbors cross holdout boundary against policy -> reject;
14. source after cutoff enters G0 -> reject;
15. missing carrier state converted to zero -> reject;
16. conflicting compatible experiments -> retain both;
17. dynamically unstable high-zT calculation -> preserve transport evidence and stability failure independently;
18. nonconverged high-fidelity candidate omitted -> reject complete census;
19. generator candidate treated as scientific evidence -> reject authority promotion;
20. benchmark PASS promoted to synthesized material/device efficiency -> reject authority promotion.

## Claim ceiling

TE-BENCH-001A establishes only benchmark design/custody requirements. A later qualified benchmark may establish a frozen retrospective half-Heusler evaluation corpus under declared split, condition, source and exposure rules. It does not establish any thermoelectric material as stable, high-zT, synthesizable, experimentally validated, manufacturable, device-efficient, commercially useful, or prospectively discovered.