# CIV-ENG-BEARING-001A — Integrated Precision-Bearing Benchmark

Issue: #6240  
Parent: CIV-ENG-BEARING-001 #6238  
Schema: `civ-eng-bearing-001a-reference-v1`

## Purpose

Freeze a synthetic docs/data benchmark for the first end-to-end CIV→engineering/materials→component→closure feedback case.

This source defines no bearing design, real dimensions, process recipe, operating rating, manufacturing plan, physical test, procurement action, or machine authority.

## Governing theorem

```text
CIV blocker
!= material bottleneck
!= material candidate
!= process-conditioned material state
!= precision article
!= interface/contact state
!= functionally qualified bearing
!= productive bearing route
!= G2+ renewal
!= downstream closure
```

The benchmark has two symbolic demand profiles:

- `B-L`: low-demand benign fixture/service profile;
- `B-P`: precision productive-equipment profile.

A result valid for `B-L` does not transfer automatically to `B-P`.

## Composition boundary

Canonical owners remain external:

- CIV-BOOT/CIV-ENG: blocker, exact demand, closure generation, closure delta;
- IND-COMP: bearing/component functional qualification;
- MAT-OPPORTUNITY/MAT-ENG/MAT-PSPP/MAT-UQ/MAT-INTERFACE: bounded material/process/interface investigation;
- ENG-MAT/ENG-TOL/ENG-MEAS: exact material applicability, precision and measurement;
- MFG-PROC/MFG-LIFE: process and lifecycle;
- SCI-EXP: experimental-unit and confirmatory-inference discipline;
- PROD-EQP/CIV-BOOT-003: productive-equipment and generational renewal.

This benchmark does not own their physics or authority.

## Corpus encoding

The canonical corpus is stored at:

`docs/engineering/data/civ_eng_bearing_001a_reference_v1.json`

Each case is represented by a shared `base` object plus per-case `o` overrides. An independent oracle must reconstruct raw inputs, derive all results without case-ID branching, and compare against `e` only after derivation.

The ordered axes are:

1. `civ`
2. `bottleneck`
3. `material_research`
4. `material_state`
5. `process`
6. `precision`
7. `interface`
8. `function`
9. `experiment`
10. `uncertainty`
11. `return`
12. `generation`
13. `effect`
14. `authority`

No `bearing_ok`, `self_sufficient`, `localization_score`, or universal readiness scalar exists.

## Routing rules

Declared blocker→owner routes are frozen through the corpus `route_map`.

Key non-equivalences:

```text
high CIV import leverage + no positive material bottleneck evidence
-> no strong materials-research promotion
```

```text
precision/metrology bottleneck
-> ENG-TOL / ENG-MEAS
not generic material discovery
```

```text
surface/interface bottleneck
-> MAT-INTERFACE + component/process owner
```

```text
functional evidence missing
-> IND-COMP admission blocked
regardless of good material or geometry evidence
```

## Material-state boundary

The benchmark keeps separate:

```text
CandidateOnly
QualifiedForArticleProjection
ApplicabilityMismatch
Insufficient
NotApplicable
```

A DFT result, candidate ranking, handbook number, coupon result, or nominal alloy label cannot directly become bearing functional evidence.

For a material-attributed `B-P` return, exact article applicability is required.

## Process boundary

Process capability is independent of material state and can be:

```text
QualifiedUnderProfile
Insufficient
StaleOrMismatched
VenueBlocked
NotApplicable
```

A good material does not compensate for missing precision process capability or venue incompatibility.

## Precision / metrology boundary

The benchmark distinguishes:

```text
FitOnly
QualifiedCurrent
StaleCalibration
Insufficient
NotApplicable
```

Dimensional fit alone is insufficient for the precision productive-equipment profile.

Calibration/currentness remains independently binding.

## Interface boundary

Bulk material support does not establish rolling-contact/interface support.

The benchmark distinguishes current interface support, lifecycle-unresolved interface support, insufficient support, and stale/mismatched support.

Required lubrication/consumable dependencies remain explicit.

## Functional component boundary

IND-COMP remains the owner of component-function qualification.

The benchmark only projects consequences equivalent to:

```text
FunctionObservedDutyUnresolved
LowerDemandOnly
QualifiedUnderExactProfile
Insufficient
NotApplicable
```

`B-L` can legitimately accept a bounded lower-demand substitute while `B-P` remains unsupported.

## Experimental-inference boundary

The benchmark preserves:

```text
100 readings on one article
!= 100 independent bearing articles

many articles from one manufacturing batch
!= independent process-batch replication

adaptive process tuning
!= fresh confirmation
```

Narrow exact-article evidence may remain scientifically useful without supporting a process-population claim.

## Uncertainty boundary

The benchmark separately represents:

```text
RobustSupport
NominalOnly
CrossesHardRequirement
MeasurementDominates
Insufficient
NotApplicable
```

A nominal `B-P` pass whose admitted uncertainty crosses a hard requirement is not a strict capability return.

## Capability return

A `B-P` capability return requires the exact generation/profile and the required precision, interface, function, experiment, uncertainty, process and currentness conditions.

A `B-L` result may be boundedly eligible under weaker declared requirements without becoming equivalent to `B-P`.

Candidate/material/coupon evidence cannot return directly to CIV.

## Generational renewal

The benchmark freezes:

```text
G1Only
G2PlusSupported
GenerationalDegradation
Unresolved
NotEvaluated
```

Local production today does not imply renewable tooling, metrology, material feed, calibration references, abrasives, or consumables.

`G1 success != G2+ renewal`.

## Closure calibration

An admitted return creates a new CIV closure generation. The benchmark preserves:

- no closure gain;
- exact observed closure gain;
- predicted gain overestimate;
- predicted gain underestimate;
- bottleneck migration;
- unresolved effect.

Old closure reports remain historical.

## Frozen 40-case corpus

Cases C01–C40 span:

- blocker/routing attribution;
- material/process evidence;
- precision/metrology;
- surface/interface;
- functional component evidence;
- pseudoreplication/process-generalization controls;
- uncertainty robustness;
- G1/G2+ renewal;
- closure calibration;
- physical-authority non-amplification.

No case contains a real load, speed, geometry, material recipe, heat-treatment recipe, grinding recipe, lubricant recipe, product rating, or safety-critical use.

## Canonical identity

Canonical JSON is UTF-8, recursively sorted keys, separators exactly `,` and `:`, with no insignificant whitespace and no trailing newline.

SHA-256:

`f9558e789ae9095830b50b0066cf5dca6eea0fd23c2ae1b2e3f51984bb46fab6`

Case order is exactly C01…C40.

## Qualification sequence

```text
source/docs/data freeze
-> independent stdlib oracle
-> exact hosted qualification
-> only after qualified/current upstream owners: executable integration adapters
```

As of source freeze, CIV-ENG-001A1, IND-COMP-001A1 and SELF-BOOT-001A1 exact-head hosted qualifiers are queued. Their existence or queue state is not PASS.

## Required independent qualifier properties

A later oracle must:

- hard-bind exact source head/parent and exact doc/corpus blobs;
- verify canonical corpus SHA-256 and schema;
- verify exact axes/codebook/route-map shape;
- verify exact C01–C40 order;
- reconstruct `base + overrides`;
- validate raw-key closure and types;
- independently derive every axis;
- validate blocker→route consistency;
- compare expected codes only after derivation;
- run case-ID-independent mutation tests;
- prove caller-provided authority input cannot change authority output;
- verify qualifier-only diff scope and clean postflight.

## Claim ceiling

A future exact-source PASS may establish only deterministic synthetic integration semantics for blocker attribution, material/process routing, precision/metrology, interface, component function, experiment design, uncertainty, capability return, generational renewal and closure calibration.

It establishes no real bearing rating, real manufacturing process, material truth, load/speed/lifetime performance, local industrial capability, self-sufficiency, economic merit, procurement/fabrication decision, product safety or physical execution authority.