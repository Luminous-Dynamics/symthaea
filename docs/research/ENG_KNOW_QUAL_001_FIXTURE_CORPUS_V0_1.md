# ENG-KNOW-QUAL-001 — fixture corpus v0.1

**Tracking:** [issue #10652](https://github.com/Luminous-Dynamics/symthaea/issues/10652)  
**Benchmark specification:** [benchmarks/engineering-knowability/README.md](../../benchmarks/engineering-knowability/README.md)  
**Machine-readable manifest:** [manifest.json](../../benchmarks/engineering-knowability/manifest.json)  
**Semantic mapping matrix:** [mapping-matrix.csv](../../benchmarks/engineering-knowability/mapping-matrix.csv)

## Purpose

This corpus evaluates whether an engineering agent can identify an admissible evidence path for a claim on the final realized system. It is not enough to show that a nominal feature exists in CAD or that a sensor symbol appears in a design.

The benchmark is a composition layer over canonical owners:
- ENG-DESIGN-006 #6124: design-for-knowability intent/review semantics.
- ENG-MEAS #4919: measurands, equations and uncertainty propagation.
- ENG-TOL #4883: tolerances, fit and as-built conformance.
- FIELD/SE-OBS #3695: physical observation and lineage.
- ENG-REF #6121: model–fixture–article campaign closure.
- ETK #1738/#1730/#1760: evidence admission, applicability/currentness and authority.
- ROB-REALIZE #4859: article/as-built/metrology lineage.
- CAD-HISTORY #10510 and HARDWARE-TOPOLOGY #10530: reused geometry/assembly/ECAD source fixtures.

No second metrology engine, observability authority, calibration authority, inspection database, experiment scheduler or lifecycle evidence system is created.

## Public case inventory

| Case | Reused fixture | Evaluation target |
|---|---|---|
| KNOW-P01 | CADH-P01 | Hole/datum measurement paths before and after the length intervention |
| KNOW-P02 | CADH-P03 | Hidden cavity properties before versus after enclosure closure |
| KNOW-P03 | ASM-P01 | Post/base, beam/post, tie/post and gusset interfaces in assembled state |
| KNOW-P04 | ASM-P02 | Bearing/shaft alignment after pedestal remount |
| KNOW-P05 | CADH-P01, ASM-P01 | Common datum, instrument and calibration dependencies |
| KNOW-P06 | CADH-P01, CADH-P04 | Measurement uncertainty against case-specific tolerance and decision boundary |
| KNOW-P07 | ECAD-P01 | Logical-net correctness versus physical testability and instrumentation access |
| KNOW-P08 | ASM-P01, CADH-P06 | Component service/replacement and requalification of registration/identity |
| KNOW-N01 | CADH-P03 | Inaccessible post-closure cavity falsely claimed as measured |
| KNOW-N02 | ASM-P02 | Stale/misbound calibration hidden behind an unchanged numeric result |
| KNOW-N03 | ECAD-P01 | Physical diagnosis claimed from logical connectivity without routed board or test evidence |
| KNOW-N04 | CADH-P01 | Contact measurement materially perturbs a compliant feature without a model |

All 12 cases are public blueprints. They are not adjudicated gold records, sealed holdout cases or scored tasks.

## Evidence-path record expected from a candidate

For each critical claim, the candidate should identify exact references for:
1. Claim, requirement and target configuration/article generation.
2. Physical measurand and direct versus derived observation.
3. Feature, datum, coordinate frame and lifecycle state.
4. Measurement method/instrument family and access path.
5. Instrument, fixture, calibration and reference artifact.
6. Measurement equation, uncertainty and relevant covariance/common-mode dependencies.
7. Decision rule and tolerance scope.
8. Back-action or disturbance, where material.
9. Service/remount/repair requalification consequences.
10. Evidence class, limitations and Unknown/Unsupported state.

A proposed evidence path is not a measurement-system capability result. A calibration reference is not current merely because it exists. Two readings are not independent merely because they are separately recorded.

## Standard-scope notes

The manifest references ISO 14253-1:2017, ISO 10360-2:2009, JCGM 100:2008/Amd.1:2026, JCGM GUM-5:2026 and JCGM 106:2012.

ISO 10360-2:2009 is specifically for Cartesian CMMs using contacting discrete-point probing for linear dimensions. It is not a generic optical scanner or non-Cartesian instrument standard. ISO 14253-1:2017 is used for in-scope quantity-valued conformity decisions taking measurement uncertainty into account. No universal 4:1 uncertainty-to-tolerance ratio is assumed.

## Claim ceiling

A passing synthetic result would establish only that the exact candidate profile planned admissible evidence paths on the evaluated fixtures. It would not establish actual instrument capability, calibration, measurement uncertainty, physical observability, product conformity, safety, approval or release authority.
