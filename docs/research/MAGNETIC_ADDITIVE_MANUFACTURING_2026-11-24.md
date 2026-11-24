# MAG-AM-001 — Recycled anisotropic Nd–Fe–B additive manufacturing benchmark

**Revision:** 0.1.0  
**Prepared:** 2026-11-24  
**Status:** literature-baseline admission and prospective protocol; not an experiment, reproduction, or SOTA result  
**Owner:** Symthaea materials research, composed with Symtropy process-twin work

## Executive thesis

Treat a permanent magnet as the outcome of a coupled **feedstock → process → realized microstructure → conditioned magnetic properties → component performance** chain.

The first bounded test case is recycled end-of-life (EOL) hot-deformed Nd–Fe–B powder in a PA12 matrix, fabricated by laser powder-bed fusion (LPBF). This is not the rare-earth-free Fe–Co–X discovery campaign. It is a circular-materials and process-engineering problem involving a rare-earth-containing magnet.

The relevant paradigm shift is to stop optimizing chemistry or a single headline magnetic number in isolation. Co-design particle morphology, particle-size distribution (PSD), packing, layer deposition, geometric/crystallographic texture, magnetic-field history, and binder/thermal process conditions. Judge the result on a Pareto surface that includes remanence, coercivity, energy product, porosity, dimensional accuracy, durability, manufacturing yield, cost, and circularity.

No claim of a new material, improved process, or industry-leading performance is made by this register.

## Why this is a high-value first experiment

Schäfer et al. published a reproducible, unusually specific starting point on 5 January 2026:

- Article: [Fabrication of Sustainable 3D Printed Anisotropic Bonded Magnets Using Recycled Nd–Fe–B Powder and Low CO2 Footprint Polyamide 12](https://doi.org/10.1002/aesr.202500339)
- Journal: *Advanced Energy and Sustainability Research*
- DOI: 10.1002/aesr.202500339
- Source access: open article; supporting data are available from the corresponding author on reasonable request, not declared as an unrestricted public raw-data repository.

The study directly uses EOL hot-deformed Nd–Fe–B. It compares hydrogen-decrepitated/desorbed (HD) and milled powder routes, then LPBF-prints PA12-bonded composites. It reports printed remanence of 0.34 T and coercivity of 1,238 kA/m for the HD-derived route, with 32.6% porosity and an alignment degree of 0.75. The milled route reports 0.32 T, 1,226 kA/m, 28.3% porosity and alignment degree 0.65.

The authors state that remanence remains about 60% below their cited state-of-the-art commercial anisotropic bonded-magnet comparator, while coercivity is in the commercial anisotropic bonded-magnet range. They attribute the remanence gap primarily to LPBF porosity and incomplete particle alignment. They identify PSD blending, smaller layer height, and a homogeneous magnetic field during printing as research directions. These are hypotheses to test, not established process improvements.

A September 2026 review, [State-of-the-art additive manufacturing of NdFeB permanent magnets: challenges and pathways toward high performance](https://doi.org/10.1016/j.jsamd.2026.101226), reports that remanence remains a limiting factor across printed NdFeB routes. It identifies material extrusion as the highest-remanence route among those reviewed and notes the difficulty of LPBF in producing magnetic texture. This is a cross-study literature comparison, not a matched head-to-head result.

## What is already owned elsewhere

This work should compose—not reimplement—the following:

| Owner | Existing responsibility | Use here |
|---|---|---|
| Symthaea MAG-001 #4347 | Rare-earth-free Fe–Co–X discovery | Adjacent discovery campaign; do not merge its chemistry scope |
| Symthaea MAG-002 #4437 | Intrinsic vs microstructure-bound magnetic properties | Keep Js/Ms/K1/Tc distinct from Hc/Br/(BH)max |
| Symthaea MAG-BENCH-002 #5115 | Leakage/contamination and holdout boundaries | Freeze target visibility, splits, source and model exposure |
| Symthaea MAT-PSPP #6224 | Process → structure → property → performance | Reuse process/microstructure lineage |
| Symthaea MAT-ENG #6185 | Engineering demand ↔ materials discovery ↔ engineering qualification | Prevent candidate-to-engineering evidence shortcuts |
| Symthaea MAT-CLOSEDLOOP #6300 | Closed-loop materials proving campaigns | Use its magnet/motor pilot composition |
| Symthaea ENG-MAG #5677 | Magnetic assembly/model/field semantics | Reuse predicted/measured field separation |
| Symtropy PWR-TX-MFG #2933 | Manufacturing digital thread | Reuse product, lot, process, inspection and receipt lineage |
| Symtropy WPT magnetothermal #3487/#3490 | Ferrite WPT magnetic/thermal co-design | Reuse matched measurement and Pareto methods only; different material and device |

Symtropy's role is process scenarios, deterministic replay, process-twin integration, sensitivity and evidence linkage. It is not assumed to be a validated DEM, magnetostatic, micromagnetic, or thermal solver. Any such numerical backend must be independently qualified and bound to exact inputs, version, configuration, boundary conditions and execution receipts.

## Source-reported baseline

### Feedstock powders

| Property | HD + desorbed | Milled | Unit |
|---|---:|---:|---|
| D10 | 43 | 27 | μm |
| D50 | 102 | 114 | μm |
| D90 | 276 | 210 | μm |
| Mean particle aspect ratio | 2.8 | 2.2 | dimensionless |
| Theoretical density | 7.6 | 7.6 | g/cm³ |
| Remanence | 1.22 | 1.12 | T |
| Coercivity | 1,269 | 1,290 | kA/m |

### LPBF printed composites

| Property | HD-derived | Milled | Unit |
|---|---:|---:|---|
| Magnetic filler fraction | 89.6 | 89.8 | wt% |
| Theoretical density | 4.6 | 4.6 | g/cm³ |
| Geometrical density | 3.1 | 3.3 | g/cm³ |
| Porosity | 32.6 | 28.3 | % |
| Remanence | 0.34 | 0.32 | T |
| Coercivity | 1,238 | 1,226 | kA/m |
| Magnetic alignment degree | 0.75 | 0.65 | dimensionless |

The paper's alignment degree is calculated from the remanent-polarization ratio between hard and easy directions using the stated Fernengel-derived relationship. Preserve the original definition when reproducing; do not silently substitute a different texture statistic.

**Source transcription discrepancy:** Table 2 reports HD + desorbed powder D90 = 276 μm, while the later discussion text states D90 = 279 μm. The machine-readable case records both and treats the value as unresolved. Use a full source PSD or explicit uncertainty for particle-resolved modelling; do not silently choose a single exact D90.

### Reported LPBF setup

| Parameter | Source value |
|---|---:|
| Machine | Sinterit Lisa Pro |
| Laser | 5 W diode, 808 nm |
| Powder-surface temperature | 177.5 °C |
| Print-chamber temperature | 140.0 °C |
| Laser spot | 0.4 mm |
| Hatch distance | 0.4 mm |
| Layer height | 150 μm |
| Characterization coupon | 5 mm cube |
| Nominal magnetic powder fraction | 90 wt% |
| Powder mixing | 15 min in a TURBULA mixer |

These values describe the reported experiment. They are not a validated recipe for another machine, another powder lot, or another feedstock distribution.

### Measurement notes

The paper reports PPMS-VSM measurements of bonded-magnet curves parallel and perpendicular to the build direction at room temperature, up to 5 T. It reports a demagnetization factor N = 0.2 for the correction used in its measurement context. Any replication must verify the applicability of that correction to the exact coupon geometry and measurement setup rather than copy it uncritically.

Powder PSD was measured by laser diffraction; particle morphology was evaluated by SEM; filler content was checked by TGA; polished composite microstructure was inspected by SEM; particle-domain structure was examined by Kerr microscopy. A custom three-axis Hall-probe setup was used for stray-field mapping of printed shapes.

IEC 60404-18:2025 is a relevant reference for open-circuit DC magnetic-property measurements using a superconducting magnet, with SCM-VSM and extraction methods and self-demagnetization correction. Its scope and sample assumptions must be checked before claiming conformity; it is not a blanket standard for every printed coupon, field map, or component test.

## Falsifiable hypotheses

**H1 — packing/texture co-design.** A controlled multimodal PSD can improve packing and reduce porosity without unacceptable loss of particle alignment and remanence.

**H2 — field-assisted particle orientation.** A controlled and measured fabrication-field history can improve the orientation distribution and remanence. It could also increase segregation or create spatially nonuniform orientation; field assistance is not presumed beneficial.

**H3 — layer-height/process interaction.** Layer height interacts with particle aspect ratio, particle torque, packing and deposition. Smaller is not automatically better; the response must be measured across the feasible process window.

**H4 — active experiment selection.** Symthaea-guided, uncertainty-aware selection can find useful process windows with fewer experiments than random, space-filling, standard DOE, and simple physics/descriptor baselines at the same experiment and compute budget.

**H5 — manufactured usefulness.** Improvements in coupon Br or alignment matter only if they transfer to a specified component geometry, field map, mechanical envelope and lifecycle requirement.

## Proposed experiment sequence

### Stage 0 — source admission and target custody

1. Freeze the exact article DOI/version, source capture date, supporting-information identity and rights/access conditions.
2. Record which facts are directly reported, digitized from figures, inferred, or not available.
3. Do not train or tune on the held-out evaluation answers. Unknown prior training exposure stays unknown.
4. Seek author-provided data through legitimate channels; if unavailable, use only source-published values with declared transcription uncertainty.
5. Do not fabricate raw measurement traces, particle distributions, or missing uncertainty estimates.

### Stage 1 — reproduction

Reproduce the two powder-route orderings and reported composite property ranges with source-matched geometry and methods to the extent the inputs permit. Establish independent material lots and builds. A paper-value transcription is not an experimental reproduction.

The first physical test protocol should characterize the powder before printing, then the printed specimen:
- PSD and full distribution, not D10/D50/D90 alone;
- aspect ratio, shape and morphology distributions;
- phase/composition and oxidation state where measurable;
- density/porosity and spatial distribution;
- filler fraction and binder distribution;
- easy/hard direction hysteresis, Br, Hc, (BH)max and temperature;
- 3D field mapping with calibrated probe, grid, frame and uncertainty;
- dimensional tolerance and mechanical integrity.

### Stage 2 — causal process DOE

Pre-register a blocked design with factors:
- powder route and lot;
- PSD blend proportions based on measured feedstock;
- layer height over a machine-qualified range;
- field-off sham and controlled field-assisted cases, only after the field apparatus and measurement envelope are independently characterized;
- selected LPBF process variables justified by equipment capability and safety review.

Keep powder lot, binder lot, geometry, build position, environmental state and measurement order blocked or explicitly modeled. Include independent builds, randomized run order, repeated baseline controls, and failed/aborted runs in the denominator. Choose sample size from variance and power analysis. Do not select the best coupon after the fact.

### Stage 3 — process-model and active-learning evaluation

Compare:
- simple analytical packing/orientation models;
- DEM-only packing;
- coupled DEM + magnetic particle-orientation models;
- thermal/process simulation where relevant;
- Symthaea/Symtropy experiment selection;
- random, space-filling, factorial/fractional-factorial and conventional Bayesian/descriptor baselines.

Use equal compute/experiment budgets and the same allowed inputs. Freeze train/validation/test splits by powder lot, build and process window. Report prediction error, uncertainty calibration/coverage, OOD detection/abstention, Pareto improvement per experiment, total compute and wall-clock cost. A model must not see held-out build measurements or target answers during selection.

### Stage 4 — component transfer

Only after coupon-level evidence qualifies, bind an exact component geometry and application demand. Compare component field maps, force/torque where applicable, thermal/environmental durability, mechanical retention, losses and system performance. A coupon result does not qualify a motor, actuator or power-conversion device.

## Benchmark scorecard

Do not collapse the dimensions into one weighted magnet score.

| Axis | Primary measurements | Essential comparison discipline |
|---|---|---|
| Magnetic performance | Br, Hc, intrinsic coercivity where measured, full hysteresis, (BH)max, temperature drift/loss | Same specimen geometry, temperature, direction, field range and demagnetization method |
| Texture | Alignment degree, orientation distribution, microscopy/texture evidence | Preserve source definition and independently measure texture |
| Packing/structure | Porosity, density, phase fraction, binder distribution, defects | Report spatial statistics and measurement method, not only one average |
| Field performance | Calibrated 3D field map, peak/RMS vector components | Same grid, coordinate frame, distance, probe and uncertainty |
| Process | Build yield, dimensional accuracy, cycle time, energy, scrap, rework | Cost and energy per accepted part, not nominal build |
| Reliability | Mechanical performance, thermal cycles, humidity/corrosion, particle retention | Application-specific qualification; accelerated ageing is not field life |
| Circularity | Recycled feedstock fraction, recovery yield, material loss, lifecycle CO2e | Declared functional unit, system boundary, allocation and source quality |
| Intelligence | Error, calibration, OOD abstention, Pareto quality, experiments-to-target | Held-out lot/build, equal input information and equal budget |

## Proposed ambition

The first prospective stretch objective is to close at least half of the *matched, source-defined* remanence gap to a commercial anisotropic bonded-magnet comparator, without a material coercivity penalty, after geometry, temperature, measurement direction, demagnetization correction and method are matched.

This is a research target, not a forecast. Do not infer an absolute commercial remanence from the article's approximate 60% gap statement; obtain the comparator's grade, data sheet or direct measurement and exact test basis first.

A claim of superiority requires:
1. independent replication across builds and ultimately feedstock lots;
2. held-out predictive validation;
3. matched commercial and conventional-process comparators;
4. uncertainty intervals and failed-run accounting;
5. component-level transfer where the intended claim concerns a component;
6. independent review of the measurement/evidence chain.

## Deliverables

1. Literature-source register and machine-readable source-reported baseline.
2. Frozen source/target-visibility and data-admission manifest.
3. Process/structure/property experiment constitution and DOE plan.
4. External solver-adapter capability matrix for DEM, magnetic torque/field, thermal and microstructure prediction.
5. Baseline reproduction report with missing-data and reproduction dispositions.
6. Held-out, matched active-experiment-selection benchmark.
7. Component-transfer report only after independent physical measurements.

## Safety and authority boundary

This is a literature, modelling, software, and protocol workstream. It does not authorize powder handling, hydrogen processing, laser operation, energized coils, machine changes, or physical fabrication. Such work requires suitably qualified facilities, process-hazard assessment, machine interlocks, material-handling controls, and independent authorization. A simulation cannot authorize a physical process.

## Current status

- **Implemented in this tranche:** issue-level research scope and source-pinned benchmark register.
- **Source verified:** the paper's reported settings, feedstock values, composite values, and discussion of the remanence gap and proposed follow-up variables.
- **Not performed:** author data acquisition, local reproduction, solver execution, DEM/EM coupling, laboratory fabrication, physical measurement, or independent replication.
- **Not claimed:** SOTA, new material discovery, process improvement, or product qualification.

## References

1. Schäfer et al. (2026), DOI [10.1002/aesr.202500339](https://doi.org/10.1002/aesr.202500339).
2. *State-of-the-art additive manufacturing of NdFeB permanent magnets: challenges and pathways toward high performance* (2026), DOI [10.1016/j.jsamd.2026.101226](https://doi.org/10.1016/j.jsamd.2026.101226).
3. IEC 60404-18:2025, [Permanent magnet materials — measurement in an open magnetic circuit using a superconducting magnet](https://webstore.iec.ch/en/publication/76495).
4. Symthaea [MAG-AM-001 #13506](https://github.com/Luminous-Dynamics/symthaea/issues/13506).
5. Symtropy [MAG-AM-PROC-001 #7296](https://github.com/Luminous-Dynamics/symtropy/issues/7296).
