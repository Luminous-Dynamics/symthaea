# Subterra Asset Observability Benchmark v0.1

**Research cut:** 17 November 2026  
**Status:** benchmark proposal; no sensor system, defect detector, volumetric estimator, simulator or field deployment has been qualified.

This benchmark is tracked by [#12609](https://github.com/Luminous-Dynamics/symthaea/issues/12609).

## Why this benchmark

A 2026 review of underground mine robotics identifies dedicated pipe/cable inspection and drawpoint hang-up volumetric assessment as largely unaddressed compared with post-blast re-entry and ground-condition assessment. This creates a more defensible research wedge than another generic tunnel-navigation benchmark.

Reference: Naheem, Frimpong & Antwi, *Underground Mine of the Future: A Review of Robotic Technology Intervention for Infrastructure Resilience*, Frontiers in Built Environment (2026), DOI: 10.3389/fbuil.2026.2017682.

The benchmark deliberately separates two different physical problems:

- Utility corridor: surface/asset defect detection and condition assessment.
- Drawpoint: partial-observation 3-D occupancy and blockage-volume estimation.

Do not combine their scores into one metric.

## Paradigm: asset-condition observability

The system must report not only what it believes is present, but:
- which surfaces and volumes were observed;
- which regions are occluded, inaccessible or unobserved;
- which sensor and calibration produced each observation;
- whether sensors share a common failure mode;
- how uncertainty changes with viewpoint and sensing modality;
- when it must abstain rather than label an asset clear.

A waypoint visit is not inspection completion. A detection is not a verified defect. A reconstructed free-space volume is not permission to enter or manipulate.

## Track A — utility corridor

### Scenario

Generate synthetic pipe/cable corridors with bends, junctions, hangers, supports, crossings, mixed materials, reflective/wet surfaces and inaccessible spans.

Defect families:
- corrosion-like surface loss and pitting;
- cracks and deformation;
- missing, displaced or failed supports;
- insulation damage / exposed-conductor proxy;
- thermal anomaly;
- leak or wetness proxy.

Inputs may include RGB, depth/LiDAR, thermal, acoustic/vibration and electromagnetic measurements. Every modality must carry explicit calibration, noise, dropout and field-of-view metadata.

### Outputs

- defect class and severity band;
- 3-D position, extent and localization uncertainty;
- calibrated probability / abstention;
- observed surface coverage and unknown region;
- sensor lineage, calibration revision and evidence digest.

### Baselines

1. Human inspection under a declared time budget.
2. Image-only detector.
3. Geometry-only inspection.
4. Multimodal late fusion.
5. Calibrated probabilistic fusion.
6. Active viewpoint / value-of-information inspection.
7. Independent reviewer using the same evidence budget.

## Track B — drawpoint hang-up

### Scenario

Generate drawpoint/orepass geometry with variable block size and shape, interlocking topology, fines, dust, water, changing illumination and safe stand-off constraints.

### Outputs

- blocked/occupied volume and uncertainty interval;
- topology and likely blockage region;
- estimated free-space volume;
- observed/occluded volume coverage;
- explicit abstention when geometry is not identifiable.

### Baselines

1. Single-view depth.
2. Multi-view depth fusion.
3. LiDAR reconstruction.
4. Human volumetric estimate.
5. Active viewpoint selection.
6. Sensor-ensemble estimator with calibration.
7. Independent evaluator with sealed ground truth.

No estimate may authorize dislodging, blasting, entry or manipulation.

## Fault model

Freeze per-track cases for:
- dust, water, low light, reflective surfaces and thermal drift;
- missing modality and sensor misregistration;
- occlusion, partial views and limited safe viewpoints;
- stale calibration and coordinate-frame/unit mismatch;
- correlated sensors that share a common-mode failure;
- adversarial or duplicated observations;
- material/asset change between views;
- out-of-distribution materials and defect morphology;
- communication loss and delayed evidence.

Split train/validation/test by site, asset geometry, material family and capture session—not adjacent frames.

## Metrics

Report per track and per fault stratum:
- per-class precision/recall and severity-weighted miss cost;
- recall at fixed false-alarm rate;
- 3-D localization error and defect extent error;
- volumetric IoU and absolute volume error;
- surface/volume coverage and unknown fraction;
- Brier score, ECE and reliability curves;
- OOD detection and risk-coverage / abstention curves;
- common-mode fault detection;
- time, energy, revisit burden and evidence latency.

Do not hide false negatives or false-clear claims inside a single weighted score.

## Proposed targets — hypotheses, not results

- Zero false claims of complete inspection when required surfaces/volumes are occluded in the frozen negative-control suite.
- Zero conversion of unknown/unobserved geometry into verified clear space.
- 100% rejection of stale calibration and frame/unit mismatch controls.
- Publish calibration and risk-coverage curves, not accuracy alone.
- Stretch: at least 20% lower inspection time or energy than matched human/robot baselines at non-inferior defect recall and false-alarm rate, with confidence intervals and held-out sites.

These are research targets, not claims of current capability.

## External comparison context

- The 2025 review *Autonomous Mobile Inspection Robots in Deep Underground Mining* (Sensors 25(12), 3598) surveys mobile inspection platforms including legged, tracked and aerial systems.
- The 2025 gecko-inspired underground cable inspection study demonstrates a robot-plus-vision defect-detection approach; it is a comparator, not a universal industrial baseline.
- The 2026 Frontiers review explicitly identifies dedicated pipe/cable and drawpoint volumetric assessment as gaps.
- The 2026 multi-robot mine inspection planning paper reports simulation-based charging-aware task allocation and path planning. Its reported cost reduction (7.20–11.16% against its compared heuristics) is paper-specific and should not be treated as field performance.
- CAVERS and SubT datasets are useful for navigation and sensor observability, but do not provide ground truth for these defect or hang-up classes.

## System ownership

- Symtropy: synthetic geometry, material/asset state and sensor physics.
- Symthaea: inspection policies, active sensing, uncertainty and abstention.
- Mycelix: identity, authority, provenance and event reconciliation.
- Sol-Atlas: rights-aware asset/utility spatial context.
- Independent evaluator: sealed ground truth, score computation and audit trail.

## Claim ceiling

Synthetic success qualifies only the declared synthetic task and sensor models. It does not establish real mine inspection reliability, electrical safety, structural integrity, drawpoint safety, geotechnical competence or permission to intervene. Field validation requires mine-owner authorization, specialist review, safe test facilities and independently measured ground truth.
