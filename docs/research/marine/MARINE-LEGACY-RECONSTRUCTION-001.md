# MARINE-LEGACY-RECONSTRUCTION-001
## Rights-aware vessel geometry recovery from 2D plans

**Status:** evidence-contract and reproduction protocol v0.1.0  
**Created:** 2026-11-15  
**Tracking issue:** [#12362](https://github.com/Luminous-Dynamics/symthaea/issues/12362)  
**Typed record:** [marine-legacy-reconstruction-evidence.schema.json](./marine-legacy-reconstruction-evidence.schema.json)  
**Synthetic known-answer fixture:** [marine-legacy-reconstruction-synthetic-0001.json](./fixtures/marine-legacy-reconstruction-synthetic-0001.json)

> This work defines evidence handling and benchmark contracts. It does not assert that the synthetic fixture reconstructs a real vessel, that a drawing-derived candidate is as-built geometry, or that any output is a certified CAD model.

## 1. Objective

Recover defensible hull geometry from archival general-arrangement drawings, body/lines plans, sheer and half-breadth plans, offset tables, specifications and partial measurements—while preserving the distinction between what the source says, what the algorithm infers, what remains uncertain, and what has been measured on the actual vessel.

The system must produce an **admissible family or distribution of candidate geometries**, not force one apparently precise hull from incomplete or contradictory evidence.

This work complements forward generative hull geometry in [#12302](https://github.com/Luminous-Dynamics/symthaea/issues/12302) and geometry generation/loft/hydrostatics in [PR #12326](https://github.com/Luminous-Dynamics/symthaea/pull/12326), [#12339](https://github.com/Luminous-Dynamics/symthaea/pull/12339), [#12361](https://github.com/Luminous-Dynamics/symthaea/pull/12361), and [#12369](https://github.com/Luminous-Dynamics/symthaea/pull/12369). Do not fork those geometry-admission and hydrostatic owners.

## 2. Why reconstruction deserves a separate product capability

Forward design begins with declared parameters and requirements. Legacy reconstruction begins with incomplete, noisy and potentially rights-restricted evidence:
- views may be missing, cropped, skewed, damaged or mislabeled;
- scans may not have a trustworthy physical scale;
- annotations and hull lines may overlap;
- different drawings may describe different revisions;
- offsets and dimensions may contradict one another;
- historical design geometry may differ materially from current as-built condition;
- some geometry may be impossible to identify uniquely from the available views.

A reconstruction platform therefore needs document identity, rights, calibration, cross-view constraint reconciliation, uncertainty, human review and later as-built fusion before its geometry can enter the forward engineering pipeline.

## 3. Published baseline and research gap

Raster2Hull (IEEE Access, 31 July 2026; DOI [10.1109/ACCESS.2026.3717961](https://doi.org/10.1109/ACCESS.2026.3717961)) introduces ShipPlan, a synthetic dataset of 5,000 paired rasterized ship drawings and 3D hull point clouds, and a multi-view/template-deformation reconstruction method.

The published work is a valuable baseline, but the dataset and output should not be over-promoted:
- ShipPlan is synthetic and standardized; it does not by itself establish performance on degraded real archival drawings.
- The output is a point cloud, not a watertight mesh, NURBS surface or exact B-rep.
- A point-cloud fit can appear visually good while missing bow/stern geometry, stations, offsets, hydrostatic coefficients or engineering metadata.
- Synthetic in-domain test accuracy does not measure real-fleet sim-to-real generalization.
- Dataset license, code availability, redistribution rights and access to raw point clouds must be independently checked before reproduction or reuse.

First reproduce the published method only to the extent that the necessary code, data, revisions and rights are verifiably accessible. Otherwise document the exact blocker and reproduce the benchmark protocol with an independently implemented, clearly named baseline. Do not imply the paper has been reproduced from a citation alone.

## 4. Evidence and artifact contract

The JSON schema records four distinct layers.

### 4.1 Original source artifacts

Every drawing, table, scan, photograph or survey is identified by:
- stable artifact ID, source type, locator, revision/date, MIME type and byte length;
- SHA-256 digest over the exact preserved bytes;
- acquisition method/time and whether the original was preserved;
- rights status, permitted operations and restrictions reference.

Rights states are explicit: license confirmed, permission granted, synthetic-owned, restricted, denied or unknown. Unknown rights do not authorize transformation, training, redistribution or publication of source content. The operation PUBLISH_METADATA is not the same as publishing the underlying drawing.

### 4.2 Views and geometric calibration

Each extracted view identifies its source artifact, page, view type, pixel dimensions, rotation/skew, coordinate-frame ID and calibration state.

Calibration binds pixel anchors to physical coordinates, units and uncertainty. It distinguishes uncalibrated, scale-only, affine, projective, engineering-datum-bound and synthetic-known-answer cases. A visually plausible scale bar is not treated as a confirmed dimension unless its origin and method are recorded.

### 4.3 Constraints, conflicts and unresolved regions

Every dimension, station, line/point or annotation carries its own source locator, view, units, interpretation status and uncertainty. The status distinguishes direct transcription, geometrically calibrated, inferred, ambiguous, conflicted and synthetic.

Cross-view conflicts remain first-class records. The system does not choose the newest, clearest or most convenient line without an authority or measured basis. Missing/occluded bow and stern regions, annotations-over-lines, damaged scans and unsupported station geometry remain explicit.

Uncertainty may be NOT_QUANTIFIED; it must not be replaced with zero. A numeric confidence score is not a substitute for calibrated dimensional uncertainty.

### 4.4 Candidate reconstruction

A candidate binds:
- exact source artifact IDs and bytes;
- method/model version and configuration digest;
- output artifact digest, representation, units and coordinate frame;
- uncertainty and geometry-admission status;
- explicit statement that drawing reconstruction alone does not establish as-built condition.

Representations are distinct: point cloud, section curves, parametric surface, NURBS surface, B-rep and triangle mesh. Passing an earlier representation does not imply a later one. Point-cloud fit is not B-rep validity; a B-rep is not hydrostatic, structural or class approval.

## 5. Reconstruction pipeline

1. **Rights and immutable intake:** preserve original bytes, revision, digest and permitted uses before any model sees the material.
2. **Document segmentation:** separate hull curves, grid, text, dimensions, tables, stamps, annotations, folds and scan damage; preserve region-level provenance.
3. **View classification and calibration:** classify body, sheer, half-breadth, lines, GA and offset-table views; estimate scale/rotation/skew and uncertainty from multiple anchors.
4. **Constraint extraction:** recover station identities, buttocks, waterlines, keel, sheer, stem, stern, transom, deck edge, dimensions, offsets and coordinate frames.
5. **Multi-view reconciliation:** solve an overdetermined constraint problem; retain conflict sets and alternate solutions rather than silently averaging incompatible data.
6. **Geometry reconstruction:** generate section curves and a candidate 3D representation from supported constraints. Regions without supporting evidence remain parameterized as unknown or bounded alternatives.
7. **Independent admission:** use the existing G0/G1 geometry-integrity owner; then progressively require surface fairness, self-intersection, CAD/B-rep and hydrostatic checks under their own profiles.
8. **Human review:** expose the source views, extracted constraints, candidate geometry, residuals, uncertainty and unresolved regions together. Human correction becomes a new revision with provenance.
9. **As-built fusion:** only later combine drawing priors with LiDAR/photogrammetry/inspection evidence. Register frames with uncertainty and preserve conflicts; do not overwrite the historical design record.

## 6. Benchmark design

### Data partitions

At minimum, report separate results for:
- synthetic in-domain;
- synthetic out-of-domain (unseen hull families, view combinations and drawing styles);
- clean real archival material with paired reference geometry;
- degraded real archival material (skew, crop, fold, annotation clutter, scan noise);
- partial/as-built survey fusion cases.

Split by original hull lineage, drawing source and vessel family—not by image crop, page, view or point-cloud fragment. No views or revisions from the same hull lineage may leak across training and holdout partitions. Freeze the test split before tuning. Where no paired real 3D truth exists, report reviewer agreement and uncertainty/coverage; do not manufacture a geometric accuracy score.

### Baselines

Compare:
1. raster-to-point-cloud/template-deformation baseline (Raster2Hull, if rights/access permit);
2. section/offset-table constrained reconstruction;
3. deterministic parametric inverse geometry fit;
4. multi-view point-cloud reconstruction;
5. learned image-to-geometry candidate, evaluated against the same held-out lineage split.

An image-only model must not be compared against an offset-table method that has additional privileged inputs without disclosing the information advantage. Publish input modality, preprocessing, compute, model weights/license and exclusions.

### Metrics

Measure separately:
- per-view pixel/curve alignment and line detection precision/recall;
- station, buttock and waterline offset errors with units and uncertainty;
- 3D Chamfer distance and F-score at declared sampling resolution;
- normal consistency and surface/section curvature residuals;
- bow/stern section error and transom/deck-edge reconstruction;
- displacement volume, waterplane, LCB/VCB and selected hydrostatic-coefficient error where geometry supports those calculations;
- mesh/B-rep validity, self-intersection, repair count and unsupported-face fraction;
- uncertainty calibration/coverage and abstention quality;
- conflict detection and unsupported-region detection;
- human review time and number/severity of manual edits;
- source lineage and rights completeness.

Do not combine these into a single weighted score. Report per-geometry class and view availability. Geometric coverage and uncertainty are as important as pointwise fit.

## 7. Falsification and adversarial cases

The frozen test corpus must include:
- missing or ambiguous scale;
- inconsistent scales across views;
- incorrect or mirrored view labels;
- contradictory station numbers and offset tables;
- skewed, rotated, cropped, folded and damaged scans;
- obscured bow/stern or missing body plan;
- mixed units/datums and left/right-handed frame confusion;
- annotation text mistaken for hull lines;
- multiple historical revisions in one drawing bundle;
- rights absent, restricted or uncertain;
- unseen vessel types and unusual stern/appendage forms;
- multiple equally plausible reconstructions;
- plausible-looking but wrong hull that passes image overlap yet fails hydrostatic known answers;
- point-cloud candidate falsely promoted to a watertight B-rep or as-built model.

For each negative control, expected disposition is INCOMPLETE, REJECTED or REVIEW_REQUIRED—never silent automatic repair.

## 8. First implementation slice

This branch adds:
- the versioned JSON Schema for source rights, views, calibrations, constraints, conflicts, unresolved regions, candidate lineage and claim ceilings;
- a synthetic fixture with two orthogonal views, exact byte digests, one explicit conflicting station observation and an occluded bow region;
- separate synthetic source, candidate point-cloud and pipeline-configuration bytes so each digest can be recomputed.

The fixture is intentionally INCOMPLETE, NOT_REVIEWED, has partial uncertainty, and does not claim geometry admission or as-built identity. It is a contract test fixture, not a Raster2Hull reproduction or a real-vessel reconstruction.

## 9. Ownership

- **Symthaea:** document interpretation, view calibration, constraint extraction, reconstruction candidates and uncertainty.
- **Symtropy:** downstream as-built inspection, refit/repair/fabrication feasibility and process qualification.
- **Sol-Atlas:** survey, site, spatial and metocean context where relevant.
- **Mycelix:** source rights/provenance, immutable identity and authority/evidence hand-offs.
- **Luminous:** end-to-end workflow, deployment and engineering lifecycle.

## 10. Claim ceiling

A structurally valid evidence record proves only that declared fields and source references satisfy the selected data contract. A synthetic test proves no real-world accuracy. A point cloud is not a CAD surface; a CAD surface is not a watertight displacement volume; a displacement mesh is not a full external hull. Drawing-derived geometry does not establish as-built condition, structural integrity, stability, class compliance, seaworthiness, DFM, operational suitability or approval. Those claims require separate, independently qualified evidence for the exact vessel and revision.
