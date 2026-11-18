# CROP-INTEL-001 — Benchmark and qualification protocol

**Version:** 0.1 proposal  
**Status:** protocol design; no benchmark has been run.

## 1. Benchmark principle

No single "crop circle score" is appropriate. A method can locate a formation accurately while decoding nothing; detect a geometric regularity while misclassifying its origin; or detect a plant difference without identifying its cause. Evaluate each task separately and publish the full error profile.

All candidate systems receive the same frozen assets, metadata, allowed transformations and compute budget. Split by formation event (not image), with photographer/source/year and geographic leakage audits. Near-duplicate aerial images of one formation must never cross train/test boundaries.

## 2. Evaluation tracks

### Track A — corpus integrity and provenance

Measure:
- fraction of records with source URL, original report identifier, capture/discovery date, crop type and location evidence;
- duplicate precision/recall on adjudicated pairs;
- fraction of assertions preserved without lossy merging;
- fraction of images with explicit rights and attribution state;
- fraction of uncertain or conflicting fields preserved as unknown/conflicting rather than silently filled.

**Target:** 100% of benchmark rows retain immutable source-assertion IDs, provenance, rights state and an uncertainty status. No unattributed pixels enter the benchmark. Do not optimize apparent corpus size at the expense of false merges.

### Track B — geometry extraction and reconstruction

Ground truth should be produced independently by at least two human annotators, with adjudication and measured uncertainty. For surveyed fields, use independent ground-control/check points and metre units.

Metrics:
- primitive-level precision, recall and F1 for circles/arcs/lines/polygons/regions;
- intersection-over-union for binary standing/flattened masks;
- symmetric Chamfer and Hausdorff distances for reconstructed boundaries;
- centre/radius/angle/length error normalized by formation diameter;
- image-to-field registration error at independent checkpoints;
- inter-annotator agreement and reconstruction uncertainty;
- failure/abstention rate under low resolution, perspective, shadows, crop growth and incomplete imagery.

**Initial targets (to be calibrated on the pilot):**
- F1 ≥ 0.95 on synthetic, surveyed, high-quality geometry controls;
- median boundary error ≤ 1% of formation diameter on held-out surveyed examples;
- no registration accepted solely on optimizer residual: it must pass independent spatial checkpoints;
- report real-world and synthetic results separately.

Baselines:
1. manual vector tracing by trained annotators;
2. conventional image processing / Hough and contour pipelines;
3. OpenCV feature matching and RANSAC registration where appropriate;
4. external crop-circle-atlas geolocation workflow as an engineering comparator, not as a ground-truth oracle;
5. proposed multi-view/provenance-aware Symthaea pipeline.

### Track C — geometric information and encoding

Separate these sub-tasks:
1. **Descriptor recovery:** estimate symmetry, radial spacing, graph structure, motif recurrence, spectrum and multiscale complexity.
2. **Code detection:** determine whether a predeclared encoding family is supported above matched nulls.
3. **Payload recovery:** recover an exact held-out payload under a fixed decoder, if one exists.

Metrics:
- geometric descriptor error and confidence interval coverage;
- MDL gain over matched null models;
- decoder exact-bit/symbol accuracy;
- exact held-out payload recovery;
- sensitivity to rotation, reflection, scale, perspective, crop-mask threshold, occlusion and alternate reconstruction;
- family-wise false-positive rate across all tested decoders;
- false-discovery rate when ranking candidate messages;
- calibration / selective risk when the correct output is "no supported decoding".

**Hard gate for an S2 code-like claim:** a preregistered decoder must beat the maximum-statistic null distribution for the entire search family at family-wise α ≤ 0.01, remain stable under predeclared perturbations, and replicate from an independent reconstruction.

**Hard gate for an S3 semantic-payload claim:** in addition to S2, the method must recover a held-out or independently specified payload exactly (or meet a declared error-correcting-code criterion), and an independent analyst must reproduce the result without being told the intended message.

A post-hoc ratio or sentence is a hypothesis only. Do not promote it based on visual plausibility.

#### Null and control families

- randomized occupancy patterns preserving area fraction, perimeter, component count and spatial resolution;
- row/column-margin-preserving 2x2 switch nulls (preserve each row/column occupancy and total density, but not perimeter, connected components or higher-order texture; use only as one control family);
- shuffled angular/radial arrangements preserving the same primitive inventory;
- geometric designs from human artists and documented circle-makers;
- algorithmic ornaments: L-systems, Voronoi, spirals, regular tilings, fractals and procedurally generated designs;
- natural lodging / wind-damage images and non-formation crop patterns;
- images of known synthetic bitmaps encoded into field-like layouts;
- human-made formations with a known author or construction plan where such provenance is genuinely available;
- matched random codebooks and alternative rotations, reflections, segmentation thresholds and row/column orientations.

A null must preserve the nuisance structure relevant to the question. Uniform random pixels alone are too easy and are not a sufficient null. The initial Rust 2x2-switch generator is deterministic and preserves row/column margins, but its finite-step output is **not yet claimed to be uniform, well-mixed, or exchangeable**. Validate reachability/mixing and compare multiple chain lengths before using it for inferential p-values; supplement it with nulls preserving perimeter, connected components, and other relevant nuisance structure.

#### Decoder tournament rules

- Register code families, symbol maps, reading directions, orientation handling, segmentation rules, allowed error correction and payload interpretation before challenge labels are revealed.
- Record every tested variant and its result.
- Use a maximum-statistic or equivalent family-wise correction across all tried decoders.
- Include an explicit "no decodable message" option.
- Keep authorship/origin labels hidden from decoders.
- Do not count an analyst-designed codebook as evidence of a pre-existing code.
- Treat Chilbolton and Crabwood as contested challenge cases. They are not positive labels merely because an interpretation has been published.

### Track D — crop-flow orientation-field descriptors

Use measurements of stem/crop-flow directions at known positions; do not infer stem direction from an uncalibrated overhead outline.

Synthetic controls should include:
- uniform flow, outward/inward radial flow, clockwise/counter-clockwise tangential flow, opposing sectors and smoothly varying fields;
- multimodal directions with a zero/undefined circular mean;
- missing samples, clustered sampling, position duplicates, measurement weights and angle-wrap perturbations;
- field-coordinate rotation/reflection and known angular measurement noise.

Measure:
- circular mean direction error using wrapped angular difference;
- resultant-length and direction-entropy error;
- nearest-neighbour directional-alignment error;
- centre-relative radial and tangential alignment error, with explicit centre-location uncertainty;
- bend-magnitude mean/variance error;
- robustness to missingness, uneven sampling, spatial clustering and quality weights;
- abstention when there is no stable global direction.

The first Rust implementation supplies descriptive summaries only. Its optional centre-relative metrics test radial/inward/outward and tangential/clockwise/counter-clockwise alignment against a supplied calibrated centre. It is not an interpolated vector-field estimator, does not estimate vorticity/divergence, and does not attribute a flow field to any mechanism.

**Synthetic target:** exact recovery within numerical tolerance for noiseless analytic fields; for noisy cases, publish error/coverage curves against the declared noise level. Do not choose a universal real-world degree threshold before instrument and sampling uncertainty are characterized.

### Track E — open-world anomaly detection and scientific reasoning

Metrics:
- AUROC and AUPRC on carefully adjudicated synthetic and real controls;
- Brier score, log score, expected calibration error and reliability diagrams;
- selective risk versus coverage, including abstention quality;
- false-confident anomaly rate on ordinary, ambiguous and out-of-domain cases;
- proper-score improvement over simple baselines;
- robustness under missing metadata, duplicated reports, photographer/source bias and image-quality degradation;
- hypothesis diversity without unsupported proliferation;
- counterfactual prediction accuracy on withheld observations;
- expected information gain and cost of the recommended next experiment.

**Target:** no forced classification on incomplete or out-of-domain evidence; confidence must decrease under provenance degradation. Report unknown and ambiguous cases in the denominator.

Reuse the existing research/falsification-power-reference where its declared assumptions fit. Do not use its known-variance Gaussian power result as a substitute for hierarchical, spatially correlated, sequential or blinded field-study power analysis.

### Track F — physical mechanism discrimination

Compare explicit predictions from:
- human mechanical flattening;
- wind/rain/ordinary crop lodging;
- terrain- and weather-dependent mechanisms;
- any additional mechanism with a well-specified governing model.

Metrics:
- predictive log score on held-out spatial observations;
- residual spatial correlation and boundary-profile error;
- prediction interval coverage;
- robustness to uncertain weather, crop stage and soil inputs;
- energy/momentum/thermal budget closure;
- parameter identifiability and sensitivity;
- ability to specify an observation that would discriminate between two live mechanisms.

A visually similar simulated output is not a mechanism validation. A proposed mechanism must survive conservation and feasibility checks, predict more than the observed outline, and be vulnerable to a predeclared falsification test.

### Track G — plant response and biological replication

Use a randomized, blinded, multi-condition field protocol. The basic sampling design should include:
- within-formation positions sampled on a fixed spatial grid;
- boundary and multiple outside-distance controls;
- matched upright plants in the same field;
- mechanically flattened controls using declared methods;
- naturally lodged controls;
- field, cultivar, phenological stage, moisture, soil and weather covariates.

Record per plant: specimen ID, coordinates relative to the formation, crop species/cultivar, growth stage, stem/node measurements, bend and break state, sample time, collector, handling, preservation and complete custody transfers.

Primary endpoints must be declared in advance. Secondary endpoints include histology, seed viability and germination. Use hierarchical models for plant-within-location-within-field clustering; do not treat each stem as an independent formation-level replicate. Calculate power from the smallest effect of interest, expected intraclass correlation and a prospective sampling plan.

No field result is qualified by software-only tests. Independent laboratory replication is mandatory before generalizing across formations.

## 3. Baselines and industry comparison

The initial competitive landscape is fragmented; no recognized standardized crop-formation leaderboard was found in the public research reviewed for this programme.

Relevant baseline categories:
- **GIS / remote sensing:** QGIS/ArcGIS-style georeferencing, coordinate transforms, spatial joins, terrain/weather overlays and field context.
- **Computer vision / photogrammetry:** manual vectorization, contour and Hough methods, SIFT/ORB-style feature extraction, RANSAC registration and multi-view reconstruction.
- **Pattern recognition:** symmetry descriptors, graph matching, shape grammars, signal processing, fractal estimators and compression/MDL methods.
- **Scientific inference:** preregistration, blinded analysis, hierarchical statistics, negative controls, proper scoring rules and independent replication.
- **Evidence engineering:** immutable source references, rights management, sample custody and reproducible transformation receipts.

The public MYTbrain crop-circle-atlas project is a useful existing engineering baseline for catalogue reconciliation and geolocation. Its published readme distinguishes its synthetic mathematics benchmark from real-world effectiveness and describes a convenience-sample pipeline QA figure. We should compare feature-by-feature and avoid presenting its own internal figures—or ours—as an independent external validation.

The Symthaea differentiator should not be another image classifier or gallery. It should be the combination of:
- provenance-preserving evidence assertions;
- uncertainty-aware parametric geometry;
- preregistered, multiplicity-corrected information tests;
- open-world causal hypotheses;
- experiment selection by expected information gain and statistical power;
- independent replay and claim-ceiling enforcement.

## 4. Release gates

| Gate | Required evidence | Permitted claim |
|---|---|---|
| G0 | Protocol, rights rules, versioned schemas | Research plan exists |
| G1 | Synthetic controls and known-answer tests pass | Implementation behaves on declared synthetic inputs |
| G2 | Independent geometry labels and surveyed checkpoints | Geometry reconstruction qualified for scoped data |
| G3 | Blind decoder benchmark and family-wise null gate | S2 code-like structure for a specific formation |
| G4 | Exact held-out payload + independent decoder replication | S3 payload for a specific formation |
| G5 | Preregistered, controlled biological replication | Scoped plant effect reproduced |
| G6 | Mechanism-specific predictions and independent physical replication | Scoped mechanism supported; origin claim still separate |

Every report records input hashes, source versions, tool commit, parameters, seeds, excluded observations, all tested hypotheses/decoders, and independent reviewer status. Skipped or unrun tests are not passes.

## 5. First benchmark sprint

1. Build synthetic field-like raster/vector fixtures with known geometry and known/absent payloads.
2. Create matched null sets; vary resolution, perspective, occlusion, crop mask and orientation.
3. Implement a simple, transparent geometry baseline before an ML/LLM approach.
4. Freeze one decoder tournament with a finite codebook and exact recovery target.
5. Blindly score the pipeline on the controls.
6. Publish all false positives and abstentions before analysing real contested cases.

**Stop condition:** if the detector calls messages in the matched null set, or if decoding depends on post-hoc orientation/codebook changes, the S2/S3 claim is rejected and the protocol is repaired before real-case interpretation.
