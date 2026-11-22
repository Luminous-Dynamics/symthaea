# Mycelial Reconstruction Scorecard

**Date:** 2026-11-22  
**Status:** First annotated graph scorecard implemented; exact-head Rust/Clippy validation pending.  
**Code:** `crates/domains/symthaea-canvas/src/mycelial_reconstruction_metrics.rs`

## 1. Why graph construction needs its own benchmark

A downstream network can have attractive cycle rank, directness or resilience metrics while the underlying reconstruction contains missed branch points, incorrect endpoints or false anastomoses. This is an upstream error: every path, transport and resilience result inherits the graph-construction mistakes.

The research literature now has directly relevant comparators:

- Sten et al., *Three-dimensional graph reconstruction of filamentous structures from z-stack images*, Ecological Informatics 90, 103317 (December 2025), DOI https://doi.org/10.1016/j.ecoinf.2025.103317. The paper reports a method for reconstructing 3D filament graphs from weak-contrast z-stacks and an overlap-detection F1 of approximately 0.91–0.92 on annotated real fungal images. This is a useful target-setting comparator, not yet a directly matched baseline for Symthaea.
- Aguilera López et al., *Distinct mycelial network architectures during early development of Trichoderma atroviride and Neurospora crassa*, Fungal Genetics and Biology 186, 104117 (September 2026), DOI https://doi.org/10.1016/j.fgb.2026.104117. Their graph descriptors distinguish more hierarchical and more decentralised organisations, associating them with different spatial-expansion and robustness characteristics. This argues against one universal best topology.
- Oyarte Galvez et al., *A travelling-wave strategy for plant–fungal trade*, Nature 639, 172–180 (2025), DOI https://doi.org/10.1038/s41586-025-08614-x, links time-resolved graph construction to actual transport observations.

## 2. Implementation contract

### Annotated graph object

`AnnotatedMycelialSkeleton` records:

- schema version and frame ID;
- immutable source image/volume digest;
- 2D or 3D dimensionality;
- coordinate/length unit (pixels, micrometres, millimetres or organism-local model units);
- producer ID and version;
- junctions with stable IDs, coordinates and classes;
- hypha segments with stable IDs, endpoint junction IDs and length.

Junction classes include:
- origin;
- branch;
- tip;
- anastomosis;
- projected crossing;
- uncertain.

The validator rejects invalid digests, duplicate IDs, missing endpoints, self-segments, unbounded/non-finite positions, invalid lengths and non-zero depth for a 2D annotation graph.

### Explicit identity matching

The score function takes a named matching-policy ID and a one-to-one mapping from predicted junction IDs to ground-truth IDs.

It deliberately does **not** hide a nearest-neighbour or assignment method inside the metric. A caller can supply annotation-ID matching, Hungarian assignment, an image-derived tracking correspondence, or another versioned approach. Every supplied match must satisfy a configured coordinate-distance tolerance. The matching policy and complete mapping are retained in the receipt.

This is important because correspondence errors can dominate apparent segmentation quality. Matching method, tolerance and split policy must be identical when comparing approaches.

### Separate metrics

| Layer | Metrics |
|---|---|
| Junction detection | TP/FP/FN, precision, recall, F1 |
| Junction classes | per-class TP/FP/FN and F1; macro-F1 |
| Fusion safety | false anastomosis junction count |
| Hypha topology | edge TP/FP/FN, precision, recall, F1 |
| Position | mean/median/P95/maximum localization error |
| Length | matched-segment MAE and mean absolute relative error |
| Graph | junctions, segments, components, cycle rank |
| Integrity | frame/image digest, units, graph digests, matching policy digest |

An incorrect junction type does not erase successful junction detection: detection and class scores are separately reported. A false fusion can therefore be visible both as a class false positive and as an extra edge / topological cycle if it changes the graph.

Hypha segments are matched after endpoint junction mapping using undirected endpoint pairs and multiplicity. Parallel hypha segments remain distinct; the evaluator does not collapse them into a simple graph.

## 3. Receipt and reproducibility

The graph canonical digest binds sorted junction identity, class and coordinates, sorted segment identity/endpoints/lengths, frame/source identity, producer version and units.

The score receipt binds:
- ground-truth and prediction graph digests;
- the named matching policy and full one-to-one identity map;
- independent count metrics and per-class scores;
- topology summaries and deltas;
- coordinate and length error summaries;
- matching tolerance and relative-error epsilon.

The evaluator validates one-to-one matching, score count invariants, class-count totals, graph count totals, topology deltas and error-summary consistency before returning a receipt.

## 4. What is not implemented

- Image segmentation or z-stack reconstruction itself.
- Automatic 2D/3D junction matching; the matching algorithm is an explicit input.
- Annotation tooling or inter-annotator agreement analysis.
- Physical coordinate calibration from microscope metadata.
- Confidence intervals over independent biological replicates.
- Dataset ingestion or redistribution of published source images.
- Transport/flow or conductance measurement.

This scorecard tests graph annotations after a matching step. It is not yet an end-to-end image-to-graph benchmark and no microscopy data have been run through it.

## 5. Proposed targets (not achieved)

These are aspirational research targets and must be measured on a frozen, matched held-out dataset.

| Metric | Proposed target |
|---|---:|
| Junction detection F1 | ≥0.95 |
| Hypha edge F1 | ≥0.95 |
| Macro-F1 across junction classes | ≥0.95 |
| False fusion rate on crossing-only negatives | ≤0.5% |
| False accepted connections on deterministic crossing conformance controls | 0 |
| Graph topology exact agreement on synthetic fixtures | 100% |
| Coordinate localization P95 | ≤1 voxel-equivalent or predeclared physical tolerance |
| Segment length mean absolute relative error | ≤5% on calibrated annotated data |

The 0.95 targets are not comparable to the published 0.91–0.92 overlap-detection F1 until the same dataset, annotation policy, task definition, matching policy and train/test split are reproduced. A fair first step is reimplementation of the published comparator on the same licensed data, then an independent held-out test.

## 6. Benchmark design

- Separate 2D and 3D tracks.
- Separate object/segment segmentation, junction detection, junction type classification, overlap-vs-fusion discrimination and graph topology.
- Freeze annotation rules and disclose inter-annotator disagreement.
- Split by organism/replicate, not by adjacent frames, to avoid temporal leakage.
- Test crossings, overlaps, occlusion, weak contrast, branching, tip contacts, true anastomoses and uncertain junctions separately.
- Report per-class confusion, not just a single aggregate F1.
- Keep uncertain and occluded evidence distinct from negative examples.
- Publish raw edge precision/recall and false-fusion counts beside graph-level metrics.
- Do not let candidate contacts silently become accepted graph edges.
- Report confidence intervals by replicate and stratify by species/morphology regime.

## 7. Connection to network growth and resilience

The scorecard qualifies the upstream graph. It should feed into, but remain separate from:

- [#13145](https://github.com/Luminous-Dynamics/symthaea/issues/13145): AMF geometric efficiency, cost and empirical transport comparison;
- [#13144](https://github.com/Luminous-Dynamics/symthaea/issues/13144): failure/repair functional resilience;
- [#13157](https://github.com/Luminous-Dynamics/symthaea/issues/13157): time-indexed directness/coverage/cost Pareto growth;
- [#13180](https://github.com/Luminous-Dynamics/symthaea/issues/13180): temporal cycle recurrence and zigzag persistence.

A reconstruction model can be excellent at recovering topology but poor at predicting transport. A graph can be resilient under node deletion but poorly suited to exploration. Keep each layer separately measured.

## 8. Research direction

The intended shift is from "generate a plausible mycelial graph" to **produce an evidence-backed, uncertainty-aware reconstruction whose downstream claims can be traced to annotated source observations**.

That makes it possible to investigate whether Symthaea can outperform current 3D reconstruction methods without trading away crossing/fusion correctness, and whether a network architecture's measured benefits persist after reconstruction error is accounted for.
