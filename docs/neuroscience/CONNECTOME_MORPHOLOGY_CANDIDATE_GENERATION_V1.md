# Morphology-conditioned candidate generation benchmark V1

**Status:** research contract; no morphology generator, biological evaluation, or candidate-recall improvement is claimed.
**Tracking:** [SCI-CONNECTOMICS-007 / #9241](https://github.com/Luminous-Dynamics/symthaea/issues/9241).
**Independent evaluator:** [SCI-CONNECTOMICS-006 / #9204](https://github.com/Luminous-Dynamics/symthaea/issues/9204).

## 1. Goal and separation of concerns

Build a candidate generator that discovers eligible proofreading errors absent from the current proposal pool. Keep three functions separate:

1. **Discovery:** emits possible error locations and correction hypotheses from image, geometry, morphology, and topology evidence.
2. **Ranking:** orders frozen proposals for limited expert time.
3. **Evaluation:** independently searches probability-sampled spatial/neuron primary sampling units (PSUs) and estimates errors covered and missed.

The #9204 audit is the evaluator. Its seed, sampled locations, verified error labels, candidate-to-error mappings, and impact weights must never enter discovery, ranking, or their tuning loop.

## 2. Representation boundary: benchmark data are not full morphology

ConnectomeBench2 V2 contains operation-level examples with rendered EM views and rendered geometry. Its card describes geometry payloads as three 2D views with seven channels at 224×224, and EM views as image slices. This is useful for controlled local classification and evidence ablations, but it is not a complete 3D neuron mesh, skeleton, or NEURD decomposition graph.

Therefore:
- Use ConnectomeBench2 V2 for its declared local proofreading tasks and task-specific modality comparisons.
- For full morphology and NEURD-compatible discovery, resolve the sample's provenance to the original segmentation / volume / mesh / skeleton / synapse data.
- Pin each original dataset, volume, segmentation revision, annotation source, and licensing basis independently.
- Never present a rendered local geometry crop as if it supplied whole-neuron morphology or complete downstream topology.

NEURD is relevant because it decomposes neuron meshes into annotated graph representations and uses node-, subgraph-, and graph-level evidence in proofreading. Its reported 99% and 95% human agreement concerns high-confidence subsets of two merge-error rules, not all proofreading cases. The same paper reports materially different synapse recall by compartment (approximately 99% for dendrites versus approximately 60% for axons in the MICrONS single-soma setting); aggressive pruning therefore needs explicit axonal false-removal controls.

Primary reference: [NEURD, Nature (9 April 2025)](https://www.nature.com/articles/s41586-025-08660-5).

## 3. Candidate and evidence contract

Each candidate must carry:

- stable candidate ID and a typed hypothesis (split, merge, reassign, or morphology-alert-only);
- source volume, segmentation revision, root/neuron identifiers, coordinate frame, and transform provenance;
- exact mesh, skeleton, synapse-table, and graph revision/digests when available;
- feature-extractor binary/version/configuration digest;
- raw feature values, units, scale, missingness, quality flags, and uncertainty; no silent imputation;
- detector identity, rule/model version, threshold, and per-detector contribution;
- affected spatial region, proposed cut/join locus, and deterministic pointer to the relevant morphology subgraph;
- evidence digests and a reproducible extraction receipt;
- an explicit abstention / insufficient-context state.

Candidate IDs must be stable for the same frozen source revision and canonical evidence. Preserve detector provenance when proposals are deduplicated. The generator and ranker receive no evaluator-only annotations, review outcomes, sample seed, sampled-PSU list, or impact oracle.

## 4. Detector ladder

### A. Segmentation and topology

Detect disconnected components, implausible attachments, inconsistent root identity, multi-soma components, orphan fragments, stitching seams, and suspicious component-boundary transitions.

### B. Mesh and skeleton geometry

Measure local caliber distributions, abrupt width ratios, branch angles, geodesic distance, tortuosity, branch order, local curvature, soma-root continuity, and morphology discontinuities across a proposed join. Record spatial context and measurement quality. A feature's abnormality is not itself proof of an error.

### C. Compartment and synapse context

Condition on axon/dendrite/soma/unknown compartment with uncertainty. Evaluate pre/post synapse density per cable length, synapse-to-branchpoint distance, compartment-conditioned distributions, and synapse attachment consistency. Missing or uncertain synapse annotations remain explicit.

### D. Connectome graph context

Assess whether a morphology-supported correction changes predeclared directed edges, paths, and circuit queries. Degree or motif counts alone must not trigger a correction. Use spatial-opportunity and compartment-conditioned nulls from #9156 before making claims about motif significance.

### E. High-recall union and later ranking

Combine specialist outputs by union before ranking. Keep feature-source provenance and disagreement signals. Tune the candidate generator for recall under a declared candidate-burden / compute budget; tune the ranker separately for useful corrections per expert minute.

## 5. Preregistered error-family vocabulary

The initial vocabulary to review with domain experts is:

- multi-soma merge;
- axon–axon merge;
- axon–dendrite merge;
- orphan neurite attachment;
- glia / vessel contamination;
- false split across a continuous branch;
- compartment misassignment;
- synapse-to-segment attachment error;
- cross-slab / stitching seam error;
- incomplete / truncated morphology;
- unresolved / other.

Do not collapse rare families into other after evaluation starts. A family absent from the probability sample must remain visible with an undefined family recall, not disappear from the report.

## 6. Evaluation matrix

Freeze the search frame, source revisions, PSU definition, generator binaries/configurations, proposal pool, feature-extraction receipts, family vocabulary, and sample commitment before outcome collection.

Compare on the same declared eligible frame:

1. current candidate generator;
2. EM-only specialist;
3. rendered-geometry-only specialist;
4. morphology-only full-mesh/skeleton rules;
5. NEURD-compatible rules / decomposition features;
6. fused specialist union;
7. leave-one-feature-family-out ablations.

Use volume / spatial block / neuron / source-operation grouping according to the true dependence structure. Do not randomly split image crops from the same source operation across development and evaluation.

Primary metrics:
- design-weighted ordinary candidate-generation recall;
- design-weighted impact-weighted candidate-generation recall;
- design-adjusted lower confidence bounds by error family, region, species, and impact tier;
- estimated missed error count and missed impact mass;
- false proposals per neuron / PSU and review burden at matched recall;
- feature extraction compute, memory, storage, data movement, and latency.

Required adverse controls:
- incomplete or decimated mesh;
- truncated axon and orphan fragment;
- poor / fragmented skeleton;
- missing or shifted synapse annotations;
- coordinate-frame transform error;
- root-ID changes and stale segment IDs;
- glia/vessel contamination;
- regional, species, and acquisition-domain shifts;
- false morphology alarms caused by normal caliber transitions or branching variation.

The independent probability audit must include PSUs in which the generator emits zero candidates. Candidate recall is bounded to the declared searchable frame and protocol; it is not automatically whole-connectome recall.

## 7. Promotion gates (proposals, not observed results)

- High-impact candidate-generation recall point estimate ≥95% and design-adjusted lower 95% bound ≥90% on the declared frame.
- Every critical family has preregistered minimum support and a separate recall bound; no pooled average can hide a missing family.
- No unacceptable increase in false accepted corrections or morphology-driven axonal deletion.
- At matched coverage, reduce expert burden or improve impact-weighted verified yield.
- Count feature extraction, data movement, random audit, adjudication, and reviewer costs in the all-in comparison.

No candidate generator is promoted based on the 99%/95% NEURD high-confidence subset figures alone. No biological win is claimed until this protocol is run on independently adjudicated data.
