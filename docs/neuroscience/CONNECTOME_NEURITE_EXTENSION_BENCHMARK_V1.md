# Incomplete neurite extension and false-split continuation benchmark V1

**Status:** research protocol; no continuation model or biological result is implemented.
**Tracking:** [SCI-CONNECTOMICS-010 / #9249](https://github.com/Luminous-Dynamics/symthaea/issues/9249).
**Related discovery work:** [#9241 morphology-conditioned candidate generation](https://github.com/Luminous-Dynamics/symthaea/issues/9241).

## 1. Why extension must be its own benchmark

Merge-error cleaning and incomplete-process recovery are not symmetric operations. A system can remove incorrect neurite material while still leaving true axons or dendrites disconnected from their parent neuron. NEURD's current paper explicitly says its implementation is focused on false-merge removal and does not fix incomplete neural processes. That creates a concrete opportunity, but not an empty field: RoboEM learns 3D neurite continuation from EM; a mesh-tip extension pipeline exists; and Autoproof reports learned fragment attachment on the Drosophila male CNS. These must be treated as comparators.

References:
- NEURD: https://www.nature.com/articles/s41586-025-08660-5
- RoboEM: https://www.nature.com/articles/s41592-024-02226-5
- Mesh-tip extension: https://pmc.ncbi.nlm.nih.gov/articles/PMC10634717/
- Autoproof: https://arxiv.org/abs/2509.26585

## 2. Task definition

Input: a versioned neuron reconstruction with a candidate terminal endpoint and surrounding source EM volume, segmentation fragments, mesh/skeleton, synapse annotations, and coordinate transforms.

Output: a ranked set of continuation hypotheses, including existing orphan-fragment attachment, newly traced path, multiple plausible branches, and explicit no-continuation / abstention. A candidate is a hypothesis, never a correction by itself.

Separate at least four cases:
- Internal false split: a true continuous process is segmented into separate fragments inside the imaged volume.
- Truncated at volume boundary: continuation lies outside available imaging and cannot be proven from current data.
- Biological terminal: the neurite genuinely ends.
- Unknown / insufficient image evidence: do not force a label.

These classes cannot be merged into a single endpoint-error label without changing the scientific meaning of the evaluation.

## 3. Multi-hypothesis continuation architecture

1. Endpoint proposal: detect suspicious tips using skeleton/mesh endpoints, branch order, caliber, local curvature, nearby fragments, section boundaries, synapse density, and source-volume boundaries.
2. Candidate continuation generation: retrieve nearby orphan fragments, create plausible paths through EM voxels, and preserve more than one candidate where ambiguity exists.
3. Evidence scoring: combine raw EM evidence, geometric tangent/curvature continuity, caliber, branch topology, compartment probability, and synapse consistency. Keep each evidence stream separately inspectable.
4. Constraint checking: reject loops, impossible self-intersections, incompatible root histories, stale coordinate frames, unsupported jumps, and inconsistent branchpoint changes.
5. Abstention: represent true terminal, boundary-truncated, uncertain, and no-candidate states explicitly. Do not convert missing evidence into negative evidence.
6. Proposal packaging: bind each hypothesis to exact volume/segmentation/skeleton/mesh/synapse revisions, spatial coordinates, feature-extractor digests, and a reversible proposed edit.

Learned tracing may use local EM patches, but long-range continuation must also track path-level uncertainty and drift. Repeated locally plausible steps can accumulate into a globally incorrect path; therefore evaluate complete reconstructed paths and their resulting synapses, not only next-step classification.

## 4. Benchmark arms

- Baseline segmentation with no extension.
- Nearest-fragment and endpoint-distance heuristic.
- Mesh-tip morphology rule pipeline.
- RoboEM-like 3D flight / continuation, where compatible source data and code are available.
- Learned local EM continuation.
- Multi-hypothesis EM + morphology + synapse + graph continuation.
- Expert manual continuation with independent blinded adjudication.

Report compatibility gaps for every arm. Do not score an unavailable method as a failed method.

## 5. Sampling and ground truth

Freeze the eligible endpoint frame independently of any method's proposals. Probability-sample endpoints / neurons / spatial blocks, including cases with no proposal from any arm. Use non-overlapping dependence units and split by neuron, source operation, region, volume and species.

Use two complementary evaluation sets:
- Natural incomplete-process cases with independently adjudicated evidence and explicit uncertainty.
- Blinded semi-synthetic truncation controls where known continuous paths are cut at controlled positions. These provide exact path truth but are not a substitute for natural errors.

Boundary-truncated examples must be labeled as unobservable continuations unless additional volume is acquired. A system must not be rewarded for inventing an out-of-frame continuation.

## 6. Metrics

Primary:
- Probability-sampled endpoint recall and impact-weighted endpoint recall.
- Recovered true cable length and synapses, with precision and recall separately.
- False-join and false-branch creation rates.
- Branchpoint preservation and path continuity.
- Expert minutes per independently verified recovered synapse and per recovered cable length.
- Risk/coverage frontier including abstained endpoints.

Secondary:
- Calibration of path-level probability and expected path length.
- Long-range drift, loop and self-intersection rate.
- Axonal versus dendritic results, thin versus thick caliber, and branch order.
- Cross-volume/species transfer.
- Compute, memory, volume access, feature extraction, audit, and adjudication costs.

## 7. Adversarial controls

- Thin axons, low-contrast regions, anisotropic voxels, folds, tears and missing sections.
- Parallel nearby neurites, crossings, and plausible but incorrect adjacent fragments.
- True biological termini versus endpoints near volume boundaries.
- Genuine branch bifurcations versus accidental branch creation.
- Pathological caliber changes and valid sharp turns.
- Coordinate-frame drift, stale root IDs, and segmentation revision changes.
- Sparse / incorrect synapse annotations and compartment misclassification.
- Long path accumulation, loops, self-intersections and erroneous branch merges.

## 8. Proposed promotion gates (not results)

- Design-adjusted lower 95% bound for critical-family endpoint recall ≥90% within the declared frame.
- False-join rate no worse than the strongest frozen baseline at matched recovered completeness.
- No material loss of verified axonal synapses or valid branchpoints.
- At least 20% lower actual expert time to a preregistered completeness target at matched independent quality and coverage.
- Human acceptance, independent validation, and reversible application remain mandatory.

Do not report a single path-recovery score without false joins, boundary-truncation handling, compartment-stratified synapse precision/recall, and abstention. No automatic correction or biological performance claim is made by this protocol.