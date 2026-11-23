# Mycelial fusion counterfactuals and decision-value review

**Status:** research specification; proposed targets are not achieved results.  
**Scope:** the evidence-bearing graph layer, not image segmentation, fungal physiology, or fusion authorization.

## Executive direction

The next useful step is not another topology descriptor. It is a paired, single-decision counterfactual:

> Hold the acquired observation, reconstruction, all accepted edges, candidate inventory, declared demands, and failure set fixed. Change exactly one candidate fusion from rejected to accepted-for-analysis. Recompute the consequences and preserve both views and their evidence.

This converts the current annotation queue from a heuristic ranking into a review instrument that can state what a disputed label would change. The counterfactual is read-only. It must never modify the accepted graph, infer a biological fusion from proximity, or authorize a fusion.

A second paradigm shift follows: allocate reviewer time by **expected reduction in downstream decision loss**, not confidence or graph importance alone. Keep uncertainty, topology impact, review cost, and expected value of information as separate reported quantities.

## Evidence boundary and graph operations

The current annotated skeleton records junction coordinates and scalar segment lengths. It does not retain a Bezier centerline or an independently measured fusion-bridge geometry. Consequently:

- Tip-to-tip: add one candidate edge between the two declared tip junctions in the analysis-only view.
- Tip-to-side: split exactly one declared target segment at an explicit normalized *arc-length fraction*, create a deterministic split-junction identity, and add the candidate link from the source tip. The two split lengths must sum exactly to the original segment length within the declared numeric representation.
- A source-curve Bezier parameter is not interchangeable with normalized arc length. A future geometry-aware adapter must invert the arc-length map before transferring a curve parameter into this graph contract.
- The candidate bridge length must be supplied as measured or explicitly declared evidence. Do not silently use Euclidean endpoint distance as hyphal material length.
- A split junction's coordinate may be interpolated only as a display/graph anchor when source centerline geometry is unavailable. Mark it as interpolated; do not call it a measured 3D contact point.

The decision object should bind: candidate ID; source observation, calibration and graph digests; candidate-inventory digest; source tip; target tip or segment plus parameter; candidate class probability and calibration-policy ID; conflict-group identity; candidate bridge-length evidence; and the graph/schema version.

The result should contain two independently materialized, read-only views:
1. **Rejected:** the frozen source graph, unchanged.
2. **Accepted for analysis:** the rejected view plus only this candidate's edge (and, for tip-to-side, only the deterministic segment split).

It must also carry a negative-control assertion that the source skeleton digest and accepted-topology digest are identical before and after analysis.

## Counterfactual scorecard

Report the pairwise difference, not just the accepted-view value:

| Quantity | Required result |
|---|---|
| Components / Betti-0 | Exact component count and delta |
| Cycle rank / Betti-1 | Exact cycle rank and delta |
| Cycle witness | Existing source-to-target path if the candidate closes a cycle |
| Segment cost | Original and split segment totals; candidate bridge cost; explicit unit |
| Shortest paths | Before/after distance and delta for every declared pair |
| Root reachability | Before/after only for caller-declared roots and demand sites |
| Edge-disjoint paths | Bounded exact count for declared pairs; report the cap |
| Failure delivery | Same frozen disruption scenarios applied to both views |
| Provenance | Source, candidate, inventory, configuration, and output digests |

A newly created cycle is not itself evidence of improved function. A candidate can increase Betti-1 while leaving every declared demand path unchanged or making some paths longer. A candidate can merge disconnected components without creating a cycle. These cases must be distinguishable.

### Structural category

Classify each candidate against the rejected graph as one of:

- **Component merge:** the endpoints were disconnected; the toggle reduces beta-0 by one and leaves beta-1 unchanged.
- **Cycle closing:** the endpoints were already connected; beta-0 is unchanged and beta-1 increases by one. Retain a witness path in the rejected graph.
- **Duplicate/no-op:** an equivalent edge already exists; do not report a new cycle or benefit.
- **Conflict:** the candidate shares an exclusive contact claim with another candidate. Analyze candidates individually; never assume the conflict group can be accepted wholesale.
- **Invalid/unresolved:** endpoint, parameter, probability provenance, source digest, or geometry does not validate. Fail closed.

For a simple undirected graph, the implementation should check the identity beta-1 = E - V + beta-0 after every view. Segment splitting changes V and E by +1 and +1 respectively before the candidate edge is added, so it does not itself change cycle rank. The candidate edge then produces the single component-merge or cycle-closing transition.

## Beyond a hand-weighted review queue: decision value of information

For a candidate with calibrated posterior outcomes, define review value as the expected reduction in a declared downstream loss after a reviewer resolves the label, minus review cost. The loss must be declared by the experiment: for example, false-fusion penalty, missed-branch penalty, root-to-demand disconnection, or error in a measured transport target. Do not silently collapse these into a universal score.

The report should expose:
- predictive class probabilities and calibration provenance;
- false-fusion and missed-fusion risk;
- exact candidate-specific topology deltas;
- each declared downstream loss under each outcome;
- expected value of information and review cost in their own units;
- conflict dependencies and the marginal value conditional on already selected reviews.

For multiple candidates, independent per-item values can double-count shared consequences. Use conflict constraints and evaluate the joint selected set against the frozen candidate graph. Start with exact enumeration on small fixtures, then compare a budgeted knapsack/greedy policy against exhaustive optimal selection. Publish the optimality gap. Only pursue submodular guarantees after the actual loss function and interaction structure satisfy the required assumptions.

This is the research distinction between "uncertain" and "worth reviewing." A low-confidence tip may be cheap to resolve but irrelevant to the declared task. A moderately uncertain candidate that changes the only root-to-demand path may be much more valuable.

## Independent oracle and adversarial fixture plan

Build a small-graph oracle independently of the production toggle code. Use a simple edge-list representation, disjoint-set component counts, and Floyd-Warshall or an independently implemented Dijkstra for all-pairs shortest paths. Compare production output to the oracle over generated small multigraphs and fixed named cases.

Required cases:

1. Rejected/no-op and exact duplicate connection.
2. Component-merging tip-to-tip candidate.
3. Cycle-closing tip-to-tip candidate with a verifiable witness path.
4. Tip-to-side split preserving total original hypha length.
5. Candidate that creates a cycle but improves no declared demand pair.
6. Candidate that creates a cycle and adds a usable alternate path under a declared edge failure.
7. Candidate that merges components but fails a declared root-to-demand pair because the demand endpoints were not those fused.
8. Two candidates claiming the same tip; only one may be selected in a feasible decision set.
9. Same source tip, two target segments at distinct parameters.
10. Duplicate candidates with different IDs but identical endpoints/contact identity.
11. Candidate probability vector with invalid sum, non-finite value, or stale calibration digest.
12. Mismatched source-observation, calibration, graph, or candidate-inventory digest.
13. Tip-to-side parameter at 0, 1, or outside the declared interior margin.
14. Zero/negative/non-finite bridge length and unit mismatch.
15. Failure set that removes the candidate edge, the original target segment, a split segment, or a declared junction.

For every case, assert that rejected and accepted-for-analysis views share the same immutable source digest. For conflict cases, assert that each candidate is analyzed separately and no infeasible joint acceptance is emitted.

## Proposed benchmark targets

These are research gates, not current results.

| Gate | Initial target |
|---|---:|
| beta-0 / beta-1 delta agreement against independent oracle | 100% on all bounded exhaustive fixtures |
| Shortest-path / reachability delta agreement | 100% on the independent small-graph corpus |
| Candidate / source / calibration / inventory digest validation | 100% |
| Unauthorized source or accepted-graph mutations | 0 |
| Duplicate or mutually exclusive candidate links silently counted | 0 |
| Tip-to-side source-length conservation | absolute error <= 1e-12 in normalized double fixtures; unit-aware tolerance for measured units |
| Counterfactual report recomputation | 100% from retained evidence |
| Review-budget graph-loss reduction | >=20% relative to uncertainty-only at equal reviewer time on held-out replicates |
| False-fusion residual at 10% review budget | >=50% lower than random review; >=20% lower than confidence-only review |
| Decision-value policy optimality gap on bounded fixtures | 0% for exhaustive fixtures; report actual gap for larger runs |

Report counts and denominators for false fusions, missed fusions, projected crossings and unresolved labels. Provide bootstrap confidence intervals clustered by biological replicate, not by image patch or adjacent frame.

## Matched SOTA and industry comparison

### Published reconstruction baseline

Sten et al. (2025), *Three-dimensional graph reconstruction of filamentous structures from z-stack images*, Ecological Informatics 90, 103317, report overlap-detection F1 of 0.91–0.92 on six manually annotated fungal images, with 474 branch points and 83 overlap points. Their pipeline combines per-plane ridge detection, graph merging/matching, shape-from-focus depth estimation and local steerable-filter overlap analysis. The fair first comparison is to reproduce their task, annotation tolerance, sample selection, and splits—not compare our synthetic graph fixtures to their biological-image score.

### Longitudinal morphology and transport

Oyarte Galvez et al. (2025), *A travelling-wave strategy for plant–fungal trade*, Nature 639, 172–180, report tracking more than 500,000 fungal nodes and approximately 100,000 cytoplasmic-flow trajectories, and compare observed networks with minimum-spanning-tree and Delaunay references. This is a higher bar than static topology: the next scientific leap is to test whether geometry/topology predicts measured flow and resource delivery on matched observations. Geometric directness is not transport, conductivity or biological function.

A 2026 comparison of *Trichoderma atroviride* and *Neurospora crassa* reports different emerging network organization—decentralized versus hierarchical. This argues against optimizing every organism toward one "ideal" topology. Report species- and developmental-stage-conditioned Pareto fronts instead.

### Commercial and open tooling

Commercial platforms already offer substantial image-analysis capability. Oxford Instruments' Imaris Filament Tracer supports tracing filamentous structures such as neurons and vessels; Imaris 10 release notes describe faster tracing of larger images and machine-learning-based versatility. ZEISS arivis Pro provides multidimensional AI-assisted segmentation, 3D/4D analysis, large-data handling and batch workflows. napari provides an open multi-dimensional viewer and interactive 3D annotation ecosystem.

Do not claim superiority over these platforms because our graph reports are more auditable. Compare them on the same raw volumes, acquisition metadata, ground-truth annotations, hardware, reviewer workflow and total analyst time. Our differentiated claim should be **reproducible evidence-to-graph assurance and decision-specific review accounting**, integrated with—not in place of—strong segmentation and tracing.

## Dataset and experimental governance

- The Sten et al. code/data package is publicly downloadable at Zenodo record 15780623 (version v2, published 1 July 2025). It contains overlap-test annotations, *R. irregularis* 2D images and Z-stacks, 3D-printed filament prototypes, time-lapse images and the MATLAB reproduction scripts. The Zenodo record grants MIT terms to the software package but separately labels several image/data subsets as Copyright 2024 Oscar Sten or Copyright 2020 Cardini et al. Treat software and data rights as separate: create a per-file rights manifest, do not mirror or redistribute images, and obtain any additional permission required for commercial benchmark use before incorporating those data into a commercial training/evaluation pipeline.
- Zenodo record 23135747 (published 4 October 2026) describes a larger bright-field dataset: 21 annotated Z-stacks, 1,205 annotated slices, 7,657 retained mosaic samples, and a separate 48-timepoint 3D time-lapse at approximately 30-minute intervals over 25 hours, plus one final-timepoint manual segmentation. It is 2.6 TB and explicitly embargoed until **8 October 2027**. The public record does not currently expose those files or establish an unrestricted dataset license. Do not use it for present benchmark execution; track it as a future acquisition candidate and verify license/access after release.
- The first executable external baseline should therefore reproduce Sten et al.'s six-image overlap evaluation with its original annotation protocol and exact manually resolved exceptions. Start with a legally reviewed local copy and publish only derived metrics/receipts unless redistribution rights are clear.
- Split train/calibration/test by organism, isolate, biological replicate, acquisition session and time-series group. Never split adjacent frames across train and test.
- Freeze the prediction and calibration artifacts before running an active-review policy. Hidden labels are used only after selection to measure the actual error reduction.
- Keep 2D projection and 3D z-stack results separate. A 2D crossing is not a 3D fusion.
- Record annotator disagreement, image quality, occlusion, depth uncertainty, reviewer time and abstention outcomes.

## Follow-on paradigm shifts

1. **Graph posterior rather than one guessed graph.** Represent uncertain contacts as a constrained distribution over feasible graph hypotheses. Report posterior intervals for beta-0, beta-1, root reachability and path length. Model contact dependencies; do not multiply marginal probabilities as if candidates were independent.
2. **Evidence-aware topology over time.** The current temporal cycle recurrence tracks exact edge-set recurrence in windows; it is not zigzag persistence. A later version can model additions, removals and unknown intervals with explicit correspondence and censoring, and validate it against small exhaustive temporal graph oracles.
3. **Function-grounded networks.** Add tracer/flow observations and calibrated edge conductance only when measured. Compare predicted delivery, observed flow, construction length, exploration coverage and resilience as a multi-objective surface.
4. **Acquisition-aware active sensing.** Before asking a person to label an ambiguous contact, estimate whether an additional z-plane, higher-resolution crop, alternate stain or timepoint could resolve it more cheaply. Compare the value of another observation against the value of human annotation.
5. **Audit-native interoperability.** Export source-linked graph annotations and counterfactual receipts in a documented, versioned schema that can sit beside OME-TIFF/OME-Zarr image data and be consumed by microscopy workbenches.

## References

- Sten et al. (2025). *Three-dimensional graph reconstruction of filamentous structures from z-stack images*. Ecological Informatics 90, 103317. https://doi.org/10.1016/j.ecoinf.2025.103317. Data/code: https://doi.org/10.5281/zenodo.15780623
- Oyarte Galvez et al. (2025). *A travelling-wave strategy for plant–fungal trade*. Nature 639, 172–180. https://doi.org/10.1038/s41586-025-08614-x. Open text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11882455/
- Aguilera López et al. (2026). *Distinct mycelial network architectures during early development of Trichoderma atroviride and Neurospora crassa*. Fungal Genetics and Biology 186, 104117. https://doi.org/10.1016/j.fgb.2026.104117
- Oxford Instruments. Imaris 10.0 release notes: https://imaris.oxinst.com/support/imaris-release-notes/10-0-0
- ZEISS. arivis Pro: https://www.zeiss.com/microscopy/en/products/software/arivis-pro.html
- napari documentation: https://napari.org/stable/getting_started/quick_start.html

## First source implementation slice — 2026-11-23

Added `crates/domains/symthaea-canvas/src/mycelial_fusion_counterfactual.rs` and exported its public contract from `symthaea-canvas`.

Public entry point:

```rust
evaluate_mycelial_fusion_counterfactuals(
    skeleton,
    candidates,
    declared_demand_pairs,
    frozen_disruption_scenarios,
)
```

The source slice currently provides:
- candidate identity, calibrated fusion probability/policy/evidence digest, explicit measured-versus-declared bridge-length basis and bridge-length unit;
- tip-to-tip and tip-to-side target contracts, with tip-to-side parameter explicitly defined as normalized arc-length fraction (not Bezier parameter);
- two read-only paired views. For a side contact, the same target-segment split is applied to both views, and only the candidate fusion edge differs;
- length-conserving target split with an explicitly interpolated *display anchor* because the current annotated skeleton does not retain source centerlines;
- exact connected-component/cycle-rank deltas, existing cycle witness, declared-demand shortest paths/directness, and the same frozen source-segment/junction disruption sets applied to both views;
- automatic conflict discovery for shared source tips, shared tip endpoints, identical side-contact identities, in addition to caller-supplied conflict groups;
- complete source frame/calibration metadata, candidate target/bridge details, evidence digests and exact disruption sets in the report;
- independent recomputation through `validate_against`, read-only source semantics, and bounded demand-evaluation work;
- an independent test-only Floyd–Warshall oracle for small tip-to-tip and tip-to-side examples.

**Important claim boundary:** the side-contact split is graph-level only. The interpolated anchor is not a measured 3D centerline point, and normalized arc-length fraction is not transferable to a Bezier curve parameter without the missing centerline geometry. The bridge may be measured or explicitly declared; the report preserves which basis applies. No fusion is automatically accepted.

### Current validation status

- The previous exact-head run on `f548a8ee4fec46ce8daef70a29ea9e1c6a8f5add` stopped at `cargo fmt --check`; the formatting output was applied to that source slice.
- The subsequent GitHub API commits have advanced the PR head, but no workflow run or commit status is currently returned for the latest API-created head. The current module has **not** been compiled, unit-tested, Clippy-checked or independently executed locally. The local container has no `rustc`, `cargo` or `rustfmt` executable. Do not treat source tests as executed results.
- Re-run the workflow from GitHub Actions on the current PR head before merging; its current workflow trigger is `pull_request` / `workflow_dispatch`.
