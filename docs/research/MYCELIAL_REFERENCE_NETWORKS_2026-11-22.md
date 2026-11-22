# Mycelial Reference Network Benchmarks

**Date:** 2026-11-22  
**Status:** Observed-network minimum-spanning-forest comparator implemented; exact-head validation pending.

## 1. Reference networks answer different questions

Do not treat “MST” and “Delaunay” as one interchangeable baseline.

### Observed-network minimum spanning forest

This reference is constructed from the **same observed hypha-segment edges** as the source network. It selects a minimum-total-weight spanning tree in every connected component. Segment traversal length is the weight (curve arc length plus geometric weld offsets). Material-like cost is separately reported as selected curve arc length, excluding weld offsets.

This is the closest graph-theoretic analogue of the weighted minimum spanning tree in the AMF travelling-wave analysis.

### Euclidean minimum spanning tree

This is a different object: it creates new straight-line connections between measured junction locations, regardless of whether a hypha segment exists. It is a spatial lower-cost reference, not an observed biological network. It can connect components that were disconnected in the measured graph.

It is not implemented in this source slice.

### Delaunay reference

A Delaunay graph constructs candidate spatial adjacencies from measured junction locations. It depends on whether the source is 2D or 3D, duplicate/coincident points, degeneracy handling and tie-breaking. It is not interchangeable with a graph of observed hyphae.

A robust, explicitly dimensioned Delaunay adapter remains future work. Current renderer geometry is three-dimensional and contains coincident origins; silently applying a planar Delaunay triangulation would merge or perturb those points and would not be an honest baseline.

## 2. Current implementation

Module:

`crates/domains/symthaea-canvas/src/mycelial_reference_networks.rs`

Public builder:

`build_observed_network_minimum_spanning_forest(graph, root_junction_ids)`

Implementation:
- validates the source junction–hypha graph before reference construction;
- sorts source hypha segments by traversal weight, stable segment ID, and endpoint index;
- uses deterministic Kruskal selection with stable union-find tie-breaking;
- preserves the connected-component structure (a disconnected input becomes a spanning forest, not a single invented connected network);
- records selected segment IDs in selection order;
- reports selected curve arc length and traversal cost separately;
- computes global geometric directness and declared-root directness with the same junction set and denominators as the source graph;
- reports disconnected pairs rather than silently assigning them a zero path.

For an input with (V) junctions and (C) connected components, the forest must select exactly:

[
|E_{MSF}| = V-C.
]

The unit is inherited from the source graph and is currently `organism_local_model_unit`. It is not a physical measurement.

## 3. Exact AMF comparator inspected

The public repository `Cocopyth/AMFTravellingWave` includes:
- `Figure_plots/Fig4/Efficiency/NetworkEfficiency.m`
- `Figure_plots/Fig4/Efficiency/NetworkEfficiencyLoop.m`

The published analysis:
1. loads measured edge origin/end labels and hyphal lengths in micrometres;
2. converts image pixel distances using the experiment calibration factor (`conv_px2mum = 1.725`);
3. constructs the observed weighted graph using measured `length_mum`;
4. constructs a minimum-spanning-tree graph from those observed edges and weights;
5. constructs a Delaunay reference from junction coordinates with Euclidean edge weights;
6. calculates direct Euclidean distance / weighted shortest-path ratios for reachable unordered pairs;
7. calculates root-directed efficiency from each terminal tip to the nearest reachable root;
8. repeats these measurements over successive image time points to obtain efficiency and construction-cost trajectories.

Repository:
https://github.com/Cocopyth/AMFTravellingWave

Paper:
https://doi.org/10.1038/s41586-025-08614-x

The repository README identifies Figshare DOI 10.6084/m9.figshare.27889143 as the source-data package. The inspected code licence is MIT; the dataset licence and redistribution rights have not been independently verified. Do not copy or redistribute the dataset until the relevant licence has been checked.

## 4. Fair-comparison requirements

Before numerical comparison with the paper, the Symthaea data adapter must provide:

- calibrated image-to-physical scale;
- measured hyphal edge lengths;
- an explicit 2D skeleton graph and crossing/fusion policy;
- explicit root and tip identities (not inferred from renderer-local origins);
- the same eligible-pair and nearest-root denominator rules;
- matched MST weight definitions;
- a Delaunay baseline with matched spatial dimension and point set;
- time-indexed observed snapshots with consistent edge identities;
- source and processing provenance for every measurement.

The current renderer-generated graph is not a substitute for this dataset. Its values are only useful for deterministic contract tests and performance scaling.

## 5. Proposed comparative table

For each time point and replicate, report:

| Family | Network | Metrics |
|---|---|---|
| Observed | measured hypha graph | global/root directness, total length/cost, components |
| Observed-edge baseline | weighted MST forest | global/root directness, selected edge cost, component count |
| Spatial baseline | Delaunay graph | global/root directness and total link length |
| Spatial lower bound | Euclidean MST | total Euclidean construction length |
| Proposed | Mycelial growth policy | directness, coverage, cost, persistent cycles, resilience |

Keep the distinction between graph-theoretic directness and physical transport. The latter requires measured widths, calibrated conductance assumptions and flow validation.

## 6. Proposed targets (not achieved)

- Exact MST edge count (V-C) for every connected-component configuration.
- Exact total-weight agreement with an independent Kruskal/Prim oracle on frozen small graphs.
- 100% deterministic selected-edge identities under input-order permutations.
- Zero source-topology mutation.
- Identical reachable-pair denominators between observed and MST reports.
- No Delaunay comparison until dimension, duplicate-point and degeneracy handling are explicit.
- On matched AMF data, reproduce the reference's reported global/root directness and cost curves within a predeclared tolerance before making a comparative performance claim.
- Only after replication, test whether the proposed growth policies improve the multi-objective Pareto frontier (directness, coverage, cost and resilience) on held-out time points/replicates.

## 7. Next extensions

1. Independent small-graph Kruskal/Prim oracle.
2. Euclidean MST, explicitly separate from observed-edge MST.
3. 2D Delaunay reference with robust duplicate/co-circular handling and a declared 2D-only track.
4. 3D Delaunay reference separately, with robust tetrahedral degeneracy handling.
5. Physical radius/volume/conductance profiles only from measured widths and calibrated assumptions.
6. Matched data adapter and time-indexed replication of the published AMF benchmark.

The core rule is to reproduce published definitions and denominators first, then test improvements. A more expressive 3D synthetic renderer is not itself evidence of superiority over a measured 2D biological benchmark.
