# Mycelial Junction–Hypha Graph Contract

**Date:** 2026-11-21  
**Status:** First renderer-neutral source implementation committed; exact-head CI pending.  
**Scope:** Convert a cubic-filament morphology into a graph whose vertices are junctions and whose weighted edges are hyphal curve intervals.

## 1. Why a second graph representation is necessary

The existing `MycelialNetworkGraph` uses **filaments as vertices** and parent/anastomosis relations as edges. That is useful for filament-level failure experiments. It is not the same abstraction as a biological mycelial skeleton, where **tips, branch points and anastomoses are vertices, and hyphal segments are weighted edges**.

The Mycelium-as-a-Network review describes the junction/link representation and notes a key danger: in 2D, a geometric crossing can be mistakenly treated as a fully connected junction even when no physical anastomosis exists. This implementation makes the authority explicit: parent IDs establish branch connectivity; an explicit accepted fusion record establishes anastomosis connectivity; coincident coordinates alone never create an edge.

Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC11687498/

## 2. Source implementation

New module:

`crates/domains/symthaea-canvas/src/mycelial_junction_graph.rs`

Public module and re-exports:

`crates/domains/symthaea-canvas/src/lib.rs`

The primary builder is:

`build_mycelial_junction_hypha_graph(model, accepted_anastomoses, config)`

The builder is a read-only projection. It does not modify MycelialModel, MycelialNetworkGraph, or CognitiveTopology.

### Junction kinds

- `Origin`: a filament start without a declared parent. Coincident origins remain distinct.
- `Branch`: a parent's end and its explicitly parented child starts.
- `Tip`: a terminal filament endpoint without a fusion.
- `Anastomosis`: an explicitly accepted fusion point.

The current model does not identify biological plant roots. Root-directed efficiency therefore requires the caller to supply a declared set of junction IDs. The adapter does not silently treat every renderer-local origin as a host root.

### Hypha segments

Each segment preserves:

- stable source filament ID;
- a graph-level, sorted `source_filament_ids` list that lets deserialized validation detect a wholly omitted source curve;
- quantized `t_start` and `t_end` parameters at one-millionth resolution;
- start and end junction identities;
- actual cubic-Bezier arc-length estimate;
- endpoint weld offsets to the canonical junction anchors;
- interpolated renderer-model radius at both ends.

A tip-to-side fusion splits the target filament into two adjacent parameter intervals. All source curves must be accounted for by contiguous intervals covering exactly [0,1]. Missing, overlapping or discontinuous intervals fail validation.

### Junction contact geometry

The fusion record supplies a declared junction anchor. The builder checks the source and target curve contact positions against that anchor. The maximum accepted gap is configurable and bounded. Each segment retains its own endpoint offset to the anchor.

Two different lengths are kept separate:

`hypha material proxy = sum(curve arc lengths)`

`graph traversal length = arc length + start weld offset + end weld offset`

This prevents a near-contact from creating a graph path shorter than the represented geometry. Weld offsets are reported separately and are not treated as constructed hyphal material.

### Arc length

The current source uses a fixed 64-interval composite-Simpson estimate of cubic-Bezier speed. It is deterministic and bounded, but is an approximation. A future accuracy study should compare this quadrature with an independently implemented adaptive integrator and report the error against high-precision fixtures.

## 3. Current bounds and validation

Initial source limits:

| Contract | Current bound |
|---|---:|
| Source filaments | Existing mesh maximum |
| Junction vertices | 8,192 |
| Hypha segments | 16,384 |
| Junction weld gap | Configurable, default 0.05 model units |
| Parameter quantization | 1e-6 |
| Arc-length integration | 64 Simpson subintervals |
| Geometric-efficiency graph | Maximum 2,048 junctions |

Fail-closed checks include:

- source model version;
- source filament ID uniqueness;
- parent-reference existence and parent-cycle rejection;
- finite, bounded curve coordinates and radii;
- terminal flag consistency with the explicit child set;
- parent-child start/end contact tolerance;
- accepted-fusion validation through the existing network contract;
- exact source/target contact parameter constraints;
- maximum declared-junction gap;
- collision of two fusion records at the same filament/parameter contact;
- segment endpoint references and incidence consistency;
- junction metadata consistency (terminal IDs are incident, fusion IDs exist only on anastomosis junctions);
- bounded IDs, coordinates, arc lengths, weld offsets and radii;
- full, gap-free, non-overlapping parameter coverage for every declared source filament, including detection of a completely omitted curve after deserialization;
- serialization round-trip digest equality.

The current implementation conservatively rejects two separate fusion records that claim the same curve contact. Multi-party anastomosis junctions are a follow-on; they must merge through an explicit accepted multi-contact record or a separately validated union rule, not by coordinate proximity.

## 4. Metrics currently implemented

### Structural metrics

- junction count;
- hypha segment count;
- connected components;
- cycle rank:

[
eta_1 = E - V + C
]

- total hypha curve arc length in renderer-local model units;
- total junction weld offset;
- maximum declared junction gap.

The cycle-rank value is an invariant of this graph representation; it is not by itself a resilience or biological-function score.

### Geometric path metrics

Weighted Dijkstra shortest paths use arc length plus endpoint offsets.

Global geometric directness is:

[
E_g = operatorname{mean}_{(i,j)in R}
rac{d_E(i,j)}{d_G(i,j)}
]

where (R) is the set of connected unordered junction pairs, (d_E) is Euclidean separation, and (d_G) is the weighted graph shortest path.

The report includes eligible, reachable and unreachable pair counts. Unreachable pairs are not silently treated as zero-length paths.

Root-directed directness takes a caller-provided set of root junction IDs. For each junction containing a terminal filament contact, it selects the reachable declared root with minimum weighted path length and compares graph path length with direct Euclidean distance. This follows the nearest-root pairing logic in the published AMF analysis while avoiding invented root semantics.

The result is explicitly a **geometric directness measurement in model units**. It is not a measured transport rate or physical efficiency until a calibrated embedding and transport model/data are supplied.

## 5. Research comparators

### Empirical AMF development and trade-offs

Oyarte Galvez et al., *A travelling-wave strategy for plant–fungal trade*, Nature 639, 172–180 (published 26 February 2025), analysed time-indexed fungal network growth and compared global/root geometric efficiency and network construction cost against minimum-spanning-tree and Delaunay reference networks.

References:
- https://doi.org/10.1038/s41586-025-08614-x
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11882455/
- Authors' code: https://github.com/Cocopyth/AMFTravellingWave

The paper's efficiency quantities compare direct spatial distance with shortest-path distance. Its observed networks exhibit different temporal trajectories for global efficiency, root efficiency and cost. Therefore the Symthaea benchmark should compare complete **time-indexed Pareto trajectories**, not one final graph score.

### Directness and coverage during network growth

Sebastiao et al., *The trade-off between directness and coverage in transport network growth*, Applied Network Science (published 27 March 2026), found distinct effects from random, centrality, directness and coverage-based link-order strategies. Directness-oriented greedy selection can become trapped in local optima.

Reference: https://doi.org/10.1007/s41109-026-00792-5

This motivates a temporal benchmark in which every accepted link produces a recorded intermediate graph and a new directness/coverage/cost/resilience report. Compare local greedy methods against bounded look-ahead or Pareto-aware growth strategies.

### Tree-to-cycle transition scale

Stark and Godin's 6 September 2026 bioRxiv preprint proposes the radius of the thickest edge outside a maximum spanning tree as a descriptor of the cost accepted for redundancy. It is an exploratory, non-peer-reviewed hypothesis.

Reference: https://doi.org/10.64898/2026.09.02.748799

Do not apply it to the current stylized renderer radii. It requires calibrated physical width and cost data.

## 6. Benchmark programme

Targets below are proposed, not achieved.

### Adapter correctness

| Metric | Initial gate |
|---|---:|
| Source filament interval coverage | 100% |
| Overlapping/missing intervals accepted | 0 |
| Duplicate/dangling junction references | 0 |
| Geometric crossing promoted to connection | 0 |
| Unaccepted candidate promoted to fusion | 0 |
| Invalid/non-finite geometry accepted | 0 |
| Semantic digest changed by this projection | 0 |
| Deterministic source and graph digests | 100% on frozen fixtures |

### Numerical accuracy

- A deterministic high-resolution chord-length oracle now checks a high-curvature cubic-Bezier fixture against the 64-interval Simpson implementation. This is an initial regression check, not an exhaustive numerical qualification.
- Compare Simpson arc length against an independent adaptive integrator on lines, circles approximated by cubic Beziers, high-curvature curves, inflections and near-degenerate curves.
- Report relative and absolute integration error over a frozen [0,1] curve-parameter corpus.
- Initial target: median relative error ≤1e-6 and P99 ≤1e-4 for non-degenerate curves; these are proposed targets to verify, not current claims.
- Compare Dijkstra results with an independent small-graph all-pairs reference implementation.
- Require geometric directness not to exceed 1 beyond a declared numerical tolerance; any violation fails closed and reports the responsible junction/segment IDs.

### Empirical network metrics

On a matched physical junction–hypha dataset, report:

- global and root-directed geometric directness;
- total hyphal arc length and, if width is measured, material-volume proxy;
- MST and Delaunay reference bounds with matching dimension, terminal set, units and tie-breaking;
- cycle rank, connectivity, minimum cuts and failure response;
- directness and coverage after every growth event;
- cost and efficiency trajectories over time.

Do not compare renderer-local model-unit results numerically with physical millimetre results.

## 7. Next extensions

1. Add an explicit model-coordinate calibration contract before any physical units are emitted.
2. Support multiway fusion contacts through a versioned, explicit junction record.
3. Add a 2D/3D measurement adapter with a declared topology policy for crossings and overlaps.
4. Implement matched MST and Delaunay reference baselines with deterministic tie-breaking and explicit dimensionality.
5. Add a physical width/radius data field only when sourced from measurements.
6. Add temporal network-growth policies and report the directness/coverage/cost/resilience Pareto frontier after each accepted edge.
7. Compare Simpson arc length against an independent adaptive integrator.
8. Add image/data ingestion only after source integrity, licensing and redistribution conditions are verified.

## 8. Claim boundaries

- Renderer-local model units are not physical units.
- Stylized model radii are not measured hyphal widths.
- Geometric directness is not cytoplasmic flow.
- Cycle rank is not sufficient evidence of resilience.
- A topology adapter is not a biological growth simulator.
- A graph crossing is not an anastomosis.
- Root-directed metrics require declared root IDs.
- No semantic CognitiveTopology relationship is inferred from geometric proximity.

The intended contribution is a deterministic, evidence-conscious graph conversion and benchmark surface on which later physical, biological and cognitive experiments can be compared without silently mixing different representations or physical assumptions.
