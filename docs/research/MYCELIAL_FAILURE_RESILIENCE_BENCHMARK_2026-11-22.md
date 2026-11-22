# Mycelial Failure Resilience Benchmark

**Date:** 2026-11-22  
**Status:** First controlled junction/segment deletion evaluator committed; exact-head validation pending.  
**Code:** `crates/domains/symthaea-canvas/src/mycelial_failure_resilience.rs`

## 1. The shift: from structural redundancy to functional disruption

A high cycle rank or visually dense network does not prove that important routes survive damage. The decisive question is whether declared network demand remains connected and how much longer its paths become after a controlled intervention.

This evaluator works on the junction–hypha graph, not the filament-adjacency graph. It removes explicitly named junctions and/or hypha segments from a derived graph view and measures the surviving topology and demand-pair paths. The source graph and evidence are immutable.

## 2. Input contract

### Demand pairs

Each `MycelialDemandPair` specifies:
- stable demand ID;
- source and target junction IDs;
- positive integer demand weight.

Demand pairs must be sorted by unique ID. The total demand weight is bounded to (2^{53}), so the integer weights can be represented exactly during IEEE-754 weighted aggregation. No normalization or inferred demand distribution is applied.

### Failure scenarios

A `MycelialFailureScenario` contains:
- a stable scenario ID;
- sorted unique failed hypha segment IDs;
- sorted unique failed junction IDs.

Removing a junction also removes all its incident segments. Combined scenarios can remove both junctions and segments. Unknown IDs, duplicates, unsorted failure inventories and empty non-baseline scenarios fail closed.

### Reproducibility

The profile binds:
- source junction–hypha graph digest;
- complete sorted demand inventory and demand digest;
- exact failed segment/junction IDs for every scenario;
- a per-demand outcome record (direct distance and surviving network path, or both explicitly absent);
- baseline and intervention summaries;
- canonical profile digest.

Profile validation recomputes each scenario's reachable/unreachable pair counts, weights, weighted reachability, directness and path-length aggregates from the per-demand outcomes. It rejects a graph path shorter than the Euclidean endpoint separation instead of omitting that demand from the summary.

## 3. Metrics

For each baseline or failure scenario:

### Structural topology

- remaining junction count;
- remaining hypha segment count;
- connected components;
- cycle rank:

[
eta_1 = E - V + C
]

### Demand functionality

For demand pair (q) with positive weight (w_q), the result stores its direct Euclidean distance (d_E(q)) and surviving graph shortest-path length (d_G(q)), or two explicit absent values if unreachable.

Weighted reachability:

[
R_w = rac{sum_{qin Q_{mathrm{reachable}}} w_q}
{sum_{qin Q} w_q}
]

Demand-weighted geometric directness:

[
D_w = rac{
sum_{qin Q_{mathrm{reachable}}}
w_q,d_E(q)/d_G(q)}
{sum_{qin Q_{mathrm{reachable}}}w_q}
]

Demand-weighted network path length:

[
L_w = rac{
sum_{qin Q_{mathrm{reachable}}}w_q,d_G(q)}
{sum_{qin Q_{mathrm{reachable}}}w_q}
]

Unreachable pair count and unreachable demand weight are reported separately. Directness and path length are undefined when no demand weight remains reachable; they are not set to zero.

The implementation rejects an impossible path where graph path length is shorter than Euclidean endpoint separation.

### What this means

- (R_w) measures structural reachability under the declared demand set.
- (D_w) measures geometric path directness among the reachable demand.
- (L_w) measures the weighted shortest-path distance among the reachable demand.
- (eta_1) describes graph cycle rank, not functional survival.

These are geometric graph quantities in the source graph's declared length unit. For the current generated Mycelial Presence model that unit is an organism-local model unit. No physical transport, conductance, material flow or biological viability is inferred.

## 4. Implemented operations

- `build_mycelial_failure_resilience_profile`: evaluate caller-supplied sorted controlled failure scenarios.
- `build_single_failure_resilience_profile`: generate one scenario per source segment and one per source junction, then evaluate the exhaustive single-failure sweep.
- Dijkstra is computed once per distinct demand source for each scenario.
- The graph is indexed in stable junction/segment ID order.
- Demand outcomes are stored individually and profile validation recomputes their counts, weights, reachability and directness aggregates.
- The exact failed identity lists are part of the receipt digest.

## 5. Bounds

| Contract | Bound |
|---|---:|
| Junctions | 256 |
| Hypha segments | 512 |
| Demand pairs | 256 |
| Distinct demand sources | 32 |
| Failure scenarios | 768 |
| Total integer demand weight | (2^{53}) |
| ID length | 128 bytes |

The single-failure sweep contains at most (V+E) scenarios: one per junction and one per segment. These bounds are intended for conformance/small-to-medium graph qualification. A larger-scale implementation should use a separately qualified incremental-dynamic-connectivity or batched shortest-path algorithm with equivalent receipts.

## 6. Synthetic ring-versus-tree control

The unit test compares a three-junction ring with a two-edge tree under removal of the same edge:

- the ring retains all declared demand pairs through the alternate route;
- the tree loses the demand pairs that depended on the removed bridge.

This is a deterministic structural conformance control, not a biological result. It demonstrates why cycle presence is only a hypothesis until the network is subjected to a defined failure.

The synthetic example compares a three-junction ring with the corresponding two-edge tree under the same bridge-edge failure. The ring retains all declared demand-pair reachability by using the alternate route; the tree disconnects the pairs that depended on the removed edge.

Run the synthetic benchmark:

```bash
cargo run -p symthaea-canvas --example mycelial_failure_resilience_bench -- 100
```

It reports the baseline and every single-junction/single-segment deletion, the weighted demand reachability/directness, profile digest and P50/P95 runtime.

## 7. Research benchmarks to pursue

The first source benchmark is the exact single-failure sweep. The broader programme should compare:

1. Random single-edge failures (distribution over seeds).
2. Random single-junction failures.
3. Degree-targeted and betweenness-targeted removal.
4. Articulation/min-cut-targeted removal.
5. Spatially correlated region failure, once the graph adapter retains calibrated curve geometry in a qualified coordinate frame.
6. Multiple simultaneous failures and common-cause events.
7. Recovery/repair: minimum added cost to restore a declared target reachability or directness threshold.
8. Temporal disruption: only after a time-respecting graph/flow model is measured.

Compare against static cycle rank, edge/node connectivity, articulation counts, minimum cuts, global directness, MST and Delaunay reference structures where assumptions match.

All policies must use the same frozen source graph, demand pairs, failure events and failure seeds. Report baseline-to-failure deltas and full distributions, not one best-case robustness score.

## 8. Acceptance gates

Proposed, not achieved:

| Metric | Initial target |
|---|---:|
| Deterministic profile digest on frozen graph/demand/scenario inputs | 100% |
| Missing/duplicate failure IDs accepted | 0 |
| Demand pairs omitted from scenario outcomes | 0 |
| Weighted aggregates inconsistent with per-demand outcomes | 0 |
| Network path accepted below Euclidean endpoint distance | 0 |
| Source graph digest changes after derived failure evaluation | 0 |
| Independent small-graph reachability oracle disagreement | 0 |

## 9. Claim boundaries

This module currently evaluates structural reachability and geometric shortest paths. It does not yet simulate or measure:
- cytoplasmic flow;
- pressure/flow conductance;
- nutrient transport;
- biological organism viability;
- adaptation or regrowth;
- repair cost or repair success;
- spatiotemporal correlated damage.

A network can preserve all demand-pair connectivity while its true transport capacity collapses. Functional biological resilience therefore requires calibrated edge conductance or measured transport, independent intervention outcomes and recovery observations. The structural graph is a necessary but insufficient layer.
