# MARINE-MOD-010 — independent A* / Dijkstra oracle

Status: test implementation authored; Rust tests and hosted checks have not been executed successfully in this environment.

## Why this matters

Re-running the same A* implementation to verify a receipt checks reproducibility, not optimality. We need a second algorithm with independent transition logic before changing the geometry representation or making performance claims. This increment adds a test-only Dijkstra reference for small 2D orthogonal grids and compares it to the production A* search.

## Implementation

The test-only reference implementation deliberately does not call production A* neighbor generation, segment intersection, cell-to-coordinate conversion, or path metrics. It uses a uniform-cost search over cell-plus-direction states, calculates bends from transitions, and implements a separate rectangle/segment intersection routine.

Frozen fixture cases cover:
- Open 3×3 grid with zero bend cost (expected cost 40 mm in the synthetic unit model).
- Open 3×3 grid with positive bend cost (47).
- Different source-escape and terminal-approach directions (47).
- Center-cell detour (54).
- A fully blocked column (no path).
- A thin barrier between grid nodes that must still block crossing segments (no path).

The Rust test definitions also cover every distinct endpoint pair on an open 3×3 grid across all 3×3 combinations of source/terminal directions and four bend penalties (2,304 cases), plus 160 deterministic seeded 5×5 obstacle maps. Each produced A* path is independently checked for endpoint identity, grid-step length, blocked cells, and segment crossings. A negative control removes the thin wall and checks that reachability returns.

The frozen values have also been checked with a separate JavaScript Dijkstra/A* model. That is a supporting model-level calculation, not evidence that the Rust test suite compiles or passes.

## Paradigm shifts and next research

**1. Distinguish algorithmic correctness from reproducibility.** Keep recomputation receipts, but compare the optimizer against an implementation-independent oracle. Frozen expected values detect drift in both implementations; randomized maps broaden the differential test domain with a pinned seed.

**2. Change where search nodes come from, not just how quickly they are searched.** A uniform grid is an excellent transparent reference, but it spends states in open space. The next candidate is a sparse orthogonal visibility/topology graph built from inflated obstacle corners, explicit service-port escape points and route-segment constraints. This is inspired by 2026 ship-piping research on knowledge-guided sparse A* and engineering-shaped topology generation. Do not copy claims of completeness without proving the candidate-node construction complete for Symthaea's exact obstacle and routing semantics.

**3. Preserve a grid challenger.** Every sparse-graph result should be compared against Dijkstra/A* on tiny grids where the exact answer is known. If sparse-graph completeness is not established for a case, report it as a performance challenger, not as an exact replacement.

**4. Expand only after the reference is trustworthy.** Model shared-route separation, terminal/junction compatibility and clearance as hard constraints. Later move into explicit 3D mesh geometry, bend radii, vertical penetrations, service-specific separation and maintenance envelopes, each with their own test fixtures and discipline-specific evidence. Geometry alone does not establish hydraulic/electrical performance or vessel compliance.

## Proposed benchmarks (targets, not results)

| Benchmark | Target |
|---|---:|
| Frozen fixture agreement | 100% expected disposition and cost |
| Open-grid exhaustive differential | 100% agreement across all bounded cases |
| Seeded obstacle maps | 100% A* / Dijkstra cost and reachability agreement |
| Independent emitted-path checks | 0 invalid paths or obstacle crossings |
| Negative-control sensitivity | 100% for removed wall, cost mutation and tampered path |
| Sparse graph vs. grid oracle | 0 cost gap for cases with a proven complete sparse graph |
| Search-efficiency ambition | At least 2× lower p95 expanded states on a pinned 100k-cell workload family, without cost gap on exact cases |
| Reporting | p50/p95 runtime, expanded states, peak memory, vertices/edges, success rate, timeout/truncation and exactness coverage |

The 2× performance number is an aspirational benchmark. No sparse graph has been implemented or benchmarked in this increment.

## Verification boundary

- The production route implementation is unchanged here; the new Dijkstra code is inside the routing module's test module.
- The work remains synthetic 2D path optimization. It does not prove 3D routing, physical installation lengths, pipe/cable hydraulics/electrical adequacy, bend radius, service separation rules, fire stopping, maintenance access, evacuation, SOLAS/class compliance, stability, seaworthiness or authority to build/operate.
- All values and controls are synthetic. The oracle validates algorithmic conformance, not supplier/as-built coordinate authenticity.

## Research context (2026)

- Dechen Liu et al., “Ship engine room pipe layout optimization based on a knowledge-driven sparse A* algorithm,” *Ocean Engineering* 358 (2026), DOI: [10.1016/j.oceaneng.2026.125839](https://doi.org/10.1016/j.oceaneng.2026.125839). The paper motivates sparse topology, obstacle-informed escape seeds and minimum straight-segment constraints.
- Hongshuo Zhang et al., “An intelligent and engineering-oriented optimization framework for ship pipe collaborative layout design in realistic 3D compartment environments,” *Applied Ocean Research* 173 (2026), article 105129, DOI: [10.1016/j.apor.2026.105129](https://doi.org/10.1016/j.apor.2026.105129). The paper motivates system-level multi-route coupling and realistic 3D obstacle models.

These papers motivate research directions. They do not validate this code, which has not run under Rust in this environment.
