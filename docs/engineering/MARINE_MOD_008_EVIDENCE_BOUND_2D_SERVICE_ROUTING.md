# MARINE-MOD-008 — Evidence-bound 2D service routing

**Status:** research prototype / synthetic routing oracle  
**Module:** crates/domains/symthaea-marine-bench/src/modular_interior_service_routing.rs  
**Example:** cargo run -p symthaea-marine-bench --example modular_interior_service_routing  
**Frozen oracle:** fixtures/modular-interiors/service-routing-oracle-v1.json

## Why this increment

The configuration model describes connected service endpoints and the earlier work searches module variants and placements. It still lacks authoritative physical coordinates for each module port. Without those anchors, drawing a pipe/cable line between inferred points would make an invented route look like engineering evidence.

This increment therefore accepts explicit routing anchors and requires evidence references. A module-port anchor must identify its exact module instance and port, name the outward side of its envelope, lie on that side (not a corner), and match the current connection's endpoint IDs and medium. Both anchors must declare the same Z plane; cross-elevation and cross-deck transfers are rejected rather than invented. The router does not authenticate supplier drawings or measurements: the evidence class and references remain declared input, not verified facts.

## Contract

1. **Bound the problem.** Up to 32 requests; grid cells capped at 250,000; expanded A* states capped at 750,000 per route; clearance at most 5,000 mm; bend penalty at most 100,000 mm. Caller-provided tighter limits are expected.
2. **Use a declared grid.** The grid pitch must exactly divide the chosen deck's XY bounds; every anchor and outward-clearance step must align to that grid. Grid cells use integer millimetres and orthogonal neighbours only.
3. **Build conservative obstacles.** Every module's declared envelope and every reserved region block route cells. Route clearance inflates obstacle envelopes. Reserved regions marked requires_review are blocked too in this initial version, rather than silently treated as free space.
4. **Do not fake port locations.** Port anchors are external input. The module schema currently supplies port IDs and interfaces, but not authoritative local port coordinates or facing. A module anchor must carry a declared outward side; the router creates an explicit straight approach through that module's own clearance envelope and checks the corridor against every other obstacle.
5. **Search deterministically.** A* explores four-connected grid cells. The objective is grid distance plus the caller's bend penalty. The source approach axis and the final connection into a terminal port are included in the bend-cost boundary conditions.
6. **Keep route interactions visible without confusing junctions with collisions.** Requests are routed independently and then tested for exact shared-cell conflicts. A sole shared point is not reported as a route collision when both inputs explicitly identify the same platform-service ID, deck and exact terminal coordinate; sharing any additional path cell still triggers a conflict. This is not joint multi-route optimization, and declared common terminals do not model a qualified manifold or junction.
7. **Bind results to exact inputs.** The report hash binds the configuration digest, normalized request manifest, grid pitch, hard limits, paths, evidence classes, route conflicts and claim ceiling. Verification repeats routing and compares the report. Invalid generated paths fail closed instead of being emitted as routed. This is integrity/reproducibility, not a signature or proof of source authenticity.
8. **Keep the claim narrow.** Do not write the 2D proxy length back into the configuration's route_length_m, and do not imply that it is installed cable or pipe length. Vertical risers, pipe/cable diameter, bend radius, slopes, pressure drop, cable segregation, supports, fire stopping, penetrations, maintenance access, service capacity, stability and regulatory qualification are out of scope.

## Frozen synthetic oracle

The first oracle tests two behaviours:

- **WETROOM-DETOUR-001:** the explicit route from a declared sleeping-pod port anchor to the declared AC-bus service point must detour around the wet-room envelope expanded by the configured 100 mm clearance. The expected path length is at least 3,700 mm on the 100 mm grid. The synthetic anchors and module geometry are not real-vessel evidence.
- **SHARED-CELL-CONFLICT-002:** a route from the wet-room AC port to a different declared tap coordinate on the same service ID shares a multi-cell path segment with the cabin route. Both routes must list each other as conflicts.
- **SHARED-SERVICE-TERMINAL-003:** two routes that share only the exact declared AC-bus service endpoint must not be called a collision just because they terminate at the same point. Additional shared path cells remain conflicts. This is a conservative graph distinction, not proof that a physical junction or manifold is approved.

Additional tests cover connection-ID binding, port-side validation, reserved-zone endpoint blockage, route-request uniqueness, search-budget exhaustion, and recomputed-report integrity. These are authored Rust tests, not executed test results.

## Benchmark targets (proposals, not measured results)

| Benchmark | Target / acceptance gate |
|---|---|
| Shortest-path correctness | 100% agreement with independent Dijkstra on exhaustive tiny grids across the full test matrix of obstacle layouts and bend penalties |
| Path validity | 100% of emitted adjacent points are exactly one grid step apart; all non-terminal cells avoid clearance-inflated obstacles and reserved regions |
| Endpoint contract | 100% detection of stale IDs, wrong service/port medium, off-side anchors, grid misalignment and missing evidence references |
| Clearance and no-path | 100% sensitivity on frozen blocked-endpoint, closed-corridor, obstacle and budget negative controls |
| Multi-route interactions | 100% detection of exact shared-cell overlaps in the frozen oracle; later expand to minimum separation, swept volumes, route class and approved junction rules |
| Determinism | 100% byte-identical report digests for 100 permutations of equivalent request inputs |
| Scaling | Measure grid node tiers of 100, 1,000, 10,000, 100,000 and 250,000; record expanded states, route length proxy, bends, p50/p95 runtime and peak memory |
| Generalization | Evaluate at least three reviewed layout/route cases with source-verified anchors before making any engineering capability claim |

A future target is to route one 10,000-cell problem within one second on a pinned host, with exact path validity and the same deterministic output. This is a profiling target only—not an observed result or SLA. A more important early result is a small exhaustive reference suite that shows the A* result is optimal for the declared graph and cost model. Optimizing against the wrong or incomplete graph is not progress.

## Research context

1. The 2026 paper [Ship engine room pipe layout optimization based on a knowledge-driven sparse A* algorithm](https://www.sciencedirect.com/science/article/pii/S0029801826016732) reduces search dimensionality through sparse topology guidance, orthogonal escape vectors, dynamic boundaries and explicit path/rule objectives. It supports the roadmap toward sparse routing and benchmarked rule compliance, but does not imply this grid prototype matches its performance.
2. The August 2026 open-access paper [An intelligent and engineering-oriented optimization framework for ship pipe collaborative layout design in realistic 3D compartment environments](https://www.sciencedirect.com/science/article/pii/S0141118726002129) treats irregular 3D geometry, obstacle interference and coupling between multiple pipes. That is the long-term destination; this implementation is only a bounded 2D step.
3. The 2026 [case-based engine-room arrangement and passageway study](https://www.sciencedirect.com/science/article/pii/S2092678226000269) combines graph structure, passageway generation, accessibility and service-aware arrangement rather than optimizing isolated pipes.
4. IMO [SOLAS overview](https://www.imo.org/en/about/conventions/pages/international-convention-for-the-safety-of-life-at-sea-%28solas%29%2C-1974.aspx) and [fire-protection guidance](https://www.imo.org/en/ourwork/safety/pages/fireprotection-default.aspx) emphasize fire containment and protection of means of escape. Which provisions apply depends on vessel class and service; routing and reserved-space rules must be mapped by qualified people before a design can be assessed for compliance.

## Next paradigm shifts

### 1. Shared multi-route optimization

Move beyond independently finding shortest paths and reporting cell conflicts. Jointly optimize a bundle with explicit system priorities, protected spaces, separation classes, approved junctions, route ownership, and maintenance pull/access envelopes. Because route ordering can change feasibility, preserve exhaustive multi-route oracles for small instances.

### 2. Real 3D clearance and topology

Replace 2D rectangles with sourced compartment surfaces, structural penetrations, obstacles and equipment envelopes. Require riser/vertical-transition evidence. A 2D route must never be silently promoted to a 3D route.

### 3. Domain-specific physical models

After topology is credible, add separate solvers for pipe hydraulics, exhaust/HVAC losses, electrical cable sizing/segregation, bend-radius/installation constraints, supports, thermal/fire containment and maintainability. Report the governing inputs, model versions and uncertainty for each discipline. Geometry alone does not establish adequacy.

### 4. Evidence-directed engineering

If a route is blocked or unresolved, report which missing input would change the decision: port coordinates, a supplier connector drawing, a surveyed penetration, a service capacity, or a verified reserved-space boundary. The goal should be to select the next useful measurement, not to invent a workaround.

### 5. Independent verification

Check A* against Dijkstra on tiny grids, fuzz obstacle boundaries, and inject false port anchors, route clearance violations, stale connection references, no-path walls, and shared-cell conflicts. Add a second route checker that reconstructs and independently validates every emitted path from raw inputs.

## Verification boundary

Rust build, rustfmt, unit-test execution, example execution and exact-head hosted checks are separate qualification gates. Queued, skipped, absent or stale checks are not passes. The service-route result is an explicit-anchor, 2D orthogonal path proxy only. It does not establish installed routing, 3D feasibility, service performance, maintainability, SOLAS compliance, stability, seaworthiness or permission to operate.
