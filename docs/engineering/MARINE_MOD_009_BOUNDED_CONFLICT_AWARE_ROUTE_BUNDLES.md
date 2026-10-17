# MARINE-MOD-009 — bounded conflict-aware service-route bundles

Status: research implementation; local Rust execution and hosted checks are not asserted.

## Why this changes the problem

The independent service router can emit locally good paths that are jointly incompatible. This layer treats each route as a small candidate catalog, then chooses one path per request under a hard shared-cell constraint. It does not trade a safety conflict against a shorter length using a weighted penalty.

Candidate discovery currently uses a deterministic edge-ban deviation heuristic, with per-route and aggregate deviation-probe caps (the default aggregate cap is 128 probes) and a global A* expansion-state cap (2,000,000 by default; hard maximum 25,000,000 across the whole bundle):

1. Run the existing evidence-bound A* router for the base path.
2. For each eligible interior edge of each discovered path, add a temporary 1 mm obstacle that bans that edge (with the original route clearance applied by the router).
3. Rerun the same router, record the exact altered-configuration digest and outcome, independently audit each returned path against the original configuration, and keep only novel valid paths.
4. Repeat from each newly discovered path until the finite deviation frontier is exhausted or an explicit bound/verification failure stops it. Every base search and deviation search consumes the same aggregate expanded-state budget; the report records effective per-call caps and expanded-state counts, including an explicit record if a search is not launched because the budget is exhausted.
5. Enumerate candidate combinations in stable route-ID order. Reject any bundle with a shared grid cell except one exact shared platform-service terminal with the same service ID, deck, elevation, and medium. Minimize total declared path cost; break ties by length, bend count, then canonical path coordinates.

This is **not** Yen's algorithm, a proven k-shortest algorithm, nor exhaustive path enumeration. A ban probe returns at most one route from the existing deterministic A* search; some feasible paths will never enter the catalog. The field discovery_frontier_exhausted means all generated single-edge-ban probes within this procedure were processed, not that all routes were found. The field bundle_catalog_product_exhausted means all combinations of the catalog actually emitted were evaluated. The selected bundle is optimal only over the discovered catalog.

## Evidence and claim ceiling

- Anchors, module-port sides and service tap coordinates remain explicit caller inputs. The router does not authenticate them against supplier drawings, as-built surveys or an installation authority.
- Candidate paths are checked against the original configuration by verify_service_route_geometry, not merely against the temporary edge-ban configuration.
- Each alternative records a digest of its path result; each deviation probe records the altered configuration digest, banned edge and result/disposition. Duplicate and failed probes are retained.
- The selection treats shared path cells as a hard incompatibility. Only the same explicit platform-service terminal may be shared, and only for the same service ID, deck, Z and medium. Any additional shared cells are rejected.
- Reports are digested and verification recomputes candidate generation and selection. This is integrity/reproducibility verification, not a cryptographic signature or proof of physical truth.
- The current implementation remains 2D orthogonal grid screening. It does not establish pipe/cable installation length, hydraulic/electrical capacity, pressure drop, bend radius, thermal expansion, fire stopping, maintainability, evacuation, 3D penetrations, SOLAS/class compliance, stability, seaworthiness or operating approval.

## Paradigm-shift roadmap

**Near term — measurable algorithmic correctness**

- Freeze tiny candidate catalogs and compare the optimizer against a separately written exhaustive Cartesian-product oracle.
- Freeze tiny grids and compare route costs/validity against an independent Dijkstra implementation; do not treat the existing route recomputation verifier as an independent algorithm oracle.
- Improve catalog diversity, provenance and negative controls before increasing candidate count.
- Add endpoint-compatible junction classes and explicit route-separation policy instead of treating all geometric overlaps as equivalent.

**Next — change the search representation**

The grid is useful as a transparent reference oracle but scales poorly. A sparse orthogonal visibility/topology graph built from obstacle corners, explicit port escape vectors, and minimum straight-segment constraints could reduce search states without quietly lowering clearance checks. Every sparse-graph result should still be checked by the independent geometry auditor and compared with the grid oracle on cases where both can finish. Do not claim solution-space completeness until the graph construction has an argued completeness condition for its stated geometric model.

**Then — move from 2D prototypes to actual engineering disciplines**

Use a versioned 3D obstacle representation (e.g. triangle mesh plus BVH) and explicit, reviewed adapters for penetrations, vertical risers, bend radii, installation tolerances, service-specific separation, maintenance envelopes, fire zones and compartment boundaries. Keep each discipline constraint hard unless a qualified ruleset explicitly classifies it otherwise. Keep geometric, hydraulic/electrical, access, and regulatory evidence separate rather than folding them into an opaque weighted score.

## Benchmark contract

All values below are proposed targets, not measurements.

| Benchmark | Target | Method |
|---|---:|---|
| Tiny route-bundle oracle | 100% agreement | Exhaust every candidate combination for frozen 1–3-route catalogs and compare with the production selector |
| Tiny-grid route oracle | 100% cost/validity agreement on bounded complete cases | Independent Dijkstra over grids small enough for exhaustive evaluation |
| Selected-bundle shared-cell conflicts | 0 | Independent post-selection checker; any forbidden overlap invalidates selection |
| Compatible terminal control | 100% accepted when otherwise valid | Exact common-terminal negative/positive pair: same endpoint alone allowed, an extra shared cell rejected |
| Obstacle, route endpoint, grid-step and clearance checks | 100% of frozen controls | Include thin/sub-grid barriers, boundary cases, budget exhaustion and tampered polylines |
| Report determinism | 100/100 equivalent request-order permutations | Same canonical input/configuration must yield the same digest |
| Tamper detection | 100% of frozen mutations | Mutate candidate route, selected index, probe record, objective and input manifest |
| Candidate coverage | Reported, not hidden | Count distinct valid alternatives, duplicate probes, failed probes, budget stops and truncation reason; retain per-route/global probe counts plus aggregate A* expanded states and per-probe search caps |
| Bundle workloads | 2, 3, 4, 6 routes; 8 candidates per route | Report success rate, combinations evaluated, conflict rejections and whether catalog-product enumeration completed |
| CPU/memory scaling | 10k, 50k, 100k, 250k grid-cell tiers | Pin host/toolchain; report p50/p95 runtime, peak RSS, expanded states and timeouts |
| Initial latency target | 10k-cell, one-route base search below 1 s on a pinned host | Provisional engineering target only; never a pass/fail substitute for geometry validity |

For bundle quality, report the aggregate declared path cost, total length and total bend count separately. Report the cost gap to the exact oracle only for cases where the oracle has genuinely exhausted the stated path/bundle universe. Do not call route success rate “safety” or report a physical minimum separation from the current shared-cell check.

## Research context (2026)

- Liu et al., “Ship engine room pipe layout optimization based on a knowledge-driven sparse A* algorithm,” *Ocean Engineering* (2026): sparse topology and orthogonal escape vectors are a useful direction for lowering search dimensionality. [Publisher page](https://www.sciencedirect.com/science/article/pii/S0029801826016732)
- Zhang et al., “An intelligent and engineering-oriented optimization framework for ship pipe collaborative layout design in realistic 3D compartment environments,” *Applied Ocean Research* (August 2026): multi-trajectory solutions, mesh/BVH geometry and system-level multi-pipe coupling motivate the progression beyond independent 2D paths. [Publisher page](https://www.sciencedirect.com/science/article/pii/S0141118726002129)

These papers motivate the research questions. They do not validate this implementation or establish that it matches the published algorithms.
