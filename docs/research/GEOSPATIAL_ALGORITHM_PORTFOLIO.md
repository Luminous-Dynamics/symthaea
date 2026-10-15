# Geospatial algorithm portfolio and evaluation gates

Status: research plan plus a dependency-free throughput baseline for the existing spherical distance and bearing functions. This document does not claim that Symthaea currently implements the full portfolio below.

## Design rule

Keep numerical kernels deterministic and independently testable. Symthaea may forecast workload and uncertainty, choose a spatial representation, tune bounded parameters, or select among kernels; it must not silently change a coordinate reference system, relax geometry predicates, or use a learned prediction as a substitute for a distance / topology invariant.

## Priority order

### P0 — coordinate and geodesic correctness

The `symthaea-geodesy` crate currently exposes spherical-Earth Haversine distance and initial bearing. Its declared model is useful for inexpensive global estimates, but it is not an ellipsoidal WGS84 inverse geodesic.

- Preserve the spherical implementation as a named, explicit model.
- Include valid-range validation / typed coordinate wrappers as a separate API proposal; do not silently normalize invalid latitude inputs.
- Compare spherical results to an ellipsoidal reference on near, far, antimeridian, high-latitude, and near-antipodal pairs.
- If high accuracy is required, evaluate a robust ellipsoidal solver such as GeographicLib / Karney's algorithm as the reference rather than implementing a fragile Vincenty-only solver. The inverse solution in GeographicLib is designed to remain convergent for difficult cases: https://geographiclib.sourceforge.io/html/C/ and https://geographiclib.sourceforge.io/html/python/geodesics.html.
- Keep the recent antipodal clamp regression and antimeridian test in the correctness corpus.

Run the microbenchmark:

```sh
cargo bench -p symthaea-geodesy --bench spherical_geodesy
```

The harness runs one million deterministic distance calls and one million bearing calls over a fixed global-coordinate corpus. Record CPU, OS, Rust toolchain, commit, and raw output. This is only a throughput baseline: it does not establish accuracy or superiority over ellipsoidal algorithms.

### P1 — global spatial indexing

Benchmark rather than assume one universal index:

- R-tree / bounding-volume hierarchy for arbitrary geometries and intersection queries.
- H3 for hierarchical hexagonal aggregation, neighborhood operations, and flow analysis: https://h3geo.org/docs/.
- S2 for global hierarchical cells with spherical-geodesic cell boundaries: https://s2geometry.io/devguide/s2cell_hierarchy.
- Geohash / quadtree or Morton / Hilbert ordering where locality, compact storage, or incremental updates are more important than cell shape.

For each, measure build cost, point lookup, radius and polygon queries, neighboring-cell traversal, memory, update cost, and false-positive refinement work. Make differences between logical parent/child relations and exact geographic containment explicit; H3 itself notes that hierarchical containment is logically exact while geometric containment is approximate across some resolutions.

### P2 — routing on geographic graphs

Compare Dijkstra, A*, ALT / landmark heuristics, and incremental replanning (e.g. D* Lite) over identical road, terrain, and synthetic graphs. Record path cost versus an exact baseline, explored nodes, update/replan time, memory, and latency under edge-cost changes.

An A* heuristic based on geodesic distance must be admissible relative to the edge cost model. If terrain, tolls, closures, or risk change the cost, do not assume straight-line distance proves optimality unless the scaling and lower bound are documented. Time-dependent routing should record departure-time assumptions and FIFO / non-FIFO handling.

### P3 — terrain, visibility, coverage, and hydrology

Create independently scored tasks for:
- line-of-sight / viewshed over elevation rasters;
- terrain-aware path cost;
- watershed and drainage consistency;
- radio / sensor coverage and placement;
- area and boundary calculations that cross the antimeridian or enclose a pole.

Tests must include poles, antimeridian-crossing polygons, near-degenerate edges, coordinate singularities, and multiple coordinate reference systems. Do not compare planar area or distance against ellipsoidal ground truth without stating the projection and expected error.

### P4 — adaptive resolution via Symthaea

Use prediction to prioritize computation, not to falsify the geometry. Candidate policy: coarse-to-fine refinement based on uncertainty and task sensitivity, with explicit error budgets and a deterministic fallback. Evaluate fixed-resolution, simple threshold, and learned/predictive policies on held-out regions and query traces. Report the quality-versus-latency frontier, not just mean speed.

## Evaluation protocol

- Version every input corpus and its license / provenance.
- Separate training / tuning regions from held-out test regions.
- Include uniform, clustered, sparse, and adversarial spatial distributions.
- Report p50/p95/p99 latency, memory, index build / update cost, distance or area error, path suboptimality, and failure counts.
- Include comparisons against a credible external implementation where practical.
- Keep coordinate units, datum, ellipsoid, longitude wrap rules, and altitude treatment explicit.
- No aggregate score may hide failure in high-latitude, antimeridian, or near-antipodal cases.
- Publish exact commit, toolchain, command line, raw outputs, seed, and test-corpus counts. Queued or skipped CI is not a pass.

## Proposed acceptance gates

1. No NaN / infinity for valid finite coordinates in the defined input domain.
2. Distance symmetry and coincident-point invariants hold within documented tolerance.
3. Reported geodesic model and units are explicit in APIs and serialized outputs.
4. A spatial-index optimization preserves exact post-filter results (false positives are acceptable in a broadphase; false negatives are not).
5. Adaptive selection improves a predeclared quality/latency objective on held-out workloads without breaching error budgets.
