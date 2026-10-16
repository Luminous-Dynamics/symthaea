# Geospatial algorithm portfolio and evaluation gates

Status: research plan plus a repeat-sampled throughput harness for the existing spherical distance and bearing functions. This document does not claim that Symthaea currently implements the full portfolio below, nor that the benchmark has been executed.

## Design rule

Keep numerical kernels deterministic and independently testable. Symthaea may forecast workload and uncertainty, choose a spatial representation, tune bounded parameters, or select among kernels; it must not silently change a coordinate reference system, relax geometry predicates, or use a learned prediction as a substitute for a distance / topology invariant.

## Priority order

### P0 — coordinate and geodesic correctness

The `symthaea-geodesy` crate currently exposes spherical-Earth Haversine distance and initial bearing. Its declared model is useful for inexpensive global estimates, but it is not an ellipsoidal WGS84 inverse geodesic.

- Preserve the spherical implementation as a named, explicit model.
- Include valid-range validation / typed coordinate wrappers as a separate API proposal; do not silently normalize invalid latitude inputs.
- Compare spherical results to an ellipsoidal reference on near, far, antimeridian, high-latitude, and near-antipodal pairs.
- Use GeographicLib / Karney's algorithm as the trusted reference for difficult ellipsoidal inverse cases rather than treating Vincenty-only iteration as a universal oracle. GeographicLib documents a convergent inverse solution and near-roundoff accuracy for WGS84: https://geographiclib.sourceforge.io/html/C/ and https://geographiclib.sourceforge.io/html/Fortran/geodesic_8for.html.
- Keep the antipodal clamp regression and antimeridian test in the correctness corpus. A clamp prevents roundoff-induced NaN; it does not establish the true ellipsoidal distance or resolve the mathematical ambiguity of initial bearing at exact antipodes.

Run the microbenchmark:

```sh
cargo bench -p symthaea-geodesy --bench spherical_geodesy
```

The harness now runs 7 repeats of one million distance calls and 7 repeats of one million bearing calls over a fixed global-coordinate corpus, with 10,000-operation timed batches. It emits mean throughput and p50/p95/p99 batch-mean latency. The percentiles describe batch means, not individual-call latency; small fixed input arrays also make this a hot-cache throughput microbenchmark. Record CPU, OS, Rust toolchain, exact commit and raw output. This is only a throughput baseline: it does not establish accuracy or superiority over ellipsoidal algorithms.

The next P0 deliverable is a **separate correctness corpus** checked against GeographicLib reference values, with tolerances declared by intended use. Do not mix reference correctness scoring into the timed loop.

### P1 — global spatial indexing

Benchmark rather than assume one universal index:

- R-tree / bounding-volume hierarchy for arbitrary geometries and intersection queries.
- H3 for hierarchical hexagonal aggregation, neighborhood operations, and flow analysis: https://h3geo.org/docs/.
- S2 for global hierarchical cells with spherical-geodesic cell boundaries: https://s2geometry.io/devguide/s2cell_hierarchy.
- Geohash / quadtree or Morton / Hilbert ordering where locality, compact storage, or incremental updates are more important than cell shape.

These are not interchangeable candidates for every task: compare indexes on a shared task only when their query semantics match. For each applicable index, measure build cost, point lookup, radius and polygon queries, neighboring-cell traversal, memory, update cost, and false-positive refinement work. Make differences between logical parent/child relations and exact geographic containment explicit; H3 notes that hierarchical containment is logical and can differ from geometric containment across resolutions.

### P2 — routing on geographic graphs

Compare Dijkstra, A*, ALT / landmark heuristics, and incremental replanning (e.g. D* Lite) over identical road, terrain, and synthetic graphs. Record path cost versus an exact baseline, explored nodes, update/replan time, memory, and latency under edge-cost changes.

An A* heuristic based on geodesic distance must be admissible relative to the edge-cost model. If terrain, tolls, closures, or risk change the cost, do not assume straight-line distance proves optimality unless the scaling and lower bound are documented. Time-dependent routing should record departure-time assumptions and FIFO / non-FIFO handling.

### P3 — terrain, visibility, coverage, and hydrology

Create independently scored tasks for:
- line-of-sight / viewshed over elevation rasters;
- terrain-aware path cost;
- watershed and drainage consistency;
- radio / sensor coverage and placement;
- area and boundary calculations that cross the antimeridian or enclose a pole.

Tests must include poles, antimeridian-crossing polygons, near-degenerate edges, coordinate singularities, and multiple coordinate reference systems. Do not compare planar area or distance against ellipsoidal ground truth without stating the projection and expected error.

### P4 — adaptive resolution via Symthaea

Use prediction to prioritize computation, not to falsify the geometry. Candidate policy: coarse-to-fine refinement based on uncertainty and task sensitivity, with explicit error budgets and a deterministic fallback. Evaluate fixed-resolution, simple-threshold, and learned/predictive policies on held-out regions and query traces. Report the quality-versus-latency frontier, not just mean speed.

The key paradigm shift is **bounded computation with an error certificate**: the policy may decide *where to spend more work*, but the final result must either satisfy a declared deterministic error/correctness bound or return an explicit abstention / unresolved result. A model confidence score alone is not an error certificate.

## Evaluation protocol

- Version every input corpus and its license / provenance.
- Separate tuning regions and traces from held-out regions and traces; never tune on final test cases.
- Include uniform, clustered, sparse, high-latitude, boundary, and adversarial spatial distributions.
- Report p50/p95/p99 latency, memory, index build / update cost, distance or area error, path suboptimality, and failure counts.
- Include comparisons against a credible external implementation where practical.
- Keep coordinate units, datum, ellipsoid, longitude wrap rules, and altitude treatment explicit.
- No aggregate score may hide failure in high-latitude, antimeridian, or near-antipodal cases.
- Publish exact commit, toolchain, command line, raw outputs, seed, and test-corpus counts. Queued or skipped CI is not a pass.
- Define a primary metric and minimum practical improvement before each optimization; retain per-workload regressions instead of collapsing them into a single score.

## Proposed acceptance gates

1. No NaN / infinity for valid finite coordinates in the defined input domain.
2. Distance symmetry, coincident-point invariants, and documented degeneracy behavior hold within declared tolerances.
3. Reported geodesic model and units are explicit in APIs and serialized outputs.
4. A spatial-index optimization preserves exact post-filter results (false positives are acceptable in a broadphase; false negatives are not).
5. Adaptive selection improves a predeclared quality/latency objective on held-out workloads without breaching error budgets.
6. The learned/predictive component demonstrates value over a fixed-threshold policy in an ablation; otherwise prefer the simpler policy.
