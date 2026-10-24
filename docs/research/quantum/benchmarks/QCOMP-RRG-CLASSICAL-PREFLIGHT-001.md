# QCOMP-RRG-CLASSICAL-PREFLIGHT-001: exploratory classical controls on random 3-regular graphs

Status: **exploratory JavaScript preflight; not Rust execution, not QGreedy reproduction, no timing claim**

## Research question

Before attempting to reproduce fixed-angle QAOA-guided greedy MIS, how much variance and apparent improvement can a weak classical policy obtain from randomized tie-breaking alone, and what is the marginal value of a small local improvement on the same graph?

This is motivated by Wybo et al., *A scalable quantum-enhanced greedy algorithm for maximum independent set problems on random regular graphs*, accepted by Physical Review Research on 19 October 2026 (https://doi.org/10.1103/ds1n-wsdm; https://arxiv.org/abs/2601.21923). The paper studies fixed-angle QAOA expectation-guided greedy on random 3-regular graphs at N = 50, 100, 200, 500, 1,000, 2,000 and 5,000. It compares with classical greedy and Marino–Kirkpatrick linear-time prioritized search. The current preflight implements neither QAOA nor Marino–Kirkpatrick and must not be presented as a paper reproduction.

## Cohort and generation method

- 70 generated simple random 3-regular graph instances: 20 each at N = 50, 100 and 200; 10 at N = 500.
- Instances are generated via the configuration model: create d stubs per vertex, Fisher–Yates shuffle with seeded XorShift64*, pair the stubs, and discard the full pairing if it contains a self-loop or repeated undirected edge. Canonical edges are sorted by (u, v).
- Graph seed formula: 202610230000 + N * 1000003 + zero_based_instance_index * 7919.
- Graph identity in the JSON is FNV-1a 64-bit over canonical ASCII DIMACS header and sorted one-based edge lines, without comment metadata. FNV-1a is non-cryptographic; it is a fast identity check, not a security hash. For the canonical experiment artifact, emit DIMACS and compute SHA-256 separately.
- This is an exploratory JavaScript reference, not Rust-generated data. Consolidation found a seed-stream mismatch in the first Rust candidate: its retry loop shuffled the previously failed permutation, while the reference resets to the canonical stub multiset on every attempt. The current generator in [PR #8324](https://github.com/Luminous-Dynamics/symthaea/pull/8324) has been changed to clone canonical stubs for each attempt and uses the documented rejection-based bounded sampler. A 70-vector Rust regression test is now authored for all frozen records: each test checks expected accepted attempt count, edge count, regular degree, and canonical DIMACS FNV-1a fingerprint. Separately, an independent JavaScript reimplementation of this corrected algorithm matched all 70 recorded attempt counts and fingerprints. This supports alignment of the reference and reviewed algorithm, but is **not Rust execution**: the Rust test has not been compiled or run because cargo/rustc are unavailable. The CLI preserves workload identity on retry exhaustion and emits a structured JSON failure receipt on stderr while keeping graph stdout empty. Rust-to-reference runtime/output parity therefore remains unproven until the binary is executed.
- Every graph was checked to have exactly N vertices, 3N/2 unique undirected edges, and degree exactly three at every vertex.

## Policies and cost boundary

For each graph:

1. Deterministic minimum-residual-degree greedy, ties by lowest vertex ID.
2. Sixteen seeded random tie-breaking minimum-residual-degree runs, each using a derived seed recorded per restart. The JSON preserves every per-run objective, not only the best.
3. A best-of-16 restart result, which is explicitly a 16-run budget and must not be compared against one greedy run as if the cost were equal.
4. For N <= 200, deterministic repeated feasible 1-for-2 improvement from the one-run deterministic greedy result. It removes one selected vertex and adds two mutually nonadjacent candidates compatible with the rest of the selected set, stopping when no such move is found. The pair-test cap is 2,000,000; exhausted runs would be labelled as unfinished, not locally optimal.

The random-tie preflight uses a seeded XorShift64* family plus unbiased bounded sampling for tie selection. This is deterministic reference code in a transient JavaScript environment, not a production algorithm or the Rust next_usize call path. No wall-clock timings were recorded.

## Results

Means below are across graph instances, not across all vertices or all repeated runs. Best-of-16 is the per-instance maximum over 16 restart runs.

| N | Graphs | Min-degree greedy mean ratio | Mean of 16 random-tie ratios (first average within each graph) | Best-of-16 mean ratio | Paired best-of-16 win / tie / loss vs one greedy |
|---:|---:|---:|---:|---:|---:|
| 50 | 20 | 0.4180 | 0.4235 | 0.4380 | 15 / 5 / 0 |
| 100 | 20 | 0.4260 | 0.426188 | 0.4405 | 16 / 4 / 0 |
| 200 | 20 | 0.43075 | 0.430078 | 0.43975 | 17 / 2 / 1 |
| 500 | 10 | 0.4296 | 0.430712 | 0.4374 | 9 / 1 / 0 |

For N <= 200, one-for-two improvement applied to the deterministic greedy set yielded mean ratios of 0.4210, 0.4270 and 0.43175 respectively, with mean cardinality gains of 0.15, 0.10 and 0.20 vertices per graph. Every local-improvement run completed its pair search and was labelled locally optimal under this particular 1-for-2 neighborhood; this is not a global-optimality statement. This weak move alone did not materially close the gap from greedy.

## Post-hoc graph-level uncertainty diagnostic

To make the descriptive differences more transparent, the JSON companion now also records a graph-level stratified bootstrap over the 70-instance JavaScript cohort. The 16 random-tie trajectories remain nested within a graph: the bootstrap resamples graph instances within each N stratum and never treats the 1,120 restart records as additional independent graph samples.

| N | Graphs | Mean objective gain: best-of-16 minus single greedy | Paired win / tie / loss |
|---:|---:|---:|---:|
| 50 | 20 | 1.00 | 15 / 5 / 0 |
| 100 | 20 | 1.45 | 16 / 4 / 0 |
| 200 | 20 | 1.80 | 17 / 2 / 1 |
| 500 | 10 | 3.90 | 9 / 1 / 0 |
| All | 70 | 1.77 | 57 / 12 / 1 |

The post-hoc stratified graph bootstrap gives a 95% percentile interval of [1.46, 2.09] vertices for the overall mean objective gain. For the local 1-for-2 improvement, which was run only at N≤200 (60 graphs), mean gain is 0.15 vertices; the corresponding post-hoc stratified bootstrap interval is [0.067, 0.25], with 9 wins, 51 ties, 0 losses.

**Interpretation boundary:** these intervals were calculated after inspecting this exploratory data and are not preregistered confirmatory inference. Best-of-16 costs 16 randomized greedy trajectories versus one deterministic trajectory, so the objective gain cannot be called an efficiency improvement. No wall-clock timing was measured. The important design implication is to compare quantum-informed methods against best-of-budget randomized greedy and strong local-search/prioritized-search baselines under a matched total budget.

## What this changes about the research plan

1. **Random tie-breaking is a necessary paid baseline.** In this small cohort, choosing the best of 16 random-tie trajectories raises average independence ratio relative to a single deterministic greedy run. That gain costs 16 full greedy runs. Any QAOA-informed comparison that reports a win against only one deterministic tie-break is incomplete unless it also reports equal-budget controls.
2. **A small 1-for-2 neighborhood is a useful control but may be too weak.** It makes few gains from the deterministic minimum-degree solution in this cohort. The next classical work should test stronger local search / prioritized search, not treat the failure of this one move as evidence that a quantum signal has value.
3. **The decisive classical baseline is still missing here.** Reproduce or faithfully implement Marino–Kirkpatrick prioritized search from its paper (https://arxiv.org/abs/2003.12293), freeze its implementation/configuration, then compare it on the exact generated graphs. Do not call a simplified imitation an exact reproduction.
4. **Run the experimental treatment on identical graph bytes.** Freeze graph files and SHA-256 digests, and pass exactly the same bytes to each policy. Seeds are convenient for replay, but source bytes plus cryptographic digest should anchor the final experiment.
5. **Separate quality from cost.** Report independence ratio and feasibility, all seed-level outcomes, best-of-budget curves, CPU work, shots, circuit batches, compile/transpile, calibration, queueing and end-to-end elapsed time. No timing is included in this preflight.

## Interpretation boundary

This is a small exploratory cohort, not a pre-registered confirmatory sample. It excludes QAOA scores, fixed angle tables, tensor networks, actual QPU execution and Marino–Kirkpatrick. It records objective quality but no runtime; it cannot establish cost-to-target superiority. The random graph cohort is separate from QOBLIB's named graphs and separate from qReduMIS's kernelization problem setting. There is no SOTA improvement or quantum-advantage claim.
