# QCOMP-RRG-COHORT-001: size-held-out random-regular MIS workloads

Status: **cohort protocol and independent preflight authored; canonical Rust generator is in [PR #8324](https://github.com/Luminous-Dynamics/symthaea/pull/8324); large graph instances not yet generated**

## Why this track exists

An accepted Physical Review Research paper dated 19 October 2026, *Scalable quantum-enhanced greedy algorithm for maximum independent set problems* (DOI 10.1103/ds1n-wsdm), describes QAOA-derived local expectations guiding sequential minimum-degree greedy choices on random-regular graphs. The paper abstract reports a 20-qubit IQM superconducting-device implementation on graphs with thousands of vertices, and comparisons against classical greedy and more sophisticated approximation methods.

QOBLIB is still the public shared benchmark for its named MIS instances; it does not substitute for a separately controlled random-regular cohort. The new track isolates graph distribution and size generalization instead of pooling results that answer different questions. This is a proposed independent evaluation, not a claim to have reproduced that paper.

## Paper-shaped cohort and stronger classical baseline

The public [preprint](https://arxiv.org/abs/2601.21923) reports a study of **3-regular graphs** at N = 50, 100, 200, 500, 1,000, 2,000 and 5,000, with 200 instances per size for its simulation comparisons. It uses fixed QAOA angles derived from tree neighborhoods and feeds local expectations into sequential minimum-degree greedy selection. The paper identifies **Marino–Kirkpatrick prioritized search** ([arXiv:2003.12293](https://arxiv.org/abs/2003.12293)) as the strong classical state-of-the-art baseline. Plain minimum-degree greedy alone is not a credible SOTA comparator.

The JSON therefore contains a separate 1,400-workload, degree-3 seed manifest matching those reported sizes and counts. These are **newly derived seeds**, not the paper authors' exact graphs: an exact public graph/seed list was not located during this pass. This is replication-scale independent sampling, not graph-for-graph reproduction. The original 96-workload degree-3/4, size-held-out manifest remains an extension/generalization track and must not be pooled with the paper-shaped cohort as if they were one distribution.

The larger SOTA campaign is specified separately as 100,000 3-regular instances per size (700,000 total seed-defined workloads). It is a streaming campaign specification, not 700,000 materialized graph files and not a completed experiment. Follow the paper's reported significance scope carefully: pre-register N ≤ 2,000 as the primary comparison, and report N = 5,000 separately as a scaling/sensitivity result.

### Required identity and controls

- **p = 1 control:** the paper derives the p=1 selection rule as equivalent to classical minimum-degree greedy (with randomized tie handling). Require an identity test on a common graph fixture before interpreting p > 1.
- **Fixed angles:** pin the exact regular-tree angle values, depth, mixer/cost conventions and implementation artifact. Merely setting p=4 is not a reproduction.
- **Fixed-instance paired panel:** for each *completed graph byte artifact*, compare p=4 hybrid, the p=1 identity control, minimum-degree greedy, seeded multi-start greedy, and a fixed-instance classical local-search/ILS baseline under identical graph identity and preregistered resource budgets. These are methods that actually consume a completed graph.
- **Marino–Kirkpatrick compatibility:** the original prioritized method is a deferred-decision process. It reveals/pairs graph stubs while constructing the random d-regular graph and makes virtual-node decisions from the partial graph. Its native output is therefore a graph-plus-independent-set sample from the declared distribution, not a result on an arbitrary already-frozen graph. Compare the original method with QGreedy at the **distribution level** using independent samples, matching N, d, graph-distribution/acceptance rules and reporting unpaired uncertainty. Do not assign the same graph-pair key or use paired inference unless identical completed-graph bytes are genuinely supplied to both methods. An input-conditioned adaptation is a separate algorithm, not an exact reproduction of the original.
- **Hardware vs simulation:** the paper uses tensor-network simulation for larger circuits and notes that some p=3 light cones exceed a 20-qubit device's capacity, so noiseless simulated expectations were substituted for those cases. Such rows must be labelled mixed hardware/simulator, never hardware-only.
- **Generalization:** new sizes and degree-4 workloads test extensions; they do not replace the paper-shaped degree-3 comparison.

### Pinned Marino–Kirkpatrick source and licensing boundary

The reference repository is pinned in the companion JSON to commit ae59badc1c07fc27d12088593546e8b98e909c8a (2020-03-27), including Git blob identities for main.cpp, Random.hpp, Header.h, and README.md. The source files include author copyright notices; no explicit open-source license was found in that pinned repository tree. The source is recorded for provenance and review, not copied into Symthaea. Do not vendor or redistribute it without permission. If a faithful executable baseline is needed and permission is unavailable, write an independent implementation from the published paper's algorithm and separately validate it rather than copying the upstream C++.

## Comparison compatibility: paired solvers versus generator-coupled procedures

The original Marino–Kirkpatrick algorithm (arXiv:2003.12293, Sections 2–3) uses deferred decisions: it generates a uniformly sampled simple d-regular graph while choosing labels/virtual nodes and completing edge pairings. Its state includes partially revealed graph structure, so the algorithm is not natively a function that maps an arbitrary completed graph file to an independent set. This is not just a statistical detail: demanding byte-identical graph inputs for that original procedure silently changes the algorithm being tested.

Accordingly, maintain two separate designs:

1. **Fixed-instance solver panel (paired):** freeze each graph's bytes and digest; run QGreedy p=1/p=4, ordinary and multi-start greedy, and a suitable classical local-search/ILS baseline on those same graphs. Pair by graph digest and replicate seed; bootstrap/cluster by graph. This estimates the value of each decision policy conditional on the same instance.
2. **Generator-coupled distribution panel (unpaired):** run the original Marino–Kirkpatrick deferred-decision procedure as specified, preserving its edge-generation/acceptance rules, and compare its independence-ratio distribution against QGreedy over separately sampled graphs from the same claimed random simple d-regular distribution. Match N/d cohorts, verify every emitted graph is simple and d-regular, report unpaired confidence intervals and distribution quantiles, and audit distributional compatibility. The published asymptotic reference r_infinity approximately 0.445330 is a theory/lower-bound line, not a same-instance result.

Do not pair by seed label when the output graphs differ; a shared integer seed is not a shared workload identity. If a completed-graph variant of the prioritized method is implemented, give it a distinct algorithm ID and report that it is a new input-conditioned variant. This separation follows the paper's own deferred-pairing description and protects the comparison from false precision.

## Reusable setup cost changes the economic question

The paper builds reusable local-QAOA expectation information for subgraph topologies and reuses it across graph instances. Our comparison must consequently separate (1) one-time or cohort-fixed table/circuit acquisition cost, (2) amortized per-instance inference cost and (3) total end-to-end deployment cost.

Report the exact number of graphs served by each shared table, the measured or estimated setup expense, marginal cost per graph, and break-even cohort size against the classical baseline. Include measured shots, circuit batches, transpilation/compilation, calibration, queue, classical preprocessing, CPU/GPU compute and end-to-end wall time. Do not charge a reusable setup cost to every graph when it was paid once, but do not hide it either. Re-run at multiple cohort sizes or calculate the break-even point from the fixed and marginal terms. The QOBLIB named-instance track remains separately reported with QOBLIB's own runtime conventions alongside a broader all-in cost view.

## Generator

The canonical dependency-free Rust API in [PR #8324](https://github.com/Luminous-Dynamics/symthaea/pull/8324) is `generate_mis_random_regular(MisRandomRegularConfig)`. It uses configuration-model stub pairing, rejects self-loops and duplicate edges, and returns a canonical `MisGraph` with exact degree validation. The `qcomp_rrg_generate` CLI serializes one-based DIMACS and records the generator version, seed and rejection-attempt count.

The generator has a bounded rejection budget. If it cannot obtain a simple graph within that budget, it returns an explicit error. It must not silently relax the requested degree or switch to a different seed. It caps each call at 100,000 vertices and 1,000,000 stubs before allocation; these are far above the paper-shaped maximum (N=5,000, degree 3, 15,000 stubs), but protect the public API from accidental huge allocations. Larger campaigns should change those ceilings only through an explicit reviewed setting with memory evidence. The default rejection budget is 100,000 complete pairings; a caller may configure a smaller cap. The underlying XorShift64* stream is deterministic and non-cryptographic; reproducibility alone is not proof of adequate sampling quality. A generator-quality audit is required before confirmatory statistical claims.

## Frozen 96-workload manifest

QCOMP-RRG-COHORT-001.json defines 96 deterministic workload identities:
- **Train:** 128, 256 and 512 vertices.
- **Validation:** 1,024 vertices.
- **Test:** 2,048 and 4,096 vertices.
- Both degree 3 and degree 4, with eight recorded seeds for each size-degree cell.

The size-held-out partition tests extrapolation within this regular-graph family. It does not establish out-of-distribution generalization to unit-disk graphs, social networks or arbitrary QOBLIB instances. Each row lists the derived seed and maximum attempt count before any graph generation is performed.

## Experimental arms

1. Deterministic minimum-residual-degree greedy baseline.
2. Strong classical local search / iterated local search, with per-run and aggregate budgets disclosed.
3. The published QAOA-informed greedy setup, once the complete paper and angle/measurement configuration are pinned. The abstract specifies precomputed angles from depth-p QAOA on regular trees and reports results at p=4, but that alone is not enough to claim a reproduction.
4. Seeded randomized greedy and local-improvement controls.
5. qReduMIS-style quantum-informed reduction as a distinct algorithm track; do not conflate its mechanism with a QAOA-marginal-informed greedy policy.
6. Physical QPU, ideal-circuit simulation, tensor-network simulation and classical replay as distinct evidence classes.

For the fixed-instance solver panel, graph digest, degree, candidate pool, stopping budget and replicate schedule must be identical across arms; pair by graph digest, not seed labels alone. The generator-coupled Marino–Kirkpatrick comparison is separately unpaired at the distribution level. Freeze all QAOA angle parameters, marginal estimation, tie-breaking, sample counts, shot allocation and classical budgets before evaluating the held-out sizes.

## Required outputs

Every result should include the workload ID and graph byte digest; generator version, seed and attempts; solver/configuration version; feasibility verification; MIS cardinality; objective gap when a proven optimum is known; return status; timeouts and failures; and raw-output digest.

Report quality distributions and feasibility rates across each graph/seed cohort, not merely the best run. Record circuit batches and physical shots separately. Include preprocessing, compilation/transpilation, calibration, queue time, classical compute, QPU time and end-to-end wall time. For practical utility, measure probability of hitting a preregistered target and time-to-target at matched total cost, not just QPU execution time.

Graph-generation overhead should be reported separately from solve time. Include it in the deployment objective only when the application genuinely requires on-demand graph generation.

## Generator smoke result

A bounded independent JavaScript preflight created the first 20 recorded seeds at each of the seven paper-shaped sizes (140 graphs total). All were simple degree-3 graphs with exactly 3N/2 edges, all connected, all unique by exact canonical edge-list comparison within each size, and all generated in 1–31 attempts (mean 6.686). Full per-seed records and component/triangle summaries are in [QCOMP-RRG-PREFLIGHT-001.json](QCOMP-RRG-PREFLIGHT-001.json).

A second independent JavaScript translation generated 100,000 six-vertex degree-three graphs to check a small-N reference distribution. Exhaustive enumeration over all 32,768 labelled simple graphs on six vertices retained exactly 70 cubic graphs: 10 K3,3 (no triangles) and 60 triangular prisms (with triangles). The observed generated sample had 14,277 K3,3 and 85,723 prism graphs, matching the expected 1/7 versus 6/7 split closely; mean rejection attempts were 10.593 versus an exact simple-pairing acceptance probability of about 0.09478. See the updated [QCOMP-RRG-PREFLIGHT-001.json](QCOMP-RRG-PREFLIGHT-001.json) and [interpretation](QCOMP-RRG-PREFLIGHT-001.md).

This is still a JavaScript translation, not execution of the Rust generator or distributional certification. It detects gross small-N bias only; it does not prove RNG quality, absence of subtle sampling bias at larger sizes, Rust byte parity, or replication of the paper's samples. A deterministic Rust unit test now samples 5,000 seeds at N=6, degree 3, with a loose frequency tolerance to catch major regressions, but it has not run. Before release, the Rust implementation must regenerate the preflight seeds and compare exact outputs/metadata; then record SHA-256 for each serialized DIMACS file.

## Release gates before confirmatory results

1. Compile and execute the generator unit tests under a recorded Rust toolchain.
2. Run deterministic replay checks: same seed/configuration must produce identical DIMACS bytes and metadata.
3. Generate all workloads exactly once, independently validate simple d-regular invariants and record SHA-256 byte digest, byte length and attempts.
4. Audit generator statistics over a larger exploratory seed sweep: duplicate graph rates, connected-component behavior, cycle/triangle counts and other relevant graph statistics. Do not call the outputs an unbiased benchmark distribution without this audit.
5. Pin the paper's full methods and angle/configuration artifacts before naming a run a reproduction.
6. For the extension manifest, tune only on its train partition, select policy settings on validation, and keep the 2,048/4,096-vertex test outcomes out of tuning and selector feedback.
7. For the 1,400 paper-shaped workload manifest, reuse each graph ID across p=1/2/3/4 arms; do not call these graph-for-graph paper reproduction until the authors' graph instances/seed list and exact settings are pinned.
8. For the 700,000-instance classical reference campaign, retain the method-specific generator/seed stream for the original deferred-decision Marino–Kirkpatrick algorithm and compare its distribution-level results against the separately generated fixed-graph QGreedy cohort. Do not call these outcomes paired unless graph-byte equality is actually established. For the fixed-instance solver panel, stream the exact same pinned graph digests through all completed-graph arms and preserve paired outcomes. In both panels, pre-register stopping/statistical procedures and record seed ranges/failures so interrupted execution can resume without resampling earlier cases.
9. Preserve exhausted attempts, invalid output, timeout, abstention and infeasible solution outcomes in all reports.

## Validation boundary

This contribution defines a generator API, a 96-workload extension, a 1,400-workload paper-shaped seed manifest, and a 700,000-instance streaming SOTA campaign specification. No complete cohort graph files have been materialized or checked in; a separate 140-graph JavaScript smoke preflight was run and recorded, but the Rust generator has not been compiled/executed. No solver was run, no simulator was called, and no physical hardware was executed. The Rust tests are authored but not compiled or executed in the authoring environment. The paper-shaped seeds use new workload identities rather than authors' original graphs, so this is not graph-for-graph reproduction. It provides no SOTA, quantum-utility or quantum-advantage result.
