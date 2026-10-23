# QCOMP-RRG-COHORT-001: size-held-out random-regular MIS workloads

Status: **generator implementation and cohort manifest authored; graph instances not yet generated**

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
- **Same instances across arms:** for each generated graph, compare p=4 hybrid, Marino–Kirkpatrick, minimum-degree greedy and randomized classical controls under paired graph identity and explicit budgets.
- **Hardware vs simulation:** the paper uses tensor-network simulation for larger circuits and notes that some p=3 light cones exceed a 20-qubit device's capacity, so noiseless simulated expectations were substituted for those cases. Such rows must be labelled mixed hardware/simulator, never hardware-only.
- **Generalization:** new sizes and degree-4 workloads test extensions; they do not replace the paper-shaped degree-3 comparison.

### Reusable setup cost changes the economic question

The paper builds reusable local-QAOA expectation information for subgraph topologies and reuses it across graph instances. Our comparison must consequently separate (1) one-time or cohort-fixed table/circuit acquisition cost, (2) amortized per-instance inference cost and (3) total end-to-end deployment cost.

Report the exact number of graphs served by each shared table, the measured or estimated setup expense, marginal cost per graph, and break-even cohort size against the classical baseline. Include measured shots, circuit batches, transpilation/compilation, calibration, queue, classical preprocessing, CPU/GPU compute and end-to-end wall time. Do not charge a reusable setup cost to every graph when it was paid once, but do not hide it either. Re-run at multiple cohort sizes or calculate the break-even point from the fixed and marginal terms. The QOBLIB named-instance track remains separately reported with QOBLIB's own runtime conventions alongside a broader all-in cost view.

## Generator

The dependency-free Rust API generate_random_regular_mis_graph uses a configuration-model stub-pairing sampler with explicit rejection of self-loops and duplicate undirected edges. Successful outputs are canonical sorted simple graphs whose degrees are checked exactly. DIMACS export uses one-based vertex IDs and records the generator version, seed, attempt count and requested degree in a comment.

The generator has a bounded rejection budget. If it cannot obtain a simple graph within that budget, it returns an explicit error. It must not silently relax the requested degree or switch to a different seed. The underlying XorShift64* stream is deterministic and non-cryptographic; reproducibility alone is not proof of adequate sampling quality. A generator-quality audit is required before confirmatory statistical claims.

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

The workload treatment must stay fixed while graph, degree, seed, candidate pool and stopping budget are paired. Freeze all angle parameters, marginal estimation, tie-breaking, sample counts, shot allocation and classical budgets before evaluating the held-out sizes.

## Required outputs

Every result should include the workload ID and graph byte digest; generator version, seed and attempts; solver/configuration version; feasibility verification; MIS cardinality; objective gap when a proven optimum is known; return status; timeouts and failures; and raw-output digest.

Report quality distributions and feasibility rates across each graph/seed cohort, not merely the best run. Record circuit batches and physical shots separately. Include preprocessing, compilation/transpilation, calibration, queue time, classical compute, QPU time and end-to-end wall time. For practical utility, measure probability of hitting a preregistered target and time-to-target at matched total cost, not just QPU execution time.

Graph-generation overhead should be reported separately from solve time. Include it in the deployment objective only when the application genuinely requires on-demand graph generation.

## Release gates before confirmatory results

1. Compile and execute the generator unit tests under a recorded Rust toolchain.
2. Run deterministic replay checks: same seed/configuration must produce identical DIMACS bytes and metadata.
3. Generate all workloads exactly once, independently validate simple d-regular invariants and record SHA-256 byte digest, byte length and attempts.
4. Audit generator statistics over a larger exploratory seed sweep: duplicate graph rates, connected-component behavior, cycle/triangle counts and other relevant graph statistics. Do not call the outputs an unbiased benchmark distribution without this audit.
5. Pin the paper's full methods and angle/configuration artifacts before naming a run a reproduction.
6. Tune only on train, select policy settings on validation, and keep the 2,048/4,096-vertex test outcomes out of all tuning and selector feedback.
7. Preserve exhausted attempts, invalid output, timeout, abstention and infeasible solution outcomes in all reports.

## Validation boundary

This contribution defines a generator API and a cohort manifest; it has not generated or stored the 96 graph files, run a solver, called a simulator, or executed physical hardware. The Rust tests are authored but not compiled or executed in the authoring environment. It provides no SOTA, quantum-utility or quantum-advantage result.
