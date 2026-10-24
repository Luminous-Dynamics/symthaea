# QCOMP-RRG-001: source-reproducible random-regular MIS track

Status: **seeded graph generator authored; no solver comparison result claimed**

## Why a separate track exists

Wybo et al., A scalable quantum-enhanced greedy algorithm for maximum independent set problems, was accepted by Physical Review Research on 19 October 2026 (https://doi.org/10.1103/ds1n-wsdm; preprint https://arxiv.org/abs/2601.21923). It combines minimal-degree greedy MIS with fixed-angle QAOA expectation values derived from regular-tree models. The accepted-paper abstract reports experiments on a 20-qubit IQM superconducting device and tensor-network studies beyond device size.

The preprint compares against random/minimal-degree greedy and Marino–Kirkpatrick linear-time prioritized search. It describes graph sizes N = 50, 100, 200, 500, 1,000, 2,000 and 5,000, with 200 graph instances per size for QGreedy analyses (a 100-graph subset for one p=3 hardware condition) and reports improvement over the linear-prioritized baseline for p=4 through N <= 2,000 in a larger 100,000-instance-per-size comparison. These are paper-reported results, not a Symthaea reproduction.

This is a different experimental track from qReduMIS, which uses a quantum-informed signal to alter classical kernelization, and from QOBLIB's fixed named graph instances. Do not pool their outcomes as one undifferentiated leaderboard.

## Implementation added

- mis_random_regular.rs: deterministic seeded simple d-regular graph generation using the configuration model, with whole-pairing rejection for self-loops or duplicate edges.
- qcomp_rrg_generate binary: emits canonical DIMACS edge-format graph bytes to stdout, with configuration and generator comments embedded.
- The generator reports seed and rejection attempts, rejects invalid degree/parity and a zero retry budget, supports degree zero for fixtures, and caps retries rather than silently returning a non-regular graph. Before allocation, it also fails closed above 100,000 vertices or 1,000,000 stubs; the published-paper-shaped maximum (N=5,000, degree 3) is 15,000 stubs. The default rejection budget is 100,000 full pairings, with a lower limit configurable per CLI/API call.
- Unit tests cover degree and edge counts, deterministic replay, invalid requests, vertex/stub allocation ceilings and the coarse small-N distribution check. Retry-budget exhaustion errors preserve vertex count, degree, seed and attempt limit for a useful failure receipt. The CLI emits structured JSON failure metadata on stderr for parse/input/generation failures and keeps stdout free of graph records unless a valid graph was produced.

The underlying XorShift64* generator is seeded and non-cryptographic. The mathematical statement that conditioning a uniform configuration-model pairing on simplicity is uniform over simple d-regular graphs assumes a uniform stub permutation; this implementation offers deterministic pseudo-random replay, not a formal randomness guarantee or proof about finite PRNG output. The graph file itself, rather than the seed alone, is the experiment's shared input artifact.

## Reproduce a generated graph locally

After executable validation is available:

    cargo run -p symthaea-quantum-comp --bin qcomp_rrg_generate -- 1000 3 12345 > rrg-n1000-d3-seed12345.dimacs
    sha256sum rrg-n1000-d3-seed12345.dimacs

For alternate platforms use an equivalent SHA-256 utility. Store the exact file, its SHA-256 digest, generator version, seed, degree, vertex count, accepted attempt count, and toolchain revision in the experiment manifest. Pass the identical file to every baseline and retain the generated DIMACS bytes as the canonical cohort artifact. The generator's DIMACS comments contain seed and attempt metadata, but the cryptographic digest must be recorded separately.

## Proposed preregistered experiment

### Workloads and replication

- Degree: d = 3 for the primary comparison, so the method matches the published fixed-angle QGreedy track.
- Sizes: N = 50, 100, 200, 500, 1,000, 2,000, 5,000.
- Pilot: 20 independently generated instances per size to validate generator, solver integration and runtime/cost accounting.
- Confirmatory target: 200 instances per size, subject to a preregistered power/precision calculation and available resources; report any deviation from this target.
- A separate large-instance comparison can target 100,000 instances per size only if its purpose, statistical endpoint and resource budget are fixed before running. Do not invoke this sample count just because it appears in the paper; justify it with the variance and target precision.
- Use a published deterministic seed derivation or frozen seed list. Never regenerate a graph independently in each solver arm; generate once, hash, and share the exact graph.
- Keep a held-out seed set for confirmation. This helps prevent instance leakage, but does not turn public algorithm code/angles into a blind test or establish generalization to other graph families.

### Two comparison tracks: same graph versus same distribution

Do not force every method into one paired table if its algorithmic interface differs.

**Track A — fixed-graph decision quality.** Generate each completed DIMACS graph once, hash it, and pass the exact bytes to every method that accepts an existing graph. Use paired per-graph outcomes for deterministic/randomized min-degree greedy, local search, fixed-graph classical heuristics, and any QGreedy implementation that uses that graph as input. The randomized-greedy restart-budget curve already illustrates why budgets must be matched.

**Track B — distribution-level deferred-decision methods.** Marino–Kirkpatrick's 2020 paper develops a deferred-decision algorithm that builds a random regular graph while deciding the independent set, using virtual-node contractions. Do not silently assume its paper-native algorithm is a fixed-input solver. First establish from the exact implementation/protocol whether a faithful fixed-graph version exists. If the method must co-generate its graph, retain the graph it generated and independently verify its output; compare graph-level objective-ratio distributions against independently sampled completed-graph methods using the unpaired inference path, not by matching seed numbers. A fixed-graph adaptation may enter Track A only with an explicit algorithm identity and validation showing how the deferred-decision choices changed.

The primary estimand, uncertainty method, and resource boundary must be specified separately for the two tracks. A fixed-graph paired improvement answers a different question from an across-distribution mean ratio comparison. Matching seed integers never creates a pair unless the actual graph identity is the same.

### Baseline arms

1. Uniform-random tie-breaking minimal-degree greedy (paper-aligned baseline).
2. Deterministic minimal-degree greedy with frozen tie-breaking.
3. Best-of-budget independent greedy restarts, charging the full restart budget.
4. Marino–Kirkpatrick linear-time prioritized search, the paper's named strong classical baseline for random regular graphs.
5. At least one additional strong eligible classical heuristic/solver selected before confirmatory runs; publish build/version and tune it under an equal disclosed budget.
6. Fixed-angle QAOA-informed greedy, initially replay/simulator only if the required angle tables and exact local expectations are available; label any independent implementation as a replication, not an exact reproduction unless artifacts match.
7. Noisy/physical-QPU QGreedy as a separate evidence arm only when raw measurement/circuit/backend receipts and the full workflow resource account are recorded.

Controls:
- Randomized vertex-score permutation preserving the score multiset.
- A constant or random score signal.
- p=1 equivalence/control expected by the method's design where applicable.
- Classical exact MIS on tiny graphs only as an oracle, not as a scalable competitor.
- For noisy expectations, use matched ideal/noisy and shot-count controls, and preserve backend calibration/time metadata.

### Primary metrics

- Independent-set cardinality and independence ratio |I| / N, checked by the verifier.
- Difference in paired independence ratio per graph, median and mean paired effect with suitable intervals, win/tie/loss counts, and confidence-aware uncertainty.
- Time-to-target/success probability under an agreed target, across multiple runs where the policy is stochastic.
- Runtime and resource accounting: QOBLIB-compatible runtime fields (average workflow wall time; CPU, GPU, QPU and other hardware; compilation and queue excluded from those fields) plus a separate all-in operational account that includes compile/transpile, calibration, queue, host orchestration, retries and full elapsed time.
- For quantum arms, show shots, unique circuits, batches, circuit depth/width after transpilation, local expectation estimation error, and the classical preprocessing/continuation cost.
- Performance by N and selected graph-structural strata, not just a pooled average.

### Decision gates

1. Generator correctness: every vertex has degree d, edges are unique, no loops exist, edge count equals N*d/2, and graph file replay from a frozen seed is bit-identical.
2. Parser/checker correctness: generated DIMACS is parsed by Symthaea and the official QOBLIB checker where applicable.
3. Algorithm correctness: every solution independently verified.
4. Baseline adequacy: compare with Marino–Kirkpatrick or clearly state why it could not be reproduced. Greedy-only wins are not enough for a state-of-the-art claim.
5. Evidence correctness: simulation, tensor-network calculation, circuit export and actual hardware execution are never blended into one score.
6. Statistical correctness: use graph-instance pairing only for identical completed graph bytes. For independently generated/deferred-decision graph cohorts use unpaired distribution-level inference; nested repetitions are not counted as additional independent graphs. Freeze the analysis plan, cohort, and multiplicity family before confirmatory runs.
7. Practical-advantage claim: demonstrate an improved final solution quality or lower all-in cost to the same target under matched conditions. Matching a paper's approximation ratio or beating one implementation is not, by itself, quantum advantage.

## Generator-quality regression

The canonical generator now includes a deterministic Rust test over 5,000 seeds at N=6, degree 3. It checks that the no-triangle K3,3 class occurs within ±0.04 of the exact 1/7 reference. The exact class counts were independently derived by enumerating all 32,768 labelled simple graphs on six vertices. A separate JavaScript translation of the generation algorithm sampled 100,000 graphs and observed 14,277 K3,3 versus 14,286 expected, but this is a small-N gross-bias diagnostic, not a proof of generator quality. The Rust test remains uncompiled/unexecuted because the authoring environment lacks cargo/rustc.

## Pinned Marino–Kirkpatrick classical reference (2026-10-24)

The source search found the author-maintained implementation linked by the 2020 paper: [RaffaeleMarino's repository](https://github.com/RaffaeleMarino/Large-Independent-Set-on-random-d-regular-graphs-with-small-and-fixed-connectivity-d), pinned at commit `ae59badc1c07fc27d12088593546e8b98e909c8a`. Its C++ tree contains `main.cpp`, `Random.hpp`, `Header.h`, and `README.md`; its root has no license file and GitHub reports no license metadata. The exact file-object IDs and sizes are recorded in [QCOMP-MK-REFERENCE-001.json](QCOMP-MK-REFERENCE-001.json).

**Do not vendor or redistribute the upstream code without clarified reuse terms.** The pinned source is a reproducible source reference, not a Symthaea implementation or a completed reproduction. If permission cannot be established, implement a clean-room Rust reimplementation directly from the published algorithm description, then validate it against independently generated small fixtures and declared paper-level invariants before it is treated as a reference baseline.

A crucial methodology distinction: Marino–Kirkpatrick's deferred-decision approach co-generates the random regular graph and its independent-set labels using virtual nodes and priority operations. Do not silently treat it as a fixed-input solver. Maintain two tracks: (A) paired, fixed-graph comparisons for policies that accept the same completed DIMACS bytes; (B) distribution-level comparisons for the native co-generated deferred-decision method, retaining every generated graph and independently verifying its output. Matching seed integers does not create a paired observation unless the completed graph bytes are identical.

The 2026 QAOA-enhanced greedy paper explicitly compares against this stronger prioritized baseline, reporting statistically significant p=4 gains on its considered sizes up to N ≤ 2000 while also noting that finite-size simulations had not exceeded the paper's asymptotic 0.445330 value. Those are external paper claims, not Symthaea measurements. A minimal-degree greedy win alone is therefore insufficient for an SOTA claim on the random-3-regular track.

## Classical restart-budget preflight

A post-hoc quality-versus-budget analysis of the 70 graph / 16-restart JavaScript preflight is now recorded in [QCOMP-RRG-BUDGET-CURVE-001](QCOMP-RRG-BUDGET-CURVE-001.md), with machine-readable output in [the JSON record](QCOMP-RRG-BUDGET-CURVE-001.json). It computes the expected best objective over every size-k subset of the 16 observed restart results for k = 1, 2, 4, 8, 16, then uses a graph-level stratified bootstrap that preserves each N stratum's sample size.

On this exploratory cohort, randomized tie-break expected-best deltas against one deterministic greedy run are 0.125, 0.688, 1.120, 1.467, and 1.771 vertices for budgets 1, 2, 4, 8, and 16. The corresponding 95% stratified graph-bootstrap intervals are approximately [-0.171, 0.415], [0.387, 0.980], [0.822, 1.413], [1.169, 1.767], and [1.457, 2.086]. These are post-hoc descriptive estimates on the finite preflight data, not controlled runtime comparisons. They show why best-of-16 should never be presented as a one-run algorithmic gain.

This is still JavaScript-reference evidence, not Rust execution or a quantum comparison. No runtime was recorded, so restart count is only a budget proxy. Confirmatory work still needs the Rust generator/runner to reproduce the frozen inputs and must compare against Marino–Kirkpatrick and a strong eligible classical heuristic under a disclosed equal all-in budget.

## Validation boundary

The generator, DIMACS export binary, structured failure receipt path, and Rust unit tests are authored in the verifier workstream. The current RRG generator now resets each rejected pairing attempt to the canonical stub list, matching the reference seed-stream contract; it also has a 70-vector regression test covering each frozen instance's accepted-attempt count, degree, edge count and canonical FNV-1a fingerprint. The fixture set was reconstructed with an independent JavaScript implementation and all 70 attempt/hash pairs matched the recorded reference values. This is an independent cross-language algorithm check, **not execution of the Rust implementation**: cargo/rustc are unavailable here, so these Rust tests remain uncompiled and unrun. There is no generated Rust artifact, measured runtime, hardware execution, paper reproduction, SOTA result, or quantum-advantage claim in this contribution.
