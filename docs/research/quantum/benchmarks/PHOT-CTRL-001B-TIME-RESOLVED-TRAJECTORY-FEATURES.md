# PHOT-CTRL-001B: time-resolved trajectory features for MIS search

Status: implementation draft; Rust build/tests and held-out solver comparisons not yet verified.

Related:
- Parent issue: https://github.com/Luminous-Dynamics/symthaea/issues/8689
- PR: https://github.com/Luminous-Dynamics/symthaea/pull/8690
- Mapping contract: https://github.com/Luminous-Dynamics/symthaea/pull/8618
- XY-to-Ising dynamics reference: https://github.com/Luminous-Dynamics/symthaea/pull/8629
- QOBLIB: https://zib-aopt.github.io/QOBLIB/

## Motivation and novelty boundary

Kim & Yamamoto, Scientific Reports 16, 10396 (2026), report a classical numerical study in which XY-to-Ising continuation and adaptive returns to XY improve TTS on Wishart-planted Ising problems. The study is a motivation for investigating spin-flip and stabilization history; it is not evidence that those time-resolved features improve MIS solving or that Symthaea has measured hardware performance:
https://doi.org/10.1038/s41598-026-41315-7

qReduMIS already demonstrates the broad quantum-informed MIS reduction pattern: classical kernelization combined with quantum-derived information about vertices likely or unlikely to occur in large independent sets, evaluated on neutral-atom hardware. That architecture is prior art; this work does not claim it as new:
https://doi.org/10.1103/nh3b-1wv5

The narrower hypothesis is that time-resolved sign-change activity, last-flip time, observed stability, and terminal disagreement from XY-to-Ising trajectories may provide information beyond final-spin inclusion frequencies and graph structure. Whether they add value is unknown and must be tested against controls.

QOBLIB's 2026 publication sets a relevant norm: application benchmarking must compare model-independent instances with suitable methods and enough detail to support meaningful conclusions; algorithm and hardware benchmarking answer different questions:
https://doi.org/10.1038/s43588-026-00991-1

## Contract

The new Rust module `trajectory_history.rs` accepts per-run samples of in-phase quadratures `x_i(t)`, normalized to the planned run interval.

- Time must be finite, within [0, 1], and strictly increasing.
- The fixed absolute zero tolerance is part of the summary. Values within it are ambiguous for that vertex and are not assigned a spin.
- A spin flip is counted only across an adjacent sampled pair whose two endpoint signs are both observed and differ. The module does not bridge ambiguous samples.
- Failed and timed-out attempts remain in the global denominator and may contribute partial-history activity statistics.
- Terminal-frequency statistics use completed-run final samples only, per vertex and only when the final x value is outside the zero tolerance.
- Stability summaries use completed histories and stop at the last valid sign observation. An ambiguous terminal sample cannot be counted as evidence of stability through the rest of the run.
- Late activity means observed sign flips at normalized planned-run time >= 0.75; this clock is not rescaled for early timeouts.
- Every ordering result is a permutation containing all vertices. It is a suggestion for variable ordering only. The summary cannot fix vertices, prune branches, issue reduction certificates, or claim optimality.

The current contract does not model optical measurement noise, infer physical energy, prove statistical independence, or establish any calibration between spin frequency and optimal-set membership.

## Hypotheses to test

H1. The mean late-flip rate adds held-out predictive value beyond graph structure and final-spin statistics for identifying useful branching / proof-query orders.

H2. The time of the final observed flip and the stability suffix add predictive value beyond total flip count alone.

H3. A controller that uses these features can reduce end-to-end time-to-target at fixed verified MIS quality, after charging for trajectory generation and feature extraction.

H4. Any improvement transfers across at least one held-out graph family, rather than reflecting only the graph generator used for tuning.

Each hypothesis can fail independently. Feature importance or predictive accuracy alone is not an optimization advantage; the downstream decision metric is paired all-in verified performance.

## Related trajectory-diagnostics literature (checked 2026-10-28)

Hellstern, Schacht, and Zaefferer, arXiv:2610.35126 (posted 26 October 2026), compare branch-and-cut, simulated annealing, parallel tempering, DMRG-based tensor networks, and fixed-schedule LR-QAOA. They report that some trajectory/structure diagnostics explain aspects of tensor-network behavior but do not yield a validated solver-selection rule; their tested LR-QAOA runs do not improve solution quality over the classical comparators. Their controlled optimum-placement and Wishart experiments also show that solver ranking depends on instance construction and the target metric. This is a preprint and not a result on our MIS trajectory features, but it is a direct caution against presenting an explanatory time-series feature as a useful controller before downstream validation: https://arxiv.org/abs/2610.35126

QOBLIB's June 2026 paper distinguishes application, algorithm, and system benchmarking and explicitly states that the released baseline runs are not claimed to be the best results achievable by classical computers. Therefore PHOT-CTRL-001B must use QOBLIB for public reproducibility while separately comparing against strong specialized MIS methods and a matched classical algorithm-selection baseline: https://doi.org/10.1038/s43588-026-00991-1

## Feature identifiability positive control

The unit-test corpus includes two synthetic trajectory cohorts with identical terminal selection frequency (the only vertex is selected in every completed run) but different dynamics:
- early-flip cohort: the spin changes at normalized time 0.25 and stays selected;
- late-flip cohort: the spin changes at normalized time 1.0.

The expected final-spin frequency is 1.0 for both cohorts, so a terminal-frequency-only prior cannot distinguish them. The expected mean late-flip rate and last-flip time differ. This is a *feature identifiability control*: it demonstrates that the extractor can represent temporal distinctions not encoded by final states. It does not demonstrate that these distinctions predict MIS optimum membership, reduce solver time, or transfer to real trajectories.

## Experiment design

### Frozen cohorts

1. Tiny all-labelled graphs for exhaustive optimum and algorithm-invariance tests.
2. Public QOBLIB MIS instances with repository revision, source paths, and instance digests frozen.
3. Separate random-regular, dense, unit-disk, and documented real-world graph cohorts, each with stable generation/source and separate reporting.
4. Held-out confirmation by graph family, not merely fresh random seeds from the same generator.

### Matched arms

- Strong classical MIS baseline with no trajectory calls.
- Classical baseline plus degree-only ordering.
- Classical baseline plus final-spin frequency ordering.
- Classical baseline plus time-resolved features: late activity, recent flip, and terminal disagreement.
- Shuffled-feature and seeded-random ordering controls.
- Trajectory simulator/hardware proposal plus the same downstream classical controller.
- A classical-only controller with equal tuning and budget to avoid attributing generic meta-selection value to the photonic-inspired signal.

Where license/build compatibility allows, include specialist MIS algorithms such as KaMIS/ReduMIS and a solver with documented best-known status. Do not use greedy-only or generic MIP-only comparisons as an SOTA claim.

### Required output

For every original instance and every run, preserve:
- graph and configuration digests;
- solver and simulator source revisions, seeds, and declared budgets;
- all trajectory frames or a content-addressed raw artifact, including partial, ambiguous, failed, and timed-out histories;
- per-vertex statistics and exact feature-policy/version digest;
- pre-repair and post-repair candidates separately;
- independent original-graph feasibility verification, objective, bound, and optimality/proof status;
- wall time for every stage, memory, host/accelerator utilization, energy measured or unknown, and failure/retry counts;
- graph-level paired result rows and uncertainty analysis.

### Primary endpoint

Predeclare one primary endpoint: paired all-in time to a fixed verified cardinality/approximation target on held-out instances. Report probability of meeting that target before budget expiry and quality-over-time curves as secondary endpoints. For exact-solver ordering, also report search nodes and bound evaluations, but never substitute those counts for end-to-end wall time.

Repeated seeds on the same graph are nested trials, not independent graph instances. Use graph-level pairing, report ties/wins/losses and effect sizes, and freeze the confirmation set before tuning.

## Go/no-go

Continue only if at least one time-resolved feature provides reproducible held-out benefit over the strongest matched classical / final-spin / shuffled controls after all trajectory and controller costs are included. If the advantage disappears after adding the feature generation cost, or fails on held-out families, stop tuning this policy family and publish the negative result.

## Explicit non-claims

This is a testable feature contract and research protocol, not a measured result. No Rust compilation or tests, QOBLIB experiment, physical optical run, speedup, SOTA result, energy advantage, or quantum advantage is claimed until separate evidence supports it.
