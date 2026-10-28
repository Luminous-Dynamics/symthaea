# PHOT-CTRL-003A: family-held-out ablation protocol for trajectory-guided MIS

**Status:** pre-registration template; not evidence of an optimization improvement.  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8753  
**Related implementation drafts:** [trajectory features PR #8719](https://github.com/Luminous-Dynamics/symthaea/pull/8719), [adaptive acquisition PR #8727](https://github.com/Luminous-Dynamics/symthaea/pull/8727), [matched evaluator PR #8735](https://github.com/Luminous-Dynamics/symthaea/pull/8735).

## 1. Falsifiable question

Does the addition of time-resolved XY-to-Ising trajectory features improve *verified time-to-target* for maximum independent set (MIS), beyond final-spin frequencies and strong classical graph methods, on graph families not used to tune the method, after all acquisition and downstream costs are charged?

This protocol tests incremental value. It does not assume that trajectory features are predictive, that adaptive acquisition saves time, or that the XY-to-Ising approach is competitive with the best available MIS solvers.

## 2. Freeze before confirmation runs

Create and version a completed copy of `PHOT-CTRL-003A-PREREGISTRATION-TEMPLATE.json` before opening any confirmation result. Every required field marked `null` or `TBD` must be replaced, or explicitly marked not applicable with a reason.

Freeze and hash:
- QOBLIB release/commit, problem category, instance identifiers, source URLs, canonical graph encoding, graph digests, and any supplemental instance generators.
- Family-based development / validation / confirmation splits. Keep generator families out of confirmation where feasible; also publish a separate within-family instance split. Never tune against confirmation results.
- Solver source revisions, build flags, container/Nix derivations, dependency lockfiles, hyperparameters, hardware, CPU/GPU/QPU model, thread count, and resource limits.
- Independent verifier source digest and exact acceptance rule; target cardinality for every instance; timeout and per-arm compute budget; tuning budget; random seeds; acquisition schedule; data-transfer boundaries; energy measurement scope.
- One primary estimator, one primary comparator, the non-inferiority margin, the uncertainty method, and the minimum effect size required for an engineering-success claim.

No interim confirmation results may change arm definitions, targets, thresholds, or stopping rules. Corrections that alter outcomes require a new protocol digest and a fresh confirmation set.

## 3. Instances and targets

### Public, reproducible instance sets

1. Start with maximum-independent-set instances from a pinned QOBLIB revision, recording each canonical input digest. Submit feasible solutions and useful negative results using QOBLIB's current contribution format.
2. Add supplemental DIMACS/BHOSLIB or generated instances only with provenance, license review, exact generator revision, seed, parameters, and digest. Report each source separately.
3. Include easy calibration instances and a tractable subset with independently established optimums, but do not let them substitute for difficult confirmation instances.
4. Publish the complete instance manifest and all inclusion/exclusion rules. Do not silently drop an instance because it is unfavorable or difficult.

### Target tiers

- **Tier A — proven optimum:** target is the optimum independently established by an exact method or accepted proof.
- **Tier B — frozen best-known target:** target is a cardinality fixed from a cited, digest-pinned reference at protocol freeze. Label attainment as reaching the frozen target, not proving optimality.
- **Tier C — prespecified application threshold:** target is set before runs from the application requirement. Report its rationale.

For every tier, only an independent verifier can establish that the reported vertex set is a conflict-free independent set and meets the frozen cardinality. A larger feasible solution may update best-known information after the trial, but must not retroactively redefine that trial's target.

## 4. Required experiment arms

All arms use the same original graph, target, hardware allocation policy, solver budget, tuning allowance, independent verifier, and paired trial schedule. A trajectory arm must pay for trajectory generation, acquisition, feature extraction, transfer, retries, and any repair it requires.

| Arm | Method | What it controls |
|---|---|---|
| A | Strong eligible specialist MIS solver (KaMIS/ReduMIS or a demonstrably stronger suitable solver) | Primary classical comparator |
| B | Classical solver portfolio / selector, trained only on development families | Tests whether trajectory guidance beats algorithm selection |
| C | Strong graph-only ordering, using the selected baseline's native order or a frozen degree/core order | Graph structure without trajectory data |
| D | Final-spin inclusion-frequency ordering | Whether time resolution adds value beyond the simpler proposed feature |
| E | Time-resolved trajectory feature ordering | Treatment arm |
| F | Shuffled trajectory feature values, with deterministic published shuffle seeds | Feature-specific negative control |
| G | Seeded-random ordering | Simple negative control |
| H | Fixed-K trajectory acquisition | Control for adaptive acquisition |
| I | Adaptive acquisition stopping gate | Treatment for acquisition policy; compare at the same total budget |
| J | Exact solver on tractable instances | Optimality and gap calibration |
| K | Oracle-feature analysis ceiling, if used | Diagnostic only; never an operational competitor or a deployable result |

Record all arms even if a solver is unavailable or ineligible; mark it unavailable with the reason instead of quietly selecting a weaker comparator. Record solver license/availability and the basis for calling a method the strongest eligible competitor at the freeze date.

The classical portfolio is selected and tuned only on development families. The frozen portfolio, not the best-performing competitor selected after inspecting confirmation results, is the comparison champion. Report specialist baseline comparisons too.

### Baseline inventory checked 2026-10-28

The public [QOBLIB Maximum Independent Set page](https://zib-aopt.github.io/QOBLIB/problem/07/) currently lists 50 unweighted MIS instances spanning 17–4,000 vertices, with 39 marked optimally solved. It lists classical submissions including **Quicopt** (50 instances), **ParallelILS** (50), **MIP** (50), and **Abs2** (50), alongside quantum-simulator and quantum-hardware submissions. Treat these as concrete candidate baselines to audit—not as proof that any one is the strongest eligible solver.

Before confirmation runs, record for every selected candidate its source revision or executable digest, license/availability, graph representation, preprocessing and postprocessing, hardware, timeout, and configuration. Reproduce results where possible. If a published submission's implementation or settings cannot be recovered, explicitly record the limitation and compare the remaining eligible specialist baselines; do not silently cite its leaderboard score as a matched runtime comparison. QOBLIB reports both total runtime and time-to-solution, and its coverage differs by method/metric, so compare only complete, semantically compatible measurements.

The catalog snapshot is now tied to QOBLIB commit [`1e0e7c523cd9707dd541cc9e5a3086125d4da3c7`](https://github.com/ZIB-AOPT/QOBLIB/tree/1e0e7c523cd9707dd541cc9e5a3086125d4da3c7/07-independentset/instances), whose latest commit date is 19 October 2026. It records 50 graph paths, Git blob SHA-1s and byte sizes, plus the pinned DIMACS format spec and CC-BY-4.0 dataset license. **Git blob SHA-1 is not our SHA-256 evidence digest**: raw source SHA-256 and canonical graph SHA-256 remain null until a loader verifies the actual bytes. The preregistration template remains `template-not-preregistered` until those SHA-256 values, family splits, targets, solver revisions and budgets are frozen.

## 5. Pairing, repetitions, and scheduling

- Use at least **10 independent stochastic runs per arm and instance** where feasible, with all seeds published. This exceeds QOBLIB's minimum of five and follows its recommendation of ten or more.
- Pair attempts by an explicit `pair_id`, graph digest, target, budget, protocol digest, verifier digest, and seed identity. A matching seed label does not prove that two different solvers use equivalent random streams; do not claim common-random-number variance reduction unless the streams are actually coupled.
- Randomize or interleave arm order to reduce machine-load and thermal-order bias. Pin CPU affinity and thread counts when practical. Record queue delay separately from process runtime.
- Every attempt gets a terminal outcome: verified target hit, feasible below target, infeasible, verifier failure, timeout, crash, abort, or missing output. Never drop failures from the denominator.
- Treat instances and held-out graph families as the generalization units. Seeds repeat on the same graph and are not independent graph examples. Report per-family effects; use a family-clustered paired bootstrap or another predeclared suitable method. If too few independent families exist to estimate family-level uncertainty, say the evidence is insufficient for a cross-family claim.

## 6. Metrics and cost boundaries

### Primary outcome

Use a **failure-inclusive restricted mean of directly measured deployment time-to-verified-target**, over a frozen horizon H. For each attempt, record the target state and a direct monotonic-clock interval from request/job entry (including queue delay) through persistence of the independently verified result. A within-horizon hit contributes its observed deployment elapsed time only if the target is independently verified and the clock reading is present and ≤ H. Every other outcome contributes exactly H: infeasible/unverified output, target miss, timeout/crash/abort, missing deployment clock, or hit after H.

This is a fixed-horizon operational score, not a Kaplan–Meier survival estimate. Report within-horizon target-hit probability separately from the restricted mean. A hit exactly at H counts as a hit but contributes H to the restricted-time mean. If neither arm has a verified hit within H, report both means as H and do not present a restricted-mean ratio as a useful speedup.

### Mandatory secondary outcomes

- **Directly observed standardized algorithm runtime** and **directly observed deployment all-in latency** as separate clock intervals.
- Target hit rate, verified feasible-solution rate, unverified/malformed output rate, timeout/crash rate, and paired hit-rate difference.
- Best verified cardinality versus direct elapsed time; normalized shortfall from the frozen target; incumbent curves and time-to-first-feasible.
- Exact-optimum gap and time-to-proof for Tier A instances where the relevant solver can certify them.
- Acquisition, preprocessing, feature extraction, transfer, compilation/initialization, solve, repair, independent verification, and receipt/persistence costs as **stage diagnostics**; preserve missing values as unknown.
- Total-system energy only when both arms use the same explicitly measured system boundary. Never compare device-only energy with full-system energy.
- Memory use, processor utilization, throughput, and scaling by vertex/edge count as descriptive engineering metrics.

**Standardized algorithm runtime** is measured directly using the frozen QOBLIB-compatible interval that excludes queue wait and compilation/initialization. **Deployment all-in latency** is measured directly from request/job entry through persistence of the verified result, including queue and amortized compile/init. Use monotonic clocks, record the exact interval boundaries, and report the timing convention for every number.

Stage durations may overlap or run concurrently. **Never sum per-stage durations and call the sum end-to-end latency.** Missing stage diagnostics do not erase a directly observed wall-clock interval. Missing direct clock intervals remain unknown and cannot produce a clock ratio or a within-horizon hit; unknown is never zero.

Conditional speedup on trials where both arms hit a target is a secondary diagnostic only; it is survivor-biased. Always include the outcomes and hit rates for every attempted pair.

## 7. Statistical and engineering decision gates

The numerical thresholds below are proposed **engineering goals**, not literature-derived guarantees. They must be accepted and frozen before confirmatory runs.

1. **Correctness gate:** zero false target certificates in the reported confirmation set; every counted target hit passes the pinned independent verifier. Any false certificate blocks an advantage claim until root-caused and re-tested.
2. **No quality hiding:** the trajectory arm's paired target-hit probability must be non-inferior to the frozen classical champion, using a predeclared absolute margin no looser than **−1 percentage point** at the selected common budget.
3. **Effect-size goal:** aim for at least **1.25×** improvement in restricted-mean all-in verified time-to-target versus the frozen champion (equivalent to at least 20% lower time when the ratio is champion / candidate). The family-clustered 95% uncertainty interval must exclude no improvement; also report the point estimate and interval rather than only a threshold pass/fail.
4. **Generalization gate:** report results on untouched graph families. A within-family win alone is not sufficient for a cross-family claim.
5. **Ablation gate:** time-resolved features must improve over final-spin-only and graph-only controls. Adaptive acquisition must be compared with fixed-K at matched total budget; stable ranking alone is not evidence of saved solving time.
6. **Negative controls:** shuffled features should not reproduce the treatment effect. If they do, investigate leakage, confounding, and unaccounted compute before interpreting the feature as causal.
7. **Multiplicity:** declare one primary comparison and estimator. Apply a prespecified correction or clearly label other arm/metric comparisons exploratory.
8. **Reproducibility:** publish exact heads, build provenance, configuration, seeds, full per-attempt records, verifier receipts, logs, and all exclusions. Report negative outcomes.

Failing a gate is a useful result. Do not relax a gate after seeing the data; revise the hypothesis and run a new, separately identified confirmation experiment.

## 8. QOBLIB reporting and cost accounting

QOBLIB provides model-independent public instances, accepted result submissions, feasibility validation, and optional objective time series for convergence / time-to-solution analysis. Its current participation guide asks for complete workflow, hardware, repeat-count, and runtime details; for stochastic algorithms it requires at least five runs and recommends 10+. Its standardized runtime guidance excludes compile and queue time. Keep QOBLIB-compatible algorithm numbers separate from the deployment all-in metric needed for an end-user service.

QOBLIB's published classical baselines are reference results, not a guarantee that every specialized classical approach has been exhausted. Compare against strong MIS-specific methods as well as the public reference values.

## 9. What would count as progress?

- **Positive:** trajectory guidance clears the correctness and non-inferiority gates, improves the primary all-in metric versus the frozen champion, and replicates on held-out families. Then submit a reproducible QOBLIB contribution and describe exactly which hardware and cost boundary the result supports.
- **Useful negative:** trajectories add no value, the feature-only control performs similarly, acquisition costs dominate, or a specialist solver wins. Publish this, identify the break-even region, and stop expanding the feature set without a new hypothesis.
- **Inconclusive:** too few independent families, missing timing stages, weak or ineligible baseline, or wide uncertainty. Improve the experiment; do not call it parity or advantage.

## References

- QOBLIB portal: https://zib-aopt.github.io/QOBLIB/
- QOBLIB participation requirements and objective time-series format: https://zib-aopt.github.io/QOBLIB/participate.html
- Koch et al., *The Quantum Optimization Benchmarking Library*, Nature Computational Science (23 June 2026), https://doi.org/10.1038/s43588-026-00991-1
- KarlsruheMIS specialist methods: https://github.com/KarlsruheMIS
- Hellstern, Schacht & Zaefferer, *Comparing classical and quantum QUBO solvers: optimality, trajectory diagnostics, and limits of solver selection*, arXiv:2610.35126 (posted 26 October 2026): https://arxiv.org/abs/2610.35126

The diagnostic preprint is not a result on our MIS features; it is relevant prior caution that explaining solver trajectories does not, by itself, validate a solver-selection policy.
