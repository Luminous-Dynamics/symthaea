# QCOMP-PORTFOLIO-001 — Workload-aware classical/quantum/hybrid compute selection

**Status:** Research contract / draft. Not implemented or qualified.  
**Owner:** Symthaea research and workload-orchestration layer.  
**Related work:** [PHYS-001 neutral physical-computation backend contract](https://github.com/Luminous-Dynamics/symthaea/pull/3624), [PHYS-003 matched substrate-neutral controls](https://github.com/Luminous-Dynamics/symthaea/pull/3679), [QCOMP-RT-001](https://github.com/Luminous-Dynamics/symthaea/issues/5806), and [symthaea-quantum-comp](https://github.com/Luminous-Dynamics/symthaea/tree/main/crates/domains/symthaea-quantum-comp).

## 1. Research question

For a specific problem instance and a declared service contract, can Symthaea select and coordinate the available classical, GPU, quantum, and hybrid methods to produce a valid result at the lowest observed end-to-end cost—without violating quality, deadline, energy-accounting, or evidence constraints?

The service contract must declare, at minimum:

- an exact problem/instance identity and independent solution checker;
- acceptable solution quality (exact, bounded gap, or a predeclared task-specific threshold);
- a deadline and, where relevant, monetary and energy budgets;
- eligible execution backends and their current evidence/calibration status;
- repeatability, risk, and data-handling requirements.

The router may return **no qualified route** when available evidence is inadequate. It must not select a quantum method simply because the workload is labelled “quantum,” or promote a predicted speedup into a measured result.

## 2. Why this is a new workstream

Do not create a competing benchmark corpus or another generic quantum SDK. Reuse established, public test suites and make the added contribution the **decision policy, workload-level accounting, uncertainty, and reproducible evidence** across candidate methods.

The public state of the art already has valuable components:

| Existing work | What it establishes | What this work should add |
|---|---|---|
| [QOBLIB, Nature Computational Science (published 23 June 2026)](https://www.nature.com/articles/s43588-026-00991-1) | Ten model-independent optimization problem classes, public instances/track records, and state-of-the-art classical reference results. | Use frozen instances/results to test portfolio selection and end-to-end cost-to-target; never relabel its baselines as our own runs. |
| [MQT Bench 2.3.0](https://mqt-bench.app/) | Quantum circuits across algorithmic, target-independent, native-gate, and mapped levels for many algorithm families and device targets. | Carry circuit/compilation results into a whole-workflow comparison only when that workload profile applies. |
| [QED-C application benchmarks](https://quantumconsortium.org/publication/application-oriented-performance-benchmarks-for-quantum-computing/) | Application circuit width/depth, result quality/fidelity, and aspects of pipeline execution and time to solution. | Compare against the best eligible classical method for the same required output and quality threshold. |
| [IBM hardware metrics (16 July 2026)](https://www.ibm.com/quantum/blog/hardware-metrics-2026) | Quantum hardware scale, quality, and speed: programmable qubits, complex qubit operations, and circuit throughput. | Keep hardware metrics as predictors/diagnostics, not substitutes for workload-level advantage. |
| [IBM hybrid workflow scheduling paper (QCE 2026, 13 September)](https://research.ibm.com/publications/practical-example-of-optimal-scheduling-of-hybrid-quantum-classical-workflows) | Hybrid workflow scheduling with task dependencies, data movement, resource contention, queueing, and resource topology. | Test policy decisions against measured or explicitly modelled end-to-end costs, and label models versus observations separately. |
| [Nature Reviews Physics, “Benchmarking quantum computers” (7 January 2025)](https://www.nature.com/articles/s42254-024-00796-z) | Benchmark design affects what performance claims can legitimately be made. | Preregister the claim, task, budget, quality threshold, and comparison set before looking at results. |

The 2026 QOBLIB paper explicitly distinguishes application benchmarking (searching for the best method for an instance), algorithm benchmarking (testing one algorithm family), and system benchmarking (running a fixed algorithm on a specific platform). It identifies application-level comparisons as the level needed to demonstrate quantum advantage. This proposal complements, rather than replaces, that framework.

## 3. Paradigm shifts

1. **From qubit count to useful workload capability.** Record physical/logical qubits and hardware quality, but rank routes by verified work completed within the declared budget.
2. **From one winner to a solver portfolio.** Treat exact solvers, classical heuristics, GPU methods, tensor-network simulation, quantum methods, and hybrid pipelines as eligible candidates. The winning method may differ by instance and constraint.
3. **From one leaderboard score to a Pareto frontier.** Report time, money, energy (with its measurement boundary), solution quality, success probability, and reliability separately. Do not collapse incomparable measures into an opaque universal score.
4. **From static selection to uncertainty-aware routing.** Predict outcome distributions, include calibration age and queue/availability where relevant, and abstain or fall back when the predicted advantage is not robust.
5. **From kernel timing to end-to-end accounting.** Include model conversion, compilation/transpilation, calibration, queueing, transfers, state preparation, shots/retries, error mitigation, post-processing, and independent verification. Also report compute-only time separately for diagnosis.
6. **From “quantum versus classical” to co-designed heterogeneous computing.** Use a QPU only when the expected contribution survives interface overhead and beats the best feasible alternative for the declared task.
7. **From impressive demos to prospective evidence.** Freeze a benchmark cohort and evaluation policy before tuning; keep held-out instances genuinely held out; preserve negative and inconclusive outcomes.

These shifts are hypotheses for a useful engineering platform, not claims that a routing advantage has already been demonstrated.

## 4. First experiment: a narrow, falsifiable pilot

### 4.1 Benchmark family

Start with **maximum independent set (MIS)** from QOBLIB because feasibility is independently checkable and small instances can anchor correctness against exact results. Use the QOBLIB instance IDs, source revision, model formulation, source-file hashes, and known-optimum/best-known status as the manifest. Keep the remaining QOBLIB classes in the source registry but out of the first headline result.

Do not cherry-pick instances based on a candidate solver's result. Preregister the eligible cohort, exclusions, target sizes, time limits, memory limits, and stopping rule. Split by instance family/graph construction—not merely random seed—so variants of the same underlying instance cannot leak between tuning and test sets.

### Baseline reality check (snapshot reviewed 22 October 2026)

QOBLIB's live site currently reports **1,264 total instances, 533 optimally solved instances, and 3,226 submissions** across ten problem classes; its Maximum Independent Set class lists **50 instances** (counts are a changing upstream snapshot, not constants to hard-code). An upstream classical submission published on 6 August 2026 reports cooperative parallel iterated local search across all 50 MIS instances, with ten independent trials per instance. It reports best-known-or-better objectives on all 50 and two new best-known values, on `frb53-24-1` (53 vs. 52) and `frb100-40` (96 vs. 94), on an Apple M5 Max with 18 CPU cores. Reported total runtimes range from fractions of a millisecond for small instances to 120 seconds for some larger cases. See the [QOBLIB live dashboard](https://zib-aopt.github.io/QOBLIB/) and [the full ParallelILS result record](https://github.com/ZIB-AOPT/QOBLIB/blob/main/07-independentset/submissions/20260806_ParallelILS_Hegade/README.md).

A particularly instructive same-instance example is `es60fst02` (186 decision variables, best-known objective 88). The ParallelILS submission reports objective 88 with 10/10 successful runs and 0.0006 seconds total runtime; the [HSQC hybrid quantum-hardware submission](https://github.com/ZIB-AOPT/QOBLIB/blob/main/07-independentset/submissions/20260804_HSQC_Hegade/es60fst02/README.md) reports objective 88 with 5/5 successful runs and 2.5608 seconds total runtime on an IBM Heron R3 QPU plus an Apple M5 Max CPU. The [QAOA hardware submission](https://github.com/ZIB-AOPT/QOBLIB/blob/main/07-independentset/submissions/20260907_QAOA_Vivek/es60fst02/README.md) also reports objective 88, but its record lists 23,365.8542 seconds total runtime across 20 runs, 78 seconds QPU runtime, and five successful runs. The records differ in hardware, algorithms and run conventions, so these numbers are **illustrative upstream reports, not a controlled head-to-head speedup experiment**. They nonetheless show exactly why CPU preprocessing, repetitions, quality threshold, QPU time and end-to-end time must all remain visible.

Treat upstream submissions as **published records, not runs independently reproduced by this project**. They set the baseline bar: our bounded exact solver is solely a correctness oracle for tiny fixtures. A performance claim must be tested against the strongest relevant QOBLIB classical records plus currently available eligible solvers, on matched instances and disclosed hardware/resource budgets. An easy-instance win or a comparison against only one weak heuristic is not evidence of state-of-the-art improvement.

### 4.2 Candidate portfolio

Begin with methods that are locally executable and whose versions and budgets can be pinned:

- exact reference methods on small instances, for correctness anchors;
- at least one established classical exact/MIP method where available;
- at least two distinct classical heuristic/anytime methods;
- Symthaea-proposed candidate(s), kept as untrusted candidates until independently evaluated.

Add a QPU or quantum-inspired solver adapter only when its identity, executable path, compilation/model conversion, device profile, calibration context, and result validation can be recorded. Record commercial solver access/licensing when used; an open-source-only portfolio is not evidence that the commercial state of the art was beaten.

For the first local run, the valid deliverable is a **classical portfolio and its measured selection regret**. No quantum-advantage claim can follow from it. Actual external quantum execution is a later evidence stage.

### 4.3 Router baselines

Compare the learned/adaptive selector against at least:

- best fixed solver across the cohort;
- simple feature-based routing with preregistered rules;
- round-robin or uniform-budget portfolio;
- per-instance measured oracle (used only as an offline upper bound, never presented as a deployable policy);
- a safe fallback policy that selects the best validated eligible method under conservative cost assumptions.

The main question is not whether a selector beats a weak solver; it is how closely it approaches the best eligible method while using only information available at decision time.

## 5. Required metrics

Report the raw run table and distributions, not only averages.

### Result quality and correctness

- **Independent feasibility/validity rate:** every returned answer passes a checker not implemented by the candidate solver.
- **Objective value and gap:** report optimum gap only when an optimum is known/proven. If only a best-known value exists, label it gap-to-best-known, not optimality gap.
- **Quality-at-budget curve:** best verified quality reached by increasing time/money/energy budgets.
- **Time-to-target:** wall time to a predeclared acceptable result, including retries, with success probability and uncertainty.

For a maximization problem with known optimum (f^* > 0), one possible normalized optimality gap is \((f^* - f)/(\max(|f^*|, \epsilon))\). The implementation must define edge cases and use the same formula across all candidates. This is a proposal, not an already implemented metric.

### End-to-end resources

- user-visible wall-clock time (queue wait included);
- active compute time and queue time separately;
- direct execution charge and amortized/setup charges separately;
- energy only with an explicit boundary: MeasuredWholeSystem, MeasuredDeviceOnly, Modelled, or Unmeasured;
- memory, CPU/GPU allocation, data transfer volume, shots, retries, compilation and calibration overhead;
- failures, timeouts, invalid outputs, and human intervention.

Never compare whole-system measured energy for one route with a device-only estimate for another as if they were equivalent.

### Selection quality

- **Route regret:** selected route's achieved cost-to-target relative to the best route actually measured and eligible for that instance/budget.
- **Coverage:** fraction of eligible test cases for which the selector can make a qualified recommendation.
- **Abstention calibration:** when the selector says “uncertain,” record whether candidate rankings genuinely overlap or reverse.
- **Frontier recall:** how often the selector identifies at least one candidate on the measured quality/cost Pareto frontier.
- **OOD safety:** behavior on instance families, problem sizes, hardware profiles, and calibration states excluded from training.

### Hardware/system diagnostics

For any quantum execution, report modality and device identity, date, programmable qubits, relevant operation quality, circuit throughput where available, native/mapped circuit depth, gates, shots, readout/SPAM assumptions, calibration snapshot, error-mitigation procedure, and device/system energy-accounting boundary. Hardware metrics are not interchangeable with application advantage.

## 6. Fair-comparison protocol

Every public comparison must publish:

1. exact instance identity, formulation, source hash, and train/validation/test assignment;
2. exact solver/backend, version, arguments, environment, hardware, and resource limits;
3. tuning budget and whether tuning used any held-out instance or test result;
4. warm-up, compilation, calibration, queueing, repetitions, retries, and stopping rules;
5. all valid and invalid outputs, failures, timeouts, and exclusions with reasons;
6. independent validation results and raw per-run observations;
7. uncertainty intervals for quality, time-to-target, cost, and route regret;
8. provenance hashes for binaries, configs, inputs, outputs, and analysis code.

Run two comparison views: **matched-budget** (declared equal time/cost/energy constraints where measurement makes that comparison valid) and **best-available** (the strongest credible method with its actual disclosed resources). Do not silently treat these as the same experiment.

At least 30 independent seeded trials per stochastic solver/instance is a starting target where computationally feasible; the final sample size must be based on variance/power and resource limits, not used as a ritual threshold. Deterministic solver runs should be repeated to assess environmental/runtime variation even when outputs are identical.

## 7. Adversarial and negative-control cases

The evaluation must fail closed on cases such as:

- a fast solution is infeasible or fails an independent checker;
- a quantum route looks faster only because classical preprocessing, queue time, or error mitigation was omitted;
- a solver's best-known result is incorrectly labelled as a proven optimum;
- a model conversion changes the instance or makes the quantum formulation much larger without accounting for it;
- a selector was tuned on test-instance labels or test outcomes;
- a route beats the baseline only after using an unreported retry budget;
- a stale calibration profile makes a route ineligible but the selector still chooses it;
- an energy comparison mixes measured whole-system energy and a device-only model;
- simulation or circuit export is mislabelled as physical-device execution;
- a route wins an aggregate average while materially harming one declared workload family;
- the best solver differs across instances and the portfolio is compared only to one weak global baseline;
- the selector returns a confident recommendation when estimates are missing or rankings are statistically indistinguishable.

## 8. Cross-project responsibilities

- **Symthaea:** problem manifests, solver candidates, policy/routing, statistical evaluation, independent checking, and evidence-bounded research proposals. Reuse the neutral physical-computation contract in PHYS-001 rather than defining a second incompatible backend/evidence taxonomy.
- **Symtropy:** later digital-twin studies of resource topology, data movement, scheduling, thermal/energy models, and fault/resource availability. Simulation outputs must remain Modelled, not be substituted for observed latency or energy.
- **Mycelix:** optional later provenance/receipt and multi-party reproduction layer for frozen input, executable, configuration, result, and reviewer attestations. It must not be required to run the initial local benchmark or silently upgrade a scientific claim's status.

The initial pilot must not require physical hardware ownership, a custom semiconductor, a new quantum SDK, or a running external CI job. Local execution and independently checked fixtures can establish implementation correctness and classical selection quality; only a later external device observation can establish quantum hardware performance.

## 9. Acceptance gates and target benchmarks

These are **proposed internal targets**, not claims about current performance or published industry results.

### Gate 0 — Reproducible benchmark definition

- 100% of selected instances have frozen identity, formulation, source lineage, and a declared optimum/best-known/unknown status.
- 100% of run records preserve solver version, seed or determinism declaration, resource budget, and raw outcomes.
- 100% of solution outputs are independently checked; invalid outputs count as failures, not missing samples.
- No claims of hardware execution, quantum advantage, or SOTA improvement.

### Gate 1 — Competitive classical portfolio

- Reproduce selected public reference results where tools, versions, and hardware are available; otherwise mark the specific reference NotReproduced with reason.
- Compare against the strongest accessible and properly tuned eligible baselines, including commercial baselines where legitimate access exists.
- Publish per-instance quality-versus-budget curves and the per-instance measured-oracle bound.
- Produce at least one fully replayable run on a local machine without depending on CI results.

### Gate 2 — Selector quality on held-out instances

Initial aspirational target: on at least 90% of eligible held-out instances, the selector achieves a valid result within 1.25× the best measured eligible route's cost-to-target, or explicitly abstains when no qualified recommendation is supported. Define how zero-cost, timeout, and unsolved cases are treated before testing. Report confidence intervals and the cohort size.

### Gate 3 — Demonstrated practical improvement

A meaningful win requires at least one preregistered, useful instance family where a new route/portfolio gives either:

- at least **2× lower end-to-end cost-to-target** at non-inferior verified output quality; or
- materially better verified solution quality under the same declared end-to-end budget.

The result must persist on held-out instances, include uncertainty, survive the strongest available baseline comparison, and be independently replayable. The 2× value is a research target, not a guarantee or universal pass criterion.

### Gate 4 — Quantum or hybrid advantage

Only eligible after actual device observations exist. Require the same task, instance, quality threshold, and disclosed budget; the strongest credible classical comparison; end-to-end accounting; successful independent verification; uncertainty; and reproducibility. A qubit-count, quantum-volume, entanglement, fidelity, simulation, or circuit-compilation result alone cannot pass this gate.

### Release rule

Queued, missing, not-run, or simulated evidence must never be reported as PASS. A failed/inconclusive result is retained and explained; thresholds may not be weakened after seeing results. A newly discovered limitation should create a versioned profile or follow-up experiment, not retroactively rewrite prior evidence.

## 10. Implementation sequence

1. **P0 — Source registry and protocol:** freeze selected QOBLIB MIS instances and known-result semantics; define manifest/report schema and claim gates.
2. **P1 — Independent checker + adapter contract:** validate MIS feasibility/objective; make budgets and output status typed. Reuse established backend/evidence contracts.
3. **P2 — Classical baseline runner:** add reproducible adapters for eligible exact and heuristic solvers; capture timeouts, failed runs, and time-to-target.
4. **P3 — Portfolio measurement:** report per-instance curves, oracle bounds, uncertainty, and negative controls.
5. **P4 — Selector:** train/tune only on the designated training/validation groups; freeze before held-out evaluation; compare with fixed and uniform-portfolio baselines.
6. **P5 — Symtropy scheduling model:** only after empirical baselines exist, predict queue/resource/data-movement trade-offs and compare predicted against realized outcomes.
7. **P6 — QPU/hybrid integration:** use existing ecosystem tooling or an explicitly qualified adapter; run a prospective external-device evaluation and preserve all backend receipts.
8. **P7 — Mycelix reproduction receipts:** add optional attestations for input, code, output, and independent review without changing scientific claim semantics.

## Nonclaims

This document reports **no measured performance result**. It does not assert that Symthaea currently beats a state-of-the-art solver, routes QPU workloads, improves quantum hardware, or demonstrates quantum advantage. It proposes a falsifiable, staged route to finding out. The initial win may be a better classical portfolio or a justified “do not use the QPU” decision; that is useful evidence, not a failure of the research program.
