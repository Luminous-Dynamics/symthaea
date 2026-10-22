# CASTING-002B — Deterministic Synthetic Benchmark Harness

**Status:** Test-only implementation drafted; Rust execution pending  
**Parent protocol:** [CASTING-002A / issue #8420](https://github.com/Luminous-Dynamics/symthaea/issues/8420)  
**Claim ceiling:** Synthetic harness semantics and accounting only

## What is implemented

The test-only module `physics::casting_optimization_benchmark` implements a deterministic, finite-cost evaluation harness with five sampling strategies:

- seeded uniform random sampling;
- uniform grid sampling;
- seeded two-dimensional Latin-hypercube sampling with one sample in each stratum per dimension;
- an **adaptive nearest-neighbor explore/exploit baseline** with a four-point LHS warm start, followed by a frozen candidate-lattice score: nearest observed objective minus a fixed distance-based exploration bonus;
- a minimal **Gaussian-process lower-confidence-bound (GP-LCB) Bayesian-optimization baseline** with a fixed RBF kernel, empirical constant prior mean, fixed observation-noise variance and diagonal jitter, and a fixed LCB exploration coefficient.

It defines eleven synthetic cases with frozen analytical references:

1. smooth bowl;
2. coupled-factor ridge;
3. bounded feasible island;
4. heteroscedastic observation noise;
5. biased cheap/low-fidelity observations versus an expensive/high-fidelity oracle;
6. a deterministic failed-query corner;
7. a non-convex Rastrigin landscape with a known global optimum and many local minima;
8. the standard Beale analytical benchmark over its declared domain;
9. McCormick's non-separable objective on its published rectangular domain;
10. the Eggholder many-local-minima function, including its boundary global minimum;
11. the Holder-table function with four global minima.

Every record stores the chosen point, status, fidelity, evaluation-only ground-truth value, observed objective where available, cost units and cumulative cost. A run receipt retains all records and a domain-separated BLAKE3 transcript digest. Failed and infeasible attempts are recorded and charged. Zero budget and cumulative-cost overflow fail closed.

The nearest-neighbor strategy remains a transparent heuristic, not Bayesian optimization. The new GP-LCB baseline fits a small Gaussian-process posterior from observed feasible records using fixed hyperparameters and a fixed lower-confidence-bound acquisition. It is a reference implementation for method comparison, not a tuned or production-qualified optimizer: hyperparameters are not learned, the search is a finite 16×16 lattice, and it does not choose fidelity adaptively. Both sequential selectors accept a restricted `ProposalObservation` view containing only coordinates, trial status and observed objective; evaluation-only ground truth is excluded by the input type. Negative-control tests change that hidden field and verify the next proposal is unchanged.

The paired comparison helper runs all five strategies over the same ordered seed list and reports per-strategy run count, median simple regret, target hit rate, median cost-to-target among successful runs, and mean cost units per run. The target hit rate is essential because the conditional cost among successful runs alone would omit strategies that never reached the target. Target achievement is defined relative to each case's frozen reference minimum plus a 0.05 objective gap, and is confirmed against noise-free objective truth only after each query; that truth is never an acquisition input.

## Important scope limits

This is the **first executable baseline tranche**, not the complete CASTING-002A benchmark. It does not yet implement learned GP hyperparameters, expected-improvement acquisition, multi-fidelity acquisition, a DOE/response-surface solver, or a foundry solver adapter. The fidelity discrepancy exists to stress later method-selection code; no adaptive method currently decides when to purchase a low- or high-fidelity evaluation.

The synthetic objective is not a casting-physics model. No observed objective, regret score, cost unit, or synthetic failure corresponds to real material, equipment, energy, yield, defect rate, or production cost. The ground-truth objective is retained in evaluation receipts for the evaluator; query selection does not use unqueried candidate truth. The adaptive proposer is explicitly checked not to read the evaluation-only truth field from prior records.

The GP-LCB constants are deliberately fixed: RBF length scale 0.25, signal variance 1.0, observation-noise variance 1e-6, diagonal jitter 1e-10 and LCB exploration coefficient 1.5. These are benchmark constants, not tuned results. The arbitrary cost schedule is deliberately explicit: one unit per low-fidelity call and ten per high-fidelity call. These are normalized experimental design weights, **not currency, compute-energy measurements, or a realistic foundry cost model**. Future work must use a preregistered schedule grounded in the actual comparison.

## Test inventory

The 24 authored tests cover:
- known-answer minima and feasibility at each declared optimum;
- equal query budgets, complete status counts, cumulative cost accounting and non-negative regret;
- exact replay of records, metrics and transcript digest at a fixed seed;
- changed transcript under a changed random seed;
- Latin-hypercube stratum coverage in both dimensions;
- explicit charges for failed and infeasible queries;
- low-/high-fidelity cost and intentionally biased low-fidelity observations;
- repeatable, seed-sensitive heteroscedastic noise;
- zero-budget refusal;
- paired-seed summaries and rejection of an empty seed set;
- adaptive proposal independence from the evaluation-only truth field and no repeated candidate points;
- adaptive determinism and inclusion in paired summaries;
- exploration fallback when the warm start contains no usable observations;
- explicit exhaustion of the finite candidate lattice;
- RBF-kernel, Cholesky factorization and triangular-solve known answers;
- one-observation GP posterior mean/variance known answer;
- duplicate-location covariance regularization and non-finite/asymmetric covariance rejection;
- GP acquisition deterministic replay and truth-leakage negative control;
- rejection of non-finite observations, malformed/non-positive-definite covariance and accidental use of sequential strategies as fixed samplers;
- a known-answer Rastrigin global minimum for non-convex optimization testing;
- the standard Beale benchmark's known global minimum;
- raw reference minima and closed-boundary handling for McCormick, Eggholder and Holder table.

Test command:

```sh
cargo test -p symthaea-core casting_optimization_benchmark
```

Tests are authored but have not been compiled or run in the current authoring environment because Rust/Cargo is unavailable there. Exact-head CI must pass before the module is treated as executable evidence. If the crate-wide test target has unrelated failures, report them separately; do not hide them by weakening the test or shrinking the required evidence.

## Promotion ladder

1. **Harness integrity:** run the exact-head tests; independently review objective truth, sampling semantics, cost accounting and transcript digest.
2. **Analytical parity:** the initial suite now includes Beale, McCormick, Eggholder and Holder table (the four analytical functions used in the 2026 continuous-casting BO comparison) plus Rastrigin. Independently validate each formula/domain/reference point and preserve raw objective values; add independent benchmark-oracle receipts before comparing optimizers.
3. **DOE comparison:** add a preregistered response-surface/factorial baseline for cases where its assumptions and factor design are appropriate.
4. **GP-BO validation:** independently review the current fixed-hyperparameter GP-LCB baseline; run exact-head tests and compare it against the nearest-neighbor heuristic over the frozen suite, including the multimodal Rastrigin fixture.
5. **Multi-fidelity:** compare low-/high-fidelity acquisition under a frozen discrepancy model and explicit costs. Penalize choices that over-trust a cheap but biased model.
6. **Statistical report:** at least 100 paired seeds per synthetic case when practical, including per-run data, medians/quantiles, paired differences and uncertainty intervals. If runtime forces a smaller count, preregister and disclose it before seeing results.
7. **Matched physical validation:** choose a named, non-critical part/process with competent foundry and metrology review. Freeze inspection, units, sample units, hard constraints, cost boundaries, holdout, stopping policy and failed-run handling before data collection.

## Research motivation

The 2026 continuous-casting BO paper compares open-source BO with commercial CAESES using McCormick, Beale, Eggholder and Holder-table analytical functions, then applies the workflow to a submerged-entry-nozzle geometry under a single-phase water approximation. The current synthetic suite now includes those four analytical objective families, but has not yet reproduced the paper's precise candidate domain, solver coupling, hyperparameters, or reported results.

A 2026 review of computational modeling, AI, and digital twins in metal casting highlights the difficulty of reproducible benchmarking, uncertainty quantification, generalizing across alloys/geometries, and industrial deployment. NIST's 2026 report on uncertainty quantification and metrology argues that model-credibility effort should scale with model influence and decision consequence. The benchmark consequently separates cheap synthetic harness checks from evidence needed for an industrial claim.

- Essam et al. (2026), casting simulation / AI / digital twin review: https://link.springer.com/article/10.1007/s43939-026-00685-5
- NIST (5 October 2026), *On Methods of Uncertainty Quantification in Computational Modeling and Simulation*: https://www.nist.gov/publications/methods-uncertainty-quantification-computational-modeling-and-simulation-harmonizing
- Pei et al. (2025), revised LHS + Bayesian optimization + phase-field case study: https://www.nature.com/articles/s41524-025-01524-6
- SFU Virtual Library of Simulation Experiments, optimization function catalogue: https://www.sfu.ca/~ssurjano/optimization.html
- Eggholder domain and many-local-minima description: https://www.sfu.ca/~ssurjano/egg.html
- McCormick domain/reference: https://www.sfu.ca/~ssurjano/mccorm.html
- 2026 continuous-casting BO versus CAESES paper: https://www.sciencedirect.com/science/article/pii/S0045793026002276
- A Bayesian optimization-driven framework for CFD problems applied to continuous casting (Computers & Fluids, 30 August 2026): compared open-source BO against commercial CAESES on analytical benchmark functions and a single-phase-water submerged-entry-nozzle geometry case: https://www.sciencedirect.com/science/article/pii/S0045793026002276
- Bayesian optimization-driven framework for CFD problems applied to continuous casting, *Computers & Fluids* (30 August 2026), compared an open-source BO framework to commercial CAESES on a submerged-entry-nozzle geometry problem using a single-phase water approximation. Treat as an algorithmic comparator, not matched molten-metal physical validation: https://www.sciencedirect.com/science/article/pii/S0045793026002276

## Nonclaims

This module does not prove that one sampling design beats another, that Bayesian optimization or HDC outperforms state of the art, that low fidelity can safely replace a physics solver, or that a foundry process has improved. No synthetic benchmark has been run here. No commercial solver comparison, physical experiment, part qualification, machine control or process recommendation is included.
