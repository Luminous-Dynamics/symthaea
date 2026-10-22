# CASTING-002B — Deterministic Synthetic Benchmark Harness

**Status:** Test-only implementation drafted; Rust execution pending  
**Parent protocol:** [CASTING-002A / issue #8420](https://github.com/Luminous-Dynamics/symthaea/issues/8420)  
**Claim ceiling:** Synthetic harness semantics and accounting only

## What is implemented

The test-only module `physics::casting_optimization_benchmark` implements a deterministic, finite-cost evaluation harness with four sampling strategies:

- seeded uniform random sampling;
- uniform grid sampling;
- seeded two-dimensional Latin-hypercube sampling with one sample in each stratum per dimension;
- an **adaptive nearest-neighbor explore/exploit baseline** with a four-point LHS warm start, followed by a frozen candidate-lattice score: nearest observed objective minus a fixed distance-based exploration bonus.

It defines six synthetic cases with known objective minima:

1. smooth bowl;
2. coupled-factor ridge;
3. bounded feasible island;
4. heteroscedastic observation noise;
5. biased cheap/low-fidelity observations versus an expensive/high-fidelity oracle;
6. a deterministic failed-query corner.

Every record stores the chosen point, status, fidelity, evaluation-only ground-truth value, observed objective where available, cost units and cumulative cost. A run receipt retains all records and a domain-separated BLAKE3 transcript digest. Failed and infeasible attempts are recorded and charged. Zero budget and cumulative-cost overflow fail closed.

The adaptive strategy is **not Bayesian optimization**: it does not fit a Gaussian process or calibrated posterior. It is a transparent sequential baseline for later comparison with a properly specified BO implementation. The selector accepts a restricted `ProposalObservation` view containing only candidate coordinates, trial status and observed objective; the evaluation-only ground-truth field is excluded by the input type. A negative-control test changes that hidden field in the evaluator transcript and verifies that the projected proposal history and next candidate remain unchanged.

The paired comparison helper runs all four strategies over the same ordered seed list and reports per-strategy run count, median simple regret, target hit rate, median cost-to-target among successful runs, and mean cost units per run. The target hit rate is essential because the conditional cost among successful runs alone would omit strategies that never reached the target. In synthetic cases with noisy observations, target achievement is judged against the noise-free objective only after each query; the adaptive selector is not given that truth as an acquisition input.

## Important scope limits

This is the **first executable baseline tranche**, not the complete CASTING-002A benchmark. It does not yet implement Bayesian optimization, multi-fidelity acquisition, a DOE/response-surface solver, or a foundry solver adapter. The fidelity discrepancy exists to stress later method-selection code; no adaptive method currently decides when to purchase a low- or high-fidelity evaluation.

The synthetic objective is not a casting-physics model. No observed objective, regret score, cost unit, or synthetic failure corresponds to real material, equipment, energy, yield, defect rate, or production cost. The ground-truth objective is retained in evaluation receipts for the evaluator; query selection does not use unqueried candidate truth. The adaptive proposer is explicitly checked not to read the evaluation-only truth field from prior records.

The arbitrary cost schedule is deliberately explicit: one unit per low-fidelity call and ten per high-fidelity call. These are normalized experimental design weights, **not currency, compute-energy measurements, or a realistic foundry cost model**. Future work must use a preregistered schedule grounded in the actual comparison.

## Test inventory

The 15 authored tests cover:
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
- explicit exhaustion of the finite candidate lattice.

Test command:

```sh
cargo test -p symthaea-core casting_optimization_benchmark
```

Tests are authored but have not been compiled or run in the current authoring environment because Rust/Cargo is unavailable there. Exact-head CI must pass before the module is treated as executable evidence. If the crate-wide test target has unrelated failures, report them separately; do not hide them by weakening the test or shrinking the required evidence.

## Promotion ladder

1. **Harness integrity:** run the exact-head tests; independently review objective truth, sampling semantics, cost accounting and transcript digest.
2. **DOE comparison:** add a preregistered response-surface/factorial baseline for cases where its assumptions and factor design are appropriate.
3. **Adaptive single-fidelity:** add a simple, independently reviewed Bayesian-optimization baseline; ensure optimizer receives only queried observations and predeclared constraints, never evaluation-only truth.
4. **Multi-fidelity:** compare low-/high-fidelity acquisition under a frozen discrepancy model and explicit costs. Penalize choices that over-trust a cheap but biased model.
5. **Statistical report:** at least 100 paired seeds per synthetic case when practical, including per-run data, medians/quantiles, paired differences and uncertainty intervals. If runtime forces a smaller count, preregister and disclose it before seeing results.
6. **Matched physical validation:** choose a named, non-critical part/process with competent foundry and metrology review. Freeze inspection, units, sample units, hard constraints, cost boundaries, holdout, stopping policy and failed-run handling before data collection.

## Research motivation

A 2026 review of computational modeling, AI, and digital twins in metal casting highlights the difficulty of reproducible benchmarking, uncertainty quantification, generalizing across alloys/geometries, and industrial deployment. NIST's 2026 report on uncertainty quantification and metrology argues that model-credibility effort should scale with model influence and decision consequence. The benchmark consequently separates cheap synthetic harness checks from evidence needed for an industrial claim.

- Essam et al. (2026), casting simulation / AI / digital twin review: https://link.springer.com/article/10.1007/s43939-026-00685-5
- NIST (5 October 2026), *On Methods of Uncertainty Quantification in Computational Modeling and Simulation*: https://www.nist.gov/publications/methods-uncertainty-quantification-computational-modeling-and-simulation-harmonizing
- Pei et al. (2025), revised LHS + Bayesian optimization + phase-field case study: https://www.nature.com/articles/s41524-025-01524-6
- Bayesian optimization-driven framework for CFD problems applied to continuous casting, *Computers & Fluids* (30 August 2026), compared an open-source BO framework to commercial CAESES on a submerged-entry-nozzle geometry problem using a single-phase water approximation. Treat as an algorithmic comparator, not matched molten-metal physical validation: https://www.sciencedirect.com/science/article/pii/S0045793026002276

## Nonclaims

This module does not prove that one sampling design beats another, that Bayesian optimization or HDC outperforms state of the art, that low fidelity can safely replace a physics solver, or that a foundry process has improved. No synthetic benchmark has been run here. No commercial solver comparison, physical experiment, part qualification, machine control or process recommendation is included.
