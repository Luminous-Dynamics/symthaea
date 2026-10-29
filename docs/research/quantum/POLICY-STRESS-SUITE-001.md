# QARCH-009 — Frozen multi-scenario decision-regret stress suite

**Status:** draft analytical reference suite; code tests authored, not executed  
**Issue:** [#9056](https://github.com/Luminous-Dynamics/symthaea/issues/9056)  
**Depends on:** QARCH-008 / [PR #9032](https://github.com/Luminous-Dynamics/symthaea/pull/9032)  
**Canonical fixture:** `docs/research/quantum/benchmarks/QARCH-009-POLICY-STRESS-SUITE.json`

## Why the paradigm shift matters

Information gain is not decision value; nominal expected value of sample information is only as trustworthy as its model; robustness is not a universal free lunch. A serious policy benchmark must include cases where robust selection helps, where it hurts, where the assumed model family misses the truth, and where the best action is to take no measurement.

This makes **decision regret under declared distribution shift** the primary target, while keeping utility, resource cost, calibration failure, and unsupported outcome mass as separate axes. It prevents the benchmark from rewarding a method merely for selecting a high-entropy experiment.

The 2026 Quantum Optimization Benchmarking Library (QOBLIB) offers a useful industry standard for experimental discipline: shared problem classes, strong classical references, verified feasibility, and reproducible result reporting. Open QBench adds benchmarking across hybrid, high-level, and compiled quantum-classical stack levels. QARCH-009 does not replace either; it adds a synthetic test surface for the *experiment policy that chooses measurements*. See [QOBLIB's 2026 paper](https://www.nature.com/articles/s43588-026-00991-1), [live QOBLIB results](https://zib-aopt.github.io/QOBLIB/), and [Open QBench](https://quantum.psnc.pl/en/openqbench/).

## Frozen scenario matrix

All five scenarios use two hypotheses and two actions, with utility 1 for a correct choice and 0 otherwise. Candidate A is nominally perfect in the default planning model; B is 80%-informative in both default planning scenarios. The held-out truth changes by scenario and remains evaluator-only.

| Case | Stressor | Expected lesson |
|---|---|---|
| Q009-01 | Nominal A collapses to uninformative | Robust B can reduce held-out regret |
| Q009-02 | Prior shifts to 0.9/0.1 | Even robust B can be worse than no measurement |
| Q009-03 | A is 65%-informative, B only 55% | Robustification can lose 0.10 utility vs nominal choice |
| Q009-04 | A detector is reversed outside the ambiguity set | Robust B reduces damage but still has 0.45 regret |
| Q009-05 | All candidates are uninformative | No measurement ties the oracle; no false advantage |

The suite compares nominal EVSI/cost, nominal information gain/cost, robust worst-case EVSI/cost, fixed-first, an exact uniform-random expectation, and **no measurement**. The oracle uses evaluator-only truth and is a regret reference, never an available selector.

## Exact analytical reference

| Policy | Mean regret over these 5 cases | Worst-case regret |
|---|---:|---:|
| Nominal EVSI/cost | 0.340 | 1.000 |
| Nominal information gain/cost | 0.340 | 1.000 |
| Robust worst-case EVSI/cost | 0.130 | 0.450 |
| Fixed first | 0.340 | 1.000 |
| Uniform random (exact expectation) | 0.235 | 0.725 |
| No measurement | 0.190 | 0.500 |

These numbers are computed from the fixture's analytical cases by construction. **They are not evidence of empirical superiority or a general policy ranking.** In Q009-03, robust selection loses 0.10 utility against nominal EVSI; in Q009-02, no measurement is the optimal action. The point is to measure both the robustness benefit and its cost, not to crown a winner on a curated synthetic set.

## Contract and acceptance

- Held-out truth never reaches the selector.
- Priors and likelihoods may shift; hypothesis labels, actions, utilities, candidate IDs, outcomes, and cost units remain contract-compatible.
- Report per-case utility, regret, cost, unsupported outcome mass, no-measurement utility, and oracle reference.
- Keep cost separate from utility unless an exchange rate is preregistered.
- Preserve deterministic tie-breaking and scenario-order invariance.
- Keep analytic, simulated, calibrated, and measured evidence classes distinct.
- Exact arithmetic checks must match the frozen fixture; malformed contracts must fail closed.
- Do not claim SOTA or quantum advantage from these tests. Use QARCH-007 predictive adequacy before physical scenario evaluation and QSIM-001 only through a separately qualified physics-to-likelihood adapter.

The Rust source adds five stress tests plus an explicit `NoMeasurement` policy result. The source tests instantiate the matrices directly; this JSON is the canonical human-/machine-readable reference fixture, not yet a runtime-loaded test input.

## Current validation boundary

The code was committed to the feature branch and the fixture JSON was validated for syntax during creation. The local environment does not have `cargo` or `rustc`; Rust compilation/tests have not been executed here. Exact-head GitHub Actions checks must pass before treating this as validated implementation evidence.
