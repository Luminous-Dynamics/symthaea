# QARCH-008: exact regret benchmark for experiment-selection policies

**Status:** draft implementation with synthetic analytic fixture  
**Code:** crates/domains/symthaea-quantum-comp/src/experiment_policy_benchmark.rs  
**Frozen fixture:** docs/research/quantum/benchmarks/QARCH-008-EXPERIMENT-POLICY-BAKEOFF.json  
**Related:** QARCH-005 decision-focused EVSI; QARCH-006 robust model-ensemble ranking; QARCH-007 held-out predictive adequacy; Symtropy QSIM-001.

## Why this is the next gap

QARCH-005 reports which experiment scores best under one nominal model; QARCH-006 scores experiments across a declared finite model ensemble. Neither ranking alone tells us whether that choice actually leads to the best downstream decision when the model used for planning differs from the data-generating process.

QARCH-008 evaluates candidate experiment choices against an evaluator-only held-out truth with a known finite outcome distribution. It calculates expected utility exactly, without Monte Carlo variance, and reports decision regret, cost, and unsupported outcome mass. It is an experiment-planner benchmark, not a quantum-device benchmark.

This matches an important distinction in current quantum benchmark practice. The 2026 Quantum Optimization Benchmarking Library (QOBLIB) argues for model-independent problem instances, strong classical references, and reproducible reporting across methods and hardware; its ten problem classes and accompanying instances create a common comparison surface ([Nature Computational Science, 2026](https://www.nature.com/articles/s43588-026-00991-1); [live benchmark library](https://zib-aopt.github.io/QOBLIB/)). Open QBench provides an interoperable framework across hybrid, high-level and compiled layers for gate-based, photonic, and annealing systems ([Open QBench](https://quantum.psnc.pl/en/openqbench/)). QARCH-008 should complement these application benchmarks by evaluating whether our experiment-selection policy chooses useful measurements under model mismatch.

Relevant active-learning research also warns that informativeness alone can amplify model misspecification:
- [Tang, Sloman & Kaski, AISTATS 2026: Representative, Informative, and De-Amplifying](https://proceedings.mlr.press/v300/tang26d.html).

## Policies compared

The first implementation compares five strategies on the same candidate pool:

1. **Nominal EVSI/cost:** QARCH-005's top candidate under the nominal model.
2. **Nominal information gain/cost:** entropy-based selection that ignores direct downstream decision value.
3. **Robust worst-case EVSI/cost:** QARCH-006's top candidate across the declared model ensemble.
4. **Uniform random:** the exact expectation across all candidates, not a sampled estimate.
5. **Fixed first:** the first candidate in nominal declaration order, representing a one-measurement fixed-order baseline.

An evaluator-only oracle is also calculated. It knows the held-out true model and chooses the best candidate and posterior action rule. It is a reference for regret, not a deployable policy or a fair competitor with the same information.

## Evaluation contract

All planning models and the hidden evaluation model must share the same hypothesis labels/order, action labels/order, utility matrix and utility unit, measurement-cost unit, candidate IDs, outcome labels/order, and declared candidate costs. Prior probabilities and likelihood matrices may differ. Invalid or mismatched contracts fail closed.

To isolate **experiment selection** from action-policy differences, every deployable candidate policy uses the same equal-weight model-averaged posterior from the planning ensemble to select an action after observing an outcome. The uniform-random policy's reported results are averaged exactly over every candidate.

For each candidate, calculate its true marginal outcome probabilities from the evaluator-only model. Then choose the downstream action using only the planning model ensemble, and accumulate exact true expected utility. If the true model assigns positive probability to an outcome that every planning model assigns zero probability, the report records that probability mass as unsupported_outcome_probability. The planner uses its best prior action on that outcome, rather than fabricating a posterior. This models a fail-closed fallback; a production system may instead explicitly abstain.

Costs and utilities are reported separately. The harness does not convert seconds or dollars into arbitrary utility points. The oracle maximizes expected decision utility over either making no measurement or running one candidate; it does not automatically solve a cost-constrained multi-step experiment-budget problem.

## Frozen analytic counterexample

The fixture has two equiprobable hypotheses and two actions. Each action earns one utility point if the corresponding hypothesis is true.

Two candidate experiments each cost one declared measurement-cost unit:

- **A — nominally perfect:** perfectly distinguishes hypotheses in the nominal model, but is uninformative in the held-out true model.
- **B — moderately informative:** produces 80% discriminating outcomes under both the nominal model and the held-out truth.

The robust model ensemble contains a plausible alternate where A is uninformative. The exact expected results are:

| Policy | Selected experiment | Expected utility under held-out truth | Decision regret |
|---|---|---:|---:|
| Nominal EVSI/cost | A | 0.50 | 0.30 |
| Nominal information gain/cost | A | 0.50 | 0.30 |
| Robust worst-case EVSI/cost | B | 0.80 | 0.00 |
| Fixed first | A | 0.50 | 0.30 |
| Uniform random, exact expectation | — | 0.65 | 0.15 |
| Evaluator-only oracle | B | 0.80 | 0.00 |

The values are synthetic analytical expectations. They check that the benchmark can expose a failure mode by construction; they do **not** demonstrate a real-world improvement, hardware calibration, or quantum advantage. The fixture is intended to be frozen before policy implementations are modified in response to its outcomes. A broader suite must prevent overfitting to this one pattern.

## Benchmark acceptance gates

### Exact contract correctness

- The baked-in counterexample reproduces each expected policy, utility, regret, and cost within a predeclared numerical tolerance.
- Uniform-random results are the exact mean across candidates and do not depend on an RNG seed.
- Nominal EVSI and entropy-based policies are invariant to candidate order except under the documented lexical tie-break.
- Robust ranking rejects duplicate/missing nominal scenarios and mismatched utility or candidate contracts.
- Ground truth with an outcome unsupported by all planning models produces visible unsupported mass.
- A workload/hypothesis/action/utility/unit mismatch is a hard error, never a silently comparable result.

### Decision regret evaluation

Expand beyond the known counterexample into a frozen suite spanning:
- nominal-best winners that collapse under misspecification;
- stable moderate experiments under all plausible models;
- prior shift;
- altered noise and detector calibration;
- an out-of-ensemble hidden mechanism;
- sparse or rare outcomes;
- unequal experiment costs;
- cases where robustness hurts nominal utility;
- cases where every candidate is low-value and abstention/no-measurement is the right outcome.

Report mean and worst-case decision regret, expected utility, measurement cost, unsupported-outcome mass, nominal-regret cost of robustification, and policy rank flips. Give per-scenario results and aggregate uncertainty. Do not select a policy from one specially designed fixture.

### Baselines

Compare the same frozen tasks against nominal EVSI/cost, entropy-only selection, random choice, fixed order/sweeps, and a robust Bayesian experimental-design baseline where its assumptions match. Model calibration must be evaluated separately via QARCH-007 before any results from physical scenarios are treated as evidence.

### Physical and application integration

- Use Symtropy QSIM-001 only for the physical quantities covered by its valid model.
- A geometry perturbation becomes an optical likelihood only through a separately qualified and appropriately calibrated adapter.
- Keep synthetic, analytic, simulated, independently simulated, calibrated, and measured evidence classes distinct.
- Application-level quantum utility still requires matched classical baselines, accepted output quality, and complete end-to-end resource accounting, including setup, encoding, retries, readout, mitigation, post-processing, and verification.

## Proposed targets

- 100% agreement with frozen analytic fixtures.
- Zero false accepts on malformed model contracts and unsupported-outcome negative controls.
- Lower worst-case regret than nominal EVSI on preregistered model-mismatch scenarios, while reporting nominal performance lost.
- No worse decision quality than the strongest applicable baseline on held-out measured scenarios, with sample size, calibration, and uncertainty reported.
- Any claimed empirical improvement reproduced independently on a frozen suite.

These are proposed criteria, not achieved results or an industry standard.

## Current implementation boundary

The Rust module defines the exact finite evaluator and five unit tests are authored. The JSON reference fixture was parsed and its expected values independently recalculated. The code has not been compiled or executed in this environment. It does not ingest physical traces, prove an experiment model calibrated, model repeated adaptive experiments, or establish a quantum-computing advantage.
