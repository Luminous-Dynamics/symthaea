# Constraint-aware adaptive audit allocation: matched heuristic comparison

**Date:** 2026-10-29  
**Status:** Synthetic research benchmark; not production certification  
**Protocol:** 100 fixed-seed repeats; 8 independent Bernoulli risk cells; four allocation policies.

## Question

The preceding adaptive audit scheduler allocates the next batch to the cell with the largest upper bound on absolute model error. It reduced audit volume compared with uniform sampling, but the allocation rule was heuristic. This study asks whether sampling the widest confidence interval or the highest estimated marginal interval-width reduction is a better decision-aware allocation policy.

All policies use the same per-cell beta-mixture confidence sequence, same fixed support, same 90% familywise error target, and same per-repeat random streams. The model is admissible only when all cellwise upper error bounds fit the declared allowance; it is rejected if any cell's lower error bound exceeds the allowance; otherwise, the scheduler can terminate with an unresolved refusal if the per-cell cap is reached.

## Protocol

- Seed: 20261029; 100 repeats.
- Eight Bernoulli cell means: [0.015, 0.035, 0.075, 0.14, 0.27, 0.50, 0.73, 0.94], chosen to create heterogeneous variances.
- Risk range: [0, 1] in normalized units (not bits).
- Familywise delta: 0.10 across eight fixed cells; each cell receives delta/8.
- Per-cell pilot: 128 observations; batch: 128; cap: 8,192 per cell.
- Maximum allowed absolute model-cell error: 0.05.
- Planning-matrix conditions: exact model, +0.02 offset on the high-variance cell, +0.051 offset, and +0.07 offset.
- Strategies: uniform; largest active cell error upper bound (previous heuristic); widest active confidence interval; estimated marginal interval-width reduction (CI width / observations).

The simulator mean is used to generate samples and score whether an admission was unsafe, not to compute confidence intervals. All results are conditional on independent Bernoulli streams and complete fixed support.

## Results

### Correct model

| Strategy | Admission | Mean total observations | Median | P95 | Saved versus uniform |
|---|---:|---:|---:|---:|---:|
| Uniform | 100/100 | 21,760 | 19,968 | 31,744 | — |
| Largest error upper bound | 100/100 | 9,545 | 9,408 | 11,661 | 56.1% |
| Widest active CI | 100/100 | 9,545 | 9,408 | 11,661 | 56.1% |
| Marginal CI reduction | 100/100 | 9,545 | 9,408 | 11,661 | 56.1% |

The three adaptive strategies select the same cell sequence in this exact-model case under this seed: the highest-variance cell dominates the certificate width. This is a reminder that policy differences only appear when uncertainty structure and decision margins disagree.

### Safe offset

| Strategy | Admit | Unresolved refusal | Mean total observations | Saved versus uniform |
|---|---:|---:|---:|---:|
| Uniform | 93/100 | 7/100 | 35,707 | — |
| Largest error upper bound | 93/100 | 7/100 | 11,415 | 68.0% |
| Widest active CI | 93/100 | 7/100 | 11,473 | 67.9% |
| Marginal CI reduction | 93/100 | 7/100 | 11,473 | 67.9% |

### Near-budget and clear violations

| Condition | Strategy | Mean total observations | Reject | Unresolved refusal | Unsafe admission |
|---|---|---:|---:|---:|---:|
| True error 0.051 | Uniform | 65,536 | 0/100 | 100/100 | 0/100 |
| True error 0.051 | Largest error upper bound | 11,909 | 0/100 | 100/100 | 0/100 |
| True error 0.051 | Widest active CI | 15,284 | 0/100 | 100/100 | 0/100 |
| True error 0.051 | Marginal CI reduction | 15,284 | 0/100 | 100/100 | 0/100 |
| True error 0.07 | Uniform | 59,709 | 28/100 | 72/100 | 0/100 |
| True error 0.07 | Largest error upper bound | 9,919 | 28/100 | 72/100 | 0/100 |
| True error 0.07 | Widest active CI | 14,487 | 28/100 | 72/100 | 0/100 |
| True error 0.07 | Marginal CI reduction | 14,532 | 28/100 | 72/100 | 0/100 |

**Finding:** the largest-error-upper-bound heuristic was more sample-efficient than the two generic width-based variants for the two violation scenarios, while the observed reject/refuse split remained identical. The width heuristics spent more samples on noisy cells that were not as decision-critical. In the safe-offset scenario, the gain over those heuristics was small (about 58 observations per run).

False admissions were 0/100 across all methods and scenarios. With zero observed events in 100 repeats, the Wilson 95% upper limit is about 3.70%, so these runs do not demonstrate a low population false-admission rate. The near-budget case was refused unresolved, not positively rejected; audit savings in that case indicate cheaper refusal, not better detection.

## Literature comparison

This is not a direct implementation of published best-arm/threshold-identification methods. Related work defines the next external benchmark set:

- [An Anytime Algorithm for Good Arm Identification (JMLR 2026)](https://www.jmlr.org/papers/v27/24-0680.html) proposes APGAI for anytime threshold identification, with bounds on its error probability at arbitrary times and adaptive gains in detecting that no good arm exists.
- [Rate-optimal Design for Anytime Best Arm Identification (AISTATS 2026)](https://proceedings.mlr.press/v300/komiyama26a.html) presents Almost Tracking, a closed-form anytime sampling design with minimax-rate guarantees, and reports comparisons on synthetic and real datasets.
- [Balancing Performance and Costs in Best Arm Identification (NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/hash/b0dfe8236981ba84b4d98a7ed0426f0b-Abstract-Conference.html) explicitly balances observation cost against the penalty for a wrong recommendation.
- [Best Arm Identification with Possibly Biased Offline Data (UAI 2025)](https://proceedings.mlr.press/v286/yang25b.html) shows why adaptive algorithms need a known bias bound when using potentially biased offline evidence.

Official APGAI implementation: [clreda/apgai](https://github.com/clreda/apgai). External published code was not run in this environment, and these results are not a direct SOTA comparison.

## Paradigm shift

The goal should be to allocate evidence according to **the active decision constraint and its expected cost to resolve**, rather than a generic variance or uncertainty score. The current maximum-error-upper-bound heuristic remains the strongest prototype here. A better next step is an approximate sample-complexity allocation that estimates, for every unresolved constraint, its required observations at the target tolerance, then samples the largest unmet deficit. That policy must still use valid anytime confidence sequences and fail closed on a capped unresolved blocker.

The full scorecard should include total observation cost, P95 cost, admission rate on safe models, conclusive rejection rate on bad models, unresolved-refusal rate, false-admission interval, per-cell spend, and sensitivity to an unknown high-variance blocker. Report every metric at matched risk and stopping guarantees.

## Reproduction and evidence

- Source: decision_risk_constraint_aware_allocation.py
- Test: test_decision_risk_constraint_aware_allocation.py
- Raw results: DECISION_RISK_CONSTRAINT_AWARE_ALLOCATION_RESULTS_2026-10-29_100.json
- Focused tests: **3/3 passed** (including a 50-repeat fixed-seed regression); full 100-repeat comparison completed in 20.820 seconds.
- Source SHA-256: c0cf1be21604318b8781353702f598ab07ef606137dcf8ccd46af3cbbc24b7d9; Git blob a23d22cb67d5084726b7022ac0edd0a7077d1f36.
- Test SHA-256: 580198a8f0466acc14b994d0fda0ea30e3f8309ef25da086bba56645d524d9be; Git blob cd547fa0694dc6f435d64e5f461fe4f505a0b82a.
- Raw JSON SHA-256: f63627af72409e9b17a91a5a021c00fc6491a9f8670b10be5ddb800dba165728.

## Limitations

This is an 8-cell iid Bernoulli toy with 100 repeats and a 90% familywise confidence target. The audit risk units are synthetic. The allocation methods are heuristic; no optimality guarantee is proven. Shared audit bias, missing support, changed policy identity, arbitrary non-Bernoulli observations, and real-world SCM shifts are not covered. The benchmark does not compile or integrate the native Rust planner/dispatch gate. No external baseline implementation, production deployment evaluation, Rust build, remote CI pass, or SOTA claim is made.
