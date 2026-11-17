# EGDDM ShiftBench D6.17.2.2.4-alpha — Exact finite stochastic adaptive policy search

**Date:** 2026-11-17  
**Status:** bounded finite-tree enumerator prototype; deterministic-limit known-answer fixture; Rust execution pending  
**Related:** issue #12683, programme issue #12630, stochastic policy scorer PR #12686, shared-latent model PR #12689.

## 1. Why a search oracle is the next step

D6.17.2.2.1 evaluates a submitted policy tree. That is necessary but insufficient: a policy generator still needs a trustworthy comparator. Learned/amortized or receding-horizon policies should be measured against an exact oracle wherever the tree is small enough to enumerate.

The field already has strong work on this:
- GO-CBED learns non-myopic sequential intervention policies targeted at a causal quantity of interest, rather than global causal-model information gain. TMLR 2026: https://uq.engin.umich.edu/causal-learning.html
- Cheng and Huan formulate sequential Bayesian experimental design as a joint stopping/design MDP, with learned value/policy approximations. https://arxiv.org/abs/2509.21734
- Wen et al. study robust sequential experimental design under model misspecification for A/B testing and contextual/dynamic settings, ICML 2026: https://proceedings.mlr.press/v306/wen26i.html
- NIST is developing adversarial evaluation probes and machine-readable evidence trails for agent workflows: https://www.nist.gov/programs-projects/building-evaluation-probes-agentic-ai

The EGDDM search work does not claim to supersede those learned or large-scale methods. It adds a reproducible finite-instance oracle against which they can be tested.

## 2. Search contract

The Rust binary stochastic-policy-search.rs uses the existing D6.17.2.2.1 scorer for every candidate policy. It does not reimplement posterior risk, hard-gate logic, likelihood propagation, or cost optimization.

Given:
- an evaluator-approved option registry and per-option outcome alphabet;
- an evaluator-owned action list, bounded prior, joint kernels, safety labels, costs and decision cutoff;
- a candidate maximum depth;
- an evaluator-owned maximum generated-policy cap;

it enumerates all policy trees in the registered finite class:
1. Every node may stop with one of the registered actions, including read-only defer.
2. Every acquisition node may select any option not already used on its current path.
3. Every registered outcome must have exactly one child branch.
4. Each outcome branch may independently choose its own next measurement or stop.
5. The scorer counts root depth as zero, so the maximum acquisition count on a path equals the candidate max-depth value; a terminal stop is allowed at that depth.
6. Trees above the scorer's 64-node limit are excluded from the legal search class and counted.
7. Each tree is passed to D6.17.2.2.1; policy feasibility is determined by its robust-risk and budget checks.

This is exhaustive enumeration of finite adaptive observation trees in the declared bounded class. It is not exhaustive over unbounded time, continuous measurement settings, stochastic actions, arbitrary POMDP state or policies larger than the registered limits.

The search uses decision-cutoff timestamps for generated nodes. The alpha searches acquisition order and outcome-adaptive structure, not fine-grained wall-clock scheduling. Real measurement/queue/retry timing must be assessed through D6.17.2.3 run receipts.

## 3. Lexicographic objective

The ranking is fixed and published:

1. Feasible before infeasible: zero scorer risk violations and zero over-budget leaves.
2. Maximize lower bound of justified-action coverage over the registered product ambiguity set.
3. Maximize nominal justified-action coverage.
4. Minimize worst-case expected acquisition cost.
5. Minimize nominal expected acquisition cost.
6. Minimize worst-case path cost.
7. Deterministic canonical-tree ordering to break exact ties.

A Pareto frontier is reported for feasible trees across robust lower coverage, nominal coverage, worst-case expected cost and worst path cost. The frontier output is capped at 50 entries; the full frontier count and truncation flag are retained.

The submitted candidate policy is scored separately and compared to the best enumerated policy. Exact regret is reported only if the policy generator completed the entire declared tree class. It reports robust/nominal coverage shortfalls and worst/nominal/path-cost excesses relative to the best tree.

## 4. Search completeness and failure semantics

An evaluator-owned cap controls the maximum generated policies (1–50,000). Candidate input cannot raise this cap.

- If every tree in the registered finite class is enumerated, search_complete=true and the output may claim exact_optimum_over_registered_finite_tree_class.
- If the generator reaches the cap or any subtree enumeration is truncated, search_complete=false, optimality_claim=none_search_incomplete, and exact_regret=null.
- The best partial policy may still be shown as a candidate, but it is not called optimal.
- Node-limit exclusions are counted. The exact optimum claim is restricted to the legal tree class whose trees fit the evaluator's 64-node limit.
- Every generated tree is scored by the authoritative stochastic policy evaluator. A generated tree failing scorer validation is an evaluator error, not silently dropped.

## 5. Deterministic-limit known answer

The first fixture reuses the four-hypothesis, two-gate D6.17.2.1 deterministic scenario and its submitted adaptive tree.

Expected finite class size:
- two stop-only policies (release, read-only defer);
- with one acquisition remaining, 10 policies over the two options;
- with two acquisition levels and adaptive branches, 74 total policies.

The exact best policy must preserve the D6.17.2.1 deterministic baseline:
- robust lower justified-action coverage: 5,000 bps;
- nominal justified-action coverage: 6,000 bps;
- nominal expected cost: 1,300 micro-USD;
- worst-case expected cost: 1,420 micro-USD;
- worst-case path cost: 1,500 micro-USD.

The submitted baseline is the same adaptive tree, so the expected regret vector is zero on these metrics. This is a deterministic-limit parity control, not a new performance result.

## 6. Comparison protocol

Compare candidate generators under the same evaluator and task instances:
- full revalidation and D6.15 minimum-sufficient acquisition;
- nominal greedy EVSI and robust one-step EVSI;
- GO-CBED-style non-myopic causal target design;
- Cheng–Huan joint design/stopping policy;
- robust sequential A/B design under misspecification;
- learned/amortized stochastic policy;
- receding-horizon robust policy;
- Fréchet-only robust joint-event fallback;
- independence multiplication as a deliberate unsafe negative control;
- exact finite search for small instances.

Report:
- exact regret when enumeration is complete;
- certified gaps (not heuristic regret) when future branch-and-bound or relaxations are added;
- robust lower/nominal/upper coverage and unsafe-action mass;
- risk, hard-gate and budget violations;
- worst/nominal expected cost and path cost;
- search CPU/time/memory and candidate count;
- end-to-end execution cost from D6.17.2.3 receipts;
- calibration and dependence validity on held-out common-cause families.

## 7. Proposed targets — not achieved

- 100% exact agreement with independent enumeration on registered small instances.
- Zero risk or hard-gate violations among policies labelled feasible.
- Zero exact-optimality claims after any search truncation.
- Exact zero regret for the baseline policy on deterministic parity fixtures.
- On learned/heuristic policies, median cost regret <=1% versus the exact oracle at matched robust coverage on a pre-registered small-instance suite.
- On larger problems, provide valid lower/upper objective bounds and a certified optimality gap rather than describing a capped best-so-far candidate as optimal.

## 8. Limitations and next steps

The current search evaluates D6.17.2.2.1's product-set ambiguity model. It does not yet search shared-latent/non-rectangular policies from D6.17.2.2.3. It also treats option costs as fixed; D6.17.2.3 receipts will supply measured execution costs later.

Next:
1. Add stochastic common-cause fixtures to search and compare product-set versus shared-latent rankings.
2. Integrate D6.17.2.2.3 risk probes into the policy scorer and search oracle.
3. Add branch-and-bound with admissible bounds for larger tree spaces.
4. Compare amortized policy learning and goal-oriented stopping against exact small-instance regret.
5. Add run receipt joins and paired-seed cost-regret evaluation.

## 9. Claim ceiling

The prototype enumerates finite adaptive trees within the registered option, depth, node and evaluator candidate-cap limits, and scores each using D6.17.2.2.1. It makes an exact-optimum claim only when enumeration completes. It does not optimize outside that finite class, validate empirical calibration, handle cross-hypothesis shared latent ambiguity, model stochastic execution cost, prove hypothesis completeness, or authorize actions.

Rust compilation, tests, rustfmt, Clippy and CI have not been executed here. Keep the branch draft until exact-head toolchain validation and independent enumeration replay.
