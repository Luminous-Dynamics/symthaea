# SCI-ADAPTIVE-TEST-ALLOCATION-001 — Cost-aware, statistically valid adaptive evaluator testing

**Status:** Research protocol and proposed contract. No adaptive scheduler or empirical result is implemented by this document.

**Tracking:** [Issue #9276](https://github.com/Luminous-Dynamics/symthaea/issues/9276).  
**Depends on:** SCI-INTERVENTION-QUALITY-001 / [PR #9274](https://github.com/Luminous-Dynamics/symthaea/pull/9274).  
**Principle:** adaptive screening may reduce cost; it must not silently weaken confirmatory evidence.

## 1. Why this is the next frontier

The candidate pool built by the intervention-quality layer can be large and expensive to execute. Exhaustive execution remains the benchmark reference, but may be too costly for repeated monitoring or rapid iteration.

Recent work makes adaptive evaluation credible, while also clarifying its limits:

- **ATLAS** (ICML 2026) applies item-response theory and Fisher-information-guided item selection. In its published experiments, it reports matching whole-bank HellaSwag ability estimates using 41 of 5,600 items (0.157 MAE), with reductions up to 90% across evaluated benchmarks. These item banks and accuracy-style endpoints are not equivalent to adversarial evaluator-fault testing. [PMLR](https://proceedings.mlr.press/v306/li26fx.html)
- **Computerized adaptive testing for medical LLM benchmarking** (npj Digital Medicine, 18 May 2026) reports correlation 0.988 with full-bank results using 1.3% of items and Spearman rank preservation in its particular medical item bank. The authors explicitly frame CAT as screening/monitoring, not clinical validation. [Nature](https://www.nature.com/articles/s41746-026-02671-w)
- **Active Evaluation Acquisition for Efficient LLM Benchmarking** (ICML 2025) uses dependencies among evaluation examples and a learned acquisition policy to reduce the number of executed prompts while reconstructing benchmark outcomes. [PMLR](https://proceedings.mlr.press/v267/li25bp.html)
- **Adaptive Learn-then-Test** (ICML 2025) uses e-processes for sequential, data-dependent testing with finite-sample statistical guarantees and early stopping. [PMLR](https://proceedings.mlr.press/v267/zecchin25a.html)

The lesson is not “run 1% of the tests.” The lesson is to estimate where each next test adds information, preserve coverage of rare but important failure modes, and keep adaptive inference valid.

## 2. Proposed architecture

The candidate-bank flow is:

1. An intervention-quality-qualified candidate bank is frozen.
2. Preregistered hard safety/control tests are always executed.
3. A randomized exploration reserve samples non-hard-gate strata.
4. A frozen information allocator selects additional candidates under a budget.
5. Selection receipts record propensities before outcomes are observed.
6. Anytime-valid screening estimates guide further sampling or stopping.
7. The allocator is frozen before an untouched confirmatory lockbox is executed.

This is a two-stage measurement design:

1. **Adaptive screening** estimates which judge/failure-mode pairs need further investigation, under a declared budget.
2. **Confirmatory evaluation** runs on an untouched, precommitted holdout that never influenced the adaptive policy.

The adaptive output is a screening decision, not proof that the judge is qualified.

## 3. Candidate bank and frozen selection state

The allocator consumes candidate cases admitted by SCI-INTERVENTION-QUALITY-001. It must not bypass the semantic-validity gate.

Every candidate has:
- candidate ID and exact input digest;
- target failure mode and transformation family;
- semantic-cluster identity and semantic relation;
- intervention-quality adjudication state;
- expected execution cost and resource budget;
- severity / mandatory-control classification;
- estimated difficulty and discrimination features, if calibrated;
- bank membership: hard control, exploration pool or confirmatory lockbox.

The pre-run plan commits to:
- candidate-bank digest and item-feature digest;
- exact allocator source revision and policy digest;
- policy parameters, prior/posterior initialization and update rule;
- RNG algorithm and seed commitment;
- per-stratum coverage floors;
- exploration probability and budget;
- stopping rule and inference family;
- fixed confirmatory lockbox commitment;
- model, prompt, rubric, judge and reference-label revisions.

The plan must be committed and externally anchored before the first outcome is observed.

## 4. Selection policy candidates

Compare several strategies under identical candidate banks and budgets:

1. **Exhaustive:** execute every eligible candidate; reference result for evaluation cost/measurement-error comparisons.
2. **Uniform random:** simple baseline with known inclusion probabilities.
3. **Stratified random:** random within transformation family, severity and failure-mode strata.
4. **Greedy coverage / set cover:** maximize new semantic clusters and failure modes subject to costs.
5. **IRT / Fisher information:** use only where item parameters are calibrated on a defensible population and the latent trait has a coherent interpretation.
6. **Expected information gain:** choose the candidate with highest expected reduction in uncertainty about a preregistered judge/failure-mode state, divided by expected cost.
7. **Contextual bandit:** Thompson sampling or UCB with a frozen context, update rule, reward and exploration floor.
8. **Hybrid:** mandatory hard gates, stratified exploration, adaptive allocation, then confirmatory lockbox.

The reward cannot simply be “find more failures.” Otherwise the policy can overselect noisy or semantically ambiguous cases. The objective must incorporate valid unique test yield, uncertainty reduction, failure-mode coverage, false-alarm cost and execution cost.

A candidate's expected information gain must be conditioned on a predeclared failure model, not reverse-engineered from a single judge's observed answer after the fact.

## 5. Propensities, exploration and off-policy estimation

For each round, the receipt records:
- the full eligible candidate-set digest;
- candidate scores used by the selector;
- each candidate's selection probability before observing its outcome;
- selected candidate IDs and draw order;
- RNG / policy commitments;
- realized execution, timeout and cost outcomes.

A nonzero selection probability is required for every stratum about which the report intends to make a population-level estimate. If the design assigns zero probability to a stratum, that stratum is unidentifiable from the adaptive sample. The report must say so.

Where inclusion probabilities support the estimand, report:
- Horvitz–Thompson and/or Hájek estimates;
- inverse-propensity weight distribution;
- effective sample size, defined as (sum of weights)^2 divided by sum of squared weights;
- maximum weight and concentration;
- positivity violations;
- unweighted executed-sample results alongside weighted estimates.

Weight clipping or trimming must be preregistered and reported as a changed estimand / sensitivity analysis, never as an invisible numerical fix.

The randomized exploration reserve is proposed at a minimum 10% of eligible non-hard-gate execution budget for initial tests. This is an engineering starting point, not a universal optimum.

## 6. Statistical validity under repeated looks

Ordinary fixed-sample p-values or confidence intervals cannot be reused naively after repeated outcome-driven selection and optional stopping.

V1 should select one explicit inferential regime:
- fixed-horizon randomized sampling with a preregistered estimator and variance model; or
- a justified sequential design using e-processes / confidence sequences with an explicit filtration and sampling assumptions.

Task, judge and semantic-cluster dependence must be represented. Candidate rows sharing one underlying semantic task are not independent observations merely because their input hashes differ.

The report must include:
- stopping boundary and the look at which it fired;
- number of interim looks and candidates examined;
- coverage and interval width;
- type-I error from repeated null simulations;
- power / detection rate under seeded effects;
- failure to stop within the budget;
- sensitivity to distribution shift and dependence misspecification.

## 7. Non-negotiable coverage constraints

Adaptive efficiency must not create blind spots:

- execute 100% of preregistered hard safety/critical-control items;
- meet a minimum per-failure-mode coverage floor;
- retain randomized exploration among eligible non-hard-gate strata;
- reserve a confirmatory lockbox not visible to the allocator;
- report every unselected candidate and its inclusion probability;
- do not let evaluator outcomes leak back into generation or lockbox labels;
- freeze the final allocator and stop policy before confirmatory execution.

An adaptive pre-screen cannot clear a system for safety-critical deployment. It can prioritize follow-up testing and detect regressions efficiently.

## 8. Baselines and target metrics

All comparisons use identical candidate-bank versions, judge revisions, prompts, rubrics, model settings, execution budgets and reference snapshots.

### Efficiency / fidelity

- median and p95 cost, wall-clock and call count;
- cost reduction against exhaustive execution;
- absolute error in reconstructed full-bank fault rate;
- rank correlation and top-k inversions relative to exhaustive judge rankings;
- interval coverage and width;
- propensity ESS and weight concentration.

### Fault coverage / risk

- failure-mode recall by severity;
- detection rate on seeded faulty judges;
- false-positive rate on neutral controls;
- mandatory-control completion;
- rare-stratum miss rate;
- performance under shifted task/failure distributions.

### Initial qualification targets

Proposed targets to preregister after pilot variance and sample-size analysis:

- at least 50% median cost reduction versus exhaustive eligible-bank execution;
- full-bank fault-rate reconstruction absolute error at most 2 percentage points on held-out synthetic and empirical panels;
- Spearman rank correlation at least 0.98 against exhaustive rankings;
- no more than 2% top-k rank inversions under the stated comparison margin;
- 100% mandatory hard-control execution and 100% high-severity stratum coverage;
- exploration of at least 10% of eligible non-hard-gate budget, unless a stronger statistical design is justified;
- sequential null simulations maintain the predeclared type-I error bound (proposed 0.05);
- no confirmatory lockbox leakage.

These targets are not measured results, and no published CAT headline should be interpreted as a guarantee for evaluator-bias testing.

## 9. Hostile controls

Reject:
- missing/duplicate/out-of-bank selections;
- missing or zero propensities for estimands that require positivity;
- probabilities that do not sum to the declared selection distribution;
- altered policy revision, parameters, seed commitment or candidate-bank digest;
- post-outcome propensity mutation;
- budget overrun hidden by dropping timeouts;
- omission of mandatory hard controls;
- lockbox candidate selected during screening;
- outcome leakage or use of unplanned stopping rules;
- impossible or invalid cost/resource receipts.

A structurally valid adaptive receipt may reveal a poor allocator. It must not be conflated with a successful measurement outcome.

## 10. Claim boundary

A verified selection receipt proves only that declared selections and probabilities satisfy the committed plan. It does not prove true randomization, hidden-state integrity, judge execution authenticity, exchangeability, representativeness, population generalization or real-world safety.

The final report must state the target population and estimand, assumptions, positivity support, effective sample size, distribution-shift limits and what remains unidentifiable.
