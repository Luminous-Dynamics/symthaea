# AOBENCH-017: matched anytime-valid inference method league

**Date:** 2026-11-15  
**Parent programme:** [AOBENCH-013 / #12197](https://github.com/Luminous-Dynamics/symthaea/issues/12197)  
**Method track:** [AOBENCH-017 / #12242](https://github.com/Luminous-Dynamics/symthaea/issues/12242)  
**Calibration harness:** [AOBENCH-013C / #12248](https://github.com/Luminous-Dynamics/symthaea/issues/12248), [PR #12250](https://github.com/Luminous-Dynamics/symthaea/pull/12250)  
**Experimental design:** [AOBENCH-011 / #12173](https://github.com/Luminous-Dynamics/symthaea/issues/12173)  
**Partial identification:** [AOBENCH-013 / #12206](https://github.com/Luminous-Dynamics/symthaea/issues/12206)

## Purpose

This document specifies the next executable research increment: a matched, preregistered comparison of sequential inference procedures and their stopping behavior.

The benchmark must answer two different questions without conflating them:

1. **Validity:** does a procedure maintain its declared error and coverage guarantees under its actual assumptions, including repeated monitoring and adaptive sampling?
2. **Efficiency:** among procedures that pass the validity gate, how much evidence, time, computation and intervention cost do they need to reach a correct decision?

A method that returns the full parameter space, abstains forever, or never opens an experiment must not win by obtaining nominal coverage at zero cost. A method that reaches a narrow interval by invalid optional stopping must not win on efficiency. Validity is a qualification gate; efficiency is compared only after that gate is passed.

This is a research protocol, not a claim that any candidate method has already been implemented, calibrated, or shown superior in AOBench.

## 1. Freeze the estimand and decision contract

Each confirmatory episode must freeze:

- estimand ID and exact population/arm definition;
- outcome support and units;
- intervention and allocation contract;
- target effect and effect direction;
- minimum practically meaningful effect, delta;
- equivalence region, if used;
- decision thresholds and allowed terminal actions;
- family-wise alpha and the registered query family;
- missingness, censoring, delayed outcome and noncompliance semantics;
- sample, cost, risk, compute and human-review budgets;
- candidate inference methods and their applicability conditions;
- episode generator, PRNG version, seed derivation and manifest digest;
- stopping and multiplicity rules;
- evaluator implementation and analysis digests.

Changing the estimand, target population, endpoint, effect threshold, or query family after outcomes are observed is a protocol deviation. The evaluator retains the attempt but does not count it as a preregistered confirmatory success.

### Decision semantics

For a benefit threshold delta > 0, a two-sided effect interval supports:

- Benefit only when its lower bound is strictly greater than +delta.
- Harm only when its upper bound is strictly less than -delta.
- No material effect only when the entire interval lies inside [-delta, +delta], provided this equivalence claim is supported by the registered procedure.
- Otherwise, Inconclusive.

A point estimate beyond delta is not sufficient. An interval that intersects a decision boundary is not a decisive result.

At the exact boundary, qualification concerns false decisive claims, not an 80% power target. Power targets apply to preregistered alternatives separated from the boundary, such as 1.5 delta and 2 delta. Near-boundary power and required sample size remain part of the reported power curve.

## 2. Separate the policy layer from the inference layer

Use a factorial experiment rather than comparing opaque agent systems.

### Allocation / experiment policies

- Fixed balanced allocation.
- Random allocation under the declared assignment probabilities.
- Greedy expected information gain per cost.
- Decision-focused expected value of sample information per cost.
- Variance-adaptive or balanced-power allocation where its assumptions hold.
- Partial-identification / max-potency experiment selection.
- Expert- or classical-design reference.
- Evaluator oracle allocation as an upper bound only.

### Inference procedures

- Current Bernoulli Hoeffding alpha-spending implementation.
- Telescoping alpha-spending schedule.
- Stitched or mixture confidence sequences.
- Empirical-Bernstein confidence sequences for bounded outcomes.
- Betting-based confidence sequences for bounded means.
- Bernoulli likelihood-ratio / e-process procedures.
- Regression-adjusted anytime-valid inference on a separately declared covariate-bearing randomized lane.
- Naive repeated fixed-time monitoring as a deliberately invalid negative control.
- Always-defer / always-wide as a degeneracy control.
- Oracle interval and oracle stopping as unattainable upper bounds.

Cross each allocation policy with only inference procedures whose mathematical assumptions cover that allocation and data process. Publish an applicability matrix. An incompatible method is not silently treated as a failed statistical test; it must be marked inapplicable or must return an unqualified result.

The method-only comparison should hold the allocation path fixed. The joint policy comparison should compare allocation-plus-inference stacks and report their assumptions and budgets separately.

## 3. Exact alpha accounting

For Q registered queries and two arm-specific mean sequences, a transparent family-wise allocation is:

\[
\alpha_{\mathrm{arm,query}} = \frac{\alpha_{\mathrm{FWER}}}{2Q}.
\]

Two valid but generally different time allocations are:

**Telescoping allocation**

\[
\alpha_n = \frac{\alpha_{\mathrm{arm,query}}}{n(n+1)},
\qquad
\sum_{n=1}^{\infty}\alpha_n
= \alpha_{\mathrm{arm,query}}.
\]

**Inverse-square allocation**

\[
\alpha_n = \frac{6\alpha_{\mathrm{arm,query}}}{\pi^2 n^2},
\qquad
\sum_{n=1}^{\infty}\alpha_n
= \alpha_{\mathrm{arm,query}}.
\]

These schedules must be evaluated as separate procedures. The telescoping allocation is not a free tightening of an existing interval: it changes the finite-sample allocation of error across looks.

For the arm-wise Hoeffding construction with bounded outcomes in [L,U], the radius at qualified sample count n is:

\[
r_n = (U-L)
\sqrt{\frac{\log(2/\alpha_n)}{2n}}.
\]

The arm interval is the sample mean plus or minus this radius, clipped to the declared support. For a treatment-minus-control contrast, subtract the control interval from the treatment interval. This is a transparent conservative construction, not a claim of optimality.

For every method, report:
- exact alpha allocation across arms, queries and looks;
- whether its guarantee is finite-sample or asymptotic;
- whether it is marginal or simultaneous;
- whether it remains valid under predictable adaptive allocation;
- whether the contrast is direct or obtained by subtracting arm intervals;
- all additional assumptions and any variance, nuisance or model estimation.

Do not compare a direct contrast procedure with a Bonferroni arm-wise construction without identifying that the estimand construction as well as the confidence-sequence method has changed.

## 4. Matched simulation and common random numbers

A comparison is only informative when methods face the same underlying difficulty.

### Fixed-allocation method league

For a given episode, create deterministic arm-specific outcome streams indexed by episode ID, world ID, arm and observation index. Every method receives the same qualified observation prefix for a given arm and sample count.

This supports paired comparisons of:
- interval width at a common sample count;
- first valid stopping time;
- false decisive claims;
- final decision and abstention;
- resource use at stopping.

Do not let execution order, stratum order, parallel scheduling or method-specific random-number consumption change the observations.

### Adaptive-allocation league

When allocation policies differ, use a simulator with explicit potential-outcome or structural-equation semantics. The episode's underlying exogenous randomness is keyed independently of the candidate policy. The policy only observes the outcome for the assignment actually made.

The evaluator retains unobserved potential outcomes and hidden world parameters. They are never exposed as agent-visible observations.

This makes the resulting policy comparison paired by world rather than falsely claiming that every adaptive policy saw an identical realized trajectory.

### Information boundary

Represent agent-visible observations, evaluator-only truth, allocation receipts and qualification adjudications as distinct typed records. The policy must not receive:

- hidden arm probabilities or potential outcomes;
- future random draws;
- evaluator-only stratum labels;
- post-episode truth or scorer fields;
- unassigned-arm outcomes;
- manifest metadata that uniquely reveals the hidden effect.

Add serialization tests proving that evaluator-only values are absent from the agent-facing task and observation payloads. A prose comment that a field is hidden is not a security boundary.

## 5. Matched method requirements

| Method family | Candidate | Qualification conditions |
|---|---|---|
| Simple bounded | Existing Hoeffding spending | Bounded outcomes; declared predictable allocation and mean estimand |
| Simple bounded | Telescoping spending | Same contract; independent schedule and evaluator implementation |
| Mixture / stitching | Howard-style mixture or stitched CS | Match bounded/sub-Gaussian assumptions and tuning to the episode |
| Variance-adaptive | Empirical-Bernstein CS | Bounded outcomes; method-specific assumptions and predictable process |
| Betting | Betting CS | Bounded outcomes; exact wealth process and inversion independently reproduced |
| Bernoulli-specific | Likelihood-ratio / e-process CS | Bernoulli likelihood and composite-null treatment-effect construction |
| Regression-adjusted | AVLM-style procedure | Randomized linear/covariate-bearing task family; pre-treatment covariates and correct asymptotic/exact guarantee labels |
| Negative control | Repeated fixed-time interval | Deliberately invalid under arbitrary optional stopping; must be shown to fail on stress controls |
| Degenerate control | Always-wide / always-defer | Must not receive coverage or decision-quality credit merely for refusing to decide |

The AVLM lane is not a direct comparator on a bare Bernoulli task without covariates. Its exact, parametric and heteroscedastic-robust asymptotic variants must be distinguished. A method's paper-level result does not imply that a local implementation inherits the guarantee.

## 6. Statistical controls and challenge families

At minimum, freeze separate strata for:

1. Null effects.
2. Equivalence-interior effects.
3. Effects just inside the decision margin.
4. Effects just outside the decision margin.
5. Alternatives at 1.5 delta and 2 delta.
6. Positive and negative effects.
7. Low-variance and high-variance bounded outcomes.
8. Rare-event and imbalanced Bernoulli outcomes.
9. Multiple registered queries with a family-wise error contract.
10. Predictable adaptive allocation.
11. Covariate imbalance and valid pre-treatment covariate adjustment.
12. Missing, delayed and censored observations.
13. Invalid assignment, delivery, isolation or measurement receipts.
14. Interference, carryover and common-cause failures.
15. Drift and nonstationarity.
16. Model misspecification and hidden mechanisms outside the registered model class.
17. Partial-identification tasks where multiple worlds support different causal explanations but a useful decision bound may still be available.

Separate identifiable, partially identifiable and non-identifiable tasks. A confidence sequence for a declared mean does not repair confounding, interference, invalid randomization, informative missingness or a misspecified estimand.

Every invalid evidence control needs a condition-removed positive twin. Rejected or unavailable evidence must never enter an inferential update. A valid statistical interval on an invalidly assigned or measured sample is not a qualified causal claim.

## 7. Metrics: report a vector, not a blended score

### Validity gates

- Empirical simultaneous coverage across every look and registered query.
- Final-look coverage, reported separately.
- Family-wise false decisive-action rate.
- Null false-benefit and false-harm rates.
- False equivalence rate.
- Coverage conditional on the declared valid evidence process.
- Rate of accepted claims narrower than the evaluator-recomputed sequence.
- Number of statistical updates from rejected or unavailable evidence (target zero).
- Method applicability / assumption-violation detection.

For coverage, the episode-level outcome is whether the true target was covered at every eligible look. Do not count each timestep as an independent replicate. Zero-qualified-observation episodes remain in process, abstention and decision denominators, but cannot inflate a coverage numerator or denominator.

### Efficiency frontier

- Power at every preregistered separated alternative.
- Full power curve over effect distance from the decision margin.
- Mean, median and quantile stopping sample counts.
- Expected and tail intervention cost, risk and compute.
- Interval width at matched sample and cost budgets.
- Time / sample count to a correct decisive outcome.
- Correct inconclusive and correct-defer rates.
- Decision regret against a clearly defined constrained oracle.
- Realized versus expected information gain.
- Unnecessary continuation after a robust decision is already available.

### Robustness and transfer

- Worst-family and worst-effect-size validity.
- Calibration across variance and outcome families.
- Performance on held-out mechanism families and combinations of assumption violations.
- Family-wise versus marginal error.
- Missingness, drift, interference, allocation and model-adequacy stress performance.

A single average cannot compensate for a failed null-control, undercoverage, invalid evidence update or false decisive action.

## 8. Monte Carlo precision and comparison rules

Monte Carlo uncertainty is part of the result.

For a rough planning calculation for a binomial operating-characteristic rate p and absolute half-width e:

\[
N \approx \frac{1.96^2p(1-p)}{e^2}.
\]

Near a 5% or 95% rate, estimating the rate to roughly one percentage point takes on the order of 1,825 independent episodes under this approximation. A half-percentage-point target takes roughly four times as many. Freeze the actual episode count with exact Wilson or another preregistered interval method; these values are planning heuristics, not acceptance evidence.

Use:
- independent episode-level confidence intervals for each rate;
- paired episode comparisons for methods sharing the same frozen streams;
- paired bootstrap or another justified paired analysis for width and stopping-cost differences;
- generator-family holdouts and family-level uncertainty, rather than timestep pseudo-replication;
- separate seed blocks for reproducibility and confirmatory replication.

When the number of generator families is too small for reliable cluster inference, publish each family result and the worst-family result rather than implying a precise population-of-families estimate.

Do not tune method hyperparameters on the sealed confirmatory manifests. Freeze the method version, alpha schedule, policy, seed list, episode count and analysis code before running confirmatory calibration.

## 9. Qualification and superiority gates

Proposed targets, not achieved results:

1. Exact algebraic and deterministic contract fixtures: 100% correct.
2. Nominal 95% simultaneous coverage: empirical coverage assessed against a preregistered tolerance and with an uncertainty interval, overall and per family.
3. False decisive-action rate: at or below the preregistered 5% target on valid null and threshold-straddling controls, with uncertainty reported.
4. Separated alternatives: at least 80% power at preregistered alternatives such as 1.5 delta and 2 delta, under sample budgets frozen by power/precision analysis.
5. Rejected/unavailable assignment, delivery, isolation or observation evidence: zero accepted statistical updates.
6. Evaluator replay: zero claims accepted with intervals narrower than evaluator recomputation.
7. Degenerate agents: zero coverage credit without qualified looks; always-defer cannot satisfy discovery or decision-utility targets.
8. Superiority: no method is declared better until it passes validity gates and improves the preregistered efficiency frontier at matched decision quality, with uncertainty on the paired improvement.

The 10–15% cost reduction target in #12197 remains a hypothesis for the broader adaptive-design programme. It should not be silently applied to every method family or used to override a validity failure.

## 10. Paradigm-shifting extension: choose experiments by decision impact

After the method league is calibrated, combine it with #12173 and #12206.

A next-experiment policy should consider at least three quantities:

- expected statistical information from the experiment;
- expected change in the decision under the posterior or identified set;
- worst-case change in the robust decision across admissible models.

For partially identified tasks, compare expected information gain with max-potency: the guaranteed reduction in target-effect bound width before outcomes are seen. Some experiments reveal many irrelevant relationships; others tighten the policy-relevant bound.

A useful decision rule is to continue only when an affordable, qualified experiment has sufficient expected decision value or robust bound-tightening value to justify its cost and risk. Otherwise, return the best supported action or Inconclusive. An oracle policy is a diagnostic upper bound, not an agent baseline.

## 11. Execution sequence

1. Keep #12216's Bernoulli Hoeffding method as the transparent baseline.
2. Finish exact-head runtime qualification and run the frozen #12250 manifest; publish per-stratum traces and uncertainty.
3. Add telescoping alpha-spending as a separate implementation and reproduce its sum-to-alpha contract exactly.
4. Add one empirical-Bernstein/betting method using a published reference construction; independently test its wealth/coverage logic before connecting it to the benchmark.
5. Add one stitched/mixture method as a separate implementation, not as an alias for betting.
6. Build the common-stream evaluator and paired comparison report.
7. Add family-wise multiple-query episodes and a multiple-testing negative control.
8. Add an independent covariate-bearing randomized lane for AVLM-style procedures.
9. Connect allocation policies and partial-identification / max-potency experiments.
10. Add sealed held-out generators and an independent reproduction protocol.

Each method increment must include exact assumptions, a small deterministic oracle fixture, a negative control, a positive twin, trace replay and an explicit claim boundary. Do not wait for a long CI workflow to design the next independent track, but do not mark the code qualified until exact-head runtime evidence exists.

## 12. External research anchors

- Howard, Ramdas, McAuliffe & Sekhon (2021), *Time-Uniform, Nonparametric, Nonasymptotic Confidence Sequences*, Annals of Statistics. https://arxiv.org/abs/1810.08240
- Waudby-Smith & Ramdas (2024), *Estimating means of bounded random variables by betting*, JRSS B. https://doi.org/10.1093/jrsssb/qkad009
- Lindon, Ham, Tingley & Bojinov (2026), *Anytime-Valid Inference in Linear Models with Applications to Regression-Adjusted Causal Inference*, JASA. https://doi.org/10.1080/01621459.2026.2692052
- Molitor & Gold (2025), *Anytime-Valid Inference in Adaptive Experiments: Covariate Adjustment and Balanced Power*. https://arxiv.org/abs/2506.20523
- Causal-Discovery-Bench (2026). https://arxiv.org/abs/2611.09934
- Petri Bench v2. https://petri-labs.org/bench/

## Claim boundary

This protocol defines a fair comparison design. It does not assert that any method has passed it. A method's theoretical time-uniform guarantee applies only under its mathematical assumptions; simulation calibration estimates finite-sample behavior only for the frozen generators. Neither substitutes for authentic allocation, intervention delivery, isolation, measurement qualification, causal identification or external validation.
