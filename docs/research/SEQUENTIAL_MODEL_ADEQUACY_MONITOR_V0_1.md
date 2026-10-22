# Finite-Horizon Sequential Model-Adequacy Monitoring v0.1

**Status:** research baseline added in a stacked draft PR; exact-head Rust validation is pending. No empirical performance or SOTA claim is made.

This follow-on builds on the calibrated batch audit in PR #8439 and joint prior/model/noise robust acquisition in PR #8361. It adds a policy for repeated audits; it does not replace either component.

## 1. Why sequential monitoring is a different problem

A batch conformal p-value at level 0.05 is not permission to run the same test indefinitely at level 0.05. Repeated thresholding inflates the probability of at least one false alarm. A detector that continually adds post-change observations to its reference set can also dilute evidence for a real shift.

This baseline keeps the null calibration reference fixed and allocates an explicit fraction of the total error budget to each audit. It prioritizes a small, auditable contract over adaptivity or power.

## 2. Contract and bound

Let (H) be the declared maximum number of audit batches and (alphain(0,1)) the total family-wise false-alarm budget. At audit (t), compute a conservative batch-level conformal rank p-value against the frozen calibration scores and alarm only if:

[
p_t leq alpha/H.
]

If every (p_t) is marginally super-uniform under the clean null, then by the union bound:

[
Pr(	ext{one or more false alarms in }1,ldots,H)
leq sum_{t=1}^{H}Pr(p_tleqalpha/H)
leq alpha.
]

The bound does **not** require independence among alarms. It does require each marginal p-value to be valid. That means the calibration scores and each fresh audit score must obey the same exchangeability/null protocol with the same predeclared score function. A changed target distribution, adaptive score tuning, model refit using audit data, or audit selection that breaks this protocol invalidates the stated claim. Start a new epoch and collect fresh calibration data when the model, scorer, or target population changes.

This is a finite-horizon Bonferroni guarantee, not an e-process, not a conformal test martingale, and not an anytime-valid conditional guarantee. It is intentionally conservative.

## 3. Implementation

`sequential_adequacy.rs` provides:

- a monitor with a fixed reference, total alpha and maximum audit count;
- an explicit resolution precondition: the smallest attainable rank p-value (1/(n+1)) must be no larger than (alpha/H);
- a typed state machine: `Monitoring`, `Alarmed`, or `HorizonExhausted`;
- a sticky terminal alarm (no continued testing inside the same epoch);
- errors for non-independent or numerically invalid audits, and for attempting to reuse an exhausted/alarmed monitor;
- per-audit results containing the p-value, threshold, audit index, and cumulative Bonferroni bound.

For (alpha=0.05), (H=5), the per-audit threshold is 0.01. At least 99 calibration batches are necessary just to make that threshold attainable. **This is a resolution floor, not a power recommendation.** In practice, the calibration count must be selected from a preregistered power study; a minimally attainable p-value does not imply useful detection probability.

A non-alarm is not a model-adequacy certificate. If the horizon expires without an alarm, the outcome is inconclusive and the calling policy should remain unresolved or run a separately predeclared validation step. The monitor currently uses a single RMS standardized-residual score; it can miss structured residuals, autocorrelation, subgroup failures, and regime-specific errors.

## 4. Test and protocol obligations

The focused Rust tests cover constructor resolution, invalid budgets/horizons, correct alpha spending through the horizon, terminal behavior after an alarm, terminal behavior after a non-alarming horizon, and the rule that invalid audit input does not consume the budget.

The caller still has to provide auditable provenance. A real run receipt should record the epoch ID; source commit; candidate/model digest; score-function version; fixed calibration-batch IDs and digest; target distribution; audit-batch IDs; sample/measurement costs; alpha and horizon; audit order; seeds; terminal decision; and a digest of the policy-visible receipt. Hidden simulated truth belongs in a separate evaluator artifact.

Before a scientific deployment, expand the score family and include explicit negative controls. Tests of decision-state semantics are not evidence of detection quality.

## 5. Benchmark plan and external comparators

Freeze candidate pools, scoring statistics, calibration policy, target distribution, and seeds before confirmatory evaluation. Split by held-out *structural world families*, not merely new random seeds from the same generator.

Compare at matched total audit/measurement budgets:

1. one-shot batch conformal at a single predeclared endpoint;
2. the finite-horizon Bonferroni-spent baseline implemented here;
3. standard residual CUSUM/Page-Hinkley-style monitors, labeled as non-distribution-free comparators unless separately justified;
4. conditional conformal test martingales (CCTM), particularly fixed-reference conditional testing;
5. predictive-rank martingales (PRM), which account for dependence induced by repeated ranks against a fixed calibration set;
6. if available, a preregistered multi-statistic or weighted conformal detector with its own calibration and multiplicity rules.

Relevant research:
- Callahan & Catanach, AISTATS 2026, [On the Misinformation in a Statistical Experiment](https://proceedings.mlr.press/v300/callahan26a.html): information maximization can amplify inferential bias under misspecification.
- Tang, Sloman & Kaski, AISTATS 2026, [Representative, Informative, and De-Amplifying (R-IDeA)](https://proceedings.mlr.press/v300/tang26d.html): an acquisition design that separates representativeness, informativeness, and error amplification.
- Shaer et al., ICML 2026, [Testing For Distribution Shifts with Conditional Conformal Test Martingales](https://proceedings.mlr.press/v306/shaer26a.html): fixed-reference conditional shift testing with anytime-valid guarantees under its assumptions.
- Kuang & Xia, September 2026 preprint, [Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales](https://arxiv.org/abs/2609.00536): models conditional dependence of reused fixed-reference ranks and reports comparisons against CCTM.

These papers are comparators and research targets, not methods implemented in this PR. First reproduce their evaluation protocols and confirm licenses/dependencies before porting code.

### Metrics

Report separately, never as one blended score:

- family-wise false-alarm rate over the declared horizon, with binomial confidence intervals;
- sensitivity by predeclared discrepancy family and magnitude;
- median and 90th-percentile detection delay in audit batches and in actual query cost;
- missed-shift and unresolved rates;
- clean-world structural recovery after an alarm and after no alarm;
- calibration batch count, query budget, runtime, and recalibration overhead;
- performance under heteroscedastic/heavy-tailed noise, covariate shift, score-distribution shift, rare-region failures, and change points;
- contamination resistance: post-change observations must never silently enter the fixed null reference.

Suggested confirmatory program: at least 60 held-out worlds across 6 or more structural families, 20 paired seeds per world, and at least 10,000 independent clean-null streams for Monte Carlo evaluation of the family-wise false-alarm rate. Keep all paired runs, negative controls, confidence intervals, protocol versions, and raw receipts. Detection sensitivity targets should be set for a domain-relevant discrepancy before opening the held-out set; do not tune the target after seeing benchmark outcomes.

### Advancement gates

Proposed targets, not results:

- empirical family-wise false-alarm probability at or below 0.05 under the predeclared null, with uncertainty intervals and no hidden retuning;
- at least 80% sensitivity for the preregistered practically relevant shift on held-out structural families, at matched total cost;
- no more than a 5-percentage-point regression in clean-world structural recovery relative to the batch-audit policy;
- lower median total cost to a valid recovery/explicit unresolved outcome than the best simple baseline, with paired world-clustered uncertainty intervals reported;
- no false statement that a no-alarm result proves adequacy or that the detector handles arbitrary non-exchangeable shift.

If it cannot meet these gates, retain it as an explicit conservative baseline and report the limitation rather than weakening the protocol.

## 6. Next paradigm shift

The strongest next implementation is not another threshold wrapper. It is to reproduce a true fixed-reference CCTM and PRM comparator behind the same monitor interface, then benchmark multi-statistic detectors and connect alarms to a typed discovery transition:

**select experiment → collect independent batch → audit adequacy → continue / reopen model family / validate / report unresolved**.

Only the policy-visible score may affect the decision. Hidden truth, discrepancy amplitude, and evaluator labels must remain isolated. An alarm should trigger a predeclared model-expansion or validation protocol, not automatic selection of whichever law fits the audit batch.
