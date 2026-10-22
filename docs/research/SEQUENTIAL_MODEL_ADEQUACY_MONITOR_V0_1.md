# Finite-Horizon Sequential Model-Adequacy Monitoring v0.1

**Status:** research implementation in a stacked draft PR; exact-head Rust validation is pending. The risk-weighted alpha allocation is a predeclared baseline, not a new distribution-free guarantee, and no empirical performance or SOTA claim is made.

This follow-on builds on the calibrated batch audit in PR #8439 and joint prior/model/noise robust acquisition in PR #8361. It adds explicit fixed-horizon error-budget allocation; it does not replace either component.

## 1. Research problem and paradigm shift

Scientific discovery often audits a model more than once: after new experiments, new conditions, or changes in instrument state. A single batch conformal p-value at level 0.05 is not permission to repeat the same test indefinitely at level 0.05. Repeated thresholding inflates the probability of at least one false alarm. A detector that keeps appending post-change observations to its reference can also dilute evidence of real shift.

The baseline therefore makes three things explicit and auditable:

1. Keep the null calibration reference and score definition fixed during one monitoring epoch.
2. Declare the finite audit horizon and allocate one portion of a total false-alarm budget to each audit before observing audit scores.
3. Stop after an alarm or after the declared horizon. A no-alarm horizon is unresolved, not proof of adequacy.

The new optional risk-weighted schedule lets a domain protocol spend more error budget at audits that are especially costly to delay. It does **not** guarantee greater power overall: the allocation must be justified and selected before evaluation. Equal Bonferroni spending remains the default.

## 2. Contract and bound

Let \(H\) be the predeclared maximum audit count and \(\alpha\in(0,1)\) the family-wise false-alarm budget. Choose fixed weights \(w_1,\ldots,w_H>0\) satisfying \(\sum_{t=1}^{H} w_t\leq 1\). At audit \(t\), compute a conservative batch-level conformal rank p-value against the frozen calibration scores and alarm only if

\[
p_t \leq \alpha_t, \qquad \alpha_t=\alpha w_t.
\]

If each audit p-value is marginally super-uniform under the clean null, the union bound gives

\[
\Pr(\text{one or more false alarms by }H)
\leq \sum_{t=1}^{H}\Pr(p_t\leq\alpha_t)
\leq \sum_{t=1}^{H}\alpha_t
\leq \alpha.
\]

This bound does **not** require independence among audit p-values. It does require their marginal validity. Calibration scores and each fresh audit score must follow the same exchangeability/null protocol under the same predeclared score function. Changed target distributions, adaptive score tuning, model refitting using audit data, or audit selection that breaks the protocol invalidates the claim. Start a new epoch and collect fresh calibration data when the model, scorer, or target population changes.

For equal spending, \(w_t=1/H\), so \(\alpha_t=\alpha/H\). For a risk-weighted schedule such as \([0.5,0.25,0.15,0.1]\), half the error budget is allocated to the first audit and the remaining half is distributed later. Such a schedule can improve early detection for some alternatives, but reduces the thresholds available later. It is only appropriate if the risk profile is declared in advance and the calibration resolution can attain every allocated threshold. A schedule whose smallest threshold is below \(1/(n+1)\), with \(n\) calibration batches, is rejected because that audit could never alarm.

This is a finite-horizon union-bound guarantee, not an e-process, not a conformal test martingale, and not an anytime-valid conditional guarantee. It is an auditable baseline against which more powerful methods should be compared.

## 3. Implementation

`sequential_adequacy.rs` provides:

- equal alpha spending using `SequentialAdequacyMonitor::new`;
- risk-weighted spending via `new_with_alpha_spending`, with fixed positive weights summing to at most one;
- an explicit resolution precondition: every allocated threshold must be at least the smallest attainable conservative rank p-value \(1/(n+1)\);
- a typed state machine: `Monitoring`, `Alarmed`, or `HorizonExhausted`;
- a sticky terminal alarm, and terminal inconclusive outcome after the horizon;
- errors for invalid/non-independent audit input, invalid schedules, inadequate p-value resolution, and attempted reuse of an exhausted/alarmed monitor;
- per-audit results containing the p-value, the actual audit threshold, the audit index, and cumulative union-bound spending.

For \(\alpha=0.05\), equal spending over five audits requires at least 99 calibration batches just to make the threshold \(0.01\) attainable. The example weighted schedule above ends at threshold \(0.005\), requiring at least 199 calibration batches to make every threshold attainable. These are **resolution floors, not power recommendations**. In practice, calibration size and allocation must be selected from a preregistered power study.

The monitor accepts one scalar score per fresh batch. `standardized_rms_residual` remains the simple baseline. `standardized_residual_diagnostics` reports four predeclared views—RMS error, normalized peak residual, normalized mean bias, and normalized lag-one dependence—and `multi_view_standardized_residual_score` aggregates them by a maximum statistic. This may be more sensitive to some alternatives and less sensitive to others; score fusion does not guarantee better power.

The second fusion path exposes `minimum_component_p_fusion_score` and `sum_log_component_p_fusion_score`. It ranks each diagnostic against an independent normalization split, aggregates the component tail scores, and then requires a *separate untouched outer calibration split* to calibrate the aggregate. This second stage is crucial: the minimum p-value or sum of negative log p-values is not, by itself, a final calibrated p-value. The outer rank calibration avoids independence assumptions between diagnostic views, conditional on the fixed normalization split, at the cost of additional clean calibration data. All score functions, component order, batch length, target distribution, and split roles must remain fixed; changing them requires fresh calibration. None of the scores can detect unrepresented subgroup failures unless subgroup diagnostics are explicitly included. A non-alarm never certifies the model.

## 4. Contract tests and receipt requirements

Focused Rust tests cover equal-spending resolution, invalid budgets/horizons, weighted schedule validation, threshold ordering, early-versus-late decision semantics, cumulative budget bounds, terminal behavior, and the rule that invalid audit input does not consume the horizon. A fixed-seed 20,000-stream null smoke test compares the corrected policy against repeatedly reusing an unadjusted 0.05 threshold with the same calibration sizes and audit scores. This is a regression check, not a confidence interval, distribution-free proof, or substitute for a larger preregistered evaluation.

A real run receipt should record the epoch ID; source commit; candidate/model digest; score-function version; fixed calibration-batch IDs and digest; target distribution; audit-batch IDs; sample and measurement costs; alpha and the full spending schedule; audit order; seeds; terminal decision; and a digest of the policy-visible receipt. Hidden simulated truth belongs in a separate evaluator artifact and must never influence model selection or audit policy.

## 5. Benchmark plan and external comparators

Freeze candidate pools, score statistics, calibration policy, target distribution, alpha schedule, and seeds before confirmatory evaluation. Split by held-out **structural world families**, not merely new seeds from the same generator.

Compare under matched total audit/measurement budgets:

1. one-shot conformal audit at a single predeclared endpoint;
2. RMS-only residual score with equal-spending finite-horizon Bonferroni baseline;
3. maximum-statistic multi-view score with the same calibrator, batches, alpha and horizon;
4. two-stage minimum-p and sum-log score fusion, with matched total clean-calibration budget;
5. equal versus risk-weighted finite-horizon spending, with weights preregistered for an explicitly stated risk/time-to-signal objective;
6. residual CUSUM/Page-Hinkley-style monitors, clearly labeled as non-distribution-free unless separately justified;
7. conditional conformal test martingales (CCTM), which target anytime-valid control using a fixed reference;
8. predictive-rank martingales (PRM), which account for dependence in repeated ranks against a fixed calibration reference;
9. recent parameter-free conditional conformal martingales for shifts at unknown times.

Relevant research frontiers:

- Shaer et al., ICML 2026, [Testing For Distribution Shifts with Conditional Conformal Test Martingales](https://proceedings.mlr.press/v306/shaer26a.html): a fixed-reference conditional method designed to avoid test-time reference contamination, with anytime-valid type-I control under its assumptions.
- Kuang & Xia, September 2026, [Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales](https://arxiv.org/abs/2609.00536): derives the conditional null distribution of reused fixed-reference ranks, constructs a predictive-rank martingale, and reports stronger detection performance than CCTM in its own experiments. Reproduce its protocol before comparing results.
- Liu, Prinster & Orabona, October 12, 2026, [Monitoring for Shifts at Unknown Times: Parameter-Free Conditional Conformal Martingales](https://arxiv.org/abs/2610.17343): proposes a restarting universal portfolio approach intended to avoid slow adaptation after long stationary periods, with reported computational improvements over a naive implementation.
- [Optimal Alpha Spending for Sequential Analysis with Binomial Data](https://academic.oup.com/jrsssb/article/82/4/1141/7056022): shows that spending shape should be selected for a predeclared operating objective such as power, expected sample size, or expected time to signal.
- [When Does Score Fusion Help? Conformally Certified Out-of-Distribution Detection for Camera and LiDAR Sensors](https://pmc.ncbi.nlm.nih.gov/articles/PMC13469369/): emphasizes that fusion gains depend on the detector and modality; a combined score must be compared with its strongest constituent rather than assumed superior.

CCTM, PRM and restart-based monitoring remain comparators, **not implementations in this PR**. Risk-weighted alpha spending is established statistical practice. The new work provides typed Rust score diagnostics, maximum-statistic aggregation, split-calibrated score-fusion primitives, and explicit monitoring contracts; none is claimed to be a new statistical guarantee or SOTA result. Any implementation of CCTM/PRM/restarting methods needs mathematical review, reproducible reference cases, dependency/license review, and tests against its published algorithm—not a name-only approximation.

### Report separately

- family-wise false-alarm rate over the declared horizon, with binomial confidence intervals;
- sensitivity by predeclared discrepancy family and magnitude;
- median and 90th-percentile detection delay in audit batches **and** query/measurement cost;
- missed-shift and unresolved rates;
- structural law recovery on hidden truth after alarm and after no alarm;
- calibration count, query budget, runtime, memory, and recalibration overhead;
- performance under heteroscedastic/heavy-tailed noise, covariate shift, score-distribution shift, rare-region failure, and change points;
- contamination resistance: post-change observations must never silently enter the frozen null reference;
- family- and world-clustered confidence intervals, with every paired run and negative control retained.

Suggested confirmatory program: at least 60 held-out worlds across 6 or more structural families, 20 paired seeds per world, and at least 10,000 independent clean-null streams for Monte Carlo evaluation of family-wise false alarms. Keep raw receipts and protocol versions. Set a domain-relevant discrepancy and target sensitivity before opening the held-out set.

### Proposed advancement gates (targets, not results)

- empirical family-wise false-alarm probability at or below 0.05 under the preregistered null, with uncertainty intervals and no hidden retuning;
- at least 80% sensitivity for the preregistered practically relevant shift on held-out structural families at matched cost;
- no more than a 5-percentage-point regression in clean-world structural recovery versus the one-shot audit policy;
- lower median total cost to a valid recovery/explicit unresolved outcome than the best simple baseline, with paired world-clustered uncertainty intervals;
- no claim that a no-alarm result proves adequacy or that the method covers arbitrary non-exchangeable shift.

If these gates fail, retain the method as an explicit conservative baseline rather than weakening the protocol.

### Exploratory synthetic pilot (independent JavaScript calculation; Rust not executed)

A deterministic, exploratory paired pilot compared four score policies at nominal alpha 0.05. Each replication used 64 residuals per batch, 99 clean batches in a component-normalization split A, 99 independent clean batches in aggregate-calibration split B, and one audit batch (198 calibration batches total for each method). There were 1,200 replications per condition with seed `0x9bf1d3a7254e8c61`. The RMS and raw-maximum policies used all 198 clean scores directly; the two-stage fusion policies used split A to rank each component and split B to calibrate the fused statistic. Alternative batches had one randomly positioned +6-standard-deviation outlier, a +0.4 standard-deviation mean shift, or Gaussian AR(1) residuals with \(\rho=0.55\). The clean-null batches were independent standard Gaussian residuals.

| Scenario | RMS only | Raw maximum | Minimum-p fusion | Sum-log fusion |
|---|---:|---:|---:|---:|
| Clean-null alarm rate | 3.50% | 3.83% | 2.08% | 4.08% |
| One +6σ outlier | 82.33% | 34.17% | 66.17% | 92.25% |
| +0.4σ mean shift | 20.17% | 84.00% | 48.50% | 53.25% |
| AR(1), \(\rho=0.55\) | 8.75% | 97.08% | 62.25% | 48.08% |

For the clean-null rates, the unpaired 95% Wilson intervals were respectively 2.60–4.70%, 2.89–5.08%, 1.42–3.06%, and 3.10–5.36%. Ties make these empirical rates conservative in some policies. This was a small exploratory Monte Carlo with 1,200 replicates, not a confirmatory benchmark; these numbers are **independent JavaScript calculations of the stated formulas, not executed Rust tests or published scientific results**.

The key finding is the trade-off: raw maximum is highly sensitive to mean bias and serial dependence but misses many single-spike alternatives; sum-log fusion performs best on this one-spike scenario but is weaker on the AR(1) alternative. No policy uniformly dominates. That agrees with recent score-fusion research, which explicitly finds that combining detectors can fail to beat the best constituent depending on the data and modality ([score-fusion study](https://pmc.ncbi.nlm.nih.gov/articles/PMC13469369/)). Next, reproduce this pilot in Rust, add heavy-tailed and heteroscedastic nulls, perform at least 10,000 clean-null streams, report paired confidence intervals for differences, and include matched-cost baselines before choosing a default policy.

## 6. Next paradigm shift

The highest-value next step is a reproducible head-to-head implementation of fixed-reference PRM and CCTM, then the October 2026 restart/parameter-free method, under one `SequentialMonitor` interface. Benchmark at matched false-alarm budget, calibration size, and measurement cost; evaluate both early and late shifts, while keeping exact conditional-rank dependence and reference uncertainty intact.

The long-term scientific-discovery loop should be:

**select experiment → collect independent batch → audit model adequacy → continue / reopen model family / validate / report unresolved**.

Only policy-visible measurements and a predeclared score may affect decisions. Hidden truth and evaluator labels stay isolated. An alarm should trigger a predeclared model-expansion or validation protocol, not automatic selection of whichever law fits the audit batch.
