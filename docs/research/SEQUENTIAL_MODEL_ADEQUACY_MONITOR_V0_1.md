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

The monitor accepts one scalar score per fresh batch. `standardized_rms_residual` remains the simple baseline; the optional `multi_view_standardized_residual_score` aggregates four predeclared diagnostics into one scalar: RMS error, a normalized maximum residual, a sample-size-scaled mean residual, and sample-size-scaled lag-one residual correlation. This is a sensitivity-oriented score family, not a new analytic p-value. Its normalizers are heuristics; the conformal guarantee applies only to the final aggregate when that exact score function, diagnostic family, batch length, ordering convention, and target-distribution protocol are frozen and applied to independent clean calibration batches and future audits. Changing any part requires fresh calibration. The score still cannot detect unrepresented subgroup failures unless subgroup diagnostics are explicitly part of the frozen score family. A non-alarm never certifies the model.

## 4. Contract tests and receipt requirements

Focused Rust tests cover equal-spending resolution, invalid budgets/horizons, weighted schedule validation, threshold ordering, early-versus-late decision semantics, cumulative budget bounds, terminal behavior, and the rule that invalid audit input does not consume the horizon. A fixed-seed 20,000-stream null smoke test compares the corrected policy against repeatedly reusing an unadjusted 0.05 threshold with the same calibration sizes and audit scores. This is a regression check, not a confidence interval, distribution-free proof, or substitute for a larger preregistered evaluation.

A real run receipt should record the epoch ID; source commit; candidate/model digest; score-function version; fixed calibration-batch IDs and digest; target distribution; audit-batch IDs; sample and measurement costs; alpha and the full spending schedule; audit order; seeds; terminal decision; and a digest of the policy-visible receipt. Hidden simulated truth belongs in a separate evaluator artifact and must never influence model selection or audit policy.

## 5. Benchmark plan and external comparators

Freeze candidate pools, score statistics, calibration policy, target distribution, alpha schedule, and seeds before confirmatory evaluation. Split by held-out **structural world families**, not merely new seeds from the same generator.

Compare under matched total audit/measurement budgets:

1. one-shot conformal audit at a single predeclared endpoint;
2. RMS-only residual score with equal-spending finite-horizon Bonferroni baseline;
3. the frozen multi-view aggregate residual score with the same calibrator, audit batches, alpha and horizon;
4. equal versus risk-weighted finite-horizon spending, with weights preregistered for an explicitly stated risk/time-to-signal objective;
4. residual CUSUM/Page-Hinkley-style monitors, clearly labeled as non-distribution-free unless separately justified;
5. conditional conformal test martingales (CCTM), which target anytime-valid control using a fixed reference;
6. predictive-rank martingales (PRM), which account for dependence in repeated ranks against a fixed calibration reference;
7. recent parameter-free conditional conformal martingales for shifts at unknown times.

Relevant research frontiers:

- Shaer et al., ICML 2026, [Testing For Distribution Shifts with Conditional Conformal Test Martingales](https://proceedings.mlr.press/v306/shaer26a.html): a fixed-reference conditional method designed to avoid test-time reference contamination, with anytime-valid type-I control under its assumptions.
- Kuang & Xia, September 2026, [Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales](https://arxiv.org/abs/2609.00536): derives the conditional null distribution of reused fixed-reference ranks, constructs a predictive-rank martingale, and reports stronger detection performance than CCTM in its own experiments. Reproduce its protocol before comparing results.
- Liu, Prinster & Orabona, October 12, 2026, [Monitoring for Shifts at Unknown Times: Parameter-Free Conditional Conformal Martingales](https://arxiv.org/abs/2610.17343): proposes a restarting universal portfolio approach intended to avoid slow adaptation after long stationary periods, with reported computational improvements over a naive implementation.
- [Optimal Alpha Spending for Sequential Analysis with Binomial Data](https://academic.oup.com/jrsssb/article/82/4/1141/7056022): shows that spending shape should be selected for a predeclared operating objective such as power, expected sample size, or expected time to signal.
- [When Does Score Fusion Help? Conformally Certified Out-of-Distribution Detection for Camera and LiDAR Sensors](https://pmc.ncbi.nlm.nih.gov/articles/PMC13469369/): emphasizes that fusion gains depend on the detector and modality; a combined score must be compared with its strongest constituent rather than assumed superior.

These are comparators and research targets, **not methods implemented by this PR**. Risk-weighted alpha spending is established statistical practice; our contribution here is the typed Rust contract, resolution/terminal-state checks, audit receipts, and a benchmark interface aligned to scientific model-adequacy decisions. Any implementation of CCTM/PRM/restarting methods needs mathematical review, reproducible reference cases, dependency/license review, and tests against its published algorithm—not a name-only approximation.

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

## 6. Next paradigm shift

The highest-value next step is a reproducible head-to-head implementation of fixed-reference PRM and CCTM, then the October 2026 restart/parameter-free method, under one `SequentialMonitor` interface. Benchmark at matched false-alarm budget, calibration size, and measurement cost; evaluate both early and late shifts, while keeping exact conditional-rank dependence and reference uncertainty intact.

The long-term scientific-discovery loop should be:

**select experiment → collect independent batch → audit model adequacy → continue / reopen model family / validate / report unresolved**.

Only policy-visible measurements and a predeclared score may affect decisions. Hidden truth and evaluator labels stay isolated. An alarm should trigger a predeclared model-expansion or validation protocol, not automatic selection of whichever law fits the audit batch.
