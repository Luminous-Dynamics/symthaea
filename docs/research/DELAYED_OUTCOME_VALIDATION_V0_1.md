# Delayed-Label Outcome Validation v0.1

Status: exploratory research protocol and first Rust monitor; not an empirical result.

## Research question

A feature-distribution alarm and a harmful outcome are different events. A detector can be statistically sensitive to benign covariate movement, while a consequential concept change can occur with weak marginal feature movement. Labels may also arrive late, be sampled selectively, or never arrive. Treating an unlabeled event as harmless creates optimistic bias; treating every feature alarm as harmful creates unnecessary intervention.

This work adds a standalone downstream outcome-validation layer that can consume an alarm receipt. This PR does not integrate a self-probe detector panel or a production prediction pipeline. It asks:

1. What changed in the observed stream?
2. Did a predeclared operational outcome deteriorate in a predeclared window after the alarm?
3. How much of the required outcome data had actually arrived at decision time?
4. Is the comparison unresolved because the window, labels, or clean baseline are incomplete?

The monitor only reports temporal association. It does not prove that the alarm caused an outcome change, and it does not estimate an unobserved counterfactual.

## Research context

- Cerqueira et al., “A Framework for Evaluating and Benchmarking Concept Drift Detection Methods,” accepted at KDD 2026, benchmark 14 methods on seven datasets and emphasize timing-aware metrics and leave-one-dataset-out tuning: https://arxiv.org/abs/2606.07789
- Alencar et al., “Concept drift detection in delayed and partially labeled data streams: An experimental survey,” *Digital Signal Processing*, published 15 October 2026, evaluates delayed and partially available labels: https://doi.org/10.1016/j.dsp.2026.106325
- Lu et al., “Autonomous Concept Drift Threshold Determination,” AAAI 2026, investigates adapting thresholds rather than fixing one threshold globally: https://ojs.aaai.org/index.php/AAAI/article/view/39586
- Boström, “Detecting Label Shift for Large Language Models with Conformal Test Martingales,” COPA 2026, evaluates false-alarm control and detection time using model-output proxies: https://proceedings.mlr.press/v329/bostrom26b.html

These works motivate the benchmark dimensions; they do not establish a performance result for Symthaea.


## Industry baseline (documentation checked 23 October 2026)

The adjacent field is not empty. The fair comparison should acknowledge what production monitoring already does:

| System | Publicly documented scope | Comparison implication |
| --- | --- | --- |
| Amazon SageMaker Model Monitor | Baseline-based data-quality checks, model-quality comparisons once ground-truth labels are ingested, bias drift and feature-attribution drift. AWS currently says the legacy Model Monitor service is no longer open to new customers and is not receiving new features. | Use as a historical/product capability reference, not as an assumed available new deployment option. Its label-merge pattern is a baseline for our delayed-outcome workflow. |
| Arize ML Monitoring | Model drift, performance, outliers, data quality and broader observability linking validation and production. | A shift alarm alone is not competitive differentiation; root-cause and performance linkage are already industry expectations. |
| NannyML | Monitoring documentation includes estimated accuracy/performance when labels are unavailable and later concept-shift validation using ground truth. | Compare against label-free performance estimation as well as feature-shift detection; the Rust pre/post mean monitor is not a substitute for those estimators. |
| Evidently | Data-quality and drift tests plus task-specific classification, regression, ranking and LLM evaluations. | A benchmark should include evaluation quality, not only distribution-distance scores. |

References: AWS data-quality and model-quality documentation (https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-data-quality.html; https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality.html), Arize ML Monitoring (https://arize.com/products/ml-monitoring/), NannyML Cloud monitoring (https://docs.nannyml.com/cloud), and Evidently's public repository overview (https://github.com/evidentlyai/evidently).

### What would be a credible Symthaea advantage?

Not the existence of a drift alert. A defensible advantage would have to be demonstrated by the combined protocol: (1) explicit per-probe-family coverage and visible uncovered shift families; (2) a strict event-time ledger that rejects retrospective alarms and backdated labels; (3) outcome confirmation that keeps missing/delayed labels unresolved; and (4) reproducible equal-budget comparisons across null, benign-shift, harmful-shift, OOD and delayed-label worlds. Industry systems should be benchmarked where a faithful runnable comparison exists, and otherwise kept as a capability comparison rather than assigned fabricated scores. The current implementation is only a small first step toward this target.

## Rust API contract

DelayedOutcomeMonitor is a small, dependency-free, standalone consumer of alarm receipts in symthaea-physics-bridge. This version is not yet wired directly into DetectorPanelScorer or a production feature pipeline; a caller supplies the alarm and the matching outcomes.

- Stream positions must be registered consecutively and in order.
- Every alarm refers to an already observed position.
- Every outcome is attached to its original observation and is accepted only at the current stream position; future and retrospectively backdated label-arrival indices are rejected.
- Pre-alarm and post-alarm windows are fixed before evaluation: [t - w, t) and [t, t + w).
- Metric direction is explicit (HigherIsWorse or LowerIsWorse) and the minimum material effect is fixed in configuration.
- Each window must meet both a minimum labelled count and a minimum label-coverage fraction.
- Incomplete post windows, absent baseline history, overlapping alarms, inadequate label coverage, and numerical overflow are explicit non-outcome states. None is silently coerced to NoMaterialChange.
- The assessed values use only labels received so far. A later assessment may change when delayed labels arrive.

Targeted test command:

    cargo test -p symthaea-physics-bridge delayed_outcome_validation --lib

This command has been specified but has **not** been executed in the current authoring environment. Exact-head CI remains the source of compile, format, and test evidence.

## Benchmark protocol

### Controlled stream worlds

Use paired seeds and the same underlying streams across every detector. Hold the outcome labels and ground truth in the evaluator, never in the detector or online monitor.

1. **Benign covariate shift:** feature distribution changes, conditional outcome mechanism does not.
2. **Harmful concept shift:** outcome relationship changes and a predeclared outcome worsens.
3. **Feature-quiet harmful shift:** outcome relationship changes with little detectable marginal feature movement.
4. **Confounded temporal change:** both the feature distribution and outcome base rate change for a separate, known reason.
5. **Recovery:** the outcome mechanism returns to baseline after a specified interval.
6. **Unprobed shift:** operational drift affects a channel or family absent from the declared self-probe suite.
7. **Null stream:** stationary data under the detector's stated assumptions.

For each world, vary abrupt versus gradual transitions, shift magnitude, duration, dimensionality, outcome variance, and window length. Inject label-arrival patterns independently of outcome values for MCAR/MAR-style tests, then add explicitly outcome-dependent missingness as a stress test. Include fixed delays, heavy-tailed delays, 5/20/50% observed labels, and bursty label delivery. Keep data-generation seeds and streams paired across systems.

### Required baselines

Compare the proposed pipeline to:
- fixed threshold on the existing feature-shift score;
- the best single detector selected only on training/probe data;
- pooled and stratified self-probe detector panels at equal probe-score budget;
- established streaming drift detectors represented in the chosen benchmark implementation;
- a label-aware detector evaluated only on the labels available at that point in the replay;
- a no-alarm policy and an always-alarm policy as sanity bounds.

Baseline configuration must be frozen without inspecting held-out evaluation streams. Use leave-one-dataset/domain-out tuning where feasible. Where reproductions cannot be run faithfully, record the exact implementation, configuration and limitation rather than borrowing published headline numbers.

### Metrics to publish separately

- feature-shift alarm precision, recall, false alarms per 1,000 observations, and average run length on null streams;
- delay from true feature-shift onset to first alarm;
- outcome deterioration precision/recall and false confirmations per 1,000 observations;
- time from true outcome deterioration to a confirmed association, plus time from feature alarm to outcome confirmation;
- unresolved fraction at decision deadlines, label coverage per window, label cost, and label-arrival distribution;
- performance split by shift family, label-delay regime, outcome severity, and detector/probe configuration;
- CPU time, allocation/peak memory, and throughput by stream size and feature dimension;
- paired confidence intervals across independent stream seeds, not only best-seed tables.

A single aggregated accuracy number is inadequate. A system that alarms quickly but cannot demonstrate outcome relevance should not be ranked as equivalent to one that confirms deterioration, and neither should be allowed to hide false alarms in its average.

## Proposed phase gates (targets, not achieved results)

1. **Contract gate:** zero look-ahead violations, duplicate outcome consumption, hidden dropped stream positions, or unresolved states silently reported as no change in adversarial replay tests.
2. **Calibration gate:** on held-out null streams that satisfy each method's declared assumptions, compare false-alarm rates against the predeclared nominal level and report confidence intervals. For methods without a defensible nominal guarantee, report measured false-alarm rates without calling them calibrated.
3. **Coverage gate:** publish outcomes only when each window passes both label-count and label-fraction gates; report every excluded/unresolved trial in the denominator.
4. **Fair-comparison gate:** compare at equal label budget and, where applicable, equal probe and compute budgets. Tune on training domains and use held-out domains for the headline table.
5. **Advantage gate:** only claim a win if paired held-out confidence intervals support a predeclared useful improvement in outcome-confirmation delay or outcome precision while false alarms and unresolved trials do not regress beyond agreed margins.
6. **Causal-claim gate:** never describe a pre/post association as causal harm. Causal attribution requires a defensible control/counterfactual design, stronger identification assumptions, or randomized intervention.

The team should choose utility margins with an application owner before looking at test results. There is no universal economically meaningful effect threshold.

### Illustrative numerical targets to preregister (not achieved results)

These targets should be reviewed against the application’s costs before any confirmatory run:

- **Null control:** for methods claiming nominal alpha = 0.05 under an exchangeable null, use at least 1,000 independent null streams and require the 95% upper confidence bound on the per-stream false-alarm probability to be at most 0.07. Also report sequential false alarms per 1,000 observations; per-stream calibration must not hide repeated alerting.
- **In-scope power:** at the minimum effect size declared operationally meaningful, target at least 90% detection by the preregistered deadline, with a lower 95% interval bound above 0.85, separately for each primary shift family—not merely on the pooled average.
- **Probe coverage:** target at least 90% detection/coverage across predeclared in-scope probe families at a fixed total probe-score budget. Every uncovered or unprobed family remains visible; this is not a promise of generalization to unknown shifts.
- **Delayed-outcome utility:** at the same false-alarm rate and label budget, target either a 20% reduction in median time-to-outcome-confirmation or a 5 percentage-point gain in outcome precision. Require the paired 95% interval to exclude zero improvement and constrain any regression in the companion metric to at most 2 percentage points.
- **No-label invariant:** exactly zero outcomes may be classified as deteriorated/improved when either window fails the configured label gates; every such event is unresolved. This is a correctness gate, not a statistical score.
- **Compute:** publish p50/p95/p99 latency and peak memory for fixed stream lengths and window sizes before setting deployment limits. Do not claim real-time readiness from unit tests or asymptotic complexity alone.

The numeric thresholds above are proposed go/no-go hypotheses, not industry norms. If the team changes them after seeing results, label the analysis exploratory and rerun on a fresh holdout.

## Known limitations and next work

The first Rust monitor uses window means. It does not correct for outcome-dependent missingness, time trends, seasonality, repeated testing, or interference among nearby episodes beyond flagging direct window overlap. Its per-alarm status is descriptive and is not a sequential hypothesis test. The simple pre/post contrast can mistake unrelated time changes for alarm-related effects. Numeric means can also be a poor outcome summary for rare failures or heavy-tailed costs.

Next benchmark revisions should add: (a) confidence intervals/effect-size uncertainty; (b) explicit censoring and missingness diagnostics; (c) seasonality-matched or contemporaneous control windows; (d) recurrent and overlapping-event handling with a predeclared policy; (e) multiple-alarm multiplicity control; and (f) real delayed-label streams with evaluator-held outcome ground truth. Causal claims require a separate design, not just a more sensitive detector.

No SOTA, real-world safety, or empirical superiority claim is made in v0.1.
