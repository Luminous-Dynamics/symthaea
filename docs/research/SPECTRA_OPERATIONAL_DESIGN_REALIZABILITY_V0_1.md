# SPECTRA operational design and realizability contract v0.1

**Status:** research specification; not an approved human-subject protocol  
**Issue:** [#10464](https://github.com/Luminous-Dynamics/symthaea/issues/10464)  
**Related:** [#10423](https://github.com/Luminous-Dynamics/symthaea/issues/10423), [#10424](https://github.com/Luminous-Dynamics/symthaea/issues/10424), [#10179](https://github.com/Luminous-Dynamics/symthaea/issues/10179), [#10187](https://github.com/Luminous-Dynamics/symthaea/issues/10187), draft [#10411](https://github.com/Luminous-Dynamics/symthaea/pull/10411), draft [#10426](https://github.com/Luminous-Dynamics/symthaea/pull/10426)

## 1. Executive decision

The expected-Fisher category search is a useful inner mathematical tool. It can compare the local information content of different *latent evidence-threshold configurations* under a declared Gaussian UVSD model.

It does not yet establish that an investigator can set those latent criteria in a participant, nor that the same thresholds are stable across people, sessions, payoff conditions, or measurement devices.

Therefore:

- Results from the current threshold enumerator are classified **LATENT_CONDITIONAL_ONLY**.
- The thresholds may be varied in simulation to learn where the model is informative.
- They must not be presented as directly selectable experimental settings.
- A design becomes operationally evaluable only after its controllable protocol actions are connected to a declared observer/measurement-channel model.
- A numerical optimizer cannot upgrade an absent action-to-observation mapping into scientific evidence.

This is a qualification boundary, not a rejection of the current information calculation.

## 2. Why this distinction matters

The current design search enumerates five ordered thresholds for six categories. Its expected-Fisher interface takes mu, sigma, thresholds, and signal/noise trial counts. The expected-information implementation includes threshold nuisance parameters when calculating the information matrix, which is appropriate for the conditional statistical calculation.

The unresolved layer is operational: what does a researcher do to make a participant produce those latent thresholds?

A confidence scale exposes ordered response options. Instructions, anchors, feedback, incentives, and task context can influence use of the scale, but they do not set the participant's internal evidence criteria to known numerical values. The criteria remain observer parameters inferred from the resulting observations.

RT offers more directly selectable measurement rules, but an RT quantile is not itself a known evidence criterion. Its relationship to evidence ordering depends on the task, response mapping, the population used to calculate quantiles, timing resolution, ties, excluded trials, and whether faster RT actually carries confidence/accuracy information in that domain.

A pre-study information calculation must model this chain:

    protocol action
      -> participant/task response process
      -> measurement channel and missingness
      -> ordered observed categories
      -> estimator and diagnostics
      -> uncertainty / model adequacy / claim disposition

Skipping the middle layers yields a conditional mathematical result, not an actionable study recommendation.

## 3. Typed variable taxonomy

Every design input must be typed before optimization.

| Type | Meaning | Examples | Design treatment |
|---|---|---|---|
| ExperimenterControlled | Directly selected by the protocol | trial allocation, block order, response-scale display, stimulus schedule, timing hardware, RT binning algorithm | May be optimized subject to protocol and burden constraints |
| InfluenceableButLatent | A participant state that protocol actions may influence, but cannot set exactly | confidence criteria, speed-accuracy policy, criterion adaptation, fatigue, lapses | Model as random/latent; propagate uncertainty |
| EstimatedObserverParameter | A latent property inferred from observations | mu, sigma, criteria, participant effects, RT/confidence coupling | Estimate with explicit identifiability and uncertainty |
| ObservedMeasurement | A recorded response or event | binary choice, ordinal confidence, RT, timeout, no response, missing rating | Preserve raw value, timestamp, units, and missingness reason |
| UnqualifiedMapping | No evidence-backed or explicitly conservative mapping exists | assumed payoff-to-criterion shift with no calibration | Block operational recommendation |

Do not encode an observer threshold as an ExperimenterControlled field merely because a design-search API accepts a numeric threshold.

### 3.1 Example: confidence

Controllable:
- number and wording of response options;
- response interface and order;
- whether confidence follows the primary binary choice;
- instructions, feedback, and incentive schedule;
- trial count and block sequence.

Latent or uncertain:
- the participant's criterion between options;
- criterion drift over time;
- differences in use of the scale across participants;
- whether scale anchors have comparable meaning between participants.

### 3.2 Example: RT

Controllable:
- event timing and clock source;
- response deadline;
- fixed-ms cut-points or the predeclared quantile algorithm;
- treatment of tied times, non-responses and invalid measurements;
- whether quantiles are pooled, participant-specific, session-specific, or block-specific.

Latent or uncertain:
- the relation between RT and confidence/accuracy;
- changes in RT distributions with stimulus difficulty, motor response, fatigue, or strategy;
- timing error and effective temporal resolution;
- the participant population that determines an adaptive quantile threshold.

A pooled quantile computed after collection is an observation transformation, not a pre-study guarantee of a fixed evidence criterion.

## 4. Operational design object

An operational design should be the tuple:

    D = (A, M, P, B, E)

where:

- A is the set of protocol actions the investigator controls;
- M is the observer/measurement-channel mapping from actions to observed category probabilities;
- P is the frozen uncertainty set or prior over observer and channel parameters;
- B is the participant-time, trial, equipment, and session burden model;
- E is the estimand and qualification/evaluation plan.

The latent threshold vector belongs inside M or P, not in A.

Each design object must declare:
- task domain and stimulus/response semantics;
- estimand and analysis unit;
- signal/noise allocation and randomization unit;
- confidence scale, RT measurement and category-construction algorithm;
- observer/channel assumptions and their evidence class;
- nuisance parameters and permitted variation;
- missingness and protocol-deviation handling;
- trial and participant-minute budgets;
- calibration, development, and locked holdout partitions;
- required approvals before participant-facing execution.

If M is absent, contradictory, or outside its qualification envelope, the result is NOT_OPERATIONALIZABLE or LATENT_CONDITIONAL_ONLY.

## 5. Keep estimands separate

The same observed counts can be used in different analyses, but the estimands and assumptions are not interchangeable.

| Analysis | Unit / information source | What it may estimate | Must not be conflated with |
|---|---|---|---|
| Single-observer UVSD | One observer/session with multiple ordered criteria/categories | Within-observer mu, sigma, d_a and criteria | Population-level variance from between-person heterogeneity |
| Repeated-session UVSD | Repeated observations from one observer | Within-person change and criterion drift, with dependence modeled | Independent replication across sessions |
| Population/hierarchical hUVSD | Random sample of participants with binary responses and between-person rate variation | Population-level distribution under hierarchical assumptions | Individual confidence/RT UVSD |
| RT-derived UVSD | RT-thresholded ROC/category data | UVSD estimates when the task supports the RT ordering assumption | Universal replacement for confidence |
| Confidence-derived UVSD | Ordered confidence responses | UVSD estimates conditional on confidence categories | Evidence that response levels are equally spaced |
| Model-free ROC/AUC | Observed rankings and labels | Discrimination summaries with fewer distributional assumptions | A parametric estimate of sigma |

In particular, the hierarchical binary model in Lages (2024) uses between-participant variability in hit and false-alarm rates. It is a separate population-level strategy, not an alternative implementation of the same single-observer likelihood.

## 6. Robust information and design objectives

The information calculation should be retained, but its interpretation and optimization should be expanded.

### 6.1 Local information

For each frozen observer/channel regime, report:
- expected Fisher matrix and parameterization;
- eigenvalue spectrum and effective rank;
- nuisance-adjusted variance for the prespecified estimand;
- d_a standard error and its sensitivity to nuisance parameters;
- category probabilities and expected sparse-cell counts;
- condition diagnostics with parameter scaling documented.

A raw matrix condition number is coordinate-dependent. It must not be described as a universal identifiability measure without a declared parameter scaling. The d_a variance calculation should be reported alongside the full spectrum; good d_a precision does not imply that every nuisance parameter is well estimated.

### 6.2 Competing design criteria

Report at least:
- C-optimality for the prespecified scalar estimand such as d_a;
- D-optimality for joint parameter-volume reduction;
- E-optimality for the weakest information direction;
- minimax / lower-tail performance over observer/channel uncertainty;
- prior-averaged performance only where the prior is justified and frozen.

Do not rank every study by one generic determinant or one average standard error. Show the Pareto frontier.

### 6.3 Burden-aware objective

Compare designs at equal *participant-time* and disclose:
- total participants;
- trials per participant and class;
- number of sessions and blocks;
- extra rating time;
- response deadline and timeout burden;
- calibration/instrumentation overhead;
- expected missingness and unusable-trial rates.

Equal trial counts do not necessarily imply equal participant burden.

### 6.4 Current threshold grid

The 13-point grid and 1,287 five-threshold combinations are appropriate as an initial conditional search. The 4 x 4 mu/sigma grid is a transparent sensitivity grid, not a validated population prior.

Keep the equal-probability and fixed-spacing baselines. Add conventional published RT-quantile and confidence-scale baselines only when their exact construction can be reproduced. Do not call one candidate a universal optimum.

## 7. Finite-sample, held-out qualification

Expected Fisher information is an asymptotic/local diagnostic. It is not a finite-sample recovery benchmark.

A full qualification campaign must generate participant/session/trial observations through the actual measurement channel, run the frozen adapter, fit the frozen estimator, and report all outcomes.

### 7.1 Partitioning

Use three non-interchangeable partitions:

1. Calibration: select observer/channel ranges, design candidates and optimizer settings.
2. Validation: compare candidate designs and resolve implementation defects.
3. Locked holdout: evaluate once after the design, assumptions and analysis are frozen.

A design must not be chosen and scored on the same seeds. Hold out complete participants and sessions where the estimand requires them; do not split adjacent trials from one session across train and test.

### 7.2 Required regimes

Cross at least:
- sigma < 1, sigma = 1, and sigma > 1, including near-one departures;
- low/medium/high sensitivity;
- category count and trial budget;
- balanced and imbalanced signal/noise allocation;
- sparse and empty observed cells;
- confidence criteria variation and drift;
- RT/confidence coupling variation, including invalid RT channels;
- lapses, response omissions, timeouts and informative missingness;
- measurement timing uncertainty and RT ties;
- participant and session heterogeneity;
- Gaussian and predeclared misspecified alternatives, including lapse/mixture and tail-shape alternatives.

### 7.3 Required metrics

Per regime and per design:
- bias and RMSE for mu, sigma, criteria, and d_a;
- median, 90th and 95th percentile absolute error;
- fit disposition counts and all failure rates;
- expected-information rank/conditioning;
- interval availability and interval coverage separately;
- interval width and coverage uncertainty;
- false UVSD declaration rate at sigma = 1;
- power at predeclared meaningful alternatives;
- held-out predictive log score and multinomial residual/adequacy diagnostics;
- participant-minutes, trials, and analysis cost.

For an interval procedure, report:
- number of generated outer campaigns;
- number of full-data fits by each disposition;
- number of intervals returned;
- conditional coverage among returned intervals;
- interval availability among all attempted campaigns;
- the joint fraction of all campaigns that both returned an interval and covered the truth.

A non-returned interval is not a covered interval. Conditional coverage alone can look good after difficult cases have been selectively excluded.

### 7.4 Monte Carlo precision

State the outer campaign count and a binomial confidence interval for each coverage/power/false-positive estimate. As a planning approximation, estimating a true 95% coverage rate to within roughly +/-0.5 percentage points at 95% confidence requires about 7,300 independent outer campaigns per regime. A 50 x 100 pilot is a smoke/calibration run only.

This calculation does not remove the need to account for dependence, stratification, or multiple comparisons.

## 8. Bootstrap and profile-likelihood method separation

The current [#10430](https://github.com/Luminous-Dynamics/symthaea/pull/10430) implementation is a *nonparametric, class-stratified category-count bootstrap*: it resamples observed category assignments within each class and preserves class totals. It is a useful conditional resampling method.

It is not the parametric bootstrap requested in #10424. Keep these methods separately named and explain their assumptions:

- Nonparametric conditional bootstrap: resample observed categorical assignments within signal/noise classes; retains the empirical category distribution.
- Parametric bootstrap: generate new class-specific multinomial counts from a fitted declared model; appropriate for model-based finite-sample calibration, including null calibration for equal-vs-unequal model tests.
- Profile likelihood: fix the parameter of interest, re-optimize nuisance parameters, and invert a declared likelihood-ratio criterion; report boundary/non-identifiable profiles explicitly.
- Wald / expected-Fisher: local approximation, useful as a fast diagnostic but not a coverage certificate.

The coverage runner should not present these as interchangeable. The next inferential gate remains profile-likelihood intervals and a parametric-bootstrap-calibrated equal-vs-unequal variance test.

## 9. Model adequacy is a separate gate

A numerically converged fit and a full-rank Fisher matrix do not establish that the Gaussian UVSD model describes the observations.

At minimum, compare:
- fitted multinomial category probabilities against observed category counts;
- posterior/predictive or parametric-bootstrap discrepancy statistics;
- calibration under equal-variance controls;
- lapse/mixture and tail alternatives;
- RT-derived and confidence-derived estimates where both channels are valid;
- model-free ROC summaries.

An adequate local fit cannot excuse systematic residual patterns or a measurement channel that violates its assumptions. Return MODEL_MISMATCH or NOT_EVALUABLE where warranted.

## 10. External parity and industry position

SPECTRA should meet existing methods before claiming a new capability.

| Comparator | Established capability / relevance | Required SPECTRA treatment |
|---|---|---|
| Miyoshi, Rahnev & Lau (2026) | RT-derived UVSD and comparison with confidence on visual-detection data; published R implementation | Reproduce the six-category example and all 11 datasets individually; preserve task/domain limits |
| rt-da (2026) | Independent Python implementation; reported R agreement on 80 subject-level fits, bootstrap intervals, RT-validity diagnostics and group workflow | Treat as a serious software-parity target; match exact fixtures, correction, category ordering and exclusions |
| SDT Assistant V3.1 | Established equal/unequal-variance normal ROC fitting for rating data and binary-response analyses | Match rating-scale UVSD behavior on shared fixtures; do not claim Rust implementation alone is better |
| Lages (2024) hierarchical hUVSD | Uses between-participant binary-response variability to estimate population variance | Compare on matched hierarchical/binary simulations with separate population estimands |

SPECTRA's potential differentiator is not simply "UVSD in Rust". It is a single coherent, auditable workflow that can qualify measurement design, identify weakly informative studies before data collection, distinguish methods and estimands, quantify uncertainty, test model adequacy, and produce reproducible per-study evidence.

That is a hypothesis to test, not a current superiority claim.

## 11. Proposed qualification targets

All are prospective targets, not achieved results.

| Dimension | Proposed target |
|---|---|
| External parity | Published R example within predeclared numerical tolerances; all 80 rt-da/R subject comparisons matched within the reference's declared optimizer tolerance |
| Dataset replication | All 11 Miyoshi visual-detection datasets reproduced individually, with exact exclusion/preprocessing notes |
| 95% interval calibration | 93–97% empirical coverage in each predeclared well-identified regime, Monte Carlo interval reported |
| Interval availability | At least 90% in each regime designated operationally supported; otherwise the regime is explicitly limited |
| Equal-variance false declaration | Study-level/familywise false UVSD declaration <=5% at sigma = 1 under the declared multiplicity policy |
| Useful alternative power | At least 90% for predeclared practically meaningful departures, with burden and Monte Carlo uncertainty disclosed |
| Design improvement | At least 15% reduction in median and 90th-percentile d_a RMSE over a frozen conventional-bin baseline at equal participant-time budget, without worse interval availability/coverage |
| Operational mapping | Zero operational recommendations without a qualified action-to-observation mapping |
| Robustness | No false upgrade under timing error, missingness, drift, class imbalance, RT-invalidity and model-misspecification controls |
| Reproducibility | Per-dataset, per-regime receipts with exact data, categories, correction mode, software revision, seeds and tool versions |

Do not tune these numbers after seeing locked holdout results. If a target proves infeasible under realistic burden, report the trade-off and revise only through a new preregistered version.

## 12. Required fail-closed dispositions

- OPERATIONAL_DESIGN_QUALIFIED: action-to-observation mapping and all declared study-design gates passed for the exact scope.
- LATENT_CONDITIONAL_ONLY: useful information calculation over assumed latent thresholds, without an operational mapping.
- NOT_OPERATIONALIZABLE: required measurement/action mapping absent or unsupported.
- WEAKLY_IDENTIFIED: declared local or finite-sample information gate failed.
- MODEL_MISMATCH: observed/simulated discrepancy is incompatible with the qualified model envelope.
- NOT_EVALUABLE: invalid input, missing source evidence, failed run, or unsupported configuration.

These dispositions are not a single quality score. They preserve where the inference chain stopped.

## 13. Work sequence

1. Preserve #10426 as a conditional expected-information kernel and label its candidate outputs accordingly.
2. Specify the operational design object and typed variable taxonomy (this document / #10464).
3. Add observer/measurement-channel generators for confidence and RT; reuse #10179 semantics.
4. Add participant/session heterogeneity and randomized block manipulation models.
5. Build finite-sample held-out design comparison at equal participant-time cost.
6. Add parametric bootstrap and profile-likelihood/model-comparison work under #10424.
7. Reproduce R, rt-da, Miyoshi and SDT Assistant fixtures before external-performance claims.
8. Only after synthetic and software gates: submit a separate ethics-reviewed protocol for any participant-facing study.

## 14. Source references

- Miyoshi, K., Rahnev, D., & Lau, H. (2026). *Correcting for unequal variance in signal detection models using response time*. iScience, 29, 114998. https://doi.org/10.1016/j.isci.2026.114998
- Authors' R reference: https://github.com/kiyomiyoshi/rt_type1_roc
- Independent rt-da software, validation suite and fixtures: https://github.com/trevcaru/rt-da
- SDT Assistant V3.1: https://hautus.org/sdt-assistant/
- Lages, M. (2024). *A hierarchical signal detection model with unequal variance for binary responses*. Psychonomic Bulletin & Review, 31, 2534–2557. https://doi.org/10.3758/s13423-024-02504-5

## 15. Claim boundary

This document is a research and engineering specification. It contains no human-subject results, no empirical participant validation, no compiled Rust evidence, no approval to alter a registered protocol, and no claim that SPECTRA exceeds current software or scientific practice.

A future qualified design will be conditional on its declared observer/channel model, population, task, protocol, instrumentation, and data scope. The system must state these boundaries with every recommendation.
