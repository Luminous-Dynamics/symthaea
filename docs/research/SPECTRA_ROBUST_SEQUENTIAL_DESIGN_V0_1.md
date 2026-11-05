# SPECTRA Robust Sequential Design Contract v0.1

**Status:** research specification; not a live experiment controller  
**Tracking:** [#10490](https://github.com/Luminous-Dynamics/symthaea/issues/10490)  
**Composes with:** #10464 (operational realizability), #10467 (UVSD calibration), #10423/#10424 (design and inference), #10179 (measurement observability), PR #10469 (latent-threshold finite-sample recovery)

## 1. Executive decision

SPECTRA should evolve from a static expected-Fisher threshold search into a **goal-directed, robust, sequential experiment designer**.

The existing latent-threshold search remains valuable as a conditional inner calculation. It is not a deployable protocol recommender. The next system should choose only experimenter-controlled actions, update from recorded observations, and select the next action only when the action-to-measurement mapping and current inference state are qualified.

The unit of optimization becomes a policy:

\[
\pi_t(a_t \mid H_t, \mathcal{M}, \mathcal{Q}, \mathcal{E}),
\]

where:

- \(a_t\) is an allowed experimenter action at time \(t\);
- \(H_t\) is the immutable, recorded history available before that action;
- \(\mathcal{M}\) is the declared set of candidate observation/data-generating models;
- \(\mathcal{Q}\) is the declared uncertainty or ambiguity set;
- \(\mathcal{E}\) is the estimand and decision objective.

This is not permission to adaptively manipulate a participant's latent confidence criterion. The experimenter can choose a stimulus, instruction condition, payoff schedule, response-scale presentation, trial allocation or acquisition setting. The participant's criterion, lapse propensity, fatigue, learning and speed-accuracy policy remain latent/influenceable processes that must be modeled or measured.

## 2. Why static Fisher design is not the final objective

Expected Fisher information is a local, model-conditional diagnostic. It is useful for screening candidate threshold vectors and inspecting local identifiability. It cannot by itself establish:

- that an investigator can implement the thresholds in a real task;
- that the observation model is correctly specified;
- that a candidate works away from the local parameter point;
- that finite-sample estimates or intervals are calibrated;
- that the design remains representative of the target task/population;
- that an adaptive stopping rule preserves the intended error guarantee.

Retain Fisher information, rank, scaled condition diagnostics and local standard errors as one layer in the evidence packet. Do not allow that layer to override finite-sample recovery, operational validity, model-adequacy or failure dispositions.

## 3. Research basis and comparator boundary

The following work motivates the research direction:

1. Watson (2017), [QUEST+: A general multidimensional Bayesian adaptive psychometric method](https://doi.org/10.1167/17.3.10), provides a mature adaptive psychometric comparator for threshold and multi-parameter psychometric tasks. It is not automatically an apples-to-apples UVSD estimator.
2. [Adaptive stimulus selection for multi-alternative psychometric functions with lapses](https://pmc.ncbi.nlm.nih.gov/articles/PMC6222824/) demonstrates information-theoretic sequential selection with explicit lapse and omission structure.
3. Go & Isaac (2022), [Robust expected information gain using ambiguity sets](https://proceedings.mlr.press/v180/go22a.html), shows why prior/model uncertainty can change EIG rankings and develops a robust alternative.
4. Tang, Sloman & Kaski (AISTATS/PMLR 2026), [Representative, Informative, and De-Amplifying](https://proceedings.mlr.press/v300/tang26d.html), identifies that information-only acquisition can interact badly with model misspecification and distribution shift; its R-IDeA acquisition combines representativeness, informativeness and error de-amplification.
5. Abdulsamad et al. (2026), [Maximin Robust Bayesian Experimental Design](https://arxiv.org/abs/2603.14094), frames robust design as a max-min problem and derives a robust information objective with finite-sample PAC-Bayes bounds for stochastic policies.
6. Doumbouya, Vidard & Zahm (2026), [Certified Bayesian optimal experimental design: from Expected Information Gain to Signal-to-Noise Ratio](https://arxiv.org/abs/2609.32282), develops upper/lower EIG bounds from signal-to-noise and Fisher-information structure to reduce nested Monte Carlo cost in its stated settings.
7. Lages (2024), [A hierarchical signal detection model with unequal variance for binary responses](https://doi.org/10.3758/s13423-024-02504-5), is a different population-level binary-response estimand and a useful comparator for heterogeneity, not a substitute for single-observer rating-scale UVSD.
8. Wen et al. (ICML/PMLR 2026), [Robust Sequential Experimental Design for A/B Testing](https://proceedings.mlr.press/v306/wen26i.html), develops robust sequential design for contextual-bandit and dynamic settings, proves a worst-case MSE bound for treatment-effect estimation, and evaluates on synthetic and technology-company data. This is a strong architectural comparator, but its treatment-effect estimand is not a UVSD estimand; SPECTRA should compare design principles and only run numerical parity when an exactly shared target can be defined.
9. Callahan & Catanach (AISTATS/PMLR 2026), [On the Misinformation in a Statistical Experiment](https://proceedings.mlr.press/v300/callahan26a.html), shows that under model and inferential misspecification classical experimental-utility comparisons can break down and highly informative designs can amplify bias into confident incorrect inference. This strengthens the requirement that model adequacy and misspecification stress gates sit outside, and can veto, an information objective.
10. van Meer (JRSSB 2026), [Anytime validity is free: inducing sequential tests](https://doi.org/10.1093/jrsssb/qkag050), gives a way to induce an anytime-valid procedure from a fixed-horizon test while retaining its terminal behavior. This is not a license to assume every sequential method is power-neutral: the paper also discusses the power costs of common sequential tests. Benchmark whole-policy power and type-I error at every stopping horizon, as well as terminal power.

These results define baselines and ideas to test. They do not establish SPECTRA performance or superiority.



### Comparator interpretation (2026 literature)

SPECTRA's contribution cannot be framed as the first robust sequential experiment designer. Wen et al. already provide a 2026 ICML treatment-effect framework for contextual-bandit and dynamic A/B testing, including a worst-case MSE bound and an industry dataset evaluation. Tang et al. provide an acquisition function explicitly designed around representativeness, informativeness and de-amplification. Callahan & Catanach warn that information-based utility comparisons themselves can fail under model/inference misspecification. These are serious comparators. Their estimands and evaluation domains differ from single-observer ordinal UVSD, so compare architectural properties now and reserve numerical superiority claims for a common estimand and locked data-generating process.

Likewise, sequential validity is not a single interchangeable checkbox. Compare fixed-horizon tests, alpha-spending, e-process/confidence-sequence procedures, and induced anytime-valid versions of a fixed terminal test. Measure the full power-versus-stopping-time curve; do not infer either free power or unavoidable power loss from the label "anytime-valid".

## 4. Typed operational action space

Every policy candidate must come from a frozen action schema. Each variable must be labelled as one of:

| Type | Examples | Policy may set it? |
|---|---|---|
| Experimenter-controlled | stimulus intensity/difficulty, randomized condition, trial allocation, block schedule, response-scale display, instruction condition, payoff/base-rate schedule, RT acquisition window | Yes, only within approved bounds |
| Influenceable but latent | confidence criteria, speed-accuracy strategy, criterion adaptation, lapse propensity, fatigue, learning | No; estimate or model |
| Estimated observer parameter | \(\mu\), \(\sigma\), \(d_a\), criterion offsets/drift, participant/session effects, channel noise | No; infer |
| Observed measurement | category, binary choice, RT, timeout, omitted response, missing confidence, protocol deviation | No; observe and record |
| Unqualified mapping | undocumented action-to-category relation, unvalidated RT-to-evidence transform | No; block recommendation |

An operational action must preserve the declared estimand. If changing stimulus intensity changes both sensitivity and lapse, the response model must represent both. If changing instructions shifts criteria or changes the decision process, that effect must not be silently attributed to \(\sigma\).

Unqualified action-to-observation mappings result in `NOT_OPERATIONALIZABLE` or `LATENT_CONDITIONAL_ONLY`, never a recommendation.

## 5. Goal-oriented acquisition

The policy's target must be declared before optimization:

- parameter target, such as \(d_a\) or \(\sigma\);
- scientific contrast, such as equal versus unequal variance;
- predictive quantity at a declared operating point;
- or a downstream decision with an explicit, defensible utility matrix.

Do not maximize entropy over every model parameter by default. Learning nuisance parameters is useful only to the extent that it improves the target, distinguishes observation models, or diagnoses model failure.

Use a multi-objective acquisition record rather than hiding every trade-off in one scalar:

\[
\mathbf{U}(a;H_t) =
\left[
U_{\text{target}},
U_{\text{model-discrimination}},
U_{\text{representativeness}},
-U_{\text{burden}},
-U_{\text{worst-case regret}}
\right].
\]

Candidate components include:

- goal-oriented expected information gain about the declared target;
- information for distinguishing equal-variance, unequal-variance and channel/lapse hypotheses;
- robust lower-tail utility over a declared ambiguity set;
- representativeness and audit-stratum coverage;
- expected participant-time and compute cost;
- policy regret under predeclared stress families.

A scalarized objective may be investigated, but weights, normalization, ambiguity radius, prior sensitivity and objective trade-offs must be frozen and exposed. Do not claim a universal optimum from an arbitrary weighted sum.

A generic robust acquisition candidate is:

\[
a_t^* = \arg\max_{a\in\mathcal{A}_t}
\inf_{Q\in\mathcal{Q}_t}
\mathbb{E}_{Y\sim p_Q(\cdot\mid a,H_t)}
\left[\Delta U_{\mathcal{E}}(a,Y;H_t)\right],
\]

subject to hard action, participant-time, audit-coverage and safety constraints. This is a conceptual contract, not an implemented estimator. The ambiguity set and numerical approximation require their own calibration.

## 6. Open-world model and stress registry

The simulator and benchmark must include separate, named data-generating families for:

1. Equal-variance and unequal-variance Gaussian SDT.
2. Random-guess lapses, contaminant responses and stimulus-dependent lapses.
3. Criterion offsets, payoff/base-rate shifts and within-session criterion drift.
4. Participant and session heterogeneity, including correlation between sensitivity, criterion and signal variance.
5. Signal/noise class imbalance and changing prevalence.
6. Confidence-channel noise, ordinal scale use and instruction-induced scale changes.
7. RT validity conditions: monotonicity failure, ties, clock quantization, onset uncertainty, timeout, non-decision time and response omissions.
8. Missing-at-random and missing-not-at-random sensitivity scenarios.
9. Non-Gaussian/heavy-tail and mixture alternatives.
10. Sequential dependence / history effects where applicable.

Do not pool models that target different estimands into one opaque ensemble. For example, single-observer ordinal UVSD, RT-derived UVSD and population hUVSD must remain separate analysis tracks with explicit mapping assumptions.

Every scenario must carry a version, parameter domain, generating mechanism, estimand, severity tier, seed family and known failure modes.

## 7. Representativeness, randomized audits and selection receipts

An adaptive selector can over-focus on actions that are informative under its current model and neglect regions where that model is wrong. Prevent this with an explicit audit design:

- randomized exploration among eligible actions;
- mandatory sentinel conditions and minimum signal/noise/condition counts;
- an independent randomized confirmatory sample or other untouched validation path;
- fixed subgroup/participant/session coverage targets;
- all selected and eligible-but-unselected action counts.

An initial 10% randomized exploration share may be evaluated as a pilot setting, not treated as a universal constant. Sweep the floor and show the efficiency/representativeness trade-off. Mandatory safety or validity controls are hard constraints, not terms that can be traded away for information.

Every action receipt should record:

- policy identifier, version and content digest;
- action schema and allowed action-set digest;
- policy random seed and PRNG/version;
- complete pre-action history digest;
- candidate actions and selected-action probability;
- selected action, tie-break and fallback reason;
- observed measurement, missingness/timeout status and protocol deviations;
- estimator/model revision and inference disposition;
- participant-time / compute cost.

For adaptive likelihood inference, the complete action history and selector state must be available. When action selection is a function only of recorded past observations and frozen policy randomness, its conditional selection mechanism can often be conditioned on in the likelihood. If decisions also depend on unrecorded information or latent participant state, that dependency must be modeled; do not assume it cancels.

## 8. Sequential stopping and inferential validity

Predeclare a maximum trial/time budget and distinct stopping reasons:

- precision target reached;
- evidence/model-discrimination target reached;
- futility;
- model-adequacy or data-quality stop;
- participant safety / protocol stop;
- hard budget reached.

For frequentist confirmatory claims with repeated looks, compare fixed-horizon inference with alpha-spending and anytime-valid e-process/confidence-sequence approaches. Bayesian posterior stopping is not automatically a frequentist type-I error guarantee. In either case, simulate the **entire policy plus stopping rule**, not merely the estimator at a fixed sample size.

Report early-stop rate, expected and maximum sample count, false-positive rate, power, interval coverage, interval availability, width and Monte Carlo uncertainty by scenario.

## 9. Benchmark protocols and baselines

All policy comparisons must use frozen held-out seeds and equal participant-time budgets. Use common random numbers for paired simulation comparisons where the coupling is valid and document when it is not.

| Baseline | Role | Fair comparison |
|---|---|---|
| Fixed equal-probability categories | Simple conventional baseline | Same task/response model and budget |
| Fixed-spaced categories | Conventional ordinal design | Same task/response model and budget |
| Current minimax / trade-off designs | SPECTRA static design candidates | Latent-conditional diagnostic only until operational mapping is qualified |
| QUEST+ | Mature adaptive psychometric baseline | Threshold / psychometric recovery on shared estimands |
| One-step Bayesian adaptive | EIG baseline | Same declared target and model set |
| Goal-oriented adaptive | Scientific-target efficiency | Same target estimand and decision |
| Robust / ambiguity-set adaptive | Model/prior robustness | Same ambiguity set and stress bank |
| Robust + audit hybrid | Proposed SPECTRA policy | Same budget, audit floor and held-out seeds |

Never compare unequal-variance model recovery with a QUEST+ threshold estimate and call the one with smaller parameter error superior. Use a shared target, or report them as complementary capabilities.

## 10. Required scorecard

Report each scenario and the aggregate with its uncertainty:

- target-parameter / quantity bias, RMSE and absolute error;
- 95% interval coverage and Monte Carlo confidence interval;
- interval availability, width, boundary/one-sided status and failure dispositions;
- false unequal-variance declaration at \(\sigma=1\) and power at prespecified alternatives;
- model-selection accuracy and misspecification detection sensitivity/specificity;
- predictive log score and predictive calibration;
- sample count, median/p95 participant-time, wall-clock and compute cost;
- worst-case and lower-tail utility, policy regret and prior/ambiguity sensitivity;
- audit-stratum coverage and action-distribution representativeness;
- lapse, missingness, timeout, protocol deviation and failure rates;
- subgroup, session and heterogeneity performance.

Never report only conditional coverage among successful fits. Include all campaign outcomes, including nonconvergence, nonidentifiability, boundary fits and unsupported regimes.

## 11. Proposed qualification targets

These are research targets, not achieved results:

| Gate | Proposed target |
|---|---|
| Type-I error | Preserve the declared 5% null bound under the entire sequential policy, with Monte Carlo uncertainty reported |
| Interval coverage | 93–97% empirical coverage for nominal 95% intervals in each predeclared well-identified regime |
| Interval availability | At least 90% in explicitly operationally supported regimes, reported separately from coverage |
| Efficiency | At least 15% lower median target RMSE or participant-time to a fixed precision target than the best fixed baseline |
| Robustness | No more than 5% relative target-RMSE degradation in each declared stress family versus the best safe baseline, otherwise mark unsupported |
| Audit coverage | At least 95% coverage of predeclared audit strata and 100% completion of mandatory controls |
| Replay | Exact action and stopping replay from immutable receipts |
| Operationalization | Zero operational recommendations when the action-to-observation mapping is unqualified |

These targets must be finalized before the confirmatory seed bank is opened. Do not revise thresholds after inspecting held-out results.

## 12. Execution plan and release gates

1. Freeze the action schema, estimand and scenario registry.
2. Build a policy simulator with explicit history conditioning and immutable action receipts.
3. Implement fixed baselines and a QUEST+-style psychometric comparator for its proper target.
4. Add goal-oriented and model-discrimination acquisitions.
5. Add ambiguity-set/maximin stress evaluation and representativeness audits.
6. Add sequential stopping and calibrate the complete procedure.
7. Run independent held-out campaigns; publish seeds, code revision, scenario registry, failures, Monte Carlo intervals and machine-readable outputs.
8. Require independent methodological review and exact-head Rust verification.
9. Only after those gates, ethics approval and a separately reviewed protocol may shadow-mode or human pilot work be considered.

### Release dispositions

- `EXPLORATORY_ONLY`: candidate policy has not passed the held-out benchmark.
- `LATENT_CONDITIONAL_ONLY`: results condition on unobservable/untestable latent action values.
- `NOT_OPERATIONALIZABLE`: no qualified action-to-measurement mapping exists.
- `MODEL_ADEQUACY_FAILED`: predictive checks or stress gates reject the declared model family.
- `INFERENCE_UNQUALIFIED`: coverage, type-I error, convergence or failure-rate gates are not met.
- `SHADOW_MODE_ELIGIBLE`: simulation and independent review gates passed; not permission for human-subject work.
- `HUMAN_PILOT_ELIGIBLE`: separate ethics, safety, informed-consent and protocol approvals are documented.

No status may be promoted solely because a local Fisher-information score is favorable.

## 13. Immediate implementation tickets

- Typed, frozen `OperationalActionSet` and action-to-observation qualification status ([draft PR #10493](https://github.com/Luminous-Dynamics/symthaea/pull/10493)).
- Conservative scenario aggregation across a complete action × scenario matrix, including lower/upper score bounds, maximin/CVaR and fail-closed coverage ([issue #10509](https://github.com/Luminous-Dynamics/symthaea/issues/10509), [draft PR #10511](https://github.com/Luminous-Dynamics/symthaea/pull/10511)).
- Known-truth observer and measurement-channel DGP (Gaussian / variance-standardized Student-t, lapse, criterion drift, independent participant effects, evidence-dependent missingness and quantized RT) — [issue #10555](https://github.com/Luminous-Dynamics/symthaea/issues/10555), [draft PR #10559](https://github.com/Luminous-Dynamics/symthaea/pull/10559).
- Simultaneous bounded-score Monte Carlo uncertainty with one-sided Hoeffding bounds and Bonferroni familywise control — [issue #10557](https://github.com/Luminous-Dynamics/symthaea/issues/10557), [draft PR #10560](https://github.com/Luminous-Dynamics/symthaea/pull/10560).
- Per-participant fail-closed conversion of observed categories to UVSD fitter tables, with explicit missingness dispositions — [issue #10561](https://github.com/Luminous-Dynamics/symthaea/issues/10561), [draft PR #10562](https://github.com/Luminous-Dynamics/symthaea/pull/10562).
- Policy interface with deterministic replay and propensity receipts ([draft PR #10493](https://github.com/Luminous-Dynamics/symthaea/pull/10493)).
- Fixed, QUEST+-style and one-step EIG baselines.
- Goal-oriented and model-discrimination utilities.
- Robust ambiguity-set / maximin scoring and prior-sensitivity analysis.
- Randomized audit floor and sentinel-coverage accounting.
- Sequential stopping module with full-policy null and coverage simulation.
- Machine-readable scorecard and frozen benchmark campaign manifest.

## 14. Claim boundary

SPECTRA has not yet demonstrated adaptive-design superiority over QUEST+, PSI/Psi, robust BOED or other current methods. No participant-level adaptive results are claimed by this specification. The proposed paradigm is a testable architecture and benchmark plan; the advantage must be established on held-out data, equal budgets, calibrated inference and operationally realizable actions.
