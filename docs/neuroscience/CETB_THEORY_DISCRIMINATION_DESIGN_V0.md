# CETB theory-discriminating experimental design V0

**Status:** Proposed research design method; no new experimental predictions or measurements are claimed.  
**Date:** 2026-11-21

## 1. The paradigm shift

CETB should not only evaluate theories after studies have run. It should help choose the next experiment before data collection by asking:

> Which feasible, ethical experiment is expected to reduce uncertainty between the competing theories the most, after accounting for measurement quality, nuisance uncertainty, missingness and acquisition cost?

This turns theory disagreement into a prospective design contract. The method does not tell us which theory is true. It identifies experiments whose possible outcomes would be more discriminating under the declared predictive commitments.

## 2. Theory prediction cards

Each theory must provide a versioned predictive distribution for every candidate experiment and outcome, not only a qualitative direction. Cards must be frozen before the design is selected and include:
- theory/version and source passage;
- predicted outcome distribution and its parameter uncertainty;
- task, intervention, modality, population and analysis window;
- nuisance assumptions, exclusions and missingness model;
- expected measurement reliability and artifact sensitivity;
- predicted distribution under plausible alternative explanations;
- a predeclared interpretation of each possible outcome, including null and ambiguous outcomes.

Use only commitments that the theory proponents or an independently appointed theory panel accept as fair operationalizations. An evaluator should not invent a weak or straw-man prediction for a theory.

## 3. Expected information gain

Let (T) be the theory identity with prior (pi_t), and let (Y) be the predeclared observable outcome for candidate design (d). Each theory supplies a likelihood (P(Y=ymid T=t,d)).

The outcome marginal is:

[
P(ymid d)=sum_t pi_t P(ymid t,d).
]

The expected information gain about theory identity is:

[
operatorname{EIG}(d)=
sum_t pi_tsum_y P(ymid t,d)
log_2rac{P(ymid t,d)}{P(ymid d)}.
]

Equivalently, this is the mutual information (I(T;Ymid d)) and is bounded by the entropy of the prior over theories. For two theories it is closely related to the Jensen–Shannon divergence between their predicted outcome distributions under the declared prior.

EIG is a prospective design quantity, not a goodness-of-fit statistic and not posterior evidence from observed data. A design with high EIG is useful only if the predictions are calibrated and the outcome is measurable.

## 4. Nuisance uncertainty and robust discrimination

Theories often predict outcomes conditional on uncertain parameters, preprocessing, population, dose, timing or measurement assumptions. Let (eta) denote nuisance/model parameters.

Do not calculate EIG from one convenient parameter setting. Report:
- EIG integrated over predeclared plausible nuisance distributions;
- a sensitivity range across defensible nuisance priors;
- a conservative lower quantile or worst-case EIG over the qualified nuisance set;
- which assumptions cause the design ranking to change.

A robust design utility can be defined as:

[
U(d)=rac{Q_{alpha}[operatorname{EIG}(dmideta)]}
{C(d)}
]

where (Q_alpha) is a predeclared lower quantile (or a conservative minimum where justified), and (C(d)) is a positive acquisition cost. The quantile is a policy parameter, not a scientific constant. Also report EIG and cost separately; a ratio can obscure whether a design is simply cheap or genuinely informative.

Do not treat a single worst-case value as universally robust when the nuisance set is unbounded or implausible. Freeze the nuisance set and explain it.

## 5. Measurement failure is an outcome

If a design has a chance of unusable recordings, task non-compliance, aborted intervention or missing report, the predictive model must include these outcomes or explicitly model the missingness mechanism. Do not silently renormalize only successful outcomes.

Separate:
- technical acquisition failure;
- participant withdrawal or inability to complete;
- ambiguous/low-quality measurements;
- missing behavioural report;
- valid null or negative result.

Missingness may itself be informative, but it is not automatically evidence for one theory. The design must model how failure depends on participant, state, site, intervention and signal quality.

## 6. Ethical and operational constraints

The selected design must satisfy minimum gates before EIG is optimized:
- ethics and participant-safety constraints;
- intervention feasibility and reversibility where relevant;
- meaningful task and outcome validity;
- measurement reliability and required acquisition quality;
- data rights and privacy;
- feasible sample size, duration, operator burden and cost;
- independent analysis and data governance.

A highly discriminating but unethical, unsafe or unmeasurable design is not eligible. Cost should include participant burden and safety, not only machine time.

## 7. Avoid theory-selection circularity

- Freeze prediction cards before viewing confirmatory outcomes.
- Publish priors and conduct prior-sensitivity analysis; do not use equal priors by default and then describe the posterior as objective truth.
- Score predictions using proper predictive scores (log score, Brier/CRPS where appropriate) on outcomes not used to tune those predictions.
- Include a null/agnostic model and plausible shared-mechanism alternatives.
- Allow more than two theories and explicitly model shared predictions.
- Keep "theories differ" separate from "the experiment can measure the difference."
- Report when all candidate theories make nearly identical predictions; the correct decision may be that no available design is discriminating.

## 8. Connection to CETB and Symthaea

Use existing Symthaea typed forecast distributions and proper scoring rules where their outcome semantics fit. The design tool should consume prediction cards and return a design ranking plus a frozen receipt; it must not author its own theory predictions or independently grade its own hypotheses.

The result should include:
- prediction-card and design-manifest digests;
- prior and nuisance distributions;
- per-theory predicted outcomes;
- expected and robust EIG;
- acquisition costs and safety-gate results;
- missingness and measurement-quality assumptions;
- selected design and rank stability under prior/nuisance sensitivity;
- preregistration and evaluation authority.

## 9. Proposed benchmark targets

These are engineering goals, not achieved results:
1. Exact arithmetic parity against hand-computed EIG fixtures.
2. Zero probability-mass loss: each theory/outcome distribution including missingness sums to one within a declared tolerance.
3. Prior sensitivity and nuisance sensitivity are always reported.
4. The ranking is reproducible from immutable card/design receipts.
5. The design selector cannot change theory cards or scoring after the design is selected.
6. All completed and failed experiments are reported, not only those that discriminate.
7. Predictive calibration is measured on historical or held-out outcomes before EIG is trusted.

## 10. What would make this a field-level contribution?

The central contribution would be a reusable, adversarially governed process linking theory commitments to experiment choice, actual measurement quality, preregistered analysis and independently scored outcomes. A successful implementation could reduce under-informative studies and post-hoc theory reinterpretation. It would not prove that the winning theory is true, and a higher EIG estimate does not guarantee real-world discrimination if the theory likelihoods are misspecified.

## 11. Status

This document defines the method only. No new IIT, GNWT, predictive-processing or other theory distributions have been authored in this increment. No new experiment has been run. The accompanying JSON contract is a synthetic design template, not a forecast about human participants.
