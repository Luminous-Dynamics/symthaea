# SDCB covariance-model uncertainty tournament v1

**Status:** preregistration proposal. No dataset or physical measurements were evaluated in this change.

## 1. Research objective

Test whether maintaining uncertainty over competing temporal/covariance models improves scientific posterior calibration and experiment decisions compared with selecting one model or averaging only within-model uncertainty.

Three different risks must be measured separately:
1. uncertainty conditional on one model;
2. uncertainty about which model is adequate;
3. misspecification because every registered model is wrong or incomplete.

Bayesian model averaging handles the first two only within the supplied model set. It cannot protect against omitted models without an explicit discrepancy or support-set treatment.

## 2. Benchmark families

| ID | Task | Required falsification |
|---|---|---|
| CM0 | One correctly specified Gaussian model | Mixture reduces exactly to single-model posterior |
| CM1 | Two models with different OU time constants | Predictive evidence updates weights; total variance includes model spread |
| CM2 | Similar model likelihoods / weak identifiability | Do not report decisive model weights from uninformative observations |
| CM3 | Finite-data model-weight uncertainty | Block-bootstrap or posterior uncertainty in weights is visible |
| CM4 | Covariance-matrix misspecification | Plug-in covariance does not pass as certain; coverage is measured |
| CM5 | Abrupt regime shift | Prequential score degradation triggers stale-model warning / abstention |
| CM6 | Heavy-tailed or outlier observations | Gaussian model probabilities do not become spuriously decisive |
| CM7 | Omitted covariance mechanism | All-registered-model poor predictive score triggers model-set inadequacy warning |
| CM8 | Near-zero posterior model weight | Robust support membership is not silently pruned |
| CM9 | Prior sensitivity | Report posterior and decision sensitivity to plausible model priors |
| CM10 | Adaptive model discrimination | Choose observations that separate decision-relevant models under hard budgets |
| CM11 | Generalized Bayesian update | Compare likelihood-based and loss-based updates under declared misspecification |
| CM12 | Process/measurand confounding | Do not relabel true process variation as instrument error |
| CM13 | Reference-chain ancestry uncertainty | Compare models with partial shared calibration ancestry |
| CM14 | Dynamic vector cross-covariance | Recover coupled state propagation and cross-factor uncertainty |

## 3. Required method arms

All approaches receive the same chronological data, action set, reference budget, model candidates, observation windows and terminal utility.

- Plug-in covariance / single selected model.
- AIC/BIC or held-out predictive model selection where assumptions fit.
- Bayesian model averaging using prequential predictive likelihood.
- Predictive stacking / cross-validated combination where the target is predictive performance rather than posterior model probability.
- Finite-scenario robust maximin support set (#13882).
- Model-weight sensitivity and blocked bootstrap around calibration evidence.
- Generalized Bayesian / Gibbs updates using robust losses, including GBOED for model-misspecified experimental design.
- Exact finite mixture evaluator (this reference).
- Decision-focused experiment selection that trades off target uncertainty, model discrimination and terminal loss.

Do not compare a robust worst-case risk, a Bayesian posterior variance and a predictive score as though they share units.

## 4. Primary controls

### 4.1 Total variance

For every task, retain:

\[
V_{\mathrm{mix}} =
\sum_s w_s V_s +
\sum_s w_s(m_s-\bar m)^2.
\]

The second term is not optional. Include constructed tasks where it dominates the within-model contribution.

### 4.2 Weight robustness

Repeat the entire model-weight calculation over block-resampled chronological calibration sets and plausible prior weights. Report:
- weight intervals / distributions;
- rank reversal frequency;
- mixture-mean and total-variance sensitivity;
- experiment-policy changes caused by weight uncertainty.

A point posterior weight from one calibration split is not evidence that model ranking is stable.

### 4.3 Model-set adequacy

Monitor prequential negative log predictive density, probability integral transform / residual diagnostics, interval coverage and change-point evidence. When every registered model has poor predictive support:
- do not force normalized weights to imply that one candidate is adequate;
- emit a model-set inadequacy state;
- widen or suspend model-dependent scientific claims;
- request model expansion / discrepancy review.

Normalized weights always sum to one; that mathematical fact does not imply the model set contains a good model.

### 4.4 Generalized Bayes comparison

Compare ordinary likelihood updating with generalized Bayesian / Gibbs updates under contamination, heavy tails, and misspecified covariance. Freeze the loss, learning-rate / temperature parameter and calibration procedure before the confirmatory test. Robust losses must not be tuned using the held-out test outcomes.

A promising 2025 preprint, *Robust Experimental Design via Generalised Bayesian Inference*, derives Gibbs expected information gain for robust design and reports robustness to outliers and incorrect outcome-noise assumptions. Reproduce its published experiments where possible; this is a directly relevant comparator, not an SDCB novelty claim. DOI/arXiv: https://arxiv.org/abs/2511.07671.

A 2026 *Journal of Computational Physics* paper also addresses Bayesian experimental design for model-discrepancy calibration using an auto-differentiable ensemble Kalman inversion approach. Compare its treatment of discrepancy and high-dimensional nuisance models when the original experimental protocol can be reproduced: https://doi.org/10.1016/j.jcp.2025.114469.

## 5. Metrics

**Predictive calibration**
- prequential log score / negative log predictive density;
- CRPS and PIT diagnostics where supported;
- nominal 90/95% interval coverage and sharpness;
- coverage by chronological regime, device, location and reference chain.

**Model uncertainty**
- model-weight calibration across repeated known-generator trials;
- weight stability under block bootstrap and plausible priors;
- posterior model entropy, without treating entropy as a scientific objective;
- frequency of incorrect decisive model selection;
- between-model variance / total variance ratio.

**Scientific decisions**
- target posterior bias and interval coverage;
- expected terminal decision loss and regret;
- false confidence and abstention frequency;
- model-discrimination value at equal reference cost;
- robust worst-case loss over the frozen support set.

**Evidence integrity**
- exact fixture agreement;
- no invalid covariance acceptance;
- replay and state-version consistency;
- explicit model-set inadequacy on omitted-model controls.

## 6. Proposed gates (targets, not achievements)

- 100% agreement with exact finite mixture references on frozen known-answer fixtures.
- Zero omission of between-model variance in report outputs.
- Zero silent robust-support pruning caused solely by Bayesian posterior weight.
- Zero automatic scientific acceptance when all candidate models fail the preregistered predictive adequacy checks.
- Nominal 90% and 95% coverage assessed chronologically and by task family.
- At most 2% regret against exact adaptive decision oracles on tractable instances.
- At least 10% lower decision loss per equal reference cost than the strongest matched plug-in / single-model / fixed-policy baseline on preregistered held-out model-uncertainty tasks, with paired confidence intervals excluding zero before superiority claims.
- Report robustness and model-weight uncertainty; no single leaderboard score may hide model-set failure.

## 7. Non-claims

The finite model-mixture oracle does not discover the true covariance model, validate model priors, establish physical causal mechanisms, or protect against omitted models. The robust support set is only as defensible as its construction. All benchmark thresholds are internal targets until independently reproduced.
