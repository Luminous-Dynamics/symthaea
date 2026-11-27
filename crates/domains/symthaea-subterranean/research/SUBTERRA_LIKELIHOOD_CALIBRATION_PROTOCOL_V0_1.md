# Subterra empirical likelihood calibration protocol v0.1

**Date:** 2026-11-16  
**Status:** Research protocol and small empirical reference estimator authored; not executed  
**Purpose:** prevent EVSI, robust EVSI and survival planning from consuming guessed or overconfident observation likelihoods.

## 1. Research motivation

The one-step EVSI and finite-horizon planners consume values of:

\`P(observation outcome | geological hypothesis, measurement action)\`

These probabilities are not derivable from qualitative site states such as Candidate, Supported or DirectlyObserved. They must be estimated from independently adjudicated observations and evaluated on held-out sites.

Recent 2026 work supports this design:
- The world-model survey argues that planner-driven data collection shifts the state/action distribution away from offline training data and recommends evaluating one-step plus multi-step predictive accuracy (horizons 1, 5, 10 and 25), NLL/proper scoring rules and uncertainty calibration under OOD state/action conditions. https://doi.org/10.1007/s44163-026-02122-1
- The active-visual-perception review identifies the next sensing action, its observation-value model and the downstream task as a closed decision loop rather than a perception-only score. https://doi.org/10.3390/s26206544
- The planetary-cave exploration review emphasizes the need to design sensing around scientific questions, environmental constraints and heterogeneous teams rather than assuming one platform or sensor suite fits every site. https://doi.org/10.1002/rob.70233
- The sensor-centric uncertainty survey highlights calibration drift, synchronization, OOD conditions and integrity monitoring as sources of overconfident multi-sensor decisions. https://doi.org/10.3390/s26123801

These papers do not validate this estimator. They motivate the evaluation protocol.

## 2. Data contract and split isolation

Every case represents one independently adjudicated site/action observation and contains:
- stable case ID;
- site ID;
- unique physical acquisition-lineage ID;
- measurement/action ID;
- independently adjudicated reference hypothesis ID;
- observed outcome ID;
- split: Training, Calibration or Evaluation;
- opaque reference-label evidence reference;
- opaque observation evidence reference.

Required invariants:
1. The hypothesis set includes an explicit \`unknown\` class.
2. A whole site belongs to exactly one split. No site may leak between Training, Calibration or Evaluation, even if its observations or modalities differ.
3. An acquisition-lineage ID may be used only once. Reprocessing one physical acquisition cannot increase effective sample count.
4. One site/action pair contributes at most one case in v0.1. Repeated surveys require a later hierarchical/repeated-measures model, not duplicated iid rows.
5. Action outcome categories are frozen before scoring and form an exhaustive categorical outcome set.
6. The reference hypothesis and observed outcome must be supported by separately recorded evidence references. The adapter does not authenticate these references.
7. Unknown examples must be genuinely adjudicated OOD/unclassified geology or measurement cases; they must not be synthetic probability padding.

## 3. Split responsibilities

### Training

Used only to estimate conditional likelihoods. The v0.1 reference uses a symmetric Dirichlet categorical estimator:

\`P(o_j | h,a) = (n_j + α) / (N + αK)\`

where \`n_j\` is the Training count for outcome \`j\`, \`N\` is the total independent Training count for that hypothesis/action cell, \`K\` is the number of frozen outcomes, and \`α > 0\` is a declared smoothing parameter.

The reference defaults are not implied: \`α\` and the minimum independent site count are explicit caller parameters. A cell below the minimum has no emitted probability vector. It must not be passed into EVSI as a fabricated likelihood.

### Calibration

A separate site-disjoint split used to assess reliability and, in a future version, choose pre-registered calibration/threshold procedures. The v0.1 reference only scores this split; it does not fit a temperature, isotonic map or other post-hoc transformation.

### Evaluation

A final site-disjoint holdout. It must not influence the hypothesis ontology, outcome categories, smoothing parameter, support threshold, loss matrix, resource costs or planner thresholds. Any tuning after looking at Evaluation invalidates that Evaluation split for final claims.

## 4. Proper scoring and coverage

The v0.1 reference reports:
- multiclass Brier score (sum of squared errors over the frozen outcome categories);
- negative log-likelihood (NLL);
- total cases, scored cases, unscored cases and coverage fraction;
- per-action / per-hypothesis score cells.

Brier and NLL are proper scoring rules. Report both; do not substitute classification accuracy alone. A model that abstains on under-supported hypothesis/action cells must report that lost coverage instead of quietly removing those cases from the denominator.

Future versions should add:
- site-cluster bootstrap confidence intervals;
- per-outcome and per-modality reliability diagrams;
- calibration intercept/slope where appropriate;
- expected calibration error as a secondary descriptive statistic, not a proper scoring rule;
- prediction-set coverage and selective risk for an explicit OOD/abstention policy;
- confidence intervals for calibration and evaluation coverage.

The current implementation does not calculate confidence intervals, ECE, conformal sets or OOD detection.

## 5. Distribution-shift and closed-loop evaluation

An offline likelihood table may perform well on random held-out frames while failing once a planner selects observations. The benchmark must therefore include two experiments:

### Open-loop held-out evaluation

Freeze Training/Calibration/Evaluation by complete site, estimate likelihoods on Training only, score Calibration and then score Evaluation once. Report all action/hypothesis cells, including unsupported ones.

### Planner-induced distribution shift

Freeze the likelihood model before running the planner. Let the planner select actions and sites from the declared admissible set. Compare predicted outcome distributions with the observations actually acquired under that policy. Report:
- one-step and k-step predictive NLL/Brier at horizons 1, 5, 10 and 25 where the scenario supports those horizons;
- predictive error under passive/fixed, random, entropy-greedy, EVSI and survival-conditioned policies;
- uncertainty / abstention when a chosen action-state combination lies outside the calibration envelope;
- site-level and geology-class-level score changes;
- energy/time/data/compute/bandwidth budgets and actual observations returned.

Do not refit the likelihood model on Evaluation data mid-comparison. Online adaptation requires a separate, versioned prequential evaluation protocol with delayed labels and immutable model-update receipts.

## 6. Required baselines

Compare:
1. Unsmoothed empirical maximum likelihood.
2. Symmetric Dirichlet categorical estimate (the v0.1 reference).
3. Uniform outcome likelihood.
4. A preregistered modality/expert baseline where credible.
5. Hierarchical/Bayesian or ensemble likelihood model as later research.
6. An explicit unknown/OOD/abstention model.

All baselines receive the same Training cases, hypothesis/outcome schema and resource/compute assumptions.

## 7. Proposed benchmark targets

These are programme targets, not achieved results or universal statistical standards.

| Benchmark | Proposed acceptance target |
|---|---|
| SiteSplitLeakageBench | Zero site overlap across Training/Calibration/Evaluation |
| AcquisitionLineageBench | Zero duplicate acquisition-lineage rows contributing independent counts |
| UnknownLikelihoodBench | Unknown class remains explicit; under-supported unknown cells emit no likelihood vector |
| ProperScoreBench | Report Brier and NLL; beat uniform and unsmoothed-MLE baselines on Evaluation with site-cluster confidence intervals |
| CalibrationShiftBench | Report calibration degradation across leave-site-out and leave-geology-class-out evaluations; abstain when support is insufficient |
| PlannerInducedShiftBench | Report open-loop versus closed-loop score gap under matched policy/action budgets |
| LikelihoodCoverageBench | Always publish scored/total coverage; never omit unsupported cells from denominator |
| ModelVersionLineageBench | Each likelihood table binds dataset ID, estimator ID, α, minimum support and Training-only provenance |

A headline superiority claim requires the evaluation sites, split manifest, outcomes, estimator configuration and score implementation to be frozen before Evaluation.

## 8. Implementation status

Standalone Rust package:
\`crates/domains/symthaea-subterranean/research/subterra-likelihood-calibration/\`

The dependency-free reference implements:
- site-disjoint split validation;
- one case per site/action;
- unique acquisition lineage and case IDs;
- mandatory explicit unknown hypothesis;
- Training-only Dirichlet categorical likelihood estimation;
- minimum support with abstention/no emitted vector;
- separate Calibration and Evaluation Brier/NLL scoring;
- table re-derivation before holdout scoring to reject tampered likelihood cells;
- guarded conversion to one-step EVSI inputs using caller-supplied priors, decision losses, resource costs and external gate dispositions.

Nineteen unit-test cases are authored. Rust formatting, compilation, tests and Clippy have not been executed in the current authoring environment because cargo/rustc/rustfmt are absent. No empirical calibration claim is made.

## 9. Non-claims

This estimator does not authenticate labels or observations, infer a prior over geological hypotheses, solve sensor correlation, model temporal drift, detect OOD by itself, guarantee calibration under distribution shift, qualify a planetary site or establish that a selected measurement is safe. It produces a bounded empirical likelihood reference whose adequacy must be established on independent site-held-out evidence before planner use. A guarded helper converts only fully supported cells into one-step EVSI inputs; it does not authenticate or recalibrate the empirical likelihoods.
