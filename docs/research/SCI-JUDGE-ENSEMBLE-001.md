# SCI-JUDGE-ENSEMBLE-001 — Correlation-aware, calibrated judge ensembles

**Status:** Protocol specification and synthetic qualification plan. No ensemble implementation, human study, external replication, or empirical Symthaea result is claimed.

**Tracking:** [Issue #9431](https://github.com/Luminous-Dynamics/symthaea/issues/9431).

## 1. Purpose and non-duplication

This work qualifies an ensemble as a decision policy under correlated evaluator errors and distribution shift. It is deliberately separate from:

- [EVAL-JUDGE-METROLOGY-001 / #9022](https://github.com/Luminous-Dynamics/symthaea/issues/9022): reliability of an individual judge;
- [SCI-JUDGE-GEOMETRY-001 / #9130](https://github.com/Luminous-Dynamics/symthaea/issues/9130): matched judge–human score-space geometry;
- [SCI-FACTORIAL-JUDGE-BIAS-001 / #9243](https://github.com/Luminous-Dynamics/symthaea/pull/9243): causal nuisance-factor contrasts;
- [SCI-INTERVENTION-QUALITY-001 / #9270](https://github.com/Luminous-Dynamics/symthaea/issues/9270): validity of generated tests;
- [SCI-REFERENCE-ASSURANCE-001 / #9081](https://github.com/Luminous-Dynamics/symthaea/issues/9081): reference provenance and adjudication.

The output is not a new scalar judge score. It is a calibrated policy with an explicit uncertainty and abstention contract.

## 2. Estimands

Keep three targets separate.

### 2.1 Fixed-benchmark performance

For a frozen benchmark of (n) items, report the finite-set mean:

\[
\widehat{\mu}_{B}=\frac{1}{n}\sum_{i=1}^{n}Y_i.
\]

This describes performance on those items. It does not, by itself, estimate performance on new tasks.

### 2.2 Generalized performance

Declare the target task-family distribution (P(T)) and within-family item distribution (P(I\mid T)). The generalized estimand is:

\[
\mu_{P}=\mathbb{E}_{T\sim P(T),\ I\sim P(I\mid T)}[Y(T,I)].
\]

Report the sampling frame, family weights, item weights, exclusions and missingness. A benchmark convenience sample cannot be relabeled as a representative population without evidence.

### 2.3 Decision risk

For decision policy \(\pi\), outcome \(y\), action \(a\), and predeclared loss matrix \(L(a,y)\):

\[
R_{P}(\pi)=\mathbb{E}_{X,Y\sim P}\left[\sum_a \pi(a\mid X)L(a,Y)\right].
\]

Choose the loss matrix before seeing test results. Report false-positive and false-negative risks separately when their consequences differ. Accuracy alone is not a decision objective.

## 3. Correlated-error model

Do not infer judge independence from model display names.

Record a dependency graph containing, when known:

- base model family and version;
- provider / serving stack;
- shared fine-tuning or evaluation lineage;
- prompt and rubric ancestry;
- retrieval or reference source;
- tool chain and preprocessing;
- judge calibration-data overlap.

Report pairwise error correlation and double-fault rates, but do not use pairwise correlation as a complete model of higher-order dependence. Include explicit common-mode stress cases.

At minimum, compare:

1. strongest single judge, selected on calibration data only;
2. unweighted majority / plurality;
3. equal-weight judge-family aggregation;
4. confidence-weighted aggregation with separately calibrated confidence;
5. cross-fitted human-grounded error-rate weighting;
6. dependence-aware weighting or family-level influence caps;
7. disagreement-only / label-free weighting as a research comparator;
8. test-label oracle as a non-deployable diagnostic upper bound.

All arms must state tie, invalid-output, timeout, retry and abstention handling.

## 4. Calibration and information separation

Use three disjoint partitions:

- **Calibration:** estimate judge-specific error rates, confidence calibration and ensemble weights.
- **Policy selection:** select thresholds and any admissible model hyperparameters using a predeclared process.
- **Held-out evaluation:** estimate final performance once, without test-label-dependent weight or threshold changes.

Split by task cluster, source and family before any item-level expansion. Near-duplicates, paraphrases, generated variants and shared source documents remain in one partition. Add a temporal split and leave-task-family-out analysis when the claim concerns future or new-family generalization.

Where feasible, obtain blinded independent human adjudication or mechanical known-answer labels. Preserve individual votes and unresolved disagreement. Human labels are a reference process with uncertainty, not unquestionable truth.

A disagreement-only weighting rule is not treated as identifying judge accuracy. In the presence of correlated errors, latent-truth models need explicit identifiability assumptions and external anchors.

## 5. Calibration and selective decisions

For each judge and each ensemble arm, report:

- Brier score and log loss;
- calibration intercept and slope;
- reliability plots under a fixed, disclosed binning policy;
- ECE only alongside bin counts, bin policy and proper scoring rules;
- class-wise calibration where sample sizes permit;
- risk–coverage curves;
- risk at predeclared coverage levels;
- coverage at predeclared risk ceilings;
- abstention rate, reason and correctness among answered cases.

Confidence must have one frozen interpretation (for example, probability that the selected verdict is correct). Do not mix confidence in evidence sufficiency, preference margin, and correctness probability.

A policy must not win by abstaining on nearly every difficult item. Always show the full risk–coverage frontier and matched-coverage comparisons.

## 6. Statistical analysis

The analysis plan must distinguish a fixed-item summary from population inference.

A possible binary correctness model, subject to preregistered adequacy diagnostics, is:

\[
\operatorname{logit}\Pr(Y_{fij}=1)=
\beta_0+\beta_{\text{arm}}+\beta_{\text{shift}}+
\beta_{\text{arm}\times\text{shift}}+
\beta_{\text{difficulty}}+
u_f+v_{fi}+b_j.
\]

Here, \(u_f\) is a task-family effect, \(v_{fi}\) an item/cluster effect, and \(b_j\) a judge effect for judge-level observations. The actual random-effects structure must follow the sampling and repeated-measure design; do not include unidentifiable or unsupported terms merely to make the model look sophisticated.

For ensemble-level comparisons, preserve the paired item structure and cluster uncertainty at the highest independently sampled unit. Report model convergence, singularity, residual and influence diagnostics, variance components, weighting assumptions and sensitivity analyses.

When there are too few independent families or clusters to support a mixed-effects model, use a predeclared cluster bootstrap or a conservative finite-sample bound and narrow the generalization claim. Never bootstrap individual judge calls as though repeated calls were independent tasks.

Use multiplicity control or explicitly label exploratory subgroup analyses. Report the effective number of independent task families, not just the raw number of trials.

## 7. Matched-budget and shift design

Match total query count, token budget, latency budget, monetary cost, energy and human-review time. If exact matching is impossible, publish a Pareto frontier rather than a single cost-normalized number.

Predeclare shift families:

- task-domain and task-family mixture change;
- task difficulty shift;
- temporal drift;
- rubric or policy revision;
- model/provider version change;
- prompt and formatting perturbation;
- evidence provenance and authority cues;
- examinee-capability-dependent leniency;
- judge-family removal and shared-lineage concentration.

Separate covariate shift, label/preference shift, concept shift and measurement-instrument drift. They are not interchangeable stressors.

## 8. Synthetic known-answer controls

Before empirical scoring, the executable fixture must include at least:

- independent judges with heterogeneous error rates;
- partially correlated judges;
- a fully common-mode judge family;
- a consistently wrong majority;
- one high-accuracy but capability-lenient judge;
- a confident-but-wrong judge;
- selective abstention concentrated on difficult examples;
- task-family and temporal distribution shift;
- a judge family removed after calibration;
- duplicated calibration clusters and train/test leakage;
- malformed, missing, duplicated and post-hoc changed observations.

The suite must verify exact numerators and denominators, not just that a command exits successfully. All fixture records must be marked synthetic and non-scoring.

## 9. Metrics and acceptance logic

Report, by task family and judge dependency group where adequately powered:

- fixed-benchmark and generalized accuracy;
- balanced accuracy, precision, recall, sensitivity and specificity;
- Brier score, log loss, calibration slope and intercept;
- risk–coverage and selective risk;
- pairwise dependence, double-fault and common-mode failure;
- matched-budget cost, energy, latency and human effort;
- abstentions, missingness, failures, retries and effective independent sample counts;
- confidence intervals / credible intervals and multiplicity policy.

Use a constrained / Pareto decision rule. A mean accuracy gain cannot compensate for a critical false-positive ceiling breach, severe calibration failure, a failed important subgroup or unusable coverage.

The minimum worthwhile effect, risk ceilings, required coverage and power target must be selected before empirical data collection from the intended use case. The published 2026 results are replication targets, not universal acceptance thresholds.

## 10. Implementation sequence

1. Freeze this protocol and the sampling / adjudication plan.
2. Add strict plan, receipt and report schemas with immutable dependency and split commitments.
3. Add deterministic synthetic fixtures and hostile controls.
4. Add an independent Rust analyzer for structural integrity, counts, dependence summaries and cost accounting.
5. Add a pinned inferential backend for mixed-effects / calibration analysis, with known-answer synthetic data and independent result checks.
6. Run a blinded pilot to estimate cluster structure, variance components, calibration requirements and sample size.
7. Run the preregistered matched-budget study and a separate external replication.

The Rust analyzer should validate the declared study and statistical output; it must not pretend that a hand-coded mean or a digest is a validated GLMM implementation.

## 11. Research references

- Zhang et al. (2026), *Can We Trust LLM Judges: A Study of Capability-Dependent Biases and Multi-Judge Ensemble for Bias Calibration*: https://arxiv.org/abs/2609.12002
- Yang et al. (2026), *FairJudge: An Adaptive, Debiased, and Consistent LLM-as-a-Judge*: https://proceedings.mlr.press/v306/yang26n.html
- Roy et al. (2026), *PReMISE: Policy Rubrics as Measurement Specifications for LLM Judges*: https://aclanthology.org/2026.emnlp-main.2653/
- Xu et al. (2026), *Am I More Pointwise or Pairwise?*: https://aclanthology.org/2026.findings-emnlp.1444/
- NIST AI 800-3 (2026), *Expanding the AI Evaluation Toolbox with Statistical Models*: https://www.nist.gov/publications/expanding-ai-evaluation-toolbox-statistical-models

## 12. Claim boundary

A committed plan, passing schema test, synthetic known-answer fixture or judge consensus does not prove evaluator identity, independent execution, reviewer competence, reference truth, model adequacy, representativeness or generalization. No empirical SOTA claim is made by this protocol.
