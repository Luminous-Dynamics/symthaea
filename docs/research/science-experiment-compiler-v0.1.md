# Symthaea scientific experiment compiler v0.1

Date: 2026-11-08  
Status: research specification; no general-purpose adaptive experiment compiler claimed  
Parent: [SCIENCE-DESIGN-001 #11132](https://github.com/Luminous-Dynamics/symthaea/issues/11132)  
Programme: [CTGP #10802](https://github.com/Luminous-Dynamics/symthaea/pull/10802)

## 1. Objective

Compile a physical question into a falsifiable, resource-bounded experiment plan. The compiler should not simply rank theories, optimize a metric, or choose the cheapest sensor. It should identify measurements that separate competing explanations while exposing where the available data, calibration and measurement geometry cannot do so.

This programme composes current primitives:
- `symthaea-science-evidence::theory_prediction`: derivation-backed model/observable prediction coverage and comparability.
- `symthaea-science-evidence::model_discrimination`: scalar Gaussian model-identity expected information gain (EIG), Simpson quadrature, prior-sensitivity scenarios and cost-normalized ranking.
- `symthaea-uap-observatory::budgeted_design`: budgeted multifidelity experiment selection with greedy and exhaustive subset reference.
- `symthaea-uap-observatory::observability`: scaled data-only one-sided Jacobi SVD.
- `symthaea-uap-observatory::global_identifiability`: discrete candidate-solution ambiguity screening.
- `symthaea-science-evidence`: content-addressed manifests and exact-head verification receipts.

It is an integration and qualification workstream. Bayesian optimal experimental design, active learning, model-discriminating stimuli and factor-graph estimation are established fields; no novelty claim is made for those general ideas.

## 2. Current literature and crucial objective distinction

Golan, Schütt and Kriegeskorte, *Making models disagree to learn how brains compute*, Nature Reviews Neuroscience (published 2026-08-28), DOI 10.1038/s41583-026-01070-0, reviews optimizing stimulus sets that make competing models disagree, with a prior over admissible stimuli to balance ecological validity against model-comparison power. It emphasizes that model capacity can make models difficult to discriminate using ordinary training-distribution samples; out-of-distribution generalization is a severe test; model ensembles must represent theoretical variation separately from nuisance variation; and performance profiles across stimulus sets should not always be collapsed into one scalar.

The review makes a point that is central to this workstream: three popular design objectives are not generally equivalent.

### A. Bayesian model-identity information gain

    U_EIG(e) = I(M; Y | e)
             = H(M) - E_Y[H(M | Y, e)]

This measures expected uncertainty reduction about a discrete model label under the declared prior and predictive distributions.

### A.1 Measurement availability and missingness

Each candidate declares a measurement availability probability `q_e` in [0,1].
The initial decision model makes one explicit simplifying assumption: a
missing measurement is uninformative about model identity. Under that
assumption,

    I_expected(M; outcome | e) = q_e * I(M; Y | usable observation, e)

    E[H(M | outcome, e)] = H(M) - q_e * I(M; Y | usable observation, e)

The implementation retains conditional EIG, availability-adjusted EIG and
expected posterior entropy separately for every prior scenario. Candidate
ranking by cost uses availability-adjusted EIG per cost; a zero-availability
candidate remains visible with its conditional EIG but has zero expected
information.

This scalar multiplier is not a general missing-data model. Occlusion,
non-detection, sensor-health state, radar cross-section and visibility may
themselves carry information. Such outcomes must be included explicitly in a
future predictive likelihood rather than treated as uninformative dropout.

### B. Expected predictive discrepancy

For a data-generating model i and alternative j, an example is:

    U_KL(e; i, j) = E_{Y~p_i(.|e)}[log p_i(Y|e) - log p_j(Y|e)]
                  = D_KL(p_i(.|e) || p_j(.|e))

For multiple alternatives, choose and report the aggregation rule. A maximin variant uses the most confusable alternative:

    U_minKL(e; i) = min_{j != i} D_KL(p_i || p_j)

This measures how difficult the worst alternative is to distinguish from i. It is not generally equal to EIG.

### C. Model-recovery accuracy

    U_recovery = P(analysis procedure selects generating model)

This is an end-to-end property of the *design plus data-generation plus analysis pipeline*, not a direct information-theoretic quantity. Estimate it by repeated planted-model simulation with a frozen model-selection procedure, including model fitting, nuisance estimation, calibration and abstention.

A design that maximizes EIG need not maximize expected KL, recovery accuracy, decision utility or robustness. The benchmark must measure them separately where the oracle permits.

### D. Parameter information

Continuous-parameter uncertainty is distinct from discrete model identity. Data-only Fisher information, Jacobian rank/SVD, posterior parameter uncertainty and parameter EIG require their own estimands and comparability contracts. The current scalar model-identity EIG module is not a general parameter-design engine.

## 3. Open-world model universe

Physical research is often M-open: the generating law may be absent from the named candidate set. The explicit `OTHER_UNRESOLVED` model is not assigned an arbitrary numeric predictive distribution.

The compiler must distinguish:
- `NamedModelComparable`: all relevant candidate models provide calibrated quantitative predictions for the declared observable.
- `PartiallyComparable`: some pairwise/model predictions are missing or outside validated domain; expose exactly which comparisons are blocked.
- `OpenWorld`: unenumerated models remain possible.
- `Unresolved`: no candidate is presently adequate or identifiable from the proposed experiment.

A high score within an incomplete model set is not evidence that the top-ranked theory is true. A new observation may invalidate all current hypotheses. The benchmark includes planted M-open cases where none of the named models generated the data.

## 4. Controversial experiments without nuisance exploitation

A candidate experiment is “controversial” when legitimate physical models make meaningfully different observable predictions under the same calibrated measurement definition.

For every candidate, retain:
- the full pairwise prediction-difference matrix;
- predictive distributions and uncertainty;
- derivation provenance and domain of validity;
- feature family or physical mechanism responsible for disagreement;
- calibration/noise assumptions and nuisance parameter sensitivities;
- prior over admissible experiments and its source;
- ecological/operational validity constraints.

A candidate must not win solely because of:
- an uncalibrated observable;
- an out-of-domain extrapolation;
- a nuisance feature (sensor artifact, aliasing, clock mismatch, preprocessing leak);
- unequal information access or unequal compute given to alternative models;
- arbitrary priors or post-hoc selection of the experiment set.

Use ensembles of model instances to separate variation of theoretical interest from nuisance variation such as initialization, optimizer seed or small hyperparameter changes. A robust controversial experiment should separate model families, not just two arbitrary instances of the same family.

## 5. End-to-end compilation stages

1. **Question compiler** — convert the scientific question into named, operational observables, physical units, scope and falsification conditions.
2. **Prediction ledger** — require versioned theories, source SHAs, model-spec artifacts, quantitative derivations, uncertainty and validated domain for each observable.
3. **Comparability gate** — preserve missing/not-derived/non-Gaussian/uncalibrated states; never coerce them to zero or convenient likelihoods.
4. **Experiment space** — define candidate protocols, controls, sensor configurations, nuisance conditions, resource costs, mutual exclusion, access and safety constraints.
5. **Observability gate** — data-only rank/singular spectrum after declared gauge fixing; global-ambiguity candidate generation and tests; prior-dominated directions stay marked prior-dominated.
6. **Objective estimation** — calculate EIG, expected predictive discrepancy, model-recovery simulations, parameter information and decision utility as separate outputs.
7. **Robust portfolio selection** — optimize across prior scenarios, nuisance uncertainty, feasibility and cost; use Pareto reporting before choosing scalar weights.
8. **Execution contract** — freeze source SHA, protocol, apparatus, calibration lineage, baselines, controls, stopping rule, raw output format, exclusion rules, compute/time budget and output digests.
9. **Evidence ingestion** — retain original measurements, calibration updates, missing data, environmental context, exclusions, uncertainty and provenance.
10. **Model criticism** — update predictions, identify contradicted assumptions, preserve failed models and permit open-world unresolved outcomes.

The compiler must make the plan executable and reviewable; it does not autonomously operate physical equipment in this phase.

## 5.1 Executable Gaussian predictive-discrepancy reference

The science-evidence crate now exports:

`gaussian_predictive_kl_nats(left_mean, left_predictive_sigma, right_mean, right_predictive_sigma, measurement_sigma)`

It computes the analytic directed KL divergence between the two scalar predictive Gaussians after adding the same independent, calibrated Gaussian measurement noise to both:

    s_left  = sqrt(predictive_sigma_left^2  + measurement_sigma^2)
    s_right = sqrt(predictive_sigma_right^2 + measurement_sigma^2)

    D_KL(p || q)
      = log(s_right / s_left)
        + [s_left^2 + (mean_left - mean_right)^2] / (2 s_right^2)
        - 1/2

The function is a reference for **expected predictive discrepancy**, not model-identity EIG. It checks finite arithmetic and positive uncertainty parameters; it does not replace the model ledger's derivation/comparability validation, nor does it support correlated, non-Gaussian or vector observations yet. The caller must ensure the inputs originate from validated, same-observable predictive distributions.

The corresponding tests cover the equal-variance closed form, directed asymmetry, identical-distribution zero, and invalid/overflow inputs. Next, compare this analytic function against independent high-precision quadrature and use it to construct a held-out maximin KL design alongside (not instead of) the current EIG scorer.

## 5.2 Executable planted-model recovery

The crate now also exports `model_recovery::evaluate_model_recovery`, a deterministic, bounded scalar-Gaussian simulation of the *frozen MAP/abstention analysis procedure*.

For each supplied generator, it:
1. draws a fixed number of synthetic observations from the generator's predictive Gaussian plus calibrated measurement noise;
2. evaluates the named candidate likelihoods under the declared prior;
3. chooses the maximum-posterior candidate only when its posterior reaches a predeclared threshold, otherwise abstains;
4. records a generator-by-inference-model confusion matrix with a separate abstention column.

It reports equal-weight named-model accuracy, prior-weighted named-model accuracy, coverage and selective accuracy. An optional `OTHER_UNRESOLVED` generator is used only to create M-open test observations; it is **not** an inference candidate and is not assigned a numeric likelihood. Its false-confident assignment rate measures how often the frozen named-model procedure nevertheless forces a named answer when the true generating distribution is outside that set.

The harness is capped at one million total draws and requires a fully comparable, calibrated scalar Gaussian experiment. It does not simulate fitting model parameters, non-Gaussian/vector/correlated measurements or physical observations. Its accuracy is conditional on the provided synthetic generator family and frozen classifier; it is not an estimate of real-world theory truth.

The recovery simulator uses the same `availability_probability` as the EIG
scorer. A draw with no usable observation enters the confusion matrix's
abstention column, remains in the named-model accuracy/coverage denominator,
and increments a separate unavailable-observation count for that generator.
Recovery metrics are therefore end-to-end under the declared availability
model rather than conditionally assuming every sensor read succeeds. The v0.1
assumption still treats non-detection as uninformative; if non-detection has a
physical likelihood, the outcome model must be extended.

The executable `model_recovery_reference` is now registered in the science-evidence crate and run by the CTGP workflow. It complements rather than replaces EIG and KL.

## 6. Benchmark matrix

### Synthetic model families

- identical quantitative predictions (EIG and KL should be zero);
- well-separated, equally uncertain Gaussian predictions;
- asymmetric predictive uncertainty;
- overlapping or near-indistinguishable distributions;
- multi-modal and non-Gaussian controls;
- correlated multi-observable predictions;
- different candidate models with shared nuisance variation;
- missing derivation / outside-domain / uncalibrated prediction;
- open-world M-open generator absent from the named model set;
- prior shifts and deliberately misspecified prior scenarios;
- globally ambiguous geometry where local Jacobian rank appears full;
- model-family ensembles with held-out instances and nuisance seeds.

### Experiment families

- natural/operational protocols sampled from a declared prior;
- deliberately synthesized controversial measurement settings;
- out-of-distribution protocols held out from candidate generation;
- calibration and reference measurements;
- negative controls designed to reveal sensor artifacts;
- repeated measurements for model-recovery and coverage analysis.

### Baselines

1. random feasible experiment;
2. nominal EIG;
3. worst-prior-scenario EIG;
4. mean EIG/cost;
5. maximin pairwise expected KL;
6. expected model-recovery accuracy (estimated by simulation);
7. data-only D-optimal / A-optimal / E-optimal design;
8. budgeted greedy selection;
9. exhaustive subset oracle on small cases;
10. proposed multiobjective Pareto portfolio.

Every baseline receives the same model predictions, candidate set, calibration information, priors, nuisance assumptions, execution budget and compute cap.

## 7. Required metrics

### Predictive discrimination
- model-identity EIG;
- pairwise KL / worst-confusable-alternative KL;
- expected log-likelihood-ratio separation;
- full pairwise controversy matrix;
- predictive overlap and prior sensitivity.

### End-to-end recovery
- model-recovery confusion matrix;
- false confident selection rate;
- correct abstention rate;
- calibration of model posterior probabilities;
- recovery under held-out model instances;
- recovery in M-open (true model not in list) settings.

### Parameter and geometric identification
- data-only Jacobian rank and singular spectrum;
- weak/null parameter combinations;
- global ambiguity count and equivalent candidate residuals;
- posterior parameter interval coverage;
- sensitivity to calibration and nuisance assumptions.

### Decision and operations
- expected decision regret under a declared downstream loss;
- budget consumed, acquisition time, setup/reset/duty cycle;
- greedy-to-exhaustive optimality gap;
- feasibility rejection reasons;
- Pareto frontier across information, cost, robustness, calibration burden and operational/ecological validity.

No headline composite metric may hide failed comparability, false confident selection, poor coverage or global ambiguity.

## 8. Prospective qualification gates

- Identical predictions yield zero pairwise predictive discrepancy and zero conditional / availability-adjusted EIG within numerical tolerance.
- Availability probability is finite and in [0,1]; q=0 preserves prior entropy, q=1 recovers conditional EIG, and intermediate q scales expected EIG.
- Missing observations are retained as abstentions in end-to-end recovery; they are not discarded from the denominator.
- The analytic Gaussian special cases match independent high-precision references.
- Quadrature reports coarse/refined discrepancy, convergence and failure states.
- EIG, expected KL and model-recovery are never conflated.
- Planted true models are recovered with calibrated uncertainty on held-out model instances and design families.
- In the M-open track, the system can reject all named models and mark the result unresolved.
- Nuisance-only model variation does not produce a false theoretical discrimination claim.
- Prior sensitivity is explicit; no one prior is silently privileged.
- Locally full-rank but globally ambiguous solutions are exposed by the global screen or an explicitly incomplete ambiguity-generation status.
- Portfolio selection respects budgets, exclusivity, setup cost and feasibility; greedy optimality gap is measured against exhaustive small-case oracle.
- Out-of-distribution validation is conducted on frozen held-out domains, not selected after observing results.
- Real-science replay has pinned data provenance, calibration, license, candidate predictions, and outcome-blind candidate generation.

## 9. Next implementation increments

1. Add independent reference functions for pairwise Gaussian KL and validate exact analytic formulas, separate from EIG.
2. Add a planted-model Monte Carlo recovery harness using a frozen likelihood-ratio/posterior decision rule and explicit abstention threshold.
3. Add nuisance-ensemble aggregation: report between-family separation and within-family variability.
4. Add an M-open track and ensure named-model confidence does not force a closed-world selection.
5. Pass the same candidate experiments into both the theory-EIG scorer and budgeted portfolio selector; compare greedy vs exhaustive on the same objective contract.
6. Add observability/global ambiguity as hard feasibility constraints and retain rejected candidates with typed reasons.
7. Produce Pareto fronts for model discrimination, recovery, cost, calibration burden and operational validity.
8. Bind every candidate set, model version, experiment, seed, prior and output to science-evidence receipts.

## 10. Claim ceiling

A pass validates the design and model-selection procedure on the declared candidate family and data generator. It does not establish that the candidate family is complete, that a selected theory is true, or that a real experiment will behave as simulated. The correct output can be **none of the current models are sufficiently supported**.
