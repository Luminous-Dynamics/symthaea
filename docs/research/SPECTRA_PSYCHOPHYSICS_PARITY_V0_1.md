# SPECTRA Psychophysical Adaptive-Design Parity — Research Specification v0.1

**Status:** Research specification; no method execution or superiority claim.
**Issue:** [#10636 — benchmark NEST, AEPsych and adaptive psychometric baselines](https://github.com/Luminous-Dynamics/symthaea/issues/10636)
**Programme:** [#10490 — robust goal-directed sequential design](https://github.com/Luminous-Dynamics/symthaea/issues/10490)
**Benchmark owner:** [#10601 — locked allocator comparison](https://github.com/Luminous-Dynamics/symthaea/issues/10601)

## 1. Objective

Compare SPECTRA against current psychophysical adaptive-design methods on matched tasks, data-generating processes, estimands and resource budgets.

Comparing only against QUEST+ and older GP approaches would omit relevant 2026 competition. The benchmark must include NEST and AEPsych, while separating shared psychometric-function tasks from SPECTRA's more specific UVSD inference and model-discrimination task.

## 2. NEST (2026): direct high-dimensional comparator

Bruin, Kosinka & Tursun, *NEST: Neural estimation by sequential testing*, Vision Research 238 (January 2026), DOI [10.1016/j.visres.2025.108710](https://doi.org/10.1016/j.visres.2025.108710).

The authors describe a neural-network psychometric function estimator with a novel acquisition function. Their paper reports up to 85% fewer trials to convergence in its multidimensional psychophysical settings, compares with GP active-learning methods and QUEST+ on applicable tasks, and demonstrates real psychovisual use cases.

Public implementation: [Bruin96/nest-1, v0.9.1](https://github.com/Bruin96/nest-1/tree/v0.9.1).

The 85% figure is a result within the paper's own tasks and protocol. It is not a transferable expected improvement for SPECTRA or UVSD. Reproduce the published tasks and comparators before testing any extension.

## 3. AEPsych: maintained psychophysics framework

[AEPsych's current documentation](https://aepsych.org/docs/introduction) describes a human-in-the-loop framework with flexible GP psychometric models, active-learning objectives and high-dimensional psychometric-field estimation. However, the current PyPI metadata lists version 0.8.0, released 11 April 2025. Therefore, treat the tagged package and the moving upstream GitHub branch as distinct comparator artifacts: pin both the package version and the exact source commit used, and report API/dependency differences. Do not assume that current documentation implies a recent PyPI release cadence.

## 3A. Psyphy (Flatiron Institute): JAX/GPU Bayesian active learning

[Psyphy](https://github.com/flatironinstitute/psyphy) is an open-source JAX framework for psychophysics, using GPU acceleration and approximate inference for Bayesian active-learning experiments. Its current focus is human color perception, with stated adaptation potential to other modalities. The upstream repository declares an MIT license and includes a committed `uv.lock`; record the exact commit and lockfile used rather than relying on a floating main branch. Its documentation includes examples on recovering Weber's law, Wishart psychophysical-process modeling and reproducing a 2025 study.

Include Psyphy as a serious contemporary comparator on its supported color-perception / spatially varying psychophysical tasks. Pin its exact source commit, `uv.lock`, JAX/CUDA stack and data. Its GPU-oriented inference architecture makes it a useful compute/latency comparator against NEST, AEPsych and SPECTRA. Do not assume its present color-perception model or evaluation applies directly to UVSD parameter recovery.

[AEPsych](https://aepsych.org/docs/introduction) is a psychophysics framework with flexible GP models, active-learning objectives, a benchmark suite, a client/server presentation architecture, and integrations for PsychToolbox/PsychoPy and BoTorch/GPyTorch.

Its documentation describes applications with six or eight stimulus dimensions and positions its models as flexible alternatives to parametric procedures such as QUEST+ and psignifit.

The current PyPI listing identifies AEPsych 0.8.0, released 11 April 2025, and a CC-BY-NC 4.0 license. Pin the package release and upstream source commit separately; record the exact model, acquisition function, inference method, configuration and transitive dependencies. Treat the release age as a maintenance / reproducibility variable, and do not incorporate AEPsych into a commercial SPECTRA production stack without separate legal permission. The framework is a comparator, not a required SPECTRA production dependency.

## 4. Comparator registry

Include only on tasks where the method's assumptions and estimands apply:

- NEST v0.9.1 (MIT-licensed public implementation; pin exact tag/commit).
- AEPsych 0.8.0 (PyPI, CC-BY-NC 4.0) and a separately pinned upstream source commit, with pinned model and acquisition function; comparator-only unless commercial permission is obtained.
- Psyphy (Flatiron Institute; MIT-licensed, JAX/GPU; pin exact commit and committed `uv.lock`), on its supported color-perception and psychophysical-process tasks.
- Psyphy with a pinned source commit, `uv.lock`, JAX version and CPU/GPU configuration.
- QUEST+ and other parametric Bayesian threshold procedures.
- Psi, Psi-marginal, Psi-grid and Psi-marg-grid.
- ZEST and standard adaptive staircases.
- Method of constant stimuli (MOCS), as a non-adaptive reference.
- GP active-learning baselines such as BALD, expected absolute volume change (EAVC) and GlobalMI where the implementation supports the task.
- SPECTRA's current UVSD procedure and robust/audit-aware variants.

The 2026 *Comparison of methods for quick estimation of psychometric thresholds* study provides an additional comparator design for bias, precision, repeatability and random-responder detection. Reproduce its task family and assumptions before extending it; keep monotonic and non-monotonic profiles separate.

## 5. Track A — shared psychometric-function parity

Run tasks where all applicable methods estimate the same psychometric function or threshold.

Begin with NEST's published synthetic functions where available:
- WEI1D–WEI4D;
- SIN2D and MAX2D;
- DN2D, NV2D and ON2D;
- HART6 and PS8D.

Do not require QUEST+ on functions that do not have the parametric form it requires. Report inapplicable rather than failed.

For every method, freeze:
- the oracle function and parameter ranges;
- noise, lapse, observer heterogeneity and response-channel settings;
- candidate stimuli / feasible action set;
- prior and initialization;
- hyperparameter tuning and development-data budget;
- trial, participant-time and compute budgets;
- prediction grid, threshold estimand and evaluation loss.

Use common random numbers only where explicit matched replication identities are available and valid. No comparator may access held-out oracle parameters or confirmatory seeds.

## 6. Track B — UVSD scientific parity

Evaluate SPECTRA's actual UVSD estimands and equal-variance versus unequal-variance model discrimination separately.

NEST and AEPsych can be valuable adaptive psychometric function estimators, but their published Bernoulli/psychometric surface results do not establish UVSD parameter recovery or the validity of UVSD model selection.

Use the known-truth observer DGP and observed-only fitting path. Include:
- equal-variance null and unequal-variance alternatives;
- near-boundary variance ratios;
- criterion drift, lapse and stimulus-dependent lapse;
- participant and session heterogeneity;
- evidence-dependent missingness and class imbalance;
- heavy-tailed evidence and mixtures;
- confidence/category and RT channels;
- timeout and timing quantization;
- joint mechanism interactions and held-out target-distribution shift.

Keep oracle parameters and truth labels evaluator-only.

## 7. Metrics

Report full trial-budget learning curves and failure-inclusive outcomes.

### Estimation and predictive quality
- target bias, RMSE and interval coverage/availability;
- threshold and surface integrated RMSE;
- held-out log score, Brier score and calibration;
- model-discrimination type-I error, power and false unequal-variance declarations;
- lapse / random-responder detection sensitivity and specificity.

### Scientific defensibility
- confidence amplification and false-adequacy rate;
- target-distribution and worst-scenario risk;
- performance under model misspecification and distribution shift;
- unsupported regime and non-evaluable disposition rates.

### Operational efficiency
- trial count and participant-time to a predeclared precision target;
- compute time, memory and per-decision latency;
- setup / tuning cost and reproducibility burden;
- stability and audit coverage of selected actions.

Reproduce NEST's AUC-based measures using the same task definitions and integration convention. Also report fixed-budget endpoints and each task independently. Never translate a cross-paper AUC difference directly into a number of trials saved on a different task.

## 8. Fairness, versioning and reproducibility

- Pin NEST v0.9.1 and its MIT license; AEPsych 0.8.0 plus exact upstream commit and CC-BY-NC 4.0 license; Psyphy source commit / `uv.lock` / JAX and CUDA configuration and MIT license; QUEST+/Psi implementation and all transitive dependencies. Record license compatibility separately from scientific performance.
- Give every comparator equal access to development data and comparable tuning budgets.
- Do not use oracle parameters, test seeds or evaluator-only stress worlds in model selection or hyperparameter tuning.
- Preserve numerical failures, convergence failures, timeouts and unsupported settings in denominators.
- Keep participant-time, CPU time, memory and setup effort as separate resources.
- Use a sealed confirmatory evaluation after all methods/configurations have been frozen.
- Target at least 10,000 outer simulations per primary confirmatory regime where computationally feasible; otherwise justify campaign size from a preregistered Monte Carlo precision target.

## 9. Proposed qualification targets — not results

- Reproduce published baseline rankings on the matched published tasks within uncertainty and documented implementation differences.
- In UVSD tasks, no method is superior on threshold RMSE alone if it fails model-discrimination, coverage or model-adequacy gates.
- At least 15% lower participant-time to a preregistered target precision than the best applicable baseline in supported regimes, without worse coverage or false-positive control.
- No more than 5% false unequal-variance declarations under the equal-variance null at nominal 5%, with Monte Carlo confidence bounds.
- Publish per-task performance frontiers, including cases where SPECTRA loses.
- 100% replay of SPECTRA's action and allocation receipts.

## 10. Relationship to existing SPECTRA tracks

- #10555 supplies the known-truth observer and measurement-channel generator.
- #10572 supplies observed-only UVSD recovery.
- #10582 supplies held-out predictive model adequacy.
- #10602 supplies known-truth inferential error amplification.
- #10601 owns the locked allocation / ranking-and-selection campaign.
- #10618 owns hidden scenario-bank and out-of-bank robustness.

Do not combine shared psychometric-function parity, UVSD recovery, Monte Carlo PCS and scenario-bank assurance into a single aggregate score.

## 11. Research references

- Bruin, Kosinka & Tursun (2026), [NEST: Neural estimation by sequential testing](https://doi.org/10.1016/j.visres.2025.108710), [implementation v0.9.1](https://github.com/Bruin96/nest-1/tree/v0.9.1).
- [AEPsych documentation](https://aepsych.org/docs/introduction).
- Watson (2017), [QUEST+: A general multidimensional Bayesian adaptive psychometric method](https://doi.org/10.1167/17.3.10).
- Fink, König & Frey (2026), [Multidimensional Bayesian adaptive testing](https://doi.org/10.3758/s13428-026-03123-9). This is an architectural comparator for Bayesian item/person parameter uncertainty in adaptive testing, not automatic UVSD parity.
- [Comparison of methods for quick estimation of psychometric thresholds](https://pmc.ncbi.nlm.nih.gov/articles/PMC13439673/), for bias, precision, repeatability and random-responder detection on its supported task family.

## Non-goals

- No direct transfer of NEST's reported trial reduction to UVSD.
- No treating QUEST+ as applicable to nonparametric functions outside its model class.
- No oracle information to adaptive policies.
- No production dependency changes or human-subject authorization.
- No SOTA/superiority statement before task-matched, held-out, failure-inclusive replication.
