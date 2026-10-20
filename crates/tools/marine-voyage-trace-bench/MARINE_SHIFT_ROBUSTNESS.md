# Marine conditional-calibration and regime-shift benchmark (V0)

Status: protocol proposal; no benchmark results claimed.

## Purpose

This benchmark is the adversarial sequel to the pointwise validation scorecard and central prediction-interval calibration evaluator. It asks whether a method's coverage, interval score and engineering decision remain trustworthy when operating conditions differ from the calibration distribution.

A pooled nominal-coverage result is insufficient. A model can achieve good aggregate coverage while systematically under-covering a critical regime. Conversely, broad intervals can achieve coverage without providing useful information. Both failures must remain visible.

This document specifies a benchmark, not an implementation of conformal prediction, a CFD solver, a method for deriving physical uncertainty bounds, or a release authority. Use the runtime scorer in `prediction_interval_calibration.rs` for the already implemented per-slice coverage/score contract. Implement additional shift diagnostics only behind this distinct benchmark owner and after review; do not duplicate an existing conformal/uncertainty owner.

## Recent research basis

1. Braun, Holzmüller, Jordan & Bach (ICML 2026), *Conditional Coverage Diagnostics for Conformal Prediction*, introduce excess risk of target coverage (ERT). Their paper notes that marginal conformal guarantees alone do not diagnose local/conditional coverage and reports greater statistical power than simpler coverage-gap diagnostics in their experiments. This is a research metric to compare against, not a claim that this crate implements ERT: https://proceedings.mlr.press/v306/braun26c.html
2. Han & Qu (2026), *The Label Complexity of Class-Conditional Coverage under Distribution Shift*, reports a marginal-coverage/per-class-coverage disconnect and an identifiability limit under joint covariate-and-outcome shift with only source labels and unlabeled target observations. This motivates a negative control: under unidentifiable shift, an honest policy must be able to say Indeterminate or request target labels, not claim a guarantee from unlabeled covariates alone: https://arxiv.org/abs/2607.18088
3. Laghuvarapu, Deb & Sun (UAI 2026), *KMM-CP*, studies kernel mean matching for covariate-shift correction and selective correction in regions with sufficient support overlap. Its reported >50% coverage-gap reduction is an outcome on its evaluated molecular-property benchmarks, not an expected marine result: https://proceedings.mlr.press/v337/laghuvarapu26a.html
4. ITTC's 2024 procedure *Uncertainty Analysis in CFD, Examples for Resistance and Flow* explicitly frames V&V around a specified code, geometry, conditions and available benchmark information. Marine evidence must preserve the actual response variable and operating condition: https://ittc.info/media/11954/75-03-02-01.pdf
5. The 2026 Duisburg Test Case (DTC) paper reports calm-water resistance and self-propulsion model experiments for a post-Panamax container ship at water-depth/draught ratios h/T=4.0, 2.0, and 1.2. It is a promising regime-shift target, but the raw files, rights, uncertainty fields, units and group mapping are not admitted here: https://doi.org/10.1080/09377255.2026.2644800

## Shift taxonomy — do not conflate these cases

| Shift family | What changes | Which evidence can support a correction? | Required honest behavior |
|---|---|---|---|
| IID/exchangeable control | Nothing material by design | Ordinary held-out calibration groups | Baselines should not invent benefits from adaptation |
| Covariate shift | Condition mix changes while the conditional outcome/error law is assumed stable | Support overlap plus a justified density-ratio or weighting estimate | Evaluate weighted methods only under their stated assumption |
| Heteroscedasticity | Error spread differs by speed, depth, load, trim or observable | Relevant calibration evidence in each stratum or a validated model | Show local undercoverage; don't let high-coverage strata mask it |
| Conditional/subgroup miss | A subset under-covers while pooled coverage appears acceptable | Conditional diagnostics and enough group-labelled evaluation data | Report worst-stratum undercoverage and subgroup uncertainty |
| Joint covariate-and-outcome shift | Both conditions and error law can change | Target labels/measurements or defensible external constraints | No label-free universal conditional-coverage promise |
| Support gap / new regime | Target conditions lie outside calibration support | New reference data, validated physics, or explicit limits | Return OutOfDomain/Indeterminate or abstain |
| Systematic bias/drift | Center prediction moves relative to truth over time/campaign | Independent residual monitoring and new calibration evidence | Don't repair the interval by silently calling bias random noise |
| Dependent samples | Repeated waves/frames/runs share a campaign or environmental history | Group-aware analysis / reviewed dependence model | Count independent experimental groups, not raw rows |
| Sparse/missing/invalid evidence | A critical slice lacks sufficient or valid labels | Additional qualified measurement | Keep the slice incomplete or indeterminate |
| No-shift negative control | Adaptation is applied without a real distribution change | Same held-out truth, paired policy replay | Report unnecessary width/cost or degradation |

Hydrodynamic coordinates to retain include depth/draught, speed/Froude number, displacement/loading, trim, hull/appendage/propulsor generation, water conditions, measured observable, scale, solver/method revision and evidence origin. A depth ratio by itself is not a complete applicability descriptor. Model-scale validity cannot be silently promoted to full-scale accuracy.

## Arms and equal-budget protocol

Use exactly the same frozen source calibration groups and held-out evaluation cases for each comparable arm:

1. Plain split/residual-quantile interval (baseline).
2. Group/Mondrian residual quantiles where group counts are sufficient and the grouping was prespecified.
3. Importance-weighted conformal only for a preregistered covariate-shift case with validated support-overlap assumptions.
4. A selective support-aware correction that can reject/abstain outside overlap.
5. The candidate adaptive method.
6. An oracle using hidden labels/truth, only as an explicitly non-deployable upper bound.

Freeze each method revision, prediction artifact, group assignments, scenario-generator version, seed manifest, split manifest, nominal levels, cost basis and evaluation manifest before opening the final outcomes. Calibration, tuning and final confirmation must be disjoint at the level of the independent experimental group, not merely disjoint rows. For methods with a different target-label budget, report that budget rather than declaring their coverage improvement free.

No method is expected to win on all shifts. Report where assumptions hold and fail. A density-ratio correction must not run as if the ratio is trustworthy where support overlap is poor; a source-only method must not claim to identify arbitrary joint shift.

## Metrics — separate, per shift stratum

- Empirical coverage and Wilson 95% bounds at declared nominal 50%, 80%, 90% and 95% levels.
- One-sided undercoverage and overcoverage; pooled coverage; worst-stratum coverage; max conditional-coverage gap; conditional diagnostics such as ERT when reproduced by a qualified implementation.
- Mean width and the proper central interval score; compare them at equal target coverage / equal label budget where applicable.
- False-safe / false-feasible and missed-feasible outcomes for every hard engineering constraint.
- Applicability, support-overlap, unknown and abstention/Indeterminate rates.
- Per-group paired score differences, wall time, solver/measurement cost and target-label burden.
- Every attempted, failed, invalid, missing or censored sample remains in accounting.
- Deterministic replay, split leakage, stale digest, changed model/method, duplicate group and malformed-interval controls.

Never combine these into a single untraceable leaderboard score. If a project-level decision requires utility weights, publish the weights as a separate decision contract and retain the full underlying metrics.

## Preregistered acceptance goals (not results)

- Zero false passes on known-answer support-gap, joint-shift, stale-profile, duplicate-group, invalid-interval and false-feasible controls.
- No per-slice pass may be inferred from pooled coverage alone.
- For known covariate shift with sufficient overlap, compare weighted calibration and plain split calibration on the same outcomes. Claim improvement only when preregistered paired effects and uncertainty support it without violating sharpness, false-safe or cost gates.
- Under unidentifiable joint shift and no target labels, allow only bounded claims, Indeterminate or target-label requests—not a universal coverage guarantee.
- Test a no-shift control; adaptation must not be presumed beneficial.
- Eventual 95% coverage studies target at least 10,000 independent held-out groups per major predeclared stratum, with the design justified for the intended confidence width. Smaller deterministic fixtures qualify software semantics only.
- Use at least three separately versioned synthetic generators (e.g., smooth, heteroscedastic and regime-boundary), and reserve a final unmodified confirmation corpus.
- Physical superiority requires rights-reviewed raw reference data, qualified uncertainty semantics, condition-grouped hold-out, physical verification/validation, and independent expert review.

## Physical DTC admission checklist

The 2026 DTC shallow-water article page says experimental data are publicly available. That statement alone does not create an admitted dataset. Before a physical benchmark can run, an independent intake review must record:

- Exact source URL/version, raw artifact bytes and reproducible SHA-256 checksums.
- Rights/reuse basis and limitations for each file.
- Original measured observable, units, measurement chain, uncertainty method, and any omitted uncertainty terms.
- Water depth, draught, speed/Froude number, loading, trim, hull/propeller configuration, model scale and test-condition identifiers.
- Experimental/campaign group IDs, dependence concerns, missing/invalid/censored outcomes and calibration metadata.
- The exact train/calibration/hold-out split at group level, model-freeze time and proof that the hold-out was not used for tuning.
- Separation of raw experimental measurements, corrected trial estimates, CFD outputs, synthetic fixtures and derived values.

Until those checks pass, use only synthetic fixtures for software tests and keep every DTC physical-performance claim blocked.

## Implementation boundary

This protocol should initially reuse the frozen per-slice predictive interval evaluator in `prediction_interval_calibration.rs`. A follow-up runtime evaluator is justified only for genuinely new outputs (paired policy deltas, max conditional coverage gap and shift-assumption/abstention accounting). Do not copy conformal algorithms from research packages into production Rust without licensing, maintenance and independent correctness review. Do not add a second selector: experiment-choice policy remains under #7641 / #7803.

## Claim ceiling

A passing software fixture says only that the protocol/evaluator handled its declared synthetic scenarios. It does not guarantee conditional coverage under arbitrary shift, validate a marine model, demonstrate lower drag/energy, establish full-scale performance, or authorize a vessel to operate.
