# Marine held-out prediction-interval calibration (V0)

## Purpose

This evaluator asks whether a model's declared **central predictive interval** achieves its stated coverage on an independent hold-out, and whether that coverage is obtained without needlessly wide intervals.

It is deliberately separate from:
- point residuals and RMSE in the validation scorecard;
- conservative numerical/reference bounds in the interval comparator;
- model-discrepancy, input-condition, measurement, numerical/discretization and dependence assessments;
- experiment-selection policy and design optimization.

A conservative error bound is not a probability interval. Do not use this module to score standard deviations, expanded uncertainty, confidence intervals with a different estimand, tolerance limits, or one-sided bounds as though they were central prediction intervals.

## Metrics

For a central interval \([l_i,u_i]\) with declared nominal coverage \(1-\alpha\) and observed outcome \(y_i\), coverage is the inclusive event \(l_i \le y_i \le u_i\).

For every frozen operating-case × observable × unit × nominal-level slice the report contains:
- sample count and covered count;
- empirical coverage plus a Wilson 95% score interval;
- mean interval width and width normalized by a predeclared unit-consistent scale;
- mean central interval score and normalized score;
- counts by applicability status;
- per-slice coverage/score gates and an overall fail-closed disposition.

The central interval score is

\[
S_\alpha(l,u;y) = (u-l) + \frac{2}{\alpha}(l-y)\mathbf{1}(y<l) + \frac{2}{\alpha}(y-u)\mathbf{1}(y>u).
\]

It rewards narrow intervals when they cover and penalizes misses in proportion to distance and nominal tail probability. It prevents a policy from optimizing coverage by making intervals arbitrarily broad. Coverage, width and score are reported separately; none is replaced by one global scalar.

Wilson bounds are evaluated over declared independent evaluation groups. To block easy inflation of nominal sample size, the code rejects repeated records for the same evaluation group and slice; all duplicate records for that group/slice are excluded. This is a conservative contract, not a universal method for aggregating repeated laboratory measurements. Build a separate, reviewed aggregation/dependence model when one experimental run legitimately yields repeats or multiple correlated samples.

## Evidence and failure semantics

The frozen profile binds the calibration manifest, evaluation manifest, prediction artifact, model configuration, split manifest, interval method/revision, disjoint calibration/evaluation group IDs, and every per-slice threshold. The canonical profile digest changes when any acceptance-relevant field changes.

The evaluator rejects:
- matching calibration/evaluation manifests or overlapping declared groups;
- a profile that says the model was not frozen before hold-out use;
- duplicate observation identities or repeated group/slice rows;
- mismatched model, prediction artifact, evaluation dataset or interval method;
- noncentral/post-outcome predictions, invalid nominal level, nonfinite/reversed limits or unknown slice;
- a missing declared evaluation group or insufficient per-slice count.

A valid but borderline/out-of-domain/unknown applicability status still receives descriptive metrics, but forces DomainNotEstablished rather than a calibration pass. A clear per-slice threshold failure is retained even if a different slice lacks coverage.

The digests are caller-supplied identity bindings unless raw artifact bytes are independently hashed and checked upstream. A split digest does not prove lawful access, source authenticity, no prior hold-out inspection, or independence of the declared groups; the freeze flag is an attestation. The Wilson interpretation requires an appropriate independent-group sampling design. Coverage on one empirical distribution does not guarantee coverage after arbitrary domain shift.

## Research and benchmark plan

### Synthetic falsification first

Use deterministic synthetic outcomes with a hidden scorer and policy-inaccessible answer key. Freeze model/prediction artifact and thresholds before evaluating the hold-out. Include:
1. calibrated central intervals at 50%, 80%, 90%, and 95%;
2. undercoverage from biased centers or under-dispersed interval widths;
3. overwide intervals with nominal coverage but poor interval score;
4. heteroscedastic error by speed/depth/load stratum;
5. group-correlated repeats and duplicated group/slice rows;
6. out-of-domain shallow-water/depth transitions;
7. missing, censored, invalid and nonfinite observations;
8. changed model, split, prediction artifact and post-outcome predictions;
9. no-effect and misleadingly favorable aggregate cases where one critical slice fails.

### Proposed gates, not achieved results

- Unit/property tests for known-answer interval scores and Wilson bounds, profile digest sensitivity and input-permutation determinism.
- Zero false passes for stale digest, split overlap, duplicate independent-group, invalid interval, wrong prediction artifact/model, or unknown/out-of-domain controls.
- A large synthetic benchmark should target at least 10,000 independent held-out groups per major predeclared stratum for the 95% calibration study. A target empirical-coverage band around 93–97% may be a starting hypothesis, but the frozen protocol must assess the Wilson interval against its declared band and separately report sharpness/score.
- Compare predictive methods at identical held-out groups; include a point-only residual model, fixed-width interval baseline, residual-quantile baseline and any proposed adaptive interval method. Use paired per-group comparisons and uncertainty on differences.
- Keep calibration data, threshold tuning and final confirmation disjoint. Never tune an interval method after viewing final confirmation outcomes and then call that set held out.
- Do not aggregate across hull families, depth/draught ratios, observables or operating regimes to conceal a local failure.

### Physical reference path

The 2026 Duisburg Test Case paper describes publicly available calm-water resistance and self-propulsion experiments for a post-Panamax container-ship model at depth/draught ratios \(h/T=4.0, 2.0, 1.2\). The source is a promising validation target because shallow-water regime change can expose optimistic extrapolation. The raw tables/files, exact byte hashes, reuse rights, units, experimental uncertainty and group/split mapping have **not** been admitted by this implementation. They must be reviewed before any values are used. Do not substitute the paper abstract for raw evidence.

Research references:
- Gneiting & Raftery (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*, Journal of the American Statistical Association: https://doi.org/10.1198/016214506000001437
- Gneiting, Balabdaoui & Raftery (2007), *Probabilistic Forecasts, Calibration and Sharpness*: https://doi.org/10.1111/j.1467-9868.2007.00587.x
- ITTC 7.5-03-01-01 (2024 Revision 05), *Uncertainty Analysis in CFD Verification and Validation*: https://www.ittc.info/media/11950/75-03-01-01.pdf
- ASME VVUQ 20.1:2024 describes a multivariate validation metric; this crate does not implement that standard: https://www.asme.org/codes-standards/vvuq-standards
- Lopes et al. (2025), *Resistance prediction using CFD at model- and full-scale and comparison with measurements*, Ocean Engineering: https://doi.org/10.1016/j.oceaneng.2025.120367
- *Duisburg test case: post-Panamax container ship for benchmarking resistance and propulsion under shallow water conditions* (published online 27 March 2026): https://doi.org/10.1080/09377255.2026.2644800

## Claim ceiling

A passing synthetic fixture establishes only that this implementation enforces the declared score/profile contract on those test cases. It does not establish real marine predictive calibration, CFD accuracy, a resistance or energy improvement, scale-effect accuracy, economic/environmental superiority, seaworthiness, certification, or operating authority. Those claims require appropriate admitted data, domain-specific verification/validation, traceable measurement uncertainty, dependence review and qualified engineering judgment.

## Local verification status

The implementation includes authored Rust tests, but they have not been compiled or run in this environment because Rust/Cargo tooling is unavailable. The PR must remain draft until **cargo metadata --locked**, **cargo test --locked**, formatting and Clippy run successfully.
