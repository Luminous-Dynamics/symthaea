# Atelier Lightshell — Robust Zone Candidate Decision Contract v0.1

## Purpose

Convert frozen zone requirements and existing evidence references into auditable candidate dispositions and a conservative, physical-only Pareto shortlist. This is the application-level decision adapter between #12742 zone requirements, #12751 coupon evidence, #12759 method applicability and #12761 mission profiles.

It composes #11746 inverse textile design, #6225 MAT-UQ, #6226 SCI-EXP and #6298 MAT-ACTIVE. It does not implement their generic optimization, calibration, DOE or experiment-selection machinery.

## Why this contract is necessary

Prediction accuracy and reported uncertainty do not guarantee reliable engineering decisions under material, lot, machine or process shift.

Dale et al. (24 October 2026) evaluate uncertainty estimation/calibration in materials active learning and report that calibration generalization to OOD materials varies, and the evaluated calibrated uncertainties did not reliably reduce OOD data requirements relative to random or uncalibrated sampling. This is a warning against over-trusting uncertainty, not textile-specific performance evidence. https://doi.org/10.1038/s41524-026-02368-4

Wan et al. (25 October 2026) evaluate controlled distribution shifts in materials-property prediction and report that ID calibration does not reliably transfer and errors can grow faster than predicted uncertainty. https://doi.org/10.1016/j.commatsci.2026.115072

Lan et al. (29 May 2026) examine bootstrap uncertainty bias in small-sample materials active learning. Their finding that bootstrap sampling can overestimate uncertainty around influential peaks reinforces that uncertainty values require exact estimator and domain semantics. https://doi.org/10.1038/s41524-026-02157-z

## Decision stages

- `exploratory`: proposed requirements or calibrated predictions may create a research shortlist. Outputs remain predictions or insufficient evidence; no module-test eligibility is granted.
- `downselection`: requires frozen robot/geometry/zone/mission bindings, exact candidate construction/process identity, manufactured article or coupon-batch identity, transition identity where applicable, and qualified manufacturing route, physical coupon evidence and applicable methods.
- `confirmatory`: additionally requires a preregistered confirmatory plan. Adaptive exploratory results cannot silently become confirmation.

## Candidate evidence contract

Each physical property interval binds:
- exact zone metric, unit and interval semantics;
- MAT-UQ uncertainty profile;
- Lightshell coupon evidence and test-run references;
- method-applicability selection and raw-data digest;
- exact material lot, process revision and construction identity;
- manufactured article or coupon-batch identity for physical down-selection;
- exact transition identity whenever a transition-specific metric is used.

A calibrated model prediction instead binds the exact predictor generation, MAT-UQ calibration receipt, calibration domain and uncertainty profile. The receipt's predictor generation and calibration domain must match the prediction and decision target. MAT-UQ remains responsible for validating those upstream receipts and their statistical calibration; this adapter checks identity and consumes the declared applicability state.

Simulation, supplier claims and synthetic examples are useful for exploratory research but cannot become physical evidence or physical Pareto-front members.

## Robust hard-gate semantics

Let a measured or admitted interval be `[L, U]`.

- Higher-is-better requirement with minimum `T`: robustly pass only if `L >= T`; definitely fail if `U < T`; otherwise uncertainty-blocked.
- Lower-is-better requirement with maximum `T`: robustly pass only if `U <= T`; definitely fail if `L > T`; otherwise uncertainty-blocked.
- Bounded requirement `[A, B]`: robustly pass only if `L >= A` and `U <= B`; definitely fail if `U < A` or `L > B`; otherwise uncertainty-blocked.

An interval that crosses a hard limit does not pass. A mean or best-case value cannot override a violating conservative bound.

## Conservative Pareto order

Do not combine impact, airflow, vibration, mass and repair into one weighted score. Pareto objectives are separate, zone-specific properties.

Candidate A robustly dominates candidate B only if, for every objective:
- for higher-is-better, A's lower bound is at least B's upper bound;
- for lower-is-better, A's upper bound is at most B's lower bound;
- and at least one objective is strictly better under those interval comparisons.

Overlapping intervals remain incomparable. The result is a set of candidates, not a forced winner. The physical Pareto front includes only candidates that are robustly feasible for further module testing and have physical, in-domain, identity-matched evidence.

## Dispositions

- `robustly_feasible_for_module_test`: every hard gate is fully inside its acceptable conservative interval; physical evidence and process capability are bound; ready for the next module-level test only.
- `predicted_robust_feasible_exploratory`: calibrated, in-domain prediction intervals clear hard gates during exploration; never eligible as physical test evidence.
- `infeasible`: a hard gate is demonstrably violated.
- `uncertainty_blocked`: one or more intervals overlap a hard-gate boundary.
- `process_blocked`: manufacturing route/capability is missing, pending or not capable.
- `applicability_blocked`: evidence is near-boundary, OOD or applicability-unknown.
- `insufficient_evidence`: required metric, UQ, specimen, method or provenance evidence is absent or non-physical.

## P0 adversarial controls

Authored tests cover missing hard-gate property; interval crossing a limit; definite hard-gate failure; process pending; OOD narrow prediction; prediction calibration receipt with wrong predictor generation or domain; material-lot/process-revision mismatch; supplier-claim promotion; overlapping Pareto intervals; robust dominance; frozen robot/zone binding; confirmatory-plan requirement; and attempted weighted-score policy weakening.

## Benchmark programme

1. **False-feasible rate:** zero false physical passes on an immutable adversarial interval/evidence corpus.
2. **UQ transfer:** calibration/coverage and interval-width diagnostics split by construction family, material lot, machine and process revision; never rely only on random in-distribution splits.
3. **OOD abstention:** measure error, coverage and abstention as construction/process/lot shifts increase.
4. **Pareto correctness:** compare implementation output against exhaustive interval-arithmetic oracle fixtures.
5. **Robustness:** evaluate decision stability across independently manufactured lots and process runs.
6. **Physical hit rate:** proportion of down-selected candidates that satisfy frozen requirements on held-out confirmatory lots.
7. **Experiment efficiency:** physical specimens required per module-test candidate, decision regret, and reduction in unresolved candidate ambiguity.

Compare against point-estimate weighted scoring, conventional surrogate optimization, random candidate selection, Bayesian/ensemble acquisition and an expert baseline under equal material/test budgets. Weighted scoring is a comparator only, not the decision policy.

Numeric improvement targets must follow a frozen physical baseline, lot/process hierarchy, measurement-system analysis, and protected holdout partition.

## Claim boundary

`robustly_feasible_for_module_test` is not textile qualification, PPE, robot safety approval, product certification or authorization to manufacture. It only makes a candidate eligible for the next physical engineering test. The engineering owner retains the final material decision.

Physical evidence is rejected when its lot, process revision or construction differs from the candidate record. A candidate using transition-specific metrics must bind the exact transition candidate/interface identity. Down-selection candidates with a qualified process must also bind the exact manufactured article or coupon-batch identity. These checks prevent a good measurement from being silently reused for a different physical article.
