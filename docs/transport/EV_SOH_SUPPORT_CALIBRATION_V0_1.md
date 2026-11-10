# SOH Support-Conditioned Calibration Audit v0.1

**Status:** Rust-first support/uncertainty evidence audit. Not an SOH predictor or proof of calibration under arbitrary distribution shift.
**Owner:** [EV-SOH-SUPPORT-CALIBRATION-001 #11390](https://github.com/Luminous-Dynamics/symthaea/issues/11390).
**Parent benchmark contract:** [SOH benchmark manifest v0.1 PR #11342](https://github.com/Luminous-Dynamics/symthaea/pull/11342).

## Research anchor

Wu et al., “Uncertainty-Aware Transformer for Lithium-Ion Battery State-of-Health Estimation,” first published 1 October 2026: https://doi.org/10.3233/ATDE260884

The paper evaluates four NASA PCoE NCA cells using 200-sample partial-charge voltage windows. It reports a within-cell chronological 95% interval PICP of 0.653; under leave-one-cell-out, in-range held-out cells had aggregate PICP 0.965, while out-of-range cells had PICP 0.697. B0006 ages to about 0.57 SOH, below the training range. These are results for four cells and one protocol, not general guarantees. They motivate support-conditioned reporting, not an assumption that this model or its detector is generally calibrated.

## Paradigm

A single marginal coverage number can conceal undercoverage in the region that matters most. This audit therefore binds every held-out prediction and target to a predeclared training-support score, classifies samples into in-support, boundary and extrapolation strata, and independently replays each stratum's coverage, width and error.

At deployment the true target is not available, so the system cannot directly observe live coverage. It can compute a label-free support score and choose to abstain. Held-out labels are used only to evaluate whether that score predicts conditional miscoverage.

## Contract and hard gates

- Exact parent benchmark-manifest, dataset, split and model digests.
- Group-disjoint unseen-cell/module/pack/vehicle/lab/domain evaluation modes in v0.1.
- Physical train, validation/calibration and test group identity sets are explicit and disjoint.
- Support-score method, feature schema and score outputs have digests; fit scope is train-only; validation/test labels are forbidden.
- Support direction is fixed: larger score means less support. In-support and extrapolation thresholds are frozen before the benchmark split.
- Every test sample carries physical group identity, support score/evidence, observed target/evidence and nested 90%/95% predicted intervals.
- Each physical group must stay inside one support stratum. The audit requires at least two independent test groups in each of in-support, boundary and extrapolation.
- Validation-only calibration groups must be a subset of the validation partition and disjoint from test. External calibration must bind a source digest and remain disjoint from development and test groups.
- Abstention cutoffs are predeclared and strictly increasing.

## Computed outputs

For each support stratum the auditor computes:
- PICP at 90% and 95%;
- Wilson 95% lower confidence bounds for empirical interval coverage;
- mean interval width at both levels;
- MAE and RMSE;
- 90% and 95% interval scores (proper interval scoring rules);
- undercoverage flags using predeclared engineering tolerances.

The auditor also computes a risk-coverage curve at the frozen support-score cutoffs: retained sample/group count, retained fraction, MAE, RMSE and worst-group MAE. A model with good marginal accuracy but poor extrapolation coverage is accepted as a *reproducible audit record* with an explicit SUPPORT_STRATUM_UNDERCOVERAGE_DETECTED status. That is not a successful calibration gate and cannot be advertised as calibrated deployment performance.

## Frozen synthetic corpus

Ten fixtures: two accepted results (one support-conditional result within tolerance, one with OOD undercoverage that must be surfaced) and eight rejected controls covering test-label leakage into support scoring, threshold freeze after the split, calibration/test group overlap, missing extrapolation coverage, duplicate sample identity, test/validation group overlap, non-nested intervals and non-monotonic abstention cutoffs.

These fixtures validate arithmetic and evidence gates only; they are not physical battery evidence.

## Limits

Split conformal and other calibration methods do not automatically provide distribution-shift-proof guarantees. Their assumptions (e.g. exchangeability or a specified covariate-shift model) must be stated and tested. The audit does not train the support detector, estimate physical battery uncertainty, determine that an OOD score is causal, or authorize BMS action.
