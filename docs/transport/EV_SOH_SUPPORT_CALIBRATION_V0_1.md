# SOH Support-Conditioned Calibration Audit v0.1

**Status:** Rust-first support/uncertainty evidence audit. Not an SOH predictor or proof of calibration under arbitrary distribution shift.
**Owner:** [EV-SOH-SUPPORT-CALIBRATION-001 #11390](https://github.com/Luminous-Dynamics/symthaea/issues/11390).
**Parent benchmark contract:** [SOH benchmark manifest v0.1 PR #11342](https://github.com/Luminous-Dynamics/symthaea/pull/11342).

## Research anchor

Wu et al., “Uncertainty-Aware Transformer for Lithium-Ion Battery State-of-Health Estimation,” first published 1 October 2026: https://doi.org/10.3233/ATDE260884

The paper evaluates four NASA PCoE NCA cells using 200-sample partial-charge voltage windows. It reports a within-cell chronological 95% interval PICP of 0.653; under leave-one-cell-out, in-range held-out cells had aggregate PICP 0.965, while out-of-range cells had PICP 0.697. B0006 ages to about 0.57 SOH, below the training range. These are results for four cells and one protocol, not general guarantees. They motivate support-conditioned reporting, not an assumption that this model or its detector is generally calibrated.

## Paradigm

A single marginal coverage number can conceal undercoverage in the region that matters most. This audit therefore binds every held-out prediction and target to both (1) a label-free support score produced by the detector and (2) a separate offline reference support state derived from the frozen train/test relation. Calibration is measured within reference states; the detector is evaluated against those states. This prevents a detector from defining the very OOD groups used to claim that it works.

At deployment the true target is not available, so the system cannot directly observe live coverage. It can compute a label-free support score and choose to abstain. Held-out labels are used only to evaluate whether that score predicts conditional miscoverage.

## Contract and hard gates

- Exact parent benchmark-manifest, dataset, split and model digests.
- Group-disjoint unseen-cell/module/pack/vehicle/lab/domain evaluation modes in v0.1.
- Physical train, validation/calibration and test group identity sets are explicit and disjoint.
- Support-score method, feature schema and score outputs have digests; fit scope is train-only; validation/test labels are forbidden.
- Support direction is fixed: larger score means less support. In-support and extrapolation thresholds are frozen before the benchmark split.
- Every test sample carries physical group identity, a label-free support score/evidence, a separate offline reference support state/evidence, observed target/evidence and nested 90%/95% predicted intervals. The reference state is used only to evaluate whether the support score predicts miscoverage; it is never used to calculate the score. Calibration is fitted after the development split is frozen and before any test access.
- A physical battery may move between reference support states over its aging trajectory; the same cell may therefore contribute different observations to in-support, boundary and extrapolation strata. The audit requires at least two distinct physical groups represented in each state to exercise the synthetic arithmetic, and computes coverage per state while clustering pass-rate evidence by physical group. A protocol-enforced floor of 30 independent groups per state is mandatory before coverage can be labelled sufficient; the manifest author cannot lower it. Thirty is only an initial benchmark floor, not a statistical guarantee. The required group count should be preregistered from application tolerance, within-group dependence and a power analysis.
- Validation-only calibration groups must be a subset of the validation partition and disjoint from test. External calibration must bind a source digest and remain disjoint from development and test groups.
- Abstention cutoffs are predeclared and strictly increasing.

## Computed outputs

For each support stratum the auditor computes:
- PICP at 90% and 95%;
- Per-physical-group coverage pass rates and Wilson 95% lower bounds over independent groups (not over correlated sample windows);
- mean interval width at both levels;
- MAE and RMSE;
- 90% and 95% interval scores (proper interval scoring rules);
- undercoverage flags using predeclared engineering tolerances.

The auditor evaluates the detector at both sample and physical-group level: sample extrapolation recall/false-positive rate/AUROC, plus group-level recall/false-positive rate/AUROC using a predeclared worst-support aggregation across that cell's test observations. It computes a risk-coverage curve at frozen support-score cutoffs: retained sample/group count, retained fraction, MAE, RMSE and worst-group MAE. Calibration strata are grouped by offline reference state, never by the detector's own thresholds. A model with good marginal accuracy but poor extrapolation coverage is preserved as a reproducible audit record with SUPPORT_STRATUM_UNDERCOVERAGE_DETECTED. If the detector also misses an extrapolating physical group, the status becomes OOD_MISDETECTION_AND_UNDERCOVERAGE. A tiny synthetic corpus with only two independent groups per state is always labelled INSUFFICIENT_PHYSICAL_GROUP_EVIDENCE, even when observed PICP is perfect. None of these states constitutes a successful calibration gate or calibrated deployment claim.

## Frozen synthetic corpus

Twelve fixtures: three accepted audit records and nine rejected controls. The accepted records include a temporal cell trajectory that crosses from in-support to extrapolation but remains underpowered, an OOD undercoverage case, and an OOD undercoverage case where the detector misses an extrapolating observation. The nine rejected controls cover test-label leakage into support scoring, post-freeze thresholds, calibration/test asset overlap, missing reference support strata, duplicate sample identity, train/test group overlap, invalid interval nesting, non-monotonic abstention cutoffs and an attempt to lower the protocol's 30-group minimum.

These fixtures validate arithmetic and evidence gates only; they are not physical battery evidence.

## Limits

Split conformal and other calibration methods do not automatically provide distribution-shift-proof guarantees. Their assumptions (e.g. exchangeability or a specified covariate-shift model) must be stated and tested. The audit does not train the support detector, estimate physical battery uncertainty, determine that an OOD score is causal, or authorize BMS action.
