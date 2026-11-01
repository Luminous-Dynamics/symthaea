use std::collections::BTreeSet;

use crate::{OnlineCostDimension, ResourceVector};

/// One example in a fixed initial labeled holdout. Independence from the
/// selection policy and predictor training is an experimental-design
/// requirement; these digests identify declared partitions but do not
/// authenticate that requirement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapCalibrationPoint {
    pub example_id: u64,
    pub holdout_partition_digest: [u8; 32],
    pub training_partition_digest: [u8; 32],
    pub model_digest: [u8; 32],
    pub score_model_digest: [u8; 32],
    pub feature_cohort_digest: [u8; 32],
    pub selection_score: u64,
    pub predicted_cost: u128,
    pub observed_remaining_to_verified_proof: ResourceVector,
    pub meter_receipt_digest: [u8; 32],
    pub completion_verifier_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapCurrentPrediction {
    pub example_id: u64,
    pub frontier_digest: [u8; 32],
    pub model_digest: [u8; 32],
    pub score_model_digest: [u8; 32],
    pub selection_policy_digest: [u8; 32],
    pub feature_cohort_digest: [u8; 32],
    pub selection_score: u64,
    pub predicted_cost: u128,
}

/// Decision-driven threshold: select when score <= base + increment times
/// the number of previously selected episodes. This rule depends on prior
/// binary decisions only, not prior costs or holdout labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecisionDrivenCapConfig {
    pub resource_dimension: OnlineCostDimension,
    pub alpha_numerator: u32,
    pub alpha_denominator: u32,
    pub base_threshold: u64,
    pub threshold_increment_per_selected: u64,
    pub holdout_partition_digest: [u8; 32],
    pub model_digest: [u8; 32],
    pub score_model_digest: [u8; 32],
    pub selection_policy_digest: [u8; 32],
    pub feature_cohort_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecisionDrivenCapState {
    selected_count: u64,
}

impl DecisionDrivenCapState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn selected_count(&self) -> u64 {
        self.selected_count
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapInterval {
    pub lower: u128,
    /// None means no finite upper endpoint is available.
    pub upper: Option<u128>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionDrivenCapResult {
    pub example_id: u64,
    pub selected: bool,
    pub selection_threshold: u64,
    pub prior_selected_count: u64,
    pub selection_policy_digest: [u8; 32],
    pub calibration_example_ids: Vec<u64>,
    pub residual_quantile: Option<u128>,
    pub interval: Option<CapInterval>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionDrivenCapError {
    InvalidAlpha,
    MissingHoldoutPartitionDigest,
    MissingTrainingPartitionDigest,
    OverlappingTrainingAndHoldoutPartitions,
    MissingModelDigest,
    MissingScoreModelDigest,
    MissingSelectionPolicyDigest,
    MissingFeatureCohortDigest,
    InvalidCurrentExampleId,
    MissingCurrentFrontierDigest,
    CurrentModelMismatch,
    CurrentScoreModelMismatch,
    CurrentSelectionPolicyMismatch,
    CurrentCohortMismatch,
    CurrentHoldoutOverlap,
    InvalidCalibrationExampleId,
    DuplicateCalibrationExampleId,
    CalibrationPartitionMismatch,
    CalibrationModelMismatch,
    CalibrationScoreModelMismatch,
    CalibrationCohortMismatch,
    MissingMeterReceipt,
    MissingCompletionVerifier,
    ThresholdOverflow,
    SelectedCountOverflow,
    QuantileArithmeticOverflow,
}

impl std::fmt::Display for DecisionDrivenCapError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for DecisionDrivenCapError {}

/// Fixed-initial-holdout CAP for one scalar resource dimension.
///
/// The finite-sample selection-conditional coverage result and real-time FCR
/// control apply only under the CAP assumptions: i.i.d. holdout/test examples,
/// a predictor and score function fixed independently of them, and this
/// decision-driven selection rule independent of the initial holdout. This
/// implementation excludes censoring, partial feedback, distribution shift,
/// arbitrary adaptive selection, and simultaneous vector coverage.
pub fn apply_decision_driven_cap(
    config: DecisionDrivenCapConfig,
    state: &mut DecisionDrivenCapState,
    holdout: &[CapCalibrationPoint],
    current: CapCurrentPrediction,
) -> Result<DecisionDrivenCapResult, DecisionDrivenCapError> {
    validate_config(config)?;
    validate_current(config, current)?;
    if holdout
        .iter()
        .any(|point| point.example_id == current.example_id)
    {
        return Err(DecisionDrivenCapError::CurrentHoldoutOverlap);
    }

    let prior_selected_count = state.selected_count;
    let selection_threshold = config
        .threshold_increment_per_selected
        .checked_mul(prior_selected_count)
        .and_then(|step| config.base_threshold.checked_add(step))
        .ok_or(DecisionDrivenCapError::ThresholdOverflow)?;
    let selected = current.selection_score <= selection_threshold;

    if !selected {
        return Ok(DecisionDrivenCapResult {
            example_id: current.example_id,
            selected: false,
            selection_threshold,
            prior_selected_count,
            selection_policy_digest: config.selection_policy_digest,
            calibration_example_ids: Vec::new(),
            residual_quantile: None,
            interval: None,
        });
    }

    validate_holdout(config, holdout)?;
    let mut residuals = Vec::<(u64, u128)>::new();
    let mut calibration_example_ids = Vec::new();

    // Apply the same current threshold to the fixed initial holdout. The
    // holdout is not updated with later selected examples.
    for point in holdout {
        if point.selection_score <= selection_threshold {
            let observed = dimension_value(
                config.resource_dimension,
                point.observed_remaining_to_verified_proof,
            );
            residuals.push((point.example_id, point.predicted_cost.abs_diff(observed)));
            calibration_example_ids.push(point.example_id);
        }
    }
    residuals.sort_by_key(|(_, residual)| *residual);
    calibration_example_ids.sort_unstable();

    let residual_quantile =
        conformal_residual_quantile(config.alpha_numerator, config.alpha_denominator, &residuals)?;
    let interval = match residual_quantile {
        Some(radius) => Some(CapInterval {
            lower: current.predicted_cost.saturating_sub(radius),
            upper: current.predicted_cost.checked_add(radius),
        }),
        None => Some(CapInterval {
            lower: 0,
            upper: None,
        }),
    };

    state.selected_count = state
        .selected_count
        .checked_add(1)
        .ok_or(DecisionDrivenCapError::SelectedCountOverflow)?;

    Ok(DecisionDrivenCapResult {
        example_id: current.example_id,
        selected: true,
        selection_threshold,
        prior_selected_count,
        selection_policy_digest: config.selection_policy_digest,
        calibration_example_ids,
        residual_quantile,
        interval,
    })
}

fn validate_config(config: DecisionDrivenCapConfig) -> Result<(), DecisionDrivenCapError> {
    if config.alpha_numerator == 0
        || config.alpha_denominator == 0
        || config.alpha_numerator >= config.alpha_denominator
    {
        return Err(DecisionDrivenCapError::InvalidAlpha);
    }
    if is_zero(config.holdout_partition_digest) {
        return Err(DecisionDrivenCapError::MissingHoldoutPartitionDigest);
    }
    if is_zero(config.training_partition_digest) {
        return Err(DecisionDrivenCapError::MissingTrainingPartitionDigest);
    }
    if config.training_partition_digest == config.holdout_partition_digest {
        return Err(DecisionDrivenCapError::OverlappingTrainingAndHoldoutPartitions);
    }
    if is_zero(config.model_digest) {
        return Err(DecisionDrivenCapError::MissingModelDigest);
    }
    if is_zero(config.score_model_digest) {
        return Err(DecisionDrivenCapError::MissingScoreModelDigest);
    }
    if is_zero(config.selection_policy_digest) {
        return Err(DecisionDrivenCapError::MissingSelectionPolicyDigest);
    }
    if is_zero(config.feature_cohort_digest) {
        return Err(DecisionDrivenCapError::MissingFeatureCohortDigest);
    }
    Ok(())
}

fn validate_current(
    config: DecisionDrivenCapConfig,
    current: CapCurrentPrediction,
) -> Result<(), DecisionDrivenCapError> {
    if current.example_id == 0 {
        return Err(DecisionDrivenCapError::InvalidCurrentExampleId);
    }
    if is_zero(current.frontier_digest) {
        return Err(DecisionDrivenCapError::MissingCurrentFrontierDigest);
    }
    if current.model_digest != config.model_digest {
        return Err(DecisionDrivenCapError::CurrentModelMismatch);
    }
    if current.score_model_digest != config.score_model_digest {
        return Err(DecisionDrivenCapError::CurrentScoreModelMismatch);
    }
    if current.selection_policy_digest != config.selection_policy_digest {
        return Err(DecisionDrivenCapError::CurrentSelectionPolicyMismatch);
    }
    if current.feature_cohort_digest != config.feature_cohort_digest {
        return Err(DecisionDrivenCapError::CurrentCohortMismatch);
    }
    Ok(())
}

fn validate_holdout(
    config: DecisionDrivenCapConfig,
    holdout: &[CapCalibrationPoint],
) -> Result<(), DecisionDrivenCapError> {
    let mut ids = BTreeSet::new();
    for point in holdout {
        if point.example_id == 0 {
            return Err(DecisionDrivenCapError::InvalidCalibrationExampleId);
        }
        if !ids.insert(point.example_id) {
            return Err(DecisionDrivenCapError::DuplicateCalibrationExampleId);
        }
        if point.holdout_partition_digest != config.holdout_partition_digest {
            return Err(DecisionDrivenCapError::CalibrationPartitionMismatch);
        }
        if point.model_digest != config.model_digest {
            return Err(DecisionDrivenCapError::CalibrationModelMismatch);
        }
        if point.score_model_digest != config.score_model_digest {
            return Err(DecisionDrivenCapError::CalibrationScoreModelMismatch);
        }
        if point.feature_cohort_digest != config.feature_cohort_digest {
            return Err(DecisionDrivenCapError::CalibrationCohortMismatch);
        }
        if is_zero(point.meter_receipt_digest) {
            return Err(DecisionDrivenCapError::MissingMeterReceipt);
        }
        if is_zero(point.completion_verifier_digest) {
            return Err(DecisionDrivenCapError::MissingCompletionVerifier);
        }
    }
    Ok(())
}

fn conformal_residual_quantile(
    alpha_numerator: u32,
    alpha_denominator: u32,
    sorted_residuals: &[(u64, u128)],
) -> Result<Option<u128>, DecisionDrivenCapError> {
    let count = u128::try_from(sorted_residuals.len())
        .map_err(|_| DecisionDrivenCapError::QuantileArithmeticOverflow)?;
    let numerator = u128::from(alpha_denominator - alpha_numerator)
        .checked_mul(
            count
                .checked_add(1)
                .ok_or(DecisionDrivenCapError::QuantileArithmeticOverflow)?,
        )
        .ok_or(DecisionDrivenCapError::QuantileArithmeticOverflow)?;
    let denominator = u128::from(alpha_denominator);
    let rank = numerator
        .checked_add(denominator - 1)
        .ok_or(DecisionDrivenCapError::QuantileArithmeticOverflow)?
        / denominator;
    if rank == 0 || rank > count {
        return Ok(None);
    }
    let index = usize::try_from(rank - 1)
        .map_err(|_| DecisionDrivenCapError::QuantileArithmeticOverflow)?;
    Ok(sorted_residuals.get(index).map(|(_, residual)| *residual))
}

fn dimension_value(dimension: OnlineCostDimension, vector: ResourceVector) -> u128 {
    match dimension {
        OnlineCostDimension::Tokens => vector.tokens,
        OnlineCostDimension::ToolCalls => vector.tool_calls,
        OnlineCostDimension::CpuMs => vector.cpu_ms,
        OnlineCostDimension::GpuMs => vector.gpu_ms,
        OnlineCostDimension::WallMs => vector.wall_ms,
        OnlineCostDimension::MicroUsd => vector.micro_usd,
    }
}

fn is_zero(digest: [u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> DecisionDrivenCapConfig {
        DecisionDrivenCapConfig {
            resource_dimension: OnlineCostDimension::Tokens,
            alpha_numerator: 1,
            alpha_denominator: 4,
            base_threshold: 5,
            threshold_increment_per_selected: 2,
            holdout_partition_digest: [1; 32],
            training_partition_digest: [9; 32],
            model_digest: [2; 32],
            score_model_digest: [7; 32],
            selection_policy_digest: [8; 32],
            feature_cohort_digest: [3; 32],
        }
    }

    fn point(id: u64, score: u64, predicted: u128, observed: u128) -> CapCalibrationPoint {
        CapCalibrationPoint {
            example_id: id,
            holdout_partition_digest: [1; 32],
            model_digest: [2; 32],
            score_model_digest: [7; 32],
            selection_policy_digest: [8; 32],
            feature_cohort_digest: [3; 32],
            selection_score: score,
            predicted_cost: predicted,
            observed_remaining_to_verified_proof: ResourceVector::new(observed, 0, 0, 0, 0, 0),
            meter_receipt_digest: [4; 32],
            completion_verifier_digest: [5; 32],
        }
    }

    fn current(id: u64, score: u64, predicted: u128) -> CapCurrentPrediction {
        CapCurrentPrediction {
            example_id: id,
            frontier_digest: [6; 32],
            model_digest: [2; 32],
            score_model_digest: [7; 32],
            feature_cohort_digest: [3; 32],
            selection_score: score,
            predicted_cost: predicted,
        }
    }

    #[test]
    fn fixed_holdout_cap_applies_the_current_threshold_to_holdout_and_test() {
        let holdout = [
            point(1, 4, 10, 11),
            point(2, 5, 10, 12),
            point(3, 5, 10, 13),
            point(4, 6, 10, 40),
        ];
        let mut state = DecisionDrivenCapState::new();
        let result =
            apply_decision_driven_cap(config(), &mut state, &holdout, current(10, 5, 20)).unwrap();

        assert!(result.selected);
        assert_eq!(result.selection_threshold, 5);
        assert_eq!(result.prior_selected_count, 0);
        assert_eq!(result.calibration_example_ids, vec![1, 2, 3]);
        assert_eq!(result.residual_quantile, Some(3));
        assert_eq!(
            result.interval,
            Some(CapInterval {
                lower: 17,
                upper: Some(23),
            })
        );
        assert_eq!(state.selected_count(), 1);
    }

    #[test]
    fn threshold_uses_only_the_prior_binary_selection_count() {
        let holdout = [point(1, 4, 10, 11), point(2, 6, 10, 12)];
        let mut state = DecisionDrivenCapState::new();
        let first =
            apply_decision_driven_cap(config(), &mut state, &holdout, current(10, 5, 20)).unwrap();
        assert!(first.selected);

        let second =
            apply_decision_driven_cap(config(), &mut state, &holdout, current(11, 6, 20)).unwrap();
        assert_eq!(second.prior_selected_count, 1);
        assert_eq!(second.selection_threshold, 7);
        assert!(second.selected);
        assert_eq!(second.calibration_example_ids, vec![1, 2]);
        assert_eq!(state.selected_count(), 2);
    }

    #[test]
    fn unselected_candidate_does_not_advance_state_or_issue_an_interval() {
        let mut state = DecisionDrivenCapState::new();
        let result =
            apply_decision_driven_cap(config(), &mut state, &[], current(10, 6, 20)).unwrap();
        assert!(!result.selected);
        assert_eq!(result.interval, None);
        assert_eq!(state.selected_count(), 0);
    }

    #[test]
    fn insufficient_selected_holdout_support_returns_unbounded_interval() {
        let holdout = [point(1, 4, 10, 11)];
        let mut state = DecisionDrivenCapState::new();
        let result =
            apply_decision_driven_cap(config(), &mut state, &holdout, current(10, 5, 20)).unwrap();
        assert!(result.selected);
        assert_eq!(result.residual_quantile, None);
        assert_eq!(
            result.interval,
            Some(CapInterval {
                lower: 0,
                upper: None,
            })
        );
    }

    #[test]
    fn holdout_permutation_does_not_change_the_interval() {
        let first_holdout = [
            point(1, 4, 10, 13),
            point(2, 5, 10, 11),
            point(3, 5, 10, 12),
        ];
        let mut second_holdout = first_holdout;
        second_holdout.reverse();
        let mut first_state = DecisionDrivenCapState::new();
        let mut second_state = DecisionDrivenCapState::new();
        let first = apply_decision_driven_cap(
            config(),
            &mut first_state,
            &first_holdout,
            current(10, 5, 20),
        )
        .unwrap();
        let second = apply_decision_driven_cap(
            config(),
            &mut second_state,
            &second_holdout,
            current(10, 5, 20),
        )
        .unwrap();
        assert_eq!(first.interval, second.interval);
        assert_eq!(first.residual_quantile, second.residual_quantile);
        assert_eq!(
            first.calibration_example_ids,
            second.calibration_example_ids
        );
    }

    #[test]
    fn duplicate_provenance_and_model_mismatches_fail_closed() {
        let mut state = DecisionDrivenCapState::new();
        let duplicate = [point(1, 4, 10, 11), point(1, 5, 10, 12)];
        assert_eq!(
            apply_decision_driven_cap(config(), &mut state, &duplicate, current(10, 5, 20)),
            Err(DecisionDrivenCapError::DuplicateCalibrationExampleId)
        );

        let mut wrong_partition = point(1, 4, 10, 11);
        wrong_partition.holdout_partition_digest = [99; 32];
        assert_eq!(
            apply_decision_driven_cap(config(), &mut state, &[wrong_partition], current(10, 5, 20)),
            Err(DecisionDrivenCapError::CalibrationPartitionMismatch)
        );

        let mut wrong_model = point(1, 4, 10, 11);
        wrong_model.model_digest = [99; 32];
        assert_eq!(
            apply_decision_driven_cap(config(), &mut state, &[wrong_model], current(10, 5, 20)),
            Err(DecisionDrivenCapError::CalibrationModelMismatch)
        );

        let mut missing_meter = point(1, 4, 10, 11);
        missing_meter.meter_receipt_digest = [0; 32];
        assert_eq!(
            apply_decision_driven_cap(config(), &mut state, &[missing_meter], current(10, 5, 20)),
            Err(DecisionDrivenCapError::MissingMeterReceipt)
        );

        let mut wrong_score = point(1, 4, 10, 11);
        wrong_score.score_model_digest = [99; 32];
        assert_eq!(
            apply_decision_driven_cap(config(), &mut state, &[wrong_score], current(10, 5, 20)),
            Err(DecisionDrivenCapError::CalibrationScoreModelMismatch)
        );
    }

    #[test]
    fn predictor_mismatch_and_holdout_overlap_are_rejected() {
        let mut state = DecisionDrivenCapState::new();
        let mut wrong_model = current(10, 5, 20);
        wrong_model.model_digest = [99; 32];
        assert_eq!(
            apply_decision_driven_cap(config(), &mut state, &[], wrong_model),
            Err(DecisionDrivenCapError::CurrentModelMismatch)
        );
        assert_eq!(
            apply_decision_driven_cap(
                config(),
                &mut state,
                &[point(10, 4, 10, 11)],
                current(10, 5, 20)
            ),
            Err(DecisionDrivenCapError::CurrentHoldoutOverlap)
        );
    }

    #[test]
    fn overlapping_model_training_and_holdout_partitions_are_rejected() {
        let mut cfg = config();
        cfg.training_partition_digest = cfg.holdout_partition_digest;
        let mut state = DecisionDrivenCapState::new();
        assert_eq!(
            apply_decision_driven_cap(cfg, &mut state, &[], current(10, 5, 20)),
            Err(DecisionDrivenCapError::OverlappingTrainingAndHoldoutPartitions)
        );
    }

    #[test]
    fn threshold_overflow_fails_without_mutating_state() {
        let mut cfg = config();
        cfg.base_threshold = u64::MAX;
        cfg.threshold_increment_per_selected = 1;
        let mut state = DecisionDrivenCapState { selected_count: 1 };
        assert_eq!(
            apply_decision_driven_cap(cfg, &mut state, &[], current(10, 1, 20)),
            Err(DecisionDrivenCapError::ThresholdOverflow)
        );
        assert_eq!(state.selected_count(), 1);
    }
}
