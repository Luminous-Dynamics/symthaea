use std::collections::BTreeSet;

use crate::{FrontierCostPrediction, ResourceVector, ShiftStatus};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompetingProofTerminal {
    PermissionRevoked,
    ExplicitAbort,
    PolicyAbandoned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LandmarkTerminal {
    ProofCompleted {
        remaining_resource_cost: ResourceVector,
        verifier_digest: [u8; 32],
    },
    RightCensored {
        observed_resource_lower_bound: ResourceVector,
    },
    CompetingTerminal(CompetingProofTerminal),
}

/// Out-of-fold censoring survival estimate evaluated at an observed event
/// time using the episode's time-varying history. The two partition digests
/// must differ: nuisance fitting and calibration outcomes are separated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CensoringWeightReceipt {
    pub censor_model_digest: [u8; 32],
    pub nuisance_training_partition_digest: [u8; 32],
    pub calibration_partition_digest: [u8; 32],
    pub observed_history_digest: [u8; 32],
    pub population_cohort_digest: [u8; 32],
    /// Estimated probability of remaining uncensored through the proof event.
    pub censoring_survival_probability: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LandmarkEpisode {
    pub episode_id: u64,
    /// Work consumed before this forecast landmark.
    pub landmark_work_units: u64,
    /// min(proof-completion work, censoring work, or competing-terminal work).
    pub observed_endpoint_work_units: u64,
    pub prediction: FrontierCostPrediction,
    /// Digest of the full time-varying history available at the endpoint.
    pub observed_history_digest: [u8; 32],
    pub terminal: LandmarkTerminal,
    /// Present only when proof completion is observed after the landmark.
    pub censoring_receipt: Option<CensoringWeightReceipt>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LandmarkIpcwError {
    EmptyCalibrationSet,
    InvalidMiscoverage,
    LandmarkMismatch,
    InvalidPositivityFloor,
    MissingModelDigest,
    MissingCensorModelDigest,
    MissingCalibrationPartitionDigest,
    MissingTrainingPartitionDigest,
    MissingPopulationCohortDigest,
    DuplicateEpisode,
    InvalidEpisodeId,
    PredictionModelMismatch,
    PredictionCohortMismatch,
    ShiftedPrediction,
    InvalidFrontierDigest,
    MissingObservedHistoryDigest,
    MissingCompletionVerifier,
    MissingCensoringReceipt,
    UnexpectedCensoringReceipt,
    CensoringReceiptModelMismatch,
    CensoringReceiptPartitionMismatch,
    CensoringReceiptHistoryMismatch,
    CensoringReceiptCohortMismatch,
    InvalidCensoringProbability,
    PositivityViolation,
    CompetingRiskRequiresSeparateEstimator,
    InvalidPredictionScale,
    NonFiniteWeight,
    NoObservedProofEventsAfterLandmark,
    NoFeasibleCalibrationThreshold,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LandmarkCoverageRegime {
    PaacUnderConditionalIndependentCensoringAndPositivity,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LandmarkIpcwCalibration {
    pub landmark_work_units: u64,
    pub target_coverage: f64,
    pub normalized_residual_threshold: f64,
    pub at_risk_count: usize,
    pub proof_events_observed_after_landmark: usize,
    pub right_censored_after_landmark: usize,
    pub exited_before_landmark: usize,
    pub total_ipcw_weight: f64,
    pub effective_sample_size: f64,
    pub maximum_ipcw_weight: f64,
    pub minimum_censoring_survival_probability: f64,
    pub model_digest: [u8; 32],
    pub censor_model_digest: [u8; 32],
    pub calibration_partition_digest: [u8; 32],
    pub nuisance_training_partition_digest: [u8; 32],
    pub population_cohort_digest: [u8; 32],
    pub coverage_regime: LandmarkCoverageRegime,
}

impl LandmarkIpcwCalibration {
    /// Applies the landmark-specific joint vector threshold to a new forecast.
    /// The returned upper bound remains a research interval; the hard ledger
    /// and completion-probability gate are separate controls.
    pub fn upper_resource_bound(
        &self,
        prediction: FrontierCostPrediction,
    ) -> Result<ResourceVector, LandmarkIpcwError> {
        validate_prediction(prediction, self.model_digest, self.population_cohort_digest)?;
        upper_vector(
            prediction.point_cost,
            prediction.uncertainty_scale,
            self.normalized_residual_threshold,
        )
    }
}

/// HAPS-style IPCW calibration for a nested family of one-sided joint
/// resource-vector envelopes.
///
/// For observed proof events after landmark tau, the estimating balance is:
///
/// sum_i I(X_i > tau) Delta_i / Ghat_i *
///     ( I(score_i <= theta) - (1 - alpha) ).
///
/// The smallest observed score threshold with nonnegative balance is selected.
/// Right-censored survivors contribute to at-risk diagnostics but not to the
/// event-observed sum; IPCW reweights the observed proof events. Competing
/// terminal events after tau are rejected rather than treated as ordinary
/// censoring. This is a research adaptation, not a theorem reproduction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LandmarkIpcwConfig {
    pub landmark_work_units: u64,
    pub miscoverage: f64,
    pub minimum_censoring_survival_probability: f64,
    pub model_digest: [u8; 32],
    pub censor_model_digest: [u8; 32],
    pub calibration_partition_digest: [u8; 32],
    pub nuisance_training_partition_digest: [u8; 32],
    pub population_cohort_digest: [u8; 32],
}

pub fn calibrate_landmark_ipcw(
    episodes: &[LandmarkEpisode],
    config: LandmarkIpcwConfig,
) -> Result<LandmarkIpcwCalibration, LandmarkIpcwError> {
    let LandmarkIpcwConfig {
        landmark_work_units,
        miscoverage,
        minimum_censoring_survival_probability,
        model_digest,
        censor_model_digest,
        calibration_partition_digest,
        nuisance_training_partition_digest,
        population_cohort_digest,
    } = config;
    if episodes.is_empty() {
        return Err(LandmarkIpcwError::EmptyCalibrationSet);
    }
    if !miscoverage.is_finite() || miscoverage <= 0.0 || miscoverage >= 1.0 {
        return Err(LandmarkIpcwError::InvalidMiscoverage);
    }
    if !minimum_censoring_survival_probability.is_finite()
        || minimum_censoring_survival_probability <= 0.0
        || minimum_censoring_survival_probability > 1.0
    {
        return Err(LandmarkIpcwError::InvalidPositivityFloor);
    }
    if is_zero(model_digest) {
        return Err(LandmarkIpcwError::MissingModelDigest);
    }
    if is_zero(censor_model_digest) {
        return Err(LandmarkIpcwError::MissingCensorModelDigest);
    }
    if is_zero(calibration_partition_digest) {
        return Err(LandmarkIpcwError::MissingCalibrationPartitionDigest);
    }
    if is_zero(nuisance_training_partition_digest) {
        return Err(LandmarkIpcwError::MissingTrainingPartitionDigest);
    }
    if nuisance_training_partition_digest == calibration_partition_digest {
        return Err(LandmarkIpcwError::CensoringReceiptPartitionMismatch);
    }
    if is_zero(population_cohort_digest) {
        return Err(LandmarkIpcwError::MissingPopulationCohortDigest);
    }

    let mut episode_ids = BTreeSet::new();
    let mut scores_and_weights = Vec::<(f64, f64)>::new();
    let mut at_risk_count = 0usize;
    let mut right_censored_after_landmark = 0usize;
    let mut exited_before_landmark = 0usize;
    let mut minimum_observed_g = 1.0_f64;
    let mut maximum_weight = 0.0_f64;
    let mut sum_weights = 0.0_f64;
    let mut sum_squared_weights = 0.0_f64;

    for episode in episodes {
        if episode.episode_id == 0 {
            return Err(LandmarkIpcwError::InvalidEpisodeId);
        }
        if episode.landmark_work_units != landmark_work_units {
            return Err(LandmarkIpcwError::LandmarkMismatch);
        }
        if !episode_ids.insert(episode.episode_id) {
            return Err(LandmarkIpcwError::DuplicateEpisode);
        }
        if matches!(episode.terminal, LandmarkTerminal::CompetingTerminal(_)) {
            return Err(LandmarkIpcwError::CompetingRiskRequiresSeparateEstimator);
        }
        if episode.observed_endpoint_work_units <= episode.landmark_work_units {
            exited_before_landmark += 1;
            if let LandmarkTerminal::ProofCompleted {
                verifier_digest, ..
            } = episode.terminal
            {
                if is_zero(verifier_digest) {
                    return Err(LandmarkIpcwError::MissingCompletionVerifier);
                }
            }
            continue;
        }

        at_risk_count += 1;
        validate_prediction(episode.prediction, model_digest, population_cohort_digest)?;

        match episode.terminal {
            LandmarkTerminal::RightCensored { .. } => {
                if episode.censoring_receipt.is_some() {
                    return Err(LandmarkIpcwError::UnexpectedCensoringReceipt);
                }
                right_censored_after_landmark += 1;
            }
            LandmarkTerminal::CompetingTerminal(_) => {
                unreachable!("competing terminals rejected above")
            }
            LandmarkTerminal::ProofCompleted {
                remaining_resource_cost,
                verifier_digest,
            } => {
                if is_zero(verifier_digest) {
                    return Err(LandmarkIpcwError::MissingCompletionVerifier);
                }
                if is_zero(episode.observed_history_digest) {
                    return Err(LandmarkIpcwError::MissingObservedHistoryDigest);
                }
                let receipt = episode
                    .censoring_receipt
                    .ok_or(LandmarkIpcwError::MissingCensoringReceipt)?;
                if receipt.censor_model_digest != censor_model_digest {
                    return Err(LandmarkIpcwError::CensoringReceiptModelMismatch);
                }
                if receipt.nuisance_training_partition_digest != nuisance_training_partition_digest
                    || receipt.calibration_partition_digest != calibration_partition_digest
                {
                    return Err(LandmarkIpcwError::CensoringReceiptPartitionMismatch);
                }
                if receipt.observed_history_digest != episode.observed_history_digest {
                    return Err(LandmarkIpcwError::CensoringReceiptHistoryMismatch);
                }
                if receipt.population_cohort_digest != population_cohort_digest {
                    return Err(LandmarkIpcwError::CensoringReceiptCohortMismatch);
                }
                let g = receipt.censoring_survival_probability;
                if !g.is_finite() || g <= 0.0 || g > 1.0 {
                    return Err(LandmarkIpcwError::InvalidCensoringProbability);
                }
                if g < minimum_censoring_survival_probability {
                    return Err(LandmarkIpcwError::PositivityViolation);
                }
                let weight = 1.0 / g;
                if !weight.is_finite() {
                    return Err(LandmarkIpcwError::NonFiniteWeight);
                }
                let score = max_normalized_residual(
                    remaining_resource_cost,
                    episode.prediction.point_cost,
                    episode.prediction.uncertainty_scale,
                )?;
                scores_and_weights.push((score, weight));
                minimum_observed_g = minimum_observed_g.min(g);
                maximum_weight = maximum_weight.max(weight);
                sum_weights += weight;
                sum_squared_weights += weight * weight;
                if !sum_weights.is_finite() || !sum_squared_weights.is_finite() {
                    return Err(LandmarkIpcwError::NonFiniteWeight);
                }
            }
        }
    }

    if scores_and_weights.is_empty() {
        return Err(LandmarkIpcwError::NoObservedProofEventsAfterLandmark);
    }

    scores_and_weights.sort_by(|left, right| left.0.total_cmp(&right.0));
    let target_coverage = 1.0 - miscoverage;
    let required_covered_weight = target_coverage * sum_weights;
    let mut cumulative_covered_weight = 0.0_f64;
    let mut candidate_threshold = None;
    let mut index = 0;
    while index < scores_and_weights.len() {
        let threshold = scores_and_weights[index].0;
        while index < scores_and_weights.len()
            && scores_and_weights[index].0.total_cmp(&threshold).is_eq()
        {
            cumulative_covered_weight += scores_and_weights[index].1;
            index += 1;
        }
        if !cumulative_covered_weight.is_finite() {
            return Err(LandmarkIpcwError::NonFiniteWeight);
        }
        if cumulative_covered_weight >= required_covered_weight {
            candidate_threshold = Some(threshold);
            break;
        }
    }
    let normalized_residual_threshold =
        candidate_threshold.ok_or(LandmarkIpcwError::NoFeasibleCalibrationThreshold)?;
    let effective_sample_size = (sum_weights * sum_weights) / sum_squared_weights;
    if !effective_sample_size.is_finite() {
        return Err(LandmarkIpcwError::NonFiniteWeight);
    }

    Ok(LandmarkIpcwCalibration {
        landmark_work_units,
        target_coverage,
        normalized_residual_threshold,
        at_risk_count,
        proof_events_observed_after_landmark: scores_and_weights.len(),
        right_censored_after_landmark,
        exited_before_landmark,
        total_ipcw_weight: sum_weights,
        effective_sample_size,
        maximum_ipcw_weight: maximum_weight,
        minimum_censoring_survival_probability: minimum_observed_g,
        model_digest,
        censor_model_digest,
        calibration_partition_digest,
        nuisance_training_partition_digest,
        population_cohort_digest,
        coverage_regime:
            LandmarkCoverageRegime::PaacUnderConditionalIndependentCensoringAndPositivity,
    })
}

fn validate_prediction(
    prediction: FrontierCostPrediction,
    model_digest: [u8; 32],
    population_cohort_digest: [u8; 32],
) -> Result<(), LandmarkIpcwError> {
    if prediction.model_digest != model_digest {
        return Err(LandmarkIpcwError::PredictionModelMismatch);
    }
    if prediction.population_cohort_digest != population_cohort_digest {
        return Err(LandmarkIpcwError::PredictionCohortMismatch);
    }
    if prediction.shift_status != ShiftStatus::InDomain {
        return Err(LandmarkIpcwError::ShiftedPrediction);
    }
    if is_zero(prediction.frontier_digest) {
        return Err(LandmarkIpcwError::InvalidFrontierDigest);
    }
    if !resource_vector_positive(prediction.uncertainty_scale) {
        return Err(LandmarkIpcwError::InvalidPredictionScale);
    }
    Ok(())
}

fn max_normalized_residual(
    observed: ResourceVector,
    point: ResourceVector,
    scale: ResourceVector,
) -> Result<f64, LandmarkIpcwError> {
    let actual = components(observed);
    let estimate = components(point);
    let uncertainty = components(scale);
    let mut maximum = 0.0_f64;
    for index in 0..6 {
        if uncertainty[index] == 0 {
            return Err(LandmarkIpcwError::InvalidPredictionScale);
        }
        let score = (actual[index] as f64 - estimate[index] as f64) / uncertainty[index] as f64;
        if !score.is_finite() {
            return Err(LandmarkIpcwError::NonFiniteWeight);
        }
        maximum = maximum.max(score.max(0.0));
    }
    Ok(maximum)
}

fn upper_vector(
    point: ResourceVector,
    scale: ResourceVector,
    threshold: f64,
) -> Result<ResourceVector, LandmarkIpcwError> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(LandmarkIpcwError::NonFiniteWeight);
    }
    let point = components(point);
    let scale = components(scale);
    let mut values = [0_u128; 6];
    for index in 0..6 {
        let value = point[index] as f64 + threshold * scale[index] as f64;
        if !value.is_finite() || value < 0.0 || value >= u128::MAX as f64 {
            return Err(LandmarkIpcwError::NonFiniteWeight);
        }
        values[index] = value.ceil() as u128;
    }
    Ok(ResourceVector::new(
        values[0], values[1], values[2], values[3], values[4], values[5],
    ))
}

fn components(vector: ResourceVector) -> [u128; 6] {
    [
        vector.tokens,
        vector.tool_calls,
        vector.cpu_ms,
        vector.gpu_ms,
        vector.wall_ms,
        vector.micro_usd,
    ]
}

fn resource_vector_positive(vector: ResourceVector) -> bool {
    components(vector).into_iter().all(|value| value > 0)
}

fn is_zero(digest: [u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrontierCostPrediction, ShiftStatus};

    const MODEL: [u8; 32] = [1; 32];
    const CENSOR_MODEL: [u8; 32] = [2; 32];
    const CALIBRATION: [u8; 32] = [3; 32];
    const TRAINING: [u8; 32] = [4; 32];
    const COHORT: [u8; 32] = [5; 32];

    fn vector(value: u128) -> ResourceVector {
        ResourceVector::new(value, value, value, value, value, value)
    }

    fn prediction() -> FrontierCostPrediction {
        FrontierCostPrediction {
            features: crate::ProofFrontierFeatures {
                unresolved_obligations: 2,
                mandatory_obligations: 1,
                maximum_dependency_depth: 2,
                dependency_edges: 1,
                admissible_alternatives: 2,
                missing_evidence_requirements: 1,
                distinct_operation_families: 1,
                distinct_evidence_families: 1,
                mandatory_verifier_operations: 1,
            },
            frontier_digest: [6; 32],
            model_digest: MODEL,
            population_cohort_digest: COHORT,
            point_cost: vector(10),
            uncertainty_scale: vector(2),
            shift_status: ShiftStatus::InDomain,
        }
    }

    fn completed(id: u64, landmark: u64, endpoint: u64, score: u128, g: f64) -> LandmarkEpisode {
        let predicted = prediction();
        let residual = vector(score * 2);
        let receipt = CensoringWeightReceipt {
            censor_model_digest: CENSOR_MODEL,
            nuisance_training_partition_digest: TRAINING,
            calibration_partition_digest: CALIBRATION,
            observed_history_digest: [7; 32],
            population_cohort_digest: COHORT,
            censoring_survival_probability: g,
        };
        LandmarkEpisode {
            episode_id: id,
            landmark_work_units: landmark,
            observed_endpoint_work_units: endpoint,
            prediction: predicted,
            observed_history_digest: [7; 32],
            terminal: LandmarkTerminal::ProofCompleted {
                remaining_resource_cost: predicted.point_cost.checked_add(residual).unwrap(),
                verifier_digest: [8; 32],
            },
            censoring_receipt: Some(receipt),
        }
    }

    fn censored(id: u64, landmark: u64, endpoint: u64) -> LandmarkEpisode {
        LandmarkEpisode {
            episode_id: id,
            landmark_work_units: landmark,
            observed_endpoint_work_units: endpoint,
            prediction: prediction(),
            observed_history_digest: [9; 32],
            terminal: LandmarkTerminal::RightCensored {
                observed_resource_lower_bound: vector(4),
            },
            censoring_receipt: None,
        }
    }

    fn calibrate(rows: &[LandmarkEpisode]) -> Result<LandmarkIpcwCalibration, LandmarkIpcwError> {
        calibrate_landmark_ipcw(
            rows,
            LandmarkIpcwConfig {
                landmark_work_units: 5,
                miscoverage: 0.10,
                minimum_censoring_survival_probability: 0.05,
                model_digest: MODEL,
                censor_model_digest: CENSOR_MODEL,
                calibration_partition_digest: CALIBRATION,
                nuisance_training_partition_digest: TRAINING,
                population_cohort_digest: COHORT,
            },
        )
    }

    #[test]
    fn weighted_balance_selects_smallest_feasible_joint_threshold() {
        let rows = [
            completed(1, 5, 6, 1, 1.0),
            completed(2, 5, 7, 2, 0.5),
            completed(3, 5, 8, 3, 0.25),
            censored(4, 5, 9),
        ];
        let calibration = calibrate(&rows).unwrap();
        assert_eq!(calibration.normalized_residual_threshold, 3.0);
        assert_eq!(calibration.at_risk_count, 4);
        assert_eq!(calibration.proof_events_observed_after_landmark, 3);
        assert_eq!(calibration.right_censored_after_landmark, 1);
        assert_eq!(calibration.effective_sample_size, 49.0 / 21.0);
        assert_eq!(calibration.maximum_ipcw_weight, 4.0);
        assert_eq!(
            calibration.coverage_regime,
            LandmarkCoverageRegime::PaacUnderConditionalIndependentCensoringAndPositivity
        );
    }

    #[test]
    fn upper_bound_is_conditioned_on_the_new_landmark_prediction() {
        let rows = [
            completed(1, 5, 6, 1, 1.0),
            completed(2, 5, 7, 2, 1.0),
            completed(3, 5, 8, 3, 1.0),
        ];
        let calibration = calibrate(&rows).unwrap();
        let upper = calibration.upper_resource_bound(prediction()).unwrap();
        assert_eq!(upper, vector(16));
    }

    #[test]
    fn events_and_censoring_before_landmark_are_not_at_risk() {
        let rows = [
            completed(1, 5, 4, 1, 1.0),
            censored(2, 5, 5),
            completed(3, 5, 8, 3, 1.0),
        ];
        let calibration = calibrate(&rows).unwrap();
        assert_eq!(calibration.exited_before_landmark, 2);
        assert_eq!(calibration.at_risk_count, 1);
        assert_eq!(calibration.proof_events_observed_after_landmark, 1);
        assert_eq!(calibration.normalized_residual_threshold, 3.0);
    }

    #[test]
    fn insufficient_censoring_survival_violates_positivity_without_weight_clipping() {
        let rows = [completed(1, 5, 6, 1, 0.01), completed(2, 5, 7, 2, 0.5)];
        assert_eq!(
            calibrate(&rows),
            Err(LandmarkIpcwError::PositivityViolation)
        );
    }

    #[test]
    fn competing_terminal_events_are_not_mislabelled_as_right_censoring() {
        let mut rows = [completed(1, 5, 6, 1, 1.0), completed(2, 5, 7, 2, 1.0)];
        rows[1].terminal =
            LandmarkTerminal::CompetingTerminal(CompetingProofTerminal::PermissionRevoked);
        rows[1].censoring_receipt = None;
        assert_eq!(
            calibrate(&rows),
            Err(LandmarkIpcwError::CompetingRiskRequiresSeparateEstimator)
        );
    }

    #[test]
    fn history_and_partition_identity_must_match_censoring_receipt() {
        let mut rows = [completed(1, 5, 6, 1, 1.0), completed(2, 5, 7, 2, 1.0)];
        let mut receipt = rows[0].censoring_receipt.unwrap();
        receipt.observed_history_digest = [99; 32];
        rows[0].censoring_receipt = Some(receipt);
        assert_eq!(
            calibrate(&rows),
            Err(LandmarkIpcwError::CensoringReceiptHistoryMismatch)
        );
    }

    #[test]
    fn nuisance_and_calibration_partitions_must_be_separate() {
        let rows = [completed(1, 5, 6, 1, 1.0), completed(2, 5, 7, 2, 1.0)];
        assert_eq!(
            calibrate_landmark_ipcw(
                &rows,
                LandmarkIpcwConfig {
                    landmark_work_units: 5,
                    miscoverage: 0.10,
                    minimum_censoring_survival_probability: 0.05,
                    model_digest: MODEL,
                    censor_model_digest: CENSOR_MODEL,
                    calibration_partition_digest: CALIBRATION,
                    nuisance_training_partition_digest: CALIBRATION,
                    population_cohort_digest: COHORT,
                },
            ),
            Err(LandmarkIpcwError::CensoringReceiptPartitionMismatch)
        );
    }

    #[test]
    fn episode_rows_must_share_the_requested_landmark() {
        let rows = [completed(1, 5, 6, 1, 1.0), completed(2, 6, 7, 2, 1.0)];
        assert_eq!(calibrate(&rows), Err(LandmarkIpcwError::LandmarkMismatch));
    }

    #[test]
    fn competing_terminals_are_rejected_even_before_the_landmark() {
        let mut row = completed(1, 5, 6, 1, 1.0);
        row.observed_endpoint_work_units = 4;
        row.terminal = LandmarkTerminal::CompetingTerminal(CompetingProofTerminal::ExplicitAbort);
        row.censoring_receipt = None;
        assert_eq!(
            calibrate(&[row]),
            Err(LandmarkIpcwError::CompetingRiskRequiresSeparateEstimator)
        );
    }

    #[test]
    fn shifted_frontier_or_different_population_abstains() {
        let rows = [completed(1, 5, 6, 1, 1.0), completed(2, 5, 7, 2, 1.0)];
        let mut shifted = rows[0];
        shifted.prediction.shift_status = ShiftStatus::SuspectedShift;
        assert_eq!(
            calibrate(&[shifted, rows[1]]),
            Err(LandmarkIpcwError::ShiftedPrediction)
        );

        let mut other_population = rows[0];
        let mut pred = other_population.prediction;
        pred.population_cohort_digest = [88; 32];
        other_population.prediction = pred;
        assert_eq!(
            calibrate(&[other_population, rows[1]]),
            Err(LandmarkIpcwError::PredictionCohortMismatch)
        );
    }

    #[test]
    fn invalid_digest_and_weight_receipts_fail_closed() {
        let rows = [completed(1, 5, 6, 1, 1.0), completed(2, 5, 7, 2, 1.0)];
        assert_eq!(
            calibrate_landmark_ipcw(
                &rows,
                LandmarkIpcwConfig {
                    landmark_work_units: 5,
                    miscoverage: 0.10,
                    minimum_censoring_survival_probability: 0.05,
                    model_digest: MODEL,
                    censor_model_digest: CENSOR_MODEL,
                    calibration_partition_digest: [0; 32],
                    nuisance_training_partition_digest: TRAINING,
                    population_cohort_digest: COHORT,
                },
            ),
            Err(LandmarkIpcwError::MissingCalibrationPartitionDigest)
        );

        let mut missing = rows[0];
        missing.censoring_receipt = None;
        assert_eq!(
            calibrate(&[missing, rows[1]]),
            Err(LandmarkIpcwError::MissingCensoringReceipt)
        );
    }

    #[test]
    fn duplicate_ids_and_invalid_probabilities_are_rejected() {
        let duplicate = [completed(1, 5, 6, 1, 1.0), completed(1, 5, 7, 2, 1.0)];
        assert_eq!(
            calibrate(&duplicate),
            Err(LandmarkIpcwError::DuplicateEpisode)
        );

        let rows = [completed(1, 5, 6, 1, 1.0)];
        assert_eq!(
            calibrate_landmark_ipcw(
                &rows,
                LandmarkIpcwConfig {
                    landmark_work_units: 5,
                    miscoverage: 1.0,
                    minimum_censoring_survival_probability: 0.05,
                    model_digest: MODEL,
                    censor_model_digest: CENSOR_MODEL,
                    calibration_partition_digest: CALIBRATION,
                    nuisance_training_partition_digest: TRAINING,
                    population_cohort_digest: COHORT,
                },
            ),
            Err(LandmarkIpcwError::InvalidMiscoverage)
        );
    }
}
