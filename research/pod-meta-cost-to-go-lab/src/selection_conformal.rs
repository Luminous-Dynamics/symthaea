use std::collections::BTreeSet;

use crate::{FrontierCostPrediction, ResourceVector, ShiftStatus};

/// Immutable pre-execution receipt for the operation actually selected by a
/// scheduler from a candidate pool. All identities are caller-supplied digests
/// in this research simulator; they are not cryptographic attestations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectionReceipt {
    pub episode_id: u64,
    pub decision_sequence: u64,
    pub candidate_pool_digest: [u8; 32],
    pub selection_policy_digest: [u8; 32],
    pub selected_population_digest: [u8; 32],
    pub pre_state_digest: [u8; 32],
    pub frontier_digest: [u8; 32],
    pub selected_operation_id: String,
    pub selected_operation_family: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedProofCalibrationExample {
    pub selection: SelectionReceipt,
    pub prediction: FrontierCostPrediction,
    /// Remaining resource cost from the selected pre-action state through
    /// independently verified sufficient proof.
    pub observed_remaining_to_verified_proof: ResourceVector,
    pub meter_receipt_digest: [u8; 32],
    pub completion_verifier_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionCalibrationError {
    EmptyCalibrationSet,
    InvalidMiscoverage,
    MissingModelDigest,
    MissingSelectionPolicyDigest,
    MissingSelectedPopulationDigest,
    MissingCalibrationDigest,
    DuplicateEpisode,
    InvalidEpisodeId,
    InvalidDecisionSequence,
    MissingCandidatePoolDigest,
    MissingPreStateDigest,
    MissingFrontierDigest,
    MissingSelectedOperationId,
    PredictionModelMismatch,
    PredictionFrontierMismatch,
    PredictionCohortMismatch,
    SelectionPolicyMismatch,
    SelectedPopulationMismatch,
    MissingCompletionVerifier,
    MissingMeterReceipt,
    ShiftedCalibrationPrediction,
    InvalidPredictionScale,
    ResourceOverflow,
    InvalidPredictionIdentity,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelectionThreshold {
    Finite(f64),
    Unbounded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionUnboundedReason {
    CalibrationSampleTooSmall,
    ShiftedFrontier,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectedProofUpperBound {
    pub upper: ResourceVector,
    pub target_coverage: f64,
    pub calibration_count: usize,
    pub normalized_residual_threshold: f64,
    pub model_digest: [u8; 32],
    pub selection_policy_digest: [u8; 32],
    pub selected_population_digest: [u8; 32],
    pub calibration_digest: [u8; 32],
    pub frontier_digest: [u8; 32],
    pub selected_operation_family: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SelectedProofEnvelope {
    Bounded(Box<SelectedProofUpperBound>),
    Unbounded {
        target_coverage: f64,
        calibration_count: usize,
        selection_policy_digest: [u8; 32],
        reason: SelectionUnboundedReason,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectionConditionedConformalCalibrator {
    target_coverage: f64,
    threshold: SelectionThreshold,
    model_digest: [u8; 32],
    selection_policy_digest: [u8; 32],
    selected_population_digest: [u8; 32],
    calibration_digest: [u8; 32],
    selected_operation_family: u32,
    calibration_count: usize,
}

impl SelectionReceipt {
    fn validate(&self) -> Result<(), SelectionCalibrationError> {
        if self.episode_id == 0 {
            return Err(SelectionCalibrationError::InvalidEpisodeId);
        }
        if self.decision_sequence == 0 {
            return Err(SelectionCalibrationError::InvalidDecisionSequence);
        }
        if is_zero(self.candidate_pool_digest) {
            return Err(SelectionCalibrationError::MissingCandidatePoolDigest);
        }
        if is_zero(self.pre_state_digest) {
            return Err(SelectionCalibrationError::MissingPreStateDigest);
        }
        if is_zero(self.frontier_digest) {
            return Err(SelectionCalibrationError::MissingFrontierDigest);
        }
        if self.selected_operation_id.trim().is_empty() {
            return Err(SelectionCalibrationError::MissingSelectedOperationId);
        }
        if is_zero(self.selection_policy_digest) {
            return Err(SelectionCalibrationError::MissingSelectionPolicyDigest);
        }
        if is_zero(self.selected_population_digest) {
            return Err(SelectionCalibrationError::MissingSelectedPopulationDigest);
        }
        Ok(())
    }
}

/// Split-conformal calibration on the distribution of actions selected by one
/// frozen policy. This is not CAP and does not itself control FCR across
/// repeated online selections. It targets marginal simultaneous coverage for
/// held-out exchangeable selected episodes from the same declared policy
/// version and selected population.
impl SelectionConditionedConformalCalibrator {
    pub fn fit(
        examples: &[SelectedProofCalibrationExample],
        miscoverage: f64,
        model_digest: [u8; 32],
        selection_policy_digest: [u8; 32],
        selected_population_digest: [u8; 32],
        calibration_digest: [u8; 32],
    ) -> Result<Self, SelectionCalibrationError> {
        if examples.is_empty() {
            return Err(SelectionCalibrationError::EmptyCalibrationSet);
        }
        if !miscoverage.is_finite() || miscoverage <= 0.0 || miscoverage >= 1.0 {
            return Err(SelectionCalibrationError::InvalidMiscoverage);
        }
        if is_zero(model_digest) {
            return Err(SelectionCalibrationError::MissingModelDigest);
        }
        if is_zero(selection_policy_digest) {
            return Err(SelectionCalibrationError::MissingSelectionPolicyDigest);
        }
        if is_zero(selected_population_digest) {
            return Err(SelectionCalibrationError::MissingSelectedPopulationDigest);
        }
        if is_zero(calibration_digest) {
            return Err(SelectionCalibrationError::MissingCalibrationDigest);
        }

        let mut episode_ids = BTreeSet::new();
        let mut selected_family = None;
        let mut scores = Vec::with_capacity(examples.len());
        for example in examples {
            example.selection.validate()?;
            if !episode_ids.insert(example.selection.episode_id) {
                return Err(SelectionCalibrationError::DuplicateEpisode);
            }
            if example.selection.selection_policy_digest != selection_policy_digest {
                return Err(SelectionCalibrationError::SelectionPolicyMismatch);
            }
            if example.selection.selected_population_digest != selected_population_digest
                || example.prediction.population_cohort_digest != selected_population_digest
            {
                return Err(SelectionCalibrationError::SelectedPopulationMismatch);
            }
            if example.prediction.model_digest != model_digest {
                return Err(SelectionCalibrationError::PredictionModelMismatch);
            }
            if example.prediction.frontier_digest != example.selection.frontier_digest {
                return Err(SelectionCalibrationError::PredictionFrontierMismatch);
            }
            if example.prediction.shift_status != ShiftStatus::InDomain {
                return Err(SelectionCalibrationError::ShiftedCalibrationPrediction);
            }
            if !resource_vector_positive(example.prediction.uncertainty_scale) {
                return Err(SelectionCalibrationError::InvalidPredictionScale);
            }
            if is_zero(example.completion_verifier_digest) {
                return Err(SelectionCalibrationError::MissingCompletionVerifier);
            }
            if is_zero(example.meter_receipt_digest) {
                return Err(SelectionCalibrationError::MissingMeterReceipt);
            }
            if example.prediction.features.validate().is_err() {
                return Err(SelectionCalibrationError::InvalidPredictionIdentity);
            }
            match selected_family {
                Some(family) if family != example.selection.selected_operation_family => {
                    return Err(SelectionCalibrationError::SelectedPopulationMismatch);
                }
                None => selected_family = Some(example.selection.selected_operation_family),
                _ => {}
            }
            scores.push(max_normalized_residual(
                example.observed_remaining_to_verified_proof,
                example.prediction.point_cost,
                example.prediction.uncertainty_scale,
            )?);
        }

        scores.sort_by(f64::total_cmp);
        let rank = (((examples.len() + 1) as f64) * (1.0 - miscoverage)).ceil() as usize;
        let threshold = if rank == 0 || rank > scores.len() {
            SelectionThreshold::Unbounded
        } else {
            SelectionThreshold::Finite(scores[rank - 1])
        };

        Ok(Self {
            target_coverage: 1.0 - miscoverage,
            threshold,
            model_digest,
            selection_policy_digest,
            selected_population_digest,
            calibration_digest,
            selected_operation_family: selected_family
                .ok_or(SelectionCalibrationError::EmptyCalibrationSet)?,
            calibration_count: examples.len(),
        })
    }

    pub fn threshold(&self) -> SelectionThreshold {
        self.threshold
    }

    pub fn calibration_count(&self) -> usize {
        self.calibration_count
    }

    pub fn predict_upper(
        &self,
        selection: &SelectionReceipt,
        prediction: FrontierCostPrediction,
    ) -> Result<SelectedProofEnvelope, SelectionCalibrationError> {
        selection.validate()?;
        if selection.selection_policy_digest != self.selection_policy_digest {
            return Err(SelectionCalibrationError::SelectionPolicyMismatch);
        }
        if selection.selected_population_digest != self.selected_population_digest
            || prediction.population_cohort_digest != self.selected_population_digest
        {
            return Err(SelectionCalibrationError::SelectedPopulationMismatch);
        }
        if selection.selected_operation_family != self.selected_operation_family {
            return Err(SelectionCalibrationError::SelectedPopulationMismatch);
        }
        if prediction.model_digest != self.model_digest {
            return Err(SelectionCalibrationError::PredictionModelMismatch);
        }
        if prediction.frontier_digest != selection.frontier_digest {
            return Err(SelectionCalibrationError::PredictionFrontierMismatch);
        }
        if !resource_vector_positive(prediction.uncertainty_scale) {
            return Err(SelectionCalibrationError::InvalidPredictionScale);
        }
        if prediction.features.validate().is_err() {
            return Err(SelectionCalibrationError::InvalidPredictionIdentity);
        }
        if prediction.shift_status != ShiftStatus::InDomain {
            return Ok(SelectedProofEnvelope::Unbounded {
                target_coverage: self.target_coverage,
                calibration_count: self.calibration_count,
                selection_policy_digest: self.selection_policy_digest,
                reason: SelectionUnboundedReason::ShiftedFrontier,
            });
        }
        match self.threshold {
            SelectionThreshold::Unbounded => Ok(SelectedProofEnvelope::Unbounded {
                target_coverage: self.target_coverage,
                calibration_count: self.calibration_count,
                selection_policy_digest: self.selection_policy_digest,
                reason: SelectionUnboundedReason::CalibrationSampleTooSmall,
            }),
            SelectionThreshold::Finite(threshold) => {
                let upper = upper_resource_vector(
                    prediction.point_cost,
                    prediction.uncertainty_scale,
                    threshold,
                )?;
                Ok(SelectedProofEnvelope::Bounded(Box::new(SelectedProofUpperBound {
                    upper,
                    target_coverage: self.target_coverage,
                    calibration_count: self.calibration_count,
                    normalized_residual_threshold: threshold,
                    model_digest: self.model_digest,
                    selection_policy_digest: self.selection_policy_digest,
                    selected_population_digest: self.selected_population_digest,
                    calibration_digest: self.calibration_digest,
                    frontier_digest: selection.frontier_digest,
                    selected_operation_family: self.selected_operation_family,
                })))
            }
        }
    }
}

fn max_normalized_residual(
    observed: ResourceVector,
    point: ResourceVector,
    scale: ResourceVector,
) -> Result<f64, SelectionCalibrationError> {
    let actual = components(observed);
    let estimate = components(point);
    let uncertainty = components(scale);
    let mut maximum = 0.0_f64;
    for index in 0..6 {
        if uncertainty[index] == 0 {
            return Err(SelectionCalibrationError::InvalidPredictionScale);
        }
        let score = (actual[index] as f64 - estimate[index] as f64) / uncertainty[index] as f64;
        if !score.is_finite() {
            return Err(SelectionCalibrationError::ResourceOverflow);
        }
        maximum = maximum.max(score.max(0.0));
    }
    Ok(maximum)
}

fn upper_resource_vector(
    point: ResourceVector,
    scale: ResourceVector,
    threshold: f64,
) -> Result<ResourceVector, SelectionCalibrationError> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(SelectionCalibrationError::ResourceOverflow);
    }
    let point = components(point);
    let scale = components(scale);
    let mut upper = [0_u128; 6];
    for index in 0..6 {
        let value = point[index] as f64 + threshold * scale[index] as f64;
        if !value.is_finite() || value < 0.0 || value >= u128::MAX as f64 {
            return Err(SelectionCalibrationError::ResourceOverflow);
        }
        upper[index] = value.ceil() as u128;
    }
    Ok(ResourceVector::new(
        upper[0], upper[1], upper[2], upper[3], upper[4], upper[5],
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
    use crate::{FrontierCostPrediction, ProofFrontierFeatures, ShiftStatus};

    const MODEL: [u8; 32] = [1; 32];
    const POLICY: [u8; 32] = [2; 32];
    const POPULATION: [u8; 32] = [3; 32];
    const CALIBRATION: [u8; 32] = [4; 32];

    fn vector(value: u128) -> ResourceVector {
        ResourceVector::new(value, value, value, value, value, value)
    }

    fn prediction() -> FrontierCostPrediction {
        FrontierCostPrediction {
            features: ProofFrontierFeatures {
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
            frontier_digest: [5; 32],
            model_digest: MODEL,
            population_cohort_digest: POPULATION,
            point_cost: vector(10),
            uncertainty_scale: vector(2),
            shift_status: ShiftStatus::InDomain,
        }
    }

    fn selection(id: u64, family: u32) -> SelectionReceipt {
        SelectionReceipt {
            episode_id: id,
            decision_sequence: 1,
            candidate_pool_digest: [6; 32],
            selection_policy_digest: POLICY,
            selected_population_digest: POPULATION,
            pre_state_digest: [7; 32],
            frontier_digest: [5; 32],
            selected_operation_id: format!("operation-{id}"),
            selected_operation_family: family,
        }
    }

    fn example(id: u64, score: u128) -> SelectedProofCalibrationExample {
        SelectedProofCalibrationExample {
            selection: selection(id, 9),
            prediction: prediction(),
            observed_remaining_to_verified_proof: vector(10 + score * 2),
            meter_receipt_digest: [9; 32],
            completion_verifier_digest: [8; 32],
        }
    }

    fn calibrator(count: usize, alpha: f64) -> SelectionConditionedConformalCalibrator {
        let examples = (1..=count)
            .map(|id| example(id as u64, id as u128))
            .collect::<Vec<_>>();
        SelectionConditionedConformalCalibrator::fit(
            &examples,
            alpha,
            MODEL,
            POLICY,
            POPULATION,
            CALIBRATION,
        )
        .unwrap()
    }

    #[test]
    fn conformal_threshold_is_fitted_on_selected_policy_cohort() {
        let calibrator = calibrator(9, 0.10);
        assert_eq!(calibrator.threshold(), SelectionThreshold::Finite(9.0));
        assert_eq!(calibrator.calibration_count(), 9);
        let receipt = selection(101, 9);
        let envelope = calibrator.predict_upper(&receipt, prediction()).unwrap();
        let SelectedProofEnvelope::Bounded(bound) = envelope else {
            panic!("nine selected calibration examples support the finite 90% rank");
        };
        assert_eq!(bound.target_coverage, 0.90);
        assert_eq!(bound.selected_operation_family, 9);
        assert_eq!(bound.selection_policy_digest, POLICY);
        assert!(vector(28).fits_within(bound.upper));
    }

    #[test]
    fn insufficient_selected_sample_returns_unbounded_envelope() {
        let calibrator = calibrator(8, 0.10);
        assert_eq!(calibrator.threshold(), SelectionThreshold::Unbounded);
        assert!(matches!(
            calibrator
                .predict_upper(&selection(101, 9), prediction())
                .unwrap(),
            SelectedProofEnvelope::Unbounded {
                reason: SelectionUnboundedReason::CalibrationSampleTooSmall,
                ..
            }
        ));
    }

    #[test]
    fn shifted_selected_frontier_returns_unbounded_envelope() {
        let calibrator = calibrator(9, 0.10);
        let mut shifted = prediction();
        shifted.shift_status = ShiftStatus::SuspectedShift;
        assert!(matches!(
            calibrator
                .predict_upper(&selection(101, 9), shifted)
                .unwrap(),
            SelectedProofEnvelope::Unbounded {
                reason: SelectionUnboundedReason::ShiftedFrontier,
                ..
            }
        ));
    }

    #[test]
    fn policy_version_mismatch_is_rejected_at_fit_and_prediction() {
        let mut row = example(20, 1);
        row.selection.selection_policy_digest = [99; 32];
        assert_eq!(
            SelectionConditionedConformalCalibrator::fit(
                &[row],
                0.10,
                MODEL,
                POLICY,
                POPULATION,
                CALIBRATION
            ),
            Err(SelectionCalibrationError::SelectionPolicyMismatch)
        );

        let calibrator = calibrator(9, 0.10);
        let mut changed = selection(101, 9);
        changed.selection_policy_digest = [99; 32];
        assert_eq!(
            calibrator.predict_upper(&changed, prediction()),
            Err(SelectionCalibrationError::SelectionPolicyMismatch)
        );
    }

    #[test]
    fn candidate_pool_and_selected_operation_identity_are_mandatory() {
        let mut missing_pool = example(21, 1);
        missing_pool.selection.candidate_pool_digest = [0; 32];
        assert_eq!(
            SelectionConditionedConformalCalibrator::fit(
                &[missing_pool],
                0.10,
                MODEL,
                POLICY,
                POPULATION,
                CALIBRATION
            ),
            Err(SelectionCalibrationError::MissingCandidatePoolDigest)
        );

        let mut missing_operation = example(22, 1);
        missing_operation.selection.selected_operation_id.clear();
        assert_eq!(
            SelectionConditionedConformalCalibrator::fit(
                &[missing_operation],
                0.10,
                MODEL,
                POLICY,
                POPULATION,
                CALIBRATION
            ),
            Err(SelectionCalibrationError::MissingSelectedOperationId)
        );
    }

    #[test]
    fn duplicate_episodes_and_missing_verifier_are_rejected() {
        let duplicate = [example(1, 1), example(1, 2)];
        assert_eq!(
            SelectionConditionedConformalCalibrator::fit(
                &duplicate,
                0.10,
                MODEL,
                POLICY,
                POPULATION,
                CALIBRATION
            ),
            Err(SelectionCalibrationError::DuplicateEpisode)
        );

        let mut missing = example(2, 1);
        missing.completion_verifier_digest = [0; 32];
        assert_eq!(
            SelectionConditionedConformalCalibrator::fit(
                &[missing],
                0.10,
                MODEL,
                POLICY,
                POPULATION,
                CALIBRATION
            ),
            Err(SelectionCalibrationError::MissingCompletionVerifier)
        );
    }

    #[test]
    fn meter_receipt_is_required_for_selected_resource_cost() {
        let mut row = example(23, 1);
        row.meter_receipt_digest = [0; 32];
        assert_eq!(
            SelectionConditionedConformalCalibrator::fit(
                &[row],
                0.10,
                MODEL,
                POLICY,
                POPULATION,
                CALIBRATION
            ),
            Err(SelectionCalibrationError::MissingMeterReceipt)
        );
    }

    #[test]
    fn frontier_model_and_population_identity_must_match() {
        let calibrator = calibrator(9, 0.10);
        let mut receipt = selection(101, 9);
        receipt.frontier_digest = [99; 32];
        assert_eq!(
            calibrator.predict_upper(&receipt, prediction()),
            Err(SelectionCalibrationError::PredictionFrontierMismatch)
        );

        let mut pred = prediction();
        pred.model_digest = [99; 32];
        assert_eq!(
            calibrator.predict_upper(&selection(101, 9), pred),
            Err(SelectionCalibrationError::PredictionModelMismatch)
        );
    }

    #[test]
    fn mixed_operation_families_are_not_pooled() {
        let rows = [example(1, 1), example(2, 2)];
        let mut second = rows[1].clone();
        second.selection.selected_operation_family = 10;
        assert_eq!(
            SelectionConditionedConformalCalibrator::fit(
                &[rows[0].clone(), second],
                0.10,
                MODEL,
                POLICY,
                POPULATION,
                CALIBRATION
            ),
            Err(SelectionCalibrationError::SelectedPopulationMismatch)
        );
    }

    #[test]
    fn miscoverage_and_calibration_digests_are_validated() {
        let rows = [example(1, 1)];
        assert_eq!(
            SelectionConditionedConformalCalibrator::fit(
                &rows,
                0.0,
                MODEL,
                POLICY,
                POPULATION,
                CALIBRATION
            ),
            Err(SelectionCalibrationError::InvalidMiscoverage)
        );
        assert_eq!(
            SelectionConditionedConformalCalibrator::fit(
                &rows, 0.1, MODEL, POLICY, POPULATION, [0; 32]
            ),
            Err(SelectionCalibrationError::MissingCalibrationDigest)
        );
    }

    #[test]
    fn predictor_with_different_population_is_rejected() {
        let calibrator = calibrator(9, 0.10);
        let mut pred = prediction();
        pred.population_cohort_digest = [77; 32];
        assert_eq!(
            calibrator.predict_upper(&selection(101, 9), pred),
            Err(SelectionCalibrationError::SelectedPopulationMismatch)
        );
    }
}
