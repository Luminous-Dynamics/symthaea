use std::collections::BTreeSet;

use crate::{ResourceVector, ShiftStatus};

/// Compact, planner-visible summary of the live proof frontier.
///
/// Counts must be computed from the authoritative pre-action frontier and
/// frozen into the forecast receipt. This summary deliberately contains no
/// hidden evaluator labels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProofFrontierFeatures {
    pub unresolved_obligations: u64,
    pub mandatory_obligations: u64,
    pub maximum_dependency_depth: u64,
    pub dependency_edges: u64,
    pub admissible_alternatives: u64,
    pub missing_evidence_requirements: u64,
    pub distinct_operation_families: u64,
    pub distinct_evidence_families: u64,
    pub mandatory_verifier_operations: u64,
}

impl ProofFrontierFeatures {
    pub fn validate(self) -> Result<(), FrontierModelError> {
        if self.mandatory_obligations > self.unresolved_obligations {
            return Err(FrontierModelError::InconsistentFeatureCounts);
        }
        if self.unresolved_obligations == 0
            && (self.maximum_dependency_depth != 0
                || self.dependency_edges != 0
                || self.admissible_alternatives != 0
                || self.missing_evidence_requirements != 0
                || self.distinct_operation_families != 0
                || self.distinct_evidence_families != 0
                || self.mandatory_verifier_operations != 0)
        {
            return Err(FrontierModelError::NonemptyFeaturesForEmptyFrontier);
        }
        Ok(())
    }

    fn counts(self) -> [u64; 9] {
        [
            self.unresolved_obligations,
            self.mandatory_obligations,
            self.maximum_dependency_depth,
            self.dependency_edges,
            self.admissible_alternatives,
            self.missing_evidence_requirements,
            self.distinct_operation_families,
            self.distinct_evidence_families,
            self.mandatory_verifier_operations,
        ]
    }
}

/// Nonnegative linear coefficients for one six-dimensional resource output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrontierCostCoefficients {
    pub intercept: ResourceVector,
    pub per_unresolved_obligation: ResourceVector,
    pub per_mandatory_obligation: ResourceVector,
    pub per_dependency_depth: ResourceVector,
    pub per_dependency_edge: ResourceVector,
    pub per_admissible_alternative: ResourceVector,
    pub per_missing_evidence_requirement: ResourceVector,
    pub per_operation_family: ResourceVector,
    pub per_evidence_family: ResourceVector,
    pub per_mandatory_verifier: ResourceVector,
}

impl FrontierCostCoefficients {
    fn evaluate(
        self,
        features: ProofFrontierFeatures,
    ) -> Result<ResourceVector, FrontierModelError> {
        let counts = features.counts();
        let coefficients = [
            self.per_unresolved_obligation,
            self.per_mandatory_obligation,
            self.per_dependency_depth,
            self.per_dependency_edge,
            self.per_admissible_alternative,
            self.per_missing_evidence_requirement,
            self.per_operation_family,
            self.per_evidence_family,
            self.per_mandatory_verifier,
        ];
        let mut total = self.intercept;
        for (coefficient, count) in coefficients.into_iter().zip(counts) {
            let contribution =
                scale_resource_vector(coefficient, count).ok_or(FrontierModelError::ArithmeticOverflow)?;
            total = total
                .checked_add(contribution)
                .ok_or(FrontierModelError::ArithmeticOverflow)?;
        }
        Ok(total)
    }
}

/// Deterministic structural baseline. Coefficients are supplied by the
/// experimenter / fitted outside this type and must be frozen before
/// calibration. This crate does not pretend the coefficients are trained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearProofFrontierCostModel {
    pub model_digest: [u8; 32],
    pub point_cost: FrontierCostCoefficients,
    pub uncertainty_scale: FrontierCostCoefficients,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrontierModelError {
    MissingModelDigest,
    MissingFrontierDigest,
    MissingPopulationCohortDigest,
    InconsistentFeatureCounts,
    NonemptyFeaturesForEmptyFrontier,
    ArithmeticOverflow,
    ZeroUncertaintyScale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrontierCostPrediction {
    pub features: ProofFrontierFeatures,
    pub frontier_digest: [u8; 32],
    pub model_digest: [u8; 32],
    /// Declared exchangeability cohort (e.g. task-family/runtime/time partition).
    pub population_cohort_digest: [u8; 32],
    pub point_cost: ResourceVector,
    pub uncertainty_scale: ResourceVector,
    pub shift_status: ShiftStatus,
}

impl LinearProofFrontierCostModel {
    pub fn predict(
        self,
        features: ProofFrontierFeatures,
        frontier_digest: [u8; 32],
        population_cohort_digest: [u8; 32],
        shift_status: ShiftStatus,
    ) -> Result<FrontierCostPrediction, FrontierModelError> {
        if self.model_digest.iter().all(|byte| *byte == 0) {
            return Err(FrontierModelError::MissingModelDigest);
        }
        if frontier_digest.iter().all(|byte| *byte == 0) {
            return Err(FrontierModelError::MissingFrontierDigest);
        }
        if population_cohort_digest.iter().all(|byte| *byte == 0) {
            return Err(FrontierModelError::MissingPopulationCohortDigest);
        }
        features.validate()?;
        let point_cost = self.point_cost.evaluate(features)?;
        let uncertainty_scale = self.uncertainty_scale.evaluate(features)?;
        if !resource_vector_is_positive(uncertainty_scale) {
            return Err(FrontierModelError::ZeroUncertaintyScale);
        }
        Ok(FrontierCostPrediction {
            features,
            frontier_digest,
            model_digest: self.model_digest,
            population_cohort_digest,
            point_cost,
            uncertainty_scale,
            shift_status,
        })
    }
}

/// Calibration rows must come from independently verified completed episodes.
/// Censored / competing-terminal observations belong in survival analysis, not
/// in this completed-case split-conformal residual set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompletedCalibrationExample {
    pub episode_id: u64,
    pub prediction: FrontierCostPrediction,
    pub observed_remaining: ResourceVector,
    pub completion_verifier_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConformalThreshold {
    Finite(f64),
    Unbounded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorConformalError {
    InvalidMiscoverage,
    EmptyCalibrationSet,
    MissingCalibrationDigest,
    MissingCohortDigest,
    DuplicateEpisode,
    InvalidEpisodeId,
    PredictionFromDifferentModel,
    DifferentCalibrationCohort,
    MissingFrontierDigest,
    InvalidFrontierFeatures,
    MissingCompletionVerifier,
    ShiftedCalibrationExample,
    InvalidPredictionScale,
    ResourceOverflow,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointCostBound {
    pub upper: ResourceVector,
    pub target_coverage: f64,
    pub calibration_count: usize,
    pub conformal_threshold: f64,
    pub model_digest: [u8; 32],
    pub calibration_digest: [u8; 32],
    pub calibration_cohort_digest: [u8; 32],
    pub frontier_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JointCostEnvelope {
    Bounded(JointCostBound),
    Unbounded {
        target_coverage: f64,
        calibration_count: usize,
        calibration_digest: [u8; 32],
        reason: UnboundedEnvelopeReason,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnboundedEnvelopeReason {
    CalibrationSampleTooSmall,
    ShiftedContext,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointBudgetDecision {
    Fits,
    DoesNotFit,
    UnboundedEnvelope,
    ExistingBudgetInvariantBroken,
    ArithmeticOverflow,
}

/// Split-conformal calibration for a simultaneous, one-sided six-resource
/// envelope. One score takes the maximum positive normalized residual across
/// every resource dimension; therefore the calibration event is joint.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitVectorConformalCalibrator {
    target_coverage: f64,
    threshold: ConformalThreshold,
    model_digest: [u8; 32],
    calibration_digest: [u8; 32],
    calibration_cohort_digest: [u8; 32],
    calibration_count: usize,
}

impl SplitVectorConformalCalibrator {
    pub fn fit(
        examples: &[CompletedCalibrationExample],
        miscoverage: f64,
        model_digest: [u8; 32],
        calibration_digest: [u8; 32],
        calibration_cohort_digest: [u8; 32],
    ) -> Result<Self, VectorConformalError> {
        if !miscoverage.is_finite() || !(0.0..1.0).contains(&miscoverage) {
            return Err(VectorConformalError::InvalidMiscoverage);
        }
        if examples.is_empty() {
            return Err(VectorConformalError::EmptyCalibrationSet);
        }
        if model_digest.iter().all(|byte| *byte == 0) {
            return Err(VectorConformalError::PredictionFromDifferentModel);
        }
        if calibration_digest.iter().all(|byte| *byte == 0) {
            return Err(VectorConformalError::MissingCalibrationDigest);
        }
        if calibration_cohort_digest.iter().all(|byte| *byte == 0) {
            return Err(VectorConformalError::MissingCohortDigest);
        }

        let mut episode_ids = BTreeSet::new();
        let mut scores = Vec::with_capacity(examples.len());
        for example in examples {
            if example.episode_id == 0 {
                return Err(VectorConformalError::InvalidEpisodeId);
            }
            if !episode_ids.insert(example.episode_id) {
                return Err(VectorConformalError::DuplicateEpisode);
            }
            if example.completion_verifier_digest.iter().all(|byte| *byte == 0) {
                return Err(VectorConformalError::MissingCompletionVerifier);
            }
            validate_prediction(example.prediction)?;
            if example.prediction.model_digest != model_digest {
                return Err(VectorConformalError::PredictionFromDifferentModel);
            }
            if example.prediction.population_cohort_digest != calibration_cohort_digest {
                return Err(VectorConformalError::DifferentCalibrationCohort);
            }
            if example.prediction.shift_status != ShiftStatus::InDomain {
                return Err(VectorConformalError::ShiftedCalibrationExample);
            }
            scores.push(max_normalized_upper_residual(
                example.observed_remaining,
                example.prediction.point_cost,
                example.prediction.uncertainty_scale,
            ));
        }

        scores.sort_by(f64::total_cmp);
        // The (n + 1) conformal rank is intentional. If it exceeds n, the
        // finite-sample-valid upper threshold is +infinity, not max(scores).
        let rank = (((examples.len() + 1) as f64) * (1.0 - miscoverage)).ceil() as usize;
        let threshold = if rank == 0 || rank > scores.len() {
            ConformalThreshold::Unbounded
        } else {
            ConformalThreshold::Finite(scores[rank - 1])
        };

        Ok(Self {
            target_coverage: 1.0 - miscoverage,
            threshold,
            model_digest,
            calibration_digest,
            calibration_cohort_digest,
            calibration_count: examples.len(),
        })
    }

    pub fn threshold(&self) -> ConformalThreshold {
        self.threshold
    }

    pub fn calibration_count(&self) -> usize {
        self.calibration_count
    }

    pub fn predict_upper(
        &self,
        prediction: FrontierCostPrediction,
    ) -> Result<JointCostEnvelope, VectorConformalError> {
        validate_prediction(prediction)?;
        if prediction.model_digest != self.model_digest {
            return Err(VectorConformalError::PredictionFromDifferentModel);
        }
        if prediction.population_cohort_digest != self.calibration_cohort_digest {
            return Err(VectorConformalError::DifferentCalibrationCohort);
        }
        if prediction.shift_status != ShiftStatus::InDomain {
            return Ok(JointCostEnvelope::Unbounded {
                target_coverage: self.target_coverage,
                calibration_count: self.calibration_count,
                calibration_digest: self.calibration_digest,
                reason: UnboundedEnvelopeReason::ShiftedContext,
            });
        }
        match self.threshold {
            ConformalThreshold::Unbounded => Ok(JointCostEnvelope::Unbounded {
                target_coverage: self.target_coverage,
                calibration_count: self.calibration_count,
                calibration_digest: self.calibration_digest,
                reason: UnboundedEnvelopeReason::CalibrationSampleTooSmall,
            }),
            ConformalThreshold::Finite(threshold) => {
                let upper = upper_resource_vector(
                    prediction.point_cost,
                    prediction.uncertainty_scale,
                    threshold,
                )?;
                Ok(JointCostEnvelope::Bounded(JointCostBound {
                    upper,
                    target_coverage: self.target_coverage,
                    calibration_count: self.calibration_count,
                    conformal_threshold: threshold,
                    model_digest: self.model_digest,
                    calibration_digest: self.calibration_digest,
                    calibration_cohort_digest: self.calibration_cohort_digest,
                    frontier_digest: prediction.frontier_digest,
                }))
            }
        }
    }
}

/// Tests whether the simultaneous upper envelope plus mandatory verification
/// reserve fits the actual ledger. Completion probability is a separate gate;
/// this function makes no claim about it and never enforces the physical cap.
pub fn assess_joint_resource_fit(
    envelope: JointCostEnvelope,
    mandatory_verification_reserve: ResourceVector,
    ledger: &crate::BudgetLedger,
) -> JointBudgetDecision {
    if !ledger.hard_budget_invariant_holds() {
        return JointBudgetDecision::ExistingBudgetInvariantBroken;
    }
    let JointCostEnvelope::Bounded(bound) = envelope else {
        return JointBudgetDecision::UnboundedEnvelope;
    };
    let Some(required) = bound.upper.checked_add(mandatory_verification_reserve) else {
        return JointBudgetDecision::ArithmeticOverflow;
    };
    if required.fits_within(ledger.remaining_capacity()) {
        JointBudgetDecision::Fits
    } else {
        JointBudgetDecision::DoesNotFit
    }
}

fn validate_prediction(
    prediction: FrontierCostPrediction,
) -> Result<(), VectorConformalError> {
    if prediction.frontier_digest.iter().all(|byte| *byte == 0) {
        return Err(VectorConformalError::MissingFrontierDigest);
    }
    if prediction.features.validate().is_err() {
        return Err(VectorConformalError::InvalidFrontierFeatures);
    }
    if !resource_vector_is_positive(prediction.uncertainty_scale) {
        return Err(VectorConformalError::InvalidPredictionScale);
    }
    Ok(())
}

fn max_normalized_upper_residual(
    observed: ResourceVector,
    point: ResourceVector,
    scale: ResourceVector,
) -> f64 {
    let observed_values = resource_components(observed);
    let point_values = resource_components(point);
    let scale_values = resource_components(scale);
    observed_values
        .into_iter()
        .zip(point_values)
        .zip(scale_values)
        .map(|((actual, estimate), uncertainty)| {
            ((actual as f64 - estimate as f64) / uncertainty as f64).max(0.0)
        })
        .fold(0.0, f64::max)
}

fn upper_resource_vector(
    point: ResourceVector,
    scale: ResourceVector,
    threshold: f64,
) -> Result<ResourceVector, VectorConformalError> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(VectorConformalError::ResourceOverflow);
    }
    let point = resource_components(point);
    let scale = resource_components(scale);
    let mut upper = [0_u128; 6];
    for index in 0..6 {
        let value = point[index] as f64 + threshold * scale[index] as f64;
        if !value.is_finite() || value < 0.0 || value >= u128::MAX as f64 {
            return Err(VectorConformalError::ResourceOverflow);
        }
        upper[index] = value.ceil() as u128;
    }
    Ok(ResourceVector::new(
        upper[0], upper[1], upper[2], upper[3], upper[4], upper[5],
    ))
}

fn resource_components(vector: ResourceVector) -> [u128; 6] {
    [
        vector.tokens,
        vector.tool_calls,
        vector.cpu_ms,
        vector.gpu_ms,
        vector.wall_ms,
        vector.micro_usd,
    ]
}

fn resource_vector_is_positive(vector: ResourceVector) -> bool {
    resource_components(vector).into_iter().all(|value| value > 0)
}

fn scale_resource_vector(vector: ResourceVector, factor: u64) -> Option<ResourceVector> {
    let factor = factor as u128;
    Some(ResourceVector::new(
        vector.tokens.checked_mul(factor)?,
        vector.tool_calls.checked_mul(factor)?,
        vector.cpu_ms.checked_mul(factor)?,
        vector.gpu_ms.checked_mul(factor)?,
        vector.wall_ms.checked_mul(factor)?,
        vector.micro_usd.checked_mul(factor)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BudgetLedger, ShiftStatus};

    const MODEL: [u8; 32] = [1; 32];
    const CALIBRATION: [u8; 32] = [2; 32];
    const COHORT: [u8; 32] = [3; 32];

    fn vector(value: u128) -> ResourceVector {
        ResourceVector::new(value, value, value, value, value, value)
    }

    fn features(unresolved: u64) -> ProofFrontierFeatures {
        ProofFrontierFeatures {
            unresolved_obligations: unresolved,
            mandatory_obligations: unresolved / 2,
            maximum_dependency_depth: unresolved.min(8),
            dependency_edges: unresolved.saturating_sub(1),
            admissible_alternatives: unresolved,
            missing_evidence_requirements: unresolved / 3,
            distinct_operation_families: unresolved.min(3),
            distinct_evidence_families: unresolved.min(4),
            mandatory_verifier_operations: unresolved / 4,
        }
    }

    fn model() -> LinearProofFrontierCostModel {
        let unit = vector(1);
        let scale = vector(2);
        LinearProofFrontierCostModel {
            model_digest: MODEL,
            point_cost: FrontierCostCoefficients {
                intercept: vector(5),
                per_unresolved_obligation: unit,
                per_mandatory_obligation: unit,
                per_dependency_depth: unit,
                per_dependency_edge: unit,
                per_admissible_alternative: unit,
                per_missing_evidence_requirement: unit,
                per_operation_family: unit,
                per_evidence_family: unit,
                per_mandatory_verifier: unit,
            },
            uncertainty_scale: FrontierCostCoefficients {
                intercept: scale,
                per_unresolved_obligation: ResourceVector::ZERO,
                per_mandatory_obligation: ResourceVector::ZERO,
                per_dependency_depth: ResourceVector::ZERO,
                per_dependency_edge: ResourceVector::ZERO,
                per_admissible_alternative: ResourceVector::ZERO,
                per_missing_evidence_requirement: ResourceVector::ZERO,
                per_operation_family: ResourceVector::ZERO,
                per_evidence_family: ResourceVector::ZERO,
                per_mandatory_verifier: ResourceVector::ZERO,
            },
        }
    }

    fn prediction(unresolved: u64) -> FrontierCostPrediction {
        model()
            .predict(features(unresolved), [4; 32], COHORT, ShiftStatus::InDomain)
            .unwrap()
    }

    fn example(id: u64, score: u128) -> CompletedCalibrationExample {
        let predicted = prediction(4);
        let scaled = ResourceVector::new(
            score * predicted.uncertainty_scale.tokens,
            score * predicted.uncertainty_scale.tool_calls,
            score * predicted.uncertainty_scale.cpu_ms,
            score * predicted.uncertainty_scale.gpu_ms,
            score * predicted.uncertainty_scale.wall_ms,
            score * predicted.uncertainty_scale.micro_usd,
        );
        CompletedCalibrationExample {
            episode_id: id,
            prediction: predicted,
            observed_remaining: predicted.point_cost.checked_add(scaled).unwrap(),
            completion_verifier_digest: [9; 32],
        }
    }

    fn calibrator(count: usize, alpha: f64) -> SplitVectorConformalCalibrator {
        let examples = (1..=count)
            .map(|id| example(id as u64, id as u128))
            .collect::<Vec<_>>();
        SplitVectorConformalCalibrator::fit(&examples, alpha, MODEL, CALIBRATION, COHORT)
            .unwrap()
    }

    #[test]
    fn frontier_baseline_changes_cost_with_obligation_structure() {
        let small = model()
            .predict(features(2), [4; 32], COHORT, ShiftStatus::InDomain)
            .unwrap();
        let large = prediction(7);
        assert!(small.point_cost.tokens < large.point_cost.tokens);
        assert!(small.point_cost.cpu_ms < large.point_cost.cpu_ms);
        assert_eq!(small.model_digest, large.model_digest);
    }

    #[test]
    fn frontier_features_reject_inconsistent_mandatory_counts() {
        let invalid = ProofFrontierFeatures {
            unresolved_obligations: 2,
            mandatory_obligations: 3,
            ..ProofFrontierFeatures::default()
        };
        assert_eq!(
            model().predict(invalid, [4; 32], COHORT, ShiftStatus::InDomain),
            Err(FrontierModelError::InconsistentFeatureCounts)
        );
    }

    #[test]
    fn frontier_can_require_multiple_mandatory_verifiers_for_one_obligation() {
        let feature_set = ProofFrontierFeatures {
            unresolved_obligations: 1,
            mandatory_obligations: 1,
            mandatory_verifier_operations: 3,
            ..ProofFrontierFeatures::default()
        };
        assert!(model()
            .predict(feature_set, [4; 32], COHORT, ShiftStatus::InDomain)
            .is_ok());
    }

    #[test]
    fn max_score_uses_the_worst_normalized_residual_across_all_resources() {
        let predicted = prediction(4);
        let observed = ResourceVector::new(
            predicted.point_cost.tokens + 2 * predicted.uncertainty_scale.tokens,
            predicted.point_cost.tool_calls + 6 * predicted.uncertainty_scale.tool_calls,
            predicted.point_cost.cpu_ms + predicted.uncertainty_scale.cpu_ms,
            predicted.point_cost.gpu_ms,
            predicted.point_cost.wall_ms + 3 * predicted.uncertainty_scale.wall_ms,
            predicted.point_cost.micro_usd,
        );
        let score = max_normalized_upper_residual(
            observed,
            predicted.point_cost,
            predicted.uncertainty_scale,
        );
        assert_eq!(score, 6.0);
    }

    #[test]
    fn conformal_order_statistic_uses_n_plus_one_rank() {
        let calibrator = calibrator(9, 0.10);
        assert_eq!(calibrator.threshold(), ConformalThreshold::Finite(9.0));
    }

    #[test]
    fn too_small_calibration_set_returns_an_unbounded_envelope() {
        let calibrator = calibrator(8, 0.10);
        assert_eq!(calibrator.threshold(), ConformalThreshold::Unbounded);
        assert!(matches!(
            calibrator.predict_upper(prediction(4)).unwrap(),
            JointCostEnvelope::Unbounded {
                reason: UnboundedEnvelopeReason::CalibrationSampleTooSmall,
                ..
            }
        ));
    }

    #[test]
    fn one_envelope_covers_all_six_dimensions_simultaneously() {
        let calibrator = calibrator(9, 0.10);
        let predicted = prediction(4);
        let envelope = calibrator.predict_upper(predicted).unwrap();
        let JointCostEnvelope::Bounded(bound) = envelope else {
            panic!("nine calibration samples support a finite 90% rank");
        };
        let observed = ResourceVector::new(
            predicted.point_cost.tokens + 9 * predicted.uncertainty_scale.tokens,
            predicted.point_cost.tool_calls + 9 * predicted.uncertainty_scale.tool_calls,
            predicted.point_cost.cpu_ms + 9 * predicted.uncertainty_scale.cpu_ms,
            predicted.point_cost.gpu_ms + 9 * predicted.uncertainty_scale.gpu_ms,
            predicted.point_cost.wall_ms + 9 * predicted.uncertainty_scale.wall_ms,
            predicted.point_cost.micro_usd + 9 * predicted.uncertainty_scale.micro_usd,
        );
        assert!(observed.fits_within(bound.upper));
        assert_eq!(bound.target_coverage, 0.90);
        assert_eq!(bound.calibration_count, 9);
    }

    #[test]
    fn frontier_shift_causes_unbounded_prediction_not_stale_calibration() {
        let calibrator = calibrator(9, 0.10);
        let shifted = model()
            .predict(features(4), [4; 32], COHORT, ShiftStatus::SuspectedShift)
            .unwrap();
        assert!(matches!(
            calibrator.predict_upper(shifted).unwrap(),
            JointCostEnvelope::Unbounded {
                reason: UnboundedEnvelopeReason::ShiftedContext,
                ..
            }
        ));
    }

    #[test]
    fn different_predictor_digest_cannot_reuse_calibration() {
        let calibrator = calibrator(9, 0.10);
        let mut changed = prediction(4);
        changed.model_digest = [8; 32];
        assert_eq!(
            calibrator.predict_upper(changed),
            Err(VectorConformalError::PredictionFromDifferentModel)
        );
    }

    #[test]
    fn calibration_and_prediction_must_share_the_declared_population_cohort() {
        let mut changed_cohort = prediction(4);
        changed_cohort.population_cohort_digest = [99; 32];
        let example = CompletedCalibrationExample {
            episode_id: 55,
            prediction: changed_cohort,
            observed_remaining: changed_cohort.point_cost,
            completion_verifier_digest: [9; 32],
        };
        assert_eq!(
            SplitVectorConformalCalibrator::fit(
                &[example],
                0.10,
                MODEL,
                CALIBRATION,
                COHORT
            ),
            Err(VectorConformalError::DifferentCalibrationCohort)
        );

        let calibrator = calibrator(9, 0.10);
        assert_eq!(
            calibrator.predict_upper(changed_cohort),
            Err(VectorConformalError::DifferentCalibrationCohort)
        );
    }

    #[test]
    fn duplicate_episodes_and_unverified_completed_cases_are_rejected() {
        let duplicate = [example(1, 1), example(1, 2)];
        assert_eq!(
            SplitVectorConformalCalibrator::fit(
                &duplicate,
                0.10,
                MODEL,
                CALIBRATION,
                COHORT
            ),
            Err(VectorConformalError::DuplicateEpisode)
        );

        let mut unverified = example(2, 1);
        unverified.completion_verifier_digest = [0; 32];
        assert_eq!(
            SplitVectorConformalCalibrator::fit(
                &[unverified],
                0.10,
                MODEL,
                CALIBRATION,
                COHORT
            ),
            Err(VectorConformalError::MissingCompletionVerifier)
        );
    }

    #[test]
    fn model_and_calibration_digests_are_required() {
        let row = [example(1, 1)];
        assert_eq!(
            SplitVectorConformalCalibrator::fit(&row, 0.0, MODEL, CALIBRATION, COHORT),
            Err(VectorConformalError::InvalidMiscoverage)
        );
        assert_eq!(
            SplitVectorConformalCalibrator::fit(&row, 0.1, [0; 32], CALIBRATION, COHORT),
            Err(VectorConformalError::PredictionFromDifferentModel)
        );
        assert_eq!(
            SplitVectorConformalCalibrator::fit(&row, 0.1, MODEL, [0; 32], COHORT),
            Err(VectorConformalError::MissingCalibrationDigest)
        );
    }

    #[test]
    fn simultaneous_envelope_plus_verification_reserve_uses_actual_ledger() {
        let calibrator = calibrator(9, 0.10);
        let envelope = calibrator.predict_upper(prediction(4)).unwrap();
        let ledger = BudgetLedger::new(vector(10_000));
        assert_eq!(
            assess_joint_resource_fit(envelope, vector(1), &ledger),
            JointBudgetDecision::Fits
        );
        let tight = BudgetLedger::new(vector(1));
        assert_eq!(
            assess_joint_resource_fit(envelope, vector(1), &tight),
            JointBudgetDecision::DoesNotFit
        );
    }

    #[test]
    fn large_envelope_conversion_fails_closed_on_overflow() {
        assert_eq!(
            upper_resource_vector(vector(u128::MAX - 2), vector(10), 1.0),
            Err(VectorConformalError::ResourceOverflow)
        );
    }
}
