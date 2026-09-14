// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Threshold-free falsification coordinates for validity-memory capacity.
//!
//! This module freezes the *comparisons* that should be inspected after the
//! preregistered experiment, rather than inventing them after results are known.
//! It aligns three pre-result surfaces:
//!
//! 1. the score-moment reconstruction (target + in-vocabulary distractor);
//! 2. the independent-Gaussian cleanup-accuracy null;
//! 3. the never-written shadow-distractor reconstruction.
//!
//! No pass/fail threshold or favorable regime is defined here. The output is a
//! residual vector intended to localize which approximation failed if empirical
//! behavior departs from the null hierarchy.

use crate::validity_capacity::{ValidityCapacityCase, ValidityCapacityPlan};
use crate::validity_capacity_accuracy_theory::ValidityCapacityAccuracyNullModel;
use crate::validity_capacity_score_moments::{
    ValidityCapacityScoreMomentError, measure_validity_capacity_score_moments,
};
use crate::validity_capacity_shadow_probe::{
    ValidityCapacityShadowError, measure_validity_capacity_shadow_distractors,
};
use crate::validity_capacity_theory::ValidityCapacityTheoryError;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct ValidityCapacityFalsificationObservation {
    pub case: ValidityCapacityCase,
    pub seed: u64,
    pub total_queries: u64,

    /// Winner-take-all empirical cleanup accuracy from the score reconstruction.
    pub empirical_accuracy: f64,
    /// Pre-result independent-Gaussian prediction from the frozen score null.
    pub predicted_accuracy: f64,
    /// Signed empirical-minus-null cleanup residual.
    pub accuracy_residual: f64,
    /// Numerical coarse/fine integration diagnostic for the accuracy null.
    pub accuracy_integration_refinement_delta: f64,

    /// Target-score null residuals around the predicted mean `1`.
    pub target_mean_bias: f64,
    pub target_mean_squared_residual: f64,
    pub target_mse_ratio_to_null: f64,

    /// In-vocabulary distractor residuals around the predicted mean `0`.
    pub vocabulary_distractor_mean_bias: f64,
    pub vocabulary_distractor_mean_squared_residual: f64,
    pub vocabulary_distractor_mse_ratio_to_null: f64,

    /// Never-written shadow-distractor residuals around the same null mean `0`.
    pub shadow_distractor_mean_bias: f64,
    pub shadow_distractor_mean_squared_residual: f64,
    pub shadow_distractor_mse_ratio_to_null: f64,
    pub max_abs_shadow_candidate_similarity: f64,

    /// Paired differences frozen before result interpretation. Positive values
    /// mean the in-vocabulary probe departed more strongly than the never-written
    /// shadow under the corresponding residual metric; negative values mean the
    /// opposite. No success threshold is attached to either sign.
    pub vocabulary_minus_shadow_mse: f64,
    pub vocabulary_minus_shadow_mse_ratio: f64,
    pub vocabulary_minus_shadow_variance: f64,

    /// Signed correctness-margin diagnostics retained from #2813.
    pub mean_true_margin: f64,
    pub smallest_true_margin: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidityCapacityFalsificationResult {
    pub observations: Vec<ValidityCapacityFalsificationObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidityCapacityFalsificationError {
    Score(String),
    Shadow(String),
    Theory(String),
    ObservationCountMismatch { score: usize, shadow: usize },
    ObservationAlignmentMismatch { index: usize },
}

impl fmt::Display for ValidityCapacityFalsificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Score(error) => write!(f, "falsification score path failed: {error}"),
            Self::Shadow(error) => write!(f, "falsification shadow path failed: {error}"),
            Self::Theory(error) => write!(f, "falsification theory path failed: {error}"),
            Self::ObservationCountMismatch { score, shadow } => write!(
                f,
                "falsification observation count mismatch: score={score}, shadow={shadow}"
            ),
            Self::ObservationAlignmentMismatch { index } => write!(
                f,
                "falsification observation alignment mismatch at index {index}"
            ),
        }
    }
}

impl std::error::Error for ValidityCapacityFalsificationError {}

impl From<ValidityCapacityScoreMomentError> for ValidityCapacityFalsificationError {
    fn from(value: ValidityCapacityScoreMomentError) -> Self {
        Self::Score(value.to_string())
    }
}

impl From<ValidityCapacityShadowError> for ValidityCapacityFalsificationError {
    fn from(value: ValidityCapacityShadowError) -> Self {
        Self::Shadow(value.to_string())
    }
}

impl From<ValidityCapacityTheoryError> for ValidityCapacityFalsificationError {
    fn from(value: ValidityCapacityTheoryError) -> Self {
        Self::Theory(value.to_string())
    }
}

pub fn measure_validity_capacity_falsification_surface(
    plan: &ValidityCapacityPlan,
) -> Result<ValidityCapacityFalsificationResult, ValidityCapacityFalsificationError> {
    let score = measure_validity_capacity_score_moments(plan)?;
    let shadow = measure_validity_capacity_shadow_distractors(plan)?;

    if score.observations.len() != shadow.observations.len() {
        return Err(ValidityCapacityFalsificationError::ObservationCountMismatch {
            score: score.observations.len(),
            shadow: shadow.observations.len(),
        });
    }

    let mut observations = Vec::with_capacity(score.observations.len());
    for (index, (score, shadow)) in score
        .observations
        .iter()
        .zip(&shadow.observations)
        .enumerate()
    {
        if score.case != shadow.case
            || score.seed != shadow.seed
            || score.total_queries != shadow.total_queries
        {
            return Err(ValidityCapacityFalsificationError::ObservationAlignmentMismatch {
                index,
            });
        }

        let accuracy_null = ValidityCapacityAccuracyNullModel::from_case(score.case)?;
        let target_bias = score.target_scores.mean - 1.0;
        let target_mse = score.target_scores.variance + target_bias * target_bias;
        let vocabulary_bias = score.probe_distractor_scores.mean;
        let vocabulary_mse =
            score.probe_distractor_scores.variance + vocabulary_bias * vocabulary_bias;
        let target_mse_ratio = ratio_or_infinity(target_mse, score.null_target_noise_variance);
        let vocabulary_mse_ratio =
            ratio_or_infinity(vocabulary_mse, score.null_distractor_noise_variance);

        observations.push(ValidityCapacityFalsificationObservation {
            case: score.case,
            seed: score.seed,
            total_queries: score.total_queries,
            empirical_accuracy: score.accuracy,
            predicted_accuracy: accuracy_null.predicted_accuracy,
            accuracy_residual: score.accuracy - accuracy_null.predicted_accuracy,
            accuracy_integration_refinement_delta: accuracy_null.integration_refinement_delta,
            target_mean_bias: target_bias,
            target_mean_squared_residual: target_mse,
            target_mse_ratio_to_null: target_mse_ratio,
            vocabulary_distractor_mean_bias: vocabulary_bias,
            vocabulary_distractor_mean_squared_residual: vocabulary_mse,
            vocabulary_distractor_mse_ratio_to_null: vocabulary_mse_ratio,
            shadow_distractor_mean_bias: shadow.shadow_mean_bias_from_zero,
            shadow_distractor_mean_squared_residual: shadow.shadow_mean_squared_residual,
            shadow_distractor_mse_ratio_to_null: shadow.shadow_mse_ratio_to_null,
            max_abs_shadow_candidate_similarity: shadow.max_abs_shadow_candidate_similarity,
            vocabulary_minus_shadow_mse: vocabulary_mse
                - shadow.shadow_mean_squared_residual,
            vocabulary_minus_shadow_mse_ratio: vocabulary_mse_ratio
                - shadow.shadow_mse_ratio_to_null,
            vocabulary_minus_shadow_variance: score.probe_distractor_scores.variance
                - shadow.shadow_scores.variance,
            mean_true_margin: score.mean_true_margin,
            smallest_true_margin: score.smallest_true_margin,
        });
    }

    Ok(ValidityCapacityFalsificationResult { observations })
}

fn ratio_or_infinity(observed: f64, predicted: f64) -> f64 {
    if predicted == 0.0 {
        if observed == 0.0 { 1.0 } else { f64::INFINITY }
    } else {
        observed / predicted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_surface_is_complete_finite_and_aligned() {
        let plan = ValidityCapacityPlan::smoke();
        let result = measure_validity_capacity_falsification_surface(&plan).unwrap();
        assert_eq!(result.observations.len(), plan.cases.len() * plan.replicate_seeds.len());

        for observation in result.observations {
            assert_eq!(
                observation.total_queries,
                observation.case.key_count as u64 * observation.case.horizon
            );
            assert!((0.0..=1.0).contains(&observation.empirical_accuracy));
            assert!((0.0..=1.0).contains(&observation.predicted_accuracy));
            assert!(observation.accuracy_residual.is_finite());
            assert!(observation.accuracy_integration_refinement_delta.is_finite());
            assert!(observation.target_mean_bias.is_finite());
            assert!(observation.target_mean_squared_residual.is_finite());
            assert!(observation.target_mse_ratio_to_null.is_finite());
            assert!(observation.vocabulary_distractor_mean_bias.is_finite());
            assert!(observation.vocabulary_distractor_mean_squared_residual.is_finite());
            assert!(observation.vocabulary_distractor_mse_ratio_to_null.is_finite());
            assert!(observation.shadow_distractor_mean_bias.is_finite());
            assert!(observation.shadow_distractor_mean_squared_residual.is_finite());
            assert!(observation.shadow_distractor_mse_ratio_to_null.is_finite());
            assert!((0.0..=1.0).contains(&observation.max_abs_shadow_candidate_similarity));
            assert!(observation.vocabulary_minus_shadow_mse.is_finite());
            assert!(observation.vocabulary_minus_shadow_mse_ratio.is_finite());
            assert!(observation.vocabulary_minus_shadow_variance.is_finite());
            assert!(observation.mean_true_margin.is_finite());
            assert!(observation.smallest_true_margin.is_finite());
        }
    }

    #[test]
    fn derived_residual_identities_are_exact() {
        let result =
            measure_validity_capacity_falsification_surface(&ValidityCapacityPlan::smoke()).unwrap();
        let score =
            measure_validity_capacity_score_moments(&ValidityCapacityPlan::smoke()).unwrap();
        let shadow =
            measure_validity_capacity_shadow_distractors(&ValidityCapacityPlan::smoke()).unwrap();

        for ((combined, score), shadow) in result
            .observations
            .iter()
            .zip(&score.observations)
            .zip(&shadow.observations)
        {
            let target_bias = score.target_scores.mean - 1.0;
            let target_mse = score.target_scores.variance + target_bias * target_bias;
            let vocabulary_mse = score.probe_distractor_scores.variance
                + score.probe_distractor_scores.mean * score.probe_distractor_scores.mean;

            assert_eq!(combined.target_mean_bias.to_bits(), target_bias.to_bits());
            assert_eq!(combined.target_mean_squared_residual.to_bits(), target_mse.to_bits());
            assert_eq!(
                combined.vocabulary_distractor_mean_squared_residual.to_bits(),
                vocabulary_mse.to_bits()
            );
            assert_eq!(
                combined.vocabulary_minus_shadow_mse.to_bits(),
                (vocabulary_mse - shadow.shadow_mean_squared_residual).to_bits()
            );
            assert_eq!(
                combined.vocabulary_minus_shadow_variance.to_bits(),
                (score.probe_distractor_scores.variance - shadow.shadow_scores.variance).to_bits()
            );
        }
    }

    #[test]
    fn repeated_smoke_surface_is_bit_deterministic() {
        let plan = ValidityCapacityPlan::smoke();
        let first = measure_validity_capacity_falsification_surface(&plan).unwrap();
        let second = measure_validity_capacity_falsification_surface(&plan).unwrap();
        assert_eq!(first.observations.len(), second.observations.len());

        for (left, right) in first.observations.iter().zip(&second.observations) {
            assert_eq!(left.case, right.case);
            assert_eq!(left.seed, right.seed);
            assert_eq!(left.empirical_accuracy.to_bits(), right.empirical_accuracy.to_bits());
            assert_eq!(left.predicted_accuracy.to_bits(), right.predicted_accuracy.to_bits());
            assert_eq!(left.accuracy_residual.to_bits(), right.accuracy_residual.to_bits());
            assert_eq!(
                left.vocabulary_minus_shadow_mse.to_bits(),
                right.vocabulary_minus_shadow_mse.to_bits()
            );
        }
    }
}
