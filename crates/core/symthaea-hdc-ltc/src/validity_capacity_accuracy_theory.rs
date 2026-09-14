// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Idealized cleanup-accuracy null derived from the frozen score-moment model.
//!
//! [`crate::validity_capacity_theory::ValidityCapacityNullModel`] predicts the
//! target and single-distractor score variances under deliberately strong
//! independent-interference assumptions. This module adds one more falsifiable
//! prediction before `research_v0` is interpreted: the winner-take-all cleanup
//! accuracy implied when those score distributions are additionally approximated
//! as mutually independent Gaussians.
//!
//! Let
//!
//! `X_target ~ Normal(1, sigma_t^2)`
//!
//! and each of the `C - 1` distractors satisfy
//!
//! `Y_j ~ Normal(0, sigma_d^2)`.
//!
//! Under mutual independence,
//!
//! `P(correct) = integral phi(z) * Phi((1 + sigma_t*z)/sigma_d)^(C-1) dz`.
//!
//! This is an **idealized null prediction**, not a theorem about the real archive.
//! Real candidate scores share one memory state and semantic codewords can be
//! reused across historical facts, so correlated departures are expected to be
//! scientifically informative rather than treated as implementation failures.
//!
//! Numerical evaluation is deterministic. The integral is truncated to `[-10,10]`
//! standard deviations (omitted Gaussian mass is negligible for this use) and
//! evaluated with composite Simpson quadrature at two fixed resolutions. The
//! normal CDF uses the classic five-coefficient rational approximation whose
//! absolute error is below roughly `8e-8`; public reference tests therefore pin
//! cleanup predictions at a tolerance appropriate to that approximation rather
//! than pretending to machine-precision probability values.

use crate::validity_capacity::ValidityCapacityCase;
use crate::validity_capacity_theory::{
    ValidityCapacityNullModel, ValidityCapacityTheoryError,
};
#[cfg(test)]
use std::f64::consts::PI;

const INTEGRATION_BOUND: f64 = 10.0;
const COARSE_INTERVALS: usize = 4_096;
const FINE_INTERVALS: usize = 8_192;
const STANDARD_NORMAL_SCALE: f64 = 0.398_942_280_401_432_7;

/// Pre-result cleanup-success prediction under an independent-Gaussian score null.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidityCapacityAccuracyNullModel {
    /// Exact finite-dimensional score-moment null inherited from #2802.
    pub score_model: ValidityCapacityNullModel,
    /// Fine-grid deterministic quadrature result for `P(correct)`.
    pub predicted_accuracy: f64,
    /// Coarser quadrature result, retained only as a numerical convergence check.
    pub coarse_predicted_accuracy: f64,
    /// Absolute difference between the fixed fine and coarse quadratures.
    pub integration_refinement_delta: f64,
    /// Symmetric standard-normal integration bound (`[-bound, bound]`).
    pub integration_bound: f64,
    /// Fine composite-Simpson interval count.
    pub integration_intervals: usize,
}

impl ValidityCapacityAccuracyNullModel {
    pub fn from_case(case: ValidityCapacityCase) -> Result<Self, ValidityCapacityTheoryError> {
        Self::new(case.dim, case.key_count, case.candidate_count, case.horizon)
    }

    pub fn new(
        dim: usize,
        key_count: usize,
        candidate_count: usize,
        horizon: u64,
    ) -> Result<Self, ValidityCapacityTheoryError> {
        let score_model =
            ValidityCapacityNullModel::new(dim, key_count, candidate_count, horizon)?;
        Ok(Self::from_score_model(score_model))
    }

    pub fn from_score_model(score_model: ValidityCapacityNullModel) -> Self {
        let coarse_predicted_accuracy =
            integrate_cleanup_accuracy(&score_model, COARSE_INTERVALS);
        let predicted_accuracy = integrate_cleanup_accuracy(&score_model, FINE_INTERVALS);
        let integration_refinement_delta =
            (predicted_accuracy - coarse_predicted_accuracy).abs();

        Self {
            score_model,
            predicted_accuracy,
            coarse_predicted_accuracy,
            integration_refinement_delta,
            integration_bound: INTEGRATION_BOUND,
            integration_intervals: FINE_INTERVALS,
        }
    }
}

fn integrate_cleanup_accuracy(model: &ValidityCapacityNullModel, intervals: usize) -> f64 {
    debug_assert!(intervals > 0 && intervals % 2 == 0);
    let lower = -INTEGRATION_BOUND;
    let upper = INTEGRATION_BOUND;
    let step = (upper - lower) / intervals as f64;

    let integrand = |z: f64| -> f64 {
        let target_score = 1.0 + model.target_noise_std * z;
        let one_distractor_below_target =
            normal_cdf(target_score / model.distractor_noise_std);
        standard_normal_pdf(z)
            * one_distractor_below_target.powf((model.candidate_count - 1) as f64)
    };

    let mut weighted_sum = integrand(lower) + integrand(upper);
    for index in 1..intervals {
        let z = lower + index as f64 * step;
        weighted_sum += if index % 2 == 0 {
            2.0 * integrand(z)
        } else {
            4.0 * integrand(z)
        };
    }
    weighted_sum * step / 3.0
}

#[inline]
fn standard_normal_pdf(value: f64) -> f64 {
    STANDARD_NORMAL_SCALE * (-0.5 * value * value).exp()
}

/// Deterministic standard-normal CDF approximation.
///
/// This is the well-known Abramowitz-Stegun 26.2.17 / 7.1.26-style five-term
/// rational approximation. Its error scale (~`1e-7`) is made explicit in this
/// module's tests and is intentionally much smaller than the scientific effects
/// this exploratory null is meant to resolve.
fn normal_cdf(value: f64) -> f64 {
    if value == 0.0 {
        return 0.5;
    }
    if value < 0.0 {
        return 1.0 - normal_cdf(-value);
    }
    if value >= 10.0 {
        return 1.0;
    }

    const P: f64 = 0.231_641_9;
    const B1: f64 = 0.319_381_530;
    const B2: f64 = -0.356_563_782;
    const B3: f64 = 1.781_477_937;
    const B4: f64 = -1.821_255_978;
    const B5: f64 = 1.330_274_429;

    let t = 1.0 / (1.0 + P * value);
    let polynomial = ((((B5 * t + B4) * t + B3) * t + B2) * t + B1) * t;
    1.0 - standard_normal_pdf(value) * polynomial
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validity_capacity::ValidityCapacityPlan;

    const REFERENCE_TOLERANCE: f64 = 2.0e-7;

    #[test]
    fn normal_cdf_approximation_matches_public_reference_points() {
        let references = [
            (-3.0, 0.001_349_898_031_630_094_5),
            (-2.0, 0.022_750_131_948_179_21),
            (-1.0, 0.158_655_253_931_457_07),
            (0.0, 0.5),
            (1.0, 0.841_344_746_068_542_9),
            (2.0, 0.977_249_868_051_820_8),
            (3.0, 0.998_650_101_968_369_9),
        ];
        for (x, expected) in references {
            let actual = normal_cdf(x);
            assert!(
                (actual - expected).abs() < 8.0e-8,
                "x={x}, expected={expected}, actual={actual}"
            );
        }
    }

    #[test]
    fn frozen_central_case_has_preregistered_cleanup_accuracy_prediction() {
        let model = ValidityCapacityAccuracyNullModel::new(4096, 8, 8, 128).unwrap();
        let reference = 0.892_433_735_974_979_3;
        assert!((model.predicted_accuracy - reference).abs() < REFERENCE_TOLERANCE);
        assert!(model.integration_refinement_delta < 1.0e-10);
        assert_eq!(model.integration_bound, 10.0);
        assert_eq!(model.integration_intervals, FINE_INTERVALS);
    }

    #[test]
    fn frozen_horizon_axis_predictions_match_pre_result_references() {
        let references = [
            (32, 0.999_734_289_885_740_8),
            (64, 0.985_556_192_959_088_1),
            (128, 0.892_433_735_974_979_3),
            (256, 0.709_945_549_531_542_3),
            (512, 0.524_039_710_809_305_0),
        ];
        for (horizon, reference) in references {
            let model = ValidityCapacityAccuracyNullModel::new(4096, 8, 8, horizon).unwrap();
            assert!(
                (model.predicted_accuracy - reference).abs() < REFERENCE_TOLERANCE,
                "H={horizon}, predicted={}, reference={reference}",
                model.predicted_accuracy
            );
            assert!(model.integration_refinement_delta < 1.0e-10);
        }
    }

    #[test]
    fn more_candidate_competitors_lower_the_independent_gaussian_prediction() {
        let two = ValidityCapacityAccuracyNullModel::new(4096, 8, 2, 128).unwrap();
        let eight = ValidityCapacityAccuracyNullModel::new(4096, 8, 8, 128).unwrap();
        let thirty_two = ValidityCapacityAccuracyNullModel::new(4096, 8, 32, 128).unwrap();
        assert!(two.predicted_accuracy > eight.predicted_accuracy);
        assert!(eight.predicted_accuracy > thirty_two.predicted_accuracy);
    }

    #[test]
    fn larger_dimension_improves_prediction_at_fixed_fact_count() {
        let small = ValidityCapacityAccuracyNullModel::new(1024, 8, 8, 128).unwrap();
        let medium = ValidityCapacityAccuracyNullModel::new(4096, 8, 8, 128).unwrap();
        let large = ValidityCapacityAccuracyNullModel::new(8192, 8, 8, 128).unwrap();
        assert!(small.predicted_accuracy < medium.predicted_accuracy);
        assert!(medium.predicted_accuracy < large.predicted_accuracy);
    }

    #[test]
    fn equal_rho_does_not_force_equal_finite_dimensional_accuracy() {
        let a = ValidityCapacityAccuracyNullModel::new(4096, 8, 8, 128).unwrap();
        let b = ValidityCapacityAccuracyNullModel::new(4096, 4, 8, 256).unwrap();
        assert_eq!(a.score_model.facts_per_dimension, b.score_model.facts_per_dimension);
        assert_ne!(
            a.score_model.target_noise_variance.to_bits(),
            b.score_model.target_noise_variance.to_bits()
        );
        assert_ne!(a.predicted_accuracy.to_bits(), b.predicted_accuracy.to_bits());
    }

    #[test]
    fn two_candidate_case_matches_difference_of_independent_gaussians() {
        let model = ValidityCapacityAccuracyNullModel::new(4096, 8, 2, 128).unwrap();
        let difference_std = (model.score_model.target_noise_variance
            + model.score_model.distractor_noise_variance)
            .sqrt();
        let analytic_two_candidate = normal_cdf(1.0 / difference_std);
        assert!(
            (model.predicted_accuracy - analytic_two_candidate).abs() < REFERENCE_TOLERANCE,
            "quadrature={}, analytic={analytic_two_candidate}",
            model.predicted_accuracy
        );
    }

    #[test]
    fn research_v0_cases_all_produce_finite_probabilities_and_converged_quadrature() {
        for case in ValidityCapacityPlan::research_v0().cases {
            let model = ValidityCapacityAccuracyNullModel::from_case(case).unwrap();
            assert!(model.predicted_accuracy.is_finite());
            assert!((0.0..=1.0).contains(&model.predicted_accuracy));
            assert!(
                model.integration_refinement_delta < 1.0e-10,
                "case={case:?}, delta={}",
                model.integration_refinement_delta
            );
        }
    }

    #[test]
    fn normal_pdf_constant_is_self_consistent() {
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert!((STANDARD_NORMAL_SCALE - expected).abs() < 1.0e-16);
    }
}
