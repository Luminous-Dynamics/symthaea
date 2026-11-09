// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! One-step hypothesis-discriminating experiment design.
//!
//! For a scalar Gaussian measurement, this module ranks candidate experiments
//! by expected mutual information between a discrete hypothesis and the
//! possible observation. Independent measurement and calibration uncertainty
//! are combined into one marginal variance for this one observation.
//!
//! Important limits:
//! - this is not a trajectory estimator or a physical hypothesis classifier;
//! - missing observations are assumed uninformative in v0.1;
//! - calibration uncertainty is marginal Gaussian and independent for this
//!   one measurement; repeated measurements with a shared calibration bias
//!   require a joint correlated model;
//! - local Fisher information and discrete global ambiguities remain separate
//!   observability checks and are not replaced by EIG.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

const COARSE_QUADRATURE_INTERVALS: usize = 1024;
const FINE_QUADRATURE_INTERVALS: usize = 2048;
const GAUSSIAN_TAIL_SIGMAS: f64 = 10.0;
const LOG_SQRT_TWO_PI: f64 = 0.918_938_533_204_672_7;
const PRIOR_SUM_TOLERANCE: f64 = 1.0e-9;
const NUMERICAL_EIG_TOLERANCE: f64 = 1.0e-7;
pub const MAX_ACTIVE_HYPOTHESES: usize = 64;
pub const MAX_ACTIVE_CANDIDATES: usize = 256;

/// Prior probability for one explicit competing explanation.
#[derive(Clone, Debug, PartialEq)]
pub struct HypothesisPrior {
    pub hypothesis_id: String,
    pub probability: f64,
}

/// Predicted scalar measurement under one hypothesis.
#[derive(Clone, Debug, PartialEq)]
pub struct HypothesisPrediction {
    pub hypothesis_id: String,
    pub expected_measurement: f64,
}

/// Candidate measurement with cost, availability and uncertainty.
#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentCandidate {
    pub candidate_id: String,
    pub cost: f64,
    /// Probability that the candidate produces a usable observation.
    ///
    /// In v0.1, a missing observation is assumed to carry no information.
    pub availability_probability: f64,
    /// Independent one-sigma measurement noise, in measurement units.
    pub measurement_sigma: f64,
    /// Independent one-sigma calibration nuisance, in the same units.
    ///
    /// For one observation this is marginalized by adding its variance.
    /// A persistent/common calibration bias across multiple observations
    /// requires a correlated joint model and is outside this API.
    pub calibration_sigma: f64,
    /// Exactly one prediction for each prior hypothesis. Order is immaterial;
    /// IDs, not vector positions, establish correspondence.
    pub predictions: Vec<HypothesisPrediction>,
}

/// Invalid inputs or numerical failure in active experiment design.
#[derive(Clone, Debug, PartialEq)]
pub enum ActiveDesignError {
    NoHypotheses,
    NoCandidates,
    TooManyHypotheses { count: usize, maximum: usize },
    TooManyCandidates { count: usize, maximum: usize },
    EmptyHypothesisId,
    EmptyCandidateId,
    DuplicateHypothesisId(String),
    DuplicateCandidateId(String),
    InvalidPrior { hypothesis_id: String },
    PriorProbabilitiesDoNotSumToOne { sum: f64 },
    InvalidCandidateCost { candidate_id: String },
    InvalidAvailability { candidate_id: String },
    InvalidMeasurementSigma { candidate_id: String },
    InvalidCalibrationSigma { candidate_id: String },
    NonFinitePrediction { candidate_id: String, hypothesis_id: String },
    PredictionSetMismatch { candidate_id: String },
    InvalidBudget,
    InvalidInformationGainFloor,
    NumericalOverflow { candidate_id: String },
    QuadratureOutOfBounds { candidate_id: String, eig: f64, entropy: f64 },
}

impl fmt::Display for ActiveDesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoHypotheses => write!(f, "at least one hypothesis prior is required"),
            Self::NoCandidates => write!(f, "at least one experiment candidate is required"),
            Self::TooManyHypotheses { count, maximum } => {
                write!(f, "{count} hypotheses exceed the maximum of {maximum}")
            }
            Self::TooManyCandidates { count, maximum } => {
                write!(f, "{count} candidates exceed the maximum of {maximum}")
            }
            Self::EmptyHypothesisId => write!(f, "hypothesis ID must not be empty"),
            Self::EmptyCandidateId => write!(f, "candidate ID must not be empty"),
            Self::DuplicateHypothesisId(id) => write!(f, "hypothesis ID {id} is duplicated"),
            Self::DuplicateCandidateId(id) => write!(f, "candidate ID {id} is duplicated"),
            Self::InvalidPrior { hypothesis_id } => {
                write!(f, "prior for hypothesis {hypothesis_id} is invalid")
            }
            Self::PriorProbabilitiesDoNotSumToOne { sum } => {
                write!(f, "hypothesis prior probabilities sum to {sum}, not one")
            }
            Self::InvalidCandidateCost { candidate_id } => {
                write!(f, "cost for candidate {candidate_id} must be finite and positive")
            }
            Self::InvalidAvailability { candidate_id } => write!(
                f,
                "availability for candidate {candidate_id} must lie in [0, 1]"
            ),
            Self::InvalidMeasurementSigma { candidate_id } => write!(
                f,
                "measurement sigma for candidate {candidate_id} must be finite and positive"
            ),
            Self::InvalidCalibrationSigma { candidate_id } => write!(
                f,
                "calibration sigma for candidate {candidate_id} must be finite and non-negative"
            ),
            Self::NonFinitePrediction { candidate_id, hypothesis_id } => write!(
                f,
                "prediction for candidate {candidate_id} and hypothesis {hypothesis_id} is not finite"
            ),
            Self::PredictionSetMismatch { candidate_id } => write!(
                f,
                "prediction IDs for candidate {candidate_id} do not exactly match the hypothesis set"
            ),
            Self::InvalidBudget => write!(f, "experiment budget must be finite and positive"),
            Self::InvalidInformationGainFloor => {
                write!(f, "information-gain floor must be finite and non-negative")
            }
            Self::NumericalOverflow { candidate_id } => {
                write!(f, "information calculation overflowed for candidate {candidate_id}")
            }
            Self::QuadratureOutOfBounds { candidate_id, eig, entropy } => write!(
                f,
                "candidate {candidate_id} EIG {eig} exceeds prior entropy {entropy}"
            ),
        }
    }
}

impl Error for ActiveDesignError {}

/// Information score for one candidate under the declared model.
#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentScore {
    pub candidate_id: String,
    /// Mutual information conditional on a usable observation, in nats.
    pub conditional_eig_nats: f64,
    /// EIG multiplied by availability under the v0.1 uninformative-missingness
    /// assumption, in nats.
    pub availability_adjusted_eig_nats: f64,
    /// Expected posterior entropy after accounting for the possibility that
    /// the experiment produces no usable observation, in nats.
    pub expected_posterior_entropy_nats: f64,
    pub prior_entropy_nats: f64,
    pub information_per_cost_nats: f64,
    pub quadrature_error_estimate_nats: f64,
    pub effective_sigma: f64,
    pub cost: f64,
    pub availability_probability: f64,
    pub within_budget: bool,
    pub eligible_for_selection: bool,
}

/// Ranked one-step design, or an explicit abstention if nothing is useful.
#[derive(Clone, Debug, PartialEq)]
pub struct ActiveDesignReport {
    pub ranked_candidates: Vec<ExperimentScore>,
    pub selected_candidate_id: Option<String>,
    pub prior_entropy_nats: f64,
    pub budget: f64,
    pub information_gain_floor_nats: f64,
    /// True when no within-budget, available candidate meets the floor.
    pub abstained: bool,
}

/// Rank candidate scalar measurements by expected information gain per cost.
///
/// The posterior over hypotheses is computed from Gaussian likelihoods with
/// variance `measurement_sigma^2 + calibration_sigma^2`. The expected
/// information gain is evaluated by deterministic composite Simpson
/// quadrature at 1,024 and 2,048 intervals; the difference is reported as a
/// convergence diagnostic.
///
/// A candidate can be selected only if it is within the budget, has non-zero
/// availability, and its availability-adjusted EIG meets the declared floor.
/// Ties are broken deterministically by conditional EIG and then candidate ID.
pub fn rank_experiments(
    priors: &[HypothesisPrior],
    candidates: &[ExperimentCandidate],
    budget: f64,
    information_gain_floor_nats: f64,
) -> Result<ActiveDesignReport, ActiveDesignError> {
    if priors.is_empty() {
        return Err(ActiveDesignError::NoHypotheses);
    }
    if candidates.is_empty() {
        return Err(ActiveDesignError::NoCandidates);
    }
    if priors.len() > MAX_ACTIVE_HYPOTHESES {
        return Err(ActiveDesignError::TooManyHypotheses {
            count: priors.len(),
            maximum: MAX_ACTIVE_HYPOTHESES,
        });
    }
    if candidates.len() > MAX_ACTIVE_CANDIDATES {
        return Err(ActiveDesignError::TooManyCandidates {
            count: candidates.len(),
            maximum: MAX_ACTIVE_CANDIDATES,
        });
    }
    if !budget.is_finite() || budget <= 0.0 {
        return Err(ActiveDesignError::InvalidBudget);
    }
    if !information_gain_floor_nats.is_finite() || information_gain_floor_nats < 0.0 {
        return Err(ActiveDesignError::InvalidInformationGainFloor);
    }

    let mut prior_ids = BTreeSet::new();
    let mut prior_sum = 0.0;
    for prior in priors {
        if prior.hypothesis_id.trim().is_empty() {
            return Err(ActiveDesignError::EmptyHypothesisId);
        }
        if !prior_ids.insert(prior.hypothesis_id.as_str()) {
            return Err(ActiveDesignError::DuplicateHypothesisId(
                prior.hypothesis_id.clone(),
            ));
        }
        if !prior.probability.is_finite()
            || prior.probability <= 0.0
            || prior.probability > 1.0
        {
            return Err(ActiveDesignError::InvalidPrior {
                hypothesis_id: prior.hypothesis_id.clone(),
            });
        }
        prior_sum += prior.probability;
    }
    if !prior_sum.is_finite() || (prior_sum - 1.0).abs() > PRIOR_SUM_TOLERANCE {
        return Err(ActiveDesignError::PriorProbabilitiesDoNotSumToOne { sum: prior_sum });
    }

    let prior_entropy_nats = -priors
        .iter()
        .map(|prior| prior.probability * prior.probability.ln())
        .sum::<f64>();
    let priors_by_id: BTreeMap<&str, f64> = priors
        .iter()
        .map(|prior| (prior.hypothesis_id.as_str(), prior.probability))
        .collect();

    let mut candidate_ids = BTreeSet::new();
    let mut ranked_candidates = Vec::with_capacity(candidates.len());

    for candidate in candidates {
        if candidate.candidate_id.trim().is_empty() {
            return Err(ActiveDesignError::EmptyCandidateId);
        }
        if !candidate_ids.insert(candidate.candidate_id.as_str()) {
            return Err(ActiveDesignError::DuplicateCandidateId(
                candidate.candidate_id.clone(),
            ));
        }
        if !candidate.cost.is_finite() || candidate.cost <= 0.0 {
            return Err(ActiveDesignError::InvalidCandidateCost {
                candidate_id: candidate.candidate_id.clone(),
            });
        }
        if !candidate.availability_probability.is_finite()
            || !(0.0..=1.0).contains(&candidate.availability_probability)
        {
            return Err(ActiveDesignError::InvalidAvailability {
                candidate_id: candidate.candidate_id.clone(),
            });
        }
        if !candidate.measurement_sigma.is_finite() || candidate.measurement_sigma <= 0.0 {
            return Err(ActiveDesignError::InvalidMeasurementSigma {
                candidate_id: candidate.candidate_id.clone(),
            });
        }
        if !candidate.calibration_sigma.is_finite() || candidate.calibration_sigma < 0.0 {
            return Err(ActiveDesignError::InvalidCalibrationSigma {
                candidate_id: candidate.candidate_id.clone(),
            });
        }

        let effective_sigma = candidate
            .measurement_sigma
            .hypot(candidate.calibration_sigma);
        if !effective_sigma.is_finite() || effective_sigma <= 0.0 {
            return Err(ActiveDesignError::InvalidMeasurementSigma {
                candidate_id: candidate.candidate_id.clone(),
            });
        }

        let mut prediction_by_id = BTreeMap::new();
        for prediction in &candidate.predictions {
            if prediction.hypothesis_id.trim().is_empty()
                || !prediction.expected_measurement.is_finite()
            {
                return Err(ActiveDesignError::NonFinitePrediction {
                    candidate_id: candidate.candidate_id.clone(),
                    hypothesis_id: prediction.hypothesis_id.clone(),
                });
            }
            if prediction_by_id
                .insert(prediction.hypothesis_id.as_str(), prediction.expected_measurement)
                .is_some()
            {
                return Err(ActiveDesignError::PredictionSetMismatch {
                    candidate_id: candidate.candidate_id.clone(),
                });
            }
        }
        if prediction_by_id.len() != priors.len()
            || prediction_by_id.keys().any(|id| !priors_by_id.contains_key(id))
        {
            return Err(ActiveDesignError::PredictionSetMismatch {
                candidate_id: candidate.candidate_id.clone(),
            });
        }

        let means: Vec<f64> = priors
            .iter()
            .map(|prior| prediction_by_id[prior.hypothesis_id.as_str()])
            .collect();
        let probabilities: Vec<f64> = priors
            .iter()
            .map(|prior| prior.probability)
            .collect();
        let coarse = gaussian_hypothesis_eig(
            &probabilities,
            &means,
            effective_sigma,
            COARSE_QUADRATURE_INTERVALS,
        );
        let fine = gaussian_hypothesis_eig(
            &probabilities,
            &means,
            effective_sigma,
            FINE_QUADRATURE_INTERVALS,
        );
        let (conditional_eig_nats, quadrature_error_estimate_nats) = match (coarse, fine) {
            (Some(coarse), Some(fine)) => (fine, (fine - coarse).abs()),
            _ => {
                return Err(ActiveDesignError::NumericalOverflow {
                    candidate_id: candidate.candidate_id.clone(),
                });
            }
        };

        if conditional_eig_nats > prior_entropy_nats + NUMERICAL_EIG_TOLERANCE {
            return Err(ActiveDesignError::QuadratureOutOfBounds {
                candidate_id: candidate.candidate_id.clone(),
                eig: conditional_eig_nats,
                entropy: prior_entropy_nats,
            });
        }
        let conditional_eig_nats = conditional_eig_nats.clamp(0.0, prior_entropy_nats);
        let availability_adjusted_eig_nats =
            candidate.availability_probability * conditional_eig_nats;
        let expected_posterior_entropy_nats =
            prior_entropy_nats - availability_adjusted_eig_nats;
        let information_per_cost_nats =
            availability_adjusted_eig_nats / candidate.cost;
        let within_budget = candidate.cost <= budget;
        let eligible_for_selection = within_budget
            && candidate.availability_probability > 0.0
            && availability_adjusted_eig_nats >= information_gain_floor_nats;

        ranked_candidates.push(ExperimentScore {
            candidate_id: candidate.candidate_id.clone(),
            conditional_eig_nats,
            availability_adjusted_eig_nats,
            expected_posterior_entropy_nats,
            prior_entropy_nats,
            information_per_cost_nats,
            quadrature_error_estimate_nats,
            effective_sigma,
            cost: candidate.cost,
            availability_probability: candidate.availability_probability,
            within_budget,
            eligible_for_selection,
        });
    }

    ranked_candidates.sort_by(|left, right| {
        right
            .eligible_for_selection
            .cmp(&left.eligible_for_selection)
            .then_with(|| {
                right
                    .information_per_cost_nats
                    .total_cmp(&left.information_per_cost_nats)
            })
            .then_with(|| {
                right
                    .availability_adjusted_eig_nats
                    .total_cmp(&left.availability_adjusted_eig_nats)
            })
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });

    let selected_candidate_id = ranked_candidates
        .iter()
        .find(|score| score.eligible_for_selection)
        .map(|score| score.candidate_id.clone());

    Ok(ActiveDesignReport {
        abstained: selected_candidate_id.is_none(),
        ranked_candidates,
        selected_candidate_id,
        prior_entropy_nats,
        budget,
        information_gain_floor_nats,
    })
}

fn gaussian_hypothesis_eig(
    probabilities: &[f64],
    means: &[f64],
    sigma: f64,
    intervals: usize,
) -> Option<f64> {
    if probabilities.len() != means.len()
        || probabilities.is_empty()
        || intervals == 0
        || intervals % 2 != 0
        || !sigma.is_finite()
        || sigma <= 0.0
    {
        return None;
    }

    // Integrate the expectation under each hypothesis in its own standard
    // normal coordinate z. This remains well-resolved when model means are
    // many thousands of sigma apart; integrating across the global span
    // would undersample the narrow likelihood peaks.
    let step = (2.0 * GAUSSIAN_TAIL_SIGMAS) / intervals as f64;
    let log_sigma = sigma.ln();
    let normalizer = log_sigma + LOG_SQRT_TWO_PI;
    let mut eig = 0.0;

    for hypothesis_index in 0..means.len() {
        let mean_i = means[hypothesis_index];
        let probability_i = probabilities[hypothesis_index];
        let integrand = |z: f64| -> Option<f64> {
            let observation = mean_i + sigma * z;
            if !observation.is_finite() {
                return None;
            }

            let log_pdf_i = -0.5 * z * z - normalizer;
            let mut log_weights = Vec::with_capacity(means.len());
            let mut maximum_log_weight = f64::NEG_INFINITY;

            for (probability, mean) in probabilities.iter().zip(means) {
                let standardized = (observation - mean) / sigma;
                let log_pdf = -0.5 * standardized * standardized - normalizer;
                let log_weight = probability.ln() + log_pdf;
                if !log_weight.is_finite() && log_weight != f64::NEG_INFINITY {
                    return None;
                }
                maximum_log_weight = maximum_log_weight.max(log_weight);
                log_weights.push(log_weight);
            }

            if !maximum_log_weight.is_finite() {
                return None;
            }
            let mixture_scaled = log_weights
                .iter()
                .map(|log_weight| (*log_weight - maximum_log_weight).exp())
                .sum::<f64>();
            if !mixture_scaled.is_finite() || mixture_scaled <= 0.0 {
                return None;
            }
            let log_mixture = maximum_log_weight + mixture_scaled.ln();
            let standard_normal_density = (-0.5 * z * z - LOG_SQRT_TWO_PI).exp();
            let value = standard_normal_density * (log_pdf_i - log_mixture);
            if value.is_finite() { Some(value) } else { None }
        };

        // Composite Simpson quadrature over z in [-10, 10]. The omitted
        // standard-normal tails are negligible for this declared model.
        let lower = -GAUSSIAN_TAIL_SIGMAS;
        let upper = GAUSSIAN_TAIL_SIGMAS;
        let mut integral = integrand(lower)? + integrand(upper)?;
        for index in 1..intervals {
            let z = lower + index as f64 * step;
            let value = integrand(z)?;
            let weight = if index % 2 == 0 { 2.0 } else { 4.0 };
            integral += weight * value;
        }
        let conditional_kl = integral * step / 3.0;
        if !conditional_kl.is_finite() {
            return None;
        }
        eig += probability_i * conditional_kl;
    }

    if eig.is_finite() {
        Some(eig.max(0.0))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn priors() -> Vec<HypothesisPrior> {
        vec![
            HypothesisPrior {
                hypothesis_id: "ballistic".to_string(),
                probability: 0.5,
            },
            HypothesisPrior {
                hypothesis_id: "maneuvering".to_string(),
                probability: 0.5,
            },
        ]
    }

    fn candidate(
        id: &str,
        first: f64,
        second: f64,
        measurement_sigma: f64,
        calibration_sigma: f64,
        cost: f64,
        availability_probability: f64,
    ) -> ExperimentCandidate {
        ExperimentCandidate {
            candidate_id: id.to_string(),
            cost,
            availability_probability,
            measurement_sigma,
            calibration_sigma,
            predictions: vec![
                HypothesisPrediction {
                    hypothesis_id: "ballistic".to_string(),
                    expected_measurement: first,
                },
                HypothesisPrediction {
                    hypothesis_id: "maneuvering".to_string(),
                    expected_measurement: second,
                },
            ],
        }
    }

    #[test]
    fn identical_hypothesis_predictions_return_zero_information_and_abstain() {
        let report = rank_experiments(
            &priors(),
            &[candidate("same", 10.0, 10.0, 1.0, 0.0, 1.0, 1.0)],
            10.0,
            1.0e-6,
        )
        .unwrap();

        assert!(report.abstained);
        assert_eq!(report.selected_candidate_id, None);
        assert!(report.ranked_candidates[0].conditional_eig_nats < 1.0e-10);
    }

    #[test]
    fn well_separated_hypotheses_have_positive_information() {
        let report = rank_experiments(
            &priors(),
            &[candidate("separating", -4.0, 4.0, 1.0, 0.0, 1.0, 1.0)],
            10.0,
            1.0e-6,
        )
        .unwrap();

        let score = &report.ranked_candidates[0];
        assert!(!report.abstained);
        assert!(score.conditional_eig_nats > 0.65);
        assert!(score.conditional_eig_nats <= report.prior_entropy_nats);
        assert!(score.quadrature_error_estimate_nats < 1.0e-6);
    }

    #[test]
    fn widely_separated_model_means_remain_numerically_resolved() {
        let report = rank_experiments(
            &priors(),
            &[candidate("wide", -1.0e6, 1.0e6, 1.0, 0.0, 1.0, 1.0)],
            10.0,
            0.0,
        )
        .unwrap();

        assert!((report.ranked_candidates[0].conditional_eig_nats - 2.0_f64.ln()).abs() < 1.0e-7);
    }

    #[test]
    fn information_gain_does_not_increase_with_measurement_or_calibration_noise() {
        let low_noise = rank_experiments(
            &priors(),
            &[candidate("low", -1.0, 1.0, 0.2, 0.0, 1.0, 1.0)],
            10.0,
            0.0,
        )
        .unwrap();
        let high_measurement_noise = rank_experiments(
            &priors(),
            &[candidate("high", -1.0, 1.0, 1.0, 0.0, 1.0, 1.0)],
            10.0,
            0.0,
        )
        .unwrap();
        let high_calibration_noise = rank_experiments(
            &priors(),
            &[candidate("calibration", -1.0, 1.0, 0.2, 1.0, 1.0, 1.0)],
            10.0,
            0.0,
        )
        .unwrap();

        assert!(
            low_noise.ranked_candidates[0].conditional_eig_nats
                > high_measurement_noise.ranked_candidates[0].conditional_eig_nats
        );
        assert!(
            low_noise.ranked_candidates[0].conditional_eig_nats
                > high_calibration_noise.ranked_candidates[0].conditional_eig_nats
        );
    }

    #[test]
    fn prior_entropy_bounds_information_and_expected_posterior_entropy() {
        let report = rank_experiments(
            &priors(),
            &[candidate("partial", -0.5, 0.5, 1.0, 0.4, 1.0, 0.8)],
            10.0,
            0.0,
        )
        .unwrap();
        let score = &report.ranked_candidates[0];

        assert!(score.conditional_eig_nats <= score.prior_entropy_nats);
        assert!(
            (score.expected_posterior_entropy_nats
                - (score.prior_entropy_nats - score.availability_adjusted_eig_nats))
                .abs()
                < 1.0e-12
        );
        assert!(score.expected_posterior_entropy_nats >= 0.0);
    }

    #[test]
    fn cost_availability_and_budget_change_the_selected_candidate() {
        let report = rank_experiments(
            &priors(),
            &[
                candidate("expensive", -5.0, 5.0, 0.5, 0.0, 20.0, 1.0),
                candidate("cheap", -2.0, 2.0, 0.5, 0.0, 1.0, 0.8),
                candidate("unavailable", -9.0, 9.0, 0.1, 0.0, 0.5, 0.0),
            ],
            2.0,
            0.01,
        )
        .unwrap();

        assert_eq!(report.selected_candidate_id.as_deref(), Some("cheap"));
        assert!(report.ranked_candidates.iter().any(|x| {
            x.candidate_id == "expensive" && !x.within_budget && !x.eligible_for_selection
        }));
        assert!(report.ranked_candidates.iter().any(|x| {
            x.candidate_id == "unavailable" && !x.eligible_for_selection
        }));
    }

    #[test]
    fn hypothesis_and_candidate_order_do_not_change_scores() {
        let base = rank_experiments(
            &priors(),
            &[
                candidate("a", -2.0, 2.0, 0.5, 0.2, 1.0, 0.9),
                candidate("b", -1.0, 1.0, 0.5, 0.0, 0.5, 1.0),
            ],
            10.0,
            0.0,
        )
        .unwrap();
        let reversed_priors = vec![priors()[1].clone(), priors()[0].clone()];
        let mut reversed = candidate("a", -2.0, 2.0, 0.5, 0.2, 1.0, 0.9);
        reversed.predictions.reverse();
        let reordered = rank_experiments(
            &reversed_priors,
            &[
                candidate("b", 1.0, -1.0, 0.5, 0.0, 0.5, 1.0),
                reversed,
            ],
            10.0,
            0.0,
        )
        .unwrap();

        assert_eq!(base.selected_candidate_id, reordered.selected_candidate_id);
        for left in &base.ranked_candidates {
            let right = reordered
                .ranked_candidates
                .iter()
                .find(|score| score.candidate_id == left.candidate_id)
                .unwrap();
            assert!(
                (left.conditional_eig_nats - right.conditional_eig_nats).abs() < 1.0e-10
            );
        }
    }

    #[test]
    fn invalid_prediction_sets_and_inputs_fail_closed() {
        let mut mismatched = candidate("bad-prediction", 0.0, 1.0, 1.0, 0.0, 1.0, 1.0);
        mismatched.predictions.pop();
        assert!(matches!(
            rank_experiments(&priors(), &[mismatched], 10.0, 0.0),
            Err(ActiveDesignError::PredictionSetMismatch { .. })
        ));

        assert_eq!(
            rank_experiments(&[], &[candidate("x", 0.0, 1.0, 1.0, 0.0, 1.0, 1.0)], 1.0, 0.0),
            Err(ActiveDesignError::NoHypotheses)
        );
        assert!(matches!(
            rank_experiments(
                &priors(),
                &[candidate("x", 0.0, 1.0, 1.0, 0.0, 1.0, 1.0)],
                1.0,
                f64::NAN
            ),
            Err(ActiveDesignError::InvalidInformationGainFloor)
        ));
    }
}
