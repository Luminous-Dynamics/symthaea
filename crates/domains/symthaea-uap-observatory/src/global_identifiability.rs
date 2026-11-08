// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Residual-qualified screening for discrete global solution ambiguities.
//!
//! Local Jacobian rank does not establish global uniqueness. This module
//! compares a caller-supplied set of physically distinct candidate solutions
//! against the same frozen observations and noise scales. It can find
//! observationally indistinguishable alternatives (including exact mirror
//! twins) even when each candidate is locally full rank.
//!
//! It does not enumerate all possible solutions. A report with no supplied
//! ambiguity is not a proof of global identifiability; candidate-generation
//! completeness must be established and reported by the caller.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// One candidate physical solution and its predicted measurements.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateSolution {
    pub solution_id: String,
    /// Parameter values in the physical units declared by the caller.
    pub parameters: Vec<f64>,
    /// Predicted measurements in the same order and units as the observations.
    pub predicted_observations: Vec<f64>,
}

/// Explicit tolerances for fit admission and measurement equivalence.
#[derive(Clone, Debug, PartialEq)]
pub struct GlobalIdentifiabilityConfig {
    /// Characteristic physical displacement per parameter. Used to calculate
    /// dimensionless separation between candidate solutions.
    pub parameter_scales: Vec<f64>,
    /// One-sigma measurement uncertainty for each observation.
    pub observation_sigmas: Vec<f64>,
    /// A candidate is admissible only if its RMS standardized residual is at
    /// or below this value.
    pub max_rms_standardized_residual: f64,
    /// A candidate is admissible only if no individual standardized residual
    /// exceeds this value.
    pub max_abs_standardized_residual: f64,
    /// Two admissible candidates are observationally equivalent only if the
    /// RMS standardized difference between their predictions is at or below
    /// this value.
    pub max_rms_prediction_difference: f64,
    /// Two admissible candidates are observationally equivalent only if no
    /// individual standardized prediction difference exceeds this value.
    pub max_abs_prediction_difference: f64,
    /// Candidates closer than this scaled Euclidean parameter distance are
    /// treated as the same local solution, not distinct global alternatives.
    pub min_scaled_parameter_separation: f64,
}

/// Maximum number of supplied candidates for the O(K^2 * M) pair screen.
pub const MAX_GLOBAL_IDENTIFIABILITY_CANDIDATES: usize = 512;

/// Input error or numerical failure during global ambiguity screening.
#[derive(Clone, Debug, PartialEq)]
pub enum GlobalIdentifiabilityError {
    NoObservations,
    NoCandidates,
    TooManyCandidates {
        count: usize,
        maximum: usize,
    },
    EmptySolutionId {
        index: usize,
    },
    DuplicateSolutionId(String),
    NoParameters,
    ParameterScaleCount {
        expected: usize,
        actual: usize,
    },
    ObservationSigmaCount {
        expected: usize,
        actual: usize,
    },
    ParameterCount {
        solution_id: String,
        expected: usize,
        actual: usize,
    },
    PredictionCount {
        solution_id: String,
        expected: usize,
        actual: usize,
    },
    InvalidParameterScale {
        index: usize,
    },
    InvalidObservationSigma {
        index: usize,
    },
    NonFiniteObservation {
        index: usize,
    },
    NonFiniteParameter {
        solution_id: String,
        index: usize,
    },
    NonFinitePrediction {
        solution_id: String,
        index: usize,
    },
    InvalidConfiguration,
    NumericalOverflow {
        solution_id: String,
    },
}

impl fmt::Display for GlobalIdentifiabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoObservations => write!(f, "observation vector is empty"),
            Self::NoCandidates => write!(f, "candidate solution set is empty"),
            Self::TooManyCandidates { count, maximum } => write!(
                f,
                "candidate solution set has {count} entries; maximum supported is {maximum}"
            ),
            Self::EmptySolutionId { index } => {
                write!(f, "candidate solution ID at index {index} is empty")
            }
            Self::DuplicateSolutionId(id) => {
                write!(f, "candidate solution ID {id} is duplicated")
            }
            Self::NoParameters => write!(f, "candidate parameter vector is empty"),
            Self::ParameterScaleCount { expected, actual } => {
                write!(f, "received {actual} parameter scales; expected {expected}")
            }
            Self::ObservationSigmaCount { expected, actual } => write!(
                f,
                "received {actual} observation sigmas; expected {expected}"
            ),
            Self::ParameterCount {
                solution_id,
                expected,
                actual,
            } => write!(
                f,
                "candidate {solution_id} has {actual} parameters; expected {expected}"
            ),
            Self::PredictionCount {
                solution_id,
                expected,
                actual,
            } => write!(
                f,
                "candidate {solution_id} has {actual} predictions; expected {expected}"
            ),
            Self::InvalidParameterScale { index } => {
                write!(f, "parameter scale {index} is not finite and positive")
            }
            Self::InvalidObservationSigma { index } => {
                write!(f, "observation sigma {index} is not finite and positive")
            }
            Self::NonFiniteObservation { index } => {
                write!(f, "observation {index} is not finite")
            }
            Self::NonFiniteParameter { solution_id, index } => {
                write!(
                    f,
                    "parameter {index} for candidate {solution_id} is not finite"
                )
            }
            Self::NonFinitePrediction { solution_id, index } => {
                write!(
                    f,
                    "prediction {index} for candidate {solution_id} is not finite"
                )
            }
            Self::InvalidConfiguration => {
                write!(f, "global-identifiability thresholds are invalid")
            }
            Self::NumericalOverflow { solution_id } => {
                write!(
                    f,
                    "residual or separation calculation overflowed for {solution_id}"
                )
            }
        }
    }
}

impl Error for GlobalIdentifiabilityError {}

/// Fit statistics for one supplied candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateFit {
    pub solution_id: String,
    pub chi_squared: f64,
    pub rms_standardized_residual: f64,
    pub max_abs_standardized_residual: f64,
    pub admissible: bool,
}

/// A pair of distinct, admissible candidates that the declared measurements
/// cannot distinguish at the configured noise-scaled tolerance.
#[derive(Clone, Debug, PartialEq)]
pub struct AmbiguousSolutionPair {
    pub left_solution_id: String,
    pub right_solution_id: String,
    /// Euclidean distance in dimensionless, caller-scaled parameter space.
    pub scaled_parameter_separation: f64,
    /// RMS prediction difference divided by per-observation sigma.
    pub rms_standardized_prediction_difference: f64,
    /// Maximum absolute prediction difference divided by per-observation sigma.
    pub max_abs_standardized_prediction_difference: f64,
}

/// Screening result for the exact candidate set supplied by the caller.
///
/// `no_supplied_ambiguity_found` means no pair in this candidate set passed
/// the configured test. It must never be promoted to "globally unique" unless
/// a separate, independently justified procedure proves that candidate
/// generation was exhaustive over the declared physical domain.
#[derive(Clone, Debug, PartialEq)]
pub struct GlobalIdentifiabilityReport {
    pub candidate_count: usize,
    pub admissible_candidate_count: usize,
    pub candidate_fits: Vec<CandidateFit>,
    pub ambiguous_pairs: Vec<AmbiguousSolutionPair>,
    pub no_supplied_ambiguity_found: bool,
}

/// Compare a supplied set of candidate solutions against frozen observations.
///
/// Candidate fit uses standardized residuals
/// `(prediction - observation) / sigma`. Pair equivalence uses standardized
/// differences between candidate predictions. Parameter separation uses the
/// declared characteristic scales:
///
/// `sqrt(sum_i (((left_i - right_i) / parameter_scale_i)^2))`.
///
/// This is a global-ambiguity *screen* over supplied alternatives, not a
/// nonlinear solver and not an exhaustive uniqueness proof.
pub fn screen_global_ambiguities(
    observations: &[f64],
    candidates: &[CandidateSolution],
    config: &GlobalIdentifiabilityConfig,
) -> Result<GlobalIdentifiabilityReport, GlobalIdentifiabilityError> {
    if observations.is_empty() {
        return Err(GlobalIdentifiabilityError::NoObservations);
    }
    if candidates.is_empty() {
        return Err(GlobalIdentifiabilityError::NoCandidates);
    }
    if candidates.len() > MAX_GLOBAL_IDENTIFIABILITY_CANDIDATES {
        return Err(GlobalIdentifiabilityError::TooManyCandidates {
            count: candidates.len(),
            maximum: MAX_GLOBAL_IDENTIFIABILITY_CANDIDATES,
        });
    }
    if config.parameter_scales.is_empty() {
        return Err(GlobalIdentifiabilityError::NoParameters);
    }
    let parameter_count = config.parameter_scales.len();
    if config.observation_sigmas.len() != observations.len() {
        return Err(GlobalIdentifiabilityError::ObservationSigmaCount {
            expected: observations.len(),
            actual: config.observation_sigmas.len(),
        });
    }
    if !config.max_rms_standardized_residual.is_finite()
        || config.max_rms_standardized_residual < 0.0
        || !config.max_abs_standardized_residual.is_finite()
        || config.max_abs_standardized_residual < 0.0
        || !config.max_rms_prediction_difference.is_finite()
        || config.max_rms_prediction_difference < 0.0
        || !config.max_abs_prediction_difference.is_finite()
        || config.max_abs_prediction_difference < 0.0
        || !config.min_scaled_parameter_separation.is_finite()
        || config.min_scaled_parameter_separation <= 0.0
    {
        return Err(GlobalIdentifiabilityError::InvalidConfiguration);
    }

    for (index, scale) in config.parameter_scales.iter().copied().enumerate() {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(GlobalIdentifiabilityError::InvalidParameterScale { index });
        }
    }
    for (index, sigma) in config.observation_sigmas.iter().copied().enumerate() {
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(GlobalIdentifiabilityError::InvalidObservationSigma { index });
        }
    }
    for (index, observation) in observations.iter().copied().enumerate() {
        if !observation.is_finite() {
            return Err(GlobalIdentifiabilityError::NonFiniteObservation { index });
        }
    }

    let mut ids = BTreeSet::new();
    let mut ordered_candidates: Vec<&CandidateSolution> = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        if candidate.solution_id.trim().is_empty() {
            return Err(GlobalIdentifiabilityError::EmptySolutionId { index });
        }
        if !ids.insert(candidate.solution_id.as_str()) {
            return Err(GlobalIdentifiabilityError::DuplicateSolutionId(
                candidate.solution_id.clone(),
            ));
        }
        if candidate.parameters.len() != parameter_count {
            return Err(GlobalIdentifiabilityError::ParameterCount {
                solution_id: candidate.solution_id.clone(),
                expected: parameter_count,
                actual: candidate.parameters.len(),
            });
        }
        if candidate.predicted_observations.len() != observations.len() {
            return Err(GlobalIdentifiabilityError::PredictionCount {
                solution_id: candidate.solution_id.clone(),
                expected: observations.len(),
                actual: candidate.predicted_observations.len(),
            });
        }
        for (parameter_index, value) in candidate.parameters.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(GlobalIdentifiabilityError::NonFiniteParameter {
                    solution_id: candidate.solution_id.clone(),
                    index: parameter_index,
                });
            }
        }
        for (prediction_index, value) in
            candidate.predicted_observations.iter().copied().enumerate()
        {
            if !value.is_finite() {
                return Err(GlobalIdentifiabilityError::NonFinitePrediction {
                    solution_id: candidate.solution_id.clone(),
                    index: prediction_index,
                });
            }
        }
        ordered_candidates.push(candidate);
    }
    ordered_candidates.sort_by(|left, right| left.solution_id.cmp(&right.solution_id));

    let mut candidate_fits = Vec::with_capacity(ordered_candidates.len());
    for candidate in &ordered_candidates {
        let mut chi_squared = 0.0;
        let mut max_abs_standardized_residual = 0.0_f64;
        for index in 0..observations.len() {
            let standardized = (candidate.predicted_observations[index] - observations[index])
                / config.observation_sigmas[index];
            if !standardized.is_finite() {
                return Err(GlobalIdentifiabilityError::NumericalOverflow {
                    solution_id: candidate.solution_id.clone(),
                });
            }
            chi_squared += standardized * standardized;
            max_abs_standardized_residual = max_abs_standardized_residual.max(standardized.abs());
            if !chi_squared.is_finite() {
                return Err(GlobalIdentifiabilityError::NumericalOverflow {
                    solution_id: candidate.solution_id.clone(),
                });
            }
        }
        let rms_standardized_residual = (chi_squared / observations.len() as f64).sqrt();
        let admissible = rms_standardized_residual <= config.max_rms_standardized_residual
            && max_abs_standardized_residual <= config.max_abs_standardized_residual;

        candidate_fits.push(CandidateFit {
            solution_id: candidate.solution_id.clone(),
            chi_squared,
            rms_standardized_residual,
            max_abs_standardized_residual,
            admissible,
        });
    }

    let mut ambiguous_pairs = Vec::new();
    for left_index in 0..ordered_candidates.len() {
        let left = ordered_candidates[left_index];
        let left_fit = &candidate_fits[left_index];
        if !left_fit.admissible {
            continue;
        }

        for right_index in (left_index + 1)..ordered_candidates.len() {
            let right = ordered_candidates[right_index];
            let right_fit = &candidate_fits[right_index];
            if !right_fit.admissible {
                continue;
            }

            let mut scaled_separation_squared = 0.0;
            for parameter_index in 0..parameter_count {
                let delta = (left.parameters[parameter_index] - right.parameters[parameter_index])
                    / config.parameter_scales[parameter_index];
                scaled_separation_squared += delta * delta;
            }
            if !scaled_separation_squared.is_finite() {
                return Err(GlobalIdentifiabilityError::NumericalOverflow {
                    solution_id: format!("{}:{}", left.solution_id, right.solution_id),
                });
            }
            let scaled_parameter_separation = scaled_separation_squared.sqrt();
            if scaled_parameter_separation < config.min_scaled_parameter_separation {
                continue;
            }

            let mut prediction_difference_squared = 0.0;
            let mut max_abs_standardized_prediction_difference = 0.0_f64;
            for index in 0..observations.len() {
                let standardized_difference = (left.predicted_observations[index]
                    - right.predicted_observations[index])
                    / config.observation_sigmas[index];
                if !standardized_difference.is_finite() {
                    return Err(GlobalIdentifiabilityError::NumericalOverflow {
                        solution_id: format!("{}:{}", left.solution_id, right.solution_id),
                    });
                }
                prediction_difference_squared += standardized_difference * standardized_difference;
                max_abs_standardized_prediction_difference =
                    max_abs_standardized_prediction_difference.max(standardized_difference.abs());
            }
            if !prediction_difference_squared.is_finite() {
                return Err(GlobalIdentifiabilityError::NumericalOverflow {
                    solution_id: format!("{}:{}", left.solution_id, right.solution_id),
                });
            }

            let rms_standardized_prediction_difference =
                (prediction_difference_squared / observations.len() as f64).sqrt();
            if rms_standardized_prediction_difference <= config.max_rms_prediction_difference
                && max_abs_standardized_prediction_difference
                    <= config.max_abs_prediction_difference
            {
                ambiguous_pairs.push(AmbiguousSolutionPair {
                    left_solution_id: left.solution_id.clone(),
                    right_solution_id: right.solution_id.clone(),
                    scaled_parameter_separation,
                    rms_standardized_prediction_difference,
                    max_abs_standardized_prediction_difference,
                });
            }
        }
    }

    Ok(GlobalIdentifiabilityReport {
        candidate_count: candidates.len(),
        admissible_candidate_count: candidate_fits.iter().filter(|fit| fit.admissible).count(),
        no_supplied_ambiguity_found: ambiguous_pairs.is_empty(),
        candidate_fits,
        ambiguous_pairs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze_jacobian;

    #[derive(Clone, Copy)]
    struct Point3 {
        x: f64,
        y: f64,
        z: f64,
    }

    fn range(left: Point3, right: Point3) -> f64 {
        ((left.x - right.x).powi(2) + (left.y - right.y).powi(2) + (left.z - right.z).powi(2))
            .sqrt()
    }

    fn predict_ranges(target: Point3, anchors: &[Point3]) -> Vec<f64> {
        anchors
            .iter()
            .map(|anchor| range(target, *anchor))
            .collect()
    }

    fn mirror_config(observation_count: usize) -> GlobalIdentifiabilityConfig {
        GlobalIdentifiabilityConfig {
            parameter_scales: vec![1.0, 1.0, 1.0],
            observation_sigmas: vec![0.01; observation_count],
            max_rms_standardized_residual: 1.0e-6,
            max_abs_standardized_residual: 1.0e-6,
            max_rms_prediction_difference: 1.0e-8,
            max_abs_prediction_difference: 1.0e-8,
            min_scaled_parameter_separation: 1.0,
        }
    }

    #[test]
    fn coplanar_range_anchors_expose_mirror_twin_even_with_full_local_rank() {
        let target = Point3 {
            x: 2.0,
            y: 3.0,
            z: 4.0,
        };
        let mirror = Point3 {
            x: 2.0,
            y: 3.0,
            z: -4.0,
        };
        let anchors = [
            Point3 {
                x: -1.0,
                y: 0.0,
                z: 0.0,
            },
            Point3 {
                x: 5.0,
                y: 0.0,
                z: 0.0,
            },
            Point3 {
                x: 0.0,
                y: 7.0,
                z: 0.0,
            },
        ];
        let observations = predict_ranges(target, &anchors);
        let target_predictions = predict_ranges(target, &anchors);
        let mirror_predictions = predict_ranges(mirror, &anchors);
        for (left, right) in target_predictions.iter().zip(&mirror_predictions) {
            assert!((left - right).abs() < 1.0e-12);
        }

        let jacobian: Vec<Vec<f64>> = anchors
            .iter()
            .map(|anchor| {
                let dx = target.x - anchor.x;
                let dy = target.y - anchor.y;
                let dz = target.z - anchor.z;
                let distance = dx.hypot(dy).hypot(dz);
                vec![dx / distance, dy / distance, dz / distance]
            })
            .collect();
        let local = analyze_jacobian(&jacobian, &[1.0, 1.0, 1.0], 1.0e-10).unwrap();
        assert_eq!(local.numerical_rank, 3);

        let candidates = [
            CandidateSolution {
                solution_id: "target-above-plane".to_string(),
                parameters: vec![target.x, target.y, target.z],
                predicted_observations: target_predictions,
            },
            CandidateSolution {
                solution_id: "mirror-below-plane".to_string(),
                parameters: vec![mirror.x, mirror.y, mirror.z],
                predicted_observations: mirror_predictions,
            },
        ];
        let report =
            screen_global_ambiguities(&observations, &candidates, &mirror_config(anchors.len()))
                .unwrap();

        assert_eq!(report.admissible_candidate_count, 2);
        assert_eq!(report.ambiguous_pairs.len(), 1);
        assert!(report.ambiguous_pairs[0].scaled_parameter_separation >= 8.0);
        assert!(!report.no_supplied_ambiguity_found);
    }

    #[test]
    fn a_non_coplanar_anchor_breaks_the_range_mirror_twin() {
        let target = Point3 {
            x: 2.0,
            y: 3.0,
            z: 4.0,
        };
        let mirror = Point3 {
            x: 2.0,
            y: 3.0,
            z: -4.0,
        };
        let anchors = [
            Point3 {
                x: -1.0,
                y: 0.0,
                z: 0.0,
            },
            Point3 {
                x: 5.0,
                y: 0.0,
                z: 0.0,
            },
            Point3 {
                x: 0.0,
                y: 7.0,
                z: 0.0,
            },
            Point3 {
                x: 0.0,
                y: 0.0,
                z: 3.0,
            },
        ];
        let observations = predict_ranges(target, &anchors);
        let candidates = [
            CandidateSolution {
                solution_id: "target-above-plane".to_string(),
                parameters: vec![target.x, target.y, target.z],
                predicted_observations: predict_ranges(target, &anchors),
            },
            CandidateSolution {
                solution_id: "mirror-below-plane".to_string(),
                parameters: vec![mirror.x, mirror.y, mirror.z],
                predicted_observations: predict_ranges(mirror, &anchors),
            },
        ];
        let report =
            screen_global_ambiguities(&observations, &candidates, &mirror_config(anchors.len()))
                .unwrap();

        assert_eq!(report.admissible_candidate_count, 1);
        assert!(report.ambiguous_pairs.is_empty());
        assert!(report.no_supplied_ambiguity_found);
        assert_eq!(report.candidate_fits[0].solution_id, "mirror-below-plane");
        assert!(!report.candidate_fits[0].admissible);
    }

    #[test]
    fn near_coplanar_anchor_is_a_noise_scaled_ambiguity_not_a_binary_claim() {
        let target = Point3 {
            x: 2.0,
            y: 3.0,
            z: 4.0,
        };
        let mirror = Point3 {
            x: 2.0,
            y: 3.0,
            z: -4.0,
        };
        let planar = [
            Point3 {
                x: -1.0,
                y: 0.0,
                z: 0.0,
            },
            Point3 {
                x: 5.0,
                y: 0.0,
                z: 0.0,
            },
            Point3 {
                x: 0.0,
                y: 7.0,
                z: 0.0,
            },
        ];
        let mut anchors = planar.to_vec();
        anchors.push(Point3 {
            x: 0.0,
            y: 0.0,
            z: 0.005,
        });
        let observations = predict_ranges(target, &anchors);
        let candidates = [
            CandidateSolution {
                solution_id: "target".to_string(),
                parameters: vec![target.x, target.y, target.z],
                predicted_observations: predict_ranges(target, &anchors),
            },
            CandidateSolution {
                solution_id: "mirror".to_string(),
                parameters: vec![mirror.x, mirror.y, mirror.z],
                predicted_observations: predict_ranges(mirror, &anchors),
            },
        ];

        let mut config = mirror_config(anchors.len());
        config.max_rms_standardized_residual = 1_000.0;
        config.max_abs_standardized_residual = 1_000.0;
        config.max_rms_prediction_difference = 1_000.0;
        config.max_abs_prediction_difference = 1_000.0;
        let report = screen_global_ambiguities(&observations, &candidates, &config).unwrap();
        assert_eq!(report.admissible_candidate_count, 2);
        assert_eq!(report.ambiguous_pairs.len(), 1);
        assert!(report.ambiguous_pairs[0].max_abs_standardized_prediction_difference > 0.0);
    }

    #[test]
    fn candidate_order_does_not_change_the_report() {
        let observations = [1.0, 2.0];
        let a = CandidateSolution {
            solution_id: "a".to_string(),
            parameters: vec![0.0, 0.0],
            predicted_observations: vec![1.0, 2.0],
        };
        let b = CandidateSolution {
            solution_id: "b".to_string(),
            parameters: vec![0.0, 5.0],
            predicted_observations: vec![1.0, 2.0],
        };
        let config = GlobalIdentifiabilityConfig {
            parameter_scales: vec![1.0, 1.0],
            observation_sigmas: vec![0.1, 0.1],
            max_rms_standardized_residual: 1.0,
            max_abs_standardized_residual: 1.0,
            max_rms_prediction_difference: 1.0e-6,
            max_abs_prediction_difference: 1.0e-6,
            min_scaled_parameter_separation: 1.0,
        };
        let left =
            screen_global_ambiguities(&observations, &[a.clone(), b.clone()], &config).unwrap();
        let right = screen_global_ambiguities(&observations, &[b, a], &config).unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn observation_order_is_invariant_when_all_measurement_vectors_are_permuted() {
        let observations = [1.0, 2.0, 3.0];
        let candidate = CandidateSolution {
            solution_id: "same".to_string(),
            parameters: vec![0.0, 1.0],
            predicted_observations: vec![1.01, 1.99, 3.0],
        };
        let config = GlobalIdentifiabilityConfig {
            parameter_scales: vec![1.0, 1.0],
            observation_sigmas: vec![0.1, 0.2, 0.3],
            max_rms_standardized_residual: 1.0,
            max_abs_standardized_residual: 1.0,
            max_rms_prediction_difference: 0.1,
            max_abs_prediction_difference: 0.1,
            min_scaled_parameter_separation: 1.0,
        };
        let original =
            screen_global_ambiguities(&observations, std::slice::from_ref(&candidate), &config)
                .unwrap();
        let permuted = screen_global_ambiguities(
            &[3.0, 1.0, 2.0],
            &[CandidateSolution {
                solution_id: "same".to_string(),
                parameters: candidate.parameters.clone(),
                predicted_observations: vec![3.0, 1.01, 1.99],
            }],
            &GlobalIdentifiabilityConfig {
                observation_sigmas: vec![0.3, 0.1, 0.2],
                ..config
            },
        )
        .unwrap();
        assert_eq!(original, permuted);
    }

    #[test]
    fn candidate_limit_is_enforced_before_pair_enumeration() {
        let candidate = CandidateSolution {
            solution_id: "candidate".to_string(),
            parameters: vec![0.0, 0.0, 0.0],
            predicted_observations: vec![1.0],
        };
        let candidates = vec![
            CandidateSolution {
                solution_id: "candidate-0".to_string(),
                ..candidate.clone()
            };
            MAX_GLOBAL_IDENTIFIABILITY_CANDIDATES + 1
        ];
        assert_eq!(
            screen_global_ambiguities(&[1.0], &candidates, &mirror_config(1)),
            Err(GlobalIdentifiabilityError::TooManyCandidates {
                count: MAX_GLOBAL_IDENTIFIABILITY_CANDIDATES + 1,
                maximum: MAX_GLOBAL_IDENTIFIABILITY_CANDIDATES,
            })
        );
    }

    #[test]
    fn malformed_inputs_fail_closed() {
        let config = mirror_config(1);
        assert_eq!(
            screen_global_ambiguities(&[], &[], &config),
            Err(GlobalIdentifiabilityError::NoObservations)
        );
        assert_eq!(
            screen_global_ambiguities(&[1.0], &[], &config),
            Err(GlobalIdentifiabilityError::NoCandidates)
        );
        let empty_id = CandidateSolution {
            solution_id: " ".to_string(),
            parameters: vec![1.0, 2.0, 3.0],
            predicted_observations: vec![1.0],
        };
        assert!(matches!(
            screen_global_ambiguities(&[1.0], &[empty_id], &config),
            Err(GlobalIdentifiabilityError::EmptySolutionId { .. })
        ));
        let malformed = CandidateSolution {
            solution_id: "bad".to_string(),
            parameters: vec![1.0, 2.0],
            predicted_observations: vec![1.0],
        };
        assert!(matches!(
            screen_global_ambiguities(&[1.0], &[malformed], &config),
            Err(GlobalIdentifiabilityError::ParameterCount { .. })
        ));
        let non_finite = CandidateSolution {
            solution_id: "nan".to_string(),
            parameters: vec![1.0, 2.0, 3.0],
            predicted_observations: vec![f64::NAN],
        };
        assert!(matches!(
            screen_global_ambiguities(&[1.0], &[non_finite], &config),
            Err(GlobalIdentifiabilityError::NonFinitePrediction { .. })
        ));
    }
}
