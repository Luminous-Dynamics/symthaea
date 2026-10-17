// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Spatial likelihood maps for owl-inspired auditory robotics.
//!
//! Given calibrated microphone positions and already-estimated pairwise TDOAs,
//! score a finite set of candidate source positions. This preserves multiple
//! plausible hypotheses instead of collapsing every observation to one bearing.
//!
//! Delay convention: for pair (i, j), delay_seconds means arrival_time(j) -
//! arrival_time(i). The geometric prediction is
//! (distance(source, j) - distance(source, i)) / speed_of_sound.
//!
//! The returned normalized weights are relative likelihood weights over the
//! supplied candidates under independent Gaussian timing-error assumptions and
//! an equal prior per candidate. They are NOT calibrated probabilities, and
//! irregularly spaced candidate grids do not represent a uniform spatial prior.
//! Pairwise TDOAs may be correlated, especially when they share microphones;
//! this first version does not model that covariance.
//!
//! This module does not estimate TDOA from audio waveforms, track moving sources,
//! model reflections, or prove uniqueness. A spatial grid is only as useful as
//! the microphone calibration, timing estimates, candidate coverage and noise
//! model supplied by the caller.

use core::fmt;

/// A point in three-dimensional Cartesian coordinates, in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Position3 {
    /// Construct a position in metres.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    fn distance_to(self, other: Self) -> f64 {
        (self.x - other.x)
            .hypot(self.y - other.y)
            .hypot(self.z - other.z)
    }
}

/// A measured time difference between two microphone channels.
///
/// The sign convention is arrival time at `second_microphone_index` minus
/// arrival time at `first_microphone_index`. `sigma_seconds` is the caller's
/// one-standard-deviation timing uncertainty for this measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TdoaMeasurement {
    pub first_microphone_index: usize,
    pub second_microphone_index: usize,
    pub delay_seconds: f64,
    pub sigma_seconds: f64,
}

/// One candidate source point and its relative likelihood weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandidateScore {
    pub position: Position3,
    /// Sum of Gaussian log-likelihood terms, omitting candidate-invariant constants.
    pub relative_log_likelihood: f64,
    /// Normalized weight over the candidates supplied to this call. This is
    /// not a calibrated probability and depends on candidate-grid sampling.
    pub normalized_weight: f64,
}

/// Validation and numerical failures for the sampled spatial-likelihood map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialLikelihoodError {
    NonFiniteInput,
    NonFiniteDerivedGeometry,
    EmptyMicrophoneArray,
    EmptyMeasurementSet,
    EmptyCandidateSet,
    NonPositiveSpeedOfSound,
    InvalidMicrophoneIndex,
    SameMicrophonePair,
    CoincidentMicrophones,
    NonPositiveTimingUncertainty,
    PhysicallyImpossibleDelay,
    NoFiniteCandidateLikelihood,
}

impl fmt::Display for SpatialLikelihoodError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteInput => write!(f, "all positions and measurements must be finite"),
            Self::NonFiniteDerivedGeometry => {
                write!(f, "microphone geometry produced a non-finite derived quantity")
            }
            Self::EmptyMicrophoneArray => write!(f, "microphone array must not be empty"),
            Self::EmptyMeasurementSet => write!(f, "at least one TDOA measurement is required"),
            Self::EmptyCandidateSet => write!(f, "at least one candidate position is required"),
            Self::NonPositiveSpeedOfSound => write!(f, "speed of sound must be positive"),
            Self::InvalidMicrophoneIndex => write!(f, "measurement references an unknown microphone"),
            Self::SameMicrophonePair => write!(f, "a TDOA measurement must reference two distinct microphones"),
            Self::CoincidentMicrophones => write!(f, "TDOA microphones must have nonzero separation"),
            Self::NonPositiveTimingUncertainty => write!(f, "timing uncertainty must be positive"),
            Self::PhysicallyImpossibleDelay => {
                write!(f, "measured delay exceeds the microphone-separation physical bound")
            }
            Self::NoFiniteCandidateLikelihood => {
                write!(f, "all candidate likelihoods underflowed or became non-finite")
            }
        }
    }
}

impl std::error::Error for SpatialLikelihoodError {}

/// Score a set of candidate source positions against pairwise TDOA measurements.
///
/// The candidate list is treated as a discrete hypothesis set with equal prior
/// weight per candidate. Each supplied TDOA contributes an independent Gaussian
/// timing-error term with the provided standard deviation. The function
/// normalizes these relative likelihoods using a max-subtracted exponential to
/// reduce underflow. It does not correct for correlated TDOAs, reflections,
/// source motion, or systematic channel delay.
///
/// Impossible timing observations are rejected using the triangle-inequality
/// bound |delay| <= microphone_separation / speed_of_sound; observations are
/// never silently clamped to a physically possible value.
pub fn score_spatial_candidates(
    microphones: &[Position3],
    measurements: &[TdoaMeasurement],
    candidates: &[Position3],
    speed_of_sound_m_s: f64,
) -> Result<Vec<CandidateScore>, SpatialLikelihoodError> {
    if microphones.is_empty() {
        return Err(SpatialLikelihoodError::EmptyMicrophoneArray);
    }
    if measurements.is_empty() {
        return Err(SpatialLikelihoodError::EmptyMeasurementSet);
    }
    if candidates.is_empty() {
        return Err(SpatialLikelihoodError::EmptyCandidateSet);
    }
    if !speed_of_sound_m_s.is_finite() || speed_of_sound_m_s <= 0.0 {
        return Err(if speed_of_sound_m_s.is_finite() {
            SpatialLikelihoodError::NonPositiveSpeedOfSound
        } else {
            SpatialLikelihoodError::NonFiniteInput
        });
    }
    if microphones.iter().any(|position| !position.is_finite())
        || candidates.iter().any(|position| !position.is_finite())
    {
        return Err(SpatialLikelihoodError::NonFiniteInput);
    }

    // Validate the entire observation set before scoring any candidate.
    for measurement in measurements {
        if !measurement.delay_seconds.is_finite() || !measurement.sigma_seconds.is_finite() {
            return Err(SpatialLikelihoodError::NonFiniteInput);
        }
        if measurement.sigma_seconds <= 0.0 {
            return Err(SpatialLikelihoodError::NonPositiveTimingUncertainty);
        }
        if measurement.first_microphone_index >= microphones.len()
            || measurement.second_microphone_index >= microphones.len()
        {
            return Err(SpatialLikelihoodError::InvalidMicrophoneIndex);
        }
        if measurement.first_microphone_index == measurement.second_microphone_index {
            return Err(SpatialLikelihoodError::SameMicrophonePair);
        }

        let first = microphones[measurement.first_microphone_index];
        let second = microphones[measurement.second_microphone_index];
        let baseline = first.distance_to(second);
        if !baseline.is_finite() {
            return Err(SpatialLikelihoodError::NonFiniteDerivedGeometry);
        }
        if baseline == 0.0 {
            return Err(SpatialLikelihoodError::CoincidentMicrophones);
        }

        let maximum_delay = baseline / speed_of_sound_m_s;
        if !maximum_delay.is_finite() || maximum_delay <= 0.0 {
            return Err(SpatialLikelihoodError::NonFiniteDerivedGeometry);
        }
        // Allow a tiny floating-point margin, not a material physical violation.
        let tolerance = maximum_delay * 1.0e-12;
        let measured_magnitude = measurement.delay_seconds.abs();
        if measured_magnitude > maximum_delay
            && measured_magnitude - maximum_delay > tolerance
        {
            return Err(SpatialLikelihoodError::PhysicallyImpossibleDelay);
        }
    }

    let mut log_likelihoods = Vec::with_capacity(candidates.len());
    for &candidate in candidates {
        let mut log_likelihood = 0.0;
        for measurement in measurements {
            let first = microphones[measurement.first_microphone_index];
            let second = microphones[measurement.second_microphone_index];
            let distance_first = candidate.distance_to(first);
            let distance_second = candidate.distance_to(second);
            if !distance_first.is_finite() || !distance_second.is_finite() {
                return Err(SpatialLikelihoodError::NonFiniteDerivedGeometry);
            }

            let predicted_delay = (distance_second - distance_first) / speed_of_sound_m_s;
            if !predicted_delay.is_finite() {
                return Err(SpatialLikelihoodError::NonFiniteDerivedGeometry);
            }
            let standardized_residual =
                (predicted_delay - measurement.delay_seconds) / measurement.sigma_seconds;
            if !standardized_residual.is_finite() {
                log_likelihood = f64::NEG_INFINITY;
                break;
            }
            let penalty = 0.5 * standardized_residual * standardized_residual;
            if !penalty.is_finite() {
                log_likelihood = f64::NEG_INFINITY;
                break;
            }
            log_likelihood -= penalty;
            if !log_likelihood.is_finite() {
                log_likelihood = f64::NEG_INFINITY;
                break;
            }
        }
        log_likelihoods.push(log_likelihood);
    }

    let maximum_log_likelihood = log_likelihoods
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum_log_likelihood.is_finite() {
        return Err(SpatialLikelihoodError::NoFiniteCandidateLikelihood);
    }

    let unnormalized_weights: Vec<f64> = log_likelihoods
        .iter()
        .map(|value| (value - maximum_log_likelihood).exp())
        .collect();
    let weight_sum: f64 = unnormalized_weights.iter().sum();
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(SpatialLikelihoodError::NoFiniteCandidateLikelihood);
    }

    Ok(candidates
        .iter()
        .copied()
        .zip(log_likelihoods)
        .zip(unnormalized_weights)
        .map(|((position, relative_log_likelihood), weight)| CandidateScore {
            position,
            relative_log_likelihood,
            normalized_weight: weight / weight_sum,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const C: f64 = 343.0;
    const SIGMA: f64 = 1.0e-5;

    fn delay(source: Position3, first: Position3, second: Position3) -> f64 {
        (source.distance_to(second) - source.distance_to(first)) / C
    }

    fn measurement(
        first_index: usize,
        second_index: usize,
        source: Position3,
        microphones: &[Position3],
    ) -> TdoaMeasurement {
        TdoaMeasurement {
            first_microphone_index: first_index,
            second_microphone_index: second_index,
            delay_seconds: delay(
                source,
                microphones[first_index],
                microphones[second_index],
            ),
            sigma_seconds: SIGMA,
        }
    }

    fn tetrahedral_array() -> [Position3; 4] {
        [
            Position3::new(0.0, 0.0, 0.0),
            Position3::new(0.10, 0.0, 0.0),
            Position3::new(0.0, 0.12, 0.0),
            Position3::new(0.0, 0.0, 0.14),
        ]
    }

    #[test]
    fn exact_synthetic_source_has_highest_weight() {
        let microphones = tetrahedral_array();
        let source = Position3::new(0.35, 0.23, 0.55);
        let measurements = [
            measurement(0, 1, source, &microphones),
            measurement(0, 2, source, &microphones),
            measurement(0, 3, source, &microphones),
        ];
        let candidates = [
            Position3::new(0.20, 0.20, 0.40),
            source,
            Position3::new(0.40, 0.15, 0.25),
        ];

        let scores =
            score_spatial_candidates(&microphones, &measurements, &candidates, C).unwrap();
        let best = scores
            .iter()
            .max_by(|left, right| left.normalized_weight.total_cmp(&right.normalized_weight))
            .unwrap();

        assert_eq!(best.position, source);
        assert!(best.relative_log_likelihood.abs() < 1e-12);
    }

    #[test]
    fn normalized_weights_sum_to_one() {
        let microphones = tetrahedral_array();
        let source = Position3::new(0.35, 0.23, 0.55);
        let measurements = [
            measurement(0, 1, source, &microphones),
            measurement(0, 2, source, &microphones),
            measurement(0, 3, source, &microphones),
        ];
        let candidates = [
            Position3::new(0.20, 0.20, 0.40),
            source,
            Position3::new(0.40, 0.15, 0.25),
        ];

        let scores =
            score_spatial_candidates(&microphones, &measurements, &candidates, C).unwrap();
        let sum: f64 = scores.iter().map(|score| score.normalized_weight).sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn planar_array_preserves_above_below_mirror_ambiguity() {
        let microphones = [
            Position3::new(-0.1, -0.1, 0.0),
            Position3::new(0.1, -0.1, 0.0),
            Position3::new(-0.1, 0.1, 0.0),
            Position3::new(0.1, 0.1, 0.0),
        ];
        let source_above = Position3::new(0.25, 0.30, 0.40);
        let source_below = Position3::new(0.25, 0.30, -0.40);
        let measurements = [
            measurement(0, 1, source_above, &microphones),
            measurement(0, 2, source_above, &microphones),
            measurement(0, 3, source_above, &microphones),
        ];
        let candidates = [source_above, source_below];

        let scores =
            score_spatial_candidates(&microphones, &measurements, &candidates, C).unwrap();
        assert!(
            (scores[0].normalized_weight - scores[1].normalized_weight).abs() < 1e-10,
            "mirror hypotheses should remain ambiguous: {scores:?}"
        );
    }

    #[test]
    fn impossible_delay_is_rejected_instead_of_clamped() {
        let microphones = [
            Position3::new(0.0, 0.0, 0.0),
            Position3::new(0.1, 0.0, 0.0),
        ];
        let observation = TdoaMeasurement {
            first_microphone_index: 0,
            second_microphone_index: 1,
            delay_seconds: 0.001,
            sigma_seconds: SIGMA,
        };
        assert_eq!(
            score_spatial_candidates(
                &microphones,
                &[observation],
                &[Position3::new(1.0, 1.0, 1.0)],
                C
            ),
            Err(SpatialLikelihoodError::PhysicallyImpossibleDelay)
        );
    }

    #[test]
    fn unknown_microphone_index_is_rejected() {
        let microphones = tetrahedral_array();
        let observation = TdoaMeasurement {
            first_microphone_index: 0,
            second_microphone_index: 4,
            delay_seconds: 0.0,
            sigma_seconds: SIGMA,
        };
        assert_eq!(
            score_spatial_candidates(
                &microphones,
                &[observation],
                &[Position3::new(1.0, 1.0, 1.0)],
                C
            ),
            Err(SpatialLikelihoodError::InvalidMicrophoneIndex)
        );
    }

    #[test]
    fn identical_microphone_indices_are_rejected() {
        let microphones = tetrahedral_array();
        let observation = TdoaMeasurement {
            first_microphone_index: 1,
            second_microphone_index: 1,
            delay_seconds: 0.0,
            sigma_seconds: SIGMA,
        };
        assert_eq!(
            score_spatial_candidates(
                &microphones,
                &[observation],
                &[Position3::new(1.0, 1.0, 1.0)],
                C
            ),
            Err(SpatialLikelihoodError::SameMicrophonePair)
        );
    }

    #[test]
    fn coincident_microphones_are_rejected() {
        let microphones = [
            Position3::new(0.0, 0.0, 0.0),
            Position3::new(0.0, 0.0, 0.0),
        ];
        let observation = TdoaMeasurement {
            first_microphone_index: 0,
            second_microphone_index: 1,
            delay_seconds: 0.0,
            sigma_seconds: SIGMA,
        };
        assert_eq!(
            score_spatial_candidates(
                &microphones,
                &[observation],
                &[Position3::new(1.0, 1.0, 1.0)],
                C
            ),
            Err(SpatialLikelihoodError::CoincidentMicrophones)
        );
    }

    #[test]
    fn non_positive_timing_uncertainty_is_rejected() {
        let microphones = tetrahedral_array();
        let mut observation = measurement(0, 1, Position3::new(0.35, 0.23, 0.55), &microphones);
        observation.sigma_seconds = 0.0;
        assert_eq!(
            score_spatial_candidates(
                &microphones,
                &[observation],
                &[Position3::new(1.0, 1.0, 1.0)],
                C
            ),
            Err(SpatialLikelihoodError::NonPositiveTimingUncertainty)
        );
    }

    #[test]
    fn non_finite_coordinates_are_rejected() {
        let microphones = [
            Position3::new(0.0, 0.0, 0.0),
            Position3::new(0.1, 0.0, 0.0),
        ];
        let observation = TdoaMeasurement {
            first_microphone_index: 0,
            second_microphone_index: 1,
            delay_seconds: 0.0,
            sigma_seconds: SIGMA,
        };
        assert_eq!(
            score_spatial_candidates(
                &microphones,
                &[observation],
                &[Position3::new(f64::NAN, 1.0, 1.0)],
                C
            ),
            Err(SpatialLikelihoodError::NonFiniteInput)
        );
    }

    #[test]
    fn empty_candidate_set_is_rejected() {
        let microphones = tetrahedral_array();
        let source = Position3::new(0.35, 0.23, 0.55);
        let observation = measurement(0, 1, source, &microphones);
        assert_eq!(
            score_spatial_candidates(&microphones, &[observation], &[], C),
            Err(SpatialLikelihoodError::EmptyCandidateSet)
        );
    }

    #[test]
    fn empty_measurement_set_is_rejected() {
        assert_eq!(
            score_spatial_candidates(
                &tetrahedral_array(),
                &[],
                &[Position3::new(0.2, 0.3, 0.4)],
                C
            ),
            Err(SpatialLikelihoodError::EmptyMeasurementSet)
        );
    }
}
