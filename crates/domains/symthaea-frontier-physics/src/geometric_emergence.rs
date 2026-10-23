// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Theory-neutral observatory for trajectories through probability space.
//!
//! GEOM-001 deliberately measures geometry without interpreting it as
//! consciousness, agency, gravity, or any other ontology.  The observables in
//! this module are suitable for intact/lesion/null comparisons and for systems
//! outside cognition entirely.

use super::information_geometry::fisher_rao_distance;
use serde::{Deserialize, Serialize};
use std::fmt;

const EPSILON: f64 = 1.0e-12;

/// Fail-closed validation errors for geometric measurements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeometricEmergenceError {
    EmptyDistribution,
    NonFiniteProbability,
    NegativeProbability,
    ZeroProbabilityMass,
    DimensionMismatch { expected: usize, observed: usize },
    InsufficientSamples { required: usize, observed: usize },
}

impl fmt::Display for GeometricEmergenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDistribution => write!(f, "distribution must contain at least one state"),
            Self::NonFiniteProbability => write!(f, "probabilities must be finite"),
            Self::NegativeProbability => write!(f, "probabilities must be non-negative"),
            Self::ZeroProbabilityMass => write!(f, "distribution must have positive total mass"),
            Self::DimensionMismatch { expected, observed } => write!(
                f,
                "distribution dimension mismatch: expected {expected}, observed {observed}"
            ),
            Self::InsufficientSamples { required, observed } => write!(
                f,
                "insufficient trajectory samples: required {required}, observed {observed}"
            ),
        }
    }
}

impl std::error::Error for GeometricEmergenceError {}

/// Normalize an arbitrary non-negative finite weight vector to unit mass.
///
/// Validation is intentionally strict so malformed states cannot silently
/// become geometric evidence.
pub fn normalize_distribution(
    distribution: &[f64],
) -> Result<Vec<f64>, GeometricEmergenceError> {
    if distribution.is_empty() {
        return Err(GeometricEmergenceError::EmptyDistribution);
    }

    if distribution.iter().any(|value| !value.is_finite()) {
        return Err(GeometricEmergenceError::NonFiniteProbability);
    }

    if distribution.iter().any(|&value| value < 0.0) {
        return Err(GeometricEmergenceError::NegativeProbability);
    }

    let total: f64 = distribution.iter().sum();
    if !total.is_finite() || total <= EPSILON {
        return Err(GeometricEmergenceError::ZeroProbabilityMass);
    }

    Ok(distribution.iter().map(|value| value / total).collect())
}

/// Descriptive geometry of a trajectory on the probability simplex.
///
/// None of these fields is a consciousness metric.  They describe how a
/// normalized probability state moves under the Fisher-Rao metric.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryMetrics {
    /// Number of recorded probability states.
    pub samples: usize,
    /// Dimensionality of each probability state.
    pub dimensions: usize,
    /// Sum of Fisher-Rao distances between consecutive samples.
    pub path_length: f64,
    /// Fisher-Rao distance between the first and final samples.
    pub endpoint_displacement: f64,
    /// endpoint_displacement / path_length, in [0, 1].
    ///
    /// A stationary trajectory is assigned 1.0 because it has no unnecessary
    /// geometric travel.  A closed non-stationary loop approaches 0.0.
    pub geodesic_efficiency: f64,
    /// path_length - endpoint_displacement.
    pub excess_path_length: f64,
    /// Mean Fisher-Rao step length.
    pub mean_step_length: f64,
    /// Population variance of Fisher-Rao step lengths.
    pub step_length_variance: f64,
    /// Largest Fisher-Rao step length.
    pub max_step_length: f64,
}

/// Accumulates a validated trajectory and computes theory-neutral geometry.
#[derive(Debug, Clone, Default)]
pub struct GeometricEmergenceObservatory {
    dimensions: Option<usize>,
    states: Vec<Vec<f64>>,
}

impl GeometricEmergenceObservatory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn dimensions(&self) -> Option<usize> {
        self.dimensions
    }

    pub fn clear(&mut self) {
        self.states.clear();
        self.dimensions = None;
    }

    /// Add one state to the trajectory after validation and normalization.
    pub fn push_distribution(
        &mut self,
        distribution: &[f64],
    ) -> Result<(), GeometricEmergenceError> {
        let normalized = normalize_distribution(distribution)?;
        let observed = normalized.len();

        if let Some(expected) = self.dimensions {
            if expected != observed {
                return Err(GeometricEmergenceError::DimensionMismatch { expected, observed });
            }
        } else {
            self.dimensions = Some(observed);
        }

        self.states.push(normalized);
        Ok(())
    }

    /// Compute descriptive trajectory geometry.
    pub fn metrics(&self) -> Result<TrajectoryMetrics, GeometricEmergenceError> {
        if self.states.len() < 2 {
            return Err(GeometricEmergenceError::InsufficientSamples {
                required: 2,
                observed: self.states.len(),
            });
        }

        let step_lengths: Vec<f64> = self
            .states
            .windows(2)
            .map(|pair| fisher_rao_distance(&pair[0], &pair[1]))
            .collect();

        let path_length: f64 = step_lengths.iter().sum();
        let endpoint_displacement = fisher_rao_distance(
            self.states.first().expect("length checked above"),
            self.states.last().expect("length checked above"),
        );

        let geodesic_efficiency = if path_length <= EPSILON {
            1.0
        } else {
            (endpoint_displacement / path_length).clamp(0.0, 1.0)
        };

        let mean_step_length = path_length / step_lengths.len() as f64;
        let step_length_variance = step_lengths
            .iter()
            .map(|step| (step - mean_step_length).powi(2))
            .sum::<f64>()
            / step_lengths.len() as f64;
        let max_step_length = step_lengths.iter().copied().fold(0.0_f64, f64::max);

        Ok(TrajectoryMetrics {
            samples: self.states.len(),
            dimensions: self.dimensions.expect("states imply dimensions"),
            path_length,
            endpoint_displacement,
            geodesic_efficiency,
            excess_path_length: (path_length - endpoint_displacement).max(0.0),
            mean_step_length,
            step_length_variance,
            max_step_length,
        })
    }
}

/// Fisher-Rao response magnitude between a reference and perturbed state.
pub fn perturbation_distance(
    reference: &[f64],
    perturbed: &[f64],
) -> Result<f64, GeometricEmergenceError> {
    let reference = normalize_distribution(reference)?;
    let perturbed = normalize_distribution(perturbed)?;

    if reference.len() != perturbed.len() {
        return Err(GeometricEmergenceError::DimensionMismatch {
            expected: reference.len(),
            observed: perturbed.len(),
        });
    }

    Ok(fisher_rao_distance(&reference, &perturbed))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1.0e-10;

    #[test]
    fn normalizes_weights_without_changing_ratios() {
        let normalized = normalize_distribution(&[2.0, 6.0]).expect("valid distribution");
        assert!((normalized[0] - 0.25).abs() < TOLERANCE);
        assert!((normalized[1] - 0.75).abs() < TOLERANCE);
    }

    #[test]
    fn rejects_invalid_probability_states() {
        assert_eq!(
            normalize_distribution(&[]),
            Err(GeometricEmergenceError::EmptyDistribution)
        );
        assert_eq!(
            normalize_distribution(&[0.0, 0.0]),
            Err(GeometricEmergenceError::ZeroProbabilityMass)
        );
        assert_eq!(
            normalize_distribution(&[0.5, -0.5]),
            Err(GeometricEmergenceError::NegativeProbability)
        );
        assert_eq!(
            normalize_distribution(&[0.5, f64::NAN]),
            Err(GeometricEmergenceError::NonFiniteProbability)
        );
    }

    #[test]
    fn stationary_trajectory_has_zero_length_and_unit_efficiency() {
        let mut observatory = GeometricEmergenceObservatory::new();
        observatory
            .push_distribution(&[0.25, 0.75])
            .expect("valid sample");
        observatory
            .push_distribution(&[0.25, 0.75])
            .expect("valid sample");

        let metrics = observatory.metrics().expect("two samples");
        assert!(metrics.path_length.abs() < TOLERANCE);
        assert!(metrics.endpoint_displacement.abs() < TOLERANCE);
        assert!((metrics.geodesic_efficiency - 1.0).abs() < TOLERANCE);
        assert!(metrics.excess_path_length.abs() < TOLERANCE);
    }

    #[test]
    fn monotonic_binary_path_is_geodesically_efficient() {
        let mut observatory = GeometricEmergenceObservatory::new();
        for state in [
            [0.9, 0.1],
            [0.7, 0.3],
            [0.5, 0.5],
            [0.3, 0.7],
            [0.1, 0.9],
        ] {
            observatory
                .push_distribution(&state)
                .expect("valid sample");
        }

        let metrics = observatory.metrics().expect("trajectory is valid");
        assert!(metrics.geodesic_efficiency > 1.0 - TOLERANCE);
        assert!(metrics.excess_path_length < TOLERANCE);
    }

    #[test]
    fn detour_increases_path_without_changing_endpoints() {
        let mut direct = GeometricEmergenceObservatory::new();
        for state in [[0.9, 0.1], [0.1, 0.9]] {
            direct.push_distribution(&state).expect("valid sample");
        }

        let mut detour = GeometricEmergenceObservatory::new();
        for state in [[0.9, 0.1], [0.5, 0.5], [0.9, 0.1], [0.1, 0.9]] {
            detour.push_distribution(&state).expect("valid sample");
        }

        let direct_metrics = direct.metrics().expect("trajectory is valid");
        let detour_metrics = detour.metrics().expect("trajectory is valid");

        assert!(
            (direct_metrics.endpoint_displacement - detour_metrics.endpoint_displacement).abs()
                < TOLERANCE
        );
        assert!(detour_metrics.path_length > direct_metrics.path_length);
        assert!(detour_metrics.geodesic_efficiency < direct_metrics.geodesic_efficiency);
        assert!(detour_metrics.excess_path_length > TOLERANCE);
    }

    #[test]
    fn perturbation_response_is_zero_for_self_and_positive_for_change() {
        let state = [0.8, 0.2];
        let same = perturbation_distance(&state, &state).expect("valid states");
        let changed = perturbation_distance(&state, &[0.2, 0.8]).expect("valid states");

        assert!(same.abs() < TOLERANCE);
        assert!(changed > 0.0);
    }

    #[test]
    fn dimension_mismatch_fails_closed() {
        let mut observatory = GeometricEmergenceObservatory::new();
        observatory
            .push_distribution(&[0.5, 0.5])
            .expect("valid first sample");

        assert_eq!(
            observatory.push_distribution(&[0.25, 0.25, 0.5]),
            Err(GeometricEmergenceError::DimensionMismatch {
                expected: 2,
                observed: 3,
            })
        );
    }
}
