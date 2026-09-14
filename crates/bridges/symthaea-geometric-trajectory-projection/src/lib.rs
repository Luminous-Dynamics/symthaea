// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dual-projection trajectory adapter for GEOM-001C.
//!
//! Every native 32D Symthaea thought-vector sample is projected through BOTH
//! preregistered GEOM-001B mappings, then measured by independent Fisher-Rao
//! observatories. The two reports are always returned side-by-side; this module
//! deliberately defines no combined score and no robustness verdict.

use std::fmt;

use symthaea_frontier_physics::geometric_emergence::{
    GeometricEmergenceError, GeometricEmergenceObservatory, TrajectoryMetrics,
};
use symthaea_geometric_state_projection::{StateProjectionError, project_thought_vector};

/// The two mandatory geometric reports for one native cognitive trajectory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualProjectionTrajectoryMetrics {
    pub primary_l1: TrajectoryMetrics,
    pub sensitivity_energy: TrajectoryMetrics,
}

/// Fail-closed errors for native cognitive trajectory analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitiveTrajectoryError {
    /// One native sample failed GEOM-001B projection. `sample_index` is exact.
    Projection {
        sample_index: usize,
        source: StateProjectionError,
    },
    /// The primary L1 trajectory failed geometric measurement.
    PrimaryGeometry(GeometricEmergenceError),
    /// The mandatory squared-energy trajectory failed geometric measurement.
    SensitivityGeometry(GeometricEmergenceError),
}

impl fmt::Display for CognitiveTrajectoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Projection {
                sample_index,
                source,
            } => write!(f, "native sample {sample_index} failed state projection: {source}"),
            Self::PrimaryGeometry(source) => {
                write!(f, "primary L1 trajectory geometry failed: {source}")
            }
            Self::SensitivityGeometry(source) => {
                write!(f, "squared-energy trajectory geometry failed: {source}")
            }
        }
    }
}

impl std::error::Error for CognitiveTrajectoryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Projection { source, .. } => Some(source),
            Self::PrimaryGeometry(source) | Self::SensitivityGeometry(source) => Some(source),
        }
    }
}

/// Analyze a sequence of native Symthaea thought vectors under both
/// preregistered GEOM-001B projections.
///
/// No projection may be selected post hoc. A successful result always contains
/// both Fisher-Rao metric families.
pub fn analyze_native_thought_trajectory(
    samples: &[Vec<f32>],
) -> Result<DualProjectionTrajectoryMetrics, CognitiveTrajectoryError> {
    let mut primary = GeometricEmergenceObservatory::new();
    let mut sensitivity = GeometricEmergenceObservatory::new();

    for (sample_index, sample) in samples.iter().enumerate() {
        let projected = project_thought_vector(sample).map_err(|source| {
            CognitiveTrajectoryError::Projection {
                sample_index,
                source,
            }
        })?;

        primary
            .push_distribution(&projected.primary_l1)
            .map_err(CognitiveTrajectoryError::PrimaryGeometry)?;
        sensitivity
            .push_distribution(&projected.sensitivity_energy)
            .map_err(CognitiveTrajectoryError::SensitivityGeometry)?;
    }

    let primary_l1 = primary
        .metrics()
        .map_err(CognitiveTrajectoryError::PrimaryGeometry)?;
    let sensitivity_energy = sensitivity
        .metrics()
        .map_err(CognitiveTrajectoryError::SensitivityGeometry)?;

    Ok(DualProjectionTrajectoryMetrics {
        primary_l1,
        sensitivity_energy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_geometric_state_projection::{StateProjectionError, THOUGHT_VECTOR_DIMENSIONS};

    const TOLERANCE: f64 = 1.0e-9;

    fn sample(offset: f32) -> Vec<f32> {
        (0..THOUGHT_VECTOR_DIMENSIONS)
            .map(|index| (index as f32 + 1.0 + offset) * if index % 3 == 0 { -1.0 } else { 1.0 })
            .collect()
    }

    fn assert_metric_close(left: &TrajectoryMetrics, right: &TrajectoryMetrics) {
        assert_eq!(left.samples, right.samples);
        assert_eq!(left.dimensions, right.dimensions);
        for (a, b) in [
            (left.path_length, right.path_length),
            (left.endpoint_displacement, right.endpoint_displacement),
            (left.geodesic_efficiency, right.geodesic_efficiency),
            (left.excess_path_length, right.excess_path_length),
            (left.mean_step_length, right.mean_step_length),
            (left.step_length_variance, right.step_length_variance),
            (left.max_step_length, right.max_step_length),
        ] {
            assert!((a - b).abs() < TOLERANCE, "{a} != {b}");
        }
    }

    fn assert_dual_close(
        left: &DualProjectionTrajectoryMetrics,
        right: &DualProjectionTrajectoryMetrics,
    ) {
        assert_metric_close(&left.primary_l1, &right.primary_l1);
        assert_metric_close(&left.sensitivity_energy, &right.sensitivity_energy);
    }

    #[test]
    fn stationary_native_trajectory_is_stationary_under_both_projections() {
        let state = sample(0.0);
        let samples = vec![state.clone(), state.clone(), state];
        let metrics = analyze_native_thought_trajectory(&samples).expect("valid trajectory");

        for family in [metrics.primary_l1, metrics.sensitivity_energy] {
            assert!(family.path_length.abs() < TOLERANCE);
            assert!(family.endpoint_displacement.abs() < TOLERANCE);
            assert!((family.geodesic_efficiency - 1.0).abs() < TOLERANCE);
            assert!(family.excess_path_length.abs() < TOLERANCE);
        }
    }

    #[test]
    fn independent_global_scale_and_sign_per_sample_leave_geometry_unchanged() {
        let samples = vec![sample(0.0), sample(3.0), sample(7.0), sample(11.0)];
        let transformed = vec![
            samples[0].iter().map(|value| -2.0 * value).collect(),
            samples[1].iter().map(|value| 4.0 * value).collect(),
            samples[2].iter().map(|value| -0.5 * value).collect(),
            samples[3].iter().map(|value| 8.0 * value).collect(),
        ];

        let original = analyze_native_thought_trajectory(&samples).expect("valid trajectory");
        let transformed =
            analyze_native_thought_trajectory(&transformed).expect("valid transformed trajectory");
        assert_dual_close(&original, &transformed);
    }

    #[test]
    fn fixed_coordinate_permutation_preserves_both_geometries() {
        let samples = vec![sample(0.0), sample(2.0), sample(5.0), sample(9.0)];
        let permuted: Vec<Vec<f32>> = samples
            .iter()
            .map(|state| {
                let mut state = state.clone();
                state.rotate_left(7);
                state.swap(2, 23);
                state
            })
            .collect();

        let original = analyze_native_thought_trajectory(&samples).expect("valid trajectory");
        let permuted =
            analyze_native_thought_trajectory(&permuted).expect("valid permuted trajectory");
        assert_dual_close(&original, &permuted);
    }

    #[test]
    fn time_reversal_preserves_descriptive_trajectory_geometry() {
        let samples = vec![sample(0.0), sample(1.0), sample(4.0), sample(10.0)];
        let reversed: Vec<Vec<f32>> = samples.iter().cloned().rev().collect();

        let forward = analyze_native_thought_trajectory(&samples).expect("valid trajectory");
        let backward = analyze_native_thought_trajectory(&reversed).expect("valid reverse trajectory");
        assert_dual_close(&forward, &backward);
    }

    #[test]
    fn malformed_sample_reports_exact_index() {
        let mut malformed = vec![1.0_f32; THOUGHT_VECTOR_DIMENSIONS];
        malformed[4] = f32::NAN;
        let samples = vec![sample(0.0), sample(1.0), malformed, sample(3.0)];

        assert_eq!(
            analyze_native_thought_trajectory(&samples),
            Err(CognitiveTrajectoryError::Projection {
                sample_index: 2,
                source: StateProjectionError::NonFiniteValue { index: 4 },
            })
        );
    }

    #[test]
    fn one_sample_fails_as_insufficient_geometry_not_as_projection() {
        let samples = vec![sample(0.0)];
        assert_eq!(
            analyze_native_thought_trajectory(&samples),
            Err(CognitiveTrajectoryError::PrimaryGeometry(
                GeometricEmergenceError::InsufficientSamples {
                    required: 2,
                    observed: 1,
                }
            ))
        );
    }

    #[test]
    fn primary_and_sensitivity_metrics_are_both_present_and_can_differ() {
        let mut first = vec![1.0_f32; THOUGHT_VECTOR_DIMENSIONS];
        first[0] = 8.0;
        let mut second = vec![1.0_f32; THOUGHT_VECTOR_DIMENSIONS];
        second[1] = 8.0;
        let metrics = analyze_native_thought_trajectory(&[first, second]).expect("valid trajectory");

        assert!(metrics.primary_l1.path_length > 0.0);
        assert!(metrics.sensitivity_energy.path_length > 0.0);
        assert!(
            (metrics.primary_l1.path_length - metrics.sensitivity_energy.path_length).abs()
                > TOLERANCE
        );
    }
}
