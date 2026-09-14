// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Preregistered cognitive-state projection for GEOM-001B.
//!
//! Symthaea's `CycleResult.thought_vector` is a native 32-dimensional
//! projection of the cycle HDV. This crate freezes two parameter-free mappings
//! from that signed vector onto simplex coordinates before lesion outcomes are
//! inspected.
//!
//! The resulting vectors are measurement coordinates for information geometry.
//! They are not calibrated probabilities of consciousness, beliefs, or world
//! states.

use std::fmt;

/// Current native dimensionality of `CycleResult.thought_vector`.
pub const THOUGHT_VECTOR_DIMENSIONS: usize = 32;

/// Projection failures that invalidate a GEOM observation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateProjectionError {
    DimensionMismatch { expected: usize, observed: usize },
    NonFiniteValue { index: usize },
    ZeroProjectionMass,
}

impl fmt::Display for StateProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, observed } => write!(
                f,
                "thought vector dimension mismatch: expected {expected}, observed {observed}"
            ),
            Self::NonFiniteValue { index } => {
                write!(f, "thought vector value at index {index} must be finite")
            }
            Self::ZeroProjectionMass => write!(
                f,
                "thought vector must contain non-zero magnitude for simplex projection"
            ),
        }
    }
}

impl std::error::Error for StateProjectionError {}

/// Both preregistered simplex projections for one native cognitive state.
///
/// `primary_l1` is the primary GEOM representation. `sensitivity_energy` is
/// mandatory sensitivity analysis and must not be discarded post hoc when it
/// disagrees with the primary representation.
#[derive(Debug, Clone, PartialEq)]
pub struct CognitiveStateProjection {
    pub primary_l1: Vec<f64>,
    pub sensitivity_energy: Vec<f64>,
}

/// Project a native 32D signed thought vector onto two parameter-free simplex
/// coordinate systems.
///
/// Primary L1-magnitude projection:
/// `p_i = |x_i| / sum_j |x_j|`.
///
/// Mandatory squared-energy sensitivity projection:
/// `q_i = x_i^2 / sum_j x_j^2`.
///
/// No padding, truncation, learned codebook, softmax temperature, clipping, or
/// data-dependent hyperparameter is permitted here.
pub fn project_thought_vector(
    thought_vector: &[f32],
) -> Result<CognitiveStateProjection, StateProjectionError> {
    if thought_vector.len() != THOUGHT_VECTOR_DIMENSIONS {
        return Err(StateProjectionError::DimensionMismatch {
            expected: THOUGHT_VECTOR_DIMENSIONS,
            observed: thought_vector.len(),
        });
    }

    let mut magnitudes = Vec::with_capacity(THOUGHT_VECTOR_DIMENSIONS);
    let mut energies = Vec::with_capacity(THOUGHT_VECTOR_DIMENSIONS);
    let mut magnitude_sum = 0.0_f64;
    let mut energy_sum = 0.0_f64;

    for (index, &value) in thought_vector.iter().enumerate() {
        if !value.is_finite() {
            return Err(StateProjectionError::NonFiniteValue { index });
        }
        let value = f64::from(value);
        let magnitude = value.abs();
        let energy = value * value;
        magnitudes.push(magnitude);
        energies.push(energy);
        magnitude_sum += magnitude;
        energy_sum += energy;
    }

    if magnitude_sum == 0.0 || energy_sum == 0.0 {
        return Err(StateProjectionError::ZeroProjectionMass);
    }

    let primary_l1 = magnitudes
        .into_iter()
        .map(|value| value / magnitude_sum)
        .collect();
    let sensitivity_energy = energies
        .into_iter()
        .map(|value| value / energy_sum)
        .collect();

    Ok(CognitiveStateProjection {
        primary_l1,
        sensitivity_energy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1.0e-10;

    fn assert_close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (left, right) in a.iter().zip(b.iter()) {
            assert!((left - right).abs() < TOLERANCE, "{left} != {right}");
        }
    }

    #[test]
    fn one_hot_state_projects_to_same_simplex_vertex() {
        let mut vector = vec![0.0_f32; THOUGHT_VECTOR_DIMENSIONS];
        vector[7] = -4.0;
        let projected = project_thought_vector(&vector).expect("valid state");

        for index in 0..THOUGHT_VECTOR_DIMENSIONS {
            let expected = if index == 7 { 1.0 } else { 0.0 };
            assert!((projected.primary_l1[index] - expected).abs() < TOLERANCE);
            assert!((projected.sensitivity_energy[index] - expected).abs() < TOLERANCE);
        }
    }

    #[test]
    fn equal_magnitudes_project_uniformly_despite_sign() {
        let vector: Vec<f32> = (0..THOUGHT_VECTOR_DIMENSIONS)
            .map(|index| if index % 2 == 0 { 2.0 } else { -2.0 })
            .collect();
        let projected = project_thought_vector(&vector).expect("valid state");
        let expected = 1.0 / THOUGHT_VECTOR_DIMENSIONS as f64;

        assert!(projected
            .primary_l1
            .iter()
            .all(|value| (*value - expected).abs() < TOLERANCE));
        assert!(projected
            .sensitivity_energy
            .iter()
            .all(|value| (*value - expected).abs() < TOLERANCE));
    }

    #[test]
    fn global_scale_and_sign_do_not_change_projection() {
        let vector: Vec<f32> = (0..THOUGHT_VECTOR_DIMENSIONS)
            .map(|index| index as f32 - 11.5)
            .collect();
        let scaled: Vec<f32> = vector.iter().map(|value| -4.0 * value).collect();

        let original = project_thought_vector(&vector).expect("valid state");
        let transformed = project_thought_vector(&scaled).expect("valid state");

        assert_close(&original.primary_l1, &transformed.primary_l1);
        assert_close(
            &original.sensitivity_energy,
            &transformed.sensitivity_energy,
        );
    }

    #[test]
    fn projection_is_permutation_equivariant() {
        let vector: Vec<f32> = (1..=THOUGHT_VECTOR_DIMENSIONS)
            .map(|value| value as f32)
            .collect();
        let mut permuted = vector.clone();
        permuted.swap(2, 29);

        let original = project_thought_vector(&vector).expect("valid state");
        let transformed = project_thought_vector(&permuted).expect("valid state");

        let mut expected_l1 = original.primary_l1.clone();
        expected_l1.swap(2, 29);
        let mut expected_energy = original.sensitivity_energy.clone();
        expected_energy.swap(2, 29);

        assert_close(&expected_l1, &transformed.primary_l1);
        assert_close(&expected_energy, &transformed.sensitivity_energy);
    }

    #[test]
    fn primary_and_sensitivity_are_intentionally_distinct() {
        let mut vector = vec![0.0_f32; THOUGHT_VECTOR_DIMENSIONS];
        vector[0] = 3.0;
        vector[1] = 1.0;
        let projected = project_thought_vector(&vector).expect("valid state");

        assert!((projected.primary_l1[0] - 0.75).abs() < TOLERANCE);
        assert!((projected.primary_l1[1] - 0.25).abs() < TOLERANCE);
        assert!((projected.sensitivity_energy[0] - 0.9).abs() < TOLERANCE);
        assert!((projected.sensitivity_energy[1] - 0.1).abs() < TOLERANCE);
    }

    #[test]
    fn outputs_are_normalized_simplex_coordinates() {
        let vector: Vec<f32> = (1..=THOUGHT_VECTOR_DIMENSIONS)
            .map(|value| (value as f32).sin())
            .collect();
        let projected = project_thought_vector(&vector).expect("valid state");

        let l1_sum: f64 = projected.primary_l1.iter().sum();
        let energy_sum: f64 = projected.sensitivity_energy.iter().sum();
        assert!((l1_sum - 1.0).abs() < TOLERANCE);
        assert!((energy_sum - 1.0).abs() < TOLERANCE);
        assert!(projected.primary_l1.iter().all(|value| *value >= 0.0));
        assert!(projected
            .sensitivity_energy
            .iter()
            .all(|value| *value >= 0.0));
    }

    #[test]
    fn malformed_native_states_fail_closed() {
        assert!(matches!(
            project_thought_vector(&vec![1.0; 31]),
            Err(StateProjectionError::DimensionMismatch {
                expected: THOUGHT_VECTOR_DIMENSIONS,
                observed: 31,
            })
        ));

        let mut non_finite = vec![1.0; THOUGHT_VECTOR_DIMENSIONS];
        non_finite[9] = f32::NAN;
        assert_eq!(
            project_thought_vector(&non_finite),
            Err(StateProjectionError::NonFiniteValue { index: 9 })
        );

        assert_eq!(
            project_thought_vector(&vec![0.0; THOUGHT_VECTOR_DIMENSIONS]),
            Err(StateProjectionError::ZeroProjectionMass)
        );
    }
}
