// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic, norm-preserving structured mixing operators for HDC spaces.
//!
//! This module is an experimental substrate primitive, not yet enabled in the
//! default HDC-LTC neuron. It provides low-cost alternatives to dense D×D
//! recurrent mixing so cross-coordinate coupling can be tested at controlled
//! compute budgets.

use super::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Structured transform applied to a continuous hypervector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HdcStructuredMixerKind {
    /// No transformation.
    Identity,
    /// A cyclic coordinate permutation. O(D), exactly invertible.
    CyclicPermutation { shift: usize },
    /// Normalized Walsh-Hadamard transform. O(D log D), orthogonal and self-inverse.
    WalshHadamard,
}

/// Construction or application failure for a structured HDC mixer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HdcStructuredMixerError {
    ZeroDimension,
    NonPowerOfTwo { dimension: usize },
    DimensionMismatch { expected: usize, actual: usize },
    InvalidButterflyParameterCount { expected: usize, actual: usize },
    NonFiniteButterflyParameter { index: usize },
}

impl std::fmt::Display for HdcStructuredMixerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "structured HDC mixer dimension must be non-zero"),
            Self::NonPowerOfTwo { dimension } => write!(
                f,
                "Walsh-Hadamard and butterfly mixers require a power-of-two dimension, got {dimension}"
            ),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "structured HDC mixer dimension mismatch: expected {expected}, received {actual}"
            ),
            Self::InvalidButterflyParameterCount { expected, actual } => write!(
                f,
                "orthogonal butterfly mixer expected {expected} angles, received {actual}"
            ),
            Self::NonFiniteButterflyParameter { index } => {
                write!(f, "orthogonal butterfly mixer angle {index} is not finite")
            }
        }
    }
}

impl std::error::Error for HdcStructuredMixerError {}

/// Fixed structured mixer with an exact inverse for the selected transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HdcStructuredMixer {
    kind: HdcStructuredMixerKind,
    dimension: usize,
}

impl HdcStructuredMixer {
    /// Construct a mixer, validating constraints up front.
    pub fn new(
        kind: HdcStructuredMixerKind,
        dimension: usize,
    ) -> Result<Self, HdcStructuredMixerError> {
        if dimension == 0 {
            return Err(HdcStructuredMixerError::ZeroDimension);
        }
        if matches!(kind, HdcStructuredMixerKind::WalshHadamard) && !dimension.is_power_of_two() {
            return Err(HdcStructuredMixerError::NonPowerOfTwo { dimension });
        }
        Ok(Self { kind, dimension })
    }

    pub fn kind(&self) -> HdcStructuredMixerKind {
        self.kind
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Apply the configured orthogonal transform.
    pub fn apply(&self, input: &ContinuousHV) -> Result<ContinuousHV, HdcStructuredMixerError> {
        self.validate_input(input)?;
        let values = match self.kind {
            HdcStructuredMixerKind::Identity => input.values.clone(),
            HdcStructuredMixerKind::CyclicPermutation { shift } => {
                let shift = shift % self.dimension;
                let mut output = vec![0.0; self.dimension];
                for (index, value) in input.values.iter().copied().enumerate() {
                    output[(index + shift) % self.dimension] = value;
                }
                output
            }
            HdcStructuredMixerKind::WalshHadamard => normalized_walsh_hadamard(&input.values),
        };
        Ok(ContinuousHV::from_values(values))
    }

    /// Apply the exact inverse of the configured transform.
    pub fn inverse(&self, input: &ContinuousHV) -> Result<ContinuousHV, HdcStructuredMixerError> {
        self.validate_input(input)?;
        let values = match self.kind {
            HdcStructuredMixerKind::Identity => input.values.clone(),
            HdcStructuredMixerKind::CyclicPermutation { shift } => {
                let shift = shift % self.dimension;
                let mut output = vec![0.0; self.dimension];
                for index in 0..self.dimension {
                    output[index] = input.values[(index + shift) % self.dimension];
                }
                output
            }
            HdcStructuredMixerKind::WalshHadamard => {
                // The normalized Walsh-Hadamard matrix is symmetric and orthogonal.
                normalized_walsh_hadamard(&input.values)
            }
        };
        Ok(ContinuousHV::from_values(values))
    }

    fn validate_input(&self, input: &ContinuousHV) -> Result<(), HdcStructuredMixerError> {
        if input.dim() != self.dimension {
            return Err(HdcStructuredMixerError::DimensionMismatch {
                expected: self.dimension,
                actual: input.dim(),
            });
        }
        Ok(())
    }
}

/// Parameterized orthogonal butterfly transform with a learned-parameter-ready
/// angle vector.
///
/// For dimension D=2^L this operator uses (D/2)*L rotation angles and
/// O(D log D) arithmetic. Each stage applies disjoint Givens rotations across
/// pairs separated by a power-of-two stride. The inverse applies the transpose
/// rotations in reverse stage order. Parameters are supplied by the caller;
/// this type does not train them or integrate into HDC-LTC automatically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HdcButterflyMixer {
    dimension: usize,
    angles: Vec<f32>,
}

impl HdcButterflyMixer {
    /// Construct a parameterized orthogonal butterfly mixer.
    ///
    /// The angle order is stage-major, then pair-major within each stage.
    pub fn new(dimension: usize, angles: Vec<f32>) -> Result<Self, HdcStructuredMixerError> {
        if dimension == 0 {
            return Err(HdcStructuredMixerError::ZeroDimension);
        }
        if !dimension.is_power_of_two() {
            return Err(HdcStructuredMixerError::NonPowerOfTwo { dimension });
        }
        let expected = Self::parameter_count_for_dimension(dimension);
        if angles.len() != expected {
            return Err(HdcStructuredMixerError::InvalidButterflyParameterCount {
                expected,
                actual: angles.len(),
            });
        }
        if let Some(index) = angles.iter().position(|angle| !angle.is_finite()) {
            return Err(HdcStructuredMixerError::NonFiniteButterflyParameter { index });
        }
        Ok(Self { dimension, angles })
    }

    /// Number of trainable rotation angles for the selected dimension.
    pub fn parameter_count_for_dimension(dimension: usize) -> usize {
        if dimension == 0 || !dimension.is_power_of_two() {
            return 0;
        }
        (dimension / 2) * (dimension.trailing_zeros() as usize)
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn angles(&self) -> &[f32] {
        &self.angles
    }

    pub fn parameter_count(&self) -> usize {
        self.angles.len()
    }

    /// Apply the factorized orthogonal transform.
    pub fn apply(&self, input: &ContinuousHV) -> Result<ContinuousHV, HdcStructuredMixerError> {
        self.apply_with_direction(input, 1.0, false)
    }

    /// Apply the exact transpose/inverse of the configured transform.
    pub fn inverse(&self, input: &ContinuousHV) -> Result<ContinuousHV, HdcStructuredMixerError> {
        self.apply_with_direction(input, -1.0, true)
    }

    fn apply_with_direction(
        &self,
        input: &ContinuousHV,
        direction: f32,
        reverse_stages: bool,
    ) -> Result<ContinuousHV, HdcStructuredMixerError> {
        if input.dim() != self.dimension {
            return Err(HdcStructuredMixerError::DimensionMismatch {
                expected: self.dimension,
                actual: input.dim(),
            });
        }

        let mut values = input.values.clone();
        let stage_count = self.dimension.trailing_zeros() as usize;
        let pairs_per_stage = self.dimension / 2;
        let mut stages = 0..stage_count;

        loop {
            let next_stage = if reverse_stages {
                stages.next_back()
            } else {
                stages.next()
            };
            let Some(stage) = next_stage else {
                break;
            };
            let half_width = 1usize << stage;
            let block_width = half_width * 2;
            let mut pair_index = 0usize;

            for block_start in (0..self.dimension).step_by(block_width) {
                for offset in 0..half_width {
                    let left_index = block_start + offset;
                    let right_index = left_index + half_width;
                    let angle_index = stage * pairs_per_stage + pair_index;
                    let angle = direction * self.angles[angle_index];
                    let cosine = angle.cos();
                    let sine = angle.sin();
                    let left = values[left_index];
                    let right = values[right_index];

                    values[left_index] = cosine * left - sine * right;
                    values[right_index] = sine * left + cosine * right;
                    pair_index += 1;
                }
            }
        }

        Ok(ContinuousHV::from_values(values))
    }
}

/// In-place butterfly transform followed by 1/sqrt(D) normalization.
fn normalized_walsh_hadamard(input: &[f32]) -> Vec<f32> {
    let dimension = input.len();
    let mut values = input.to_vec();
    let mut half_width = 1usize;

    while half_width < dimension {
        let block_width = half_width * 2;
        for block_start in (0..dimension).step_by(block_width) {
            for offset in 0..half_width {
                let left = values[block_start + offset];
                let right = values[block_start + offset + half_width];
                values[block_start + offset] = left + right;
                values[block_start + offset + half_width] = left - right;
            }
        }
        half_width = block_width;
    }

    let scale = 1.0 / (dimension as f32).sqrt();
    values.iter_mut().for_each(|value| *value *= scale);
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(dimension: usize, seed: u64) -> ContinuousHV {
        ContinuousHV::random(dimension, seed)
    }

    fn max_abs_difference(left: &ContinuousHV, right: &ContinuousHV) -> f32 {
        left.values
            .iter()
            .zip(right.values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn identity_and_cyclic_permutation_round_trip() {
        let input = sample(64, 10);
        for kind in [
            HdcStructuredMixerKind::Identity,
            HdcStructuredMixerKind::CyclicPermutation { shift: 17 },
            HdcStructuredMixerKind::CyclicPermutation { shift: 64 * 3 + 5 },
        ] {
            let mixer = HdcStructuredMixer::new(kind, 64).unwrap();
            let mixed = mixer.apply(&input).unwrap();
            let restored = mixer.inverse(&mixed).unwrap();
            assert!(max_abs_difference(&input, &restored) < 1e-7);
            assert!((input.norm() - mixed.norm()).abs() < 1e-5);
        }
    }

    #[test]
    fn walsh_hadamard_is_orthogonal_and_self_inverse() {
        for dimension in [1, 2, 16, 256, 16_384] {
            let input = sample(dimension, 11);
            let mixer =
                HdcStructuredMixer::new(HdcStructuredMixerKind::WalshHadamard, dimension).unwrap();
            let mixed = mixer.apply(&input).unwrap();
            let restored = mixer.inverse(&mixed).unwrap();

            assert!(
                max_abs_difference(&input, &restored) < 2e-5,
                "round-trip error at dimension {dimension}"
            );
            assert!(
                (input.norm() - mixed.norm()).abs() < 2e-4,
                "norm changed at dimension {dimension}: {} -> {}",
                input.norm(),
                mixed.norm()
            );
        }
    }

    #[test]
    fn walsh_hadamard_spreads_a_single_coordinate_across_the_space() {
        let dimension = 256;
        let input = ContinuousHV::basis(7, dimension);
        let mixer =
            HdcStructuredMixer::new(HdcStructuredMixerKind::WalshHadamard, dimension).unwrap();
        let mixed = mixer.apply(&input).unwrap();
        let expected_magnitude = 1.0 / (dimension as f32).sqrt();

        assert!(
            mixed.values.iter().all(|value| {
                (*value - expected_magnitude).abs() < 1e-6
                    || (*value + expected_magnitude).abs() < 1e-6
            }),
            "a one-hot input should spread across every mixed coordinate"
        );
        assert!(
            mixed
                .values
                .iter()
                .filter(|value| value.abs() > 1e-6)
                .count()
                == dimension
        );
    }

    #[test]
    fn butterfly_parameter_count_is_linearithmic_not_quadratic() {
        assert_eq!(
            HdcButterflyMixer::parameter_count_for_dimension(16_384),
            114_688
        );
        assert_eq!(
            HdcButterflyMixer::parameter_count_for_dimension(1_024),
            5_120
        );
        assert_eq!(HdcButterflyMixer::parameter_count_for_dimension(12), 0);
    }

    #[test]
    fn parameterized_butterfly_is_orthogonal_and_invertible() {
        for dimension in [2, 8, 64, 256, 16_384] {
            let parameter_count = HdcButterflyMixer::parameter_count_for_dimension(dimension);
            let angles = (0..parameter_count)
                .map(|index| ((index as f32 * 0.017).sin()) * 0.6)
                .collect::<Vec<_>>();
            let mixer = HdcButterflyMixer::new(dimension, angles).unwrap();
            let input = sample(dimension, 0xB077 + dimension as u64);
            let mixed = mixer.apply(&input).unwrap();
            let restored = mixer.inverse(&mixed).unwrap();

            assert!(
                max_abs_difference(&input, &restored) < 5e-5,
                "round-trip failed at dimension {dimension}"
            );
            assert!(
                (input.norm() - mixed.norm()).abs() < 5e-4,
                "orthogonality failed at dimension {dimension}"
            );
        }
    }

    #[test]
    fn zero_angle_butterfly_is_identity() {
        let dimension = 64;
        let parameter_count = HdcButterflyMixer::parameter_count_for_dimension(dimension);
        let mixer = HdcButterflyMixer::new(dimension, vec![0.0; parameter_count]).unwrap();
        let input = sample(dimension, 0x1D3A71);
        let output = mixer.apply(&input).unwrap();

        assert!(max_abs_difference(&input, &output) < 1e-7);
        assert!((input.norm() - output.norm()).abs() < 1e-7);
    }

    #[test]
    fn parameterized_butterfly_spreads_a_single_coordinate() {
        let dimension = 256;
        let count = HdcButterflyMixer::parameter_count_for_dimension(dimension);
        let mixer =
            HdcButterflyMixer::new(dimension, vec![std::f32::consts::FRAC_PI_4; count]).unwrap();
        let input = ContinuousHV::basis(7, dimension);
        let mixed = mixer.apply(&input).unwrap();
        let active = mixed
            .values
            .iter()
            .filter(|value| value.abs() > 1e-6)
            .count();

        assert_eq!(active, dimension);
        assert!((mixed.norm() - input.norm()).abs() < 1e-4);
    }

    #[test]
    fn butterfly_rejects_wrong_parameter_count_and_nonfinite_angles() {
        assert!(matches!(
            HdcButterflyMixer::new(12, Vec::new()),
            Err(HdcStructuredMixerError::NonPowerOfTwo { dimension: 12 })
        ));
        assert!(matches!(
            HdcButterflyMixer::new(64, vec![0.0; 10]),
            Err(HdcStructuredMixerError::InvalidButterflyParameterCount { .. })
        ));
        let mut angles = vec![0.0; HdcButterflyMixer::parameter_count_for_dimension(8)];
        angles[1] = f32::NAN;
        assert!(matches!(
            HdcButterflyMixer::new(8, angles),
            Err(HdcStructuredMixerError::NonFiniteButterflyParameter { index: 1 })
        ));
    }

    #[test]
    fn rejects_invalid_dimensions_and_input_shapes() {
        assert!(matches!(
            HdcStructuredMixer::new(HdcStructuredMixerKind::Identity, 0),
            Err(HdcStructuredMixerError::ZeroDimension)
        ));
        assert!(matches!(
            HdcStructuredMixer::new(HdcStructuredMixerKind::WalshHadamard, 12),
            Err(HdcStructuredMixerError::NonPowerOfTwo { dimension: 12 })
        ));

        let mixer = HdcStructuredMixer::new(HdcStructuredMixerKind::Identity, 64).unwrap();
        assert!(matches!(
            mixer.apply(&sample(32, 12)),
            Err(HdcStructuredMixerError::DimensionMismatch {
                expected: 64,
                actual: 32
            })
        ));
    }
}
