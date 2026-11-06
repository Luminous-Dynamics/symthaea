// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic, norm-preserving structured mixing operators for HDC spaces.
//!
//! This module is an experimental substrate primitive, not yet enabled in the
//! default HDC-LTC neuron. It provides low-cost alternatives to dense D×D
//! recurrent mixing so cross-coordinate coupling can be tested at controlled
//! compute budgets.

use serde::{Deserialize, Serialize};
use super::unified_hv::ContinuousHV;

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
}

impl std::fmt::Display for HdcStructuredMixerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "structured HDC mixer dimension must be non-zero"),
            Self::NonPowerOfTwo { dimension } => write!(
                f,
                "Walsh-Hadamard mixer requires a power-of-two dimension, got {dimension}"
            ),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "structured HDC mixer dimension mismatch: expected {expected}, received {actual}"
            ),
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
        if matches!(kind, HdcStructuredMixerKind::WalshHadamard)
            && !dimension.is_power_of_two()
        {
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
    pub fn apply(
        &self,
        input: &ContinuousHV,
    ) -> Result<ContinuousHV, HdcStructuredMixerError> {
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
            HdcStructuredMixerKind::WalshHadamard => {
                normalized_walsh_hadamard(&input.values)
            }
        };
        Ok(ContinuousHV::from_values(values))
    }

    /// Apply the exact inverse of the configured transform.
    pub fn inverse(
        &self,
        input: &ContinuousHV,
    ) -> Result<ContinuousHV, HdcStructuredMixerError> {
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
                HdcStructuredMixer::new(HdcStructuredMixerKind::WalshHadamard, dimension)
                    .unwrap();
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
        assert!(mixed.values.iter().filter(|value| value.abs() > 1e-6).count() == dimension);
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
