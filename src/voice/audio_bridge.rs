// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Audio Bridge: STT HDC to Cognitive Loop Conversion
//!
//! Type conversion helpers that bridge the STT audio pipeline (which produces
//! `BinaryHV` frame vectors) to the cognitive loop (which consumes `ContinuousHV`).
//!
//! ## Pipeline Position
//!
//! ```text
//! STT AudioProjector::project()
//!     │
//!     ▼
//! Vec<BinaryHV>  ──► aggregate_audio_hvs() ──► ContinuousHV
//!                                                    │
//!                                                    ▼
//!                                    CognitiveLoopService::cycle_with_hv()
//! ```
//!
//! ## Encoding
//!
//! Binary bits map to bipolar floats: `0 → -1.0`, `1 → +1.0`.
//! Multiple frame HVs are aggregated via element-wise averaging, producing
//! a continuous vector that preserves the dominant activation pattern across
//! the audio segment.

use symthaea_core::hdc::{BinaryHV, ContinuousHV};

/// Convert a single `BinaryHV` to `ContinuousHV` using bipolar encoding.
///
/// Each bit maps to: `0 → -1.0`, `1 → +1.0`.
/// This is the standard HDC binary-to-continuous conversion.
///
/// # Example
///
/// ```ignore
/// let binary = BinaryHV::zero();
/// let continuous = binary_to_continuous(&binary);
/// assert_eq!(continuous.values.len(), BinaryHV::DIM);
/// assert!(continuous.values.iter().all(|&v| v == -1.0));
/// ```
pub fn binary_to_continuous(hv: &BinaryHV) -> ContinuousHV {
    hv.to_continuous()
}

/// Aggregate multiple audio frame HVs into a single `ContinuousHV` for
/// cognitive processing.
///
/// Converts each `BinaryHV` to bipolar floats and computes the element-wise
/// average. The result is a continuous vector whose components lie in `[-1.0, 1.0]`,
/// capturing the dominant spectral pattern across all frames.
///
/// Returns a zero-valued `ContinuousHV` if `frames` is empty.
///
/// # Arguments
///
/// * `frames` - Audio frame hypervectors from `AudioProjector::project()`.
///
/// # Example
///
/// ```ignore
/// let frames = projector.project(&audio_samples);
/// let aggregated = aggregate_audio_hvs(&frames);
/// let result = cognitive_loop.cycle_with_hv(&aggregated);
/// ```
pub fn aggregate_audio_hvs(frames: &[BinaryHV]) -> ContinuousHV {
    if frames.is_empty() {
        return ContinuousHV::zero(BinaryHV::DIM);
    }

    if frames.len() == 1 {
        return binary_to_continuous(&frames[0]);
    }

    let dim = BinaryHV::DIM;
    let n = frames.len() as f32;
    let mut sum = vec![0.0_f32; dim];

    for frame in frames {
        let bipolar = frame.to_bipolar();
        for (s, &b) in sum.iter_mut().zip(bipolar.iter()) {
            *s += b;
        }
    }

    // Average
    for s in &mut sum {
        *s /= n;
    }

    ContinuousHV::from_vec(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_to_continuous_zeros() {
        let hv = BinaryHV::zero();
        let continuous = binary_to_continuous(&hv);
        assert_eq!(continuous.values.len(), BinaryHV::DIM);
        // All zero bits -> all -1.0
        assert!(
            continuous
                .values
                .iter()
                .all(|&v| (v - (-1.0)).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn test_binary_to_continuous_ones() {
        let hv = BinaryHV::ones();
        let continuous = binary_to_continuous(&hv);
        assert_eq!(continuous.values.len(), BinaryHV::DIM);
        // All one bits -> all +1.0
        assert!(
            continuous
                .values
                .iter()
                .all(|&v| (v - 1.0).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn test_aggregate_empty() {
        let result = aggregate_audio_hvs(&[]);
        assert_eq!(result.values.len(), BinaryHV::DIM);
        assert!(result.values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_aggregate_single_frame() {
        let hv = BinaryHV::ones();
        let result = aggregate_audio_hvs(&[hv]);
        assert_eq!(result.values.len(), BinaryHV::DIM);
        // Single frame: should be identical to binary_to_continuous
        let direct = binary_to_continuous(&hv);
        assert_eq!(result.values, direct.values);
    }

    #[test]
    fn test_aggregate_multiple_frames() {
        let zeros = BinaryHV::zero();
        let ones = BinaryHV::ones();
        let result = aggregate_audio_hvs(&[zeros, ones]);
        assert_eq!(result.values.len(), BinaryHV::DIM);
        // Average of -1.0 and 1.0 = 0.0 for each dimension
        for &v in &result.values {
            assert!(v.abs() < f32::EPSILON, "expected 0.0, got {}", v);
        }
    }

    #[test]
    fn test_aggregate_values_bounded() {
        // Random HVs should average to values in [-1, 1]
        let frames: Vec<BinaryHV> = (0..10).map(|i| BinaryHV::random(i as u64 + 1)).collect();
        let result = aggregate_audio_hvs(&frames);
        for &v in &result.values {
            assert!(
                (-1.0..=1.0).contains(&v),
                "aggregated value should be in [-1, 1]: got {}",
                v
            );
            assert!(v.is_finite());
        }
    }
}
