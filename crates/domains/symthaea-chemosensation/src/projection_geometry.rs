// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Full-resolution geometry for evidence-producing projection assessments.
//!
//! `ContinuousHV::similarity()` intentionally honors Symthaea's global adaptive
//! cognitive stride. That is useful for runtime throttling, but an experiment
//! receipt must not change because an unrelated power/performance controller
//! changed how many dimensions cognition samples. Projection assessment therefore
//! uses this fixed full-resolution cosine implementation instead.

use symthaea_core::hdc::unified_hv::ContinuousHV;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionGeometryError {
    DimensionMismatch { left: usize, right: usize },
    EmptyVector,
    NonFiniteVector,
    DegenerateVector,
}

/// Full-resolution cosine similarity independent of the global cognitive stride.
///
/// Accumulation uses `f64` to reduce rounding error across 16,384 dimensions;
/// the bounded result is returned as `f32` to match the surrounding HDC API.
pub(crate) fn exact_cosine(
    left: &ContinuousHV,
    right: &ContinuousHV,
) -> Result<f32, ProjectionGeometryError> {
    if left.values.len() != right.values.len() {
        return Err(ProjectionGeometryError::DimensionMismatch {
            left: left.values.len(),
            right: right.values.len(),
        });
    }
    if left.values.is_empty() {
        return Err(ProjectionGeometryError::EmptyVector);
    }

    let mut dot = 0.0f64;
    let mut left_norm_sq = 0.0f64;
    let mut right_norm_sq = 0.0f64;
    for (&x, &y) in left.values.iter().zip(&right.values) {
        if !x.is_finite() || !y.is_finite() {
            return Err(ProjectionGeometryError::NonFiniteVector);
        }
        let x = f64::from(x);
        let y = f64::from(y);
        dot += x * y;
        left_norm_sq += x * x;
        right_norm_sq += y * y;
    }

    let denominator = (left_norm_sq * right_norm_sq).sqrt();
    if !denominator.is_finite() || denominator <= 1e-20 {
        return Err(ProjectionGeometryError::DegenerateVector);
    }

    Ok((dot / denominator).clamp(-1.0, 1.0) as f32)
}

pub(crate) fn validate_non_degenerate(
    vector: &ContinuousHV,
) -> Result<(), ProjectionGeometryError> {
    // Comparing the vector with itself performs the same finite/norm validation
    // without introducing a second numerical contract.
    exact_cosine(vector, vector).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hv(values: &[f32]) -> ContinuousHV {
        ContinuousHV::from_vec(values.to_vec())
    }

    #[test]
    fn exact_geometry_has_known_cosine_answers() {
        assert!((exact_cosine(&hv(&[1.0, 0.0]), &hv(&[1.0, 0.0])).unwrap() - 1.0).abs() < 1e-6);
        assert!((exact_cosine(&hv(&[1.0, 0.0]), &hv(&[-1.0, 0.0])).unwrap() + 1.0).abs() < 1e-6);
        assert!(exact_cosine(&hv(&[1.0, 0.0]), &hv(&[0.0, 1.0])).unwrap().abs() < 1e-6);
    }

    #[test]
    fn invalid_geometry_is_rejected_not_reinterpreted() {
        assert_eq!(
            exact_cosine(&hv(&[1.0]), &hv(&[1.0, 2.0])),
            Err(ProjectionGeometryError::DimensionMismatch { left: 1, right: 2 })
        );
        assert_eq!(
            exact_cosine(&hv(&[]), &hv(&[])),
            Err(ProjectionGeometryError::EmptyVector)
        );
        assert_eq!(
            exact_cosine(&hv(&[0.0, 0.0]), &hv(&[1.0, 0.0])),
            Err(ProjectionGeometryError::DegenerateVector)
        );
        assert_eq!(
            exact_cosine(&hv(&[f32::NAN]), &hv(&[1.0])),
            Err(ProjectionGeometryError::NonFiniteVector)
        );
    }

    #[test]
    fn accumulation_is_full_resolution() {
        // Differ only in a position that a stride-based sampler could skip.
        // Full-resolution cosine must account for every component.
        let left = hv(&[1.0, 1.0, 1.0, 1.0]);
        let right = hv(&[1.0, -1.0, 1.0, 1.0]);
        let similarity = exact_cosine(&left, &right).unwrap();
        assert!((similarity - 0.5).abs() < 1e-6);
    }
}
