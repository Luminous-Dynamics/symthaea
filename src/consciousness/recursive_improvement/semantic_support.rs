// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit support diagnostics for semantic HDC contexts.
//!
//! `SemanticContextEncoder` clamps scalar inputs before level encoding. That is a
//! reasonable representation policy, but it means an out-of-range context can be
//! semantically close—or even identical in HDC space—to an in-range boundary value.
//! This module makes that limitation observable without changing encoder behavior.
//!
//! Support diagnostics are not evidence, confidence, or validation. They describe
//! whether a numeric context lies inside the nominal numeric support of one encoder.

use super::semantic_context::{ContextIdentity, SemanticContextEncoder, SemanticContextError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemanticContextSupport {
    /// Numeric context width.
    pub context_dimension: usize,
    /// Encoder absolute clamp used for every scalar coordinate.
    pub clamp_abs: f32,
    /// Largest absolute scalar observed before clamping.
    pub max_abs_input: f32,
    /// Number of coordinates whose magnitude exceeds `clamp_abs`.
    pub clipped_value_count: usize,
    /// Fraction of coordinates outside nominal encoder support.
    pub clipped_fraction: f32,
    /// Largest absolute amount by which any coordinate exceeds `clamp_abs`.
    pub max_overflow_abs: f32,
    /// Mean overflow amount across all coordinates, counting in-support values as 0.
    pub mean_overflow_abs: f32,
}

impl SemanticContextSupport {
    pub fn within_nominal_support(self) -> bool {
        self.clipped_value_count == 0
    }

    /// Diagnostic-only normalized overflow severity.
    ///
    /// `0` means fully inside support. Positive values indicate how far the worst
    /// coordinate exceeds the configured clamp relative to the clamp magnitude.
    /// This value must not be interpreted as probability or evidence confidence.
    pub fn max_overflow_ratio(self) -> f32 {
        if self.clamp_abs <= 0.0 {
            0.0
        } else {
            self.max_overflow_abs / self.clamp_abs
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticSupportError {
    Context(SemanticContextError),
}

impl From<SemanticContextError> for SemanticSupportError {
    fn from(value: SemanticContextError) -> Self {
        Self::Context(value)
    }
}

/// Assess whether a numeric context is inside one encoder's nominal scalar support.
///
/// Calling `ContextIdentity::from_context` first intentionally reuses the existing
/// empty/non-finite validation contract rather than defining a second validator.
pub fn assess_semantic_context_support(
    encoder: SemanticContextEncoder,
    context: &[f32],
) -> Result<SemanticContextSupport, SemanticSupportError> {
    ContextIdentity::from_context(context)?;

    let clamp_abs = encoder.clamp_abs();
    let mut max_abs_input = 0.0_f32;
    let mut clipped_value_count = 0usize;
    let mut max_overflow_abs = 0.0_f32;
    let mut overflow_sum = 0.0_f32;

    for value in context {
        let absolute = value.abs();
        max_abs_input = max_abs_input.max(absolute);
        let overflow = (absolute - clamp_abs).max(0.0);
        if overflow > 0.0 {
            clipped_value_count += 1;
            max_overflow_abs = max_overflow_abs.max(overflow);
            overflow_sum += overflow;
        }
    }

    let dimension = context.len();
    Ok(SemanticContextSupport {
        context_dimension: dimension,
        clamp_abs,
        max_abs_input,
        clipped_value_count,
        clipped_fraction: clipped_value_count as f32 / dimension as f32,
        max_overflow_abs,
        mean_overflow_abs: overflow_sum / dimension as f32,
    })
}

/// Paired diagnostics for a semantic retrieval comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemanticPairSupport {
    pub query: SemanticContextSupport,
    pub candidate: SemanticContextSupport,
}

impl SemanticPairSupport {
    pub fn both_within_nominal_support(self) -> bool {
        self.query.within_nominal_support() && self.candidate.within_nominal_support()
    }
}

pub fn assess_semantic_pair_support(
    encoder: SemanticContextEncoder,
    query: &[f32],
    candidate: &[f32],
) -> Result<SemanticPairSupport, SemanticSupportError> {
    Ok(SemanticPairSupport {
        query: assess_semantic_context_support(encoder, query)?,
        candidate: assess_semantic_context_support(encoder, candidate)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_range_context_is_explicitly_supported() {
        let support = assess_semantic_context_support(
            SemanticContextEncoder::default(),
            &[0.0, 0.25, -0.75, 1.0],
        )
        .unwrap();

        assert!(support.within_nominal_support());
        assert_eq!(support.clipped_value_count, 0);
        assert_eq!(support.clipped_fraction, 0.0);
        assert_eq!(support.max_overflow_abs, 0.0);
        assert_eq!(support.mean_overflow_abs, 0.0);
    }

    #[test]
    fn clamp_saturation_is_observable_without_changing_exact_identity() {
        let encoder = SemanticContextEncoder::default();
        let boundary = [1.0, -1.0, 0.0];
        let outside = [2.5, -3.0, 0.0];

        let support = assess_semantic_context_support(encoder, &outside).unwrap();
        assert!(!support.within_nominal_support());
        assert_eq!(support.clipped_value_count, 2);
        assert!((support.clipped_fraction - 2.0 / 3.0).abs() < f32::EPSILON);
        assert_eq!(support.max_overflow_abs, 2.0);
        assert!(support.mean_overflow_abs > 0.0);

        assert_ne!(
            ContextIdentity::from_context(&boundary).unwrap(),
            ContextIdentity::from_context(&outside).unwrap()
        );
    }

    #[test]
    fn support_is_relative_to_encoder_configuration() {
        let narrow = SemanticContextEncoder::new(64, 1.0).unwrap();
        let wide = SemanticContextEncoder::new(64, 3.0).unwrap();
        let context = [2.0, -2.5];

        let narrow_support = assess_semantic_context_support(narrow, &context).unwrap();
        let wide_support = assess_semantic_context_support(wide, &context).unwrap();

        assert!(!narrow_support.within_nominal_support());
        assert!(wide_support.within_nominal_support());
    }

    #[test]
    fn paired_support_requires_both_sides_inside_support() {
        let support = assess_semantic_pair_support(
            SemanticContextEncoder::default(),
            &[0.1, 0.2],
            &[0.1, 1.5],
        )
        .unwrap();

        assert!(support.query.within_nominal_support());
        assert!(!support.candidate.within_nominal_support());
        assert!(!support.both_within_nominal_support());
    }

    #[test]
    fn malformed_context_fails_through_existing_context_contract() {
        assert_eq!(
            assess_semantic_context_support(SemanticContextEncoder::default(), &[f32::NAN]),
            Err(SemanticSupportError::Context(
                SemanticContextError::NonFiniteValue { index: 0 }
            ))
        );
    }
}