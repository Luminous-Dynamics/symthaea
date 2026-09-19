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

use std::collections::HashMap;

use super::dream_feedback::ActionPrior;
use super::semantic_context::{ContextIdentity, SemanticContextEncoder, SemanticContextError};
use super::semantic_null_calibration::SemanticNullCalibration;
use super::semantic_supported_retrieval::{
    FullySupportedCalibratedSemanticPriorCandidate, FullySupportedSemanticPriorCandidate,
    SemanticQuerySupportError, SupportBoundSemanticPriorMemory,
};

pub const SEMANTIC_SUPPORT_BINDING_SCHEMA: &str =
    "symthaea.semantic-context-support-binding.v1";

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticSupportBindingError {
    Support(SemanticSupportError),
    BindingDigestMismatch,
    ContextIdentityMismatch,
    EncoderMismatch,
    SupportMismatch,
}

impl From<SemanticSupportError> for SemanticSupportBindingError {
    fn from(value: SemanticSupportError) -> Self {
        Self::Support(value)
    }
}

/// Content-bound semantic-support receipt.
///
/// The digest commits to exact context identity, encoder configuration, and every
/// diagnostic field. It proves integrity/detachment resistance only; authenticity
/// still depends on where this receipt is frozen or signed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemanticSupportBinding {
    pub context_identity: ContextIdentity,
    pub encoder_levels: u16,
    pub encoder_clamp_bits: u32,
    pub support: SemanticContextSupport,
    pub binding_digest: [u8; 32],
}

impl SemanticSupportBinding {
    pub fn create(
        encoder: SemanticContextEncoder,
        context: &[f32],
    ) -> Result<Self, SemanticSupportBindingError> {
        let context_identity = ContextIdentity::from_context(context)
            .map_err(SemanticSupportError::from)?;
        let support = assess_semantic_context_support(encoder, context)?;
        let mut binding = Self {
            context_identity,
            encoder_levels: encoder.levels(),
            encoder_clamp_bits: encoder.clamp_abs().to_bits(),
            support,
            binding_digest: [0; 32],
        };
        binding.binding_digest = binding.recompute_digest();
        Ok(binding)
    }

    pub fn encoder_clamp_abs(self) -> f32 {
        f32::from_bits(self.encoder_clamp_bits)
    }

    pub fn within_nominal_support(self) -> bool {
        self.support.within_nominal_support()
    }

    /// Verify internal field integrity. This is not an authenticity check.
    pub fn validate_self(self) -> Result<(), SemanticSupportBindingError> {
        if self.binding_digest != self.recompute_digest() {
            return Err(SemanticSupportBindingError::BindingDigestMismatch);
        }
        if self.support.clamp_abs.to_bits() != self.encoder_clamp_bits {
            return Err(SemanticSupportBindingError::SupportMismatch);
        }
        Ok(())
    }

    /// Recompute identity and support from the original context and encoder.
    pub fn validate_against_context(
        self,
        encoder: SemanticContextEncoder,
        context: &[f32],
    ) -> Result<(), SemanticSupportBindingError> {
        self.validate_self()?;

        let observed_identity = ContextIdentity::from_context(context)
            .map_err(SemanticSupportError::from)?;
        if observed_identity != self.context_identity {
            return Err(SemanticSupportBindingError::ContextIdentityMismatch);
        }
        if encoder.levels() != self.encoder_levels
            || encoder.clamp_abs().to_bits() != self.encoder_clamp_bits
        {
            return Err(SemanticSupportBindingError::EncoderMismatch);
        }

        let observed_support = assess_semantic_context_support(encoder, context)?;
        if observed_support != self.support {
            return Err(SemanticSupportBindingError::SupportMismatch);
        }
        Ok(())
    }

    fn recompute_digest(self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(SEMANTIC_SUPPORT_BINDING_SCHEMA.as_bytes());
        hasher.update(&self.context_identity.fast_hash.to_le_bytes());
        hasher.update(&self.context_identity.exact_digest);
        hasher.update(&self.encoder_levels.to_le_bytes());
        hasher.update(&self.encoder_clamp_bits.to_le_bytes());
        hash_support(&mut hasher, self.support);
        *hasher.finalize().as_bytes()
    }
}

fn hash_support(hasher: &mut blake3::Hasher, support: SemanticContextSupport) {
    hasher.update(&(support.context_dimension as u64).to_le_bytes());
    for value in [
        support.clamp_abs,
        support.max_abs_input,
        support.clipped_fraction,
        support.max_overflow_abs,
        support.mean_overflow_abs,
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    hasher.update(&(support.clipped_value_count as u64).to_le_bytes());
}

#[derive(Debug, Clone, PartialEq)]
pub enum IntegrityBoundSemanticPriorError {
    Binding(SemanticSupportBindingError),
    Retrieval(SemanticQuerySupportError),
    MissingSourceBinding { identity: ContextIdentity },
    BindingDrift { identity: ContextIdentity },
}

impl From<SemanticSupportBindingError> for IntegrityBoundSemanticPriorError {
    fn from(value: SemanticSupportBindingError) -> Self {
        Self::Binding(value)
    }
}

impl From<SemanticQuerySupportError> for IntegrityBoundSemanticPriorError {
    fn from(value: SemanticQuerySupportError) -> Self {
        Self::Retrieval(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntegrityBoundSemanticPriorCandidate {
    pub candidate: FullySupportedSemanticPriorCandidate,
    pub query_binding: SemanticSupportBinding,
    pub candidate_source_binding: SemanticSupportBinding,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntegrityBoundCalibratedSemanticPriorCandidate {
    pub candidate: FullySupportedCalibratedSemanticPriorCandidate,
    pub query_binding: SemanticSupportBinding,
    pub candidate_source_binding: SemanticSupportBinding,
}

/// V2 strict operational memory whose support metadata is cryptographically bound
/// to exact context identity and encoder configuration.
///
/// The v1 strict memory remains the inner behavioral engine. This wrapper adds
/// durable support provenance without changing the historical diagnostic/raw APIs.
#[derive(Clone, Default)]
pub struct IntegrityBoundSemanticPriorMemory {
    inner: SupportBoundSemanticPriorMemory,
    source_bindings: HashMap<ContextIdentity, SemanticSupportBinding>,
}

impl IntegrityBoundSemanticPriorMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn context_encoder(&self) -> SemanticContextEncoder {
        self.inner.context_encoder()
    }

    pub fn source_binding(&self, identity: ContextIdentity) -> Option<SemanticSupportBinding> {
        self.source_bindings.get(&identity).copied()
    }

    pub fn index_prior(
        &mut self,
        context: &[f32],
        prior: &ActionPrior,
    ) -> Result<ContextIdentity, IntegrityBoundSemanticPriorError> {
        let binding = SemanticSupportBinding::create(self.context_encoder(), context)?;
        let identity = self.inner.index_prior(context, prior)?;
        if identity != binding.context_identity {
            return Err(IntegrityBoundSemanticPriorError::BindingDrift { identity });
        }
        self.source_bindings.insert(identity, binding);
        Ok(identity)
    }

    pub fn retrieve_integrity_bound(
        &self,
        query: &[f32],
        min_similarity: f32,
        limit: usize,
    ) -> Result<Vec<IntegrityBoundSemanticPriorCandidate>, IntegrityBoundSemanticPriorError> {
        let query_binding = SemanticSupportBinding::create(self.context_encoder(), query)?;
        if !query_binding.within_nominal_support() {
            return Ok(Vec::new());
        }
        let candidates = self
            .inner
            .retrieve_fully_supported(query, min_similarity, limit)?;
        candidates
            .into_iter()
            .map(|candidate| self.bind_raw_candidate(candidate, query_binding))
            .collect()
    }

    pub fn retrieve_calibrated_integrity_bound(
        &self,
        query: &[f32],
        min_similarity: f32,
        limit: usize,
        calibration: &SemanticNullCalibration,
    ) -> Result<Vec<IntegrityBoundCalibratedSemanticPriorCandidate>, IntegrityBoundSemanticPriorError>
    {
        let query_binding = SemanticSupportBinding::create(self.context_encoder(), query)?;
        if !query_binding.within_nominal_support() {
            return Ok(Vec::new());
        }
        let candidates = self.inner.retrieve_calibrated_fully_supported(
            query,
            min_similarity,
            limit,
            calibration,
        )?;
        candidates
            .into_iter()
            .map(|candidate| self.bind_calibrated_candidate(candidate, query_binding))
            .collect()
    }

    fn bind_raw_candidate(
        &self,
        candidate: FullySupportedSemanticPriorCandidate,
        query_binding: SemanticSupportBinding,
    ) -> Result<IntegrityBoundSemanticPriorCandidate, IntegrityBoundSemanticPriorError> {
        let identity = candidate.candidate.identity;
        let source_binding = self
            .source_bindings
            .get(&identity)
            .copied()
            .ok_or(IntegrityBoundSemanticPriorError::MissingSourceBinding { identity })?;
        source_binding.validate_self()?;
        if source_binding.context_identity != identity
            || source_binding.support != candidate.candidate_source_support
            || query_binding.support != candidate.query_support
        {
            return Err(IntegrityBoundSemanticPriorError::BindingDrift { identity });
        }
        Ok(IntegrityBoundSemanticPriorCandidate {
            candidate,
            query_binding,
            candidate_source_binding: source_binding,
        })
    }

    fn bind_calibrated_candidate(
        &self,
        candidate: FullySupportedCalibratedSemanticPriorCandidate,
        query_binding: SemanticSupportBinding,
    ) -> Result<IntegrityBoundCalibratedSemanticPriorCandidate, IntegrityBoundSemanticPriorError> {
        let identity = candidate.candidate.candidate.identity;
        let source_binding = self
            .source_bindings
            .get(&identity)
            .copied()
            .ok_or(IntegrityBoundSemanticPriorError::MissingSourceBinding { identity })?;
        source_binding.validate_self()?;
        if source_binding.context_identity != identity
            || source_binding.support != candidate.candidate_source_support
            || query_binding.support != candidate.query_support
        {
            return Err(IntegrityBoundSemanticPriorError::BindingDrift { identity });
        }
        Ok(IntegrityBoundCalibratedSemanticPriorCandidate {
            candidate,
            query_binding,
            candidate_source_binding: source_binding,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::dream_feedback::hash_context;

    fn prior(context: &[f32]) -> ActionPrior {
        ActionPrior {
            context_hash: hash_context(context),
            preferred_direction: vec![0.2, -0.3, 0.4],
            strength: 0.8,
            evidence_count: 1,
        }
    }

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

    #[test]
    fn binding_round_trips_against_exact_context_and_encoder() {
        let encoder = SemanticContextEncoder::default();
        let context = [0.2, -0.4, 0.9];
        let binding = SemanticSupportBinding::create(encoder, &context).unwrap();

        binding.validate_self().unwrap();
        binding.validate_against_context(encoder, &context).unwrap();
        assert!(binding.within_nominal_support());
    }

    #[test]
    fn support_shape_cannot_be_detached_to_another_context() {
        let encoder = SemanticContextEncoder::default();
        let left = SemanticSupportBinding::create(encoder, &[0.2, 0.3]).unwrap();
        let right = SemanticSupportBinding::create(encoder, &[0.3, 0.2]).unwrap();

        assert_eq!(left.support, right.support);
        assert_ne!(left.context_identity, right.context_identity);
        assert_ne!(left.binding_digest, right.binding_digest);
    }

    #[test]
    fn encoder_configuration_is_part_of_binding_identity() {
        let context = [0.2, -0.3];
        let narrow = SemanticSupportBinding::create(
            SemanticContextEncoder::new(64, 1.0).unwrap(),
            &context,
        )
        .unwrap();
        let wide = SemanticSupportBinding::create(
            SemanticContextEncoder::new(32, 2.0).unwrap(),
            &context,
        )
        .unwrap();

        assert_eq!(narrow.context_identity, wide.context_identity);
        assert_ne!(narrow.binding_digest, wide.binding_digest);
    }

    #[test]
    fn clamp_saturated_context_is_bound_without_becoming_supported() {
        let binding = SemanticSupportBinding::create(
            SemanticContextEncoder::default(),
            &[2.5, -0.1],
        )
        .unwrap();

        binding.validate_self().unwrap();
        assert!(!binding.within_nominal_support());
        assert_eq!(binding.support.clipped_value_count, 1);
    }

    #[test]
    fn integrity_bound_memory_returns_bound_receipts_for_both_sides() {
        let context = [0.2];
        let mut memory = IntegrityBoundSemanticPriorMemory::new();
        let identity = memory.index_prior(&context, &prior(&context)).unwrap();

        let candidates = memory.retrieve_integrity_bound(&context, 0.0, 1).unwrap();
        assert_eq!(candidates.len(), 1);
        let candidate = &candidates[0];
        assert_eq!(candidate.candidate.candidate.identity, identity);
        assert_eq!(candidate.query_binding.context_identity, identity);
        assert_eq!(candidate.candidate_source_binding.context_identity, identity);
        candidate.query_binding.validate_self().unwrap();
        candidate.candidate_source_binding.validate_self().unwrap();
    }

    #[test]
    fn integrity_bound_memory_retains_unsupported_source_but_strictly_excludes_it() {
        let context = [2.0];
        let mut memory = IntegrityBoundSemanticPriorMemory::new();
        let identity = memory.index_prior(&context, &prior(&context)).unwrap();
        let source_binding = memory.source_binding(identity).unwrap();

        assert!(!source_binding.within_nominal_support());
        assert!(memory.retrieve_integrity_bound(&[1.0], 0.0, 10).unwrap().is_empty());
    }
}
