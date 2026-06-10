// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified Broca backend trait for language generation tiers.
//!
//! Three implementations exist:
//! - **BrocaLite** (always available): 512 vocab, 1024D, 12 channels
//! - **BrocaFull** (feature `broca-full`): 1024 vocab, 16384D, 24 channels
//! - **BrocaPipeline** (feature `broca-pipeline`): Trained checkpoint, full symthaea-broca
//!
//! This trait provides a common interface so SporeEngine can swap tiers
//! without changing calling code.

use crate::broca::GenerationResult;

/// Which Broca tier is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrocaTier {
    /// BrocaLite: 512 tokens, 1024D embeddings, element-wise recurrence.
    /// Always available, ~2MB, no checkpoint needed.
    Lite,
    /// BrocaFull: 1024 tokens, 16384D, 24-channel encoder with epistemic gating.
    /// Self-contained, requires `broca-full` feature.
    Full,
    /// BrocaPipeline: Production-grade, trained checkpoint (~335MB).
    /// Requires `broca-pipeline` feature + checkpoint file.
    Pipeline,
}

impl std::fmt::Display for BrocaTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BrocaTier::Lite => write!(f, "BrocaLite"),
            BrocaTier::Full => write!(f, "BrocaFull"),
            BrocaTier::Pipeline => write!(f, "BrocaPipeline"),
        }
    }
}

/// Consciousness state sufficient for any Broca tier to generate language.
///
/// This is the common denominator extracted from the three different
/// ThoughtChannels / parameter lists used by each tier.
#[derive(Debug, Clone)]
pub struct GenerationContext {
    /// Overall consciousness level [0, 1].
    pub consciousness_level: f32,
    /// Prediction error from the last cycle.
    pub prediction_error: f32,
    /// Eight Harmonies alignment [0, 1].
    pub harmony_alignment: f32,
    /// Neuromodulator levels [DA, NE, 5-HT, Oxytocin].
    pub neuromodulators: [f32; 4],
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Optional input text for contextual generation.
    pub input: Option<String>,
}

impl Default for GenerationContext {
    fn default() -> Self {
        Self {
            consciousness_level: 0.5,
            prediction_error: 0.1,
            harmony_alignment: 0.5,
            neuromodulators: [0.5, 0.5, 0.5, 0.5],
            max_tokens: 16,
            input: None,
        }
    }
}

/// Unified trait for all Broca language generation backends.
///
/// Each tier implements this trait, allowing SporeEngine to swap
/// implementations at runtime or compile-time without API changes.
pub trait BrocaBackend {
    /// Which tier this backend represents.
    fn tier(&self) -> BrocaTier;

    /// Vocabulary size of this backend.
    fn vocab_size(&self) -> usize;

    /// Whether this backend supports epistemic gating (hallucination prevention).
    fn supports_epistemic_gating(&self) -> bool;

    /// Whether this backend is ready to generate (e.g., checkpoint loaded).
    fn is_ready(&self) -> bool;

    /// Generate text from consciousness state.
    ///
    /// All tiers accept the same `GenerationContext` and return the same
    /// `GenerationResult`. Each tier internally maps the context to its
    /// native ThoughtChannels format.
    fn generate_from_context(&mut self, ctx: &GenerationContext) -> GenerationResult;

    /// Reset internal hidden state (e.g., between conversations).
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broca_tier_display() {
        assert_eq!(format!("{}", BrocaTier::Lite), "BrocaLite");
        assert_eq!(format!("{}", BrocaTier::Full), "BrocaFull");
        assert_eq!(format!("{}", BrocaTier::Pipeline), "BrocaPipeline");
    }

    #[test]
    fn generation_context_default_is_sane() {
        let ctx = GenerationContext::default();
        assert!(ctx.consciousness_level >= 0.0 && ctx.consciousness_level <= 1.0);
        assert!(ctx.max_tokens > 0);
        assert!(ctx.input.is_none());
    }
}