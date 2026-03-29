// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Builder pattern for `CognitiveLoopService`.
//!
//! Extracted from `mod.rs` to reduce file size.

use super::{CognitiveLoopConfig, CognitiveLoopService, TemporalBackend};
use anyhow::Result;

/// Builder for configuring the cognitive loop service
pub struct CognitiveLoopBuilder {
    config: CognitiveLoopConfig,
}

impl CognitiveLoopBuilder {
    pub fn new() -> Self {
        Self {
            config: CognitiveLoopConfig::default(),
        }
    }

    pub fn with_cfc_neurons(mut self, neurons: usize) -> Self {
        self.config.cfc_config.num_neurons = neurons;
        self.config.cfc_config.input_dim = neurons; // Keep in sync for train_step
        self
    }

    /// Alias for backward compatibility
    pub fn with_ltc_neurons(self, neurons: usize) -> Self {
        self.with_cfc_neurons(neurons)
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.config.cfc_config.learning_rate = lr;
        self
    }

    pub fn with_delta_t(mut self, delta_t: f32) -> Self {
        self.config.cfc_config.delta_t = delta_t;
        self
    }

    pub fn with_prediction_horizons(mut self, horizons: Vec<f32>) -> Self {
        self.config.cfc_config.prediction_horizons = horizons;
        self
    }

    pub fn with_attention_lr(mut self, lr: f32) -> Self {
        self.config.encoder_config.attention_lr = lr;
        self
    }

    pub fn with_learning_threshold(mut self, threshold: f32) -> Self {
        self.config.learning_threshold = threshold;
        self
    }

    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }

    /// Enable causal discovery integration
    ///
    /// When enabled, the cognitive loop tracks (input, output) pairs and
    /// periodically runs causal discovery to weight attention based on
    /// discovered causal structure.
    pub fn with_causal_enhancement(mut self, enabled: bool) -> Self {
        self.config.causal_enhancement = enabled;
        self
    }

    /// Set the interval (in cycles) between causal discovery runs
    ///
    /// Lower values = more frequent discovery but higher compute cost.
    /// Default is 100 cycles.
    pub fn with_causal_discovery_interval(mut self, interval: usize) -> Self {
        self.config.causal_discovery_interval = interval;
        self
    }

    /// Set a genesis phrase for deterministic initialization.
    ///
    /// When set, all HDC vectors, network weights, and exploration randomness
    /// are derived from this phrase via SHAKE-256 domain separation, making
    /// the cognitive loop fully reproducible.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let loop_a = CognitiveLoopBuilder::new()
    ///     .with_genesis_phrase("We hold these truths...")
    ///     .build()?;
    ///
    /// let loop_b = CognitiveLoopBuilder::new()
    ///     .with_genesis_phrase("We hold these truths...")
    ///     .build()?;
    ///
    /// // loop_a and loop_b will produce identical outputs for identical inputs
    /// ```
    pub fn with_genesis_phrase(mut self, phrase: impl Into<String>) -> Self {
        self.config.genesis_phrase = Some(phrase.into());
        // Disable async training for determinism (training order matters)
        self.config.async_training = false;
        self
    }

    /// Alias for `with_genesis_phrase` using the term from the Genesis module.
    pub fn seeded(self, phrase: impl Into<String>) -> Self {
        self.with_genesis_phrase(phrase)
    }

    /// Set the temporal backend (CfC or HdcLtcUnified)
    pub fn with_temporal_backend(mut self, backend: TemporalBackend) -> Self {
        self.config.temporal_backend = backend;
        self
    }

    /// Enable or disable async training
    ///
    /// Note: When a genesis phrase is set, async training is automatically
    /// disabled to ensure determinism.
    pub fn with_async_training(mut self, enabled: bool) -> Self {
        // Only allow if no genesis phrase is set
        if self.config.genesis_phrase.is_none() {
            self.config.async_training = enabled;
        }
        self
    }

    pub fn build(self) -> Result<CognitiveLoopService> {
        CognitiveLoopService::new(self.config)
    }
}

impl Default for CognitiveLoopBuilder {
    fn default() -> Self {
        Self::new()
    }
}
