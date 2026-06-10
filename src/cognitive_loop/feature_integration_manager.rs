// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Feature Integration Manager — groups neural bridge, semantic encoder,
//! school bridge, causal consciousness, and physics integration into a single CLS field.

/// Consolidated manager for feature-gated integration subsystems.
///
/// Replaces 8 top-level CLS fields (all feature-gated or Optional):
/// - `neural_bridge` — pre-computed embedding → HDC (neural-bridge)
/// - `semantic_embedding_channel` — background Qwen3 encoding (semantic-encoder)
/// - `semantic_hdc_bridge` — JL projection for semantic → BinaryHV (semantic-encoder)
/// - `pending_semantic_rx` — pending response receiver (semantic-encoder)
/// - `last_semantic_continuous` — last 16,384D semantic projection (semantic-encoder)
/// - `school_bridge` — curriculum-aware learning (school_learning)
/// - `causal_consciousness` — HSIC causal attention
/// - `physics_integration` — physics analogy HDC search (physics-bridge)
pub struct FeatureIntegrationManager {
    /// Neural bridge for projecting pre-computed embeddings into HDC space.
    #[cfg(feature = "neural-bridge")]
    pub neural_bridge: Option<crate::perception::NeuralBridge>,

    /// Background embedding channel for semantic encoding.
    #[cfg(feature = "semantic-encoder")]
    pub semantic_embedding_channel: Option<symthaea_embeddings::channel::EmbeddingChannel>,

    /// JL projection bridge for semantic embeddings → BinaryHV.
    #[cfg(feature = "semantic-encoder")]
    pub semantic_hdc_bridge: Option<symthaea_embeddings::HdcBridge>,

    /// Pending response receiver from the semantic embedding channel.
    #[cfg(feature = "semantic-encoder")]
    pub pending_semantic_rx: std::sync::Mutex<
        Option<std::sync::mpsc::Receiver<symthaea_embeddings::channel::EmbedResponse>>,
    >,

    /// Last semantic embedding projected to continuous HDC space (16,384D).
    #[cfg(feature = "semantic-encoder")]
    pub last_semantic_continuous: Option<Vec<f32>>,

    /// School bridge for curriculum-aware learning recommendations.
    #[cfg(feature = "school_learning")]
    pub school_bridge: Option<crate::school::School>,

    /// Causal consciousness: HSIC-based causal attention weighting.
    pub causal_consciousness: Option<crate::intelligence::CausalConsciousness>,

    /// Physics bridge integration: HDC semantic search for physics analogies.
    #[cfg(feature = "physics-bridge")]
    pub physics_integration: Option<super::physics_integration::PhysicsIntegration>,

    /// Analogy engine: cross-domain concept analogy discovery.
    #[cfg(feature = "analogy-engine")]
    pub analogy_integration: Option<super::analogy_integration::AnalogyIntegration>,

    /// UCL cross-domain frame detection (TRADE, CONFLICT, etc.).
    #[cfg(feature = "ucl-frames")]
    pub ucl_frame_integration: Option<super::ucl_frame_integration::UCLFrameIntegration>,
}

impl std::fmt::Debug for FeatureIntegrationManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeatureIntegrationManager")
            .field("causal_consciousness", &self.causal_consciousness.is_some())
            .finish_non_exhaustive()
    }
}

impl FeatureIntegrationManager {
    /// Reset integration state for a new episode.
    pub fn reset(&mut self) {
        // Most fields retain learned state across episodes.
        #[cfg(feature = "semantic-encoder")]
        {
            self.last_semantic_continuous = None;
        }
    }
}
