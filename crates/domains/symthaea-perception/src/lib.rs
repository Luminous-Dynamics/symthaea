#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//
// ── STATUS: ARCHIVED (workspace-excluded) 2026-07-16 ────────────────────────
// Vision review P2.2 (VISION_PROJECTION_REVIEW_2026-07-15.md): this crate is
// a real SigLIP-SO400M / Moondream2 / ocrs ONNX stack with graceful
// degradation, but it has ZERO active consumers — its only reference is
// symthaea-foveation's non-default `perception` feature, which nothing in the
// workspace enables, and it duplicates the main crate's live src/perception/
// modules. Excluded from the workspace so builds stop compiling a dead
// island. TO ADOPT: enable foveation's `perception` feature from the main
// crate's `foveation` gate (RealVentralPipeline gets a live SigLIP ventral
// stream), dedup against src/perception/, and delete this banner.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::new_without_default)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::unwrap_or_default)]

//! # Perception Capabilities
//!
//! Provides sensory capabilities for Sophia with graceful degradation.
//!
//! ## Capability Modes
//!
//! Each capability operates in one of three modes depending on model availability:
//!
//! | Capability | Model | Full | Degraded | Stub (Fallback) |
//! |-----------|-------|------|----------|-----------------|
//! | OCR | rten + ocrs (~8MB) | Full text extraction | N/A | Empty `OcrResult` |
//! | Image embedding | SigLIP-SO400M (ONNX) | 768D semantic vector | N/A | Hash-based 768D vector |
//! | Image captioning | Moondream2 VQA | Natural language caption | N/A | Empty string |
//! | Text embedding | Qwen3-Embedding-0.6B (ONNX) | 1024D semantic vector | N/A | Hash-based 768D vector |
//!
//! Stubs produce deterministic outputs seeded from the input hash, ensuring
//! reproducibility without model files. Use `PerceptionHealth::capabilities()`
//! to query availability at runtime.
//!
//! ## Model Stack
//!
//! | Component | Model | Dim | Purpose |
//! |-----------|-------|-----|---------|
//! | Text Embedding | Qwen3-Embedding-0.6B | 1024D | Semantic grounding |
//! | Image Embedding | SigLIP-SO400M | 768D | Visual understanding |
//! | Captioning | Moondream2 | - | Image-to-text |
//!
//! ## Architecture
//!
//! ```text
//! Input (Image/Text/Code)
//!         │
//!         ▼
//! ┌─────────────────┐
//! │ Model Hub       │ ← Auto-download from HuggingFace
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ ONNX Inference  │ ← SigLIP/Qwen3 via ort
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ JL Projection   │ ← 768D/1024D → 16,384D HDC
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ HDC Space       │ ← Holographic consciousness
//! └─────────────────┘
//! ```
//!
//! Foundation for tool usage and tool creation capabilities.

#![allow(dead_code, unused_variables)]

// ============================================================================
// HDC Constants (local to perception crate)
// ============================================================================

/// Standard HDC dimension for holographic representations
pub const HDC_DIMENSION: usize = 16_384;

pub mod code;
pub mod model_hub;
pub mod multi_modal;
pub mod ocr;
pub mod semantic_encoder;
pub mod semantic_vision;
pub mod visual;
// conscious_perception was removed — requires CausalMind from main crate.
#[cfg(feature = "embeddings")]
pub mod consciousness_bridge;
pub mod model_loader_actor; // NEW: Async model loading via actor pattern
pub mod resilience; // NEW: Graceful degradation layer

pub use code::{CodePerceptionCortex, CodeQualityAnalysis, ProjectStructure, RustCodeSemantics};
pub use model_hub::{ModelHub, ModelSpec};
pub use multi_modal::{
    ModalityContribution, ModalityType, ModalityWeights, MultiModalIntegrator,
    MultiModalPerception, QWEN3_DIM, SIGLIP_DIM,
};
pub use ocr::{
    ImageQuality, OcrMethod, OcrResult, OcrSystem, OcrWord, RustOcrEngine, TesseractEngine,
};
pub use semantic_encoder::SemanticEncoder;
pub use semantic_vision::{
    CacheStats, EmbeddingCache, ImageCaption, ImageEmbedding, MoondreamModel, SIGLIP_EMBEDDING_DIM,
    SIGLIP_INPUT_SIZE, SemanticVision, SigLipModel, VqaResponse,
};
pub use visual::{VisualCortex, VisualFeatures};
// pub use conscious_perception::{
//     ConsciousPerception, ConsciousPerceptionConfig, PerceptionResult, PerceptionStats,
// };
#[cfg(feature = "embeddings")]
pub use consciousness_bridge::{BridgeConfig, BridgeStats, PerceptionBridge};
pub use model_loader_actor::{
    LoadingStatusSnapshot, ModelLoadResult, ModelLoaderActor, ModelLoaderConfig, ModelLoaderHandle,
    ModelLoaderMessage,
};
pub use resilience::{
    // Availability tracking
    Availability,
    BackgroundLoader,
    // Caption fallback
    CaptionFallback,
    // Coherence gating
    CoherenceGate,
    // Background loading
    LoadingStatus,
    PerceptionCapabilities,
    // Configuration and results
    ResilienceConfig,
    // Unified manager
    ResilienceManager,
    ResilienceStats,
    ResilientResult,
};

// Surgical compatibility modules for local cross-crate simulation tracking
pub mod brain {
    pub mod prefrontal {
        use std::sync::Arc;
        #[derive(Debug, Clone, Default)]
        pub struct AttentionBid;
        impl AttentionBid {
            pub fn new<A, B>(_name: A, _payload: B) -> Self {
                Self
            }
            pub fn with_salience(self, _salience: f32) -> Self {
                self
            }
            pub fn with_urgency(self, _urgency: f32) -> Self {
                self
            }
            pub fn with_tags(self, _tags: Vec<String>) -> Self {
                self
            }
            pub fn with_hdc_semantic<T>(self, _val: Option<Arc<T>>) -> Self {
                self
            }
        }
    }
}

pub mod embeddings {
    #[derive(Debug, Clone, Default)]
    pub struct Qwen3Config;
    #[derive(Debug, Clone, Default)]
    pub struct Qwen3EmbedderResult {
        pub embedding: Vec<f32>,
    }
    #[derive(Debug, Clone, Default)]
    pub struct Qwen3Embedder;
    impl Qwen3Embedder {
        pub fn new(_cfg: Qwen3Config) -> anyhow::Result<Self> {
            Ok(Self)
        }
        pub fn embed(&mut self, _text: &str) -> anyhow::Result<Qwen3EmbedderResult> {
            Ok(Qwen3EmbedderResult::default())
        }
    }
}

pub mod perception {
    pub use crate::multi_modal::*;
    pub use crate::semantic_vision::SemanticVision;
}
