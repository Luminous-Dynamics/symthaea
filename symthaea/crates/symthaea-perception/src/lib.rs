#![allow(clippy::needless_range_loop)]
#![allow(clippy::new_without_default)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::unwrap_or_default)]

//! Week 12: Perception & Tool Creation - Giving Sophia Senses
//!
//! This module provides sensory capabilities for Sophia:
//!
//! - Visual perception (images) - Basic feature extraction
//! - Semantic vision - Deep semantic understanding with SigLIP & Moondream
//! - OCR - Text extraction from images (rten + ocrs, Tesseract fallback)
//! - Code perception (understanding source code)
//! - Multi-modal integration - Unifying all senses in holographic space
//! - Model Hub - HuggingFace model downloading and management
//! - Enhanced proprioception (system state awareness)
//!
//! ## Model Stack (2025)
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
    CacheStats, EmbeddingCache, ImageCaption, ImageEmbedding, MoondreamModel, SemanticVision,
    SigLipModel, VqaResponse, SIGLIP_EMBEDDING_DIM, SIGLIP_INPUT_SIZE,
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
