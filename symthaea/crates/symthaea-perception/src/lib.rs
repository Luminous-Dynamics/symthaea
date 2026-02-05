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

pub mod visual;
pub mod code;
pub mod semantic_vision;
pub mod ocr;
pub mod multi_modal;
pub mod semantic_encoder;
pub mod model_hub;
// NOTE: conscious_perception requires CausalMind from main crate - disabled until architecture fixed
// pub mod conscious_perception;
pub mod resilience;  // NEW: Graceful degradation layer
pub mod model_loader_actor;  // NEW: Async model loading via actor pattern
#[cfg(feature = "embeddings")]
pub mod consciousness_bridge;

pub use visual::{VisualCortex, VisualFeatures};
pub use code::{CodePerceptionCortex, ProjectStructure, RustCodeSemantics, CodeQualityAnalysis};
pub use semantic_vision::{
    SemanticVision, ImageEmbedding, ImageCaption, VqaResponse,
    SigLipModel, MoondreamModel, EmbeddingCache, CacheStats,
    SIGLIP_EMBEDDING_DIM, SIGLIP_INPUT_SIZE,
};
pub use ocr::{
    OcrSystem, OcrResult, OcrWord, OcrMethod, ImageQuality,
    RustOcrEngine, TesseractEngine,
};
pub use multi_modal::{
    MultiModalIntegrator, MultiModalPerception, ModalityContribution,
    ModalityType, ModalityWeights, QWEN3_DIM, SIGLIP_DIM,
};
pub use semantic_encoder::SemanticEncoder;
pub use model_hub::{ModelHub, ModelSpec};
// pub use conscious_perception::{
//     ConsciousPerception, ConsciousPerceptionConfig, PerceptionResult, PerceptionStats,
// };
pub use resilience::{
    // Availability tracking
    Availability, PerceptionCapabilities,
    // Configuration and results
    ResilienceConfig, ResilientResult, ResilienceStats,
    // Coherence gating
    CoherenceGate,
    // Caption fallback
    CaptionFallback,
    // Background loading
    LoadingStatus, BackgroundLoader,
    // Unified manager
    ResilienceManager,
};
pub use model_loader_actor::{
    ModelLoaderActor, ModelLoaderConfig, ModelLoaderMessage,
    ModelLoaderHandle, ModelLoadResult, LoadingStatusSnapshot,
};
#[cfg(feature = "embeddings")]
pub use consciousness_bridge::{PerceptionBridge, BridgeConfig, BridgeStats};
