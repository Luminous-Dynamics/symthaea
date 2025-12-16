//! Week 12: Perception & Tool Creation - Giving Sophia Senses
//!
//! This module provides sensory capabilities for Sophia:
//! - Visual perception (images) - Basic feature extraction
//! - Semantic vision - Deep semantic understanding with SigLIP & Moondream
//! - OCR - Text extraction from images (rten + ocrs, Tesseract fallback)
//! - Code perception (understanding source code)
//! - Multi-modal integration - Unifying all senses in holographic space 🌟
//! - Enhanced proprioception (system state awareness)
//!
//! Foundation for tool usage and tool creation capabilities.

pub mod visual;
pub mod code;
pub mod semantic_vision;
pub mod ocr;
pub mod multi_modal;

pub use visual::{VisualCortex, VisualFeatures};
pub use code::{CodePerceptionCortex, ProjectStructure, RustCodeSemantics, CodeQualityAnalysis};
pub use semantic_vision::{
    SemanticVision, ImageEmbedding, ImageCaption, VqaResponse,
    SigLipModel, MoondreamModel, EmbeddingCache, CacheStats,
    SIGLIP_EMBEDDING_DIM,
};
pub use ocr::{
    OcrSystem, OcrResult, OcrWord, OcrMethod, ImageQuality,
    RustOcrEngine, TesseractEngine,
};
pub use multi_modal::{
    MultiModalIntegrator, MultiModalPerception, ModalityContribution,
    ModalityType, ModalityWeights,
};
