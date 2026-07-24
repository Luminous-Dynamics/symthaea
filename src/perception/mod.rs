// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Perception - Multi-modal sensory processing
//!
//! Provides video capture, physiological signal processing, and
//! temporal feature extraction for consciousness-aware perception.
//!
//! ## Neural Bridge (LLM as Semantic Sensor)
//!
//! The [`neural_bridge`] module implements the "Hyperdimensional Probe" -
//! a trained linear projection that maps LLM internal activations directly
//! to Symthaea's 16,384-dimensional HDC space.
//!
//! Instead of treating the LLM as a "Conversation Partner" (Text -> Text),
//! we treat it as a "Semantic Sensor" (Activations -> Vectors).

pub mod physio;
pub mod social_trust;
pub mod video;

// Vision Manifold - Patch-based HDC video encoding with CfC temporal dynamics
#[cfg(feature = "vision-manifold")]
pub use symthaea_vision_manifold as vision_manifold;

// Audio perception - Speech recognition via symthaea-stt
pub mod audio;

// Live microphone capture → STT → ContinuousHV (feature-gated, opt-in).
// Runs on a background thread so the cognitive loop never blocks on audio I/O.
#[cfg(feature = "voice-stt-live")]
pub mod audio_stream;
#[cfg(feature = "voice-stt-live")]
pub use audio_stream::{MicCaptureConfig, MicCaptureHandle};

// Live phone-screen capture → vision frames for the loop's VisionBridge
// (feature-gated, opt-in). Background thread; loop never blocks on ADB.
#[cfg(feature = "phone")]
pub mod phone_stream;
#[cfg(feature = "phone")]
pub use phone_stream::{PhoneCaptureConfig, PhoneCaptureHandle};

// Physical-sensor fusion → ContinuousHV (feature-gated).
// Each sensor subtype is independently gated under `sensor-fusion`.
#[cfg(feature = "sensor-fusion")]
pub mod sensor_fusion;

// SHA-256 integrity verification for model weight files
pub mod model_integrity;

// Model status registry for graceful degradation
pub mod model_status;
pub use model_status::{ModelLoadError, ModelRegistry};
// Note: ModelStatus is not re-exported here to avoid name collision with
// modern_embeddings::ModelStatus (behind neural-bridge feature). Access via
// perception::model_status::ModelStatus or use the qualified path.

// Semantic encoding - Text/embedding -> HDC projection
pub mod semantic_encoder;
pub use semantic_encoder::{JLProjector, NGramEncoder, SemanticEncoder};

// Neural Bridge - LLM activation -> HDC direct projection
pub mod neural_bridge;
pub use neural_bridge::NeuralBridge;

// Neural Bridge Consciousness Probe - Topological analysis of LLM representations
pub mod neural_bridge_consciousness_probe;
pub use neural_bridge_consciousness_probe::{
    ClassComparisonResult, ClassStatistics, Concept, ConceptCorpus, ConceptProbeResult,
    ConsciousnessProbe, ProbeConfig,
};

#[cfg(feature = "neural-bridge")]
pub use neural_bridge_consciousness_probe::ConsciousnessProbeV2;

// Layer-wise activation extraction for transformer models
#[cfg(feature = "neural-bridge")]
pub mod layer_extractor;

#[cfg(feature = "neural-bridge")]
pub use layer_extractor::{AllLayerActivations, LayerActivation, LayerExtractor, PoolingMethod};

// BERT-specific layer extraction (using candle-transformers native BERT)
#[cfg(feature = "neural-bridge")]
pub mod bert_layer_extractor;

#[cfg(feature = "neural-bridge")]
pub use bert_layer_extractor::{
    BertExtractionStatus, BertExtractorConfig, BertLayerExtractor, BertPreset,
    bert_extraction_status, print_bert_status,
};

// Phenomenal Content Detector - Detect phenomenal vs functional content
#[cfg(feature = "neural-bridge")]
pub mod phenomenal_detector;

#[cfg(feature = "neural-bridge")]
pub use phenomenal_detector::{
    CalibrationResult,
    ClassLabel,
    ComparisonResult,
    ContrastiveCalibrationResult,
    ContrastiveEvaluation,
    // Contrastive training types
    ContrastiveExample,
    ContrastiveExamples,
    DetectionMethod,
    DetectorConfig,
    DocumentAnalysis,
    ExampleEvaluation,
    PhenomenalAnalysis,
    PhenomenalClassification,
    PhenomenalDetector,
};

// Scaling Findings - Research findings on phenomenal discrimination scaling
// Documents optimal model sizes and angular separation mechanisms
pub mod scaling_findings;
pub use scaling_findings::{
    Architecture, DiscriminationQuality, ModelRecommendation, OptimalModelConfig, ScalingFindings,
    ScalingMetrics, get_all_optimal_configs, get_optimal_model, get_scaling_findings,
    recommend_model,
};

// Multi-Model Extractor Framework - Cross-architecture support
pub mod multi_model_extractor;
pub use multi_model_extractor::{
    ModelArchitecture, ModelConfig, ModelPreset, PoolingStrategy, ValidationStatus,
    all_validation_status, print_support_summary,
};

// Epistemic Semantic Vectors - HDC with uncertainty metadata
pub mod epistemic_vector;
pub use epistemic_vector::{EpistemicSemanticVector, UncertaintySource};

// BGE-M3 Embedding Model (pure Rust via Candle)
#[cfg(feature = "neural-bridge")]
pub mod bge_m3;

// Neural Bridge v2 - Complete text → HDC pipeline
#[cfg(feature = "neural-bridge")]
pub mod neural_bridge_v2;

#[cfg(feature = "neural-bridge")]
pub use neural_bridge_v2::{NeuralBridgeV2, NeuralBridgeV2Builder, NeuralBridgeV2Config};

// Modern Embeddings - Unified interface for modern embedding models
// Replaces problematic BERT layer extraction with models that properly support
// intermediate layer access (BGE-M3 primary, ModernBERT secondary)
#[cfg(feature = "neural-bridge")]
pub mod modern_embeddings;

#[cfg(feature = "neural-bridge")]
pub use modern_embeddings::{
    EmbedderStats,
    // Configuration - note: PoolingMethod conflicts with layer_extractor, use modern_embeddings::PoolingMethod
    EmbeddingConfig,
    // Core trait and types
    EmbeddingModel,
    LayerAnalysisResult,
    LayerOutput,
    LayerScore,
    ModelBackend,
    // Model info
    ModelInfo,
    ModelStatus,
    PhenomenalCorridor,
    // H2 hypothesis testing
    PhenomenalLayerAnalyzer,
    // ModelArchitecture conflicts with multi_model_extractor, use modern_embeddings::ModelArchitecture
    // Main interface
    UnifiedEmbedder,
    activation_to_hv16,
    all_model_info,
    print_model_summary,
    // HDC integration
    project_to_hv16,
};

// DELETED 2026-07-16 (vision review P2.3): the `full_perception` byte-hash
// stratum — visual_cortex.rs (random-filter "hierarchical vision"),
// semantic_vision.rs (byte-histogram "embeddings" + fabricated captions with
// 0.9-1.0 confidence over hash noise), multi_modal.rs, and their only
// dependents conscious_perception.rs + resilience.rs. The feature was an
// orphan (nothing enabled it except --all-features) and every "vision"
// computation in it was confidence theater. Real replacements already exist:
// visual_features.rs (real CV below), the vision-manifold encoder, and the
// archived symthaea-perception crate's SigLIP stack (adopt path, review P2.2).

// Ported from crates/symthaea-perception (2026-02-06)
// Code semantic analysis (syn-based Rust parsing, project structure analysis)
pub mod code_perception;
// OCR with dual-engine approach (rten pure-Rust + Tesseract fallback)
pub mod ocr;
// Traditional CV feature extraction (brightness, edges, dominant colors)
pub mod visual_features;
