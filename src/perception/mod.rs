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
pub mod video;
pub mod social_trust;

// Audio perception - Speech recognition via symthaea-stt
pub mod audio;

// Model status registry for graceful degradation
pub mod model_status;
pub use model_status::{ModelLoadError, ModelRegistry};
// Note: ModelStatus is not re-exported here to avoid name collision with
// modern_embeddings::ModelStatus (behind neural-bridge feature). Access via
// perception::model_status::ModelStatus or use the qualified path.

// Semantic encoding - Text/embedding -> HDC projection
pub mod semantic_encoder;
pub use semantic_encoder::{SemanticEncoder, JLProjector, NGramEncoder};

// Neural Bridge - LLM activation -> HDC direct projection
pub mod neural_bridge;
pub use neural_bridge::NeuralBridge;

// Neural Bridge Consciousness Probe - Topological analysis of LLM representations
pub mod neural_bridge_consciousness_probe;
pub use neural_bridge_consciousness_probe::{
    ConsciousnessProbe, ConceptCorpus, Concept, ConceptProbeResult,
    ClassComparisonResult, ClassStatistics, ProbeConfig,
};

#[cfg(feature = "neural-bridge")]
pub use neural_bridge_consciousness_probe::ConsciousnessProbeV2;

// Layer-wise activation extraction for transformer models
#[cfg(feature = "neural-bridge")]
pub mod layer_extractor;

#[cfg(feature = "neural-bridge")]
pub use layer_extractor::{LayerExtractor, LayerActivation, AllLayerActivations, PoolingMethod};

// BERT-specific layer extraction (using candle-transformers native BERT)
#[cfg(feature = "neural-bridge")]
pub mod bert_layer_extractor;

#[cfg(feature = "neural-bridge")]
pub use bert_layer_extractor::{
    BertLayerExtractor, BertPreset, BertExtractorConfig,
    bert_extraction_status, BertExtractionStatus, print_bert_status,
};

// Phenomenal Content Detector - Detect phenomenal vs functional content
#[cfg(feature = "neural-bridge")]
pub mod phenomenal_detector;

#[cfg(feature = "neural-bridge")]
pub use phenomenal_detector::{
    PhenomenalDetector, DetectorConfig, DetectionMethod,
    PhenomenalClassification, PhenomenalAnalysis, ClassLabel,
    ComparisonResult, DocumentAnalysis, CalibrationResult,
    // Contrastive training types
    ContrastiveExample, ContrastiveExamples, ContrastiveCalibrationResult,
    ContrastiveEvaluation, ExampleEvaluation,
};

// Scaling Findings - Research findings on phenomenal discrimination scaling
// Documents optimal model sizes and angular separation mechanisms
pub mod scaling_findings;
pub use scaling_findings::{
    Architecture, OptimalModelConfig, ScalingMetrics, ScalingFindings,
    DiscriminationQuality, ModelRecommendation,
    get_optimal_model, get_all_optimal_configs, get_scaling_findings, recommend_model,
};

// Multi-Model Extractor Framework - Cross-architecture support
pub mod multi_model_extractor;
pub use multi_model_extractor::{
    ModelPreset, ModelConfig, ModelArchitecture, PoolingStrategy,
    ValidationStatus, all_validation_status, print_support_summary,
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
    // Core trait and types
    EmbeddingModel, LayerOutput,
    // Configuration - note: PoolingMethod conflicts with layer_extractor, use modern_embeddings::PoolingMethod
    EmbeddingConfig, ModelBackend,
    // ModelArchitecture conflicts with multi_model_extractor, use modern_embeddings::ModelArchitecture
    // Main interface
    UnifiedEmbedder, EmbedderStats,
    // H2 hypothesis testing
    PhenomenalLayerAnalyzer, LayerAnalysisResult, LayerScore, PhenomenalCorridor,
    // Model info
    ModelInfo, ModelStatus, all_model_info, print_model_summary,
    // HDC integration
    project_to_hv16, activation_to_hv16,
};

// Multi-modal integration (conditionally compiled)
#[cfg(feature = "full_perception")]
pub mod visual_cortex;
#[cfg(feature = "full_perception")]
pub mod semantic_vision;
#[cfg(feature = "full_perception")]
pub mod multi_modal;

#[cfg(feature = "full_perception")]
pub use visual_cortex::{VisualCortex, VisualCortexConfig, FeatureExtractionResult};
#[cfg(feature = "full_perception")]
pub use semantic_vision::{SemanticVision, VisionConfig, VisualFeatures, ImageEmbedding, ImageCaption, OcrSystem};
#[cfg(feature = "full_perception")]
pub use multi_modal::{MultiModalIntegrator, ModalityType};

#[cfg(feature = "full_perception")]
pub mod conscious_perception;

#[cfg(feature = "full_perception")]
pub mod resilience;
