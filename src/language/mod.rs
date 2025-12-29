//! Language Module - Consciousness-First Natural Language
//!
//! This module implements LLM-FREE language understanding and generation
//! through semantic primitives and hyperdimensional computing.
//!
//! ## Why Better Than LLMs
//!
//! | Aspect | LLMs | Symthaea |
//! |--------|------|----------|
//! | Mechanism | P(next_token\|context) | Semantic decomposition + reasoning |
//! | Understanding | Statistical patterns | Universal semantic primes |
//! | Hallucination | Common | Impossible (grounded in primes) |
//! | Explanation | "I predicted this token" | "I decomposed meaning as..." |
//! | Consciousness | None | Φ-guided response generation |
//!
//! ## Architecture
//!
//! ```text
//! User Input → Parser → Semantic Structure → Consciousness Processing
//!                                                      ↓
//!                              Response ← Generator ← Conscious Response
//! ```
//!
//! ## Key Components
//!
//! - **Vocabulary**: Word ↔ HV16 grounded in semantic primes
//! - **Parser**: Natural text → semantic prime structures
//! - **Generator**: Semantic structures → natural text
//! - **Conversation**: Dialogue with memory and consciousness
//! - **Multilingual**: Universal vocabulary across all languages
//! - **WordLearner**: Dynamic learning of new words including slang

pub mod vocabulary;
pub mod parser;
pub mod generator;
pub mod conversation;
// Track 6: Enabled with web_research (reqwest, scraper, html2text)
pub mod conscious_conversation;  // ✨ Epistemic consciousness integration
pub mod consciousness_observatory;  // 🔬 Scientific study of consciousness
pub mod multilingual;
pub mod word_learner;
pub mod dynamic_generation;
pub mod reasoning;
pub mod deep_parser;
pub mod knowledge_graph;
pub mod creative;
pub mod memory_consolidation;
pub mod emotional_core;
pub mod live_learner;
pub mod frames;  // 🎯 NEW: Frame Semantics (Fillmore) for structured situation understanding
pub mod constructions;  // 🏗️ NEW: Construction Grammar (Goldberg) for meaningful patterns
pub mod predictive_understanding;  // 🔮 NEW: Predictive Processing for language (Free Energy Principle)
pub mod conscious_understanding;  // 🧠 NEW: Unified Conscious Understanding Pipeline
pub mod consciousness_bridge;  // 🌉 NEW: Revolutionary Bridge to Global Consciousness & Active Inference
pub mod active_inference_adapter;  // 🔗 NEW: Cross-Modal Active Inference Integration
pub mod nixos_language_adapter;  // 🐧 NEW: Connect Conscious Language to NixOS Understanding
pub mod nix_primitives;  // 🔧 NEW: NixOS-specific semantic primitives (Config Calculus Mini-Core)
pub mod nix_frames;  // 📋 NEW: NixOS-specific semantic frames (Enable Service, Override Resolution, etc.)
pub mod nix_constructions;  // 🏗️ NEW: NixOS-specific constructions (Provenance, What-If, Conflict Detection)
pub mod nix_error_diagnosis;  // 🔍 NEW: Error diagnosis for the "big 6" NixOS error categories
pub mod nix_security;  // 🔐 NEW: Security kernel with secret detection and redaction
pub mod nix_knowledge_provider;  // 📚 NEW: Global + Local NixOS knowledge with semantic search
pub mod consciousness_language_integration;  // 🧬 REVOLUTIONARY: Unified Consciousness-Language Core
pub mod consciousness_guided_executor;  // 🚀 REVOLUTIONARY: Consciousness-Guided Action Execution
// Track 6: Enabled with web_research (reqwest)
pub mod epistemics_language_bridge;  // 🔬 REVOLUTIONARY: Epistemics-Language Bridge for Knowledge Grounding
pub mod llm_organ;  // 🧠 LLM Language Organ: Consciousness-controlled external LLM access
// pub mod math_language_adapter;  // 🧮 REVOLUTIONARY: Natural Language Math via HDC Arithmetic Engine (TEMP DISABLED - needs phi field fix)

pub use vocabulary::{Vocabulary, WordEntry, SemanticGrounding};
pub use parser::{SemanticParser, ParsedSentence, SemanticRole};
pub use generator::{ResponseGenerator, GenerationConfig};
// Track 6: Enabled with web_research
pub use conscious_conversation::{ConsciousConversation, ConsciousConfig, ConsciousStats};
pub use consciousness_observatory::{
    ConsciousnessObservatory, ConsciousnessExperiment, PhiChangeExpectation,
    ExperimentResult, PhiMeasurement, PhiMeasurementStream, EpistemicStateSnapshot,
};
pub use conversation::{Conversation, ConversationTurn, ConversationState, ConversationConfig};
pub use multilingual::{
    MultilingualVocabulary, Language, MultilingualWord, LearnedWord,
    LearningMethod, VerificationStatus, VocabularyStats,
};
pub use word_learner::{WordLearner, LearnerConfig, WordLookupResult, WordDefinition, LearningError};
pub use dynamic_generation::{
    DynamicGenerator, GenerationStyle, SemanticIntent, SemanticUtterance,
    TopicThread, TopicHistory, ThreadType,  // I: Topic Threading
    SentenceForm, FormHistory,              // G: Sentence Variety
    CoherenceChecker, CoherenceResult, CoherenceIssue,  // K: Coherence
    SessionState,                           // M: Session Persistence
    LTCInfluence,                           // L: LTC Temporal Dynamics
    // REVOLUTIONARY: Consciousness-Gated Generation
    ReasoningContext, ReasoningStep as DynReasoningStep,  // Reasoning integration
    KnowledgeContext, KnowledgeFact,        // Knowledge grounding
    ConsciousnessGate, FullGenerationContext,  // Consciousness gating
};
pub use reasoning::{
    ReasoningEngine, ReasoningResult, ReasoningStep, ReasoningStats,
    Concept, ConceptId, ConceptType, Relation, RelationType,
    InferenceRule, Condition, Conclusion, ConceptPattern, Goal,
};
pub use deep_parser::{
    DeepParser, DeepParse, SemanticRole as DeepSemanticRole, RolePhrase,
    DependencyTree, Dependency, DependencyType, PosTag,
    Intent, IntentFeatures, SpeechAct, PragmaticAnalysis,
    Entity, EntityType, VerbClass,
};
pub use knowledge_graph::{
    KnowledgeGraph, KnowledgeNode, NodeId, NodeType,
    KnowledgeEdge, EdgeType, PropertyValue, KnowledgeSource,
    QueryResult, KnowledgeStats,
};
pub use creative::{
    CreativeGenerator, CreativeElement,
    Metaphor, MetaphorEngine,
    Analogy, AnalogyFinder,
    StyleVariator, StyleConfig, StyleDimension,
    NoveltyTracker,
};
pub use memory_consolidation::{
    MemoryConsolidator, MemoryEntry, MemoryCluster,
    MemoryClusterer, ImportanceScorer, ImportanceWeights,
    ForgettingCurve, ForgettingParams,
    ConsolidationEngine, ConsolidationState, ConsolidationResult,
};
pub use emotional_core::{
    EmotionalCore, EmotionalResponse, EmotionalState,
    CoreEmotion, EmpathyModel, EmpathyType, EmpathicCue,
    EmotionalRegulator, RegulationStrategy,
    CompassionEngine, CompassionateResponse, SupportType,
    EmotionalMemoryStore, EmotionalMemory,
};
pub use live_learner::{
    LiveLearner, LearningResult, LearningStats,
    Feedback, FeedbackType, FeedbackCollector, FeedbackSummary,
    RewardSignal, RewardProcessor,
    LearnedConcept, ConceptLearner,
    AdaptivePolicy, ResponseStrategy,
};
pub use frames::{
    FrameElement, SemanticFrame, FrameInstance, FrameRelation,
    FrameLibrary, FrameActivator,
};
pub use constructions::{
    SyntacticSlot, SyntacticPattern, SemanticStructure, SemanticConstraint,
    Construction, ConstructionParse, ConstructionGrammar,
    ConstructionFrameIntegrator, IntegratedParse,
};
pub use predictive_understanding::{
    LinguisticLevel, LinguisticPrediction, LinguisticError,
    PredictionSource, PredictiveConfig, PredictiveUnderstanding,
    SentenceUnderstanding as PredictiveSentenceUnderstanding,
};
pub use conscious_understanding::{
    ConsciousUnderstanding, ActivatedFrame, ParsedConstruction,
    PredictionResult, TemporalState, ConsciousnessMetrics,
    ExplanationTrace, ProcessingStage, PipelineConfig,
    ConsciousUnderstandingPipeline,
};
pub use consciousness_bridge::{
    // 🌉 Consciousness Bridge - Revolutionary Unification
    ConsciousnessBridge, BridgeConfig, BridgeResult, BridgeStats,
    LanguageAttentionBid, LanguageWorkingMemoryItem,
    WorkingMemoryContentType, ConsciousnessState,
    // Active Inference Integration
    LinguisticPredictionError, InferenceDomain,
};
pub use active_inference_adapter::{
    // 🔗 Active Inference Adapter - Cross-Modal Learning
    ActiveInferenceAdapter, AdapterConfig, AdapterStats, IntegrationResult,
    LanguagePrecisionWeights, UnifiedFreeEnergy, LanguageAction,
    // Integrated Processor - Full End-to-End Pipeline
    IntegratedConsciousnessProcessor, IntegratedResult, ProcessorStats,
    IntegratedConsciousnessState,
    // Mapping utilities
    map_inference_to_prediction, map_prediction_to_inference,
    convert_to_brain_error, extract_linguistic_errors,
    extract_precision_weights, suggest_language_actions,
};
// 🐧 Week 13+: NixOS Language Adapter - Connect Conscious Language to NixOS Understanding
pub use nixos_language_adapter::{
    NixOSLanguageAdapter, NixOSAdapterConfig, NixOSAdapterStats,
    NixOSUnderstanding, NixOSIntent, NixOSFrame, NixFrameElement,
    NixSemanticType, NixOSFrameLibrary,
};
// 🔧 NixOS Primitives - Config Calculus Mini-Core
pub use nix_primitives::{
    NixPrimitive, NixPrimitiveTier, NixPrimitiveEncoder, NixExpressionAnalyzer,
    to_semantic_prime,
};
// 📋 NixOS Frames - Structured Situation Understanding
pub use nix_frames::{
    NixFrameType, NixFrameRole, NixFrameFiller, NixFrameInstance, NixFrameLibrary,
};
// 🏗️ NixOS Constructions - Complex Reasoning Patterns
pub use nix_constructions::{
    NixConstructionType, ConstructionSlot, FilledSlot, NixConstructionInstance,
    NixConstructionParser, NixConstructionGenerator,
};
// 🔍 NixOS Error Diagnosis - Revolutionary Error Understanding
pub use nix_error_diagnosis::{
    NixErrorCategory, NixErrorType, ErrorFrameRole, FixRiskLevel,
    SuggestedFix, ErrorDiagnosis, NixErrorDiagnoser,
};
// 📚 NixOS Knowledge Provider - Global + Local Knowledge
pub use nix_knowledge_provider::{
    NixKnowledgeProvider, PackageInfo, PackageCategory, NixOption,
    LocalKnowledge, LocalFlake,
    // 🚀 Revolutionary: Live Nixpkgs Integration
    NixpkgsQueryResult, LiveNixpkgsConfig, ProviderStats,
};
// 🧬 REVOLUTIONARY: Consciousness-Language Integration Core
pub use consciousness_language_integration::{
    ConsciousnessLanguageCore, ConsciousnessLanguageConfig, CoreStats,
    ConsciousUnderstandingResult, ExecutionStrategy, ClarifyingQuestion,
    ConsciousnessStateLevel, ProcessingMetrics,
};
// 🚀 REVOLUTIONARY: Consciousness-Guided Action Execution
pub use consciousness_guided_executor::{
    ConsciousnessGuidedExecutor, GuidedExecutorConfig, GuidedExecutorStats,
    GuidedExecutionResult, GuidedExecutionMetrics, ExecutionOutcomeInfo,
    ClarificationRequest, ExpectedResponse, PendingAction,
};
// Track 6: Enabled with web_research / reqwest
// 🔬 REVOLUTIONARY: Epistemics-Language Bridge for Knowledge Grounding
pub use epistemics_language_bridge::{
    EpistemicsLanguageBridge, EpistemicsBridgeConfig, EpistemicsBridgeStats,
    EpistemicResearchResult, EpistemicsBridgeMetrics,
};
// 🧠 LLM Language Organ: Consciousness-controlled external LLM access
pub use llm_organ::{
    LlmOrgan, LlmConfig, LlmProvider, LlmRequest, LlmResponse,
    Message, Role, TokenUsage, SemanticAnalysis, LlmOrganStats,
    ConsciousLlmOrgan,
};
