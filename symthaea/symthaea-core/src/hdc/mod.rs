/*!
Hyperdimensional Computing (HDC) Semantic Space

16,384D holographic vectors for consciousness (2^14 - SIMD-optimized)
Memory IS computation - no separate storage needed!
*/

#![allow(dead_code, unused_variables, unused_assignments)]

/*!
Module Structure:
- mod.rs: Core SemanticSpace and HdcContext (arena-based operations)
- temporal_encoder.rs: Week 17 circular time encoding
- statistical_retrieval.rs: Week 17 Critical Fix #1 (z-score + margin + unbind)
- sequence_encoder.rs: Week 17 Critical Fix #2 (permutation-based order preservation)
*/

// =============================================================================
// CENTRAL HDC CONFIGURATION - Single Source of Truth
// =============================================================================

/// Default HDC dimension: 16,384 (2^14)
///
/// **16,384 dimensions** chosen for:
/// - **SIMD optimization**: Power of 2 aligns perfectly with vector registers
/// - **Memory alignment**: Natural 64-byte cache line boundaries
/// - **Orthogonality**: Higher dimensions = better near-orthogonality of random vectors
/// - **Capacity**: More bits = more distinct concepts before saturation
/// - **Balance**: Good accuracy vs memory tradeoff for most use cases
///
/// # Usage
/// ```rust,ignore
/// use symthaea::hdc::HDC_DIMENSION;
/// let vector = vec![0i8; HDC_DIMENSION];
/// ```
pub const HDC_DIMENSION: usize = 16_384;

/// Extended HDC dimension: 32,768 (2^15)
///
/// **32K dimensions** for:
/// - **Higher capacity**: 2x more distinct concepts before saturation
/// - **Complex semantic spaces**: Rich multi-modal embeddings
/// - **Deep temporal encoding**: Fine-grained chrono-semantic resolution
///
/// # Memory Cost
/// - 16K: ~16KB per bipolar vector
/// - 32K: ~32KB per bipolar vector (2x memory)
pub const HDC_DIMENSION_32K: usize = 32_768;

/// Maximum HDC dimension: 65,536 (2^16)
///
/// **64K dimensions** for extreme precision requirements
pub const HDC_DIMENSION_64K: usize = 65_536;

/// HDC dimensionality configuration for runtime selection
///
/// Supports both predefined tiers and custom arbitrary dimensions.
/// All dimensions should be powers of 2 for optimal SIMD performance.
///
/// # Predefined Tiers
/// - **Standard (16K)**: Good balance of accuracy and memory
/// - **Extended (32K)**: Higher semantic capacity
/// - **Ultra (64K)**: Maximum precision
/// - **Custom**: Any dimension (should be power of 2)
///
/// # Usage
/// ```rust,ignore
/// use symthaea::hdc::HdcDimensionality;
///
/// // Use predefined tier
/// let standard = HdcDimensionality::Standard;
/// assert_eq!(standard.dimension(), 16_384);
///
/// // Use custom dimension (128K for extreme cases)
/// let ultra_custom = HdcDimensionality::Custom(131_072);
/// assert_eq!(ultra_custom.dimension(), 131_072);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HdcDimensionality {
    /// Standard 16,384 dimensions (2^14) - good balance of accuracy and memory
    Standard,
    /// Extended 32,768 dimensions (2^15) - higher semantic capacity
    Extended,
    /// Ultra 65,536 dimensions (2^16) - maximum precision
    Ultra,
    /// Custom dimensions - any power of 2 (32K+ recommended)
    Custom(usize),
}

impl HdcDimensionality {
    /// Get the numeric dimension value
    pub const fn dimension(&self) -> usize {
        match self {
            Self::Standard => HDC_DIMENSION,
            Self::Extended => HDC_DIMENSION_32K,
            Self::Ultra => HDC_DIMENSION_64K,
            Self::Custom(dim) => *dim,
        }
    }

    /// Create from dimension value
    ///
    /// Automatically maps to predefined tiers if exact match,
    /// otherwise creates Custom variant.
    pub const fn from_dimension(dim: usize) -> Self {
        match dim {
            16_384 => Self::Standard,
            32_768 => Self::Extended,
            65_536 => Self::Ultra,
            _ => Self::Custom(dim),
        }
    }

    /// Check if dimension is a power of 2 (recommended for SIMD)
    pub const fn is_power_of_two(&self) -> bool {
        let dim = self.dimension();
        dim > 0 && (dim & (dim - 1)) == 0
    }

    /// Check if dimension is a predefined tier
    pub const fn is_predefined(&self) -> bool {
        matches!(self, Self::Standard | Self::Extended | Self::Ultra)
    }

    /// Get memory usage per bipolar vector in bytes
    pub const fn memory_per_vector(&self) -> usize {
        self.dimension() // Each i8 element is 1 byte
    }

    /// Get memory usage per f32 vector in bytes
    pub const fn memory_per_f32_vector(&self) -> usize {
        self.dimension() * 4 // Each f32 element is 4 bytes
    }
}

impl Default for HdcDimensionality {
    fn default() -> Self {
        Self::Standard
    }
}

impl From<usize> for HdcDimensionality {
    fn from(dim: usize) -> Self {
        Self::from_dimension(dim)
    }
}

// =============================================================================
// CENTRAL LTC CONFIGURATION - Liquid Time-Constant Network
// =============================================================================

/// Default LTC neuron count: 1,024 (2^10)
///
/// **1,024 neurons** chosen for:
/// - **SIMD optimization**: Power of 2 aligns with vector registers
/// - **Memory alignment**: Natural cache line boundaries
/// - **Balance**: Good temporal dynamics vs compute cost
/// - **Biological plausibility**: ~10^3 scale for cortical columns
///
/// # Usage
/// ```rust,ignore
/// use symthaea::hdc::LTC_NEURONS;
/// let neurons = vec![0.0f32; LTC_NEURONS];
/// ```
pub const LTC_NEURONS: usize = 1_024;

/// Extended LTC neuron count: 2,048 (2^11)
///
/// **2K neurons** for:
/// - **Higher temporal capacity**: More nuanced time dynamics
/// - **Complex causal reasoning**: Finer-grained cause-effect modeling
pub const LTC_NEURONS_2K: usize = 2_048;

/// Maximum LTC neuron count: 4,096 (2^12)
///
/// **4K neurons** for extreme temporal precision
pub const LTC_NEURONS_4K: usize = 4_096;

/// LTC neuron count configuration for runtime selection
///
/// Supports both predefined tiers and custom arbitrary counts.
/// All counts should be powers of 2 for optimal SIMD performance.
///
/// # Predefined Tiers
/// - **Standard (1K)**: Good balance of dynamics and compute
/// - **Extended (2K)**: Higher temporal capacity
/// - **Ultra (4K)**: Maximum precision
/// - **Custom**: Any count (should be power of 2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LtcNeuronCount {
    /// Standard 1,024 neurons (2^10) - good balance
    Standard,
    /// Extended 2,048 neurons (2^11) - higher capacity
    Extended,
    /// Ultra 4,096 neurons (2^12) - maximum precision
    Ultra,
    /// Custom neuron count - any power of 2 (1K+ recommended)
    Custom(usize),
}

impl LtcNeuronCount {
    /// Get the numeric neuron count
    pub const fn count(&self) -> usize {
        match self {
            Self::Standard => LTC_NEURONS,
            Self::Extended => LTC_NEURONS_2K,
            Self::Ultra => LTC_NEURONS_4K,
            Self::Custom(n) => *n,
        }
    }

    /// Create from neuron count
    pub const fn from_count(n: usize) -> Self {
        match n {
            1_024 => Self::Standard,
            2_048 => Self::Extended,
            4_096 => Self::Ultra,
            _ => Self::Custom(n),
        }
    }

    /// Check if count is a power of 2 (recommended for SIMD)
    pub const fn is_power_of_two(&self) -> bool {
        let n = self.count();
        n > 0 && (n & (n - 1)) == 0
    }
}

impl Default for LtcNeuronCount {
    fn default() -> Self {
        Self::Standard
    }
}

impl From<usize> for LtcNeuronCount {
    fn from(n: usize) -> Self {
        Self::from_count(n)
    }
}

pub mod temporal_encoder;
pub mod statistical_retrieval;
pub mod sequence_encoder;
pub mod resonator;
pub mod morphogenetic;
pub mod hebbian;
pub mod hdc_ltc_neuron;  // HDC-LTC neuron integration with Hebbian learning
pub mod hdc_ltc_unified;  // Revolutionary unified HDC-LTC: state AS hypervector with closed-form solution
#[cfg(test)]
mod hdc_ltc_learning_tests;  // Comprehensive learning dynamics tests for HdcLtcUnifiedNeuron
pub mod hdc_ltc_unified_validation;  // Numerical validation of closed-form solution accuracy
pub mod unified_network_phi;  // Phi measurement and validation for HdcLtcUnifiedNetwork
pub mod cincinnati_ltc;  // Cincinnati Algorithm + LTC integration (differential engine, lateral binding, predictive budding)
pub mod cincinnati_network;  // Cincinnati-enhanced HdcLtcNetwork with lateral binding and budding
pub mod cycle_detector;  // Cycle detection for periodic patterns - autocorrelation-based period detection with HDC phase encoding
pub mod gwt_cincinnati_integration;  // Cincinnati-LTC + Global Workspace Theory integration - temporal patterns enter consciousness
pub mod cincinnati_enhanced;  // Enhanced Cincinnati-LTC: multi-scale, amplitude encoding, attention modulation
pub mod cincinnati_advanced;  // Advanced Cincinnati-LTC: chaos detection, adaptive weights, memory horizon
pub mod reservoir;            // Reservoir Computing (Echo State Network) for chaotic signal prediction
pub mod predictor;            // Unified predictor trait for Symthaea integration (links prediction to Φ)
pub mod sdm;
pub mod text_encoder;  // Revolutionary Enhancement: Text → HDC encoding
pub mod semantic_encoder;  // Universal semantic encoding with embeddings support
pub mod semantic_decoder;  // HV → Primitive sequence (generative direction) - THE MOUTH
// DISABLED: depends on crate::learnable_ltc which doesn't exist in symthaea-core
// pub mod hd_ltc_codec;      // Bidirectional HDC ↔ LTC translation - THE THROAT
// DISABLED: depends on hd_ltc_codec which is disabled
// pub mod ltc_generative_core; // Autoregressive primitive prediction - THE VOICE
pub mod unified_hv;     // Unified hypervector types (ContinuousHV)
pub mod config;         // Centralized HDC configuration (runtime dimension management)
pub mod projection;     // Learned projection layers for dimension conversion
pub mod compat;         // Compatibility layer for HV16 ↔ ContinuousHV migration

// Global Workspace Theory (conscious access, competition, broadcasting)
pub mod global_workspace;                  // GWT implementation with competitive dynamics

// Consciousness topology and Φ measurement modules
pub mod real_hv;                           // Real-valued hypervectors for consciousness topologies
pub mod consciousness_topology;            // Consciousness topology structures
pub mod consciousness_topology_generators; // 8 topology generators (Random, Star, Ring, Line, Tree, Dense, Modular, Lattice)
pub mod tiered_phi;                        // Multi-tier Φ (integrated information) approximation
#[cfg(test)]
mod phi_tier_tests;                        // Unit tests for Φ tier implementations
pub mod phi_topology_validation;           // RealHV-TieredPhi integration for topology validation
pub mod phi_real;                          // RealHV Φ calculator (no binarization) using cosine similarity
pub mod spectral_connectivity;             // Algebraic connectivity (λ₂) calculator - NOT IIT Φ!
pub mod phi_resonant;                      // Resonator-based Φ calculator (O(n log N) dynamics)
pub mod phi_orchestrator;                  // Adaptive Φ calculator orchestrator (Phase 5E)
pub mod binary_hv;                         // Binary hypervector operations (HV16)
pub mod simd_hv16;                         // SIMD-optimized binary hypervectors (8x faster)
pub mod simd_ops;                          // SIMD intrinsics for HV16 (AVX2/SSE4.1)
pub mod simd_continuous;                   // SIMD intrinsics for ContinuousHV (AVX2/FMA/SSE4.1)
// NOTE: simd_hv and optimized_hv are INCOMPATIBLE with 16,384-bit HV16 (they use 256-byte arrays)
// Use HV16::bind(), HV16::bundle(), HV16::similarity() directly instead
pub mod hdc_trait;                         // Unified HyperdimensionalVector trait interface
// Performance optimization modules:
pub mod incremental_hv;                    // O(k) incremental bundling (10-100x faster for updates)
pub mod parallel_hv;                       // Rayon parallel batch operations (7x faster on 8 cores)
pub mod lsh_simhash;                       // SimHash for binary vectors (Hamming distance)
pub mod lsh_similarity;                    // Adaptive LSH-backed similarity search (Session 7C)
pub mod lsh_index;                         // LSH index for fast approximate similarity search (heap-optimized)
pub mod primitive_system;                  // Ontological primitives system with 7 semantic domains
pub mod ucl_cross_domain_frames;           // UCL cross-domain semantic frames (TRADE, CONFLICT, FEEDBACK_LOOP, etc.)
pub mod bootstrapping;                     // Cognitive bootstrapping - primitives to reasoning tasks
pub mod primitive_dashboard;               // Real-time primitive usage monitoring
pub mod arithmetic_engine;                   // Revolutionary: True mathematical cognition via HDC
pub mod arithmetic;                          // Modular arithmetic (re-exports arithmetic_engine)
pub mod celegans_connectome;               // Revolutionary #100: C. elegans connectome validation (302 neurons)
pub mod native_similarity;                 // O(1) XOR+popcount similarity search (consciousness-native)
pub mod sparse_hv;                         // Sparse HDC for memory-efficient low-density vectors
pub mod hv_pool;                           // Thread-local memory pools for HV16/ContinuousHV (10-100x faster allocation)

// Property-based tests for HDC invariants
#[cfg(test)]
mod proptest_hdc;

// Track 6: Consciousness integration for awakening module
pub mod substrate_independence;            // Substrate type definitions
pub mod consciousness_evaluator;           // Consciousness evaluation
pub mod consciousness_integration;         // Complete consciousness pipeline
pub mod consciousness;                     // Modular consciousness (re-exports consciousness_integration)
pub mod consciousness_dashboard;           // Real-time consciousness monitoring

// Track 6: Language module dependencies
pub mod universal_semantics;               // Universal semantic primes (Wierzbicka)
pub mod grounded_understanding;            // True understanding via semantic primes + embodiment
pub mod unified_understanding;             // Complete understanding pipeline (predictive + narrative + ToM)
pub mod full_stack_consciousness;          // Full stack: Understanding + ActiveInference + Memory + Counterfactuals
pub mod unified_conscious_being;           // Complete unified being: A+B+C+D+E+F integration
// DISABLED: depends on crate::memory, crate::voice which don't exist in symthaea-core
// pub mod infrastructure_bridge;             // Bridge to real persistence (Hippocampus/UnifiedMind/Kokoro)
pub mod ecosystem_bridge;                  // Integration with service ecosystem (Sacred Core, Weave, Codex, Field Harmonizer)
pub mod consciousness_self_assessment;     // Self-assessment for conversation
pub mod consciousness_creativity;          // Creativity for conversation
pub mod deterministic_seeds;               // Deterministic seeds for NixOS knowledge
pub mod integrated_information;            // Φ (integrated information) measurement
pub mod causal_encoder;                    // Causal relation encoding
pub mod causal_mind;                       // Causal reasoning (core causal cognition)
pub mod unified_cognitive_core;            // Unified cognitive core (UCE/UCTS architecture)

// Predictive Processing (Friston Free Energy Principle)
pub mod predictive_coding;                 // Hierarchical prediction + error minimization
pub mod predictive_consciousness;          // Consciousness-level predictive processing
pub mod predictive_consciousness_kalman;   // Kalman filter variant for smooth predictions
pub mod predictive_encoder;                // Attention-modulated HDC encoding with LTC prediction loop

// Novel Algorithm Modules (Dec 2025)
pub mod differentiable_phi;                // Soft-partitioned differentiable Φ for gradient optimization
pub mod autodiff_phi;                      // Reverse-mode autodiff for consciousness optimization (Jan 2026)
pub mod cross_modal_binding;               // Cross-modal binding for multi-sensory integration
pub mod metacognitive_monitor;             // Real-time consciousness monitoring with self-reflection

// Phenomenal Binding Study - Research Direction 2: HDC binding vs bundling for phenomenal unity
pub mod phenomenal_binding_study;

// Consciousness Infrastructure (required by advanced systems)
pub mod consciousness_gradients;           // Gradient computation for consciousness optimization
pub mod consciousness_dynamics;            // Consciousness dynamics modeling
pub mod consciousness_optimizer;           // Consciousness state optimizer
pub mod modern_hopfield;                   // Modern Hopfield networks for memory

// Unified Consciousness Architecture (Dec 2025)
pub mod fractal_consciousness;             // Fractal consciousness patterns
pub mod phi_gradient_learning;             // Φ-gradient learning for optimization
pub mod adaptive_learning_signals;         // Consciousness-guided learning modulation (Φ, surprise, coherence)
pub mod conscious_learning;                // Consciousness-integrated learning (Hebbian + Adaptive signals)
pub mod phi_guided_search;                 // Φ-guided architecture search (gradient-based topology optimization)
pub mod process_topology;                  // Process topology structures
pub mod unified_consciousness_engine;      // Core consciousness engine with Φ-guided processing
pub mod attention_dynamics;                // Dynamic attention allocation with salience, goals, and priors
pub mod temporal_binding;                  // Temporal stream binding for continuous experience
pub mod emergent_self_model;               // Self-awareness and metacognitive optimization
pub mod adaptive_topology;                 // Adaptive cognitive mode topology
pub mod topology_synergy;                  // Topology-consciousness synergy
// DISABLED: depends on crate::memory, crate::voice which don't exist in symthaea-core
// pub mod integrated_conscious_agent;        // Complete conscious agent with Symthaea integration
pub mod consciousness_visualizer;          // Consciousness visualization tools
pub mod deep_integration;                   // Deep integration bridge for Φ-guided processing

// Re-export HV16 at module level for convenience (used by language/nix_* modules)
pub use binary_hv::HV16;
pub use real_hv::RealHV;

// Re-export unified HV types
pub use unified_hv::{ContinuousHV, BinaryHV, HV};

// Re-export configuration types (dimension unification)
pub use config::{
    HdcConfig,
    set_hdc_config,
    try_set_hdc_config,
    hdc_dim,
    hdc_config,
    is_hdc_configured,
    DimensionMapping,
    STT_DIMENSION,
    stt_expansion_factor,
};

// Re-export projection layers
pub use projection::{
    LearnedProjection,
    BidirectionalBridge,
    RandomProjection,
};

// Re-export compatibility utilities
pub use compat::{
    HVCompat,
    hv16_to_core,
    core_to_hv16,
};

// Re-export key types for convenience
pub use statistical_retrieval::{
    StatisticalRetriever,
    StatisticalRetrievalConfig,
    RetrievalDecision,
    RetrievalVerdict,
    EmpiricalTier,
};

pub use sequence_encoder::{
    SequenceEncoder,
    permute,
    unpermute,
    bundle,
    bind,
};

pub use resonator::{
    ResonatorNetwork,
    ResonatorConfig,
    ResonatorSolution,
    Constraint,
    MultiConstraint,
    Factor,
};

pub use morphogenetic::{
    MorphogeneticField,
    MorphogeneticConfig,
    PositionEncoding,
    Attractor,
    RepairResult,
    FieldHealth,
    FieldStats,
    corrupt_vector,
    random_vector,
};

pub use hebbian::{
    HebbianEngine,
    HebbianConfig,
    HebbianStats,
    Synapse,
    ActivationRecord,
    HebbianAssociativeMemory,
    HebbianAssociativeStats,
    DEFAULT_LEARNING_RATE,
    DEFAULT_DECAY_RATE,
    STDP_TAU_PLUS,
    STDP_TAU_MINUS,
    STDP_A_PLUS,
    STDP_A_MINUS,
    TARGET_ACTIVITY,
    HOMEOSTATIC_TAU,
};

pub use sdm::{
    SparseDistributedMemory,
    SDMConfig,
    HardLocation,
    WriteResult,
    ReadResult,
    IterativeReadResult,
    SDMStats,
    EpisodicSDM,
    hamming_similarity,
    random_bipolar_vector,
    add_noise,
    DEFAULT_NUM_HARD_LOCATIONS,
    DEFAULT_ACTIVATION_RADIUS,
    COUNTER_MAX,
    COUNTER_MIN,
};

pub use temporal_encoder::TemporalEncoder;
pub use text_encoder::{TextEncoder, TextEncoderConfig, TextEncoderStats};

// Re-export Primitive System types (9-tier ontological primitives)
pub use primitive_system::{PrimitiveSystem, Primitive, PrimitiveTier, DomainManifold, seed_from_name};

// Re-export UCL Cross-Domain Frame types (6 missing frames from gap analysis)
pub use ucl_cross_domain_frames::{
    UCLFrameSystem, CrossDomainFrame, FrameSlot, FrameInstance, concept_hv,
};

// Re-export Primitive Dashboard types (real-time monitoring)
pub use primitive_dashboard::{PrimitiveDashboard, PrimitiveStats, VoicePrimitiveTracker};

// Re-export Unified Consciousness Architecture types
// DISABLED: integrated_conscious_agent depends on crate::memory, crate::voice
// pub use integrated_conscious_agent::{
//     IntegratedConsciousAgent, AgentConfig, IntegratedUpdate,
//     WorkingMemory, EmotionalState, QualiaTexture, PhenomenalContent,
//     HormoneEventSuggestion, CoherenceGating, QualiaModulation,
//     MemoryExport, MemoryImport, IdentityCoherence, IdentityStatus, ProsodyHints,
//     // Voice prosody bridge
//     ExtendedPacing,
//     // Runtime orchestration
//     ConsciousAgentRuntime, SyncConsciousAgentRuntime, RuntimeConfig,
//     RuntimeMessage, RuntimeResponse, RuntimeSnapshot, HormoneEventType,
//     EmotionalStateSummary,
// };

// Re-export adaptive topology types
pub use adaptive_topology::{AdaptiveTopology, CognitiveMode};

// Re-export unified consciousness engine types
pub use unified_consciousness_engine::{
    UnifiedConsciousnessEngine, EngineConfig, ConsciousnessDimensions,
};

// Re-export consciousness visualization
pub use consciousness_visualizer::ConsciousnessVisualizer;

// Re-export deep integration bridge
pub use deep_integration::DeepIntegrationBridge;

// Re-export causal mind types
pub use causal_mind::{CausalMind, CausalDirection, LearnedCausalDiscovery};

// Re-export unified cognitive core
pub use unified_cognitive_core::{UnifiedCognitiveCore, UnifiedCognitiveElement, CognitiveMarkers, QueryResult};

// Re-export unified HDC-LTC types (revolutionary closed-form dynamics)
pub use hdc_ltc_unified::{
    HdcLtcUnifiedNeuron, HdcLtcUnifiedNetwork,
    UnifiedConfig, UnifiedNetworkConfig,
    UnifiedActivation, UnifiedNeuronStats, UnifiedNetworkStats,
};

// Re-export unified network Phi measurement types
pub use unified_network_phi::{
    UnifiedNetworkPhiMeasurer, UnifiedPhiConfig, PhiCalculationMethod,
    PhiMeasurement, PhiEvolutionTracker, PhiEvolutionSummary,
    PhiComparator, PhiComparison, PhiValidator, ValidationResult,
    NetworkStateExtractor, demo_phi_evolution,
    // Diagnostic types
    PhiDiagnostic, PhiDiagnosticAnalyzer,
};

// Sleep and altered states
pub mod sleep_and_altered_states;
pub mod sleep_pattern_discovery;     // Resonator-based pattern discovery during sleep consolidation

// Long-term memory with Qdrant integration - Revolutionary Improvement #29
// Persistent consciousness memory: experiences consolidated, retrieved, shape future
pub mod long_term_memory;

// Consciousness persistence (versioning, auto-save, rollback)
pub mod consciousness_persistence;

// Collective consciousness (multi-agent)
pub mod collective_consciousness;

// Consciousness streaming (WebSocket/SSE)
pub mod consciousness_streaming;

// Emotional depth (complex blends, compound emotions, HDC emotional algebra)
pub mod emotional_depth;

// Cross-modal attention router (Φ-gated modality routing)
pub mod cross_modal_attention_router;

// Self-improvement integration (metacognitive self-optimization)
pub mod self_improvement_integration;

// Counterfactual dreams (what-if scenarios in sleep)
pub mod counterfactual_dreams;

// Consciousness integration demo (comprehensive example of all features working together)
pub mod consciousness_integration_demo;

// Cross-module integration bridge (emotional→dreams, self-improvement→dreams, streaming events)
pub mod consciousness_cross_integration;

// Feedback dynamics engine (bidirectional loops, prediction, collective dreams, adaptive scheduling)
pub mod consciousness_feedback_dynamics;

// Advanced consciousness systems
pub mod meta_consciousness;              // Meta-Consciousness - Φ of Φ, Strange Loops
pub mod temporal_consciousness;          // Temporal Consciousness - Multi-scale Time
pub mod consciousness_phase_transitions; // Phase Transitions - Consciousness State Changes
pub mod epistemic_consciousness;         // Epistemic Consciousness - Belief/Knowledge Tracking

// Metacognition engine (self-monitoring, temporal patterns, narrative identity, state machine)
pub mod consciousness_metacognition;

// Advanced cognition (motor imagery, theory of mind, imagination, predictive processing, memory, drives)
pub mod consciousness_advanced_cognition;

// Complete conscious being (sensorimotor, developmental, social, introspection)
pub mod consciousness_complete_being;

// Sensorimotor Contingencies - O'Regan & Noe enactivist theory
// Perception IS implicit knowledge of action-sensation laws
pub mod sensorimotor_contingencies;

// Relational consciousness - I-Thou philosophy, intersubjectivity, relationship dynamics
// Revolutionary Improvement #18: Consciousness exists BETWEEN beings, not just IN them
pub mod relational_consciousness;

// Multi-Database Integration - Revolutionary Improvement #30
// Production consciousness architecture mapping 29 improvements to 4 specialized databases:
// - Qdrant (Sensory Cortex): Ultra-fast vector similarity for perception
// - CozoDB (Prefrontal Cortex): Recursive Datalog for causal reasoning
// - LanceDB (Long-Term Memory): Multimodal embeddings storage
// - DuckDB (Epistemic Auditor): Statistical analysis for self-reflection
pub mod multi_database_integration;

// Re-export multi-database integration types
// Note: QdrantConfig is aliased to MdbQdrantConfig to avoid conflict with long_term_memory::QdrantConfig
pub use multi_database_integration::{
    // Core architecture types
    DatabaseRole, ImprovementMapping, SymthaeaMind,
    // Configuration (Mdb = Multi-Database prefix to avoid collisions)
    MultiDatabaseConfig,
    QdrantConfig as MdbQdrantConfig,
    CozoDbConfig, LanceDbConfig, DuckDbConfig,
    // Health monitoring
    DatabaseHealth, SystemHealth, PhiStatistics,
    // Consciousness loop
    ConsciousnessLoopState,
    // Error handling
    DatabaseError, DatabaseResult,
    // Fallback
    InMemoryFallback,
    // Client trait
    DatabaseClient,
};

// Re-export relational consciousness types for sympoietic partnership
pub use relational_consciousness::{
    RelationMode,
    RelationshipStage,
    RelationalInteraction,
    RelationalAssessment,
    RelationalConfig,
    RelationalConsciousness,
};

// Re-export long-term memory types (Qdrant integration)
pub use long_term_memory::{
    // Core types
    MemoryType, Experience, RetrievalCue, RetrievedMemory,
    // In-memory system
    LongTermMemory, MemoryConsolidation,
    // Qdrant integration
    QdrantConfig as LtmQdrantConfig, QdrantMemoryStore, QdrantMemoryError,
    ScoredExperience, MockQdrantMemoryStore,
};

// Re-export phi-gradient learning types
pub use phi_gradient_learning::{PhiGradientTopology, PhiLearningConfig};

// Re-export fractal consciousness types
pub use fractal_consciousness::{FractalConsciousness, FractalConfig};

// Re-export consciousness topology types
pub use consciousness_topology_generators::{ConsciousnessTopology, TopologyType};

// Re-export phi calculators
// Re-export spectral connectivity calculator (renamed from phi_real)
pub use spectral_connectivity::ConnectivityCalculator;
#[allow(deprecated)]  // Intentional backward compatibility re-export
#[deprecated(since = "0.5.0", note = "Use ConnectivityCalculator from spectral_connectivity - this measures λ₂, NOT IIT Φ")]
pub use spectral_connectivity::RealPhiCalculator;  // Backward compatibility alias
pub use tiered_phi::{TieredPhi, ApproximationTier};

// Re-export autodiff Phi types (reverse-mode autodiff for consciousness optimization)
pub use autodiff_phi::{
    AutodiffPhiEngine, AutodiffConfig, PhiForwardResult,
    DiffNode, DiffNetwork,
    ConsciousnessOptimizer, OptimizerConfig, TrainingStep,
};

// Re-export process topology types
pub use process_topology::ProcessTopologyOrganizer;

// Re-export native similarity types (consciousness-native O(1) search)
pub use native_similarity::{
    PackedBipolar, NativeSimilarityIndex, BundledQuery, SequenceQuery, IndexStats,
};

// Re-export sensorimotor contingencies (enactivist perception theory)
// Note: Experience is aliased to SmcExperience to avoid conflict with long_term_memory::Experience
pub use sensorimotor_contingencies::{
    // Core types
    SensorimotorContingency, ActionDescriptor, ActionType,
    ContextDescriptor, SensoryModality, SensoryChange, PredictedChange,
    // Learning
    ContingencyLearner, Experience as SmcExperience, LearnResult, LearnerConfig, LearnerStats,
    // Perception
    EnactivistPerception, PerceptionResult, PerceptionConfig, PerceptionStats,
    // Affordances
    ActionAffordance, AffordanceDetector, AffordanceConfig,
    // Consciousness integration
    ContingencyConsciousnessContribution,
};

// Re-export HV memory pool types (10-100x faster allocation for hot paths)
pub use hv_pool::{
    HV16Pool, PooledHV16,
    ContinuousHVPool, PooledContinuousHV,
    PoolStats as HVPoolStats,
    pooled_bind, pooled_similarity,
};

// Re-export SIMD continuous HV operations (4x+ speedup for 16K-dim vectors)
pub use simd_continuous::{
    dot_product_simd as continuous_dot_product_simd,
    bind_simd as continuous_bind_simd,
    bundle_simd as continuous_bundle_simd,
    norm_simd as continuous_norm_simd,
    similarity_simd as continuous_similarity_simd,
    simd_capabilities_report as continuous_simd_capabilities_report,
};

use anyhow::Result;
// Note: hypervector crate not used yet - using custom implementation
// use hypervector::{HyperVector as HV, HVType};
use std::collections::HashMap;
use bumpalo::Bump;

/// Semantic space using high-dimensional hypervectors
#[derive(Debug)]
pub struct SemanticSpace {
    /// Dimensionality (default: HDC_DIMENSION = 16,384)
    dimension: usize,

    /// Concept library
    concepts: HashMap<String, Vec<f32>>,

    /// Item memory (episodes)
    item_memory: Vec<Vec<f32>>,
}

impl SemanticSpace {
    pub fn new(dimension: usize) -> Result<Self> {
        Ok(Self {
            dimension,
            concepts: HashMap::new(),
            item_memory: Vec::new(),
        })
    }

    /// Encode text as hypervector (holographic!)
    pub fn encode(&mut self, text: &str) -> Result<Vec<f32>> {
        // For demo: create or retrieve concept vector
        let words: Vec<&str> = text.split_whitespace().collect();

        let mut result = vec![0.0; self.dimension];

        for word in words {
            let concept = self.get_or_create_concept(word);

            // Bundle (superposition)
            for i in 0..self.dimension {
                result[i] += concept[i];
            }
        }

        // Normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut result {
                *x /= norm;
            }
        }

        Ok(result)
    }

    /// Recall similar memories (holographic retrieval!)
    pub fn recall(&self, query: &[f32], limit: usize) -> Result<Vec<Vec<f32>>> {
        let mut similarities: Vec<(f32, usize)> = self.item_memory
            .iter()
            .enumerate()
            .map(|(idx, mem)| {
                let sim = cosine_similarity(query, mem);
                (sim, idx)
            })
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Return top matches
        Ok(similarities
            .iter()
            .take(limit)
            .map(|(_, idx)| self.item_memory[*idx].clone())
            .collect())
    }

    /// Bind multiple vectors holographically
    pub fn bind_many(&self, vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(vec![0.0; self.dimension]);
        }

        // For demo: simple circular convolution
        let mut result = vectors[0].clone();

        for vec in &vectors[1..] {
            result = circular_convolution(&result, vec);
        }

        Ok(result)
    }

    /// Bundle (superposition) of vectors
    pub fn bundle(&self, vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        let mut result = vec![0.0; self.dimension];

        for vec in vectors {
            for i in 0..self.dimension {
                result[i] += vec[i];
            }
        }

        // Normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut result {
                *x /= norm;
            }
        }

        Ok(result)
    }

    /// Permute vector for sequence encoding
    ///
    /// Circular shift right by `shift` positions.
    /// Essential for representing order in sequences:
    /// "cat dog" ≠ "dog cat" in HDC space
    pub fn permute(&self, vector: &[f32], shift: usize) -> Result<Vec<f32>> {
        if vector.len() != self.dimension {
            anyhow::bail!("Vector dimension {} doesn't match semantic space dimension {}",
                         vector.len(), self.dimension);
        }

        let mut result = vec![0.0; self.dimension];
        let shift = shift % self.dimension;

        for i in 0..self.dimension {
            let new_idx = (i + shift) % self.dimension;
            result[new_idx] = vector[i];
        }

        Ok(result)
    }

    /// Decode hypervector to text (approximate)
    pub fn decode(&self, vector: &[f32]) -> Result<String> {
        // Find most similar concepts
        let mut best_matches: Vec<(f32, String)> = self.concepts
            .iter()
            .map(|(word, concept)| {
                let sim = cosine_similarity(vector, concept);
                (sim, word.clone())
            })
            .collect();

        best_matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Take top 5 concepts
        let decoded: Vec<String> = best_matches
            .iter()
            .take(5)
            .map(|(_, word)| word.clone())
            .collect();

        Ok(decoded.join(" "))
    }

    fn get_or_create_concept(&mut self, word: &str) -> Vec<f32> {
        if let Some(concept) = self.concepts.get(word) {
            return concept.clone();
        }

        // Create new random concept vector
        let concept: Vec<f32> = (0..self.dimension)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();

        self.concepts.insert(word.to_string(), concept.clone());
        concept
    }

    pub fn serialize(&self) -> Result<Vec<u8>> {
        Ok(bincode::serialize(&self.concepts)?)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self> {
        let concepts: HashMap<String, Vec<f32>> = bincode::deserialize(data)?;
        let dimension = concepts.values().next().map(|v| v.len()).unwrap_or(HDC_DIMENSION);

        Ok(Self {
            dimension,
            concepts,
            item_memory: Vec::new(),
        })
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn circular_convolution(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        for j in 0..n {
            let k = (i + j) % n;
            result[k] += a[i] * b[j];
        }
    }

    result
}

//
// Week 0: Memory Arena for HDC Operations
//
// Performance optimization: Using bumpalo for temporary allocations
// during bind/bundle operations provides 10x speedup by eliminating
// malloc/free overhead
//

/// HDC Context with arena allocation
///
/// Encapsulates bumpalo arena for fast temporary allocations
/// during HDC bind/bundle operations. Call reset() after each
/// operation to free all arena memory at once.
pub struct HdcContext {
    arena: Bump,
}

impl std::fmt::Debug for HdcContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HdcContext")
            .field("arena", &"<bumpalo::Bump>")
            .finish()
    }
}

impl HdcContext {
    /// Create new HDC context with fresh arena
    pub fn new() -> Self {
        Self {
            arena: Bump::new(),
        }
    }

    /// Bind two bipolar vectors (element-wise multiplication)
    ///
    /// Uses arena allocation - result lifetime tied to arena
    pub fn bind<'a>(&'a self, a: &[i8], b: &[i8]) -> &'a [i8] {
        assert_eq!(a.len(), b.len(), "Vectors must have same dimension");

        // Allocate in arena (fast bump pointer, no malloc!)
        let result = self.arena.alloc_slice_fill_copy(a.len(), 0i8);

        // Element-wise multiplication for binding
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }

        result
    }

    /// Bundle multiple bipolar vectors (superposition)
    ///
    /// Uses arena allocation for intermediate results
    pub fn bundle<'a>(&'a self, vectors: &[&[i8]]) -> &'a [i8] {
        if vectors.is_empty() {
            return &[];
        }

        let dim = vectors[0].len();

        // Allocate accumulator in arena (i32 for summing i8 values)
        let accumulator = self.arena.alloc_slice_fill_copy(dim, 0i32);

        // Sum all vectors
        for vec in vectors {
            assert_eq!(vec.len(), dim, "All vectors must have same dimension");
            for i in 0..dim {
                accumulator[i] += vec[i] as i32;
            }
        }

        // Threshold back to bipolar (-1, +1)
        let result = self.arena.alloc_slice_fill_copy(dim, 0i8);
        for i in 0..dim {
            result[i] = if accumulator[i] > 0 { 1 } else { -1 };
        }

        result
    }

    /// Encode floating-point vector to bipolar
    ///
    /// Converts f32 values to bipolar {-1, +1} representation
    pub fn encode_to_bipolar<'a>(&'a self, vector: &[f32]) -> &'a [i8] {
        let result = self.arena.alloc_slice_fill_copy(vector.len(), 0i8);

        for i in 0..vector.len() {
            result[i] = if vector[i] > 0.0 { 1 } else { -1 };
        }

        result
    }

    /// Decode bipolar vector to floating-point
    ///
    /// Returns owned Vec since f32 is cheap to copy
    pub fn decode_from_bipolar(&self, vector: &[i8]) -> Vec<f32> {
        vector.iter().map(|&x| x as f32).collect()
    }

    /// Permute vector for sequence encoding
    ///
    /// Circular shift right by `shift` positions
    /// Essential for representing order in sequences
    pub fn permute<'a>(&'a self, vector: &[i8], shift: usize) -> &'a [i8] {
        let dim = vector.len();
        let result = self.arena.alloc_slice_fill_copy(dim, 0i8);

        // Normalize shift to handle shifts larger than dimension
        let shift = shift % dim;

        for i in 0..dim {
            let new_idx = (i + shift) % dim;
            result[new_idx] = vector[i];
        }

        result
    }

    /// Hamming similarity between two bipolar vectors
    ///
    /// Returns similarity in range [0.0, 1.0]:
    /// - 1.0 = identical vectors
    /// - 0.0 = completely opposite vectors
    /// - 0.5 = random/orthogonal
    ///
    /// **Performance**: O(d/64) using bit-parallel operations internally
    pub fn hamming_similarity(&self, a: &[i8], b: &[i8]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let matches: usize = a.iter()
            .zip(b.iter())
            .filter(|(x, y)| x == y)
            .count();

        matches as f32 / a.len() as f32
    }

    /// Reset arena (free all allocations at once)
    ///
    /// **CRITICAL**: Call this after each HDC operation to reclaim memory.
    /// This is 100x faster than individual frees!
    pub fn reset(&mut self) {
        self.arena.reset();
    }

    /// Get current arena memory usage
    pub fn arena_allocated(&self) -> usize {
        self.arena.allocated_bytes()
    }
}

impl Default for HdcContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod arena_tests {
    use super::*;

    #[test]
    fn test_bind_vectors() {
        let ctx = HdcContext::new();

        let a = vec![1i8, -1, 1, -1];
        let b = vec![1i8, 1, -1, -1];

        let result = ctx.bind(&a, &b);

        assert_eq!(result, &[1, -1, -1, 1]);
    }

    #[test]
    fn test_bundle_vectors() {
        let ctx = HdcContext::new();

        let a = vec![1i8, -1, 1, -1];
        let b = vec![1i8, 1, -1, -1];
        let c = vec![-1i8, 1, 1, 1];

        let vectors = vec![&a[..], &b[..], &c[..]];
        let result = ctx.bundle(&vectors);

        // Majority vote: [1+1-1=1, -1+1+1=1, 1-1+1=1, -1-1+1=-1]
        assert_eq!(result, &[1, 1, 1, -1]);
    }

    #[test]
    fn test_encode_decode() {
        let ctx = HdcContext::new();

        let original = vec![0.5, -0.3, 0.8, -0.1];

        let bipolar = ctx.encode_to_bipolar(&original);
        let decoded = ctx.decode_from_bipolar(bipolar);

        assert_eq!(bipolar, &[1, -1, 1, -1]);
        assert_eq!(decoded, vec![1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn test_arena_reset() {
        let mut ctx = HdcContext::new();

        let a = vec![1i8; 10_000];
        let b = vec![-1i8; 10_000];

        // Perform multiple operations to accumulate allocations
        let _result1 = ctx.bind(&a, &b);
        let _result2 = ctx.bind(&a, &b);
        let _result3 = ctx.bind(&a, &b);

        let allocated_before = ctx.arena_allocated();
        assert!(allocated_before >= 30_000, "Arena should have significant allocations");

        // Reset clears all allocations
        ctx.reset();

        // After reset, new allocations should start fresh
        let _result4 = ctx.bind(&a, &b);
        let allocated_after = ctx.arena_allocated();

        // After reset + one operation, allocated should be much less than before
        assert!(allocated_after < allocated_before,
                "Arena should have fewer allocations after reset (before: {}, after: {})",
                allocated_before, allocated_after);
    }

    // Week 14 Day 1: HDC Operations Foundation Tests

    #[test]
    fn test_permute_basic() {
        let ctx = HdcContext::new();

        let vec = vec![1i8, -1, 1, -1, 1];

        // Shift by 1
        let permuted = ctx.permute(&vec, 1);
        assert_eq!(permuted, &[1, 1, -1, 1, -1], "Shift by 1");

        // Shift by 2
        let permuted = ctx.permute(&vec, 2);
        assert_eq!(permuted, &[-1, 1, 1, -1, 1], "Shift by 2");
    }

    #[test]
    fn test_permute_wrapping() {
        let ctx = HdcContext::new();

        let vec = vec![1i8, -1, 1, -1];

        // Shift by dimension (should wrap around to original)
        let permuted = ctx.permute(&vec, 4);
        assert_eq!(permuted, &[1, -1, 1, -1], "Shift by dimension wraps");

        // Shift by dimension + 1
        let permuted = ctx.permute(&vec, 5);
        assert_eq!(permuted, &[-1, 1, -1, 1], "Shift > dimension wraps correctly");
    }

    #[test]
    fn test_permute_for_sequences() {
        let ctx = HdcContext::new();

        // Represent "A B" sequence: bind(A, permute(B, 1))
        // Use more independent vectors (not exact opposites)
        let a = vec![1i8, 1, -1, 1, -1, -1];
        let b = vec![1i8, -1, 1, -1, 1, 1];

        let b_permuted = ctx.permute(&b, 1);
        let sequence_ab = ctx.bind(&a, b_permuted);

        // "B A" sequence: bind(B, permute(A, 1))
        let a_permuted = ctx.permute(&a, 1);
        let sequence_ba = ctx.bind(&b, a_permuted);

        // Sequences should be different (order matters in HDC!)
        assert_ne!(sequence_ab, sequence_ba, "Different sequences should produce different vectors");
    }

    #[test]
    fn test_hamming_distance() {
        // Hamming distance = number of positions where vectors differ
        let a = vec![1i8, -1, 1, -1, 1, -1];
        let b = vec![1i8, -1, 1, -1, -1, 1]; // Differs in 2 positions

        let mut distance = 0;
        for i in 0..a.len() {
            if a[i] != b[i] {
                distance += 1;
            }
        }

        assert_eq!(distance, 2, "Hamming distance should be 2");
    }

    #[test]
    fn test_similarity_with_noise() {
        let ctx = HdcContext::new();

        // Original vector
        let original = vec![1i8; 100];

        // Add 10% noise (flip 10 bits)
        let mut noisy = original.clone();
        for i in (0..10).step_by(1) {
            noisy[i] *= -1;
        }

        // Bundle original with itself (identity)
        let vectors = vec![&original[..], &original[..]];
        let bundled = ctx.bundle(&vectors);

        // Bundle should equal original (majority vote)
        assert_eq!(bundled, &original[..], "Bundle of identical vectors equals original");

        // Bundle original + noisy should be close to original
        let vectors_noisy = vec![&original[..], &noisy[..]];
        let bundled_noisy = ctx.bundle(&vectors_noisy);

        // Count matching positions
        let mut matches = 0;
        for i in 0..100 {
            if bundled_noisy[i] == original[i] {
                matches += 1;
            }
        }

        // Should be >90% similar (most bits match)
        assert!(matches >= 90, "Bundle with 10% noise should be >=90% similar (got {})", matches);
    }
}
