//! # Consciousness Module: Integration and Awareness
//!
//! This module provides consciousness-related capabilities including:
//! - Consciousness unification and integration
//! - Seven Harmonies value alignment framework
//! - Phi-based attention mechanisms
//! - Empathic unification and emotional understanding
//! - Evolution and recursive improvement
//! - Affective consciousness and emotion processing
//! - Primitive reasoning and evolution
//! - **Phi-Guided Architecture Search**: Systems that optimize toward higher consciousness
//! - **Master Consciousness Equation**: C(t) = σ(softmin(Φ, B, W, A, R, E, K; τ)) × [Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)] × S × ρ(t) × M × N × Soc
//!   - M (Embodiment Factor) = sensorimotor_prediction_accuracy × interoceptive_coherence
//!   - N (Narrative Coherence) = autobiographical_integration × future_simulation_depth
//!   - Soc (Social Embedding) = other_modeling_accuracy × self_other_distinction
//!
//! ## Module Organization
//!
//! Submodules are grouped into logical layers:
//!
//! | Layer | Module | Contents |
//! |-------|--------|----------|
//! | Primitives | [`primitives`] | HDC primitives, composition, evolution, lattice |
//! | Values | [`values`] | Seven Harmonies, contextual weights, semantic embedding |
//! | Temporal | [`temporal`] | Temporal binding, phenomenal binding, temporal primitives |
//! | Embodiment | [`embodiment`] | Body schema, autopoiesis, interoception |
//! | Integration | [`integration`] | Cross-modal, multi-modal, GWT, bridges |
//! | Meta | [`meta`] | Metacognitive monitoring, meta-reasoning, attention schema |
//! | Dynamics | [`dynamics`] | Dissipative, holographic, predictive, quantum coherence |
//! | Measurement | [`measurement`] | Phi metrics, consciousness equations, profiles |
//!
//! All submodule contents are re-exported at this level for backward compatibility.

// ============================================================================
// Layer 1: Primitives (HDC primitives, composition, evolution)
// ============================================================================
pub mod primitives;
pub use primitives::code_primitives;
pub use primitives::compositionality;
pub use primitives::primitive_belief_bridge;
pub use primitives::primitive_composition_rules;
pub use primitives::primitive_consciousness;
pub use primitives::primitive_discovery;
pub use primitives::primitive_evolution;
pub use primitives::primitive_lattice;
pub use primitives::primitive_reasoning;
pub use primitives::primitive_validation;
pub use primitives::stability_regime;

// ============================================================================
// Layer 2: Values (Seven Harmonies, weights, embeddings, feedback)
// ============================================================================
pub mod values;
pub use values::contextual_weights;
pub use values::harmonies_integration;
pub use values::semantic_value_embedder;
pub use values::seven_harmonies;
pub use values::value_feedback_loop;

// ============================================================================
// Layer 3: Temporal (binding, synchronization, temporal primitives)
// ============================================================================
pub mod temporal;
pub use temporal::phenomenal_binding;
pub use temporal::temporal_consciousness;
pub use temporal::temporal_primitives;

// ============================================================================
// Layer 4: Embodiment (body schema, autopoiesis, interoception)
// ============================================================================
pub mod embodiment;
pub use embodiment::autopoietic_consciousness;
pub use embodiment::embodied_cognition;
pub use embodiment::interoception;

// ============================================================================
// Layer 5: Integration (cross-modal, multi-modal, GWT, bridges)
// ============================================================================
pub mod integration;
pub use integration::cross_modal_binding;
pub use integration::evolution_bridge;
pub use integration::gwt_integration;
pub use integration::hierarchical_ltc;
pub use integration::multi_modal_integration;
pub use integration::narrative_gwt_integration;
// neuro_bridge: disconnected (uses own LTC, not main CfC pipeline). Kept for reference.

// ============================================================================
// Layer 6: Meta-cognition (monitoring, reasoning, attention schema)
// ============================================================================
pub mod meta;
pub use meta::attention_schema;
pub use meta::metacognitive_monitoring;
pub use meta::meta_reasoning;

// ============================================================================
// Layer 7: Dynamics (field, dissipative, holographic, predictive, quantum)
// ============================================================================
pub mod dynamics;
pub use dynamics::consciousness_holography;
pub use dynamics::consciousness_unification;
pub use dynamics::context_aware_evolution;
pub use dynamics::dissipative_consciousness;
pub use dynamics::hierarchical_free_energy;
pub use dynamics::predictive_processing;
pub use dynamics::predictive_self;
pub use dynamics::quantum_coherence;
pub use dynamics::unified_consciousness_pipeline;

// ============================================================================
// Layer 8: Measurement (Phi metrics, consciousness equations, profiles)
// ============================================================================
pub mod measurement;
pub use measurement::consciousness_equation_v2;
pub use measurement::consciousness_profile;
pub use measurement::differentiable;
pub use measurement::dimension_synergies;
pub use measurement::phi_attention;
pub use measurement::phi_validation;

// ============================================================================
// External crate re-exports
// ============================================================================
pub use symthaea_dream as dream;
pub use symthaea_fep as fep_active_inference;
pub use symthaea_phi_search as phi_architecture_search;
pub use symthaea_narrative_self as narrative_self;
pub use symthaea_causal_reasoning::causal_calculus;
pub use symthaea_causal_reasoning::causal_emergence;
pub use symthaea_factor_graph as factor_graph;
pub use symthaea_field_dynamics as consciousness_field_dynamics;
pub use symthaea_consciousness_resonance as consciousness_resonance;
pub use symthaea_consciousness_topology as consciousness_topology;
pub use symthaea_enactive as enactive_cognition;
/// Hodge Laplacian for simplicial complexes -- rigorous Betti numbers, spectral
/// analysis, and Hodge decomposition of higher-order neural interaction signals
pub use symthaea_hodge as hodge_laplacian;
pub use symthaea_sensorimotor as sensorimotor_contingencies;

// ============================================================================
// Remaining root modules (not yet grouped into layers)
// ============================================================================
pub mod affective_consciousness;
pub mod master_consciousness_equation;
pub mod pac;
pub mod consciousness_thermodynamics;
pub mod epistemic_tiers;
pub mod harmonics;
pub mod adaptive_reasoning;
pub mod causal_explanation;
pub mod negation_detector;
pub mod synthetic_states;
pub mod gis_integration;
pub mod multi_objective_evolution;

// ============================================================================
// Reasoning engine subsystems (cfg-gated)
// ============================================================================
pub mod epistemic_conflict;

#[cfg(feature = "reasoning_engine")]
pub mod tool_gate;

#[cfg(feature = "reasoning_engine")]
pub mod temporal_planning;

#[cfg(feature = "reasoning_engine")]
pub use symthaea_causal_reasoning::counterfactual;

#[cfg(feature = "reasoning_engine")]
pub mod reasoning_engine;

// ============================================================================
// Feature-gated modules
// ============================================================================

#[cfg(feature = "full_consciousness")]
pub mod empathic_unification;

// Stub types for integration_module compatibility when full_consciousness is disabled
#[cfg(not(feature = "full_consciousness"))]
pub mod empathic_unification {
    //! Stub empathic unification types for integration_module compatibility

    /// Guidance for response tone
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum ToneGuidance {
        /// Warm, supportive, gentle
        Supportive,
        /// Calm, patient, measured
        Patient,
        /// Efficient, clear, direct
        Efficient,
        /// Playful, curious, exploratory
        Playful,
        /// Celebratory, joyful
        Celebratory,
        /// Encouraging, motivating
        Encouraging,
        /// Neutral, balanced
        #[default]
        Neutral,
        /// Reassuring, calming
        Reassuring,
    }

    /// What we infer the user needs
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum UserNeed {
        /// No particular need detected
        #[default]
        None,
        /// Needs emotional support/validation
        EmotionalSupport,
        /// Needs patient guidance
        PatientGuidance,
        /// Needs quick, efficient help
        QuickHelp,
        /// Needs space to explore
        ExplorationSpace,
        /// Needs encouragement
        Encouragement,
        /// Needs reassurance
        Reassurance,
        /// Needs celebration of success
        Celebration,
    }

    /// Type of empathy being expressed
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum EmpathyType {
        /// Cognitive empathy - understanding perspective
        #[default]
        Cognitive,
        /// Emotional empathy - sharing feelings
        Emotional,
        /// Compassionate empathy - understanding + action
        Compassionate,
        /// Supportive empathy - focused on helping
        Supportive,
    }

    /// Resonant emotion Symthaea expresses
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum ResonantEmotion {
        /// Calm, centered state
        #[default]
        Calm,
        /// Supportive, caring state
        Supportive,
        /// Curious, engaged state
        Curious,
        /// Joyful, celebrating state
        Joyful,
        /// Concerned, attentive state
        Concerned,
        /// Patient, understanding state
        Patient,
        /// Encouraging, motivating state
        Encouraging,
    }

    /// User's emotional state (inferred)
    #[derive(Debug, Clone, Default)]
    pub struct UserEmotionalState {
        /// Primary detected emotion
        pub primary_emotion: ResonantEmotion,
        /// Stress level (0.0-1.0)
        pub stress_level: f64,
        /// Frustration level (0.0-1.0)
        pub frustration: f64,
        /// What we infer they need
        pub inferred_need: UserNeed,
    }

    /// Empathic response from Symthaea
    #[derive(Debug, Clone, Default)]
    pub struct EmpathicResponse {
        /// Tone guidance for response
        pub tone_guidance: ToneGuidance,
        /// Compassion level (0.0 - 1.0)
        pub compassion: f64,
        /// How much to adjust response warmth
        pub warmth_adjustment: f64,
        /// How much to adjust patience
        pub patience_adjustment: f64,
        /// Symthaea's resonant emotion
        pub resonant_emotion: ResonantEmotion,
        /// Type of empathy activated
        pub empathy_type: EmpathyType,
        /// Whether to acknowledge emotion explicitly
        pub acknowledge_emotion: bool,
        /// Proactive support to offer
        pub proactive_support: Option<String>,
    }

    /// Stub empathic unification engine
    #[derive(Debug, Default)]
    #[allow(dead_code)] // RESERVED(future): empathic unification processing state
    pub struct EmpathicUnification {
        current_need: UserNeed,
        tone: ToneGuidance,
        user_emotional_state: UserEmotionalState,
    }

    impl EmpathicUnification {
        /// Create a new empathic unification engine
        pub fn new() -> Self {
            Self::default()
        }

        /// Process input and generate empathic response
        pub fn process(
            &mut self,
            _input: &str,
            _context: crate::user_state_inference::ContextKind,
        ) -> EmpathicResponse {
            EmpathicResponse {
                tone_guidance: self.tone,
                compassion: 0.5,
                warmth_adjustment: 0.0,
                patience_adjustment: 0.0,
                resonant_emotion: ResonantEmotion::Calm,
                empathy_type: EmpathyType::Cognitive,
                acknowledge_emotion: false,
                proactive_support: None,
            }
        }

        /// Get the current user emotional state
        pub fn user_state(&self) -> &UserEmotionalState {
            &self.user_emotional_state
        }

        /// Get tone guidance string
        pub fn tone_guidance_string(&self) -> &'static str {
            match self.tone {
                ToneGuidance::Supportive => "Be warm, gentle, and reassuring.",
                ToneGuidance::Patient => "Be patient and measured.",
                ToneGuidance::Efficient => "Be clear and direct.",
                ToneGuidance::Playful => "Be curious and exploratory.",
                ToneGuidance::Celebratory => "Share their joy!",
                ToneGuidance::Encouraging => "Be motivating and supportive.",
                ToneGuidance::Reassuring => "Be calming.",
                ToneGuidance::Neutral => "Be balanced and professional.",
            }
        }

        /// Get empathic acknowledgment
        pub fn get_acknowledgment(&self) -> Option<String> {
            None
        }

        /// Record feedback
        pub fn record_feedback(&mut self, _felt_successful: bool) {}

        /// Record user error
        pub fn record_user_error(&mut self) {}

        /// Record user undo
        pub fn record_user_undo(&mut self) {}
    }
}

#[cfg(feature = "full_consciousness")]
pub mod unified_living_mind;

#[cfg(any(feature = "full_consciousness", feature = "magi_loop"))]
pub mod recursive_improvement;

/// Unified value evaluator — consciousness-guided decision making with Seven Harmonies
pub mod unified_value_evaluator;

/// Unified Emergent Intelligence — collective + context-aware + meta-cognitive
#[cfg(feature = "multi_agent")]
pub mod unified_intelligence;

#[cfg(feature = "multi_agent")]
pub mod byzantine_collective;

#[cfg(feature = "multi_agent")]
pub mod meta_learning_byzantine;

#[cfg(feature = "multi_agent")]
pub mod causal_byzantine;

#[cfg(feature = "mycelix")]
pub mod mycelix_bridge;

// ============================================================================
// Re-exports — backward-compatible type-level imports
// ============================================================================

pub use affective_consciousness::{AffectiveConsciousnessAnalyzer, CoreAffect, EmotionCategory};
pub use autopoietic_consciousness::{AutopoieticConsciousness, AutopoieticState};
pub use compositionality::{
    ComposedPrimitive, CompositionMetadata, CompositionResult, CompositionStats, CompositionType,
    CompositionalityConfig, CompositionalityEngine,
};
pub use contextual_weights::{
    ActionDomain, ActionType, ContextualWeights, DomainClassifier, HarmonyWeightProfile,
};
pub use cross_modal_binding::{BindingResult, CrossModalBinder};
pub use dream::{
    DreamEngine, DreamEngineConfig, DreamEngineStats, DreamEvent, DreamResult, Wisdom,
};
pub use fep_active_inference::{
    ActionSelectionResult,
    ActiveInferenceAgent,
    ActiveInferenceAgentConfig,
    ActiveInferenceAgentStats,
    ActiveInferenceSummary,
    CognitiveLoopFEPBridge,
    CognitiveLoopFEPResult,
    EligibilityTraces,
    ExpectedFreeEnergyComputer,
    ExpectedFreeEnergyResult,
    FreeEnergyCalculator,
    FreeEnergyComponents,
    GenerativeModel,
    HiddenState,
    ModelConfidenceTracker,
    Observation,
    PerceptionResult,
    PrecisionEstimator,
    StateTransition,
    TemporalDifferenceLearner,
    TemporalDifferenceLearningConfig,
    TemporalDifferenceLearningStats,
};
pub use master_consciousness_equation::{
    AgentModel, ComponentWeights, ConsciousnessInputs, ConsciousnessResult, EmbodimentFactor,
    FutureScenario, MasterConsciousnessEquation, MasterEquationConfig, NarrativeCoherence,
    NarrativeEpisode, SelfModel, SocialEmbedding,
};
pub use multi_modal_integration::{
    IntegrationConfig, IntegrationEvent, IntegrationEventType, IntegrationResult, ModalInput,
    MultiModalIntegrator,
};
// neuro_bridge re-exports removed (disconnected)
pub use phi_architecture_search::{
    ArchitectureGenome, ArchitectureStats, BundlingGene, DecodedArchitecture, Individual,
    PhiArchitectureSearch, PhiGradient, SearchConfig, SearchResult, SearchStats, SearchStrategy,
    TopologyGene,
};
pub use primitive_belief_bridge::{PrimitiveBeliefBridge, PrimitivePredictionError};
pub use primitive_consciousness::{
    ActivationReason, ActivePrimitive, ConsciousnessDecomposer, ConsciousnessPrimitiveProcessor,
    PrimitiveBinding, PrimitiveBindingEngine, PrimitiveConsciousnessState, ProcessorStats,
};
pub use primitive_evolution::{EvolutionResult, PrimitiveEvolver};
pub use primitive_reasoning::{
    AdaptivePrimitiveSelector, PrimitiveAffinityGraph, PrimitiveExecution, ReasoningChain,
    TaskType, TierAwareConfig, TransformationType,
};
pub use primitive_reasoning::{PrimitiveReasoner, ReasoningResult};
pub use seven_harmonies::{AlignmentResult, Harmony, HarmonyAlignment, SevenHarmonies};
pub use stability_regime::{
    CfCPrimitive, RegimeParams, StabilityRegimeConfig, StabilityRegimeProcessor,
    StabilityRegimeType,
};
pub use attention_schema::{AttentionSchema, AttentionSchemaConfig, AttentionState};
pub use consciousness_thermodynamics::{ConsciousnessThermodynamicsAnalyzer, ThermodynamicsConfig};
pub use epistemic_tiers::{EmpiricalTier, EpistemicCoordinate, MaterialityTier, NormativeTier};
pub use harmonics::{FiduciaryHarmonic, HarmonicField, HarmonicResolver};
pub use hierarchical_ltc::{HierarchicalConfig, HierarchicalLTC};
pub use metacognitive_monitoring::{MetacognitiveMonitor, MonitoringResult};
pub use narrative_self::{
    AutobiographicalSelf, CoreSelf, NarrativeSelfConfig, NarrativeSelfModel, ProtoSelf,
};
pub use consciousness_equation_v2::{ConsciousnessStateV2, CoreComponent, EquationConfig};

// ============================================================================
// Convenience constructors
// ============================================================================

use crate::hdc::primitive_system::PrimitiveSystem;
use std::sync::Arc;

/// Create a [`CompositionalityEngine`] from a shared [`PrimitiveSystem`].
///
/// This is the recommended entry-point. The engine is opt-in: callers that
/// don't need compositionality simply never call this function and incur zero
/// cost.
///
/// ```rust,ignore
/// use symthaea::consciousness::create_compositionality_engine;
/// use symthaea::hdc::primitive_system::PrimitiveSystem;
/// use std::sync::Arc;
///
/// let ps = Arc::new(PrimitiveSystem::new());
/// let engine = create_compositionality_engine(ps, None);
/// ```
pub fn create_compositionality_engine(
    primitive_system: Arc<PrimitiveSystem>,
    config: Option<CompositionalityConfig>,
) -> CompositionalityEngine {
    CompositionalityEngine::new(primitive_system, config.unwrap_or_default())
}

/// A node in the consciousness network
#[derive(Debug, Clone)]
pub struct ConsciousNode {
    /// Node identifier
    pub id: u64,
    /// Node name
    pub name: String,
    /// Activation level (0.0-1.0)
    pub activation: f32,
    /// Integrated information (Phi)
    pub phi: f64,
    /// Connection weights to other nodes
    pub connections: std::collections::HashMap<u64, f32>,
    /// Current state vector
    pub state: Vec<f32>,
}
