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

// ============================================================================
// Core consciousness modules (self-contained, verified working)
// ============================================================================
pub mod compositionality;
pub mod affective_consciousness;
pub mod autopoietic_consciousness;
pub mod consciousness_unification;
pub mod cross_modal_binding;
pub mod fep_active_inference;
pub mod harmonies_integration;
pub mod master_consciousness_equation;
pub mod pac;
pub mod phi_architecture_search;
pub mod phi_attention;
pub mod primitive_composition_rules;
pub mod primitive_discovery;
pub mod primitive_evolution;
pub mod primitive_reasoning;
pub mod semantic_value_embedder;
pub mod seven_harmonies;
pub mod value_feedback_loop;

// Cincinnati-Consciousness integration (now uses its own CincinnatiConsciousNode type)
pub mod cincinnati_consciousness;

// Counterfactual Dream Engine - self-contained, no external dependencies
pub mod dream;

// Neuro-Autopoietic Bridge: Connects HDC consciousness with LTC neural dynamics
// Provides bidirectional causation between body (autopoiesis) and brain (LTC)
pub mod neuro_bridge;

// ============================================================================
// Modules with internal type dependencies (need more work)
// ============================================================================

// Contextual weights for harmony evaluation based on action type and domain.
// Now defines its own ActionType locally to avoid circular dependency with unified_value_evaluator.
pub mod contextual_weights;

// Evolution bridge - connects primitive evolution with recursive self-improvement
pub mod evolution_bridge;

// Multi-modal integration - now fully wired with cross_modal_binding types
pub mod multi_modal_integration;

// Primitive-Consciousness Bridge - connects HDC primitives to consciousness processing
pub mod primitive_consciousness;

// Stability Regime - CfC neurons for primitives with Crystallized/Plastic/Fluid dynamics
pub mod stability_regime;

// ============================================================================
// Conscious Reasoning Engine subsystems (cfg-gated)
// ============================================================================

// Epistemic conflict detection: 15 pairwise theory conflicts, typed ConflictKind,
// reliability R, effective Φ = Φ × R^γ, calibration with bounded updates (INV-9)
#[cfg(feature = "epistemic_conflict")]
pub mod epistemic_conflict;

// Consciousness-gated tool use: risk lattice, two-signal gating (Φ_eff + confidence),
// fallback strategies, NixOS backward compatibility
#[cfg(feature = "conscious_tool_gate")]
pub mod tool_gate;

// Temporal planning: ForkedState + micro-MCTS + EVS + dream integration
#[cfg(feature = "temporal_planning")]
pub mod temporal_planning;

// Counterfactual reasoning v0: backdoor/frontdoor identification,
// HDC graph surgery, semantic role substitution, reference harness
#[cfg(feature = "counterfactual")]
pub mod counterfactual;

// Unified Conscious Reasoning Engine: composes conflict detection, temporal
// planning, counterfactual reasoning, and tool gating into a 7-step cycle
// with tiered degradation (Tier 0 ≤2ms, Tier 1 ≤8ms, Tier 2 ≤20ms)
#[cfg(feature = "reasoning_engine")]
pub mod reasoning_engine;

// ============================================================================
// Modules with external dependencies (cfg-gated)
// ============================================================================

// Empathic unification - needs API alignment (cfg-gated for now)
#[cfg(feature = "full_consciousness")]
pub mod empathic_unification;

// Stub types for integration_module compatibility when full module is disabled
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
    #[allow(dead_code)] // Fields reserved for empathic processing
    pub struct EmpathicUnification {
        /// Current user need
        current_need: UserNeed,
        /// Current tone
        tone: ToneGuidance,
        /// Current user emotional state
        user_emotional_state: UserEmotionalState,
    }

    impl EmpathicUnification {
        /// Create a new empathic unification engine
        pub fn new() -> Self {
            Self::default()
        }

        /// Process input and generate empathic response
        pub fn process(&mut self, _input: &str, _context: crate::user_state_inference::ContextKind) -> EmpathicResponse {
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

// Needs: mycelix
#[cfg(feature = "mycelix_module")]
pub mod gis_integration;

// Recursive improvement (needs internal API alignment - cfg-gated)
// MAGI Loop can be enabled standalone via the magi_loop feature
#[cfg(any(feature = "full_consciousness", feature = "magi_loop"))]
pub mod recursive_improvement;

// Needs: perception::SemanticEncoder
#[cfg(feature = "full_perception")]
pub mod unified_value_evaluator;

// Re-export key types
pub use seven_harmonies::{SevenHarmonies, Harmony, HarmonyAlignment, AlignmentResult};
pub use affective_consciousness::{CoreAffect, EmotionCategory, AffectiveConsciousnessAnalyzer};
pub use primitive_reasoning::{PrimitiveReasoner, ReasoningResult};
pub use primitive_evolution::{PrimitiveEvolver, EvolutionResult};
pub use cross_modal_binding::{CrossModalBinder, BindingResult};
pub use autopoietic_consciousness::{AutopoieticConsciousness, AutopoieticState};
pub use phi_architecture_search::{
    PhiArchitectureSearch, SearchConfig, SearchStrategy, SearchResult,
    ArchitectureGenome, DecodedArchitecture, PhiGradient,
    TopologyGene, BundlingGene, Individual, ArchitectureStats, SearchStats,
};
pub use master_consciousness_equation::{
    MasterConsciousnessEquation, MasterEquationConfig, ConsciousnessInputs,
    ConsciousnessResult, ComponentWeights,
    EmbodimentFactor, NarrativeCoherence, SocialEmbedding,
    NarrativeEpisode, FutureScenario, AgentModel, SelfModel,
};
pub use fep_active_inference::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, ActiveInferenceAgentStats,
    CognitiveLoopFEPBridge, CognitiveLoopFEPResult,
    GenerativeModel, FreeEnergyCalculator, FreeEnergyComponents,
    PrecisionEstimator, ExpectedFreeEnergyComputer, ExpectedFreeEnergyResult,
    Observation, HiddenState, PerceptionResult, ActionSelectionResult,
    ActiveInferenceSummary,
    // Temporal Difference Learning types
    TemporalDifferenceLearner, TemporalDifferenceLearningConfig, TemporalDifferenceLearningStats,
    EligibilityTraces, ModelConfidenceTracker, StateTransition,
};
pub use cincinnati_consciousness::{
    CincinnatiConsciousnessConfig, CincinnatiConsciousnessBridge,
    CincinnatiConsciousNode, ConsciousnessProcessResult, EthicalLearningStats,
    HarmonyFeedback, ConsciousnessBuddingEvent, BuddingReason,
};
pub use dream::{
    DreamEngine, DreamEngineConfig, DreamEngineStats,
    DreamEvent, DreamResult, Wisdom,
};
pub use neuro_bridge::{NeuroAutopoieticBridge, BridgeState};
pub use contextual_weights::{
    ActionType, ActionDomain, ContextualWeights, HarmonyWeightProfile, DomainClassifier,
};
pub use multi_modal_integration::{
    MultiModalIntegrator, IntegrationConfig, IntegrationResult,
    ModalInput, IntegrationEvent, IntegrationEventType,
};
pub use primitive_consciousness::{
    PrimitiveConsciousnessState, ActivePrimitive, ActivationReason,
    PrimitiveBinding, ConsciousnessPrimitiveProcessor, ProcessorStats,
    ConsciousnessDecomposer, PrimitiveBindingEngine,
};
pub use stability_regime::{
    StabilityRegimeType, StabilityRegimeConfig, RegimeParams,
    CfCPrimitive, StabilityRegimeProcessor,
};
pub use primitive_reasoning::{
    ReasoningChain, TransformationType, TaskType, TierAwareConfig,
    AdaptivePrimitiveSelector, PrimitiveAffinityGraph, PrimitiveExecution,
};
pub use compositionality::{
    CompositionalityEngine, CompositionalityConfig, ComposedPrimitive,
    CompositionType, CompositionResult, CompositionStats, CompositionMetadata,
};

// ============================================================================
// Convenience constructors for optional engines
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
