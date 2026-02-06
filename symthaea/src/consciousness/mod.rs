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

// Empathic unification - re-enabled after adding required types to emotional_core
// and ContextKind variants to user_state_inference
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

// ============================================================================
// Ported from crates/symthaea-consciousness (2026-02-06) — Tier 1 high-value
// ============================================================================

/// Hierarchical LTC architecture with 25x speedup (CSR sparse matrix)
pub mod hierarchical_ltc;

/// Narrative Self-Model (Damasio autobiographical self, Tulving autonoetic consciousness)
pub mod narrative_self;

/// Attention Schema Theory (Graziano) — attention as a control mechanism
pub mod attention_schema;

/// Consciousness Thermodynamics — free energy, entropy, phase transitions
pub mod consciousness_thermodynamics;

/// Epistemic Causal Reasoning — 3D epistemic cube (Empirical × Normative × Materiality)
pub mod epistemic_tiers;

/// Metacognitive Monitoring — self-correction via Φ observation
pub mod metacognitive_monitoring;

/// Operational Fiduciary Harmonics — Seven Harmonics optimization framework
pub mod harmonics;

/// Unified Emergent Intelligence — collective + context-aware + meta-cognitive
/// Gated: depends on meta_reasoning + context_aware_evolution (not yet ported)
#[cfg(feature = "full_consciousness")]
pub mod unified_intelligence;

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

// ── Ported Tier 1 re-exports ──────────────────────────────────────────────
pub use hierarchical_ltc::{HierarchicalLTC, HierarchicalConfig};
pub use narrative_self::{NarrativeSelfModel, NarrativeSelfConfig, ProtoSelf, CoreSelf, AutobiographicalSelf};
pub use attention_schema::{AttentionSchema, AttentionState, AttentionSchemaConfig};
pub use consciousness_thermodynamics::{ConsciousnessThermodynamicsAnalyzer, ThermodynamicsConfig};
pub use epistemic_tiers::{EpistemicCoordinate, EmpiricalTier, NormativeTier, MaterialityTier};
pub use metacognitive_monitoring::{MetacognitiveMonitor, MonitoringResult};
pub use harmonics::{FiduciaryHarmonic, HarmonicField, HarmonicResolver};

// ============================================================================
// Ported from crates/symthaea-consciousness (2026-02-06) — Tiers 2–5
// ============================================================================

// ── Tier 2: Primitive Reasoning Layer ───────────────────────────────────────
pub mod adaptive_reasoning;
pub mod causal_explanation;
pub mod negation_detector;
pub mod synthetic_states;
pub mod temporal_primitives;

pub mod primitive_validation;

// ── Tier 3: Consciousness Integration ──────────────────────────────────────
pub mod compositionality_primitives;
pub mod consciousness_signatures;
pub mod causal_emergence;

pub mod dimension_synergies;

pub mod consciousness_profile;
pub mod meta_primitives;
pub mod phi_validation;

// ── Tier 4: Dynamics & Field ───────────────────────────────────────────────
pub mod consciousness_field_dynamics;
pub mod consciousness_holography;
pub mod consciousness_resonance;
pub mod consciousness_topology;
pub mod dissipative_consciousness;
pub mod embodied_cognition;
pub mod enactive_cognition;
pub mod meta_cognitive_optimizer;
pub mod phenomenal_binding;
pub mod predictive_processing;
pub mod predictive_self;
pub mod quantum_coherence;
pub mod sensorimotor_contingencies;

// ── Tier 4 (NEEDS_CHAIN — dependencies now satisfied) ──────────────────────
pub mod consciousness_equation_v2;
pub mod context_aware_evolution;
pub mod temporal_consciousness;
pub mod unified_consciousness_pipeline;

/// Gated: depends on recursive_improvement (feature-gated)
#[cfg(any(feature = "full_consciousness", feature = "magi_loop"))]
pub mod consciousness_driven_evolution;

/// Gated: depends on autopoietic_consciousness::LifeState (not in src version)
#[cfg(feature = "full_consciousness")]
pub mod unified_living_mind;

// ── Final batch: remaining unique modules from crate ────────────────────────
pub mod differentiable;
pub mod meta_reasoning;

/// Gated: depends on unified_intelligence (gated behind full_consciousness)
#[cfg(feature = "full_consciousness")]
pub mod byzantine_collective;

/// Gated: depends on byzantine_collective (gated)
#[cfg(feature = "full_consciousness")]
pub mod meta_learning_byzantine;

/// Gated: depends on meta_learning_byzantine (gated)
#[cfg(feature = "full_consciousness")]
pub mod causal_byzantine;

/// Gated: depends on observability (gated behind observability_module)
#[cfg(feature = "observability_module")]
pub mod gwt_integration;

/// Gated: depends on gwt_integration + unified_value_evaluator (gated)
#[cfg(all(feature = "observability_module", feature = "full_perception"))]
pub mod narrative_gwt_integration;

/// Gated: depends on unified_value_evaluator (gated behind full_perception)
#[cfg(feature = "full_perception")]
pub mod mycelix_bridge;

/// Gated: depends on mycelix_bridge (gated)
#[cfg(feature = "full_perception")]
pub mod value_system_tests;

/// Gated: depends on consciousness_driven_evolution (gated)
#[cfg(any(feature = "full_consciousness", feature = "magi_loop"))]
pub mod consciousness_guided_discovery;

/// Gated: depends on consciousness_guided_discovery (gated)
#[cfg(any(feature = "full_consciousness", feature = "magi_loop"))]
pub mod meta_meta_learning;

/// Gated: depends on consciousness_profile (gated behind full_consciousness)
#[cfg(feature = "full_consciousness")]
pub mod multi_objective_evolution;

// Re-exports from consciousness_equation_v2 (needed by differentiable.rs)
pub use consciousness_equation_v2::{ConsciousnessStateV2, CoreComponent, EquationConfig};

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
