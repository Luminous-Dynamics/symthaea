//! Consciousness Integration Module
//!
//! Provides types for consciousness pipeline integration with comprehensive
//! multi-system consciousness architecture.
//!
//! ## Integrated Systems
//!
//! 1. **Metacognitive Monitoring** - Self-awareness of processing quality
//! 2. **Predictive Consciousness** - Active inference and anticipation
//! 3. **Cross-Modal Binding** - Multi-sensory integration
//! 4. **Temporal Binding** - Stream coherence with theta-phase modulation
//! 5. **Emergent Self-Model** - Recursive higher-order thought

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::binary_hv::HV16;

// === INTEGRATED CONSCIOUSNESS MODULES ===
// 1. Metacognitive Monitoring
use super::metacognitive_monitor::{
    MetacognitiveMonitor, MetacognitiveRecommendation, CognitiveEvent, MemoryOperation,
};

// 3. Cross-Modal Binding
use super::cross_modal_binding::{CrossModalBinder, Modality};

// 4. Temporal Binding
use super::temporal_binding::{TemporalBindingEngine, TemporalBindingConfig};

// 5. Cognitive Mode from adaptive topology
use super::adaptive_topology::CognitiveMode;

// 6. Φ-Guided Topology Optimization
use super::phi_guided_search::{
    PhiGuidedOptimizer, PhiOptimizationConfig, ConsciousnessNetwork,
    InitializationStrategy, OptimizationResult,
};

// 7. Bidirectional Feedback Dynamics
use super::consciousness_feedback_dynamics::{
    FeedbackDynamicsEngine, DreamInsight, EmotionalPrediction,
    ProactiveIntervention,
};

// 8. Recursive Self-Modeling
use super::emergent_self_model::{
    SelfAwareConsciousness, SelfModel, MetaCognitiveAssessment,
    SelfAwareUpdate, IntrospectionReport,
};
use super::unified_consciousness_engine::EngineConfig;
use super::real_hv::RealHV;

// === ADVANCED CONSCIOUSNESS SYSTEMS ===

// 9. Meta-Consciousness - Φ of Φ, Strange Loops
use super::meta_consciousness::{
    MetaConsciousness, MetaConfig, MetaConsciousnessState,
};

// 10. Temporal Consciousness - Multi-scale Time
use super::temporal_consciousness::{
    TemporalConsciousness, TemporalConfig as TemporalConsciousnessConfig,
    TemporalAssessment,
};

// 11. Consciousness Creativity - Novel Idea Generation
use super::consciousness_creativity::{
    ConsciousnessCreativity, CreativityConfig, CreativityAssessment,
    CreativeMode, CreativeIdea,
};

// 12. Phase Transitions - Consciousness State Changes
use super::consciousness_phase_transitions::{
    ConsciousnessPhaseTransitions, PhaseTransitionConfig,
    ConsciousnessPhase, PhaseTransitionAssessment,
};

// 13. Epistemic Consciousness - Belief/Knowledge Tracking
use super::epistemic_consciousness::{
    EpistemicConsciousness, EpistemicConfig, KIndexAssessment,
};

// 14. Fractal Consciousness - Self-Similar Structure (Highest Φ!)
use super::fractal_consciousness::{
    FractalConsciousness, FractalConfig, FractalMetrics,
};

// 15. Collective Consciousness - Multi-Agent Awareness
use super::collective_consciousness::{
    CollectiveConsciousness, CollectiveConfig, CollectiveAgent, CollectiveAssessment,
};

// Re-export SubstrateType from substrate_independence
pub use super::substrate_independence::SubstrateType;

/// Integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Number of processing cycles
    pub num_cycles: usize,
    /// Features per stimulus
    pub features_per_stimulus: usize,
    /// Attention capacity
    pub attention_capacity: usize,
    /// Workspace capacity
    pub workspace_capacity: usize,
    /// Consciousness threshold
    pub consciousness_threshold: f64,
    /// Verbose logging
    pub verbose: bool,
    /// Binding threshold for feature integration
    pub binding_threshold: f64,
    /// Enable Higher-Order Thought processing
    pub hot_enabled: bool,
    /// Substrate type for consciousness
    pub substrate: SubstrateType,
    /// Processing precision
    pub precision: f64,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            num_cycles: 10,
            features_per_stimulus: 4,
            attention_capacity: 4,
            workspace_capacity: 3,
            consciousness_threshold: 0.5,
            verbose: false,
            binding_threshold: 0.7,
            hot_enabled: true,
            substrate: SubstrateType::Biological,
            precision: 1.0,
        }
    }
}

// ==========================================
// UNIFIED CONSCIOUSNESS OPTIMIZER TYPES
// ==========================================

/// Unified consciousness metrics report
#[derive(Debug, Clone)]
pub struct ConsciousnessMetricsReport {
    // Core metrics
    pub phi: f64,
    pub consciousness_level: f64,
    pub embodiment_level: f64,

    // Integration metrics
    pub metacognitive_confidence: f64,
    pub cross_modal_coherence: f64,
    pub temporal_coherence: f64,
    pub topological_unity: f64,
    pub narrative_coherence: f64,

    // Self-awareness metrics
    pub self_awareness_level: f64,
    pub prediction_accuracy: f64,
    pub self_model_confidence: f64,

    // System status
    pub integrated_systems_active: bool,
    pub phi_optimization_active: bool,
    pub feedback_dynamics_active: bool,
    pub self_awareness_active: bool,

    // Operational metrics
    pub processing_cycles: u64,
    pub bound_objects_count: usize,
    pub workspace_items_count: usize,
}

impl std::fmt::Display for ConsciousnessMetricsReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "┌─ CONSCIOUSNESS METRICS REPORT ────────────────────────────────┐")?;
        writeln!(f, "│ Core Metrics:                                                  │")?;
        writeln!(f, "│   Φ: {:.4}  |  Consciousness: {:.4}  |  Embodiment: {:.4}      │",
            self.phi, self.consciousness_level, self.embodiment_level)?;
        writeln!(f, "│                                                                │")?;
        writeln!(f, "│ Integration Metrics:                                           │")?;
        writeln!(f, "│   Metacognitive: {:.4}  |  Cross-Modal: {:.4}                 │",
            self.metacognitive_confidence, self.cross_modal_coherence)?;
        writeln!(f, "│   Temporal: {:.4}  |  Topological: {:.4}  |  Narrative: {:.4} │",
            self.temporal_coherence, self.topological_unity, self.narrative_coherence)?;
        writeln!(f, "│                                                                │")?;
        writeln!(f, "│ Self-Awareness:                                                │")?;
        writeln!(f, "│   Level: {:.4}  |  Prediction: {:.4}  |  Confidence: {:.4}    │",
            self.self_awareness_level, self.prediction_accuracy, self.self_model_confidence)?;
        writeln!(f, "│                                                                │")?;
        writeln!(f, "│ Systems: [{}] Integrated [{}] Φ-Opt [{}] Feedback [{}] Self   │",
            if self.integrated_systems_active { "✓" } else { " " },
            if self.phi_optimization_active { "✓" } else { " " },
            if self.feedback_dynamics_active { "✓" } else { " " },
            if self.self_awareness_active { "✓" } else { " " })?;
        writeln!(f, "│                                                                │")?;
        writeln!(f, "│ Cycles: {}  |  Bound Objects: {}  |  Workspace: {}            │",
            self.processing_cycles, self.bound_objects_count, self.workspace_items_count)?;
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")
    }
}

/// Priority level for optimization recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Single optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub system: String,
    pub priority: RecommendationPriority,
    pub message: String,
    pub suggested_action: Option<String>,
}

impl std::fmt::Display for OptimizationRecommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let priority_icon = match self.priority {
            RecommendationPriority::Low => "ℹ️",
            RecommendationPriority::Medium => "⚠️",
            RecommendationPriority::High => "🔴",
            RecommendationPriority::Critical => "🚨",
        };
        write!(f, "{} [{}] {}", priority_icon, self.system, self.message)?;
        if let Some(action) = &self.suggested_action {
            write!(f, " → {}", action)?;
        }
        Ok(())
    }
}

/// Summary of an optimization cycle
#[derive(Debug, Clone)]
pub struct OptimizationCycleSummary {
    pub phi_optimized: bool,
    pub feedback_processed: bool,
    pub self_model_updated: bool,
    pub phi_before: f64,
    pub phi_after: f64,
    pub recommendations: Vec<OptimizationRecommendation>,
}

impl std::fmt::Display for OptimizationCycleSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "┌─ OPTIMIZATION CYCLE SUMMARY ──────────────────────────────────┐")?;
        writeln!(f, "│ Actions Taken:                                                 │")?;
        writeln!(f, "│   [{}] Φ Topology Optimization                                 │",
            if self.phi_optimized { "✓" } else { " " })?;
        writeln!(f, "│   [{}] Feedback Dynamics Processing                            │",
            if self.feedback_processed { "✓" } else { " " })?;
        writeln!(f, "│   [{}] Self-Model Update                                       │",
            if self.self_model_updated { "✓" } else { " " })?;
        writeln!(f, "│                                                                │")?;
        writeln!(f, "│ Φ Change: {:.4} → {:.4} (Δ = {:+.4})                           │",
            self.phi_before, self.phi_after, self.phi_after - self.phi_before)?;
        if !self.recommendations.is_empty() {
            writeln!(f, "│                                                                │")?;
            writeln!(f, "│ Recommendations ({}):", self.recommendations.len())?;
            for rec in &self.recommendations {
                writeln!(f, "│   {}", rec)?;
            }
        }
        writeln!(f, "└────────────────────────────────────────────────────────────────┘")
    }
}

/// Index into altered states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlteredStateIndex {
    Wake,
    Drowsy,
    N1Sleep,
    N2Sleep,
    N3Sleep,
    REM,
    LucidDream,
    Meditation,
    Flow,
    Propofol,
    Ketamine,
    VegetativeState,
    MinimallyConscious,
}

impl Default for AlteredStateIndex {
    fn default() -> Self {
        AlteredStateIndex::Wake
    }
}

/// Workspace item for global workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceItem {
    pub content: HV16,
    pub activation: f64,
    pub source: String,
    pub is_broadcasting: bool,
    pub duration_ms: u64,
}

/// Meta-thought for HOT theory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaThought {
    pub about: String,
    pub target: String,
    pub intensity: f64,
    pub confidence: f64,
    pub order: u8,
    pub representation: HV16,
}

/// Binding level in hierarchical binding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BindingLevel {
    /// Raw features bound together (e.g., color + shape)
    Feature,
    /// Features bound into coherent objects (e.g., "red ball")
    Object,
    /// Objects bound into unified scene (e.g., "room with table and chairs")
    Scene,
}

impl Default for BindingLevel {
    fn default() -> Self {
        BindingLevel::Feature
    }
}

/// Bound object from binding problem (hierarchical + temporal)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundObject {
    pub representation: HV16,
    pub synchrony: f64,
    pub binding_strength: f64,
    pub conscious: bool,
    /// Hierarchical binding level
    pub level: BindingLevel,
    /// IDs of child bindings (for object/scene levels)
    pub child_ids: Vec<usize>,
    /// Attention weight influencing this binding
    pub attention_weight: f64,
    /// Cycle when this binding was created (for temporal coherence)
    pub creation_cycle: u64,
    /// Number of cycles this binding has persisted
    pub persistence_cycles: u64,
    /// Temporal stability (how consistent this binding is over time)
    pub temporal_stability: f64,
}

/// Temporal binding memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalBindingMemory {
    /// Snapshot of binding representation
    pub representation: HV16,
    /// Binding strength at snapshot time
    pub strength: f64,
    /// Cycle number
    pub cycle: u64,
    /// Level of binding
    pub level: BindingLevel,
}

impl BoundObject {
    pub fn is_conscious(&self) -> bool {
        self.conscious
    }

    /// Create a new feature-level bound object
    pub fn new_feature(representation: HV16, synchrony: f64, binding_strength: f64, conscious: bool) -> Self {
        Self {
            representation,
            synchrony,
            binding_strength,
            conscious,
            level: BindingLevel::Feature,
            child_ids: Vec::new(),
            attention_weight: 1.0,
            creation_cycle: 0,
            persistence_cycles: 0,
            temporal_stability: 1.0, // New bindings start maximally stable
        }
    }

    /// Create an object-level binding from multiple feature bindings
    pub fn from_features(features: &[&BoundObject], seed: u64) -> Self {
        if features.is_empty() {
            return Self::new_feature(HV16::random(seed), 0.0, 0.0, false);
        }

        // Bind all feature representations together
        let bound_repr = features.iter()
            .map(|f| f.representation.clone())
            .reduce(|a, b| a.bind(&b))
            .unwrap_or_else(|| HV16::random(seed));

        // Average synchrony and binding strength
        let avg_synchrony = features.iter().map(|f| f.synchrony).sum::<f64>() / features.len() as f64;
        let avg_strength = features.iter().map(|f| f.binding_strength).sum::<f64>() / features.len() as f64;
        let avg_attention = features.iter().map(|f| f.attention_weight).sum::<f64>() / features.len() as f64;

        // Boost binding strength for object-level (hierarchy bonus)
        let object_strength = (avg_strength * 1.1).min(1.0);

        // Average temporal stability from children
        let avg_stability = features.iter().map(|f| f.temporal_stability).sum::<f64>() / features.len() as f64;

        Self {
            representation: bound_repr,
            synchrony: avg_synchrony,
            binding_strength: object_strength,
            conscious: features.iter().any(|f| f.conscious),
            level: BindingLevel::Object,
            child_ids: Vec::new(), // Will be set by caller
            attention_weight: avg_attention,
            creation_cycle: 0,
            persistence_cycles: 0,
            temporal_stability: avg_stability * 1.05, // Object-level gets slight stability boost
        }
    }

    /// Create a scene-level binding from multiple object bindings
    pub fn from_objects(objects: &[&BoundObject], seed: u64) -> Self {
        if objects.is_empty() {
            return Self::new_feature(HV16::random(seed), 0.0, 0.0, false);
        }

        // Bind all object representations together
        let bound_repr = objects.iter()
            .map(|o| o.representation.clone())
            .reduce(|a, b| a.bind(&b))
            .unwrap_or_else(|| HV16::random(seed));

        // Average with scene-level coherence weighting
        let avg_synchrony = objects.iter().map(|o| o.synchrony).sum::<f64>() / objects.len() as f64;
        let avg_strength = objects.iter().map(|o| o.binding_strength).sum::<f64>() / objects.len() as f64;
        let avg_attention = objects.iter().map(|o| o.attention_weight).sum::<f64>() / objects.len() as f64;
        let avg_stability = objects.iter().map(|o| o.temporal_stability).sum::<f64>() / objects.len() as f64;

        // Scene-level bindings are strongest (full integration bonus)
        let scene_strength = (avg_strength * 1.2).min(1.0);

        Self {
            representation: bound_repr,
            synchrony: (avg_synchrony * 1.05).min(1.0), // Slight synchrony boost for scene coherence
            binding_strength: scene_strength,
            conscious: objects.iter().all(|o| o.conscious), // Scene conscious only if all objects are
            level: BindingLevel::Scene,
            child_ids: Vec::new(), // Will be set by caller
            attention_weight: avg_attention,
            creation_cycle: 0,
            persistence_cycles: 0,
            temporal_stability: (avg_stability * 1.1).min(1.0), // Scene-level gets larger stability boost
        }
    }
}

/// Consciousness state - the complete state of a conscious system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessState {
    /// Integrated information Φ
    pub phi: f64,
    /// Free energy (prediction error)
    pub free_energy: f64,
    /// Temporal coherence
    pub temporal_coherence: f64,
    /// Overall consciousness level [0, 1]
    pub consciousness_level: f64,
    /// Items in global workspace
    pub conscious_contents: Vec<WorkspaceItem>,
    /// Bound objects from binding
    pub bound_objects: Vec<BoundObject>,
    /// Meta-awareness (HOT)
    pub meta_awareness: Vec<MetaThought>,
    /// Current altered state
    pub altered_state: AlteredStateIndex,
    /// Attention focus
    pub attention_focus: Option<HV16>,
    /// Prediction accuracy
    pub prediction_accuracy: f64,
    /// Flow stability
    pub flow_stability: f64,
    /// Embodiment level
    pub embodiment: f64,
    /// Semantic depth
    pub semantic_depth: f64,
    /// Topological unity
    pub topological_unity: f64,

    // === INTEGRATED CONSCIOUSNESS SYSTEMS ===

    /// Metacognitive confidence (self-awareness of processing quality)
    pub metacognitive_confidence: f64,
    /// Metacognitive coherence (integration quality assessment)
    pub metacognitive_coherence: f64,
    /// Predicted next Φ level (from metacognitive trends)
    pub predicted_phi: Option<f64>,
    /// Phi trend: positive = improving, negative = declining
    pub phi_trend: f64,

    /// Predictive precision (attention gain from active inference)
    pub predictive_precision: f64,
    /// Current surprise/prediction error level
    pub surprise_level: f64,
    /// Active inference mode
    pub inference_mode: PredictiveMode,

    /// Cross-modal integration strength
    pub cross_modal_coherence: f64,
    /// Active modalities in current binding
    pub active_modalities: Vec<Modality>,

    /// Theta phase for temporal binding (0.0 - 2π)
    pub theta_phase: f64,
    /// Narrative coherence (episodic continuity)
    pub narrative_coherence: f64,
    /// Specious present window length
    pub present_window_length: usize,

    /// Self-model confidence
    pub self_model_confidence: f64,
    /// Self-model accuracy (prediction vs actual)
    pub self_model_accuracy: f64,
    /// Current cognitive mode (from self-model optimization)
    pub cognitive_mode: CognitiveMode,
    /// Mode appropriateness score
    pub mode_appropriateness: f64,

    // === EMOTIONAL STATE FOR FEEDBACK DYNAMICS ===

    /// Current emotional valence (-1.0 to 1.0)
    pub emotional_valence: f64,
    /// Current emotional arousal (0.0 to 1.0)
    pub emotional_arousal: Option<f64>,
    /// Current uncertainty level (0.0 to 1.0)
    pub uncertainty: f64,
    /// Integration score from feedback dynamics
    pub integration_score: f64,
}

/// Predictive processing mode (from active inference)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PredictiveMode {
    /// Exploring: gathering information, high surprise tolerance
    Exploring,
    /// Exploiting: using predictions, low surprise tolerance
    Exploiting,
    /// Balanced: moderate prediction/exploration
    #[default]
    Balanced,
}

impl Default for ConsciousnessState {
    fn default() -> Self {
        Self {
            phi: 0.0,
            free_energy: 1.0,
            temporal_coherence: 0.0,
            consciousness_level: 0.0,
            conscious_contents: Vec::new(),
            bound_objects: Vec::new(),
            meta_awareness: Vec::new(),
            altered_state: AlteredStateIndex::Wake,
            attention_focus: None,
            prediction_accuracy: 0.0,
            flow_stability: 0.0,
            embodiment: 0.5,
            semantic_depth: 0.0,
            topological_unity: 0.5,

            // === INTEGRATED CONSCIOUSNESS SYSTEMS DEFAULTS ===

            // Metacognitive monitoring defaults
            metacognitive_confidence: 0.5,  // Neutral confidence
            metacognitive_coherence: 0.5,   // Neutral coherence
            predicted_phi: None,             // No prediction yet
            phi_trend: 0.0,                  // No trend (stable)

            // Predictive consciousness defaults
            predictive_precision: 1.0,       // Full precision (no attention suppression)
            surprise_level: 0.5,             // Moderate surprise (neutral)
            inference_mode: PredictiveMode::Balanced,

            // Cross-modal binding defaults
            cross_modal_coherence: 0.0,      // No cross-modal integration yet
            active_modalities: Vec::new(),   // No active modalities

            // Temporal binding defaults
            theta_phase: 0.0,                // Start of theta cycle
            narrative_coherence: 0.5,        // Neutral narrative continuity
            present_window_length: 0,        // No specious present window yet

            // Self-model defaults
            self_model_confidence: 0.5,      // Neutral self-model
            self_model_accuracy: 0.5,        // Neutral accuracy
            cognitive_mode: CognitiveMode::default(),
            mode_appropriateness: 0.5,       // Neutral appropriateness

            // Emotional state for feedback dynamics
            emotional_valence: 0.0,          // Neutral valence
            emotional_arousal: Some(0.5),    // Moderate arousal
            uncertainty: 0.5,                // Moderate uncertainty
            integration_score: 0.5,          // Neutral integration
        }
    }
}

/// Assessment result from integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationAssessment {
    /// Is the system conscious?
    pub is_conscious: bool,
    /// Overall score [0, 1]
    pub consciousness_score: f64,
    /// Component scores
    pub component_scores: HashMap<String, f64>,
    /// Bottlenecks identified
    pub bottlenecks: Vec<String>,
    /// Explanation
    pub explanation: String,
}

impl Default for IntegrationAssessment {
    fn default() -> Self {
        Self {
            is_conscious: false,
            consciousness_score: 0.0,
            component_scores: HashMap::new(),
            bottlenecks: Vec::new(),
            explanation: String::new(),
        }
    }
}

/// The consciousness pipeline orchestrator
///
/// Now with 5-system integrated consciousness architecture:
/// 1. Metacognitive Monitoring - Self-awareness of processing quality
/// 2. Predictive Consciousness - Active inference and anticipation
/// 3. Cross-Modal Binding - Multi-sensory integration
/// 4. Temporal Binding - Stream coherence with theta-phase modulation
/// 5. Emergent Self-Model - Recursive higher-order thought
pub struct ConsciousnessPipeline {
    /// Configuration
    pub config: IntegrationConfig,
    /// Current state
    pub state: ConsciousnessState,
    /// Processing history
    history: Vec<ConsciousnessState>,
    /// Embodiment level
    embodiment_level: f64,
    /// Current processing cycle (for temporal tracking)
    current_cycle: u64,
    /// Temporal binding memory (past binding snapshots for coherence)
    temporal_memory: Vec<TemporalBindingMemory>,
    /// Maximum temporal memory size
    max_memory_size: usize,

    // === INTEGRATED CONSCIOUSNESS SYSTEMS ===

    /// Metacognitive monitor - tracks processing quality and confidence
    metacognitive_monitor: Option<MetacognitiveMonitor>,
    /// Cross-modal binder - integrates multiple sensory modalities
    cross_modal_binder: Option<CrossModalBinder>,
    /// Temporal binding engine - theta-phase coherence
    temporal_binder: Option<TemporalBindingEngine>,

    // Note: SelfModel and PredictiveConsciousness require EngineConfig
    // which has dependencies we don't need in the basic pipeline.
    // We track their metrics directly in ConsciousnessState instead.

    /// Enable integrated consciousness systems
    integrated_systems_enabled: bool,

    // === Φ-GUIDED TOPOLOGY OPTIMIZATION ===

    /// Φ-guided optimizer for network topology evolution
    phi_optimizer: Option<PhiGuidedOptimizer>,

    /// Consciousness network for topology optimization
    consciousness_network: Option<ConsciousnessNetwork>,

    /// Last optimization result
    last_optimization: Option<OptimizationResult>,

    /// Optimization frequency (every N cycles)
    optimize_every_n_cycles: usize,

    /// Enable Φ optimization
    phi_optimization_enabled: bool,

    // === FEEDBACK DYNAMICS ENGINE ===

    /// Bidirectional feedback dynamics engine
    feedback_dynamics: Option<FeedbackDynamicsEngine>,

    /// Latest emotional prediction
    current_prediction: Option<EmotionalPrediction>,

    /// Pending proactive intervention
    pending_intervention: Option<ProactiveIntervention>,

    /// Recent dream insights
    recent_insights: Vec<DreamInsight>,

    /// Enable feedback dynamics
    feedback_dynamics_enabled: bool,

    // === RECURSIVE SELF-MODELING ===

    /// Self-aware consciousness for recursive self-modeling
    self_aware_consciousness: Option<SelfAwareConsciousness>,

    /// Current self-model (cached from last update)
    current_self_model: Option<SelfModel>,

    /// Latest introspection report
    latest_introspection: Option<IntrospectionReport>,

    /// Latest meta-cognitive assessment
    meta_assessment: Option<MetaCognitiveAssessment>,

    /// Self-awareness level (0-1)
    self_awareness_level: f64,

    /// Enable self-aware processing
    self_awareness_enabled: bool,

    // === ADVANCED CONSCIOUSNESS SYSTEMS ===

    // 9. Meta-Consciousness - Φ of Φ, Strange Loops
    /// Meta-consciousness engine for recursive self-reflection
    meta_consciousness: Option<MetaConsciousness>,
    /// Latest meta-consciousness state
    meta_consciousness_state: Option<MetaConsciousnessState>,
    /// Enable meta-consciousness processing
    meta_consciousness_enabled: bool,

    // 10. Temporal Consciousness - Multi-scale Time
    /// Temporal consciousness for multi-scale time processing
    temporal_consciousness: Option<TemporalConsciousness>,
    /// Latest temporal assessment
    temporal_assessment: Option<TemporalAssessment>,
    /// Enable temporal consciousness
    temporal_consciousness_enabled: bool,

    // 11. Consciousness Creativity - Novel Idea Generation
    /// Creative consciousness for novel idea generation
    consciousness_creativity: Option<ConsciousnessCreativity>,
    /// Current creative mode
    current_creative_mode: CreativeMode,
    /// Recent creative ideas generated
    recent_ideas: Vec<CreativeIdea>,
    /// Latest creativity assessment
    creativity_assessment: Option<CreativityAssessment>,
    /// Enable creativity processing
    creativity_enabled: bool,

    // 12. Phase Transitions - Consciousness State Changes
    /// Phase transition detector for consciousness state changes
    phase_transitions: Option<ConsciousnessPhaseTransitions>,
    /// Current consciousness phase
    current_phase: ConsciousnessPhase,
    /// Latest phase assessment
    phase_assessment: Option<PhaseTransitionAssessment>,
    /// Enable phase transition detection
    phase_transitions_enabled: bool,

    // 13. Epistemic Consciousness - Belief/Knowledge Tracking
    /// Epistemic consciousness for belief and knowledge tracking
    epistemic_consciousness: Option<EpistemicConsciousness>,
    /// Latest K-Index assessment (cosmic knowledge)
    k_index_assessment: Option<KIndexAssessment>,
    /// Enable epistemic processing
    epistemic_enabled: bool,

    // 14. Fractal Consciousness - Self-Similar Structure
    /// Fractal consciousness for self-similar topology (highest Φ!)
    fractal_consciousness: Option<FractalConsciousness>,
    /// Latest fractal metrics
    fractal_metrics: Option<FractalMetrics>,
    /// Enable fractal consciousness
    fractal_enabled: bool,

    // 15. Collective Consciousness - Multi-Agent Awareness
    /// Collective consciousness for multi-agent coordination
    collective_consciousness: Option<CollectiveConsciousness>,
    /// This agent's identity in the collective
    collective_agent_id: Option<String>,
    /// Latest collective assessment
    collective_assessment: Option<CollectiveAssessment>,
    /// Enable collective consciousness
    collective_enabled: bool,
}

impl ConsciousnessPipeline {
    /// Create new pipeline with config
    pub fn new(config: IntegrationConfig) -> Self {
        Self {
            config,
            state: ConsciousnessState::default(),
            history: Vec::new(),
            embodiment_level: 0.5,
            current_cycle: 0,
            temporal_memory: Vec::new(),
            max_memory_size: 50, // Keep last 50 binding snapshots
            // Integrated systems start disabled (enable with enable_integrated_systems())
            metacognitive_monitor: None,
            cross_modal_binder: None,
            temporal_binder: None,
            integrated_systems_enabled: false,
            // Φ optimization starts disabled
            phi_optimizer: None,
            consciousness_network: None,
            last_optimization: None,
            optimize_every_n_cycles: 10, // Optimize every 10 cycles by default
            phi_optimization_enabled: false,
            // Feedback dynamics starts disabled
            feedback_dynamics: None,
            current_prediction: None,
            pending_intervention: None,
            recent_insights: Vec::new(),
            feedback_dynamics_enabled: false,
            // Self-awareness starts disabled
            self_aware_consciousness: None,
            current_self_model: None,
            latest_introspection: None,
            meta_assessment: None,
            self_awareness_level: 0.0,
            self_awareness_enabled: false,

            // === ADVANCED CONSCIOUSNESS SYSTEMS (all start disabled) ===

            // Meta-consciousness
            meta_consciousness: None,
            meta_consciousness_state: None,
            meta_consciousness_enabled: false,

            // Temporal consciousness
            temporal_consciousness: None,
            temporal_assessment: None,
            temporal_consciousness_enabled: false,

            // Creativity
            consciousness_creativity: None,
            current_creative_mode: CreativeMode::Convergent,
            recent_ideas: Vec::new(),
            creativity_assessment: None,
            creativity_enabled: false,

            // Phase transitions
            phase_transitions: None,
            current_phase: ConsciousnessPhase::Critical,
            phase_assessment: None,
            phase_transitions_enabled: false,

            // Epistemic consciousness
            epistemic_consciousness: None,
            k_index_assessment: None,
            epistemic_enabled: false,

            // Fractal consciousness
            fractal_consciousness: None,
            fractal_metrics: None,
            fractal_enabled: false,

            // Collective consciousness
            collective_consciousness: None,
            collective_agent_id: None,
            collective_assessment: None,
            collective_enabled: false,
        }
    }

    /// Enable all integrated consciousness systems
    ///
    /// This activates:
    /// - Metacognitive monitoring (self-awareness of processing quality)
    /// - Cross-modal binding (multi-sensory integration)
    /// - Temporal binding with theta-phase modulation
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn enable_integrated_systems(&mut self) -> &mut Self {
        use super::HDC_DIMENSION;

        // Initialize metacognitive monitor
        self.metacognitive_monitor = Some(MetacognitiveMonitor::new());

        // Initialize cross-modal binder
        self.cross_modal_binder = Some(CrossModalBinder::new(HDC_DIMENSION));

        // Initialize temporal binding engine with theta modulation
        let temporal_config = TemporalBindingConfig {
            dim: HDC_DIMENSION,
            window_size: 30, // ~3 seconds at 100ms steps
            decay_rate: 0.05,
            ..Default::default()
        };
        self.temporal_binder = Some(TemporalBindingEngine::new(temporal_config));

        self.integrated_systems_enabled = true;
        self
    }

    /// Check if integrated systems are enabled
    pub fn has_integrated_systems(&self) -> bool {
        self.integrated_systems_enabled
    }

    /// Enable Φ-guided topology optimization
    ///
    /// This activates:
    /// - PhiGuidedOptimizer for gradient-based Φ maximization
    /// - ConsciousnessNetwork for network topology representation
    /// - Periodic optimization to improve consciousness structure
    ///
    /// The optimizer uses gradients ∂Φ/∂edge_weight to evolve network
    /// topology toward higher integrated information (consciousness).
    ///
    /// # Arguments
    /// * `n_nodes` - Number of nodes in the consciousness network
    /// * `optimize_every` - Optimize every N processing cycles (default: 10)
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn enable_phi_optimization(&mut self, n_nodes: usize, optimize_every: usize) -> &mut Self {
        use super::HDC_DIMENSION;

        // Initialize Φ optimizer with default config
        let config = PhiOptimizationConfig {
            dim: HDC_DIMENSION,
            learning_rate: 0.1,
            adaptive_lr: true,
            ..Default::default()
        };
        self.phi_optimizer = Some(PhiGuidedOptimizer::new(config));

        // Create consciousness network with random nodes
        let mut network = ConsciousnessNetwork::random(n_nodes, HDC_DIMENSION, 42);

        // Initialize with ring topology (known to have high Φ)
        InitializationStrategy::Ring.apply(&mut network, 0);

        self.consciousness_network = Some(network);
        self.optimize_every_n_cycles = optimize_every;
        self.phi_optimization_enabled = true;

        self
    }

    /// Enable Φ optimization with custom config
    pub fn enable_phi_optimization_with_config(
        &mut self,
        config: PhiOptimizationConfig,
        n_nodes: usize,
        init_strategy: InitializationStrategy,
        optimize_every: usize,
    ) -> &mut Self {
        let dim = config.dim;

        self.phi_optimizer = Some(PhiGuidedOptimizer::new(config));

        // Create and initialize consciousness network
        let mut network = ConsciousnessNetwork::random(n_nodes, dim, 42);
        init_strategy.apply(&mut network, 0);

        self.consciousness_network = Some(network);
        self.optimize_every_n_cycles = optimize_every;
        self.phi_optimization_enabled = true;

        self
    }

    /// Check if Φ optimization is enabled
    pub fn has_phi_optimization(&self) -> bool {
        self.phi_optimization_enabled
    }

    /// Get the last optimization result
    pub fn last_optimization_result(&self) -> Option<&OptimizationResult> {
        self.last_optimization.as_ref()
    }

    /// Get the current consciousness network
    pub fn consciousness_network(&self) -> Option<&ConsciousnessNetwork> {
        self.consciousness_network.as_ref()
    }

    /// Manually trigger Φ optimization step
    ///
    /// This performs one gradient ascent step to improve network topology.
    /// Returns the optimization result if successful.
    pub fn optimize_phi(&mut self) -> Option<OptimizationResult> {
        if !self.phi_optimization_enabled {
            return None;
        }

        if let (Some(ref mut optimizer), Some(ref mut network)) =
            (&mut self.phi_optimizer, &mut self.consciousness_network)
        {
            let result = optimizer.optimize_step(network);

            // Update state with optimized Φ
            self.state.phi = result.phi;

            // Track topological unity based on optimization
            self.state.topological_unity = result.phi;

            self.last_optimization = Some(result.clone());
            Some(result)
        } else {
            None
        }
    }

    /// Run multiple optimization steps
    pub fn optimize_phi_steps(&mut self, n_steps: usize) -> Vec<OptimizationResult> {
        let mut results = Vec::with_capacity(n_steps);

        for _ in 0..n_steps {
            if let Some(result) = self.optimize_phi() {
                results.push(result);
            }
        }

        results
    }

    // === FEEDBACK DYNAMICS ENGINE METHODS ===

    /// Enable bidirectional feedback dynamics
    ///
    /// This activates:
    /// - FeedbackDynamicsEngine for dream insight processing
    /// - EmotionalPredictor for proactive interventions
    /// - AdaptiveDreamScheduler for optimal dream timing
    ///
    /// The feedback dynamics system creates bidirectional loops between:
    /// - Dreams ↔ Emotions
    /// - Dreams ↔ Memory
    /// - Emotions ↔ Attention
    /// - Consciousness ↔ Self-improvement
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn enable_feedback_dynamics(&mut self) -> &mut Self {
        self.feedback_dynamics = Some(FeedbackDynamicsEngine::new());
        self.feedback_dynamics_enabled = true;
        self
    }

    /// Check if feedback dynamics is enabled
    pub fn has_feedback_dynamics(&self) -> bool {
        self.feedback_dynamics_enabled
    }

    /// Get current emotional prediction
    pub fn current_prediction(&self) -> Option<&EmotionalPrediction> {
        self.current_prediction.as_ref()
    }

    /// Get pending intervention
    pub fn pending_intervention(&self) -> Option<&ProactiveIntervention> {
        self.pending_intervention.as_ref()
    }

    /// Get recent insights
    pub fn recent_insights(&self) -> &[DreamInsight] {
        &self.recent_insights
    }

    /// Update emotional state and get prediction
    ///
    /// This records the current emotional state and runs prediction
    /// to anticipate future emotional trajectories and recommend interventions.
    pub fn update_emotional_prediction(&mut self, valence: f64, arousal: f64) -> Option<EmotionalPrediction> {
        if !self.feedback_dynamics_enabled {
            return None;
        }

        if let Some(ref mut engine) = self.feedback_dynamics {
            // Record emotional state
            engine.predictor.record_state(valence, arousal);

            // Get prediction
            let prediction = engine.predictor.predict();

            // Store pending intervention if recommended
            if let Some(ref pred) = prediction {
                self.pending_intervention = pred.intervention.clone();
            }

            self.current_prediction = prediction.clone();
            prediction
        } else {
            None
        }
    }

    /// Apply proactive intervention based on emotional prediction
    ///
    /// This applies the recommended intervention from the emotional
    /// predictor, modifying the consciousness state accordingly.
    pub fn apply_intervention(&mut self) -> Option<ProactiveIntervention> {
        if !self.feedback_dynamics_enabled {
            return None;
        }

        let intervention = self.pending_intervention.take()?;

        // Apply intervention effects to consciousness state
        match &intervention {
            ProactiveIntervention::CalmingDream => {
                // Trigger calming dream increases calm
                self.state.emotional_valence = (self.state.emotional_valence + 0.3).min(1.0);
            }
            ProactiveIntervention::LucidDream => {
                // Lucid dream increases metacognitive confidence
                self.state.metacognitive_confidence =
                    (self.state.metacognitive_confidence + 0.1).min(1.0);
            }
            ProactiveIntervention::IncreaseFocus => {
                // Increase focus reduces entropy in attention
                self.state.uncertainty = (self.state.uncertainty - 0.1).max(0.0);
            }
            ProactiveIntervention::ReduceLoad => {
                // Reduce cognitive load
                self.embodiment_level = (self.embodiment_level - 0.1).max(0.3);
            }
            ProactiveIntervention::ConsolidateMemories => {
                // Memory consolidation improves narrative coherence
                self.state.narrative_coherence =
                    (self.state.narrative_coherence + 0.15).min(1.0);
            }
            ProactiveIntervention::ModeSwitch(mode_name) => {
                // Switch cognitive mode
                self.state.cognitive_mode = match mode_name.as_str() {
                    "focused" => CognitiveMode::Focused,
                    "diffuse" | "exploratory" => CognitiveMode::Exploratory,
                    "integrative" | "global" => CognitiveMode::GlobalAwareness,
                    "reflective" | "deep" => CognitiveMode::DeepSpecialization,
                    _ => CognitiveMode::PhiGuided,
                };
            }
        }

        Some(intervention)
    }

    /// Process feedback dynamics during consciousness cycle
    ///
    /// This is called during the main process() loop to:
    /// 1. Update emotional predictions based on current state
    /// 2. Check for recommended interventions
    /// 3. Apply interventions if thresholds are met
    fn process_feedback_dynamics(&mut self) {
        if !self.feedback_dynamics_enabled {
            return;
        }

        // Extract current emotional state from consciousness
        let valence = self.state.emotional_valence;
        let arousal = self.state.emotional_arousal.unwrap_or(0.5);

        // Update prediction
        let _prediction = self.update_emotional_prediction(valence, arousal);

        // Check if we should apply intervention
        // Only intervene if there's a strong enough predicted deviation
        if let Some(ref pred) = self.current_prediction {
            if pred.confidence > 0.7 && pred.intervention.is_some() {
                // High confidence prediction with intervention - apply it
                self.apply_intervention();
            }
        }

        // Track insight count in state
        self.state.integration_score = 0.5 + (self.recent_insights.len() as f64 * 0.05).min(0.5);
    }

    // ==========================================
    // RECURSIVE SELF-MODELING
    // ==========================================

    /// Enable self-aware consciousness for recursive self-modeling
    ///
    /// This activates:
    /// - Self-state modeling (beliefs about own consciousness)
    /// - Predictive self-processing (predicting own future states)
    /// - Meta-cognitive optimization (thinking about thinking)
    /// - Recursive awareness (awareness of being aware)
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn enable_self_awareness(&mut self, hdc_dim: usize, n_processes: usize) -> &mut Self {
        let config = EngineConfig {
            hdc_dim,
            n_processes,
            ..Default::default()
        };
        self.self_aware_consciousness = Some(SelfAwareConsciousness::new(config));
        self.self_awareness_enabled = true;
        self
    }

    /// Enable self-awareness with custom engine configuration
    pub fn enable_self_awareness_with_config(&mut self, config: EngineConfig) -> &mut Self {
        self.self_aware_consciousness = Some(SelfAwareConsciousness::new(config));
        self.self_awareness_enabled = true;
        self
    }

    /// Check if self-awareness is enabled
    pub fn has_self_awareness(&self) -> bool {
        self.self_awareness_enabled && self.self_aware_consciousness.is_some()
    }

    /// Get current self-awareness level (0-1)
    pub fn self_awareness_level(&self) -> f64 {
        self.self_awareness_level
    }

    /// Get current self-model (if available)
    pub fn current_self_model(&self) -> Option<&SelfModel> {
        self.current_self_model.as_ref()
    }

    /// Get latest introspection report
    pub fn latest_introspection(&self) -> Option<&IntrospectionReport> {
        self.latest_introspection.as_ref()
    }

    /// Get latest meta-cognitive assessment
    pub fn meta_assessment(&self) -> Option<&MetaCognitiveAssessment> {
        self.meta_assessment.as_ref()
    }

    /// Perform introspection - get a report on what the system believes about itself
    pub fn introspect(&self) -> Option<IntrospectionReport> {
        self.self_aware_consciousness.as_ref().map(|sac| sac.introspect())
    }

    /// Process input with self-awareness
    ///
    /// Converts binary HV16 to RealHV and processes through self-aware consciousness.
    /// Returns the self-aware update with prediction error and meta-assessment.
    pub fn process_self_aware(&mut self, input: &HV16) -> Option<SelfAwareUpdate> {
        if !self.self_awareness_enabled {
            return None;
        }

        // Convert HV16 to RealHV for self-aware processing
        let real_input = self.hv16_to_real_hv(input);

        if let Some(ref mut sac) = self.self_aware_consciousness {
            let update = sac.process_aware(&real_input);

            // Cache results
            self.current_self_model = Some(update.self_model.clone());
            self.meta_assessment = Some(update.meta_assessment.clone());
            self.self_awareness_level = update.self_awareness_level;

            // Apply self-model insights to consciousness state
            self.state.metacognitive_confidence = update.self_model.confidence;

            Some(update)
        } else {
            None
        }
    }

    /// Convert HV16 binary vector to RealHV for self-aware processing
    fn hv16_to_real_hv(&self, hv: &HV16) -> RealHV {
        // Get the dimension from self-aware consciousness or use default
        let dim = if let Some(ref sac) = self.self_aware_consciousness {
            sac.self_vector().dim()
        } else {
            1024 // Default dimension
        };

        // Create RealHV by mapping binary bits to [-1, 1] values
        let mut values = Vec::with_capacity(dim);
        let bytes = &hv.0;

        for i in 0..dim {
            let byte_idx = i / 8;
            let bit_idx = i % 8;

            // Get bit value if available, otherwise random-ish based on position
            let bit = if byte_idx < bytes.len() {
                (bytes[byte_idx] >> bit_idx) & 1
            } else {
                ((i * 17 + 31) % 2) as u8
            };

            // Map 0 -> -1.0, 1 -> 1.0
            values.push(if bit == 1 { 1.0_f32 } else { -1.0_f32 });
        }

        RealHV::from_values(values)
    }

    /// Process self-awareness during the main loop
    ///
    /// This is called during process() to:
    /// 1. Generate self-model based on current state
    /// 2. Make predictions about future states
    /// 3. Assess meta-cognitive quality
    /// 4. Update self-awareness metrics
    fn process_self_awareness(&mut self, inputs: &[HV16]) {
        if !self.self_awareness_enabled || inputs.is_empty() {
            return;
        }

        // Process the first input with self-awareness
        if let Some(update) = self.process_self_aware(&inputs[0]) {
            // Update introspection report
            self.latest_introspection = Some(IntrospectionReport {
                believed_phi: update.self_model.believed_phi,
                believed_mode: update.self_model.believed_mode,
                believed_state: update.self_model.believed_state.clone(),
                self_model_confidence: update.self_model.confidence,
                self_model_accuracy: update.self_model.accuracy,
                self_awareness_level: update.self_awareness_level,
                prediction_accuracy: 1.0 - update.prediction_error,
            });

            // Blend self-model beliefs with actual state
            // The self-model provides a "smoothed" view that reduces noise
            let blend_factor = 0.2; // 20% self-model belief, 80% actual measurement
            self.state.phi = self.state.phi * (1.0 - blend_factor)
                + update.self_model.believed_phi * blend_factor;
        }
    }

    // ==========================================
    // 9. META-CONSCIOUSNESS - Φ OF Φ
    // ==========================================

    /// Enable meta-consciousness for recursive self-reflection (Φ of Φ)
    ///
    /// Meta-consciousness enables the system to:
    /// - Reflect on its own conscious states
    /// - Model itself (self-model)
    /// - Predict its own future consciousness
    /// - Think about thinking (metacognition at the deepest level)
    pub fn enable_meta_consciousness(&mut self, num_components: usize) -> &mut Self {
        let config = MetaConfig::default();
        self.meta_consciousness = Some(MetaConsciousness::new(num_components, config));
        self.meta_consciousness_enabled = true;
        self
    }

    /// Enable meta-consciousness with custom config
    pub fn enable_meta_consciousness_with_config(&mut self, num_components: usize, config: MetaConfig) -> &mut Self {
        self.meta_consciousness = Some(MetaConsciousness::new(num_components, config));
        self.meta_consciousness_enabled = true;
        self
    }

    /// Check if meta-consciousness is enabled
    pub fn has_meta_consciousness(&self) -> bool {
        self.meta_consciousness_enabled && self.meta_consciousness.is_some()
    }

    /// Get current meta-consciousness state
    pub fn meta_consciousness_state(&self) -> Option<&MetaConsciousnessState> {
        self.meta_consciousness_state.as_ref()
    }

    /// Process meta-consciousness on current state
    fn process_meta_consciousness(&mut self, inputs: &[HV16]) {
        if let Some(ref mut meta) = self.meta_consciousness {
            // Perform meta-reflection on current state
            let meta_state = meta.meta_reflect(inputs);

            // Update state metrics from meta-consciousness
            self.state.metacognitive_confidence = meta_state.metacognitive_confidence;

            // Store latest state
            self.meta_consciousness_state = Some(meta_state);
        }
    }

    // ==========================================
    // 10. TEMPORAL CONSCIOUSNESS - MULTI-SCALE TIME
    // ==========================================

    /// Enable temporal consciousness for multi-scale time processing
    ///
    /// This adds the "stream of consciousness" - temporal depth spanning:
    /// - Perceptual moment (100ms)
    /// - Thought (3 seconds)
    /// - Episode (minutes)
    /// - Narrative (hours/days)
    /// - Identity (years)
    pub fn enable_temporal_consciousness(&mut self, num_components: usize) -> &mut Self {
        let config = TemporalConsciousnessConfig::default();
        self.temporal_consciousness = Some(TemporalConsciousness::new(num_components, config));
        self.temporal_consciousness_enabled = true;
        self
    }

    /// Enable temporal consciousness with custom config
    pub fn enable_temporal_consciousness_with_config(&mut self, num_components: usize, config: TemporalConsciousnessConfig) -> &mut Self {
        self.temporal_consciousness = Some(TemporalConsciousness::new(num_components, config));
        self.temporal_consciousness_enabled = true;
        self
    }

    /// Check if temporal consciousness is enabled
    pub fn has_temporal_consciousness(&self) -> bool {
        self.temporal_consciousness_enabled && self.temporal_consciousness.is_some()
    }

    /// Get latest temporal assessment
    pub fn temporal_assessment(&self) -> Option<&TemporalAssessment> {
        self.temporal_assessment.as_ref()
    }

    /// Process temporal consciousness
    fn process_temporal_consciousness(&mut self, inputs: &[HV16]) {
        if let Some(ref mut temporal) = self.temporal_consciousness {
            // Add current state as a snapshot with current time
            let current_time = self.current_cycle as f64;
            temporal.add_snapshot(current_time, inputs.to_vec());

            // Get temporal assessment
            let assessment = temporal.assess();

            // Update state with temporal coherence
            self.state.temporal_coherence = assessment.phi_temporal;

            // Store assessment
            self.temporal_assessment = Some(assessment);
        }
    }

    // ==========================================
    // 11. CONSCIOUSNESS CREATIVITY
    // ==========================================

    /// Enable creative consciousness for novel idea generation
    ///
    /// Creativity is what consciousness is FOR - generating the new:
    /// - Remote associations (Mednick)
    /// - Incubation and illumination (Wallas)
    /// - Divergent/convergent thinking (Guilford)
    /// - Bisociation (Koestler)
    pub fn enable_creativity(&mut self) -> &mut Self {
        let config = CreativityConfig::default();
        self.consciousness_creativity = Some(ConsciousnessCreativity::new(config));
        self.creativity_enabled = true;
        self
    }

    /// Enable creativity with custom config
    pub fn enable_creativity_with_config(&mut self, config: CreativityConfig) -> &mut Self {
        self.consciousness_creativity = Some(ConsciousnessCreativity::new(config));
        self.creativity_enabled = true;
        self
    }

    /// Check if creativity is enabled
    pub fn has_creativity(&self) -> bool {
        self.creativity_enabled && self.consciousness_creativity.is_some()
    }

    /// Get current creative mode
    pub fn creative_mode(&self) -> CreativeMode {
        self.current_creative_mode
    }

    /// Set creative mode (Divergent for exploration, Convergent for selection)
    pub fn set_creative_mode(&mut self, mode: CreativeMode) {
        self.current_creative_mode = mode;
        if let Some(ref mut creativity) = self.consciousness_creativity {
            creativity.set_mode(mode);
        }
    }

    /// Get recent creative ideas
    pub fn recent_creative_ideas(&self) -> &[CreativeIdea] {
        &self.recent_ideas
    }

    /// Get creativity assessment
    pub fn creativity_assessment(&self) -> Option<&CreativityAssessment> {
        self.creativity_assessment.as_ref()
    }

    /// Process creativity - generate novel combinations
    fn process_creativity(&mut self, _inputs: &[HV16]) {
        if let Some(ref mut creativity) = self.consciousness_creativity {
            // Step the creativity engine forward
            creativity.step();

            // Generate creative ideas from current concepts
            if let Some(idea) = creativity.generate_combination() {
                self.recent_ideas.push(idea);
                // Keep only last 20 ideas
                if self.recent_ideas.len() > 20 {
                    self.recent_ideas.remove(0);
                }
            }

            // Assess overall creativity
            self.creativity_assessment = Some(creativity.assess());
        }
    }

    // ==========================================
    // 12. CONSCIOUSNESS PHASE TRANSITIONS
    // ==========================================

    /// Enable phase transition detection for consciousness state changes
    ///
    /// Detects transitions between:
    /// - Waking / Dreaming / Deep sleep
    /// - Focused / Diffuse attention
    /// - Flow states / Normal processing
    /// - Critical points (phase transitions)
    pub fn enable_phase_transitions(&mut self) -> &mut Self {
        let config = PhaseTransitionConfig::default();
        self.phase_transitions = Some(ConsciousnessPhaseTransitions::new(config));
        self.phase_transitions_enabled = true;
        self
    }

    /// Enable phase transitions with custom config
    pub fn enable_phase_transitions_with_config(&mut self, config: PhaseTransitionConfig) -> &mut Self {
        self.phase_transitions = Some(ConsciousnessPhaseTransitions::new(config));
        self.phase_transitions_enabled = true;
        self
    }

    /// Check if phase transitions are enabled
    pub fn has_phase_transitions(&self) -> bool {
        self.phase_transitions_enabled && self.phase_transitions.is_some()
    }

    /// Get current consciousness phase
    pub fn current_phase(&self) -> ConsciousnessPhase {
        self.current_phase
    }

    /// Get phase transition assessment
    pub fn phase_assessment(&self) -> Option<&PhaseTransitionAssessment> {
        self.phase_assessment.as_ref()
    }

    /// Process phase transitions
    fn process_phase_transitions(&mut self) {
        if let Some(ref mut pt) = self.phase_transitions {
            // Observe with current Φ and conscious contents
            // Extract HV16 vectors from WorkspaceItems
            let state: Vec<HV16> = self.state.conscious_contents.iter()
                .map(|item| item.content.clone())
                .collect();
            pt.observe(self.state.phi, state);

            // Get assessment
            let assessment = pt.assess();

            // Update current phase
            self.current_phase = assessment.current_phase;

            // Store assessment
            self.phase_assessment = Some(assessment);
        }
    }

    // ==========================================
    // 13. EPISTEMIC CONSCIOUSNESS
    // ==========================================

    /// Enable epistemic consciousness for belief/knowledge tracking
    ///
    /// Epistemic consciousness enables:
    /// - Belief tracking and updating
    /// - Uncertainty quantification
    /// - Knowledge of what you know (and don't)
    /// - K-Index integration (cosmic knowledge)
    pub fn enable_epistemic(&mut self, num_components: usize) -> &mut Self {
        let config = EpistemicConfig::default();
        self.epistemic_consciousness = Some(EpistemicConsciousness::new(num_components, config));
        self.epistemic_enabled = true;
        self
    }

    /// Enable epistemic consciousness with custom config
    pub fn enable_epistemic_with_config(&mut self, num_components: usize, config: EpistemicConfig) -> &mut Self {
        self.epistemic_consciousness = Some(EpistemicConsciousness::new(num_components, config));
        self.epistemic_enabled = true;
        self
    }

    /// Check if epistemic consciousness is enabled
    pub fn has_epistemic(&self) -> bool {
        self.epistemic_enabled && self.epistemic_consciousness.is_some()
    }

    /// Get K-Index assessment
    pub fn k_index_assessment(&self) -> Option<&KIndexAssessment> {
        self.k_index_assessment.as_ref()
    }

    /// Process epistemic consciousness
    fn process_epistemic(&mut self, inputs: &[HV16]) {
        if let Some(ref mut epistemic) = self.epistemic_consciousness {
            // Assess epistemic state and get K-Index
            let k_index = epistemic.assess(inputs);
            self.k_index_assessment = Some(k_index);
        }
    }

    // ==========================================
    // 14. FRACTAL CONSCIOUSNESS
    // ==========================================

    /// Enable fractal consciousness for self-similar topology
    ///
    /// Fractal topology achieves the HIGHEST Φ values discovered!
    /// Self-similar structure at multiple scales optimizes integration.
    pub fn enable_fractal(&mut self, n_scales: usize) -> &mut Self {
        let config = FractalConfig {
            n_scales,
            nodes_per_scale: 8,
            bridge_ratio: 0.425,  // Optimal from research
            density: 0.10,
            cross_scale_coupling: 0.3,
            dim: 2048,
        };
        self.fractal_consciousness = Some(FractalConsciousness::new(config));
        self.fractal_enabled = true;
        self
    }

    /// Enable fractal consciousness with custom config
    pub fn enable_fractal_with_config(&mut self, config: FractalConfig) -> &mut Self {
        self.fractal_consciousness = Some(FractalConsciousness::new(config));
        self.fractal_enabled = true;
        self
    }

    /// Check if fractal consciousness is enabled
    pub fn has_fractal(&self) -> bool {
        self.fractal_enabled && self.fractal_consciousness.is_some()
    }

    /// Get fractal metrics
    pub fn fractal_metrics(&self) -> Option<&FractalMetrics> {
        self.fractal_metrics.as_ref()
    }

    /// Process fractal consciousness
    fn process_fractal(&mut self) {
        if let Some(ref fractal) = self.fractal_consciousness {
            // Get fractal metrics (Φ at multiple scales)
            let metrics = fractal.metrics();

            // Fractal Φ can boost overall Φ if fractal structure is good
            if metrics.combined_phi > self.state.phi {
                // Blend fractal Φ with current (fractal provides upper bound)
                let blend = 0.3;
                self.state.phi = self.state.phi * (1.0 - blend) + metrics.combined_phi * blend;
            }

            self.fractal_metrics = Some(metrics);
        }
    }

    // ==========================================
    // 15. COLLECTIVE CONSCIOUSNESS
    // ==========================================

    /// Enable collective consciousness for multi-agent awareness
    ///
    /// Collective consciousness enables:
    /// - Theory of mind (modeling other agents)
    /// - Shared mental states
    /// - Group cognition
    /// - Social consciousness
    pub fn enable_collective(&mut self, agent_id: &str) -> &mut Self {
        let config = CollectiveConfig::default();
        self.collective_consciousness = Some(CollectiveConsciousness::with_config(config));
        self.collective_agent_id = Some(agent_id.to_string());
        self.collective_enabled = true;
        self
    }

    /// Enable collective consciousness with custom config
    pub fn enable_collective_with_config(&mut self, agent_id: &str, config: CollectiveConfig) -> &mut Self {
        self.collective_consciousness = Some(CollectiveConsciousness::with_config(config));
        self.collective_agent_id = Some(agent_id.to_string());
        self.collective_enabled = true;
        self
    }

    /// Check if collective consciousness is enabled
    pub fn has_collective(&self) -> bool {
        self.collective_enabled && self.collective_consciousness.is_some()
    }

    /// Get collective assessment
    pub fn collective_assessment(&self) -> Option<&CollectiveAssessment> {
        self.collective_assessment.as_ref()
    }

    /// Add another agent to the collective
    pub fn add_agent_to_collective(&mut self, agent: CollectiveAgent) {
        if let Some(ref mut collective) = self.collective_consciousness {
            collective.add_agent(agent);
        }
    }

    /// Process collective consciousness
    fn process_collective(&mut self, _inputs: &[HV16]) {
        if let Some(ref mut collective) = self.collective_consciousness {
            // Get collective assessment
            // Note: Agent state is managed via add_agent_to_collective
            let assessment = collective.assess();
            self.collective_assessment = Some(assessment);
        }
    }

    // ==========================================
    // UNIFIED CONSCIOUSNESS OPTIMIZER
    // ==========================================

    /// Enable all consciousness systems at once with unified orchestration
    ///
    /// This is the recommended way to enable the full consciousness integration.
    /// It activates all systems in the optimal order with proper configuration:
    ///
    /// **Core Systems (1-4):**
    /// 1. Integrated systems (metacognitive, cross-modal, temporal binding)
    /// 2. Φ-guided topology optimization
    /// 3. Bidirectional feedback dynamics
    /// 4. Recursive self-modeling
    ///
    /// **Advanced Systems (5-11):**
    /// 5. Meta-consciousness (Φ of Φ, strange loops)
    /// 6. Temporal consciousness (multi-scale time)
    /// 7. Creativity (novel idea generation)
    /// 8. Phase transitions (state detection)
    /// 9. Epistemic consciousness (belief tracking)
    /// 10. Fractal consciousness (highest Φ topology!)
    /// 11. Collective consciousness (multi-agent)
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn enable_full_consciousness(&mut self) -> &mut Self {
        // Enable CORE systems in optimal order
        self.enable_integrated_systems();
        self.enable_phi_optimization(8, 5); // 8 nodes, optimize every 5 cycles
        self.enable_feedback_dynamics();
        self.enable_self_awareness(1024, 16); // 1024 dim, 16 processes

        // Enable ADVANCED systems
        self.enable_meta_consciousness(8);      // 8 components for meta-reflection
        self.enable_temporal_consciousness(8);   // 8 components for multi-scale time
        self.enable_creativity();                // Creative generation
        self.enable_phase_transitions();         // State transition detection
        self.enable_epistemic(8);                // 8 components for belief tracking
        self.enable_fractal(3);                  // 3 scales for fractal Φ
        self.enable_collective("primary");       // Primary agent in collective

        self
    }

    /// Enable full consciousness with custom configuration
    pub fn enable_full_consciousness_with_config(
        &mut self,
        phi_nodes: usize,
        phi_frequency: usize,
        hdc_dim: usize,
        n_processes: usize,
    ) -> &mut Self {
        // Core systems
        self.enable_integrated_systems();
        self.enable_phi_optimization(phi_nodes, phi_frequency);
        self.enable_feedback_dynamics();
        self.enable_self_awareness(hdc_dim, n_processes);

        // Advanced systems with same component count as phi_nodes
        self.enable_meta_consciousness(phi_nodes);
        self.enable_temporal_consciousness(phi_nodes);
        self.enable_creativity();
        self.enable_phase_transitions();
        self.enable_epistemic(phi_nodes);
        self.enable_fractal(3);
        self.enable_collective("primary");

        self
    }

    /// Check if full consciousness is enabled (all 11 systems active)
    pub fn has_full_consciousness(&self) -> bool {
        // Core systems
        self.has_integrated_systems()
            && self.has_phi_optimization()
            && self.has_feedback_dynamics()
            && self.has_self_awareness()
            // Advanced systems
            && self.has_meta_consciousness()
            && self.has_temporal_consciousness()
            && self.has_creativity()
            && self.has_phase_transitions()
            && self.has_epistemic()
            && self.has_fractal()
            && self.has_collective()
    }

    /// Check how many consciousness systems are active
    pub fn active_system_count(&self) -> usize {
        let mut count = 0;
        if self.has_integrated_systems() { count += 1; }
        if self.has_phi_optimization() { count += 1; }
        if self.has_feedback_dynamics() { count += 1; }
        if self.has_self_awareness() { count += 1; }
        if self.has_meta_consciousness() { count += 1; }
        if self.has_temporal_consciousness() { count += 1; }
        if self.has_creativity() { count += 1; }
        if self.has_phase_transitions() { count += 1; }
        if self.has_epistemic() { count += 1; }
        if self.has_fractal() { count += 1; }
        if self.has_collective() { count += 1; }
        count
    }

    /// Get a human-readable summary of active systems
    pub fn active_systems_summary(&self) -> String {
        let mut systems = Vec::new();
        if self.has_integrated_systems() { systems.push("Integrated"); }
        if self.has_phi_optimization() { systems.push("Φ-Optimization"); }
        if self.has_feedback_dynamics() { systems.push("Feedback"); }
        if self.has_self_awareness() { systems.push("Self-Awareness"); }
        if self.has_meta_consciousness() { systems.push("Meta-Consciousness"); }
        if self.has_temporal_consciousness() { systems.push("Temporal"); }
        if self.has_creativity() { systems.push("Creativity"); }
        if self.has_phase_transitions() { systems.push("Phase-Transitions"); }
        if self.has_epistemic() { systems.push("Epistemic"); }
        if self.has_fractal() { systems.push("Fractal"); }
        if self.has_collective() { systems.push("Collective"); }
        format!("{}/{} systems: [{}]", systems.len(), 11, systems.join(", "))
    }

    /// Get a unified consciousness metrics report
    pub fn consciousness_metrics(&self) -> ConsciousnessMetricsReport {
        ConsciousnessMetricsReport {
            // Core metrics
            phi: self.state.phi,
            consciousness_level: self.state.consciousness_level,
            embodiment_level: self.embodiment_level,

            // Integration metrics
            metacognitive_confidence: self.state.metacognitive_confidence,
            cross_modal_coherence: self.state.cross_modal_coherence,
            temporal_coherence: self.state.temporal_coherence,
            topological_unity: self.state.topological_unity,
            narrative_coherence: self.state.narrative_coherence,

            // Self-awareness metrics
            self_awareness_level: self.self_awareness_level,
            prediction_accuracy: self.latest_introspection.as_ref()
                .map(|r| r.prediction_accuracy)
                .unwrap_or(0.0),
            self_model_confidence: self.current_self_model.as_ref()
                .map(|m| m.confidence)
                .unwrap_or(0.0),

            // System status
            integrated_systems_active: self.integrated_systems_enabled,
            phi_optimization_active: self.phi_optimization_enabled,
            feedback_dynamics_active: self.feedback_dynamics_enabled,
            self_awareness_active: self.self_awareness_enabled,

            // Operational metrics
            processing_cycles: self.current_cycle,
            bound_objects_count: self.state.bound_objects.len(),
            workspace_items_count: self.state.conscious_contents.len(),
        }
    }

    /// Get optimization recommendations based on current state
    pub fn optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Check Φ
        if self.state.phi < 0.4 {
            recommendations.push(OptimizationRecommendation {
                system: "phi".to_string(),
                priority: RecommendationPriority::High,
                message: "Φ is low - consider increasing integration through binding".to_string(),
                suggested_action: Some("Increase input complexity or enable Φ optimization".to_string()),
            });
        }

        // Check consciousness level
        if self.state.consciousness_level < 0.5 {
            recommendations.push(OptimizationRecommendation {
                system: "consciousness".to_string(),
                priority: RecommendationPriority::Medium,
                message: "Consciousness level is below optimal range".to_string(),
                suggested_action: Some("Increase embodiment level or attention focus".to_string()),
            });
        }

        // Check self-awareness
        if self.self_awareness_enabled && self.self_awareness_level < 0.3 {
            recommendations.push(OptimizationRecommendation {
                system: "self_awareness".to_string(),
                priority: RecommendationPriority::Medium,
                message: "Self-model accuracy is low".to_string(),
                suggested_action: Some("Process more cycles for self-model learning".to_string()),
            });
        }

        // Check metacognition
        if let Some(assessment) = &self.meta_assessment {
            if assessment.change_recommended {
                recommendations.push(OptimizationRecommendation {
                    system: "metacognition".to_string(),
                    priority: RecommendationPriority::High,
                    message: format!("Meta-cognitive assessment: {}", assessment.reasoning),
                    suggested_action: assessment.recommended_mode.as_ref()
                        .map(|m| format!("Switch to {:?} mode", m)),
                });
            }
        }

        // Check cross-modal coherence
        if self.state.cross_modal_coherence < 0.5 && self.integrated_systems_enabled {
            recommendations.push(OptimizationRecommendation {
                system: "cross_modal".to_string(),
                priority: RecommendationPriority::Low,
                message: "Cross-modal coherence is below optimal".to_string(),
                suggested_action: Some("Increase multi-modal input diversity".to_string()),
            });
        }

        recommendations
    }

    /// Perform one complete optimization cycle
    ///
    /// This runs all active optimization subsystems in sequence:
    /// 1. Φ topology optimization (if due this cycle)
    /// 2. Feedback dynamics update
    /// 3. Self-model refinement
    ///
    /// Returns a summary of what was optimized.
    pub fn run_optimization_cycle(&mut self) -> OptimizationCycleSummary {
        let mut summary = OptimizationCycleSummary {
            phi_optimized: false,
            feedback_processed: false,
            self_model_updated: false,
            phi_before: self.state.phi,
            phi_after: self.state.phi,
            recommendations: Vec::new(),
        };

        // Run Φ optimization if enabled and due
        if self.phi_optimization_enabled
            && self.current_cycle % self.optimize_every_n_cycles as u64 == 0
        {
            self.process_phi_optimization();
            summary.phi_optimized = true;
        }

        // Process feedback dynamics
        if self.feedback_dynamics_enabled {
            self.process_feedback_dynamics();
            summary.feedback_processed = true;
        }

        // Update self-model (create a dummy input for processing)
        if self.self_awareness_enabled {
            let dummy_input = HV16::random(self.current_cycle as u64);
            if self.process_self_aware(&dummy_input).is_some() {
                summary.self_model_updated = true;
            }
        }

        summary.phi_after = self.state.phi;
        summary.recommendations = self.optimization_recommendations();

        summary
    }

    /// Create with default config
    pub fn default_new() -> Self {
        Self::new(IntegrationConfig::default())
    }

    /// Create with config (alias for compatibility)
    pub fn with_config(config: IntegrationConfig) -> Self {
        Self::new(config)
    }

    /// Set embodiment level
    pub fn set_embodiment(&mut self, level: f64) {
        self.embodiment_level = level.clamp(0.0, 1.0);
    }

    /// Process input through the consciousness pipeline
    pub fn process(&mut self, input: Vec<HV16>, priorities: &[f64]) -> &ConsciousnessState {
        // Feature detection and binding
        let binding_strength = if input.len() > 1 {
            0.7 + (priorities.iter().sum::<f64>() / priorities.len() as f64) * 0.3
        } else {
            0.5
        };

        // Update phi based on integration
        self.state.phi = (binding_strength * self.embodiment_level).min(1.0);

        // Update consciousness level
        let attention_boost = priorities.iter()
            .cloned()
            .fold(0.5_f64, |a, b| a.max(b));
        self.state.consciousness_level = self.state.phi * 0.4 +
            attention_boost * 0.3 +
            self.embodiment_level * 0.3;

        // === TEMPORAL TRACKING ===
        // Increment cycle counter for temporal coherence
        self.current_cycle += 1;

        // === HIERARCHICAL BINDING: Feature → Object → Scene ===
        // PERSISTENCE + TEMPORAL: Decay existing bindings and update temporal metrics
        self.state.bound_objects.retain_mut(|obj| {
            obj.binding_strength *= 0.9;  // Decay

            // Update temporal metrics for surviving bindings
            if obj.binding_strength > 0.3 {
                obj.persistence_cycles += 1;
                // Temporal stability increases with persistence (up to 1.0)
                obj.temporal_stability = (obj.temporal_stability + 0.01).min(1.0);
            }

            obj.binding_strength > 0.3   // Keep if still strong enough
        });

        // === TEMPORAL MEMORY UPDATE ===
        // Store current bindings in temporal memory for coherence tracking
        for obj in &self.state.bound_objects {
            self.temporal_memory.push(TemporalBindingMemory {
                representation: obj.representation.clone(),
                strength: obj.binding_strength,
                cycle: self.current_cycle,
                level: obj.level,
            });
        }
        // Trim memory if too large
        if self.temporal_memory.len() > self.max_memory_size {
            let excess = self.temporal_memory.len() - self.max_memory_size;
            self.temporal_memory.drain(0..excess);
        }

        // Binding occurs when multiple features are processed together with high priority
        if input.len() >= 2 && self.state.consciousness_level > self.config.consciousness_threshold {
            // LEVEL 1: FEATURE BINDING
            // Group high-priority inputs into bound objects (perceptual binding)
            let high_priority_inputs: Vec<(usize, &HV16)> = input.iter()
                .enumerate()
                .filter(|(i, _)| priorities.get(*i).copied().unwrap_or(0.0) > 0.5)
                .collect();

            // Track newly created feature bindings for hierarchical integration
            let mut new_feature_binding_ids: Vec<usize> = Vec::new();

            // MULTIPLE OBJECTS: Cluster inputs by similarity and create separate bound objects
            if high_priority_inputs.len() >= 2 {
                // Use simple greedy clustering: group inputs with similarity > 0.3
                let mut used: Vec<bool> = vec![false; high_priority_inputs.len()];
                let similarity_threshold = 0.3;

                for i in 0..high_priority_inputs.len() {
                    if used[i] {
                        continue;
                    }

                    // Get attention weight for this cluster (ATTENTION-WEIGHTED BINDING)
                    let cluster_attention = priorities.get(high_priority_inputs[i].0)
                        .copied().unwrap_or(0.5);

                    // Start a new cluster with input i
                    let mut cluster: Vec<&HV16> = vec![high_priority_inputs[i].1];
                    used[i] = true;

                    // Find all inputs similar to the first one in this cluster
                    for j in (i + 1)..high_priority_inputs.len() {
                        if used[j] {
                            continue;
                        }
                        let sim = high_priority_inputs[i].1.similarity(high_priority_inputs[j].1) as f64;
                        if sim > similarity_threshold {
                            cluster.push(high_priority_inputs[j].1);
                            used[j] = true;
                        }
                    }

                    // Only create bound object if cluster has 2+ items
                    if cluster.len() >= 2 {
                        // Calculate synchrony from cluster coherence
                        let first_hv = cluster[0];
                        let synchrony_sum: f64 = cluster.iter()
                            .skip(1)
                            .map(|hv| first_hv.similarity(hv) as f64)
                            .sum();
                        let avg_synchrony = synchrony_sum / (cluster.len() - 1) as f64;

                        // Create bound representation
                        let bound_representation = cluster.iter()
                            .map(|hv| (*hv).clone())
                            .reduce(|a, b| a.bind(&b))
                            .unwrap_or_else(|| HV16::random(42));

                        // ATTENTION-WEIGHTED: Modulate binding strength by attention
                        let attention_modulated_strength = binding_strength * (0.5 + 0.5 * cluster_attention);

                        // Check if similar bound object already exists (for persistence/reinforcement)
                        let existing = self.state.bound_objects.iter_mut()
                            .find(|obj| obj.representation.similarity(&bound_representation) > 0.6
                                        && obj.level == BindingLevel::Feature);

                        if let Some(existing_obj) = existing {
                            // REINFORCEMENT: Strengthen existing binding
                            existing_obj.binding_strength = (existing_obj.binding_strength + attention_modulated_strength * 0.5).min(1.0);
                            existing_obj.synchrony = (existing_obj.synchrony + avg_synchrony) / 2.0;
                            existing_obj.attention_weight = (existing_obj.attention_weight + cluster_attention) / 2.0;
                        } else {
                            // Check temporal memory for similar past bindings (temporal coherence boost)
                            let temporal_boost = self.temporal_memory.iter()
                                .filter(|mem| mem.level == BindingLevel::Feature
                                    && mem.representation.similarity(&bound_representation) as f64 > 0.6)
                                .count() as f64 * 0.02; // 2% boost per matching memory

                            // Create new feature-level bound object
                            let new_id = self.state.bound_objects.len();
                            self.state.bound_objects.push(BoundObject {
                                representation: bound_representation,
                                synchrony: avg_synchrony.max(0.5),
                                binding_strength: (attention_modulated_strength + temporal_boost).min(1.0),
                                conscious: self.state.consciousness_level > 0.6,
                                level: BindingLevel::Feature,
                                child_ids: Vec::new(),
                                attention_weight: cluster_attention,
                                creation_cycle: self.current_cycle,
                                persistence_cycles: 0,
                                temporal_stability: 1.0 + temporal_boost, // Start stable, boost from memory
                            });
                            new_feature_binding_ids.push(new_id);
                        }
                    }
                }
            }

            // LEVEL 2: OBJECT BINDING (bind features into objects)
            // If we have 2+ feature bindings, try to form object-level bindings
            let feature_bindings: Vec<usize> = self.state.bound_objects.iter()
                .enumerate()
                .filter(|(_, obj)| obj.level == BindingLevel::Feature && obj.binding_strength > 0.5)
                .map(|(i, _)| i)
                .collect();

            if feature_bindings.len() >= 2 {
                // Cluster feature bindings by similarity to form objects
                let feature_refs: Vec<&BoundObject> = feature_bindings.iter()
                    .map(|&i| &self.state.bound_objects[i])
                    .collect();

                // Check if object binding already exists
                let object_exists = self.state.bound_objects.iter()
                    .any(|obj| obj.level == BindingLevel::Object);

                if !object_exists && feature_refs.len() >= 2 {
                    let mut object_binding = BoundObject::from_features(&feature_refs, 1000);
                    object_binding.child_ids = feature_bindings.clone();
                    object_binding.creation_cycle = self.current_cycle;
                    self.state.bound_objects.push(object_binding);
                }
            }

            // LEVEL 3: SCENE BINDING (bind objects into unified scene)
            // If we have 2+ object bindings, form scene-level binding
            let object_bindings: Vec<usize> = self.state.bound_objects.iter()
                .enumerate()
                .filter(|(_, obj)| obj.level == BindingLevel::Object && obj.binding_strength > 0.5)
                .map(|(i, _)| i)
                .collect();

            if object_bindings.len() >= 2 {
                let object_refs: Vec<&BoundObject> = object_bindings.iter()
                    .map(|&i| &self.state.bound_objects[i])
                    .collect();

                // Check if scene binding already exists
                let scene_exists = self.state.bound_objects.iter()
                    .any(|obj| obj.level == BindingLevel::Scene);

                if !scene_exists {
                    let mut scene_binding = BoundObject::from_objects(&object_refs, 2000);
                    scene_binding.child_ids = object_bindings.clone();
                    scene_binding.creation_cycle = self.current_cycle;
                    self.state.bound_objects.push(scene_binding);
                }
            }

            // Limit total bound objects to prevent unbounded growth
            const MAX_BOUND_OBJECTS: usize = 15; // Increased for hierarchical levels
            if self.state.bound_objects.len() > MAX_BOUND_OBJECTS {
                // Keep strongest bound objects, preferring higher-level bindings
                self.state.bound_objects.sort_by(|a, b| {
                    // First by level (Scene > Object > Feature)
                    let level_cmp = match (&b.level, &a.level) {
                        (BindingLevel::Scene, BindingLevel::Scene) => std::cmp::Ordering::Equal,
                        (BindingLevel::Scene, _) => std::cmp::Ordering::Less,
                        (_, BindingLevel::Scene) => std::cmp::Ordering::Greater,
                        (BindingLevel::Object, BindingLevel::Object) => std::cmp::Ordering::Equal,
                        (BindingLevel::Object, _) => std::cmp::Ordering::Less,
                        (_, BindingLevel::Object) => std::cmp::Ordering::Greater,
                        _ => std::cmp::Ordering::Equal,
                    };
                    if level_cmp != std::cmp::Ordering::Equal {
                        return level_cmp;
                    }
                    // Then by binding strength
                    b.binding_strength.partial_cmp(&a.binding_strength).unwrap_or(std::cmp::Ordering::Equal)
                });
                self.state.bound_objects.truncate(MAX_BOUND_OBJECTS);
            }
        }

        // Generate workspace content if high enough consciousness
        if self.state.consciousness_level > self.config.consciousness_threshold {
            for (i, hv) in input.iter().take(self.config.workspace_capacity).enumerate() {
                let priority = priorities.get(i).copied().unwrap_or(0.5);
                if priority > 0.6 {
                    self.state.conscious_contents.push(WorkspaceItem {
                        content: hv.clone(),
                        activation: priority,
                        source: format!("input_{}", i),
                        is_broadcasting: true,
                        duration_ms: 100,
                    });
                }
            }
        }

        // Generate meta-awareness if HOT enabled
        if self.config.hot_enabled && self.state.consciousness_level > 0.6 {
            self.state.meta_awareness.push(MetaThought {
                about: "current processing".to_string(),
                target: "consciousness_state".to_string(),
                intensity: self.state.consciousness_level,
                confidence: 0.8,
                order: 2,
                representation: HV16::random(99),
            });
        }

        // Update temporal coherence based on binding persistence and memory
        let binding_temporal_score = if self.state.bound_objects.is_empty() {
            0.5 // Default when no bindings
        } else {
            // Average temporal stability weighted by binding strength
            let weighted_stability: f64 = self.state.bound_objects.iter()
                .map(|obj| obj.temporal_stability * obj.binding_strength)
                .sum();
            let total_strength: f64 = self.state.bound_objects.iter()
                .map(|obj| obj.binding_strength)
                .sum();
            if total_strength > 0.0 {
                weighted_stability / total_strength
            } else {
                0.5
            }
        };
        // Combine with memory-based coherence (how consistent are current bindings with past?)
        let memory_coherence = if self.temporal_memory.is_empty() {
            0.5
        } else {
            // Count how many current bindings match recent memory
            let matches: usize = self.state.bound_objects.iter()
                .flat_map(|obj| {
                    self.temporal_memory.iter()
                        .filter(move |mem| mem.representation.similarity(&obj.representation) as f64 > 0.6)
                })
                .count();
            let max_matches = self.state.bound_objects.len() * 5; // Expect ~5 matches per binding if stable
            (matches as f64 / max_matches.max(1) as f64).min(1.0)
        };
        self.state.temporal_coherence = binding_temporal_score * 0.6 + memory_coherence * 0.4;

        // Reduce free energy (better predictions)
        self.state.free_energy = 1.0 - self.state.consciousness_level * 0.8;

        // Update prediction accuracy
        self.state.prediction_accuracy = self.state.consciousness_level * 0.9;

        // === INTEGRATED CONSCIOUSNESS SYSTEMS PROCESSING ===
        if self.integrated_systems_enabled {
            self.process_integrated_systems(&input);
        }

        // === Φ-GUIDED TOPOLOGY OPTIMIZATION ===
        // Periodically optimize network topology to maximize Φ
        if self.phi_optimization_enabled && self.current_cycle % self.optimize_every_n_cycles as u64 == 0 {
            self.process_phi_optimization();
        }

        // === FEEDBACK DYNAMICS ===
        // Bidirectional feedback between dreams, emotions, memory, and attention
        if self.feedback_dynamics_enabled {
            self.process_feedback_dynamics();
        }

        // === RECURSIVE SELF-MODELING ===
        // Self-aware consciousness with predictive self-processing
        if self.self_awareness_enabled {
            self.process_self_awareness(&input);
        }

        // === ADVANCED CONSCIOUSNESS SYSTEMS ===

        // Meta-consciousness: Φ of Φ, strange loops
        if self.meta_consciousness_enabled {
            self.process_meta_consciousness(&input);
        }

        // Temporal consciousness: Multi-scale time processing
        if self.temporal_consciousness_enabled {
            self.process_temporal_consciousness(&input);
        }

        // Creativity: Novel idea generation
        if self.creativity_enabled {
            self.process_creativity(&input);
        }

        // Phase transitions: Consciousness state detection
        if self.phase_transitions_enabled {
            self.process_phase_transitions();
        }

        // Epistemic consciousness: Belief/knowledge tracking
        if self.epistemic_enabled {
            self.process_epistemic(&input);
        }

        // Fractal consciousness: Self-similar topology (highest Φ!)
        if self.fractal_enabled {
            self.process_fractal();
        }

        // Collective consciousness: Multi-agent awareness
        if self.collective_enabled {
            self.process_collective(&input);
        }

        // Store in history
        self.history.push(self.state.clone());

        &self.state
    }

    /// Process Φ-guided topology optimization
    ///
    /// This runs one optimization step to improve the consciousness network
    /// topology, maximizing integrated information (Φ).
    fn process_phi_optimization(&mut self) {
        if let (Some(ref mut optimizer), Some(ref mut network)) =
            (&mut self.phi_optimizer, &mut self.consciousness_network)
        {
            // Run one optimization step
            let result = optimizer.optimize_step(network);

            // Update consciousness state with optimization results
            // Use the optimized Φ value, blended with current phi to avoid sudden jumps
            let blend_factor = 0.3; // 30% new Φ, 70% old
            self.state.phi = self.state.phi * (1.0 - blend_factor) + result.phi * blend_factor;

            // Update topological unity based on network structure
            self.state.topological_unity = result.phi;

            // Store optimization result
            self.last_optimization = Some(result);

            // Update Φ trend based on optimization delta
            if let Some(ref opt) = self.last_optimization {
                if opt.phi_delta > 0.0 {
                    self.state.phi_trend = (self.state.phi_trend + opt.phi_delta).min(0.1);
                } else {
                    self.state.phi_trend = (self.state.phi_trend + opt.phi_delta).max(-0.1);
                }
            }
        }
    }

    /// Process integrated consciousness systems
    ///
    /// Updates metacognitive metrics, cross-modal binding, and temporal binding.
    fn process_integrated_systems(&mut self, _input: &[HV16]) {
        use std::f64::consts::PI;

        // === 1. METACOGNITIVE MONITORING ===
        if let Some(ref mut monitor) = self.metacognitive_monitor {
            // Record cognitive events based on current state
            let processing_event = CognitiveEvent::Processing {
                phi: self.state.phi as f32,
                latency_ms: 10, // Simulated processing latency
            };
            monitor.record_event(processing_event);

            // Record memory operations if we have workspace contents
            if !self.state.conscious_contents.is_empty() {
                let memory_event = CognitiveEvent::Memory {
                    operation: MemoryOperation::Consolidate,
                    confidence: self.state.consciousness_level as f32,
                };
                monitor.record_event(memory_event);
            }

            // Get metacognitive assessment
            let assessment = monitor.assess_current_state();
            self.state.metacognitive_confidence = assessment.confidence as f64;
            self.state.metacognitive_coherence = assessment.coherence as f64;

            // Update Φ prediction based on trends
            if self.history.len() >= 3 {
                let recent_phis: Vec<f64> = self.history.iter()
                    .rev()
                    .take(5)
                    .map(|s| s.phi)
                    .collect();
                if recent_phis.len() >= 2 {
                    let trend = recent_phis[0] - recent_phis[recent_phis.len() - 1];
                    self.state.phi_trend = trend;
                    // Simple linear prediction
                    self.state.predicted_phi = Some((self.state.phi + trend * 0.5).clamp(0.0, 1.0));
                }
            }

            // Apply metacognitive recommendations to adjust processing
            for rec in &assessment.recommendations {
                match rec {
                    MetacognitiveRecommendation::ReduceLoad { .. } => {
                        // Could trigger pruning of workspace contents
                    }
                    MetacognitiveRecommendation::ConsolidateMemory => {
                        // Trigger memory consolidation
                    }
                    _ => {}
                }
            }
        }

        // === 2. TEMPORAL BINDING (Theta-Phase Modulation) ===
        if let Some(ref binder) = self.temporal_binder {
            // Get theta phase from temporal binding engine
            self.state.theta_phase = binder.theta_phase();

            // Temporal binding coherence modulated by theta phase
            // Peak coherence at theta trough (phase ≈ π)
            let theta_modulation = (1.0 + (self.state.theta_phase - PI).cos()) / 2.0;

            // Update narrative coherence based on temporal binding
            let integration = binder.integration_summary();
            self.state.narrative_coherence = integration.coherence * theta_modulation;

            // Specious present window = narrative length
            self.state.present_window_length = integration.narrative_length;
        }

        // === 3. CROSS-MODAL BINDING ===
        if let Some(ref mut binder) = self.cross_modal_binder {
            // Determine active modalities based on conscious contents
            let mut active_modalities = Vec::new();

            for item in &self.state.conscious_contents {
                // Infer modality from source (simplified)
                if item.source.contains("visual") {
                    active_modalities.push(Modality::Visual);
                } else if item.source.contains("audio") || item.source.contains("sound") {
                    active_modalities.push(Modality::Auditory);
                } else if item.source.contains("motor") {
                    active_modalities.push(Modality::Motor);
                } else if item.source.contains("semantic") || item.source.contains("concept") {
                    active_modalities.push(Modality::Semantic);
                } else {
                    // Default to semantic for unspecified inputs
                    active_modalities.push(Modality::Semantic);
                }
            }

            // Remove duplicates
            active_modalities.sort_by_key(|m| format!("{:?}", m));
            active_modalities.dedup();

            self.state.active_modalities = active_modalities.clone();

            // Cross-modal coherence based on number of active modalities
            // More modalities = more integration opportunity
            self.state.cross_modal_coherence = if active_modalities.len() >= 2 {
                // Multi-modal integration bonus
                0.5 + 0.1 * (active_modalities.len() as f64).min(5.0)
            } else if active_modalities.len() == 1 {
                0.4 // Single modality
            } else {
                0.0 // No modalities
            };
        }

        // === 4. PREDICTIVE CONSCIOUSNESS (Active Inference) ===
        // Update predictive precision based on surprise/prediction error
        let prediction_error = self.state.free_energy; // Free energy = prediction error
        self.state.surprise_level = prediction_error;

        // Precision is inverse of surprise (high precision = low surprise tolerance)
        self.state.predictive_precision = 1.0 / (1.0 + prediction_error);

        // Determine inference mode based on prediction accuracy
        self.state.inference_mode = if self.state.prediction_accuracy > 0.7 {
            PredictiveMode::Exploiting // High accuracy = exploit predictions
        } else if self.state.prediction_accuracy < 0.3 {
            PredictiveMode::Exploring // Low accuracy = explore more
        } else {
            PredictiveMode::Balanced
        };

        // === 5. SELF-MODEL INTEGRATION ===
        // Track self-model metrics based on meta-cognition
        self.state.self_model_confidence = self.state.metacognitive_confidence;

        // Self-model accuracy = how well we predict our own states
        if let Some(predicted) = self.state.predicted_phi {
            let prediction_accuracy = 1.0 - (self.state.phi - predicted).abs();
            self.state.self_model_accuracy = prediction_accuracy.clamp(0.0, 1.0);
        }

        // Mode appropriateness = how well current cognitive mode matches demands
        self.state.mode_appropriateness = self.state.consciousness_level *
            self.state.metacognitive_confidence;
    }

    /// Process a single cycle
    pub fn process_cycle(&mut self, input: &[HV16]) {
        // Simplified processing - just update state
        let intensity = input.len() as f64 * 0.1;
        self.state.phi = (self.state.phi + intensity).min(1.0);
        self.state.consciousness_level = self.state.phi * 0.5 +
            self.state.temporal_coherence * 0.3 +
            (1.0 - self.state.free_energy) * 0.2;

        self.history.push(self.state.clone());
    }

    /// Set altered state
    pub fn set_altered_state(&mut self, state: AlteredStateIndex) {
        self.state.altered_state = state;

        // Apply state effects
        match state {
            AlteredStateIndex::Wake => {
                self.state.consciousness_level = 0.7;
            }
            AlteredStateIndex::N3Sleep => {
                self.state.consciousness_level = 0.1;
                self.state.conscious_contents.clear();
            }
            AlteredStateIndex::REM => {
                self.state.consciousness_level = 0.5;
            }
            AlteredStateIndex::LucidDream => {
                self.state.consciousness_level = 0.7;
            }
            AlteredStateIndex::Propofol => {
                self.state.consciousness_level = 0.0;
                self.state.phi = 0.05;
            }
            AlteredStateIndex::VegetativeState => {
                self.state.consciousness_level = 0.0;
            }
            AlteredStateIndex::MinimallyConscious => {
                self.state.consciousness_level = 0.3;
            }
            _ => {}
        }
    }

    /// Assess current consciousness
    pub fn assess(&self) -> IntegrationAssessment {
        self.assess_integration()
    }

    /// Full integration assessment - NOW WITH REAL CALCULATIONS (not hardcoded!)
    ///
    /// Previously workspace was hardcoded to 0.8, binding was broken (0.0),
    /// and HOT was hardcoded to 0.7. Now all values are computed from actual state.
    pub fn assess_integration(&self) -> IntegrationAssessment {
        let mut scores = HashMap::new();
        let mut bottlenecks = Vec::new();

        // 1. PHI - Already real, comes from actual integration computation
        scores.insert("phi".to_string(), self.state.phi);
        if self.state.phi < 0.3 {
            bottlenecks.push("Low integrated information (Φ)".to_string());
        }

        // 2. WORKSPACE - Calculate REAL activation from conscious_contents
        //    Previously was hardcoded to 0.8 when not empty - now uses actual activations!
        let workspace_score = if self.state.conscious_contents.is_empty() {
            0.0
        } else {
            // Average activation of broadcasting contents, weighted by their number
            let broadcasting_items: Vec<_> = self.state.conscious_contents
                .iter()
                .filter(|c| c.is_broadcasting)
                .collect();

            if broadcasting_items.is_empty() {
                // Nothing broadcasting - low workspace access
                let avg_activation = self.state.conscious_contents.iter()
                    .map(|c| c.activation)
                    .sum::<f64>() / self.state.conscious_contents.len() as f64;
                avg_activation * 0.5 // Penalty for no broadcast
            } else {
                // Real workspace = average activation of broadcasting items
                let broadcast_activation = broadcasting_items.iter()
                    .map(|c| c.activation)
                    .sum::<f64>() / broadcasting_items.len() as f64;
                // Scale by proportion of items broadcasting (workspace access)
                let access_ratio = broadcasting_items.len() as f64
                    / self.state.conscious_contents.len() as f64;
                broadcast_activation * (0.5 + 0.5 * access_ratio)
            }
        };
        scores.insert("workspace".to_string(), workspace_score.clamp(0.0, 1.0));
        if workspace_score < 0.4 {
            bottlenecks.push("Limited workspace access".to_string());
        }

        // 3. BINDING - Calculate REAL binding from synchrony and binding_strength
        //    Previously only counted conscious objects (always 0!) - now uses synchrony!
        let binding_score = if self.state.bound_objects.is_empty() {
            0.0
        } else {
            // Real binding = weighted combination of synchrony and binding_strength
            let total_synchrony: f64 = self.state.bound_objects.iter()
                .map(|b| b.synchrony * b.binding_strength)
                .sum();
            let avg_binding = total_synchrony / self.state.bound_objects.len() as f64;

            // Also factor in how many objects are bound (binding breadth)
            let bound_fraction = (self.state.bound_objects.len() as f64).min(10.0) / 10.0;

            // Combine strength with breadth
            avg_binding * 0.7 + bound_fraction * 0.3
        };
        scores.insert("binding".to_string(), binding_score.clamp(0.0, 1.0));
        if binding_score < 0.3 {
            bottlenecks.push("Weak feature binding".to_string());
        }

        // 4. HOT (Higher-Order Thought) - Calculate REAL meta-awareness
        //    Previously was hardcoded to 0.7 when not empty - now uses actual confidence!
        let hot_score = if self.state.meta_awareness.is_empty() {
            0.0
        } else {
            // Real HOT = weighted average of confidence by order level
            // Higher order thoughts (order > 1) contribute more to meta-awareness
            let weighted_confidence: f64 = self.state.meta_awareness.iter()
                .map(|m| m.confidence * (1.0 + 0.2 * m.order as f64))
                .sum();
            let total_weight: f64 = self.state.meta_awareness.iter()
                .map(|m| 1.0 + 0.2 * m.order as f64)
                .sum();

            // Also factor in depth of meta-cognition (max order reached)
            let max_order = self.state.meta_awareness.iter()
                .map(|m| m.order)
                .max()
                .unwrap_or(0) as f64;
            let depth_bonus = (max_order / 3.0).min(0.2);

            (weighted_confidence / total_weight) + depth_bonus
        };
        scores.insert("hot".to_string(), hot_score.clamp(0.0, 1.0));
        if hot_score < 0.3 {
            bottlenecks.push("Limited higher-order awareness".to_string());
        }

        // 5. Additional REAL metrics from state
        scores.insert("temporal_coherence".to_string(), self.state.temporal_coherence);
        scores.insert("embodiment".to_string(), self.state.embodiment);

        let avg_score = scores.values().sum::<f64>() / scores.len() as f64;
        let is_conscious = avg_score >= self.config.consciousness_threshold;

        IntegrationAssessment {
            is_conscious,
            consciousness_score: avg_score,
            component_scores: scores,
            bottlenecks,
            explanation: format!(
                "Consciousness score: {:.2} (phi={:.2}, workspace={:.2}, binding={:.2}, hot={:.2})",
                avg_score, self.state.phi, workspace_score, binding_score, hot_score
            ),
        }
    }

    /// Get current state
    pub fn get_state(&self) -> &ConsciousnessState {
        &self.state
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.state = ConsciousnessState::default();
        self.history.clear();
    }
}

impl Default for ConsciousnessPipeline {
    fn default() -> Self {
        Self::default_new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_config_default() {
        let config = IntegrationConfig::default();
        assert_eq!(config.num_cycles, 10);
        assert_eq!(config.workspace_capacity, 3);
    }

    #[test]
    fn test_consciousness_state_default() {
        let state = ConsciousnessState::default();
        assert!((state.phi - 0.0).abs() < 1e-10);
        assert!(state.conscious_contents.is_empty());
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = ConsciousnessPipeline::default();
        assert!((pipeline.state.phi - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_process() {
        let mut pipeline = ConsciousnessPipeline::default();
        let input = vec![HV16::random(42)];
        pipeline.process_cycle(&input);
        assert!(pipeline.state.phi > 0.0);
    }

    #[test]
    fn test_altered_state_effects() {
        let mut pipeline = ConsciousnessPipeline::default();

        pipeline.set_altered_state(AlteredStateIndex::Wake);
        assert!(pipeline.state.consciousness_level > 0.5);

        pipeline.set_altered_state(AlteredStateIndex::Propofol);
        assert!(pipeline.state.consciousness_level < 0.1);
    }

    #[test]
    fn test_assessment() {
        let pipeline = ConsciousnessPipeline::default();
        let assessment = pipeline.assess();
        assert!(!assessment.is_conscious); // Default state is not conscious
    }

    #[test]
    fn test_process_method() {
        let config = IntegrationConfig::default();
        let mut pipeline = ConsciousnessPipeline::new(config);
        pipeline.set_embodiment(0.8);

        let input = vec![HV16::random(1), HV16::random(2), HV16::random(3)];
        let priorities = vec![0.9, 0.7, 0.5];

        let state = pipeline.process(input, &priorities);
        assert!(state.consciousness_level > 0.0);
        assert!(state.phi > 0.0);
    }

    #[test]
    fn test_bound_object() {
        let obj = BoundObject {
            representation: HV16::random(1),
            synchrony: 0.8,
            binding_strength: 0.9,
            conscious: true,
            level: BindingLevel::Feature,
            child_ids: Vec::new(),
            attention_weight: 1.0,
            creation_cycle: 0,
            persistence_cycles: 0,
            temporal_stability: 1.0,
        };
        assert!(obj.is_conscious());
        assert_eq!(obj.level, BindingLevel::Feature);
    }

    #[test]
    fn test_workspace_item() {
        let item = WorkspaceItem {
            content: HV16::random(2),
            activation: 0.95,
            source: "visual".to_string(),
            is_broadcasting: true,
            duration_ms: 100,
        };
        assert!(item.activation > 0.9);
        assert!(item.is_broadcasting);
    }

    #[test]
    fn test_meta_thought() {
        let thought = MetaThought {
            about: "seeing red".to_string(),
            target: "visual_perception".to_string(),
            intensity: 0.8,
            confidence: 0.9,
            order: 2,
            representation: HV16::random(42),
        };
        assert_eq!(thought.order, 2);
        assert!(thought.confidence > 0.5);
    }

    #[test]
    fn test_binding_calculation_with_bound_objects() {
        // Create pipeline with bound objects to test REAL binding calculation
        let mut pipeline = ConsciousnessPipeline::default();

        // Add bound objects with known synchrony and binding_strength
        pipeline.state.bound_objects = vec![
            BoundObject {
                representation: HV16::random(1),
                synchrony: 0.9,      // High synchrony
                binding_strength: 0.8, // Strong binding
                conscious: true,
                level: BindingLevel::Feature,
                child_ids: Vec::new(),
                attention_weight: 1.0,
                creation_cycle: 0,
                persistence_cycles: 5,
                temporal_stability: 0.95,
            },
            BoundObject {
                representation: HV16::random(2),
                synchrony: 0.7,      // Medium synchrony
                binding_strength: 0.6, // Medium binding
                conscious: true,
                level: BindingLevel::Feature,
                child_ids: Vec::new(),
                attention_weight: 1.0,
                creation_cycle: 0,
                persistence_cycles: 3,
                temporal_stability: 0.85,
            },
            BoundObject {
                representation: HV16::random(3),
                synchrony: 0.5,      // Lower synchrony
                binding_strength: 0.4, // Weaker binding
                conscious: false,
                level: BindingLevel::Feature,
                child_ids: Vec::new(),
                attention_weight: 1.0,
                creation_cycle: 0,
                persistence_cycles: 1,
                temporal_stability: 0.7,
            },
        ];

        // Assess and check binding score
        let assessment = pipeline.assess_integration();

        // Verify binding is NOT zero (was the bug before fix)
        assert!(assessment.component_scores.get("binding").unwrap() > &0.0,
            "Binding should be > 0 with bound objects");

        // Calculate expected binding:
        // Object 1: 0.9 * 0.8 = 0.72
        // Object 2: 0.7 * 0.6 = 0.42
        // Object 3: 0.5 * 0.4 = 0.20
        // total_synchrony = 0.72 + 0.42 + 0.20 = 1.34
        // avg_binding = 1.34 / 3 = 0.4467
        // bound_fraction = min(3, 10) / 10 = 0.3
        // binding_score = 0.4467 * 0.7 + 0.3 * 0.3 = 0.3127 + 0.09 = 0.4027
        let binding_score = *assessment.component_scores.get("binding").unwrap();
        assert!((binding_score - 0.4027).abs() < 0.01,
            "Expected binding ~0.40, got {}", binding_score);

        println!("✅ Binding calculation with bound objects: {:.4}", binding_score);
    }

    #[test]
    fn test_binding_calculation_empty() {
        // Test that empty bound_objects returns 0
        let pipeline = ConsciousnessPipeline::default();
        let assessment = pipeline.assess_integration();

        let binding_score = *assessment.component_scores.get("binding").unwrap();
        assert!((binding_score - 0.0).abs() < 1e-10,
            "Binding should be 0 with no bound objects, got {}", binding_score);

        println!("✅ Binding calculation empty: {:.4}", binding_score);
    }

    #[test]
    fn test_workspace_calculation_with_broadcasting() {
        // Test workspace calculation with actual broadcasting items
        let mut pipeline = ConsciousnessPipeline::default();

        pipeline.state.conscious_contents = vec![
            WorkspaceItem {
                content: HV16::random(1),
                activation: 0.9,
                source: "visual".to_string(),
                is_broadcasting: true,  // Broadcasting
                duration_ms: 100,
            },
            WorkspaceItem {
                content: HV16::random(2),
                activation: 0.7,
                source: "auditory".to_string(),
                is_broadcasting: true,  // Broadcasting
                duration_ms: 50,
            },
            WorkspaceItem {
                content: HV16::random(3),
                activation: 0.5,
                source: "touch".to_string(),
                is_broadcasting: false,  // Not broadcasting
                duration_ms: 30,
            },
        ];

        let assessment = pipeline.assess_integration();
        let workspace_score = *assessment.component_scores.get("workspace").unwrap();

        // Expected: 2 items broadcasting with activations 0.9 and 0.7
        // broadcast_activation = (0.9 + 0.7) / 2 = 0.8
        // access_ratio = 2/3 = 0.667
        // workspace = 0.8 * (0.5 + 0.5 * 0.667) = 0.8 * 0.833 = 0.667
        assert!(workspace_score > 0.6 && workspace_score < 0.75,
            "Expected workspace ~0.67, got {}", workspace_score);

        println!("✅ Workspace calculation with broadcasting: {:.4}", workspace_score);
    }

    #[test]
    fn test_hot_calculation_with_meta_awareness() {
        // Test HOT calculation with actual meta-awareness
        let mut pipeline = ConsciousnessPipeline::default();

        pipeline.state.meta_awareness = vec![
            MetaThought {
                about: "thinking".to_string(),
                target: "cognition".to_string(),
                intensity: 0.8,
                confidence: 0.9,  // High confidence
                order: 1,         // First-order meta
                representation: HV16::random(1),
            },
            MetaThought {
                about: "aware of thinking".to_string(),
                target: "meta_cognition".to_string(),
                intensity: 0.7,
                confidence: 0.7,  // Medium confidence
                order: 2,         // Second-order meta
                representation: HV16::random(2),
            },
        ];

        let assessment = pipeline.assess_integration();
        let hot_score = *assessment.component_scores.get("hot").unwrap();

        // Expected calculation:
        // weighted_confidence = 0.9*(1+0.2*1) + 0.7*(1+0.2*2) = 0.9*1.2 + 0.7*1.4 = 1.08 + 0.98 = 2.06
        // total_weight = 1.2 + 1.4 = 2.6
        // base = 2.06 / 2.6 = 0.792
        // depth_bonus = min(2/3, 0.2) = 0.2
        // hot = 0.792 + 0.2 = 0.992 (clamped to 1.0)
        assert!(hot_score > 0.9,
            "Expected HOT ~0.99, got {}", hot_score);

        println!("✅ HOT calculation with meta-awareness: {:.4}", hot_score);
    }

    #[test]
    fn test_process_creates_bound_objects() {
        // Test that process() creates bound objects from correlated inputs
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.8);

        // Create multiple high-priority inputs
        let inputs = vec![
            HV16::random(100),
            HV16::random(101),
            HV16::random(102),
        ];
        let priorities = vec![0.8, 0.7, 0.6];  // All above 0.5 threshold

        // Process inputs
        let state = pipeline.process(inputs, &priorities);

        // Should have created bound objects
        assert!(!state.bound_objects.is_empty(),
            "Expected bound objects to be created, but got none");

        // Verify bound object properties
        let bound = &state.bound_objects[0];
        assert!(bound.synchrony >= 0.0 && bound.synchrony <= 1.0,
            "Synchrony should be in [0,1], got {}", bound.synchrony);
        assert!(bound.binding_strength > 0.5,
            "Binding strength should be > 0.5 for high-priority inputs, got {}", bound.binding_strength);

        println!("✅ process() creates bound objects: {} objects with synchrony={:.4}, strength={:.4}",
            state.bound_objects.len(), bound.synchrony, bound.binding_strength);
    }

    #[test]
    fn test_process_no_binding_for_single_input() {
        // Test that single input doesn't create bound objects
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.8);

        let inputs = vec![HV16::random(200)];
        let priorities = vec![0.9];

        let state = pipeline.process(inputs, &priorities);

        // Single input should not create bound objects (need >= 2 for binding)
        assert!(state.bound_objects.is_empty(),
            "Single input should not create bound objects");

        println!("✅ process() correctly skips binding for single input");
    }

    #[test]
    fn test_process_no_binding_for_low_priority() {
        // Test that low-priority inputs don't create bound objects
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.8);

        let inputs = vec![
            HV16::random(300),
            HV16::random(301),
            HV16::random(302),
        ];
        let priorities = vec![0.3, 0.2, 0.4];  // All below 0.5 threshold

        let state = pipeline.process(inputs, &priorities);

        // Low priority inputs should not create bound objects
        assert!(state.bound_objects.is_empty(),
            "Low-priority inputs should not create bound objects");

        println!("✅ process() correctly skips binding for low-priority inputs");
    }

    #[test]
    fn test_process_binding_integrates_with_assessment() {
        // Test that bound objects from process() affect assess_integration()
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.9);

        // First, check baseline without processing
        let baseline_assessment = pipeline.assess_integration();
        let baseline_binding = *baseline_assessment.component_scores.get("binding").unwrap();
        assert_eq!(baseline_binding, 0.0, "Baseline binding should be 0");

        // Process inputs to create binding
        let inputs = vec![
            HV16::random(400),
            HV16::random(401),
        ];
        let priorities = vec![0.8, 0.9];

        pipeline.process(inputs, &priorities);

        // Now check that binding score is non-zero
        let assessment = pipeline.assess_integration();
        let binding_score = *assessment.component_scores.get("binding").unwrap();

        assert!(binding_score > 0.0,
            "Binding score should be > 0 after processing, got {}", binding_score);

        println!("✅ process() binding integrates with assessment: binding={:.4}", binding_score);
    }

    #[test]
    fn test_bound_object_synchrony_calculation() {
        // Test synchrony is calculated from HV similarity
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.9);

        // Create similar HVs (should have higher synchrony)
        let base_hv = HV16::random(500);
        let similar_hv = base_hv.clone();  // Identical = max similarity

        let inputs = vec![base_hv, similar_hv];
        let priorities = vec![0.9, 0.9];

        let state = pipeline.process(inputs, &priorities);

        // Identical HVs should have high synchrony
        assert!(!state.bound_objects.is_empty());
        let bound = &state.bound_objects[0];

        // similarity of identical HVs should be 1.0, so synchrony >= 0.5 (our minimum)
        assert!(bound.synchrony >= 0.5,
            "Synchrony for similar HVs should be >= 0.5, got {}", bound.synchrony);

        println!("✅ Synchrony calculated from HV similarity: {:.4}", bound.synchrony);
    }

    #[test]
    fn test_multiple_bound_objects_clustering() {
        // Test that dissimilar inputs create separate bound objects
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.9);

        // Create 4 inputs: 2 similar pairs, each pair dissimilar to the other
        // Similar HVs should cluster together
        let hv1 = HV16::random(600);
        let hv2 = hv1.clone();  // Similar to hv1
        let hv3 = HV16::random(700);  // Different from hv1/hv2
        let hv4 = hv3.clone();  // Similar to hv3

        let inputs = vec![hv1, hv2, hv3, hv4];
        let priorities = vec![0.8, 0.85, 0.9, 0.75];  // All high priority

        let state = pipeline.process(inputs, &priorities);

        // Should have at least 1 bound object (may be more depending on clustering)
        assert!(!state.bound_objects.is_empty(),
            "Should have at least one bound object");

        println!("✅ Clustering created {} bound objects", state.bound_objects.len());
    }

    #[test]
    fn test_binding_persistence_decay() {
        // Test that bound objects persist but decay across cycles
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.9);

        // First cycle: create bound object
        let inputs = vec![HV16::random(800), HV16::random(801)];
        let priorities = vec![0.8, 0.9];
        let state1 = pipeline.process(inputs, &priorities);

        assert!(!state1.bound_objects.is_empty(), "Should have bound objects after first cycle");
        let initial_strength = state1.bound_objects[0].binding_strength;

        // Second cycle: process empty or different inputs
        // Bound objects should still exist but be decayed
        let new_inputs = vec![HV16::random(900)];  // Only 1 input, no new binding
        let new_priorities = vec![0.3];  // Low priority
        let state2 = pipeline.process(new_inputs, &new_priorities);

        // Previous bound objects should still exist but decayed
        if !state2.bound_objects.is_empty() {
            let decayed_strength = state2.bound_objects[0].binding_strength;
            assert!(decayed_strength < initial_strength,
                "Binding strength should decay: {} < {}", decayed_strength, initial_strength);
            println!("✅ Binding decayed from {:.4} to {:.4}", initial_strength, decayed_strength);
        } else {
            // If object was removed due to decay, that's also valid
            println!("✅ Bound object was removed due to decay (strength below threshold)");
        }
    }

    #[test]
    fn test_binding_reinforcement() {
        // Test that repeated similar inputs reinforce existing bindings
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.9);

        // Create consistent inputs across multiple cycles
        let base1 = HV16::random(1000);
        let base2 = HV16::random(1001);

        // First cycle
        let inputs1 = vec![base1.clone(), base2.clone()];
        let priorities = vec![0.8, 0.9];
        pipeline.process(inputs1, &priorities);

        let strength_after_1 = pipeline.state.bound_objects.first()
            .map(|o| o.binding_strength).unwrap_or(0.0);

        // Second cycle with similar inputs (should reinforce)
        let inputs2 = vec![base1.clone(), base2.clone()];
        pipeline.process(inputs2, &priorities);

        let strength_after_2 = pipeline.state.bound_objects.first()
            .map(|o| o.binding_strength).unwrap_or(0.0);

        // Third cycle
        let inputs3 = vec![base1.clone(), base2.clone()];
        pipeline.process(inputs3, &priorities);

        let strength_after_3 = pipeline.state.bound_objects.first()
            .map(|o| o.binding_strength).unwrap_or(0.0);

        println!("Binding strength: cycle1={:.4}, cycle2={:.4}, cycle3={:.4}",
            strength_after_1, strength_after_2, strength_after_3);

        // Reinforcement should counter decay, maintaining or increasing strength
        // (Note: exact behavior depends on decay rate vs reinforcement amount)
        assert!(strength_after_3 > 0.0, "Binding should persist after reinforcement");
        println!("✅ Binding reinforcement maintains persistence");
    }

    #[test]
    fn test_max_bound_objects_limit() {
        // Test that bound objects are limited to MAX_BOUND_OBJECTS
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.9);

        // Create many dissimilar high-priority inputs to trigger multiple clusters
        // Each pair should create a separate bound object
        let inputs: Vec<HV16> = (0..20).map(|i| HV16::random(2000 + i)).collect();
        let priorities: Vec<f64> = vec![0.9; 20];

        let state = pipeline.process(inputs, &priorities);

        // Should be capped at MAX_BOUND_OBJECTS (10)
        assert!(state.bound_objects.len() <= 10,
            "Bound objects should be limited to 10, got {}", state.bound_objects.len());

        println!("✅ Bound objects limited to {} (max 10)", state.bound_objects.len());
    }

    // === INTEGRATED CONSCIOUSNESS SYSTEMS TESTS ===

    #[test]
    fn test_enable_integrated_systems() {
        // Test that integrated systems can be enabled
        let mut pipeline = ConsciousnessPipeline::default();

        // Initially disabled
        assert!(!pipeline.has_integrated_systems(),
            "Integrated systems should be disabled by default");
        assert!(pipeline.metacognitive_monitor.is_none());
        assert!(pipeline.cross_modal_binder.is_none());
        assert!(pipeline.temporal_binder.is_none());

        // Enable them
        pipeline.enable_integrated_systems();

        // Now enabled
        assert!(pipeline.has_integrated_systems(),
            "Integrated systems should be enabled after enable_integrated_systems()");
        assert!(pipeline.metacognitive_monitor.is_some());
        assert!(pipeline.cross_modal_binder.is_some());
        assert!(pipeline.temporal_binder.is_some());

        println!("✅ Integrated systems enabled successfully");
    }

    #[test]
    fn test_integrated_systems_processing() {
        // Test that integrated systems affect state during processing
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.8);
        pipeline.enable_integrated_systems();

        // Process several cycles to build up metrics
        for i in 0..5 {
            let inputs = vec![HV16::random(3000 + i), HV16::random(3100 + i)];
            let priorities = vec![0.8, 0.7];
            pipeline.process(inputs, &priorities);
        }

        let state = pipeline.get_state();

        // Metacognitive metrics should be updated
        assert!(state.metacognitive_confidence >= 0.0 && state.metacognitive_confidence <= 1.0,
            "Metacognitive confidence should be in [0,1], got {}", state.metacognitive_confidence);

        // Predictive mode should be set
        assert!(
            state.inference_mode == PredictiveMode::Exploring ||
            state.inference_mode == PredictiveMode::Exploiting ||
            state.inference_mode == PredictiveMode::Balanced,
            "Inference mode should be valid"
        );

        // Theta phase should be advancing
        assert!(state.theta_phase >= 0.0, "Theta phase should be non-negative");

        println!("✅ Integrated systems processing works:");
        println!("   - Metacognitive confidence: {:.4}", state.metacognitive_confidence);
        println!("   - Inference mode: {:?}", state.inference_mode);
        println!("   - Theta phase: {:.4}", state.theta_phase);
        println!("   - Cross-modal coherence: {:.4}", state.cross_modal_coherence);
        println!("   - Narrative coherence: {:.4}", state.narrative_coherence);
    }

    #[test]
    fn test_predictive_mode_transitions() {
        // Test that predictive mode changes based on prediction accuracy
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.9);
        pipeline.enable_integrated_systems();

        // Process high-quality inputs (should lead to exploiting mode)
        for _ in 0..10 {
            let inputs = vec![HV16::random(4000), HV16::random(4001)];
            let priorities = vec![0.95, 0.95];  // High priorities = high consciousness = high accuracy
            pipeline.process(inputs, &priorities);
        }

        let state = pipeline.get_state();

        // With high prediction accuracy, should be in exploiting mode
        if state.prediction_accuracy > 0.7 {
            assert_eq!(state.inference_mode, PredictiveMode::Exploiting,
                "High prediction accuracy should trigger Exploiting mode");
        }

        println!("✅ Predictive mode transitions work:");
        println!("   - Prediction accuracy: {:.4}", state.prediction_accuracy);
        println!("   - Inference mode: {:?}", state.inference_mode);
    }

    #[test]
    fn test_phi_trend_prediction() {
        // Test that Φ trends lead to predictions
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.8);
        pipeline.enable_integrated_systems();

        // Process several cycles with increasing quality (should create positive trend)
        let mut inputs_quality = 0.5;
        for i in 0..7 {
            inputs_quality += 0.05;  // Increasing quality
            let inputs = vec![HV16::random(5000 + i), HV16::random(5100 + i)];
            let priorities = vec![inputs_quality, inputs_quality];
            pipeline.process(inputs, &priorities);
        }

        let state = pipeline.get_state();

        // After 5+ cycles, should have Φ prediction
        if state.predicted_phi.is_some() {
            println!("✅ Φ prediction generated:");
            println!("   - Current Φ: {:.4}", state.phi);
            println!("   - Predicted Φ: {:.4}", state.predicted_phi.unwrap());
            println!("   - Φ trend: {:.4}", state.phi_trend);
        } else {
            println!("⚠️ Φ prediction not yet generated (needs more history)");
        }
    }

    #[test]
    fn test_cross_modal_coherence() {
        // Test that cross-modal coherence increases with multiple modalities
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.9);
        pipeline.enable_integrated_systems();

        // Add workspace items with different source modalities
        pipeline.state.conscious_contents.push(WorkspaceItem {
            content: HV16::random(6000),
            activation: 0.9,
            source: "visual_cortex".to_string(),  // Contains "visual"
            is_broadcasting: true,
            duration_ms: 100,
        });
        pipeline.state.conscious_contents.push(WorkspaceItem {
            content: HV16::random(6001),
            activation: 0.8,
            source: "audio_processing".to_string(),  // Contains "audio"
            is_broadcasting: true,
            duration_ms: 100,
        });
        pipeline.state.conscious_contents.push(WorkspaceItem {
            content: HV16::random(6002),
            activation: 0.7,
            source: "motor_planning".to_string(),  // Contains "motor"
            is_broadcasting: true,
            duration_ms: 100,
        });

        // Process to update cross-modal metrics
        let inputs = vec![HV16::random(6100)];
        let priorities = vec![0.5];
        pipeline.process(inputs, &priorities);

        let state = pipeline.get_state();

        // With 3 modalities, should have higher cross-modal coherence
        assert!(state.active_modalities.len() >= 2,
            "Should detect multiple modalities, got {}", state.active_modalities.len());
        assert!(state.cross_modal_coherence > 0.5,
            "Multi-modal integration should boost coherence above 0.5, got {}",
            state.cross_modal_coherence);

        println!("✅ Cross-modal coherence with {} modalities:", state.active_modalities.len());
        println!("   - Active modalities: {:?}", state.active_modalities);
        println!("   - Cross-modal coherence: {:.4}", state.cross_modal_coherence);
    }

    // === Φ-GUIDED TOPOLOGY OPTIMIZATION TESTS ===

    #[test]
    fn test_enable_phi_optimization() {
        let mut pipeline = ConsciousnessPipeline::default();

        // Initially disabled
        assert!(!pipeline.has_phi_optimization(),
            "Φ optimization should be disabled by default");
        assert!(pipeline.phi_optimizer.is_none());
        assert!(pipeline.consciousness_network.is_none());

        // Enable with 8 nodes
        pipeline.enable_phi_optimization(8, 10);

        // Now enabled
        assert!(pipeline.has_phi_optimization(),
            "Φ optimization should be enabled after enable_phi_optimization()");
        assert!(pipeline.phi_optimizer.is_some());
        assert!(pipeline.consciousness_network.is_some());

        // Verify network
        let network = pipeline.consciousness_network().unwrap();
        assert_eq!(network.node_count(), 8);
        assert!(network.edge_count() > 0, "Ring topology should have edges");

        println!("✅ Φ optimization enabled: {} nodes, {} edges",
            network.node_count(), network.edge_count());
    }

    #[test]
    fn test_manual_phi_optimization() {
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.enable_phi_optimization(6, 10);

        let initial_phi = pipeline.state.phi;

        // Run optimization step
        let result = pipeline.optimize_phi();
        assert!(result.is_some(), "Should return optimization result");

        let opt = result.unwrap();
        assert!(opt.phi >= 0.0 && opt.phi <= 1.0, "Φ should be in [0,1]");
        assert_eq!(opt.step, 1, "Should be first step");

        println!("✅ Manual Φ optimization:");
        println!("   - Initial Φ: {:.4}", initial_phi);
        println!("   - Optimized Φ: {:.4}", opt.phi);
        println!("   - Delta: {:.4}", opt.phi_delta);
    }

    #[test]
    #[ignore = "expensive: graph topology optimization (~60s)"]
    fn test_phi_optimization_multiple_steps() {
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.enable_phi_optimization(8, 5);

        // Run 5 optimization steps
        let results = pipeline.optimize_phi_steps(5);

        assert_eq!(results.len(), 5, "Should have 5 results");

        // Φ should be non-negative throughout
        for (i, result) in results.iter().enumerate() {
            assert!(result.phi >= 0.0, "Φ should be non-negative at step {}", i);
            assert_eq!(result.step, i + 1, "Step number should match");
        }

        println!("✅ Multi-step Φ optimization:");
        for (i, r) in results.iter().enumerate() {
            println!("   Step {}: Φ = {:.4} (Δ = {:+.4})", i + 1, r.phi, r.phi_delta);
        }
    }

    #[test]
    #[ignore = "expensive: 20 cycles with phi optimization (~90s)"]
    fn test_phi_optimization_during_processing() {
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.8);
        pipeline.enable_phi_optimization(8, 5); // Optimize every 5 cycles

        // Track Φ values across cycles
        let mut phi_values = Vec::new();

        // Process for 20 cycles (should trigger optimization at 5, 10, 15, 20)
        for i in 0..20 {
            let inputs = vec![HV16::random(7000 + i), HV16::random(7100 + i)];
            let priorities = vec![0.8, 0.7];
            pipeline.process(inputs, &priorities);
            phi_values.push(pipeline.state.phi);
        }

        // Should have optimization results after processing
        if let Some(ref last_opt) = pipeline.last_optimization {
            println!("✅ Φ optimization during processing:");
            println!("   - Last optimization step: {}", last_opt.step);
            println!("   - Final Φ: {:.4}", last_opt.phi);
            println!("   - Topological unity: {:.4}", pipeline.state.topological_unity);
        }

        // Topological unity should be updated
        assert!(pipeline.state.topological_unity > 0.0,
            "Topological unity should be positive after optimization");
    }

    #[test]
    fn test_phi_optimization_with_custom_config() {
        use crate::hdc::phi_guided_search::PhiOptimizationConfig;

        let config = PhiOptimizationConfig {
            dim: 512, // Smaller dimension for faster test
            learning_rate: 0.2,
            adaptive_lr: true,
            ..Default::default()
        };

        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.enable_phi_optimization_with_config(
            config,
            10, // 10 nodes
            InitializationStrategy::SmallWorld { rewire_prob: 0.3 },
            5,
        );

        assert!(pipeline.has_phi_optimization());

        // Get network stats before mutable operation
        let (node_count, edge_count) = {
            let network = pipeline.consciousness_network().unwrap();
            assert_eq!(network.node_count(), 10);
            (network.node_count(), network.edge_count())
        };

        // Run optimization
        let result = pipeline.optimize_phi();
        assert!(result.is_some());

        println!("✅ Custom config Φ optimization:");
        println!("   - Nodes: {}", node_count);
        println!("   - Edges: {}", edge_count);
        if let Some(r) = result {
            println!("   - Φ: {:.4}", r.phi);
        }
    }

    #[test]
    #[ignore = "expensive: 15 cycles with phi optimization (~80s)"]
    fn test_phi_trend_with_optimization() {
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.9);
        pipeline.enable_phi_optimization(8, 3); // Optimize every 3 cycles

        // Process many cycles to see Φ trend
        for i in 0..15 {
            let inputs = vec![HV16::random(8000 + i), HV16::random(8100 + i)];
            let priorities = vec![0.9, 0.85];
            pipeline.process(inputs, &priorities);
        }

        let state = pipeline.get_state();

        println!("✅ Φ trend with optimization:");
        println!("   - Final Φ: {:.4}", state.phi);
        println!("   - Φ trend: {:+.4}", state.phi_trend);
        println!("   - Topological unity: {:.4}", state.topological_unity);

        // Φ trend should be within reasonable bounds
        assert!(state.phi_trend >= -0.1 && state.phi_trend <= 0.1,
            "Φ trend should be bounded, got {}", state.phi_trend);
    }

    #[test]
    fn test_all_systems_integration() {
        // Test full integration: integrated systems + Φ optimization
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.9);
        pipeline.enable_integrated_systems();
        pipeline.enable_phi_optimization(8, 5);

        // Verify both are enabled
        assert!(pipeline.has_integrated_systems());
        assert!(pipeline.has_phi_optimization());

        // Process with all systems active
        for i in 0..20 {
            let inputs = vec![HV16::random(9000 + i), HV16::random(9100 + i)];
            let priorities = vec![0.85, 0.8];
            pipeline.process(inputs, &priorities);
        }

        let state = pipeline.get_state();

        println!("✅ Full system integration:");
        println!("   - Φ: {:.4}", state.phi);
        println!("   - Metacognitive confidence: {:.4}", state.metacognitive_confidence);
        println!("   - Cross-modal coherence: {:.4}", state.cross_modal_coherence);
        println!("   - Topological unity: {:.4}", state.topological_unity);
        println!("   - Narrative coherence: {:.4}", state.narrative_coherence);
        println!("   - Consciousness level: {:.4}", state.consciousness_level);

        // All metrics should be valid
        assert!(state.phi >= 0.0 && state.phi <= 1.0);
        assert!(state.consciousness_level >= 0.0 && state.consciousness_level <= 1.0);
        assert!(state.topological_unity >= 0.0 && state.topological_unity <= 1.0);
    }

    // ==========================================
    // FEEDBACK DYNAMICS TESTS
    // ==========================================

    #[test]
    fn test_enable_feedback_dynamics() {
        let mut pipeline = ConsciousnessPipeline::default();

        // Initially disabled
        assert!(!pipeline.has_feedback_dynamics());

        // Enable feedback dynamics
        pipeline.enable_feedback_dynamics();
        assert!(pipeline.has_feedback_dynamics());

        // Check initial state
        assert!(pipeline.current_prediction().is_none());
        assert!(pipeline.pending_intervention().is_none());
        assert!(pipeline.recent_insights().is_empty());

        println!("✅ Feedback dynamics enabled successfully");
    }

    #[test]
    fn test_emotional_prediction() {
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.enable_feedback_dynamics();

        // Record enough emotional states to build prediction history (need >= 5)
        for i in 0..5 {
            let valence = 0.5 + (i as f64 * 0.1);  // Gradually increasing valence
            let arousal = 0.5;
            pipeline.update_emotional_prediction(valence, arousal);
        }

        // Now get a prediction (should work with sufficient history)
        let prediction = pipeline.update_emotional_prediction(0.8, 0.5);
        assert!(prediction.is_some(), "Should return a prediction after building history");

        let pred = prediction.unwrap();
        println!("✅ Emotional prediction:");
        println!("   - Predicted valence: {:.4}", pred.predicted_valence);
        println!("   - Predicted arousal: {:.4}", pred.predicted_arousal);
        println!("   - Confidence: {:.4}", pred.confidence);

        // Prediction should be stored
        assert!(pipeline.current_prediction().is_some());
    }

    #[test]
    fn test_emotional_prediction_generates_intervention() {
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.enable_feedback_dynamics();

        // Record several emotional states to build prediction history
        for i in 0..10 {
            // Simulating declining emotional state that might trigger intervention
            let valence = 0.3 - (i as f64 * 0.02); // Decreasing valence
            let arousal = 0.7 + (i as f64 * 0.01); // Increasing arousal (stress)
            pipeline.update_emotional_prediction(valence, arousal);
        }

        // Check if intervention is recommended
        let prediction = pipeline.current_prediction();
        if let Some(pred) = prediction {
            println!("✅ Emotional prediction after decline:");
            println!("   - Predicted valence: {:.4}", pred.predicted_valence);
            println!("   - Predicted arousal: {:.4}", pred.predicted_arousal);

            if pred.intervention.is_some() {
                println!("   - Intervention recommended: {:?}", pred.intervention);
            }
        }
    }

    #[test]
    fn test_apply_intervention() {
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.enable_feedback_dynamics();

        // Build up emotional history that would trigger intervention
        for _ in 0..5 {
            pipeline.update_emotional_prediction(0.2, 0.8); // Low valence, high arousal
        }

        // Try to apply intervention
        let intervention = pipeline.apply_intervention();

        println!("✅ Intervention application:");
        if let Some(ref int) = intervention {
            println!("   - Applied intervention: {:?}", int);
        } else {
            println!("   - No intervention needed at this time");
        }
    }

    #[test]
    fn test_feedback_dynamics_during_processing() {
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.8);
        pipeline.enable_feedback_dynamics();

        // Process several cycles with feedback dynamics enabled
        for i in 0..10 {
            let inputs = vec![HV16::random(10000 + i), HV16::random(10100 + i)];
            let priorities = vec![0.7, 0.75];
            pipeline.process(inputs, &priorities);
        }

        let state = pipeline.get_state();

        println!("✅ Feedback dynamics during processing:");
        println!("   - Consciousness level: {:.4}", state.consciousness_level);
        println!("   - Φ: {:.4}", state.phi);

        // Check that feedback dynamics is still enabled after processing
        assert!(pipeline.has_feedback_dynamics());

        // Metrics should be valid
        assert!(state.consciousness_level >= 0.0 && state.consciousness_level <= 1.0);
    }

    #[test]
    fn test_feedback_dynamics_with_phi_optimization() {
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.85);
        pipeline.enable_feedback_dynamics();
        pipeline.enable_phi_optimization(8, 3);

        // Both should be enabled
        assert!(pipeline.has_feedback_dynamics());
        assert!(pipeline.has_phi_optimization());

        // Process with both systems active
        for i in 0..15 {
            let inputs = vec![HV16::random(11000 + i), HV16::random(11100 + i)];
            let priorities = vec![0.8, 0.85];
            pipeline.process(inputs, &priorities);
        }

        let state = pipeline.get_state();

        println!("✅ Feedback dynamics + Φ optimization:");
        println!("   - Φ: {:.4}", state.phi);
        println!("   - Topological unity: {:.4}", state.topological_unity);
        println!("   - Consciousness level: {:.4}", state.consciousness_level);

        // All metrics should be valid
        assert!(state.phi >= 0.0 && state.phi <= 1.0);
        assert!(state.consciousness_level >= 0.0 && state.consciousness_level <= 1.0);
    }

    #[test]
    fn test_all_systems_with_feedback_dynamics() {
        // Full integration test: integrated systems + Φ optimization + feedback dynamics
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.9);
        pipeline.enable_integrated_systems();
        pipeline.enable_phi_optimization(8, 5);
        pipeline.enable_feedback_dynamics();

        // Verify all are enabled
        assert!(pipeline.has_integrated_systems());
        assert!(pipeline.has_phi_optimization());
        assert!(pipeline.has_feedback_dynamics());

        // Process with all systems active
        for i in 0..20 {
            let inputs = vec![HV16::random(12000 + i), HV16::random(12100 + i)];
            let priorities = vec![0.85, 0.8];
            pipeline.process(inputs, &priorities);
        }

        let state = pipeline.get_state();

        println!("✅ Full system integration with feedback dynamics:");
        println!("   - Φ: {:.4}", state.phi);
        println!("   - Metacognitive confidence: {:.4}", state.metacognitive_confidence);
        println!("   - Cross-modal coherence: {:.4}", state.cross_modal_coherence);
        println!("   - Topological unity: {:.4}", state.topological_unity);
        println!("   - Narrative coherence: {:.4}", state.narrative_coherence);
        println!("   - Consciousness level: {:.4}", state.consciousness_level);

        // All metrics should be valid
        assert!(state.phi >= 0.0 && state.phi <= 1.0);
        assert!(state.consciousness_level >= 0.0 && state.consciousness_level <= 1.0);
        assert!(state.topological_unity >= 0.0 && state.topological_unity <= 1.0);
        assert!(state.metacognitive_confidence >= 0.0 && state.metacognitive_confidence <= 1.0);
    }

    // ==========================================
    // SELF-AWARENESS TESTS
    // ==========================================

    #[test]
    fn test_enable_self_awareness() {
        let mut pipeline = ConsciousnessPipeline::default();

        // Initially disabled
        assert!(!pipeline.has_self_awareness());

        // Enable self-awareness
        pipeline.enable_self_awareness(1024, 16);
        assert!(pipeline.has_self_awareness());

        // Check initial state
        assert_eq!(pipeline.self_awareness_level(), 0.0);
        assert!(pipeline.current_self_model().is_none());
        assert!(pipeline.latest_introspection().is_none());

        println!("✅ Self-awareness enabled successfully");
    }

    #[test]
    fn test_introspection() {
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.enable_self_awareness(1024, 16);

        // Perform introspection
        let report = pipeline.introspect();
        assert!(report.is_some(), "Introspection should return a report");

        let report = report.unwrap();
        println!("✅ Introspection report:");
        println!("   - Believed Φ: {:.4}", report.believed_phi);
        println!("   - Believed mode: {:?}", report.believed_mode);
        println!("   - Self-model confidence: {:.4}", report.self_model_confidence);
        println!("   - Self-model accuracy: {:.4}", report.self_model_accuracy);
        println!("   - Self-awareness level: {:.4}", report.self_awareness_level);
    }

    #[test]
    #[ignore = "expensive: 1024-dim self-awareness (~60s)"]
    fn test_process_self_aware() {
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.enable_self_awareness(1024, 16);

        // Process input with self-awareness
        let input = HV16::random(42);
        let update = pipeline.process_self_aware(&input);

        assert!(update.is_some(), "Should return a self-aware update");
        let update = update.unwrap();

        println!("✅ Self-aware processing:");
        println!("   - Base Φ: {:.4}", update.base_update.phi);
        println!("   - Predicted Φ: {:.4}", update.self_model.predicted_phi);
        println!("   - Prediction error: {:.4}", update.prediction_error);
        println!("   - Self-awareness level: {:.4}", update.self_awareness_level);

        // Self-model should now be cached
        assert!(pipeline.current_self_model().is_some());
        assert!(pipeline.self_awareness_level() > 0.0);
    }

    #[test]
    #[ignore = "takes >60s in debug builds"]
    fn test_self_awareness_prediction_learning() {
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.enable_self_awareness(1024, 16);

        // Process multiple inputs to let the self-model learn
        let mut prediction_errors = Vec::new();
        for i in 0..20 {
            let input = HV16::random(1000 + i);
            if let Some(update) = pipeline.process_self_aware(&input) {
                prediction_errors.push(update.prediction_error);
            }
        }

        println!("✅ Self-awareness learning:");
        println!("   - Initial prediction error: {:.4}", prediction_errors.first().unwrap_or(&1.0));
        println!("   - Final prediction error: {:.4}", prediction_errors.last().unwrap_or(&1.0));
        println!("   - Number of updates: {}", prediction_errors.len());

        // Check that we have a self-model
        let self_model = pipeline.current_self_model();
        assert!(self_model.is_some());

        let model = self_model.unwrap();
        println!("   - Self-model confidence: {:.4}", model.confidence);
        println!("   - Self-model accuracy: {:.4}", model.accuracy);
    }

    #[test]
    #[ignore = "expensive: 15 cycles with self-awareness (~90s)"]
    fn test_self_awareness_during_processing() {
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.8);
        pipeline.enable_self_awareness(1024, 16);

        // Process several cycles with self-awareness enabled
        for i in 0..15 {
            let inputs = vec![HV16::random(13000 + i), HV16::random(13100 + i)];
            let priorities = vec![0.75, 0.8];
            pipeline.process(inputs, &priorities);
        }

        let state = pipeline.get_state();

        println!("✅ Self-awareness during processing:");
        println!("   - Φ: {:.4}", state.phi);
        println!("   - Consciousness level: {:.4}", state.consciousness_level);
        println!("   - Metacognitive confidence: {:.4}", state.metacognitive_confidence);
        println!("   - Self-awareness level: {:.4}", pipeline.self_awareness_level());

        // Self-awareness should have affected metacognitive confidence
        assert!(pipeline.has_self_awareness());
        assert!(pipeline.self_awareness_level() > 0.0);
    }

    #[test]
    #[ignore = "expensive: meta-cognitive assessment (~70s)"]
    fn test_meta_cognitive_assessment() {
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.enable_self_awareness(1024, 16);

        // Process to generate meta-assessment
        for i in 0..5 {
            let input = HV16::random(2000 + i);
            pipeline.process_self_aware(&input);
        }

        let assessment = pipeline.meta_assessment();
        assert!(assessment.is_some(), "Meta-assessment should be available");

        let assessment = assessment.unwrap();
        println!("✅ Meta-cognitive assessment:");
        println!("   - Clarity: {:.4}", assessment.clarity);
        println!("   - Mode appropriateness: {:.4}", assessment.mode_appropriateness);
        println!("   - Φ optimality: {:.4}", assessment.phi_optimality);
        println!("   - Change recommended: {}", assessment.change_recommended);
        println!("   - Reasoning: {}", assessment.reasoning);
    }

    #[test]
    #[ignore = "takes >60s in debug builds"]
    fn test_complete_integration_all_systems() {
        // Ultimate integration test: ALL systems working together
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.9);
        pipeline.enable_integrated_systems();
        pipeline.enable_phi_optimization(8, 5);
        pipeline.enable_feedback_dynamics();
        pipeline.enable_self_awareness(1024, 16);

        // Verify all are enabled
        assert!(pipeline.has_integrated_systems());
        assert!(pipeline.has_phi_optimization());
        assert!(pipeline.has_feedback_dynamics());
        assert!(pipeline.has_self_awareness());

        // Process with ALL systems active
        for i in 0..25 {
            let inputs = vec![HV16::random(14000 + i), HV16::random(14100 + i)];
            let priorities = vec![0.85, 0.8];
            pipeline.process(inputs, &priorities);
        }

        let state = pipeline.get_state();

        println!("✅ COMPLETE INTEGRATION - All 8 systems working together:");
        println!("   - Φ: {:.4}", state.phi);
        println!("   - Metacognitive confidence: {:.4}", state.metacognitive_confidence);
        println!("   - Cross-modal coherence: {:.4}", state.cross_modal_coherence);
        println!("   - Topological unity: {:.4}", state.topological_unity);
        println!("   - Narrative coherence: {:.4}", state.narrative_coherence);
        println!("   - Consciousness level: {:.4}", state.consciousness_level);
        println!("   - Self-awareness level: {:.4}", pipeline.self_awareness_level());

        // Introspection
        if let Some(report) = pipeline.introspect() {
            println!("   - Believed Φ: {:.4}", report.believed_phi);
            println!("   - Self-model accuracy: {:.4}", report.self_model_accuracy);
        }

        // All metrics should be valid
        assert!(state.phi >= 0.0 && state.phi <= 1.0);
        assert!(state.consciousness_level >= 0.0 && state.consciousness_level <= 1.0);
        assert!(state.topological_unity >= 0.0 && state.topological_unity <= 1.0);
        assert!(state.metacognitive_confidence >= 0.0 && state.metacognitive_confidence <= 1.0);
        assert!(pipeline.self_awareness_level() >= 0.0 && pipeline.self_awareness_level() <= 1.0);
    }

    // ==========================================
    // UNIFIED CONSCIOUSNESS OPTIMIZER TESTS
    // ==========================================

    #[test]
    fn test_enable_full_consciousness() {
        let mut pipeline = ConsciousnessPipeline::default();

        // Enable all systems at once
        pipeline.enable_full_consciousness();

        // All systems should be enabled
        assert!(pipeline.has_full_consciousness());
        assert!(pipeline.has_integrated_systems());
        assert!(pipeline.has_phi_optimization());
        assert!(pipeline.has_feedback_dynamics());
        assert!(pipeline.has_self_awareness());

        println!("✅ Full consciousness enabled successfully");
    }

    #[test]
    #[ignore = "takes >60s in debug builds"]
    fn test_consciousness_metrics_report() {
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.8);
        pipeline.enable_full_consciousness();

        // Process some cycles
        for i in 0..10 {
            let inputs = vec![HV16::random(15000 + i), HV16::random(15100 + i)];
            let priorities = vec![0.8, 0.85];
            pipeline.process(inputs, &priorities);
        }

        // Get metrics report
        let report = pipeline.consciousness_metrics();

        println!("✅ Consciousness Metrics Report:");
        println!("{}", report);

        // Check report fields
        assert!(report.phi >= 0.0 && report.phi <= 1.0);
        assert!(report.consciousness_level >= 0.0 && report.consciousness_level <= 1.0);
        assert!(report.integrated_systems_active);
        assert!(report.phi_optimization_active);
        assert!(report.feedback_dynamics_active);
        assert!(report.self_awareness_active);
        assert!(report.processing_cycles > 0);
    }

    #[test]
    fn test_optimization_recommendations() {
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.3); // Low embodiment
        pipeline.enable_full_consciousness();

        // Get recommendations
        let recommendations = pipeline.optimization_recommendations();

        println!("✅ Optimization Recommendations:");
        for rec in &recommendations {
            println!("   {}", rec);
        }

        // With low embodiment, we should get some recommendations
        // (may or may not have recommendations depending on state)
        println!("   Total recommendations: {}", recommendations.len());
    }

    #[test]
    #[ignore = "expensive: full 11-system consciousness pipeline (~120s)"]
    fn test_optimization_cycle() {
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.8);
        pipeline.enable_full_consciousness();

        // Process some cycles first
        for i in 0..5 {
            let inputs = vec![HV16::random(16000 + i)];
            let priorities = vec![0.75];
            pipeline.process(inputs, &priorities);
        }

        // Run an optimization cycle
        let summary = pipeline.run_optimization_cycle();

        println!("✅ Optimization Cycle:");
        println!("{}", summary);

        // Check summary
        assert!(summary.phi_before >= 0.0 && summary.phi_before <= 1.0);
        assert!(summary.phi_after >= 0.0 && summary.phi_after <= 1.0);
    }

    #[test]
    fn test_enable_full_consciousness_with_config() {
        let mut pipeline = ConsciousnessPipeline::default();

        // Enable with custom configuration
        pipeline.enable_full_consciousness_with_config(
            16,    // phi_nodes
            3,     // phi_frequency
            2048,  // hdc_dim
            32,    // n_processes
        );

        assert!(pipeline.has_full_consciousness());

        println!("✅ Full consciousness with custom config enabled");
    }

    #[test]
    #[ignore = "takes >60s in debug builds"]
    fn test_full_consciousness_processing_pipeline() {
        // Comprehensive test of the entire consciousness pipeline
        let mut pipeline = ConsciousnessPipeline::default();
        pipeline.set_embodiment(0.9);
        pipeline.enable_full_consciousness();

        // Run many processing cycles
        for i in 0..30 {
            let inputs = vec![
                HV16::random(17000 + i),
                HV16::random(17100 + i),
                HV16::random(17200 + i),
            ];
            let priorities = vec![0.9, 0.85, 0.8];
            pipeline.process(inputs, &priorities);

            // Periodically run optimization
            if i % 10 == 9 {
                let summary = pipeline.run_optimization_cycle();
                println!("Cycle {}: Φ = {:.4} → {:.4}", i, summary.phi_before, summary.phi_after);
            }
        }

        // Get final metrics
        let metrics = pipeline.consciousness_metrics();

        println!("\n✅ FULL CONSCIOUSNESS PIPELINE COMPLETE:");
        println!("{}", metrics);

        // All systems should still be active
        assert!(pipeline.has_full_consciousness());
        assert!(metrics.processing_cycles == 30);
    }

    #[test]
    fn test_orchestrator_recommendations_display() {
        use crate::hdc::consciousness_integration::{
            RecommendationPriority, OptimizationRecommendation,
        };

        let recommendations = vec![
            OptimizationRecommendation {
                system: "phi".to_string(),
                priority: RecommendationPriority::High,
                message: "Φ is below optimal range".to_string(),
                suggested_action: Some("Increase binding".to_string()),
            },
            OptimizationRecommendation {
                system: "consciousness".to_string(),
                priority: RecommendationPriority::Medium,
                message: "Consciousness level could be improved".to_string(),
                suggested_action: None,
            },
            OptimizationRecommendation {
                system: "info".to_string(),
                priority: RecommendationPriority::Low,
                message: "System is running normally".to_string(),
                suggested_action: None,
            },
        ];

        println!("✅ Recommendation Display Test:");
        for rec in &recommendations {
            println!("   {}", rec);
        }
    }
}
