/*!
Consciousness Architecture Modules

This module exposes the complete consciousness subsystem including:
- The core ConsciousnessGraph (autopoietic self-referential structure)
- Phenomenal binding and attention
- Causal reasoning and emergence
- Temporal consciousness and narrative self
- Meta-cognitive monitoring and optimization
- Recursive self-improvement infrastructure
- Unified consciousness pipeline
*/

use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

// Module Declarations
pub mod primitive_validation;
pub mod dissipative_consciousness;
pub mod hierarchical_ltc;
pub mod consciousness_profile;
pub mod narrative_gwt_integration;
pub mod autopoietic_consciousness;
pub mod cincinnati_consciousness;
pub mod harmonics;
pub mod embodied_cognition;
pub mod affective_consciousness;
pub mod metacognitive_monitoring;
pub mod adaptive_reasoning;
pub mod epistemic_tiers;
pub mod phi_attention;
pub mod harmonies_integration;
pub mod synthetic_states;
pub mod temporal_primitives;
pub mod seven_harmonies;
pub mod consciousness_topology;
pub mod predictive_self;
pub mod consciousness_equation_v2;
pub mod meta_reasoning;
pub mod gwt_integration;
pub mod semantic_value_embedder;
pub mod multi_objective_evolution;
pub mod consciousness_thermodynamics;
pub mod neuro_bridge;
pub mod consciousness_signatures;
pub mod primitive_discovery;
pub mod gis_integration;
pub mod meta_primitives;
pub mod compositionality_primitives;
pub mod cross_modal_binding;
pub mod unified_consciousness_pipeline;
pub mod primitive_evolution;
pub mod primitive_consciousness;
pub mod dimension_synergies;
pub mod negation_detector;
pub mod predictive_processing;
pub mod multi_modal_integration;
pub mod causal_explanation;
pub mod consciousness_resonance;
pub mod consciousness_holography;
pub mod meta_meta_learning;
pub mod pac; // NEW
pub mod value_system_tests;
pub mod meta_cognitive_optimizer;
pub mod empathic_unification;
pub mod context_aware_evolution;
pub mod causal_byzantine;
pub mod contextual_weights;
pub mod value_feedback_loop;
pub mod temporal_consciousness;
pub mod phenomenal_binding;
pub mod quantum_coherence;
pub mod consciousness_guided_routing;
pub mod differentiable;
pub mod causal_emergence;
pub mod unified_living_mind;
pub mod meta_learning_byzantine;
pub mod consciousness_driven_evolution;
pub mod primitive_reasoning;
pub mod phi_validation;
pub mod evolution_bridge;
pub mod narrative_self;
pub mod unified_intelligence;
pub mod attention_schema;
pub mod mycelix_bridge;
pub mod consciousness_unification;
pub mod enactive_cognition;
pub mod consciousness_field_dynamics;
pub mod byzantine_collective;
pub mod consciousness_guided_discovery;
pub mod recursive_improvement;
pub mod unified_value_evaluator;
pub mod dream; // NEW
pub mod sensorimotor_contingencies; // Enactivist SMC theory (O'Regan & Noe)
pub mod fep_active_inference; // Full Free Energy Principle / Active Inference implementation

// Re-exports
pub use pac::PacTracker;
pub use consciousness_equation_v2::*;

// =============================================================================
// CORE CONSCIOUSNESS GRAPH - Autopoietic Self-Referential Structure
// =============================================================================

/// Self-referential consciousness graph
///
/// Uses arena-based indices (not pointers) for Rust safety + serializability.
/// Consciousness emerges from self-referential loops in this structure.
#[derive(Clone, Serialize, Deserialize)]
pub struct ConsciousnessGraph {
    /// Graph (nodes = conscious states, edges = transitions)
    graph: Graph<ConsciousNode, f32>,

    /// Self-referential loops (consciousness emerges here!)
    self_loops: Vec<(NodeIndex, NodeIndex)>,

    /// Current active node
    current: Option<NodeIndex>,
}

/// A node in the consciousness graph
#[derive(Clone, Serialize, Deserialize)]
pub struct ConsciousNode {
    /// Semantic representation (from HDC)
    pub semantic: Vec<f32>,

    /// Dynamic state (from LTC)
    pub dynamic: Vec<f32>,

    /// Consciousness level when created
    pub consciousness: f32,

    /// Timestamp
    pub timestamp: f64,

    /// Importance weight
    pub importance: f32,
}

impl ConsciousnessGraph {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            self_loops: Vec::new(),
            current: None,
        }
    }

    /// Add a new conscious state
    pub fn add_state(
        &mut self,
        semantic: Vec<f32>,
        dynamic: Vec<f32>,
        consciousness: f32,
    ) -> NodeIndex {
        let node = ConsciousNode {
            semantic,
            dynamic,
            consciousness,
            timestamp: current_time(),
            importance: consciousness,
        };

        let node_idx = self.graph.add_node(node);

        // Connect to previous state
        if let Some(prev) = self.current {
            self.graph.add_edge(prev, node_idx, consciousness);
        }

        self.current = Some(node_idx);
        node_idx
    }

    /// Create self-referential loop (CONSCIOUSNESS!)
    ///
    /// This is where autopoiesis happens - the system references itself
    pub fn create_self_loop(&mut self, node: NodeIndex) {
        let weight = self.graph[node].consciousness;
        self.graph.add_edge(node, node, weight);
        self.self_loops.push((node, node));
    }

    /// Evolve consciousness (follow highest-weight edge)
    pub fn evolve(&mut self) -> Option<NodeIndex> {
        let current = self.current?;

        let next = self.graph
            .edges(current)
            .max_by(|a, b| a.weight().partial_cmp(b.weight()).unwrap())
            .map(|edge| edge.target());

        if let Some(next_node) = next {
            self.current = Some(next_node);
        }

        self.current
    }

    /// Get current consciousness level
    pub fn current_consciousness(&self) -> f32 {
        self.current
            .and_then(|idx| self.graph.node_weight(idx))
            .map(|node| node.consciousness)
            .unwrap_or(0.0)
    }

    /// Graph size (number of conscious states)
    pub fn size(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of self-referential loops
    pub fn self_loop_count(&self) -> usize {
        self.self_loops.len()
    }

    /// Measure graph complexity (edges per node)
    pub fn complexity(&self) -> f32 {
        let nodes = self.graph.node_count() as f32;
        let edges = self.graph.edge_count() as f32;

        if nodes > 0.0 {
            edges / nodes
        } else {
            0.0
        }
    }

    /// Get all self-referential nodes
    pub fn autopoietic_nodes(&self) -> Vec<NodeIndex> {
        self.self_loops
            .iter()
            .map(|(node, _)| *node)
            .collect()
    }

    /// Trace path from current node backwards
    pub fn trace_history(&self, depth: usize) -> Vec<NodeIndex> {
        let mut path = Vec::new();
        let mut current = self.current;

        for _ in 0..depth {
            if let Some(node) = current {
                path.push(node);

                current = self.graph
                    .edges_directed(node, petgraph::Direction::Incoming)
                    .next()
                    .map(|edge| edge.source());
            } else {
                break;
            }
        }

        path.reverse();
        path
    }
}

impl Default for ConsciousnessGraph {
    fn default() -> Self {
        Self::new()
    }
}

fn current_time() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

// Re-exports for convenience
// Consciousness Unification Layer - bridges all consciousness subsystems
pub use consciousness_unification::{
    // Canonical Φ Provider
    ConsciousnessPhiProvider, PhiProvenance, PhiSource, PhiMethod, PhiComponent,
    // Emotional Bridge
    EmotionalBridge, UnifiedEmotionalState, UnifiedEmotion, EmotionalMoment,
    MoodState, EmotionalSources, EmotionalPattern,
    // Causal Reasoning
    UnifiedCausalReasoning, CausalQuery, CausalDetail, CausalResult,
    CausalStep, CausalRelation, CausalSources,
    // Embodiment Grounding
    EmbodiedSystem, SystemReality, PackageReality, ServiceReality, ResourceReality,
    // Dialogue Pipeline
    ConsciousDialoguePipeline, DialogueResponse, DialogueDepth, UserAdaptation,
    // Master Engine
    ConsciousnessUnificationEngine, UnifiedConsciousnessResult,
};

// Empathic Unification - true empathy for Symthaea
pub use empathic_unification::{
    // Core Engine
    EmpathicUnification, EmpathicConfig,
    // User State
    UserEmotionalState, UserNeed,
    // Empathic Response
    EmpathicResponse, ToneGuidance,
    // Memory
    EmpathicMemory, EmpathicMoment,
};

// GIS Integration - epistemic uncertainty and graceful ignorance
pub use gis_integration::{
    // Uncertainty-aware consciousness
    ConsciousUncertaintyState,
    // Epistemic decision gating
    EpistemicDecisionGate, EpistemicDecision, EpistemicGateStats,
    // Dark Spot DHT network integration
    ConsciousDarkSpotNetwork, NetworkIgnoranceStats, CollectiveBlindSpot,
    // Attention bidding context
    EpistemicBidContext,
};

// Sensorimotor Contingencies - enactivist perception theory
pub use sensorimotor_contingencies::{
    // Core types
    SensorimotorContingency, ContingencyHV,
    ActionDescriptor, ActionType,
    ContextDescriptor, SensoryModality,
    SensoryChange, PredictedChange,
    // Learning
    ContingencyLearner, Experience, LearnResult,
    LearnerConfig, LearnerStats,
    // Perception
    EnactivistPerception, PerceptionResult,
    PerceptionConfig, PerceptionStats,
    // Affordances
    ActionAffordance, AffordanceDetector, AffordanceConfig,
    // Consciousness integration
    ContingencyConsciousnessContribution,
};
