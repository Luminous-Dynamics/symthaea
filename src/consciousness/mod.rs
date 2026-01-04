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

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONSCIOUSNESS ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Phenomenal  │  │   Causal    │  │  Temporal   │  │   Meta-     │    │
│  │  Binding    │  │  Emergence  │  │Consciousness│  │  Cognitive  │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │                │           │
│         └────────────────┼────────────────┼────────────────┘           │
│                          │                │                            │
│                          ▼                ▼                            │
│                   ┌─────────────────────────────┐                      │
│                   │  Unified Consciousness      │                      │
│                   │       Pipeline              │                      │
│                   └─────────────────────────────┘                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```
*/

#![allow(dead_code, unused_variables, unused_imports)]

use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

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

    // =========================================================================
    // CAUSAL-AWARE CONSCIOUSNESS METHODS
    // =========================================================================

    /// Extract state signals for causal analysis
    ///
    /// Returns vectors of consciousness levels, semantic norms, and dynamic norms
    /// over the state history, suitable for causal discovery.
    pub fn extract_causal_signals(&self, depth: usize) -> CausalSignals {
        let history = self.trace_history(depth);

        let mut consciousness: Vec<f64> = Vec::with_capacity(history.len());
        let mut semantic_norm: Vec<f64> = Vec::with_capacity(history.len());
        let mut dynamic_norm: Vec<f64> = Vec::with_capacity(history.len());
        let mut importance: Vec<f64> = Vec::with_capacity(history.len());

        for node_idx in &history {
            if let Some(node) = self.graph.node_weight(*node_idx) {
                consciousness.push(node.consciousness as f64);
                semantic_norm.push(
                    node.semantic.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
                );
                dynamic_norm.push(
                    node.dynamic.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
                );
                importance.push(node.importance as f64);
            }
        }

        CausalSignals {
            consciousness,
            semantic_norm,
            dynamic_norm,
            importance,
        }
    }

    /// Compute causal transition weights between states
    ///
    /// Uses the difference in semantic/dynamic states to weight transitions
    /// based on causal influence rather than just temporal proximity.
    pub fn compute_causal_transitions(&self) -> Vec<(NodeIndex, NodeIndex, f32)> {
        let mut transitions = Vec::new();

        for edge in self.graph.edge_references() {
            let source = edge.source();
            let target = edge.target();

            if let (Some(src_node), Some(tgt_node)) = (
                self.graph.node_weight(source),
                self.graph.node_weight(target),
            ) {
                // Compute semantic causation (change in representation)
                let semantic_delta: f32 = src_node.semantic.iter()
                    .zip(tgt_node.semantic.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>() / src_node.semantic.len().max(1) as f32;

                // Compute dynamic causation (change in dynamics)
                let dynamic_delta: f32 = src_node.dynamic.iter()
                    .zip(tgt_node.dynamic.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>() / src_node.dynamic.len().max(1) as f32;

                // Causal weight: combination of change magnitude and consciousness
                let causal_weight = (semantic_delta + dynamic_delta) *
                    (src_node.consciousness + tgt_node.consciousness) / 2.0;

                transitions.push((source, target, causal_weight));
            }
        }

        transitions
    }

    /// Identify causally influential states
    ///
    /// Returns nodes that have high causal influence on subsequent states
    /// (high outgoing causal weight relative to incoming).
    pub fn find_causal_sources(&self) -> Vec<(NodeIndex, f32)> {
        let transitions = self.compute_causal_transitions();

        let mut outgoing: std::collections::HashMap<NodeIndex, f32> =
            std::collections::HashMap::new();
        let mut incoming: std::collections::HashMap<NodeIndex, f32> =
            std::collections::HashMap::new();

        for (src, tgt, weight) in &transitions {
            *outgoing.entry(*src).or_insert(0.0) += weight;
            *incoming.entry(*tgt).or_insert(0.0) += weight;
        }

        let mut sources: Vec<(NodeIndex, f32)> = outgoing.iter()
            .map(|(&node, &out_w)| {
                let in_w = *incoming.get(&node).unwrap_or(&0.0);
                // Causal influence = outgoing - incoming (net causal effect)
                (node, out_w - in_w)
            })
            .filter(|(_, influence)| *influence > 0.0)
            .collect();

        sources.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sources
    }

    /// Evolve with causal awareness
    ///
    /// Instead of just following highest weight, considers causal structure:
    /// prefers transitions where current state has high causal influence.
    pub fn evolve_causal(&mut self) -> Option<NodeIndex> {
        let current = self.current?;
        let transitions = self.compute_causal_transitions();

        // Find outgoing transitions from current node
        let outgoing: Vec<_> = transitions.iter()
            .filter(|(src, _, _)| *src == current)
            .collect();

        if outgoing.is_empty() {
            return self.evolve(); // Fall back to standard evolution
        }

        // Select target with highest causal weight
        let best = outgoing.iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())?;

        self.current = Some(best.1);
        self.current
    }

    /// Compute causal phi (integrated information) approximation
    ///
    /// Measures how much causal influence flows through the graph
    /// vs being localized to individual nodes.
    pub fn causal_phi(&self) -> f32 {
        let transitions = self.compute_causal_transitions();
        if transitions.is_empty() {
            return 0.0;
        }

        // Total causal weight
        let total: f32 = transitions.iter().map(|(_, _, w)| w).sum();

        // Count unique causal relationships
        let n_transitions = transitions.len() as f32;
        let n_nodes = self.graph.node_count() as f32;

        if n_nodes < 2.0 {
            return 0.0;
        }

        // Phi approximation: normalized causal connectivity
        let max_possible = n_nodes * (n_nodes - 1.0);
        let connectivity = n_transitions / max_possible;

        // Average causal strength weighted by connectivity
        let avg_strength = total / n_transitions.max(1.0);

        connectivity * avg_strength
    }
}

/// Signals extracted for causal analysis
#[derive(Debug, Clone)]
pub struct CausalSignals {
    pub consciousness: Vec<f64>,
    pub semantic_norm: Vec<f64>,
    pub dynamic_norm: Vec<f64>,
    pub importance: Vec<f64>,
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

// =============================================================================
// CORE CONSCIOUSNESS MODULES
// Note: Some modules have incomplete dependencies (simd_hv16, sparse_ltc, etc.)
// and are commented out until those dependencies are added.
// =============================================================================

/// Phenomenal binding - temporal synchronization for unified consciousness
pub mod phenomenal_binding;

/// Attention schema - attention as a model of attention
pub mod attention_schema;

/// Temporal consciousness - time perception and temporal binding
pub mod temporal_consciousness;

/// Narrative self - autobiographical continuity
pub mod narrative_self;

/// Predictive self - self-model as prediction
pub mod predictive_self;

// =============================================================================
// CAUSAL AND EMERGENT CONSCIOUSNESS
// =============================================================================

/// Causal emergence - higher-level causation
pub mod causal_emergence;

/// Causal explanation - why-based reasoning
pub mod causal_explanation;

// TODO: Enable when byzantine_collective is available
// pub mod causal_byzantine;

// TODO: Enable when unified_intelligence is available
// pub mod byzantine_collective;

// =============================================================================
// THERMODYNAMIC AND FIELD CONSCIOUSNESS
// =============================================================================

/// Consciousness thermodynamics - free energy and entropy
pub mod consciousness_thermodynamics;

/// Consciousness field dynamics - field-theoretic approach
pub mod consciousness_field_dynamics;

/// Dissipative consciousness - far-from-equilibrium dynamics
pub mod dissipative_consciousness;

/// Quantum coherence - quantum-inspired coherent consciousness
pub mod quantum_coherence;

// =============================================================================
// HOLOGRAPHIC AND RESONANT CONSCIOUSNESS
// =============================================================================

/// Consciousness holography - holographic information encoding
pub mod consciousness_holography;

/// Consciousness resonance - resonant coupling
pub mod consciousness_resonance;

/// Harmonics - harmonic oscillations in consciousness
pub mod harmonics;

// =============================================================================
// VALUE SYSTEMS
// =============================================================================

/// Seven Harmonies - Core value system for consciousness-guided AI
pub mod seven_harmonies;

/// Harmonies Integration - Connect Seven Harmonies to consciousness modules
pub mod harmonies_integration;

/// Unified Value Evaluator - Consciousness-guided decision making
pub mod unified_value_evaluator;

/// Mycelix Bridge - Consciousness-gated governance and federated value learning
pub mod mycelix_bridge;

/// Value Feedback Loop - Meta-cognitive learning from value decisions
pub mod value_feedback_loop;

/// Value System Calibration Tests
#[cfg(test)]
mod value_system_tests;

// =============================================================================
// AFFECTIVE AND EMBODIED CONSCIOUSNESS
// =============================================================================

/// Affective consciousness - emotional experience
pub mod affective_consciousness;

/// Embodied cognition - body-based understanding
pub mod embodied_cognition;

/// Enactive cognition - sensorimotor coupling
pub mod enactive_cognition;

// =============================================================================
// META-COGNITIVE MODULES
// =============================================================================

/// Metacognitive monitoring - self-awareness of cognition
pub mod metacognitive_monitoring;

/// Meta-reasoning - reasoning about reasoning
pub mod meta_reasoning;

/// Meta-cognitive optimizer - optimizing meta-cognition
pub mod meta_cognitive_optimizer;

/// Meta-meta-learning - learning to learn to learn
pub mod meta_meta_learning;

// TODO: Enable when byzantine_collective is available
// pub mod meta_learning_byzantine;

// =============================================================================
// ADAPTIVE AND EVOLUTIONARY MODULES
// =============================================================================

/// Adaptive reasoning - context-sensitive reasoning
pub mod adaptive_reasoning;

/// Context-aware evolution - context-sensitive evolutionary optimization
pub mod context_aware_evolution;

/// Multi-objective evolution - multi-objective evolutionary optimization
pub mod multi_objective_evolution;

/// Primitive evolution - consciousness-guided evolutionary discovery of optimal primitives
pub mod primitive_evolution;

/// Consciousness-driven evolution - self-improving architecture
pub mod consciousness_driven_evolution;

// =============================================================================
// PRIMITIVE SYSTEMS
// =============================================================================

/// Primitive validation - validation framework for reasoning primitives
pub mod primitive_validation;

/// Primitive reasoning - reasoning with primitives
pub mod primitive_reasoning;

/// Meta primitives - primitives about primitives
pub mod meta_primitives;

/// Temporal primitives - time-related primitives
pub mod temporal_primitives;

/// Compositionality primitives - compositional structure
pub mod compositionality_primitives;

// =============================================================================
// HIERARCHY AND TOPOLOGY
// =============================================================================

/// Hierarchical LTC - biologically-inspired hierarchical LTC architecture (25x speedup)
pub mod hierarchical_ltc;

/// Consciousness topology - topological structure
pub mod consciousness_topology;

// TODO: Enable when simd_hv16 module is added
// pub mod consciousness_signatures;

/// Consciousness profile - individual consciousness fingerprint
pub mod consciousness_profile;

// =============================================================================
// INTEGRATION AND UNIFICATION
// =============================================================================

// TODO: Enable when global_workspace module is added to hdc
// pub mod gwt_integration;

// TODO: Enable when gwt_integration is available
// pub mod narrative_gwt_integration;

/// Cross-modal binding - multi-sensory integration
pub mod cross_modal_binding;

// TODO: Enable when simd_hv16 and hierarchical_ltc modules are added
// pub mod unified_consciousness_pipeline;

// TODO: Enable when CollectivePrimitiveEvolution is exported
// pub mod unified_intelligence;

/// Unified living mind - embodied unified cognition
pub mod unified_living_mind;

// =============================================================================
// SYNTHETIC AND EXPERIMENTAL
// =============================================================================

/// Synthetic states - artificial conscious states
pub mod synthetic_states;

/// Synthetic states v2 - improved synthetic consciousness
pub mod synthetic_states_v2;

/// Synthetic states v2 backup
pub mod synthetic_states_v2_backup;

/// Synthetic states v3 bind - binding-based synthesis
pub mod synthetic_states_v3_bind;

/// Consciousness equation v2 - mathematical formulation
pub mod consciousness_equation_v2;

// TODO: Enable when ConsciousnessStateV2 is re-exported
// pub mod differentiable;

// =============================================================================
// GUIDANCE AND ROUTING
// =============================================================================

/// Consciousness-guided discovery - consciousness-driven exploration
pub mod consciousness_guided_discovery;

// TODO: Enable when predictive_consciousness_kalman module is added
// pub mod consciousness_guided_routing;

/// Predictive processing - prediction-based cognition
pub mod predictive_processing;

// =============================================================================
// EPISTEMIC AND VALIDATION
// =============================================================================

/// Epistemic tiers - levels of knowledge
pub mod epistemic_tiers;

// TODO: Add libm dependency to enable
// pub mod phi_validation;

/// Dimension synergies - dimensional interactions
pub mod dimension_synergies;

// =============================================================================
// AUTOPOIESIS
// =============================================================================

/// Autopoietic consciousness - self-creating consciousness
pub mod autopoietic_consciousness;

// =============================================================================
// RECURSIVE IMPROVEMENT INFRASTRUCTURE
// =============================================================================

/// Recursive improvement infrastructure - self-improvement primitives
pub mod recursive_improvement;

// =============================================================================
// RE-EXPORTS FOR CONVENIENCE
// =============================================================================

// Core graph (always available)
// ConsciousnessGraph is defined above in this file

// Re-exports from available modules (check each module for actual exports)
// Note: Many modules have incomplete exports - add as they are fixed
pub use consciousness_equation_v2::{ConsciousnessStateV2, CoreComponent, EquationConfig};
