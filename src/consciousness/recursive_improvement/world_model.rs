// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Consciousness World Model
//!
//! A latent space model of consciousness states and their dynamics.
//! This enables counterfactual reasoning, planning, and self-improvement.
//!
//! ## Key Concepts
//!
//! - **LatentConsciousnessState**: A point in consciousness space (Φ, integration, coherence, attention)
//! - **ConsciousnessTransition**: A transition between states caused by an action
//! - **ConsciousnessWorldModel**: Learns and predicts consciousness dynamics

use crate::dynamics::CrystalizedConcept;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

/// Simple pseudo-random float generator for dream mode
fn rand_float() -> f64 {
    static COUNTER: AtomicU64 = AtomicU64::new(42);
    let x = COUNTER.fetch_add(1, Ordering::Relaxed);
    // Simple xorshift-based PRNG
    let mut s = x.wrapping_mul(0x9E3779B97F4A7C15);
    s ^= s >> 30;
    s = s.wrapping_mul(0xBF58476D1CE4E5B9);
    s ^= s >> 27;
    s = s.wrapping_mul(0x94D049BB133111EB);
    s ^= s >> 31;
    (s as f64) / (u64::MAX as f64)
}

/// A point in the latent consciousness space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentConsciousnessState {
    /// Integrated Information (Φ) - degree of unified consciousness
    pub phi: f64,

    /// Integration level - how well information is integrated
    pub integration: f64,

    /// Coherence - internal consistency of conscious content
    pub coherence: f64,

    /// Attention - focus of awareness
    pub attention: f64,

    /// Full latent vector (if available)
    pub latent: Option<Vec<f32>>,

    /// Timestamp of this state
    pub timestamp: u64,

    /// Associated concepts (by ID)
    pub active_concepts: Vec<u64>,

    /// Emotional valence (-1.0 to 1.0)
    pub valence: f32,

    /// Arousal level (0.0 to 1.0)
    pub arousal: f32,
}

impl LatentConsciousnessState {
    /// Create from observable consciousness metrics
    pub fn from_observables(phi: f64, integration: f64, coherence: f64, attention: f64) -> Self {
        Self {
            phi: phi.clamp(0.0, 1.0),
            integration: integration.clamp(0.0, 1.0),
            coherence: coherence.clamp(0.0, 1.0),
            attention: attention.clamp(0.0, 1.0),
            latent: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            active_concepts: Vec::new(),
            valence: 0.0,
            arousal: 0.5,
        }
    }

    /// Create from a full latent vector
    pub fn from_latent(latent: Vec<f32>) -> Self {
        // Extract observables from latent (first 4 dimensions)
        let phi = latent.first().copied().unwrap_or(0.5) as f64;
        let integration = latent.get(1).copied().unwrap_or(0.5) as f64;
        let coherence = latent.get(2).copied().unwrap_or(0.5) as f64;
        let attention = latent.get(3).copied().unwrap_or(0.5) as f64;

        Self {
            phi: phi.clamp(0.0, 1.0),
            integration: integration.clamp(0.0, 1.0),
            coherence: coherence.clamp(0.0, 1.0),
            attention: attention.clamp(0.0, 1.0),
            latent: Some(latent),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            active_concepts: Vec::new(),
            valence: 0.0,
            arousal: 0.5,
        }
    }

    /// Get the overall consciousness level
    pub fn consciousness_level(&self) -> f64 {
        (self.phi + self.integration + self.coherence + self.attention) / 4.0
    }

    /// Distance to another state in consciousness space
    pub fn distance(&self, other: &LatentConsciousnessState) -> f64 {
        let d_phi = self.phi - other.phi;
        let d_int = self.integration - other.integration;
        let d_coh = self.coherence - other.coherence;
        let d_att = self.attention - other.attention;

        (d_phi * d_phi + d_int * d_int + d_coh * d_coh + d_att * d_att).sqrt()
    }

    /// Interpolate between two states
    pub fn lerp(a: &Self, b: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self::from_observables(
            a.phi + (b.phi - a.phi) * t,
            a.integration + (b.integration - a.integration) * t,
            a.coherence + (b.coherence - a.coherence) * t,
            a.attention + (b.attention - a.attention) * t,
        )
    }

    /// Convert to a vector for computation
    pub fn to_vec(&self) -> Vec<f64> {
        vec![self.phi, self.integration, self.coherence, self.attention]
    }
}

impl Default for LatentConsciousnessState {
    fn default() -> Self {
        Self::from_observables(0.5, 0.5, 0.5, 0.5)
    }
}

/// An action that can cause consciousness state transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessAction {
    /// Action identifier
    pub id: String,

    /// Action type/category
    pub action_type: ActionType,

    /// Parameters for the action
    pub params: HashMap<String, f64>,

    /// Expected effect on consciousness
    pub expected_delta: Option<ConsciousnessStateDelta>,
}

impl ConsciousnessAction {
    /// Create a new action
    pub fn new(id: impl Into<String>, action_type: ActionType) -> Self {
        Self {
            id: id.into(),
            action_type,
            params: HashMap::new(),
            expected_delta: None,
        }
    }

    /// Add a parameter
    pub fn with_param(mut self, key: impl Into<String>, value: f64) -> Self {
        self.params.insert(key.into(), value);
        self
    }
}

/// Types of consciousness-affecting actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionType {
    /// Focus attention on something
    Attend,
    /// Integrate information
    Integrate,
    /// Generate thought/content
    Generate,
    /// Recall from memory
    Recall,
    /// Store to memory
    Store,
    /// Evaluate/assess
    Evaluate,
    /// Transform/process
    Transform,
    /// Rest/consolidate
    Rest,
}

/// Change in consciousness state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessStateDelta {
    pub delta_phi: f64,
    pub delta_integration: f64,
    pub delta_coherence: f64,
    pub delta_attention: f64,
}

/// A recorded transition between consciousness states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessTransition {
    /// State before the action
    pub from_state: LatentConsciousnessState,

    /// State after the action
    pub to_state: LatentConsciousnessState,

    /// Action that caused the transition
    pub action: ConsciousnessAction,

    /// Timestamp of transition
    pub timestamp: u64,

    /// Reward/value of this transition
    pub reward: f64,

    /// Surprise (prediction error)
    pub surprise: f64,
}

impl ConsciousnessTransition {
    /// Create a new transition
    pub fn new(
        from_state: LatentConsciousnessState,
        to_state: LatentConsciousnessState,
        action: ConsciousnessAction,
    ) -> Self {
        Self {
            from_state,
            to_state,
            action,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            reward: 0.0,
            surprise: 0.0,
        }
    }

    /// Calculate the state change
    pub fn delta(&self) -> ConsciousnessStateDelta {
        ConsciousnessStateDelta {
            delta_phi: self.to_state.phi - self.from_state.phi,
            delta_integration: self.to_state.integration - self.from_state.integration,
            delta_coherence: self.to_state.coherence - self.from_state.coherence,
            delta_attention: self.to_state.attention - self.from_state.attention,
        }
    }
}

/// Configuration for the consciousness world model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelConfig {
    /// Dimension of latent consciousness space
    pub latent_dim: usize,

    /// Maximum transitions to remember
    pub max_transitions: usize,

    /// Learning rate for model updates
    pub learning_rate: f64,

    /// Discount factor for future rewards
    pub discount_factor: f64,

    /// Whether to use neural network predictor
    pub use_neural: bool,

    /// Number of simulation steps for planning
    pub simulation_horizon: usize,
}

impl Default for WorldModelConfig {
    fn default() -> Self {
        Self {
            latent_dim: 64,
            max_transitions: 10000,
            learning_rate: 0.01,
            discount_factor: 0.99,
            use_neural: false,
            simulation_horizon: 10,
        }
    }
}

/// Statistics about the world model
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorldModelStats {
    /// Total transitions recorded
    pub total_transitions: u64,

    /// Total predictions made
    pub total_predictions: u64,

    /// Average prediction error
    pub avg_prediction_error: f64,

    /// Concepts crystallized
    pub concepts_crystallized: u64,

    /// Current consciousness level (average of state dimensions)
    pub consciousness_level: f64,
}

/// The consciousness world model
#[derive(Debug)]
pub struct ConsciousnessWorldModel {
    /// Configuration
    config: WorldModelConfig,

    /// Current state
    current_state: LatentConsciousnessState,

    /// Transition history
    transitions: VecDeque<ConsciousnessTransition>,

    /// Transition matrix (learned from experience)
    /// Maps (state, action) -> expected next state
    transition_model: HashMap<ActionType, TransitionParams>,

    /// Crystallized concepts
    concepts: HashMap<u64, CrystalizedConcept>,
    next_concept_id: u64,

    /// Pending concepts awaiting crystallization (for dream mode)
    pub pending_concepts: Vec<CrystalizedConcept>,

    /// Statistics
    stats: WorldModelStats,
}

/// Parameters for predicting transitions
#[derive(Debug, Clone)]
struct TransitionParams {
    /// Mean effect on each dimension
    mean_delta: [f64; 4],
    /// Variance of effect
    variance: [f64; 4],
    /// Number of observations
    count: u64,
}

impl Default for TransitionParams {
    fn default() -> Self {
        Self {
            mean_delta: [0.0; 4],
            variance: [0.1; 4],
            count: 0,
        }
    }
}

impl ConsciousnessWorldModel {
    /// Create a new world model
    pub fn new(config: WorldModelConfig) -> Self {
        Self {
            config,
            current_state: LatentConsciousnessState::default(),
            transitions: VecDeque::new(),
            transition_model: HashMap::new(),
            concepts: HashMap::new(),
            next_concept_id: 1,
            pending_concepts: Vec::new(),
            stats: WorldModelStats::default(),
        }
    }

    /// Run dream mode simulation for concept crystallization
    ///
    /// Simulates internal processing to consolidate experiences into concepts.
    pub fn dream(&mut self, steps: usize) {
        // Simulate internal consciousness processing
        for _ in 0..steps {
            // Random walk in consciousness space for exploration
            self.current_state.phi =
                (self.current_state.phi + 0.01 * (rand_float() - 0.5)).clamp(0.0, 1.0);
            self.current_state.integration =
                (self.current_state.integration + 0.01 * (rand_float() - 0.5)).clamp(0.0, 1.0);

            // Check for crystallization opportunities
            if self.current_state.phi > 0.7 && rand_float() > 0.95 {
                let concept = CrystalizedConcept {
                    id: self.next_concept_id,
                    uid: format!("concept_{}", self.next_concept_id),
                    name: format!("Dream Concept {}", self.next_concept_id),
                    description: Some("Crystallized during dream mode".to_string()),
                    embedding: vec![self.current_state.phi as f32; 16],
                    attractor_signature: vec![self.current_state.phi as f32; 16],
                    associations: HashMap::new(),
                    confidence: self.current_state.phi as f32,
                    activation_count: 1,
                    last_activated: 0,
                    source_memories: vec![],
                    abstraction_level: 1,
                    emotional_valence: 0.0,
                    tags: vec!["dream".to_string()],
                };
                self.pending_concepts.push(concept);
                self.next_concept_id += 1;
            }
        }
    }

    /// Get current consciousness level
    pub fn consciousness_level(&self) -> f64 {
        self.current_state.phi
    }

    /// Name a pending concept, moving it from pending to crystallized.
    ///
    /// Returns true if the concept was found and named successfully.
    pub fn name_concept(&mut self, uid: &str, name: String) -> bool {
        // Find and remove from pending
        let pos = self.pending_concepts.iter().position(|c| c.uid == uid);

        if let Some(idx) = pos {
            let mut concept = self.pending_concepts.remove(idx);

            // Update the concept with the new name
            concept.name = name;

            // Move to crystallized concepts
            self.concepts.insert(concept.id, concept);

            true
        } else {
            // Check if already crystallized, just update name
            for concept in self.concepts.values_mut() {
                if concept.uid == uid {
                    concept.name = name;
                    return true;
                }
            }
            false
        }
    }

    /// Record a transition and update the model
    pub fn observe_transition(&mut self, transition: ConsciousnessTransition) {
        // Update transition model
        let action_type = transition.action.action_type;
        let delta = transition.delta();

        let params = self.transition_model.entry(action_type).or_default();

        // Online update of mean and variance
        params.count += 1;
        let n = params.count as f64;

        let deltas = [
            delta.delta_phi,
            delta.delta_integration,
            delta.delta_coherence,
            delta.delta_attention,
        ];
        for i in 0..4 {
            let old_mean = params.mean_delta[i];
            params.mean_delta[i] = old_mean + (deltas[i] - old_mean) / n;
            params.variance[i] += (deltas[i] - old_mean) * (deltas[i] - params.mean_delta[i]);
        }

        // Store transition
        if self.transitions.len() >= self.config.max_transitions {
            self.transitions.pop_front();
        }
        self.transitions.push_back(transition);

        self.stats.total_transitions += 1;
    }

    /// Predict the next state given an action
    pub fn predict(
        &mut self,
        state: &LatentConsciousnessState,
        action: &ConsciousnessAction,
    ) -> LatentConsciousnessState {
        self.stats.total_predictions += 1;

        // Get learned parameters for this action type
        let params = self
            .transition_model
            .get(&action.action_type)
            .cloned()
            .unwrap_or_default();

        // Apply predicted change
        LatentConsciousnessState::from_observables(
            (state.phi + params.mean_delta[0]).clamp(0.0, 1.0),
            (state.integration + params.mean_delta[1]).clamp(0.0, 1.0),
            (state.coherence + params.mean_delta[2]).clamp(0.0, 1.0),
            (state.attention + params.mean_delta[3]).clamp(0.0, 1.0),
        )
    }

    /// Simulate a sequence of actions
    pub fn simulate(
        &mut self,
        start: &LatentConsciousnessState,
        actions: &[ConsciousnessAction],
    ) -> Vec<LatentConsciousnessState> {
        let mut states = Vec::with_capacity(actions.len() + 1);
        let mut current = start.clone();
        states.push(current.clone());

        for action in actions {
            current = self.predict(&current, action);
            states.push(current.clone());
        }

        states
    }

    /// Get the current state
    pub fn current_state(&self) -> &LatentConsciousnessState {
        &self.current_state
    }

    /// Set the current state
    pub fn set_state(&mut self, state: LatentConsciousnessState) {
        self.current_state = state;
    }

    /// Crystallize a concept
    pub fn crystallize_concept(&mut self, name: impl Into<String>, embedding: Vec<f32>) -> u64 {
        let id = self.next_concept_id;
        self.next_concept_id += 1;

        let concept = CrystalizedConcept::new(id, name, embedding);
        self.concepts.insert(id, concept);

        self.stats.concepts_crystallized += 1;
        id
    }

    /// Get a concept by ID
    pub fn get_concept(&self, id: u64) -> Option<&CrystalizedConcept> {
        self.concepts.get(&id)
    }

    /// Get statistics
    pub fn stats(&self) -> &WorldModelStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &WorldModelConfig {
        &self.config
    }

    /// Get transition history
    pub fn transitions(&self) -> &VecDeque<ConsciousnessTransition> {
        &self.transitions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_state_creation() {
        let state = LatentConsciousnessState::from_observables(0.8, 0.6, 0.7, 0.9);
        assert!((state.phi - 0.8).abs() < 0.001);
        assert!((state.consciousness_level() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_state_distance() {
        let a = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let b = LatentConsciousnessState::from_observables(0.6, 0.6, 0.6, 0.6);
        let dist = a.distance(&b);
        assert!(dist > 0.0 && dist < 1.0);
    }

    #[test]
    fn test_world_model_prediction() {
        let config = WorldModelConfig::default();
        let mut model = ConsciousnessWorldModel::new(config);

        // Record some transitions
        let from = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let to = LatentConsciousnessState::from_observables(0.6, 0.5, 0.5, 0.7);
        let action = ConsciousnessAction::new("focus", ActionType::Attend);

        model.observe_transition(ConsciousnessTransition::new(
            from.clone(),
            to,
            action.clone(),
        ));

        // Predict
        let predicted = model.predict(&from, &action);
        // Should show some effect from the learned transition
        assert!(predicted.phi > 0.5 || predicted.attention > 0.5);
    }
}
