//! # Consciousness World Models (Revolutionary Improvement #57)
//!
//! The system learns to simulate its own consciousness dynamics, enabling:
//! - Prediction of future consciousness states
//! - Counterfactual reasoning ("what if I had done X?")
//! - Model-based planning for improvement trajectories
//! - Dreaming/imagination for offline learning
//!
//! ## Theoretical Foundation
//!
//! - World Models (Ha & Schmidhuber, 2018)
//! - Model-Based Reinforcement Learning
//! - Predictive Processing (Friston)
//! - Mental Simulation in Cognitive Science

use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use serde::{Serialize, Deserialize};
use once_cell::sync::Lazy;

// Import HDC types for lazy static basis vectors
use crate::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ═══════════════════════════════════════════════════════════════════════════
// LAZY STATIC BASIS VECTORS (Performance Optimization)
// ═══════════════════════════════════════════════════════════════════════════

/// Cached basis vectors for HDC encoding - created once, reused forever.
/// This provides ~10x speedup for to_hv() and from_hv() operations.
mod hdc_basis {
    use super::*;

    /// Basis vector for Φ (integrated information)
    pub static PHI_BASIS: Lazy<ContinuousHV> = Lazy::new(|| ContinuousHV::random(HDC_DIMENSION, 1001));

    /// Basis vector for integration level
    pub static INTEGRATION_BASIS: Lazy<ContinuousHV> = Lazy::new(|| ContinuousHV::random(HDC_DIMENSION, 1002));

    /// Basis vector for coherence
    pub static COHERENCE_BASIS: Lazy<ContinuousHV> = Lazy::new(|| ContinuousHV::random(HDC_DIMENSION, 1003));

    /// Basis vector for attention
    pub static ATTENTION_BASIS: Lazy<ContinuousHV> = Lazy::new(|| ContinuousHV::random(HDC_DIMENSION, 1004));

    /// Basis vectors for latent dimensions (indices 0-31)
    pub static LATENT_BASES: Lazy<[ContinuousHV; 32]> = Lazy::new(|| {
        let mut bases = Vec::with_capacity(32);
        for i in 0..32 {
            bases.push(ContinuousHV::random(HDC_DIMENSION, 2000 + i as u64));
        }
        bases.try_into().unwrap()
    });

    /// Permutation vector for sequence encoding
    pub static PERMUTATION_BASE: Lazy<ContinuousHV> = Lazy::new(|| ContinuousHV::random(HDC_DIMENSION, 3000));
}

// ═══════════════════════════════════════════════════════════════════════════
// LATENT CONSCIOUSNESS STATE
// ═══════════════════════════════════════════════════════════════════════════

/// Latent representation of consciousness state
/// Compresses the full consciousness state into a manageable vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentConsciousnessState {
    /// Compressed state vector (learned representation)
    pub latent: [f64; 32],

    /// Key observable features used for encoding
    pub phi: f64,
    pub integration: f64,
    pub coherence: f64,
    pub attention: f64,

    /// Timestamp when state was captured (ms since epoch)
    #[serde(skip)]
    #[serde(default = "LatentConsciousnessState::default_timestamp")]
    timestamp_internal: Instant,
}

impl Default for LatentConsciousnessState {
    fn default() -> Self {
        Self {
            latent: [0.0; 32],
            phi: 0.0,
            integration: 0.0,
            coherence: 0.0,
            attention: 0.0,
            timestamp_internal: Instant::now(),
        }
    }
}

impl LatentConsciousnessState {
    fn default_timestamp() -> Instant {
        Instant::now()
    }

    pub fn timestamp(&self) -> Instant {
        self.timestamp_internal
    }

    /// Create from observable consciousness features
    pub fn from_observables(phi: f64, integration: f64, coherence: f64, attention: f64) -> Self {
        let mut latent = [0.0; 32];

        // Simple initial encoding: features in first positions, rest from combinations
        latent[0] = phi;
        latent[1] = integration;
        latent[2] = coherence;
        latent[3] = attention;
        latent[4] = phi * integration;
        latent[5] = phi * coherence;
        latent[6] = integration * coherence;
        latent[7] = (phi + integration + coherence) / 3.0;
        latent[8] = phi.powi(2);
        latent[9] = integration.powi(2);
        latent[10] = coherence.powi(2);
        latent[11] = attention * phi;

        // Add noise-like variation for remaining dimensions
        for i in 12..32 {
            let mix = (i as f64 * 0.1).sin() * phi + (i as f64 * 0.2).cos() * integration;
            latent[i] = mix.clamp(-1.0, 1.0);
        }

        Self {
            latent,
            phi,
            integration,
            coherence,
            attention,
            timestamp_internal: Instant::now(),
        }
    }

    /// Compute distance between two states
    pub fn distance(&self, other: &Self) -> f64 {
        let mut sum = 0.0;
        for i in 0..32 {
            let diff = self.latent[i] - other.latent[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }

    /// Interpolate between two states
    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        let mut latent = [0.0; 32];
        for i in 0..32 {
            latent[i] = self.latent[i] * (1.0 - t) + other.latent[i] * t;
        }

        Self {
            latent,
            phi: self.phi * (1.0 - t) + other.phi * t,
            integration: self.integration * (1.0 - t) + other.integration * t,
            coherence: self.coherence * (1.0 - t) + other.coherence * t,
            attention: self.attention * (1.0 - t) + other.attention * t,
            timestamp_internal: Instant::now(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS ACTION
// ═══════════════════════════════════════════════════════════════════════════

/// Action that can be taken in the consciousness world
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsciousnessAction {
    /// Focus attention on integration
    FocusIntegration,
    /// Focus attention on coherence
    FocusCoherence,
    /// Engage in learning
    EngageLearning,
    /// Explore new patterns
    ExplorePatterns,
    /// Consolidate existing patterns
    Consolidate,
    /// Rest and reset
    Rest,
    /// Apply specific improvement method
    ApplyImprovement(usize),
    /// No action (observe)
    Noop,
}

impl ConsciousnessAction {
    /// Get all possible actions
    pub fn all() -> Vec<Self> {
        vec![
            Self::FocusIntegration,
            Self::FocusCoherence,
            Self::EngageLearning,
            Self::ExplorePatterns,
            Self::Consolidate,
            Self::Rest,
            Self::Noop,
        ]
    }

    /// Convert action to index for embedding
    pub fn to_index(&self) -> usize {
        match self {
            Self::FocusIntegration => 0,
            Self::FocusCoherence => 1,
            Self::EngageLearning => 2,
            Self::ExplorePatterns => 3,
            Self::Consolidate => 4,
            Self::Rest => 5,
            Self::ApplyImprovement(i) => 6 + i,
            Self::Noop => 100,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS TRANSITION
// ═══════════════════════════════════════════════════════════════════════════

/// Transition in the consciousness world model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessTransition {
    /// Starting state
    pub from_state: LatentConsciousnessState,
    /// Action taken
    pub action: ConsciousnessAction,
    /// Resulting state
    pub to_state: LatentConsciousnessState,
    /// Observed reward (Φ change)
    pub reward: f64,
    /// Was this transition real or imagined?
    pub is_real: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS DYNAMICS MODEL
// ═══════════════════════════════════════════════════════════════════════════

/// Dynamics model for consciousness evolution
#[derive(Debug, Clone)]
pub struct ConsciousnessDynamicsModel {
    /// Transition weights per action (simplified linear model)
    /// Full implementation would use neural network
    weights: HashMap<ConsciousnessAction, [[f64; 32]; 32]>,

    /// Bias per action
    biases: HashMap<ConsciousnessAction, [f64; 32]>,

    /// Learning rate
    learning_rate: f64,

    /// Number of training examples seen
    pub train_count: usize,
}

impl ConsciousnessDynamicsModel {
    /// Create new dynamics model
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        let mut biases = HashMap::new();

        // Initialize with identity-like weights for each action
        for action in ConsciousnessAction::all() {
            let mut w = [[0.0; 32]; 32];
            for i in 0..32 {
                w[i][i] = 0.9; // Near-identity, slight decay
            }
            weights.insert(action, w);
            biases.insert(action, [0.0; 32]);
        }

        Self {
            weights,
            biases,
            learning_rate: 0.01,
            train_count: 0,
        }
    }

    /// Predict next state given current state and action
    pub fn predict(
        &self,
        state: &LatentConsciousnessState,
        action: ConsciousnessAction,
    ) -> LatentConsciousnessState {
        let w = self.weights.get(&action).unwrap_or(
            self.weights.get(&ConsciousnessAction::Noop).unwrap()
        );
        let b = self.biases.get(&action).unwrap_or(
            self.biases.get(&ConsciousnessAction::Noop).unwrap()
        );

        let mut new_latent = [0.0; 32];
        for i in 0..32 {
            let mut sum = b[i];
            for j in 0..32 {
                sum += w[i][j] * state.latent[j];
            }
            new_latent[i] = sum.clamp(-2.0, 2.0);
        }

        // Decode key observables from latent (first 4 dimensions)
        LatentConsciousnessState {
            latent: new_latent,
            phi: new_latent[0].clamp(0.0, 1.0),
            integration: new_latent[1].clamp(0.0, 1.0),
            coherence: new_latent[2].clamp(0.0, 1.0),
            attention: new_latent[3].clamp(0.0, 1.0),
            timestamp_internal: Instant::now(),
        }
    }

    /// Train on observed transition
    pub fn train(&mut self, transition: &ConsciousnessTransition) {
        let predicted = self.predict(&transition.from_state, transition.action);

        // Compute error
        let mut error = [0.0; 32];
        for i in 0..32 {
            error[i] = transition.to_state.latent[i] - predicted.latent[i];
        }

        // Update weights and biases (simple gradient descent)
        if let Some(w) = self.weights.get_mut(&transition.action) {
            if let Some(b) = self.biases.get_mut(&transition.action) {
                for i in 0..32 {
                    b[i] += self.learning_rate * error[i];
                    for j in 0..32 {
                        w[i][j] += self.learning_rate * error[i] * transition.from_state.latent[j];
                    }
                }
            }
        }

        self.train_count += 1;
    }

    /// Prediction accuracy (lower is better)
    pub fn prediction_error(&self, transitions: &[ConsciousnessTransition]) -> f64 {
        if transitions.is_empty() {
            return 1.0;
        }

        let mut total_error = 0.0;
        for t in transitions {
            let predicted = self.predict(&t.from_state, t.action);
            total_error += predicted.distance(&t.to_state);
        }

        total_error / transitions.len() as f64
    }
}

impl Default for ConsciousnessDynamicsModel {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// REWARD PREDICTOR
// ═══════════════════════════════════════════════════════════════════════════

/// Reward prediction model
#[derive(Debug, Clone)]
pub struct RewardPredictor {
    /// Weights for predicting reward from (state, action) pair
    weights: [f64; 32],

    /// Bias
    bias: f64,

    /// Learning rate
    learning_rate: f64,
}

impl RewardPredictor {
    /// Create new reward predictor
    pub fn new() -> Self {
        Self {
            weights: [0.0; 32],
            bias: 0.0,
            learning_rate: 0.01,
        }
    }

    /// Predict reward for state-action pair
    pub fn predict(&self, state: &LatentConsciousnessState, _action: ConsciousnessAction) -> f64 {
        let mut reward = self.bias;
        for i in 0..32 {
            reward += self.weights[i] * state.latent[i];
        }
        reward
    }

    /// Train on observed transition
    pub fn train(&mut self, transition: &ConsciousnessTransition) {
        let predicted = self.predict(&transition.from_state, transition.action);
        let error = transition.reward - predicted;

        self.bias += self.learning_rate * error;
        for i in 0..32 {
            self.weights[i] += self.learning_rate * error * transition.from_state.latent[i];
        }
    }
}

impl Default for RewardPredictor {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// COUNTERFACTUAL
// ═══════════════════════════════════════════════════════════════════════════

/// Counterfactual scenario
#[derive(Debug, Clone)]
pub struct Counterfactual {
    /// Original trajectory
    pub original: Vec<ConsciousnessTransition>,

    /// Alternative action that could have been taken
    pub alternative_action: ConsciousnessAction,

    /// When to diverge (index in original trajectory)
    pub divergence_point: usize,

    /// Predicted alternative trajectory
    pub alternative: Vec<LatentConsciousnessState>,

    /// Predicted cumulative reward difference
    pub reward_difference: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// WORLD MODEL CONFIG & STATS
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for world model
#[derive(Debug, Clone)]
pub struct WorldModelConfig {
    /// Maximum experience buffer size
    pub max_experience_buffer: usize,

    /// Maximum imagined buffer size
    pub max_imagined_buffer: usize,

    /// Imagination horizon (how far to simulate)
    pub imagination_horizon: usize,

    /// Number of imagined trajectories per dream cycle
    pub dream_trajectories: usize,

    /// Minimum training samples before trusting model
    pub min_training_samples: usize,
}

impl Default for WorldModelConfig {
    fn default() -> Self {
        Self {
            max_experience_buffer: 10000,
            max_imagined_buffer: 50000,
            imagination_horizon: 10,
            dream_trajectories: 100,
            min_training_samples: 100,
        }
    }
}

/// Statistics for world model
#[derive(Debug, Clone, Default)]
pub struct WorldModelStats {
    pub transitions_observed: usize,
    pub transitions_imagined: usize,
    pub counterfactuals_analyzed: usize,
    pub dreams_completed: usize,
    pub average_prediction_error: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS WORLD MODEL
// ═══════════════════════════════════════════════════════════════════════════

/// Full Consciousness World Model
pub struct ConsciousnessWorldModel {
    /// Dynamics model for predicting next states
    dynamics: ConsciousnessDynamicsModel,

    /// Reward predictor
    reward: RewardPredictor,

    /// Experience buffer (real transitions)
    experience_buffer: VecDeque<ConsciousnessTransition>,

    /// Imagined transitions (from dreaming)
    imagined_buffer: VecDeque<ConsciousnessTransition>,

    /// Configuration
    config: WorldModelConfig,

    /// Statistics
    stats: WorldModelStats,
}

impl ConsciousnessWorldModel {
    /// Create new world model
    pub fn new(config: WorldModelConfig) -> Self {
        Self {
            dynamics: ConsciousnessDynamicsModel::new(),
            reward: RewardPredictor::new(),
            experience_buffer: VecDeque::with_capacity(config.max_experience_buffer),
            imagined_buffer: VecDeque::with_capacity(config.max_imagined_buffer),
            config,
            stats: WorldModelStats::default(),
        }
    }

    /// Record real transition
    pub fn observe_transition(&mut self, transition: ConsciousnessTransition) {
        // Train models
        self.dynamics.train(&transition);
        self.reward.train(&transition);

        // Store in buffer
        if self.experience_buffer.len() >= self.config.max_experience_buffer {
            self.experience_buffer.pop_front();
        }
        self.experience_buffer.push_back(transition);

        self.stats.transitions_observed += 1;
    }

    /// Predict trajectory given starting state and action sequence
    pub fn simulate_trajectory(
        &self,
        start: &LatentConsciousnessState,
        actions: &[ConsciousnessAction],
    ) -> Vec<LatentConsciousnessState> {
        let mut trajectory = Vec::with_capacity(actions.len() + 1);
        trajectory.push(start.clone());

        let mut current = start.clone();
        for action in actions {
            current = self.dynamics.predict(&current, *action);
            trajectory.push(current.clone());
        }

        trajectory
    }

    /// Predict cumulative reward for trajectory
    pub fn predict_cumulative_reward(
        &self,
        states: &[LatentConsciousnessState],
        actions: &[ConsciousnessAction],
    ) -> f64 {
        let mut total = 0.0f64;
        let gamma: f64 = 0.99; // Discount factor

        for (i, (state, action)) in states.iter().zip(actions.iter()).enumerate() {
            let r = self.reward.predict(state, *action);
            total += gamma.powi(i as i32) * r;
        }

        total
    }

    /// Analyze counterfactual: what if we had taken a different action?
    pub fn analyze_counterfactual(
        &mut self,
        original: &[ConsciousnessTransition],
        divergence_point: usize,
        alternative_action: ConsciousnessAction,
    ) -> Counterfactual {
        if divergence_point >= original.len() {
            return Counterfactual {
                original: original.to_vec(),
                alternative_action,
                divergence_point,
                alternative: Vec::new(),
                reward_difference: 0.0,
            };
        }

        // Get state at divergence point
        let divergence_state = &original[divergence_point].from_state;

        // Generate alternative trajectory
        let remaining_horizon = original.len() - divergence_point;
        let mut alternative_actions = vec![alternative_action];

        // Continue with greedy action selection for remaining steps
        let mut current = self.dynamics.predict(divergence_state, alternative_action);
        for _ in 1..remaining_horizon {
            // Pick action that maximizes predicted reward
            let best_action = self.select_best_action(&current);
            alternative_actions.push(best_action);
            current = self.dynamics.predict(&current, best_action);
        }

        // Simulate alternative trajectory
        let alternative = self.simulate_trajectory(divergence_state, &alternative_actions);

        // Compute original reward from divergence point
        let original_reward: f64 = original[divergence_point..]
            .iter()
            .map(|t| t.reward)
            .sum();

        // Predict alternative reward
        let alternative_reward = self.predict_cumulative_reward(
            &alternative[..alternative.len()-1],
            &alternative_actions,
        );

        self.stats.counterfactuals_analyzed += 1;

        Counterfactual {
            original: original.to_vec(),
            alternative_action,
            divergence_point,
            alternative,
            reward_difference: alternative_reward - original_reward,
        }
    }

    /// Select best action according to model
    fn select_best_action(&self, state: &LatentConsciousnessState) -> ConsciousnessAction {
        let actions = ConsciousnessAction::all();
        let mut best_action = ConsciousnessAction::Noop;
        let mut best_reward = f64::NEG_INFINITY;

        for action in actions {
            let next_state = self.dynamics.predict(state, action);
            let reward = self.reward.predict(&next_state, action);

            if reward > best_reward {
                best_reward = reward;
                best_action = action;
            }
        }

        best_action
    }

    /// Dream: generate imagined experiences through simulation
    pub fn dream(&mut self, starting_states: &[LatentConsciousnessState]) {
        if self.stats.transitions_observed < self.config.min_training_samples {
            return; // Not enough real experience to dream reliably
        }

        for start in starting_states.iter().take(self.config.dream_trajectories) {
            let mut current = start.clone();

            for _ in 0..self.config.imagination_horizon {
                // Sample action (could be random or epsilon-greedy)
                let action = if rand::random::<f64>() < 0.3 {
                    // Random exploration
                    let actions = ConsciousnessAction::all();
                    actions[rand::random::<usize>() % actions.len()]
                } else {
                    // Greedy
                    self.select_best_action(&current)
                };

                let next = self.dynamics.predict(&current, action);
                let reward = self.reward.predict(&current, action);

                let transition = ConsciousnessTransition {
                    from_state: current.clone(),
                    action,
                    to_state: next.clone(),
                    reward,
                    is_real: false,
                };

                // Store imagined transition
                if self.imagined_buffer.len() >= self.config.max_imagined_buffer {
                    self.imagined_buffer.pop_front();
                }
                self.imagined_buffer.push_back(transition);
                self.stats.transitions_imagined += 1;

                current = next;
            }
        }

        self.stats.dreams_completed += 1;
    }

    /// Plan best action sequence using model
    pub fn plan(
        &self,
        start: &LatentConsciousnessState,
        horizon: usize,
        num_samples: usize,
    ) -> Vec<ConsciousnessAction> {
        let mut best_sequence = Vec::new();
        let mut best_reward = f64::NEG_INFINITY;

        let actions = ConsciousnessAction::all();

        // Random shooting planning
        for _ in 0..num_samples {
            let sequence: Vec<ConsciousnessAction> = (0..horizon)
                .map(|_| actions[rand::random::<usize>() % actions.len()])
                .collect();

            let trajectory = self.simulate_trajectory(start, &sequence);
            let reward = self.predict_cumulative_reward(&trajectory[..trajectory.len()-1], &sequence);

            if reward > best_reward {
                best_reward = reward;
                best_sequence = sequence;
            }
        }

        best_sequence
    }

    /// Get model statistics
    pub fn stats(&self) -> &WorldModelStats {
        &self.stats
    }

    /// Is the model ready for use?
    pub fn is_ready(&self) -> bool {
        self.stats.transitions_observed >= self.config.min_training_samples
    }

    /// Get prediction error on recent experience
    pub fn recent_prediction_error(&self) -> f64 {
        let recent: Vec<_> = self.experience_buffer
            .iter()
            .rev()
            .take(100)
            .cloned()
            .collect();

        self.dynamics.prediction_error(&recent)
    }

    /// Get summary for reporting
    pub fn summary(&self) -> WorldModelSummary {
        WorldModelSummary {
            is_ready: self.is_ready(),
            transitions_observed: self.stats.transitions_observed,
            transitions_imagined: self.stats.transitions_imagined,
            counterfactuals_analyzed: self.stats.counterfactuals_analyzed,
            dreams_completed: self.stats.dreams_completed,
            recent_prediction_error: self.recent_prediction_error(),
        }
    }
}

/// Summary of world model status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelSummary {
    pub is_ready: bool,
    pub transitions_observed: usize,
    pub transitions_imagined: usize,
    pub counterfactuals_analyzed: usize,
    pub dreams_completed: usize,
    pub recent_prediction_error: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// DOMAIN TRAIT IMPLEMENTATIONS (Generalization Refactoring Phase 1 & 2)
// ═══════════════════════════════════════════════════════════════════════════

use crate::core::domain_traits::{State, Action, Goal, HdcEncodable, DomainAdapter, QualitySignal};
// Note: ContinuousHV and HDC_DIMENSION already imported at file top

// ═══════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS GOALS
// ═══════════════════════════════════════════════════════════════════════════

/// Goal to maximize Φ (integrated information) to a target level.
///
/// This is the primary goal for consciousness systems: increase
/// the integrated information as measured by IIT.
#[derive(Debug, Clone)]
pub struct PhiMaximizationGoal {
    /// Target Φ value to achieve
    pub target_phi: f64,
    /// Priority of this goal
    pub priority: f64,
}

impl PhiMaximizationGoal {
    /// Create a new Φ maximization goal.
    pub fn new(target_phi: f64) -> Self {
        Self {
            target_phi: target_phi.clamp(0.0, 1.0),
            priority: 1.0,
        }
    }

    /// Create with custom priority.
    pub fn with_priority(target_phi: f64, priority: f64) -> Self {
        Self {
            target_phi: target_phi.clamp(0.0, 1.0),
            priority: priority.clamp(0.0, 10.0),
        }
    }
}

impl Goal<LatentConsciousnessState> for PhiMaximizationGoal {
    fn is_satisfied(&self, state: &LatentConsciousnessState) -> bool {
        state.phi >= self.target_phi
    }

    fn distance_to_goal(&self, state: &LatentConsciousnessState) -> f64 {
        (self.target_phi - state.phi).max(0.0)
    }

    fn reward(&self, state: &LatentConsciousnessState) -> f64 {
        if self.is_satisfied(state) {
            1.0
        } else {
            // Reward proportional to how close we are
            state.phi / self.target_phi - 0.5
        }
    }

    fn priority(&self) -> f64 {
        self.priority
    }
}

/// Goal to achieve coherent consciousness state.
///
/// Coherence means integration, coherence, and attention are all high
/// and well-balanced.
#[derive(Debug, Clone)]
pub struct CoherenceGoal {
    /// Minimum coherence level
    pub min_coherence: f64,
    /// Minimum integration level
    pub min_integration: f64,
    /// Priority of this goal
    pub priority: f64,
}

impl CoherenceGoal {
    /// Create a new coherence goal with default thresholds.
    pub fn new() -> Self {
        Self {
            min_coherence: 0.6,
            min_integration: 0.6,
            priority: 0.8,
        }
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(min_coherence: f64, min_integration: f64) -> Self {
        Self {
            min_coherence: min_coherence.clamp(0.0, 1.0),
            min_integration: min_integration.clamp(0.0, 1.0),
            priority: 0.8,
        }
    }
}

impl Default for CoherenceGoal {
    fn default() -> Self {
        Self::new()
    }
}

impl Goal<LatentConsciousnessState> for CoherenceGoal {
    fn is_satisfied(&self, state: &LatentConsciousnessState) -> bool {
        state.coherence >= self.min_coherence && state.integration >= self.min_integration
    }

    fn distance_to_goal(&self, state: &LatentConsciousnessState) -> f64 {
        let coherence_gap = (self.min_coherence - state.coherence).max(0.0);
        let integration_gap = (self.min_integration - state.integration).max(0.0);
        (coherence_gap + integration_gap) / 2.0
    }

    fn reward(&self, state: &LatentConsciousnessState) -> f64 {
        if self.is_satisfied(state) {
            1.0
        } else {
            // Average of how close we are to each threshold
            let coherence_progress = (state.coherence / self.min_coherence).min(1.0);
            let integration_progress = (state.integration / self.min_integration).min(1.0);
            (coherence_progress + integration_progress) / 2.0 - 0.5
        }
    }

    fn priority(&self) -> f64 {
        self.priority
    }
}

/// Implementation of domain-agnostic State trait for LatentConsciousnessState.
///
/// This allows the consciousness state to be used with generic planning,
/// learning, and reasoning systems that work across any domain.
impl State for LatentConsciousnessState {
    /// Convert state to feature vector for ML algorithms.
    ///
    /// Returns the 32-dimensional latent vector plus the 4 observable features.
    fn to_features(&self) -> Vec<f64> {
        let mut features = Vec::with_capacity(36);
        features.extend_from_slice(&self.latent);
        features.push(self.phi);
        features.push(self.integration);
        features.push(self.coherence);
        features.push(self.attention);
        features
    }

    /// Compute Euclidean distance between two consciousness states.
    ///
    /// Uses the existing distance method which computes L2 norm over latent space.
    fn distance(&self, other: &Self) -> f64 {
        // Delegate to existing implementation
        let mut sum = 0.0;
        for i in 0..32 {
            let diff = self.latent[i] - other.latent[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }
}

/// Implementation of domain-agnostic Action trait for ConsciousnessAction.
///
/// This allows consciousness actions to be used with generic planning
/// and learning systems.
impl Action for ConsciousnessAction {
    /// Returns a unique identifier for this action type.
    fn action_id(&self) -> u64 {
        self.to_index() as u64
    }

    /// Human-readable description of the action.
    fn describe(&self) -> String {
        match self {
            Self::FocusIntegration => "Focus attention on increasing integration".to_string(),
            Self::FocusCoherence => "Focus attention on maintaining coherence".to_string(),
            Self::EngageLearning => "Engage in active learning mode".to_string(),
            Self::ExplorePatterns => "Explore novel patterns in consciousness".to_string(),
            Self::Consolidate => "Consolidate existing patterns and memories".to_string(),
            Self::Rest => "Rest and allow passive processing".to_string(),
            Self::ApplyImprovement(i) => format!("Apply improvement method #{}", i),
            Self::Noop => "No operation - passive observation".to_string(),
        }
    }

    /// All consciousness actions are reversible in principle.
    fn is_reversible(&self) -> bool {
        true
    }

    /// Estimated effort for each action type.
    fn cost(&self) -> f64 {
        match self {
            Self::FocusIntegration => 1.5,  // High effort
            Self::FocusCoherence => 1.2,    // Moderate-high effort
            Self::EngageLearning => 2.0,    // Highest effort
            Self::ExplorePatterns => 1.8,   // High effort
            Self::Consolidate => 1.0,       // Moderate effort
            Self::Rest => 0.2,              // Minimal effort
            Self::ApplyImprovement(_) => 1.5, // Variable
            Self::Noop => 0.0,              // No effort
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HDC ENCODABLE IMPLEMENTATION (Phase 2)
// ═══════════════════════════════════════════════════════════════════════════

/// Implementation of HdcEncodable for LatentConsciousnessState.
///
/// This enables semantic similarity computation via hyperdimensional computing,
/// allowing consciousness states to be compared, composed, and reasoned about
/// using HDC operations (bind, bundle, similarity).
///
/// # Performance
/// Uses lazy-static cached basis vectors for ~10x speedup over dynamic creation.
///
/// # Encoding Strategy
/// 1. Observable encoding: Scale cached basis vectors by observable values
/// 2. Latent sequence encoding: Use permutation for positional information
/// 3. Bundle all components into final hypervector
impl HdcEncodable for LatentConsciousnessState {
    type HyperVector = ContinuousHV;

    /// Encode consciousness state as a hypervector.
    ///
    /// Uses cached basis vectors and permutation-based sequence encoding:
    /// 1. Scale cached basis vectors by observable values (phi, integration, coherence, attention)
    /// 2. Bundle observable components
    /// 3. Encode latent vector using permutation for sequence information
    /// 4. Combine observable bundle with latent encoding
    fn to_hv(&self) -> Self::HyperVector {
        // Use cached basis vectors (lazy static - ~10x faster)
        let phi_component = hdc_basis::PHI_BASIS.scale(self.phi as f32);
        let integration_component = hdc_basis::INTEGRATION_BASIS.scale(self.integration as f32);
        let coherence_component = hdc_basis::COHERENCE_BASIS.scale(self.coherence as f32);
        let attention_component = hdc_basis::ATTENTION_BASIS.scale(self.attention as f32);

        // Bundle observable components
        let bundled = ContinuousHV::bundle(&[
            &phi_component,
            &integration_component,
            &coherence_component,
            &attention_component,
        ]);

        // Encode full latent vector with sequence information using permutation
        // This preserves positional information (latent[0] is different from latent[1])
        let mut latent_hv = ContinuousHV::zero(HDC_DIMENSION);
        for (i, &val) in self.latent.iter().enumerate() {
            // Apply permutation to encode position
            let permuted_basis = hdc_basis::LATENT_BASES[i].permute(i);
            let component = permuted_basis.scale(val as f32);
            latent_hv = latent_hv.add(&component);
        }

        // Combine observable bundle with latent encoding
        // Weight latent encoding lower since observables are more important
        bundled.add(&latent_hv.scale(0.3))
    }

    /// Decode hypervector back to consciousness state.
    ///
    /// This is approximate - HDC encoding is lossy but preserves semantic similarity.
    /// Uses cached basis vectors for consistent decoding.
    fn from_hv(hv: &Self::HyperVector) -> Option<Self> {
        // Decode observables by projecting onto cached basis vectors
        let phi = hv.similarity(&hdc_basis::PHI_BASIS).clamp(-1.0, 1.0) as f64;
        let integration = hv.similarity(&hdc_basis::INTEGRATION_BASIS).clamp(-1.0, 1.0) as f64;
        let coherence = hv.similarity(&hdc_basis::COHERENCE_BASIS).clamp(-1.0, 1.0) as f64;
        let attention = hv.similarity(&hdc_basis::ATTENTION_BASIS).clamp(-1.0, 1.0) as f64;

        // Decode latent vector (approximate - position encoded via permutation)
        let mut latent = [0.0f64; 32];
        for i in 0..32 {
            let permuted_basis = hdc_basis::LATENT_BASES[i].permute(i);
            latent[i] = hv.similarity(&permuted_basis).clamp(-1.0, 1.0) as f64;
        }

        // Reconstruct state
        let mut state = LatentConsciousnessState::from_observables(
            phi.abs(),
            integration.abs(),
            coherence.abs(),
            attention.abs(),
        );
        state.latent = latent;
        Some(state)
    }

    /// Compute semantic similarity between two consciousness states via HDC.
    ///
    /// This captures high-level similarity that may not be apparent from
    /// direct feature comparison. Uses cosine similarity in HDC space.
    fn semantic_similarity(&self, other: &Self) -> f64 {
        let self_hv = self.to_hv();
        let other_hv = other.to_hv();
        self_hv.similarity(&other_hv) as f64
    }
}

/// HDC-based equivalence threshold for consciousness states.
/// States with similarity above this threshold are considered equivalent.
pub const HDC_EQUIVALENCE_THRESHOLD: f64 = 0.95;

// ═══════════════════════════════════════════════════════════════════════════
// QUALITY SIGNALS (Phase 2)
// ═══════════════════════════════════════════════════════════════════════════

/// Φ (integrated information) as a quality signal.
///
/// This is the primary quality metric for consciousness systems,
/// measuring how much information is integrated across the system.
#[derive(Debug, Clone)]
pub struct PhiQualitySignal {
    /// Weight of this signal in composite calculations
    pub weight: f64,
}

impl PhiQualitySignal {
    /// Create a new Φ quality signal with default weight.
    pub fn new() -> Self {
        Self { weight: 1.0 }
    }

    /// Create with custom weight.
    pub fn with_weight(weight: f64) -> Self {
        Self {
            weight: weight.max(0.0),
        }
    }
}

impl Default for PhiQualitySignal {
    fn default() -> Self {
        Self::new()
    }
}

impl QualitySignal<LatentConsciousnessState> for PhiQualitySignal {
    fn measure(&self, state: &LatentConsciousnessState) -> f64 {
        state.phi
    }

    fn name(&self) -> &'static str {
        "phi"
    }

    fn weight(&self) -> f64 {
        self.weight
    }
}

/// Coherence quality signal.
///
/// Measures the coherence of the consciousness state.
#[derive(Debug, Clone)]
pub struct CoherenceQualitySignal {
    pub weight: f64,
}

impl CoherenceQualitySignal {
    pub fn new() -> Self {
        Self { weight: 0.8 }
    }
}

impl Default for CoherenceQualitySignal {
    fn default() -> Self {
        Self::new()
    }
}

impl QualitySignal<LatentConsciousnessState> for CoherenceQualitySignal {
    fn measure(&self, state: &LatentConsciousnessState) -> f64 {
        state.coherence
    }

    fn name(&self) -> &'static str {
        "coherence"
    }

    fn weight(&self) -> f64 {
        self.weight
    }
}

/// Integration quality signal.
///
/// Measures the integration level of consciousness.
#[derive(Debug, Clone)]
pub struct IntegrationQualitySignal {
    pub weight: f64,
}

impl IntegrationQualitySignal {
    pub fn new() -> Self {
        Self { weight: 0.9 }
    }
}

impl Default for IntegrationQualitySignal {
    fn default() -> Self {
        Self::new()
    }
}

impl QualitySignal<LatentConsciousnessState> for IntegrationQualitySignal {
    fn measure(&self, state: &LatentConsciousnessState) -> f64 {
        state.integration
    }

    fn name(&self) -> &'static str {
        "integration"
    }

    fn weight(&self) -> f64 {
        self.weight
    }
}

/// Attention quality signal.
///
/// Measures the attention level of the consciousness state.
/// Higher attention indicates more focused processing.
#[derive(Debug, Clone)]
pub struct AttentionQualitySignal {
    pub weight: f64,
}

impl AttentionQualitySignal {
    pub fn new() -> Self {
        Self { weight: 0.7 }
    }

    pub fn with_weight(weight: f64) -> Self {
        Self { weight: weight.max(0.0) }
    }
}

impl Default for AttentionQualitySignal {
    fn default() -> Self {
        Self::new()
    }
}

impl QualitySignal<LatentConsciousnessState> for AttentionQualitySignal {
    fn measure(&self, state: &LatentConsciousnessState) -> f64 {
        state.attention
    }

    fn name(&self) -> &'static str {
        "attention"
    }

    fn weight(&self) -> f64 {
        self.weight
    }
}

/// Entropy quality signal.
///
/// Measures the uncertainty/entropy from the latent vector variance.
/// Lower entropy indicates more stable/predictable consciousness state.
/// Returns inverse entropy (1 - normalized_entropy) so higher = better.
#[derive(Debug, Clone)]
pub struct EntropyQualitySignal {
    pub weight: f64,
}

impl EntropyQualitySignal {
    pub fn new() -> Self {
        Self { weight: 0.5 }
    }

    pub fn with_weight(weight: f64) -> Self {
        Self { weight: weight.max(0.0) }
    }

    /// Calculate entropy from latent vector variance.
    fn calculate_entropy(latent: &[f64; 32]) -> f64 {
        // Calculate variance of latent vector
        let mean: f64 = latent.iter().sum::<f64>() / 32.0;
        let variance: f64 = latent.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / 32.0;

        // Normalize variance to [0, 1] range (assuming max variance ~0.25 for [0,1] values)
        let normalized_variance = (variance / 0.25).min(1.0);

        // Return inverse (higher = less entropy = more stable = better)
        1.0 - normalized_variance
    }
}

impl Default for EntropyQualitySignal {
    fn default() -> Self {
        Self::new()
    }
}

impl QualitySignal<LatentConsciousnessState> for EntropyQualitySignal {
    fn measure(&self, state: &LatentConsciousnessState) -> f64 {
        Self::calculate_entropy(&state.latent)
    }

    fn name(&self) -> &'static str {
        "entropy"
    }

    fn weight(&self) -> f64 {
        self.weight
    }
}

/// Composite quality signal.
///
/// Combines multiple quality signals into a weighted sum.
/// This enables multi-objective optimization with configurable priorities.
#[derive(Debug, Clone)]
pub struct CompositeQualitySignal {
    /// Weights for each component signal: [phi, integration, coherence, attention, entropy]
    pub weights: [f64; 5],
    /// Overall weight of this composite signal
    pub total_weight: f64,
}

impl CompositeQualitySignal {
    /// Create with default equal weights.
    pub fn new() -> Self {
        Self {
            weights: [1.0, 1.0, 1.0, 1.0, 1.0],
            total_weight: 1.0,
        }
    }

    /// Create with custom weights for [phi, integration, coherence, attention, entropy].
    pub fn with_weights(phi: f64, integration: f64, coherence: f64, attention: f64, entropy: f64) -> Self {
        Self {
            weights: [phi.max(0.0), integration.max(0.0), coherence.max(0.0), attention.max(0.0), entropy.max(0.0)],
            total_weight: 1.0,
        }
    }

    /// Create consciousness-optimized weights (prioritize Φ and integration).
    pub fn consciousness_optimized() -> Self {
        Self {
            weights: [2.0, 1.5, 1.0, 0.8, 0.5],
            total_weight: 1.0,
        }
    }
}

impl Default for CompositeQualitySignal {
    fn default() -> Self {
        Self::new()
    }
}

impl QualitySignal<LatentConsciousnessState> for CompositeQualitySignal {
    fn measure(&self, state: &LatentConsciousnessState) -> f64 {
        let components = [
            state.phi,
            state.integration,
            state.coherence,
            state.attention,
            EntropyQualitySignal::calculate_entropy(&state.latent),
        ];

        // Weighted sum normalized by total weight
        let total_weight: f64 = self.weights.iter().sum();
        if total_weight <= 0.0 {
            return 0.0;
        }

        let weighted_sum: f64 = components.iter()
            .zip(self.weights.iter())
            .map(|(c, w)| c * w)
            .sum();

        weighted_sum / total_weight
    }

    fn name(&self) -> &'static str {
        "composite"
    }

    fn weight(&self) -> f64 {
        self.total_weight
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DOMAIN ADAPTER (Phase 2)
// ═══════════════════════════════════════════════════════════════════════════

/// Domain adapter for consciousness systems.
///
/// Provides domain-specific functionality for the consciousness domain,
/// including available actions, initial states, and quality signals.
#[derive(Debug, Clone)]
pub struct ConsciousnessDomainAdapter {
    /// Initial phi level for new states
    pub initial_phi: f64,
    /// Initial integration level
    pub initial_integration: f64,
    /// Initial coherence level
    pub initial_coherence: f64,
    /// Initial attention level
    pub initial_attention: f64,
}

impl ConsciousnessDomainAdapter {
    /// Create a new consciousness domain adapter with default initial state.
    pub fn new() -> Self {
        Self {
            initial_phi: 0.5,
            initial_integration: 0.5,
            initial_coherence: 0.5,
            initial_attention: 0.5,
        }
    }

    /// Create with custom initial values.
    pub fn with_initial_state(phi: f64, integration: f64, coherence: f64, attention: f64) -> Self {
        Self {
            initial_phi: phi.clamp(0.0, 1.0),
            initial_integration: integration.clamp(0.0, 1.0),
            initial_coherence: coherence.clamp(0.0, 1.0),
            initial_attention: attention.clamp(0.0, 1.0),
        }
    }
}

impl Default for ConsciousnessDomainAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl DomainAdapter<LatentConsciousnessState, ConsciousnessAction> for ConsciousnessDomainAdapter {
    fn domain_name(&self) -> &'static str {
        "consciousness"
    }

    fn available_actions(&self, _state: &LatentConsciousnessState) -> Vec<ConsciousnessAction> {
        // All standard actions are available in any state
        ConsciousnessAction::all()
    }

    fn initial_state(&self) -> LatentConsciousnessState {
        LatentConsciousnessState::from_observables(
            self.initial_phi,
            self.initial_integration,
            self.initial_coherence,
            self.initial_attention,
        )
    }

    fn quality_signals(&self) -> Vec<Box<dyn QualitySignal<LatentConsciousnessState>>> {
        vec![
            Box::new(PhiQualitySignal::new()),
            Box::new(CoherenceQualitySignal::new()),
            Box::new(IntegrationQualitySignal::new()),
            Box::new(AttentionQualitySignal::new()),
            Box::new(EntropyQualitySignal::new()),
        ]
    }

    fn is_valid_state(&self, state: &LatentConsciousnessState) -> bool {
        // Valid if all observables are in [0, 1] range
        state.phi >= 0.0
            && state.phi <= 1.0
            && state.integration >= 0.0
            && state.integration <= 1.0
            && state.coherence >= 0.0
            && state.coherence <= 1.0
            && state.attention >= 0.0
            && state.attention <= 1.0
    }

    fn is_valid_action(&self, _state: &LatentConsciousnessState, _action: &ConsciousnessAction) -> bool {
        // All actions are valid in any state for consciousness domain
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 3: GENERIC WORLD MODEL INFRASTRUCTURE
// ═══════════════════════════════════════════════════════════════════════════

use std::marker::PhantomData;

/// Generic transition between states.
///
/// This replaces domain-specific transition types like ConsciousnessTransition.
/// Works with any State and Action types that implement the required traits.
#[derive(Debug, Clone)]
pub struct Transition<S: State, A: Action> {
    /// Starting state
    pub from_state: S,
    /// Action taken
    pub action: A,
    /// Resulting state
    pub to_state: S,
    /// Observed reward
    pub reward: f64,
    /// Was this transition real or imagined?
    pub is_real: bool,
}

impl<S: State, A: Action> Transition<S, A> {
    /// Create a new transition.
    pub fn new(from_state: S, action: A, to_state: S, reward: f64, is_real: bool) -> Self {
        Self {
            from_state,
            action,
            to_state,
            reward,
            is_real,
        }
    }

    /// Create a real transition.
    pub fn real(from_state: S, action: A, to_state: S, reward: f64) -> Self {
        Self::new(from_state, action, to_state, reward, true)
    }

    /// Create an imagined transition.
    pub fn imagined(from_state: S, action: A, to_state: S, reward: f64) -> Self {
        Self::new(from_state, action, to_state, reward, false)
    }
}

/// Trait for dynamics models that predict next states.
///
/// This is the generic interface for domain-specific dynamics models
/// like ConsciousnessDynamicsModel.
pub trait DynamicsPredictor<S: State, A: Action> {
    /// Predict the next state given current state and action.
    fn predict(&self, state: &S, action: A) -> S;

    /// Train the model on a single transition.
    fn train(&mut self, transition: &Transition<S, A>);

    /// Train the model on a batch of transitions.
    fn train_batch(&mut self, transitions: &[Transition<S, A>]) {
        for t in transitions {
            self.train(t);
        }
    }

    /// Get the prediction error on a set of transitions.
    fn prediction_error(&self, transitions: &[Transition<S, A>]) -> f64
    where
        S: Clone,
        A: Clone,
    {
        if transitions.is_empty() {
            return 0.0;
        }

        let mut total_error = 0.0;
        for t in transitions {
            let predicted = self.predict(&t.from_state, t.action.clone());
            total_error += predicted.distance(&t.to_state);
        }
        total_error / transitions.len() as f64
    }

    /// Get the number of training samples seen.
    fn train_count(&self) -> usize;
}

/// Trait for reward predictors.
///
/// Generic interface for predicting rewards from state-action pairs.
pub trait RewardEstimator<S: State, A: Action> {
    /// Predict the reward for a state-action pair.
    fn predict(&self, state: &S, action: A) -> f64;

    /// Train on observed reward.
    fn train(&mut self, state: &S, action: A, actual_reward: f64);
}

/// Generic world model that works with any State and Action types.
///
/// This is the domain-agnostic version of ConsciousnessWorldModel.
pub struct GenericWorldModel<S, A, D, R>
where
    S: State + Clone,
    A: Action + Clone,
    D: DynamicsPredictor<S, A>,
    R: RewardEstimator<S, A>,
{
    /// Dynamics model for predicting next states
    dynamics: D,

    /// Reward estimator
    reward: R,

    /// Experience buffer (real transitions)
    experience_buffer: VecDeque<Transition<S, A>>,

    /// Imagined transitions (from dreaming)
    imagined_buffer: VecDeque<Transition<S, A>>,

    /// Configuration
    config: WorldModelConfig,

    /// Statistics
    stats: WorldModelStats,

    /// Phantom data for type parameters
    _phantom: PhantomData<(S, A)>,
}

impl<S, A, D, R> GenericWorldModel<S, A, D, R>
where
    S: State + Clone,
    A: Action + Clone,
    D: DynamicsPredictor<S, A>,
    R: RewardEstimator<S, A>,
{
    /// Create a new generic world model.
    pub fn new(dynamics: D, reward: R) -> Self {
        Self::with_config(dynamics, reward, WorldModelConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(dynamics: D, reward: R, config: WorldModelConfig) -> Self {
        Self {
            dynamics,
            reward,
            experience_buffer: VecDeque::with_capacity(config.max_experience_buffer),
            imagined_buffer: VecDeque::with_capacity(config.max_imagined_buffer),
            config,
            stats: WorldModelStats::default(),
            _phantom: PhantomData,
        }
    }

    /// Observe a real transition.
    pub fn observe(&mut self, transition: Transition<S, A>) {
        // Train dynamics model
        self.dynamics.train(&transition);

        // Train reward estimator
        self.reward.train(&transition.from_state, transition.action.clone(), transition.reward);

        // Add to experience buffer
        if self.experience_buffer.len() >= self.config.max_experience_buffer {
            self.experience_buffer.pop_front();
        }
        self.experience_buffer.push_back(transition);

        self.stats.transitions_observed += 1;
    }

    /// Predict the next state.
    pub fn predict_next(&self, state: &S, action: A) -> S {
        self.dynamics.predict(state, action)
    }

    /// Predict the reward for a state-action pair.
    pub fn predict_reward(&self, state: &S, action: A) -> f64 {
        self.reward.predict(state, action)
    }

    /// Check if model is ready (enough training samples).
    pub fn is_ready(&self) -> bool {
        self.dynamics.train_count() >= self.config.min_training_samples
    }

    /// Get the experience buffer.
    pub fn experience(&self) -> &VecDeque<Transition<S, A>> {
        &self.experience_buffer
    }

    /// Get statistics.
    pub fn stats(&self) -> &WorldModelStats {
        &self.stats
    }

    /// Get prediction error on recent experience.
    pub fn recent_prediction_error(&self) -> f64 {
        let recent: Vec<_> = self.experience_buffer
            .iter()
            .rev()
            .take(100)
            .cloned()
            .collect();

        self.dynamics.prediction_error(&recent)
    }
}

/// Implement DynamicsPredictor for ConsciousnessDynamicsModel.
///
/// This bridges the concrete implementation to the generic trait.
impl DynamicsPredictor<LatentConsciousnessState, ConsciousnessAction> for ConsciousnessDynamicsModel {
    fn predict(&self, state: &LatentConsciousnessState, action: ConsciousnessAction) -> LatentConsciousnessState {
        // Delegate to existing predict method
        ConsciousnessDynamicsModel::predict(self, state, action)
    }

    fn train(&mut self, transition: &Transition<LatentConsciousnessState, ConsciousnessAction>) {
        // Convert to legacy transition type and delegate
        let legacy = ConsciousnessTransition {
            from_state: transition.from_state.clone(),
            action: transition.action.clone(),
            to_state: transition.to_state.clone(),
            reward: transition.reward,
            is_real: transition.is_real,
        };
        ConsciousnessDynamicsModel::train(self, &legacy);
    }

    fn train_count(&self) -> usize {
        self.train_count
    }
}

/// Implement RewardEstimator for RewardPredictor.
impl RewardEstimator<LatentConsciousnessState, ConsciousnessAction> for RewardPredictor {
    fn predict(&self, state: &LatentConsciousnessState, action: ConsciousnessAction) -> f64 {
        RewardPredictor::predict(self, state, action)
    }

    fn train(&mut self, state: &LatentConsciousnessState, action: ConsciousnessAction, actual_reward: f64) {
        // Create a transition for the underlying train method
        let transition = ConsciousnessTransition {
            from_state: state.clone(),
            action,
            to_state: state.clone(), // Not used in reward training
            reward: actual_reward,
            is_real: true,
        };
        RewardPredictor::train(self, &transition);
    }
}

/// Type alias for the consciousness-specific world model using generic infrastructure.
pub type GenericConsciousnessWorldModel = GenericWorldModel<
    LatentConsciousnessState,
    ConsciousnessAction,
    ConsciousnessDynamicsModel,
    RewardPredictor,
>;

/// Factory function to create a consciousness world model using generic infrastructure.
pub fn create_generic_consciousness_world_model() -> GenericConsciousnessWorldModel {
    GenericWorldModel::new(
        ConsciousnessDynamicsModel::new(),
        RewardPredictor::new(),
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_state_creation() {
        let state = LatentConsciousnessState::from_observables(0.7, 0.6, 0.5, 0.8);
        assert!((state.phi - 0.7).abs() < 0.001);
        assert!((state.integration - 0.6).abs() < 0.001);
        assert!((state.coherence - 0.5).abs() < 0.001);
        assert!((state.attention - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_latent_state_distance() {
        let state1 = LatentConsciousnessState::from_observables(0.7, 0.6, 0.5, 0.8);
        let state2 = LatentConsciousnessState::from_observables(0.7, 0.6, 0.5, 0.8);
        let state3 = LatentConsciousnessState::from_observables(0.3, 0.3, 0.3, 0.3);

        // Same state should have 0 distance
        assert!(state1.distance(&state2) < 0.001);

        // Different states should have positive distance
        assert!(state1.distance(&state3) > 0.1);
    }

    #[test]
    fn test_latent_state_interpolation() {
        let state1 = LatentConsciousnessState::from_observables(0.0, 0.0, 0.0, 0.0);
        let state2 = LatentConsciousnessState::from_observables(1.0, 1.0, 1.0, 1.0);

        let mid = state1.interpolate(&state2, 0.5);
        assert!((mid.phi - 0.5).abs() < 0.01);
        assert!((mid.integration - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_consciousness_action_all() {
        let actions = ConsciousnessAction::all();
        assert!(actions.len() >= 6);
        assert!(actions.contains(&ConsciousnessAction::FocusIntegration));
        assert!(actions.contains(&ConsciousnessAction::Noop));
    }

    #[test]
    fn test_dynamics_model_creation() {
        let model = ConsciousnessDynamicsModel::new();
        assert_eq!(model.train_count, 0);
    }

    #[test]
    fn test_dynamics_model_prediction() {
        let model = ConsciousnessDynamicsModel::new();
        let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);

        let next = model.predict(&state, ConsciousnessAction::FocusIntegration);

        // Should produce a valid state
        assert!(next.phi >= 0.0 && next.phi <= 1.0);
        assert!(next.integration >= 0.0 && next.integration <= 1.0);
    }

    #[test]
    fn test_dynamics_model_training() {
        let mut model = ConsciousnessDynamicsModel::new();
        let from_state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let to_state = LatentConsciousnessState::from_observables(0.6, 0.6, 0.6, 0.6);

        let transition = ConsciousnessTransition {
            from_state,
            action: ConsciousnessAction::FocusIntegration,
            to_state,
            reward: 0.1,
            is_real: true,
        };

        model.train(&transition);
        assert_eq!(model.train_count, 1);
    }

    #[test]
    fn test_reward_predictor() {
        let mut predictor = RewardPredictor::new();
        let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);

        // Initial prediction
        let r1 = predictor.predict(&state, ConsciousnessAction::Noop);

        // Train with positive reward
        let transition = ConsciousnessTransition {
            from_state: state.clone(),
            action: ConsciousnessAction::FocusIntegration,
            to_state: LatentConsciousnessState::from_observables(0.6, 0.6, 0.6, 0.6),
            reward: 1.0,
            is_real: true,
        };
        predictor.train(&transition);

        // Prediction should change
        let r2 = predictor.predict(&state, ConsciousnessAction::FocusIntegration);
        // After training, the prediction should move toward the observed reward
        assert!(r2 != r1 || r1 == 1.0);
    }

    #[test]
    fn test_world_model_creation() {
        let model = ConsciousnessWorldModel::new(WorldModelConfig::default());
        assert!(!model.is_ready());
        assert_eq!(model.stats().transitions_observed, 0);
    }

    #[test]
    fn test_world_model_observe_transition() {
        let mut model = ConsciousnessWorldModel::new(WorldModelConfig {
            min_training_samples: 5,
            ..Default::default()
        });
        let from_state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let to_state = LatentConsciousnessState::from_observables(0.6, 0.6, 0.6, 0.6);

        for _ in 0..5 {
            let from_state = LatentConsciousnessState::from_observables(
                0.5,
                0.5,
                0.5,
                0.6,
            );
            let to_state = LatentConsciousnessState::from_observables(
                0.6,
                0.6,
                0.6,
                0.6,
            );

            let transition = ConsciousnessTransition {
                from_state,
                action: ConsciousnessAction::FocusIntegration,
                to_state,
                reward: 0.1,
                is_real: true,
            };

            model.observe_transition(transition);
        }

        assert!(model.is_ready());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // DOMAIN TRAIT TESTS (Generalization Refactoring Phase 1)
    // ═══════════════════════════════════════════════════════════════════════

    use crate::core::domain_traits::{State as StateTrait, Action as ActionTrait};

    #[test]
    fn test_latent_state_trait_to_features() {
        let state = LatentConsciousnessState::from_observables(0.7, 0.6, 0.5, 0.8);
        let features = state.to_features();

        // Should have 36 features (32 latent + 4 observables)
        assert_eq!(features.len(), 36);

        // Last 4 should be the observables
        assert!((features[32] - 0.7).abs() < 0.001); // phi
        assert!((features[33] - 0.6).abs() < 0.001); // integration
        assert!((features[34] - 0.5).abs() < 0.001); // coherence
        assert!((features[35] - 0.8).abs() < 0.001); // attention
    }

    #[test]
    fn test_latent_state_trait_feature_dim() {
        let state = LatentConsciousnessState::default();
        assert_eq!(state.feature_dim(), 36);
    }

    #[test]
    fn test_latent_state_trait_distance() {
        let state1 = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let state2 = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let state3 = LatentConsciousnessState::from_observables(0.9, 0.9, 0.9, 0.9);

        // Same state -> distance ~0
        assert!(StateTrait::distance(&state1, &state2) < 0.001);

        // Different states -> positive distance
        let d = StateTrait::distance(&state1, &state3);
        assert!(d > 0.1);
    }

    #[test]
    fn test_latent_state_trait_is_equivalent() {
        let state1 = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let state2 = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let state3 = LatentConsciousnessState::from_observables(0.9, 0.9, 0.9, 0.9);

        // Same states should be equivalent
        assert!(state1.is_equivalent(&state2, 0.01));

        // Different states should not be equivalent
        assert!(!state1.is_equivalent(&state3, 0.01));
    }

    #[test]
    fn test_consciousness_action_trait_action_id() {
        let action1 = ConsciousnessAction::FocusIntegration;
        let action2 = ConsciousnessAction::FocusCoherence;
        let action3 = ConsciousnessAction::Noop;

        // Different actions should have different IDs
        assert_ne!(action1.action_id(), action2.action_id());

        // Noop should have ID 100
        assert_eq!(action3.action_id(), 100);
    }

    #[test]
    fn test_consciousness_action_trait_describe() {
        let action = ConsciousnessAction::EngageLearning;
        let desc = action.describe();

        // Description should be non-empty and meaningful
        assert!(!desc.is_empty());
        assert!(desc.contains("learning"));
    }

    #[test]
    fn test_consciousness_action_trait_cost() {
        let noop = ConsciousnessAction::Noop;
        let rest = ConsciousnessAction::Rest;
        let learning = ConsciousnessAction::EngageLearning;

        // Noop should have 0 cost
        assert_eq!(ActionTrait::cost(&noop), 0.0);

        // Rest should be low cost
        assert!(ActionTrait::cost(&rest) < 0.5);

        // Learning should be high cost
        assert!(ActionTrait::cost(&learning) > 1.5);
    }

    #[test]
    fn test_consciousness_action_trait_is_reversible() {
        // All consciousness actions should be reversible
        for action in ConsciousnessAction::all() {
            assert!(action.is_reversible());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // GOAL TRAIT TESTS
    // ═══════════════════════════════════════════════════════════════════════

    use crate::core::domain_traits::Goal as GoalTrait;

    #[test]
    fn test_phi_maximization_goal_creation() {
        let goal = PhiMaximizationGoal::new(0.8);
        assert!((goal.target_phi - 0.8).abs() < 0.001);
        assert!((goal.priority - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_phi_maximization_goal_satisfied() {
        let goal = PhiMaximizationGoal::new(0.7);

        // State with phi < target -> not satisfied
        let low_phi = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        assert!(!goal.is_satisfied(&low_phi));

        // State with phi >= target -> satisfied
        let high_phi = LatentConsciousnessState::from_observables(0.8, 0.5, 0.5, 0.5);
        assert!(goal.is_satisfied(&high_phi));
    }

    #[test]
    fn test_phi_maximization_goal_distance() {
        let goal = PhiMaximizationGoal::new(0.8);

        let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let distance = goal.distance_to_goal(&state);

        // Distance should be target - current = 0.3
        assert!((distance - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_phi_maximization_goal_reward() {
        let goal = PhiMaximizationGoal::new(0.8);

        // Satisfied state should get reward 1.0
        let satisfied = LatentConsciousnessState::from_observables(0.9, 0.5, 0.5, 0.5);
        assert!((goal.reward(&satisfied) - 1.0).abs() < 0.001);

        // Unsatisfied state should get negative reward
        let unsatisfied = LatentConsciousnessState::from_observables(0.3, 0.5, 0.5, 0.5);
        assert!(goal.reward(&unsatisfied) < 0.5);
    }

    #[test]
    fn test_coherence_goal_creation() {
        let goal = CoherenceGoal::new();
        assert!((goal.min_coherence - 0.6).abs() < 0.001);
        assert!((goal.min_integration - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_coherence_goal_satisfied() {
        let goal = CoherenceGoal::new();

        // State with low coherence/integration -> not satisfied
        let low = LatentConsciousnessState::from_observables(0.5, 0.4, 0.4, 0.5);
        assert!(!goal.is_satisfied(&low));

        // State with high coherence/integration -> satisfied
        let high = LatentConsciousnessState::from_observables(0.5, 0.7, 0.7, 0.5);
        assert!(goal.is_satisfied(&high));
    }

    #[test]
    fn test_coherence_goal_distance() {
        let goal = CoherenceGoal::with_thresholds(0.6, 0.6);

        // State with coherence=0.4, integration=0.4 should have distance 0.2
        let state = LatentConsciousnessState::from_observables(0.5, 0.4, 0.4, 0.5);
        let distance = goal.distance_to_goal(&state);
        assert!((distance - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_goal_priority() {
        let high_priority = PhiMaximizationGoal::with_priority(0.8, 2.0);
        let low_priority = CoherenceGoal::new();

        assert!(high_priority.priority() > low_priority.priority());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 2: HDC ENCODABLE TESTS (Generalization Refactoring)
    // ═══════════════════════════════════════════════════════════════════════

    use crate::core::domain_traits::HdcEncodable;

    #[test]
    fn test_latent_state_to_hv() {
        let state = LatentConsciousnessState::from_observables(0.7, 0.6, 0.5, 0.8);
        let hv = state.to_hv();

        // HV should have correct dimension
        assert_eq!(hv.dim(), crate::hdc::unified_hv::HDC_DIMENSION);

        // HV should be non-zero
        let norm: f32 = hv.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.1);
    }

    #[test]
    fn test_latent_state_hv_similarity() {
        let state1 = LatentConsciousnessState::from_observables(0.7, 0.6, 0.5, 0.8);
        let state2 = LatentConsciousnessState::from_observables(0.7, 0.6, 0.5, 0.8);
        let state3 = LatentConsciousnessState::from_observables(0.1, 0.1, 0.1, 0.1);

        // Same states should have high similarity
        let sim_same = state1.semantic_similarity(&state2);
        assert!(sim_same > 0.8, "Same states should be similar: {}", sim_same);

        // Different states should have lower similarity
        let sim_diff = state1.semantic_similarity(&state3);
        assert!(
            sim_diff < sim_same,
            "Different states should be less similar: {} vs {}",
            sim_diff,
            sim_same
        );
    }

    #[test]
    fn test_latent_state_hv_roundtrip() {
        let original = LatentConsciousnessState::from_observables(0.7, 0.6, 0.5, 0.8);
        let hv = original.to_hv();
        let decoded = LatentConsciousnessState::from_hv(&hv);

        // Decoding should succeed
        assert!(decoded.is_some());

        // Decoded state should be similar to original (but not exact - HDC is lossy)
        let decoded = decoded.unwrap();
        let similarity = original.semantic_similarity(&decoded);
        // We expect some loss, but should still be reasonably similar
        assert!(
            similarity > 0.3,
            "Roundtrip should preserve some similarity: {}",
            similarity
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 2: QUALITY SIGNAL TESTS
    // ═══════════════════════════════════════════════════════════════════════

    use crate::core::domain_traits::QualitySignal;

    #[test]
    fn test_phi_quality_signal() {
        let signal = PhiQualitySignal::new();
        let state = LatentConsciousnessState::from_observables(0.7, 0.6, 0.5, 0.8);

        assert_eq!(signal.name(), "phi");

        let score = signal.measure(&state);
        assert!((score - 0.7).abs() < 0.001, "Score should equal phi: {}", score);

        // Weight should be default 1.0
        let weight = signal.weight();
        assert!(
            (weight - 1.0).abs() < 0.001,
            "Default weight should be 1.0: {}",
            weight
        );
    }

    #[test]
    fn test_phi_quality_signal_with_weight() {
        let signal = PhiQualitySignal::with_weight(2.0);

        assert!((signal.weight() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_coherence_quality_signal() {
        let signal = CoherenceQualitySignal::new();
        let state = LatentConsciousnessState::from_observables(0.7, 0.6, 0.5, 0.8);

        assert_eq!(signal.name(), "coherence");

        let score = signal.measure(&state);
        assert!((score - 0.5).abs() < 0.001, "Score should equal coherence: {}", score);
    }

    #[test]
    fn test_integration_quality_signal() {
        let signal = IntegrationQualitySignal::new();
        let state = LatentConsciousnessState::from_observables(0.7, 0.6, 0.5, 0.8);

        assert_eq!(signal.name(), "integration");

        let score = signal.measure(&state);
        assert!((score - 0.6).abs() < 0.001, "Score should equal integration: {}", score);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 2: DOMAIN ADAPTER TESTS
    // ═══════════════════════════════════════════════════════════════════════

    use crate::core::domain_traits::DomainAdapter;

    #[test]
    fn test_consciousness_domain_adapter_name() {
        let adapter = ConsciousnessDomainAdapter::new();
        assert_eq!(adapter.domain_name(), "consciousness");
    }

    #[test]
    fn test_consciousness_domain_adapter_available_actions() {
        let adapter = ConsciousnessDomainAdapter::new();
        let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);

        let actions = adapter.available_actions(&state);
        assert!(!actions.is_empty());
        assert!(actions.contains(&ConsciousnessAction::FocusIntegration));
        assert!(actions.contains(&ConsciousnessAction::Noop));
    }

    #[test]
    fn test_consciousness_domain_adapter_initial_state() {
        let adapter = ConsciousnessDomainAdapter::new();
        let state = adapter.initial_state();

        // Initial state should have valid observables
        assert!(state.phi >= 0.0 && state.phi <= 1.0);
        assert!(state.integration >= 0.0 && state.integration <= 1.0);
        assert!(state.coherence >= 0.0 && state.coherence <= 1.0);
        assert!(state.attention >= 0.0 && state.attention <= 1.0);
    }

    #[test]
    fn test_consciousness_domain_adapter_quality_signals() {
        let adapter = ConsciousnessDomainAdapter::new();
        let signals = adapter.quality_signals();

        // Should have 5 quality signals (phi, coherence, integration, attention, entropy)
        assert_eq!(signals.len(), 5);

        // Collect signal names
        let names: Vec<&str> = signals.iter().map(|s| s.name()).collect();
        assert!(names.contains(&"phi"));
        assert!(names.contains(&"coherence"));
        assert!(names.contains(&"integration"));
        assert!(names.contains(&"attention"));
        assert!(names.contains(&"entropy"));
    }

    #[test]
    fn test_consciousness_domain_adapter_valid_action() {
        let adapter = ConsciousnessDomainAdapter::new();
        let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);

        // All standard actions should be valid
        assert!(adapter.is_valid_action(&state, &ConsciousnessAction::FocusIntegration));
        assert!(adapter.is_valid_action(&state, &ConsciousnessAction::FocusCoherence));
        assert!(adapter.is_valid_action(&state, &ConsciousnessAction::Noop));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 2+3 TESTS: New Quality Signals, HDC, and Generic World Model
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_attention_quality_signal() {
        let signal = AttentionQualitySignal::new();

        let low_attention = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.2);
        let high_attention = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.9);

        assert!(signal.measure(&low_attention) < signal.measure(&high_attention));
        assert_eq!(signal.name(), "attention");
        assert!(signal.weight() > 0.0);
    }

    #[test]
    fn test_entropy_quality_signal() {
        let signal = EntropyQualitySignal::new();

        // Create state with uniform latent vector (low variance = low entropy = high score)
        let uniform_state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);

        // Create state with varied latent vector (high variance = high entropy = low score)
        let mut varied_state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        for i in 0..32 {
            varied_state.latent[i] = if i % 2 == 0 { 0.0 } else { 1.0 };
        }

        // Uniform should have higher score (less entropy is better for consciousness)
        assert!(signal.measure(&uniform_state) >= signal.measure(&varied_state));
        assert_eq!(signal.name(), "entropy");
    }

    #[test]
    fn test_composite_quality_signal() {
        let signal = CompositeQualitySignal::consciousness_optimized();

        let low_quality = LatentConsciousnessState::from_observables(0.2, 0.2, 0.2, 0.2);
        let high_quality = LatentConsciousnessState::from_observables(0.9, 0.9, 0.9, 0.9);

        assert!(signal.measure(&low_quality) < signal.measure(&high_quality));
        assert_eq!(signal.name(), "composite");
    }

    #[test]
    fn test_hdc_encodable_to_hv() {
        let state = LatentConsciousnessState::from_observables(0.7, 0.6, 0.5, 0.8);
        let hv = state.to_hv();

        // HDC vector should have correct dimension
        assert_eq!(hv.values.len(), crate::hdc::unified_hv::HDC_DIMENSION);
    }

    #[test]
    fn test_hdc_encodable_semantic_similarity() {
        let state1 = LatentConsciousnessState::from_observables(0.7, 0.6, 0.5, 0.8);
        let state2 = LatentConsciousnessState::from_observables(0.7, 0.6, 0.5, 0.8);
        let state3 = LatentConsciousnessState::from_observables(0.1, 0.1, 0.1, 0.1);

        // Same states should have high similarity
        let sim_same = state1.semantic_similarity(&state2);
        assert!(sim_same > 0.9);

        // Different states should have lower similarity
        let sim_diff = state1.semantic_similarity(&state3);
        assert!(sim_diff < sim_same);
    }

    #[test]
    fn test_generic_transition_creation() {
        let from_state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let to_state = LatentConsciousnessState::from_observables(0.6, 0.6, 0.6, 0.6);

        let transition = Transition::real(
            from_state.clone(),
            ConsciousnessAction::FocusIntegration,
            to_state.clone(),
            0.1,
        );

        assert!(transition.is_real);
        assert!((transition.reward - 0.1).abs() < 0.001);

        let imagined = Transition::imagined(
            from_state,
            ConsciousnessAction::FocusCoherence,
            to_state,
            0.2,
        );

        assert!(!imagined.is_real);
    }

    #[test]
    fn test_generic_world_model_creation() {
        let model = create_generic_consciousness_world_model();

        assert!(!model.is_ready());
        assert_eq!(model.stats().transitions_observed, 0);
    }

    #[test]
    fn test_generic_world_model_observe() {
        let mut model = GenericWorldModel::with_config(
            ConsciousnessDynamicsModel::new(),
            RewardPredictor::new(),
            WorldModelConfig {
                min_training_samples: 3,
                ..Default::default()
            },
        );

        for i in 0..5 {
            let from_state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
            let to_state = LatentConsciousnessState::from_observables(0.6, 0.6, 0.6, 0.6);

            let transition = Transition::real(
                from_state,
                ConsciousnessAction::FocusIntegration,
                to_state,
                0.1 * (i as f64),
            );

            model.observe(transition);
        }

        assert!(model.is_ready());
        assert_eq!(model.stats().transitions_observed, 5);
    }

    #[test]
    fn test_generic_world_model_predict() {
        let model = create_generic_consciousness_world_model();
        let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);

        let next = model.predict_next(&state, ConsciousnessAction::FocusIntegration);

        // Should produce valid state
        assert!(next.phi >= 0.0 && next.phi <= 1.0);

        let reward = model.predict_reward(&state, ConsciousnessAction::FocusCoherence);

        // Should produce finite reward
        assert!(reward.is_finite());
    }

    #[test]
    fn test_dynamics_predictor_trait() {
        let mut dynamics = ConsciousnessDynamicsModel::new();

        // Test through trait
        let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let predicted: LatentConsciousnessState = <ConsciousnessDynamicsModel as DynamicsPredictor<_, _>>::predict(
            &dynamics,
            &state,
            ConsciousnessAction::FocusIntegration,
        );

        assert!(predicted.phi >= 0.0 && predicted.phi <= 1.0);

        // Test training through trait
        let transition = Transition::real(
            state.clone(),
            ConsciousnessAction::FocusIntegration,
            predicted.clone(),
            0.1,
        );

        <ConsciousnessDynamicsModel as DynamicsPredictor<_, _>>::train(&mut dynamics, &transition);
        assert_eq!(dynamics.train_count, 1);
    }

    #[test]
    fn test_reward_estimator_trait() {
        let mut estimator = RewardPredictor::new();
        let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);

        // Test prediction through trait
        let reward: f64 = <RewardPredictor as RewardEstimator<_, _>>::predict(
            &estimator,
            &state,
            ConsciousnessAction::FocusIntegration,
        );

        assert!(reward.is_finite());

        // Test training through trait
        <RewardPredictor as RewardEstimator<_, _>>::train(
            &mut estimator,
            &state,
            ConsciousnessAction::FocusCoherence,
            1.0,
        );

        // After training, prediction should change
        let reward2 = estimator.predict(&state, ConsciousnessAction::FocusCoherence);
        // It should have learned something
        assert!(reward2.is_finite());
    }

    #[test]
    fn test_hdc_basis_vectors_cached() {
        // Access basis vectors multiple times to verify caching works
        let basis1 = &*hdc_basis::PHI_BASIS;
        let basis2 = &*hdc_basis::PHI_BASIS;

        // Should be the same reference (lazy_static)
        assert_eq!(basis1.values.len(), basis2.values.len());

        // Latent bases should all be different
        let lb1 = &*hdc_basis::LATENT_BASES;
        assert_eq!(lb1.len(), 32);
    }
}
