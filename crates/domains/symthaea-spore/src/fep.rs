// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Free Energy Principle (FEP) active inference for the Spore WASM kernel.
//!
//! Simplified port of `symthaea-fep` — fully WASM-compatible (no `std::time`,
//! deterministic xorshift RNG). Implements the perception-action loop:
//!
//! ```text
//! observe -> update beliefs (minimize F) -> select action (minimize G) -> act
//! ```
//!
//! # Core equation
//!
//! Variational free energy:  F = Complexity - Accuracy
//!   - Accuracy  = E_q[ln p(o|s)]  (how well beliefs explain observations)
//!   - Complexity = D_KL[q(s) || p(s)]  (divergence from prior)
//!
//! Expected free energy for action selection:
//!   G(a) = pragmatic_value + epistemic_value
//!
//! The agent picks the action minimizing G (softmax with temperature).

use serde::{Deserialize, Serialize};

// =============================================================================
// RNG — deterministic xorshift64 (WASM-safe, no std::time)
// =============================================================================

fn xorshift(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

// =============================================================================
// MOTOR COMMAND TYPES
// =============================================================================

/// The 8 cognitive/motor actions available to the active inference agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotorCommandType {
    /// Redirect attention to a different sensory modality or feature.
    AttentionShift,
    /// Modulate the model learning rate up or down.
    LearningRateAdjust,
    /// Trigger exploratory behavior to reduce epistemic uncertainty.
    ExplorationTrigger,
    /// Initiate metacognitive reflection on current beliefs.
    ReflectionInitiate,
    /// Consolidate current representations into stable memory.
    MemoryConsolidate,
    /// Reset cached expectations (model mismatch recovery).
    ExpectationReset,
    /// Emit an external motor/effector command.
    MotorOutput,
    /// Maintain current state — no action needed.
    NoOp,
}

impl MotorCommandType {
    /// Map action index (0..7) to command type.
    pub fn from_action_index(index: usize) -> Self {
        match index {
            0 => Self::AttentionShift,
            1 => Self::LearningRateAdjust,
            2 => Self::ExplorationTrigger,
            3 => Self::ReflectionInitiate,
            4 => Self::MemoryConsolidate,
            5 => Self::ExpectationReset,
            6 => Self::MotorOutput,
            _ => Self::NoOp,
        }
    }

    /// Map command type back to action index.
    pub fn to_action_index(&self) -> usize {
        match self {
            Self::AttentionShift => 0,
            Self::LearningRateAdjust => 1,
            Self::ExplorationTrigger => 2,
            Self::ReflectionInitiate => 3,
            Self::MemoryConsolidate => 4,
            Self::ExpectationReset => 5,
            Self::MotorOutput => 6,
            Self::NoOp => 7,
        }
    }
}

// =============================================================================
// MOTOR COMMAND
// =============================================================================

/// A motor command with intensity and confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorCommand {
    /// Type of action to take.
    pub command_type: MotorCommandType,
    /// Intensity of the command (clamped 0.0..=1.0).
    pub intensity: f32,
    /// Confidence in this command being the right choice (0.0..=1.0).
    pub confidence: f32,
}

impl MotorCommand {
    pub fn new(command_type: MotorCommandType, intensity: f32, confidence: f32) -> Self {
        Self {
            command_type,
            intensity: intensity.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    pub fn noop() -> Self {
        Self {
            command_type: MotorCommandType::NoOp,
            intensity: 0.0,
            confidence: 1.0,
        }
    }
}

// =============================================================================
// OBSERVATION
// =============================================================================

/// An observation vector with precision (inverse variance / confidence).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Raw observation values.
    pub values: Vec<f32>,
    /// Observation precision — higher means more trustworthy.
    pub precision: f32,
}

impl Observation {
    pub fn new(values: Vec<f32>, precision: f32) -> Self {
        Self {
            values,
            precision: precision.max(0.01),
        }
    }

    /// Construct from the four consciousness observables.
    pub fn from_consciousness(phi: f32, integration: f32, coherence: f32, attention: f32) -> Self {
        Self {
            values: vec![phi, integration, coherence, attention],
            precision: 1.0,
        }
    }
}

// =============================================================================
// HIDDEN STATE (beliefs)
// =============================================================================

/// Gaussian belief over hidden states: q(s) = N(mean, diag(1/precision)).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenState {
    /// Mean of the belief distribution.
    pub mean: Vec<f32>,
    /// Per-dimension precision (inverse variance).
    pub precision: Vec<f32>,
}

impl HiddenState {
    pub fn new(dim: usize) -> Self {
        Self {
            mean: vec![0.5; dim],
            precision: vec![1.0; dim],
        }
    }

    /// Differential entropy of the multivariate diagonal Gaussian.
    ///
    /// H = 0.5 * (d + d*ln(2pi) + sum(ln(1/precision_i)))
    pub fn entropy(&self) -> f32 {
        let d = self.mean.len() as f32;
        let log_det: f32 = self.precision.iter().map(|p| -(p.max(0.001)).ln()).sum();
        0.5 * (d + d * (core::f32::consts::TAU).ln() + log_det)
    }

    /// Scalar confidence in [0, 1] — average precision clamped.
    pub fn confidence(&self) -> f32 {
        if self.precision.is_empty() {
            return 0.0;
        }
        let avg = self.precision.iter().sum::<f32>() / self.precision.len() as f32;
        avg.clamp(0.0, 1.0)
    }
}

// =============================================================================
// FREE ENERGY COMPONENTS
// =============================================================================

/// Decomposed variational free energy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeEnergyComponents {
    /// Total free energy F = complexity - accuracy.
    pub total: f32,
    /// Accuracy: E_q[ln p(o|s)].
    pub accuracy: f32,
    /// Complexity: D_KL[q(s) || p(s)].
    pub complexity: f32,
    /// Magnitude of prediction error vector.
    pub prediction_error: f32,
}

impl Default for FreeEnergyComponents {
    fn default() -> Self {
        Self {
            total: 0.0,
            accuracy: 0.0,
            complexity: 0.0,
            prediction_error: 0.0,
        }
    }
}

// =============================================================================
// GENERATIVE MODEL
// =============================================================================

/// Simplified generative model P(o,s) = P(o|s) P(s'|s).
///
/// Uses dense matrices stored as `Vec<Vec<f32>>`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerativeModel {
    /// Observation likelihood: observation[i][j] = P(o_j | s_i).
    pub observation: Vec<Vec<f32>>,
    /// State transition: transition[i][j] = P(s'_j | s_i).
    pub transition: Vec<Vec<f32>>,
    /// Hidden state dimensionality.
    pub state_dim: usize,
    /// Observation dimensionality.
    pub obs_dim: usize,
}

// Matrix rows/cols are stored as Vec<Vec<f32>>, and every loop below uses the
// index itself (i == j diagonal checks, row/col position) rather than purely
// for element access, so a zip/enumerate rewrite would be less readable.
#[allow(clippy::needless_range_loop)]
impl GenerativeModel {
    /// Create a new model with identity-like (diagonal-dominant) matrices.
    pub fn new(state_dim: usize, obs_dim: usize) -> Self {
        // Observation matrix: near-identity mapping state -> obs
        let mut observation = vec![vec![0.0f32; obs_dim]; state_dim];
        for i in 0..state_dim {
            for j in 0..obs_dim {
                if i == j && i < obs_dim {
                    observation[i][j] = 0.8;
                } else {
                    // Small off-diagonal coupling
                    observation[i][j] = 0.2 / (state_dim.max(obs_dim)) as f32;
                }
            }
        }

        // Transition matrix: high self-transition, small diffusion
        let mut transition = vec![vec![0.0f32; state_dim]; state_dim];
        for i in 0..state_dim {
            for j in 0..state_dim {
                if i == j {
                    transition[i][j] = 0.8;
                } else {
                    transition[i][j] = 0.2 / (state_dim - 1).max(1) as f32;
                }
            }
        }

        Self {
            observation,
            transition,
            state_dim,
            obs_dim,
        }
    }

    /// Predicted observation from state: E[o] = A * s.
    pub fn predict(&self, state: &[f32]) -> Vec<f32> {
        let mut predicted = vec![0.0f32; self.obs_dim];
        for i in 0..self.state_dim.min(state.len()) {
            for j in 0..self.obs_dim {
                predicted[j] += self.observation[i][j] * state[i];
            }
        }
        predicted
    }

    /// Predicted next state: E[s'] = B * s.
    pub fn transition(&self, state: &[f32]) -> Vec<f32> {
        let mut next = vec![0.0f32; self.state_dim];
        for i in 0..self.state_dim.min(state.len()) {
            for j in 0..self.state_dim {
                next[j] += self.transition[i][j] * state[i];
            }
        }
        next
    }

    /// Hebbian-like learning: adjust observation matrix toward observed data.
    pub fn learn(&mut self, state: &[f32], observation: &[f32], lr: f32) {
        let predicted = self.predict(state);
        for i in 0..self.state_dim.min(state.len()) {
            for j in 0..self.obs_dim.min(observation.len()) {
                let error = observation[j] - predicted[j];
                let gradient = error * state[i];
                self.observation[i][j] += lr * gradient;
                self.observation[i][j] = self.observation[i][j].clamp(0.0, 1.0);
            }
        }
    }
}

// =============================================================================
// FREE ENERGY CALCULATOR
// =============================================================================

/// Computes variational free energy F = Complexity - Accuracy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeEnergyCalculator;

impl FreeEnergyCalculator {
    /// Compute free energy given an observation, belief state, and generative model.
    pub fn compute(
        &self,
        obs: &Observation,
        belief: &HiddenState,
        model: &GenerativeModel,
    ) -> FreeEnergyComponents {
        let predicted = model.predict(&belief.mean);

        // Accuracy: -0.5 * precision * ||o - o_hat||^2
        let mut sum_sq = 0.0f32;
        for (p, o) in predicted.iter().zip(obs.values.iter()) {
            sum_sq += (p - o).powi(2);
        }
        let accuracy = -0.5 * obs.precision * sum_sq;

        // Complexity: KL(q || prior), prior = N(0.5, I)
        let mut kl = 0.0f32;
        for i in 0..belief.mean.len() {
            let var_q = 1.0 / belief.precision[i].max(0.001);
            let mean_diff = belief.mean[i] - 0.5;
            // KL for univariate Gaussians (prior variance = 1.0)
            kl += 0.5 * (var_q + mean_diff.powi(2) - 1.0 - var_q.max(0.001).ln());
        }
        let complexity = kl.max(0.0);

        let prediction_error = sum_sq.sqrt();
        let total = complexity - accuracy;

        FreeEnergyComponents {
            total,
            accuracy,
            complexity,
            prediction_error,
        }
    }
}

// =============================================================================
// FEP CYCLE RESULT
// =============================================================================

/// Result of one full FEP perception-action cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FepCycleResult {
    /// Total variational free energy.
    pub free_energy: f32,
    /// Magnitude of prediction error.
    pub prediction_error: f32,
    /// The selected motor command.
    pub motor_command: MotorCommand,
    /// Whether prediction error exceeds surprise threshold.
    pub is_surprised: bool,
    /// Whether the agent is in exploration mode (epistemic > pragmatic).
    pub exploration_mode: bool,
    /// Confidence of the current belief state.
    pub belief_confidence: f32,
}

// =============================================================================
// ACTIVE INFERENCE CONFIG
// =============================================================================

/// Configuration for the active inference agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceConfig {
    /// Hidden state dimensionality (default 8).
    pub state_dim: usize,
    /// Observation dimensionality (default 4).
    pub obs_dim: usize,
    /// Number of available actions (default 8, one per MotorCommandType).
    pub num_actions: usize,
    /// Belief update iterations per perception step.
    pub inference_iterations: usize,
    /// Learning rate for belief gradient descent.
    pub belief_lr: f32,
    /// Softmax temperature for action selection (lower = more greedy).
    pub action_temperature: f32,
}

impl Default for ActiveInferenceConfig {
    fn default() -> Self {
        Self {
            state_dim: 8,
            obs_dim: 4,
            num_actions: 8,
            inference_iterations: 5,
            belief_lr: 0.1,
            action_temperature: 1.0,
        }
    }
}

// =============================================================================
// ACTIVE INFERENCE AGENT
// =============================================================================

/// Active inference agent implementing the full perception-action loop.
///
/// On each cycle:
/// 1. Perceive: update beliefs to minimize free energy F.
/// 2. Act: select action minimizing expected free energy G.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceAgent {
    config: ActiveInferenceConfig,
    belief: HiddenState,
    model: GenerativeModel,
    fe_calc: FreeEnergyCalculator,
    last_fe: FreeEnergyComponents,
    cycle: u64,
    rng_state: u64,
}

impl ActiveInferenceAgent {
    /// Create agent with explicit configuration.
    pub fn new(config: ActiveInferenceConfig) -> Self {
        let belief = HiddenState::new(config.state_dim);
        let model = GenerativeModel::new(config.state_dim, config.obs_dim);
        Self {
            config,
            belief,
            model,
            fe_calc: FreeEnergyCalculator,
            last_fe: FreeEnergyComponents::default(),
            cycle: 0,
            rng_state: 0x9E3779B97F4A7C15, // golden ratio hash
        }
    }

    /// Create agent with sensible defaults (state_dim=8, obs_dim=4, 8 actions).
    pub fn with_defaults() -> Self {
        Self::new(ActiveInferenceConfig::default())
    }

    /// Perception step: update beliefs via gradient descent on free energy.
    ///
    /// Returns the free energy components after convergence.
    pub fn perceive(&mut self, obs: &Observation) -> FreeEnergyComponents {
        for _ in 0..self.config.inference_iterations {
            self.update_belief(obs);
        }

        let fe = self.fe_calc.compute(obs, &self.belief, &self.model);
        self.last_fe = fe.clone();

        // Learn: update generative model toward observed data
        let lr = self.config.belief_lr * 0.1; // model learns slower than beliefs
        self.model.learn(&self.belief.mean, &obs.values, lr);

        fe
    }

    /// Single belief update iteration (gradient descent on F).
    fn update_belief(&mut self, obs: &Observation) {
        let predicted = self.model.predict(&self.belief.mean);

        // Prediction error weighted by observation precision
        let mut weighted_error = vec![0.0f32; self.config.obs_dim];
        for ((we, &val), &pred) in weighted_error
            .iter_mut()
            .zip(obs.values.iter())
            .zip(predicted.iter())
        {
            *we = (val - pred) * obs.precision;
        }

        // Gradient w.r.t. belief mean: dF/dmu = A^T * weighted_error + prior_pull
        for i in 0..self.config.state_dim {
            let mut grad = 0.0f32;
            if let Some(row) = self.model.observation.get(i) {
                for (&obs_ij, &we_j) in row.iter().zip(weighted_error.iter()) {
                    grad += obs_ij * we_j;
                }
            }
            // Prior pull toward 0.5
            let prior_grad = 0.1 * (0.5 - self.belief.mean[i]);
            let delta = self.config.belief_lr * (grad + prior_grad);
            self.belief.mean[i] = (self.belief.mean[i] + delta).clamp(0.0, 1.0);
        }

        // Update precision based on local prediction accuracy
        for i in 0..self.belief.precision.len() {
            let error_i = if i < weighted_error.len() {
                weighted_error[i].abs()
            } else {
                0.5
            };
            let new_prec = 1.0 / (1.0 + error_i);
            self.belief.precision[i] = 0.9 * self.belief.precision[i] + 0.1 * new_prec;
        }
    }

    /// Action selection: compute expected free energy for each action, softmax sample.
    ///
    /// Returns `(action_index, MotorCommand)`.
    pub fn select_action(&mut self) -> (usize, MotorCommand) {
        let num = self.config.num_actions;

        // Compute expected free energy G(a) for each action
        let mut g_values = Vec::with_capacity(num);
        for a in 0..num {
            g_values.push(self.expected_free_energy(a));
        }

        // Softmax over -G (lower G is better, so we negate for exp)
        let max_neg_g = g_values
            .iter()
            .map(|g| -g)
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = g_values
            .iter()
            .map(|g| ((-g - max_neg_g) / self.config.action_temperature).exp())
            .collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|e| e / sum_exp).collect();

        // Sample from distribution via xorshift
        let u = xorshift(&mut self.rng_state) as f32;
        let mut cumulative = 0.0f32;
        let mut selected = num - 1;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if u < cumulative {
                selected = i;
                break;
            }
        }

        let intensity = probs[selected]; // probability as proxy for intensity
        let confidence = self.belief.confidence();
        let cmd = MotorCommand::new(
            MotorCommandType::from_action_index(selected),
            intensity,
            confidence,
        );

        (selected, cmd)
    }

    /// Expected free energy G(a) for a single action.
    ///
    /// G(a) = pragmatic (distance from preferred obs) + epistemic (entropy change)
    fn expected_free_energy(&self, action: usize) -> f32 {
        // Predict next state under action
        // Simplified: each action biases the transition differently
        let mut biased_state = self.belief.mean.clone();
        if action < biased_state.len() {
            biased_state[action] += 0.1;
            biased_state[action] = biased_state[action].clamp(0.0, 1.0);
        }
        let next_state = self.model.transition(&biased_state);
        let expected_obs = self.model.predict(&next_state);

        // Pragmatic value: squared distance from preferred observation (all 0.8)
        let pragmatic: f32 = expected_obs.iter().map(|o| (o - 0.8).powi(2)).sum();

        // Epistemic value: predicted entropy change
        let current_entropy = self.belief.entropy();
        let next_belief = HiddenState {
            mean: next_state,
            precision: self.belief.precision.clone(),
        };
        let predicted_entropy = next_belief.entropy();
        let epistemic = predicted_entropy - current_entropy;

        pragmatic + 0.5 * epistemic
    }

    /// Run a full perception-action cycle from consciousness observables.
    pub fn cycle(
        &mut self,
        phi: f32,
        integration: f32,
        coherence: f32,
        attention: f32,
    ) -> FepCycleResult {
        self.cycle += 1;

        let obs = Observation::from_consciousness(phi, integration, coherence, attention);
        let fe = self.perceive(&obs);
        let (_, motor_command) = self.select_action();

        let is_surprised = fe.prediction_error > 0.5;
        let exploration_mode = motor_command.command_type == MotorCommandType::ExplorationTrigger
            || fe.prediction_error > 0.3;

        FepCycleResult {
            free_energy: fe.total,
            prediction_error: fe.prediction_error,
            motor_command,
            is_surprised,
            exploration_mode,
            belief_confidence: self.belief.confidence(),
        }
    }

    /// Current total free energy.
    pub fn free_energy(&self) -> f32 {
        self.last_fe.total
    }

    /// Reference to current belief state.
    pub fn belief(&self) -> &HiddenState {
        &self.belief
    }

    /// Whether the agent is in a surprised state (prediction error > 0.5).
    pub fn is_surprised(&self) -> bool {
        self.last_fe.prediction_error > 0.5
    }

    /// Current belief confidence (0.0-1.0).
    pub fn belief_confidence(&self) -> f32 {
        self.belief.confidence()
    }

    /// Current cycle count.
    pub fn cycle_count(&self) -> u64 {
        self.cycle
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_update_converges() {
        let mut agent = ActiveInferenceAgent::with_defaults();
        let obs = Observation::from_consciousness(0.7, 0.8, 0.6, 0.9);

        let fe1 = agent.perceive(&obs);
        let _fe2 = agent.perceive(&obs);
        let fe3 = agent.perceive(&obs);

        // Free energy should decrease (or at least not increase much) as beliefs converge
        // After repeated exposure to the same observation, prediction error should drop
        assert!(
            fe3.prediction_error <= fe1.prediction_error + 0.01,
            "prediction error should decrease: first={}, third={}",
            fe1.prediction_error,
            fe3.prediction_error
        );
    }

    #[test]
    fn test_action_selection_distribution() {
        let mut agent = ActiveInferenceAgent::with_defaults();
        let obs = Observation::from_consciousness(0.5, 0.5, 0.5, 0.5);
        agent.perceive(&obs);

        // Run many action selections and verify all 8 types can appear
        let mut counts = [0u32; 8];
        // Reset RNG each time to get varied results across many samples
        for seed in 0..200u64 {
            agent.rng_state = 0x9E3779B97F4A7C15 ^ seed.wrapping_mul(6364136223846793005);
            let (idx, cmd) = agent.select_action();
            assert!(idx < 8, "action index out of range: {}", idx);
            assert_eq!(cmd.command_type, MotorCommandType::from_action_index(idx));
            counts[idx] += 1;
        }

        // At least some actions should be selected (not all go to one bucket)
        let nonzero = counts.iter().filter(|c| **c > 0).count();
        assert!(
            nonzero >= 2,
            "expected at least 2 different actions selected, got {}: {:?}",
            nonzero,
            counts
        );
    }

    #[test]
    fn test_free_energy_computation() {
        let fe_calc = FreeEnergyCalculator;
        let model = GenerativeModel::new(4, 4);
        let belief = HiddenState::new(4);

        // Observation matching the belief prior (0.5 everywhere)
        let matching_obs = Observation::new(vec![0.5; 4], 1.0);
        let fe_match = fe_calc.compute(&matching_obs, &belief, &model);

        // Observation far from belief
        let distant_obs = Observation::new(vec![1.0, 0.0, 1.0, 0.0], 1.0);
        let fe_distant = fe_calc.compute(&distant_obs, &belief, &model);

        // Distant observation should produce higher free energy
        assert!(
            fe_distant.total > fe_match.total,
            "distant obs FE ({}) should exceed matching obs FE ({})",
            fe_distant.total,
            fe_match.total
        );

        // Distant observation should have higher prediction error
        assert!(fe_distant.prediction_error > fe_match.prediction_error);
    }

    #[test]
    fn test_motor_command_types_roundtrip() {
        let types = [
            MotorCommandType::AttentionShift,
            MotorCommandType::LearningRateAdjust,
            MotorCommandType::ExplorationTrigger,
            MotorCommandType::ReflectionInitiate,
            MotorCommandType::MemoryConsolidate,
            MotorCommandType::ExpectationReset,
            MotorCommandType::MotorOutput,
            MotorCommandType::NoOp,
        ];

        for (i, t) in types.iter().enumerate() {
            assert_eq!(t.to_action_index(), i);
            assert_eq!(MotorCommandType::from_action_index(i), *t);
        }

        // Out-of-range index maps to NoOp
        assert_eq!(
            MotorCommandType::from_action_index(99),
            MotorCommandType::NoOp
        );
    }

    #[test]
    fn test_observation_from_consciousness() {
        let obs = Observation::from_consciousness(0.7, 0.8, 0.6, 0.9);
        assert_eq!(obs.values.len(), 4);
        assert!((obs.values[0] - 0.7).abs() < f32::EPSILON);
        assert!((obs.values[3] - 0.9).abs() < f32::EPSILON);
        assert!((obs.precision - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_full_cycle() {
        let mut agent = ActiveInferenceAgent::with_defaults();

        let result = agent.cycle(0.6, 0.7, 0.5, 0.8);
        assert!(result.free_energy.is_finite());
        assert!(result.prediction_error >= 0.0);
        assert!(result.belief_confidence >= 0.0 && result.belief_confidence <= 1.0);
        assert_eq!(agent.cycle_count(), 1);

        // Run several more cycles to ensure stability
        for i in 0..10 {
            let r = agent.cycle(0.6 + i as f32 * 0.01, 0.7, 0.5, 0.8);
            assert!(
                r.free_energy.is_finite(),
                "FE not finite at cycle {}",
                i + 2
            );
        }
        assert_eq!(agent.cycle_count(), 11);
    }

    #[test]
    fn test_hidden_state_entropy_and_confidence() {
        let state = HiddenState::new(4);
        let entropy = state.entropy();
        assert!(entropy.is_finite() && entropy > 0.0, "entropy={}", entropy);

        let confidence = state.confidence();
        assert!(
            (0.0..=1.0).contains(&confidence),
            "confidence={}",
            confidence
        );

        // Higher precision -> higher confidence
        let precise = HiddenState {
            mean: vec![0.5; 4],
            precision: vec![0.9; 4],
        };
        let vague = HiddenState {
            mean: vec![0.5; 4],
            precision: vec![0.1; 4],
        };
        assert!(precise.confidence() > vague.confidence());
        // Higher precision -> lower entropy
        assert!(precise.entropy() < vague.entropy());
    }
}
