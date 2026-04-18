// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Counterfactual Dream Engine
===========================

Implements "Counterfactual Dreaming" - the ability to simulate alternative pasts
to learn from mistakes not made.

Algorithm:
1. Record high-surprise events (wake).
2. During low-load (sleep/idle), sample an event.
3. Generate N alternative actions.
4. Simulate outcomes using a simple world model.
5. If alternative yields higher Phi than reality -> Consolidate as "Wisdom".

# Example

```ignore
use symthaea_dream::{DreamEngine, DreamEngineConfig};

let config = DreamEngineConfig::default();
let mut engine = DreamEngine::new(config);

// Record a surprising event during wakefulness
let state = vec![0.5; 64];
let action = vec![0.1; 32]; // Implements DreamableAction
let outcome = vec![0.3; 64];
engine.record(&state, action, &outcome, 0.5); // surprise = 0.5

// Run a dream cycle during sleep/idle
let insights = engine.dream().unwrap();
println!("Gained {} insights from dreaming", insights);
```
*/

#![deny(unsafe_code)]

pub mod motor_trajectory;

use anyhow::Result;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::fmt::Debug;

/// Trait for actions that can be dreamed about
pub trait DreamableAction: Clone + Debug + Serialize + DeserializeOwned + Send + Sync {
    /// Generate a perturbed version of this action (counterfactual)
    fn perturb(&self, seed: u64) -> Self;

    /// Predict the outcome vector of this action given a state
    fn predict_outcome(&self, state: &[f32]) -> Vec<f32>;

    /// Quantify the magnitude/cost of the action (for statistics)
    fn magnitude(&self) -> f32;
}

/// Default implementation for float vectors (backward compatibility)
impl DreamableAction for Vec<f32> {
    fn perturb(&self, seed: u64) -> Self {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(&seed.to_le_bytes());
        for v in self {
            hasher.update(&v.to_le_bytes());
        }

        let mut bytes = vec![0u8; self.len() * 4];
        let mut xof = hasher.finalize_xof();
        xof.fill(&mut bytes);

        self.iter()
            .zip(bytes.chunks_exact(4))
            .map(|(&val, chunk)| {
                let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let noise = (bits as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32;
                // Perturb by 20% max
                (val + noise * 0.2).clamp(-1.0, 1.0)
            })
            .collect()
    }

    fn predict_outcome(&self, state: &[f32]) -> Vec<f32> {
        let action_influence = self.magnitude();
        state
            .iter()
            .map(|&s| (s * 0.7 + action_influence * 0.1).clamp(-1.0, 1.0))
            .collect()
    }

    fn magnitude(&self) -> f32 {
        if self.is_empty() {
            0.0
        } else {
            self.iter().map(|a| a.abs()).sum::<f32>() / self.len() as f32
        }
    }
}

/// Configuration for the dream engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamEngineConfig {
    /// Minimum surprise threshold for recording events (default: 0.1)
    pub surprise_threshold: f32,
    /// Number of counterfactual simulations per dream cycle (default: 5)
    pub counterfactual_count: usize,
    /// Phi improvement threshold for wisdom consolidation (default: 0.01)
    pub wisdom_threshold: f32,
    /// Maximum events to store in memory (default: 1000)
    pub max_memory_size: usize,
    /// Default dimension for state vectors (default: 64)
    pub state_dim: usize,
}

impl Default for DreamEngineConfig {
    fn default() -> Self {
        Self {
            surprise_threshold: 0.1,
            counterfactual_count: 5,
            wisdom_threshold: 0.01,
            max_memory_size: 1000,
            state_dim: 64,
        }
    }
}

/// A recorded event from waking experience
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "A: DeserializeOwned", serialize = "A: Serialize"))]
pub struct DreamEvent<A: DreamableAction = Vec<f32>> {
    /// State at time of event
    pub state: Vec<f32>,
    /// Action taken
    pub action: A,
    /// Actual outcome observed
    pub actual_outcome: Vec<f32>,
    /// Surprise level (0.0-1.0)
    pub surprise: f32,
    /// Timestamp (optional)
    pub timestamp: Option<u64>,
}

/// Result of a dream cycle
#[derive(Debug, Clone, Default)]
pub struct DreamResult {
    /// Number of insights gained (counterfactuals that improved Phi)
    pub insights: usize,
    /// Number of events processed
    pub events_processed: usize,
    /// Total counterfactual simulations run
    pub simulations_run: usize,
    /// Best Phi improvement found
    pub best_phi_improvement: f32,
    /// Best macro-level Effective Information improvement
    pub best_ei_improvement: f32,
}

/// Consolidated wisdom from dreaming
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "A: DeserializeOwned", serialize = "A: Serialize"))]
pub struct Wisdom<A: DreamableAction> {
    /// Original state context
    pub context_state: Vec<f32>,
    /// Alternative action that would have been better
    pub better_action: A,
    /// Expected Phi improvement
    pub phi_improvement: f32,
    /// Macro-level Effective Information gain
    pub effective_information: f32,
    /// Confidence in this wisdom (0.0-1.0)
    pub confidence: f32,
}

/// Distribution of predicted outcomes for precognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeDistribution {
    pub expected_phi: f32,
    pub failure_probability: f32,
    pub confidence: f32,
}

/// A simple associative world model that learns from experience
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransitionMemory {
    /// Maps action fingerprints to observed state deltas and outcomes
    pub observations: Vec<CausalLink>,
}

/// A single causal link discovered from experience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalLink {
    pub action_fingerprint: u64,
    pub state_context: Vec<f32>,
    pub outcome: Vec<f32>,
    pub weight: f32,
}

/// Statistics for the dream engine
#[derive(Debug, Clone, Default)]
pub struct DreamEngineStats {
    /// Total events recorded
    pub events_recorded: u64,
    /// Total events rejected (below surprise threshold)
    pub events_rejected: u64,
    /// Total dream cycles run
    pub dream_cycles: u64,
    /// Total insights gained
    pub total_insights: u64,
    /// Total simulations run
    pub total_simulations: u64,
}

/// Counterfactual Dream Engine
///
/// Simulates alternative pasts to learn from mistakes not made.
/// Records high-surprise events during "wakefulness" and processes
/// them during "sleep" to discover better actions.
#[derive(Debug)]
pub struct DreamEngine<A: DreamableAction = Vec<f32>> {
    /// Configuration
    config: DreamEngineConfig,
    /// Memory of recorded events
    memory: Vec<DreamEvent<A>>,
    /// Consolidated wisdom from dreaming
    wisdom: Vec<Wisdom<A>>,
    /// Learned world model
    pub world_model: TransitionMemory,
    /// Statistics
    stats: DreamEngineStats,
}

impl<A: DreamableAction> DreamEngine<A> {
    /// Create a new dream engine with the given configuration
    pub fn new(config: DreamEngineConfig) -> Self {
        Self {
            config,
            memory: Vec::new(),
            wisdom: Vec::new(),
            world_model: TransitionMemory::default(),
            stats: DreamEngineStats::default(),
        }
    }

    /// Create a dream engine with default configuration
    pub fn with_defaults() -> Self {
        Self::new(DreamEngineConfig::default())
    }

    /// Record a waking event for later dreaming
    ///
    /// Only events with surprise above the threshold are recorded.
    pub fn record(&mut self, state: &[f32], action: A, outcome: &[f32], surprise: f32) {
        if surprise > self.config.surprise_threshold {
            // Update learned world model
            let fingerprint = self.hash_action(&action);
            self.world_model.observations.push(CausalLink {
                action_fingerprint: fingerprint,
                state_context: state.to_vec(),
                outcome: outcome.to_vec(),
                weight: surprise,
            });

            if self.world_model.observations.len() > self.config.max_memory_size {
                self.world_model.observations.remove(0);
            }

            // Enforce memory limit
            if self.memory.len() >= self.config.max_memory_size {
                // Remove oldest event
                self.memory.remove(0);
            }

            self.memory.push(DreamEvent {
                state: state.to_vec(),
                action,
                actual_outcome: outcome.to_vec(),
                surprise,
                timestamp: Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                ),
            });
            self.stats.events_recorded += 1;
        } else {
            self.stats.events_rejected += 1;
        }
    }

    fn hash_action(&self, action: &A) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut s = DefaultHasher::new();
        format!("{:?}", action).hash(&mut s);
        s.finish()
    }

    /// Record an event with explicit timestamp
    pub fn record_with_timestamp(
        &mut self,
        state: &[f32],
        action: A,
        outcome: &[f32],
        surprise: f32,
        timestamp: u64,
    ) {
        if surprise > self.config.surprise_threshold {
            if self.memory.len() >= self.config.max_memory_size {
                self.memory.remove(0);
            }

            self.memory.push(DreamEvent {
                state: state.to_vec(),
                action,
                actual_outcome: outcome.to_vec(),
                surprise,
                timestamp: Some(timestamp),
            });
            self.stats.events_recorded += 1;
        } else {
            self.stats.events_rejected += 1;
        }
    }

    /// Record a GWT-triggered consolidation event (Dehaene & Changeux 2011).
    ///
    /// When the global workspace broadcasts, the current cognitive state should
    /// be consolidated as a high-salience memory. This bypasses the normal
    /// surprise threshold to ensure broadcast-worthy content is always recorded.
    pub fn record_consolidation_event(&mut self, state: &[f32], action: A, prediction_error: f32) {
        // Use prediction_error boosted above threshold so it always records.
        // The state doubles as outcome (self-referential snapshot).
        let boosted_surprise = prediction_error.max(self.config.surprise_threshold + 0.01);
        self.record(state, action, state, boosted_surprise);
    }

    /// Run a dream cycle (Counterfactual Simulation)
    ///
    /// Processes stored events and generates counterfactual scenarios
    /// to discover better actions that would have led to higher Phi.
    pub fn dream(&mut self) -> Result<DreamResult> {
        if self.memory.is_empty() {
            return Ok(DreamResult::default());
        }

        let mut result = DreamResult::default();
        self.stats.dream_cycles += 1;

        // Process the most surprising event first
        // (In production, could sample randomly weighted by surprise)
        let event_idx = self.find_most_surprising_event();
        let event = &self.memory[event_idx];

        // 1. Evaluate actual Phi and EI
        let actual_phi = Self::estimate_phi(&event.actual_outcome);
        let actual_ei = Self::estimate_ei(&event.actual_outcome);

        // 2. Generate counterfactual actions and simulate outcomes
        let mut best_phi_improvement = 0.0f32;
        let mut best_ei_improvement = 0.0f32;
        let mut best_action: Option<A> = None;

        for i in 0..self.config.counterfactual_count {
            // Generate alternative action by perturbing the original
            let alt_action = self.generate_counterfactual_action(&event.action, i as u64);

            // Simulate outcome using simple predictive model
            let predicted_outcome = self.simulate_outcome(&event.state, &alt_action);
            let alt_phi = Self::estimate_phi(&predicted_outcome);
            let alt_ei = Self::estimate_ei(&predicted_outcome);

            result.simulations_run += 1;
            self.stats.total_simulations += 1;

            // Check if this counterfactual is better (improvement in Phi OR EI)
            let phi_improvement = alt_phi - actual_phi;
            let ei_improvement = alt_ei - actual_ei;

            if phi_improvement > self.config.wisdom_threshold || ei_improvement > 0.01 {
                result.insights += 1;
                self.stats.total_insights += 1;

                if phi_improvement > best_phi_improvement {
                    best_phi_improvement = phi_improvement;
                    best_ei_improvement = ei_improvement;
                    best_action = Some(alt_action);
                }
            }
        }

        result.events_processed = 1;
        result.best_phi_improvement = best_phi_improvement;
        result.best_ei_improvement = best_ei_improvement;

        // Consolidate wisdom if we found a significantly better action
        if let Some(action) = best_action {
            self.wisdom.push(Wisdom {
                context_state: event.state.clone(),
                better_action: action,
                phi_improvement: best_phi_improvement,
                effective_information: best_ei_improvement,
                confidence: (best_phi_improvement / 0.5 + best_ei_improvement).min(1.0),
            });
        }

        Ok(result)
    }

    /// Run multiple dream cycles
    pub fn dream_session(&mut self, cycles: usize) -> Result<Vec<DreamResult>> {
        let mut results = Vec::with_capacity(cycles);
        for _ in 0..cycles {
            results.push(self.dream()?);
        }
        Ok(results)
    }

    /// Clear processed events from memory
    pub fn clear_memory(&mut self) {
        self.memory.clear();
    }

    /// PRECOGNITION: Predict outcome distribution for an action
    ///
    /// Runs counterfactual simulations to estimate the likelihood of failure
    /// and the expected outcome before taking an action.
    pub fn predict_outcome_distribution(&self, state: &[f32], action: &A) -> OutcomeDistribution {
        let mut simulations = Vec::with_capacity(self.config.counterfactual_count);
        let mut failure_count = 0;
        let mut total_phi = 0.0;

        for i in 0..self.config.counterfactual_count {
            // Generate a slightly perturbed action to account for environmental noise
            let perturbed = action.perturb(i as u64);
            let outcome = self.simulate_outcome(state, &perturbed);
            let phi = Self::estimate_phi(&outcome);

            total_phi += phi;
            // Failure heuristic: Phi < 0.2 is considered a failure
            if phi < 0.2 {
                failure_count += 1;
            }
            simulations.push(outcome);
        }

        OutcomeDistribution {
            expected_phi: total_phi / self.config.counterfactual_count as f32,
            failure_probability: failure_count as f32 / self.config.counterfactual_count as f32,
            confidence: 1.0 - (failure_count as f32 / self.config.counterfactual_count as f32),
        }
    }

    /// Get accumulated wisdom
    pub fn wisdom(&self) -> &[Wisdom<A>] {
        &self.wisdom
    }

    /// Get statistics
    pub fn stats(&self) -> &DreamEngineStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &DreamEngineConfig {
        &self.config
    }

    /// Get number of stored events
    pub fn memory_size(&self) -> usize {
        self.memory.len()
    }

    /// Find the index of the most surprising event
    fn find_most_surprising_event(&self) -> usize {
        self.memory
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.surprise
                    .partial_cmp(&b.surprise)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Generate a counterfactual action by perturbing the original
    fn generate_counterfactual_action(&self, original: &A, seed: u64) -> A {
        original.perturb(seed)
    }

    /// Simulate outcome given state and action
    fn simulate_outcome(&self, state: &[f32], action: &A) -> Vec<f32> {
        let fingerprint = self.hash_action(action);

        // Causal reasoning: find the most similar observed state for this action
        let mut best_sim = -1.0f32;
        let mut best_outcome = None;

        for observation in &self.world_model.observations {
            if observation.action_fingerprint == fingerprint {
                let sim = self.cosine_similarity(state, &observation.state_context);
                if sim > best_sim {
                    best_sim = sim;
                    best_outcome = Some(&observation.outcome);
                }
            }
        }

        if let Some(obs_outcome) = best_outcome {
            // Pearl counterfactual: Blend observed reality with heuristic prediction
            let predicted = action.predict_outcome(state);
            obs_outcome
                .iter()
                .zip(predicted.iter())
                .map(|(&o, &p)| (o * 0.6 + p * 0.4).clamp(-1.0, 1.0))
                .collect()
        } else {
            // Fallback to heuristic prediction if action never seen
            action.predict_outcome(state)
        }
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }
        dot / (mag_a * mag_b)
    }

    /// Estimate Phi (integrated information) from outcome vector
    ///
    /// For this demo/MVP, we use the mean magnitude of the vector as a proxy
    /// for the quality/integration of the result.
    fn estimate_phi(outcome: &[f32]) -> f32 {
        if outcome.is_empty() {
            return 0.0;
        }

        let n = outcome.len() as f32;
        let sum: f32 = outcome.iter().map(|x| x.abs()).sum();
        let mean_magnitude = sum / n;
        mean_magnitude.clamp(0.0, 1.0)
    }

    /// Estimate Effective Information (causal power) from outcome vector
    ///
    /// For this demo/MVP, we use the "Determinism" of the outcome
    /// as a proxy for its causal effectiveness.
    fn estimate_ei(outcome: &[f32]) -> f32 {
        if outcome.is_empty() {
            return 0.0;
        }

        let n = outcome.len() as f32;
        let mean: f32 = outcome.iter().sum::<f32>() / n;

        // High determinism = low variance around a strong signal (near 1 or -1)
        let variance: f32 = outcome.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;

        // EI proxy: inverse of variance (how "certain" is the outcome)
        (1.0 - variance.sqrt()).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dream_engine_creation() {
        let engine = DreamEngine::<Vec<f32>>::with_defaults();
        assert_eq!(engine.memory_size(), 0);
        assert_eq!(engine.wisdom().len(), 0);
    }

    #[test]
    fn test_event_recording() {
        let mut engine = DreamEngine::<Vec<f32>>::with_defaults();

        let state = vec![0.5; 64];
        let action = vec![0.1; 32];
        let outcome = vec![0.3; 64];

        // Low surprise - should be rejected
        engine.record(&state, action.clone(), &outcome, 0.05);
        assert_eq!(engine.memory_size(), 0);
        assert_eq!(engine.stats().events_rejected, 1);

        // High surprise - should be recorded
        engine.record(&state, action, &outcome, 0.5);
        assert_eq!(engine.memory_size(), 1);
        assert_eq!(engine.stats().events_recorded, 1);
    }

    #[test]
    fn test_dream_cycle() {
        let mut engine = DreamEngine::<Vec<f32>>::with_defaults();

        // Record a surprising event
        let state = vec![0.5; 64];
        let action = vec![0.1; 32];
        let outcome = vec![0.3; 64];
        engine.record(&state, action, &outcome, 0.8);

        // Run dream cycle
        let result = engine.dream().unwrap();
        assert_eq!(result.events_processed, 1);
        assert!(result.simulations_run > 0);
    }

    #[test]
    fn test_phi_estimation() {
        // Zero vector should have low Phi
        let zero_outcome = vec![0.0; 64];
        let phi_zero = DreamEngine::<Vec<f32>>::estimate_phi(&zero_outcome);
        assert!(phi_zero < 0.1, "Zero vector should have low Phi");

        // Varied vector should have higher Phi
        let varied_outcome: Vec<f32> = (0..64).map(|i| (i as f32 / 64.0) * 2.0 - 1.0).collect();
        let phi_varied = DreamEngine::<Vec<f32>>::estimate_phi(&varied_outcome);
        assert!(
            phi_varied > phi_zero,
            "Varied vector should have higher Phi"
        );
    }

    #[test]
    fn test_memory_limit() {
        let config = DreamEngineConfig {
            max_memory_size: 5,
            ..Default::default()
        };
        let mut engine = DreamEngine::<Vec<f32>>::new(config);

        // Record more events than the limit
        for i in 0..10 {
            let state = vec![i as f32 / 10.0; 64];
            let action = vec![0.1; 32];
            let outcome = vec![0.3; 64];
            engine.record(&state, action, &outcome, 0.5);
        }

        assert_eq!(
            engine.memory_size(),
            5,
            "Memory should be capped at max_memory_size"
        );
    }

    #[test]
    fn test_counterfactual_generation() {
        let engine = DreamEngine::<Vec<f32>>::with_defaults();
        let original = vec![0.5; 32];

        let alt1 = engine.generate_counterfactual_action(&original, 0);
        let alt2 = engine.generate_counterfactual_action(&original, 1);

        // Different seeds should produce different actions
        assert_ne!(
            alt1, alt2,
            "Different seeds should produce different counterfactuals"
        );

        // Actions should be perturbed but not drastically different
        for (o, a) in original.iter().zip(alt1.iter()) {
            assert!(
                (o - a).abs() < 0.5,
                "Counterfactual should not be too different from original"
            );
        }
    }

    #[test]
    fn test_dream_session() {
        let mut engine = DreamEngine::<Vec<f32>>::with_defaults();

        // Record several events
        for i in 0..5 {
            let state = vec![(i as f32) / 5.0; 64];
            let action = vec![0.1; 32];
            let outcome = vec![0.3; 64];
            engine.record(&state, action, &outcome, 0.3 + i as f32 * 0.1);
        }

        // Run multiple dream cycles
        let results = engine.dream_session(3).unwrap();
        assert_eq!(results.len(), 3);

        // Stats should reflect the session
        assert_eq!(engine.stats().dream_cycles, 3);
    }

    #[test]
    fn test_dream_engine_config_default_values() {
        let config = DreamEngineConfig::default();
        assert_eq!(config.surprise_threshold, 0.1);
        assert_eq!(config.counterfactual_count, 5);
        assert_eq!(config.wisdom_threshold, 0.01);
        assert_eq!(config.max_memory_size, 1000);
        assert_eq!(config.state_dim, 64);
    }

    #[test]
    fn test_dream_engine_custom_config() {
        let config = DreamEngineConfig {
            surprise_threshold: 0.5,
            counterfactual_count: 10,
            max_memory_size: 50,
            ..Default::default()
        };
        let engine = DreamEngine::<Vec<f32>>::new(config);
        assert_eq!(engine.config().surprise_threshold, 0.5);
        assert_eq!(engine.config().counterfactual_count, 10);
        assert_eq!(engine.config().max_memory_size, 50);
    }

    #[test]
    fn test_record_at_exact_threshold_rejected() {
        let config = DreamEngineConfig {
            surprise_threshold: 0.5,
            ..Default::default()
        };
        let mut engine = DreamEngine::<Vec<f32>>::new(config);
        engine.record(&vec![0.5; 64], vec![0.1; 32], &vec![0.3; 64], 0.5);
        assert_eq!(engine.memory_size(), 0);
        engine.record(&vec![0.5; 64], vec![0.1; 32], &vec![0.3; 64], 0.501);
        assert_eq!(engine.memory_size(), 1);
    }

    #[test]
    fn test_record_with_timestamp() {
        let mut engine = DreamEngine::<Vec<f32>>::with_defaults();
        engine.record_with_timestamp(&vec![0.5; 64], vec![0.1; 32], &vec![0.3; 64], 0.5, 12345);
        assert_eq!(engine.memory_size(), 1);
    }

    #[test]
    fn test_record_with_timestamp_below_threshold_rejected() {
        let mut engine = DreamEngine::<Vec<f32>>::with_defaults();
        engine.record_with_timestamp(&vec![0.5; 64], vec![0.1; 32], &vec![0.3; 64], 0.05, 12345);
        assert_eq!(engine.memory_size(), 0);
        assert_eq!(engine.stats().events_rejected, 1);
    }

    #[test]
    fn test_dream_empty_memory_returns_default() {
        let mut engine = DreamEngine::<Vec<f32>>::with_defaults();
        let result = engine.dream().unwrap();
        assert_eq!(result.events_processed, 0);
        assert_eq!(result.simulations_run, 0);
        assert_eq!(result.insights, 0);
    }

    #[test]
    fn test_clear_memory() {
        let mut engine = DreamEngine::<Vec<f32>>::with_defaults();
        for _ in 0..5 {
            engine.record(&vec![0.5; 64], vec![0.1; 32], &vec![0.3; 64], 0.8);
        }
        assert_eq!(engine.memory_size(), 5);
        engine.clear_memory();
        assert_eq!(engine.memory_size(), 0);
    }

    #[test]
    fn test_phi_estimation_empty_input() {
        assert_eq!(DreamEngine::<Vec<f32>>::estimate_phi(&[]), 0.0);
    }

    #[test]
    fn test_phi_estimation_single_element() {
        let phi = DreamEngine::<Vec<f32>>::estimate_phi(&[0.5]);
        assert!(phi >= 0.0 && phi <= 1.0);
    }

    #[test]
    fn test_phi_estimation_high_variance_higher() {
        let low_var = vec![0.5; 64];
        let high_var: Vec<f32> = (0..64)
            .map(|i| if i % 2 == 0 { 0.9 } else { -0.9 })
            .collect();
        let phi_low = DreamEngine::<Vec<f32>>::estimate_phi(&low_var);
        let phi_high = DreamEngine::<Vec<f32>>::estimate_phi(&high_var);
        assert!(phi_high > phi_low, "Higher variance should give higher phi");
    }

    #[test]
    fn test_dream_processes_most_surprising_event() {
        let mut engine = DreamEngine::<Vec<f32>>::with_defaults();
        for i in 0..5 {
            engine.record(
                &vec![(i as f32) / 5.0; 64],
                vec![0.1; 32],
                &vec![0.3; 64],
                0.2 + i as f32 * 0.15,
            );
        }
        let result = engine.dream().unwrap();
        assert_eq!(result.events_processed, 1);
        assert_eq!(result.simulations_run, engine.config().counterfactual_count);
    }

    #[test]
    fn test_wisdom_accumulates() {
        let config = DreamEngineConfig {
            counterfactual_count: 20,
            wisdom_threshold: 0.001,
            ..Default::default()
        };
        let mut engine = DreamEngine::<Vec<f32>>::new(config);
        for i in 0..10 {
            let state: Vec<f32> = (0..64)
                .map(|j| ((i * 7 + j) as f32 / 100.0).sin())
                .collect();
            let action: Vec<f32> = (0..32).map(|j| ((i * 3 + j) as f32 / 50.0).cos()).collect();
            let outcome: Vec<f32> = (0..64)
                .map(|j| ((i * 11 + j) as f32 / 80.0).sin().abs())
                .collect();
            engine.record(&state, action, &outcome, 0.3 + (i as f32) * 0.05);
        }
        engine.dream_session(5).unwrap();
        assert_eq!(engine.stats().dream_cycles, 5);
        assert!(engine.stats().total_simulations > 0);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let engine = DreamEngine::<Vec<f32>>::with_defaults();
        let a = vec![1.0, 2.0, 3.0];
        let sim = engine.cosine_similarity(&a, &a);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "sim(a,a) should be 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let engine = DreamEngine::<Vec<f32>>::with_defaults();
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = engine.cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "Orthogonal vectors should have sim ~0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let engine = DreamEngine::<Vec<f32>>::with_defaults();
        let a = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = a.iter().map(|x| -x).collect();
        let sim = engine.cosine_similarity(&a, &b);
        assert!(
            (sim - (-1.0)).abs() < 1e-6,
            "sim(a,-a) should be -1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let engine = DreamEngine::<Vec<f32>>::with_defaults();
        let a = vec![1.0, 2.0, 3.0];
        let zero = vec![0.0, 0.0, 0.0];
        let sim = engine.cosine_similarity(&a, &zero);
        assert_eq!(sim, 0.0, "Zero vector should return 0.0");
    }

    #[test]
    fn test_ei_estimation_empty() {
        let ei = DreamEngine::<Vec<f32>>::estimate_ei(&[]);
        assert_eq!(ei, 0.0, "EI of empty should be 0.0");
    }

    #[test]
    fn test_ei_estimation_deterministic_high() {
        // Uniform signal = zero variance = high EI
        let uniform = vec![0.8; 64];
        let ei = DreamEngine::<Vec<f32>>::estimate_ei(&uniform);
        assert!(
            ei > 0.9,
            "Uniform signal should have high EI (low variance), got {}",
            ei
        );
    }

    #[test]
    fn test_wisdom_confidence_bounded() {
        let config = DreamEngineConfig {
            counterfactual_count: 20,
            wisdom_threshold: 0.001,
            ..Default::default()
        };
        let mut engine = DreamEngine::<Vec<f32>>::new(config);
        for i in 0..10 {
            let state: Vec<f32> = (0..64)
                .map(|j| ((i * 7 + j) as f32 / 100.0).sin())
                .collect();
            let action: Vec<f32> = (0..32).map(|j| ((i * 3 + j) as f32 / 50.0).cos()).collect();
            let outcome: Vec<f32> = (0..64)
                .map(|j| ((i * 11 + j) as f32 / 80.0).sin().abs())
                .collect();
            engine.record(&state, action, &outcome, 0.5 + (i as f32) * 0.05);
        }
        engine.dream_session(5).unwrap();
        for w in engine.wisdom() {
            assert!(
                w.confidence >= 0.0 && w.confidence <= 1.0,
                "Wisdom confidence {} not in [0, 1]",
                w.confidence
            );
        }
    }

    #[test]
    fn test_world_model_observation_count() {
        let mut engine = DreamEngine::<Vec<f32>>::with_defaults();
        for i in 0..5 {
            engine.record(
                &vec![i as f32 * 0.1; 64],
                vec![0.1; 32],
                &vec![0.3; 64],
                0.5,
            );
        }
        assert_eq!(
            engine.world_model.observations.len(),
            5,
            "World model should accumulate observations"
        );
    }

    #[test]
    fn test_predict_outcome_fallback() {
        let engine = DreamEngine::<Vec<f32>>::with_defaults();
        // No prior observations => fallback to heuristic
        let state = vec![0.5; 64];
        let action = vec![0.2; 32];
        let outcome = engine.simulate_outcome(&state, &action);
        assert_eq!(
            outcome.len(),
            state.len(),
            "Predicted outcome should match state length"
        );
        // All values should be bounded
        for v in &outcome {
            assert!(
                *v >= -1.0 && *v <= 1.0,
                "Predicted value {} out of bounds",
                v
            );
        }
    }

    #[test]
    fn test_dream_stats_event_count() {
        let mut engine = DreamEngine::<Vec<f32>>::with_defaults();
        engine.record(&vec![0.5; 64], vec![0.1; 32], &vec![0.3; 64], 0.5);
        engine.record(&vec![0.6; 64], vec![0.2; 32], &vec![0.4; 64], 0.6);
        engine.record(&vec![0.7; 64], vec![0.3; 32], &vec![0.5; 64], 0.05); // rejected
        assert_eq!(engine.stats().events_recorded, 2);
        assert_eq!(engine.stats().events_rejected, 1);
    }

    #[test]
    fn test_action_perturbation_bounded() {
        let action = vec![0.5; 32];
        let perturbed = action.perturb(42);
        assert_eq!(perturbed.len(), action.len());
        for v in &perturbed {
            assert!(
                *v >= -1.0 && *v <= 1.0,
                "Perturbed value {} out of [-1,1]",
                v
            );
        }
    }

    #[test]
    fn test_most_surprising_event_selection() {
        let mut engine = DreamEngine::<Vec<f32>>::with_defaults();
        engine.record(&vec![0.1; 64], vec![0.1; 32], &vec![0.1; 64], 0.3);
        engine.record(&vec![0.2; 64], vec![0.2; 32], &vec![0.2; 64], 0.9); // most surprising
        engine.record(&vec![0.3; 64], vec![0.3; 32], &vec![0.3; 64], 0.5);
        let idx = engine.find_most_surprising_event();
        assert_eq!(idx, 1, "Should select event with highest surprise");
    }
}
