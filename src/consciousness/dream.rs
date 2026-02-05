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

```rust
use symthaea::consciousness::dream::{DreamEngine, DreamEngineConfig};

let config = DreamEngineConfig::default();
let mut engine = DreamEngine::new(config);

// Record a surprising event during wakefulness
let state = vec![0.5; 64];
let action = vec![0.1; 32];
let outcome = vec![0.3; 64];
engine.record(&state, &action, &outcome, 0.5); // surprise = 0.5

// Run a dream cycle during sleep/idle
let insights = engine.dream().unwrap();
println!("Gained {} insights from dreaming", insights);
```
*/

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Configuration for the dream engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamEngineConfig {
    /// Minimum surprise threshold for recording events (default: 0.1)
    pub surprise_threshold: f32,
    /// Number of counterfactual simulations per dream cycle (default: 5)
    pub counterfactual_count: usize,
    /// Phi improvement threshold for wisdom consolidation (default: 0.05)
    pub wisdom_threshold: f32,
    /// Maximum events to store in memory (default: 1000)
    pub max_memory_size: usize,
    /// Default dimension for state vectors (default: 64)
    pub state_dim: usize,
    /// Default dimension for action vectors (default: 32)
    pub action_dim: usize,
}

impl Default for DreamEngineConfig {
    fn default() -> Self {
        Self {
            surprise_threshold: 0.1,
            counterfactual_count: 5,
            wisdom_threshold: 0.05,
            max_memory_size: 1000,
            state_dim: 64,
            action_dim: 32,
        }
    }
}

/// A recorded event from waking experience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamEvent {
    /// State at time of event
    pub state: Vec<f32>,
    /// Action taken
    pub action: Vec<f32>,
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
}

/// Consolidated wisdom from dreaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wisdom {
    /// Original state context
    pub context_state: Vec<f32>,
    /// Alternative action that would have been better
    pub better_action: Vec<f32>,
    /// Expected Phi improvement
    pub phi_improvement: f32,
    /// Confidence in this wisdom (0.0-1.0)
    pub confidence: f32,
}

/// Counterfactual Dream Engine
///
/// Simulates alternative pasts to learn from mistakes not made.
/// Records high-surprise events during "wakefulness" and processes
/// them during "sleep" to discover better actions.
#[derive(Debug)]
pub struct DreamEngine {
    /// Configuration
    config: DreamEngineConfig,
    /// Memory of recorded events
    memory: Vec<DreamEvent>,
    /// Consolidated wisdom from dreaming
    wisdom: Vec<Wisdom>,
    /// Statistics
    stats: DreamEngineStats,
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

impl DreamEngine {
    /// Create a new dream engine with the given configuration
    pub fn new(config: DreamEngineConfig) -> Self {
        Self {
            config,
            memory: Vec::new(),
            wisdom: Vec::new(),
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
    pub fn record(&mut self, state: &[f32], action: &[f32], outcome: &[f32], surprise: f32) {
        if surprise > self.config.surprise_threshold {
            // Enforce memory limit
            if self.memory.len() >= self.config.max_memory_size {
                // Remove oldest event
                self.memory.remove(0);
            }

            self.memory.push(DreamEvent {
                state: state.to_vec(),
                action: action.to_vec(),
                actual_outcome: outcome.to_vec(),
                surprise,
                timestamp: Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0)
                ),
            });
            self.stats.events_recorded += 1;
        } else {
            self.stats.events_rejected += 1;
        }
    }

    /// Record an event with explicit timestamp
    pub fn record_with_timestamp(
        &mut self,
        state: &[f32],
        action: &[f32],
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
                action: action.to_vec(),
                actual_outcome: outcome.to_vec(),
                surprise,
                timestamp: Some(timestamp),
            });
            self.stats.events_recorded += 1;
        } else {
            self.stats.events_rejected += 1;
        }
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

        // 1. Evaluate actual Phi (integrated information approximation)
        let actual_phi = Self::estimate_phi(&event.actual_outcome);

        // 2. Generate counterfactual actions and simulate outcomes
        let mut best_improvement = 0.0f32;
        let mut best_action: Option<Vec<f32>> = None;

        for i in 0..self.config.counterfactual_count {
            // Generate alternative action by perturbing the original
            let alt_action = self.generate_counterfactual_action(&event.action, i as u64);

            // Simulate outcome using simple predictive model
            let predicted_outcome = self.simulate_outcome(&event.state, &alt_action);
            let alt_phi = Self::estimate_phi(&predicted_outcome);

            result.simulations_run += 1;
            self.stats.total_simulations += 1;

            // Check if this counterfactual is better
            let improvement = alt_phi - actual_phi;
            if improvement > self.config.wisdom_threshold {
                result.insights += 1;
                self.stats.total_insights += 1;

                if improvement > best_improvement {
                    best_improvement = improvement;
                    best_action = Some(alt_action);
                }
            }
        }

        result.events_processed = 1;
        result.best_phi_improvement = best_improvement;

        // Consolidate wisdom if we found a significantly better action
        if let Some(action) = best_action {
            self.wisdom.push(Wisdom {
                context_state: event.state.clone(),
                better_action: action,
                phi_improvement: best_improvement,
                confidence: (best_improvement / 0.5).min(1.0), // Normalize confidence
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

    /// Get accumulated wisdom
    pub fn wisdom(&self) -> &[Wisdom] {
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
                a.surprise.partial_cmp(&b.surprise).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Generate a counterfactual action by perturbing the original
    fn generate_counterfactual_action(&self, original: &[f32], seed: u64) -> Vec<f32> {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(&seed.to_le_bytes());
        for v in original {
            hasher.update(&v.to_le_bytes());
        }

        let mut bytes = vec![0u8; original.len() * 4];
        let mut xof = hasher.finalize_xof();
        xof.fill(&mut bytes);

        original
            .iter()
            .zip(bytes.chunks_exact(4))
            .map(|(&val, chunk)| {
                let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let noise = (bits as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32;
                // Perturb by 20% max
                (val + noise * 0.2).clamp(-1.0, 1.0)
            })
            .collect()
    }

    /// Simulate outcome given state and action
    ///
    /// This is a simplified world model. In production, this would use
    /// the HierarchicalCfCWorldModel for more accurate predictions.
    fn simulate_outcome(&self, state: &[f32], action: &[f32]) -> Vec<f32> {
        // Simple linear combination as placeholder world model
        // Real implementation would use CfC networks
        let action_influence: f32 = action.iter().map(|a| a.abs()).sum::<f32>() / action.len() as f32;

        state
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                // Add some action-dependent transformation
                let action_component = if i < action.len() {
                    action[i] * 0.3
                } else {
                    0.0
                };
                (s * 0.7 + action_component + action_influence * 0.1).clamp(-1.0, 1.0)
            })
            .collect()
    }

    /// Estimate Phi (integrated information) from outcome vector
    ///
    /// This is an approximation based on:
    /// - Variance (diversity of states)
    /// - Correlation structure (integration)
    /// - Non-linearity (complexity)
    fn estimate_phi(outcome: &[f32]) -> f32 {
        if outcome.is_empty() {
            return 0.0;
        }

        let n = outcome.len() as f32;

        // Mean
        let mean: f32 = outcome.iter().sum::<f32>() / n;

        // Variance (diversity)
        let variance: f32 = outcome.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;

        // Entropy approximation (using variance-based estimate)
        let entropy = if variance > 0.0 {
            0.5 * (1.0 + (2.0 * std::f32::consts::PI * variance).ln())
        } else {
            0.0
        };

        // Integration estimate (consecutive element correlation)
        let mut integration = 0.0f32;
        if outcome.len() > 1 {
            for i in 0..(outcome.len() - 1) {
                integration += (outcome[i] * outcome[i + 1]).abs();
            }
            integration /= (outcome.len() - 1) as f32;
        }

        // Combine into Phi estimate
        // Higher variance + higher integration = higher consciousness


        (entropy.max(0.0) * 0.5 + integration * 0.5).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dream_engine_creation() {
        let engine = DreamEngine::with_defaults();
        assert_eq!(engine.memory_size(), 0);
        assert_eq!(engine.wisdom().len(), 0);
    }

    #[test]
    fn test_event_recording() {
        let mut engine = DreamEngine::with_defaults();

        let state = vec![0.5; 64];
        let action = vec![0.1; 32];
        let outcome = vec![0.3; 64];

        // Low surprise - should be rejected
        engine.record(&state, &action, &outcome, 0.05);
        assert_eq!(engine.memory_size(), 0);
        assert_eq!(engine.stats().events_rejected, 1);

        // High surprise - should be recorded
        engine.record(&state, &action, &outcome, 0.5);
        assert_eq!(engine.memory_size(), 1);
        assert_eq!(engine.stats().events_recorded, 1);
    }

    #[test]
    fn test_dream_cycle() {
        let mut engine = DreamEngine::with_defaults();

        // Record a surprising event
        let state = vec![0.5; 64];
        let action = vec![0.1; 32];
        let outcome = vec![0.3; 64];
        engine.record(&state, &action, &outcome, 0.8);

        // Run dream cycle
        let result = engine.dream().unwrap();
        assert_eq!(result.events_processed, 1);
        assert!(result.simulations_run > 0);
    }

    #[test]
    fn test_phi_estimation() {
        // Zero vector should have low Phi
        let zero_outcome = vec![0.0; 64];
        let phi_zero = DreamEngine::estimate_phi(&zero_outcome);
        assert!(phi_zero < 0.1, "Zero vector should have low Phi");

        // Varied vector should have higher Phi
        let varied_outcome: Vec<f32> = (0..64).map(|i| (i as f32 / 64.0) * 2.0 - 1.0).collect();
        let phi_varied = DreamEngine::estimate_phi(&varied_outcome);
        assert!(phi_varied > phi_zero, "Varied vector should have higher Phi");
    }

    #[test]
    fn test_memory_limit() {
        let config = DreamEngineConfig {
            max_memory_size: 5,
            ..Default::default()
        };
        let mut engine = DreamEngine::new(config);

        // Record more events than the limit
        for i in 0..10 {
            let state = vec![i as f32 / 10.0; 64];
            let action = vec![0.1; 32];
            let outcome = vec![0.3; 64];
            engine.record(&state, &action, &outcome, 0.5);
        }

        assert_eq!(engine.memory_size(), 5, "Memory should be capped at max_memory_size");
    }

    #[test]
    fn test_counterfactual_generation() {
        let engine = DreamEngine::with_defaults();
        let original = vec![0.5; 32];

        let alt1 = engine.generate_counterfactual_action(&original, 0);
        let alt2 = engine.generate_counterfactual_action(&original, 1);

        // Different seeds should produce different actions
        assert_ne!(alt1, alt2, "Different seeds should produce different counterfactuals");

        // Actions should be perturbed but not drastically different
        for (o, a) in original.iter().zip(alt1.iter()) {
            assert!((o - a).abs() < 0.5, "Counterfactual should not be too different from original");
        }
    }

    #[test]
    fn test_dream_session() {
        let mut engine = DreamEngine::with_defaults();

        // Record several events
        for i in 0..5 {
            let state = vec![(i as f32) / 5.0; 64];
            let action = vec![0.1; 32];
            let outcome = vec![0.3; 64];
            engine.record(&state, &action, &outcome, 0.3 + i as f32 * 0.1);
        }

        // Run multiple dream cycles
        let results = engine.dream_session(3).unwrap();
        assert_eq!(results.len(), 3);

        // Stats should reflect the session
        assert_eq!(engine.stats().dream_cycles, 3);
    }
}
