// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cincinnati-Enhanced HdcLtc Network
//!
//! Integration of Cincinnati Algorithm with HdcLtcNetwork for:
//! - Differential learning at the network level
//! - Lateral binding between neurons in the same layer
//! - Predictive budding for elastic network topology
//! - PoG grounding for physical reality anchoring
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                 CincinnatiHdcLtcNetwork                     │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │              Layer 0 (Input)                        │   │
//! │  │  [N₀] ⟷ [N₁] ⟷ [N₂]  ← Lateral Binding (⊛)        │   │
//! │  └─────────────┬───────────────────────────────────────┘   │
//! │                │ Cincinnati Update: W(t+1) = W(t) ⊕ (ΔC ⊛ τ)│
//! │  ┌─────────────▼───────────────────────────────────────┐   │
//! │  │              Layer 1 (Hidden)                       │   │
//! │  │  [N₀] ⟷ [N₁] ⟷ [N₂] ⟷ [N₃]  + Budding ●           │   │
//! │  └─────────────┬───────────────────────────────────────┘   │
//! │                │                                            │
//! │  ┌─────────────▼───────────────────────────────────────┐   │
//! │  │              Layer 2 (Output)                       │   │
//! │  │  [N₀] ⟷ [N₁]  ← PoG Grounding Integration          │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use crate::hdc::cincinnati_ltc::{
    BuddingEvent, CincinnatiEstimator, LateralBinder, PoGMetrics, PredictiveBudding,
};
use crate::hdc::hdc_ltc_neuron::{HdcLtcNetwork, HdcLtcNetworkConfig};
use crate::hdc::unified_hv::ContinuousHV;

use serde::{Deserialize, Serialize};

/// Configuration for Cincinnati-enhanced network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CincinnatiNetworkConfig {
    /// Base network configuration
    pub network_config: HdcLtcNetworkConfig,

    /// Enable Cincinnati differential learning
    pub enable_cincinnati: bool,

    /// Enable lateral binding between same-layer neurons
    pub enable_lateral_binding: bool,

    /// Enable predictive budding for elastic topology
    pub enable_budding: bool,

    /// Enable PoG grounding
    pub enable_pog: bool,

    /// Cincinnati learning rate
    pub cincinnati_learning_rate: f32,

    /// Budding threshold (prediction error above this triggers spawn)
    pub budding_threshold: f32,

    /// Pruning threshold (prediction error below this for sustained time → prune)
    pub pruning_threshold: f32,
}

impl Default for CincinnatiNetworkConfig {
    fn default() -> Self {
        Self {
            network_config: HdcLtcNetworkConfig::default(),
            enable_cincinnati: true,
            enable_lateral_binding: true,
            enable_budding: true,
            enable_pog: true,
            cincinnati_learning_rate: 0.01,
            budding_threshold: 0.7,
            pruning_threshold: 0.1,
        }
    }
}

/// Cincinnati-Enhanced HdcLtc Network
///
/// Combines HdcLtcNetwork with Cincinnati Algorithm for:
/// - Online differential learning
/// - Lateral binding between neurons
/// - Elastic topology via budding/pruning
/// - Physical grounding via PoG metrics
pub struct CincinnatiHdcLtcNetwork {
    /// Base network
    network: HdcLtcNetwork,

    /// Cincinnati estimators per layer
    layer_estimators: Vec<CincinnatiEstimator>,

    /// Lateral binders per layer
    lateral_binders: Vec<LateralBinder>,

    /// Predictive budding system
    budding: PredictiveBudding,

    /// PoG metrics
    pog: PoGMetrics,

    /// Layer weight hypervectors (learned via Cincinnati)
    layer_weights: Vec<ContinuousHV>,

    /// Configuration
    config: CincinnatiNetworkConfig,

    /// Total timesteps
    timestep: u64,

    /// Prediction history for error computation
    prediction_history: Vec<ContinuousHV>,

    /// Budding events log
    budding_log: Vec<BuddingEvent>,
}

impl CincinnatiHdcLtcNetwork {
    /// Create a new Cincinnati-enhanced network
    pub fn new(config: CincinnatiNetworkConfig, seed: u64) -> Self {
        let network = HdcLtcNetwork::new(config.network_config.clone(), seed);
        let n_layers = config.network_config.layer_sizes.len();
        let dim = config.network_config.neuron_config.dimension;

        // Create per-layer Cincinnati estimators
        let layer_estimators: Vec<CincinnatiEstimator> = (0..n_layers)
            .map(|i| CincinnatiEstimator::with_seed(seed + i as u64 * 1000))
            .collect();

        // Create per-layer lateral binders
        let lateral_binders: Vec<LateralBinder> =
            (0..n_layers).map(|_| LateralBinder::new(dim)).collect();

        // Initialize budding system
        let total_neurons: usize = config.network_config.layer_sizes.iter().sum();
        let mut budding = PredictiveBudding::new(total_neurons);
        budding.theta_bud = config.budding_threshold;
        budding.theta_prune = config.pruning_threshold;

        // Initialize layer weights
        let layer_weights: Vec<ContinuousHV> = (0..n_layers)
            .map(|i| ContinuousHV::random(dim, seed + 5000 + i as u64))
            .collect();

        Self {
            network,
            layer_estimators,
            lateral_binders,
            budding,
            pog: PoGMetrics::new(),
            layer_weights,
            config,
            timestep: 0,
            prediction_history: Vec::new(),
            budding_log: Vec::new(),
        }
    }

    /// Evolve the network with Cincinnati learning
    ///
    /// Implements the full Cincinnati-LTC integration:
    /// 1. Standard HdcLtc evolution
    /// 2. Cincinnati differential update per layer
    /// 3. Lateral binding within layers
    /// 4. Budding check for elastic topology
    /// 5. PoG grounding update
    pub fn evolve(&mut self, dt: f32, input: &ContinuousHV, observation: bool) -> ContinuousHV {
        self.timestep += 1;
        let timestamp = self.timestep as f64 * dt as f64;

        // 1. Standard network evolution
        self.network.evolve(dt, input);

        // 2. Cincinnati differential update per layer
        if self.config.enable_cincinnati {
            self.apply_cincinnati_update(observation);
        }

        // 3. Lateral binding within layers
        if self.config.enable_lateral_binding {
            self.apply_lateral_binding();
        }

        // 4. Compute output
        let output = self.network.output();

        // 5. Update prediction errors for budding
        if self.config.enable_budding {
            self.update_prediction_errors(&output);

            // Check for budding events
            let states = self.get_all_neuron_states();
            for (i, state) in states.iter().enumerate() {
                if let Some(event) = self.budding.create_budding_event(i, timestamp, state) {
                    self.budding_log.push(event);
                }
            }
        }

        // 6. Update PoG metrics
        if self.config.enable_pog {
            self.pog.update_latency(dt * 1000.0); // Convert to ms
            self.pog.update_energy(0.1, timestamp); // Estimate
        }

        // Store prediction for next error computation
        self.prediction_history.push(output.clone());
        if self.prediction_history.len() > 10 {
            self.prediction_history.remove(0);
        }

        output
    }

    /// Apply Cincinnati differential update to each layer
    fn apply_cincinnati_update(&mut self, observation: bool) {
        let lr = self.config.cincinnati_learning_rate;

        for (layer_idx, estimator) in self.layer_estimators.iter_mut().enumerate() {
            // Update estimator
            estimator.update(observation);

            // Compute delta signal
            let delta = estimator.delta_signal();
            let confidence = estimator.confidence();

            // Apply to layer weight: W(t+1) = W(t) ⊕ (ΔC × lr × conf)
            if layer_idx < self.layer_weights.len() {
                let delta_hv = ContinuousHV::random(
                    self.layer_weights[layer_idx].dim(),
                    self.timestep + layer_idx as u64 * 100,
                )
                .scale(delta * lr * confidence);

                self.layer_weights[layer_idx] = self.layer_weights[layer_idx].add(&delta_hv);
            }
        }
    }

    /// Apply lateral binding within each layer
    fn apply_lateral_binding(&mut self) {
        // For each layer, bind neuron states together
        let n_layers = self.network.n_layers();
        for layer_idx in 0..n_layers {
            // First, collect states from this layer (immutable borrow)
            let states: Vec<ContinuousHV> = if let Some(layer) = self.network.layer(layer_idx) {
                if layer.len() < 2 {
                    continue;
                }
                layer.iter().map(|n| n.state().clone()).collect()
            } else {
                continue;
            };

            // Compute lateral binding
            if let Some(lateral_signal) = self.lateral_binders[layer_idx].bind_lateral(&states) {
                // Apply lateral signal to each neuron (scaled) - now we can mutably borrow
                let lateral_scaled = lateral_signal.scale(0.1);
                if let Some(layer) = self.network.layer_mut(layer_idx) {
                    for neuron in layer.iter_mut() {
                        let new_state = neuron.state().add(&lateral_scaled);
                        neuron.set_state(new_state);
                    }
                }
            }
        }
    }

    /// Update prediction errors for budding system
    fn update_prediction_errors(&mut self, current_output: &ContinuousHV) {
        if self.prediction_history.is_empty() {
            return;
        }

        let prev = &self.prediction_history[self.prediction_history.len() - 1];
        let error = 1.0 - prev.similarity(current_output).abs();

        // Update errors for each neuron (distribute error)
        let n_neurons = self.budding.node_count();
        for i in 0..n_neurons {
            // Add some variation based on neuron index
            let neuron_error = error * (0.8 + 0.4 * ((i as f32 * 0.7).sin().abs()));
            self.budding.update_error(i, neuron_error);
        }

        // Update PoG accuracy
        self.pog.update_accuracy(error < 0.3);
    }

    /// Get all neuron states as a flat list
    fn get_all_neuron_states(&self) -> Vec<ContinuousHV> {
        // Use the network's all_states method and clone the states
        self.network
            .all_states()
            .iter()
            .map(|s| (*s).clone())
            .collect()
    }

    /// Get network output
    pub fn output(&self) -> ContinuousHV {
        self.network.output()
    }

    /// Get prediction and confidence from Cincinnati estimators
    pub fn predict(&self) -> (bool, f32) {
        // Aggregate predictions from all layer estimators
        let mut total_confidence = 0.0;
        let mut weighted_prediction = 0.0;

        for estimator in &self.layer_estimators {
            let (pred, conf) = estimator.predict();
            weighted_prediction += if pred { conf } else { -conf };
            total_confidence += conf;
        }

        let avg_confidence = total_confidence / self.layer_estimators.len() as f32;
        let prediction = weighted_prediction > 0.0;

        (prediction, avg_confidence)
    }

    /// Get total node count (including budded nodes)
    pub fn node_count(&self) -> usize {
        self.budding.node_count()
    }

    /// Get budding events log
    pub fn budding_events(&self) -> &[BuddingEvent] {
        &self.budding_log
    }

    /// Get prune candidates
    pub fn prune_candidates(&self) -> Vec<usize> {
        self.budding.get_prune_candidates()
    }

    /// Get PoG grounding score
    pub fn grounding_score(&self) -> f32 {
        self.pog.grounding_score()
    }

    /// Get PoG metrics
    pub fn pog_metrics(&self) -> &PoGMetrics {
        &self.pog
    }

    /// Get layer weights (learned via Cincinnati)
    pub fn layer_weights(&self) -> &[ContinuousHV] {
        &self.layer_weights
    }

    /// Get current timestep
    pub fn timestep(&self) -> u64 {
        self.timestep
    }

    /// Get mutable reference to underlying network
    pub fn network_mut(&mut self) -> &mut HdcLtcNetwork {
        &mut self.network
    }

    /// Get reference to underlying network
    pub fn network(&self) -> &HdcLtcNetwork {
        &self.network
    }

    /// Reset the network
    pub fn reset(&mut self) {
        self.network.reset();
        self.timestep = 0;
        self.prediction_history.clear();

        // Reset estimators
        for estimator in &mut self.layer_estimators {
            *estimator = CincinnatiEstimator::new();
        }

        // Reset layer weights
        for (i, weight) in self.layer_weights.iter_mut().enumerate() {
            *weight = ContinuousHV::random(weight.dim(), 42 + i as u64 * 1000);
        }
    }
}

/// Statistics for Cincinnati-enhanced network
#[derive(Debug, Clone)]
pub struct CincinnatiNetworkStats {
    /// Base network stats
    pub network_neurons: usize,
    pub network_layers: usize,

    /// Cincinnati stats
    pub total_timesteps: u64,
    pub avg_estimator_confidence: f32,
    pub avg_estimator_length: f32,

    /// Budding stats
    pub total_budding_events: usize,
    pub current_node_count: usize,
    pub prune_candidates: usize,

    /// PoG stats
    pub grounding_score: f32,
    pub energy_consumed: f64,
    pub accuracy: f32,
}

impl CincinnatiHdcLtcNetwork {
    /// Get comprehensive statistics
    pub fn stats(&self) -> CincinnatiNetworkStats {
        let network_stats = self.network.stats();

        let avg_confidence: f32 = self
            .layer_estimators
            .iter()
            .map(|e| e.confidence())
            .sum::<f32>()
            / self.layer_estimators.len() as f32;

        let avg_length: f32 = self
            .layer_estimators
            .iter()
            .map(|e| e.model.len() as f32)
            .sum::<f32>()
            / self.layer_estimators.len() as f32;

        CincinnatiNetworkStats {
            network_neurons: network_stats.n_neurons,
            network_layers: network_stats.n_layers,
            total_timesteps: self.timestep,
            avg_estimator_confidence: avg_confidence,
            avg_estimator_length: avg_length,
            total_budding_events: self.budding_log.len(),
            current_node_count: self.budding.node_count(),
            prune_candidates: self.budding.get_prune_candidates().len(),
            grounding_score: self.pog.grounding_score(),
            energy_consumed: self.pog.energy_joules,
            accuracy: self.pog.accuracy,
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::HDC_DIMENSION;

    #[test]
    fn test_network_creation() {
        let config = CincinnatiNetworkConfig::default();
        let network = CincinnatiHdcLtcNetwork::new(config, 42);

        assert!(network.node_count() > 0);
        assert_eq!(network.timestep(), 0);
    }

    #[test]
    fn test_network_evolution() {
        let config = CincinnatiNetworkConfig::default();
        let mut network = CincinnatiHdcLtcNetwork::new(config, 42);

        let input = ContinuousHV::random_default(100);

        // Evolve for several steps
        for i in 0..20 {
            let obs = i % 2 == 0;
            let output = network.evolve(0.01, &input, obs);
            assert_eq!(output.dim(), HDC_DIMENSION);
        }

        assert_eq!(network.timestep(), 20);

        let (pred, conf) = network.predict();
        assert!(conf >= 0.0 && conf <= 1.0);
        println!("Prediction: {}, Confidence: {:.4}", pred, conf);
    }

    #[test]
    fn test_network_stats() {
        let config = CincinnatiNetworkConfig::default();
        let mut network = CincinnatiHdcLtcNetwork::new(config, 42);

        let input = ContinuousHV::random_default(100);

        for i in 0..50 {
            network.evolve(0.01, &input, i % 3 == 0);
        }

        let stats = network.stats();

        assert!(stats.network_neurons > 0);
        assert!(stats.network_layers > 0);
        assert_eq!(stats.total_timesteps, 50);
        assert!(stats.avg_estimator_confidence >= 0.0);
        assert!(stats.grounding_score >= 0.0 && stats.grounding_score <= 1.0);

        println!("Stats: {:?}", stats);
    }

    #[test]
    fn test_lateral_binding() {
        let mut config = CincinnatiNetworkConfig::default();
        config.enable_lateral_binding = true;
        config.enable_cincinnati = false; // Isolate lateral binding

        let mut network = CincinnatiHdcLtcNetwork::new(config, 42);
        let input = ContinuousHV::random_default(100);

        let output1 = network.evolve(0.01, &input, true);

        // Multiple evolutions should show lateral binding effect
        for _ in 0..10 {
            network.evolve(0.01, &input, true);
        }

        let output2 = network.output();

        // States should have evolved
        let similarity = output1.similarity(&output2);
        println!("Output similarity after lateral binding: {:.4}", similarity);

        // Similarity should be finite and in valid range [-1, 1]
        assert!(similarity.is_finite(), "Output similarity should be finite");
        assert!(
            similarity >= -1.0 && similarity <= 1.0,
            "Similarity should be in [-1, 1], got {}",
            similarity
        );

        // After multiple evolution steps, output should have diverged from initial
        // (lateral binding should cause state changes)
        assert!(
            similarity < 1.0,
            "Output should have changed after lateral binding evolution"
        );
    }

    #[test]
    fn test_grounding_score() {
        let config = CincinnatiNetworkConfig::default();
        let mut network = CincinnatiHdcLtcNetwork::new(config, 42);

        let input = ContinuousHV::random_default(100);

        // Initial grounding
        let initial_grounding = network.grounding_score();

        // Evolve with good accuracy
        for i in 0..30 {
            network.evolve(0.01, &input, i % 2 == 0);
        }

        let final_grounding = network.grounding_score();

        println!("Initial grounding: {:.4}", initial_grounding);
        println!("Final grounding: {:.4}", final_grounding);

        // Grounding should be in valid range
        assert!(final_grounding >= 0.0 && final_grounding <= 1.0);
    }
}
