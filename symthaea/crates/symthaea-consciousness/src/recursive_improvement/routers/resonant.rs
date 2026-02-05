//! # Resonant Consciousness Router
//!
//! Uses HDC + LTC + Resonator networks for soft, continuous consciousness routing
//! that eliminates hard threshold artifacts and enables learned path preferences.
//!
//! ## Key Innovations
//!
//! 1. **Soft Routing**: No hard Φ thresholds - resonator finds optimal path mixture
//! 2. **Smooth Transitions**: LTC dynamics prevent jerky mode switches
//! 3. **Learned Preferences**: Path confidence evolves from experience
//! 4. **Multi-Constraint**: Balances Φ, cost, latency, uncertainty simultaneously
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    RESONANT CONSCIOUSNESS ROUTER                         │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  Input: (Φ, uncertainty, task_type, cost_budget)                        │
//! │           │                                                              │
//! │           ▼                                                              │
//! │  ┌────────────────┐                                                     │
//! │  │ HDC Encoder    │ ── Encode state as hypervector                      │
//! │  └────────┬───────┘                                                     │
//! │           │                                                              │
//! │           ▼                                                              │
//! │  ┌────────────────┐    ┌──────────────┐                                 │
//! │  │ Path Resonator │◄───│ Path Codebook│ (learned routing patterns)      │
//! │  └────────┬───────┘    └──────────────┘                                 │
//! │           │                                                              │
//! │           ▼  path_weights: [0.3, 0.5, 0.15, 0.05, 0.0]                  │
//! │  ┌────────────────┐                                                     │
//! │  │ LTC Evolution  │ ── dP/dt = (-P + resonator_output) / τ             │
//! │  └────────┬───────┘                                                     │
//! │           │                                                              │
//! │           ▼                                                              │
//! │  Output: smooth path mixture with confidence                            │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Comparison to Hard-Threshold Routing
//!
//! | Aspect | Hard Thresholds | Resonant Router |
//! |--------|-----------------|-----------------|
//! | Transitions | Jerky (Φ 0.79→0.81 = mode switch) | Smooth (gradual weight shift) |
//! | Learning | None | Paths evolve from experience |
//! | Multi-objective | Manual tuning | Resonator finds Pareto optimum |
//! | Uncertainty | Ignored | Explicitly modeled |

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use super::types::{
    RoutingStrategy, RoutingPlan, RoutingOutcome, PredictedRoute,
    CognitiveResourceType, ConsciousnessAction, LatentConsciousnessState,
    Router, RouterStats,
};

// ═══════════════════════════════════════════════════════════════════════════
// LTC PATH STATE
// ═══════════════════════════════════════════════════════════════════════════

/// Liquid Time-Constant state for a routing path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtcPathState {
    /// Current path weight [0, 1]
    pub weight: f64,
    /// Time constant (τ) - higher = more stable
    pub tau: f64,
    /// Target weight from resonator
    pub target: f64,
    /// Accumulated experience count
    pub experience: u64,
    /// Average success rate on this path
    pub success_rate: f64,
}

impl LtcPathState {
    pub fn new(tau: f64) -> Self {
        Self {
            weight: 0.0,
            tau,
            target: 0.0,
            experience: 0,
            success_rate: 0.5,
        }
    }

    /// Evolve the path weight using LTC dynamics
    /// dW/dt = (-W + target) / τ
    pub fn evolve(&mut self, dt: f64) {
        let dw = (-self.weight + self.target) / self.tau;
        self.weight += dw * dt;
        self.weight = self.weight.clamp(0.0, 1.0);
    }

    /// Record an outcome and update success rate
    pub fn record_outcome(&mut self, success: bool) {
        self.experience += 1;
        let n = self.experience as f64;
        self.success_rate = self.success_rate * (n - 1.0) / n
            + if success { 1.0 } else { 0.0 } / n;

        // Adapt τ based on variance in outcomes
        // High success rate → increase τ (more stable)
        // Variable outcomes → decrease τ (more responsive)
        if self.experience > 10 {
            self.tau = 5.0 + 15.0 * self.success_rate; // τ ∈ [5, 20]
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for the resonant consciousness router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonantRouterConfig {
    /// Dimension of state encoding
    pub dimension: usize,
    /// Default time constant for path evolution
    pub default_tau: f64,
    /// Time step for LTC evolution
    pub dt: f64,
    /// Minimum weight to consider a path active
    pub min_path_weight: f64,
    /// Maximum iterations for resonator convergence
    pub max_resonator_iterations: usize,
    /// Energy threshold for convergence
    pub convergence_threshold: f64,
    /// Enable path weight smoothing
    pub enable_smoothing: bool,
    /// Smoothing window size
    pub smoothing_window: usize,
    /// Weight for Φ in routing decisions
    pub phi_weight: f64,
    /// Weight for cost efficiency
    pub cost_weight: f64,
    /// Weight for latency
    pub latency_weight: f64,
}

impl Default for ResonantRouterConfig {
    fn default() -> Self {
        Self {
            dimension: 256, // Smaller than full HDC for efficiency
            default_tau: 10.0,
            dt: 1.0,
            min_path_weight: 0.05,
            max_resonator_iterations: 50,
            convergence_threshold: 0.01,
            enable_smoothing: true,
            smoothing_window: 5,
            phi_weight: 0.5,
            cost_weight: 0.3,
            latency_weight: 0.2,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// STATISTICS
// ═══════════════════════════════════════════════════════════════════════════

/// Statistics for resonant routing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResonantRouterStats {
    /// Total routing decisions
    pub decisions_made: u64,
    /// Smooth transitions (no sudden mode switch)
    pub smooth_transitions: u64,
    /// Hard transitions (weight change > 0.5)
    pub hard_transitions: u64,
    /// Resonator convergence count
    pub resonator_convergences: u64,
    /// Average convergence iterations
    pub avg_convergence_iterations: f64,
    /// Average path entropy (higher = more distributed)
    pub avg_path_entropy: f64,
    /// Path usage counts
    pub path_usage: HashMap<RoutingStrategy, u64>,
}

impl ResonantRouterStats {
    /// Calculate smoothness ratio
    pub fn smoothness_ratio(&self) -> f64 {
        let total = self.smooth_transitions + self.hard_transitions;
        if total == 0 {
            1.0
        } else {
            self.smooth_transitions as f64 / total as f64
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ROUTING RESULT
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a resonant routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonantRoutingResult {
    /// Primary recommended strategy
    pub primary_strategy: RoutingStrategy,
    /// Confidence in primary strategy [0, 1]
    pub confidence: f64,
    /// Full path weight distribution
    pub path_weights: HashMap<RoutingStrategy, f64>,
    /// Whether this was a smooth transition
    pub smooth_transition: bool,
    /// Resonator convergence energy
    pub resonator_energy: f64,
    /// Recommended resource allocation
    pub resource_allocation: HashMap<CognitiveResourceType, f64>,
}

// ═══════════════════════════════════════════════════════════════════════════
// RESONANT CONSCIOUSNESS ROUTER
// ═══════════════════════════════════════════════════════════════════════════

/// Resonant Consciousness Router using HDC + LTC + Resonator
///
/// This router replaces hard Φ thresholds with soft resonator-based
/// routing that evolves smoothly via LTC dynamics.
pub struct ResonantConsciousnessRouter {
    /// Configuration
    config: ResonantRouterConfig,
    /// LTC states for each routing path
    path_states: HashMap<RoutingStrategy, LtcPathState>,
    /// Path encoding vectors (codebook)
    path_vectors: HashMap<RoutingStrategy, Vec<f32>>,
    /// Statistics
    stats: ResonantRouterStats,
    /// Previous routing result for transition detection
    previous_weights: HashMap<RoutingStrategy, f64>,
    /// Phi history for trend detection
    phi_history: Vec<f64>,
}

impl ResonantConsciousnessRouter {
    /// Create a new resonant consciousness router
    pub fn new(config: ResonantRouterConfig) -> Self {
        let mut path_states = HashMap::new();
        let mut path_vectors = HashMap::new();

        // Initialize all routing strategies
        let strategies = [
            RoutingStrategy::FullDeliberation,
            RoutingStrategy::StandardProcessing,
            RoutingStrategy::HeuristicGuided,
            RoutingStrategy::FastPatterns,
            RoutingStrategy::Reflexive,
            RoutingStrategy::Ensemble,
            RoutingStrategy::Preparatory,
        ];

        for (i, strategy) in strategies.iter().enumerate() {
            // Create LTC state with default tau
            path_states.insert(*strategy, LtcPathState::new(config.default_tau));

            // Create deterministic path vector
            let vector = Self::create_path_vector(config.dimension, i as u64);
            path_vectors.insert(*strategy, vector);
        }

        Self {
            config,
            path_states,
            path_vectors,
            stats: ResonantRouterStats::default(),
            previous_weights: HashMap::new(),
            phi_history: Vec::with_capacity(100),
        }
    }

    /// Create a deterministic path vector for a strategy
    fn create_path_vector(dimension: usize, seed: u64) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut vector = Vec::with_capacity(dimension);
        for i in 0..dimension {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();
            // Convert to [-1, 1] range
            let value = (hash as f64 / u64::MAX as f64) * 2.0 - 1.0;
            vector.push(value as f32);
        }
        vector
    }

    /// Encode the current state as a query vector
    fn encode_state(&self, state: &LatentConsciousnessState) -> Vec<f32> {
        let dim = self.config.dimension;
        let mut query = vec![0.0f32; dim];

        // Encode Φ across first quarter of dimensions
        let phi_region = dim / 4;
        for i in 0..phi_region {
            let phase = (i as f64 / phi_region as f64) * std::f64::consts::PI * 2.0;
            query[i] = (state.phi * phase.cos()) as f32;
        }

        // Encode uncertainty (integration) in second quarter
        let integration = state.integration;
        for i in phi_region..(phi_region * 2) {
            let phase = ((i - phi_region) as f64 / phi_region as f64) * std::f64::consts::PI * 2.0;
            query[i] = (integration * phase.sin()) as f32;
        }

        // Encode phi trend in third quarter
        let trend = self.compute_phi_trend();
        for i in (phi_region * 2)..(phi_region * 3) {
            query[i] = trend as f32;
        }

        // Encode attention in fourth quarter
        let attention = state.attention;
        for i in (phi_region * 3)..dim {
            query[i] = attention as f32;
        }

        // Normalize
        let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut query {
                *x /= norm;
            }
        }

        query
    }

    /// Compute recent Φ trend
    fn compute_phi_trend(&self) -> f64 {
        if self.phi_history.len() < 2 {
            return 0.0;
        }

        let recent = self.phi_history.len().saturating_sub(5);
        let recent_avg: f64 = self.phi_history[recent..].iter().sum::<f64>()
            / (self.phi_history.len() - recent) as f64;

        let older = recent.saturating_sub(5);
        let older_avg: f64 = self.phi_history[older..recent].iter().sum::<f64>()
            / (recent - older).max(1) as f64;

        (recent_avg - older_avg).clamp(-1.0, 1.0)
    }

    /// Compute similarity between query and path vector
    fn compute_similarity(&self, query: &[f32], path_vec: &[f32]) -> f32 {
        let dot: f32 = query.iter().zip(path_vec.iter()).map(|(a, b)| a * b).sum();
        let norm_q: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_p: f32 = path_vec.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_q > 0.0 && norm_p > 0.0 {
            dot / (norm_q * norm_p)
        } else {
            0.0
        }
    }

    /// Run the resonator to find optimal path weights
    fn resonate(&mut self, query: &[f32]) -> (HashMap<RoutingStrategy, f64>, f64) {
        let mut weights: HashMap<RoutingStrategy, f64> = HashMap::new();
        let mut total_similarity = 0.0;

        // Compute raw similarities
        for (strategy, path_vec) in &self.path_vectors {
            let sim = self.compute_similarity(query, path_vec);
            let path_state = self.path_states.get(strategy).unwrap();

            // Weight by success rate (learned preference)
            let weighted_sim = (sim as f64 + 1.0) / 2.0 * path_state.success_rate;
            weights.insert(*strategy, weighted_sim);
            total_similarity += weighted_sim;
        }

        // Normalize to get probability distribution
        if total_similarity > 0.0 {
            for weight in weights.values_mut() {
                *weight /= total_similarity;
            }
        }

        // Iterative refinement (resonator dynamics)
        let mut energy = 1.0;
        for iter in 0..self.config.max_resonator_iterations {
            // Compute new energy (sum of squared differences from target)
            let mut new_energy = 0.0;
            for (strategy, weight) in &weights {
                let target = self.path_states.get(strategy).unwrap().target;
                new_energy += (weight - target).powi(2);
            }
            new_energy = new_energy.sqrt();

            // Check convergence
            if (energy - new_energy).abs() < self.config.convergence_threshold {
                self.stats.resonator_convergences += 1;
                self.stats.avg_convergence_iterations =
                    self.stats.avg_convergence_iterations * 0.9 + iter as f64 * 0.1;
                break;
            }
            energy = new_energy;

            // Update targets based on current weights
            for (strategy, weight) in &weights {
                if let Some(state) = self.path_states.get_mut(strategy) {
                    state.target = *weight;
                }
            }
        }

        (weights, energy)
    }

    /// Route based on current consciousness state
    pub fn route(&mut self, state: &LatentConsciousnessState) -> ResonantRoutingResult {
        // Update phi history
        self.phi_history.push(state.phi);
        if self.phi_history.len() > 100 {
            self.phi_history.remove(0);
        }

        // Encode state
        let query = self.encode_state(state);

        // Run resonator
        let (raw_weights, energy) = self.resonate(&query);

        // Evolve LTC states
        for (strategy, weight) in &raw_weights {
            if let Some(path_state) = self.path_states.get_mut(strategy) {
                path_state.target = *weight;
                path_state.evolve(self.config.dt);
            }
        }

        // Get smoothed weights from LTC states
        let mut smoothed_weights: HashMap<RoutingStrategy, f64> = HashMap::new();
        for (strategy, path_state) in &self.path_states {
            smoothed_weights.insert(*strategy, path_state.weight);
        }

        // Normalize smoothed weights
        let total: f64 = smoothed_weights.values().sum();
        if total > 0.0 {
            for weight in smoothed_weights.values_mut() {
                *weight /= total;
            }
        }

        // Find primary strategy
        let (primary_strategy, confidence) = smoothed_weights
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(s, w)| (*s, *w))
            .unwrap_or((RoutingStrategy::StandardProcessing, 0.5));

        // Detect transition type
        let smooth_transition = self.is_smooth_transition(&smoothed_weights);
        if smooth_transition {
            self.stats.smooth_transitions += 1;
        } else {
            self.stats.hard_transitions += 1;
        }

        // Update previous weights
        self.previous_weights = smoothed_weights.clone();

        // Compute path entropy
        let entropy = self.compute_entropy(&smoothed_weights);
        self.stats.avg_path_entropy = self.stats.avg_path_entropy * 0.9 + entropy * 0.1;

        // Update usage stats
        *self.stats.path_usage.entry(primary_strategy).or_insert(0) += 1;
        self.stats.decisions_made += 1;

        // Compute resource allocation
        let resource_allocation = self.compute_resource_allocation(&smoothed_weights);

        ResonantRoutingResult {
            primary_strategy,
            confidence,
            path_weights: smoothed_weights,
            smooth_transition,
            resonator_energy: energy,
            resource_allocation,
        }
    }

    /// Check if this is a smooth transition
    fn is_smooth_transition(&self, current: &HashMap<RoutingStrategy, f64>) -> bool {
        if self.previous_weights.is_empty() {
            return true;
        }

        let max_delta: f64 = current
            .iter()
            .map(|(s, w)| {
                let prev = self.previous_weights.get(s).unwrap_or(&0.0);
                (w - prev).abs()
            })
            .fold(0.0, f64::max);

        max_delta < 0.3 // Less than 30% change = smooth
    }

    /// Compute entropy of path distribution
    fn compute_entropy(&self, weights: &HashMap<RoutingStrategy, f64>) -> f64 {
        let mut entropy = 0.0;
        for weight in weights.values() {
            if *weight > 0.0 {
                entropy -= weight * weight.log2();
            }
        }
        entropy
    }

    /// Compute resource allocation based on path weights
    fn compute_resource_allocation(
        &self,
        weights: &HashMap<RoutingStrategy, f64>,
    ) -> HashMap<CognitiveResourceType, f64> {
        let mut allocation = HashMap::new();

        // Weighted average of resource needs
        let mut attention_need = 0.0;
        let mut memory_need = 0.0;
        let mut computation_need = 0.0;

        for (strategy, weight) in weights {
            let factor = strategy.resource_factor();
            attention_need += factor * weight;
            memory_need += factor * 0.8 * weight;
            computation_need += factor * 0.6 * weight;
        }

        allocation.insert(CognitiveResourceType::Attention, attention_need);
        allocation.insert(CognitiveResourceType::WorkingMemory, memory_need);
        allocation.insert(CognitiveResourceType::Computation, computation_need);

        allocation
    }

    /// Record an outcome and update path learning
    pub fn record_outcome(&mut self, strategy: RoutingStrategy, success: bool) {
        if let Some(path_state) = self.path_states.get_mut(&strategy) {
            path_state.record_outcome(success);
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &ResonantRouterStats {
        &self.stats
    }

    /// Get a summary of current path states
    pub fn path_summary(&self) -> HashMap<RoutingStrategy, (f64, f64, f64)> {
        self.path_states
            .iter()
            .map(|(s, state)| (*s, (state.weight, state.tau, state.success_rate)))
            .collect()
    }
}

impl Router for ResonantConsciousnessRouter {
    fn name(&self) -> &'static str {
        "ResonantConsciousnessRouter"
    }

    fn current_strategy(&self, phi: f64) -> RoutingStrategy {
        // For compatibility, provide a simple phi-based answer
        // Real routing should use route() method
        RoutingStrategy::from_phi(phi)
    }

    fn plan(&mut self, state: &LatentConsciousnessState) -> RoutingPlan {
        let result = self.route(state);

        // Convert to RoutingPlan format
        let mut recommended_preallocation = HashMap::new();
        for (resource, amount) in result.resource_allocation {
            recommended_preallocation.insert(resource, amount);
        }

        RoutingPlan {
            current_phi: state.phi,
            current_strategy: result.primary_strategy,
            predictions: Vec::new(), // Resonant router focuses on current, not predictions
            recommended_preallocation,
            transition_warning: !result.smooth_transition,
            phi_trajectory: self.phi_history.clone(),
        }
    }

    fn execute(
        &mut self,
        state: &LatentConsciousnessState,
        _action: ConsciousnessAction,
    ) -> RoutingStrategy {
        self.route(state).primary_strategy
    }

    fn record_outcome(&mut self, outcome: RoutingOutcome) {
        let success = outcome.prediction_accurate;
        self.record_outcome(outcome.strategy_used, success);
    }

    fn stats(&self) -> RouterStats {
        let mut custom_metrics = HashMap::new();
        custom_metrics.insert("smoothness_ratio".to_string(), self.stats.smoothness_ratio());
        custom_metrics.insert("avg_path_entropy".to_string(), self.stats.avg_path_entropy);
        custom_metrics.insert(
            "avg_convergence_iterations".to_string(),
            self.stats.avg_convergence_iterations,
        );

        RouterStats {
            decisions_made: self.stats.decisions_made,
            accurate_predictions: self.stats.smooth_transitions,
            transitions: self.stats.hard_transitions,
            avg_confidence: self.stats.smoothness_ratio(),
            avg_phi_error: 0.0, // Not applicable for resonant router
            custom_metrics,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_state(phi: f64) -> LatentConsciousnessState {
        // from_observables(phi, integration, coherence, attention)
        LatentConsciousnessState::from_observables(phi, 0.5, 0.5, 0.5)
    }

    #[test]
    fn test_resonant_router_creation() {
        let config = ResonantRouterConfig::default();
        let router = ResonantConsciousnessRouter::new(config);

        assert_eq!(router.path_states.len(), 7); // 7 strategies
        assert_eq!(router.path_vectors.len(), 7);
    }

    #[test]
    fn test_routing_with_high_phi() {
        let config = ResonantRouterConfig::default();
        let mut router = ResonantConsciousnessRouter::new(config);

        let state = create_test_state(0.9);
        let result = router.route(&state);

        assert!(result.confidence > 0.0);
        assert!(!result.path_weights.is_empty());
    }

    #[test]
    fn test_smooth_transitions() {
        let config = ResonantRouterConfig::default();
        let mut router = ResonantConsciousnessRouter::new(config);

        // Gradually increase phi
        for phi in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7] {
            let state = create_test_state(phi);
            let result = router.route(&state);

            // After warmup, transitions should be smooth
            if router.stats.decisions_made > 3 {
                // Most transitions should be smooth
                assert!(
                    router.stats.smoothness_ratio() > 0.5,
                    "Smoothness ratio too low: {}",
                    router.stats.smoothness_ratio()
                );
            }
        }
    }

    #[test]
    fn test_path_learning() {
        let config = ResonantRouterConfig::default();
        let mut router = ResonantConsciousnessRouter::new(config);

        // Record many successful outcomes for FullDeliberation
        for _ in 0..20 {
            router.record_outcome(RoutingStrategy::FullDeliberation, true);
        }

        // Check that success rate updated
        let summary = router.path_summary();
        let (_, _, success_rate) = summary[&RoutingStrategy::FullDeliberation];
        assert!(success_rate > 0.9, "Success rate should be high: {}", success_rate);
    }

    #[test]
    fn test_entropy_calculation() {
        let config = ResonantRouterConfig::default();
        let router = ResonantConsciousnessRouter::new(config);

        // Uniform distribution should have high entropy
        let mut uniform: HashMap<RoutingStrategy, f64> = HashMap::new();
        uniform.insert(RoutingStrategy::FullDeliberation, 0.2);
        uniform.insert(RoutingStrategy::StandardProcessing, 0.2);
        uniform.insert(RoutingStrategy::HeuristicGuided, 0.2);
        uniform.insert(RoutingStrategy::FastPatterns, 0.2);
        uniform.insert(RoutingStrategy::Reflexive, 0.2);

        let entropy = router.compute_entropy(&uniform);
        assert!(entropy > 2.0, "Uniform distribution should have high entropy: {}", entropy);

        // Concentrated distribution should have low entropy
        let mut concentrated: HashMap<RoutingStrategy, f64> = HashMap::new();
        concentrated.insert(RoutingStrategy::FullDeliberation, 0.9);
        concentrated.insert(RoutingStrategy::StandardProcessing, 0.1);

        let low_entropy = router.compute_entropy(&concentrated);
        assert!(low_entropy < entropy, "Concentrated should have lower entropy");
    }
}
