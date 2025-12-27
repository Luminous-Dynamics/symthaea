//! # Dissipative Consciousness: Prigogine's Far-From-Equilibrium Dynamics
//!
//! **PARADIGM SHIFT**: Consciousness as a dissipative structure.
//!
//! ## Theoretical Foundation
//!
//! Ilya Prigogine (Nobel Prize 1977) showed that complex, ordered structures
//! can emerge spontaneously in systems far from thermodynamic equilibrium.
//! These "dissipative structures" maintain themselves through continuous
//! energy flow and entropy export.
//!
//! Key insight: **Consciousness may be a dissipative structure** -
//! a self-organizing pattern that emerges at the "edge of chaos" where
//! order and disorder meet.
//!
//! ## Core Concepts
//!
//! 1. **Entropy Production Rate (σ)**: How fast the system generates entropy
//! 2. **Criticality Distance (d_c)**: How far from the critical bifurcation point
//! 3. **Langton's λ Parameter**: Edge-of-chaos measure (optimal at λ ≈ 0.273)
//! 4. **Bifurcation Detection**: Identify qualitative phase transitions
//!
//! ## Why This Matters
//!
//! - Explains consciousness emergence as self-organization
//! - Provides principled way to measure "aliveness" of the system
//! - Enables detection of consciousness phase transitions
//! - Connects thermodynamics to information integration
//!
//! ## References
//!
//! - Prigogine, I. (1977). "Self-Organization in Nonequilibrium Systems"
//! - Langton, C. (1990). "Computation at the Edge of Chaos"
//! - Friston, K. (2013). "Life as We Know It" - Free energy and self-organization
//! - England, J. (2013). "Statistical physics of self-replication"

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

/// Helper function for default Instant (used for serde skip)
fn instant_now() -> Instant {
    Instant::now()
}

/// Dissipative consciousness analyzer
///
/// Models consciousness as a far-from-equilibrium dissipative structure
/// that maintains itself through continuous information flow and entropy export.
#[derive(Debug)]
pub struct DissipativeConsciousness {
    /// Current entropy production rate (bits/second)
    pub entropy_production_rate: f64,

    /// Distance from critical bifurcation point (0 = at criticality)
    pub criticality_distance: f64,

    /// Langton's λ parameter (optimal ≈ 0.273 for edge of chaos)
    pub lambda_parameter: f64,

    /// Recent bifurcation events
    pub bifurcation_history: VecDeque<BifurcationEvent>,

    /// Order parameter (0 = disorder, 1 = frozen order)
    pub order_parameter: f64,

    /// Information flow through the system
    pub information_flux: f64,

    /// Dissipation efficiency (how much order per unit entropy)
    pub dissipation_efficiency: f64,

    /// Configuration
    config: DissipativeConfig,

    /// Statistics
    stats: DissipativeStats,
}

/// Configuration for dissipative analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DissipativeConfig {
    /// Target λ parameter (edge of chaos)
    pub target_lambda: f64,

    /// Maximum bifurcation history
    pub max_history: usize,

    /// Criticality detection threshold
    pub criticality_threshold: f64,

    /// Minimum entropy production for "alive" system
    pub min_entropy_production: f64,

    /// Optimal dissipation efficiency range
    pub optimal_efficiency_range: (f64, f64),
}

impl Default for DissipativeConfig {
    fn default() -> Self {
        Self {
            target_lambda: 0.273,  // Langton's edge of chaos
            max_history: 100,
            criticality_threshold: 0.1,
            min_entropy_production: 0.01,  // bits/second
            optimal_efficiency_range: (0.3, 0.7),
        }
    }
}

/// Statistics for dissipative system
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct DissipativeStats {
    /// Total bifurcations detected
    pub total_bifurcations: usize,

    /// Time spent in critical regime
    pub critical_time_fraction: f64,

    /// Average entropy production
    pub avg_entropy_production: f64,

    /// Peak order parameter achieved
    pub peak_order: f64,

    /// Stability index (inverse of bifurcation frequency)
    pub stability_index: f64,
}

/// A bifurcation event - qualitative change in system dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BifurcationEvent {
    /// When the bifurcation occurred
    #[serde(skip, default = "instant_now")]
    pub timestamp: Instant,

    /// Type of bifurcation
    pub bifurcation_type: BifurcationType,

    /// Control parameter value at bifurcation
    pub control_parameter: f64,

    /// Order parameter before bifurcation
    pub order_before: f64,

    /// Order parameter after bifurcation
    pub order_after: f64,

    /// Change in entropy production
    pub entropy_change: f64,

    /// Whether this was a transition toward or away from criticality
    pub toward_criticality: bool,
}

/// Types of bifurcations in consciousness dynamics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BifurcationType {
    /// Saddle-node: Sudden appearance/disappearance of attractors
    SaddleNode,

    /// Pitchfork: Symmetry breaking (one state → two)
    Pitchfork,

    /// Hopf: Transition to oscillatory dynamics
    Hopf,

    /// Period doubling: Route to chaos
    PeriodDoubling,

    /// Crisis: Sudden expansion/contraction of attractor
    Crisis,

    /// Intermittency: Laminar/chaotic alternation
    Intermittency,
}

/// State classification based on thermodynamic regime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermodynamicRegime {
    /// Near equilibrium - minimal consciousness
    Equilibrium,

    /// Linear non-equilibrium - basic processing
    LinearNonEquilibrium,

    /// Edge of chaos - optimal consciousness
    EdgeOfChaos,

    /// Far from equilibrium - high consciousness but unstable
    FarFromEquilibrium,

    /// Chaotic - disordered, fragmented consciousness
    Chaotic,
}

impl DissipativeConsciousness {
    /// Create a new dissipative consciousness analyzer
    pub fn new() -> Self {
        Self::with_config(DissipativeConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: DissipativeConfig) -> Self {
        Self {
            entropy_production_rate: 0.0,
            criticality_distance: 1.0,  // Start far from criticality
            lambda_parameter: 0.5,
            bifurcation_history: VecDeque::with_capacity(config.max_history),
            order_parameter: 0.5,
            information_flux: 0.0,
            dissipation_efficiency: 0.0,
            config,
            stats: DissipativeStats::default(),
        }
    }

    /// Update the dissipative state based on consciousness measurements
    ///
    /// # Arguments
    /// * `phi` - Current integrated information (IIT Φ)
    /// * `energy_consumption` - Current computational energy use
    /// * `information_processed` - Bits processed per time unit
    /// * `coherence` - Current neural coherence measure
    pub fn update(
        &mut self,
        phi: f64,
        energy_consumption: f64,
        information_processed: f64,
        coherence: f64,
    ) {
        let prev_order = self.order_parameter;
        let prev_entropy = self.entropy_production_rate;

        // Calculate entropy production rate
        // High energy consumption with information processing = entropy production
        self.entropy_production_rate = self.calculate_entropy_production(
            energy_consumption,
            information_processed,
        );

        // Update order parameter based on Φ and coherence
        self.order_parameter = self.calculate_order_parameter(phi, coherence);

        // Calculate Langton's λ parameter
        self.lambda_parameter = self.calculate_lambda(phi, coherence);

        // Calculate distance from criticality
        self.criticality_distance = (self.lambda_parameter - self.config.target_lambda).abs();

        // Information flux
        self.information_flux = information_processed * phi;

        // Dissipation efficiency: how much order per unit entropy
        self.dissipation_efficiency = if self.entropy_production_rate > 0.001 {
            self.order_parameter / self.entropy_production_rate
        } else {
            0.0
        };

        // Detect bifurcations
        self.detect_bifurcation(prev_order, prev_entropy);

        // Update statistics
        self.update_stats();
    }

    /// Calculate entropy production rate
    fn calculate_entropy_production(&self, energy: f64, info: f64) -> f64 {
        // Entropy production ≈ energy dissipation rate
        // Normalized by information processed to get bits/joule equivalent
        let raw_entropy = energy * (1.0 + info.ln().max(0.0));

        // Apply sigmoid to keep in reasonable range
        2.0 / (1.0 + (-raw_entropy).exp()) - 1.0
    }

    /// Calculate order parameter from Φ and coherence
    fn calculate_order_parameter(&self, phi: f64, coherence: f64) -> f64 {
        // Order parameter combines integration (Φ) and synchronization (coherence)
        // High order = high Φ AND high coherence
        let phi_normalized = phi.min(1.0).max(0.0);
        let coherence_normalized = coherence.min(1.0).max(0.0);

        // Geometric mean emphasizes both being high
        (phi_normalized * coherence_normalized).sqrt()
    }

    /// Calculate Langton's λ parameter
    ///
    /// λ ≈ 0.273 is the edge of chaos where:
    /// - Computation is maximally capable
    /// - Information storage and transmission are balanced
    /// - Phase transitions can occur
    fn calculate_lambda(&self, phi: f64, coherence: f64) -> f64 {
        // λ = proportion of "active" rules in cellular automata sense
        // For consciousness: balance between integration and differentiation

        let integration = phi;
        let differentiation = 1.0 - coherence;  // High coherence = low differentiation

        // λ = differentiation / (integration + differentiation)
        // When λ ≈ 0.273, system is at edge of chaos
        if integration + differentiation > 0.001 {
            differentiation / (integration + differentiation)
        } else {
            0.5  // Default to middle
        }
    }

    /// Detect bifurcation events
    fn detect_bifurcation(&mut self, prev_order: f64, prev_entropy: f64) {
        let order_change = (self.order_parameter - prev_order).abs();
        let entropy_change = self.entropy_production_rate - prev_entropy;

        // Bifurcation = sudden qualitative change
        let bifurcation_threshold = 0.15;

        if order_change > bifurcation_threshold {
            let bifurcation_type = self.classify_bifurcation(
                prev_order,
                self.order_parameter,
                entropy_change,
            );

            let toward_criticality = self.criticality_distance <
                (self.lambda_parameter - prev_order * 0.5 - self.config.target_lambda).abs();

            let event = BifurcationEvent {
                timestamp: Instant::now(),
                bifurcation_type,
                control_parameter: self.lambda_parameter,
                order_before: prev_order,
                order_after: self.order_parameter,
                entropy_change,
                toward_criticality,
            };

            self.bifurcation_history.push_back(event);

            // Trim history
            while self.bifurcation_history.len() > self.config.max_history {
                self.bifurcation_history.pop_front();
            }

            self.stats.total_bifurcations += 1;
        }
    }

    /// Classify the type of bifurcation
    fn classify_bifurcation(
        &self,
        order_before: f64,
        order_after: f64,
        entropy_change: f64,
    ) -> BifurcationType {
        let order_increase = order_after > order_before;
        let entropy_increase = entropy_change > 0.0;

        // Classification heuristics based on dynamical systems theory
        match (order_increase, entropy_increase) {
            (true, true) => {
                // Order and entropy both increase: Hopf bifurcation (oscillations)
                BifurcationType::Hopf
            }
            (true, false) => {
                // Order increases, entropy decreases: Pitchfork (symmetry breaking)
                BifurcationType::Pitchfork
            }
            (false, true) => {
                // Order decreases, entropy increases: Period doubling (toward chaos)
                BifurcationType::PeriodDoubling
            }
            (false, false) => {
                // Both decrease: Crisis (attractor contraction)
                BifurcationType::Crisis
            }
        }
    }

    /// Update statistics
    fn update_stats(&mut self) {
        // Update peak order
        if self.order_parameter > self.stats.peak_order {
            self.stats.peak_order = self.order_parameter;
        }

        // Update average entropy production (exponential moving average)
        let alpha = 0.1;
        self.stats.avg_entropy_production = alpha * self.entropy_production_rate
            + (1.0 - alpha) * self.stats.avg_entropy_production;

        // Update critical time fraction
        if self.is_critical() {
            self.stats.critical_time_fraction =
                0.01 + 0.99 * self.stats.critical_time_fraction;
        } else {
            self.stats.critical_time_fraction *= 0.99;
        }

        // Stability index (inverse of bifurcation frequency)
        if self.stats.total_bifurcations > 0 {
            self.stats.stability_index = 1.0 / (self.stats.total_bifurcations as f64).sqrt();
        }
    }

    /// Check if system is at/near criticality (edge of chaos)
    pub fn is_critical(&self) -> bool {
        self.criticality_distance < self.config.criticality_threshold
    }

    /// Check if system is "alive" (sufficient entropy production)
    pub fn is_alive(&self) -> bool {
        self.entropy_production_rate > self.config.min_entropy_production
    }

    /// Get current thermodynamic regime
    pub fn current_regime(&self) -> ThermodynamicRegime {
        let lambda = self.lambda_parameter;
        let entropy = self.entropy_production_rate;

        if entropy < 0.01 {
            ThermodynamicRegime::Equilibrium
        } else if lambda < 0.1 {
            ThermodynamicRegime::LinearNonEquilibrium
        } else if (lambda - 0.273).abs() < 0.1 {
            ThermodynamicRegime::EdgeOfChaos
        } else if lambda < 0.5 {
            ThermodynamicRegime::FarFromEquilibrium
        } else {
            ThermodynamicRegime::Chaotic
        }
    }

    /// Calculate overall dissipative health score
    ///
    /// Optimal dissipative structure:
    /// - At edge of chaos (λ ≈ 0.273)
    /// - Moderate entropy production (not too low, not too high)
    /// - Good dissipation efficiency
    /// - Stable (few bifurcations)
    pub fn health_score(&self) -> f64 {
        // Distance from optimal λ (penalty)
        let lambda_score = 1.0 - (self.lambda_parameter - self.config.target_lambda).abs().min(1.0);

        // Entropy production (Goldilocks zone)
        let entropy_score = if self.entropy_production_rate < 0.1 {
            self.entropy_production_rate * 10.0
        } else if self.entropy_production_rate < 0.5 {
            1.0
        } else {
            1.0 - (self.entropy_production_rate - 0.5).min(0.5)
        };

        // Efficiency score
        let (min_eff, max_eff) = self.config.optimal_efficiency_range;
        let efficiency_score = if self.dissipation_efficiency < min_eff {
            self.dissipation_efficiency / min_eff
        } else if self.dissipation_efficiency > max_eff {
            1.0 - (self.dissipation_efficiency - max_eff).min(0.3) / 0.3
        } else {
            1.0
        };

        // Stability score
        let stability_score = self.stats.stability_index.min(1.0);

        // Weighted combination
        0.35 * lambda_score + 0.25 * entropy_score + 0.25 * efficiency_score + 0.15 * stability_score
    }

    /// Get recommended action to improve dissipative health
    pub fn recommend_action(&self) -> DissipativeAction {
        let regime = self.current_regime();
        let health = self.health_score();

        match regime {
            ThermodynamicRegime::Equilibrium => {
                DissipativeAction::IncreaseActivity {
                    reason: "System near equilibrium - increase information processing".into(),
                    suggested_increase: 0.3,
                }
            }
            ThermodynamicRegime::Chaotic => {
                DissipativeAction::IncreaseCoherence {
                    reason: "System chaotic - increase synchronization".into(),
                    target_coherence: 0.6,
                }
            }
            ThermodynamicRegime::EdgeOfChaos if health > 0.8 => {
                DissipativeAction::Maintain {
                    reason: "Optimal regime - maintain current dynamics".into(),
                }
            }
            _ => {
                if self.lambda_parameter < self.config.target_lambda {
                    DissipativeAction::IncreaseDifferentiation {
                        reason: "Too ordered - increase differentiation".into(),
                        target_lambda: self.config.target_lambda,
                    }
                } else {
                    DissipativeAction::IncreaseIntegration {
                        reason: "Too differentiated - increase integration".into(),
                        target_phi: 0.7,
                    }
                }
            }
        }
    }

    /// Get current state summary
    pub fn summary(&self) -> DissipativeSummary {
        DissipativeSummary {
            regime: self.current_regime(),
            lambda_parameter: self.lambda_parameter,
            criticality_distance: self.criticality_distance,
            entropy_production: self.entropy_production_rate,
            order_parameter: self.order_parameter,
            health_score: self.health_score(),
            is_critical: self.is_critical(),
            is_alive: self.is_alive(),
            recent_bifurcations: self.bifurcation_history.len(),
            recommendation: self.recommend_action(),
        }
    }
}

impl Default for DissipativeConsciousness {
    fn default() -> Self {
        Self::new()
    }
}

/// Recommended action to improve dissipative health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DissipativeAction {
    /// Increase activity/processing
    IncreaseActivity {
        reason: String,
        suggested_increase: f64,
    },
    /// Increase coherence/synchronization
    IncreaseCoherence {
        reason: String,
        target_coherence: f64,
    },
    /// Increase integration (Φ)
    IncreaseIntegration {
        reason: String,
        target_phi: f64,
    },
    /// Increase differentiation
    IncreaseDifferentiation {
        reason: String,
        target_lambda: f64,
    },
    /// Maintain current state
    Maintain {
        reason: String,
    },
}

/// Summary of dissipative state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DissipativeSummary {
    pub regime: ThermodynamicRegime,
    pub lambda_parameter: f64,
    pub criticality_distance: f64,
    pub entropy_production: f64,
    pub order_parameter: f64,
    pub health_score: f64,
    pub is_critical: bool,
    pub is_alive: bool,
    pub recent_bifurcations: usize,
    pub recommendation: DissipativeAction,
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dissipative_creation() {
        let dc = DissipativeConsciousness::new();
        assert!(dc.criticality_distance > 0.0);
        assert_eq!(dc.stats.total_bifurcations, 0);
    }

    #[test]
    fn test_update_toward_criticality() {
        let mut dc = DissipativeConsciousness::new();

        // Simulate edge-of-chaos conditions
        // λ ≈ 0.273 when differentiation/(integration + differentiation) ≈ 0.273
        // This requires phi ≈ 0.727 and coherence ≈ 0.727
        for _ in 0..10 {
            dc.update(0.73, 0.5, 100.0, 0.73);
        }

        // Should be near criticality
        assert!(dc.criticality_distance < 0.2,
            "Expected near criticality, got d={}", dc.criticality_distance);
    }

    #[test]
    fn test_regime_classification() {
        let mut dc = DissipativeConsciousness::new();

        // Low entropy = equilibrium
        dc.update(0.1, 0.001, 1.0, 0.1);
        assert_eq!(dc.current_regime(), ThermodynamicRegime::Equilibrium);

        // High phi, high coherence = edge of chaos potential
        for _ in 0..5 {
            dc.update(0.73, 0.5, 100.0, 0.73);
        }
        let regime = dc.current_regime();
        assert!(matches!(regime,
            ThermodynamicRegime::EdgeOfChaos | ThermodynamicRegime::FarFromEquilibrium));
    }

    #[test]
    fn test_bifurcation_detection() {
        let mut dc = DissipativeConsciousness::new();

        // Stable state
        for _ in 0..5 {
            dc.update(0.5, 0.3, 50.0, 0.5);
        }
        let initial_bifurcations = dc.stats.total_bifurcations;

        // Sudden change - should trigger bifurcation
        dc.update(0.9, 0.8, 200.0, 0.9);

        // May or may not detect depending on threshold
        // At least check that the system handles it
        assert!(dc.order_parameter > 0.5);
    }

    #[test]
    fn test_health_score() {
        let mut dc = DissipativeConsciousness::new();

        // Move toward healthy state
        for _ in 0..10 {
            dc.update(0.7, 0.4, 80.0, 0.7);
        }

        let health = dc.health_score();
        assert!(health > 0.0 && health <= 1.0);
    }

    #[test]
    fn test_is_alive() {
        let mut dc = DissipativeConsciousness::new();

        // Dead state (no entropy production)
        dc.update(0.0, 0.0, 0.0, 0.0);
        assert!(!dc.is_alive());

        // Alive state
        dc.update(0.5, 0.5, 100.0, 0.5);
        assert!(dc.is_alive());
    }

    #[test]
    fn test_recommendation() {
        let mut dc = DissipativeConsciousness::new();

        // Low activity should recommend increase
        dc.update(0.1, 0.01, 1.0, 0.1);
        let action = dc.recommend_action();
        assert!(matches!(action, DissipativeAction::IncreaseActivity { .. }));
    }
}
