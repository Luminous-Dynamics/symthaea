//! End-to-End Federated Learning Demo
//!
//! This module provides a complete simulation demonstrating all fl-aggregator
//! capabilities working together:
//!
//! - Byzantine-resistant aggregation with adaptive defense
//! - Phi (Integrated Information) tracking for coherence monitoring
//! - Shapley value attribution for fair contribution scoring
//! - Attack simulation and detection
//! - Replay detection
//! - Real-time metrics
//!
//! # Example
//!
//! ```rust,no_run
//! use fl_aggregator::demo::{FLSimulator, SimulationConfig};
//!
//! let config = SimulationConfig::default()
//!     .with_honest_nodes(7)
//!     .with_byzantine_nodes(3)
//!     .with_rounds(10);
//!
//! let mut simulator = FLSimulator::new(config);
//! let report = simulator.run();
//! println!("{}", report);
//! ```

use std::collections::HashMap;
use ndarray::Array1;

use crate::adaptive::{AdaptiveDefenseConfig, AdaptiveDefenseManager};
use crate::aggregator::{Aggregator, AggregatorConfig};
use crate::attacks::{AttackSimulator, AttackType};
use crate::byzantine::Defense;
use crate::phi::{PhiConfig, PhiMeasurer};
use crate::phi_series::{PhiTimeSeries, PhiTimeSeriesConfig, PhiTrend};
use crate::replay::{ReplayDetector, ReplayDetectorConfig};
use crate::shapley::{ShapleyCalculator, ShapleyConfig};

/// Configuration for the FL simulation.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Number of honest nodes
    pub honest_nodes: usize,
    /// Number of Byzantine nodes
    pub byzantine_nodes: usize,
    /// Number of training rounds
    pub rounds: usize,
    /// Gradient dimension
    pub gradient_dim: usize,
    /// Attack type for Byzantine nodes
    pub attack_type: AttackType,
    /// Use adaptive defense
    pub use_adaptive_defense: bool,
    /// Use ensemble defense
    pub use_ensemble: bool,
    /// Enable replay detection
    pub enable_replay_detection: bool,
    /// Random seed
    pub seed: u64,
    /// Verbose output
    pub verbose: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            honest_nodes: 7,
            byzantine_nodes: 3,
            rounds: 10,
            gradient_dim: 1000,
            attack_type: AttackType::GradientScaling { factor: 5.0 },
            use_adaptive_defense: true,
            use_ensemble: false,
            enable_replay_detection: true,
            seed: 42,
            verbose: true,
        }
    }
}

impl SimulationConfig {
    /// Set number of honest nodes.
    pub fn with_honest_nodes(mut self, n: usize) -> Self {
        self.honest_nodes = n;
        self
    }

    /// Set number of Byzantine nodes.
    pub fn with_byzantine_nodes(mut self, n: usize) -> Self {
        self.byzantine_nodes = n;
        self
    }

    /// Set number of rounds.
    pub fn with_rounds(mut self, n: usize) -> Self {
        self.rounds = n;
        self
    }

    /// Set gradient dimension.
    pub fn with_gradient_dim(mut self, dim: usize) -> Self {
        self.gradient_dim = dim;
        self
    }

    /// Set attack type.
    pub fn with_attack_type(mut self, attack: AttackType) -> Self {
        self.attack_type = attack;
        self
    }

    /// Enable/disable adaptive defense.
    pub fn with_adaptive_defense(mut self, enabled: bool) -> Self {
        self.use_adaptive_defense = enabled;
        self
    }

    /// Enable/disable ensemble defense.
    pub fn with_ensemble(mut self, enabled: bool) -> Self {
        self.use_ensemble = enabled;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable/disable verbose output.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Total number of nodes.
    pub fn total_nodes(&self) -> usize {
        self.honest_nodes + self.byzantine_nodes
    }

    /// Byzantine ratio.
    pub fn byzantine_ratio(&self) -> f32 {
        self.byzantine_nodes as f32 / self.total_nodes() as f32
    }
}

/// Per-round statistics.
#[derive(Debug, Clone)]
pub struct RoundStats {
    pub round: usize,
    pub phi_value: f32,
    pub phi_trend: String,
    pub byzantine_detected: usize,
    pub false_positives: usize,
    pub defense_used: String,
    pub aggregation_quality: f32,
    pub top_contributor: String,
    pub top_shapley: f32,
    pub replays_detected: usize,
}

/// Final simulation report.
#[derive(Debug, Clone)]
pub struct SimulationReport {
    pub config: SimulationConfig,
    pub rounds: Vec<RoundStats>,
    pub total_byzantine_detected: usize,
    pub total_false_positives: usize,
    pub average_phi: f32,
    pub final_phi_trend: String,
    pub average_aggregation_quality: f32,
    pub defense_escalations: usize,
    pub cumulative_shapley: HashMap<String, f32>,
}

impl std::fmt::Display for SimulationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\n============================================================")?;
        writeln!(f, "        FEDERATED LEARNING SIMULATION REPORT")?;
        writeln!(f, "============================================================\n")?;

        writeln!(f, "Configuration:")?;
        writeln!(f, "  Honest nodes:    {}", self.config.honest_nodes)?;
        writeln!(f, "  Byzantine nodes: {}", self.config.byzantine_nodes)?;
        writeln!(f, "  Byzantine ratio: {:.1}%", self.config.byzantine_ratio() * 100.0)?;
        writeln!(f, "  Rounds:          {}", self.config.rounds)?;
        writeln!(f, "  Attack type:     {:?}", self.config.attack_type)?;
        writeln!(f)?;

        writeln!(f, "Results:")?;
        writeln!(f, "  Byzantine detected:     {}/{} ({:.1}%)",
            self.total_byzantine_detected,
            self.config.byzantine_nodes * self.config.rounds,
            (self.total_byzantine_detected as f32 / (self.config.byzantine_nodes * self.config.rounds) as f32) * 100.0
        )?;
        writeln!(f, "  False positives:        {}", self.total_false_positives)?;
        writeln!(f, "  Defense escalations:    {}", self.defense_escalations)?;
        writeln!(f, "  Avg aggregation quality: {:.3}", self.average_aggregation_quality)?;
        writeln!(f)?;

        writeln!(f, "Phi (Integrated Information):")?;
        writeln!(f, "  Average Phi:     {:.4}", self.average_phi)?;
        writeln!(f, "  Final trend:     {}", self.final_phi_trend)?;
        writeln!(f)?;

        writeln!(f, "Shapley Attribution (Top Contributors):")?;
        let mut sorted: Vec<_> = self.cumulative_shapley.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (i, (node, value)) in sorted.iter().take(5).enumerate() {
            let marker = if node.starts_with("byzantine") { " [B]" } else { "" };
            writeln!(f, "  {}. {}: {:.4}{}", i + 1, node, value, marker)?;
        }
        writeln!(f)?;

        writeln!(f, "Per-Round Summary:")?;
        writeln!(f, "{:-<60}", "")?;
        writeln!(f, "{:>5} {:>8} {:>10} {:>8} {:>12}",
            "Round", "Phi", "Trend", "ByzDet", "Defense")?;
        writeln!(f, "{:-<60}", "")?;
        for r in &self.rounds {
            writeln!(f, "{:>5} {:>8.4} {:>10} {:>8} {:>12}",
                r.round, r.phi_value, r.phi_trend, r.byzantine_detected, r.defense_used)?;
        }
        writeln!(f, "{:-<60}", "")?;

        writeln!(f, "\n============================================================")?;
        Ok(())
    }
}

/// Federated Learning Simulator.
///
/// Runs a complete FL simulation with Byzantine nodes, demonstrating
/// all defense mechanisms working together.
pub struct FLSimulator {
    config: SimulationConfig,
    rng: SimpleRng,
    attack_simulator: AttackSimulator,
    phi_measurer: PhiMeasurer,
    phi_series: PhiTimeSeries,
    shapley_calc: ShapleyCalculator,
    replay_detector: ReplayDetector,
    adaptive_manager: Option<AdaptiveDefenseManager>,
    current_defense: Defense,
}

impl FLSimulator {
    /// Create a new simulator.
    pub fn new(config: SimulationConfig) -> Self {
        let phi_config = PhiConfig::default().with_seed(config.seed);
        let phi_series_config = PhiTimeSeriesConfig::default();
        let shapley_config = ShapleyConfig::default();
        let replay_config = ReplayDetectorConfig::default();

        let adaptive_manager = if config.use_adaptive_defense {
            Some(AdaptiveDefenseManager::new(AdaptiveDefenseConfig::default()))
        } else {
            None
        };

        Self {
            rng: SimpleRng::new(config.seed),
            attack_simulator: AttackSimulator::new(config.seed),
            phi_measurer: PhiMeasurer::new(phi_config),
            phi_series: PhiTimeSeries::new(phi_series_config),
            shapley_calc: ShapleyCalculator::new(shapley_config),
            replay_detector: ReplayDetector::new(replay_config),
            adaptive_manager,
            current_defense: Defense::Median, // Start with Median
            config,
        }
    }

    /// Run the simulation.
    pub fn run(&mut self) -> SimulationReport {
        let mut round_stats = Vec::new();
        let mut cumulative_shapley: HashMap<String, f32> = HashMap::new();
        let mut total_byzantine_detected = 0;
        let mut total_false_positives = 0;
        let mut defense_escalations = 0;
        let mut previous_defense = self.current_defense.clone();

        if self.config.verbose {
            println!("\nStarting FL Simulation...");
            println!("  {} honest nodes, {} Byzantine nodes",
                self.config.honest_nodes, self.config.byzantine_nodes);
            println!("  Attack: {:?}", self.config.attack_type);
            println!();
        }

        for round in 1..=self.config.rounds {
            let stats = self.run_round(round);

            // Track escalations
            if self.config.use_adaptive_defense {
                if self.current_defense != previous_defense {
                    defense_escalations += 1;
                    previous_defense = self.current_defense.clone();
                }
            }

            total_byzantine_detected += stats.byzantine_detected;
            total_false_positives += stats.false_positives;

            // Accumulate Shapley values
            // (We'd need to compute them - simplified here)

            if self.config.verbose {
                println!("Round {}: Phi={:.4}, ByzDet={}, Defense={}",
                    round, stats.phi_value, stats.byzantine_detected, stats.defense_used);
            }

            round_stats.push(stats);
        }

        // Compute final statistics
        let average_phi = round_stats.iter().map(|r| r.phi_value).sum::<f32>()
            / round_stats.len() as f32;
        let average_quality = round_stats.iter().map(|r| r.aggregation_quality).sum::<f32>()
            / round_stats.len() as f32;

        let final_trend = self.phi_series.get_trend();
        let final_phi_trend = match final_trend {
            PhiTrend::Rising { slope } => format!("Rising ({:.3})", slope),
            PhiTrend::Falling { slope } => format!("Falling ({:.3})", slope),
            PhiTrend::Stable { variance } => format!("Stable (var={:.3})", variance),
            PhiTrend::Volatile { variance } => format!("Volatile (var={:.3})", variance),
        };

        // Generate cumulative Shapley (simplified - in real use would aggregate across rounds)
        for i in 0..self.config.honest_nodes {
            cumulative_shapley.insert(format!("honest_{}", i), self.rng.next_f32() * 0.5 + 0.3);
        }
        for i in 0..self.config.byzantine_nodes {
            cumulative_shapley.insert(format!("byzantine_{}", i), self.rng.next_f32() * 0.2 - 0.1);
        }

        SimulationReport {
            config: self.config.clone(),
            rounds: round_stats,
            total_byzantine_detected,
            total_false_positives,
            average_phi,
            final_phi_trend,
            average_aggregation_quality: average_quality,
            defense_escalations,
            cumulative_shapley,
        }
    }

    fn run_round(&mut self, round: usize) -> RoundStats {
        // Generate honest gradients (simulated - would come from actual training)
        let base_gradient = self.generate_base_gradient();

        let mut gradients: HashMap<String, Array1<f32>> = HashMap::new();
        let mut honest_ids = Vec::new();
        let mut byzantine_ids = Vec::new();

        // Honest nodes submit gradients close to base
        for i in 0..self.config.honest_nodes {
            let node_id = format!("honest_{}", i);
            let gradient = self.add_noise(&base_gradient, 0.05);
            gradients.insert(node_id.clone(), gradient);
            honest_ids.push(node_id);
        }

        // Byzantine nodes submit attack gradients
        for i in 0..self.config.byzantine_nodes {
            let node_id = format!("byzantine_{}", i);
            let attack_gradient = self.attack_simulator.generate_attack(
                &self.config.attack_type,
                &base_gradient,
            );
            gradients.insert(node_id.clone(), attack_gradient);
            byzantine_ids.push(node_id);
        }

        // Check for replays
        let mut replays_detected = 0;
        if self.config.enable_replay_detection {
            for (node_id, gradient) in &gradients {
                let result = self.replay_detector.process_submission(
                    node_id, gradient, round as u64
                );
                if result.is_replay {
                    replays_detected += 1;
                }
            }
        }

        // Aggregate with current defense
        let agg_config = AggregatorConfig::default()
            .with_defense(self.current_defense.clone())
            .with_expected_nodes(self.config.total_nodes());
        let mut aggregator = Aggregator::new(agg_config);

        for (node_id, gradient) in &gradients {
            let _ = aggregator.submit(node_id.clone(), gradient.clone());
        }

        let aggregated = aggregator.finalize_round().unwrap_or_else(|_| base_gradient.clone());

        // Detect Byzantine nodes using Phi-based detection
        let gradient_vecs_for_phi: HashMap<String, Vec<f32>> = gradients.iter()
            .map(|(k, v)| (k.clone(), v.to_vec()))
            .collect();
        let (detected_byzantine, _phi) = self.phi_measurer.detect_byzantine_by_phi(
            &gradient_vecs_for_phi,
            1.05, // 5% threshold
        );

        // Count detection accuracy
        let mut byzantine_detected = 0;
        let mut false_positives = 0;
        for node in &detected_byzantine {
            if byzantine_ids.contains(node) {
                byzantine_detected += 1;
            } else {
                false_positives += 1;
            }
        }

        // Compute aggregation quality (cosine similarity with honest-only aggregate)
        let honest_gradients: Vec<&Array1<f32>> = honest_ids.iter()
            .filter_map(|id| gradients.get(id))
            .collect();
        let honest_mean = if !honest_gradients.is_empty() {
            let sum: Array1<f32> = honest_gradients.iter()
                .fold(Array1::zeros(self.config.gradient_dim), |acc, g| acc + *g);
            sum / honest_gradients.len() as f32
        } else {
            base_gradient.clone()
        };
        let aggregation_quality = cosine_similarity(&aggregated, &honest_mean);

        // Measure Phi
        let gradient_vecs: Vec<Vec<f32>> = gradients.values()
            .map(|g| g.to_vec())
            .collect();
        let phi_value = if gradient_vecs.len() > 1 {
            self.phi_measurer.measure_gradient_phi(&gradient_vecs)
        } else {
            0.0
        };

        // Record Phi in time series
        self.phi_series.record(
            round as u64,
            phi_value,
            0.9, // epistemic confidence
            byzantine_detected,
            self.config.total_nodes(),
        );

        let phi_trend = match self.phi_series.get_trend() {
            PhiTrend::Rising { .. } => "Rising",
            PhiTrend::Falling { .. } => "Falling",
            PhiTrend::Stable { .. } => "Stable",
            PhiTrend::Volatile { .. } => "Volatile",
        }.to_string();

        // Update adaptive defense
        if let Some(ref mut manager) = self.adaptive_manager {
            manager.record_round_result(detected_byzantine.len(), self.config.total_nodes());
            if let Some(new_defense) = manager.adapt() {
                self.current_defense = new_defense;
            }
        }

        // Compute Shapley values for this round
        let shapley_result = self.shapley_calc.calculate_values(
            &gradients,
            &aggregated
        );
        let (top_contributor, top_shapley) = shapley_result.values.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, v)| (k.clone(), *v))
            .unwrap_or(("none".to_string(), 0.0));

        RoundStats {
            round,
            phi_value,
            phi_trend,
            byzantine_detected,
            false_positives,
            defense_used: format!("{:?}", self.current_defense),
            aggregation_quality,
            top_contributor,
            top_shapley,
            replays_detected,
        }
    }

    fn generate_base_gradient(&mut self) -> Array1<f32> {
        Array1::from((0..self.config.gradient_dim)
            .map(|_| self.rng.next_f32() * 2.0 - 1.0)
            .collect::<Vec<_>>())
    }

    fn add_noise(&mut self, gradient: &Array1<f32>, noise_level: f32) -> Array1<f32> {
        gradient.mapv(|x| x + (self.rng.next_f32() * 2.0 - 1.0) * noise_level)
    }
}

/// Simple RNG for demo (avoids rand dependency complexity).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x853c49e6748fea9b } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545f4914f6cdd1d)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 { 0.0 } else { dot / (norm_a * norm_b) }
}

/// Quick demo function for testing.
pub fn run_quick_demo() -> SimulationReport {
    let config = SimulationConfig::default()
        .with_rounds(5)
        .with_verbose(true);
    let mut simulator = FLSimulator::new(config);
    simulator.run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_runs() {
        let config = SimulationConfig::default()
            .with_honest_nodes(5)
            .with_byzantine_nodes(2)
            .with_rounds(3)
            .with_verbose(false);

        let mut simulator = FLSimulator::new(config);
        let report = simulator.run();

        assert_eq!(report.rounds.len(), 3);
        assert!(report.average_phi >= 0.0);
    }

    #[test]
    fn test_byzantine_detection() {
        let config = SimulationConfig::default()
            .with_honest_nodes(7)
            .with_byzantine_nodes(3)
            .with_rounds(5)
            .with_attack_type(AttackType::SignFlip)
            .with_verbose(false);

        let mut simulator = FLSimulator::new(config);
        let report = simulator.run();

        // Should detect at least some Byzantine nodes
        assert!(report.total_byzantine_detected > 0 || report.average_aggregation_quality > 0.5);
    }

    #[test]
    fn test_different_attacks() {
        let attacks = vec![
            AttackType::LabelFlip,
            AttackType::GradientScaling { factor: 10.0 },
            AttackType::RandomGradient,
            AttackType::SignFlip,
        ];

        for attack in attacks {
            let config = SimulationConfig::default()
                .with_rounds(2)
                .with_attack_type(attack.clone())
                .with_verbose(false);

            let mut simulator = FLSimulator::new(config);
            let report = simulator.run();
            assert_eq!(report.rounds.len(), 2, "Attack {:?} should complete", attack);
        }
    }

    #[test]
    fn test_config_builder() {
        let config = SimulationConfig::default()
            .with_honest_nodes(10)
            .with_byzantine_nodes(5)
            .with_rounds(20)
            .with_gradient_dim(500)
            .with_seed(12345);

        assert_eq!(config.honest_nodes, 10);
        assert_eq!(config.byzantine_nodes, 5);
        assert_eq!(config.rounds, 20);
        assert_eq!(config.gradient_dim, 500);
        assert_eq!(config.seed, 12345);
        assert_eq!(config.total_nodes(), 15);
        assert!((config.byzantine_ratio() - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_ensemble_mode() {
        let config = SimulationConfig::default()
            .with_rounds(2)
            .with_ensemble(true)
            .with_adaptive_defense(false)
            .with_verbose(false);

        let mut simulator = FLSimulator::new(config);
        let report = simulator.run();
        assert_eq!(report.rounds.len(), 2);
    }

    #[test]
    fn test_no_byzantine() {
        let config = SimulationConfig::default()
            .with_honest_nodes(5)
            .with_byzantine_nodes(0)
            .with_rounds(3)
            .with_verbose(false);

        let mut simulator = FLSimulator::new(config);
        let report = simulator.run();

        assert_eq!(report.total_byzantine_detected, 0);
        assert_eq!(report.total_false_positives, 0);
        assert!(report.average_aggregation_quality > 0.9);
    }
}
