// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Ported from fl-aggregator/src/attacks.rs at commit feat/fl-consolidation
//! Attack Simulator for Testing Byzantine Defenses.
//!
//! WASM-compatible implementation using `Vec<f32>` instead of ndarray.
//! Implements 11 Byzantine attacks for testing federated learning defenses.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Types of Byzantine attacks that can be simulated.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AttackType {
    /// Gradient pointing in wrong direction (negated).
    LabelFlip,
    /// Multiply gradient by a large factor.
    GradientScaling { factor: f32 },
    /// Add Gaussian noise to gradient.
    GradientNoise { std_dev: f32 },
    /// Flip all gradient signs.
    SignFlip,
    /// Submit zero gradients (free-riding without learning).
    ZeroGradient,
    /// Submit random gradients.
    RandomGradient,
    /// Inject backdoor pattern into gradient.
    Backdoor { trigger_pattern: Vec<f32> },
    /// Multiple attackers coordinate to shift aggregation.
    CollisionAttack { num_colluders: usize },
    /// Attack that evades detection (stays close to mean).
    AdaptiveAttack,
    /// Gradually shift gradient over rounds.
    SlowPoisoning { rate: f32 },
    /// Copy other nodes' gradients with small noise.
    FreeRider,
}

impl std::fmt::Display for AttackType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttackType::LabelFlip => write!(f, "LabelFlip"),
            AttackType::GradientScaling { factor } => write!(f, "GradientScaling({}x)", factor),
            AttackType::GradientNoise { std_dev } => write!(f, "GradientNoise(σ={})", std_dev),
            AttackType::SignFlip => write!(f, "SignFlip"),
            AttackType::ZeroGradient => write!(f, "ZeroGradient"),
            AttackType::RandomGradient => write!(f, "RandomGradient"),
            AttackType::Backdoor { .. } => write!(f, "Backdoor"),
            AttackType::CollisionAttack { num_colluders } => {
                write!(f, "CollisionAttack({})", num_colluders)
            }
            AttackType::AdaptiveAttack => write!(f, "AdaptiveAttack"),
            AttackType::SlowPoisoning { rate } => write!(f, "SlowPoisoning(rate={})", rate),
            AttackType::FreeRider => write!(f, "FreeRider"),
        }
    }
}

/// XorShift64 PRNG for WASM-compatible randomness.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E3779B97F4A7C15, // Avoid seed-0 fixed point
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate f32 in [-1.0, 1.0]
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
    }

    /// Generate f32 in [0.0, 1.0)
    fn next_unit_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }

    /// Box-Muller Gaussian noise with given std_dev.
    fn next_gaussian(&mut self, std_dev: f32) -> f32 {
        let u1 = self.next_unit_f32().max(1e-10);
        let u2 = self.next_unit_f32();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        z * std_dev
    }
}

/// L2 norm of a gradient vector.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity between two gradient vectors.
#[cfg(test)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Attack simulator for generating Byzantine gradients.
///
/// Uses a deterministic PRNG (XorShift64) for reproducibility and WASM compatibility.
pub struct AttackSimulator {
    rng: XorShift64,
    round_counter: HashMap<String, usize>,
    gradient_history: Vec<Vec<f32>>,
}

impl AttackSimulator {
    /// Create a new attack simulator with a seed for reproducibility.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: XorShift64::new(seed),
            round_counter: HashMap::new(),
            gradient_history: Vec::new(),
        }
    }

    /// Reset the simulator state.
    pub fn reset(&mut self) {
        self.round_counter.clear();
        self.gradient_history.clear();
    }

    /// Update gradient history for adaptive attacks.
    pub fn update_history(&mut self, gradients: &[Vec<f32>]) {
        self.gradient_history = gradients.to_vec();
    }

    /// Generate an attack gradient based on the attack type.
    pub fn generate_attack(
        &mut self,
        attack_type: &AttackType,
        honest_gradient: &[f32],
    ) -> Vec<f32> {
        match attack_type {
            AttackType::LabelFlip => self.label_flip(honest_gradient),
            AttackType::GradientScaling { factor } => self.gradient_scaling(honest_gradient, *factor),
            AttackType::GradientNoise { std_dev } => self.gradient_noise(honest_gradient, *std_dev),
            AttackType::SignFlip => self.sign_flip(honest_gradient),
            AttackType::ZeroGradient => vec![0.0; honest_gradient.len()],
            AttackType::RandomGradient => self.random_gradient(honest_gradient),
            AttackType::Backdoor { trigger_pattern } => {
                self.backdoor(honest_gradient, trigger_pattern)
            }
            AttackType::CollisionAttack { .. } => self.collision_single(honest_gradient),
            AttackType::AdaptiveAttack => self.adaptive(honest_gradient),
            AttackType::SlowPoisoning { rate } => {
                self.slow_poisoning(honest_gradient, *rate, "default")
            }
            AttackType::FreeRider => self.free_rider(honest_gradient),
        }
    }

    /// Generate multiple attack gradients (batch mode).
    pub fn generate_attack_batch(
        &mut self,
        attack_type: &AttackType,
        honest_gradients: &[Vec<f32>],
        num_attackers: usize,
    ) -> Vec<Vec<f32>> {
        if honest_gradients.is_empty() {
            return vec![];
        }

        match attack_type {
            AttackType::CollisionAttack { num_colluders } => {
                self.collision_batch(honest_gradients, *num_colluders)
            }
            _ => (0..num_attackers)
                .map(|i| {
                    let idx = i % honest_gradients.len();
                    self.generate_attack(attack_type, &honest_gradients[idx])
                })
                .collect(),
        }
    }

    fn label_flip(&self, honest: &[f32]) -> Vec<f32> {
        honest.iter().map(|x| -x).collect()
    }

    fn gradient_scaling(&self, honest: &[f32], factor: f32) -> Vec<f32> {
        honest.iter().map(|x| x * factor).collect()
    }

    fn gradient_noise(&mut self, honest: &[f32], std_dev: f32) -> Vec<f32> {
        honest
            .iter()
            .map(|x| x + self.rng.next_gaussian(std_dev))
            .collect()
    }

    fn sign_flip(&self, honest: &[f32]) -> Vec<f32> {
        honest.iter().map(|x| -x).collect()
    }

    fn random_gradient(&mut self, honest: &[f32]) -> Vec<f32> {
        let norm = l2_norm(honest);
        let random: Vec<f32> = (0..honest.len()).map(|_| self.rng.next_f32()).collect();
        let random_norm = l2_norm(&random);
        if random_norm > 1e-10 {
            let scale = norm / random_norm;
            random.iter().map(|x| x * scale).collect()
        } else {
            random
        }
    }

    fn backdoor(&self, honest: &[f32], trigger: &[f32]) -> Vec<f32> {
        let mut result = honest.to_vec();
        for (i, &val) in trigger.iter().enumerate() {
            if i < result.len() {
                result[i] += val;
            }
        }
        result
    }

    fn collision_single(&mut self, honest: &[f32]) -> Vec<f32> {
        honest.iter().map(|x| x * 5.0).collect()
    }

    fn collision_batch(
        &mut self,
        honest_gradients: &[Vec<f32>],
        num_colluders: usize,
    ) -> Vec<Vec<f32>> {
        if honest_gradients.is_empty() {
            return vec![];
        }

        let dim = honest_gradients[0].len();

        // Compute mean of honest gradients
        let mut mean = vec![0.0f32; dim];
        for g in honest_gradients {
            for (i, &v) in g.iter().enumerate() {
                mean[i] += v;
            }
        }
        let n = honest_gradients.len() as f32;
        for v in &mut mean {
            *v /= n;
        }

        // Random attack direction
        let attack_dir: Vec<f32> = (0..dim).map(|_| self.rng.next_f32()).collect();
        let attack_norm = l2_norm(&attack_dir);
        let normalized: Vec<f32> = if attack_norm > 1e-10 {
            attack_dir.iter().map(|x| x / attack_norm).collect()
        } else {
            attack_dir
        };

        let mean_norm = l2_norm(&mean);
        let base_attack: Vec<f32> = normalized.iter().map(|x| x * mean_norm * 3.0).collect();

        // All colluders submit similar gradients with small variations
        (0..num_colluders)
            .map(|_| {
                base_attack
                    .iter()
                    .map(|x| x + self.rng.next_gaussian(0.1))
                    .collect()
            })
            .collect()
    }

    fn adaptive(&mut self, honest: &[f32]) -> Vec<f32> {
        if !self.gradient_history.is_empty() {
            let dim = honest.len();
            let mut mean = vec![0.0f32; dim];
            let mut count = 0;
            for g in &self.gradient_history {
                if g.len() == dim {
                    for (i, &v) in g.iter().enumerate() {
                        mean[i] += v;
                    }
                    count += 1;
                }
            }
            if count > 0 {
                for v in &mut mean {
                    *v /= count as f32;
                }
            }

            let attack_dir: Vec<f32> = honest.iter().map(|x| -x).collect();
            let attack_norm = l2_norm(&attack_dir);
            let mean_norm = l2_norm(&mean);
            let attack_strength = 0.3;

            if attack_norm > 1e-10 {
                let scale = mean_norm * attack_strength / attack_norm;
                mean.iter()
                    .zip(attack_dir.iter())
                    .map(|(m, a)| m + a * scale)
                    .collect()
            } else {
                mean
            }
        } else {
            honest
                .iter()
                .map(|x| x + self.rng.next_gaussian(0.1))
                .collect()
        }
    }

    fn slow_poisoning(&mut self, honest: &[f32], rate: f32, attacker_id: &str) -> Vec<f32> {
        let round = self
            .round_counter
            .entry(attacker_id.to_string())
            .or_insert(0);
        *round += 1;

        let poison_strength = (rate * *round as f32).min(1.0);

        honest
            .iter()
            .map(|x| x * (1.0 - poison_strength) + (-x) * poison_strength)
            .collect()
    }

    fn free_rider(&mut self, honest: &[f32]) -> Vec<f32> {
        honest
            .iter()
            .map(|x| x + self.rng.next_gaussian(0.01))
            .collect()
    }
}

/// Attack scenario configuration for defense testing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttackScenario {
    /// Number of honest nodes.
    pub honest_nodes: usize,
    /// Number of Byzantine nodes.
    pub byzantine_nodes: usize,
    /// Attack types and their counts.
    pub attack_types: Vec<(AttackType, usize)>,
    /// Number of training rounds to simulate.
    pub rounds: usize,
}

impl AttackScenario {
    /// Create a simple scenario with a single attack type.
    pub fn simple(
        honest_nodes: usize,
        byzantine_nodes: usize,
        attack_type: AttackType,
        rounds: usize,
    ) -> Self {
        Self {
            honest_nodes,
            byzantine_nodes,
            attack_types: vec![(attack_type, byzantine_nodes)],
            rounds,
        }
    }

    /// Create a mixed attack scenario.
    pub fn mixed(
        honest_nodes: usize,
        attack_types: Vec<(AttackType, usize)>,
        rounds: usize,
    ) -> Self {
        let byzantine_nodes = attack_types.iter().map(|(_, count)| count).sum();
        Self {
            honest_nodes,
            byzantine_nodes,
            attack_types,
            rounds,
        }
    }

    /// Get total number of nodes.
    pub fn total_nodes(&self) -> usize {
        self.honest_nodes + self.byzantine_nodes
    }

    /// Get Byzantine fraction.
    pub fn byzantine_fraction(&self) -> f32 {
        self.byzantine_nodes as f32 / self.total_nodes() as f32
    }
}

/// Metrics for attack effectiveness.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttackMetrics {
    /// Fraction of rounds where attack shifted aggregation.
    pub attack_success_rate: f32,
    /// Fraction of Byzantine nodes correctly detected.
    pub detection_rate: f32,
    /// Estimated accuracy impact.
    pub accuracy_impact: f32,
    /// Cosine similarity between attacked and honest-only aggregation.
    pub aggregation_similarity: f32,
}

impl AttackMetrics {
    pub fn new(attack_success_rate: f32, detection_rate: f32, aggregation_similarity: f32) -> Self {
        let accuracy_impact = (1.0 - aggregation_similarity).max(0.0);
        Self {
            attack_success_rate,
            detection_rate,
            accuracy_impact,
            aggregation_similarity,
        }
    }

    /// Check if attack was effective (high success, low detection).
    pub fn is_attack_effective(&self) -> bool {
        self.attack_success_rate > 0.5 && self.detection_rate < 0.5
    }

    /// Check if defense was effective (low success, high detection).
    pub fn is_defense_effective(&self) -> bool {
        self.attack_success_rate < 0.3 && self.aggregation_similarity > 0.8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_gradient() -> Vec<f32> {
        vec![1.0, 2.0, 3.0, 4.0, 5.0]
    }

    fn test_gradients(count: usize) -> Vec<Vec<f32>> {
        (0..count)
            .map(|i| {
                vec![
                    1.0 + i as f32 * 0.1,
                    2.0 + i as f32 * 0.1,
                    3.0 + i as f32 * 0.1,
                    4.0 + i as f32 * 0.1,
                    5.0 + i as f32 * 0.1,
                ]
            })
            .collect()
    }

    #[test]
    fn test_label_flip() {
        let mut sim = AttackSimulator::new(42);
        let honest = test_gradient();
        let attack = sim.generate_attack(&AttackType::LabelFlip, &honest);

        for (h, a) in honest.iter().zip(attack.iter()) {
            assert!((a + h).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gradient_scaling() {
        let mut sim = AttackSimulator::new(42);
        let honest = test_gradient();
        let factor = 10.0;
        let attack = sim.generate_attack(&AttackType::GradientScaling { factor }, &honest);

        for (h, a) in honest.iter().zip(attack.iter()) {
            assert!((a - h * factor).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gradient_noise() {
        let mut sim = AttackSimulator::new(42);
        let honest = test_gradient();
        let attack = sim.generate_attack(&AttackType::GradientNoise { std_dev: 1.0 }, &honest);

        let diff: f32 = honest.iter().zip(attack.iter()).map(|(h, a)| (h - a).abs()).sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_sign_flip() {
        let mut sim = AttackSimulator::new(42);
        let honest = test_gradient();
        let attack = sim.generate_attack(&AttackType::SignFlip, &honest);

        for (h, a) in honest.iter().zip(attack.iter()) {
            assert!((a + h).abs() < 1e-6);
        }
    }

    #[test]
    fn test_zero_gradient() {
        let mut sim = AttackSimulator::new(42);
        let honest = test_gradient();
        let attack = sim.generate_attack(&AttackType::ZeroGradient, &honest);

        for a in &attack {
            assert!(a.abs() < 1e-6);
        }
    }

    #[test]
    fn test_random_gradient() {
        let mut sim = AttackSimulator::new(42);
        let honest = test_gradient();
        let attack = sim.generate_attack(&AttackType::RandomGradient, &honest);

        let honest_norm = l2_norm(&honest);
        let attack_norm = l2_norm(&attack);
        assert!((attack_norm - honest_norm).abs() < honest_norm * 0.5);

        let sim_val = cosine_similarity(&honest, &attack);
        assert!(sim_val < 0.99);
    }

    #[test]
    fn test_backdoor() {
        let mut sim = AttackSimulator::new(42);
        let honest = test_gradient();
        let trigger = vec![10.0, 20.0];
        let attack = sim.generate_attack(
            &AttackType::Backdoor {
                trigger_pattern: trigger.clone(),
            },
            &honest,
        );

        assert!((attack[0] - (honest[0] + trigger[0])).abs() < 1e-6);
        assert!((attack[1] - (honest[1] + trigger[1])).abs() < 1e-6);
        for i in 2..honest.len() {
            assert!((attack[i] - honest[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_collision_batch() {
        let mut sim = AttackSimulator::new(42);
        let honest = test_gradients(5);
        let attacks = sim.generate_attack_batch(
            &AttackType::CollisionAttack { num_colluders: 3 },
            &honest,
            3,
        );

        assert_eq!(attacks.len(), 3);

        // All colluders should produce similar gradients
        for i in 1..attacks.len() {
            let cos = cosine_similarity(&attacks[0], &attacks[i]);
            assert!(cos > 0.9, "Colluders should have similar gradients, got {}", cos);
        }
    }

    #[test]
    fn test_adaptive_attack() {
        let mut sim = AttackSimulator::new(42);
        let honest_gradients = test_gradients(5);
        sim.update_history(&honest_gradients);

        let honest = test_gradient();
        let attack = sim.generate_attack(&AttackType::AdaptiveAttack, &honest);

        let mut mean = vec![0.0f32; honest.len()];
        for g in &honest_gradients {
            for (i, &v) in g.iter().enumerate() {
                mean[i] += v;
            }
        }
        let n = honest_gradients.len() as f32;
        for v in &mut mean {
            *v /= n;
        }

        let cos = cosine_similarity(&attack, &mean);
        assert!(cos > 0.3, "Adaptive attack should stay close to mean, got {}", cos);
    }

    #[test]
    fn test_slow_poisoning() {
        let mut sim = AttackSimulator::new(42);
        let honest = test_gradient();
        let rate = 0.1;

        let attack1 = sim.generate_attack(&AttackType::SlowPoisoning { rate }, &honest);
        let attack2 = sim.generate_attack(&AttackType::SlowPoisoning { rate }, &honest);
        let attack3 = sim.generate_attack(&AttackType::SlowPoisoning { rate }, &honest);

        let norm1 = l2_norm(&attack1);
        let norm2 = l2_norm(&attack2);
        let norm3 = l2_norm(&attack3);
        let honest_norm = l2_norm(&honest);

        assert!(norm1 < honest_norm, "Round 1 should have reduced norm");
        assert!(norm2 < norm1, "Round 2 should be smaller than round 1");
        assert!(norm3 < norm2, "Round 3 should be smaller than round 2");
    }

    #[test]
    fn test_free_rider() {
        let mut sim = AttackSimulator::new(42);
        let honest = test_gradient();
        let attack = sim.generate_attack(&AttackType::FreeRider, &honest);

        let cos = cosine_similarity(&attack, &honest);
        assert!(cos > 0.99, "Free rider should closely match honest gradient");
    }

    #[test]
    fn test_scenario_creation() {
        let scenario = AttackScenario::simple(7, 3, AttackType::SignFlip, 10);

        assert_eq!(scenario.honest_nodes, 7);
        assert_eq!(scenario.byzantine_nodes, 3);
        assert_eq!(scenario.total_nodes(), 10);
        assert!((scenario.byzantine_fraction() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_attack_metrics() {
        let metrics = AttackMetrics::new(0.6, 0.3, 0.5);
        assert!(metrics.is_attack_effective());
        assert!(!metrics.is_defense_effective());

        let good_defense = AttackMetrics::new(0.1, 0.9, 0.95);
        assert!(!good_defense.is_attack_effective());
        assert!(good_defense.is_defense_effective());
    }
}
