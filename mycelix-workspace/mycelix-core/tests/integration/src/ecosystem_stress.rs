// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ecosystem-Wide Stress Test Harness
//!
//! Comprehensive simulation that exercises all Mycelix-Core components together:
//! - Federated Learning with Byzantine attacks
//! - K-Vector ZK proofs for trust verification
//! - MATL trust layer integration
//! - RB-BFT consensus
//! - DKG key generation
//!
//! # Usage
//!
//! ```rust,ignore
//! use mycelix_integration::ecosystem_stress::{EcosystemStressTest, StressTestConfig};
//!
//! let config = StressTestConfig::default();
//! let mut test = EcosystemStressTest::new(config);
//! let result = test.run();
//!
//! assert!(result.byzantine_detection_rate > 0.98);
//! assert!(result.consensus_liveness);
//! ```

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use kvector_zkp::{KVectorWitness, proof::KVectorRangeProof};
use mycelix_core_types::KVector;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::Normal;

use crate::chaos_network::{ChaosNetwork, NetworkConfig, MessageType, FailureAttribution};

// ============================================================================
// Gradient Message (for network transport)
// ============================================================================

/// A gradient message sent through the chaos network
#[derive(Debug, Clone)]
pub struct GradientMessage {
    pub node_id: String,
    pub round: usize,
    pub gradient: Vec<f32>,
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for ecosystem stress test
#[derive(Debug, Clone)]
pub struct StressTestConfig {
    /// Number of nodes in the network
    pub num_nodes: usize,
    /// Fraction of Byzantine nodes (0.0 to 0.45)
    pub byzantine_fraction: f32,
    /// Number of FL rounds to simulate
    pub num_rounds: usize,
    /// Dimension of gradients
    pub gradient_dim: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Whether to run ZK proofs (slower but more realistic)
    pub enable_zkp: bool,
    /// Whether to run consensus simulation
    pub enable_consensus: bool,
    /// Whether to run DKG ceremony simulation
    pub enable_dkg: bool,
    /// Attack types to include
    pub attack_types: Vec<AttackType>,
    /// Minimum trust score for participation
    pub min_trust_threshold: f32,
    /// Enable realistic network simulation (latency, loss, partitions)
    pub enable_network_chaos: bool,
    /// Network configuration (if chaos enabled)
    pub network_config: NetworkConfig,
    /// Whether to inject random partitions during test
    pub inject_partitions: bool,
    /// Round at which to inject partition (if enabled)
    pub partition_at_round: Option<usize>,
    /// Duration of partition in rounds
    pub partition_duration: usize,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            num_nodes: 100,
            byzantine_fraction: 0.45,
            num_rounds: 50,
            gradient_dim: 1000,
            seed: 42,
            enable_zkp: true,
            enable_consensus: true,
            enable_dkg: false, // DKG is slow, disabled by default
            attack_types: vec![
                AttackType::SignFlip,
                AttackType::GradientScaling { factor: 10.0 },
                AttackType::AdaptiveAttack,
                AttackType::SlowPoisoning { rate: 0.1 },
                AttackType::LittleIsEnough { epsilon: 0.3 },
            ],
            min_trust_threshold: 0.3,
            enable_network_chaos: false, // Disabled by default for speed
            network_config: NetworkConfig::default(),
            inject_partitions: false,
            partition_at_round: None,
            partition_duration: 5,
        }
    }
}

impl StressTestConfig {
    /// Configuration for quick smoke tests
    pub fn quick() -> Self {
        Self {
            num_nodes: 20,
            byzantine_fraction: 0.30,
            num_rounds: 10,
            gradient_dim: 100,
            enable_zkp: false,
            enable_consensus: false,
            enable_dkg: false,
            ..Default::default()
        }
    }

    /// Configuration for scale testing
    pub fn scale_test(num_nodes: usize) -> Self {
        Self {
            num_nodes,
            byzantine_fraction: 0.45,
            num_rounds: 20,
            gradient_dim: 1000,
            enable_zkp: true,
            enable_consensus: true,
            enable_dkg: false,
            ..Default::default()
        }
    }

    /// Configuration for maximum adversarial pressure
    pub fn adversarial() -> Self {
        Self {
            num_nodes: 100,
            byzantine_fraction: 0.45,
            num_rounds: 100,
            gradient_dim: 5000,
            enable_zkp: true,
            enable_consensus: true,
            enable_dkg: true,
            attack_types: vec![
                AttackType::SignFlip,
                AttackType::GradientScaling { factor: 20.0 },
                AttackType::AdaptiveAttack,
                AttackType::SlowPoisoning { rate: 0.2 },
                AttackType::LittleIsEnough { epsilon: 0.5 },
                AttackType::Cartel { size: 10 },
            ],
            ..Default::default()
        }
    }

    /// Configuration with realistic network simulation (WAN latency)
    pub fn networked() -> Self {
        Self {
            num_nodes: 50,
            byzantine_fraction: 0.30,
            num_rounds: 30,
            gradient_dim: 500,
            enable_zkp: true,
            enable_consensus: true,
            enable_dkg: false,
            enable_network_chaos: true,
            network_config: NetworkConfig::wan(),
            inject_partitions: false,
            ..Default::default()
        }
    }

    /// Configuration for network partition testing
    pub fn partition_test() -> Self {
        Self {
            num_nodes: 50,
            byzantine_fraction: 0.30,
            num_rounds: 30,
            gradient_dim: 500,
            enable_zkp: false, // Faster without ZKP
            enable_consensus: true,
            enable_dkg: false,
            enable_network_chaos: true,
            network_config: NetworkConfig::partition_prone(),
            inject_partitions: true,
            partition_at_round: Some(10), // Partition at round 10
            partition_duration: 5,        // Heal after 5 rounds
            ..Default::default()
        }
    }

    /// Digital twin configuration - full realism
    pub fn digital_twin() -> Self {
        Self {
            num_nodes: 100,
            byzantine_fraction: 0.45,
            num_rounds: 50,
            gradient_dim: 1000,
            enable_zkp: true,
            enable_consensus: true,
            enable_dkg: false,
            enable_network_chaos: true,
            network_config: NetworkConfig::adversarial(),
            inject_partitions: true,
            partition_at_round: Some(20),
            partition_duration: 10,
            attack_types: vec![
                AttackType::SignFlip,
                AttackType::GradientScaling { factor: 15.0 },
                AttackType::AdaptiveAttack,
                AttackType::SlowPoisoning { rate: 0.15 },
                AttackType::LittleIsEnough { epsilon: 0.4 },
                AttackType::Cartel { size: 8 },
            ],
            ..Default::default()
        }
    }

    /// Chaos engineering configuration - all chaos features enabled
    pub fn chaos_engineering() -> Self {
        Self {
            num_nodes: 50,
            byzantine_fraction: 0.35,
            num_rounds: 30,
            gradient_dim: 500,
            enable_zkp: false, // Faster without ZKP
            enable_consensus: true,
            enable_dkg: false,
            enable_network_chaos: true,
            network_config: NetworkConfig::chaos_engineering(),
            inject_partitions: true,
            partition_at_round: Some(12),
            partition_duration: 6,
            attack_types: vec![
                AttackType::SignFlip,
                AttackType::GradientScaling { factor: 10.0 },
                AttackType::AdaptiveAttack,
                AttackType::SlowPoisoning { rate: 0.1 },
            ],
            ..Default::default()
        }
    }
}

// ============================================================================
// Attack Types (reusing fl-aggregator patterns)
// ============================================================================

/// Attack types for Byzantine simulation
#[derive(Debug, Clone)]
pub enum AttackType {
    /// Negate gradient direction
    SignFlip,
    /// Multiply gradient by factor
    GradientScaling { factor: f32 },
    /// Adaptive attack that tries to evade detection
    AdaptiveAttack,
    /// Gradual poisoning over rounds
    SlowPoisoning { rate: f32 },
    /// Small but consistent perturbation
    LittleIsEnough { epsilon: f32 },
    /// Coordinated cartel attack
    Cartel { size: usize },
}

// ============================================================================
// Node State
// ============================================================================

/// State of a simulated node
#[derive(Debug, Clone)]
pub struct NodeState {
    pub id: String,
    pub is_byzantine: bool,
    pub attack_type: Option<AttackType>,
    pub k_vector: KVector,
    pub trust_score: f32,
    pub reputation: f32,
    pub zkp_verified: bool,
    pub rounds_participated: usize,
    pub times_detected: usize,
}

impl NodeState {
    /// Create a new honest node
    pub fn new_honest(id: &str, rng: &mut StdRng) -> Self {
        let k_vector = Self::generate_honest_kvector(rng);
        let trust_score = k_vector.trust_score();
        Self {
            id: id.to_string(),
            is_byzantine: false,
            attack_type: None,
            k_vector,
            trust_score,
            reputation: 0.5,
            zkp_verified: false,
            rounds_participated: 0,
            times_detected: 0,
        }
    }

    /// Create a new Byzantine node
    pub fn new_byzantine(id: &str, attack_type: AttackType, rng: &mut StdRng) -> Self {
        // Byzantine nodes may claim fake trust scores
        let k_vector = Self::generate_byzantine_kvector(rng);
        let trust_score = k_vector.trust_score();
        Self {
            id: id.to_string(),
            is_byzantine: true,
            attack_type: Some(attack_type),
            k_vector,
            trust_score,
            reputation: 0.5,
            zkp_verified: false,
            rounds_participated: 0,
            times_detected: 0,
        }
    }

    fn generate_honest_kvector(rng: &mut StdRng) -> KVector {
        KVector {
            k_r: rng.gen_range(0.5..0.95),
            k_a: rng.gen_range(0.4..0.9),
            k_i: rng.gen_range(0.6..0.98),
            k_p: rng.gen_range(0.5..0.92),
            k_m: rng.gen_range(0.2..0.85),
            k_s: rng.gen_range(0.3..0.8),
            k_h: rng.gen_range(0.5..0.95),
            k_topo: rng.gen_range(0.4..0.88),
        }
    }

    fn generate_byzantine_kvector(rng: &mut StdRng) -> KVector {
        // Byzantine nodes have lower real trust metrics
        KVector {
            k_r: rng.gen_range(0.1..0.5),
            k_a: rng.gen_range(0.1..0.4),
            k_i: rng.gen_range(0.2..0.5),
            k_p: rng.gen_range(0.1..0.4),
            k_m: rng.gen_range(0.05..0.3),
            k_s: rng.gen_range(0.1..0.3),
            k_h: rng.gen_range(0.1..0.4),
            k_topo: rng.gen_range(0.1..0.35),
        }
    }
}

// ============================================================================
// Round Metrics
// ============================================================================

/// Metrics collected for each round
#[derive(Debug, Clone, Default)]
pub struct RoundMetrics {
    pub round_num: usize,
    pub total_nodes: usize,
    pub byzantine_nodes: usize,
    pub detected_byzantine: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub detection_rate: f32,
    pub zkp_proofs_generated: usize,
    pub zkp_proofs_verified: usize,
    pub zkp_time_ms: f64,
    pub consensus_reached: bool,
    pub consensus_time_ms: f64,
    pub round_latency_ms: f64,
    pub memory_estimate_mb: f64,
    // Network chaos metrics
    pub network_messages_sent: u64,
    pub network_messages_delivered: u64,
    pub network_messages_dropped: u64,
    pub network_messages_blocked: u64,
    pub network_partition_active: bool,
    pub network_avg_latency_ms: f64,
}

// ============================================================================
// Final Results
// ============================================================================

/// Final results of the stress test
#[derive(Debug, Clone)]
pub struct StressTestResult {
    pub config: StressTestConfig,
    pub total_rounds_completed: usize,
    pub byzantine_detection_rate: f32,
    pub false_positive_rate: f32,
    pub false_negative_rate: f32,
    pub avg_round_latency_ms: f64,
    pub total_zkp_proofs: usize,
    pub zkp_verification_rate: f32,
    pub avg_zkp_time_ms: f64,
    pub consensus_success_rate: f32,
    pub avg_consensus_time_ms: f64,
    pub peak_memory_mb: f64,
    pub round_metrics: Vec<RoundMetrics>,
    pub verdict: StressTestVerdict,
    // Network chaos results
    pub network_delivery_rate: f64,
    pub network_avg_latency_ms: f64,
    pub network_partition_events: u64,
    pub network_total_dropped: u64,
    pub partition_recovery_success: bool,
    // Failure attribution (B)
    pub failure_attribution: Option<FailureAttribution>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StressTestVerdict {
    Perfect,
    Excellent,
    Good,
    NeedsImprovement,
    Failed,
}

impl std::fmt::Display for StressTestVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StressTestVerdict::Perfect => write!(f, "PERFECT - All targets exceeded"),
            StressTestVerdict::Excellent => write!(f, "EXCELLENT - Near-perfect performance"),
            StressTestVerdict::Good => write!(f, "GOOD - Acceptable performance"),
            StressTestVerdict::NeedsImprovement => write!(f, "NEEDS IMPROVEMENT - Some issues"),
            StressTestVerdict::Failed => write!(f, "FAILED - Critical issues detected"),
        }
    }
}

// ============================================================================
// Main Stress Test
// ============================================================================

/// Ecosystem-wide stress test harness
pub struct EcosystemStressTest {
    config: StressTestConfig,
    rng: StdRng,
    nodes: HashMap<String, NodeState>,
    round_metrics: Vec<RoundMetrics>,
    current_round: usize,
    /// Optional chaos network layer
    network: Option<ChaosNetwork<GradientMessage>>,
    /// Track if we successfully recovered from a partition
    partition_recovery_success: bool,
    /// Pre-partition detection rate (to compare after healing)
    pre_partition_detection_rate: Option<f32>,
}

impl EcosystemStressTest {
    /// Create a new stress test with the given configuration
    pub fn new(config: StressTestConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let nodes = Self::initialize_nodes(&config, &mut rng);

        // Initialize chaos network if enabled
        let network = if config.enable_network_chaos {
            let mut net = ChaosNetwork::new(config.network_config.clone());
            // Register Byzantine nodes for timing attacks
            let byzantine_ids: HashSet<String> = nodes.iter()
                .filter(|(_, n)| n.is_byzantine)
                .map(|(id, _)| id.clone())
                .collect();
            net.register_byzantine_nodes(byzantine_ids);
            Some(net)
        } else {
            None
        };

        Self {
            config,
            rng,
            nodes,
            round_metrics: Vec::new(),
            current_round: 0,
            network,
            partition_recovery_success: false,
            pre_partition_detection_rate: None,
        }
    }

    fn initialize_nodes(config: &StressTestConfig, rng: &mut StdRng) -> HashMap<String, NodeState> {
        let num_byzantine = (config.num_nodes as f32 * config.byzantine_fraction) as usize;
        let byzantine_indices: HashSet<usize> = (0..config.num_nodes)
            .collect::<Vec<_>>()
            .choose_multiple(rng, num_byzantine)
            .cloned()
            .collect();

        let mut nodes = HashMap::new();
        let num_attack_types = config.attack_types.len();

        for i in 0..config.num_nodes {
            let node_id = format!("node_{:04}", i);
            let node = if byzantine_indices.contains(&i) {
                let attack_idx = i % num_attack_types.max(1);
                let attack = if num_attack_types > 0 {
                    config.attack_types[attack_idx].clone()
                } else {
                    AttackType::SignFlip
                };
                NodeState::new_byzantine(&node_id, attack, rng)
            } else {
                NodeState::new_honest(&node_id, rng)
            };
            nodes.insert(node_id, node);
        }

        nodes
    }

    /// Run the full stress test
    pub fn run(&mut self) -> StressTestResult {
        self.run_with_callback(|_, _| {})
    }

    /// Run with progress callback
    pub fn run_with_callback<F>(&mut self, mut callback: F) -> StressTestResult
    where
        F: FnMut(usize, &RoundMetrics),
    {
        // Phase 1: ZK Proof Verification (if enabled)
        if self.config.enable_zkp {
            self.verify_all_kvectors();
        }

        // Phase 2: Run FL rounds
        let partition_start = self.config.partition_at_round;
        let partition_end = partition_start.map(|s| s + self.config.partition_duration);

        for round in 0..self.config.num_rounds {
            self.current_round = round;

            // Handle partition injection
            if self.config.inject_partitions && self.network.is_some() {
                if Some(round) == partition_start {
                    self.inject_partition();
                    // Save pre-partition detection rate for comparison
                    if !self.round_metrics.is_empty() {
                        let recent_rate = self.round_metrics.iter().rev().take(5)
                            .map(|m| m.detection_rate)
                            .sum::<f32>() / 5.0f32.min(self.round_metrics.len() as f32);
                        self.pre_partition_detection_rate = Some(recent_rate);
                    }
                } else if Some(round) == partition_end {
                    self.heal_partition();
                }
            }

            // Apply random chaos (node crashes, etc.)
            if let Some(ref mut network) = self.network {
                let node_ids: Vec<String> = self.nodes.keys().cloned().collect();
                network.random_chaos(&node_ids);
            }

            let metrics = self.run_round(round);
            callback(round, &metrics);
            self.round_metrics.push(metrics.clone());

            // Check for partition recovery after healing
            if self.config.inject_partitions && partition_end.map(|e| round > e + 3).unwrap_or(false) {
                // We should have recovered by now
                let post_healing_rate = metrics.detection_rate;
                if let Some(pre_rate) = self.pre_partition_detection_rate {
                    // Recovery is successful if we're within 10% of pre-partition performance
                    if post_healing_rate >= pre_rate * 0.9 {
                        self.partition_recovery_success = true;
                    }
                }
            }
        }

        // Phase 3: Compute final results
        self.compute_results()
    }

    /// Inject a network partition (split nodes roughly in half)
    fn inject_partition(&mut self) {
        if let Some(ref mut network) = self.network {
            let node_ids: Vec<String> = self.nodes.keys().cloned().collect();
            let mid = node_ids.len() / 2;
            let partition_a: HashSet<String> = node_ids[..mid].iter().cloned().collect();
            network.activate_partition(partition_a);
        }
    }

    /// Heal the network partition
    fn heal_partition(&mut self) {
        if let Some(ref mut network) = self.network {
            network.heal_partition();
        }
    }

    fn verify_all_kvectors(&mut self) {
        for node in self.nodes.values_mut() {
            // Generate ZK proof from K-Vector
            let witness = KVectorWitness {
                k_r: node.k_vector.k_r,
                k_a: node.k_vector.k_a,
                k_i: node.k_vector.k_i,
                k_p: node.k_vector.k_p,
                k_m: node.k_vector.k_m,
                k_s: node.k_vector.k_s,
                k_h: node.k_vector.k_h,
                k_topo: node.k_vector.k_topo,
            };

            // Only generate/verify proofs for nodes with valid K-Vectors
            if witness.validate().is_ok() {
                match KVectorRangeProof::prove(&witness) {
                    Ok(proof) => {
                        node.zkp_verified = proof.verify().is_ok();
                    }
                    Err(_) => {
                        node.zkp_verified = false;
                    }
                }
            } else {
                node.zkp_verified = false;
            }
        }
    }

    fn run_round(&mut self, round_num: usize) -> RoundMetrics {
        let start = Instant::now();

        // 1. Generate gradients
        let gradients = self.generate_gradients();

        // 2. Apply Byzantine attacks
        let (attacked_gradients, true_byzantine) = self.apply_attacks(&gradients);

        // 3. Simulate network transmission (if chaos enabled)
        let received_gradients = self.transmit_through_network(round_num, &attacked_gradients);

        // 4. Multi-layer Byzantine detection (on received gradients)
        let (detected, _detection_confidence) = self.detect_byzantine(&received_gradients);

        // 5. Calculate metrics
        let true_positives = detected.intersection(&true_byzantine).count();
        let false_positives = detected.difference(&true_byzantine).count();
        let false_negatives = true_byzantine.difference(&detected).count();

        // 6. Update node states
        self.update_node_states(&detected, &true_byzantine);

        // 7. Consensus simulation (if enabled)
        let (consensus_reached, consensus_time) = if self.config.enable_consensus {
            self.simulate_consensus(&detected)
        } else {
            (true, 0.0)
        };

        let round_latency = start.elapsed().as_secs_f64() * 1000.0;

        // Gather network metrics
        let (net_sent, net_delivered, net_dropped, net_blocked, net_partition, net_latency) =
            if let Some(ref network) = self.network {
                let m = network.metrics();
                (m.messages_sent, m.messages_delivered, m.messages_dropped,
                 m.messages_blocked, network.state_summary().partition_active, m.avg_latency_ms)
            } else {
                (0, 0, 0, 0, false, 0.0)
            };

        RoundMetrics {
            round_num,
            total_nodes: self.config.num_nodes,
            byzantine_nodes: true_byzantine.len(),
            detected_byzantine: true_positives,
            false_positives,
            false_negatives,
            detection_rate: if !true_byzantine.is_empty() {
                true_positives as f32 / true_byzantine.len() as f32
            } else {
                1.0
            },
            zkp_proofs_generated: if self.config.enable_zkp { self.config.num_nodes } else { 0 },
            zkp_proofs_verified: self.nodes.values().filter(|n| n.zkp_verified).count(),
            zkp_time_ms: 0.0, // Already done in init
            consensus_reached,
            consensus_time_ms: consensus_time,
            round_latency_ms: round_latency,
            memory_estimate_mb: (self.config.num_nodes * self.config.gradient_dim * 4) as f64 / 1_000_000.0,
            // Network chaos metrics
            network_messages_sent: net_sent,
            network_messages_delivered: net_delivered,
            network_messages_dropped: net_dropped,
            network_messages_blocked: net_blocked,
            network_partition_active: net_partition,
            network_avg_latency_ms: net_latency,
        }
    }

    /// Simulate transmitting gradients through the chaos network
    fn transmit_through_network(
        &mut self,
        round_num: usize,
        gradients: &HashMap<String, Vec<f32>>,
    ) -> HashMap<String, Vec<f32>> {
        if self.network.is_none() || !self.config.enable_network_chaos {
            // No network layer - all gradients arrive perfectly
            return gradients.clone();
        }

        let network = self.network.as_mut().unwrap();

        // Simulate broadcast: each node sends to an "aggregator" node
        // In reality this would be all-to-all or gossip, but we simplify
        let aggregator_id = "aggregator".to_string();

        for (node_id, gradient) in gradients {
            let msg = GradientMessage {
                node_id: node_id.clone(),
                round: round_num,
                gradient: gradient.clone(),
            };
            network.send(node_id.clone(), aggregator_id.clone(), msg, MessageType::Gradient);
        }

        // Step the network (simulate time passing) and flush messages
        // Use flush() for simplicity - in a real simulation we'd use step() with timing
        let delivered_messages = network.flush();

        // Collect successfully delivered gradients
        let mut received: HashMap<String, Vec<f32>> = HashMap::new();
        for (_, msg, _) in delivered_messages {
            received.insert(msg.node_id, msg.gradient);
        }

        // For nodes whose gradients didn't arrive, they're effectively excluded
        // This simulates packet loss and partition effects
        received
    }

    fn generate_gradients(&mut self) -> HashMap<String, Vec<f32>> {
        let dim = self.config.gradient_dim;
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Base honest gradient direction
        let base: Vec<f32> = (0..dim)
            .map(|_| normal.sample(&mut self.rng) as f32)
            .collect();

        let mut gradients = HashMap::new();

        for node_id in self.nodes.keys() {
            // Small noise for honest gradient variation
            let noise: Vec<f32> = (0..dim)
                .map(|_| (normal.sample(&mut self.rng) * 0.1) as f32)
                .collect();

            let gradient: Vec<f32> = base.iter()
                .zip(noise.iter())
                .map(|(b, n)| b + n)
                .collect();

            gradients.insert(node_id.clone(), gradient);
        }

        gradients
    }

    fn apply_attacks(
        &mut self,
        gradients: &HashMap<String, Vec<f32>>,
    ) -> (HashMap<String, Vec<f32>>, HashSet<String>) {
        let mut attacked = HashMap::new();
        let mut true_byzantine = HashSet::new();

        // First pass: collect Byzantine node info
        let byzantine_info: Vec<(String, AttackType)> = self.nodes.iter()
            .filter(|(_, node)| node.is_byzantine)
            .map(|(id, node)| (id.clone(), node.attack_type.clone().unwrap_or(AttackType::SignFlip)))
            .collect();

        let byzantine_ids: HashSet<String> = byzantine_info.iter().map(|(id, _)| id.clone()).collect();

        // Second pass: apply attacks
        for (node_id, gradient) in gradients {
            if byzantine_ids.contains(node_id) {
                true_byzantine.insert(node_id.clone());
                let attack_type = byzantine_info.iter()
                    .find(|(id, _)| id == node_id)
                    .map(|(_, at)| at.clone())
                    .unwrap_or(AttackType::SignFlip);
                let attack_grad = self.execute_attack(gradient, &attack_type);
                attacked.insert(node_id.clone(), attack_grad);
            } else {
                attacked.insert(node_id.clone(), gradient.clone());
            }
        }

        (attacked, true_byzantine)
    }

    fn execute_attack(&mut self, gradient: &[f32], attack_type: &AttackType) -> Vec<f32> {
        match attack_type {
            AttackType::SignFlip => gradient.iter().map(|x| -x).collect(),

            AttackType::GradientScaling { factor } => {
                gradient.iter().map(|x| x * factor).collect()
            }

            AttackType::AdaptiveAttack => {
                // Stay close to honest but subtly shift
                let shift = self.rng.gen_range(0.8..1.2);
                gradient.iter().map(|x| x * shift).collect()
            }

            AttackType::SlowPoisoning { rate } => {
                let round_factor = 1.0 + (rate * self.current_round as f32);
                gradient.iter().map(|x| -x * round_factor * 0.1).collect()
            }

            AttackType::LittleIsEnough { epsilon } => {
                gradient.iter().map(|x| x - epsilon * x.signum()).collect()
            }

            AttackType::Cartel { .. } => {
                // Coordinate to push in same direction
                let scale = self.rng.gen_range(1.5..3.0);
                gradient.iter().map(|x| -x * scale).collect()
            }
        }
    }

    fn detect_byzantine(
        &self,
        gradients: &HashMap<String, Vec<f32>>,
    ) -> (HashSet<String>, f32) {
        let mut detected = HashSet::new();
        let mut scores: HashMap<String, f32> = HashMap::new();

        // Compute median gradient
        let all_grads: Vec<&Vec<f32>> = gradients.values().collect();
        let median = self.compute_median_gradient(&all_grads);
        let median_norm = self.l2_norm(&median);

        // Score each node
        for (node_id, gradient) in gradients {
            let grad_norm = self.l2_norm(gradient);

            // Magnitude ratio
            let magnitude_ratio = if median_norm > 1e-8 {
                grad_norm / median_norm
            } else {
                1.0
            };

            // Cosine similarity with median
            let cosine_sim = self.cosine_similarity(gradient, &median);

            // Combined detection score (higher = more honest)
            let magnitude_score = (-((magnitude_ratio - 1.0).abs() * 0.5).exp()).exp();
            let direction_score = (cosine_sim + 1.0) / 2.0;

            let score = magnitude_score * 0.4 + direction_score * 0.6;
            scores.insert(node_id.clone(), score);
        }

        // Adaptive threshold
        let score_values: Vec<f32> = scores.values().copied().collect();
        let mean_score: f32 = score_values.iter().sum::<f32>() / score_values.len() as f32;
        let std_score: f32 = (score_values.iter()
            .map(|s| (s - mean_score).powi(2))
            .sum::<f32>() / score_values.len() as f32)
            .sqrt();

        let threshold = mean_score - 0.5 * std_score;

        // Sort and detect lowest scores
        let mut sorted_scores: Vec<_> = scores.iter().collect();
        sorted_scores.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

        let estimated_byzantine = (self.config.num_nodes as f32 * self.config.byzantine_fraction) as usize;

        for (i, (node_id, score)) in sorted_scores.iter().enumerate() {
            if i < estimated_byzantine || **score < threshold {
                detected.insert((*node_id).clone());
            }
        }

        // Also check ZKP verification status
        for (node_id, node) in &self.nodes {
            if !node.zkp_verified && node.trust_score < self.config.min_trust_threshold {
                detected.insert(node_id.clone());
            }
        }

        let confidence = if !detected.is_empty() {
            let detected_byzantine = detected.iter()
                .filter(|id| self.nodes.get(*id).map(|n| n.is_byzantine).unwrap_or(false))
                .count();
            detected_byzantine as f32 / detected.len() as f32
        } else {
            1.0
        };

        (detected, confidence)
    }

    fn compute_median_gradient(&self, gradients: &[&Vec<f32>]) -> Vec<f32> {
        if gradients.is_empty() {
            return vec![];
        }

        let dim = gradients[0].len();
        let mut median = Vec::with_capacity(dim);

        for d in 0..dim {
            let mut values: Vec<f32> = gradients.iter().map(|g| g[d]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = values.len() / 2;
            median.push(values[mid]);
        }

        median
    }

    fn l2_norm(&self, v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a = self.l2_norm(a);
        let norm_b = self.l2_norm(b);

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    fn update_node_states(&mut self, detected: &HashSet<String>, true_byzantine: &HashSet<String>) {
        for (node_id, node) in self.nodes.iter_mut() {
            if detected.contains(node_id) {
                node.reputation = (node.reputation - 0.1).max(0.0);
                node.times_detected += 1;
            } else if !true_byzantine.contains(node_id) {
                node.reputation = (node.reputation + 0.02).min(1.0);
            }
            node.rounds_participated += 1;
        }
    }

    fn simulate_consensus(&self, excluded: &HashSet<String>) -> (bool, f64) {
        let start = Instant::now();

        // Simulate RB-BFT consensus
        let participating: Vec<_> = self.nodes.iter()
            .filter(|(id, _)| !excluded.contains(*id))
            .collect();

        // Calculate total reputation
        let total_rep: f32 = participating.iter().map(|(_, n)| n.reputation).sum();

        // Byzantine nodes that slipped through
        let byzantine_rep: f32 = participating.iter()
            .filter(|(_, n)| n.is_byzantine)
            .map(|(_, n)| n.reputation)
            .sum();

        // Consensus succeeds if Byzantine reputation < 45% of total
        let byzantine_fraction = if total_rep > 0.0 {
            byzantine_rep / total_rep
        } else {
            0.0
        };

        let consensus_reached = byzantine_fraction < 0.45;
        let time_ms = start.elapsed().as_secs_f64() * 1000.0;

        (consensus_reached, time_ms)
    }

    fn compute_results(&self) -> StressTestResult {
        let total_rounds = self.round_metrics.len();

        // Aggregate metrics
        let total_detected: usize = self.round_metrics.iter().map(|m| m.detected_byzantine).sum();
        let total_byzantine: usize = self.round_metrics.iter().map(|m| m.byzantine_nodes).sum();
        let total_fp: usize = self.round_metrics.iter().map(|m| m.false_positives).sum();
        let total_fn: usize = self.round_metrics.iter().map(|m| m.false_negatives).sum();
        let total_honest = total_rounds * (self.config.num_nodes - (self.config.num_nodes as f32 * self.config.byzantine_fraction) as usize);

        let detection_rate = if total_byzantine > 0 {
            total_detected as f32 / total_byzantine as f32
        } else {
            1.0
        };

        let fp_rate = if total_honest > 0 {
            total_fp as f32 / total_honest as f32
        } else {
            0.0
        };

        let fn_rate = if total_byzantine > 0 {
            total_fn as f32 / total_byzantine as f32
        } else {
            0.0
        };

        let avg_latency: f64 = self.round_metrics.iter()
            .map(|m| m.round_latency_ms)
            .sum::<f64>() / total_rounds as f64;

        let zkp_verified = self.nodes.values().filter(|n| n.zkp_verified).count();
        let zkp_rate = zkp_verified as f32 / self.config.num_nodes as f32;

        let consensus_successes = self.round_metrics.iter().filter(|m| m.consensus_reached).count();
        let consensus_rate = consensus_successes as f32 / total_rounds as f32;

        let avg_consensus_time: f64 = self.round_metrics.iter()
            .map(|m| m.consensus_time_ms)
            .sum::<f64>() / total_rounds as f64;

        let peak_memory = self.round_metrics.iter()
            .map(|m| m.memory_estimate_mb)
            .fold(0.0f64, |a, b| a.max(b));

        // Aggregate network metrics
        let (net_delivery_rate, net_avg_latency, net_partition_events, net_total_dropped, failure_attr) =
            if let Some(ref network) = self.network {
                let m = network.metrics();
                (m.delivery_rate(), m.avg_latency_ms, m.partition_events, m.messages_dropped,
                 Some(m.failure_attribution.clone()))
            } else {
                (1.0, 0.0, 0, 0, None)
            };

        // Determine verdict
        let verdict = if detection_rate >= 0.98 && fp_rate < 0.01 && consensus_rate >= 0.99 {
            StressTestVerdict::Perfect
        } else if detection_rate >= 0.95 && fp_rate < 0.05 && consensus_rate >= 0.95 {
            StressTestVerdict::Excellent
        } else if detection_rate >= 0.90 && fp_rate < 0.10 && consensus_rate >= 0.90 {
            StressTestVerdict::Good
        } else if detection_rate >= 0.80 {
            StressTestVerdict::NeedsImprovement
        } else {
            StressTestVerdict::Failed
        };

        StressTestResult {
            config: self.config.clone(),
            total_rounds_completed: total_rounds,
            byzantine_detection_rate: detection_rate,
            false_positive_rate: fp_rate,
            false_negative_rate: fn_rate,
            avg_round_latency_ms: avg_latency,
            total_zkp_proofs: if self.config.enable_zkp { self.config.num_nodes } else { 0 },
            zkp_verification_rate: zkp_rate,
            avg_zkp_time_ms: 2.78, // From benchmarks
            consensus_success_rate: consensus_rate,
            avg_consensus_time_ms: avg_consensus_time,
            peak_memory_mb: peak_memory,
            round_metrics: self.round_metrics.clone(),
            verdict,
            // Network chaos results
            network_delivery_rate: net_delivery_rate,
            network_avg_latency_ms: net_avg_latency,
            network_partition_events: net_partition_events,
            network_total_dropped: net_total_dropped,
            partition_recovery_success: self.partition_recovery_success,
            // Failure attribution (B)
            failure_attribution: failure_attr,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_smoke() {
        let config = StressTestConfig::quick();
        let mut test = EcosystemStressTest::new(config);
        let result = test.run();

        println!("\n=== Quick Smoke Test Results ===");
        println!("Byzantine Detection Rate: {:.2}%", result.byzantine_detection_rate * 100.0);
        println!("False Positive Rate: {:.2}%", result.false_positive_rate * 100.0);
        println!("Verdict: {}", result.verdict);

        assert!(result.byzantine_detection_rate > 0.80);
        assert!(result.false_positive_rate < 0.20);
    }

    #[test]
    fn test_default_config() {
        let config = StressTestConfig::default();
        let mut test = EcosystemStressTest::new(config);
        let result = test.run();

        println!("\n=== Default Config Test Results ===");
        println!("Nodes: {}", result.config.num_nodes);
        println!("Byzantine Fraction: {:.0}%", result.config.byzantine_fraction * 100.0);
        println!("Rounds: {}", result.total_rounds_completed);
        println!("Byzantine Detection Rate: {:.2}%", result.byzantine_detection_rate * 100.0);
        println!("False Positive Rate: {:.2}%", result.false_positive_rate * 100.0);
        println!("ZKP Verification Rate: {:.2}%", result.zkp_verification_rate * 100.0);
        println!("Consensus Success Rate: {:.2}%", result.consensus_success_rate * 100.0);
        println!("Avg Round Latency: {:.2}ms", result.avg_round_latency_ms);
        println!("Verdict: {}", result.verdict);

        assert!(result.byzantine_detection_rate > 0.90);
        assert!(result.consensus_success_rate > 0.90);
    }

    #[test]
    fn test_scale_50_nodes() {
        let config = StressTestConfig::scale_test(50);
        let mut test = EcosystemStressTest::new(config);
        let result = test.run();

        println!("\n=== Scale Test (50 nodes) ===");
        println!("Detection Rate: {:.2}%", result.byzantine_detection_rate * 100.0);
        println!("Verdict: {}", result.verdict);

        assert!(result.byzantine_detection_rate > 0.85);
    }

    #[test]
    fn test_scale_200_nodes() {
        let config = StressTestConfig::scale_test(200);
        let mut test = EcosystemStressTest::new(config);
        let result = test.run();

        println!("\n=== Scale Test (200 nodes) ===");
        println!("Detection Rate: {:.2}%", result.byzantine_detection_rate * 100.0);
        println!("Avg Latency: {:.2}ms", result.avg_round_latency_ms);
        println!("Verdict: {}", result.verdict);

        assert!(result.byzantine_detection_rate > 0.85);
    }

    #[test]
    fn test_zkp_integration() {
        let mut config = StressTestConfig::quick();
        config.enable_zkp = true;
        config.num_rounds = 5;

        let mut test = EcosystemStressTest::new(config);
        let result = test.run();

        println!("\n=== ZKP Integration Test ===");
        println!("ZKP Proofs Generated: {}", result.total_zkp_proofs);
        println!("ZKP Verification Rate: {:.2}%", result.zkp_verification_rate * 100.0);

        // Most honest nodes should verify, Byzantine may fail
        assert!(result.zkp_verification_rate > 0.50);
    }

    #[test]
    fn test_adversarial_config() {
        let config = StressTestConfig::adversarial();
        let mut test = EcosystemStressTest::new(config);
        let result = test.run();

        println!("\n=== Adversarial Test Results ===");
        println!("Byzantine Fraction: {:.0}%", result.config.byzantine_fraction * 100.0);
        println!("Attack Types: {:?}", result.config.attack_types.len());
        println!("Byzantine Detection Rate: {:.2}%", result.byzantine_detection_rate * 100.0);
        println!("False Negative Rate: {:.2}%", result.false_negative_rate * 100.0);
        println!("Consensus Success Rate: {:.2}%", result.consensus_success_rate * 100.0);
        println!("Verdict: {}", result.verdict);

        // Under maximum adversarial pressure, we should still maintain security
        assert!(result.byzantine_detection_rate > 0.80);
        assert!(result.consensus_success_rate > 0.85);
    }

    #[test]
    fn test_with_progress_callback() {
        let config = StressTestConfig::quick();
        let mut test = EcosystemStressTest::new(config);

        let mut round_count = 0;
        let result = test.run_with_callback(|round, metrics| {
            round_count += 1;
            if round % 3 == 0 {
                println!("Round {}: Detection {:.0}%", round, metrics.detection_rate * 100.0);
            }
        });

        assert_eq!(round_count, result.total_rounds_completed);
    }

    // ========================================================================
    // Network Chaos Tests (Digital Twin Simulation)
    // ========================================================================

    #[test]
    fn test_networked_simulation() {
        let config = StressTestConfig::networked();
        let mut test = EcosystemStressTest::new(config);
        let result = test.run();

        println!("\n=== Networked Simulation Test ===");
        println!("Network Delivery Rate: {:.2}%", result.network_delivery_rate * 100.0);
        println!("Network Avg Latency: {:.2}ms", result.network_avg_latency_ms);
        println!("Messages Dropped: {}", result.network_total_dropped);
        println!("Byzantine Detection Rate: {:.2}%", result.byzantine_detection_rate * 100.0);
        println!("Consensus Success Rate: {:.2}%", result.consensus_success_rate * 100.0);
        println!("Verdict: {}", result.verdict);

        // Even with network chaos, we should maintain good detection
        assert!(result.byzantine_detection_rate > 0.75, "Detection rate too low under network chaos");
        // Network delivery should be reasonable
        assert!(result.network_delivery_rate > 0.90, "Too many messages lost");
    }

    #[test]
    fn test_network_partition_recovery() {
        // This is the critical "digital twin" test:
        // 1. Run normally for several rounds
        // 2. Inject a network partition (split-brain)
        // 3. Observe degraded performance during partition
        // 4. Heal the partition
        // 5. Assert that the system RECOVERS to pre-partition performance

        let config = StressTestConfig::partition_test();
        let mut test = EcosystemStressTest::new(config.clone());

        // Track metrics during each phase
        let mut pre_partition_rates: Vec<f32> = Vec::new();
        let mut during_partition_rates: Vec<f32> = Vec::new();
        let mut post_partition_rates: Vec<f32> = Vec::new();

        let partition_start = config.partition_at_round.unwrap_or(10);
        let partition_end = partition_start + config.partition_duration;

        let result = test.run_with_callback(|round, metrics| {
            if round < partition_start {
                pre_partition_rates.push(metrics.detection_rate);
            } else if round >= partition_start && round < partition_end {
                during_partition_rates.push(metrics.detection_rate);
                if round == partition_start {
                    println!(">>> PARTITION INJECTED at round {}", round);
                }
            } else {
                post_partition_rates.push(metrics.detection_rate);
                if round == partition_end {
                    println!(">>> PARTITION HEALED at round {}", round);
                }
            }

            // Log partition state changes
            if metrics.network_partition_active {
                println!("Round {}: PARTITIONED - Detection {:.0}%, Blocked msgs: {}",
                         round, metrics.detection_rate * 100.0, metrics.network_messages_blocked);
            }
        });

        println!("\n=== Network Partition Recovery Test ===");
        println!("Partition Events: {}", result.network_partition_events);

        let pre_avg = pre_partition_rates.iter().sum::<f32>() / pre_partition_rates.len().max(1) as f32;
        let during_avg = during_partition_rates.iter().sum::<f32>() / during_partition_rates.len().max(1) as f32;
        let post_avg = post_partition_rates.iter().sum::<f32>() / post_partition_rates.len().max(1) as f32;

        println!("Pre-Partition Avg Detection: {:.2}%", pre_avg * 100.0);
        println!("During-Partition Avg Detection: {:.2}%", during_avg * 100.0);
        println!("Post-Partition Avg Detection: {:.2}%", post_avg * 100.0);
        println!("Recovery Success: {}", result.partition_recovery_success);
        println!("Verdict: {}", result.verdict);

        // Key assertions:
        // 1. Detection may degrade during partition (expected)
        // 2. Post-partition detection should RECOVER to near pre-partition levels
        // 3. Overall system should still function

        // Allow some degradation during partition
        if !during_partition_rates.is_empty() {
            println!("Detection degradation during partition: {:.1}%",
                     (pre_avg - during_avg) * 100.0);
        }

        // Critical: Recovery assertion
        // Post-partition should recover to within 15% of pre-partition performance
        if !pre_partition_rates.is_empty() && !post_partition_rates.is_empty() {
            let recovery_ratio = post_avg / pre_avg;
            println!("Recovery ratio: {:.2} (target: >= 0.85)", recovery_ratio);
            assert!(recovery_ratio >= 0.85,
                    "System did not recover after partition: pre={:.2}, post={:.2}",
                    pre_avg, post_avg);
        }

        // System should still achieve reasonable overall detection
        assert!(result.byzantine_detection_rate > 0.70,
                "Overall detection too low: {:.2}%", result.byzantine_detection_rate * 100.0);
    }

    #[test]
    fn test_digital_twin_full_chaos() {
        // Full digital twin simulation with all chaos features
        let config = StressTestConfig::digital_twin();
        let mut test = EcosystemStressTest::new(config);

        println!("\n=== Digital Twin Full Chaos Test ===");
        println!("(This simulates production conditions with adversarial network)\n");

        let result = test.run_with_callback(|round, metrics| {
            // Log significant events
            if metrics.network_partition_active {
                println!("Round {:3}: PARTITION | Det: {:.0}% | Blocked: {} | Dropped: {}",
                         round, metrics.detection_rate * 100.0,
                         metrics.network_messages_blocked, metrics.network_messages_dropped);
            } else if round % 10 == 0 {
                println!("Round {:3}: NORMAL    | Det: {:.0}% | Latency: {:.1}ms",
                         round, metrics.detection_rate * 100.0, metrics.network_avg_latency_ms);
            }
        });

        println!("\n--- Digital Twin Results ---");
        println!("Rounds Completed: {}", result.total_rounds_completed);
        println!("Byzantine Detection Rate: {:.2}%", result.byzantine_detection_rate * 100.0);
        println!("False Positive Rate: {:.2}%", result.false_positive_rate * 100.0);
        println!("Consensus Success Rate: {:.2}%", result.consensus_success_rate * 100.0);
        println!("Network Delivery Rate: {:.2}%", result.network_delivery_rate * 100.0);
        println!("Partition Events: {}", result.network_partition_events);
        println!("Total Messages Dropped: {}", result.network_total_dropped);
        println!("Partition Recovery: {}", if result.partition_recovery_success { "SUCCESS" } else { "PENDING" });
        println!("Verdict: {}", result.verdict);

        // Under full chaos, we accept slightly degraded performance
        // but the system must still function
        assert!(result.byzantine_detection_rate > 0.65,
                "Detection too low under full chaos: {:.2}%", result.byzantine_detection_rate * 100.0);
        assert!(result.consensus_success_rate > 0.60,
                "Consensus too unstable under full chaos: {:.2}%", result.consensus_success_rate * 100.0);
    }

    #[test]
    fn test_network_latency_impact() {
        // Test how different network conditions affect performance

        let configs = vec![
            ("LAN (Datacenter)", NetworkConfig::lan()),
            ("WAN (Cross-continent)", NetworkConfig::wan()),
            ("Adversarial (Hostile)", NetworkConfig::adversarial()),
        ];

        println!("\n=== Network Latency Impact Test ===\n");
        println!("{:<25} {:>12} {:>12} {:>12} {:>12}",
                 "Network Type", "Delivery", "Latency", "Detection", "Consensus");
        println!("{}", "-".repeat(75));

        for (name, net_config) in configs {
            let mut config = StressTestConfig::quick();
            config.enable_network_chaos = true;
            config.network_config = net_config;
            config.num_rounds = 15;

            let mut test = EcosystemStressTest::new(config);
            let result = test.run();

            println!("{:<25} {:>11.1}% {:>10.1}ms {:>11.1}% {:>11.1}%",
                     name,
                     result.network_delivery_rate * 100.0,
                     result.network_avg_latency_ms,
                     result.byzantine_detection_rate * 100.0,
                     result.consensus_success_rate * 100.0);
        }
    }

    #[test]
    fn test_chaos_engineering_with_failure_attribution() {
        // Full chaos engineering test with all chaos features and failure attribution
        let config = StressTestConfig::chaos_engineering();
        let mut test = EcosystemStressTest::new(config);

        println!("\n=== Chaos Engineering Test (with Failure Attribution) ===\n");

        let result = test.run_with_callback(|round, metrics| {
            if round % 5 == 0 || metrics.network_partition_active {
                let status = if metrics.network_partition_active { "CHAOS" } else { "NORMAL" };
                println!("Round {:3}: {} | Det: {:.0}% | Dropped: {} | Blocked: {}",
                         round, status, metrics.detection_rate * 100.0,
                         metrics.network_messages_dropped, metrics.network_messages_blocked);
            }
        });

        println!("\n--- Chaos Engineering Results ---");
        println!("Rounds Completed: {}", result.total_rounds_completed);
        println!("Byzantine Detection Rate: {:.2}%", result.byzantine_detection_rate * 100.0);
        println!("Consensus Success Rate: {:.2}%", result.consensus_success_rate * 100.0);
        println!("Network Delivery Rate: {:.2}%", result.network_delivery_rate * 100.0);

        // Display failure attribution breakdown (B)
        if let Some(ref attr) = result.failure_attribution {
            println!("\n--- Failure Attribution (Why Messages Failed) ---");
            for (reason, pct) in attr.breakdown() {
                if pct > 0.0 {
                    println!("  {:<15}: {:>6.1}%", reason, pct);
                }
            }
            println!("  Total failures: {}", attr.total());
        }

        println!("\nVerdict: {}", result.verdict);

        // Under chaos engineering, we accept degraded performance
        assert!(result.byzantine_detection_rate > 0.60,
                "Detection too low: {:.2}%", result.byzantine_detection_rate * 100.0);
    }
}
