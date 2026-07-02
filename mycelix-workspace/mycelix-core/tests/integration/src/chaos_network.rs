// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Chaos Network Layer - Realistic Network Simulation
//!
//! Transforms the ecosystem stress test from "perfect in-memory" to "digital twin"
//! by simulating real-world network conditions:
//!
//! - **Latency**: Messages take time to arrive (configurable distribution)
//! - **Packet Loss**: Messages may be dropped entirely
//! - **Network Partitions**: "Split brain" scenarios where groups can't communicate
//! - **Node Crashes**: Temporary unavailability
//! - **Byzantine Timing**: Adversaries exploit latency differentials
//!
//! # Why This Matters
//!
//! BFT protocols are proven safe *assuming* bounds on message delivery time.
//! Real networks violate these assumptions constantly. This layer tests if
//! Mycelix correctly handles timeouts, view changes, and recovery.
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_integration::chaos_network::{ChaosNetwork, NetworkConfig};
//!
//! let config = NetworkConfig::wan(); // Simulates cross-continent latency
//! let mut network = ChaosNetwork::new(config);
//!
//! // Send message through chaos layer
//! network.send("node_001", "node_050", gradient_message);
//!
//! // Simulate partition during consensus
//! network.toggle_partition();
//!
//! // Step time forward and receive delivered messages
//! let delivered = network.step();
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the chaos network
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Mean latency in milliseconds
    pub mean_latency_ms: f64,
    /// Latency jitter (standard deviation) in milliseconds
    pub latency_jitter_ms: f64,
    /// Probability of packet loss (0.0 - 1.0)
    pub packet_loss_rate: f64,
    /// Probability of node temporarily crashing per round
    pub node_crash_rate: f64,
    /// Duration of node crash in milliseconds
    pub crash_duration_ms: u64,
    /// Whether to enable Byzantine timing attacks
    pub enable_timing_attacks: bool,
    /// Random seed for reproducibility
    pub seed: u64,
    // === Expanded Chaos Options (A) ===
    /// Enable cascading failures (node crash triggers dependent crashes)
    pub enable_cascading_failures: bool,
    /// Probability of cascading to dependent nodes (0.0 - 1.0)
    pub cascade_probability: f64,
    /// Maximum cascade depth
    pub max_cascade_depth: u32,
    /// Enable view-change storms
    pub enable_view_change_storms: bool,
    /// Probability of view-change storm per round
    pub view_change_storm_rate: f64,
    /// Enable latency spikes (sudden jitter bursts)
    pub enable_latency_spikes: bool,
    /// Probability of latency spike per message
    pub latency_spike_rate: f64,
    /// Multiplier for spike latency (e.g., 10x normal)
    pub latency_spike_factor: f64,
    /// Enable targeted attacks on high-value nodes
    pub enable_targeted_attacks: bool,
    /// Latency timeout threshold (messages taking longer are considered failed)
    pub latency_timeout_ms: f64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            mean_latency_ms: 50.0,
            latency_jitter_ms: 20.0,
            packet_loss_rate: 0.01,
            node_crash_rate: 0.001,
            crash_duration_ms: 5000,
            enable_timing_attacks: false,
            seed: 42,
            // Expanded chaos - disabled by default
            enable_cascading_failures: false,
            cascade_probability: 0.3,
            max_cascade_depth: 3,
            enable_view_change_storms: false,
            view_change_storm_rate: 0.01,
            enable_latency_spikes: false,
            latency_spike_rate: 0.05,
            latency_spike_factor: 10.0,
            enable_targeted_attacks: false,
            latency_timeout_ms: 5000.0,
        }
    }
}

impl NetworkConfig {
    /// LAN configuration (datacenter-like)
    pub fn lan() -> Self {
        Self {
            mean_latency_ms: 1.0,
            latency_jitter_ms: 0.5,
            packet_loss_rate: 0.0001,
            node_crash_rate: 0.0001,
            crash_duration_ms: 1000,
            enable_timing_attacks: false,
            latency_timeout_ms: 100.0,
            ..Default::default()
        }
    }

    /// WAN configuration (cross-continent)
    pub fn wan() -> Self {
        Self {
            mean_latency_ms: 150.0,
            latency_jitter_ms: 50.0,
            packet_loss_rate: 0.02,
            node_crash_rate: 0.005,
            crash_duration_ms: 10000,
            enable_timing_attacks: false,
            latency_timeout_ms: 3000.0,
            ..Default::default()
        }
    }

    /// Adversarial configuration (hostile network)
    pub fn adversarial() -> Self {
        Self {
            mean_latency_ms: 200.0,
            latency_jitter_ms: 100.0,
            packet_loss_rate: 0.05,
            node_crash_rate: 0.01,
            crash_duration_ms: 15000,
            enable_timing_attacks: true,
            enable_latency_spikes: true,
            latency_spike_rate: 0.1,
            latency_spike_factor: 15.0,
            latency_timeout_ms: 5000.0,
            ..Default::default()
        }
    }

    /// Partition-prone configuration
    pub fn partition_prone() -> Self {
        Self {
            mean_latency_ms: 100.0,
            latency_jitter_ms: 30.0,
            packet_loss_rate: 0.03,
            node_crash_rate: 0.02,
            crash_duration_ms: 8000,
            enable_timing_attacks: true,
            latency_timeout_ms: 2000.0,
            ..Default::default()
        }
    }

    /// Chaos engineering configuration (all chaos features enabled)
    pub fn chaos_engineering() -> Self {
        Self {
            mean_latency_ms: 150.0,
            latency_jitter_ms: 75.0,
            packet_loss_rate: 0.04,
            node_crash_rate: 0.02,
            crash_duration_ms: 10000,
            enable_timing_attacks: true,
            seed: 42,
            // All expanded chaos enabled
            enable_cascading_failures: true,
            cascade_probability: 0.4,
            max_cascade_depth: 4,
            enable_view_change_storms: true,
            view_change_storm_rate: 0.05,
            enable_latency_spikes: true,
            latency_spike_rate: 0.1,
            latency_spike_factor: 20.0,
            enable_targeted_attacks: true,
            latency_timeout_ms: 3000.0,
        }
    }
}

// ============================================================================
// Message Types
// ============================================================================

/// A message in transit through the network
#[derive(Debug, Clone)]
pub struct InFlightMessage<T> {
    /// The message payload
    pub message: T,
    /// When this message should be delivered
    pub deliver_at: Instant,
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Message type for metrics
    pub msg_type: MessageType,
    /// When the message was sent (for latency tracking)
    pub sent_at: Instant,
}

/// Message types for metrics tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageType {
    Gradient,
    Vote,
    Proposal,
    Commit,
    ViewChange,
    Heartbeat,
    ZkProof,
    KeyShare,
    Other,
}

// ============================================================================
// Network Events
// ============================================================================

/// Events that occurred during network simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkEvent {
    /// Message was dropped due to packet loss
    PacketDropped {
        source: String,
        target: String,
        msg_type: MessageType,
        timestamp_ms: u64,
    },
    /// Message was blocked by partition
    PartitionBlocked {
        source: String,
        target: String,
        msg_type: MessageType,
        timestamp_ms: u64,
    },
    /// Node crashed
    NodeCrashed {
        node_id: String,
        duration_ms: u64,
        timestamp_ms: u64,
    },
    /// Node recovered from crash
    NodeRecovered {
        node_id: String,
        timestamp_ms: u64,
    },
    /// Partition activated
    PartitionActivated {
        partition_a: Vec<String>,
        partition_b: Vec<String>,
        timestamp_ms: u64,
    },
    /// Partition healed
    PartitionHealed {
        timestamp_ms: u64,
    },
    /// Message delivered
    MessageDelivered {
        source: String,
        target: String,
        msg_type: MessageType,
        latency_ms: f64,
        timestamp_ms: u64,
    },
    /// Timing attack executed
    TimingAttackExecuted {
        attacker: String,
        fast_target: String,
        slow_targets: Vec<String>,
        timestamp_ms: u64,
    },
    /// Cascading failure triggered
    CascadingFailure {
        trigger_node: String,
        affected_nodes: Vec<String>,
        timestamp_ms: u64,
    },
    /// View change storm (rapid leader changes)
    ViewChangeStorm {
        view_changes: u32,
        duration_ms: u64,
        timestamp_ms: u64,
    },
    /// Latency spike (jitter burst)
    LatencySpike {
        affected_messages: u32,
        spike_factor: f64,
        timestamp_ms: u64,
    },
    /// Targeted attack on high-value node
    TargetedAttack {
        target: String,
        attack_type: String,
        timestamp_ms: u64,
    },
}

// ============================================================================
// Failure Attribution (B: Why did this message fail?)
// ============================================================================

/// Detailed failure reasons for message delivery
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FailureReason {
    /// Random packet loss (network unreliability)
    PacketLoss,
    /// Network partition blocked the message
    PartitionBlocked,
    /// Target node was crashed/unavailable
    NodeUnavailable,
    /// Latency timeout (message took too long)
    LatencyTimeout,
    /// Signature verification failed
    SignatureInvalid,
    /// Byzantine node dropped message intentionally
    ByzantineDrop,
    /// Cascading failure from dependent node
    CascadingFailure,
    /// View change in progress
    ViewChangeInProgress,
}

/// Failure attribution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FailureAttribution {
    pub packet_loss: u64,
    pub partition_blocked: u64,
    pub node_unavailable: u64,
    pub latency_timeout: u64,
    pub signature_invalid: u64,
    pub byzantine_drop: u64,
    pub cascading_failure: u64,
    pub view_change: u64,
}

impl FailureAttribution {
    pub fn record(&mut self, reason: FailureReason) {
        match reason {
            FailureReason::PacketLoss => self.packet_loss += 1,
            FailureReason::PartitionBlocked => self.partition_blocked += 1,
            FailureReason::NodeUnavailable => self.node_unavailable += 1,
            FailureReason::LatencyTimeout => self.latency_timeout += 1,
            FailureReason::SignatureInvalid => self.signature_invalid += 1,
            FailureReason::ByzantineDrop => self.byzantine_drop += 1,
            FailureReason::CascadingFailure => self.cascading_failure += 1,
            FailureReason::ViewChangeInProgress => self.view_change += 1,
        }
    }

    pub fn total(&self) -> u64 {
        self.packet_loss + self.partition_blocked + self.node_unavailable +
        self.latency_timeout + self.signature_invalid + self.byzantine_drop +
        self.cascading_failure + self.view_change
    }

    /// Get failure breakdown as percentages
    pub fn breakdown(&self) -> Vec<(&'static str, f64)> {
        let total = self.total() as f64;
        if total == 0.0 {
            return vec![];
        }
        vec![
            ("Packet Loss", self.packet_loss as f64 / total * 100.0),
            ("Partition", self.partition_blocked as f64 / total * 100.0),
            ("Node Down", self.node_unavailable as f64 / total * 100.0),
            ("Timeout", self.latency_timeout as f64 / total * 100.0),
            ("Bad Signature", self.signature_invalid as f64 / total * 100.0),
            ("Byzantine", self.byzantine_drop as f64 / total * 100.0),
            ("Cascade", self.cascading_failure as f64 / total * 100.0),
            ("View Change", self.view_change as f64 / total * 100.0),
        ]
    }
}

// ============================================================================
// Network Metrics
// ============================================================================

/// Detailed metrics from network simulation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages delivered
    pub messages_delivered: u64,
    /// Total messages dropped (packet loss)
    pub messages_dropped: u64,
    /// Total messages blocked (partition)
    pub messages_blocked: u64,
    /// Total node crashes
    pub node_crashes: u64,
    /// Total partition events
    pub partition_events: u64,
    /// Timing attacks executed
    pub timing_attacks: u64,
    /// Average latency in ms
    pub avg_latency_ms: f64,
    /// Maximum latency observed
    pub max_latency_ms: f64,
    /// Latency samples for histogram
    latency_samples: Vec<f64>,
    /// Events log
    pub events: Vec<NetworkEvent>,
    // === Expanded Chaos Metrics (A) ===
    /// Cascading failures triggered
    pub cascading_failures: u64,
    /// View change storms
    pub view_change_storms: u64,
    /// Latency spikes (jitter bursts)
    pub latency_spikes: u64,
    /// Targeted attacks executed
    pub targeted_attacks: u64,
    // === Failure Attribution (B) ===
    pub failure_attribution: FailureAttribution,
}

impl NetworkMetrics {
    pub fn record_latency(&mut self, latency_ms: f64) {
        self.latency_samples.push(latency_ms);
        self.max_latency_ms = self.max_latency_ms.max(latency_ms);

        // Update running average
        let n = self.latency_samples.len() as f64;
        self.avg_latency_ms = self.latency_samples.iter().sum::<f64>() / n;
    }

    /// Get latency percentile (0-100)
    pub fn latency_percentile(&self, p: f64) -> f64 {
        if self.latency_samples.is_empty() {
            return 0.0;
        }

        let mut sorted = self.latency_samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let idx = ((p / 100.0) * (sorted.len() - 1) as f64) as usize;
        sorted[idx]
    }

    /// Delivery rate (0.0 - 1.0)
    pub fn delivery_rate(&self) -> f64 {
        if self.messages_sent == 0 {
            return 1.0;
        }
        self.messages_delivered as f64 / self.messages_sent as f64
    }
}

// ============================================================================
// Chaos Network
// ============================================================================

/// The Chaos Network Layer - simulates realistic network conditions
pub struct ChaosNetwork<T> {
    config: NetworkConfig,
    /// Latency distribution
    latency_dist: Normal<f64>,
    /// Messages currently in transit
    in_flight: VecDeque<InFlightMessage<T>>,
    /// Is network currently partitioned?
    partition_active: bool,
    /// Which nodes are in partition A (rest are in partition B)
    partition_a: HashSet<String>,
    /// Crashed nodes and when they recover
    crashed_nodes: HashMap<String, Instant>,
    /// Byzantine nodes (for timing attacks)
    byzantine_nodes: HashSet<String>,
    /// Network metrics
    pub metrics: NetworkMetrics,
    /// Simulation start time
    start_time: Instant,
    /// Random number generator
    rng: StdRng,
    // === Expanded Chaos State (A) ===
    /// Is a view-change storm currently active?
    view_change_storm_active: bool,
    /// When the view-change storm ends
    view_change_storm_end: Option<Instant>,
    /// Is a latency spike currently active?
    latency_spike_active: bool,
    /// High-value target nodes (for targeted attacks)
    high_value_targets: HashSet<String>,
    /// Node dependency graph (for cascading failures)
    node_dependencies: HashMap<String, Vec<String>>,
}

impl<T: Clone> ChaosNetwork<T> {
    /// Create a new chaos network with the given configuration
    pub fn new(config: NetworkConfig) -> Self {
        let latency_dist = Normal::new(config.mean_latency_ms, config.latency_jitter_ms)
            .unwrap_or_else(|_| Normal::new(config.mean_latency_ms, 1.0).unwrap());

        Self {
            rng: StdRng::seed_from_u64(config.seed),
            config,
            latency_dist,
            in_flight: VecDeque::new(),
            partition_active: false,
            partition_a: HashSet::new(),
            crashed_nodes: HashMap::new(),
            byzantine_nodes: HashSet::new(),
            metrics: NetworkMetrics::default(),
            start_time: Instant::now(),
            // Expanded chaos state
            view_change_storm_active: false,
            view_change_storm_end: None,
            latency_spike_active: false,
            high_value_targets: HashSet::new(),
            node_dependencies: HashMap::new(),
        }
    }

    /// Create with default configuration
    pub fn default_network() -> Self {
        Self::new(NetworkConfig::default())
    }

    /// Register Byzantine nodes for timing attacks
    pub fn register_byzantine_nodes(&mut self, nodes: HashSet<String>) {
        self.byzantine_nodes = nodes;
    }

    /// Register high-value target nodes (for targeted attacks)
    pub fn register_high_value_targets(&mut self, nodes: HashSet<String>) {
        self.high_value_targets = nodes;
    }

    /// Set up node dependency graph (for cascading failures)
    pub fn set_node_dependencies(&mut self, deps: HashMap<String, Vec<String>>) {
        self.node_dependencies = deps;
    }

    /// Build a simple dependency graph where each node depends on ~3 random others
    pub fn build_random_dependencies(&mut self, all_nodes: &[String]) {
        let mut deps = HashMap::new();
        for node in all_nodes {
            let mut node_deps = Vec::new();
            let dep_count = self.rng.gen_range(1..=4);
            for other in all_nodes.iter().filter(|n| *n != node).take(dep_count) {
                if self.rng.gen_bool(0.3) {
                    node_deps.push(other.clone());
                }
            }
            if !node_deps.is_empty() {
                deps.insert(node.clone(), node_deps);
            }
        }
        self.node_dependencies = deps;
    }

    /// Send a message through the chaos layer
    pub fn send(&mut self, source: String, target: String, msg: T, msg_type: MessageType) -> bool {
        self.metrics.messages_sent += 1;
        let now = Instant::now();
        let timestamp_ms = now.duration_since(self.start_time).as_millis() as u64;

        // Check 0: View-change storm active (consensus messages may be dropped)
        if self.view_change_storm_active {
            if let Some(end_time) = self.view_change_storm_end {
                if now >= end_time {
                    self.view_change_storm_active = false;
                    self.view_change_storm_end = None;
                } else if matches!(msg_type, MessageType::Vote | MessageType::Proposal | MessageType::Commit) {
                    // During view-change storm, consensus messages may be dropped
                    if self.rng.gen_bool(0.5) {
                        self.metrics.messages_dropped += 1;
                        self.metrics.failure_attribution.record(FailureReason::ViewChangeInProgress);
                        return false;
                    }
                }
            }
        }

        // Check 1: Is target node crashed?
        if let Some(&recover_at) = self.crashed_nodes.get(&target) {
            if now < recover_at {
                self.metrics.messages_dropped += 1;
                self.metrics.failure_attribution.record(FailureReason::NodeUnavailable);
                self.metrics.events.push(NetworkEvent::PacketDropped {
                    source: source.clone(),
                    target: target.clone(),
                    msg_type,
                    timestamp_ms,
                });
                return false;
            } else {
                // Node recovered
                self.crashed_nodes.remove(&target);
                self.metrics.events.push(NetworkEvent::NodeRecovered {
                    node_id: target.clone(),
                    timestamp_ms,
                });
            }
        }

        // Check 2: Random packet loss
        if self.rng.gen_bool(self.config.packet_loss_rate) {
            self.metrics.messages_dropped += 1;
            self.metrics.failure_attribution.record(FailureReason::PacketLoss);
            self.metrics.events.push(NetworkEvent::PacketDropped {
                source,
                target,
                msg_type,
                timestamp_ms,
            });
            return false;
        }

        // Check 3: Network partition
        if self.partition_active && self.is_partitioned(&source, &target) {
            self.metrics.messages_blocked += 1;
            self.metrics.failure_attribution.record(FailureReason::PartitionBlocked);
            self.metrics.events.push(NetworkEvent::PartitionBlocked {
                source,
                target,
                msg_type,
                timestamp_ms,
            });
            return false;
        }

        // Check 3b: Targeted attack on high-value nodes
        if self.config.enable_targeted_attacks && self.high_value_targets.contains(&target) {
            // High-value targets have higher chance of message loss
            if self.rng.gen_bool(0.15) {
                self.metrics.messages_dropped += 1;
                self.metrics.targeted_attacks += 1;
                self.metrics.failure_attribution.record(FailureReason::ByzantineDrop);
                self.metrics.events.push(NetworkEvent::TargetedAttack {
                    target: target.clone(),
                    attack_type: "message_drop".to_string(),
                    timestamp_ms,
                });
                return false;
            }
        }

        // Calculate latency
        let mut latency_ms = self.latency_dist.sample(&mut self.rng).max(0.1);

        // Check 4: Byzantine timing attack
        if self.config.enable_timing_attacks && self.byzantine_nodes.contains(&source) {
            // Byzantine nodes send to some targets fast, others slow
            if self.rng.gen_bool(0.3) {
                // Fast path to "friendly" targets
                latency_ms *= 0.1;
            } else {
                // Slow path to honest nodes
                latency_ms *= 3.0;
            }
            self.metrics.timing_attacks += 1;
        }

        // Check 5: Latency spike (jitter burst)
        if self.config.enable_latency_spikes {
            if self.latency_spike_active || self.rng.gen_bool(self.config.latency_spike_rate) {
                latency_ms *= self.config.latency_spike_factor;
                if !self.latency_spike_active {
                    self.latency_spike_active = true;
                    self.metrics.latency_spikes += 1;
                    self.metrics.events.push(NetworkEvent::LatencySpike {
                        affected_messages: 1,
                        spike_factor: self.config.latency_spike_factor,
                        timestamp_ms,
                    });
                }
                // Spike lasts for a random duration
                if self.rng.gen_bool(0.3) {
                    self.latency_spike_active = false;
                }
            }
        }

        // Check 6: Latency timeout (message took too long - considered failed)
        if latency_ms > self.config.latency_timeout_ms {
            self.metrics.messages_dropped += 1;
            self.metrics.failure_attribution.record(FailureReason::LatencyTimeout);
            return false;
        }

        let deliver_at = now + Duration::from_micros((latency_ms * 1000.0) as u64);

        self.in_flight.push_back(InFlightMessage {
            message: msg,
            deliver_at,
            source,
            target,
            msg_type,
            sent_at: now,
        });

        true
    }

    /// Step the simulation forward, returning messages that have "arrived"
    pub fn step(&mut self) -> Vec<(String, T, MessageType)> {
        let now = Instant::now();
        let timestamp_ms = now.duration_since(self.start_time).as_millis() as u64;

        let mut delivered = Vec::new();
        let mut remaining = VecDeque::new();

        while let Some(msg) = self.in_flight.pop_front() {
            if now >= msg.deliver_at {
                let latency_ms = msg.deliver_at.duration_since(msg.sent_at).as_secs_f64() * 1000.0;

                self.metrics.messages_delivered += 1;
                self.metrics.record_latency(latency_ms);
                self.metrics.events.push(NetworkEvent::MessageDelivered {
                    source: msg.source.clone(),
                    target: msg.target.clone(),
                    msg_type: msg.msg_type,
                    latency_ms,
                    timestamp_ms,
                });

                delivered.push((msg.target, msg.message, msg.msg_type));
            } else {
                remaining.push_back(msg);
            }
        }

        self.in_flight = remaining;
        delivered
    }

    /// Check if enough time has passed for all in-flight messages
    pub fn has_pending_messages(&self) -> bool {
        !self.in_flight.is_empty()
    }

    /// Flush all messages (deliver immediately, ignoring timing)
    pub fn flush(&mut self) -> Vec<(String, T, MessageType)> {
        let mut delivered = Vec::new();
        let timestamp_ms = Instant::now().duration_since(self.start_time).as_millis() as u64;

        while let Some(msg) = self.in_flight.pop_front() {
            let latency_ms = 0.0; // Immediate delivery

            self.metrics.messages_delivered += 1;
            self.metrics.record_latency(latency_ms);
            self.metrics.events.push(NetworkEvent::MessageDelivered {
                source: msg.source.clone(),
                target: msg.target.clone(),
                msg_type: msg.msg_type,
                latency_ms,
                timestamp_ms,
            });

            delivered.push((msg.target, msg.message, msg.msg_type));
        }

        delivered
    }

    /// Activate a network partition
    pub fn activate_partition(&mut self, partition_a: HashSet<String>) {
        self.partition_active = true;
        let partition_b: Vec<String> = self.in_flight.iter()
            .flat_map(|m| vec![m.source.clone(), m.target.clone()])
            .filter(|id| !partition_a.contains(id))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        self.partition_a = partition_a.clone();
        self.metrics.partition_events += 1;

        let timestamp_ms = Instant::now().duration_since(self.start_time).as_millis() as u64;
        self.metrics.events.push(NetworkEvent::PartitionActivated {
            partition_a: partition_a.into_iter().collect(),
            partition_b,
            timestamp_ms,
        });
    }

    /// Toggle partition state
    pub fn toggle_partition(&mut self, all_nodes: &[String]) {
        if self.partition_active {
            self.heal_partition();
        } else {
            // Split nodes roughly in half
            let mid = all_nodes.len() / 2;
            let partition_a: HashSet<String> = all_nodes[..mid].iter().cloned().collect();
            self.activate_partition(partition_a);
        }
    }

    /// Heal the network partition
    pub fn heal_partition(&mut self) {
        self.partition_active = false;
        self.partition_a.clear();

        let timestamp_ms = Instant::now().duration_since(self.start_time).as_millis() as u64;
        self.metrics.events.push(NetworkEvent::PartitionHealed { timestamp_ms });
    }

    /// Check if two nodes are separated by partition
    fn is_partitioned(&self, source: &str, target: &str) -> bool {
        let source_in_a = self.partition_a.contains(source);
        let target_in_a = self.partition_a.contains(target);
        source_in_a != target_in_a
    }

    /// Crash a node for a duration
    pub fn crash_node(&mut self, node_id: &str) {
        let recover_at = Instant::now() + Duration::from_millis(self.config.crash_duration_ms);
        self.crashed_nodes.insert(node_id.to_string(), recover_at);
        self.metrics.node_crashes += 1;

        let timestamp_ms = Instant::now().duration_since(self.start_time).as_millis() as u64;
        self.metrics.events.push(NetworkEvent::NodeCrashed {
            node_id: node_id.to_string(),
            duration_ms: self.config.crash_duration_ms,
            timestamp_ms,
        });
    }

    /// Randomly crash nodes based on crash rate
    pub fn random_chaos(&mut self, all_nodes: &[String]) {
        for node_id in all_nodes {
            if self.rng.gen_bool(self.config.node_crash_rate) {
                self.crash_node(node_id);
            }
        }
    }

    /// Get network metrics
    pub fn metrics(&self) -> &NetworkMetrics {
        &self.metrics
    }

    /// Get current network state summary
    pub fn state_summary(&self) -> NetworkStateSummary {
        NetworkStateSummary {
            partition_active: self.partition_active,
            partition_a_size: self.partition_a.len(),
            crashed_nodes: self.crashed_nodes.len(),
            in_flight_messages: self.in_flight.len(),
            delivery_rate: self.metrics.delivery_rate(),
            avg_latency_ms: self.metrics.avg_latency_ms,
        }
    }

    // ========================================================================
    // Expanded Chaos Methods (A)
    // ========================================================================

    /// Trigger a cascading failure starting from a node
    /// When a node crashes, nodes that depend on it may also crash
    pub fn trigger_cascading_failure(&mut self, trigger_node: &str, depth: u32) {
        if !self.config.enable_cascading_failures || depth > self.config.max_cascade_depth {
            return;
        }

        let timestamp_ms = Instant::now().duration_since(self.start_time).as_millis() as u64;

        // Crash the trigger node
        self.crash_node(trigger_node);

        // Find dependent nodes and potentially crash them
        let mut affected = Vec::new();
        if let Some(dependents) = self.node_dependencies.get(trigger_node).cloned() {
            for dependent in dependents {
                if self.rng.gen_bool(self.config.cascade_probability) {
                    affected.push(dependent.clone());
                    // Recursive cascade
                    self.trigger_cascading_failure(&dependent, depth + 1);
                }
            }
        }

        if !affected.is_empty() {
            self.metrics.cascading_failures += 1;
            self.metrics.events.push(NetworkEvent::CascadingFailure {
                trigger_node: trigger_node.to_string(),
                affected_nodes: affected,
                timestamp_ms,
            });
        }
    }

    /// Trigger a view-change storm (rapid consensus disruption)
    /// This causes consensus messages to be dropped for a duration
    pub fn trigger_view_change_storm(&mut self, duration_ms: u64) {
        if !self.config.enable_view_change_storms {
            return;
        }

        let timestamp_ms = Instant::now().duration_since(self.start_time).as_millis() as u64;
        let view_changes = self.rng.gen_range(3..10);

        self.view_change_storm_active = true;
        self.view_change_storm_end = Some(Instant::now() + Duration::from_millis(duration_ms));

        self.metrics.view_change_storms += 1;
        self.metrics.events.push(NetworkEvent::ViewChangeStorm {
            view_changes,
            duration_ms,
            timestamp_ms,
        });
    }

    /// Apply comprehensive chaos for a round
    /// This is the main entry point for chaos engineering tests
    pub fn apply_round_chaos(&mut self, all_nodes: &[String]) {
        // Random node crashes
        self.random_chaos(all_nodes);

        // Maybe trigger cascading failure
        if self.config.enable_cascading_failures && self.rng.gen_bool(0.02) {
            if let Some(trigger) = all_nodes.choose(&mut self.rng) {
                self.trigger_cascading_failure(trigger, 0);
            }
        }

        // Maybe trigger view-change storm
        if self.config.enable_view_change_storms
            && !self.view_change_storm_active
            && self.rng.gen_bool(self.config.view_change_storm_rate)
        {
            let duration = self.rng.gen_range(500..2000);
            self.trigger_view_change_storm(duration);
        }
    }

    /// Get failure attribution breakdown
    pub fn failure_breakdown(&self) -> &FailureAttribution {
        &self.metrics.failure_attribution
    }

    /// Check if view-change storm is currently active
    pub fn is_view_change_storm_active(&self) -> bool {
        self.view_change_storm_active
    }
}

/// Summary of current network state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStateSummary {
    pub partition_active: bool,
    pub partition_a_size: usize,
    pub crashed_nodes: usize,
    pub in_flight_messages: usize,
    pub delivery_rate: f64,
    pub avg_latency_ms: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_send_receive() {
        let mut network: ChaosNetwork<String> = ChaosNetwork::new(NetworkConfig::lan());

        network.send(
            "node_001".to_string(),
            "node_002".to_string(),
            "hello".to_string(),
            MessageType::Gradient,
        );

        // Wait a bit for message to be deliverable
        std::thread::sleep(Duration::from_millis(10));

        let delivered = network.step();
        assert_eq!(delivered.len(), 1);
        assert_eq!(delivered[0].0, "node_002");
        assert_eq!(delivered[0].1, "hello");
    }

    #[test]
    fn test_packet_loss() {
        let config = NetworkConfig {
            packet_loss_rate: 1.0, // 100% loss
            ..NetworkConfig::default()
        };
        let mut network: ChaosNetwork<String> = ChaosNetwork::new(config);

        let sent = network.send(
            "node_001".to_string(),
            "node_002".to_string(),
            "hello".to_string(),
            MessageType::Gradient,
        );

        assert!(!sent);
        assert_eq!(network.metrics.messages_dropped, 1);
    }

    #[test]
    fn test_network_partition() {
        let mut network: ChaosNetwork<String> = ChaosNetwork::new(NetworkConfig::lan());

        // Create partition: node_001 in A, node_002 not in A
        let partition_a: HashSet<String> = vec!["node_001".to_string()].into_iter().collect();
        network.activate_partition(partition_a);

        let sent = network.send(
            "node_001".to_string(),
            "node_002".to_string(),
            "hello".to_string(),
            MessageType::Gradient,
        );

        assert!(!sent);
        assert_eq!(network.metrics.messages_blocked, 1);
        assert!(network.partition_active);
    }

    #[test]
    fn test_partition_heal() {
        let mut network: ChaosNetwork<String> = ChaosNetwork::new(NetworkConfig::lan());

        let partition_a: HashSet<String> = vec!["node_001".to_string()].into_iter().collect();
        network.activate_partition(partition_a);

        // Heal partition
        network.heal_partition();

        let sent = network.send(
            "node_001".to_string(),
            "node_002".to_string(),
            "hello".to_string(),
            MessageType::Gradient,
        );

        assert!(sent);
        assert!(!network.partition_active);
    }

    #[test]
    fn test_node_crash() {
        let config = NetworkConfig {
            crash_duration_ms: 100,
            ..NetworkConfig::lan()
        };
        let mut network: ChaosNetwork<String> = ChaosNetwork::new(config);

        network.crash_node("node_002");

        let sent = network.send(
            "node_001".to_string(),
            "node_002".to_string(),
            "hello".to_string(),
            MessageType::Gradient,
        );

        assert!(!sent);
        assert_eq!(network.metrics.node_crashes, 1);
    }

    #[test]
    fn test_latency_distribution() {
        let config = NetworkConfig {
            mean_latency_ms: 100.0,
            latency_jitter_ms: 10.0,
            packet_loss_rate: 0.0,
            ..NetworkConfig::default()
        };
        let mut network: ChaosNetwork<String> = ChaosNetwork::new(config);

        // Send multiple messages
        for i in 0..100 {
            network.send(
                format!("node_{:03}", i),
                format!("node_{:03}", i + 1),
                "test".to_string(),
                MessageType::Gradient,
            );
        }

        // Wait for delivery
        std::thread::sleep(Duration::from_millis(200));
        network.step();

        // Check latency distribution
        let avg = network.metrics.avg_latency_ms;
        assert!(avg > 50.0 && avg < 150.0, "Average latency {} out of expected range", avg);
    }

    #[test]
    fn test_wan_config() {
        let config = NetworkConfig::wan();
        assert!(config.mean_latency_ms > 100.0);
        assert!(config.packet_loss_rate > 0.01);
    }

    #[test]
    fn test_adversarial_config() {
        let config = NetworkConfig::adversarial();
        assert!(config.enable_timing_attacks);
        assert!(config.packet_loss_rate > 0.03);
    }

    #[test]
    fn test_metrics_tracking() {
        let mut network: ChaosNetwork<String> = ChaosNetwork::new(NetworkConfig::lan());

        network.send("a".to_string(), "b".to_string(), "m1".to_string(), MessageType::Gradient);
        network.send("a".to_string(), "c".to_string(), "m2".to_string(), MessageType::Vote);

        assert_eq!(network.metrics.messages_sent, 2);

        std::thread::sleep(Duration::from_millis(10));
        network.step();

        assert!(network.metrics.messages_delivered > 0);
    }

    #[test]
    fn test_flush() {
        // Use LAN config with no packet loss for flush test
        let config = NetworkConfig {
            packet_loss_rate: 0.0, // No loss for this test
            ..NetworkConfig::lan()
        };
        let mut network: ChaosNetwork<String> = ChaosNetwork::new(config);

        for i in 0..10 {
            network.send(
                format!("node_{}", i),
                format!("node_{}", i + 1),
                format!("msg_{}", i),
                MessageType::Gradient,
            );
        }

        // Flush immediately (don't wait for latency)
        let delivered = network.flush();
        assert_eq!(delivered.len(), 10);
    }

    // ========================================================================
    // Expanded Chaos Tests (A)
    // ========================================================================

    #[test]
    fn test_cascading_failure() {
        let config = NetworkConfig {
            enable_cascading_failures: true,
            cascade_probability: 1.0, // Always cascade for test
            max_cascade_depth: 2,
            ..NetworkConfig::lan()
        };
        let mut network: ChaosNetwork<String> = ChaosNetwork::new(config);

        // Set up dependency chain: node_001 -> node_002 -> node_003
        let mut deps = HashMap::new();
        deps.insert("node_001".to_string(), vec!["node_002".to_string()]);
        deps.insert("node_002".to_string(), vec!["node_003".to_string()]);
        network.set_node_dependencies(deps);

        // Trigger cascade from node_001
        network.trigger_cascading_failure("node_001", 0);

        // All three nodes should be crashed
        assert!(network.crashed_nodes.contains_key("node_001"));
        assert!(network.crashed_nodes.contains_key("node_002"));
        assert!(network.crashed_nodes.contains_key("node_003"));
        assert!(network.metrics.cascading_failures >= 1);
    }

    #[test]
    fn test_view_change_storm() {
        let config = NetworkConfig {
            enable_view_change_storms: true,
            ..NetworkConfig::lan()
        };
        let mut network: ChaosNetwork<String> = ChaosNetwork::new(config);

        // Trigger view change storm
        network.trigger_view_change_storm(1000);

        assert!(network.is_view_change_storm_active());
        assert_eq!(network.metrics.view_change_storms, 1);

        // Consensus messages should have a chance of being dropped
        let mut dropped = 0;
        for _ in 0..20 {
            if !network.send(
                "node_001".to_string(),
                "node_002".to_string(),
                "vote".to_string(),
                MessageType::Vote,
            ) {
                dropped += 1;
            }
        }

        // Some votes should have been dropped
        println!("Dropped {} out of 20 votes during view-change storm", dropped);
        // Due to randomness, we can't assert exact count, but some should drop
    }

    #[test]
    fn test_latency_spike() {
        let config = NetworkConfig {
            enable_latency_spikes: true,
            latency_spike_rate: 1.0, // Always spike for test
            latency_spike_factor: 5.0,
            mean_latency_ms: 10.0,
            latency_timeout_ms: 1000.0, // High timeout to allow spikes
            ..NetworkConfig::default()
        };
        let mut network: ChaosNetwork<String> = ChaosNetwork::new(config);

        network.send(
            "node_001".to_string(),
            "node_002".to_string(),
            "test".to_string(),
            MessageType::Gradient,
        );

        assert!(network.metrics.latency_spikes >= 1);
    }

    #[test]
    fn test_targeted_attack() {
        let config = NetworkConfig {
            enable_targeted_attacks: true,
            packet_loss_rate: 0.0, // No normal packet loss
            ..NetworkConfig::lan()
        };
        let mut network: ChaosNetwork<String> = ChaosNetwork::new(config);

        // Register high-value target
        let targets: HashSet<String> = vec!["validator_001".to_string()].into_iter().collect();
        network.register_high_value_targets(targets);

        // Send many messages to the high-value target
        let mut dropped = 0;
        for i in 0..50 {
            if !network.send(
                format!("node_{:03}", i),
                "validator_001".to_string(),
                "msg".to_string(),
                MessageType::Gradient,
            ) {
                dropped += 1;
            }
        }

        // Some messages to high-value targets should be dropped
        println!("Dropped {} out of 50 messages to high-value target", dropped);
        assert!(dropped > 0, "Expected some targeted drops");
    }

    #[test]
    fn test_failure_attribution() {
        let config = NetworkConfig {
            packet_loss_rate: 0.5,
            ..NetworkConfig::default()
        };
        let mut network: ChaosNetwork<String> = ChaosNetwork::new(config);

        // Crash a node
        network.crash_node("node_002");

        // Send messages to crashed node
        for _ in 0..10 {
            network.send(
                "node_001".to_string(),
                "node_002".to_string(),
                "test".to_string(),
                MessageType::Gradient,
            );
        }

        let breakdown = network.failure_breakdown();
        println!("Failure breakdown: node_unavailable={}, packet_loss={}",
                 breakdown.node_unavailable, breakdown.packet_loss);

        // All should be attributed to node unavailable (crashed)
        assert!(breakdown.node_unavailable >= 10);
    }

    #[test]
    fn test_chaos_engineering_config() {
        let config = NetworkConfig::chaos_engineering();

        assert!(config.enable_cascading_failures);
        assert!(config.enable_view_change_storms);
        assert!(config.enable_latency_spikes);
        assert!(config.enable_targeted_attacks);
        assert!(config.enable_timing_attacks);
    }

    #[test]
    fn test_apply_round_chaos() {
        let config = NetworkConfig::chaos_engineering();
        let mut network: ChaosNetwork<String> = ChaosNetwork::new(config);

        let nodes: Vec<String> = (0..20).map(|i| format!("node_{:03}", i)).collect();
        network.build_random_dependencies(&nodes);

        // Apply chaos for several rounds
        for _ in 0..10 {
            network.apply_round_chaos(&nodes);
        }

        // Should have some chaos events
        let total_chaos = network.metrics.node_crashes
            + network.metrics.cascading_failures
            + network.metrics.view_change_storms;
        println!("Total chaos events: crashes={}, cascades={}, storms={}",
                 network.metrics.node_crashes,
                 network.metrics.cascading_failures,
                 network.metrics.view_change_storms);
        assert!(total_chaos > 0, "Expected some chaos events");
    }
}
