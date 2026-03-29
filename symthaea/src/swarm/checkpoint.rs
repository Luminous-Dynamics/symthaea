// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Federated Learning Checkpoint and Resume System
//!
//! This module provides fault-tolerant checkpointing for distributed federated learning,
//! enabling recovery from coordinator crashes, node disconnections, and network partitions.
//!
//! # Features
//!
//! - **Full Checkpoints**: Complete state snapshots including model weights, node states, and buffers
//! - **Incremental Checkpoints**: Delta-based saves for efficient storage
//! - **Coordinator Recovery**: Resume training after coordinator crashes
//! - **Node Rejoin Protocol**: Allow nodes to rejoin and catch up after disconnection
//! - **Gradient Replay**: Replay pending gradients from checkpoint on recovery
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                    CHECKPOINT SYSTEM ARCHITECTURE                            │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                              │
//! │  ┌────────────────────┐         ┌─────────────────────────┐                 │
//! │  │  CheckpointManager │         │   FederatedCheckpoint   │                 │
//! │  │                    │         │                         │                 │
//! │  │ • Periodic saves   │  ────▶  │ • Global model weights  │                 │
//! │  │ • Auto-cleanup     │         │ • Round number          │                 │
//! │  │ • Recovery detect  │         │ • Node states           │                 │
//! │  │ • Rollback support │         │ • Aggregation buffer    │                 │
//! │  └────────────────────┘         └─────────────────────────┘                 │
//! │                                                                              │
//! │  ┌─────────────────────────────────────────────────────────────────────────┐│
//! │  │                        Node-Level Checkpoints                           ││
//! │  │                                                                          ││
//! │  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 ││
//! │  │  │ Local Model │    │  Pending    │    │   Comm      │                 ││
//! │  │  │   State     │    │ Gradients   │    │   Queue     │                 ││
//! │  │  └─────────────┘    └─────────────┘    └─────────────┘                 ││
//! │  └─────────────────────────────────────────────────────────────────────────┘│
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use symthaea::swarm::checkpoint::{CheckpointManager, CheckpointConfig};
//!
//! // Configure checkpointing
//! let config = CheckpointConfig::new()
//!     .with_checkpoint_interval(10)  // Every 10 rounds
//!     .with_checkpoint_dir("/tmp/fl-checkpoints");
//!
//! let mut manager = CheckpointManager::new(config);
//!
//! // In training loop
//! manager.maybe_checkpoint(&coordinator).await?;
//!
//! // On recovery
//! let coordinator = manager.resume_from_latest().await?;
//! ```

use crate::swarm::federated_cfc::GradientMessage;
use crate::swarm::federated_network::{
    FederatedCoordinator, FederatedNetworkConfig, FederatedNode,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read as IoRead, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use tracing::{debug, info, warn};

// ============================================================================
// CHECKPOINT DATA STRUCTURES
// ============================================================================

/// Node state for checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeState {
    /// Node ID (32 bytes)
    pub node_id: [u8; 32],

    /// Last seen timestamp (ms since epoch)
    pub last_seen: u64,

    /// Number of contributions from this node
    pub contribution_count: u64,

    /// Node's trust score
    pub trust_score: f32,

    /// Node's current round
    pub current_round: u64,

    /// Whether node is active
    pub is_active: bool,

    /// Node address serialized
    pub address: String,
}

impl NodeState {
    /// Create from a FederatedNode
    pub fn from_node(node: &FederatedNode, contribution_count: u64) -> Self {
        Self {
            node_id: node.node_id,
            last_seen: node.last_heartbeat,
            contribution_count,
            trust_score: node.trust_score,
            current_round: node.current_round,
            is_active: node.is_active,
            address: node.address.to_string(),
        }
    }
}

/// Buffered gradient for replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferedGradient {
    /// The gradient message
    pub gradient: GradientMessage,

    /// Trust weight assigned during receive
    pub trust_weight: f32,

    /// Round when received
    pub round_received: u64,
}

/// Complete federated learning checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedCheckpoint {
    /// Checkpoint format version
    pub version: u32,

    /// Global model weights
    pub global_weights: Vec<f32>,

    /// Current round number
    pub round_number: u64,

    /// State of all known nodes
    pub node_states: HashMap<[u8; 32], NodeState>,

    /// Pending gradients in aggregation buffer
    pub aggregation_buffer: Vec<BufferedGradient>,

    /// Checkpoint creation timestamp (ms since epoch)
    pub timestamp: u64,

    /// Optional coordinator ID
    pub coordinator_id: Option<[u8; 32]>,

    /// Configuration snapshot
    pub config_snapshot: CheckpointedConfig,

    /// Checksum for integrity verification
    pub checksum: [u8; 32],
}

/// Serializable subset of FederatedNetworkConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointedConfig {
    pub num_nodes: usize,
    pub sync_interval_ms: u64,
    pub timeout_ms: u64,
    pub min_nodes_for_aggregation: usize,
    pub enable_dp: bool,
    pub enable_byzantine_tolerance: bool,
    pub byzantine_trim_fraction: f32,
    pub max_gradient_staleness_ms: u64,
}

impl From<&FederatedNetworkConfig> for CheckpointedConfig {
    fn from(config: &FederatedNetworkConfig) -> Self {
        Self {
            num_nodes: config.num_nodes,
            sync_interval_ms: config.sync_interval_ms,
            timeout_ms: config.timeout_ms,
            min_nodes_for_aggregation: config.min_nodes_for_aggregation,
            enable_dp: config.enable_dp,
            enable_byzantine_tolerance: config.enable_byzantine_tolerance,
            byzantine_trim_fraction: config.byzantine_trim_fraction,
            max_gradient_staleness_ms: config.max_gradient_staleness_ms,
        }
    }
}

impl FederatedCheckpoint {
    /// Current checkpoint format version
    pub const VERSION: u32 = 1;

    /// Create a new checkpoint
    pub fn new(
        weights: Vec<f32>,
        round: u64,
        node_states: HashMap<[u8; 32], NodeState>,
        aggregation_buffer: Vec<BufferedGradient>,
        config: &FederatedNetworkConfig,
        coordinator_id: Option<[u8; 32]>,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let mut checkpoint = Self {
            version: Self::VERSION,
            global_weights: weights,
            round_number: round,
            node_states,
            aggregation_buffer,
            timestamp,
            coordinator_id,
            config_snapshot: CheckpointedConfig::from(config),
            checksum: [0u8; 32],
        };

        // Compute checksum
        checkpoint.checksum = checkpoint.compute_checksum();
        checkpoint
    }

    /// Compute BLAKE3 checksum of checkpoint data
    fn compute_checksum(&self) -> [u8; 32] {
        use blake3::Hasher;

        let mut hasher = Hasher::new();

        // Hash version
        hasher.update(&self.version.to_le_bytes());

        // Hash weights
        for w in &self.global_weights {
            hasher.update(&w.to_le_bytes());
        }

        // Hash round number
        hasher.update(&self.round_number.to_le_bytes());

        // Hash timestamp
        hasher.update(&self.timestamp.to_le_bytes());

        // Hash node count
        hasher.update(&(self.node_states.len() as u64).to_le_bytes());

        // Hash buffer size
        hasher.update(&(self.aggregation_buffer.len() as u64).to_le_bytes());

        *hasher.finalize().as_bytes()
    }

    /// Verify checkpoint integrity
    pub fn verify_checksum(&self) -> bool {
        let expected = self.compute_checksum();
        // Constant-time comparison
        let mut diff = 0u8;
        for (a, b) in self.checksum.iter().zip(expected.iter()) {
            diff |= a ^ b;
        }
        diff == 0
    }

    /// Save checkpoint to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let serialized =
            bincode::serialize(self).map_err(|e| std::io::Error::other(e.to_string()))?;

        let mut file = std::fs::File::create(path)?;
        file.write_all(&serialized)?;
        file.sync_all()?;

        Ok(())
    }

    /// Load checkpoint from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let checkpoint: Self =
            bincode::deserialize(&buffer).map_err(|e| std::io::Error::other(e.to_string()))?;

        // Verify integrity
        if !checkpoint.verify_checksum() {
            return Err(std::io::Error::other(
                "Checkpoint checksum verification failed",
            ));
        }

        Ok(checkpoint)
    }

    /// Get checkpoint size in bytes (estimated)
    pub fn size_bytes(&self) -> usize {
        // Approximate size: weights + node states + buffer + overhead
        let weights_size = self.global_weights.len() * 4;
        let nodes_size = self.node_states.len() * 100; // ~100 bytes per node state
        let buffer_size = self
            .aggregation_buffer
            .iter()
            .map(|b| b.gradient.dim() * 4 + 100)
            .sum::<usize>();
        weights_size + nodes_size + buffer_size + 256 // overhead
    }
}

// ============================================================================
// INCREMENTAL CHECKPOINT
// ============================================================================

/// Incremental checkpoint containing only changes since last full checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalCheckpoint {
    /// Base checkpoint round this is relative to
    pub base_round: u64,

    /// Current round number
    pub current_round: u64,

    /// Weight deltas (sparse: only changed indices)
    pub weight_deltas: Vec<(usize, f32)>,

    /// New gradients since base
    pub new_gradients: Vec<BufferedGradient>,

    /// Node state changes
    pub node_updates: Vec<NodeState>,

    /// Removed node IDs
    pub removed_nodes: Vec<[u8; 32]>,

    /// Timestamp
    pub timestamp: u64,

    /// Checksum
    pub checksum: [u8; 32],
}

impl IncrementalCheckpoint {
    /// Create incremental checkpoint from current state and base checkpoint
    pub fn from_diff(
        current_weights: &[f32],
        base_weights: &[f32],
        base_round: u64,
        current_round: u64,
        new_gradients: Vec<BufferedGradient>,
        node_updates: Vec<NodeState>,
        removed_nodes: Vec<[u8; 32]>,
    ) -> Self {
        // Compute weight deltas (only store significant changes)
        let weight_deltas: Vec<(usize, f32)> = current_weights
            .iter()
            .zip(base_weights.iter())
            .enumerate()
            .filter(|(_, (curr, base))| (*curr - *base).abs() > 1e-7)
            .map(|(i, (curr, _))| (i, *curr))
            .collect();

        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let mut checkpoint = Self {
            base_round,
            current_round,
            weight_deltas,
            new_gradients,
            node_updates,
            removed_nodes,
            timestamp,
            checksum: [0u8; 32],
        };

        checkpoint.checksum = checkpoint.compute_checksum();
        checkpoint
    }

    fn compute_checksum(&self) -> [u8; 32] {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(&self.base_round.to_le_bytes());
        hasher.update(&self.current_round.to_le_bytes());
        hasher.update(&(self.weight_deltas.len() as u64).to_le_bytes());
        hasher.update(&self.timestamp.to_le_bytes());

        *hasher.finalize().as_bytes()
    }

    /// Save to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let serialized =
            bincode::serialize(self).map_err(|e| std::io::Error::other(e.to_string()))?;

        let mut file = std::fs::File::create(path)?;
        file.write_all(&serialized)?;
        file.sync_all()?;

        Ok(())
    }

    /// Load from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        bincode::deserialize(&buffer).map_err(|e| std::io::Error::other(e.to_string()))
    }

    /// Apply incremental checkpoint to base weights
    pub fn apply_to_weights(&self, base_weights: &mut [f32]) {
        for (idx, value) in &self.weight_deltas {
            if *idx < base_weights.len() {
                base_weights[*idx] = *value;
            }
        }
    }
}

// ============================================================================
// NODE-LEVEL CHECKPOINT
// ============================================================================

/// Checkpoint for individual node state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCheckpoint {
    /// Node ID
    pub node_id: [u8; 32],

    /// Local model weights
    pub local_weights: Vec<f32>,

    /// Pending gradients not yet sent
    pub pending_gradients: Vec<GradientMessage>,

    /// Communication queue state (message hashes awaiting ack)
    pub pending_message_hashes: Vec<[u8; 32]>,

    /// Last completed round
    pub last_round: u64,

    /// Timestamp
    pub timestamp: u64,

    /// Checksum
    pub checksum: [u8; 32],
}

impl NodeCheckpoint {
    /// Create a new node checkpoint
    pub fn new(
        node_id: [u8; 32],
        local_weights: Vec<f32>,
        pending_gradients: Vec<GradientMessage>,
        pending_message_hashes: Vec<[u8; 32]>,
        last_round: u64,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let mut checkpoint = Self {
            node_id,
            local_weights,
            pending_gradients,
            pending_message_hashes,
            last_round,
            timestamp,
            checksum: [0u8; 32],
        };

        checkpoint.checksum = checkpoint.compute_checksum();
        checkpoint
    }

    fn compute_checksum(&self) -> [u8; 32] {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(&self.node_id);
        hasher.update(&self.last_round.to_le_bytes());
        hasher.update(&self.timestamp.to_le_bytes());
        hasher.update(&(self.local_weights.len() as u64).to_le_bytes());

        *hasher.finalize().as_bytes()
    }

    /// Save to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let serialized =
            bincode::serialize(self).map_err(|e| std::io::Error::other(e.to_string()))?;

        let mut file = std::fs::File::create(path)?;
        file.write_all(&serialized)?;
        file.sync_all()?;

        Ok(())
    }

    /// Load from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let checkpoint: Self =
            bincode::deserialize(&buffer).map_err(|e| std::io::Error::other(e.to_string()))?;

        // Verify checksum
        let expected = checkpoint.compute_checksum();
        let mut diff = 0u8;
        for (a, b) in checkpoint.checksum.iter().zip(expected.iter()) {
            diff |= a ^ b;
        }
        if diff != 0 {
            return Err(std::io::Error::other("Node checkpoint checksum failed"));
        }

        Ok(checkpoint)
    }
}

// ============================================================================
// CHECKPOINT CONFIGURATION
// ============================================================================

/// Configuration for checkpoint management
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Directory to store checkpoints
    pub checkpoint_dir: PathBuf,

    /// Checkpoint every N rounds
    pub checkpoint_interval: usize,

    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,

    /// Enable incremental checkpoints between full checkpoints
    pub enable_incremental: bool,

    /// Full checkpoint every N incremental checkpoints
    pub full_checkpoint_interval: usize,

    /// Enable automatic recovery on startup
    pub auto_recover: bool,

    /// Checkpoint compression (future feature)
    pub compress: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("/tmp/symthaea-fl-checkpoints"),
            checkpoint_interval: 10,
            max_checkpoints: 5,
            enable_incremental: true,
            full_checkpoint_interval: 5,
            auto_recover: true,
            compress: false,
        }
    }
}

impl CheckpointConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set checkpoint directory
    pub fn with_checkpoint_dir<P: Into<PathBuf>>(mut self, dir: P) -> Self {
        self.checkpoint_dir = dir.into();
        self
    }

    /// Set checkpoint interval
    pub fn with_checkpoint_interval(mut self, interval: usize) -> Self {
        self.checkpoint_interval = interval;
        self
    }

    /// Set maximum checkpoints to retain
    pub fn with_max_checkpoints(mut self, max: usize) -> Self {
        self.max_checkpoints = max;
        self
    }

    /// Enable/disable incremental checkpoints
    pub fn with_incremental(mut self, enable: bool) -> Self {
        self.enable_incremental = enable;
        self
    }

    /// Create config for testing (fast checkpoints, temp directory)
    pub fn for_testing() -> Self {
        Self {
            checkpoint_dir: std::env::temp_dir()
                .join(format!("symthaea-test-checkpoint-{}", std::process::id())),
            checkpoint_interval: 2,
            max_checkpoints: 3,
            enable_incremental: false,
            full_checkpoint_interval: 1,
            auto_recover: true,
            compress: false,
        }
    }
}

// ============================================================================
// CHECKPOINT MANAGER
// ============================================================================

/// Manages checkpointing lifecycle for federated learning
pub struct CheckpointManager {
    /// Configuration
    config: CheckpointConfig,

    /// Last full checkpoint (for incremental diffs)
    last_full_checkpoint: Option<FederatedCheckpoint>,

    /// Number of incremental checkpoints since last full
    incremental_count: usize,

    /// Last checkpoint round
    last_checkpoint_round: u64,

    /// Contribution counts per node (for checkpoint)
    contribution_counts: HashMap<[u8; 32], u64>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(config: CheckpointConfig) -> Self {
        // Ensure checkpoint directory exists
        if let Err(e) = std::fs::create_dir_all(&config.checkpoint_dir) {
            warn!("Failed to create checkpoint directory: {}", e);
        }

        Self {
            config,
            last_full_checkpoint: None,
            incremental_count: 0,
            last_checkpoint_round: 0,
            contribution_counts: HashMap::new(),
        }
    }

    /// Get the checkpoint directory
    pub fn checkpoint_dir(&self) -> &Path {
        &self.config.checkpoint_dir
    }

    /// Check if checkpointing is due
    pub fn should_checkpoint(&self, current_round: u64) -> bool {
        current_round > 0
            && current_round != self.last_checkpoint_round
            && (current_round as usize).is_multiple_of(self.config.checkpoint_interval)
    }

    /// Record a contribution from a node
    pub fn record_contribution(&mut self, node_id: [u8; 32]) {
        *self.contribution_counts.entry(node_id).or_insert(0) += 1;
    }

    /// Create checkpoint from coordinator state
    pub fn create_checkpoint(
        &mut self,
        coordinator: &FederatedCoordinator,
        config: &FederatedNetworkConfig,
        aggregation_buffer: &[(GradientMessage, f32)],
    ) -> FederatedCheckpoint {
        let weights = coordinator.get_weights();
        let round = coordinator.current_round();

        // Build node states - we need to access peers through coordinator
        // Since FederatedCoordinator doesn't expose peers directly in a way we can iterate,
        // we'll use the stats and work with what we have
        let node_states: HashMap<[u8; 32], NodeState> = self
            .contribution_counts
            .iter()
            .map(|(id, count)| {
                let state = NodeState {
                    node_id: *id,
                    last_seen: SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0),
                    contribution_count: *count,
                    trust_score: 0.5, // Default since we can't access peer data
                    current_round: round,
                    is_active: true,
                    address: "unknown".to_string(),
                };
                (*id, state)
            })
            .collect();

        // Convert aggregation buffer
        let buffer: Vec<BufferedGradient> = aggregation_buffer
            .iter()
            .map(|(g, w)| BufferedGradient {
                gradient: g.clone(),
                trust_weight: *w,
                round_received: round,
            })
            .collect();

        FederatedCheckpoint::new(
            weights,
            round,
            node_states,
            buffer,
            config,
            Some(coordinator.local_node_id()),
        )
    }

    /// Save a full checkpoint
    pub fn save_checkpoint(&mut self, checkpoint: &FederatedCheckpoint) -> std::io::Result<()> {
        let filename = format!("checkpoint_round_{:08}.bin", checkpoint.round_number);
        let path = self.config.checkpoint_dir.join(&filename);

        info!("Saving checkpoint to {:?}", path);
        checkpoint.save_to_file(&path)?;

        self.last_full_checkpoint = Some(checkpoint.clone());
        self.last_checkpoint_round = checkpoint.round_number;
        self.incremental_count = 0;

        // Cleanup old checkpoints
        self.cleanup_old_checkpoints()?;

        Ok(())
    }

    /// Save an incremental checkpoint
    pub fn save_incremental(
        &mut self,
        current_weights: &[f32],
        current_round: u64,
        new_gradients: Vec<BufferedGradient>,
        node_updates: Vec<NodeState>,
        removed_nodes: Vec<[u8; 32]>,
    ) -> std::io::Result<()> {
        let base = match &self.last_full_checkpoint {
            Some(cp) => cp,
            None => {
                warn!("No base checkpoint for incremental save");
                return Err(std::io::Error::other("No base checkpoint"));
            }
        };

        let incremental = IncrementalCheckpoint::from_diff(
            current_weights,
            &base.global_weights,
            base.round_number,
            current_round,
            new_gradients,
            node_updates,
            removed_nodes,
        );

        let filename = format!(
            "incremental_base{:08}_round{:08}.bin",
            base.round_number, current_round
        );
        let path = self.config.checkpoint_dir.join(&filename);

        debug!("Saving incremental checkpoint to {:?}", path);
        incremental.save_to_file(&path)?;

        self.incremental_count += 1;
        self.last_checkpoint_round = current_round;

        Ok(())
    }

    /// Maybe checkpoint based on current round
    pub fn maybe_checkpoint(
        &mut self,
        coordinator: &FederatedCoordinator,
        config: &FederatedNetworkConfig,
        aggregation_buffer: &[(GradientMessage, f32)],
    ) -> std::io::Result<bool> {
        let current_round = coordinator.current_round();

        if !self.should_checkpoint(current_round) {
            return Ok(false);
        }

        // Decide full vs incremental
        let do_full = !self.config.enable_incremental
            || self.last_full_checkpoint.is_none()
            || self.incremental_count >= self.config.full_checkpoint_interval;

        if do_full {
            let checkpoint = self.create_checkpoint(coordinator, config, aggregation_buffer);
            self.save_checkpoint(&checkpoint)?;
            info!("Full checkpoint saved at round {}", current_round);
        } else {
            let weights = coordinator.get_weights();
            let buffer: Vec<BufferedGradient> = aggregation_buffer
                .iter()
                .map(|(g, w)| BufferedGradient {
                    gradient: g.clone(),
                    trust_weight: *w,
                    round_received: current_round,
                })
                .collect();

            self.save_incremental(&weights, current_round, buffer, vec![], vec![])?;
            info!("Incremental checkpoint saved at round {}", current_round);
        }

        Ok(true)
    }

    /// Find the latest checkpoint in the directory
    pub fn find_latest_checkpoint(&self) -> std::io::Result<Option<PathBuf>> {
        let entries: Vec<_> = std::fs::read_dir(&self.config.checkpoint_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .file_name()
                    .map(|n| n.to_string_lossy().starts_with("checkpoint_round_"))
                    .unwrap_or(false)
            })
            .collect();

        if entries.is_empty() {
            return Ok(None);
        }

        // Sort by modification time (most recent first)
        let mut paths: Vec<_> = entries.into_iter().map(|e| e.path()).collect();
        paths.sort_by(|a, b| {
            let a_time = a.metadata().and_then(|m| m.modified()).ok();
            let b_time = b.metadata().and_then(|m| m.modified()).ok();
            b_time.cmp(&a_time)
        });

        Ok(paths.into_iter().next())
    }

    /// Load the latest checkpoint
    pub fn load_latest_checkpoint(&self) -> std::io::Result<Option<FederatedCheckpoint>> {
        match self.find_latest_checkpoint()? {
            Some(path) => {
                info!("Loading checkpoint from {:?}", path);
                let checkpoint = FederatedCheckpoint::load_from_file(&path)?;
                Ok(Some(checkpoint))
            }
            None => Ok(None),
        }
    }

    /// Clean up old checkpoints, keeping only the most recent N
    fn cleanup_old_checkpoints(&self) -> std::io::Result<()> {
        let mut entries: Vec<_> = std::fs::read_dir(&self.config.checkpoint_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .file_name()
                    .map(|n| {
                        let name = n.to_string_lossy();
                        name.starts_with("checkpoint_round_") || name.starts_with("incremental_")
                    })
                    .unwrap_or(false)
            })
            .collect();

        if entries.len() <= self.config.max_checkpoints {
            return Ok(());
        }

        // Sort by modification time (oldest first)
        entries.sort_by(|a, b| {
            let a_time = a.metadata().and_then(|m| m.modified()).ok();
            let b_time = b.metadata().and_then(|m| m.modified()).ok();
            a_time.cmp(&b_time)
        });

        // Remove oldest checkpoints
        let to_remove = entries.len() - self.config.max_checkpoints;
        for entry in entries.into_iter().take(to_remove) {
            debug!("Removing old checkpoint: {:?}", entry.path());
            if let Err(e) = std::fs::remove_file(entry.path()) {
                warn!("Failed to remove old checkpoint: {}", e);
            }
        }

        Ok(())
    }
}

// ============================================================================
// RECOVERY PROTOCOL
// ============================================================================

/// Recovery protocol result
pub struct RecoveryResult {
    /// Recovered checkpoint
    pub checkpoint: FederatedCheckpoint,

    /// Gradients to replay
    pub pending_gradients: Vec<GradientMessage>,

    /// Nodes that need resync notification
    pub stale_nodes: Vec<[u8; 32]>,
}

/// Coordinator recovery from checkpoint
pub async fn recover_coordinator(
    checkpoint: FederatedCheckpoint,
    config: FederatedNetworkConfig,
) -> Result<(FederatedCoordinator, RecoveryResult), std::io::Error> {
    info!(
        "Recovering coordinator from round {} checkpoint",
        checkpoint.round_number
    );

    // Verify checkpoint integrity
    if !checkpoint.verify_checksum() {
        return Err(std::io::Error::other(
            "Checkpoint integrity verification failed",
        ));
    }

    // Create new coordinator with recovered weights
    let coordinator =
        FederatedCoordinator::new(config.clone(), checkpoint.global_weights.clone()).await;

    // Extract pending gradients for replay
    let pending_gradients: Vec<GradientMessage> = checkpoint
        .aggregation_buffer
        .iter()
        .map(|b| b.gradient.clone())
        .collect();

    // Identify stale nodes (haven't been seen recently)
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let stale_threshold = config.max_gradient_staleness_ms * 2;
    let stale_nodes: Vec<[u8; 32]> = checkpoint
        .node_states
        .iter()
        .filter(|(_, state)| now.saturating_sub(state.last_seen) > stale_threshold)
        .map(|(id, _)| *id)
        .collect();

    info!(
        "Recovery complete: {} pending gradients, {} stale nodes",
        pending_gradients.len(),
        stale_nodes.len()
    );

    let result = RecoveryResult {
        checkpoint,
        pending_gradients,
        stale_nodes,
    };

    Ok((coordinator, result))
}

/// Node rejoin protocol after disconnection
pub struct NodeRejoinRequest {
    /// Node ID
    pub node_id: [u8; 32],

    /// Last known round before disconnect
    pub last_round: u64,

    /// Local checkpoint if available
    pub local_checkpoint: Option<NodeCheckpoint>,
}

/// Response to node rejoin request
pub struct NodeRejoinResponse {
    /// Current global weights
    pub current_weights: Vec<f32>,

    /// Current round
    pub current_round: u64,

    /// Missed rounds (for logging/metrics)
    pub missed_rounds: u64,

    /// Whether local gradients should be replayed
    pub replay_gradients: bool,
}

/// Process node rejoin request
pub fn process_node_rejoin(
    request: &NodeRejoinRequest,
    current_weights: &[f32],
    current_round: u64,
    max_replay_age: u64,
) -> NodeRejoinResponse {
    let missed_rounds = current_round.saturating_sub(request.last_round);

    // Determine if gradients should be replayed
    let replay_gradients = if let Some(ref checkpoint) = request.local_checkpoint {
        // Replay if checkpoint is recent enough
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        now.saturating_sub(checkpoint.timestamp) < max_replay_age
            && !checkpoint.pending_gradients.is_empty()
    } else {
        false
    };

    info!(
        "Node {} rejoining: missed {} rounds, replay={}",
        hex::encode(&request.node_id[..8]),
        missed_rounds,
        replay_gradients
    );

    NodeRejoinResponse {
        current_weights: current_weights.to_vec(),
        current_round,
        missed_rounds,
        replay_gradients,
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // =========================================================================
    // FederatedCheckpoint Tests
    // =========================================================================

    #[test]
    fn test_checkpoint_creation() {
        let config = FederatedNetworkConfig::for_testing();
        let checkpoint = FederatedCheckpoint::new(
            vec![1.0, 2.0, 3.0],
            10,
            HashMap::new(),
            vec![],
            &config,
            Some([1u8; 32]),
        );

        assert_eq!(checkpoint.version, FederatedCheckpoint::VERSION);
        assert_eq!(checkpoint.round_number, 10);
        assert_eq!(checkpoint.global_weights, vec![1.0, 2.0, 3.0]);
        assert!(checkpoint.verify_checksum());
    }

    #[test]
    fn test_checkpoint_save_load_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test_checkpoint.bin");

        let config = FederatedNetworkConfig::for_testing();
        let original = FederatedCheckpoint::new(
            vec![0.5, -0.3, 0.8, -0.2],
            42,
            HashMap::new(),
            vec![],
            &config,
            None,
        );

        // Save
        original.save_to_file(&path).unwrap();

        // Load
        let loaded = FederatedCheckpoint::load_from_file(&path).unwrap();

        assert_eq!(loaded.round_number, original.round_number);
        assert_eq!(loaded.global_weights, original.global_weights);
        assert!(loaded.verify_checksum());
    }

    #[test]
    fn test_checkpoint_checksum_tamper_detection() {
        let temp_dir = TempDir::new().unwrap();
        let _path = temp_dir.path().join("tampered_checkpoint.bin");

        let config = FederatedNetworkConfig::for_testing();
        let mut checkpoint = FederatedCheckpoint::new(
            vec![1.0, 2.0, 3.0],
            5,
            HashMap::new(),
            vec![],
            &config,
            None,
        );

        // Tamper with data after checksum
        checkpoint.round_number = 999;

        // Checksum should fail
        assert!(!checkpoint.verify_checksum());
    }

    // =========================================================================
    // IncrementalCheckpoint Tests
    // =========================================================================

    #[test]
    fn test_incremental_checkpoint() {
        let base_weights = vec![1.0, 2.0, 3.0, 4.0];
        let current_weights = vec![1.0, 2.5, 3.0, 4.5]; // Only indices 1 and 3 changed

        let incremental = IncrementalCheckpoint::from_diff(
            &current_weights,
            &base_weights,
            10,
            15,
            vec![],
            vec![],
            vec![],
        );

        assert_eq!(incremental.base_round, 10);
        assert_eq!(incremental.current_round, 15);
        assert_eq!(incremental.weight_deltas.len(), 2); // Only 2 changed
    }

    #[test]
    fn test_incremental_apply() {
        let base_weights = vec![1.0, 2.0, 3.0, 4.0];
        let current_weights = vec![1.0, 2.5, 3.0, 4.5];

        let incremental = IncrementalCheckpoint::from_diff(
            &current_weights,
            &base_weights,
            10,
            15,
            vec![],
            vec![],
            vec![],
        );

        // Apply to base
        let mut recovered = base_weights.clone();
        incremental.apply_to_weights(&mut recovered);

        // Should match current
        for (r, c) in recovered.iter().zip(current_weights.iter()) {
            assert!((r - c).abs() < 1e-6);
        }
    }

    // =========================================================================
    // NodeCheckpoint Tests
    // =========================================================================

    #[test]
    fn test_node_checkpoint_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("node_checkpoint.bin");

        let gradient = GradientMessage::new([1u8; 32], vec![0.1, 0.2], 0.8);

        let original = NodeCheckpoint::new(
            [42u8; 32],
            vec![1.0, 2.0, 3.0],
            vec![gradient],
            vec![[99u8; 32]],
            100,
        );

        original.save_to_file(&path).unwrap();
        let loaded = NodeCheckpoint::load_from_file(&path).unwrap();

        assert_eq!(loaded.node_id, original.node_id);
        assert_eq!(loaded.last_round, original.last_round);
        assert_eq!(loaded.local_weights, original.local_weights);
        assert_eq!(loaded.pending_gradients.len(), 1);
    }

    // =========================================================================
    // CheckpointManager Tests
    // =========================================================================

    #[test]
    fn test_checkpoint_manager_should_checkpoint() {
        let config = CheckpointConfig::for_testing();
        let manager = CheckpointManager::new(config);

        // Interval is 2, so should checkpoint on rounds 2, 4, 6, etc.
        assert!(!manager.should_checkpoint(0));
        assert!(!manager.should_checkpoint(1));
        assert!(manager.should_checkpoint(2));
        assert!(!manager.should_checkpoint(3));
        assert!(manager.should_checkpoint(4));
    }

    #[test]
    fn test_checkpoint_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let config = CheckpointConfig::new()
            .with_checkpoint_dir(temp_dir.path())
            .with_max_checkpoints(2);

        let manager = CheckpointManager::new(config.clone());

        // Create 4 checkpoints
        for i in 1..=4 {
            let checkpoint = FederatedCheckpoint::new(
                vec![i as f32],
                i as u64,
                HashMap::new(),
                vec![],
                &FederatedNetworkConfig::for_testing(),
                None,
            );
            let path = temp_dir
                .path()
                .join(format!("checkpoint_round_{:08}.bin", i));
            checkpoint.save_to_file(&path).unwrap();

            // Small delay to ensure different timestamps
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Cleanup
        manager.cleanup_old_checkpoints().unwrap();

        // Should only have 2 checkpoints left
        let remaining: Vec<_> = std::fs::read_dir(temp_dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();

        assert_eq!(remaining.len(), 2);
    }

    #[test]
    fn test_find_latest_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let config = CheckpointConfig::new().with_checkpoint_dir(temp_dir.path());

        let manager = CheckpointManager::new(config);

        // Initially no checkpoints
        assert!(manager.find_latest_checkpoint().unwrap().is_none());

        // Create some checkpoints with delays
        for i in [5, 10, 15] {
            let checkpoint = FederatedCheckpoint::new(
                vec![i as f32],
                i as u64,
                HashMap::new(),
                vec![],
                &FederatedNetworkConfig::for_testing(),
                None,
            );
            let path = temp_dir
                .path()
                .join(format!("checkpoint_round_{:08}.bin", i));
            checkpoint.save_to_file(&path).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Should find the latest (round 15)
        let latest = manager.find_latest_checkpoint().unwrap().unwrap();
        assert!(latest.to_string_lossy().contains("00000015"));
    }

    // =========================================================================
    // Recovery Tests
    // =========================================================================

    #[tokio::test]
    async fn test_coordinator_recovery() {
        let config = FederatedNetworkConfig::for_testing();

        // Create a checkpoint
        let checkpoint = FederatedCheckpoint::new(
            vec![1.0, 2.0, 3.0, 4.0],
            50,
            HashMap::new(),
            vec![],
            &config,
            Some([1u8; 32]),
        );

        // Recover
        let (coordinator, result) = recover_coordinator(checkpoint.clone(), config)
            .await
            .unwrap();

        assert_eq!(coordinator.get_weights(), vec![1.0, 2.0, 3.0, 4.0]);
        assert!(result.pending_gradients.is_empty());
        assert!(result.stale_nodes.is_empty());
    }

    #[test]
    fn test_node_rejoin_protocol() {
        let request = NodeRejoinRequest {
            node_id: [1u8; 32],
            last_round: 40,
            local_checkpoint: None,
        };

        let current_weights = vec![1.0, 2.0, 3.0];
        let response = process_node_rejoin(&request, &current_weights, 50, 60000);

        assert_eq!(response.current_round, 50);
        assert_eq!(response.missed_rounds, 10);
        assert!(!response.replay_gradients);
        assert_eq!(response.current_weights, current_weights);
    }

    #[test]
    fn test_node_rejoin_with_checkpoint() {
        let gradient = GradientMessage::new([1u8; 32], vec![0.1, 0.2], 0.8);
        let node_checkpoint =
            NodeCheckpoint::new([1u8; 32], vec![1.0, 2.0, 3.0], vec![gradient], vec![], 45);

        let request = NodeRejoinRequest {
            node_id: [1u8; 32],
            last_round: 45,
            local_checkpoint: Some(node_checkpoint),
        };

        let current_weights = vec![1.5, 2.5, 3.5];
        let response = process_node_rejoin(&request, &current_weights, 50, 60000 * 10); // Long max age

        assert_eq!(response.missed_rounds, 5);
        assert!(response.replay_gradients); // Should replay since checkpoint is recent
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[tokio::test]
    async fn test_full_checkpoint_cycle() {
        let temp_dir = TempDir::new().unwrap();
        let checkpoint_config = CheckpointConfig::new()
            .with_checkpoint_dir(temp_dir.path())
            .with_checkpoint_interval(1)
            .with_max_checkpoints(3);

        let mut manager = CheckpointManager::new(checkpoint_config);
        let network_config = FederatedNetworkConfig::for_testing();
        let coordinator = FederatedCoordinator::new(network_config.clone(), vec![0.0; 10]).await;

        // Record some contributions
        manager.record_contribution([1u8; 32]);
        manager.record_contribution([2u8; 32]);
        manager.record_contribution([1u8; 32]);

        // Create and save checkpoint
        let checkpoint = manager.create_checkpoint(&coordinator, &network_config, &[]);
        manager.save_checkpoint(&checkpoint).unwrap();

        // Should be able to load it back
        let loaded = manager.load_latest_checkpoint().unwrap().unwrap();
        assert_eq!(loaded.round_number, checkpoint.round_number);
        assert!(loaded.verify_checksum());
    }

    #[tokio::test]
    async fn test_checkpoint_resume_training() {
        let temp_dir = TempDir::new().unwrap();

        // Phase 1: Train and checkpoint
        let config = FederatedNetworkConfig::for_testing();
        let initial_weights = vec![0.0; 5];

        let coordinator1 = FederatedCoordinator::new(config.clone(), initial_weights.clone()).await;
        coordinator1.update_weights(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Checkpoint
        let checkpoint = FederatedCheckpoint::new(
            coordinator1.get_weights(),
            coordinator1.current_round(),
            HashMap::new(),
            vec![],
            &config,
            Some(coordinator1.local_node_id()),
        );

        let path = temp_dir.path().join("training_checkpoint.bin");
        checkpoint.save_to_file(&path).unwrap();

        // Phase 2: "Crash" and recover
        drop(coordinator1);

        let loaded = FederatedCheckpoint::load_from_file(&path).unwrap();
        let (coordinator2, _) = recover_coordinator(loaded, config).await.unwrap();

        // Verify state was restored
        assert_eq!(coordinator2.get_weights(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_node_checkpoint_cycle() {
        let temp_dir = TempDir::new().unwrap();

        // Create node checkpoint with pending work
        let gradient1 = GradientMessage::new([1u8; 32], vec![0.1, 0.2, 0.3], 0.8);
        let gradient2 = GradientMessage::new([2u8; 32], vec![0.4, 0.5, 0.6], 0.7);

        let node_cp = NodeCheckpoint::new(
            [42u8; 32],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![gradient1.clone(), gradient2.clone()],
            vec![[99u8; 32], [100u8; 32]],
            25,
        );

        let path = temp_dir.path().join("node_state.bin");
        node_cp.save_to_file(&path).unwrap();

        // Load and verify
        let loaded = NodeCheckpoint::load_from_file(&path).unwrap();

        assert_eq!(loaded.node_id, [42u8; 32]);
        assert_eq!(loaded.last_round, 25);
        assert_eq!(loaded.local_weights.len(), 5);
        assert_eq!(loaded.pending_gradients.len(), 2);
        assert_eq!(loaded.pending_message_hashes.len(), 2);

        // Verify gradient data preserved
        assert_eq!(
            loaded.pending_gradients[0].gradient_data,
            gradient1.gradient_data
        );
        assert_eq!(
            loaded.pending_gradients[1].trust_score,
            gradient2.trust_score
        );
    }
}
