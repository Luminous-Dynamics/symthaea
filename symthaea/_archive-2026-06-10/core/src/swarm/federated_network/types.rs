// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core types for federated network communication.

use crate::swarm::federated_cfc::{DifferentialPrivacyConfig, GradientMessage};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::{Duration, SystemTime};

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for the federated network
#[derive(Debug, Clone)]
pub struct FederatedNetworkConfig {
    /// Number of nodes in the federation
    pub num_nodes: usize,

    /// Interval between synchronization rounds (milliseconds)
    pub sync_interval_ms: u64,

    /// Timeout for network operations (milliseconds)
    pub timeout_ms: u64,

    /// Minimum number of nodes required for aggregation
    pub min_nodes_for_aggregation: usize,

    /// Enable differential privacy
    pub enable_dp: bool,

    /// Differential privacy configuration
    pub dp_config: Option<DifferentialPrivacyConfig>,

    /// Enable Byzantine fault tolerance
    pub enable_byzantine_tolerance: bool,

    /// Trim fraction for Byzantine tolerance (0.0 to 0.5)
    pub byzantine_trim_fraction: f32,

    /// Heartbeat interval (milliseconds)
    pub heartbeat_interval_ms: u64,

    /// Maximum staleness for gradients (milliseconds)
    pub max_gradient_staleness_ms: u64,
}

impl Default for FederatedNetworkConfig {
    fn default() -> Self {
        Self {
            num_nodes: 3,
            sync_interval_ms: 5000,
            timeout_ms: 10000,
            min_nodes_for_aggregation: 1,
            enable_dp: false,
            dp_config: None,
            enable_byzantine_tolerance: false,
            byzantine_trim_fraction: 0.1,
            heartbeat_interval_ms: 5000,
            max_gradient_staleness_ms: 60000,
        }
    }
}

impl FederatedNetworkConfig {
    /// Create a new config with specified number of nodes
    pub fn with_num_nodes(mut self, num_nodes: usize) -> Self {
        self.num_nodes = num_nodes;
        self.min_nodes_for_aggregation = (num_nodes / 2).max(1);
        self
    }

    /// Set the synchronization interval
    pub fn with_sync_interval_ms(mut self, interval_ms: u64) -> Self {
        self.sync_interval_ms = interval_ms;
        self
    }

    /// Set the network timeout
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Enable differential privacy with the given config
    pub fn with_differential_privacy(mut self, config: DifferentialPrivacyConfig) -> Self {
        self.enable_dp = true;
        self.dp_config = Some(config);
        self
    }

    /// Enable Byzantine fault tolerance
    pub fn with_byzantine_tolerance(mut self, trim_fraction: f32) -> Self {
        self.enable_byzantine_tolerance = true;
        self.byzantine_trim_fraction = trim_fraction.clamp(0.0, 0.5);
        self
    }

    /// Create a config for testing (fast timeouts, small network)
    pub fn for_testing() -> Self {
        Self {
            num_nodes: 3,
            sync_interval_ms: 100,
            timeout_ms: 1000,
            min_nodes_for_aggregation: 1,
            enable_dp: false,
            dp_config: None,
            enable_byzantine_tolerance: false,
            byzantine_trim_fraction: 0.1,
            heartbeat_interval_ms: 500,
            max_gradient_staleness_ms: 5000,
        }
    }
}

// ============================================================================
// MESSAGE TYPES
// ============================================================================

/// Messages exchanged in the federated network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FederatedMessage {
    /// Share gradient data with the network
    GradientShare(GradientMessage),

    /// Broadcast aggregated model update
    ModelUpdate {
        /// Aggregated weights
        weights: Vec<f32>,
        /// Round number
        round: u64,
        /// Source node ID
        source_id: [u8; 32],
        /// Timestamp
        timestamp: u64,
    },

    /// Heartbeat to indicate node is alive
    Heartbeat {
        /// Source node ID
        node_id: [u8; 32],
        /// Current round number
        round: u64,
        /// Node's trust score
        trust_score: f32,
        /// Timestamp
        timestamp: u64,
    },

    /// Request to join the federation
    JoinRequest {
        /// Node ID
        node_id: [u8; 32],
        /// Node's address
        address: String,
        /// Initial trust score
        trust_score: f32,
    },

    /// Acknowledgment of join request
    JoinAck {
        /// Accepted node ID
        node_id: [u8; 32],
        /// Current round number
        current_round: u64,
        /// List of known peers
        peers: Vec<([u8; 32], String)>,
    },

    /// Node is leaving the federation
    Leave {
        /// Node ID
        node_id: [u8; 32],
        /// Reason for leaving
        reason: String,
    },

    /// Request current model weights from coordinator
    SyncRequest {
        /// Node ID
        node_id: [u8; 32],
        /// Last known round
        last_round: u64,
    },

    /// Response with current model weights
    SyncResponse {
        /// Current weights
        weights: Vec<f32>,
        /// Current round
        round: u64,
    },
}

impl FederatedMessage {
    /// Get the message type as a string for logging
    pub fn message_type(&self) -> &'static str {
        match self {
            Self::GradientShare(_) => "GradientShare",
            Self::ModelUpdate { .. } => "ModelUpdate",
            Self::Heartbeat { .. } => "Heartbeat",
            Self::JoinRequest { .. } => "JoinRequest",
            Self::JoinAck { .. } => "JoinAck",
            Self::Leave { .. } => "Leave",
            Self::SyncRequest { .. } => "SyncRequest",
            Self::SyncResponse { .. } => "SyncResponse",
        }
    }

    /// Get the source node ID if available
    pub fn source_node_id(&self) -> Option<[u8; 32]> {
        match self {
            Self::GradientShare(g) => Some(g.source_id),
            Self::ModelUpdate { source_id, .. } => Some(*source_id),
            Self::Heartbeat { node_id, .. } => Some(*node_id),
            Self::JoinRequest { node_id, .. } => Some(*node_id),
            Self::JoinAck { node_id, .. } => Some(*node_id),
            Self::Leave { node_id, .. } => Some(*node_id),
            Self::SyncRequest { node_id, .. } => Some(*node_id),
            Self::SyncResponse { .. } => None,
        }
    }
}

// ============================================================================
// FEDERATED NODE
// ============================================================================

/// A node in the federated network
#[derive(Debug, Clone)]
pub struct FederatedNode {
    /// Unique 32-byte identifier for this node
    pub node_id: [u8; 32],

    /// Network address for this node
    pub address: NodeAddress,

    /// Trust score from the network (0.0 to 1.0)
    pub trust_score: f32,

    /// Last heartbeat timestamp
    pub last_heartbeat: u64,

    /// Current synchronization round
    pub current_round: u64,

    /// Whether this node is active
    pub is_active: bool,
}

/// Address of a node (can be socket address or channel identifier)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NodeAddress {
    /// Socket address for TCP connections
    Socket(SocketAddr),

    /// Channel identifier for local simulation
    Channel(String),

    /// Placeholder for unassigned address
    Unassigned,
}

impl std::fmt::Display for NodeAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Socket(addr) => write!(f, "tcp://{addr}"),
            Self::Channel(id) => write!(f, "channel://{id}"),
            Self::Unassigned => write!(f, "unassigned"),
        }
    }
}

impl FederatedNode {
    /// Create a new federated node
    pub fn new(node_id: [u8; 32], address: NodeAddress) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            node_id,
            address,
            trust_score: 0.5, // Default trust
            last_heartbeat: timestamp,
            current_round: 0,
            is_active: true,
        }
    }

    /// Create a node with a generated random ID
    pub fn with_random_id(address: NodeAddress) -> Self {
        let mut node_id = [0u8; 32];
        let mut rng = rand::thread_rng();
        rand::Rng::fill(&mut rng, &mut node_id);
        Self::new(node_id, address)
    }

    /// Get hex-encoded short ID (first 8 bytes)
    pub fn short_id(&self) -> String {
        hex::encode(&self.node_id[..8])
    }

    /// Update heartbeat timestamp
    pub fn update_heartbeat(&mut self) {
        self.last_heartbeat = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
    }

    /// Check if node is stale (no heartbeat in given duration)
    pub fn is_stale(&self, max_staleness_ms: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        now.saturating_sub(self.last_heartbeat) > max_staleness_ms
    }
}

// ============================================================================
// NETWORK BACKEND TRAIT
// ============================================================================

/// Result type for network operations
pub type NetworkResult<T> = Result<T, NetworkError>;

/// Errors that can occur in network operations
#[derive(Debug)]
pub enum NetworkError {
    /// Timeout waiting for response
    Timeout { operation: String, timeout_ms: u64 },

    /// Connection failed
    ConnectionFailed { target: String, reason: String },

    /// Channel closed
    ChannelClosed { reason: String },

    /// Send failed
    SendFailed { reason: String },

    /// Receive failed
    ReceiveFailed { reason: String },

    /// Node not found
    NodeNotFound { node_id: String },

    /// Serialization error
    Serialization { reason: String },

    /// Internal error
    Internal { reason: String },
}

impl std::fmt::Display for NetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Timeout {
                operation,
                timeout_ms,
            } => {
                write!(f, "Timeout after {timeout_ms}ms: {operation}")
            }
            Self::ConnectionFailed { target, reason } => {
                write!(f, "Connection to {target} failed: {reason}")
            }
            Self::ChannelClosed { reason } => {
                write!(f, "Channel closed: {reason}")
            }
            Self::SendFailed { reason } => {
                write!(f, "Send failed: {reason}")
            }
            Self::ReceiveFailed { reason } => {
                write!(f, "Receive failed: {reason}")
            }
            Self::NodeNotFound { node_id } => {
                write!(f, "Node not found: {node_id}")
            }
            Self::Serialization { reason } => {
                write!(f, "Serialization error: {reason}")
            }
            Self::Internal { reason } => {
                write!(f, "Internal error: {reason}")
            }
        }
    }
}

impl std::error::Error for NetworkError {}

/// Trait for network communication backends
///
/// This abstraction allows swapping between local channels (for testing)
/// and real network protocols (TCP, QUIC) for production.
#[async_trait]
pub trait NetworkBackend: Send + Sync {
    /// Send a message to a specific node
    async fn send(&self, target: &NodeAddress, message: FederatedMessage) -> NetworkResult<()>;

    /// Broadcast a message to all known nodes
    async fn broadcast(&self, message: FederatedMessage) -> NetworkResult<usize>;

    /// Receive the next message (blocking with timeout)
    async fn receive(&self, timeout: Duration) -> NetworkResult<(NodeAddress, FederatedMessage)>;

    /// Register a new node with the backend
    async fn register_node(&self, node: &FederatedNode) -> NetworkResult<()>;

    /// Unregister a node from the backend
    async fn unregister_node(&self, node_id: &[u8; 32]) -> NetworkResult<()>;

    /// Get the local node's address
    fn local_address(&self) -> NodeAddress;

    /// Check if backend is connected/ready
    fn is_ready(&self) -> bool;
}
