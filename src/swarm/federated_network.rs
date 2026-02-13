//! Federated Network Communication Layer
//!
//! This module provides network communication primitives for federated CfC learning
//! across the Symthaea swarm. It supports both local channel-based simulation for
//! testing and a real TCP backend for distributed network deployment.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                      FEDERATED NETWORK LAYER                                 │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                              │
//! │  ┌───────────────────────┐         ┌───────────────────────┐                │
//! │  │   FederatedNode       │         │   FederatedCoordinator│                │
//! │  │                       │         │                       │                │
//! │  │ • node_id (32 bytes)  │         │ • Node registry       │                │
//! │  │ • address (socket)    │    ────▶│ • Message routing     │                │
//! │  │ • trust_score         │         │ • Sync coordination   │                │
//! │  │ • local aggregator    │         │ • Heartbeat manager   │                │
//! │  └───────────────────────┘         └───────────────────────┘                │
//! │                                                                              │
//! │  ┌───────────────────────────────────────────────────────────────────────┐  │
//! │  │                      NetworkBackend (Trait)                            │  │
//! │  ├───────────────────────────────────────────────────────────────────────┤  │
//! │  │                                                                        │  │
//! │  │  ┌─────────────────────────┐    ┌─────────────────────────┐           │  │
//! │  │  │   LocalChannelBackend   │    │      TcpBackend        │           │  │
//! │  │  │   (Testing/Simulation)  │    │   (Real TCP Network)   │           │  │
//! │  │  │                         │    │                         │           │  │
//! │  │  │ • Tokio MPSC channels   │    │ • Length-prefixed wire  │           │  │
//! │  │  │ • Zero latency          │    │ • Async TCP with tokio  │           │  │
//! │  │  │ • Perfect reliability   │    │ • Connection pooling    │           │  │
//! │  │  └─────────────────────────┘    └─────────────────────────┘           │  │
//! │  └───────────────────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use symthaea::swarm::federated_network::{
//!     FederatedCoordinator, FederatedNetworkConfig, LocalChannelBackend,
//! };
//!
//! // Create coordinator with 3 nodes
//! let config = FederatedNetworkConfig::default()
//!     .with_num_nodes(3)
//!     .with_sync_interval_ms(1000);
//!
//! let coordinator = FederatedCoordinator::new_with_backend(
//!     config,
//!     LocalChannelBackend::new(),
//! ).await?;
//!
//! // Run federated learning round
//! coordinator.run_sync_round().await?;
//! ```

use crate::swarm::federated_cfc::{
    DifferentialPrivacyConfig, FederatedAggregator, GradientMessage,
};
use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{broadcast, mpsc, oneshot};
use tracing::{debug, info, warn};

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
            min_nodes_for_aggregation: 2,
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
            min_nodes_for_aggregation: 2,
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
            Self::Socket(addr) => write!(f, "tcp://{}", addr),
            Self::Channel(id) => write!(f, "channel://{}", id),
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
                write!(f, "Timeout after {}ms: {}", timeout_ms, operation)
            }
            Self::ConnectionFailed { target, reason } => {
                write!(f, "Connection to {} failed: {}", target, reason)
            }
            Self::ChannelClosed { reason } => {
                write!(f, "Channel closed: {}", reason)
            }
            Self::SendFailed { reason } => {
                write!(f, "Send failed: {}", reason)
            }
            Self::ReceiveFailed { reason } => {
                write!(f, "Receive failed: {}", reason)
            }
            Self::NodeNotFound { node_id } => {
                write!(f, "Node not found: {}", node_id)
            }
            Self::Serialization { reason } => {
                write!(f, "Serialization error: {}", reason)
            }
            Self::Internal { reason } => {
                write!(f, "Internal error: {}", reason)
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

// ============================================================================
// LOCAL CHANNEL BACKEND
// ============================================================================

/// Message wrapper with source address for channel routing
pub struct ChannelEnvelope {
    source: NodeAddress,
    message: FederatedMessage,
}

/// Local channel-based backend for testing and simulation
///
/// This backend uses Tokio MPSC channels to simulate network communication
/// between nodes in the same process. It's ideal for:
/// - Unit and integration tests
/// - Development and debugging
/// - Performance benchmarks without network overhead
pub struct LocalChannelBackend {
    /// Local node's identifier
    local_id: String,

    /// Senders for each registered node
    senders: Arc<RwLock<HashMap<String, mpsc::Sender<ChannelEnvelope>>>>,

    /// Receiver for incoming messages
    receiver: Arc<tokio::sync::Mutex<mpsc::Receiver<ChannelEnvelope>>>,

    /// Our own sender (for others to send to us)
    self_sender: mpsc::Sender<ChannelEnvelope>,

    /// Mapping from 32-byte node IDs to sender keys (channel IDs).
    /// Populated via `register_node()` and used by `unregister_node()`.
    node_id_map: Arc<RwLock<HashMap<[u8; 32], String>>>,

    /// Whether the backend is initialized
    ready: Arc<std::sync::atomic::AtomicBool>,
}

impl LocalChannelBackend {
    /// Create a new local channel backend
    pub fn new() -> Self {
        Self::with_id(format!("node-{}", rand::random::<u32>()))
    }

    /// Create a new local channel backend with a specific ID
    pub fn with_id(id: String) -> Self {
        let (tx, rx) = mpsc::channel(1000);

        Self {
            local_id: id,
            senders: Arc::new(RwLock::new(HashMap::new())),
            receiver: Arc::new(tokio::sync::Mutex::new(rx)),
            self_sender: tx,
            node_id_map: Arc::new(RwLock::new(HashMap::new())),
            ready: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }

    /// Get a sender for this backend (for other nodes to register)
    pub fn get_sender(&self) -> mpsc::Sender<ChannelEnvelope> {
        self.self_sender.clone()
    }

    /// Register another node's sender with this backend
    pub fn register_peer_sender(&self, node_id: &str, sender: mpsc::Sender<ChannelEnvelope>) {
        self.senders.write().insert(node_id.to_string(), sender);
    }

    /// Get the local ID
    pub fn local_id(&self) -> &str {
        &self.local_id
    }
}

impl Default for LocalChannelBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl NetworkBackend for LocalChannelBackend {
    async fn send(&self, target: &NodeAddress, message: FederatedMessage) -> NetworkResult<()> {
        let target_id = match target {
            NodeAddress::Channel(id) => id.clone(),
            _ => {
                return Err(NetworkError::SendFailed {
                    reason: "LocalChannelBackend only supports Channel addresses".to_string(),
                })
            }
        };

        let sender = self.senders.read().get(&target_id).cloned();

        match sender {
            Some(tx) => {
                let envelope = ChannelEnvelope {
                    source: NodeAddress::Channel(self.local_id.clone()),
                    message,
                };

                tx.send(envelope)
                    .await
                    .map_err(|_| NetworkError::ChannelClosed {
                        reason: format!("Channel to {} closed", target_id),
                    })?;

                Ok(())
            }
            None => Err(NetworkError::NodeNotFound { node_id: target_id }),
        }
    }

    async fn broadcast(&self, message: FederatedMessage) -> NetworkResult<usize> {
        let senders: Vec<(String, mpsc::Sender<ChannelEnvelope>)> = self
            .senders
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let mut sent_count = 0;

        for (node_id, tx) in senders {
            let envelope = ChannelEnvelope {
                source: NodeAddress::Channel(self.local_id.clone()),
                message: message.clone(),
            };

            if tx.send(envelope).await.is_ok() {
                sent_count += 1;
            } else {
                debug!("Failed to send to {}", node_id);
            }
        }

        Ok(sent_count)
    }

    async fn receive(&self, timeout: Duration) -> NetworkResult<(NodeAddress, FederatedMessage)> {
        let mut rx = self.receiver.lock().await;

        match tokio::time::timeout(timeout, rx.recv()).await {
            Ok(Some(envelope)) => Ok((envelope.source, envelope.message)),
            Ok(None) => Err(NetworkError::ChannelClosed {
                reason: "Receiver channel closed".to_string(),
            }),
            Err(_) => Err(NetworkError::Timeout {
                operation: "receive".to_string(),
                timeout_ms: timeout.as_millis() as u64,
            }),
        }
    }

    async fn register_node(&self, node: &FederatedNode) -> NetworkResult<()> {
        if let NodeAddress::Channel(ref id) = node.address {
            self.node_id_map.write().insert(node.node_id, id.clone());
        }
        debug!("Node registered: {}", node.short_id());
        Ok(())
    }

    async fn unregister_node(&self, node_id: &[u8; 32]) -> NetworkResult<()> {
        if let Some(key) = self.node_id_map.write().remove(node_id) {
            self.senders.write().remove(&key);
            debug!("Node {} unregistered (key: {})", hex::encode(&node_id[..8]), key);
        }
        Ok(())
    }

    fn local_address(&self) -> NodeAddress {
        NodeAddress::Channel(self.local_id.clone())
    }

    fn is_ready(&self) -> bool {
        self.ready.load(std::sync::atomic::Ordering::SeqCst)
    }
}

// ============================================================================
// TCP BACKEND
// ============================================================================

/// Maximum wire message size: 10 MB
const TCP_MAX_MESSAGE_SIZE: usize = 10_000_000;

/// TCP-based backend for real network communication
///
/// Provides a fully functional TCP networking layer for the federated swarm.
/// Messages are framed using a 4-byte big-endian length prefix followed by
/// bincode-serialized payload:
///
/// ```text
/// ┌─────────────────────┬──────────────────────────────────┐
/// │ 4 bytes (BE u32)    │  bincode(FederatedMessage)       │
/// │ payload length      │  variable length payload         │
/// └─────────────────────┴──────────────────────────────────┘
/// ```
///
/// # Connection Management
///
/// - Incoming connections are accepted by a background listener task.
/// - Outgoing connections are created lazily on first send and cached.
/// - Each connection is split into an `OwnedReadHalf` (spawned as a reader task)
///   and an `OwnedWriteHalf` (stored for sending).
/// - Connection failures automatically evict stale entries from the cache.
pub struct TcpBackend {
    /// Actual bound local address (resolved from port 0 if needed)
    local_addr: SocketAddr,

    /// Active outgoing TCP connections to peers, keyed by remote address.
    /// Each value is the write half of a TCP stream, wrapped in a tokio Mutex
    /// to allow concurrent sends to different peers while serializing writes
    /// to the same peer.
    connections: Arc<RwLock<HashMap<SocketAddr, Arc<tokio::sync::Mutex<tokio::net::tcp::OwnedWriteHalf>>>>>,

    /// Sender side of the incoming message channel.
    /// Reader tasks (one per accepted/connected stream) push received messages here.
    incoming_tx: mpsc::Sender<(NodeAddress, FederatedMessage)>,

    /// Receiver side of the incoming message channel.
    /// Protected by a tokio Mutex so only one caller can poll `receive()` at a time.
    incoming_rx: tokio::sync::Mutex<mpsc::Receiver<(NodeAddress, FederatedMessage)>>,

    /// Mapping from 32-byte node IDs to their socket addresses.
    /// Populated via `register_node()` and consulted during `send()`.
    registered_nodes: Arc<RwLock<HashMap<[u8; 32], SocketAddr>>>,

    /// Handle for the TCP listener accept-loop task.
    /// Stored so it can be inspected or aborted on shutdown.
    #[allow(dead_code)]
    listener_handle: tokio::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,

    /// Atomic flag indicating whether the backend has bound and is ready
    /// to accept connections and send messages.
    ready: Arc<std::sync::atomic::AtomicBool>,
}

// ---------------------------------------------------------------------------
// Wire helpers
// ---------------------------------------------------------------------------

/// Write a length-prefixed bincode frame to a TCP write half.
///
/// Frame layout: `[4-byte big-endian length][bincode payload]`
///
/// Returns `NetworkError::Serialization` if bincode encoding fails,
/// `NetworkError::SendFailed` if the payload exceeds `TCP_MAX_MESSAGE_SIZE`
/// or the write itself fails.
async fn write_framed(
    writer: &mut tokio::net::tcp::OwnedWriteHalf,
    msg: &FederatedMessage,
) -> NetworkResult<()> {
    use tokio::io::AsyncWriteExt;

    let data = bincode::serialize(msg).map_err(|e| NetworkError::Serialization {
        reason: e.to_string(),
    })?;

    if data.len() > TCP_MAX_MESSAGE_SIZE {
        return Err(NetworkError::SendFailed {
            reason: format!(
                "Message too large ({} bytes, max {} bytes)",
                data.len(),
                TCP_MAX_MESSAGE_SIZE
            ),
        });
    }

    let len_bytes = (data.len() as u32).to_be_bytes();

    writer
        .write_all(&len_bytes)
        .await
        .map_err(|e| NetworkError::SendFailed {
            reason: e.to_string(),
        })?;

    writer
        .write_all(&data)
        .await
        .map_err(|e| NetworkError::SendFailed {
            reason: e.to_string(),
        })?;

    Ok(())
}

/// Read a length-prefixed bincode frame from a TCP read half.
///
/// Returns `NetworkError::ReceiveFailed` on I/O errors or if the advertised
/// length exceeds `TCP_MAX_MESSAGE_SIZE`, and `NetworkError::Serialization`
/// if bincode decoding fails.
async fn read_framed(
    reader: &mut tokio::net::tcp::OwnedReadHalf,
) -> Result<FederatedMessage, NetworkError> {
    use tokio::io::AsyncReadExt;

    let mut len_buf = [0u8; 4];
    reader
        .read_exact(&mut len_buf)
        .await
        .map_err(|e| NetworkError::ReceiveFailed {
            reason: e.to_string(),
        })?;

    let len = u32::from_be_bytes(len_buf) as usize;

    if len > TCP_MAX_MESSAGE_SIZE {
        return Err(NetworkError::ReceiveFailed {
            reason: format!(
                "Incoming message too large ({} bytes, max {} bytes)",
                len, TCP_MAX_MESSAGE_SIZE
            ),
        });
    }

    let mut data = vec![0u8; len];
    reader
        .read_exact(&mut data)
        .await
        .map_err(|e| NetworkError::ReceiveFailed {
            reason: e.to_string(),
        })?;

    bincode::deserialize(&data).map_err(|e| NetworkError::Serialization {
        reason: e.to_string(),
    })
}

// ---------------------------------------------------------------------------
// TcpBackend implementation
// ---------------------------------------------------------------------------

impl TcpBackend {
    /// Create a new TCP backend bound to `bind_addr`.
    ///
    /// This is an async constructor because it:
    /// 1. Binds a `TcpListener` (supports port 0 for OS-assigned ports).
    /// 2. Stores the actual bound address (important when port 0 is used).
    /// 3. Spawns a background accept loop that handles incoming connections.
    ///
    /// # Errors
    ///
    /// Returns `NetworkError::Internal` if the bind fails.
    pub async fn new(bind_addr: SocketAddr) -> NetworkResult<Self> {
        let listener =
            tokio::net::TcpListener::bind(bind_addr)
                .await
                .map_err(|e| NetworkError::Internal {
                    reason: format!("Failed to bind TCP listener on {}: {}", bind_addr, e),
                })?;

        let local_addr = listener.local_addr().map_err(|e| NetworkError::Internal {
            reason: format!("Failed to get local address: {}", e),
        })?;

        let (incoming_tx, incoming_rx) = mpsc::channel(4096);

        let connections: Arc<
            RwLock<
                HashMap<SocketAddr, Arc<tokio::sync::Mutex<tokio::net::tcp::OwnedWriteHalf>>>,
            >,
        > = Arc::new(RwLock::new(HashMap::new()));

        let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // Spawn the accept loop
        let accept_tx = incoming_tx.clone();
        let accept_connections = connections.clone();
        let accept_ready = ready.clone();

        let listener_handle = tokio::spawn(async move {
            // Mark ready once the listener is running
            accept_ready.store(true, std::sync::atomic::Ordering::SeqCst);

            loop {
                match listener.accept().await {
                    Ok((stream, peer_addr)) => {
                        debug!("TCP: accepted connection from {}", peer_addr);

                        let (read_half, write_half) = stream.into_split();

                        // Cache the write half so we can send back to this peer
                        accept_connections.write().insert(
                            peer_addr,
                            Arc::new(tokio::sync::Mutex::new(write_half)),
                        );

                        // Spawn a reader task for this connection
                        let reader_tx = accept_tx.clone();
                        let reader_connections = accept_connections.clone();
                        tokio::spawn(Self::reader_loop(
                            read_half,
                            peer_addr,
                            reader_tx,
                            reader_connections,
                        ));
                    }
                    Err(e) => {
                        warn!("TCP: accept error: {}", e);
                        // Brief sleep to avoid a tight error loop
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                }
            }
        });

        Ok(Self {
            local_addr,
            connections,
            incoming_tx,
            incoming_rx: tokio::sync::Mutex::new(incoming_rx),
            registered_nodes: Arc::new(RwLock::new(HashMap::new())),
            listener_handle: tokio::sync::Mutex::new(Some(listener_handle)),
            ready,
        })
    }

    /// Background reader loop for a single TCP stream.
    ///
    /// Reads framed messages and forwards them into `incoming_tx`.
    /// On EOF or error the loop exits and the connection entry is removed.
    async fn reader_loop(
        mut read_half: tokio::net::tcp::OwnedReadHalf,
        peer_addr: SocketAddr,
        incoming_tx: mpsc::Sender<(NodeAddress, FederatedMessage)>,
        connections: Arc<
            RwLock<
                HashMap<SocketAddr, Arc<tokio::sync::Mutex<tokio::net::tcp::OwnedWriteHalf>>>,
            >,
        >,
    ) {
        loop {
            match read_framed(&mut read_half).await {
                Ok(msg) => {
                    debug!(
                        "TCP: received {} from {}",
                        msg.message_type(),
                        peer_addr
                    );
                    let source = NodeAddress::Socket(peer_addr);
                    if incoming_tx.send((source, msg)).await.is_err() {
                        // Channel closed; backend is being dropped
                        debug!("TCP: incoming channel closed, stopping reader for {}", peer_addr);
                        break;
                    }
                }
                Err(e) => {
                    debug!("TCP: reader for {} ended: {}", peer_addr, e);
                    // Remove the cached connection on error/EOF
                    connections.write().remove(&peer_addr);
                    break;
                }
            }
        }
    }

    /// Get or create an outgoing TCP connection to `addr`.
    ///
    /// If a cached write-half exists for `addr`, it is returned immediately.
    /// Otherwise a new TCP connection is established, split, and the read half
    /// is handed to a spawned reader task while the write half is cached and
    /// returned.
    async fn get_or_connect(
        &self,
        addr: SocketAddr,
    ) -> NetworkResult<Arc<tokio::sync::Mutex<tokio::net::tcp::OwnedWriteHalf>>> {
        // Fast path: connection already cached
        {
            let conns = self.connections.read();
            if let Some(writer) = conns.get(&addr) {
                return Ok(Arc::clone(writer));
            }
        }

        // Slow path: establish a new connection
        debug!("TCP: connecting to {}", addr);
        let stream = tokio::net::TcpStream::connect(addr)
            .await
            .map_err(|e| NetworkError::ConnectionFailed {
                target: addr.to_string(),
                reason: e.to_string(),
            })?;

        let (read_half, write_half) = stream.into_split();
        let writer = Arc::new(tokio::sync::Mutex::new(write_half));

        // Cache the write half
        self.connections
            .write()
            .insert(addr, Arc::clone(&writer));

        // Spawn a reader task for replies on this connection
        let reader_tx = self.incoming_tx.clone();
        let reader_connections = self.connections.clone();
        tokio::spawn(Self::reader_loop(
            read_half,
            addr,
            reader_tx,
            reader_connections,
        ));

        Ok(writer)
    }

    /// Resolve a `NodeAddress` to a `SocketAddr`.
    ///
    /// Returns `NetworkError::SendFailed` if the address is not a `Socket` variant.
    fn resolve_addr(target: &NodeAddress) -> NetworkResult<SocketAddr> {
        match target {
            NodeAddress::Socket(addr) => Ok(*addr),
            other => Err(NetworkError::SendFailed {
                reason: format!(
                    "TcpBackend requires Socket addresses, got {}",
                    other
                ),
            }),
        }
    }

    /// Look up the `SocketAddr` for a node by its 32-byte ID.
    #[allow(dead_code)]
    fn lookup_node_addr(&self, node_id: &[u8; 32]) -> NetworkResult<SocketAddr> {
        self.registered_nodes
            .read()
            .get(node_id)
            .copied()
            .ok_or_else(|| NetworkError::NodeNotFound {
                node_id: hex::encode(&node_id[..8]),
            })
    }
}

#[async_trait]
impl NetworkBackend for TcpBackend {
    /// Send a message to a specific node address.
    ///
    /// The target must be a `NodeAddress::Socket`. The backend will reuse an
    /// existing connection or establish a new one. If the send fails due to a
    /// broken pipe the cached connection is evicted so the next attempt will
    /// reconnect.
    async fn send(&self, target: &NodeAddress, message: FederatedMessage) -> NetworkResult<()> {
        let addr = Self::resolve_addr(target)?;

        let writer = self.get_or_connect(addr).await?;

        let result = {
            let mut guard = writer.lock().await;
            write_framed(&mut guard, &message).await
        };

        if result.is_err() {
            // Evict the broken connection so a future send will reconnect
            self.connections.write().remove(&addr);
        }

        result
    }

    /// Broadcast a message to all registered nodes.
    ///
    /// Sends are attempted sequentially to each registered peer. Failures are
    /// logged but do not abort the broadcast. Returns the number of peers that
    /// received the message successfully.
    async fn broadcast(&self, message: FederatedMessage) -> NetworkResult<usize> {
        let targets: Vec<([u8; 32], SocketAddr)> = self
            .registered_nodes
            .read()
            .iter()
            .map(|(id, addr)| (*id, *addr))
            .collect();

        let mut success_count = 0usize;

        for (node_id, addr) in &targets {
            let target = NodeAddress::Socket(*addr);
            match self.send(&target, message.clone()).await {
                Ok(()) => {
                    success_count += 1;
                }
                Err(e) => {
                    debug!(
                        "TCP: broadcast to {} ({}) failed: {}",
                        hex::encode(&node_id[..8]),
                        addr,
                        e
                    );
                }
            }
        }

        Ok(success_count)
    }

    /// Receive the next incoming message, blocking up to `timeout`.
    ///
    /// Returns `NetworkError::Timeout` if no message arrives within the
    /// deadline, or `NetworkError::ChannelClosed` if all senders have dropped.
    async fn receive(&self, timeout: Duration) -> NetworkResult<(NodeAddress, FederatedMessage)> {
        let mut rx = self.incoming_rx.lock().await;

        match tokio::time::timeout(timeout, rx.recv()).await {
            Ok(Some(pair)) => Ok(pair),
            Ok(None) => Err(NetworkError::ChannelClosed {
                reason: "Incoming message channel closed".to_string(),
            }),
            Err(_) => Err(NetworkError::Timeout {
                operation: "tcp_receive".to_string(),
                timeout_ms: timeout.as_millis() as u64,
            }),
        }
    }

    /// Register a node so that `broadcast()` and node-ID-based sends know how
    /// to reach it.
    ///
    /// The node's address must be `NodeAddress::Socket`.
    async fn register_node(&self, node: &FederatedNode) -> NetworkResult<()> {
        let addr = Self::resolve_addr(&node.address)?;
        self.registered_nodes.write().insert(node.node_id, addr);
        debug!(
            "TCP: registered node {} at {}",
            hex::encode(&node.node_id[..8]),
            addr
        );
        Ok(())
    }

    /// Unregister a node by ID, removing it from the routing table and
    /// closing any cached connection.
    async fn unregister_node(&self, node_id: &[u8; 32]) -> NetworkResult<()> {
        if let Some(addr) = self.registered_nodes.write().remove(node_id) {
            self.connections.write().remove(&addr);
            debug!(
                "TCP: unregistered node {} (was at {})",
                hex::encode(&node_id[..8]),
                addr
            );
        }
        Ok(())
    }

    /// Return the local bound socket address wrapped in `NodeAddress::Socket`.
    fn local_address(&self) -> NodeAddress {
        NodeAddress::Socket(self.local_addr)
    }

    /// Returns `true` once the TCP listener has successfully bound and the
    /// accept loop is running.
    fn is_ready(&self) -> bool {
        self.ready.load(std::sync::atomic::Ordering::SeqCst)
    }
}

// ============================================================================
// FEDERATED COORDINATOR
// ============================================================================

/// Statistics about the federated network
#[derive(Debug, Clone, Default)]
pub struct CoordinatorStats {
    /// Number of active nodes
    pub active_nodes: usize,

    /// Current synchronization round
    pub current_round: u64,

    /// Total gradients received
    pub gradients_received: u64,

    /// Total model updates broadcast
    pub model_updates_broadcast: u64,

    /// Total heartbeats received
    pub heartbeats_received: u64,

    /// Number of failed send attempts
    pub send_failures: u64,

    /// Uptime in seconds
    pub uptime_seconds: u64,
}

/// Event broadcast by the coordinator
#[derive(Debug, Clone)]
pub enum CoordinatorEvent {
    /// New node joined
    NodeJoined {
        node_id: [u8; 32],
        address: NodeAddress,
    },

    /// Node left or was removed
    NodeLeft { node_id: [u8; 32], reason: String },

    /// Synchronization round completed
    RoundCompleted { round: u64, participants: usize },

    /// Model updated
    ModelUpdated { round: u64, weights_dim: usize },

    /// Error occurred
    Error { message: String },
}

/// Coordinator for federated learning across multiple nodes
///
/// The coordinator manages:
/// - Node registration and heartbeat monitoring
/// - Gradient collection and aggregation
/// - Model update broadcasting
/// - Synchronization rounds
pub struct FederatedCoordinator {
    /// Configuration
    config: FederatedNetworkConfig,

    /// Network backend
    backend: Arc<dyn NetworkBackend>,

    /// Local node information
    local_node: Arc<RwLock<FederatedNode>>,

    /// Known peer nodes
    peers: Arc<RwLock<HashMap<[u8; 32], FederatedNode>>>,

    /// Federated aggregator for gradient processing
    aggregator: Arc<RwLock<FederatedAggregator>>,

    /// Current round number
    current_round: Arc<std::sync::atomic::AtomicU64>,

    /// Statistics
    stats: Arc<RwLock<CoordinatorStats>>,

    /// Event broadcaster
    event_tx: broadcast::Sender<CoordinatorEvent>,

    /// Start time
    start_time: std::time::Instant,

    /// Running flag
    running: Arc<std::sync::atomic::AtomicBool>,

    /// Shutdown signal sender
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl FederatedCoordinator {
    /// Create a new coordinator with a local channel backend (for testing)
    pub async fn new(config: FederatedNetworkConfig, initial_weights: Vec<f32>) -> Self {
        Self::new_with_backend(
            config,
            initial_weights,
            Arc::new(LocalChannelBackend::new()),
        )
        .await
    }

    /// Create a new coordinator with a specific backend
    pub async fn new_with_backend(
        config: FederatedNetworkConfig,
        initial_weights: Vec<f32>,
        backend: Arc<dyn NetworkBackend>,
    ) -> Self {
        let (event_tx, _) = broadcast::channel(100);
        let (shutdown_tx, _shutdown_rx) = oneshot::channel();

        // Create local node
        let local_node = FederatedNode::with_random_id(backend.local_address());

        // Create aggregator with optional DP and Byzantine tolerance
        let mut aggregator = FederatedAggregator::new(initial_weights)
            .with_source_id(local_node.node_id)
            .with_max_staleness(config.max_gradient_staleness_ms);

        if config.enable_dp {
            if let Some(dp_config) = config.dp_config {
                aggregator = aggregator.with_differential_privacy(dp_config);
            }
        }

        if config.enable_byzantine_tolerance {
            aggregator = aggregator.with_byzantine_tolerance(config.byzantine_trim_fraction);
        }

        Self {
            config,
            backend,
            local_node: Arc::new(RwLock::new(local_node)),
            peers: Arc::new(RwLock::new(HashMap::new())),
            aggregator: Arc::new(RwLock::new(aggregator)),
            current_round: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            stats: Arc::new(RwLock::new(CoordinatorStats::default())),
            event_tx,
            start_time: std::time::Instant::now(),
            running: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            shutdown_tx: Some(shutdown_tx),
        }
    }

    /// Get the local node ID
    pub fn local_node_id(&self) -> [u8; 32] {
        self.local_node.read().node_id
    }

    /// Get the local node's short ID
    pub fn local_short_id(&self) -> String {
        self.local_node.read().short_id()
    }

    /// Subscribe to coordinator events
    pub fn subscribe_events(&self) -> broadcast::Receiver<CoordinatorEvent> {
        self.event_tx.subscribe()
    }

    /// Get current statistics
    pub fn stats(&self) -> CoordinatorStats {
        let mut stats = self.stats.read().clone();
        stats.uptime_seconds = self.start_time.elapsed().as_secs();
        stats.active_nodes = self.peers.read().len();
        stats.current_round = self.current_round.load(std::sync::atomic::Ordering::SeqCst);
        stats
    }

    /// Get current round number
    pub fn current_round(&self) -> u64 {
        self.current_round.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get number of active peers
    pub fn peer_count(&self) -> usize {
        self.peers.read().len()
    }

    /// Get current model weights
    pub fn get_weights(&self) -> Vec<f32> {
        self.aggregator.read().local_weights().to_vec()
    }

    /// Get a reference to the network backend
    pub fn backend(&self) -> &Arc<dyn NetworkBackend> {
        &self.backend
    }

    /// Update local model weights
    pub fn update_weights(&self, weights: Vec<f32>) {
        self.aggregator.write().update_local_weights(weights);
    }

    /// Register a peer node
    pub async fn register_peer(&self, node: FederatedNode) {
        let node_id = node.node_id;
        let address = node.address.clone();

        // Register with backend for routing (node_id → address mapping)
        let _ = self.backend.register_node(&node).await;

        self.peers.write().insert(node_id, node);

        info!("Peer registered: {}", hex::encode(&node_id[..8]));

        let _ = self
            .event_tx
            .send(CoordinatorEvent::NodeJoined { node_id, address });
    }

    /// Unregister a peer node
    pub async fn unregister_peer(&self, node_id: &[u8; 32], reason: &str) {
        if self.peers.write().remove(node_id).is_some() {
            info!(
                "Peer unregistered: {} ({})",
                hex::encode(&node_id[..8]),
                reason
            );

            // Also remove from backend routing
            let _ = self.backend.unregister_node(node_id).await;

            let _ = self.event_tx.send(CoordinatorEvent::NodeLeft {
                node_id: *node_id,
                reason: reason.to_string(),
            });
        }
    }

    /// Process an incoming message
    pub async fn process_message(
        &self,
        source: NodeAddress,
        message: FederatedMessage,
    ) -> NetworkResult<()> {
        debug!("Processing {} from {}", message.message_type(), source);

        match message {
            FederatedMessage::GradientShare(gradient) => {
                self.handle_gradient_share(gradient).await?;
            }
            FederatedMessage::Heartbeat {
                node_id,
                round,
                trust_score,
                timestamp,
            } => {
                self.handle_heartbeat(node_id, round, trust_score, timestamp);
            }
            FederatedMessage::JoinRequest {
                node_id,
                address,
                trust_score,
            } => {
                self.handle_join_request(node_id, address, trust_score)
                    .await?;
            }
            FederatedMessage::SyncRequest {
                node_id,
                last_round,
            } => {
                self.handle_sync_request(source, node_id, last_round)
                    .await?;
            }
            FederatedMessage::Leave { node_id, reason } => {
                self.unregister_peer(&node_id, &reason).await;
            }
            FederatedMessage::ModelUpdate { weights, round, .. } => {
                self.handle_model_update(weights, round).await?;
            }
            _ => {
                debug!("Ignoring message type: {}", message.message_type());
            }
        }

        Ok(())
    }

    /// Handle incoming gradient share
    async fn handle_gradient_share(&self, gradient: GradientMessage) -> NetworkResult<()> {
        debug!(
            "Received gradient from {}",
            hex::encode(&gradient.source_id[..8])
        );

        let accepted = self.aggregator.write().receive_gradient(gradient);

        if accepted {
            self.stats.write().gradients_received += 1;
        }

        Ok(())
    }

    /// Handle heartbeat from a peer
    fn handle_heartbeat(&self, node_id: [u8; 32], round: u64, trust_score: f32, _timestamp: u64) {
        if let Some(peer) = self.peers.write().get_mut(&node_id) {
            peer.update_heartbeat();
            peer.current_round = round;
            peer.trust_score = trust_score;
        }

        self.stats.write().heartbeats_received += 1;
    }

    /// Handle join request
    async fn handle_join_request(
        &self,
        node_id: [u8; 32],
        address: String,
        trust_score: f32,
    ) -> NetworkResult<()> {
        let node = FederatedNode {
            node_id,
            address: NodeAddress::Channel(address.clone()),
            trust_score,
            last_heartbeat: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            current_round: 0,
            is_active: true,
        };

        self.register_peer(node).await;

        // Send acknowledgment (best-effort — the joining peer may not have
        // a backend channel set up yet, e.g. in test scenarios)
        let peers: Vec<([u8; 32], String)> = self
            .peers
            .read()
            .iter()
            .map(|(id, n)| (*id, n.address.to_string()))
            .collect();

        let ack = FederatedMessage::JoinAck {
            node_id,
            current_round: self.current_round(),
            peers,
        };

        if let Err(e) = self.backend.send(&NodeAddress::Channel(address), ack).await {
            debug!("JoinAck delivery failed (peer may connect later): {}", e);
        }

        Ok(())
    }

    /// Handle sync request
    async fn handle_sync_request(
        &self,
        source: NodeAddress,
        _node_id: [u8; 32],
        _last_round: u64,
    ) -> NetworkResult<()> {
        let response = FederatedMessage::SyncResponse {
            weights: self.get_weights(),
            round: self.current_round(),
        };

        self.backend.send(&source, response).await?;

        Ok(())
    }

    /// Handle model update from coordinator
    async fn handle_model_update(&self, weights: Vec<f32>, round: u64) -> NetworkResult<()> {
        if round > self.current_round() {
            self.update_weights(weights);
            self.current_round
                .store(round, std::sync::atomic::Ordering::SeqCst);

            let _ = self.event_tx.send(CoordinatorEvent::ModelUpdated {
                round,
                weights_dim: self.get_weights().len(),
            });
        }

        Ok(())
    }

    /// Share local gradient with all peers
    pub async fn share_gradient(&self, noise_scale: f32) -> NetworkResult<usize> {
        let gradient = self.aggregator.read().export_local_gradient(noise_scale);
        let message = FederatedMessage::GradientShare(gradient);

        let sent = self.backend.broadcast(message).await?;

        debug!("Shared gradient with {} peers", sent);
        Ok(sent)
    }

    /// Send heartbeat to all peers
    pub async fn send_heartbeat(&self) -> NetworkResult<usize> {
        let local_node = self.local_node.read().clone();

        let message = FederatedMessage::Heartbeat {
            node_id: local_node.node_id,
            round: self.current_round(),
            trust_score: local_node.trust_score,
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };

        self.backend.broadcast(message).await
    }

    /// Run a single synchronization round
    ///
    /// This collects gradients from peers, aggregates them, and broadcasts
    /// the updated model.
    pub async fn run_sync_round(&self) -> NetworkResult<Option<Vec<f32>>> {
        let round = self
            .current_round
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;

        info!("Starting sync round {}", round);

        // Check if we have enough peers
        let peer_count = self.peer_count();
        if peer_count < self.config.min_nodes_for_aggregation.saturating_sub(1) {
            warn!(
                "Not enough peers for aggregation ({}/{})",
                peer_count, self.config.min_nodes_for_aggregation
            );
            return Ok(None);
        }

        // Share our gradient
        let noise_scale = if self.config.enable_dp {
            self.config
                .dp_config
                .as_ref()
                .map(|c| c.sigma() as f32)
                .unwrap_or(0.0)
        } else {
            0.0
        };

        self.share_gradient(noise_scale).await?;

        // Wait for gradients from peers (with timeout)
        let timeout = Duration::from_millis(self.config.timeout_ms);
        let deadline = std::time::Instant::now() + timeout;

        while std::time::Instant::now() < deadline {
            // Check if we have enough contributions
            if self.aggregator.read().pending_contributions()
                >= self.config.min_nodes_for_aggregation
            {
                break;
            }

            // Try to receive a message
            match self.backend.receive(Duration::from_millis(100)).await {
                Ok((source, message)) => {
                    if let Err(e) = self.process_message(source, message).await {
                        warn!("Error processing message: {}", e);
                    }
                }
                Err(NetworkError::Timeout { .. }) => {
                    // Continue waiting
                }
                Err(e) => {
                    warn!("Error receiving message: {}", e);
                }
            }
        }

        // Perform aggregation
        let aggregated = self.aggregator.write().aggregate();

        if let Some(ref new_weights) = aggregated {
            // Apply aggregated gradient
            self.aggregator.write().apply_gradient(new_weights, 0.1);

            // Broadcast updated model
            let local_node = self.local_node.read();
            let update_msg = FederatedMessage::ModelUpdate {
                weights: self.get_weights(),
                round,
                source_id: local_node.node_id,
                timestamp: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
            };

            self.backend.broadcast(update_msg).await?;
            self.stats.write().model_updates_broadcast += 1;

            // Emit event
            let _ = self.event_tx.send(CoordinatorEvent::RoundCompleted {
                round,
                participants: peer_count + 1,
            });

            info!(
                "Round {} completed with {} participants",
                round,
                peer_count + 1
            );
        }

        Ok(aggregated)
    }

    /// Shutdown the coordinator
    pub async fn shutdown(&mut self) {
        self.running
            .store(false, std::sync::atomic::Ordering::SeqCst);

        // Send leave message to all peers
        let local_node = self.local_node.read().clone();
        let leave_msg = FederatedMessage::Leave {
            node_id: local_node.node_id,
            reason: "Coordinator shutdown".to_string(),
        };

        let _ = self.backend.broadcast(leave_msg).await;

        // Take and drop the shutdown sender
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }

        info!("Coordinator shutdown complete");
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Create a fully connected local channel network for testing
///
/// Returns a vector of coordinators that can communicate with each other.
pub async fn create_test_network(
    num_nodes: usize,
    weights_dim: usize,
) -> Vec<FederatedCoordinator> {
    let config = FederatedNetworkConfig::for_testing().with_num_nodes(num_nodes);
    let initial_weights = vec![0.0f32; weights_dim];

    // Create backends first
    let backends: Vec<Arc<LocalChannelBackend>> = (0..num_nodes)
        .map(|i| Arc::new(LocalChannelBackend::with_id(format!("node-{}", i))))
        .collect();

    // Cross-register all backends
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            if i != j {
                backends[i].register_peer_sender(backends[j].local_id(), backends[j].get_sender());
            }
        }
    }

    // Create coordinators
    let mut coordinators = Vec::new();
    for backend in backends {
        let coordinator = FederatedCoordinator::new_with_backend(
            config.clone(),
            initial_weights.clone(),
            backend,
        )
        .await;
        coordinators.push(coordinator);
    }

    // Register peers with each other
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            if i != j {
                let peer_node = FederatedNode::new(
                    coordinators[j].local_node_id(),
                    NodeAddress::Channel(format!("node-{}", j)),
                );
                coordinators[i].register_peer(peer_node).await;
            }
        }
    }

    coordinators
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Configuration Tests
    // =========================================================================

    #[test]
    fn test_config_default() {
        let config = FederatedNetworkConfig::default();
        assert_eq!(config.num_nodes, 3);
        assert_eq!(config.sync_interval_ms, 5000);
        assert_eq!(config.timeout_ms, 10000);
        assert!(!config.enable_dp);
        assert!(!config.enable_byzantine_tolerance);
    }

    #[test]
    fn test_config_with_num_nodes() {
        let config = FederatedNetworkConfig::default().with_num_nodes(5);
        assert_eq!(config.num_nodes, 5);
        assert_eq!(config.min_nodes_for_aggregation, 2);
    }

    #[test]
    fn test_config_with_dp() {
        let config = FederatedNetworkConfig::default()
            .with_differential_privacy(DifferentialPrivacyConfig::moderate_privacy());
        assert!(config.enable_dp);
        assert!(config.dp_config.is_some());
    }

    #[test]
    fn test_config_for_testing() {
        let config = FederatedNetworkConfig::for_testing();
        assert_eq!(config.sync_interval_ms, 100);
        assert_eq!(config.timeout_ms, 1000);
    }

    // =========================================================================
    // Message Tests
    // =========================================================================

    #[test]
    fn test_message_types() {
        let gradient = GradientMessage::new([0u8; 32], vec![0.1; 10], 0.8);
        let msg = FederatedMessage::GradientShare(gradient);
        assert_eq!(msg.message_type(), "GradientShare");

        let msg = FederatedMessage::Heartbeat {
            node_id: [0u8; 32],
            round: 1,
            trust_score: 0.9,
            timestamp: 0,
        };
        assert_eq!(msg.message_type(), "Heartbeat");
    }

    #[test]
    fn test_message_source_node_id() {
        let gradient = GradientMessage::new([1u8; 32], vec![0.1; 10], 0.8);
        let msg = FederatedMessage::GradientShare(gradient);
        assert_eq!(msg.source_node_id(), Some([1u8; 32]));

        let msg = FederatedMessage::SyncResponse {
            weights: vec![],
            round: 0,
        };
        assert_eq!(msg.source_node_id(), None);
    }

    // =========================================================================
    // Node Tests
    // =========================================================================

    #[test]
    fn test_node_creation() {
        let node = FederatedNode::new([0u8; 32], NodeAddress::Unassigned);
        assert_eq!(node.node_id, [0u8; 32]);
        assert_eq!(node.trust_score, 0.5);
        assert!(node.is_active);
    }

    #[test]
    fn test_node_with_random_id() {
        let node = FederatedNode::with_random_id(NodeAddress::Unassigned);
        assert!(node.node_id != [0u8; 32]);
    }

    #[test]
    fn test_node_short_id() {
        let mut id = [0u8; 32];
        id[0] = 0xAB;
        id[1] = 0xCD;
        let node = FederatedNode::new(id, NodeAddress::Unassigned);
        assert_eq!(node.short_id().len(), 16);
        assert!(node.short_id().starts_with("abcd"));
    }

    #[test]
    fn test_node_staleness() {
        let mut node = FederatedNode::new([0u8; 32], NodeAddress::Unassigned);
        assert!(!node.is_stale(60000)); // Not stale within 1 minute

        // Set heartbeat to past
        node.last_heartbeat = 0;
        assert!(node.is_stale(60000)); // Should be stale now
    }

    // =========================================================================
    // Node Address Tests
    // =========================================================================

    #[test]
    fn test_node_address_display() {
        let addr = NodeAddress::Channel("test-123".to_string());
        assert_eq!(format!("{}", addr), "channel://test-123");

        let socket_addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let addr = NodeAddress::Socket(socket_addr);
        assert_eq!(format!("{}", addr), "tcp://127.0.0.1:8080");

        let addr = NodeAddress::Unassigned;
        assert_eq!(format!("{}", addr), "unassigned");
    }

    // =========================================================================
    // Local Channel Backend Tests
    // =========================================================================

    #[tokio::test]
    async fn test_local_backend_creation() {
        let backend = LocalChannelBackend::new();
        assert!(backend.is_ready());
        assert!(matches!(backend.local_address(), NodeAddress::Channel(_)));
    }

    #[tokio::test]
    async fn test_local_backend_with_id() {
        let backend = LocalChannelBackend::with_id("my-node".to_string());
        assert_eq!(backend.local_id(), "my-node");
    }

    #[tokio::test]
    async fn test_local_backend_send_to_unknown() {
        let backend = LocalChannelBackend::new();
        let result = backend
            .send(
                &NodeAddress::Channel("unknown".to_string()),
                FederatedMessage::Heartbeat {
                    node_id: [0u8; 32],
                    round: 0,
                    trust_score: 0.5,
                    timestamp: 0,
                },
            )
            .await;
        assert!(matches!(result, Err(NetworkError::NodeNotFound { .. })));
    }

    #[tokio::test]
    async fn test_local_backend_communication() {
        let backend1 = LocalChannelBackend::with_id("node-1".to_string());
        let backend2 = LocalChannelBackend::with_id("node-2".to_string());

        // Cross-register
        backend1.register_peer_sender("node-2", backend2.get_sender());
        backend2.register_peer_sender("node-1", backend1.get_sender());

        // Send from 1 to 2
        let msg = FederatedMessage::Heartbeat {
            node_id: [1u8; 32],
            round: 5,
            trust_score: 0.8,
            timestamp: 12345,
        };

        backend1
            .send(&NodeAddress::Channel("node-2".to_string()), msg.clone())
            .await
            .unwrap();

        // Receive on 2
        let (source, received) = backend2.receive(Duration::from_secs(1)).await.unwrap();
        assert_eq!(source, NodeAddress::Channel("node-1".to_string()));
        assert_eq!(received.message_type(), "Heartbeat");
    }

    #[tokio::test]
    async fn test_local_backend_broadcast() {
        let backend1 = LocalChannelBackend::with_id("node-1".to_string());
        let backend2 = LocalChannelBackend::with_id("node-2".to_string());
        let backend3 = LocalChannelBackend::with_id("node-3".to_string());

        // Register all peers with node-1
        backend1.register_peer_sender("node-2", backend2.get_sender());
        backend1.register_peer_sender("node-3", backend3.get_sender());

        // Broadcast from node-1
        let msg = FederatedMessage::Heartbeat {
            node_id: [1u8; 32],
            round: 1,
            trust_score: 0.5,
            timestamp: 0,
        };

        let sent_count = backend1.broadcast(msg).await.unwrap();
        assert_eq!(sent_count, 2);

        // Both should receive
        let _ = backend2.receive(Duration::from_secs(1)).await.unwrap();
        let _ = backend3.receive(Duration::from_secs(1)).await.unwrap();
    }

    #[tokio::test]
    async fn test_local_backend_receive_timeout() {
        let backend = LocalChannelBackend::new();
        let result = backend.receive(Duration::from_millis(10)).await;
        assert!(matches!(result, Err(NetworkError::Timeout { .. })));
    }

    // =========================================================================
    // Coordinator Tests
    // =========================================================================

    #[tokio::test]
    async fn test_coordinator_creation() {
        let config = FederatedNetworkConfig::for_testing();
        let coordinator = FederatedCoordinator::new(config, vec![0.0; 10]).await;

        assert_eq!(coordinator.current_round(), 0);
        assert_eq!(coordinator.peer_count(), 0);
        assert_eq!(coordinator.get_weights().len(), 10);
    }

    #[tokio::test]
    async fn test_coordinator_update_weights() {
        let config = FederatedNetworkConfig::for_testing();
        let coordinator = FederatedCoordinator::new(config, vec![0.0; 5]).await;

        coordinator.update_weights(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let weights = coordinator.get_weights();

        assert_eq!(weights.len(), 5);
        assert!((weights[0] - 1.0).abs() < 0.001);
        assert!((weights[4] - 5.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_coordinator_register_peer() {
        let config = FederatedNetworkConfig::for_testing();
        let coordinator = FederatedCoordinator::new(config, vec![0.0; 10]).await;

        let peer = FederatedNode::with_random_id(NodeAddress::Channel("peer-1".to_string()));
        let peer_id = peer.node_id;

        coordinator.register_peer(peer).await;
        assert_eq!(coordinator.peer_count(), 1);

        coordinator.unregister_peer(&peer_id, "test").await;
        assert_eq!(coordinator.peer_count(), 0);
    }

    #[tokio::test]
    async fn test_coordinator_stats() {
        let config = FederatedNetworkConfig::for_testing();
        let coordinator = FederatedCoordinator::new(config, vec![0.0; 10]).await;

        let stats = coordinator.stats();
        assert_eq!(stats.active_nodes, 0);
        assert_eq!(stats.current_round, 0);
        assert_eq!(stats.gradients_received, 0);
    }

    #[tokio::test]
    async fn test_coordinator_event_subscription() {
        let config = FederatedNetworkConfig::for_testing();
        let coordinator = FederatedCoordinator::new(config, vec![0.0; 10]).await;

        let mut rx = coordinator.subscribe_events();

        let peer = FederatedNode::with_random_id(NodeAddress::Channel("peer-1".to_string()));
        coordinator.register_peer(peer).await;

        let event = rx.try_recv();
        assert!(event.is_ok());
        assert!(matches!(
            event.unwrap(),
            CoordinatorEvent::NodeJoined { .. }
        ));
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[tokio::test]
    async fn test_create_test_network() {
        let coordinators = create_test_network(3, 10).await;

        assert_eq!(coordinators.len(), 3);

        for coordinator in &coordinators {
            // Each coordinator should see 2 peers
            assert_eq!(coordinator.peer_count(), 2);
            assert_eq!(coordinator.get_weights().len(), 10);
        }
    }

    #[tokio::test]
    async fn test_gradient_sharing() {
        let coordinators = create_test_network(3, 10).await;

        // Update weights on coordinator 0
        coordinators[0].update_weights(vec![1.0; 10]);

        // Share gradient from coordinator 0
        let sent = coordinators[0].share_gradient(0.0).await.unwrap();
        assert_eq!(sent, 2); // Should send to 2 peers

        // Give a bit of time for messages to arrive
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Coordinator 1 should be able to receive the gradient
        let result = coordinators[1]
            .backend
            .receive(Duration::from_millis(100))
            .await;
        assert!(result.is_ok());

        let (_, msg) = result.unwrap();
        assert_eq!(msg.message_type(), "GradientShare");
    }

    #[tokio::test]
    async fn test_heartbeat_broadcast() {
        let coordinators = create_test_network(2, 5).await;

        // Send heartbeat from coordinator 0
        let sent = coordinators[0].send_heartbeat().await.unwrap();
        assert_eq!(sent, 1); // Should send to 1 peer

        // Coordinator 1 should receive it
        let (_, msg) = coordinators[1]
            .backend
            .receive(Duration::from_millis(100))
            .await
            .unwrap();
        assert_eq!(msg.message_type(), "Heartbeat");
    }

    #[tokio::test]
    async fn test_full_sync_round() {
        let coordinators = create_test_network(3, 4).await;

        // Set different initial weights for each coordinator
        coordinators[0].update_weights(vec![1.0, 0.0, 0.0, 0.0]);
        coordinators[1].update_weights(vec![0.0, 1.0, 0.0, 0.0]);
        coordinators[2].update_weights(vec![0.0, 0.0, 1.0, 0.0]);

        // Run sync round on coordinator 0 (acting as coordinator)
        // First, the other coordinators need to share their gradients
        coordinators[1].share_gradient(0.0).await.unwrap();
        coordinators[2].share_gradient(0.0).await.unwrap();

        // Run sync round on coordinator 0
        let result = coordinators[0].run_sync_round().await;
        assert!(result.is_ok());

        // The result may or may not have aggregated weights depending on timing
        // In a real scenario, we'd need more sophisticated synchronization
    }

    #[tokio::test]
    async fn test_coordinator_shutdown() {
        let config = FederatedNetworkConfig::for_testing();
        let mut coordinator = FederatedCoordinator::new(config, vec![0.0; 10]).await;

        // Should not panic
        coordinator.shutdown().await;
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_network_error_display() {
        let err = NetworkError::Timeout {
            operation: "receive".to_string(),
            timeout_ms: 1000,
        };
        assert!(format!("{}", err).contains("1000"));
        assert!(format!("{}", err).contains("receive"));

        let err = NetworkError::NodeNotFound {
            node_id: "abc123".to_string(),
        };
        assert!(format!("{}", err).contains("abc123"));
    }

    #[test]
    fn test_network_error_timeout_construction_and_format() {
        let err = NetworkError::Timeout {
            operation: "gradient_sync".to_string(),
            timeout_ms: 5000,
        };
        let display = format!("{}", err);
        let debug = format!("{:?}", err);

        assert!(!display.is_empty());
        assert!(display.contains("5000"));
        assert!(display.contains("gradient_sync"));
        assert!(display.contains("Timeout"));
        assert!(!debug.is_empty());
        assert!(debug.contains("Timeout"));
    }

    #[test]
    fn test_network_error_connection_failed_construction_and_format() {
        let err = NetworkError::ConnectionFailed {
            target: "192.168.1.100:8080".to_string(),
            reason: "connection refused".to_string(),
        };
        let display = format!("{}", err);
        let debug = format!("{:?}", err);

        assert!(!display.is_empty());
        assert!(display.contains("192.168.1.100:8080"));
        assert!(display.contains("connection refused"));
        assert!(!debug.is_empty());
        assert!(debug.contains("ConnectionFailed"));
    }

    #[test]
    fn test_network_error_channel_closed_construction_and_format() {
        let err = NetworkError::ChannelClosed {
            reason: "receiver dropped".to_string(),
        };
        let display = format!("{}", err);
        let debug = format!("{:?}", err);

        assert!(!display.is_empty());
        assert!(display.contains("receiver dropped"));
        assert!(display.contains("Channel closed"));
        assert!(!debug.is_empty());
        assert!(debug.contains("ChannelClosed"));
    }

    #[test]
    fn test_network_error_send_failed_construction_and_format() {
        let err = NetworkError::SendFailed {
            reason: "broken pipe".to_string(),
        };
        let display = format!("{}", err);
        let debug = format!("{:?}", err);

        assert!(!display.is_empty());
        assert!(display.contains("broken pipe"));
        assert!(display.contains("Send failed"));
        assert!(!debug.is_empty());
        assert!(debug.contains("SendFailed"));
    }

    #[test]
    fn test_network_error_receive_failed_construction_and_format() {
        let err = NetworkError::ReceiveFailed {
            reason: "unexpected EOF".to_string(),
        };
        let display = format!("{}", err);
        let debug = format!("{:?}", err);

        assert!(!display.is_empty());
        assert!(display.contains("unexpected EOF"));
        assert!(display.contains("Receive failed"));
        assert!(!debug.is_empty());
        assert!(debug.contains("ReceiveFailed"));
    }

    #[test]
    fn test_network_error_node_not_found_construction_and_format() {
        let err = NetworkError::NodeNotFound {
            node_id: "abcdef0123456789".to_string(),
        };
        let display = format!("{}", err);
        let debug = format!("{:?}", err);

        assert!(!display.is_empty());
        assert!(display.contains("abcdef0123456789"));
        assert!(display.contains("Node not found"));
        assert!(!debug.is_empty());
        assert!(debug.contains("NodeNotFound"));
    }

    #[test]
    fn test_network_error_serialization_construction_and_format() {
        let err = NetworkError::Serialization {
            reason: "invalid bincode payload".to_string(),
        };
        let display = format!("{}", err);
        let debug = format!("{:?}", err);

        assert!(!display.is_empty());
        assert!(display.contains("invalid bincode payload"));
        assert!(display.contains("Serialization error"));
        assert!(!debug.is_empty());
        assert!(debug.contains("Serialization"));
    }

    #[test]
    fn test_network_error_internal_construction_and_format() {
        let err = NetworkError::Internal {
            reason: "failed to bind TCP listener".to_string(),
        };
        let display = format!("{}", err);
        let debug = format!("{:?}", err);

        assert!(!display.is_empty());
        assert!(display.contains("failed to bind TCP listener"));
        assert!(display.contains("Internal error"));
        assert!(!debug.is_empty());
        assert!(debug.contains("Internal"));
    }

    #[test]
    fn test_network_error_implements_std_error() {
        let err = NetworkError::Timeout {
            operation: "test".to_string(),
            timeout_ms: 100,
        };
        // Verify the error can be used as a dyn std::error::Error
        let err_ref: &dyn std::error::Error = &err;
        assert!(!err_ref.to_string().is_empty());
        // NetworkError has no source() impl, so source should be None
        assert!(err_ref.source().is_none());
    }

    #[test]
    fn test_all_network_error_variants_have_nonempty_display() {
        let errors: Vec<NetworkError> = vec![
            NetworkError::Timeout {
                operation: "op".to_string(),
                timeout_ms: 1,
            },
            NetworkError::ConnectionFailed {
                target: "t".to_string(),
                reason: "r".to_string(),
            },
            NetworkError::ChannelClosed {
                reason: "r".to_string(),
            },
            NetworkError::SendFailed {
                reason: "r".to_string(),
            },
            NetworkError::ReceiveFailed {
                reason: "r".to_string(),
            },
            NetworkError::NodeNotFound {
                node_id: "n".to_string(),
            },
            NetworkError::Serialization {
                reason: "r".to_string(),
            },
            NetworkError::Internal {
                reason: "r".to_string(),
            },
        ];

        assert_eq!(errors.len(), 8, "Expected 8 NetworkError variants");

        for (i, err) in errors.iter().enumerate() {
            let display = format!("{}", err);
            let debug = format!("{:?}", err);
            assert!(
                !display.is_empty(),
                "Variant {} has empty Display output",
                i
            );
            assert!(!debug.is_empty(), "Variant {} has empty Debug output", i);
        }
    }

    #[test]
    fn test_network_error_display_messages_are_distinct() {
        let errors: Vec<NetworkError> = vec![
            NetworkError::Timeout {
                operation: "op".to_string(),
                timeout_ms: 1,
            },
            NetworkError::ConnectionFailed {
                target: "t".to_string(),
                reason: "r".to_string(),
            },
            NetworkError::ChannelClosed {
                reason: "r".to_string(),
            },
            NetworkError::SendFailed {
                reason: "r".to_string(),
            },
            NetworkError::ReceiveFailed {
                reason: "r".to_string(),
            },
            NetworkError::NodeNotFound {
                node_id: "n".to_string(),
            },
            NetworkError::Serialization {
                reason: "r".to_string(),
            },
            NetworkError::Internal {
                reason: "r".to_string(),
            },
        ];

        let displays: Vec<String> = errors.iter().map(|e| format!("{}", e)).collect();

        // Each variant should produce a unique prefix (the part before the user-supplied content)
        for i in 0..displays.len() {
            for j in (i + 1)..displays.len() {
                assert_ne!(
                    displays[i], displays[j],
                    "Variants {} and {} produce identical Display output: '{}'",
                    i, j, displays[i]
                );
            }
        }
    }

    // =========================================================================
    // TCP Backend Tests
    // =========================================================================

    #[tokio::test]
    async fn test_tcp_backend_binds_and_ready() {
        let backend = TcpBackend::new("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();

        // The accept loop should have set the ready flag
        // Give it a brief moment to start
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(backend.is_ready());

        // local_address should be a Socket with a real port (not 0)
        match backend.local_address() {
            NodeAddress::Socket(addr) => {
                assert_ne!(addr.port(), 0);
                assert_eq!(addr.ip(), std::net::Ipv4Addr::LOCALHOST);
            }
            other => panic!("Expected Socket address, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_tcp_two_node_send_receive() {
        let backend1 = TcpBackend::new("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let backend2 = TcpBackend::new("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();

        // Register each with the other
        let node1 = FederatedNode::new([1u8; 32], backend1.local_address());
        let node2 = FederatedNode::new([2u8; 32], backend2.local_address());
        backend1.register_node(&node2).await.unwrap();
        backend2.register_node(&node1).await.unwrap();

        // Send a Heartbeat from backend1 to backend2
        let msg = FederatedMessage::Heartbeat {
            node_id: [1u8; 32],
            round: 1,
            trust_score: 0.5,
            timestamp: 42,
        };
        backend1
            .send(&backend2.local_address(), msg)
            .await
            .unwrap();

        // Receive on backend2
        let (_addr, received) = backend2.receive(Duration::from_secs(5)).await.unwrap();
        match received {
            FederatedMessage::Heartbeat {
                node_id,
                round,
                trust_score,
                timestamp,
            } => {
                assert_eq!(node_id, [1u8; 32]);
                assert_eq!(round, 1);
                assert!((trust_score - 0.5).abs() < f32::EPSILON);
                assert_eq!(timestamp, 42);
            }
            other => panic!("Expected Heartbeat, got {:?}", other.message_type()),
        }
    }

    #[tokio::test]
    async fn test_tcp_broadcast() {
        let backend1 = TcpBackend::new("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let backend2 = TcpBackend::new("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let backend3 = TcpBackend::new("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();

        // Register nodes 2 and 3 with node 1
        let node2 = FederatedNode::new([2u8; 32], backend2.local_address());
        let node3 = FederatedNode::new([3u8; 32], backend3.local_address());
        backend1.register_node(&node2).await.unwrap();
        backend1.register_node(&node3).await.unwrap();

        // Broadcast from node 1
        let msg = FederatedMessage::Heartbeat {
            node_id: [1u8; 32],
            round: 10,
            trust_score: 0.9,
            timestamp: 100,
        };
        let sent_count = backend1.broadcast(msg).await.unwrap();
        assert_eq!(sent_count, 2);

        // Both node 2 and node 3 should receive
        let (_addr2, received2) = backend2.receive(Duration::from_secs(5)).await.unwrap();
        assert!(matches!(
            received2,
            FederatedMessage::Heartbeat { round: 10, .. }
        ));

        let (_addr3, received3) = backend3.receive(Duration::from_secs(5)).await.unwrap();
        assert!(matches!(
            received3,
            FederatedMessage::Heartbeat { round: 10, .. }
        ));
    }

    #[tokio::test]
    async fn test_tcp_large_message() {
        let backend1 = TcpBackend::new("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let backend2 = TcpBackend::new("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();

        let node2 = FederatedNode::new([2u8; 32], backend2.local_address());
        backend1.register_node(&node2).await.unwrap();

        // Send a ModelUpdate with 10,000 weights
        let weights: Vec<f32> = (0..10_000).map(|i| i as f32 * 0.001).collect();
        let msg = FederatedMessage::ModelUpdate {
            weights: weights.clone(),
            round: 42,
            source_id: [1u8; 32],
            timestamp: 999,
        };

        backend1
            .send(&backend2.local_address(), msg)
            .await
            .unwrap();

        let (_addr, received) = backend2.receive(Duration::from_secs(5)).await.unwrap();
        match received {
            FederatedMessage::ModelUpdate {
                weights: recv_weights,
                round,
                source_id,
                ..
            } => {
                assert_eq!(round, 42);
                assert_eq!(source_id, [1u8; 32]);
                assert_eq!(recv_weights.len(), 10_000);
                // Verify first and last values survived the round-trip
                assert!((recv_weights[0] - 0.0).abs() < f32::EPSILON);
                assert!((recv_weights[9_999] - 9.999).abs() < 0.001);
            }
            other => panic!("Expected ModelUpdate, got {:?}", other.message_type()),
        }
    }

    #[tokio::test]
    async fn test_tcp_concurrent_sends() {
        let backend1 = TcpBackend::new("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let backend2 = TcpBackend::new("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();

        let node2 = FederatedNode::new([2u8; 32], backend2.local_address());
        backend1.register_node(&node2).await.unwrap();

        let target_addr = backend2.local_address();
        let backend1 = Arc::new(backend1);

        // Spawn 10 concurrent send tasks
        let num_tasks = 10usize;
        let mut handles = Vec::new();

        for i in 0..num_tasks {
            let b1 = Arc::clone(&backend1);
            let target = target_addr.clone();
            handles.push(tokio::spawn(async move {
                let msg = FederatedMessage::Heartbeat {
                    node_id: [1u8; 32],
                    round: i as u64,
                    trust_score: 0.5,
                    timestamp: i as u64,
                };
                b1.send(&target, msg).await
            }));
        }

        // Wait for all sends to complete
        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        // Receive all 10 messages on backend2
        let mut received_rounds = std::collections::HashSet::new();
        for _ in 0..num_tasks {
            let (_addr, msg) = backend2.receive(Duration::from_secs(5)).await.unwrap();
            if let FederatedMessage::Heartbeat { round, .. } = msg {
                received_rounds.insert(round);
            }
        }

        assert_eq!(received_rounds.len(), num_tasks);
        for i in 0..num_tasks {
            assert!(
                received_rounds.contains(&(i as u64)),
                "Missing round {}",
                i
            );
        }
    }

    #[tokio::test]
    async fn test_tcp_receive_timeout() {
        let backend = TcpBackend::new("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();

        let result = backend.receive(Duration::from_millis(50)).await;
        assert!(matches!(result, Err(NetworkError::Timeout { .. })));
    }

    #[tokio::test]
    async fn test_tcp_send_to_non_socket_address() {
        let backend = TcpBackend::new("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();

        let result = backend
            .send(
                &NodeAddress::Channel("some-channel".to_string()),
                FederatedMessage::Heartbeat {
                    node_id: [0u8; 32],
                    round: 0,
                    trust_score: 0.5,
                    timestamp: 0,
                },
            )
            .await;

        assert!(matches!(result, Err(NetworkError::SendFailed { .. })));
    }

    #[tokio::test]
    async fn test_tcp_register_unregister_node() {
        let backend = TcpBackend::new("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();

        let node_id = [42u8; 32];
        let addr: SocketAddr = "127.0.0.1:9999".parse().unwrap();
        let node = FederatedNode::new(node_id, NodeAddress::Socket(addr));

        // Register
        backend.register_node(&node).await.unwrap();
        assert!(backend.registered_nodes.read().contains_key(&node_id));

        // Unregister
        backend.unregister_node(&node_id).await.unwrap();
        assert!(!backend.registered_nodes.read().contains_key(&node_id));
    }

    #[tokio::test]
    async fn test_tcp_bidirectional_communication() {
        let backend1 = TcpBackend::new("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let backend2 = TcpBackend::new("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();

        // Register in both directions
        let node1 = FederatedNode::new([1u8; 32], backend1.local_address());
        let node2 = FederatedNode::new([2u8; 32], backend2.local_address());
        backend1.register_node(&node2).await.unwrap();
        backend2.register_node(&node1).await.unwrap();

        // Send from 1 -> 2
        let msg1 = FederatedMessage::Heartbeat {
            node_id: [1u8; 32],
            round: 1,
            trust_score: 0.5,
            timestamp: 0,
        };
        backend1
            .send(&backend2.local_address(), msg1)
            .await
            .unwrap();

        let (_addr, received1) = backend2.receive(Duration::from_secs(5)).await.unwrap();
        assert!(matches!(
            received1,
            FederatedMessage::Heartbeat { round: 1, .. }
        ));

        // Send from 2 -> 1
        let msg2 = FederatedMessage::Heartbeat {
            node_id: [2u8; 32],
            round: 2,
            trust_score: 0.7,
            timestamp: 0,
        };
        backend2
            .send(&backend1.local_address(), msg2)
            .await
            .unwrap();

        let (_addr, received2) = backend1.receive(Duration::from_secs(5)).await.unwrap();
        assert!(matches!(
            received2,
            FederatedMessage::Heartbeat { round: 2, .. }
        ));
    }
}
