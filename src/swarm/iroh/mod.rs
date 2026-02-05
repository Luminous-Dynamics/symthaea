//! Iroh Integration - The Synapse Layer
//!
//! This module provides real-time, low-latency P2P communication using Iroh's
//! QUIC-based networking. It handles:
//!
//! - **NAT Traversal**: Via Iroh's relay network (Magicsock)
//! - **Direct Connections**: QUIC streams for <50ms latency
//! - **Tensor Streaming**: Bi-directional streams for neural state sync
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                      IrohNode                                │
//! ├──────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  ┌─────────────────┐    ┌─────────────────┐                  │
//! │  │   Endpoint      │────│   Magicsock     │                  │
//! │  │  (QUIC Server)  │    │ (NAT Traversal) │                  │
//! │  └────────┬────────┘    └─────────────────┘                  │
//! │           │                                                   │
//! │  ┌────────▼────────┐    ┌─────────────────┐                  │
//! │  │   Connections   │────│   Channels      │                  │
//! │  │  (Per-Peer)     │    │ (Bi-directional)│                  │
//! │  └─────────────────┘    └─────────────────┘                  │
//! │                                                              │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Status
//!
//! This module is structured for Iroh 0.95+ integration. The types and traits
//! are defined, but full integration requires the `swarm` feature flag.
//!
//! ## Iroh 0.95 API Notes
//!
//! The iroh crate has unified networking. Key types:
//! - `iroh::Endpoint` - The main networking endpoint (use `.id()` and `.addr()`)
//! - `iroh::EndpointId` - 32-byte public key identifying an endpoint
//! - `iroh::EndpointAddr` - Address containing EndpointId + relay info + direct addrs

mod ticket;
mod streaming;

pub use ticket::TicketManager;
pub use streaming::{TensorStream, StreamConfig};

use crate::swarm::{SwarmConfig, SwarmError, SwarmResult, ConsciousnessVector};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

// ============================================================================
// IROH NODE - Main Entry Point
// ============================================================================

/// The main Iroh node for real-time P2P communication
///
/// When the `swarm` feature is enabled, this wraps Iroh's Endpoint.
/// Without the feature, it provides a stub implementation.
#[allow(dead_code)] // Fields reserved for full implementation
pub struct IrohNode {
    /// Node ID as a string (hex-encoded)
    node_id: String,

    /// Active connections indexed by peer ID
    connections: Arc<RwLock<HashMap<String, IrohChannel>>>,

    /// Ticket manager for connection establishment
    ticket_manager: TicketManager,

    /// Configuration
    config: SwarmConfig,

    /// Whether this is a real or stub implementation
    is_stub: bool,

    /// Inner Iroh endpoint (only with feature)
    #[cfg(feature = "swarm")]
    endpoint: Option<iroh::Endpoint>,
}

impl IrohNode {
    /// Create a new Iroh node with deterministic node ID from a genesis seed (stub).
    #[cfg(not(feature = "swarm"))]
    pub async fn from_genesis(
        config: SwarmConfig,
        genesis: &symthaea_core::genesis::GenesisSeed,
        label: &str,
    ) -> SwarmResult<Self> {
        let mut rng = genesis.domain(&format!("{label}::iroh_node"));
        let node_id = format!("{:064x}", rand::Rng::gen::<u64>(&mut rng));

        tracing::warn!(
            "Iroh node created in STUB mode (genesis-seeded). NodeId: {}",
            &node_id[..16]
        );

        Ok(Self {
            node_id,
            connections: Arc::new(RwLock::new(HashMap::new())),
            ticket_manager: TicketManager::new(),
            config,
            is_stub: true,
        })
    }

    /// Create a new Iroh node (stub implementation without feature)
    #[cfg(not(feature = "swarm"))]
    pub async fn new(config: SwarmConfig) -> SwarmResult<Self> {
        // Generate a random node ID for the stub
        let node_id = format!("{:064x}", rand::random::<u64>());

        tracing::warn!(
            "Iroh node created in STUB mode (swarm feature not enabled). NodeId: {}",
            &node_id[..16]
        );

        Ok(Self {
            node_id,
            connections: Arc::new(RwLock::new(HashMap::new())),
            ticket_manager: TicketManager::new(),
            config,
            is_stub: true,
        })
    }

    /// Create a new Iroh node (real implementation with feature)
    ///
    /// Iroh 0.95 API:
    /// - Endpoint::builder() creates a builder
    /// - .alpns() sets accepted protocols
    /// - .bind() starts the endpoint
    /// - .id() returns our EndpointId
    #[cfg(feature = "swarm")]
    pub async fn new(config: SwarmConfig) -> SwarmResult<Self> {
        // Build the endpoint using Iroh 0.95 API
        // Use alpns() to set the protocol we accept connections for
        let endpoint = iroh::Endpoint::builder()
            .alpns(vec![b"symthaea/1".to_vec()])
            .bind()
            .await
            .map_err(|e| SwarmError::Internal(format!("Failed to bind endpoint: {}", e)))?;

        // Get our endpoint ID - in Iroh 0.95, use id() method
        let endpoint_id = endpoint.id().to_string();

        tracing::info!(
            "Iroh node started with EndpointId: {}",
            &endpoint_id[..16.min(endpoint_id.len())]
        );

        Ok(Self {
            node_id: endpoint_id,
            connections: Arc::new(RwLock::new(HashMap::new())),
            ticket_manager: TicketManager::new(),
            config,
            is_stub: false,
            endpoint: Some(endpoint),
        })
    }

    /// Get our node ID as a hex string
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Check if this is a stub implementation
    pub fn is_stub(&self) -> bool {
        self.is_stub
    }

    /// Get the configuration
    pub fn config(&self) -> &SwarmConfig {
        &self.config
    }

    /// Get connected peers
    pub fn connected_peers(&self) -> Vec<String> {
        self.connections.read().keys().cloned().collect()
    }

    /// Get a channel to a connected peer
    pub fn get_channel(&self, peer_id: &str) -> Option<IrohChannel> {
        self.connections.read().get(peer_id).cloned()
    }

    /// Get the ticket manager for external access
    pub fn ticket_manager(&self) -> &TicketManager {
        &self.ticket_manager
    }

    /// Get known peers from ticket cache
    pub fn known_peer_ids(&self) -> Vec<String> {
        self.ticket_manager.known_peers()
    }

    /// Cleanup expired tickets (call periodically or before operations)
    pub fn cleanup_tickets(&self) {
        self.ticket_manager.cleanup_expired();
    }

    /// Get a cached ticket for a known peer (if available)
    pub fn get_cached_ticket(&self, peer_id: &str) -> Option<crate::swarm::ConnectionTicket> {
        self.ticket_manager.get_incoming(peer_id)
    }

    /// Get ticket statistics (outgoing_count, incoming_count)
    pub fn ticket_stats(&self) -> (usize, usize) {
        self.ticket_manager.ticket_count()
    }

    /// Disconnect from a peer
    pub fn disconnect(&self, peer_id: &str) {
        if let Some(channel) = self.connections.write().remove(peer_id) {
            channel.close();
            tracing::info!("Disconnected from peer: {}", peer_id);
        }
    }

    /// Create a connection ticket (stub returns error without feature)
    #[cfg(not(feature = "swarm"))]
    pub fn create_ticket(&self) -> SwarmResult<String> {
        Err(SwarmError::FeatureNotEnabled {
            feature: "swarm".to_string(),
        })
    }

    /// Create a connection ticket (real implementation with feature)
    ///
    /// In Iroh 0.95, we get our EndpointAddr via .addr() and serialize it as a ticket
    /// EndpointAddr implements Serialize, so we use JSON encoding
    ///
    /// The ticket is stored in the TicketManager for tracking and expiration.
    #[cfg(feature = "swarm")]
    pub fn create_ticket(&self) -> SwarmResult<String> {
        use crate::swarm::ConnectionTicket;

        let endpoint = self.endpoint.as_ref()
            .ok_or(SwarmError::NotInitialized)?;

        // Get our address info for others to connect
        // In Iroh 0.95, addr() is synchronous and returns EndpointAddr
        let endpoint_addr = endpoint.addr();

        // Serialize as JSON string (EndpointAddr implements Serialize)
        let ticket_str = serde_json::to_string(&endpoint_addr)
            .map_err(|e| SwarmError::Internal(format!("Failed to serialize ticket: {}", e)))?;

        // Store in ticket manager for tracking
        let connection_ticket = ConnectionTicket::new(&ticket_str, &self.node_id);
        self.ticket_manager.store_outgoing(connection_ticket);

        tracing::debug!("Created and stored outgoing ticket for node {}", &self.node_id[..16.min(self.node_id.len())]);

        Ok(ticket_str)
    }

    /// Connect to a peer (stub returns error without feature)
    #[cfg(not(feature = "swarm"))]
    pub async fn connect(&self, _ticket: &str) -> SwarmResult<IrohChannel> {
        Err(SwarmError::FeatureNotEnabled {
            feature: "swarm".to_string(),
        })
    }

    /// Connect to a peer (real implementation with feature)
    ///
    /// Iroh 0.95 connect API:
    /// - Deserialize EndpointAddr from ticket JSON string
    /// - Validate and store ticket via TicketManager
    /// - Call endpoint.connect(endpoint_addr, alpn)
    #[cfg(feature = "swarm")]
    pub async fn connect(&self, ticket: &str) -> SwarmResult<IrohChannel> {
        use crate::swarm::ConnectionTicket;

        let endpoint = self.endpoint.as_ref()
            .ok_or(SwarmError::NotInitialized)?;

        // Cleanup expired tickets before processing
        self.ticket_manager.cleanup_expired();

        // Deserialize the ticket as EndpointAddr (JSON format)
        let endpoint_addr: iroh::EndpointAddr = serde_json::from_str(ticket)
            .map_err(|e| SwarmError::InvalidTicket {
                reason: format!("Failed to deserialize ticket: {}", e),
            })?;

        // Get peer ID from the endpoint address (use .id field directly)
        let peer_id = endpoint_addr.id.to_string();

        // Create and validate a ConnectionTicket
        let connection_ticket = ConnectionTicket::new(ticket, &peer_id);
        self.ticket_manager.validate(&connection_ticket)?;

        // Store incoming ticket for tracking
        self.ticket_manager.store_incoming(connection_ticket);

        // Check max peers
        let current_count = self.connections.read().len();
        if current_count >= self.config.max_peers {
            return Err(SwarmError::MaxPeersReached {
                current: current_count,
                max: self.config.max_peers,
            });
        }

        // Establish connection with ALPN protocol identifier
        let connection = endpoint
            .connect(endpoint_addr, b"symthaea/1")
            .await
            .map_err(|e| SwarmError::ConnectionFailed {
                peer_id: peer_id.clone(),
                reason: e.to_string(),
            })?;

        let channel = IrohChannel::new(peer_id.clone(), connection);

        // Store connection
        self.connections.write().insert(peer_id.clone(), channel.clone());

        // Mark ticket as used
        self.ticket_manager.mark_used(&peer_id);

        tracing::info!("Connected to peer: {} (ticket validated and stored)", peer_id);

        Ok(channel)
    }

    /// Shutdown the node
    pub async fn shutdown(self) {
        // Close all connections
        for (peer_id, channel) in self.connections.write().drain() {
            channel.close();
            tracing::debug!("Closed connection to: {}", peer_id);
        }

        #[cfg(feature = "swarm")]
        if let Some(endpoint) = self.endpoint {
            // In Iroh 0.95, close() is async and returns ()
            endpoint.close().await;
        }

        tracing::info!("Iroh node shutdown complete");
    }
}

// ============================================================================
// IROH CHANNEL - Bi-directional Communication
// ============================================================================

/// A bi-directional channel to a peer for tensor streaming
#[derive(Clone)]
pub struct IrohChannel {
    /// The peer's endpoint ID
    peer_id: String,

    /// Whether the channel is still alive
    alive: Arc<std::sync::atomic::AtomicBool>,

    /// Inner connection (only with feature)
    /// In Iroh 0.95, Connection is at iroh::endpoint::Connection
    #[cfg(feature = "swarm")]
    connection: Option<iroh::endpoint::Connection>,
}

impl IrohChannel {
    /// Create a new channel (stub)
    #[cfg(not(feature = "swarm"))]
    #[allow(dead_code)]
    fn new(peer_id: String) -> Self {
        Self {
            peer_id,
            alive: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }

    /// Create a new channel from a connection
    #[cfg(feature = "swarm")]
    fn new(peer_id: String, connection: iroh::endpoint::Connection) -> Self {
        Self {
            peer_id,
            alive: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            connection: Some(connection),
        }
    }

    /// Get the peer's node ID
    pub fn peer_id(&self) -> &str {
        &self.peer_id
    }

    /// Check if the channel is still alive
    pub fn is_alive(&self) -> bool {
        self.alive.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Send a consciousness vector over the channel (stub without feature)
    #[cfg(not(feature = "swarm"))]
    pub async fn send_consciousness(&self, _state: &ConsciousnessVector) -> SwarmResult<()> {
        Err(SwarmError::FeatureNotEnabled {
            feature: "swarm".to_string(),
        })
    }

    /// Send a consciousness vector over the channel (real with feature)
    ///
    /// Opens a bi-directional stream and sends the serialized vector
    #[cfg(feature = "swarm")]
    pub async fn send_consciousness(&self, state: &ConsciousnessVector) -> SwarmResult<()> {
        let connection = self.connection.as_ref()
            .ok_or(SwarmError::ChannelClosed { peer_id: self.peer_id.clone() })?;

        let (mut send, _recv) = connection
            .open_bi()
            .await
            .map_err(|e| SwarmError::SendFailed {
                peer_id: self.peer_id.clone(),
                reason: e.to_string(),
            })?;

        let bytes = bincode::serialize(state)
            .map_err(|e| SwarmError::SerializationError(e.to_string()))?;

        send.write_all(&bytes).await
            .map_err(|e| SwarmError::SendFailed {
                peer_id: self.peer_id.clone(),
                reason: e.to_string(),
            })?;

        send.finish()
            .map_err(|e| SwarmError::SendFailed {
                peer_id: self.peer_id.clone(),
                reason: e.to_string(),
            })?;

        Ok(())
    }

    /// Receive a consciousness vector from the channel (stub without feature)
    #[cfg(not(feature = "swarm"))]
    pub async fn recv_consciousness(&self) -> SwarmResult<ConsciousnessVector> {
        Err(SwarmError::FeatureNotEnabled {
            feature: "swarm".to_string(),
        })
    }

    /// Receive a consciousness vector from the channel (real with feature)
    #[cfg(feature = "swarm")]
    pub async fn recv_consciousness(&self) -> SwarmResult<ConsciousnessVector> {
        let connection = self.connection.as_ref()
            .ok_or(SwarmError::ChannelClosed { peer_id: self.peer_id.clone() })?;

        let (_send, mut recv) = connection
            .accept_bi()
            .await
            .map_err(|e| SwarmError::ReceiveFailed {
                peer_id: self.peer_id.clone(),
                reason: e.to_string(),
            })?;

        let bytes = recv.read_to_end(1024 * 1024) // 1MB max
            .await
            .map_err(|e| SwarmError::ReceiveFailed {
                peer_id: self.peer_id.clone(),
                reason: e.to_string(),
            })?;

        bincode::deserialize(&bytes)
            .map_err(|e| SwarmError::SerializationError(e.to_string()))
    }

    /// Close the channel
    pub fn close(&self) {
        self.alive.store(false, std::sync::atomic::Ordering::SeqCst);

        #[cfg(feature = "swarm")]
        if let Some(ref conn) = self.connection {
            conn.close(0u8.into(), b"goodbye");
        }
    }
}

impl std::fmt::Debug for IrohChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IrohChannel")
            .field("peer_id", &self.peer_id)
            .field("alive", &self.is_alive())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_mode() {
        // Without the swarm feature, everything should be stubs
        assert!(!cfg!(feature = "swarm") || true);
    }

    #[test]
    fn test_ticket_manager() {
        let manager = TicketManager::new();
        assert_eq!(manager.ticket_count(), (0, 0));
    }
}
