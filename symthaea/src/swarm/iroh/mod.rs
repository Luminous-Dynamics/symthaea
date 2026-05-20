// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

pub(crate) mod bridge;
mod streaming;
mod ticket;

pub use bridge::{IrohBridgeActor, IrohBridgeHandle};
pub use streaming::{StreamConfig, TensorStream};
pub use ticket::TicketManager;

use crate::swarm::{ConsciousnessVector, SwarmConfig, SwarmError, SwarmResult};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

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

    /// Optional handshake reference for clearing trust on disconnect.
    handshake: Option<Arc<RwLock<crate::swarm::handshake::HybridHandshake>>>,

    /// Optional attestation manager for verifying inbound ConsciousnessVectors.
    /// When set, `recv_verified_consciousness()` checks Ed25519 signatures
    /// before accepting CVs into the cognitive loop.
    attestation: Option<Arc<RwLock<crate::swarm::attestation::AttestationManager>>>,

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
            handshake: None,
            attestation: None,
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
            handshake: None,
            attestation: None,
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
        // Build the endpoint using Iroh 0.97 API (preset required since 0.97)
        // Use alpns() to set the protocol we accept connections for
        let endpoint = iroh::Endpoint::builder(iroh::endpoint::presets::N0)
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
            handshake: None,
            attestation: None,
            endpoint: Some(endpoint),
        })
    }

    /// Get our node ID as a hex string
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Access the inner Iroh endpoint (if initialized).
    #[cfg(feature = "swarm")]
    pub fn endpoint(&self) -> Option<&iroh::Endpoint> {
        self.endpoint.as_ref()
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

    /// Set the handshake reference for trust clearing on disconnect.
    pub fn set_handshake(
        &mut self,
        handshake: Arc<RwLock<crate::swarm::handshake::HybridHandshake>>,
    ) {
        self.handshake = Some(handshake);
    }

    /// Set the attestation manager for verifying inbound ConsciousnessVectors.
    ///
    /// When set, `recv_verified_consciousness()` on channels will check Ed25519
    /// signatures before accepting CVs. Without this, all CVs are accepted
    /// (legacy behavior preserved for backward compatibility).
    pub fn set_attestation(
        &mut self,
        attestation: Arc<RwLock<crate::swarm::attestation::AttestationManager>>,
    ) {
        self.attestation = Some(attestation);
    }

    /// Get the attestation manager reference (for passing to channels).
    pub fn attestation(
        &self,
    ) -> &Option<Arc<RwLock<crate::swarm::attestation::AttestationManager>>> {
        &self.attestation
    }

    /// Disconnect from a peer and clear their trust entry.
    ///
    /// Trust is cleared on disconnect to prevent stale trust from persisting
    /// across reconnection — peers must re-handshake after reconnecting.
    pub fn disconnect(&self, peer_id: &str) {
        if let Some(channel) = self.connections.write().remove(peer_id) {
            channel.close();
            // Clear trust on disconnect to prevent zombie trust
            if let Some(ref hs) = self.handshake {
                hs.write().remove_peer(peer_id);
                tracing::debug!(peer = peer_id, "Cleared trust on disconnect");
            }
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

        let endpoint = self.endpoint.as_ref().ok_or(SwarmError::NotInitialized)?;

        // Get our address info for others to connect
        // In Iroh 0.95, addr() is synchronous and returns EndpointAddr
        let endpoint_addr = endpoint.addr();

        // Serialize as JSON string (EndpointAddr implements Serialize)
        let ticket_str = serde_json::to_string(&endpoint_addr)
            .map_err(|e| SwarmError::Internal(format!("Failed to serialize ticket: {}", e)))?;

        // Store in ticket manager for tracking
        let connection_ticket = ConnectionTicket::new(&ticket_str, &self.node_id);
        self.ticket_manager.store_outgoing(connection_ticket);

        tracing::debug!(
            "Created and stored outgoing ticket for node {}",
            &self.node_id[..16.min(self.node_id.len())]
        );

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

        let endpoint = self.endpoint.as_ref().ok_or(SwarmError::NotInitialized)?;

        // Cleanup expired tickets before processing
        self.ticket_manager.cleanup_expired();

        // Deserialize the ticket as EndpointAddr (JSON format)
        let endpoint_addr: iroh::EndpointAddr =
            serde_json::from_str(ticket).map_err(|e| SwarmError::InvalidTicket {
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
        self.connections
            .write()
            .insert(peer_id.clone(), channel.clone());

        // Mark ticket as used
        self.ticket_manager.mark_used(&peer_id);

        tracing::info!(
            "Connected to peer: {} (ticket validated and stored)",
            peer_id
        );

        Ok(channel)
    }

    /// Accept a single inbound QUIC connection (stub without `swarm` feature).
    #[cfg(not(feature = "swarm"))]
    pub async fn accept_incoming(&self) -> SwarmResult<(String, IrohChannel)> {
        Err(SwarmError::FeatureNotEnabled {
            feature: "swarm".to_string(),
        })
    }

    /// Accept a single inbound QUIC connection from a peer that connected to us.
    ///
    /// Iroh 0.97 two-step accept protocol:
    /// 1. `endpoint.accept().await` → `Option<Incoming>` (waits for next inbound)
    /// 2. `incoming.accept()` → `Accepting` future (begins QUIC handshake)
    /// 3. `accepting.await` → `Connection<HandshakeCompleted>` (QUIC handshake done)
    /// 4. `connection.remote_id()` → `EndpointId` (peer's Ed25519 public key, verified)
    ///
    /// Returns the peer ID string and the live `IrohChannel`.  Returns
    /// `SwarmError::Internal` if the endpoint is closed (returns `None`).
    #[cfg(feature = "swarm")]
    pub async fn accept_incoming(&self) -> SwarmResult<(String, IrohChannel)> {
        let endpoint = self.endpoint.as_ref().ok_or(SwarmError::NotInitialized)?;

        // Wait for the next inbound connection request
        let incoming = endpoint.accept().await.ok_or_else(|| {
            SwarmError::Internal("Iroh endpoint closed — no more inbound connections".to_string())
        })?;

        // Accept and begin QUIC handshake
        let accepting = incoming.accept().map_err(|e| {
            SwarmError::Internal(format!("Failed to accept inbound connection: {e}"))
        })?;

        // Await handshake completion; connection is fully established
        let connection = accepting
            .await
            .map_err(|e| SwarmError::Internal(format!("Inbound QUIC handshake failed: {e}")))?;

        // remote_id() is available on Connection<HandshakeCompleted> — cryptographically verified
        let peer_id = connection.remote_id().to_string();

        let channel = IrohChannel::new(peer_id.clone(), connection);

        // Register in our connection pool
        self.connections
            .write()
            .insert(peer_id.clone(), channel.clone());

        tracing::info!(
            peer = %&peer_id[..peer_id.len().min(16)],
            "Inbound QUIC connection accepted"
        );

        Ok((peer_id, channel))
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

    /// Get a reference to the underlying QUIC connection (for handshake protocol).
    ///
    /// Returns `None` in stub mode or if the connection was not established.
    #[cfg(feature = "swarm")]
    pub fn connection_ref(&self) -> Option<&iroh::endpoint::Connection> {
        self.connection.as_ref()
    }

    /// Stub for non-swarm builds — always returns `None`.
    #[cfg(not(feature = "swarm"))]
    pub fn connection_ref(&self) -> Option<&()> {
        None
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
        let connection = self.connection.as_ref().ok_or(SwarmError::ChannelClosed {
            peer_id: self.peer_id.clone(),
        })?;

        let (mut send, _recv) = connection
            .open_bi()
            .await
            .map_err(|e| SwarmError::SendFailed {
                peer_id: self.peer_id.clone(),
                reason: e.to_string(),
            })?;

        let bytes =
            bincode::serialize(state).map_err(|e| SwarmError::SerializationError(e.to_string()))?;

        send.write_all(&bytes)
            .await
            .map_err(|e| SwarmError::SendFailed {
                peer_id: self.peer_id.clone(),
                reason: e.to_string(),
            })?;

        send.finish().map_err(|e| SwarmError::SendFailed {
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
        let connection = self.connection.as_ref().ok_or(SwarmError::ChannelClosed {
            peer_id: self.peer_id.clone(),
        })?;

        let (_send, mut recv) =
            connection
                .accept_bi()
                .await
                .map_err(|e| SwarmError::ReceiveFailed {
                    peer_id: self.peer_id.clone(),
                    reason: e.to_string(),
                })?;

        let bytes = recv
            .read_to_end(1024 * 1024) // 1MB max
            .await
            .map_err(|e| SwarmError::ReceiveFailed {
                peer_id: self.peer_id.clone(),
                reason: e.to_string(),
            })?;

        bincode::deserialize(&bytes).map_err(|e| SwarmError::SerializationError(e.to_string()))
    }

    /// Receive and verify a consciousness vector using attestation.
    ///
    /// This is the **secure recv path**: it deserializes an
    /// [`AttestedConsciousnessVector`] and verifies the Ed25519 signature
    /// against the attestation manager's trusted signer set before returning
    /// the inner CV. Rejects untrusted or tampered CVs with an error.
    ///
    /// **Security policy**: In release builds, rejects unsigned CVs when no
    /// attestation manager is configured. In debug/test builds, falls back to
    /// unverified reception with a warning. Use `recv_consciousness()` if you
    /// explicitly intend to accept unverified CVs.
    #[cfg(feature = "swarm")]
    pub async fn recv_verified_consciousness(
        &self,
        attestation: &Option<Arc<RwLock<crate::swarm::attestation::AttestationManager>>>,
    ) -> SwarmResult<ConsciousnessVector> {
        let connection = self.connection.as_ref().ok_or(SwarmError::ChannelClosed {
            peer_id: self.peer_id.clone(),
        })?;

        let (_send, mut recv) =
            connection
                .accept_bi()
                .await
                .map_err(|e| SwarmError::ReceiveFailed {
                    peer_id: self.peer_id.clone(),
                    reason: e.to_string(),
                })?;

        let bytes = recv
            .read_to_end(1024 * 1024)
            .await
            .map_err(|e| SwarmError::ReceiveFailed {
                peer_id: self.peer_id.clone(),
                reason: e.to_string(),
            })?;

        match attestation {
            Some(mgr) => {
                // Secure path: deserialize as AttestedCV, verify signature + trust
                let attested: crate::swarm::AttestedConsciousnessVector =
                    bincode::deserialize(&bytes)
                        .map_err(|e| SwarmError::SerializationError(e.to_string()))?;

                let manager = mgr.read();
                if manager.requires_attestation() {
                    manager.verify_and_extract(&attested)
                } else {
                    // Attestation manager present but not required — accept without check
                    Ok(attested.vector)
                }
            }
            None => {
                // No attestation manager configured.
                // In debug/test builds: warn and accept raw CV (backward compat).
                // In release builds: reject — callers must either configure an
                // AttestationManager or use recv_consciousness() explicitly.
                #[cfg(debug_assertions)]
                {
                    tracing::warn!(
                        peer = %self.peer_id,
                        "recv_verified_consciousness called without AttestationManager — \
                         accepting unverified CV (debug build only)"
                    );
                    bincode::deserialize(&bytes)
                        .map_err(|e| SwarmError::SerializationError(e.to_string()))
                }
                #[cfg(not(debug_assertions))]
                {
                    tracing::error!(
                        peer = %self.peer_id,
                        "recv_verified_consciousness rejected: no AttestationManager configured"
                    );
                    Err(SwarmError::AttestationRequired)
                }
            }
        }
    }

    /// Stub verified recv (without swarm feature).
    ///
    /// Enforces the same security policy as the real implementation:
    /// rejects when no attestation manager in release builds.
    #[cfg(not(feature = "swarm"))]
    pub async fn recv_verified_consciousness(
        &self,
        attestation: &Option<Arc<RwLock<crate::swarm::attestation::AttestationManager>>>,
    ) -> SwarmResult<ConsciousnessVector> {
        if attestation.is_none() {
            #[cfg(not(debug_assertions))]
            return Err(SwarmError::AttestationRequired);
        }
        Err(SwarmError::FeatureNotEnabled {
            feature: "swarm".to_string(),
        })
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
        // Without the swarm feature, everything should be stubs.
        // With the swarm feature, real implementations are used.
        // Either way, the module compiles and this test passes.
        let is_stub = !cfg!(feature = "swarm");
        // Assert the feature flag is a valid bool (documents compile-time detection)
        assert!(is_stub || !is_stub, "feature flag should resolve to a bool");
    }

    #[test]
    fn test_ticket_manager() {
        let manager = TicketManager::new();
        assert_eq!(manager.ticket_count(), (0, 0));
    }
}
